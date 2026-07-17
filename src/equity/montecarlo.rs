//! Monte Carlo pricing engine.
//!
//! - Terminal-value simulation (exact GBM step) for European payoffs, with
//!   a low-discrepancy (Sobol) sampler by default.
//! - Path-wise simulation with Exact / Euler / Milstein discretization
//!   (seeded pseudo-random draws).
//! - American exercise via Longstaff-Schwartz least-squares regression.
//! - Greeks by central-difference bump-and-reprice with common random
//!   numbers: every price uses the same deterministic draws, so the
//!   difference isolates the sensitivity rather than sampler noise.

use std::io;
use std::str::FromStr;
use chrono::{Local, NaiveDate};
use libm::exp;

use crate::core::utils::ContractStyle;
use super::local_vol::LocalVol;
use super::vanila_option::{EquityOption, EquityOptionBase, VanillaPayoff};
use super::utils::{Engine, LongShort, Payoff};
use crate::core::trade::PutOrCall;
use crate::utils::RNG::{pseudo_normal_matrix, pseudo_normals, sobol_normals};
use crate::core::quotes::Quote;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::data_models::EquityOptionData;
use crate::core::vols::VolSurface;
use crate::core::traits::Instrument;

/// Time-stepping scheme for path-wise simulation.
/// `Exact` samples the closed-form GBM transition (no discretization bias);
/// Euler and Milstein are the standard approximate schemes (useful as the
/// basis for models without closed-form transitions, e.g. local vol/Heston).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscretizationScheme {
    Exact,
    Euler,
    Milstein,
}

impl FromStr for DiscretizationScheme {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "exact" => Ok(DiscretizationScheme::Exact),
            "euler" => Ok(DiscretizationScheme::Euler),
            "milstein" => Ok(DiscretizationScheme::Milstein),
            other => Err(format!("Invalid discretization scheme '{other}'")),
        }
    }
}

/// Draw sampler. `Sobol` (low-discrepancy, default) applies to
/// single-dimension terminal simulation; path-wise and American simulation
/// always use the seeded pseudo-random generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    Sobol,
    PseudoRandom,
}

impl FromStr for Sampler {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "sobol" | "quasi" => Ok(Sampler::Sobol),
            "pseudo" | "pseudorandom" | "pseudo_random" => Ok(Sampler::PseudoRandom),
            other => Err(format!("Invalid sampler '{other}'")),
        }
    }
}

/// Dynamics used for path generation. `Gbm` diffuses at the option's own
/// (constant) implied vol; `LocalVol` diffuses at the Dupire local
/// volatility calibrated from the option's vol surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McModel {
    Gbm,
    LocalVol,
}

impl FromStr for McModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "gbm" | "blackscholes" | "bs" => Ok(McModel::Gbm),
            "local_vol" | "localvol" | "lv" => Ok(McModel::LocalVol),
            other => Err(format!("Invalid mc_model '{other}'")),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MonteCarloConfig {
    pub paths: usize,
    /// 1 = terminal simulation (exact); > 1 = path-wise stepping.
    /// Local vol always steps path-wise (at least [`LOCAL_VOL_MIN_STEPS`]).
    pub time_steps: usize,
    pub scheme: DiscretizationScheme,
    pub sampler: Sampler,
    pub model: McModel,
    pub seed: u64,
}

pub const LOCAL_VOL_MIN_STEPS: usize = 100;

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig {
            paths: 100_000,
            time_steps: 1,
            scheme: DiscretizationScheme::Exact,
            sampler: Sampler::Sobol,
            model: McModel::Gbm,
            seed: 42,
        }
    }
}

impl MonteCarloConfig {
    pub fn from_data(data: &EquityOptionData) -> Self {
        let defaults = MonteCarloConfig::default();
        let scheme = data
            .mc_scheme
            .as_deref()
            .map(|s| s.parse::<DiscretizationScheme>().expect("Invalid mc_scheme"))
            .unwrap_or(defaults.scheme);
        // approximate schemes need real time-stepping to mean anything
        let default_steps = match scheme {
            DiscretizationScheme::Exact => 1,
            _ => 252,
        };
        let model = data
            .mc_model
            .as_deref()
            .map(|s| s.parse::<McModel>().expect("Invalid mc_model"))
            .unwrap_or(defaults.model);
        MonteCarloConfig {
            paths: data.simulation.unwrap_or(defaults.paths as u64) as usize,
            time_steps: data.mc_time_steps.unwrap_or(default_steps),
            scheme,
            sampler: data
                .mc_sampler
                .as_deref()
                .map(|s| s.parse::<Sampler>().expect("Invalid mc_sampler"))
                .unwrap_or(defaults.sampler),
            model,
            seed: data.mc_seed.unwrap_or(defaults.seed),
        }
    }
}

/// Market inputs snapshot; Greeks bump these fields and reprice with the
/// same draws (common random numbers).
#[derive(Debug, Clone, Copy)]
struct MarketParams {
    s0: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
}

fn market_params(option: &EquityOption) -> MarketParams {
    MarketParams {
        s0: option.base.underlying_price.value(),
        strike: option.base.strike_price,
        r: option.base.risk_free_rate(),
        q: option.base.dividend_yield,
        sigma: option.base.volatility(),
        t: option.time_to_maturity(),
    }
}

pub fn npv(option: &EquityOption) -> f64 {
    assert!(option.base.volatility() >= 0.0);
    assert!(option.base.time_to_maturity() >= 0.0);
    assert!(option.base.underlying_price.value >= 0.0);
    price(option, &market_params(option))
}

fn price(option: &EquityOption, p: &MarketParams) -> f64 {
    match option.payoff.exercise_style() {
        ContractStyle::American => american_npv(option, p),
        _ => european_npv(option, p),
    }
}

// ── Greeks: central-difference bumps with common random numbers ─────────

pub fn delta(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = p.s0 * 0.01;
    (price(option, &MarketParams { s0: p.s0 + h, ..p })
        - price(option, &MarketParams { s0: p.s0 - h, ..p }))
        / (2.0 * h)
}

pub fn gamma(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = p.s0 * 0.01;
    (price(option, &MarketParams { s0: p.s0 + h, ..p })
        - 2.0 * price(option, &p)
        + price(option, &MarketParams { s0: p.s0 - h, ..p }))
        / (h * h)
}

pub fn vega(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = 0.01;
    (price(option, &MarketParams { sigma: p.sigma + h, ..p })
        - price(option, &MarketParams { sigma: p.sigma - h, ..p }))
        / (2.0 * h)
}

pub fn theta(option: &EquityOption) -> f64 {
    // theta = dV/dt (calendar) = -dV/dT
    let p = market_params(option);
    let h = (1.0 / 365.0_f64).min(0.5 * p.t);
    -(price(option, &MarketParams { t: p.t + h, ..p })
        - price(option, &MarketParams { t: p.t - h, ..p }))
        / (2.0 * h)
}

pub fn rho(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = 1e-4;
    (price(option, &MarketParams { r: p.r + h, ..p })
        - price(option, &MarketParams { r: p.r - h, ..p }))
        / (2.0 * h)
}

// ── Volatility dynamics along a path ────────────────────────────────────

enum PathVol<'a> {
    Const(f64),
    Local(LocalVol<'a>),
}

impl PathVol<'_> {
    fn vol(&self, s: f64, t: f64) -> f64 {
        match self {
            PathVol::Const(v) => *v,
            PathVol::Local(lv) => lv.vol(s, t),
        }
    }
}

fn path_vol<'a>(option: &'a EquityOption, p: &MarketParams) -> PathVol<'a> {
    match option.mc.model {
        McModel::Gbm => PathVol::Const(p.sigma),
        McModel::LocalVol => PathVol::Local(LocalVol::new(
            &option.base.vol_surface,
            &option.base.discount_curve,
            // the local vol function is frozen at the calibration spot;
            // spot bumps (delta/gamma) move the path start, not the model
            option.base.underlying_price.value(),
            option.base.dividend_yield,
            // vega bumps enter as a parallel shift of the implied surface
            p.sigma - option.base.volatility(),
        )),
    }
}

fn effective_steps(cfg: &MonteCarloConfig) -> usize {
    match cfg.model {
        McModel::LocalVol => cfg.time_steps.max(LOCAL_VOL_MIN_STEPS),
        McModel::Gbm => cfg.time_steps,
    }
}

// ── European ────────────────────────────────────────────────────────────

fn european_npv(option: &EquityOption, p: &MarketParams) -> f64 {
    let df = exp(-p.r * p.t);
    let terminals = simulate_terminals(option, p);
    let sum: f64 = terminals.iter().map(|s| option.payoff.payoff(*s, p.strike)).sum();
    df * sum / terminals.len() as f64
}

/// Terminal underlying levels under the risk-neutral measure.
fn simulate_terminals(option: &EquityOption, p: &MarketParams) -> Vec<f64> {
    let cfg = &option.mc;
    let steps = effective_steps(cfg);
    if steps <= 1 {
        // exact one-step GBM transition (constant vol only)
        let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * p.t;
        let vol_sqrt_t = p.sigma * p.t.sqrt();
        let draws = match cfg.sampler {
            Sampler::Sobol => sobol_normals(cfg.paths),
            Sampler::PseudoRandom => pseudo_normals(cfg.paths, cfg.seed),
        };
        draws.iter().map(|z| p.s0 * exp(drift + vol_sqrt_t * z)).collect()
    } else {
        let vol_model = path_vol(option, p);
        let dt = p.t / steps as f64;
        let draws = pseudo_normal_matrix(cfg.paths, steps, cfg.seed);
        draws
            .iter()
            .map(|path| {
                let mut s = p.s0;
                for (i, z) in path.iter().enumerate() {
                    let sigma = vol_model.vol(s, i as f64 * dt);
                    s = step(cfg.scheme, s, dt, *z, p.r - p.q, sigma);
                }
                s
            })
            .collect()
    }
}

fn step(scheme: DiscretizationScheme, s: f64, dt: f64, z: f64, drift: f64, sigma: f64) -> f64 {
    let dw = dt.sqrt() * z;
    let next = match scheme {
        DiscretizationScheme::Exact => s * exp((drift - 0.5 * sigma * sigma) * dt + sigma * dw),
        DiscretizationScheme::Euler => s * (1.0 + drift * dt + sigma * dw),
        DiscretizationScheme::Milstein => {
            s * (1.0 + drift * dt + sigma * dw + 0.5 * sigma * sigma * (dw * dw - dt))
        }
    };
    next.max(0.0)
}

// ── American: Longstaff-Schwartz ────────────────────────────────────────

const LSMC_DEFAULT_STEPS: usize = 50;

/// Least-squares Monte Carlo (Longstaff-Schwartz 2001): backward induction
/// where the continuation value on in-the-money paths is regressed on a
/// quadratic polynomial of the (normalized) spot.
fn american_npv(option: &EquityOption, p: &MarketParams) -> f64 {
    let cfg = &option.mc;
    let steps =
        if cfg.time_steps > 1 { cfg.time_steps } else { LSMC_DEFAULT_STEPS }.max(match cfg.model {
            McModel::LocalVol => LOCAL_VOL_MIN_STEPS,
            McModel::Gbm => 1,
        });
    let dt = p.t / steps as f64;
    let disc = exp(-p.r * dt);

    // full paths (exact stepping unless an approximate scheme is requested)
    let vol_model = path_vol(option, p);
    let draws = pseudo_normal_matrix(cfg.paths, steps, cfg.seed);
    let spots: Vec<Vec<f64>> = draws
        .iter()
        .map(|path| {
            let mut s = p.s0;
            path.iter()
                .enumerate()
                .map(|(i, z)| {
                    let sigma = vol_model.vol(s, i as f64 * dt);
                    s = step(cfg.scheme, s, dt, *z, p.r - p.q, sigma);
                    s
                })
                .collect()
        })
        .collect();

    // cashflows initialized at expiry
    let mut cashflow: Vec<f64> =
        spots.iter().map(|path| option.payoff.payoff(path[steps - 1], p.strike)).collect();

    for step_idx in (0..steps - 1).rev() {
        for cf in cashflow.iter_mut() {
            *cf *= disc;
        }
        // regress discounted continuation on in-the-money paths only
        let itm: Vec<usize> = (0..spots.len())
            .filter(|&i| option.payoff.payoff(spots[i][step_idx], p.strike) > 0.0)
            .collect();
        if itm.len() < 3 {
            continue;
        }
        let xs: Vec<f64> = itm.iter().map(|&i| spots[i][step_idx] / p.s0).collect();
        let ys: Vec<f64> = itm.iter().map(|&i| cashflow[i]).collect();
        let Some(beta) = quadratic_least_squares(&xs, &ys) else { continue };
        for (k, &i) in itm.iter().enumerate() {
            let x = xs[k];
            let continuation = beta[0] + beta[1] * x + beta[2] * x * x;
            let exercise = option.payoff.payoff(spots[i][step_idx], p.strike);
            if exercise > continuation {
                cashflow[i] = exercise;
            }
        }
    }

    // discount the first exercise date back to today
    disc * cashflow.iter().sum::<f64>() / cashflow.len() as f64
}

/// Least squares fit of `y ~ b0 + b1 x + b2 x^2` via the normal equations.
/// Returns None if the system is (near-)singular.
fn quadratic_least_squares(x: &[f64], y: &[f64]) -> Option<[f64; 3]> {
    let n = x.len() as f64;
    let (mut sx, mut sx2, mut sx3, mut sx4) = (0.0, 0.0, 0.0, 0.0);
    let (mut sy, mut sxy, mut sx2y) = (0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        let xi2 = xi * xi;
        sx += xi;
        sx2 += xi2;
        sx3 += xi2 * xi;
        sx4 += xi2 * xi2;
        sy += yi;
        sxy += xi * yi;
        sx2y += xi2 * yi;
    }
    let mut a = [[n, sx, sx2, sy], [sx, sx2, sx3, sxy], [sx2, sx3, sx4, sx2y]];
    // gaussian elimination with partial pivoting
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        for row in col + 1..3 {
            let f = a[row][col] / a[col][col];
            for k in col..4 {
                a[row][k] -= f * a[col][k];
            }
        }
    }
    let mut beta = [0.0; 3];
    for row in (0..3).rev() {
        let mut acc = a[row][3];
        for k in row + 1..3 {
            acc -= a[row][k] * beta[k];
        }
        beta[row] = acc / a[row][row];
    }
    Some(beta)
}

// ── Interactive CLI helper ──────────────────────────────────────────────

pub fn option_pricing() {
    println!("Welcome to the Monte Carlo Option pricer.");
    println!("(Step 1/7) What is the current price of the underlying asset?");
    let mut curr_price = String::new();
    io::stdin()
        .read_line(&mut curr_price)
        .expect("Failed to read line");

    println!("(Step 2/7) Do you want a call option ('C') or a put option ('P') ?");
    let mut side_input = String::new();
    io::stdin()
        .read_line(&mut side_input)
        .expect("Failed to read line");

    let side: PutOrCall;
    match side_input.trim() {
        "C" | "c" | "Call" | "call" => side = PutOrCall::Call,
        "P" | "p" | "Put" | "put" => side = PutOrCall::Put,
        _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
    }

    println!("Stike price:");
    let mut strike = String::new();
    io::stdin()
        .read_line(&mut strike)
        .expect("Failed to read line");

    println!("Expected annualized volatility in %:");
    println!("E.g.: Enter 50% chance as 0.50 ");
    let mut vol = String::new();
    io::stdin()
        .read_line(&mut vol)
        .expect("Failed to read line");

    println!("Risk-free rate in %:");
    let mut rf = String::new();
    io::stdin().read_line(&mut rf).expect("Failed to read line");

    println!("Maturity date in YYYY-MM-DD format:");
    let mut expiry = String::new();
    io::stdin()
        .read_line(&mut expiry)
        .expect("Failed to read line");
    let future_date = NaiveDate::parse_from_str(&expiry.trim(), "%Y-%m-%d").expect("Invalid date format");
    println!("Dividend yield on this stock:");
    let mut div = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");

    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        rf.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .expect("Invalid risk free rate");
    let vol_surface = VolSurface::flat(
        vol.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
    )
    .expect("Invalid volatility");
    let curr_quote = Quote::new(curr_price.trim().parse::<f64>().unwrap());
    let option = EquityOptionBase {
        symbol: "ABC".to_string(),
        currency: None,
        exchange: None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: Some("ABC".to_string()),
        entry_price: 0.0,
        long_short: LongShort::LONG,
        underlying_price: curr_quote,
        current_price: Quote::new(0.0),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        vol_surface,
        maturity_date: future_date,
        discount_curve,
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        valuation_date,
        multiplier: 1.0,
    };
    println!("{:?}", option.time_to_maturity());
    let payoff = Box::new(VanillaPayoff {
        put_or_call: side,
        exercise_style: crate::core::utils::ContractStyle::European,
    });
    let equityoption = EquityOption {
        base: option,
        payoff: payoff,
        engine: Engine::MonteCarlo,
        mc: MonteCarloConfig::default(),
    };

    println!("Theoretical Price ${}", equityoption.npv());
    let mut wait = String::new();
    io::stdin()
        .read_line(&mut wait)
        .expect("Failed to read line");
}
