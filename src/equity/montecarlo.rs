//! Monte Carlo pricing engine.
//!
//! - Terminal-value simulation (exact GBM step, 1-D Sobol) for European
//!   payoffs; path-wise simulation with Exact / Euler / Milstein stepping.
//! - **Parallel, streamed path generation**: every path derives its own
//!   deterministic RNG stream from (seed, path index), so paths are
//!   generated in parallel with rayon, results are independent of thread
//!   scheduling, and no draw matrix is materialized.
//! - **Multi-dimensional quasi-Monte Carlo**: with the (default) `Sobol`
//!   sampler, path-wise routes use a low-discrepancy sequence through a
//!   **Brownian bridge**, so the best coordinates carry each path's coarse
//!   structure.
//! - Dupire local vol dynamics, Brownian-bridge barrier correction,
//!   geometric control variate for arithmetic Asians.
//! - American exercise via **two-pass Longstaff-Schwartz** (regression on
//!   one set of paths, valuation on an independent set — removes foresight
//!   bias) with a cubic polynomial basis.
//! - [`npv_with_stats`] reports the standard error alongside the price.
//! - Greeks by central-difference bump-and-reprice with common random
//!   numbers (deterministic draws make every reprice use identical paths).
//!
//! Multi-factor models (stochastic vol) plug in by widening the per-step
//! draw dimension and adding a second bridge; the per-path stream and
//! stepping structure is factor-agnostic.

use std::io;
use std::str::FromStr;
use chrono::{Local, NaiveDate};
use libm::exp;
use rayon::prelude::*;

use crate::core::utils::ContractStyle;
use super::asian::{self, AsianStrikeType, AveragingType};
use super::autocallable::AutocallablePayoff;
use super::barrier::{BarrierDirection, KnockType};
use super::heston::HestonParams;
use super::local_vol::LocalVol;
use super::vanila_option::{AsianPayoff, BarrierPayoff, EquityOption, EquityOptionBase, VanillaPayoff};
use super::utils::{Engine, LongShort, Payoff};
use crate::core::trade::PutOrCall;
use crate::utils::RNG::{
    path_normals, pseudo_normals, sobol_normals, BrownianBridge, QmcSequence,
};
use crate::core::quotes::Quote;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::data_models::EquityOptionData;
use crate::core::vols::VolSurface;
use crate::core::traits::Instrument;

/// Time-stepping scheme for path-wise simulation.
/// `Exact` samples the closed-form GBM transition (no discretization bias);
/// Euler and Milstein are the standard approximate schemes (the basis for
/// models without closed-form transitions, e.g. local vol / Heston).
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

/// Draw sampler. `Sobol` selects the low-discrepancy family: true Sobol
/// (van der Corput) in one dimension, a scrambled multi-dimensional
/// sequence through a Brownian bridge for path-wise simulation.
/// `PseudoRandom` uses seeded per-path PCG64 streams with antithetic
/// pairing. Longstaff-Schwarz always uses pseudo-random streams.
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
    Heston,
}

impl FromStr for McModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "gbm" | "blackscholes" | "bs" => Ok(McModel::Gbm),
            "local_vol" | "localvol" | "lv" => Ok(McModel::LocalVol),
            "heston" => Ok(McModel::Heston),
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
pub const HESTON_MIN_STEPS: usize = 250;
/// Minimum monitoring steps for path-dependent payoffs.
pub const PATH_DEPENDENT_MIN_STEPS: usize = 100;

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

/// Price with sampling diagnostics.
///
/// `std_err` is the standard error of the mean over paths. For the
/// low-discrepancy sampler the points are not independent, so treat it as
/// an indicative scale rather than a rigorous confidence bound; for the
/// LSMC it reflects valuation-pass noise only (not regression uncertainty).
#[derive(Debug, Clone, Copy)]
pub struct McStats {
    pub pv: f64,
    pub std_err: f64,
    pub paths: usize,
    pub steps: usize,
}

fn stats(sum: f64, sum_sq: f64, n: usize, steps: usize, offset: f64) -> McStats {
    let nf = n as f64;
    let mean = sum / nf;
    let var = (sum_sq / nf - mean * mean).max(0.0);
    McStats { pv: mean + offset, std_err: (var / nf).sqrt(), paths: n, steps }
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
        q: option.base.carry_yield(),
        sigma: option.base.volatility(),
        t: option.time_to_maturity(),
    }
}

/// Cash dividend amounts bucketed per simulation step (None if there are
/// none): path simulation subtracts them at the ex-date step.
fn dividends_per_step(option: &EquityOption, t: f64, steps: usize) -> Option<Vec<f64>> {
    if option.base.cash_dividends.is_empty() {
        return None;
    }
    let dt = t / steps as f64;
    let mut buckets = vec![0.0; steps];
    for (date, amount) in &option.base.cash_dividends {
        let td = (*date - option.base.valuation_date).num_days() as f64 / 365.0;
        if td > 0.0 && td <= t {
            let idx = (((td / dt).ceil() as usize).max(1) - 1).min(steps - 1);
            buckets[idx] += amount;
        }
    }
    Some(buckets)
}

/// Escrowed-model spot consistent with the bumped market params: rho bumps
/// shift the dividend discounting, delta bumps move the raw spot.
///
/// Cash dividends are discounted at the net carry `r - carry` (here
/// `p.r - p.q`, `p.q` being the total carry), matching the analytic engine
/// and the jump-model forward; see
/// [`EquityOptionBase::pv_cash_dividends`](super::vanila_option::EquityOptionBase::pv_cash_dividends).
fn escrowed_spot(option: &EquityOption, p: &MarketParams) -> f64 {
    let dr = p.r - option.base.risk_free_rate();
    let mut pv = 0.0;
    for (date, amount) in &option.base.cash_dividends {
        let td = (*date - option.base.valuation_date).num_days() as f64 / 365.0;
        if td > 0.0 && td <= p.t {
            // df(td) e^{-dr td} discounts at the bumped rate p.r;
            // e^{p.q td} moves it to the net carry (p.r - p.q).
            pv += amount * option.base.discount_curve.df(td) * ((p.q - dr) * td).exp();
        }
    }
    p.s0 - pv
}

pub fn npv(option: &EquityOption) -> f64 {
    npv_with_stats(option).pv
}

/// Price with standard error and simulation diagnostics.
pub fn npv_with_stats(option: &EquityOption) -> McStats {
    assert!(option.base.volatility() >= 0.0);
    assert!(option.base.time_to_maturity() >= 0.0);
    assert!(option.base.underlying_price.value >= 0.0);
    price(option, &market_params(option))
}

fn price(option: &EquityOption, p: &MarketParams) -> McStats {
    match option.payoff.exercise_style() {
        ContractStyle::American => american_npv(option, p),
        _ => european_npv(option, p),
    }
}

// ── Greeks: central-difference bumps with common random numbers ─────────

pub fn delta(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = p.s0 * 0.01;
    (price(option, &MarketParams { s0: p.s0 + h, ..p }).pv
        - price(option, &MarketParams { s0: p.s0 - h, ..p }).pv)
        / (2.0 * h)
}

pub fn gamma(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = p.s0 * 0.01;
    (price(option, &MarketParams { s0: p.s0 + h, ..p }).pv - 2.0 * price(option, &p).pv
        + price(option, &MarketParams { s0: p.s0 - h, ..p }).pv)
        / (h * h)
}

pub fn vega(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = 0.01;
    (price(option, &MarketParams { sigma: p.sigma + h, ..p }).pv
        - price(option, &MarketParams { sigma: p.sigma - h, ..p }).pv)
        / (2.0 * h)
}

pub fn theta(option: &EquityOption) -> f64 {
    // theta = dV/dt (calendar) = -dV/dT
    let p = market_params(option);
    let h = (1.0 / 365.0_f64).min(0.5 * p.t);
    -(price(option, &MarketParams { t: p.t + h, ..p }).pv
        - price(option, &MarketParams { t: p.t - h, ..p }).pv)
        / (2.0 * h)
}

pub fn rho(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let h = 1e-4;
    (price(option, &MarketParams { r: p.r + h, ..p }).pv
        - price(option, &MarketParams { r: p.r - h, ..p }).pv)
        / (2.0 * h)
}

/// Vanna, estimated with common-random-number mixed spot/volatility bumps.
pub fn vanna(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let hs = p.s0 * 0.01;
    let hv = 0.01;
    (price(option, &MarketParams { s0: p.s0 + hs, sigma: p.sigma + hv, ..p }).pv
        - price(option, &MarketParams { s0: p.s0 - hs, sigma: p.sigma + hv, ..p }).pv
        - price(option, &MarketParams { s0: p.s0 + hs, sigma: p.sigma - hv, ..p }).pv
        + price(option, &MarketParams { s0: p.s0 - hs, sigma: p.sigma - hv, ..p }).pv)
        / (4.0 * hs * hv)
}

/// Charm, the calendar-time change in delta, estimated by mixed bumps.
pub fn charm(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let hs = p.s0 * 0.01;
    let ht = (1.0 / 365.0_f64).min(0.5 * p.t);
    -(price(option, &MarketParams { s0: p.s0 + hs, t: p.t + ht, ..p }).pv
        - price(option, &MarketParams { s0: p.s0 - hs, t: p.t + ht, ..p }).pv
        - price(option, &MarketParams { s0: p.s0 + hs, t: p.t - ht, ..p }).pv
        + price(option, &MarketParams { s0: p.s0 - hs, t: p.t - ht, ..p }).pv)
        / (4.0 * hs * ht)
}

/// Zomma, estimated by applying a volatility bump to the common-random-
/// number gamma estimate.
pub fn zomma(option: &EquityOption) -> f64 {
    let p = market_params(option);
    let hs = p.s0 * 0.01;
    let hv = 0.01;
    let gamma_at_vol = |sigma: f64| {
        (price(option, &MarketParams { s0: p.s0 + hs, sigma, ..p }).pv
            - 2.0 * price(option, &MarketParams { sigma, ..p }).pv
            + price(option, &MarketParams { s0: p.s0 - hs, sigma, ..p }).pv)
            / (hs * hs)
    };
    (gamma_at_vol(p.sigma + hv) - gamma_at_vol(p.sigma - hv)) / (2.0 * hv)
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
            option.base.carry_yield(),
            // vega bumps enter as a parallel shift of the implied surface
            p.sigma - option.base.volatility(),
        )),
        McModel::Heston => unreachable!("Heston paths are generated by the dedicated routes"),
    }
}

fn effective_steps(cfg: &MonteCarloConfig) -> usize {
    match cfg.model {
        McModel::LocalVol => cfg.time_steps.max(LOCAL_VOL_MIN_STEPS),
        McModel::Heston => cfg.time_steps.max(HESTON_MIN_STEPS),
        McModel::Gbm => cfg.time_steps,
    }
}

// ── Per-path Brownian increments ────────────────────────────────────────

/// Deterministic per-path Brownian increment source. Pseudo-random paths
/// come in antithetic pairs (2k, 2k+1) from independent per-pair streams;
/// low-discrepancy paths are sequence points routed through the Brownian
/// bridge.
enum PathDraws {
    Pseudo { seed: u64, sqrt_dt: f64 },
    Qmc { seq: QmcSequence, bridge: BrownianBridge },
}

impl PathDraws {
    fn new(cfg: &MonteCarloConfig, steps: usize, dt: f64) -> Self {
        match cfg.sampler {
            Sampler::Sobol => PathDraws::Qmc {
                seq: QmcSequence::new(steps, cfg.seed),
                bridge: BrownianBridge::new(steps, dt),
            },
            Sampler::PseudoRandom => PathDraws::Pseudo { seed: cfg.seed, sqrt_dt: dt.sqrt() },
        }
    }

    fn pseudo(seed: u64, dt: f64) -> Self {
        PathDraws::Pseudo { seed, sqrt_dt: dt.sqrt() }
    }

    /// Fill `dw` with the Brownian increments of path `index`.
    fn fill(&self, index: usize, z: &mut [f64], w: &mut [f64], dw: &mut [f64]) {
        match self {
            PathDraws::Pseudo { seed, sqrt_dt } => {
                path_normals(*seed, (index / 2) as u64, z);
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                for (d, zi) in dw.iter_mut().zip(z.iter()) {
                    *d = sign * sqrt_dt * zi;
                }
            }
            PathDraws::Qmc { seq, bridge } => {
                seq.normals(index as u64 + 1, z);
                bridge.increments(z, w, dw);
            }
        }
    }
}

fn step(scheme: DiscretizationScheme, s: f64, dt: f64, dw: f64, drift: f64, sigma: f64) -> f64 {
    let next = match scheme {
        DiscretizationScheme::Exact => s * exp((drift - 0.5 * sigma * sigma) * dt + sigma * dw),
        DiscretizationScheme::Euler => s * (1.0 + drift * dt + sigma * dw),
        DiscretizationScheme::Milstein => {
            s * (1.0 + drift * dt + sigma * dw + 0.5 * sigma * sigma * (dw * dw - dt))
        }
    };
    next.max(0.0)
}

/// Paths per parallel work unit. Each chunk is summed serially in index
/// order and chunk results are folded in order, so totals are bit-exact
/// reproducible regardless of thread scheduling.
const PATH_CHUNK: usize = 4096;

/// Parallel map-reduce over paths: `eval(dw, scratch)` values one path from
/// its Brownian increments; returns (sum, sum of squares) deterministically.
fn run_paths<F>(paths: usize, steps: usize, draws: &PathDraws, eval: F) -> (f64, f64)
where
    F: Fn(&[f64], &mut Vec<f64>) -> f64 + Sync,
{
    let chunks = paths.div_ceil(PATH_CHUNK);
    let partials: Vec<(f64, f64)> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut z = vec![0.0; steps];
            let mut w = vec![0.0; steps];
            let mut dw = vec![0.0; steps];
            let mut scratch = Vec::new();
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for i in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(paths) {
                draws.fill(i, &mut z, &mut w, &mut dw);
                let v = eval(&dw, &mut scratch);
                sum += v;
                sum_sq += v * v;
            }
            (sum, sum_sq)
        })
        .collect();
    partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

// ── European ────────────────────────────────────────────────────────────

fn european_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    if option.mc.model == McModel::Heston {
        return heston_european_npv(option, p);
    }
    if option.payoff.is_path_dependent() {
        // barriers get the Brownian-bridge crossing correction; Asians get
        // the geometric control variate; anything else path-dependent uses
        // its own path_payoff with discrete monitoring
        return if let Some(barrier) = option.payoff.as_any().downcast_ref::<BarrierPayoff>() {
            barrier_npv(option, barrier, p)
        } else if let Some(asian) = option.payoff.as_any().downcast_ref::<AsianPayoff>() {
            asian_npv(option, asian, p)
        } else if let Some(auto) = option.payoff.as_any().downcast_ref::<AutocallablePayoff>() {
            autocall_npv(option, auto, p)
        } else {
            generic_path_npv(option, p)
        };
    }
    let cfg = &option.mc;
    let steps = effective_steps(cfg);
    let df = exp(-p.r * p.t);
    if steps <= 1 {
        // exact one-step GBM transition (constant vol only)
        let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * p.t;
        let vol_sqrt_t = p.sigma * p.t.sqrt();
        let s0 = escrowed_spot(option, p);
        let z = match cfg.sampler {
            Sampler::Sobol => sobol_normals(cfg.paths),
            Sampler::PseudoRandom => pseudo_normals(cfg.paths, cfg.seed),
        };
        let partials: Vec<(f64, f64)> = z
            .par_chunks(PATH_CHUNK)
            .map(|chunk| {
                let (mut sum, mut sum_sq) = (0.0, 0.0);
                for z in chunk {
                    let v = df
                        * option.payoff.payoff(s0 * exp(drift + vol_sqrt_t * z), p.strike);
                    sum += v;
                    sum_sq += v * v;
                }
                (sum, sum_sq)
            })
            .collect();
        let (sum, sum_sq) =
            partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        return stats(sum, sum_sq, cfg.paths, 1, 0.0);
    }
    let dt = p.t / steps as f64;
    let vol_model = path_vol(option, p);
    let draws = PathDraws::new(cfg, steps, dt);
    let divs = dividends_per_step(option, p.t, steps);
    let drift = p.r - p.q;
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, _| {
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            let sigma = vol_model.vol(s, i as f64 * dt);
            s = step(cfg.scheme, s, dt, *d, drift, sigma);
            if let Some(divs) = &divs {
                s = (s - divs[i]).max(1e-8);
            }
        }
        df * option.payoff.payoff(s, p.strike)
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

/// Path-dependent pricing through [`Payoff::path_payoff`] on discretely
/// monitored paths.
fn generic_path_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let cfg = &option.mc;
    let steps = effective_steps(cfg).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let df = exp(-p.r * p.t);
    let vol_model = path_vol(option, p);
    let draws = PathDraws::new(cfg, steps, dt);
    let drift = p.r - p.q;
    let divs = dividends_per_step(option, p.t, steps);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, path| {
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            let sigma = vol_model.vol(s, i as f64 * dt);
            s = step(cfg.scheme, s, dt, *d, drift, sigma);
            if let Some(divs) = &divs {
                s = (s - divs[i]).max(1e-8);
            }
            path.push(s);
        }
        df * option.payoff.path_payoff(path, p.strike)
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

/// Asian pricing. Arithmetic fixed-strike Asians under plain GBM use the
/// geometric average as a control variate: the same paths price both
/// averages, the closed-form discrete geometric value corrects the
/// difference, and the variance collapses because the two payoffs are
/// highly correlated. Every other combination (geometric, floating strike,
/// local vol, approximate schemes) prices through the generic path route.
fn asian_npv(option: &EquityOption, asian: &AsianPayoff, p: &MarketParams) -> McStats {
    let cfg = &option.mc;
    let use_control_variate = asian.averaging == AveragingType::Arithmetic
        && asian.strike_type == AsianStrikeType::FixedStrike
        && cfg.model == McModel::Gbm
        && cfg.scheme == DiscretizationScheme::Exact
        && option.base.cash_dividends.is_empty();
    if !use_control_variate {
        return generic_path_npv(option, p);
    }
    let steps = effective_steps(cfg).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let drift_dt = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let df = exp(-p.r * p.t);
    let draws = PathDraws::new(cfg, steps, dt);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, _| {
        let mut s = p.s0;
        let mut sum_s = 0.0;
        let mut log_sum = 0.0;
        for d in dw {
            s *= exp(drift_dt + p.sigma * d);
            sum_s += s;
            log_sum += s.ln();
        }
        let arithmetic = sum_s / steps as f64;
        let geometric = (log_sum / steps as f64).exp();
        df * (option.payoff.payoff(arithmetic, p.strike)
            - option.payoff.payoff(geometric, p.strike))
    });
    let geo_closed = asian::geometric_asian_price(
        p.s0,
        p.strike,
        p.r,
        p.q,
        p.sigma,
        p.t,
        Some(steps),
        *option.payoff.put_or_call(),
    );
    stats(sum, sum_sq, cfg.paths, steps, geo_closed)
}

/// Barrier pricing with a Brownian-bridge crossing correction: each path
/// carries a survival probability that accounts for the chance of touching
/// the barrier *between* monitoring points, removing the O(sqrt(dt))
/// discrete-monitoring bias and reducing variance (conditional Monte Carlo).
fn barrier_npv(option: &EquityOption, barrier: &BarrierPayoff, p: &MarketParams) -> McStats {
    let cfg = &option.mc;
    let steps = effective_steps(cfg).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let down = barrier.direction == BarrierDirection::Down;
    let out = barrier.knock == KnockType::Out;
    let h = barrier.barrier;
    let knocked_at_start = if down { p.s0 <= h } else { p.s0 >= h };
    if knocked_at_start && out {
        return McStats { pv: 0.0, std_err: 0.0, paths: cfg.paths, steps };
    }
    let df = exp(-p.r * p.t);
    let vol_model = path_vol(option, p);
    let draws = PathDraws::new(cfg, steps, dt);
    let drift = p.r - p.q;
    let divs = dividends_per_step(option, p.t, steps);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, _| {
        let mut s = p.s0;
        let mut survival = if knocked_at_start { 0.0 } else { 1.0 };
        for (i, d) in dw.iter().enumerate() {
            let sigma = vol_model.vol(s, i as f64 * dt);
            let mut s_next = step(cfg.scheme, s, dt, *d, drift, sigma);
            if let Some(divs) = &divs {
                s_next = (s_next - divs[i]).max(1e-8);
            }
            if survival > 0.0 {
                let crossed = if down { s_next <= h } else { s_next >= h };
                if crossed {
                    survival = 0.0;
                } else {
                    // probability the bridge touched the barrier inside the step
                    let (a, b) = if down {
                        ((s / h).ln(), (s_next / h).ln())
                    } else {
                        ((h / s).ln(), (h / s_next).ln())
                    };
                    survival *= 1.0 - (-2.0 * a * b / (sigma * sigma * dt)).exp();
                }
            }
            s = s_next;
        }
        let vanilla_leg = option.payoff.payoff(s, p.strike);
        let weight = if out { survival } else { 1.0 - survival };
        df * weight * vanilla_leg
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

/// Autocallable valuation: cash flows land on their own call dates, so
/// each path value is the redemption amount times the discount factor of
/// its payment date (curve discount factors, shifted consistently under
/// rho bumps). Steps are aligned so every observation falls exactly on a
/// simulation step. Runs under GBM and local vol.
fn autocall_npv(option: &EquityOption, auto: &AutocallablePayoff, p: &MarketParams) -> McStats {
    let cfg = &option.mc;
    let n_obs = auto.observations.max(1);
    let steps = effective_steps(cfg).max(PATH_DEPENDENT_MIN_STEPS).div_ceil(n_obs) * n_obs;
    let dt = p.t / steps as f64;
    let obs_idx: Vec<usize> = (1..=n_obs).map(|m| m * steps / n_obs - 1).collect();
    let dr = p.r - option.base.risk_free_rate();
    let dfs: Vec<f64> = obs_idx
        .iter()
        .map(|&i| {
            let tm = (i + 1) as f64 * dt;
            option.base.discount_curve.df(tm) * exp(-dr * tm)
        })
        .collect();
    let divs = dividends_per_step(option, p.t, steps);
    let vol_model = path_vol(option, p);
    let draws = PathDraws::new(cfg, steps, dt);
    let drift = p.r - p.q;
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, path| {
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            let sigma = vol_model.vol(s, i as f64 * dt);
            s = step(cfg.scheme, s, dt, *d, drift, sigma);
            if let Some(divs) = &divs {
                s = (s - divs[i]).max(1e-8);
            }
            path.push(s);
        }
        auto.path_value(path, &obs_idx, &dfs)
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

// ── Heston stochastic volatility paths ──────────────────────────────────

/// Full-truncation Euler simulation of the Heston model (two correlated
/// normals per step, seeded per-path pseudo-random streams; the Andersen QE
/// scheme is the planned upgrade). Vega bumps map to a parallel shift of
/// the instantaneous and long-run vol.
fn heston_european_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let hp = option
        .heston
        .expect("heston parameters are required when mc_model = heston")
        .with_vol_shift(p.sigma - option.base.volatility());
    let cfg = &option.mc;
    let steps = effective_steps(cfg);
    let dt = p.t / steps as f64;
    let df = exp(-p.r * p.t);

    if let Some(barrier) = option.payoff.as_any().downcast_ref::<BarrierPayoff>() {
        let down = barrier.direction == BarrierDirection::Down;
        let out = barrier.knock == KnockType::Out;
        let h = barrier.barrier;
        let knocked_at_start = if down { p.s0 <= h } else { p.s0 >= h };
        if knocked_at_start && out {
            return McStats { pv: 0.0, std_err: 0.0, paths: cfg.paths, steps };
        }
        let (sum, sum_sq) = run_heston_paths(option, p, &hp, steps, dt, |spots, vols| {
            let mut survival = if knocked_at_start { 0.0 } else { 1.0 };
            let mut s_prev = p.s0;
            for (i, &s_next) in spots.iter().enumerate() {
                if survival > 0.0 {
                    let crossed = if down { s_next <= h } else { s_next >= h };
                    if crossed {
                        survival = 0.0;
                    } else {
                        let (a, b) = if down {
                            ((s_prev / h).ln(), (s_next / h).ln())
                        } else {
                            ((h / s_prev).ln(), (h / s_next).ln())
                        };
                        let sigma = vols[i].max(1e-8);
                        survival *= 1.0 - (-2.0 * a * b / (sigma * sigma * dt)).exp();
                    }
                }
                s_prev = s_next;
            }
            let weight = if out { survival } else { 1.0 - survival };
            df * weight * option.payoff.payoff(s_prev, p.strike)
        });
        return stats(sum, sum_sq, cfg.paths, steps, 0.0);
    }

    if let Some(auto) = option.payoff.as_any().downcast_ref::<AutocallablePayoff>() {
        let n_obs = auto.observations.max(1);
        let steps = steps.div_ceil(n_obs) * n_obs;
        let dt = p.t / steps as f64;
        let obs_idx: Vec<usize> = (1..=n_obs).map(|m| m * steps / n_obs - 1).collect();
        let dr = p.r - option.base.risk_free_rate();
        let dfs: Vec<f64> = obs_idx
            .iter()
            .map(|&i| {
                let tm = (i + 1) as f64 * dt;
                option.base.discount_curve.df(tm) * exp(-dr * tm)
            })
            .collect();
        let (sum, sum_sq) = run_heston_paths(option, p, &hp, steps, dt, |spots, _| {
            auto.path_value(spots, &obs_idx, &dfs)
        });
        return stats(sum, sum_sq, cfg.paths, steps, 0.0);
    }

    let path_dependent = option.payoff.is_path_dependent();
    let (sum, sum_sq) = run_heston_paths(option, p, &hp, steps, dt, |spots, _| {
        let v = if path_dependent {
            option.payoff.path_payoff(spots, p.strike)
        } else {
            option.payoff.payoff(*spots.last().unwrap(), p.strike)
        };
        df * v
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

/// Parallel Heston path generation: `eval(spots, vols)` receives the path's
/// spot levels and the per-step vols (`sqrt(v)`) actually used to diffuse.
fn run_heston_paths<F>(
    option: &EquityOption,
    p: &MarketParams,
    hp: &HestonParams,
    steps: usize,
    dt: f64,
    eval: F,
) -> (f64, f64)
where
    F: Fn(&[f64], &[f64]) -> f64 + Sync,
{
    let cfg = &option.mc;
    let drift = p.r - p.q;
    let sqrt_dt = dt.sqrt();
    let rho = hp.rho;
    let rho_perp = (1.0 - rho * rho).sqrt();
    let divs = dividends_per_step(option, p.t, steps);
    let chunks = cfg.paths.div_ceil(PATH_CHUNK);
    let partials: Vec<(f64, f64)> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut z = vec![0.0; 2 * steps];
            let mut spots = vec![0.0; steps];
            let mut vols = vec![0.0; steps];
            let (mut sum, mut sum_sq) = (0.0, 0.0);
            for i in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                // antithetic pairs share a stream with negated draws
                path_normals(cfg.seed, (i / 2) as u64, &mut z);
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                let mut s = p.s0;
                let mut v = hp.v0;
                for j in 0..steps {
                    let z_s = sign * z[2 * j];
                    let z_v = rho * z_s + rho_perp * sign * z[2 * j + 1];
                    let v_pos = v.max(0.0);
                    let sqrt_v = v_pos.sqrt();
                    s *= exp((drift - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z_s);
                    if let Some(divs) = &divs {
                        s = (s - divs[j]).max(1e-8);
                    }
                    v += hp.kappa * (hp.theta - v_pos) * dt
                        + hp.vol_of_vol * sqrt_v * sqrt_dt * z_v;
                    spots[j] = s;
                    vols[j] = sqrt_v;
                }
                let value = eval(&spots, &vols);
                sum += value;
                sum_sq += value * value;
            }
            (sum, sum_sq)
        })
        .collect();
    partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1))
}

// ── American: two-pass Longstaff-Schwartz ───────────────────────────────

const LSMC_DEFAULT_STEPS: usize = 50;
const LSMC_BASIS: usize = 4;

/// Basis functions for the continuation-value regression: cubic in the
/// normalized spot. (An "include the payoff" basis is exactly collinear
/// with `[1, x]` for vanilla payoffs on in-the-money paths, so the cubic
/// term is the safe way to add flexibility.)
fn lsmc_basis(x: f64) -> [f64; LSMC_BASIS] {
    [1.0, x, x * x, x * x * x]
}

/// Two-pass least-squares Monte Carlo (Longstaff-Schwartz):
/// pass 1 fits the per-date continuation-value regressions on one set of
/// paths; pass 2 applies the fitted exercise rule to an independent set,
/// which removes the foresight (in-sample) bias of single-pass LSMC.
/// Always uses pseudo-random per-path streams.
fn american_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let cfg = &option.mc;
    if cfg.model == McModel::Heston {
        panic!("American exercise under the Heston model is not supported yet");
    }
    let steps = if cfg.time_steps > 1 { cfg.time_steps } else { LSMC_DEFAULT_STEPS }
        .max(if cfg.model == McModel::LocalVol { LOCAL_VOL_MIN_STEPS } else { 1 });
    let dt = p.t / steps as f64;
    let disc = exp(-p.r * dt);
    let vol_model = path_vol(option, p);
    let drift = p.r - p.q;
    let seed_regression = cfg.seed ^ 0xA11C_E5ED;
    let seed_valuation = cfg.seed ^ 0xB0B5_1EED;

    let simulate = |draws: &PathDraws, index: usize, bufs: &mut (Vec<f64>, Vec<f64>, Vec<f64>), path: &mut Vec<f64>| {
        let (z, w, dw) = bufs;
        draws.fill(index, z, w, dw);
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            let sigma = vol_model.vol(s, i as f64 * dt);
            s = step(cfg.scheme, s, dt, *d, drift, sigma);
            path.push(s);
        }
    };

    // ── pass 1: simulate and fit regressions backwards
    let reg_draws = PathDraws::pseudo(seed_regression, dt);
    let spots: Vec<Vec<f64>> = (0..cfg.paths)
        .into_par_iter()
        .map_init(
            || (vec![0.0; steps], vec![0.0; steps], vec![0.0; steps]),
            |bufs, i| {
                let mut path = Vec::with_capacity(steps);
                simulate(&reg_draws, i, bufs, &mut path);
                path
            },
        )
        .collect();

    let mut cashflow: Vec<f64> =
        spots.iter().map(|path| option.payoff.payoff(path[steps - 1], p.strike)).collect();
    let mut betas: Vec<Option<[f64; LSMC_BASIS]>> = vec![None; steps.saturating_sub(1)];
    for step_idx in (0..steps - 1).rev() {
        for cf in cashflow.iter_mut() {
            *cf *= disc;
        }
        let itm: Vec<usize> = (0..spots.len())
            .filter(|&i| option.payoff.payoff(spots[i][step_idx], p.strike) > 0.0)
            .collect();
        if itm.len() < LSMC_BASIS {
            continue;
        }
        let rows: Vec<([f64; LSMC_BASIS], f64)> = itm
            .iter()
            .map(|&i| {
                let s = spots[i][step_idx];
                let pay = option.payoff.payoff(s, p.strike);
                (lsmc_basis(s / p.s0), cashflow[i])
            })
            .collect();
        let Some(beta) = least_squares(&rows) else { continue };
        for &i in &itm {
            let s = spots[i][step_idx];
            let pay = option.payoff.payoff(s, p.strike);
            let continuation = dot(&beta, &lsmc_basis(s / p.s0));
            if pay > continuation {
                cashflow[i] = pay;
            }
        }
        betas[step_idx] = Some(beta);
    }
    drop(spots);
    drop(cashflow);

    // ── pass 2: apply the fitted exercise rule to independent paths
    let val_draws = PathDraws::pseudo(seed_valuation, dt);
    let partials: Vec<(f64, f64)> = (0..cfg.paths.div_ceil(PATH_CHUNK))
        .into_par_iter()
        .map(|chunk| {
            let mut bufs = (vec![0.0; steps], vec![0.0; steps], vec![0.0; steps]);
            let mut path = Vec::with_capacity(steps);
            let (mut c_sum, mut c_sum_sq) = (0.0, 0.0);
            for i in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                simulate(&val_draws, i, &mut bufs, &mut path);
                let mut value = 0.0;
                let mut exercised = false;
                for k in 0..steps - 1 {
                    let s = path[k];
                    let pay = option.payoff.payoff(s, p.strike);
                    if pay > 0.0 {
                        if let Some(beta) = &betas[k] {
                            let continuation = dot(beta, &lsmc_basis(s / p.s0));
                            if pay > continuation {
                                value = pay * disc.powi(k as i32 + 1);
                                exercised = true;
                                break;
                            }
                        }
                    }
                }
                if !exercised {
                    value = option.payoff.payoff(path[steps - 1], p.strike)
                        * disc.powi(steps as i32);
                }
                c_sum += value;
                c_sum_sq += value * value;
            }
            (c_sum, c_sum_sq)
        })
        .collect();
    let (sum, sum_sq) = partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

fn dot(a: &[f64; LSMC_BASIS], b: &[f64; LSMC_BASIS]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Least squares via the normal equations with partial-pivot Gaussian
/// elimination; None if (near-)singular.
fn least_squares(rows: &[([f64; LSMC_BASIS], f64)]) -> Option<[f64; LSMC_BASIS]> {
    let k = LSMC_BASIS;
    let mut m = [[0.0; LSMC_BASIS + 1]; LSMC_BASIS];
    for (basis, y) in rows {
        for i in 0..k {
            for j in 0..k {
                m[i][j] += basis[i] * basis[j];
            }
            m[i][k] += basis[i] * y;
        }
    }
    for col in 0..k {
        let pivot =
            (col..k).max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())?;
        if m[pivot][col].abs() < 1e-10 {
            return None;
        }
        m.swap(col, pivot);
        for row in col + 1..k {
            let f = m[row][col] / m[col][col];
            for c in col..=k {
                m[row][c] -= f * m[col][c];
            }
        }
    }
    let mut beta = [0.0; LSMC_BASIS];
    for row in (0..k).rev() {
        let mut acc = m[row][k];
        for c in row + 1..k {
            acc -= m[row][c] * beta[c];
        }
        beta[row] = acc / m[row][row];
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
        borrow_cost: 0.0,
        cash_dividends: vec![],
        futures_settlement: None,
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
        fd: crate::equity::finite_difference::FdConfig::default(),
        heston: None,
    };

    let result = npv_with_stats(&equityoption);
    println!("Theoretical Price ${} (std err {})", result.pv, result.std_err);
    let mut wait = String::new();
    io::stdin()
        .read_line(&mut wait)
        .expect("Failed to read line");
}
