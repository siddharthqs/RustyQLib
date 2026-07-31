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
//!   bias) with a cubic polynomial basis; under Heston the paths and the
//!   regression basis carry the `(spot, variance)` state, stepping the
//!   Andersen QE scheme.
//! - [`npv_with_stats`] reports the standard error alongside the price.
//! - Greeks by central-difference bump-and-reprice with common random
//!   numbers (deterministic draws make every reprice use identical paths).
//!
//! Path dynamics come from the stochastic-process layer
//! ([`core::montecarlo::process`](crate::core::montecarlo::process) +
//! [`equity::processes`](crate::equity::processes)): the SDE's
//! drift/diffusion live in the process object (GBM / local vol as a
//! [`BlackScholesProcess`], Heston as the two-factor [`HestonProcess`]),
//! and Euler / Milstein / exact stepping are generic over it. A new model
//! plugs in by implementing the process trait; the per-path stream and
//! stepping structure is factor-agnostic.

use libm::exp;
use rayon::prelude::*;

use crate::core::utils::ContractStyle;
use super::asian::{self, AsianStrikeType, AveragingType};
use super::accumulator::AccumulatorPayoff;
use super::autocallable::AutocallablePayoff;
use super::barrier::{BarrierDirection, KnockType};
use super::heston::HestonParams;
use super::local_vol::LocalVol;
use super::processes::{BlackScholesProcess, HestonProcess, HestonScheme, VolDynamics};
use super::vanilla_option::{AsianPayoff, BarrierPayoff, EquityOption, VanillaPayoff};
use super::utils::Model;
use crate::core::montecarlo::process::{StochasticProcess, StochasticProcess1D};
use crate::core::trade::PutOrCall;
use crate::core::montecarlo::{path_normals, pseudo_normals, sobol_normals, PathDraws};
use crate::core::data_models::EquityOptionData;

/// Re-exported from the asset-agnostic process layer, where the schemes
/// are defined once against any SDE's drift/diffusion coefficients.
pub use crate::core::montecarlo::process::DiscretizationScheme;

/// Re-exported from the asset-agnostic path layer. Longstaff-Schwartz
/// always uses pseudo-random streams.
pub use crate::core::montecarlo::paths::Sampler;

/// Dynamics used for path generation. `Gbm` diffuses at the option's own
/// (constant) implied vol; `LocalVol` diffuses at the Dupire local
/// volatility calibrated from the option's vol surface.


#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MonteCarloConfig {
    pub paths: usize,
    /// 1 = terminal simulation (exact); > 1 = path-wise stepping.
    /// Local vol always steps path-wise (at least [`LOCAL_VOL_MIN_STEPS`]).
    pub time_steps: usize,
    pub scheme: DiscretizationScheme,
    pub sampler: Sampler,
    pub seed: u64,
}

pub const LOCAL_VOL_MIN_STEPS: usize = 100;
/// Step floor for full-truncation Euler, whose O(dt) variance-truncation
/// bias needs a fine grid.
pub const HESTON_MIN_STEPS: usize = 250;
/// Step floor under Andersen QE, which is near bias-free on coarse grids
/// (that is its point) — the floor only keeps enough resolution for the
/// vol path itself.
pub const HESTON_QE_MIN_STEPS: usize = 25;
/// Minimum monitoring steps for path-dependent payoffs.
pub const PATH_DEPENDENT_MIN_STEPS: usize = 100;

impl Default for MonteCarloConfig {
    fn default() -> Self {
        MonteCarloConfig {
            paths: 100_000,
            time_steps: 1,
            scheme: DiscretizationScheme::Exact,
            sampler: Sampler::Sobol,
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
        MonteCarloConfig {
            paths: data.simulation.unwrap_or(defaults.paths as u64) as usize,
            time_steps: data.mc_time_steps.unwrap_or(default_steps),
            scheme,
            sampler: data
                .mc_sampler
                .as_deref()
                .map(|s| s.parse::<Sampler>().expect("Invalid mc_sampler"))
                .unwrap_or(defaults.sampler),
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
    let (mean, std_err) = crate::core::montecarlo::mean_std_err(sum, sum_sq, n);
    McStats { pv: mean + offset, std_err, paths: n, steps }
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
        s0: option.market.spot.value(),
        strike: option.base.strike_price,
        r: option.risk_free_rate(),
        q: option.carry_yield(),
        sigma: option.volatility(),
        t: option.time_to_maturity(),
    }
}

/// Cash dividend amounts bucketed per simulation step (None if there are
/// none): path simulation subtracts them at the ex-date step.
fn dividends_per_step(option: &EquityOption, t: f64, steps: usize) -> Option<Vec<f64>> {
    if option.market.cash_dividends.is_empty() {
        return None;
    }
    let dt = t / steps as f64;
    let mut buckets = vec![0.0; steps];
    for (date, amount) in &option.market.cash_dividends {
        let td = (*date - option.market.valuation_date).num_days() as f64 / 365.0;
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
/// [`EquityOptionBase::pv_cash_dividends`](super::vanilla_option::EquityOptionBase::pv_cash_dividends).
fn escrowed_spot(option: &EquityOption, p: &MarketParams) -> f64 {
    let dr = p.r - option.risk_free_rate();
    let mut pv = 0.0;
    for (date, amount) in &option.market.cash_dividends {
        let td = (*date - option.market.valuation_date).num_days() as f64 / 365.0;
        if td > 0.0 && td <= p.t {
            // df(td) e^{-dr td} discounts at the bumped rate p.r;
            // e^{p.q td} moves it to the net carry (p.r - p.q).
            pv += amount * option.market.discount_curve.df(td) * ((p.q - dr) * td).exp();
        }
    }
    p.s0 - pv
}

pub fn npv(option: &EquityOption) -> f64 {
    npv_with_stats(option).pv
}

/// Price with standard error and simulation diagnostics.
pub fn npv_with_stats(option: &EquityOption) -> McStats {
    assert!(option.volatility() >= 0.0);
    assert!(option.time_to_maturity() >= 0.0);
    assert!(option.market.spot.mid() >= 0.0);
    if let Some(barrier) = option.payoff.as_any().downcast_ref::<BarrierPayoff>() {
        assert!(
            !(barrier.rebate != 0.0 && barrier.rebate_at_hit),
            "at-hit rebates need the touch time: price on the Analytical engine              (Monte Carlo supports the at-expiry rebate convention)"
        );
    }
    price(option, &market_params(option))
}

fn price(option: &EquityOption, p: &MarketParams) -> McStats {
    match option.payoff.exercise_style() {
        ContractStyle::American | ContractStyle::Bermudan(_) => american_npv(option, p),
        _ => european_npv(option, p),
    }
}

// Greeks: central-difference bumps with common random numbers, produced
// by the central sensitivity engine (`crate::equity::greeks`) through
// [`npv_with`] — every bumped reprice reuses the base price's draws.
// Where the payoff is smooth enough, the pathwise estimator below
// replaces the delta/vega bumps.

/// Pathwise delta and vega for the exact terminal-GBM route: differentiate
/// the discounted payoff along each path instead of bumping,
///
/// ```text
/// dV/dS0    = e^{-rT} E[ payoff'(S_T) * S_T / S0 ]
/// dV/dsigma = e^{-rT} E[ payoff'(S_T) * S_T * (sqrt(T) Z - sigma T) ]
/// ```
///
/// (`S0` the escrowed spot, `payoff'` = ±indicator for a vanilla). One
/// simulation instead of four, no finite-difference bias, and the same
/// draws as the price, so the estimates are exactly reproducible.
///
/// Applies to European vanilla payoffs on constant-vol GBM with one-step
/// terminal simulation; returns `None` outside that scope (path-dependent,
/// multi-step, local vol, Heston, American), where the central bump
/// stencils take over. The kink at the strike has measure zero, so the
/// interchange of derivative and expectation is valid for vanillas.
pub(crate) fn pathwise_delta_vega(option: &EquityOption) -> Option<(f64, f64)> {
    let vanilla = option.payoff.as_any().downcast_ref::<VanillaPayoff>()?;
    if !matches!(vanilla.exercise_style, ContractStyle::European) {
        return None;
    }
    if option.model != Model::Gbm {
        return None;
    }
    let cfg = option.mc_cfg();
    if effective_steps(cfg, &option.model) > 1 {
        return None;
    }
    let p = market_params(option);
    let df = exp(-p.r * p.t);
    let drift = (p.r - p.q - 0.5 * p.sigma * p.sigma) * p.t;
    let sqrt_t = p.t.sqrt();
    let vol_sqrt_t = p.sigma * sqrt_t;
    let s0 = escrowed_spot(option, &p);
    let z = match cfg.sampler {
        Sampler::Sobol => sobol_normals(cfg.paths),
        Sampler::PseudoRandom => pseudo_normals(cfg.paths, cfg.seed),
    };
    let sign = match vanilla.put_or_call {
        PutOrCall::Call => 1.0,
        PutOrCall::Put => -1.0,
    };
    let partials: Vec<(f64, f64)> = z
        .par_chunks(PATH_CHUNK)
        .map(|chunk| {
            let (mut delta_sum, mut vega_sum) = (0.0, 0.0);
            for z in chunk {
                let s_t = s0 * exp(drift + vol_sqrt_t * z);
                let in_the_money = match vanilla.put_or_call {
                    PutOrCall::Call => s_t > p.strike,
                    PutOrCall::Put => s_t < p.strike,
                };
                if in_the_money {
                    delta_sum += sign * s_t / s0;
                    vega_sum += sign * s_t * (sqrt_t * z - p.sigma * p.t);
                }
            }
            (delta_sum, vega_sum)
        })
        .collect();
    let (delta_sum, vega_sum) =
        partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let n = cfg.paths as f64;
    Some((df * delta_sum / n, df * vega_sum / n))
}

/// First-order adjoint Greeks from **one** simulation.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AadGreeks {
    pub delta: f64,
    /// Sensitivity to a parallel implied-vol shift.
    pub vega: f64,
    pub rho: f64,
}

/// Delta, vega and rho from a backward adjoint sweep per path: the GBM
/// stepping and the payoff (via [`Payoff::path_payoff_var`]) are recorded
/// on an AAD tape with `(S0, sigma, r)` as inputs, and one reverse pass
/// per path yields all three sensitivities — the cost does not grow with
/// the number of Greeks. Holding the draws fixed these are the classical
/// pathwise estimators: unbiased for continuous payoffs, using exactly
/// the draws the price uses.
///
/// Theta is deliberately absent: differentiating with the Brownian path
/// held fixed while the time grid moves is not well-defined, so theta
/// stays with the bump stencils.
///
/// Scope: European exercise, GBM dynamics, exact stepping, and a payoff
/// that opts into AAD (vanilla, Asian, lookback, forward-start — the
/// continuous ones). Returns `None` otherwise; discontinuous payoffs
/// (barrier, binary, autocallable) never opt in because the
/// almost-everywhere derivative of their indicator is zero.
pub(crate) fn aad_greeks(option: &EquityOption) -> Option<AadGreeks> {
    use crate::core::aad::{Tape, Var};
    if !matches!(option.payoff.exercise_style(), ContractStyle::European) {
        return None;
    }
    if option.model != Model::Gbm {
        return None;
    }
    let cfg = option.mc_cfg();
    if cfg.scheme != DiscretizationScheme::Exact {
        return None;
    }
    let p = market_params(option);
    let steps = if option.payoff.is_path_dependent() {
        effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS)
    } else {
        effective_steps(cfg, &option.model).max(1)
    };
    // capability probe before spinning up the parallel loop
    {
        let tape = Tape::new();
        let probe: Vec<Var> = (0..steps.max(2)).map(|_| tape.var(p.s0)).collect();
        option.payoff.path_payoff_var(&probe, p.strike)?;
    }
    let dt = p.t / steps as f64;
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let divs = dividends_per_step(option, p.t, steps);
    let chunks = cfg.paths.div_ceil(PATH_CHUNK);
    let partials: Vec<(f64, f64, f64)> = (0..chunks)
        .into_par_iter()
        .map(|chunk| {
            let mut z = vec![0.0; steps];
            let mut w = vec![0.0; steps];
            let mut dw = vec![0.0; steps];
            let tape = Tape::new();
            let mut path_vars: Vec<Var> = Vec::with_capacity(steps);
            let (mut delta_sum, mut vega_sum, mut rho_sum) = (0.0, 0.0, 0.0);
            for i in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                draws.fill(i, &mut z, &mut w, &mut dw);
                tape.clear();
                path_vars.clear();
                let s0 = tape.var(p.s0);
                let sigma = tape.var(p.sigma);
                let r = tape.var(p.r);
                let mut s = s0;
                for (step_idx, dwi) in dw.iter().enumerate() {
                    // exact GBM step: s * exp((r - q - sigma^2/2) dt + sigma dW)
                    let exponent =
                        (r - p.q) * dt - sigma * sigma * (0.5 * dt) + sigma * *dwi;
                    s = s * exponent.exp();
                    if let Some(divs) = &divs {
                        if divs[step_idx] != 0.0 {
                            s = (s - divs[step_idx]).maxf(1e-8);
                        }
                    }
                    path_vars.push(s);
                }
                let payoff = option
                    .payoff
                    .path_payoff_var(&path_vars, p.strike)
                    .expect("the probe above guaranteed AAD support");
                let discounted = payoff * (-(r * p.t)).exp();
                let g = discounted.grad();
                delta_sum += g.wrt(s0);
                vega_sum += g.wrt(sigma);
                rho_sum += g.wrt(r);
            }
            (delta_sum, vega_sum, rho_sum)
        })
        .collect();
    let (delta_sum, vega_sum, rho_sum) = partials
        .into_iter()
        .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
    let n = cfg.paths as f64;
    Some(AadGreeks { delta: delta_sum / n, vega: vega_sum / n, rho: rho_sum / n })
}

/// Reprice under a shifted market (spot, parallel vol, rate, calendar time)
/// with the same random draws as the base price, for PnL attribution.
pub(crate) fn npv_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> f64 {
    let p = market_params(option);
    price(
        option,
        &MarketParams {
            s0: p.s0 + d_spot,
            sigma: p.sigma + d_vol,
            r: p.r + d_rate,
            t: (p.t - d_time).max(1e-6),
            ..p
        },
    )
    .pv
}

// ── Model dynamics as a stochastic process ──────────────────────────────

/// The option's dynamics under the (possibly bumped) market `p`, as a
/// [`BlackScholesProcess`] the generic stepping consumes.
fn bs_process<'a>(option: &'a EquityOption, p: &MarketParams) -> BlackScholesProcess<'a> {
    let vol = match option.model {
        Model::Gbm => VolDynamics::Const(p.sigma),
        Model::LocalVol => VolDynamics::Local(LocalVol::new(
            &option.market.vol_surface,
            &option.market.discount_curve,
            // the local vol function is frozen at the calibration spot;
            // spot bumps (delta/gamma) move the path start, not the model
            option.market.spot.value(),
            option.carry_yield(),
            // vega bumps enter as a parallel shift of the implied surface
            p.sigma - option.volatility(),
        )),
        Model::Heston(_) => unreachable!("Heston paths are generated by the dedicated routes"),
    };
    BlackScholesProcess::new(p.r - p.q, vol)
}

fn effective_steps(cfg: &MonteCarloConfig, model: &Model) -> usize {
    match model {
        Model::LocalVol => cfg.time_steps.max(LOCAL_VOL_MIN_STEPS),
        Model::Heston(_) => {
            // Exact selects the QE scheme (see `run_heston_paths`), which
            // tolerates far coarser grids than full-truncation Euler
            let floor = match cfg.scheme {
                DiscretizationScheme::Exact => HESTON_QE_MIN_STEPS,
                _ => HESTON_MIN_STEPS,
            };
            cfg.time_steps.max(floor)
        }
        Model::Gbm => cfg.time_steps,
    }
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
    if option.model .is_heston() {
        return heston_european_npv(option, p);
    }
    if option.payoff.is_path_dependent() {
        // barriers get the Brownian-bridge crossing correction; Asians get
        // the geometric control variate; anything else path-dependent uses
        // its own path_payoff with discrete monitoring
        return if let Some(barrier) = option
            .payoff
            .as_any()
            .downcast_ref::<BarrierPayoff>()
            .filter(|b| b.barrier2.is_none() && b.rebate == 0.0)
        {
            barrier_npv(option, barrier, p)
        } else if let Some(asian) = option.payoff.as_any().downcast_ref::<AsianPayoff>() {
            asian_npv(option, asian, p)
        } else if let Some(auto) = option.payoff.as_any().downcast_ref::<AutocallablePayoff>() {
            autocall_npv(option, auto, p)
        } else if let Some(accu) = option.payoff.as_any().downcast_ref::<AccumulatorPayoff>() {
            accumulator_npv(option, accu, p)
        } else {
            generic_path_npv(option, p)
        };
    }
    let cfg = option.mc_cfg();
    let steps = effective_steps(cfg, &option.model);
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
    let process = bs_process(option, p);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let divs = dividends_per_step(option, p.t, steps);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, _| {
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            s = process.evolve(cfg.scheme, i as f64 * dt, s, dt, *d);
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
    let cfg = option.mc_cfg();
    let steps = effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let df = exp(-p.r * p.t);
    let process = bs_process(option, p);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let divs = dividends_per_step(option, p.t, steps);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, path| {
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            s = process.evolve(cfg.scheme, i as f64 * dt, s, dt, *d);
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
    let cfg = option.mc_cfg();
    let use_control_variate = asian.averaging == AveragingType::Arithmetic
        && asian.strike_type == AsianStrikeType::FixedStrike
        && option.model == Model::Gbm
        && cfg.scheme == DiscretizationScheme::Exact
        && option.market.cash_dividends.is_empty();
    if !use_control_variate {
        return generic_path_npv(option, p);
    }
    let steps = effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let drift_dt = (p.r - p.q - 0.5 * p.sigma * p.sigma) * dt;
    let df = exp(-p.r * p.t);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
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
    let cfg = option.mc_cfg();
    let steps = effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS);
    let dt = p.t / steps as f64;
    let down = barrier.direction == BarrierDirection::Down;
    let out = barrier.knock == KnockType::Out;
    let h = barrier.barrier;
    let knocked_at_start = if down { p.s0 <= h } else { p.s0 >= h };
    if knocked_at_start && out {
        return McStats { pv: 0.0, std_err: 0.0, paths: cfg.paths, steps };
    }
    let df = exp(-p.r * p.t);
    let process = bs_process(option, p);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let divs = dividends_per_step(option, p.t, steps);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, _| {
        let mut s = p.s0;
        let mut survival = if knocked_at_start { 0.0 } else { 1.0 };
        for (i, d) in dw.iter().enumerate() {
            // one vol lookup serves both the step and the bridge
            // crossing probability below
            let sigma = process.vol(s, i as f64 * dt);
            let mut s_next = process.step_with_vol(cfg.scheme, i as f64 * dt, s, dt, *d, sigma);
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

/// Observation grid for an autocallable on a path of `steps` steps over
/// life `t`: per-observation path indices and discount factors. Explicit
/// `observation_times` (business-day adjusted call dates as year
/// fractions) map to the nearest grid step and discount at their exact
/// times; without them observations are equally spaced.
fn observation_grid(
    option: &EquityOption,
    n_obs: usize,
    observation_times: Option<&Vec<f64>>,
    t: f64,
    r: f64,
    steps: usize,
) -> (Vec<usize>, Vec<f64>) {
    let dr = r - option.risk_free_rate();
    let n_obs = n_obs.max(1);
    let (obs_idx, obs_times): (Vec<usize>, Vec<f64>) = match observation_times {
        Some(times) => {
            let mut idx = Vec::with_capacity(times.len());
            let mut prev: i64 = 0;
            for &tm in times {
                // nearest grid step, strictly increasing so no two
                // observations collapse onto one step
                let i = ((tm / t) * steps as f64).round().max(1.0) as i64;
                let i = i.max(prev + 1).min(steps as i64);
                idx.push(i as usize - 1);
                prev = i;
            }
            (idx, times.clone())
        }
        None => {
            let dt = t / steps as f64;
            let idx: Vec<usize> = (1..=n_obs).map(|m| m * steps / n_obs - 1).collect();
            let times = idx.iter().map(|&i| (i + 1) as f64 * dt).collect();
            (idx, times)
        }
    };
    let dfs = obs_times
        .iter()
        .map(|&tm| option.market.discount_curve.df(tm) * exp(-dr * tm))
        .collect();
    (obs_idx, dfs)
}

/// Autocallable valuation: cash flows land on their own call dates, so
/// each path value is the redemption amount times the discount factor of
/// its payment date (curve discount factors, shifted consistently under
/// rho bumps). Steps are aligned so every observation falls exactly on a
/// simulation step. Runs under GBM and local vol.
fn autocall_npv(option: &EquityOption, auto: &AutocallablePayoff, p: &MarketParams) -> McStats {
    let cfg = option.mc_cfg();
    let n_obs = auto.observations.max(1);
    let steps = effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS).div_ceil(n_obs) * n_obs;
    let dt = p.t / steps as f64;
    let (obs_idx, dfs) =
        observation_grid(option, n_obs, auto.observation_times.as_ref(), p.t, p.r, steps);
    let divs = dividends_per_step(option, p.t, steps);
    let process = bs_process(option, p);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, path| {
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            s = process.evolve(cfg.scheme, i as f64 * dt, s, dt, *d);
            if let Some(divs) = &divs {
                s = (s - divs[i]).max(1e-8);
            }
            path.push(s);
        }
        auto.path_value(path, &obs_idx, &dfs)
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

/// Accumulator valuation: daily accruals land on their own observation
/// dates (each discounted on the option's curve), with the knock-out
/// checked **discretely** at each observation — the contractual daily-
/// close convention. Steps are aligned so every observation falls exactly
/// on a simulation step. Runs under GBM and local vol; the Heston route
/// lives in `heston_european_npv`.
fn accumulator_npv(option: &EquityOption, accu: &AccumulatorPayoff, p: &MarketParams) -> McStats {
    let cfg = option.mc_cfg();
    let n_obs = accu.observations.max(1);
    let steps = effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS).div_ceil(n_obs) * n_obs;
    let dt = p.t / steps as f64;
    let (obs_idx, dfs) = observation_grid(option, n_obs, None, p.t, p.r, steps);
    let divs = dividends_per_step(option, p.t, steps);
    let process = bs_process(option, p);
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let (sum, sum_sq) = run_paths(cfg.paths, steps, &draws, |dw, path| {
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            s = process.evolve(cfg.scheme, i as f64 * dt, s, dt, *d);
            if let Some(divs) = &divs {
                s = (s - divs[i]).max(1e-8);
            }
            path.push(s);
        }
        accu.path_value(path, &obs_idx, &dfs, p.strike)
    });
    stats(sum, sum_sq, cfg.paths, steps, 0.0)
}

// ── Heston stochastic volatility paths ──────────────────────────────────

/// Heston simulation on seeded per-path pseudo-random streams. The
/// default (`Exact`) scheme selects Andersen QE with martingale
/// correction; `mc_scheme: euler`/`milstein` select full-truncation
/// Euler. Vega bumps map to a parallel shift of the instantaneous and
/// long-run vol.
fn heston_european_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let hp = option.heston_params().with_vol_shift(p.sigma - option.volatility());
    let cfg = option.mc_cfg();
    // same monitoring floor as the Black-Scholes path routes — under the
    // QE floor of 25 steps a path-dependent payoff (fixing dates, barrier
    // monitoring) would otherwise land on too coarse a grid
    let steps = if option.payoff.is_path_dependent() {
        effective_steps(cfg, &option.model).max(PATH_DEPENDENT_MIN_STEPS)
    } else {
        effective_steps(cfg, &option.model)
    };
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
        let (obs_idx, dfs) =
            observation_grid(option, n_obs, auto.observation_times.as_ref(), p.t, p.r, steps);
        let (sum, sum_sq) = run_heston_paths(option, p, &hp, steps, dt, |spots, _| {
            auto.path_value(spots, &obs_idx, &dfs)
        });
        return stats(sum, sum_sq, cfg.paths, steps, 0.0);
    }

    if let Some(accu) = option.payoff.as_any().downcast_ref::<AccumulatorPayoff>() {
        let n_obs = accu.observations.max(1);
        let steps = steps.div_ceil(n_obs) * n_obs;
        let dt = p.t / steps as f64;
        let (obs_idx, dfs) = observation_grid(option, n_obs, None, p.t, p.r, steps);
        let (sum, sum_sq) = run_heston_paths(option, p, &hp, steps, dt, |spots, _| {
            accu.path_value(spots, &obs_idx, &dfs, p.strike)
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

/// Parallel Heston path generation through the two-factor
/// [`HestonProcess`] (Andersen QE-M under the default `Exact` scheme,
/// full-truncation Euler otherwise): `eval(spots, vols)` receives the
/// path's spot levels and the per-step vols (`sqrt(v_t)` entering each
/// step).
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
    let cfg = option.mc_cfg();
    // Exact requests the best transition sampling available => Andersen
    // QE-M; Euler/Milstein keep the plain full-truncation Euler stepping
    let scheme = match cfg.scheme {
        DiscretizationScheme::Exact => HestonScheme::QuadraticExponential,
        _ => HestonScheme::FullTruncation,
    };
    let process = HestonProcess { drift_rate: p.r - p.q, params: *hp, scheme };
    let sqrt_dt = dt.sqrt();
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
                let mut x = [p.s0, hp.v0];
                let mut x_next = [0.0; 2];
                for j in 0..steps {
                    // independent increments; the process applies rho
                    let dw = [sign * sqrt_dt * z[2 * j], sign * sqrt_dt * z[2 * j + 1]];
                    vols[j] = x[1].max(0.0).sqrt();
                    process.evolve(j as f64 * dt, &x, dt, &dw, &mut x_next);
                    if let Some(divs) = &divs {
                        x_next[0] = (x_next[0] - divs[j]).max(1e-8);
                    }
                    x = x_next;
                    spots[j] = x[0];
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

/// Exercise rights per path index k (spot at time (k+1)dt): every step
/// for American, only mapped steps for Bermudan.
fn exercise_mask(option: &EquityOption, t: f64, steps: usize) -> Vec<bool> {
    match option.payoff.exercise_style() {
        ContractStyle::Bermudan(times) => {
            let mut mask = vec![false; steps.saturating_sub(1)];
            for g in crate::core::utils::times_to_grid_steps(times, t, steps) {
                if g < steps {
                    mask[g - 1] = true;
                }
            }
            mask
        }
        _ => vec![true; steps.saturating_sub(1)],
    }
}

/// Two-pass least-squares Monte Carlo (Longstaff-Schwartz):
/// pass 1 fits the per-date continuation-value regressions on one set of
/// paths; pass 2 applies the fitted exercise rule to an independent set,
/// which removes the foresight (in-sample) bias of single-pass LSMC.
/// Always uses pseudo-random per-path streams. Heston takes its own
/// route ([`heston_american_npv`]): the exercise decision there depends
/// on the variance state, so both the paths and the regression basis are
/// two-dimensional.
fn american_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let cfg = option.mc_cfg();
    if option.model.is_heston() {
        return heston_american_npv(option, p);
    }
    let steps = if cfg.time_steps > 1 { cfg.time_steps } else { LSMC_DEFAULT_STEPS }
        .max(if option.model == Model::LocalVol { LOCAL_VOL_MIN_STEPS } else { 1 });
    let dt = p.t / steps as f64;
    let allowed = exercise_mask(option, p.t, steps);
    let disc = exp(-p.r * dt);
    let process = bs_process(option, p);
    let seed_regression = cfg.seed ^ 0xA11C_E5ED;
    let seed_valuation = cfg.seed ^ 0xB0B5_1EED;

    let simulate = |draws: &PathDraws, index: usize, bufs: &mut (Vec<f64>, Vec<f64>, Vec<f64>), path: &mut Vec<f64>| {
        let (z, w, dw) = bufs;
        draws.fill(index, z, w, dw);
        path.clear();
        let mut s = p.s0;
        for (i, d) in dw.iter().enumerate() {
            s = process.evolve(cfg.scheme, i as f64 * dt, s, dt, *d);
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
        if !allowed[step_idx] {
            // no exercise right at this date: continuation only, no
            // regression fitted, so pass 2 cannot exercise here either
            continue;
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

// ── American under Heston ───────────────────────────────────────────────

const HESTON_LSMC_BASIS: usize = 6;

/// Regression basis over the two-dimensional Heston state: cubic in the
/// normalized spot plus the variance level and its spot cross term — the
/// continuation value of an American option under stochastic vol depends
/// on how much volatility is left, not just on where the spot is.
fn heston_lsmc_basis(x: f64, v: f64) -> [f64; HESTON_LSMC_BASIS] {
    [1.0, x, x * x, x * x * x, v, x * v]
}

/// Two-pass Longstaff-Schwartz under Heston dynamics, stepping the
/// two-factor [`HestonProcess`] (Andersen QE-M under the default `Exact`
/// scheme, full-truncation Euler otherwise) and regressing on the
/// `(spot, variance)` state. Same structure as [`american_npv`]:
/// regression pass on one set of antithetic pseudo-random paths,
/// valuation pass on an independent set. QE's coarse-grid accuracy is
/// what makes this affordable — the exercise grid (default
/// [`LSMC_DEFAULT_STEPS`]) is all the resolution it needs, where
/// full-truncation Euler must step at [`HESTON_MIN_STEPS`].
fn heston_american_npv(option: &EquityOption, p: &MarketParams) -> McStats {
    let hp = option.heston_params().with_vol_shift(p.sigma - option.volatility());
    let cfg = option.mc_cfg();
    let scheme = match cfg.scheme {
        DiscretizationScheme::Exact => HestonScheme::QuadraticExponential,
        _ => HestonScheme::FullTruncation,
    };
    let floor = match scheme {
        HestonScheme::QuadraticExponential => HESTON_QE_MIN_STEPS,
        HestonScheme::FullTruncation => HESTON_MIN_STEPS,
    };
    let steps = if cfg.time_steps > 1 { cfg.time_steps } else { LSMC_DEFAULT_STEPS }.max(floor);
    let dt = p.t / steps as f64;
    let allowed = exercise_mask(option, p.t, steps);
    let disc = exp(-p.r * dt);
    let process = HestonProcess { drift_rate: p.r - p.q, params: hp, scheme };
    let sqrt_dt = dt.sqrt();
    let seed_regression = cfg.seed ^ 0xA11C_E5ED;
    let seed_valuation = cfg.seed ^ 0xB0B5_1EED;

    // fill one path's spot and (truncated) variance levels; antithetic
    // pairs (2k, 2k+1) share a stream with negated draws, as everywhere
    let simulate = |seed: u64, i: usize, z: &mut [f64], spots: &mut [f64], vars: &mut [f64]| {
        path_normals(seed, (i / 2) as u64, z);
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let mut x = [p.s0, hp.v0];
        let mut x_next = [0.0; 2];
        for j in 0..steps {
            let dw = [sign * sqrt_dt * z[2 * j], sign * sqrt_dt * z[2 * j + 1]];
            process.evolve(j as f64 * dt, &x, dt, &dw, &mut x_next);
            x = x_next;
            spots[j] = x[0];
            vars[j] = x[1].max(0.0);
        }
    };

    // ── pass 1: simulate (S, v) paths and fit regressions backwards
    let paths_sv: Vec<(Vec<f64>, Vec<f64>)> = (0..cfg.paths)
        .into_par_iter()
        .map_init(
            || vec![0.0; 2 * steps],
            |z, i| {
                let mut spots = vec![0.0; steps];
                let mut vars = vec![0.0; steps];
                simulate(seed_regression, i, z, &mut spots, &mut vars);
                (spots, vars)
            },
        )
        .collect();

    let mut cashflow: Vec<f64> = paths_sv
        .iter()
        .map(|(spots, _)| option.payoff.payoff(spots[steps - 1], p.strike))
        .collect();
    let mut betas: Vec<Option<[f64; HESTON_LSMC_BASIS]>> = vec![None; steps.saturating_sub(1)];
    for step_idx in (0..steps - 1).rev() {
        for cf in cashflow.iter_mut() {
            *cf *= disc;
        }
        if !allowed[step_idx] {
            continue;
        }
        let itm: Vec<usize> = (0..paths_sv.len())
            .filter(|&i| option.payoff.payoff(paths_sv[i].0[step_idx], p.strike) > 0.0)
            .collect();
        if itm.len() < HESTON_LSMC_BASIS {
            continue;
        }
        let rows: Vec<([f64; HESTON_LSMC_BASIS], f64)> = itm
            .iter()
            .map(|&i| {
                let (spots, vars) = &paths_sv[i];
                (heston_lsmc_basis(spots[step_idx] / p.s0, vars[step_idx]), cashflow[i])
            })
            .collect();
        let Some(beta) = least_squares(&rows) else { continue };
        for &i in &itm {
            let (spots, vars) = &paths_sv[i];
            let s = spots[step_idx];
            let pay = option.payoff.payoff(s, p.strike);
            let continuation = dot(&beta, &heston_lsmc_basis(s / p.s0, vars[step_idx]));
            if pay > continuation {
                cashflow[i] = pay;
            }
        }
        betas[step_idx] = Some(beta);
    }
    drop(paths_sv);
    drop(cashflow);

    // ── pass 2: apply the fitted exercise rule to independent paths
    let partials: Vec<(f64, f64)> = (0..cfg.paths.div_ceil(PATH_CHUNK))
        .into_par_iter()
        .map(|chunk| {
            let mut z = vec![0.0; 2 * steps];
            let mut spots = vec![0.0; steps];
            let mut vars = vec![0.0; steps];
            let (mut c_sum, mut c_sum_sq) = (0.0, 0.0);
            for i in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                simulate(seed_valuation, i, &mut z, &mut spots, &mut vars);
                let mut value = 0.0;
                let mut exercised = false;
                for k in 0..steps - 1 {
                    let pay = option.payoff.payoff(spots[k], p.strike);
                    if pay > 0.0 {
                        if let Some(beta) = &betas[k] {
                            let continuation =
                                dot(beta, &heston_lsmc_basis(spots[k] / p.s0, vars[k]));
                            if pay > continuation {
                                value = pay * disc.powi(k as i32 + 1);
                                exercised = true;
                                break;
                            }
                        }
                    }
                }
                if !exercised {
                    value = option.payoff.payoff(spots[steps - 1], p.strike)
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

fn dot<const K: usize>(a: &[f64; K], b: &[f64; K]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Least squares via the normal equations with partial-pivot Gaussian
/// elimination; None if (near-)singular. Generic over the basis size so
/// the one-dimensional (spot) and Heston (spot, variance) regressions
/// share it.
fn least_squares<const K: usize>(rows: &[([f64; K], f64)]) -> Option<[f64; K]> {
    let mut m = [[0.0; K]; K];
    let mut rhs = [0.0; K];
    for (basis, y) in rows {
        for i in 0..K {
            for j in 0..K {
                m[i][j] += basis[i] * basis[j];
            }
            rhs[i] += basis[i] * y;
        }
    }
    for col in 0..K {
        let pivot =
            (col..K).max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())?;
        if m[pivot][col].abs() < 1e-10 {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in col + 1..K {
            let f = m[row][col] / m[col][col];
            for c in col..K {
                m[row][c] -= f * m[col][c];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut beta = [0.0; K];
    for row in (0..K).rev() {
        let mut acc = rhs[row];
        for c in row + 1..K {
            acc -= m[row][c] * beta[c];
        }
        beta[row] = acc / m[row][row];
    }
    Some(beta)
}

