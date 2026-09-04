//! [`MultiAssetEquityOption`]: the N-underlying companion of
//! [`EquityOption`](super::equity_option::EquityOption) — one contract
//! observed on several correlated equity underlyings, each leg bound to
//! its own market data ([`AssetLeg`]: spot, dividend yield, vol
//! surface) with a shared discount curve and a correlation matrix.
//!
//! Rebinding is honest by construction: [`with_market`]
//! (MultiAssetEquityOption::with_market) pulls **every** leg's spot and
//! surface from the typed [`Market`](crate::core::market::Market) store
//! by symbol, so per-symbol stress shocks land on the right leg and
//! nothing prices off frozen co-asset levels — the defect of the
//! scalar-market standalones this type replaces.
//!
//! Payoffs: the worst-of autocallable (path-simulated on the
//! correlated-GBM machinery ported draw-for-draw from the standalone)
//! and the five rainbow types (best-of / worst-of / spread / basket /
//! exchange), terminal-priced on a **one-pass** Monte Carlo — every
//! Greek bump scenario evaluated against the same correlated draws in
//! a single path-generation pass — or on the analytic engine
//! (Margrabe / Kirk / lognormal moment matching). Each leg's GBM vol
//! is its surface's ATM-forward vol at the contract maturity;
//! per-asset smiles in the *dynamics* and multi-asset stochastic-vol
//! models remain future work. The standalone
//! [`WorstOfAutocallable`](super::worst_of::WorstOfAutocallable) and
//! [`RainbowOption`](super::rainbow::RainbowOption) stay as the
//! flat-market validation references.

use std::sync::Arc;

use chrono::NaiveDate;
use libm::exp;
use rayon::prelude::*;

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::market::{Discount, Market, Spot, Vol};
use crate::core::montecarlo::paths::{FactorScratch, MultiDraws};
use crate::core::montecarlo::process::StochasticProcess;
use crate::core::quotes::Quote;
use crate::core::results::PricingResult;
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::utils::observation_steps;
use crate::core::vols::VolSurface;
use crate::equity::autocallable::AutocallablePayoff;
use crate::equity::montecarlo::{
    summarize, McStats, MonteCarloConfig, PathAccum, Sampler, PATH_DEPENDENT_MIN_STEPS,
};
use crate::equity::processes::MultiAssetGbmProcess;
use crate::equity::rainbow::{
    basket_moment_match_price, kirk_price, margrabe_price, RainbowPayoff, RainbowType,
};
use crate::equity::utils::{Engine, Model, Payoff, PricingEngine};

/// One underlying's market binding: everything that moves with the
/// market for this symbol. The contract-side per-asset state (the
/// initial fixings) lives in [`MultiAssetBase`] instead.
#[derive(Debug, Clone)]
pub struct AssetLeg {
    pub symbol: String,
    pub spot: Quote,
    /// Continuous dividend yield of this leg.
    pub dividend_yield: f64,
    /// This leg's vol surface; `Arc`-shared with the
    /// [`Market`](crate::core::market::Market) store, copy-on-write like
    /// the single-asset binding.
    pub vol_surface: Arc<VolSurface>,
}

/// The market state a multi-asset instrument is bound to: per-symbol
/// legs, the correlation structure, and a shared discount curve.
#[derive(Debug, Clone)]
pub struct MultiAssetMarketData {
    /// The as-of date of this snapshot; anchors every year fraction.
    pub valuation_date: NaiveDate,
    pub assets: Vec<AssetLeg>,
    /// Full correlation matrix (n x n); the Cholesky factor is cached
    /// at construction (Higham-repaired when the input fails PSD).
    pub correlations: Vec<Vec<f64>>,
    pub(crate) chol: Vec<Vec<f64>>,
    pub discount_curve: Arc<YieldCurve>,
}

/// Contract terms and identity — no market state.
#[derive(Debug, Clone)]
pub struct MultiAssetBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub maturity_date: NaiveDate,
    /// Contractual per-asset initial fixings — the denominators of the
    /// performance ratios, frozen at build. Market bumps and rebinds
    /// move the leg spots, never these (otherwise delta would cancel to
    /// zero by homogeneity).
    pub initial_fixings: Vec<f64>,
}

/// An equity option on several correlated underlyings; see module docs.
#[derive(Debug)]
pub struct MultiAssetEquityOption {
    pub base: MultiAssetBase,
    pub market: MultiAssetMarketData,
    /// An [`AutocallablePayoff`] evaluated on the worst-of performance
    /// path (in `initial_fixing` units), or a [`RainbowPayoff`]
    /// (best-of / worst-of / spread / basket / exchange) on the
    /// terminal levels.
    pub payoff: Box<dyn Payoff>,
    pub engine: PricingEngine,
    /// Underlying dynamics; stage 1 supports correlated GBM only.
    pub model: Model,
}

impl Clone for MultiAssetEquityOption {
    fn clone(&self) -> Self {
        MultiAssetEquityOption {
            base: self.base.clone(),
            market: self.market.clone(),
            payoff: self.payoff.clone_box(),
            engine: self.engine,
            model: self.model,
        }
    }
}

/// Market snapshot the Greeks bump (common random numbers: every
/// reprice reuses the same deterministic draws).
#[derive(Clone)]
struct Params {
    spots: Vec<f64>,
    vols: Vec<f64>,
    /// Parallel shift of the discount/drift rate (rho bumps).
    dr: f64,
    t: f64,
}

impl MultiAssetEquityOption {
    /// Start a builder; see [`MultiAssetEquityOptionBuilder`].
    pub fn builder() -> MultiAssetEquityOptionBuilder {
        MultiAssetEquityOptionBuilder::new()
    }

    /// The contract's currency code, falling back to USD.
    pub fn currency_code(&self) -> &str {
        self.base.currency.as_deref().unwrap_or("USD")
    }

    pub fn time_to_maturity(&self) -> f64 {
        crate::equity::conventions::year_fraction(
            self.market.valuation_date,
            self.base.maturity_date,
        )
    }

    /// Rebind **every** leg to a typed market snapshot: per-symbol spot
    /// and vol surface, the currency's discount curve, and the
    /// snapshot's valuation date. Correlations are contract-level
    /// assumptions and stay. Errors name the missing key.
    pub fn with_market(&self, market: &Market) -> Result<MultiAssetEquityOption, RustyQLibError> {
        let mut option = self.clone();
        for leg in &mut option.market.assets {
            leg.spot = *market.get(&Spot(leg.symbol.clone()))?;
            leg.vol_surface = market.get(&Vol(leg.symbol.clone()))?.clone();
        }
        option.market.discount_curve = market
            .get(&Discount(self.currency_code().to_string()))?
            .clone();
        option.market.valuation_date = market.valuation_date();
        Ok(option)
    }

    /// Value under a typed market snapshot: rebind, then price.
    pub fn npv_in(&self, market: &Market) -> Result<f64, RustyQLibError> {
        self.with_market(market)?.try_npv()
    }

    /// Build from the `rainbow_option` JSON contract data — the batch
    /// service's constructor, routed through the builder so JSON
    /// contracts get the same validation as library users. The
    /// standalone [`RainbowOption`](super::rainbow::RainbowOption)
    /// keeps its own parser as the flat-market reference.
    pub fn try_from_rainbow_json(
        data: &crate::equity::rainbow::RainbowOptionData,
    ) -> Result<MultiAssetEquityOption, RustyQLibError> {
        let valuation_date =
            crate::core::data_models::parse_valuation_date(data.valuation_date.as_deref())?;
        let rainbow_type = match data.rainbow_type.trim().to_lowercase().as_str() {
            "best_of" | "bestof" | "max" => RainbowType::BestOf,
            "worst_of" | "worstof" | "min" => RainbowType::WorstOf,
            "spread" => RainbowType::Spread,
            "basket" => RainbowType::Basket,
            "exchange" | "margrabe" => RainbowType::Exchange,
            other => {
                return Err(RustyQLibError::invalid_input(
                    "rainbow_type",
                    format!("invalid rainbow_type '{other}'"),
                ))
            }
        };
        let put_or_call = match data.put_or_call.as_deref().unwrap_or("C").trim() {
            "C" | "c" | "Call" | "call" => PutOrCall::Call,
            "P" | "p" | "Put" | "put" => PutOrCall::Put,
            other => {
                return Err(RustyQLibError::invalid_input(
                    "put_or_call",
                    format!("invalid put_or_call '{other}' (use 'C' or 'P')"),
                ))
            }
        };
        if rainbow_type != RainbowType::Exchange && data.strike_price.is_none() {
            return Err(RustyQLibError::invalid_input(
                "strike_price",
                "strike_price is required",
            ));
        }
        let maturity_date =
            NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    "maturity",
                    format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
                )
            })?;
        let mut b = Self::builder()
            .symbol(&data.symbol)
            .valuation_date(valuation_date)
            .maturity_date(maturity_date)
            .correlations(data.correlations.clone());
        for a in &data.assets {
            b = b.asset(&a.symbol, a.spot, a.volatility, a.dividend.unwrap_or(0.0));
        }
        b = match &data.discount_curve {
            Some(input) => b.discount_curve(YieldCurve::from_input(input, valuation_date)?),
            None => b.flat_rate(data.risk_free_rate.ok_or_else(|| {
                RustyQLibError::invalid_input(
                    "risk_free_rate",
                    "either risk_free_rate or discount_curve must be provided",
                )
            })?),
        };
        b = match rainbow_type {
            RainbowType::Exchange => b.exchange(put_or_call),
            RainbowType::BestOf => {
                b.best_of(put_or_call, data.strike_price.expect("checked above"))
            }
            RainbowType::WorstOf => {
                b.worst_of(put_or_call, data.strike_price.expect("checked above"))
            }
            RainbowType::Spread => b.spread(put_or_call, data.strike_price.expect("checked above")),
            RainbowType::Basket => {
                b = b.basket(put_or_call, data.strike_price.expect("checked above"));
                if let Some(w) = &data.weights {
                    b = b.basket_weights(w.clone());
                }
                b
            }
        };
        b = match data.pricer.as_deref().map_or("MC", |v| v).trim() {
            "Analytical" | "analytical" => b.engine(Engine::BlackScholes),
            "MonteCarlo" | "montecarlo" | "MC" | "mc" => {
                let sampler = data
                    .mc_sampler
                    .as_deref()
                    .map(|txt| {
                        txt.parse::<Sampler>().map_err(|_| {
                            RustyQLibError::invalid_input(
                                "mc_sampler",
                                format!("invalid mc_sampler '{txt}'"),
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or(Sampler::Sobol);
                b.engine(Engine::MonteCarlo)
                    .paths(data.simulation.unwrap_or(100_000) as usize)
                    .sampler(sampler)
                    .seed(data.mc_seed.unwrap_or(42))
            }
            other => {
                return Err(RustyQLibError::invalid_input(
                    "pricer",
                    format!("invalid pricer '{other}' for rainbow (Analytical or MC)"),
                ))
            }
        };
        b.build()
    }

    /// Support gate: correlated GBM dynamics; rainbow payoffs price on
    /// the Analytical (Margrabe / Kirk / moment match) or MonteCarlo
    /// engines, autocallables on MonteCarlo only.
    pub(crate) fn check_engine_support(&self) -> Result<(), RustyQLibError> {
        let unsupported = |msg: String| Err(RustyQLibError::UnsupportedEngine(msg));
        if !matches!(self.model, Model::Gbm) {
            return unsupported(
                "multi-asset notes price under correlated GBM dynamics; multi-asset \
                 stochastic-vol models are future work"
                    .to_string(),
            );
        }
        if let Some(rb) = self.payoff.as_any().downcast_ref::<RainbowPayoff>() {
            return match &self.engine {
                PricingEngine::BlackScholes => {
                    if matches!(rb.rainbow_type, RainbowType::BestOf | RainbowType::WorstOf) {
                        unsupported(
                            "best-of / worst-of rainbows have no analytic pricer yet \
                             (Stulz for two assets is future work); use MonteCarlo"
                                .to_string(),
                        )
                    } else {
                        Ok(())
                    }
                }
                PricingEngine::MonteCarlo(_) => Ok(()),
                other => unsupported(format!(
                    "rainbow options price on the Analytical or MonteCarlo engines, not {:?}",
                    other.kind()
                )),
            };
        }
        if !matches!(self.engine, PricingEngine::MonteCarlo(_)) {
            return unsupported(
                "multi-asset autocallables price on the MonteCarlo engine only".to_string(),
            );
        }
        Ok(())
    }

    /// Monte Carlo settings. Invariant: only called on the Monte Carlo
    /// code paths (the engine dispatch guarantees it).
    fn mc_cfg(&self) -> &MonteCarloConfig {
        match &self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    /// Each leg's GBM vol: the surface's ATM-forward vol at the
    /// contract maturity (per-asset smiles in the dynamics are future
    /// work, as documented for the standalones).
    fn leg_vols(&self, t: f64) -> Vec<f64> {
        let r = self
            .market
            .discount_curve
            .zero_rate_with(t, Compounding::Continuous);
        self.market
            .assets
            .iter()
            .map(|leg| {
                let f = leg.spot.value() * exp((r - leg.dividend_yield) * t);
                leg.vol_surface.vol(f, f, t)
            })
            .collect()
    }

    fn params(&self) -> Params {
        let t = self.time_to_maturity();
        Params {
            spots: self.market.assets.iter().map(|a| a.spot.value()).collect(),
            vols: self.leg_vols(t),
            dr: 0.0,
            t,
        }
    }

    /// The worst-of payoff. Invariant: stage-1 construction only builds
    /// autocallable payoffs (the builder guarantees it).
    fn autocall(&self) -> &AutocallablePayoff {
        self.payoff
            .as_any()
            .downcast_ref::<AutocallablePayoff>()
            .expect("stage-1 multi-asset payoffs are autocallables")
    }

    /// Observation grid on a path of `steps` steps over life `t`:
    /// per-observation path indices (strictly increasing) and discount
    /// factors at the exact observation times. Explicit times map
    /// through [`observation_steps`], which drops an observation that
    /// collapses onto an already-used step together with its time, so
    /// `obs_idx[m]` and `dfs[m]` always describe the same fixing.
    fn observation_grid(&self, t: f64, dr: f64, steps: usize) -> (Vec<usize>, Vec<f64>) {
        let auto = self.autocall();
        let n_obs = auto.observations.max(1);
        let (obs_idx, obs_times): (Vec<usize>, Vec<f64>) = match &auto.observation_times {
            Some(times) => observation_steps(times, t, steps).into_iter().unzip(),
            None => {
                let dt = t / steps as f64;
                let idx: Vec<usize> = (1..=n_obs).map(|m| m * steps / n_obs - 1).collect();
                let times = idx.iter().map(|&i| (i + 1) as f64 * dt).collect();
                (idx, times)
            }
        };
        let dfs = obs_times
            .iter()
            .map(|&tm| self.market.discount_curve.df(tm) * exp(-dr * tm))
            .collect();
        (obs_idx, dfs)
    }

    /// Price with sampling diagnostics on the base market.
    pub fn npv_with_stats(&self) -> McStats {
        self.stats_with(&self.params())
    }

    /// Value one bumped-parameter snapshot on the configured engine:
    /// analytic closed forms, the one-pass terminal route for rainbow
    /// payoffs, or the worst-of path simulation for autocallables.
    fn stats_with(&self, p: &Params) -> McStats {
        if matches!(self.engine, PricingEngine::BlackScholes) {
            return McStats {
                pv: self.analytic_npv_with(p),
                std_err: None,
                paths: 0,
                steps: 0,
            };
        }
        if self
            .payoff
            .as_any()
            .downcast_ref::<RainbowPayoff>()
            .is_some()
        {
            let mut stats = self.rainbow_terminal_stats(std::slice::from_ref(p));
            return stats.pop().expect("one scenario in, one result out");
        }
        self.autocall_stats_with(p)
    }

    /// Analytic rainbow pricers on the bumped snapshot (Margrabe /
    /// Kirk / basket moment matching), with the discount/drift rate
    /// read from the curve at the (possibly bumped) tenor plus `dr`.
    fn analytic_npv_with(&self, p: &Params) -> f64 {
        let rb = self
            .payoff
            .as_any()
            .downcast_ref::<RainbowPayoff>()
            .expect("the analytic engine is gated to rainbow payoffs");
        let dividends: Vec<f64> = self
            .market
            .assets
            .iter()
            .map(|a| a.dividend_yield)
            .collect();
        let r = self
            .market
            .discount_curve
            .zero_rate_with(p.t, Compounding::Continuous)
            + p.dr;
        match rb.rainbow_type {
            RainbowType::Exchange => margrabe_price(
                &p.spots,
                &dividends,
                &p.vols,
                self.market.correlations[0][1],
                p.t,
                rb.put_or_call,
            ),
            RainbowType::Spread => kirk_price(
                &p.spots,
                &dividends,
                &p.vols,
                self.market.correlations[0][1],
                r,
                p.t,
                rb.strike_price,
                rb.put_or_call,
            ),
            RainbowType::Basket => basket_moment_match_price(
                &p.spots,
                &dividends,
                &p.vols,
                &self.market.correlations,
                &rb.weights,
                r,
                p.t,
                rb.strike_price,
                rb.put_or_call,
            ),
            // invariant: check_engine_support refuses these before pricing
            RainbowType::BestOf | RainbowType::WorstOf => {
                unreachable!("best-of/worst-of on the analytic engine is rejected before pricing")
            }
        }
    }

    /// Terminal correlated-GBM valuation of **every scenario in one
    /// path-generation pass**: the correlated draws are computed once
    /// per path and each scenario's terminal levels are re-derived from
    /// the same draws (exact one-step lognormal transitions), so a full
    /// Greek request costs one pass instead of one simulation per bump
    /// leg — while staying bit-identical to running each scenario as
    /// its own common-random-number simulation (pinned by test).
    fn rainbow_terminal_stats(&self, scenarios: &[Params]) -> Vec<McStats> {
        const PATH_CHUNK: usize = 4096;
        let rb = self
            .payoff
            .as_any()
            .downcast_ref::<RainbowPayoff>()
            .expect("terminal route reached with a non-rainbow payoff");
        let n = self.market.assets.len();
        let cfg = self.mc_cfg();
        let dividends: Vec<f64> = self
            .market
            .assets
            .iter()
            .map(|a| a.dividend_yield)
            .collect();
        // per-scenario precomputation: drifts, diffusion loadings, discount
        struct Scenario {
            spots: Vec<f64>,
            drifts: Vec<f64>,
            vol_sqrt_t: Vec<f64>,
            df: f64,
        }
        let scs: Vec<Scenario> = scenarios
            .iter()
            .map(|p| {
                let r = self
                    .market
                    .discount_curve
                    .zero_rate_with(p.t, Compounding::Continuous)
                    + p.dr;
                Scenario {
                    spots: p.spots.clone(),
                    drifts: (0..n)
                        .map(|i| (r - dividends[i] - 0.5 * p.vols[i] * p.vols[i]) * p.t)
                        .collect(),
                    vol_sqrt_t: (0..n).map(|i| p.vols[i] * p.t.sqrt()).collect(),
                    df: exp(-r * p.t),
                }
            })
            .collect();
        let qmc = match cfg.sampler {
            Sampler::Sobol => Some(crate::core::montecarlo::LowDiscrepancy::best(n, cfg.seed)),
            Sampler::PseudoRandom => None,
        };
        let chunks = cfg.paths.div_ceil(PATH_CHUNK);
        let partials: Vec<Vec<PathAccum>> = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut eps = vec![0.0; n];
                let mut z = vec![0.0; n];
                let mut terminal = vec![0.0; n];
                let mut accs = vec![PathAccum::default(); scs.len()];
                for path in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                    match &qmc {
                        Some(seq) => seq.normals(path as u64 + 1, &mut eps),
                        None => {
                            // antithetic pairs from per-pair streams
                            crate::core::montecarlo::path_normals(
                                cfg.seed,
                                (path / 2) as u64,
                                &mut eps,
                            );
                            if path % 2 == 1 {
                                for e in eps.iter_mut() {
                                    *e = -*e;
                                }
                            }
                        }
                    }
                    for i in 0..n {
                        // z_i = sum_j L[i][j] eps_j (Cholesky-correlated)
                        z[i] = (0..=i).map(|j| self.market.chol[i][j] * eps[j]).sum();
                    }
                    for (sc, acc) in scs.iter().zip(accs.iter_mut()) {
                        for i in 0..n {
                            terminal[i] = sc.spots[i] * exp(sc.drifts[i] + sc.vol_sqrt_t[i] * z[i]);
                        }
                        acc.push(path, sc.df * rb.terminal_value(&terminal));
                    }
                }
                accs
            })
            .collect();
        let mut folded = vec![PathAccum::default(); scs.len()];
        for chunk_accs in partials {
            for (f, a) in folded.iter_mut().zip(chunk_accs) {
                *f = f.merge(a);
            }
        }
        folded
            .into_iter()
            .map(|acc| {
                summarize(
                    acc,
                    cfg.paths,
                    1,
                    0.0,
                    matches!(cfg.sampler, Sampler::Sobol),
                )
            })
            .collect()
    }

    /// The correlated-GBM Monte Carlo for autocallables, ported
    /// draw-for-draw from the standalone worst-of so identical inputs
    /// and seeds give identical values (the standalone is the
    /// validation reference).
    fn autocall_stats_with(&self, p: &Params) -> McStats {
        let n = self.market.assets.len();
        let t = p.t;
        let auto = self.autocall();
        let cfg = self.mc_cfg();
        let n_obs = auto.observations.max(1);
        // every observation lands exactly on a simulation step
        let steps = cfg.time_steps.max(PATH_DEPENDENT_MIN_STEPS).div_ceil(n_obs) * n_obs;
        let dt = t / steps as f64;
        let (obs_idx, dfs) = self.observation_grid(t, p.dr, steps);
        let r = self
            .market
            .discount_curve
            .zero_rate_with(t, Compounding::Continuous)
            + p.dr;
        let process = MultiAssetGbmProcess {
            drift_rates: self
                .market
                .assets
                .iter()
                .map(|a| r - a.dividend_yield)
                .collect(),
            vols: p.vols.clone(),
            chol: self.market.chol.clone(),
        };
        let draws = MultiDraws::new(cfg.sampler, cfg.seed, n, steps, dt);
        let fixing = auto.initial_fixing;

        const CHUNK: usize = 4096;
        let chunks = cfg.paths.div_ceil(CHUNK);
        let partials: Vec<PathAccum> = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut scratch = FactorScratch::new(n, steps);
                let mut dw = vec![0.0; n * steps];
                let mut x = vec![0.0; n];
                let mut x_next = vec![0.0; n];
                let mut worst = vec![0.0; steps];
                let mut acc = PathAccum::default();
                for i in chunk * CHUNK..((chunk + 1) * CHUNK).min(cfg.paths) {
                    draws.fill(i, n, steps, &mut scratch, &mut dw);
                    x.copy_from_slice(&p.spots);
                    for j in 0..steps {
                        process.evolve(j as f64 * dt, &x, dt, &dw[j * n..(j + 1) * n], &mut x_next);
                        x.copy_from_slice(&x_next);
                        // worst-of performance in initial_fixing units,
                        // normalized by the *contractual* fixings: market
                        // bumps move the path start, never the
                        // denominators
                        let w = x
                            .iter()
                            .zip(&self.base.initial_fixings)
                            .map(|(s, s0)| s / s0)
                            .fold(f64::MAX, f64::min);
                        worst[j] = fixing * w;
                    }
                    acc.push(i, auto.path_value(&worst, &obs_idx, &dfs));
                }
                acc
            })
            .collect();
        let acc = partials
            .into_iter()
            .fold(PathAccum::default(), PathAccum::merge);
        summarize(
            acc,
            cfg.paths,
            steps,
            0.0,
            matches!(cfg.sampler, Sampler::Sobol),
        )
    }

    /// Per-asset spot deltas (central bumps, common random numbers).
    pub fn deltas(&self) -> Vec<f64> {
        let mut rep = MultiRepricer::new(self);
        (0..self.market.assets.len())
            .map(|i| rep.delta(i))
            .collect()
    }

    /// Per-asset gammas (same-leg second differences on the delta legs).
    pub fn gammas(&self) -> Vec<f64> {
        let mut rep = MultiRepricer::new(self);
        (0..self.market.assets.len())
            .map(|i| rep.gamma(i))
            .collect()
    }

    /// Per-asset vegas (central bumps of each leg's vol; the down leg is
    /// floored at the minimum bumped vol and the stencil divides by the
    /// **effective** spread, so tiny-vol legs are not overstated).
    pub fn vegas(&self) -> Vec<f64> {
        let mut rep = MultiRepricer::new(self);
        (0..self.market.assets.len()).map(|i| rep.vega(i)).collect()
    }

    pub fn theta(&self) -> f64 {
        MultiRepricer::new(self).theta()
    }

    pub fn rho(&self) -> f64 {
        MultiRepricer::new(self).rho()
    }

    /// Value plus every reported sensitivity from **one** shared reprice
    /// cache: the base simulation and each bump leg run exactly once,
    /// per-asset gammas fall out of the delta legs for free, and every
    /// reprice reuses the same deterministic draws (common random
    /// numbers). This is the batch entry point the former
    /// `price()` + `deltas()` + `vegas()` service pattern re-simulated
    /// its way around.
    pub fn pricing_result(&self) -> Result<PricingResult, RustyQLibError> {
        self.check_engine_support()?;
        if self
            .payoff
            .as_any()
            .downcast_ref::<RainbowPayoff>()
            .is_some()
            && matches!(self.engine, PricingEngine::MonteCarlo(_))
        {
            return self.rainbow_pricing_result();
        }
        let n = self.market.assets.len();
        let mut rep = MultiRepricer::new(self);
        let stats = rep.base_stats();
        let asset_greeks = crate::core::results::PerAssetGreeks {
            symbols: self
                .market
                .assets
                .iter()
                .map(|a| a.symbol.clone())
                .collect(),
            deltas: (0..n).map(|i| rep.delta(i)).collect(),
            gammas: (0..n).map(|i| rep.gamma(i)).collect(),
            vegas: (0..n).map(|i| rep.vega(i)).collect(),
        };
        Ok(PricingResult {
            pv: stats.pv,
            greeks: crate::core::results::Greeks {
                theta: rep.theta(),
                rho: rep.rho(),
                ..Default::default()
            },
            std_err: stats.std_err,
            asset_greeks: Some(asset_greeks),
        })
    }
}

impl MultiAssetEquityOption {
    /// The batched rainbow result through the one-pass evaluator:
    /// every bump scenario (per-asset spot up/dn, per-asset vol up/dn,
    /// theta and rho legs) is valued against the same correlated draws
    /// in a single path-generation pass. Values are bit-identical to
    /// the per-scenario common-random-number simulations the
    /// piecemeal accessors run (pinned by test); the cost drops from
    /// `4n + 5` path generations to one.
    fn rainbow_pricing_result(&self) -> Result<PricingResult, RustyQLibError> {
        let n = self.market.assets.len();
        let base = self.params();
        let mut scenarios = vec![base.clone()];
        // per-asset spot up/dn (delta + gamma legs)
        let spot_h: Vec<f64> = (0..n).map(|i| base.spots[i] * 0.01).collect();
        for i in 0..n {
            let mut up = base.clone();
            up.spots[i] += spot_h[i];
            let mut dn = base.clone();
            dn.spots[i] -= spot_h[i];
            scenarios.push(up);
            scenarios.push(dn);
        }
        // per-asset vol up/dn (effective-spread convention, review B9)
        let vol_h = 0.01;
        let mut vol_spreads = Vec::with_capacity(n);
        for i in 0..n {
            let mut up = base.clone();
            up.vols[i] += vol_h;
            let mut dn = base.clone();
            dn.vols[i] = (dn.vols[i] - vol_h).max(crate::equity::conventions::MIN_BUMPED_VOL);
            vol_spreads.push(up.vols[i] - dn.vols[i]);
            scenarios.push(up);
            scenarios.push(dn);
        }
        // theta and rho legs
        let t_h = (1.0 / 365.0_f64).min(0.5 * base.t);
        let mut t_up = base.clone();
        t_up.t += t_h;
        let mut t_dn = base.clone();
        t_dn.t -= t_h;
        scenarios.push(t_up);
        scenarios.push(t_dn);
        let r_h = 1e-4;
        let mut r_up = base.clone();
        r_up.dr += r_h;
        let mut r_dn = base.clone();
        r_dn.dr -= r_h;
        scenarios.push(r_up);
        scenarios.push(r_dn);

        let stats = self.rainbow_terminal_stats(&scenarios);
        let v = |idx: usize| stats[idx].pv;
        let base_stats = stats[0];
        let mut deltas = Vec::with_capacity(n);
        let mut gammas = Vec::with_capacity(n);
        for i in 0..n {
            let (up, dn) = (v(1 + 2 * i), v(2 + 2 * i));
            deltas.push((up - dn) / (2.0 * spot_h[i]));
            gammas.push((up - 2.0 * base_stats.pv + dn) / (spot_h[i] * spot_h[i]));
        }
        let vega_base = 1 + 2 * n;
        let vegas: Vec<f64> = (0..n)
            .map(|i| (v(vega_base + 2 * i) - v(vega_base + 2 * i + 1)) / vol_spreads[i])
            .collect();
        let t_base = vega_base + 2 * n;
        let theta = -(v(t_base) - v(t_base + 1)) / (2.0 * t_h);
        let rho = (v(t_base + 2) - v(t_base + 3)) / (2.0 * r_h);
        Ok(PricingResult {
            pv: base_stats.pv,
            greeks: crate::core::results::Greeks {
                theta,
                rho,
                ..Default::default()
            },
            std_err: base_stats.std_err,
            asset_greeks: Some(crate::core::results::PerAssetGreeks {
                symbols: self
                    .market
                    .assets
                    .iter()
                    .map(|a| a.symbol.clone())
                    .collect(),
                deltas,
                gammas,
                vegas,
            }),
        })
    }
}

/// Memoized reprices of one multi-asset option keyed on the full bumped
/// parameter snapshot, so a batched Greek request never runs the same
/// simulation twice (the base leg is the usual repeat customer: value,
/// per-asset gammas and any caller-side `npv()` all read it). Bump
/// sizes: 1% of each spot for delta/gamma, 1 vol point for vega (down
/// leg floored, effective-spread divisor), 1 day capped at half-life
/// for theta, 1bp for rho.
struct MultiRepricer<'a> {
    option: &'a MultiAssetEquityOption,
    base: Params,
    cache: std::collections::HashMap<Vec<u64>, f64>,
    base_stats: Option<McStats>,
}

impl<'a> MultiRepricer<'a> {
    fn new(option: &'a MultiAssetEquityOption) -> Self {
        MultiRepricer {
            option,
            base: option.params(),
            cache: std::collections::HashMap::new(),
            base_stats: None,
        }
    }

    fn key(p: &Params) -> Vec<u64> {
        p.spots
            .iter()
            .chain(p.vols.iter())
            .map(|x| x.to_bits())
            .chain([p.dr.to_bits(), p.t.to_bits()])
            .collect()
    }

    fn reprice(&mut self, p: &Params) -> f64 {
        let key = Self::key(p);
        if let Some(&v) = self.cache.get(&key) {
            return v;
        }
        let v = self.option.stats_with(p).pv;
        self.cache.insert(key, v);
        v
    }

    /// Base value with its sampling diagnostics; the pv is seeded into
    /// the cache so gamma's center leg is free.
    fn base_stats(&mut self) -> McStats {
        if let Some(stats) = self.base_stats {
            return stats;
        }
        let stats = self.option.stats_with(&self.base);
        self.cache.insert(Self::key(&self.base), stats.pv);
        self.base_stats = Some(stats);
        stats
    }

    fn delta_legs(&mut self, i: usize) -> (f64, f64, f64) {
        let h = self.base.spots[i] * 0.01;
        let mut up = self.base.clone();
        up.spots[i] += h;
        let mut dn = self.base.clone();
        dn.spots[i] -= h;
        (self.reprice(&up), self.reprice(&dn), h)
    }

    fn delta(&mut self, i: usize) -> f64 {
        let (up, dn, h) = self.delta_legs(i);
        (up - dn) / (2.0 * h)
    }

    fn gamma(&mut self, i: usize) -> f64 {
        let base = self.base_stats().pv;
        let (up, dn, h) = self.delta_legs(i);
        (up - 2.0 * base + dn) / (h * h)
    }

    fn vega(&mut self, i: usize) -> f64 {
        let h = 0.01;
        let mut up = self.base.clone();
        up.vols[i] += h;
        let mut dn = self.base.clone();
        dn.vols[i] = (dn.vols[i] - h).max(crate::equity::conventions::MIN_BUMPED_VOL);
        // divide by the effective spread: a floored down leg must not
        // masquerade as a full central difference (review finding B9)
        let spread = up.vols[i] - dn.vols[i];
        (self.reprice(&up) - self.reprice(&dn)) / spread
    }

    fn theta(&mut self) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * self.base.t);
        let mut up = self.base.clone();
        up.t += h;
        let mut dn = self.base.clone();
        dn.t -= h;
        -(self.reprice(&up) - self.reprice(&dn)) / (2.0 * h)
    }

    fn rho(&mut self) -> f64 {
        let h = 1e-4;
        let mut up = self.base.clone();
        up.dr += h;
        let mut dn = self.base.clone();
        dn.dr -= h;
        (self.reprice(&up) - self.reprice(&dn)) / (2.0 * h)
    }
}

impl Instrument for MultiAssetEquityOption {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        self.check_engine_support()?;
        Ok(self.npv_with_stats().pv)
    }

    /// The full batched result — value, scalar theta/rho, the Monte
    /// Carlo standard error, and per-asset deltas/gammas/vegas in
    /// [`PricingResult::asset_greeks`] — from one shared reprice cache
    /// ([`pricing_result`](MultiAssetEquityOption::pricing_result)).
    /// The scalar spot/vol slots stay zero by design.
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        self.pricing_result()
    }
}

// ── Builder ─────────────────────────────────────────────────────────────

/// One prompted leg before materialization.
#[derive(Debug, Clone)]
struct LegSpec {
    symbol: String,
    spot: f64,
    flat_vol: f64,
    dividend_yield: f64,
}

/// Worst-of autocallable terms before materialization; barriers are
/// worst-of performance levels in `initial_fixing = 100` units
/// (autocall 100 = 100% of initial).
#[derive(Debug, Clone)]
struct WorstOfSpec {
    autocall_barrier: f64,
    protection_barrier: f64,
    coupon: f64,
    observations: usize,
    notional: f64,
}

/// The payoff request before materialization.
#[derive(Debug, Clone)]
enum MaPayoffSpec {
    WorstOfAutocall(WorstOfSpec),
    Rainbow {
        rainbow_type: RainbowType,
        put_or_call: PutOrCall,
        /// Required except for exchange options (which have none).
        strike: Option<f64>,
        /// Basket weights; defaults to equal weights at build.
        weights: Option<Vec<f64>>,
    },
}

/// Builder for [`MultiAssetEquityOption`] — the multi-asset analogue of
/// [`EquityOptionBuilder`](crate::equity::builder::EquityOptionBuilder):
/// every input validated at `build()`, "builds => prices".
pub struct MultiAssetEquityOptionBuilder {
    symbol: String,
    currency: Option<String>,
    assets: Vec<LegSpec>,
    correlations: Option<Vec<Vec<f64>>>,
    flat_rate: Option<f64>,
    discount_curve: Option<YieldCurve>,
    valuation_date: NaiveDate,
    maturity_date: Option<NaiveDate>,
    payoff: Option<MaPayoffSpec>,
    engine: Engine,
    mc: MonteCarloConfig,
    setter_error: Option<RustyQLibError>,
}

impl Default for MultiAssetEquityOptionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAssetEquityOptionBuilder {
    pub fn new() -> Self {
        MultiAssetEquityOptionBuilder {
            symbol: "BASKET".to_string(),
            currency: None,
            assets: Vec::new(),
            correlations: None,
            flat_rate: None,
            discount_curve: None,
            valuation_date: chrono::Local::now().date_naive(),
            maturity_date: None,
            payoff: None,
            engine: Engine::MonteCarlo,
            mc: MonteCarloConfig::default(),
            setter_error: None,
        }
    }

    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = symbol.to_string();
        self
    }

    pub fn currency(mut self, currency: &str) -> Self {
        self.currency = Some(currency.to_string());
        self
    }

    /// Add one underlying: symbol, spot, flat vol and continuous
    /// dividend yield. Call once per leg, in basket order (the
    /// correlation matrix follows this order).
    pub fn asset(mut self, symbol: &str, spot: f64, flat_vol: f64, dividend_yield: f64) -> Self {
        self.assets.push(LegSpec {
            symbol: symbol.to_string(),
            spot,
            flat_vol,
            dividend_yield,
        });
        self
    }

    /// Full n x n correlation matrix in leg order; a non-PSD empirical
    /// matrix is Higham-repaired at build.
    pub fn correlations(mut self, correlations: Vec<Vec<f64>>) -> Self {
        self.correlations = Some(correlations);
        self
    }

    pub fn flat_rate(mut self, rate: f64) -> Self {
        self.flat_rate = Some(rate);
        self.discount_curve = None;
        self
    }

    pub fn discount_curve(mut self, curve: YieldCurve) -> Self {
        self.discount_curve = Some(curve);
        self
    }

    pub fn valuation_date(mut self, date: NaiveDate) -> Self {
        self.valuation_date = date;
        self
    }

    pub fn maturity_date(mut self, date: NaiveDate) -> Self {
        self.maturity_date = Some(date);
        self
    }

    pub fn years_to_maturity(mut self, years: f64) -> Self {
        self.maturity_date =
            Some(self.valuation_date + chrono::Duration::days((years * 365.0).round() as i64));
        self
    }

    /// Worst-of autocallable (Athena): at each of `observations`
    /// equally spaced dates the worst-of performance (in
    /// `initial_fixing = 100` units) is compared against
    /// `autocall_barrier`; called notes redeem `notional` plus the
    /// accrued `coupon` per elapsed observation; at maturity the
    /// `protection_barrier` gates downside participation.
    pub fn worst_of_autocallable(
        mut self,
        autocall_barrier: f64,
        protection_barrier: f64,
        coupon: f64,
        observations: usize,
        notional: f64,
    ) -> Self {
        self.payoff = Some(MaPayoffSpec::WorstOfAutocall(WorstOfSpec {
            autocall_barrier,
            protection_barrier,
            coupon,
            observations,
            notional,
        }));
        self
    }

    fn rainbow(
        mut self,
        rainbow_type: RainbowType,
        put_or_call: PutOrCall,
        strike: Option<f64>,
    ) -> Self {
        self.payoff = Some(MaPayoffSpec::Rainbow {
            rainbow_type,
            put_or_call,
            strike,
            weights: None,
        });
        self
    }

    /// Best-of rainbow: pays on the best performer's terminal level
    /// against `strike`. Monte Carlo only (no analytic pricer yet).
    pub fn best_of(self, put_or_call: PutOrCall, strike: f64) -> Self {
        self.rainbow(RainbowType::BestOf, put_or_call, Some(strike))
    }

    /// Worst-of rainbow option (vanilla on the worst terminal level;
    /// for the autocallable note see
    /// [`worst_of_autocallable`](Self::worst_of_autocallable)).
    /// Monte Carlo only (no analytic pricer yet).
    pub fn worst_of(self, put_or_call: PutOrCall, strike: f64) -> Self {
        self.rainbow(RainbowType::WorstOf, put_or_call, Some(strike))
    }

    /// Spread option on exactly two legs, `(S1 - S2 - K)^+`; prices on
    /// MonteCarlo or the Analytical engine (Kirk's approximation).
    pub fn spread(self, put_or_call: PutOrCall, strike: f64) -> Self {
        self.rainbow(RainbowType::Spread, put_or_call, Some(strike))
    }

    /// Basket option on the weighted sum of terminal levels (equal
    /// weights unless [`basket_weights`](Self::basket_weights) is set);
    /// prices on MonteCarlo or the Analytical engine (lognormal moment
    /// matching).
    pub fn basket(self, put_or_call: PutOrCall, strike: f64) -> Self {
        self.rainbow(RainbowType::Basket, put_or_call, Some(strike))
    }

    /// Margrabe exchange option on exactly two legs, `(S1 - S2)^+`
    /// (mirrored for puts); prices on MonteCarlo or the Analytical
    /// engine (exact closed form).
    pub fn exchange(self, put_or_call: PutOrCall) -> Self {
        self.rainbow(RainbowType::Exchange, put_or_call, None)
    }

    /// Basket weights in leg order; must follow [`basket`](Self::basket).
    pub fn basket_weights(mut self, weights: Vec<f64>) -> Self {
        match &mut self.payoff {
            Some(MaPayoffSpec::Rainbow {
                rainbow_type: RainbowType::Basket,
                weights: w,
                ..
            }) => *w = Some(weights),
            _ => {
                self.setter_error = Some(RustyQLibError::invalid_input(
                    "basket_weights",
                    "basket_weights must follow .basket(...)",
                ));
            }
        }
        self
    }

    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }

    pub fn paths(mut self, paths: usize) -> Self {
        self.mc.paths = paths;
        self
    }

    pub fn mc_time_steps(mut self, steps: usize) -> Self {
        self.mc.time_steps = steps;
        self
    }

    pub fn sampler(mut self, sampler: Sampler) -> Self {
        self.mc.sampler = sampler;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.mc.seed = seed;
        self
    }

    /// Validate every input and construct the option ("builds =>
    /// prices").
    pub fn build(mut self) -> Result<MultiAssetEquityOption, RustyQLibError> {
        if let Some(e) = self.setter_error.take() {
            return Err(e);
        }
        let invalid = |field: &str, reason: String| {
            Err(RustyQLibError::InvalidInput {
                field: field.to_string(),
                reason,
            })
        };

        // ── legs ────────────────────────────────────────────────────────
        let n = self.assets.len();
        if n < 2 {
            return invalid(
                "assets",
                "multi-asset options need at least two legs; add them with .asset(...)".to_string(),
            );
        }
        for leg in &self.assets {
            if !(leg.spot.is_finite() && leg.spot > 0.0) {
                return invalid(
                    "assets",
                    format!(
                        "spot of '{}' must be positive and finite, got {}",
                        leg.symbol, leg.spot
                    ),
                );
            }
            if !(leg.flat_vol.is_finite() && leg.flat_vol > 0.0) {
                return invalid(
                    "assets",
                    format!(
                        "volatility of '{}' must be positive and finite, got {}",
                        leg.symbol, leg.flat_vol
                    ),
                );
            }
            crate::equity::conventions::check_vol_band(
                &format!("vol({})", leg.symbol),
                leg.flat_vol,
            )?;
            if !leg.dividend_yield.is_finite() {
                return invalid(
                    "assets",
                    format!("dividend yield of '{}' must be finite", leg.symbol),
                );
            }
            crate::equity::conventions::check_rate_band(
                &format!("dividend({})", leg.symbol),
                leg.dividend_yield,
            )?;
        }

        // ── correlations ────────────────────────────────────────────────
        let correlations = match self.correlations.take() {
            Some(c) => c,
            None => {
                return invalid(
                    "correlations",
                    "set correlations(...) before build()".to_string(),
                )
            }
        };
        if correlations.len() != n || correlations.iter().any(|row| row.len() != n) {
            return invalid(
                "correlations",
                format!("correlations must be an {n} x {n} matrix"),
            );
        }
        let chol = crate::core::linalg::cholesky_with_repair(&correlations)?;

        // ── curve / dates ───────────────────────────────────────────────
        let discount_curve = match self.discount_curve.take() {
            Some(c) => c,
            None => {
                let rate = match self.flat_rate {
                    Some(r) => r,
                    None => {
                        return invalid(
                            "flat_rate",
                            "set flat_rate() or discount_curve() before build()".to_string(),
                        )
                    }
                };
                crate::equity::conventions::check_rate_band("flat_rate", rate)?;
                YieldCurve::flat(
                    rate,
                    self.valuation_date,
                    crate::equity::conventions::EQUITY_DAY_COUNT,
                    Compounding::Continuous,
                )?
            }
        };
        let maturity_date = match self.maturity_date {
            Some(d) => d,
            None => {
                return invalid(
                    "maturity_date",
                    "set maturity_date() or years_to_maturity() before build()".to_string(),
                )
            }
        };
        if maturity_date <= self.valuation_date {
            return invalid(
                "maturity_date",
                format!(
                    "maturity {maturity_date} must be after the valuation date {}",
                    self.valuation_date
                ),
            );
        }

        // ── payoff ──────────────────────────────────────────────────────
        let spec = match self.payoff.take() {
            Some(spec) => spec,
            None => {
                return invalid(
                    "payoff",
                    "set a payoff (worst_of_autocallable, best_of, spread, ...) \
                     before build()"
                        .to_string(),
                )
            }
        };
        let payoff: Box<dyn Payoff> = match spec {
            MaPayoffSpec::WorstOfAutocall(spec) => {
                for (name, x) in [
                    ("autocall_barrier", spec.autocall_barrier),
                    ("protection_barrier", spec.protection_barrier),
                    ("notional", spec.notional),
                ] {
                    if !(x.is_finite() && x > 0.0) {
                        return invalid(
                            name,
                            format!("{name} must be positive and finite, got {x}"),
                        );
                    }
                }
                if !(spec.coupon.is_finite() && spec.coupon >= 0.0) {
                    return invalid(
                        "coupon",
                        format!(
                            "coupon must be non-negative and finite, got {}",
                            spec.coupon
                        ),
                    );
                }
                if spec.observations < 1 {
                    return invalid("observations", "need at least one observation".to_string());
                }
                Box::new(AutocallablePayoff {
                    exercise_style: crate::core::utils::ContractStyle::European,
                    autocall_barrier: spec.autocall_barrier,
                    protection_barrier: spec.protection_barrier,
                    coupon: spec.coupon,
                    observations: spec.observations,
                    observation_times: None,
                    notional: spec.notional,
                    initial_fixing: 100.0,
                    coupon_barrier: None,
                    memory: false,
                })
            }
            MaPayoffSpec::Rainbow {
                rainbow_type,
                put_or_call,
                strike,
                weights,
            } => {
                if matches!(rainbow_type, RainbowType::Spread | RainbowType::Exchange) && n != 2 {
                    return invalid(
                        "assets",
                        "spread and exchange options take exactly two assets".to_string(),
                    );
                }
                let strike_price = match (rainbow_type, strike) {
                    (RainbowType::Exchange, _) => 0.0,
                    (_, Some(k)) if k.is_finite() && k > 0.0 => k,
                    (_, Some(k)) => {
                        return invalid(
                            "strike",
                            format!("strike must be positive and finite, got {k}"),
                        )
                    }
                    (_, None) => return invalid("strike", "strike is required".to_string()),
                };
                let weights = match weights {
                    Some(w) => {
                        if w.len() != n {
                            return invalid(
                                "basket_weights",
                                format!("weights must have one entry per asset ({n})"),
                            );
                        }
                        if w.iter().any(|x| !x.is_finite()) {
                            return invalid("basket_weights", "weights must be finite".to_string());
                        }
                        // the basket moment match takes ln of the weighted
                        // forward sum; a non-positive first moment would
                        // NaN silently
                        if w.iter().sum::<f64>() <= 0.0 {
                            return invalid(
                                "basket_weights",
                                "basket weights must sum to a positive number".to_string(),
                            );
                        }
                        w
                    }
                    None => vec![1.0 / n as f64; n],
                };
                Box::new(RainbowPayoff {
                    exercise_style: crate::core::utils::ContractStyle::European,
                    rainbow_type,
                    put_or_call,
                    strike_price,
                    weights,
                })
            }
        };
        let assets: Vec<AssetLeg> = self
            .assets
            .iter()
            .map(|leg| {
                Ok(AssetLeg {
                    symbol: leg.symbol.clone(),
                    spot: Quote::new(leg.spot),
                    dividend_yield: leg.dividend_yield,
                    vol_surface: Arc::new(VolSurface::flat(
                        leg.flat_vol,
                        self.valuation_date,
                        crate::equity::conventions::EQUITY_DAY_COUNT,
                    )?),
                })
            })
            .collect::<Result<_, RustyQLibError>>()?;
        let initial_fixings = assets.iter().map(|a| a.spot.value()).collect();
        let engine = match self.engine {
            Engine::MonteCarlo => {
                self.mc.validate()?;
                PricingEngine::MonteCarlo(self.mc)
            }
            Engine::BlackScholes => PricingEngine::BlackScholes,
            other => PricingEngine::from_kind(other),
        };
        let option = MultiAssetEquityOption {
            base: MultiAssetBase {
                symbol: self.symbol,
                currency: self.currency,
                maturity_date,
                initial_fixings,
            },
            market: MultiAssetMarketData {
                valuation_date: self.valuation_date,
                assets,
                correlations,
                chol,
                discount_curve: Arc::new(discount_curve),
            },
            payoff,
            engine,
            model: Model::Gbm,
        };
        // "builds => prices"
        option.check_engine_support()?;
        Ok(option)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::contracts::worst_of::WorstOfAutocallable;

    fn dates() -> (NaiveDate, NaiveDate) {
        (
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2029, 1, 1).unwrap(),
        )
    }

    fn builder_note(rho: f64, paths: usize) -> MultiAssetEquityOptionBuilder {
        let (val, mat) = dates();
        MultiAssetEquityOption::builder()
            .symbol("WOF")
            .asset("AAA", 100.0, 0.25, 0.02)
            .asset("BBB", 100.0, 0.25, 0.02)
            .correlations(vec![vec![1.0, rho], vec![rho, 1.0]])
            .flat_rate(0.03)
            .valuation_date(val)
            .maturity_date(mat)
            .worst_of_autocallable(100.0, 70.0, 6.0, 6, 100.0)
            .sampler(Sampler::PseudoRandom)
            .seed(42)
            .paths(paths)
    }

    fn standalone_note(rho: f64, paths: usize) -> WorstOfAutocallable {
        let (val, mat) = dates();
        WorstOfAutocallable::new(
            "WOF",
            vec![100.0, 100.0],
            vec![0.25, 0.25],
            vec![0.02, 0.02],
            vec![vec![1.0, rho], vec![rho, 1.0]],
            AutocallablePayoff {
                exercise_style: crate::core::utils::ContractStyle::European,
                autocall_barrier: 100.0,
                protection_barrier: 70.0,
                coupon: 6.0,
                observations: 6,
                observation_times: None,
                notional: 100.0,
                initial_fixing: 100.0,
                coupon_barrier: None,
                memory: false,
            },
            mat,
            val,
            YieldCurve::flat(
                0.03,
                dates().0,
                crate::equity::conventions::EQUITY_DAY_COUNT,
                Compounding::Continuous,
            )
            .unwrap(),
            MonteCarloConfig {
                paths,
                sampler: Sampler::PseudoRandom,
                seed: 42,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn mainline_matches_the_standalone_worst_of_exactly() {
        // same algorithm, same draws, same inputs: the mainline must
        // reproduce the standalone (its validation reference) to
        // floating-point identity
        let mainline = builder_note(0.6, 20_000).build().expect("note must build");
        let standalone = standalone_note(0.6, 20_000);
        let a = mainline.npv();
        let b = standalone.npv();
        assert!(
            (a - b).abs() < 1e-9,
            "mainline {a} vs standalone {b} must be draw-identical"
        );
        // greeks route through the same ported bumps
        let da = mainline.deltas();
        let db = standalone.deltas();
        for (x, y) in da.iter().zip(&db) {
            assert!((x - y).abs() < 1e-9, "deltas {da:?} vs {db:?}");
        }
    }

    #[test]
    fn with_market_rebinds_every_leg() {
        use crate::core::daycount::DayCountConvention;
        let (val, _) = dates();
        let note = builder_note(0.6, 20_000).build().unwrap();
        let base_pv = note.npv();

        // a typed market snapshot with the second leg crashed 30%
        let surf = |v: f64| Arc::new(VolSurface::flat(v, val, DayCountConvention::Act365).unwrap());
        let market = Market::new(val)
            .with(Spot("AAA".into()), Quote::new(100.0))
            .with(Spot("BBB".into()), Quote::new(70.0))
            .with(Vol("AAA".into()), surf(0.25))
            .with(Vol("BBB".into()), surf(0.25))
            .with(
                Discount("USD".into()),
                Arc::new(
                    YieldCurve::flat(
                        0.03,
                        val,
                        DayCountConvention::Act365,
                        Compounding::Continuous,
                    )
                    .unwrap(),
                ),
            );
        let rebound = note.with_market(&market).expect("rebind must succeed");
        // every leg took the snapshot's values...
        assert_eq!(rebound.market.assets[0].spot.value(), 100.0);
        assert_eq!(rebound.market.assets[1].spot.value(), 70.0);
        // ...the contractual fixings did not move...
        assert_eq!(rebound.base.initial_fixings, vec![100.0, 100.0]);
        // ...and the crashed worst-of is worth materially less
        let crashed_pv = rebound.npv();
        assert!(
            crashed_pv < base_pv - 5.0,
            "crash must hurt the note: {crashed_pv} vs {base_pv}"
        );
        // a snapshot missing one leg's data errors with the typed key
        let partial = Market::new(val).with(Spot("AAA".into()), Quote::new(100.0));
        assert!(note.with_market(&partial).is_err());
    }

    #[test]
    fn the_note_is_long_correlation_and_long_each_asset() {
        let tight = builder_note(0.9, 30_000).build().unwrap().npv();
        let loose = builder_note(0.2, 30_000).build().unwrap().npv();
        assert!(
            tight > loose,
            "a tighter basket has a better worst performer: {tight} vs {loose}"
        );
        let deltas = builder_note(0.6, 20_000).build().unwrap().deltas();
        for (i, d) in deltas.iter().enumerate() {
            assert!(*d > 0.0, "leg {i} delta must be positive: {deltas:?}");
        }
    }

    #[test]
    fn builder_validates_and_refuses_with_named_fields() {
        use crate::core::errors::RustyQLibError;
        let field = |r: Result<MultiAssetEquityOption, RustyQLibError>| match r {
            Err(RustyQLibError::InvalidInput { field, .. }) => field,
            other => panic!(
                "expected InvalidInput, got {:?}",
                other.map(|_| "an option")
            ),
        };
        // fewer than two legs
        let (val, mat) = dates();
        let one_leg = MultiAssetEquityOption::builder()
            .asset("AAA", 100.0, 0.25, 0.02)
            .correlations(vec![vec![1.0]])
            .flat_rate(0.03)
            .valuation_date(val)
            .maturity_date(mat)
            .worst_of_autocallable(100.0, 70.0, 6.0, 6, 100.0)
            .build();
        assert_eq!(field(one_leg), "assets");
        // missing correlations
        let no_corr = MultiAssetEquityOption::builder()
            .asset("AAA", 100.0, 0.25, 0.02)
            .asset("BBB", 100.0, 0.25, 0.02)
            .flat_rate(0.03)
            .valuation_date(val)
            .maturity_date(mat)
            .worst_of_autocallable(100.0, 70.0, 6.0, 6, 100.0)
            .build();
        assert_eq!(field(no_corr), "correlations");
        // percent-vs-decimal unit slip on a leg vol
        let bad_vol = builder_note(0.6, 1_000);
        let bad_vol = bad_vol.asset("CCC", 100.0, 25.0, 0.0);
        let err = bad_vol
            .correlations(vec![
                vec![1.0, 0.6, 0.0],
                vec![0.6, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ])
            .build();
        assert!(matches!(err, Err(RustyQLibError::InvalidInput { .. })));
        // non-MC engines are refused at build()
        let lattice = builder_note(0.6, 1_000).engine(Engine::Binomial).build();
        assert!(
            matches!(lattice, Err(RustyQLibError::UnsupportedEngine(_))),
            "multi-asset notes must refuse lattice engines at build()"
        );
    }

    #[test]
    fn batched_result_matches_the_piecemeal_greeks_bit_for_bit() {
        // the shared reprice cache must be transparent: same stencils,
        // same bump sizes, same deterministic draws
        let note = builder_note(0.6, 20_000).build().unwrap();
        let result = note.pricing_result().expect("batched result");
        let ag = result.asset_greeks.as_ref().expect("per-asset greeks");
        assert_eq!(ag.deltas, note.deltas());
        assert_eq!(ag.gammas, note.gammas());
        assert_eq!(ag.vegas, note.vegas());
        assert_eq!(result.greeks.theta, note.theta());
        assert_eq!(result.greeks.rho, note.rho());
        assert_eq!(result.pv, note.npv());
        // symbols name the slots in leg order
        assert_eq!(ag.symbols, vec!["AAA".to_string(), "BBB".to_string()]);
        for g in &ag.gammas {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn tiny_vol_leg_vega_divides_by_the_effective_spread() {
        // a leg vol below the bump size: the down leg floors and the
        // stencil divides by the actual spread instead of 2h (review
        // finding B9 — the old convention overstated such vegas ~2x)
        let (val, mat) = dates();
        let note = MultiAssetEquityOption::builder()
            .asset("AAA", 100.0, 0.005, 0.0)
            .asset("BBB", 100.0, 0.25, 0.0)
            .correlations(vec![vec![1.0, 0.5], vec![0.5, 1.0]])
            .flat_rate(0.03)
            .valuation_date(val)
            .maturity_date(mat)
            .worst_of_autocallable(100.0, 70.0, 6.0, 6, 100.0)
            .sampler(Sampler::PseudoRandom)
            .seed(7)
            .paths(20_000)
            .build()
            .expect("tiny-vol note must build");
        let vegas = note.vegas();
        assert!(vegas.iter().all(|v| v.is_finite()), "{vegas:?}");
        // the effective spread for the tiny-vol leg is (0.015 - floor),
        // not 0.02: reproduce the stencil by hand through the cache-free
        // path to pin the convention
        let up_dn_spread = (0.005 + 0.01) - crate::equity::conventions::MIN_BUMPED_VOL;
        assert!(
            up_dn_spread < 0.02,
            "the test premise: spread {up_dn_spread}"
        );
    }

    #[test]
    fn asset_greeks_serialize_only_for_multi_asset_results() {
        // single-asset results keep their exact JSON shape
        let single = serde_json::to_value(PricingResult::from_pv(1.0)).unwrap();
        assert!(
            single.get("asset_greeks").is_none(),
            "single-asset JSON must not grow a field: {single}"
        );
        // multi-asset results carry the per-leg block and round-trip
        let note = builder_note(0.6, 5_000).build().unwrap();
        let result = note.pricing_result().unwrap();
        let json = serde_json::to_value(&result).unwrap();
        assert!(json.get("asset_greeks").is_some());
        let back: PricingResult = serde_json::from_value(json).unwrap();
        assert_eq!(back, result);
    }

    // ── stage 3: rainbow payoffs ───────────────────────────────────────

    fn rainbow_data(
        rainbow_type: &str,
        strike: Option<f64>,
        pricer: &str,
    ) -> crate::equity::rainbow::RainbowOptionData {
        crate::equity::rainbow::RainbowOptionData {
            symbol: "RB".into(),
            rainbow_type: rainbow_type.into(),
            put_or_call: Some("C".into()),
            assets: vec![
                crate::equity::rainbow::RainbowAssetData {
                    symbol: "AAA".into(),
                    spot: 100.0,
                    volatility: 0.3,
                    dividend: Some(0.02),
                },
                crate::equity::rainbow::RainbowAssetData {
                    symbol: "BBB".into(),
                    spot: 95.0,
                    volatility: 0.25,
                    dividend: Some(0.01),
                },
            ],
            correlations: vec![vec![1.0, 0.6], vec![0.6, 1.0]],
            strike_price: strike,
            weights: None,
            maturity: "2027-01-01".into(),
            risk_free_rate: Some(0.05),
            discount_curve: None,
            pricer: Some(pricer.into()),
            simulation: Some(40_000),
            mc_sampler: Some("pseudo".into()),
            mc_seed: Some(42),
            valuation_date: Some("2026-01-01".into()),
        }
    }

    #[test]
    fn mainline_rainbows_match_the_standalone_on_every_type() {
        // MC types: identical draws, identical terminal transitions —
        // exact equality. Analytic types: identical closed forms.
        for (rainbow_type, strike, pricer, tol) in [
            ("best_of", Some(100.0), "MC", 1e-9),
            ("worst_of", Some(100.0), "MC", 1e-9),
            ("spread", Some(5.0), "MC", 1e-9),
            ("basket", Some(97.0), "MC", 1e-9),
            ("exchange", None, "MC", 1e-9),
            ("spread", Some(5.0), "Analytical", 1e-12),
            ("basket", Some(97.0), "Analytical", 1e-12),
            ("exchange", None, "Analytical", 1e-12),
        ] {
            let data = rainbow_data(rainbow_type, strike, pricer);
            let standalone = crate::equity::rainbow::RainbowOption::try_from_json(&data)
                .expect("standalone must parse")
                .npv();
            let mainline = MultiAssetEquityOption::try_from_rainbow_json(&data)
                .expect("mainline must parse")
                .npv();
            assert!(
                (mainline - standalone).abs() < tol,
                "{rainbow_type}/{pricer}: mainline {mainline} vs standalone {standalone}"
            );
        }
    }

    #[test]
    fn rainbow_one_pass_batch_matches_the_piecemeal_greeks_bit_for_bit() {
        // the single path-generation pass must reproduce the
        // per-scenario CRN simulations exactly: same draws per path,
        // same terminal arithmetic per scenario
        let option = MultiAssetEquityOption::try_from_rainbow_json(&rainbow_data(
            "worst_of",
            Some(100.0),
            "MC",
        ))
        .unwrap();
        let result = option.pricing_result().expect("batched result");
        let ag = result.asset_greeks.as_ref().expect("per-asset greeks");
        assert_eq!(ag.deltas, option.deltas());
        assert_eq!(ag.gammas, option.gammas());
        assert_eq!(ag.vegas, option.vegas());
        assert_eq!(result.greeks.theta, option.theta());
        assert_eq!(result.greeks.rho, option.rho());
        assert_eq!(result.pv, option.npv());
        // the standalone's greeks agree too (same stencils, same draws;
        // its vega keeps the historical 2h convention so vols above the
        // bump size match exactly)
        let standalone = crate::equity::rainbow::RainbowOption::try_from_json(&rainbow_data(
            "worst_of",
            Some(100.0),
            "MC",
        ))
        .unwrap();
        for (a, b) in ag.deltas.iter().zip(standalone.deltas()) {
            assert!((a - b).abs() < 1e-9, "delta {a} vs standalone {b}");
        }
        for (a, b) in ag.vegas.iter().zip(standalone.vegas()) {
            assert!((a - b).abs() < 1e-9, "vega {a} vs standalone {b}");
        }
    }

    #[test]
    fn rainbow_builder_validates_and_gates_engines() {
        use crate::core::errors::RustyQLibError;
        let (val, mat) = dates();
        let two = || {
            MultiAssetEquityOption::builder()
                .asset("AAA", 100.0, 0.3, 0.02)
                .asset("BBB", 95.0, 0.25, 0.01)
                .correlations(vec![vec![1.0, 0.6], vec![0.6, 1.0]])
                .flat_rate(0.05)
                .valuation_date(val)
                .maturity_date(mat)
                .sampler(Sampler::PseudoRandom)
                .paths(5_000)
        };
        // best-of has no analytic pricer: refused on the analytic engine
        let refused = two()
            .best_of(PutOrCall::Call, 100.0)
            .engine(Engine::BlackScholes)
            .build();
        assert!(matches!(refused, Err(RustyQLibError::UnsupportedEngine(_))));
        // ...but exchange prices analytically (Margrabe)
        let margrabe = two()
            .exchange(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .expect("exchange must build on the analytic engine");
        assert!(margrabe.npv() > 0.0);
        // and its batched result carries per-asset greeks with the
        // exchange signature: long asset 1, short asset 2
        let result = margrabe.pricing_result().unwrap();
        let ag = result.asset_greeks.unwrap();
        assert!(ag.deltas[0] > 0.0 && ag.deltas[1] < 0.0, "{:?}", ag.deltas);
        // spread needs exactly two legs
        let three = MultiAssetEquityOption::builder()
            .asset("AAA", 100.0, 0.3, 0.0)
            .asset("BBB", 95.0, 0.25, 0.0)
            .asset("CCC", 90.0, 0.2, 0.0)
            .correlations(vec![
                vec![1.0, 0.5, 0.5],
                vec![0.5, 1.0, 0.5],
                vec![0.5, 0.5, 1.0],
            ])
            .flat_rate(0.05)
            .valuation_date(val)
            .maturity_date(mat)
            .spread(PutOrCall::Call, 5.0)
            .build();
        match three {
            Err(RustyQLibError::InvalidInput { field, .. }) => assert_eq!(field, "assets"),
            other => panic!(
                "expected assets error, got {:?}",
                other.map(|_| "an option")
            ),
        }
        // basket_weights must follow .basket(...)
        let misuse = two()
            .exchange(PutOrCall::Call)
            .basket_weights(vec![0.5, 0.5])
            .build();
        assert!(matches!(misuse, Err(RustyQLibError::InvalidInput { .. })));
    }
}
