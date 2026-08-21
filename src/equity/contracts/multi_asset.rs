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
//! Stage 1 scope: correlated-GBM Monte Carlo (the engine machinery of
//! the standalone worst-of, ported verbatim so values are
//! draw-identical), with the worst-of autocallable as the proving
//! payoff. Each leg's GBM vol is its surface's ATM-forward vol at the
//! contract maturity; per-asset smiles in the *dynamics* remain future
//! work, as they were for the standalones. The standalone
//! [`WorstOfAutocallable`](super::worst_of::WorstOfAutocallable) stays
//! as the flat-market validation reference.

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
use crate::core::traits::Instrument;
use crate::core::vols::VolSurface;
use crate::equity::autocallable::AutocallablePayoff;
use crate::equity::montecarlo::{
    summarize, McStats, MonteCarloConfig, PathAccum, Sampler, PATH_DEPENDENT_MIN_STEPS,
};
use crate::equity::processes::MultiAssetGbmProcess;
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
    /// Stage 1: an [`AutocallablePayoff`] evaluated on the worst-of
    /// performance path (in `initial_fixing` units); further multi-asset
    /// payoffs plug in through the same trait object.
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
        crate::equity::conventions::year_fraction(self.market.valuation_date, self.base.maturity_date)
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

    /// Stage-1 support gate: correlated GBM on the Monte Carlo engine.
    pub(crate) fn check_engine_support(&self) -> Result<(), RustyQLibError> {
        let unsupported = |msg: &str| Err(RustyQLibError::UnsupportedEngine(msg.to_string()));
        if !matches!(self.model, Model::Gbm) {
            return unsupported(
                "multi-asset notes price under correlated GBM dynamics; multi-asset \
                 stochastic-vol models are future work",
            );
        }
        if !matches!(self.engine, PricingEngine::MonteCarlo(_)) {
            return unsupported(
                "multi-asset autocallables price on the MonteCarlo engine only",
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
    /// factors at the exact observation times.
    fn observation_grid(&self, t: f64, dr: f64, steps: usize) -> (Vec<usize>, Vec<f64>) {
        let auto = self.autocall();
        let n_obs = auto.observations.max(1);
        let (obs_idx, obs_times): (Vec<usize>, Vec<f64>) = match &auto.observation_times {
            Some(times) => {
                let mut idx = Vec::with_capacity(times.len());
                let mut prev: i64 = 0;
                for &tm in times {
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
            .map(|&tm| self.market.discount_curve.df(tm) * exp(-dr * tm))
            .collect();
        (obs_idx, dfs)
    }

    /// Price with sampling diagnostics on the base market.
    pub fn npv_with_stats(&self) -> McStats {
        self.mc_stats_with(&self.params())
    }

    /// The correlated-GBM Monte Carlo, ported draw-for-draw from the
    /// standalone worst-of so identical inputs and seeds give identical
    /// values (the standalone is the validation reference).
    fn mc_stats_with(&self, p: &Params) -> McStats {
        let n = self.market.assets.len();
        let t = p.t;
        let auto = self.autocall();
        let cfg = self.mc_cfg();
        let n_obs = auto.observations.max(1);
        // every observation lands exactly on a simulation step
        let steps = cfg
            .time_steps
            .max(PATH_DEPENDENT_MIN_STEPS)
            .div_ceil(n_obs)
            * n_obs;
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

    fn price_with(&self, p: &Params) -> f64 {
        self.mc_stats_with(p).pv
    }

    /// Per-asset spot deltas (central bumps, common random numbers).
    pub fn deltas(&self) -> Vec<f64> {
        let base = self.params();
        (0..base.spots.len())
            .map(|i| {
                let h = base.spots[i] * 0.01;
                let mut up = base.clone();
                up.spots[i] += h;
                let mut dn = base.clone();
                dn.spots[i] -= h;
                (self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
            })
            .collect()
    }

    /// Per-asset vegas (central bumps of each leg's vol).
    pub fn vegas(&self) -> Vec<f64> {
        let base = self.params();
        (0..base.vols.len())
            .map(|i| {
                let h = 0.01;
                let mut up = base.clone();
                up.vols[i] += h;
                let mut dn = base.clone();
                dn.vols[i] = (dn.vols[i] - h).max(1e-6);
                (self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
            })
            .collect()
    }

    pub fn theta(&self) -> f64 {
        let base = self.params();
        let h = (1.0 / 365.0_f64).min(0.5 * base.t);
        let mut up = base.clone();
        up.t += h;
        let mut dn = base.clone();
        dn.t -= h;
        -(self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
    }

    pub fn rho(&self) -> f64 {
        let base = self.params();
        let h = 1e-4;
        let mut up = base.clone();
        up.dr += h;
        let mut dn = base.clone();
        dn.dr -= h;
        (self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
    }
}

impl Instrument for MultiAssetEquityOption {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        self.check_engine_support()?;
        Ok(self.npv_with_stats().pv)
    }

    /// Value, scalar theta/rho and the Monte Carlo standard error. Spot
    /// Greeks are per-asset — see [`deltas`](Self::deltas) and
    /// [`vegas`](Self::vegas) — so the scalar delta/gamma/vega slots
    /// stay zero (a per-asset result slot arrives with the shared
    /// multi-asset Greek cache in stage 2).
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        self.check_engine_support()?;
        let stats = self.npv_with_stats();
        Ok(PricingResult {
            pv: stats.pv,
            greeks: crate::core::results::Greeks {
                theta: self.theta(),
                rho: self.rho(),
                ..Default::default()
            },
            std_err: stats.std_err,
        })
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
    payoff: Option<WorstOfSpec>,
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
        self.maturity_date = Some(
            self.valuation_date + chrono::Duration::days((years * 365.0).round() as i64),
        );
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
        self.payoff = Some(WorstOfSpec {
            autocall_barrier,
            protection_barrier,
            coupon,
            observations,
            notional,
        });
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
                "multi-asset options need at least two legs; add them with .asset(...)"
                    .to_string(),
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
            crate::equity::conventions::check_vol_band(&format!("vol({})", leg.symbol), leg.flat_vol)?;
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
                    "set worst_of_autocallable(...) before build()".to_string(),
                )
            }
        };
        for (name, x) in [
            ("autocall_barrier", spec.autocall_barrier),
            ("protection_barrier", spec.protection_barrier),
            ("notional", spec.notional),
        ] {
            if !(x.is_finite() && x > 0.0) {
                return invalid(name, format!("{name} must be positive and finite, got {x}"));
            }
        }
        if !(spec.coupon.is_finite() && spec.coupon >= 0.0) {
            return invalid(
                "coupon",
                format!("coupon must be non-negative and finite, got {}", spec.coupon),
            );
        }
        if spec.observations < 1 {
            return invalid("observations", "need at least one observation".to_string());
        }

        // ── materialize ─────────────────────────────────────────────────
        let payoff: Box<dyn Payoff> = Box::new(AutocallablePayoff {
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
        });
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
        let surf = |v: f64| {
            Arc::new(VolSurface::flat(v, val, DayCountConvention::Act365).unwrap())
        };
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
            other => panic!("expected InvalidInput, got {:?}", other.map(|_| "an option")),
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
}
