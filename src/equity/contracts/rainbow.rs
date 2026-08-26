//! Rainbow (multi-asset) options: best-of, worst-of, spread, basket and
//! exchange payoffs on n correlated lognormal assets.
//!
//! Engines:
//! - **Analytic**: Margrabe (exchange, exact), Kirk's approximation
//!   (spread), moment-matched lognormal (basket). Best-of / worst-of have
//!   no analytic pricer yet (Stulz for n = 2 is future work) and price on
//!   Monte Carlo.
//! - **Monte Carlo**: correlated terminal GBM (Cholesky), low-discrepancy
//!   or antithetic pseudo-random sampling, deterministic parallel
//!   reduction, standard errors.
//!
//! Greeks: per-asset `deltas` and `vegas` by common-random-number bumps;
//! scalar theta and rho. Each asset carries a flat vol; per-asset smiles
//! for multi-asset payoffs are future work.
//!
//! The standalone [`RainbowOption`] is the flat-market **validation
//! reference**: production pricing (curve discounting, per-leg
//! surfaces, market rebinding, the one-pass batched Greeks) lives on
//! [`MultiAssetEquityOption`](super::multi_asset::MultiAssetEquityOption),
//! which the JSON service routes through. [`RainbowPayoff`] and the
//! shared closed forms in this file are what the mainline consumes.

use chrono::NaiveDate;
use libm::exp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::montecarlo::path_normals;
use crate::core::results::{Greeks, PricingResult};
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::utils::norm_cdf;
use crate::equity::montecarlo::{McStats, MonteCarloConfig, Sampler};
use crate::equity::utils::PricingEngine;

const PATH_CHUNK: usize = 4096;

// ── Contract data (JSON) ────────────────────────────────────────────────

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RainbowAssetData {
    pub symbol: String,
    pub spot: f64,
    pub volatility: f64,
    pub dividend: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RainbowOptionData {
    pub symbol: String,
    /// "best_of" | "worst_of" | "spread" | "basket" | "exchange"
    pub rainbow_type: String,
    pub put_or_call: Option<String>,
    pub assets: Vec<RainbowAssetData>,
    /// Full correlation matrix, n x n.
    pub correlations: Vec<Vec<f64>>,
    pub strike_price: Option<f64>,
    /// Basket weights (defaults to equal weights).
    pub weights: Option<Vec<f64>>,
    pub maturity: String,
    pub risk_free_rate: Option<f64>,
    pub discount_curve: Option<crate::core::curves::CurveInput>,
    pub pricer: Option<String>,
    pub simulation: Option<u64>,
    pub mc_sampler: Option<String>,
    pub mc_seed: Option<u64>,
    /// Pricing as-of date (`YYYY-MM-DD`); defaults to today.
    pub valuation_date: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainbowType {
    BestOf,
    WorstOf,
    Spread,
    Basket,
    Exchange,
}

// ── Instrument ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct RainbowOption {
    pub symbol: String,
    pub rainbow_type: RainbowType,
    pub put_or_call: PutOrCall,
    pub spots: Vec<f64>,
    pub vols: Vec<f64>,
    pub dividends: Vec<f64>,
    pub correlations: Vec<Vec<f64>>,
    pub strike_price: f64,
    pub weights: Vec<f64>,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub discount_curve: YieldCurve,
    /// The numerical method with its settings. Rainbow payoffs price on
    /// the analytic engine (Margrabe / Kirk / moment matching) or Monte
    /// Carlo (terminal correlated GBM: `paths`, `sampler` and `seed` from
    /// the config; `time_steps`/`scheme` do not apply).
    pub engine: PricingEngine,
    /// Cholesky factor of the correlation matrix (lower triangular).
    chol: Vec<Vec<f64>>,
}

/// Terminal payoff of one rainbow type on realized asset levels —
/// shared by the standalone product and [`RainbowPayoff`].
pub(crate) fn rainbow_terminal_value(
    rainbow_type: RainbowType,
    put_or_call: PutOrCall,
    strike: f64,
    weights: &[f64],
    terminal: &[f64],
) -> f64 {
    let phi = match put_or_call {
        PutOrCall::Call => 1.0,
        PutOrCall::Put => -1.0,
    };
    let k = strike;
    match rainbow_type {
        RainbowType::BestOf => {
            let best = terminal.iter().cloned().fold(f64::MIN, f64::max);
            (phi * (best - k)).max(0.0)
        }
        RainbowType::WorstOf => {
            let worst = terminal.iter().cloned().fold(f64::MAX, f64::min);
            (phi * (worst - k)).max(0.0)
        }
        RainbowType::Spread => (phi * (terminal[0] - terminal[1] - k)).max(0.0),
        RainbowType::Basket => {
            let basket: f64 = weights.iter().zip(terminal).map(|(w, s)| w * s).sum();
            (phi * (basket - k)).max(0.0)
        }
        RainbowType::Exchange => match put_or_call {
            PutOrCall::Call => (terminal[0] - terminal[1]).max(0.0),
            PutOrCall::Put => (terminal[1] - terminal[0]).max(0.0),
        },
    }
}

/// Margrabe (1978), exact: exchange option pays (S1 - S2)^+
/// (mirrored for puts).
pub(crate) fn margrabe_price(
    spots: &[f64],
    dividends: &[f64],
    vols: &[f64],
    rho: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    let (i, j) = match put_or_call {
        PutOrCall::Call => (0, 1),
        PutOrCall::Put => (1, 0),
    };
    let sigma = (vols[i] * vols[i] + vols[j] * vols[j] - 2.0 * rho * vols[i] * vols[j]).sqrt();
    let (q_i, q_j) = (dividends[i], dividends[j]);
    let st = sigma * t.sqrt();
    if st < 1e-12 {
        // perfectly correlated identical dynamics: the exchange is
        // deterministic — discounted positive forward difference
        return (spots[i] * exp(-q_i * t) - spots[j] * exp(-q_j * t)).max(0.0);
    }
    let d1 = ((spots[i] / spots[j]).ln() + (q_j - q_i + 0.5 * sigma * sigma) * t) / st;
    let d2 = d1 - st;
    spots[i] * exp(-q_i * t) * norm_cdf(d1) - spots[j] * exp(-q_j * t) * norm_cdf(d2)
}

/// Kirk's (1995) approximation for spread options (S1 - S2 - K)^+.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kirk_price(
    spots: &[f64],
    dividends: &[f64],
    vols: &[f64],
    rho: f64,
    r: f64,
    t: f64,
    strike: f64,
    put_or_call: PutOrCall,
) -> f64 {
    let f1 = spots[0] * exp((r - dividends[0]) * t);
    let f2 = spots[1] * exp((r - dividends[1]) * t);
    let k = strike;
    let w = f2 / (f2 + k);
    let sigma =
        (vols[0] * vols[0] - 2.0 * rho * vols[0] * vols[1] * w + vols[1] * vols[1] * w * w).sqrt();
    let st = sigma * t.sqrt();
    let d1 = ((f1 / (f2 + k)).ln() + 0.5 * sigma * sigma * t) / st;
    let d2 = d1 - st;
    let df = exp(-r * t);
    match put_or_call {
        PutOrCall::Call => df * (f1 * norm_cdf(d1) - (f2 + k) * norm_cdf(d2)),
        PutOrCall::Put => df * ((f2 + k) * norm_cdf(-d2) - f1 * norm_cdf(-d1)),
    }
}

/// Lognormal moment matching for basket options (Levy /
/// Turnbull-Wakeman style): match the basket forward's first two
/// moments, price with Black's formula.
#[allow(clippy::too_many_arguments)]
pub(crate) fn basket_moment_match_price(
    spots: &[f64],
    dividends: &[f64],
    vols: &[f64],
    correlations: &[Vec<f64>],
    weights: &[f64],
    r: f64,
    t: f64,
    strike: f64,
    put_or_call: PutOrCall,
) -> f64 {
    let n = spots.len();
    let fwds: Vec<f64> = (0..n)
        .map(|i| weights[i] * spots[i] * exp((r - dividends[i]) * t))
        .collect();
    let m1: f64 = fwds.iter().sum();
    let mut m2 = 0.0;
    for i in 0..n {
        for j in 0..n {
            m2 += fwds[i] * fwds[j] * exp(correlations[i][j] * vols[i] * vols[j] * t);
        }
    }
    let log_var = (m2 / (m1 * m1)).ln().max(1e-12);
    let sqrt_v = log_var.sqrt();
    let k = strike;
    let d1 = ((m1 / k).ln() + 0.5 * log_var) / sqrt_v;
    let d2 = d1 - sqrt_v;
    let df = exp(-r * t);
    match put_or_call {
        PutOrCall::Call => df * (m1 * norm_cdf(d1) - k * norm_cdf(d2)),
        PutOrCall::Put => df * (k * norm_cdf(-d2) - m1 * norm_cdf(-d1)),
    }
}

/// The rainbow payoffs as a mainline [`Payoff`], priced inside
/// [`MultiAssetEquityOption`](super::multi_asset::MultiAssetEquityOption)
/// on the terminal correlated-GBM route (best-of / worst-of / spread /
/// basket / exchange) or the analytic engine (Margrabe / Kirk / moment
/// matching). Terminal-only: the value is a function of the realized
/// levels at maturity, evaluated through
/// [`terminal_value`](Self::terminal_value). The standalone
/// [`RainbowOption`] remains the flat-market validation reference.
#[derive(Debug, Clone)]
pub struct RainbowPayoff {
    pub exercise_style: crate::core::utils::ContractStyle,
    pub rainbow_type: RainbowType,
    pub put_or_call: PutOrCall,
    /// Strike (unused by exchange options).
    pub strike_price: f64,
    /// Basket weights in leg order (equal weights by default); unused
    /// by the other types.
    pub weights: Vec<f64>,
}

impl RainbowPayoff {
    /// Payoff on the realized terminal levels, one value per leg.
    pub fn terminal_value(&self, terminal: &[f64]) -> f64 {
        rainbow_terminal_value(
            self.rainbow_type,
            self.put_or_call,
            self.strike_price,
            &self.weights,
            terminal,
        )
    }
}

impl crate::equity::utils::Payoff for RainbowPayoff {
    /// Degenerate single-asset value: zero (the payoff needs every
    /// leg's terminal level).
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, _path: &[f64], _strike: f64) -> f64 {
        panic!(
            "Rainbow payoffs read every leg's terminal level and cannot be              valued through a single-asset path; the multi-asset engines              price them via terminal_value"
        );
    }
    fn is_path_dependent(&self) -> bool {
        false
    }
    fn payoff_kind(&self) -> crate::equity::utils::PayoffType {
        crate::equity::utils::PayoffType::Rainbow
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &crate::core::utils::ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn crate::equity::utils::Payoff> {
        Box::new(self.clone())
    }
}

impl Instrument for RainbowOption {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        self.check_engine_support()?;
        Ok(match self.engine {
            PricingEngine::BlackScholes => self.analytic_npv_with(&self.params()),
            _ => self.mc_stats_with(&self.params()).pv,
        })
    }

    /// Value, scalar theta/rho and (under Monte Carlo) the standard
    /// error. Spot Greeks are per-asset for rainbows — see
    /// [`RainbowOption::deltas`] and [`RainbowOption::vegas`] — so the
    /// scalar delta/gamma/vega slots stay zero.
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        self.check_engine_support()?;
        let (pv, std_err) = match self.engine {
            PricingEngine::MonteCarlo(_) => {
                let stats = self.mc_stats_with(&self.params());
                (stats.pv, stats.std_err)
            }
            _ => (self.try_npv()?, None),
        };
        Ok(PricingResult {
            pv,
            greeks: Greeks {
                theta: self.theta(),
                rho: self.rho(),
                ..Default::default()
            },
            std_err,
            asset_greeks: None,
        })
    }
}

/// Market snapshot bumped by the Greeks (common random numbers).
#[derive(Clone)]
struct Params {
    spots: Vec<f64>,
    vols: Vec<f64>,
    r: f64,
    t: f64,
}

impl RainbowOption {
    /// Monte Carlo settings. Invariant: only called on the Monte Carlo
    /// code paths (the engine dispatch guarantees it).
    fn mc_cfg(&self) -> &MonteCarloConfig {
        match &self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`RainbowOption::try_from_json`].
    pub fn from_json(data: &RainbowOptionData) -> Box<RainbowOption> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &RainbowOptionData) -> Result<Box<RainbowOption>, RustyQLibError> {
        let valuation_date =
            crate::core::data_models::parse_valuation_date(data.valuation_date.as_deref())?;
        let n = data.assets.len();
        if n < 2 {
            return Err(RustyQLibError::invalid_input(
                "assets",
                "rainbow options need at least two assets",
            ));
        }
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
        if matches!(rainbow_type, RainbowType::Spread | RainbowType::Exchange) && n != 2 {
            return Err(RustyQLibError::invalid_input(
                "assets",
                "spread and exchange options take exactly two assets",
            ));
        }
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
        let strike_price = data.strike_price.unwrap_or(0.0);
        if rainbow_type != RainbowType::Exchange && data.strike_price.is_none() {
            return Err(RustyQLibError::invalid_input(
                "strike_price",
                "strike_price is required",
            ));
        }
        let weights = match &data.weights {
            Some(w) => {
                if w.len() != n {
                    return Err(RustyQLibError::invalid_input(
                        "weights",
                        "weights must match the number of assets",
                    ));
                }
                if w.iter().any(|x| !x.is_finite()) {
                    return Err(RustyQLibError::invalid_input(
                        "weights",
                        "weights must be finite",
                    ));
                }
                // the basket moment match takes ln of the weighted
                // forward sum; a non-positive first moment would NaN
                // silently
                if rainbow_type == RainbowType::Basket && w.iter().sum::<f64>() <= 0.0 {
                    return Err(RustyQLibError::invalid_input(
                        "weights",
                        "basket weights must sum to a positive number",
                    ));
                }
                w.clone()
            }
            None => vec![1.0 / n as f64; n],
        };
        if data.correlations.len() != n || data.correlations.iter().any(|row| row.len() != n) {
            return Err(RustyQLibError::invalid_input(
                "correlations",
                "correlations must be an n x n matrix",
            ));
        }
        // an empirical / hand-stressed matrix that fails PSD is repaired
        // with Higham's nearest-correlation projection; asymmetry or a
        // non-unit diagonal is a data error and still rejected
        let chol = crate::core::linalg::cholesky_with_repair(&data.correlations)?;
        for (i, a) in data.assets.iter().enumerate() {
            crate::equity::conventions::check_vol_band(
                &format!("assets[{i}].volatility"),
                a.volatility,
            )?;
        }
        if let Some(r) = data.risk_free_rate {
            crate::equity::conventions::check_rate_band("risk_free_rate", r)?;
        }
        let discount_curve = match &data.discount_curve {
            Some(input) => YieldCurve::from_input(input, valuation_date)?,
            None => YieldCurve::flat(
                data.risk_free_rate.ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "risk_free_rate",
                        "either risk_free_rate or discount_curve must be provided",
                    )
                })?,
                valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )?,
        };
        let maturity_date =
            NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    "maturity",
                    format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
                )
            })?;
        Ok(Box::new(RainbowOption {
            symbol: data.symbol.clone(),
            rainbow_type,
            put_or_call,
            spots: data.assets.iter().map(|a| a.spot).collect(),
            vols: data.assets.iter().map(|a| a.volatility).collect(),
            dividends: data
                .assets
                .iter()
                .map(|a| a.dividend.unwrap_or(0.0))
                .collect(),
            correlations: data.correlations.clone(),
            strike_price,
            weights,
            maturity_date,
            valuation_date,
            discount_curve,
            engine: match data.pricer.as_deref().map_or("MC", |v| v).trim() {
                "Analytical" | "analytical" => PricingEngine::BlackScholes,
                "MonteCarlo" | "montecarlo" | "MC" | "mc" => {
                    PricingEngine::MonteCarlo(MonteCarloConfig {
                        paths: data.simulation.unwrap_or(100_000) as usize,
                        sampler: data
                            .mc_sampler
                            .as_deref()
                            .map(|s| {
                                s.parse::<Sampler>().map_err(|_| {
                                    RustyQLibError::invalid_input(
                                        "mc_sampler",
                                        format!("invalid mc_sampler '{s}'"),
                                    )
                                })
                            })
                            .transpose()?
                            .unwrap_or(Sampler::Sobol),
                        seed: data.mc_seed.unwrap_or(42),
                        ..Default::default()
                    })
                }
                other => {
                    return Err(RustyQLibError::invalid_input(
                        "pricer",
                        format!("invalid pricer '{other}' for rainbow (Analytical or MC)"),
                    ))
                }
            },
            chol,
        }))
    }

    pub fn time_to_maturity(&self) -> f64 {
        crate::equity::conventions::year_fraction(self.valuation_date, self.maturity_date)
    }

    fn params(&self) -> Params {
        let t = self.time_to_maturity();
        Params {
            spots: self.spots.clone(),
            vols: self.vols.clone(),
            r: self
                .discount_curve
                .zero_rate_with(t, Compounding::Continuous),
            t,
        }
    }

    /// Terminal payoff on realized asset levels.
    fn payoff(&self, terminal: &[f64]) -> f64 {
        rainbow_terminal_value(
            self.rainbow_type,
            self.put_or_call,
            self.strike_price,
            &self.weights,
            terminal,
        )
    }

    // ── Pricing ─────────────────────────────────────────────────────────

    /// Reject engine/payoff combinations the library cannot price,
    /// with an error naming the engine that can.
    pub(crate) fn check_engine_support(&self) -> Result<(), RustyQLibError> {
        match &self.engine {
            PricingEngine::BlackScholes => {
                if matches!(
                    self.rainbow_type,
                    RainbowType::BestOf | RainbowType::WorstOf
                ) {
                    return Err(RustyQLibError::UnsupportedEngine(
                        "best-of / worst-of rainbows have no analytic pricer yet \
                         (Stulz for two assets is future work); use MonteCarlo"
                            .to_string(),
                    ));
                }
                Ok(())
            }
            PricingEngine::MonteCarlo(_) => Ok(()),
            other => Err(RustyQLibError::UnsupportedEngine(format!(
                "rainbow options price on the Analytical or MonteCarlo engines, not {:?}",
                other.kind()
            ))),
        }
    }

    pub fn npv_with_stats(&self) -> Option<McStats> {
        match self.engine {
            PricingEngine::MonteCarlo(_) => Some(self.mc_stats_with(&self.params())),
            _ => None,
        }
    }

    fn price_with(&self, p: &Params) -> f64 {
        match self.engine {
            PricingEngine::BlackScholes => self.analytic_npv_with(p),
            _ => self.mc_stats_with(p).pv,
        }
    }

    /// Per-asset spot deltas (central bumps, common random numbers).
    pub fn deltas(&self) -> Vec<f64> {
        let base = self.params();
        (0..self.spots.len())
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

    /// Per-asset vegas (central bumps of each asset's vol).
    pub fn vegas(&self) -> Vec<f64> {
        let base = self.params();
        (0..self.vols.len())
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
        up.r += h;
        let mut dn = base.clone();
        dn.r -= h;
        (self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
    }

    // ── Analytic pricers ────────────────────────────────────────────────

    fn analytic_npv_with(&self, p: &Params) -> f64 {
        match self.rainbow_type {
            RainbowType::Exchange => self.margrabe(p),
            RainbowType::Spread => self.kirk(p),
            RainbowType::Basket => self.basket_moment_match(p),
            // invariant: check_engine_support refuses these before pricing
            RainbowType::BestOf | RainbowType::WorstOf => {
                unreachable!("best-of/worst-of on the analytic engine is rejected before pricing")
            }
        }
    }

    fn margrabe(&self, p: &Params) -> f64 {
        margrabe_price(
            &p.spots,
            &self.dividends,
            &p.vols,
            self.correlations[0][1],
            p.t,
            self.put_or_call,
        )
    }

    fn kirk(&self, p: &Params) -> f64 {
        kirk_price(
            &p.spots,
            &self.dividends,
            &p.vols,
            self.correlations[0][1],
            p.r,
            p.t,
            self.strike_price,
            self.put_or_call,
        )
    }

    fn basket_moment_match(&self, p: &Params) -> f64 {
        basket_moment_match_price(
            &p.spots,
            &self.dividends,
            &p.vols,
            &self.correlations,
            &self.weights,
            p.r,
            p.t,
            self.strike_price,
            self.put_or_call,
        )
    }

    // ── Monte Carlo (correlated terminal GBM) ───────────────────────────

    fn mc_stats_with(&self, p: &Params) -> McStats {
        let n = self.spots.len();
        let t = p.t;
        let df = exp(-p.r * t);
        let sqrt_t = t.sqrt();
        let drifts: Vec<f64> = (0..n)
            .map(|i| (p.r - self.dividends[i] - 0.5 * p.vols[i] * p.vols[i]) * t)
            .collect();
        let cfg = self.mc_cfg();
        let qmc = match cfg.sampler {
            Sampler::Sobol => Some(crate::core::montecarlo::LowDiscrepancy::best(n, cfg.seed)),
            Sampler::PseudoRandom => None,
        };
        let chunks = cfg.paths.div_ceil(PATH_CHUNK);
        let partials: Vec<crate::equity::montecarlo::PathAccum> = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut eps = vec![0.0; n];
                let mut terminal = vec![0.0; n];
                let mut acc = crate::equity::montecarlo::PathAccum::default();
                for path in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(cfg.paths) {
                    match &qmc {
                        Some(seq) => seq.normals(path as u64 + 1, &mut eps),
                        None => {
                            // antithetic pairs from per-pair streams
                            path_normals(cfg.seed, (path / 2) as u64, &mut eps);
                            if path % 2 == 1 {
                                for e in eps.iter_mut() {
                                    *e = -*e;
                                }
                            }
                        }
                    }
                    for i in 0..n {
                        // z_i = sum_j L[i][j] eps_j (Cholesky-correlated)
                        let z: f64 = (0..=i).map(|j| self.chol[i][j] * eps[j]).sum();
                        terminal[i] = p.spots[i] * exp(drifts[i] + p.vols[i] * sqrt_t * z);
                    }
                    acc.push(path, df * self.payoff(&terminal));
                }
                acc
            })
            .collect();
        let acc = partials.into_iter().fold(
            crate::equity::montecarlo::PathAccum::default(),
            crate::equity::montecarlo::PathAccum::merge,
        );
        crate::equity::montecarlo::summarize(acc, cfg.paths, 1, 0.0, qmc.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::linalg::cholesky;
    use crate::equity::blackscholes::bs_price;
    use crate::equity::utils::Engine;
    use chrono::Local;

    fn two_asset(
        rainbow_type: &str,
        pc: &str,
        strike: Option<f64>,
        rho: f64,
    ) -> Box<RainbowOption> {
        RainbowOption::from_json(&RainbowOptionData {
            symbol: "RB".to_string(),
            rainbow_type: rainbow_type.to_string(),
            put_or_call: Some(pc.to_string()),
            assets: vec![
                RainbowAssetData {
                    symbol: "A".into(),
                    spot: 100.0,
                    volatility: 0.3,
                    dividend: Some(0.02),
                },
                RainbowAssetData {
                    symbol: "B".into(),
                    spot: 95.0,
                    volatility: 0.25,
                    dividend: Some(0.01),
                },
            ],
            correlations: vec![vec![1.0, rho], vec![rho, 1.0]],
            strike_price: strike,
            weights: None,
            maturity: maturity_1y(),
            risk_free_rate: Some(0.05),
            discount_curve: None,
            pricer: Some("MC".to_string()),
            simulation: Some(100_000),
            mc_sampler: None,
            mc_seed: None,
            valuation_date: None,
        })
    }

    fn maturity_1y() -> String {
        let d = Local::now().date_naive() + chrono::Duration::days(365);
        d.format("%Y-%m-%d").to_string()
    }

    #[test]
    fn unsupported_engines_error_instead_of_panicking() {
        // best-of has no analytic pricer: typed refusal, not a panic
        let mut option = two_asset("best_of", "C", Some(100.0), 0.6);
        option.engine = PricingEngine::BlackScholes;
        match option.try_npv() {
            Err(RustyQLibError::UnsupportedEngine(msg)) => {
                assert!(
                    msg.contains("MonteCarlo"),
                    "should name the right engine: {msg}"
                )
            }
            other => panic!("expected UnsupportedEngine, got {other:?}"),
        }
        // engines that never apply to rainbows are refused too
        option.engine = PricingEngine::from_kind(Engine::Binomial);
        assert!(matches!(
            option.try_npv(),
            Err(RustyQLibError::UnsupportedEngine(_))
        ));
        // and price() carries the same guarantee
        assert!(option.price().is_err());
    }

    #[test]
    fn price_reports_theta_rho_and_mc_std_err() {
        let mut option = two_asset("exchange", "C", None, 0.6);
        // pseudo sampler so a standard error is reported at all
        if let PricingEngine::MonteCarlo(cfg) = &mut option.engine {
            cfg.sampler = Sampler::PseudoRandom;
        }
        let result = option.price().unwrap();
        let se = result
            .std_err
            .expect("MC rainbow must report a standard error");
        assert!(se > 0.0 && se.is_finite());
        assert!((result.pv - option.npv()).abs() < 1e-12);
        assert_eq!(result.greeks.theta, option.theta());
        assert_eq!(result.greeks.rho, option.rho());
        assert_eq!(result.greeks.delta, 0.0, "spot Greeks are per-asset");
    }

    #[test]
    fn margrabe_matches_monte_carlo() {
        let mut option = two_asset("exchange", "C", None, 0.6);
        option.engine = PricingEngine::BlackScholes;
        let analytic = option.npv();
        option.engine = PricingEngine::from_kind(Engine::MonteCarlo);
        let mc = option.npv();
        assert!((mc - analytic).abs() < 0.05, "mc={mc} margrabe={analytic}");
        assert!(analytic > 0.0);
    }

    #[test]
    fn margrabe_vanishes_for_identical_assets() {
        let mut option = two_asset("exchange", "C", None, 1.0);
        option.spots = vec![100.0, 100.0];
        option.vols = vec![0.3, 0.3];
        option.dividends = vec![0.02, 0.02];
        option.engine = PricingEngine::BlackScholes;
        assert!(option.npv().abs() < 1e-10);
    }

    #[test]
    fn kirk_close_to_monte_carlo() {
        let mut option = two_asset("spread", "C", Some(5.0), 0.6);
        option.engine = PricingEngine::BlackScholes;
        let kirk = option.npv();
        option.engine = PricingEngine::from_kind(Engine::MonteCarlo);
        let mc = option.npv();
        // Kirk is an approximation: agreement at the few-cents level
        assert!((mc - kirk).abs() < 0.10, "mc={mc} kirk={kirk}");
    }

    #[test]
    fn spread_with_zero_strike_equals_margrabe() {
        let mut spread = two_asset("spread", "C", Some(0.0), 0.6);
        spread.engine = PricingEngine::BlackScholes;
        let mut exchange = two_asset("exchange", "C", None, 0.6);
        exchange.engine = PricingEngine::BlackScholes;
        assert!((spread.npv() - exchange.npv()).abs() < 1e-10);
    }

    #[test]
    fn basket_moment_match_close_to_monte_carlo() {
        let data = RainbowOptionData {
            symbol: "BK".into(),
            rainbow_type: "basket".into(),
            put_or_call: Some("C".into()),
            assets: vec![
                RainbowAssetData {
                    symbol: "A".into(),
                    spot: 100.0,
                    volatility: 0.3,
                    dividend: None,
                },
                RainbowAssetData {
                    symbol: "B".into(),
                    spot: 90.0,
                    volatility: 0.25,
                    dividend: None,
                },
                RainbowAssetData {
                    symbol: "C".into(),
                    spot: 110.0,
                    volatility: 0.35,
                    dividend: None,
                },
            ],
            correlations: vec![
                vec![1.0, 0.5, 0.3],
                vec![0.5, 1.0, 0.4],
                vec![0.3, 0.4, 1.0],
            ],
            strike_price: Some(100.0),
            weights: None,
            maturity: maturity_1y(),
            risk_free_rate: Some(0.05),
            discount_curve: None,
            pricer: Some("Analytical".into()),
            simulation: Some(100_000),
            mc_sampler: None,
            mc_seed: None,
            valuation_date: None,
        };
        let mut option = RainbowOption::from_json(&data);
        let analytic = option.npv();
        option.engine = PricingEngine::from_kind(Engine::MonteCarlo);
        let mc = option.npv();
        assert!(
            (mc - analytic).abs() < 0.15,
            "mc={mc} moment-match={analytic}"
        );
    }

    #[test]
    fn best_of_plus_worst_of_equals_sum_of_vanillas() {
        // max + min = S1 + S2 pathwise, so (max-K)+ + (min-K)+ = (S1-K)+ + (S2-K)+
        let k = 100.0;
        let best = two_asset("best_of", "C", Some(k), 0.6).npv();
        let worst = two_asset("worst_of", "C", Some(k), 0.6).npv();
        let t = two_asset("best_of", "C", Some(k), 0.6).time_to_maturity();
        let vanillas = bs_price(100.0, k, 0.05, 0.02, 0.3, t, PutOrCall::Call)
            + bs_price(95.0, k, 0.05, 0.01, 0.25, t, PutOrCall::Call);
        assert!(
            (best + worst - vanillas).abs() < 0.1,
            "best {best} + worst {worst} vs vanillas {vanillas}"
        );
    }

    #[test]
    fn worst_of_call_at_zero_strike_is_forward_minus_margrabe() {
        // min(S1, S2) = S2 - (S2 - S1)^+
        let worst = two_asset("worst_of", "C", Some(1e-9), 0.6);
        let t = worst.time_to_maturity();
        let worst_pv = worst.npv();
        let mut exchange_21 = two_asset("exchange", "P", None, 0.6); // pays (S2 - S1)^+
        exchange_21.engine = PricingEngine::BlackScholes;
        let expected = 95.0 * (-0.01 * t as f64).exp() - exchange_21.npv();
        assert!(
            (worst_pv - expected).abs() < 0.05,
            "{worst_pv} vs {expected}"
        );
    }

    #[test]
    fn correlation_orders_worst_of_prices() {
        // higher correlation raises the worst-of call (the min rises)
        let low = two_asset("worst_of", "C", Some(100.0), 0.0).npv();
        let high = two_asset("worst_of", "C", Some(100.0), 0.9).npv();
        assert!(high > low, "high-corr {high} must exceed low-corr {low}");
    }

    #[test]
    fn monte_carlo_is_reproducible_and_reports_stats() {
        let mut option = two_asset("worst_of", "C", Some(100.0), 0.6);
        assert_eq!(option.npv(), option.npv());
        // the default low-discrepancy sampler reports no standard error
        assert_eq!(option.npv_with_stats().unwrap().std_err, None);
        if let PricingEngine::MonteCarlo(cfg) = &mut option.engine {
            cfg.sampler = Sampler::PseudoRandom;
        }
        let stats = option.npv_with_stats().unwrap();
        let se = stats
            .std_err
            .expect("pseudo sampler reports a standard error");
        assert!(se > 0.0 && se < 0.5);
    }

    #[test]
    fn deltas_and_vegas_have_sensible_signs() {
        let option = two_asset("spread", "C", Some(5.0), 0.6);
        let deltas = option.deltas();
        assert!(deltas[0] > 0.0, "long asset 1: {deltas:?}");
        assert!(deltas[1] < 0.0, "short asset 2: {deltas:?}");
        let vegas = option.vegas();
        assert!(vegas[0] > 0.0);
    }

    #[test]
    fn cholesky_rejects_invalid_correlations() {
        assert!(cholesky(&[vec![1.0, 0.5], vec![0.4, 1.0]]).is_err()); // asymmetric
        assert!(cholesky(&[vec![2.0, 0.0], vec![0.0, 1.0]]).is_err()); // diagonal != 1
                                                                       // correlation > 1 in disguise: not positive definite
        assert!(cholesky(&[
            vec![1.0, 0.9, -0.9],
            vec![0.9, 1.0, 0.9],
            vec![-0.9, 0.9, 1.0]
        ])
        .is_err());
        assert!(cholesky(&[vec![1.0, 0.5], vec![0.5, 1.0]]).is_ok());
    }
}
