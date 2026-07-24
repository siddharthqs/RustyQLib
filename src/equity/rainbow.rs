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

use chrono::{Local, NaiveDate};
use libm::exp;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::linalg::{cholesky, nearest_correlation};
use crate::core::trade::PutOrCall;
use crate::core::utils::norm_cdf;
use crate::equity::montecarlo::{McStats, Sampler};
use crate::equity::utils::Engine;
use crate::core::montecarlo::{path_normals, QmcSequence};
use crate::core::errors::RustyQLibError;

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
    pub engine: Engine,
    pub paths: usize,
    pub sampler: Sampler,
    pub seed: u64,
    /// Cholesky factor of the correlation matrix (lower triangular).
    chol: Vec<Vec<f64>>,
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
    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`RainbowOption::try_from_json`].
    pub fn from_json(data: &RainbowOptionData) -> Box<RainbowOption> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &RainbowOptionData) -> Result<Box<RainbowOption>, RustyQLibError> {
        let valuation_date = Local::now().date_naive();
        let n = data.assets.len();
        if n < 2 {
            return Err(RustyQLibError::invalid_input("assets", "rainbow options need at least two assets"));
        }
        let rainbow_type = match data.rainbow_type.trim().to_lowercase().as_str() {
            "best_of" | "bestof" | "max" => RainbowType::BestOf,
            "worst_of" | "worstof" | "min" => RainbowType::WorstOf,
            "spread" => RainbowType::Spread,
            "basket" => RainbowType::Basket,
            "exchange" | "margrabe" => RainbowType::Exchange,
            other => return Err(RustyQLibError::invalid_input(
                "rainbow_type",
                format!("invalid rainbow_type '{other}'"),
            )),
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
            other => return Err(RustyQLibError::invalid_input(
                "put_or_call",
                format!("invalid put_or_call '{other}' (use 'C' or 'P')"),
            )),
        };
        let strike_price = data.strike_price.unwrap_or(0.0);
        if rainbow_type != RainbowType::Exchange && data.strike_price.is_none() {
            return Err(RustyQLibError::invalid_input("strike_price", "strike_price is required"));
        }
        let weights = match &data.weights {
            Some(w) => {
                if w.len() != n {
                    return Err(RustyQLibError::invalid_input(
                        "weights",
                        "weights must match the number of assets",
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
        let chol = match cholesky(&data.correlations) {
            Ok(l) => l,
            Err(RustyQLibError::NumericalError(ref msg))
                if msg.contains("positive semi-definite") =>
            {
                eprintln!(
                    "warning: correlation matrix is not PSD; \
                     projecting to the nearest correlation matrix (Higham)"
                );
                let repaired = nearest_correlation(&data.correlations, 1e-12, 200)?;
                cholesky(&repaired)?
            }
            Err(e) => return Err(e),
        };
        let discount_curve = match &data.discount_curve {
            Some(input) => YieldCurve::from_input(input, valuation_date)?,
            None => YieldCurve::flat(
                data.risk_free_rate.unwrap_or(0.0),
                valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )?,
        };
        let maturity_date = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d")
            .map_err(|_| RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            ))?;
        Ok(Box::new(RainbowOption {
            symbol: data.symbol.clone(),
            rainbow_type,
            put_or_call,
            spots: data.assets.iter().map(|a| a.spot).collect(),
            vols: data.assets.iter().map(|a| a.volatility).collect(),
            dividends: data.assets.iter().map(|a| a.dividend.unwrap_or(0.0)).collect(),
            correlations: data.correlations.clone(),
            strike_price,
            weights,
            maturity_date,
            valuation_date,
            discount_curve,
            engine: match data.pricer.as_deref().map_or("MC", |v| v).trim() {
                "Analytical" | "analytical" => Engine::BlackScholes,
                "MonteCarlo" | "montecarlo" | "MC" | "mc" => Engine::MonteCarlo,
                other => return Err(RustyQLibError::invalid_input(
                    "pricer",
                    format!("invalid pricer '{other}' for rainbow (Analytical or MC)"),
                )),
            },
            paths: data.simulation.unwrap_or(100_000) as usize,
            sampler: data
                .mc_sampler
                .as_deref()
                .map(|s| s.parse::<Sampler>().map_err(|_| RustyQLibError::invalid_input(
                    "mc_sampler",
                    format!("invalid mc_sampler '{s}'"),
                )))
                .transpose()?
                .unwrap_or(Sampler::Sobol),
            seed: data.mc_seed.unwrap_or(42),
            chol,
        }))
    }

    pub fn time_to_maturity(&self) -> f64 {
        (self.maturity_date - self.valuation_date).num_days() as f64 / 365.0
    }

    fn params(&self) -> Params {
        let t = self.time_to_maturity();
        Params {
            spots: self.spots.clone(),
            vols: self.vols.clone(),
            r: self.discount_curve.zero_rate_with(t, Compounding::Continuous),
            t,
        }
    }

    /// Terminal payoff on realized asset levels.
    fn payoff(&self, terminal: &[f64]) -> f64 {
        let phi = match self.put_or_call {
            PutOrCall::Call => 1.0,
            PutOrCall::Put => -1.0,
        };
        let k = self.strike_price;
        match self.rainbow_type {
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
                let basket: f64 =
                    self.weights.iter().zip(terminal).map(|(w, s)| w * s).sum();
                (phi * (basket - k)).max(0.0)
            }
            RainbowType::Exchange => match self.put_or_call {
                PutOrCall::Call => (terminal[0] - terminal[1]).max(0.0),
                PutOrCall::Put => (terminal[1] - terminal[0]).max(0.0),
            },
        }
    }

    // ── Pricing ─────────────────────────────────────────────────────────

    pub fn npv(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => self.analytic_npv_with(&self.params()),
            Engine::MonteCarlo => self.mc_stats_with(&self.params()).pv,
            _ => panic!("Rainbow options price on the Analytical or MonteCarlo engines"),
        }
    }

    pub fn npv_with_stats(&self) -> Option<McStats> {
        match self.engine {
            Engine::MonteCarlo => Some(self.mc_stats_with(&self.params())),
            _ => None,
        }
    }

    fn price_with(&self, p: &Params) -> f64 {
        match self.engine {
            Engine::BlackScholes => self.analytic_npv_with(p),
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
            RainbowType::BestOf | RainbowType::WorstOf => panic!(
                "best_of / worst_of have no analytic pricer yet; use the MonteCarlo engine"
            ),
        }
    }

    /// Margrabe (1978), exact: exchange option pays (S1 - S2)^+.
    fn margrabe(&self, p: &Params) -> f64 {
        let (i, j) = match self.put_or_call {
            PutOrCall::Call => (0, 1),
            PutOrCall::Put => (1, 0),
        };
        let rho = self.correlations[0][1];
        let sigma = (p.vols[i] * p.vols[i] + p.vols[j] * p.vols[j]
            - 2.0 * rho * p.vols[i] * p.vols[j])
            .sqrt();
        let (q_i, q_j) = (self.dividends[i], self.dividends[j]);
        let st = sigma * p.t.sqrt();
        if st < 1e-12 {
            // perfectly correlated identical dynamics: the exchange is
            // deterministic — discounted positive forward difference
            return (p.spots[i] * exp(-q_i * p.t) - p.spots[j] * exp(-q_j * p.t)).max(0.0);
        }
        let d1 = ((p.spots[i] / p.spots[j]).ln() + (q_j - q_i + 0.5 * sigma * sigma) * p.t) / st;
        let d2 = d1 - st;
        p.spots[i] * exp(-q_i * p.t) * norm_cdf(d1) - p.spots[j] * exp(-q_j * p.t) * norm_cdf(d2)
    }

    /// Kirk's (1995) approximation for spread options (S1 - S2 - K)^+.
    fn kirk(&self, p: &Params) -> f64 {
        let f1 = p.spots[0] * exp((p.r - self.dividends[0]) * p.t);
        let f2 = p.spots[1] * exp((p.r - self.dividends[1]) * p.t);
        let k = self.strike_price;
        let rho = self.correlations[0][1];
        let w = f2 / (f2 + k);
        let sigma = (p.vols[0] * p.vols[0] - 2.0 * rho * p.vols[0] * p.vols[1] * w
            + p.vols[1] * p.vols[1] * w * w)
            .sqrt();
        let st = sigma * p.t.sqrt();
        let d1 = ((f1 / (f2 + k)).ln() + 0.5 * sigma * sigma * p.t) / st;
        let d2 = d1 - st;
        let df = exp(-p.r * p.t);
        match self.put_or_call {
            PutOrCall::Call => df * (f1 * norm_cdf(d1) - (f2 + k) * norm_cdf(d2)),
            PutOrCall::Put => df * ((f2 + k) * norm_cdf(-d2) - f1 * norm_cdf(-d1)),
        }
    }

    /// Lognormal moment matching for basket options (Levy / Turnbull-Wakeman
    /// style): match the basket forward's first two moments, price with
    /// Black's formula.
    fn basket_moment_match(&self, p: &Params) -> f64 {
        let n = p.spots.len();
        let fwds: Vec<f64> = (0..n)
            .map(|i| self.weights[i] * p.spots[i] * exp((p.r - self.dividends[i]) * p.t))
            .collect();
        let m1: f64 = fwds.iter().sum();
        let mut m2 = 0.0;
        for i in 0..n {
            for j in 0..n {
                m2 += fwds[i]
                    * fwds[j]
                    * exp(self.correlations[i][j] * p.vols[i] * p.vols[j] * p.t);
            }
        }
        let log_var = (m2 / (m1 * m1)).ln().max(1e-12);
        let sqrt_v = log_var.sqrt();
        let k = self.strike_price;
        let d1 = ((m1 / k).ln() + 0.5 * log_var) / sqrt_v;
        let d2 = d1 - sqrt_v;
        let df = exp(-p.r * p.t);
        match self.put_or_call {
            PutOrCall::Call => df * (m1 * norm_cdf(d1) - k * norm_cdf(d2)),
            PutOrCall::Put => df * (k * norm_cdf(-d2) - m1 * norm_cdf(-d1)),
        }
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
        let qmc = match self.sampler {
            Sampler::Sobol => Some(QmcSequence::new(n, self.seed)),
            Sampler::PseudoRandom => None,
        };
        let chunks = self.paths.div_ceil(PATH_CHUNK);
        let partials: Vec<(f64, f64)> = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut eps = vec![0.0; n];
                let mut terminal = vec![0.0; n];
                let (mut sum, mut sum_sq) = (0.0, 0.0);
                for path in chunk * PATH_CHUNK..((chunk + 1) * PATH_CHUNK).min(self.paths) {
                    match &qmc {
                        Some(seq) => seq.normals(path as u64 + 1, &mut eps),
                        None => {
                            // antithetic pairs from per-pair streams
                            path_normals(self.seed, (path / 2) as u64, &mut eps);
                            if path % 2 == 1 {
                                for e in eps.iter_mut() {
                                    *e = -*e;
                                }
                            }
                        }
                    }
                    for i in 0..n {
                        // z_i = sum_j L[i][j] eps_j (Cholesky-correlated)
                        let z: f64 =
                            (0..=i).map(|j| self.chol[i][j] * eps[j]).sum();
                        terminal[i] = p.spots[i] * exp(drifts[i] + p.vols[i] * sqrt_t * z);
                    }
                    let v = df * self.payoff(&terminal);
                    sum += v;
                    sum_sq += v * v;
                }
                (sum, sum_sq)
            })
            .collect();
        let (sum, sum_sq) =
            partials.into_iter().fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let nf = self.paths as f64;
        let mean = sum / nf;
        let var = (sum_sq / nf - mean * mean).max(0.0);
        McStats { pv: mean, std_err: (var / nf).sqrt(), paths: self.paths, steps: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    fn two_asset(rainbow_type: &str, pc: &str, strike: Option<f64>, rho: f64) -> Box<RainbowOption> {
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
        })
    }

    fn maturity_1y() -> String {
        let d = Local::now().date_naive() + chrono::Duration::days(365);
        d.format("%Y-%m-%d").to_string()
    }

    #[test]
    fn margrabe_matches_monte_carlo() {
        let mut option = two_asset("exchange", "C", None, 0.6);
        option.engine = Engine::BlackScholes;
        let analytic = option.npv();
        option.engine = Engine::MonteCarlo;
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
        option.engine = Engine::BlackScholes;
        assert!(option.npv().abs() < 1e-10);
    }

    #[test]
    fn kirk_close_to_monte_carlo() {
        let mut option = two_asset("spread", "C", Some(5.0), 0.6);
        option.engine = Engine::BlackScholes;
        let kirk = option.npv();
        option.engine = Engine::MonteCarlo;
        let mc = option.npv();
        // Kirk is an approximation: agreement at the few-cents level
        assert!((mc - kirk).abs() < 0.10, "mc={mc} kirk={kirk}");
    }

    #[test]
    fn spread_with_zero_strike_equals_margrabe() {
        let mut spread = two_asset("spread", "C", Some(0.0), 0.6);
        spread.engine = Engine::BlackScholes;
        let mut exchange = two_asset("exchange", "C", None, 0.6);
        exchange.engine = Engine::BlackScholes;
        assert!((spread.npv() - exchange.npv()).abs() < 1e-10);
    }

    #[test]
    fn basket_moment_match_close_to_monte_carlo() {
        let data = RainbowOptionData {
            symbol: "BK".into(),
            rainbow_type: "basket".into(),
            put_or_call: Some("C".into()),
            assets: vec![
                RainbowAssetData { symbol: "A".into(), spot: 100.0, volatility: 0.3, dividend: None },
                RainbowAssetData { symbol: "B".into(), spot: 90.0, volatility: 0.25, dividend: None },
                RainbowAssetData { symbol: "C".into(), spot: 110.0, volatility: 0.35, dividend: None },
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
        };
        let mut option = RainbowOption::from_json(&data);
        let analytic = option.npv();
        option.engine = Engine::MonteCarlo;
        let mc = option.npv();
        assert!((mc - analytic).abs() < 0.15, "mc={mc} moment-match={analytic}");
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
        exchange_21.engine = Engine::BlackScholes;
        let expected = 95.0 * (-0.01 * t as f64).exp() - exchange_21.npv();
        assert!((worst_pv - expected).abs() < 0.05, "{worst_pv} vs {expected}");
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
        let option = two_asset("worst_of", "C", Some(100.0), 0.6);
        assert_eq!(option.npv(), option.npv());
        let stats = option.npv_with_stats().unwrap();
        assert!(stats.std_err > 0.0 && stats.std_err < 0.5);
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
