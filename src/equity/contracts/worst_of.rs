//! Worst-of autocallable: the structured-products flagship — an
//! autocallable note observed on the **worst performer** of a basket.
//!
//! At each observation the worst-of performance
//! `W(t) = min_i S_i(t) / S_i(0)` is compared against the barriers; the
//! coupon, autocall, knock-in and downside-participation logic is the
//! single-asset [`AutocallablePayoff`] evaluated on the worst-of path
//! expressed in `initial_fixing` units, so every payoff variant (Athena
//! accrued coupons, Phoenix conditional coupons with memory, explicit
//! observation schedules) carries over unchanged.
//!
//! Paths are the correlated multi-asset lognormal dynamics of
//! [`MultiAssetGbmProcess`] — exact joint transitions, so step count only
//! sets monitoring resolution — driven by the shared multi-factor draw
//! machinery (seeded pseudo-random antithetic pairs, or the
//! low-discrepancy sequence with one Brownian bridge per asset).
//!
//! Economics worth testing against: the note is **long correlation**
//! (a tighter basket has a better worst performer), and adding an asset
//! can only cheapen it.

use chrono::NaiveDate;
use libm::exp;
use rayon::prelude::*;

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::linalg::{cholesky, nearest_correlation};
use crate::core::montecarlo::paths::{FactorScratch, MultiDraws};
use crate::core::montecarlo::process::StochasticProcess;
use crate::core::results::PricingResult;
use crate::core::traits::Instrument;
use crate::equity::autocallable::AutocallablePayoff;
use crate::equity::montecarlo::{McStats, MonteCarloConfig, PATH_DEPENDENT_MIN_STEPS};
use crate::equity::processes::MultiAssetGbmProcess;

/// Autocallable note on the worst-of performance of a correlated basket.
pub struct WorstOfAutocallable {
    pub symbol: String,
    /// Current spots — also the contractual initial fixings that
    /// normalize the worst-of performance (the note is assumed priced
    /// from inception levels; Greek bumps move the market spot, never
    /// the fixing).
    pub spots: Vec<f64>,
    pub vols: Vec<f64>,
    pub dividends: Vec<f64>,
    pub correlations: Vec<Vec<f64>>,
    /// Redemption logic; its barriers are worst-of performance levels in
    /// `initial_fixing` units (e.g. fixing 100, autocall 100 = 100% of
    /// initial, protection 70 = 70% of initial).
    pub payoff: AutocallablePayoff,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub discount_curve: YieldCurve,
    pub mc: MonteCarloConfig,
    /// Lower-triangular Cholesky factor of the correlation matrix.
    chol: Vec<Vec<f64>>,
}

/// Market snapshot the Greeks bump (common random numbers: every reprice
/// reuses the same deterministic draws).
#[derive(Clone)]
struct Params {
    spots: Vec<f64>,
    vols: Vec<f64>,
    /// Parallel shift of the discount/drift rate (rho bumps).
    dr: f64,
    t: f64,
}

impl WorstOfAutocallable {
    /// Validate and construct: dimensions must agree, and a correlation
    /// matrix that fails PSD is repaired with Higham's projection (an
    /// asymmetric or non-unit-diagonal matrix is a data error and still
    /// rejected).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        symbol: &str,
        spots: Vec<f64>,
        vols: Vec<f64>,
        dividends: Vec<f64>,
        correlations: Vec<Vec<f64>>,
        payoff: AutocallablePayoff,
        maturity_date: NaiveDate,
        valuation_date: NaiveDate,
        discount_curve: YieldCurve,
        mc: MonteCarloConfig,
    ) -> Result<Self, RustyQLibError> {
        let n = spots.len();
        if n < 2 {
            return Err(RustyQLibError::invalid_input(
                "assets",
                "worst-of autocallables need at least two assets",
            ));
        }
        if vols.len() != n || dividends.len() != n {
            return Err(RustyQLibError::invalid_input(
                "assets",
                "spots, vols and dividends must have the same length",
            ));
        }
        if correlations.len() != n || correlations.iter().any(|row| row.len() != n) {
            return Err(RustyQLibError::invalid_input(
                "correlations",
                "correlations must be an n x n matrix",
            ));
        }
        let chol = match cholesky(&correlations) {
            Ok(l) => l,
            Err(RustyQLibError::NumericalError(ref msg))
                if msg.contains("positive semi-definite") =>
            {
                log::warn!(
                    "correlation matrix is not PSD; \
                     projecting to the nearest correlation matrix (Higham)"
                );
                let repaired = nearest_correlation(&correlations, 1e-12, 200)?;
                cholesky(&repaired)?
            }
            Err(e) => return Err(e),
        };
        Ok(WorstOfAutocallable {
            symbol: symbol.to_string(),
            spots,
            vols,
            dividends,
            correlations,
            payoff,
            maturity_date,
            valuation_date,
            discount_curve,
            mc,
            chol,
        })
    }

    pub fn time_to_maturity(&self) -> f64 {
        (self.maturity_date - self.valuation_date).num_days() as f64 / 365.0
    }

    fn params(&self) -> Params {
        Params {
            spots: self.spots.clone(),
            vols: self.vols.clone(),
            dr: 0.0,
            t: self.time_to_maturity(),
        }
    }

    /// Observation grid on a path of `steps` steps over life `t`:
    /// per-observation path indices (strictly increasing) and discount
    /// factors at the exact observation times.
    fn observation_grid(&self, t: f64, dr: f64, steps: usize) -> (Vec<usize>, Vec<f64>) {
        let n_obs = self.payoff.observations.max(1);
        let (obs_idx, obs_times): (Vec<usize>, Vec<f64>) = match &self.payoff.observation_times {
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
            .map(|&tm| self.discount_curve.df(tm) * exp(-dr * tm))
            .collect();
        (obs_idx, dfs)
    }

    pub fn npv_with_stats(&self) -> McStats {
        self.mc_stats_with(&self.params())
    }

    fn mc_stats_with(&self, p: &Params) -> McStats {
        let n = self.spots.len();
        let t = p.t;
        let n_obs = self.payoff.observations.max(1);
        // every observation lands exactly on a simulation step
        let steps = self
            .mc
            .time_steps
            .max(PATH_DEPENDENT_MIN_STEPS)
            .div_ceil(n_obs)
            * n_obs;
        let dt = t / steps as f64;
        let (obs_idx, dfs) = self.observation_grid(t, p.dr, steps);
        let r = self
            .discount_curve
            .zero_rate_with(t, Compounding::Continuous)
            + p.dr;
        let process = MultiAssetGbmProcess {
            drift_rates: self.dividends.iter().map(|q| r - q).collect(),
            vols: p.vols.clone(),
            chol: self.chol.clone(),
        };
        let draws = MultiDraws::new(self.mc.sampler, self.mc.seed, n, steps, dt);
        let fixing = self.payoff.initial_fixing;

        const CHUNK: usize = 4096;
        let chunks = self.mc.paths.div_ceil(CHUNK);
        let partials: Vec<(f64, f64)> = (0..chunks)
            .into_par_iter()
            .map(|chunk| {
                let mut scratch = FactorScratch::new(n, steps);
                let mut dw = vec![0.0; n * steps];
                let mut x = vec![0.0; n];
                let mut x_next = vec![0.0; n];
                let mut worst = vec![0.0; steps];
                let (mut sum, mut sum_sq) = (0.0, 0.0);
                for i in chunk * CHUNK..((chunk + 1) * CHUNK).min(self.mc.paths) {
                    draws.fill(i, n, steps, &mut scratch, &mut dw);
                    x.copy_from_slice(&p.spots);
                    for j in 0..steps {
                        process.evolve(j as f64 * dt, &x, dt, &dw[j * n..(j + 1) * n], &mut x_next);
                        x.copy_from_slice(&x_next);
                        // worst-of performance in initial_fixing units,
                        // normalized by the *contractual* fixings
                        // (self.spots): market bumps move the path start,
                        // never the denominators — otherwise delta would
                        // cancel to zero by homogeneity
                        let w = x
                            .iter()
                            .zip(&self.spots)
                            .map(|(s, s0)| s / s0)
                            .fold(f64::MAX, f64::min);
                        worst[j] = fixing * w;
                    }
                    let v = self.payoff.path_value(&worst, &obs_idx, &dfs);
                    sum += v;
                    sum_sq += v * v;
                }
                (sum, sum_sq)
            })
            .collect();
        let (sum, sum_sq) = partials
            .into_iter()
            .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        let nf = self.mc.paths as f64;
        let mean = sum / nf;
        let var = (sum_sq / nf - mean * mean).max(0.0);
        McStats {
            pv: mean,
            std_err: (var / nf).sqrt(),
            paths: self.mc.paths,
            steps,
        }
    }

    fn price_with(&self, p: &Params) -> f64 {
        self.mc_stats_with(p).pv
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
        up.dr += h;
        let mut dn = base.clone();
        dn.dr -= h;
        (self.price_with(&up) - self.price_with(&dn)) / (2.0 * h)
    }
}

impl Instrument for WorstOfAutocallable {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.npv_with_stats().pv)
    }

    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        let stats = self.npv_with_stats();
        Ok(PricingResult {
            pv: stats.pv,
            greeks: crate::core::results::Greeks {
                theta: self.theta(),
                rho: self.rho(),
                ..Default::default()
            },
            std_err: Some(stats.std_err),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use crate::core::utils::ContractStyle;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::montecarlo::Sampler;
    use crate::equity::utils::Engine;

    fn dates() -> (NaiveDate, NaiveDate) {
        (
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            NaiveDate::from_ymd_opt(2029, 1, 1).unwrap(),
        )
    }

    fn payoff() -> AutocallablePayoff {
        AutocallablePayoff {
            exercise_style: ContractStyle::European,
            autocall_barrier: 100.0,
            protection_barrier: 70.0,
            coupon: 6.0,
            observations: 6,
            observation_times: None,
            notional: 100.0,
            initial_fixing: 100.0,
            coupon_barrier: None,
            memory: false,
        }
    }

    fn note(n: usize, rho: f64, paths: usize) -> WorstOfAutocallable {
        let (val, mat) = dates();
        let correlations: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { rho }).collect())
            .collect();
        WorstOfAutocallable::new(
            "WOF",
            vec![100.0; n],
            vec![0.25; n],
            vec![0.02; n],
            correlations,
            payoff(),
            mat,
            val,
            YieldCurve::flat(
                0.03,
                val,
                DayCountConvention::Act365,
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
    fn perfect_correlation_degenerates_to_the_single_asset_note() {
        // identical assets at rho = 1 share one path, so the worst-of note
        // must price like the single-asset autocallable on the same terms
        let (val, mat) = dates();
        let single = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .dividend_yield(0.02)
            .valuation_date(val)
            .maturity_date(mat)
            .autocallable(100.0, 70.0, 6.0, 6, 100.0)
            .engine(Engine::MonteCarlo)
            .build()
            .expect("single-asset note must build")
            .npv();
        let wof = note(2, 1.0, 100_000);
        let stats = wof.npv_with_stats();
        assert!(
            (stats.pv - single).abs() < 4.0 * stats.std_err.max(0.05),
            "worst-of {} vs single-asset {} (se {})",
            stats.pv,
            single,
            stats.std_err
        );
    }

    #[test]
    fn the_note_is_long_correlation() {
        // a tighter basket has a better worst performer: value must rise
        // with correlation, and even the tightest basket stays below the
        // rho = 1 degenerate case
        let low = note(2, 0.2, 60_000).npv();
        let high = note(2, 0.8, 60_000).npv();
        let degenerate = note(2, 1.0, 60_000).npv();
        assert!(high > low + 0.1, "rho=0.8 {high} vs rho=0.2 {low}");
        assert!(degenerate > high, "rho=1 {degenerate} vs rho=0.8 {high}");
    }

    #[test]
    fn adding_an_asset_cheapens_the_note() {
        // min over three is never better than min over two of the same
        let two = note(2, 0.5, 60_000).npv();
        let three = note(3, 0.5, 60_000).npv();
        assert!(three < two - 0.1, "3-asset {three} vs 2-asset {two}");
    }

    #[test]
    fn deltas_are_positive_and_the_price_reports_stats() {
        let wof = note(2, 0.6, 20_000);
        // the holder is long each asset (higher spot => better worst-of)
        for (i, d) in wof.deltas().iter().enumerate() {
            assert!(*d > 0.0, "delta[{i}] = {d}");
        }
        let result = wof.price().unwrap();
        assert!(result.std_err.unwrap() > 0.0);
        assert!(result.pv > 0.0);
    }
}
