//! Martingale check for local-vol dynamics: does the simulated
//! underlying recover the forwards it should?
//!
//! [`martingale_report`] simulates `dS/S = r dt + sigma_loc(S, t) dW`
//! (log-Euler, antithetic pairs, fixed seed) with drift from the
//! supplied discount curve and zero dividends — the build pipeline's
//! convention — and compares the simulated mean at each expiry against
//! an **externally supplied** target forward, in z-scores against the
//! Monte Carlo standard error.
//!
//! The external targets are what make this a real test. Each log-Euler
//! step is mean-one in expectation *by construction*, so validating the
//! scheme against its own drift can only catch numerical damage (NaNs,
//! clamp-induced tail explosions). Validated against **parity forwards
//! from an option chain**, the check also measures how far the
//! zero-dividend modelling assumption sits from where the market
//! actually puts the forward — a genuine model-validation number for
//! dividend-paying names.

use serde::Serialize;

use crate::core::curves::{Compounding, YieldCurve};

/// Simulation choices for [`martingale_report`].
#[derive(Debug, Clone)]
pub struct MartingaleConfig {
    /// Simulated paths (rounded up to an even count for antithetics).
    pub paths: usize,
    /// Time steps per year (at least one step per expiry interval).
    pub steps_per_year: usize,
    /// RNG seed; runs are deterministic per seed.
    pub seed: u64,
    /// |z| at or below this passes (3 = "within 3 standard errors").
    pub z_threshold: f64,
}

impl Default for MartingaleConfig {
    fn default() -> Self {
        MartingaleConfig {
            paths: 16_384,
            steps_per_year: 104,
            seed: 42,
            z_threshold: 3.0,
        }
    }
}

/// One expiry's forward-recovery result.
#[derive(Debug, Clone, Serialize)]
pub struct MartingaleCheck {
    /// Expiry time (year fraction).
    pub t: f64,
    /// The target forward the simulation is measured against.
    pub target_forward: f64,
    /// Monte Carlo mean of the simulated underlying at `t`.
    pub simulated_mean: f64,
    /// `simulated_mean / target_forward - 1`.
    pub relative_error: f64,
    /// Standard error of `relative_error` (over antithetic pair means).
    pub standard_error: f64,
    /// `relative_error / standard_error` — how surprised to be.
    pub z_score: f64,
}

/// The martingale verdict: per-expiry checks plus the headline numbers.
#[derive(Debug, Clone, Serialize)]
pub struct MartingaleReport {
    pub checks: Vec<MartingaleCheck>,
    pub max_abs_z: f64,
    /// Largest relative forward error, signed at its maximizer.
    pub worst_relative_error: f64,
    /// All |z| within the configured threshold.
    pub within_threshold: bool,
    pub paths: usize,
    pub seed: u64,
}

/// Simulate `local_vol` dynamics off `curve` (zero dividends) from
/// `spot` and check forward recovery at each `(t, target_forward)` —
/// see the module docs for why the targets are supplied rather than
/// derived. Targets must have positive times and forwards; they are
/// checked in time order.
pub fn martingale_report(
    local_vol: &dyn Fn(f64, f64) -> f64,
    curve: &YieldCurve,
    spot: f64,
    targets: &[(f64, f64)],
    config: &MartingaleConfig,
) -> MartingaleReport {
    use rand::distributions::Distribution;
    use rand::SeedableRng;

    let mut targets: Vec<(f64, f64)> = targets
        .iter()
        .copied()
        .filter(|&(t, f)| t > 0.0 && f > 0.0)
        .collect();
    targets.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let pairs = (config.paths / 2).max(1);
    let mut report = MartingaleReport {
        checks: Vec::with_capacity(targets.len()),
        max_abs_z: 0.0,
        worst_relative_error: 0.0,
        within_threshold: true,
        paths: pairs * 2,
        seed: config.seed,
    };
    if targets.is_empty() {
        report.within_threshold = false;
        return report;
    }

    // substep grid: config resolution, but never coarser than the
    // expiry checkpoints themselves
    let mut grid: Vec<f64> = vec![0.0];
    for &(t, _) in &targets {
        let previous = *grid.last().expect("grid starts at 0");
        let substeps = (((t - previous) * config.steps_per_year as f64).ceil() as usize).max(1);
        for i in 1..=substeps {
            grid.push(previous + (t - previous) * i as f64 / substeps as f64);
        }
    }
    let checkpoints: Vec<usize> = targets
        .iter()
        .map(|&(t, _)| grid.partition_point(|&g| g < t - 1e-12))
        .collect();

    // per-step drift from the curve's continuous zeros: exact forward
    // growth over each substep under zero dividends
    let drift: Vec<f64> = grid
        .windows(2)
        .map(|w| {
            let (t0, t1) = (w[0], w[1]);
            let z1 = if t1 > 0.0 {
                curve.zero_rate_with(t1, Compounding::Continuous) * t1
            } else {
                0.0
            };
            let z0 = if t0 > 0.0 {
                curve.zero_rate_with(t0, Compounding::Continuous) * t0
            } else {
                0.0
            };
            z1 - z0
        })
        .collect();

    let mut rng = rand_pcg::Pcg64::seed_from_u64(config.seed);
    let normal = rand_distr::StandardNormal;
    // accumulate antithetic pair means per checkpoint
    let mut sums = vec![0.0_f64; targets.len()];
    let mut sum_squares = vec![0.0_f64; targets.len()];
    for _ in 0..pairs {
        let mut log_up = spot.ln();
        let mut log_down = spot.ln();
        let mut checkpoint = 0usize;
        for (step, growth) in drift.iter().enumerate() {
            let (t0, t1) = (grid[step], grid[step + 1]);
            let dt = t1 - t0;
            let sqrt_dt = dt.sqrt();
            let shock: f64 = normal.sample(&mut rng);
            let vol_up = local_vol(log_up.exp(), t0.max(1e-4));
            let vol_down = local_vol(log_down.exp(), t0.max(1e-4));
            log_up += growth - 0.5 * vol_up * vol_up * dt + vol_up * sqrt_dt * shock;
            log_down += growth - 0.5 * vol_down * vol_down * dt - vol_down * sqrt_dt * shock;
            while checkpoint < checkpoints.len() && step + 1 == checkpoints[checkpoint] {
                let pair_mean = 0.5 * (log_up.exp() + log_down.exp());
                sums[checkpoint] += pair_mean;
                sum_squares[checkpoint] += pair_mean * pair_mean;
                checkpoint += 1;
            }
        }
    }

    let n = pairs as f64;
    for (i, &(t, target)) in targets.iter().enumerate() {
        let mean = sums[i] / n;
        let variance = (sum_squares[i] / n - mean * mean).max(0.0);
        let standard_error = (variance / n).sqrt() / target;
        let relative_error = mean / target - 1.0;
        let z_score = if standard_error > 0.0 {
            relative_error / standard_error
        } else if relative_error.abs() < 1e-12 {
            0.0
        } else {
            f64::INFINITY
        };
        if z_score.abs() > report.max_abs_z {
            report.max_abs_z = z_score.abs();
        }
        if relative_error.abs() > report.worst_relative_error.abs() {
            report.worst_relative_error = relative_error;
        }
        report.checks.push(MartingaleCheck {
            t,
            target_forward: target,
            simulated_mean: mean,
            relative_error,
            standard_error,
            z_score,
        });
    }
    report.within_threshold = report.max_abs_z <= config.z_threshold;
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn flat_dynamics_recover_curve_forwards() {
        let rate = 0.04;
        let spot = 100.0;
        let curve = curve(rate);
        let targets: Vec<(f64, f64)> = [0.25, 0.5, 1.0]
            .iter()
            .map(|&t: &f64| (t, spot * (rate * t).exp()))
            .collect();
        let report = martingale_report(
            &|_, _| 0.2,
            &curve,
            spot,
            &targets,
            &MartingaleConfig::default(),
        );
        assert_eq!(report.checks.len(), 3);
        assert!(
            report.within_threshold,
            "max |z| = {}, errors {:?}",
            report.max_abs_z,
            report
                .checks
                .iter()
                .map(|c| c.relative_error)
                .collect::<Vec<_>>()
        );
        for check in &report.checks {
            assert!(check.standard_error > 0.0 && check.standard_error < 0.01);
        }
    }

    #[test]
    fn state_dependent_vol_is_still_a_martingale() {
        // a steep skew must not bias the forward (each Euler step is
        // mean-one for any adapted vol)
        let curve = curve(0.03);
        let skewed = |level: f64, _t: f64| (0.45 - 0.002 * (level - 100.0)).clamp(0.05, 0.9);
        let targets = [(0.5, 100.0 * (0.03_f64 * 0.5).exp())];
        let report = martingale_report(
            &skewed,
            &curve,
            100.0,
            &targets,
            &MartingaleConfig::default(),
        );
        assert!(
            report.within_threshold,
            "z = {}, err = {}",
            report.max_abs_z, report.worst_relative_error
        );
    }

    #[test]
    fn wrong_targets_fail_loudly_and_runs_are_deterministic() {
        let curve = curve(0.04);
        // targets 2% away from where the dynamics put the forward
        let bad_targets = [(0.5, 100.0 * (0.04_f64 * 0.5).exp() * 1.02)];
        let config = MartingaleConfig::default();
        let report = martingale_report(&|_, _| 0.2, &curve, 100.0, &bad_targets, &config);
        assert!(!report.within_threshold);
        assert!(report.max_abs_z > 5.0, "z = {}", report.max_abs_z);
        assert!(report.worst_relative_error < -0.015, "{report:?}");
        // determinism: same seed, same numbers
        let again = martingale_report(&|_, _| 0.2, &curve, 100.0, &bad_targets, &config);
        assert_eq!(
            report.checks[0].simulated_mean,
            again.checks[0].simulated_mean
        );
        // no usable targets = no verdict
        let empty = martingale_report(&|_, _| 0.2, &curve, 100.0, &[], &config);
        assert!(!empty.within_threshold);
    }
}
