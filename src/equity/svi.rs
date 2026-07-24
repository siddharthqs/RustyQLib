//! SVI and SSVI implied-volatility parameterizations (Gatheral 2004;
//! Gatheral & Jacquier 2014).
//!
//! **SVI** (raw form) parameterizes one expiry's total variance in
//! log-moneyness `k = ln(K/F)`:
//!
//! ```text
//! w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]
//! ```
//!
//! five parameters per smile: level `a`, wing slope `b`, skew `rho`,
//! shift `m`, ATM curvature `sigma`. Wings are asymptotically linear
//! with slopes `b(1 - rho)` (put side) and `b(1 + rho)` (call side).
//!
//! **SSVI** parameterizes the whole surface from the ATM total-variance
//! term structure `theta_t` and three global parameters `(rho, eta,
//! gamma)` through the power-law curvature
//! `phi(theta) = eta / (theta^gamma (1 + theta)^(1-gamma))`:
//!
//! ```text
//! w(k, t) = theta_t/2 [ 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ]
//! ```
//!
//! Both calibrate by Levenberg-Marquardt
//! ([`core::optimization`](crate::core::optimization)) in transformed
//! parameter spaces, the same pattern as
//! [`heston::calibrate`](crate::equity::heston::calibrate). Butterfly
//! arbitrage is checked through the Gatheral-Jacquier `g(k)` density
//! condition (SVI) and the power-law sufficient conditions (SSVI), and
//! fitted smiles sample into the pricing
//! [`VolSurface`](crate::core::vols::VolSurface) via
//! [`Ssvi::to_vol_surface`].

use chrono::NaiveDate;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::core::vols::{VolError, VolSurface};
use crate::core::errors::RustyQLibError;

// ── SVI: one expiry ─────────────────────────────────────────────────────

/// Raw SVI parameters for a single expiry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

/// Result of an SVI smile calibration.
#[derive(Debug, Clone)]
pub struct SviFit {
    pub params: SviParams,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl SviParams {
    /// Total variance `w(k)` at log-moneyness `k = ln(K/F)`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let d = k - self.m;
        self.a + self.b * (self.rho * d + (d * d + self.sigma * self.sigma).sqrt())
    }

    /// Implied vol at log-moneyness `k` for expiry `t`.
    pub fn vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k).max(0.0) / t).sqrt()
    }

    /// Static parameter constraints: `b >= 0`, `|rho| < 1`, `sigma > 0`
    /// and non-negative minimum variance `a + b sigma sqrt(1 - rho^2)`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.b < 0.0 {
            return Err(RustyQLibError::invalid_input("svi params", "b must be non-negative"));
        }
        if !(-1.0..1.0).contains(&self.rho) && self.rho != -1.0 {
            return Err(RustyQLibError::invalid_input("svi params", "rho must be in (-1, 1)"));
        }
        if self.sigma <= 0.0 {
            return Err(RustyQLibError::invalid_input("svi params", "sigma must be positive"));
        }
        if self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() < 0.0 {
            return Err(RustyQLibError::invalid_input("svi params", "minimum total variance is negative"));
        }
        Ok(())
    }

    /// The Gatheral-Jacquier butterfly function
    /// `g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2`,
    /// which must stay non-negative for an arbitrage-free density.
    pub fn butterfly_g(&self, k: f64) -> f64 {
        let d = k - self.m;
        let root = (d * d + self.sigma * self.sigma).sqrt();
        let w = self.a + self.b * (self.rho * d + root);
        let w1 = self.b * (self.rho + d / root);
        let w2 = self.b * self.sigma * self.sigma / (root * root * root);
        (1.0 - k * w1 / (2.0 * w)).powi(2) - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    }

    /// Minimum of `g(k)` over a wide log-moneyness scan; negative means
    /// the smile carries butterfly arbitrage.
    pub fn min_butterfly_g(&self) -> f64 {
        (0..=800)
            .map(|i| self.butterfly_g(-2.0 + i as f64 * 0.005))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn has_butterfly_arbitrage(&self) -> bool {
        self.min_butterfly_g() < 0.0
    }

    /// Calibrate to one expiry's quotes `(k, implied vol)` by
    /// Levenberg-Marquardt on total-variance residuals, with `b` and
    /// `sigma` in log space and `rho` through `tanh` so every trial is
    /// admissible.
    pub fn calibrate(quotes: &[(f64, f64)], t: f64) -> SviFit {
        assert!(quotes.len() >= 5, "SVI has five parameters; need at least five quotes");
        assert!(t > 0.0);
        let w_target: Vec<(f64, f64)> =
            quotes.iter().map(|&(k, v)| (k, v * v * t)).collect();
        let (w_min, w_max) = w_target
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, w)| (lo.min(w), hi.max(w)));
        let k_at_min = w_target
            .iter()
            .fold((0.0, f64::INFINITY), |acc, &(k, w)| if w < acc.1 { (k, w) } else { acc })
            .0;
        let k_span = quotes.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max)
            - quotes.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
        // start: level at the observed floor, gentle wings, no skew
        let x0 = vec![
            0.5 * w_min,                                          // a
            (((w_max - w_min) / k_span.max(0.1)).max(1e-3)).ln(), // ln b
            0.0,                                                  // atanh rho
            k_at_min,                                             // m
            0.2_f64.ln(),                                         // ln sigma
        ];
        let unpack = |u: &[f64]| SviParams {
            a: u[0],
            b: u[1].exp(),
            rho: u[2].tanh(),
            m: u[3],
            sigma: u[4].exp(),
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            let p = unpack(u);
            w_target.iter().map(|&(k, w)| p.total_variance(k) - w).collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let params = unpack(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(k, v)| (params.vol(k, t) - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SviFit { params, rmse, iterations: fit.iterations, converged: fit.converged }
    }
}

// ── SSVI: the whole surface ─────────────────────────────────────────────

/// SSVI surface: ATM total-variance pillars plus global `(rho, eta,
/// gamma)` with the power-law curvature.
#[derive(Debug, Clone)]
pub struct Ssvi {
    pub rho: f64,
    pub eta: f64,
    /// Power-law exponent in `(0, 1]`.
    pub gamma: f64,
    /// `(t, theta_t)` pillars, `t` and `theta` strictly increasing.
    pub theta_pillars: Vec<(f64, f64)>,
}

/// Result of an SSVI calibration.
#[derive(Debug, Clone)]
pub struct SsviFit {
    pub surface: Ssvi,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl Ssvi {
    /// ATM total variance at `t`: proportional below the first pillar
    /// (variance accrues from zero), linear between pillars, and
    /// continued with the last segment's slope beyond.
    pub fn theta(&self, t: f64) -> f64 {
        let p = &self.theta_pillars;
        let n = p.len();
        if t <= 0.0 {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1 * t / p[0].0;
        }
        if t >= p[n - 1].0 {
            if n == 1 {
                return p[0].1 * t / p[0].0;
            }
            let slope = (p[n - 1].1 - p[n - 2].1) / (p[n - 1].0 - p[n - 2].0);
            return p[n - 1].1 + slope * (t - p[n - 1].0);
        }
        let idx = p.partition_point(|&(ti, _)| ti < t);
        let (t0, w0) = p[idx - 1];
        let (t1, w1) = p[idx];
        w0 + (w1 - w0) * (t - t0) / (t1 - t0)
    }

    /// Power-law curvature `phi(theta)`.
    pub fn phi(&self, theta: f64) -> f64 {
        self.eta / (theta.powf(self.gamma) * (1.0 + theta).powf(1.0 - self.gamma))
    }

    /// Total variance `w(k, t)`.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let theta = self.theta(t);
        if theta <= 0.0 {
            return 0.0;
        }
        let phi = self.phi(theta);
        let pk = phi * k;
        0.5 * theta
            * (1.0 + self.rho * pk + ((pk + self.rho).powi(2) + 1.0 - self.rho * self.rho).sqrt())
    }

    /// Implied vol for `strike` given the `forward` at expiry `t`.
    pub fn vol(&self, strike: f64, forward: f64, t: f64) -> f64 {
        (self.total_variance((strike / forward).ln(), t) / t).sqrt()
    }

    /// Static no-arbitrage checks (Gatheral-Jacquier): admissible
    /// parameters, nondecreasing `theta` (calendar), the power-law
    /// sufficient condition `eta (1 + |rho|) <= 2`, and the per-pillar
    /// butterfly bounds `theta phi (1 + |rho|) <= 4` and
    /// `theta phi^2 (1 + |rho|) <= 4`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if !(-1.0..1.0).contains(&self.rho) {
            return Err(RustyQLibError::invalid_input("svi params", "rho must be in (-1, 1)"));
        }
        if self.eta <= 0.0 {
            return Err(RustyQLibError::invalid_input("svi params", "eta must be positive"));
        }
        if !(0.0..=1.0).contains(&self.gamma) || self.gamma == 0.0 {
            return Err(RustyQLibError::invalid_input("svi params", "gamma must be in (0, 1]"));
        }
        if self.theta_pillars.is_empty() {
            return Err(RustyQLibError::invalid_input("svi params", "need at least one theta pillar"));
        }
        if self.theta_pillars.iter().any(|&(t, w)| t <= 0.0 || w <= 0.0) {
            return Err(RustyQLibError::invalid_input("svi params", "theta pillars must have positive times and variances"));
        }
        if self.theta_pillars.windows(2).any(|p| p[1].0 <= p[0].0 || p[1].1 < p[0].1) {
            return Err(RustyQLibError::invalid_input("svi params", "theta pillars must be increasing in time and nondecreasing in variance (calendar arbitrage)"));
        }
        if self.eta * (1.0 + self.rho.abs()) > 2.0 {
            return Err(RustyQLibError::invalid_input("svi params", "eta (1 + |rho|) must not exceed 2 (static arbitrage)"));
        }
        for &(_, theta) in &self.theta_pillars {
            let phi = self.phi(theta);
            if theta * phi * (1.0 + self.rho.abs()) > 4.0
                || theta * phi * phi * (1.0 + self.rho.abs()) > 4.0
            {
                return Err(RustyQLibError::invalid_input("svi params", "butterfly bound violated at a theta pillar"));
            }
        }
        Ok(())
    }

    /// Calibrate `(rho, eta, gamma)` to surface quotes `(t, k, vol)`
    /// given the ATM total-variance pillars, by Levenberg-Marquardt on
    /// total-variance residuals (`tanh` / `exp` / logistic transforms
    /// keep every trial admissible).
    pub fn calibrate(
        quotes: &[(f64, f64, f64)],
        theta_pillars: &[(f64, f64)],
        start: (f64, f64, f64),
    ) -> SsviFit {
        assert!(quotes.len() >= 3, "need at least three quotes for three parameters");
        let make = |u: &[f64]| Ssvi {
            rho: u[0].tanh(),
            eta: u[1].exp(),
            gamma: 1.0 / (1.0 + (-u[2]).exp()),
            theta_pillars: theta_pillars.to_vec(),
        };
        let (rho0, eta0, gamma0) = start;
        let x0 = vec![
            rho0.clamp(-0.999, 0.999).atanh(),
            eta0.ln(),
            (gamma0.clamp(1e-3, 1.0 - 1e-9) / (1.0 - gamma0.clamp(1e-3, 1.0 - 1e-9))).ln(),
        ];
        let residuals = |u: &[f64]| -> Vec<f64> {
            let s = make(u);
            quotes
                .iter()
                .map(|&(t, k, v)| s.total_variance(k, t) - v * v * t)
                .collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let surface = make(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(t, k, v)| ((surface.total_variance(k, t) / t).sqrt() - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SsviFit { surface, rmse, iterations: fit.iterations, converged: fit.converged }
    }

    /// Sample the SSVI surface into the canonical pricing
    /// [`VolSurface`]: per expiry `(t, forward)`, strikes are placed at
    /// `forward * exp(k)` over the log-moneyness grid.
    pub fn to_vol_surface(
        &self,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        expiry_forwards: &[(f64, f64)],
        log_moneyness_grid: &[f64],
    ) -> Result<VolSurface, VolError> {
        let expiries: Vec<Tenor> =
            expiry_forwards.iter().map(|&(t, _)| Tenor::YearFraction(t)).collect();
        let smiles: Vec<Vec<(f64, f64)>> = expiry_forwards
            .iter()
            .map(|&(t, forward)| {
                log_moneyness_grid
                    .iter()
                    .map(|&k| {
                        let strike = forward * k.exp();
                        (strike, self.vol(strike, forward, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, reference_date, day_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sane() -> SviParams {
        SviParams { a: 0.03, b: 0.12, rho: -0.4, m: -0.02, sigma: 0.3 }
    }

    #[test]
    fn svi_shape_matches_the_closed_form_structure() {
        let p = sane();
        p.validate().unwrap();
        // total variance at k = m is a + b sigma
        assert!((p.total_variance(p.m) - (p.a + p.b * p.sigma)).abs() < 1e-14);
        // asymptotic wing slopes b (1 +- rho), measured per unit of |k|
        let far = 60.0;
        let call_slope = p.total_variance(far + 1.0) - p.total_variance(far);
        let put_slope = p.total_variance(-far - 1.0) - p.total_variance(-far);
        assert!((call_slope - p.b * (1.0 + p.rho)).abs() < 1e-3, "{call_slope}");
        assert!((put_slope - p.b * (1.0 - p.rho)).abs() < 1e-3, "{put_slope}");
    }

    #[test]
    fn vogt_example_carries_butterfly_arbitrage_and_sane_params_do_not() {
        // the classic arbitrageable SVI smile (Gatheral-Jacquier 2014 §3)
        let vogt = SviParams { a: -0.0410, b: 0.1331, rho: 0.3060, m: 0.3586, sigma: 0.4153 };
        assert!(vogt.has_butterfly_arbitrage(), "min g = {}", vogt.min_butterfly_g());
        assert!((vogt.min_butterfly_g() - -0.0329).abs() < 2e-3);
        assert!(!sane().has_butterfly_arbitrage(), "min g = {}", sane().min_butterfly_g());
    }

    #[test]
    fn svi_calibration_round_trips() {
        let truth = sane();
        let t = 0.75;
        let quotes: Vec<(f64, f64)> =
            (0..15).map(|i| -0.42 + i as f64 * 0.06).map(|k| (k, truth.vol(k, t))).collect();
        let fit = SviParams::calibrate(&quotes, t);
        assert!(fit.rmse < 1e-6, "vol rmse {} params {:?}", fit.rmse, fit.params);
        assert!(fit.params.validate().is_ok());
        // the fitted smile matches off the quote grid too
        for i in 0..=20 {
            let k = -0.5 + i as f64 * 0.05;
            assert!((fit.params.vol(k, t) - truth.vol(k, t)).abs() < 1e-4, "k = {k}");
        }
    }

    fn ssvi() -> Ssvi {
        Ssvi {
            rho: -0.55,
            eta: 0.9,
            gamma: 0.45,
            theta_pillars: vec![(0.25, 0.012), (0.5, 0.023), (1.0, 0.045), (2.0, 0.09)],
        }
    }

    #[test]
    fn ssvi_reproduces_the_atm_term_structure_and_skew_sign() {
        let s = ssvi();
        s.validate().unwrap();
        for &(t, theta) in &s.theta_pillars {
            assert!((s.total_variance(0.0, t) - theta).abs() < 1e-14, "w(0, {t})");
        }
        // negative rho: puts richer than calls
        assert!(s.total_variance(-0.2, 1.0) > s.total_variance(0.2, 1.0));
        // calendar: total variance nondecreasing in t at fixed k
        for i in 1..40 {
            let (t0, t1) = (i as f64 * 0.05, (i + 1) as f64 * 0.05);
            assert!(s.total_variance(0.15, t1) >= s.total_variance(0.15, t0), "t = {t0}");
        }
    }

    #[test]
    fn ssvi_no_arbitrage_bounds_are_enforced() {
        let mut bad = ssvi();
        bad.eta = 1.5; // eta (1 + |rho|) = 2.325 > 2
        assert!(bad.validate().is_err());
        let mut decreasing = ssvi();
        decreasing.theta_pillars[2].1 = 0.01; // calendar violation
        assert!(decreasing.validate().is_err());
    }

    #[test]
    fn ssvi_calibration_round_trips() {
        let truth = ssvi();
        let mut quotes = Vec::new();
        for &(t, _) in &truth.theta_pillars {
            for i in 0..7 {
                let k = -0.3 + i as f64 * 0.1;
                quotes.push((t, k, (truth.total_variance(k, t) / t).sqrt()));
            }
        }
        let fit = Ssvi::calibrate(&quotes, &truth.theta_pillars, (-0.2, 0.5, 0.5));
        assert!(fit.rmse < 1e-8, "vol rmse {}", fit.rmse);
        assert!((fit.surface.rho - truth.rho).abs() < 1e-4, "rho {}", fit.surface.rho);
        assert!((fit.surface.eta - truth.eta).abs() < 1e-3, "eta {}", fit.surface.eta);
        assert!(fit.surface.validate().is_ok());
    }

    #[test]
    fn sampled_vol_surface_agrees_with_the_parametric_form() {
        use chrono::NaiveDate;
        let s = ssvi();
        let reference = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let forwards = [(0.25, 101.0), (1.0, 104.0), (2.0, 108.0)];
        let grid: Vec<f64> = (0..13).map(|i| -0.3 + i as f64 * 0.05).collect();
        let surface = s
            .to_vol_surface(reference, DayCountConvention::Act365, &forwards, &grid)
            .unwrap();
        // exact at the sampled nodes
        for &(t, f) in &forwards {
            for &k in &grid {
                let strike = f * k.exp();
                let sampled = surface.vol(strike, f, t);
                let parametric = s.vol(strike, f, t);
                assert!(
                    (sampled - parametric).abs() < 1e-10,
                    "t = {t}, k = {k}: {sampled} vs {parametric}"
                );
            }
        }
    }
}
