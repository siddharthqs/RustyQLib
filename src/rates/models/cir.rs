//! Cox-Ingersoll-Ross (1985): `dr = a(b - r)dt + sigma sqrt(r) dW`.
//!
//! Square-root dynamics keep the rate non-negative (strictly positive
//! under the Feller condition `2ab >= sigma^2`) at the cost of Gaussian
//! tractability: bond prices stay affine in closed form, but the
//! transition is noncentral chi-square, simulated here with a
//! full-truncation Euler step (bias `O(dt)` — use small steps).

use crate::core::errors::RustyQLibError;
use crate::rates::models::ShortRateModel;

#[derive(Debug, Clone)]
pub struct CoxIngersollRoss {
    /// Mean-reversion speed `a > 0`.
    pub a: f64,
    /// Long-run level `b > 0`.
    pub b: f64,
    /// Volatility of `sqrt(r)` dynamics.
    pub sigma: f64,
    /// Today's short rate (non-negative).
    pub r0: f64,
}

impl CoxIngersollRoss {
    pub fn new(a: f64, b: f64, sigma: f64, r0: f64) -> Result<Self, RustyQLibError> {
        if !(a > 0.0 && a.is_finite() && b > 0.0 && b.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("a and b must be positive, got a={a}, b={b}"),
            ));
        }
        if !(sigma >= 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("sigma must be non-negative, got {sigma}"),
            ));
        }
        if !(r0 >= 0.0 && r0.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("r0 must be non-negative, got {r0}"),
            ));
        }
        Ok(CoxIngersollRoss { a, b, sigma, r0 })
    }

    /// Whether `2ab >= sigma^2`, which keeps the rate strictly positive.
    pub fn feller_condition_holds(&self) -> bool {
        2.0 * self.a * self.b >= self.sigma * self.sigma
    }

    fn gamma(&self) -> f64 {
        (self.a * self.a + 2.0 * self.sigma * self.sigma).sqrt()
    }
}

impl ShortRateModel for CoxIngersollRoss {
    fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError> {
        if maturity < t {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("maturity {maturity} must be at or after t {t}"),
            ));
        }
        if short_rate < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("the CIR short rate cannot be negative, got {short_rate}"),
            ));
        }
        let tau = maturity - t;
        if self.sigma == 0.0 {
            // deterministic mean-reverting limit
            let b_det = (1.0 - (-self.a * tau).exp()) / self.a;
            return Ok((-self.b * (tau - b_det) - b_det * short_rate).exp());
        }
        let gamma = self.gamma();
        let e = (gamma * tau).exp();
        let denominator = (gamma + self.a) * (e - 1.0) + 2.0 * gamma;
        let b_factor = 2.0 * (e - 1.0) / denominator;
        let a_factor = (2.0 * gamma * ((self.a + gamma) * tau / 2.0).exp() / denominator)
            .powf(2.0 * self.a * self.b / (self.sigma * self.sigma));
        Ok(a_factor * (-b_factor * short_rate).exp())
    }

    fn evolve(&self, _t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cir",
                format!("dt must be positive, got {dt}"),
            ));
        }
        // full-truncation Euler: negative excursions feed neither the
        // drift's pull-down nor the diffusion
        let positive_part = short_rate.max(0.0);
        Ok(short_rate
            + self.a * (self.b - positive_part) * dt
            + self.sigma * (positive_part * dt).sqrt() * z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> CoxIngersollRoss {
        // 2ab = 0.006 >= sigma^2 = 0.0036: Feller holds
        CoxIngersollRoss::new(0.3, 0.05, 0.06, 0.03).unwrap()
    }

    #[test]
    fn bond_price_is_one_at_maturity_and_yields_the_short_rate() {
        let m = model();
        assert!((m.zero_bond(1.0, 1.0, 0.03).unwrap() - 1.0).abs() < 1e-15);
        let h = 1e-6;
        let y = -m.zero_bond(0.0, h, 0.037).unwrap().ln() / h;
        assert!((y - 0.037).abs() < 1e-5, "yield {y}");
        // monotone decreasing in the rate
        assert!(m.zero_bond(0.0, 5.0, 0.05).unwrap() < m.zero_bond(0.0, 5.0, 0.02).unwrap());
    }

    #[test]
    fn sigma_zero_matches_the_deterministic_limit_continuously() {
        // the closed form must approach the sigma = 0 branch smoothly.
        // sigma cannot be taken arbitrarily small in f64: the exponent
        // 2ab/sigma^2 amplifies one ulp of its base by ~3e10 at
        // sigma = 1e-6, so probe at 1e-4/1e-5 where both the physical
        // O(sigma^2) gap and the rounding stay far below the tolerance
        let deterministic = CoxIngersollRoss::new(0.3, 0.05, 0.0, 0.03).unwrap();
        for sigma in [1e-4, 1e-5] {
            let nearly = CoxIngersollRoss::new(0.3, 0.05, sigma, 0.03).unwrap();
            for t in [1.0, 5.0, 20.0] {
                let d = deterministic.zero_bond(0.0, t, 0.03).unwrap();
                let n = nearly.zero_bond(0.0, t, 0.03).unwrap();
                assert!((d - n).abs() < 1e-6, "sigma={sigma}, t={t}: {d} vs {n}");
            }
        }
    }

    #[test]
    fn feller_condition_is_reported() {
        assert!(model().feller_condition_holds());
        let violating = CoxIngersollRoss::new(0.05, 0.02, 0.09, 0.03).unwrap();
        assert!(!violating.feller_condition_holds());
    }

    #[test]
    fn euler_step_pulls_to_the_mean_and_truncates_at_zero() {
        let m = model();
        // drift-only step from below b moves up, from above moves down
        assert!(m.evolve(0.0, 0.02, 0.1, 0.0).unwrap() > 0.02);
        assert!(m.evolve(0.0, 0.09, 0.1, 0.0).unwrap() < 0.09);
        // at r = 0 the diffusion switches off and the drift is a*b*dt
        let at_zero = m.evolve(0.0, 0.0, 0.1, -3.0).unwrap();
        assert!((at_zero - 0.3 * 0.05 * 0.1).abs() < 1e-15);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(CoxIngersollRoss::new(0.0, 0.05, 0.06, 0.03).is_err());
        assert!(CoxIngersollRoss::new(0.3, 0.05, 0.06, -0.01).is_err());
        let m = model();
        assert!(m.zero_bond(2.0, 1.0, 0.03).is_err());
        assert!(m.zero_bond(0.0, 5.0, -0.01).is_err());
        assert!(m.evolve(0.0, 0.03, 0.0, 0.0).is_err());
    }
}
