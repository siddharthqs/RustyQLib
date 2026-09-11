//! Vasicek (1977): `dr = a(b - r)dt + sigma dW`.
//!
//! Mean-reverting Gaussian short rate with fully closed forms
//! (Brigo-Mercurio ch. 3): affine bond prices `P(t,T) = A e^{-B r}`,
//! Jamshidian's zero-bond option formula, and an exact Gaussian
//! transition (an Ornstein-Uhlenbeck bridge step), so simulation carries
//! no discretization bias at any step size. Its term structure is
//! endogenous — for curve-consistent pricing use
//! [`HullWhite`](crate::rates::models::HullWhite).

use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::rates::models::{
    gaussian_bond_price_vol, gaussian_short_rate_std, gaussian_zero_bond_option,
    validate_bond_option_terms, OneFactorAffine, ShortRateModel,
};

#[derive(Debug, Clone)]
pub struct Vasicek {
    /// Mean-reversion speed `a > 0`.
    pub a: f64,
    /// Long-run level `b`.
    pub b: f64,
    /// Absolute (normal) volatility of the short rate.
    pub sigma: f64,
    /// Today's short rate.
    pub r0: f64,
}

impl Vasicek {
    pub fn new(a: f64, b: f64, sigma: f64, r0: f64) -> Result<Self, RustyQLibError> {
        if !(a > 0.0 && a.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "vasicek",
                format!("mean reversion must be positive, got {a}"),
            ));
        }
        if !(sigma >= 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "vasicek",
                format!("sigma must be non-negative, got {sigma}"),
            ));
        }
        if !b.is_finite() || !r0.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "vasicek",
                format!("b and r0 must be finite, got b={b}, r0={r0}"),
            ));
        }
        Ok(Vasicek { a, b, sigma, r0 })
    }

    /// `B(t,T) = (1 - e^{-a(T-t)}) / a`.
    pub(crate) fn b_factor(&self, t: f64, maturity: f64) -> f64 {
        crate::rates::models::b_factor(self.a, t, maturity)
    }

    /// Conditional mean of `r(t + dt)` given `r(t)`.
    pub fn expected_short_rate(&self, short_rate: f64, dt: f64) -> f64 {
        self.b + (short_rate - self.b) * (-self.a * dt).exp()
    }

    /// Conditional standard deviation of `r(t + dt)` given `r(t)`.
    pub fn short_rate_std(&self, dt: f64) -> f64 {
        gaussian_short_rate_std(self.a, self.sigma, dt)
    }
}

impl ShortRateModel for Vasicek {
    fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError> {
        if maturity < t {
            return Err(RustyQLibError::invalid_input(
                "vasicek",
                format!("maturity {maturity} must be at or after t {t}"),
            ));
        }
        let tau = maturity - t;
        let b_factor = self.b_factor(t, maturity);
        let sigma2 = self.sigma * self.sigma;
        let a2 = self.a * self.a;
        let ln_a = (self.b - sigma2 / (2.0 * a2)) * (b_factor - tau)
            - sigma2 * b_factor * b_factor / (4.0 * self.a);
        Ok((ln_a - b_factor * short_rate).exp())
    }

    fn evolve(&self, _t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "vasicek",
                format!("dt must be positive, got {dt}"),
            ));
        }
        Ok(self.expected_short_rate(short_rate, dt) + self.short_rate_std(dt) * z)
    }
}

impl OneFactorAffine for Vasicek {
    fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        self.zero_bond_exchange_option(expiry, expiry, bond_maturity, strike, put_or_call)
    }

    fn zero_bond_exchange_option(
        &self,
        expiry: f64,
        settlement: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        validate_bond_option_terms(expiry, settlement, bond_maturity, strike)?;
        let p_settlement = self.zero_bond(0.0, settlement, self.r0)?;
        let p_bond = self.zero_bond(0.0, bond_maturity, self.r0)?;
        let sigma_p =
            gaussian_bond_price_vol(self.a, self.sigma, expiry, settlement, bond_maturity);
        Ok(gaussian_zero_bond_option(
            p_settlement,
            p_bond,
            strike,
            sigma_p,
            put_or_call,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Vasicek {
        Vasicek::new(0.15, 0.05, 0.01, 0.03).unwrap()
    }

    #[test]
    fn bond_price_is_one_at_maturity_and_yields_the_short_rate() {
        let m = model();
        assert!((m.zero_bond(2.0, 2.0, 0.03).unwrap() - 1.0).abs() < 1e-15);
        // instantaneous yield -> short rate: -ln P(t, t+h) / h -> r
        let h = 1e-6;
        let y = -m.zero_bond(1.0, 1.0 + h, 0.032).unwrap().ln() / h;
        assert!((y - 0.032).abs() < 1e-6, "yield {y}");
        // price falls as the rate rises
        assert!(m.zero_bond(0.0, 5.0, 0.04).unwrap() < m.zero_bond(0.0, 5.0, 0.02).unwrap());
    }

    #[test]
    fn exact_transition_matches_the_conditional_moments() {
        let m = model();
        let (r, dt) = (0.03, 0.7);
        // z = 0 lands on the conditional mean
        let mean = m.evolve(0.0, r, dt, 0.0).unwrap();
        assert!((mean - m.expected_short_rate(r, dt)).abs() < 1e-15);
        // a one-sigma draw moves by exactly the conditional std
        let up = m.evolve(0.0, r, dt, 1.0).unwrap();
        assert!((up - mean - m.short_rate_std(dt)).abs() < 1e-15);
        // chaining conditional means equals the direct mean (linearity)
        let two_step = m.expected_short_rate(m.expected_short_rate(r, 0.35), 0.35);
        assert!((two_step - m.expected_short_rate(r, 0.7)).abs() < 1e-15);
        // long horizon pulls to the long-run level
        assert!((m.expected_short_rate(0.10, 200.0) - m.b).abs() < 1e-10);
    }

    #[test]
    fn zero_bond_option_put_call_parity_is_exact() {
        let m = model();
        let (expiry, maturity, strike) = (1.0, 4.0, 0.85);
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        let put = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let p_expiry = m.zero_bond(0.0, expiry, m.r0).unwrap();
        let p_bond = m.zero_bond(0.0, maturity, m.r0).unwrap();
        assert!(
            (call - put - (p_bond - strike * p_expiry)).abs() < 1e-15,
            "parity: {call} - {put}"
        );
        assert!(call > 0.0 && put > 0.0);
    }

    #[test]
    fn zero_vol_collapses_to_discounted_intrinsic() {
        let m = Vasicek::new(0.15, 0.05, 0.0, 0.03).unwrap();
        let (expiry, maturity) = (1.0, 4.0);
        let p_expiry = m.zero_bond(0.0, expiry, m.r0).unwrap();
        let p_bond = m.zero_bond(0.0, maturity, m.r0).unwrap();
        let strike = 0.8;
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        assert!((call - (p_bond - strike * p_expiry).max(0.0)).abs() < 1e-15);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(Vasicek::new(0.0, 0.05, 0.01, 0.03).is_err());
        assert!(Vasicek::new(0.15, 0.05, -0.01, 0.03).is_err());
        let m = model();
        assert!(m.zero_bond(2.0, 1.0, 0.03).is_err());
        assert!(m.evolve(0.0, 0.03, 0.0, 0.0).is_err());
        assert!(m.zero_bond_option(1.0, 0.5, 0.9, PutOrCall::Call).is_err());
        assert!(m.zero_bond_option(1.0, 4.0, -0.9, PutOrCall::Call).is_err());
    }
}
