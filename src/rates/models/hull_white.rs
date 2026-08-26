//! Hull-White one-factor (extended Vasicek):
//! `dr = (theta(t) - a r)dt + sigma dW`.
//!
//! The drift `theta(t)` is chosen so the model reproduces an input
//! [`YieldCurve`] **exactly** — the term-structure-consistent
//! formulation of Brigo-Mercurio ch. 3.3, where every formula is
//! written against the market discount factors `P(0,t)` and the
//! instantaneous forward `f(0,t)` instead of `theta` itself:
//!
//! ```text
//! P(t,T) = A(t,T) e^{-B(t,T) r(t)},   B = (1 - e^{-a(T-t)})/a
//! ln A   = ln(P(0,T)/P(0,t)) + B f(0,t) - sigma^2/(4a) (1-e^{-2at}) B^2
//! ```
//!
//! so `P(0,T)` equals the curve's discount factor by construction — the
//! property tested here. The transition is exact Gaussian around
//! `alpha(t) = f(0,t) + sigma^2/(2a^2) (1-e^{-at})^2`, so Monte Carlo
//! over any step size is bias-free — the property a long-dated equity
//! hybrid relies on.
//!
//! Instantaneous forwards come from a symmetric log-df difference on
//! the curve; under the curve's log-linear interpolation forwards are
//! piecewise constant, so the difference is exact away from pillars.

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::rates::models::{
    gaussian_bond_price_vol, gaussian_short_rate_std, gaussian_zero_bond_option,
    validate_bond_option_terms, OneFactorAffine, ShortRateModel,
};

/// Step for the log-df difference approximating `f(0,t)`.
const FORWARD_STEP: f64 = 1e-5;

#[derive(Debug, Clone)]
pub struct HullWhite {
    /// Mean-reversion speed `a > 0`.
    pub a: f64,
    /// Absolute (normal) volatility of the short rate.
    pub sigma: f64,
    curve: YieldCurve,
    /// `f(0, 0+)`: the model's initial short rate.
    r0: f64,
}

impl HullWhite {
    /// Fit the model to `curve`; `a` and `sigma` are the free dynamics
    /// parameters (typically calibrated to swaptions or caps).
    pub fn new(a: f64, sigma: f64, curve: YieldCurve) -> Result<Self, RustyQLibError> {
        if !(a > 0.0 && a.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "hull_white",
                format!("mean reversion must be positive, got {a}"),
            ));
        }
        if !(sigma >= 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "hull_white",
                format!("sigma must be non-negative, got {sigma}"),
            ));
        }
        let r0 = instantaneous_forward(&curve, 0.0);
        if !r0.is_finite() {
            return Err(RustyQLibError::NumericalError(
                "the curve's short-end forward is not finite".to_string(),
            ));
        }
        Ok(HullWhite {
            a,
            sigma,
            curve,
            r0,
        })
    }

    /// The fitted curve.
    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    /// `B(t,T) = (1 - e^{-a(T-t)}) / a`.
    fn b_factor(&self, t: f64, maturity: f64) -> f64 {
        crate::rates::models::b_factor(self.a, t, maturity)
    }

    /// `alpha(t) = f(0,t) + sigma^2/(2a^2)(1 - e^{-at})^2` — the mean
    /// around which the short rate reverts under the fit.
    pub fn alpha(&self, t: f64) -> f64 {
        let s = (1.0 - (-self.a * t).exp()) / self.a;
        instantaneous_forward(&self.curve, t) + 0.5 * (self.sigma * self.sigma) * s * s
    }

    /// Unconditional mean of the short rate at `t` (from `r(0) = f(0,0)`).
    pub fn expected_short_rate(&self, t: f64) -> f64 {
        // E[r(t)] = alpha(t) + (r0 - alpha(0)) e^{-at}; alpha(0) = r0
        self.alpha(t)
    }

    /// Conditional standard deviation of `r(t + dt)` given `r(t)`.
    pub fn short_rate_std(&self, dt: f64) -> f64 {
        gaussian_short_rate_std(self.a, self.sigma, dt)
    }
}

impl ShortRateModel for HullWhite {
    fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError> {
        if !(maturity >= t && t >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "hull_white",
                format!("need 0 <= t <= maturity, got t={t}, maturity={maturity}"),
            ));
        }
        let b_factor = self.b_factor(t, maturity);
        let p_ratio = self.curve.df(maturity) / self.curve.df(t);
        let forward = instantaneous_forward(&self.curve, t);
        let vol_term = self.sigma * self.sigma / (4.0 * self.a)
            * (1.0 - (-2.0 * self.a * t).exp())
            * b_factor
            * b_factor;
        let ln_a = p_ratio.ln() + b_factor * forward - vol_term;
        Ok((ln_a - b_factor * short_rate).exp())
    }

    fn evolve(&self, t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError> {
        if !(dt > 0.0 && dt.is_finite() && t >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "hull_white",
                format!("need t >= 0 and dt > 0, got t={t}, dt={dt}"),
            ));
        }
        // exact transition: r(t+dt) = alpha(t+dt)
        //   + (r(t) - alpha(t)) e^{-a dt} + std * z
        let decay = (-self.a * dt).exp();
        let mean = self.alpha(t + dt) + (short_rate - self.alpha(t)) * decay;
        Ok(mean + self.short_rate_std(dt) * z)
    }
}

impl OneFactorAffine for HullWhite {
    fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        validate_bond_option_terms(expiry, bond_maturity, strike)?;
        let p_expiry = self.curve.df(expiry);
        let p_bond = self.curve.df(bond_maturity);
        let sigma_p = gaussian_bond_price_vol(self.a, self.sigma, expiry, bond_maturity);
        Ok(gaussian_zero_bond_option(
            p_expiry,
            p_bond,
            strike,
            sigma_p,
            put_or_call,
        ))
    }
}

/// `f(0,t) = -d ln P(0,t) / dt` by symmetric difference (one-sided at
/// the anchor).
pub(crate) fn instantaneous_forward(curve: &YieldCurve, t: f64) -> f64 {
    let lo = (t - FORWARD_STEP).max(0.0);
    let hi = t + FORWARD_STEP;
    (curve.df(lo).ln() - curve.df(hi).ln()) / (hi - lo)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, 13).unwrap()
    }

    /// An upward-sloping market curve.
    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn model() -> HullWhite {
        HullWhite::new(0.1, 0.01, market_curve()).unwrap()
    }

    #[test]
    fn fitted_model_reproduces_the_input_curve_exactly() {
        // the whole point of the extension: P(0,T) = market df(T)
        let m = model();
        let r0 = m.initial_short_rate();
        for t in [0.25, 0.75, 1.0, 3.3, 7.0, 12.0, 25.0] {
            let model_df = m.zero_bond(0.0, t, r0).unwrap();
            let market_df = m.curve().df(t);
            assert!(
                (model_df / market_df - 1.0).abs() < 1e-9,
                "t={t}: {model_df} vs {market_df}"
            );
        }
    }

    #[test]
    fn zero_bond_options_satisfy_parity_against_the_market_curve() {
        let m = model();
        let (expiry, maturity, strike) = (1.0, 5.0, 0.83);
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        let put = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let parity = m.curve().df(maturity) - strike * m.curve().df(expiry);
        assert!((call - put - parity).abs() < 1e-15, "{call} vs {put}");
        assert!(call > 0.0 && put > 0.0);
        // more volatility, more option value
        let calm = HullWhite::new(0.1, 0.002, market_curve()).unwrap();
        assert!(
            calm.zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
                .unwrap()
                < call
        );
    }

    #[test]
    fn mean_path_follows_alpha_and_z_zero_evolution_tracks_it() {
        let m = model();
        // chaining z = 0 steps reproduces the unconditional mean exactly
        // (the transition is linear in the state)
        let mut r = m.initial_short_rate();
        let dt = 0.25;
        for step in 1..=20 {
            r = m.evolve((step - 1) as f64 * dt, r, dt, 0.0).unwrap();
            let t = step as f64 * dt;
            assert!(
                (r - m.expected_short_rate(t)).abs() < 1e-10,
                "t={t}: {r} vs {}",
                m.expected_short_rate(t)
            );
        }
        // the fitted mean rides above the forwards by the convexity term
        let t = 5.0;
        let forward = instantaneous_forward(m.curve(), t);
        assert!(m.expected_short_rate(t) > forward);
    }

    #[test]
    fn reconstituted_bond_prices_respond_to_the_simulated_rate() {
        let m = model();
        let r0 = m.initial_short_rate();
        // at a node with a higher short rate, bonds are cheaper
        let low = m.zero_bond(2.0, 7.0, r0 - 0.01).unwrap();
        let high = m.zero_bond(2.0, 7.0, r0 + 0.01).unwrap();
        assert!(high < low);
        // and P(t,t) = 1 whatever the state
        assert!((m.zero_bond(2.0, 2.0, 0.1).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn simulated_discounting_recovers_the_curve() {
        // the cross-asset contract: simulate the short rate with the
        // exact transition, discount 1 unit along each path with a
        // trapezoid bank account, and the Monte Carlo average must
        // reproduce the market discount factor (a hybrid equity model
        // does exactly this for its funding leg)
        use rand::{Rng, SeedableRng};
        let m = model();
        let (horizon, steps, paths) = (5.0_f64, 100usize, 20_000usize);
        let dt = horizon / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(7);
        let mut sum = 0.0;
        for _ in 0..paths {
            let mut r = m.initial_short_rate();
            let mut integral = 0.0;
            for step in 0..steps {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                let next = m.evolve(step as f64 * dt, r, dt, z).unwrap();
                integral += 0.5 * (r + next) * dt;
                r = next;
            }
            sum += (-integral).exp();
        }
        let mc_df = sum / paths as f64;
        let market_df = m.curve().df(horizon);
        assert!(
            (mc_df / market_df - 1.0).abs() < 0.005,
            "MC {mc_df} vs market {market_df}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(HullWhite::new(0.0, 0.01, market_curve()).is_err());
        assert!(HullWhite::new(0.1, -0.01, market_curve()).is_err());
        let m = model();
        assert!(m.zero_bond(3.0, 2.0, 0.03).is_err());
        assert!(m.evolve(0.0, 0.03, -1.0, 0.0).is_err());
        assert!(m.zero_bond_option(0.0, 5.0, 0.8, PutOrCall::Call).is_err());
    }
}
