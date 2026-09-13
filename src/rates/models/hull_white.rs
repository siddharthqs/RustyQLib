//! Hull-White one-factor (extended Vasicek), in its generalized form:
//! `dr = (theta(t) - a(t) r)dt + sigma(t) dW` with **piecewise-constant**
//! mean reversion and volatility.
//!
//! The drift `theta(t)` is chosen so the model reproduces an input
//! [`YieldCurve`] **exactly** — the term-structure-consistent
//! formulation of Brigo-Mercurio ch. 3.3, where every formula is
//! written against the market discount factors `P(0,t)` and the
//! instantaneous forward `f(0,t)` instead of `theta` itself. With
//! `K(s,t) = int_s^t a(u) du` (so `e^{-K}` is the mean-reversion decay):
//!
//! ```text
//! P(t,T) = A(t,T) e^{-B(t,T) r(t)},   B(t,T) = int_t^T e^{-K(t,u)} du
//! ln A   = ln(P(0,T)/P(0,t)) + B f(0,t) - B^2 V(0,t) / 2
//! V(s,t) = int_s^t sigma(u)^2 e^{-2K(u,t)} du          (variance of r(t) | r(s))
//! M(s,t) = int_s^t sigma(u)^2 e^{-K(u,t)} B(u,t) du    (the convexity / forward-measure drift)
//! ```
//!
//! so `P(0,T)` equals the curve's discount factor by construction — the
//! property tested here. The transition is exact Gaussian around
//! `alpha(t) = f(0,t) + M(0,t)`, so Monte Carlo over any step size is
//! bias-free — the property a long-dated equity hybrid relies on. Bond
//! prices stay separable (`B(u,T) = B(u,t) + e^{-K(u,t)} B(t,T)`), so
//! the zero-bond option is still Jamshidian's Black-style formula with
//! price-ratio variance `(B(e,T_b) - B(e,T_s))^2 V(0,e)`.
//!
//! Every integral is evaluated segment by segment on the merged
//! breakpoints of `a` and `sigma`, each segment having closed-form
//! exponential integrals, so the constant case reproduces the classic
//! formulas exactly. The three constructors are three names for one
//! model:
//!
//! - [`HullWhite::new`] — constant `a` and `sigma`: the textbook model.
//! - [`HullWhite::with_piecewise_sigma`] — constant `a`, piecewise
//!   `sigma`: QuantLib's **GSR**, what fits a column of swaption
//!   expiries ([`calibrate_hull_white_piecewise`]).
//! - [`HullWhite::generalized`] — both piecewise: QuantLib's
//!   **GeneralizedHullWhite**.
//!
//! Instantaneous forwards come from a symmetric log-df difference on
//! the curve; under the curve's log-linear interpolation forwards are
//! piecewise constant, so the difference is exact away from pillars.
//!
//! [`calibrate_hull_white_piecewise`]: crate::rates::models::calibration::calibrate_hull_white_piecewise

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::rates::models::{
    gaussian_zero_bond_option, validate_bond_option_terms, OneFactorAffine, ShortRateModel,
};

/// Step for the log-df difference approximating `f(0,t)`.
const FORWARD_STEP: f64 = 1e-5;

/// A piecewise-constant function of time: `values[i]` on
/// `[times[i-1], times[i])`, the last from the last breakpoint on.
#[derive(Debug, Clone, PartialEq)]
struct Piecewise {
    times: Vec<f64>,
    values: Vec<f64>,
}

impl Piecewise {
    fn new(
        name: &str,
        times: &[f64],
        values: &[f64],
        positive: bool,
    ) -> Result<Self, RustyQLibError> {
        if values.len() != times.len() + 1 {
            return Err(RustyQLibError::invalid_input(
                "hull_white",
                format!(
                    "need one {name} per interval: {} breakpoints take {} values, got {}",
                    times.len(),
                    times.len() + 1,
                    values.len()
                ),
            ));
        }
        for &v in values {
            let ok = v.is_finite() && if positive { v > 0.0 } else { v >= 0.0 };
            if !ok {
                return Err(RustyQLibError::invalid_input(
                    "hull_white",
                    format!(
                        "{name} must be {}, got {v}",
                        if positive { "positive" } else { "non-negative" }
                    ),
                ));
            }
        }
        for (i, &t) in times.iter().enumerate() {
            let previous = if i == 0 { 0.0 } else { times[i - 1] };
            if !(t > previous && t.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "hull_white",
                    format!("{name} breakpoints must be positive and increasing, got {times:?}"),
                ));
            }
        }
        Ok(Piecewise {
            times: times.to_vec(),
            values: values.to_vec(),
        })
    }

    fn at(&self, t: f64) -> f64 {
        let i = self.times.iter().take_while(|&&s| s <= t).count();
        self.values[i]
    }
}

#[derive(Debug, Clone)]
pub struct HullWhite {
    mean_reversion: Piecewise,
    sigma: Piecewise,
    curve: YieldCurve,
    /// `f(0, 0+)`: the model's initial short rate.
    r0: f64,
}

impl HullWhite {
    /// Fit the model to `curve` with constant `a` and `sigma` — the
    /// textbook model; the free dynamics parameters are typically
    /// calibrated to swaptions or caps.
    pub fn new(a: f64, sigma: f64, curve: YieldCurve) -> Result<Self, RustyQLibError> {
        Self::generalized(&[], &[a], &[], &[sigma], curve)
    }

    /// Constant `a`, piecewise-constant volatility: `sigmas[i]` on
    /// `[times[i-1], times[i])`, the last from the last breakpoint on,
    /// so `sigmas.len() == times.len() + 1`. QuantLib's GSR.
    pub fn with_piecewise_sigma(
        a: f64,
        times: &[f64],
        sigmas: &[f64],
        curve: YieldCurve,
    ) -> Result<Self, RustyQLibError> {
        Self::generalized(&[], &[a], times, sigmas, curve)
    }

    /// Piecewise-constant mean reversion **and** volatility, each on its
    /// own breakpoints (same layout as
    /// [`with_piecewise_sigma`](Self::with_piecewise_sigma)). QuantLib's
    /// GeneralizedHullWhite.
    pub fn generalized(
        a_times: &[f64],
        mean_reversions: &[f64],
        sigma_times: &[f64],
        sigmas: &[f64],
        curve: YieldCurve,
    ) -> Result<Self, RustyQLibError> {
        let mean_reversion = Piecewise::new("mean reversion", a_times, mean_reversions, true)?;
        let sigma = Piecewise::new("sigma", sigma_times, sigmas, false)?;
        let r0 = instantaneous_forward(&curve, 0.0);
        if !r0.is_finite() {
            return Err(RustyQLibError::NumericalError(
                "the curve's short-end forward is not finite".to_string(),
            ));
        }
        Ok(HullWhite {
            mean_reversion,
            sigma,
            curve,
            r0,
        })
    }

    /// The same dynamics fitted to another curve (a bumped one, say).
    pub fn refit(&self, curve: YieldCurve) -> Result<Self, RustyQLibError> {
        Self::generalized(
            &self.mean_reversion.times,
            &self.mean_reversion.values,
            &self.sigma.times,
            &self.sigma.values,
            curve,
        )
    }

    /// The fitted curve.
    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    /// The mean reversion on the first interval — *the* `a` of a model
    /// with constant mean reversion.
    pub fn a(&self) -> f64 {
        self.mean_reversion.values[0]
    }

    /// The mean reversion at `t`.
    pub fn a_at(&self, t: f64) -> f64 {
        self.mean_reversion.at(t)
    }

    /// The breakpoints of the mean-reversion term structure.
    pub fn a_times(&self) -> &[f64] {
        &self.mean_reversion.times
    }

    /// The mean reversions, one per interval.
    pub fn mean_reversions(&self) -> &[f64] {
        &self.mean_reversion.values
    }

    /// The volatility on the first interval — *the* sigma of a
    /// constant-vol model.
    pub fn sigma(&self) -> f64 {
        self.sigma.values[0]
    }

    /// The volatility at `t`.
    pub fn sigma_at(&self, t: f64) -> f64 {
        self.sigma.at(t)
    }

    /// The breakpoints of the volatility term structure.
    pub fn sigma_times(&self) -> &[f64] {
        &self.sigma.times
    }

    /// The volatilities, one per interval.
    pub fn sigmas(&self) -> &[f64] {
        &self.sigma.values
    }

    /// The constant-coefficient segments of `[t0, t1]`: `(a, sigma, lo, hi)`
    /// on the merged breakpoints of both term structures.
    fn segments(&self, t0: f64, t1: f64) -> Vec<(f64, f64, f64, f64)> {
        if t1 <= t0 {
            return Vec::new();
        }
        let mut cuts: Vec<f64> = self
            .mean_reversion
            .times
            .iter()
            .chain(self.sigma.times.iter())
            .copied()
            .filter(|&s| s > t0 && s < t1)
            .collect();
        cuts.sort_by(|x, y| x.total_cmp(y));
        cuts.dedup();
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut lo = t0;
        for hi in cuts.into_iter().chain(std::iter::once(t1)) {
            out.push((self.mean_reversion.at(lo), self.sigma.at(lo), lo, hi));
            lo = hi;
        }
        out
    }

    /// `e^{-K(s,t)}`, `K(s,t) = int_s^t a(u) du`: how much of a shock at
    /// `s` survives to `t`.
    pub fn decay(&self, s: f64, t: f64) -> f64 {
        let k: f64 = self
            .segments(s, t)
            .into_iter()
            .map(|(a, _, lo, hi)| a * (hi - lo))
            .sum();
        (-k).exp()
    }

    /// `B(t,T) = int_t^T e^{-K(t,u)} du` — the affine loading of the bond
    /// price on the short rate.
    pub fn b_factor(&self, t: f64, maturity: f64) -> f64 {
        let mut total = 0.0;
        let mut decay_to_lo = 1.0;
        for (a, _, lo, hi) in self.segments(t, maturity) {
            total += decay_to_lo * (1.0 - (-a * (hi - lo)).exp()) / a;
            decay_to_lo *= (-a * (hi - lo)).exp();
        }
        total
    }

    /// `V(t0, t1) = int_{t0}^{t1} sigma(u)^2 e^{-2K(u,t1)} du`: the
    /// conditional variance of `r(t1)` given `r(t0)`.
    pub fn short_rate_variance(&self, t0: f64, t1: f64) -> f64 {
        let segments = self.segments(t0, t1);
        let mut total = 0.0;
        // decay from each segment's end to t1, accumulated from the back
        let mut decay_hi = 1.0;
        for &(a, sigma, lo, hi) in segments.iter().rev() {
            let delta = hi - lo;
            total +=
                sigma * sigma * decay_hi * decay_hi * (1.0 - (-2.0 * a * delta).exp()) / (2.0 * a);
            decay_hi *= (-a * delta).exp();
        }
        total
    }

    /// `M(t0, t1) = int_{t0}^{t1} sigma(u)^2 e^{-K(u,t1)} B(u,t1) du`: the
    /// convexity term lifting the short rate above the forward, and the
    /// drift of `x = r - alpha` over a step under the `t1`-forward
    /// measure (which a grid engine discounting with `P(t0, t1)` needs).
    pub fn forward_measure_shift(&self, t0: f64, t1: f64) -> f64 {
        let segments = self.segments(t0, t1);
        let mut total = 0.0;
        let mut decay_hi = 1.0; // e^{-K(hi, t1)}
        let mut b_hi = 0.0; // B(hi, t1)
        for &(a, sigma, lo, hi) in segments.iter().rev() {
            let delta = hi - lo;
            // for u in the segment, w = hi - u:
            //   e^{-K(u,t1)} = decay_hi e^{-a w}
            //   B(u,t1)      = (1 - e^{-a w})/a + e^{-a w} B(hi,t1)
            let single = (1.0 - (-a * delta).exp()) / a;
            let double = (1.0 - (-2.0 * a * delta).exp()) / (2.0 * a);
            total += sigma * sigma * decay_hi * (single / a + (b_hi - 1.0 / a) * double);
            b_hi = single + (-a * delta).exp() * b_hi;
            decay_hi *= (-a * delta).exp();
        }
        total
    }

    /// `W(t0, t1) = int_{t0}^{t1} sigma(s)^2 B(s, t1)^2 ds`: the variance
    /// of the integrated state `int_{t0}^{t1} x(u) du` given `x(t0)`,
    /// whose covariance with `x(t1)` is [`forward_measure_shift`] and
    /// whose conditional mean is `x(t0) B(t0, t1)` — the moments an
    /// exact simulation of the bank account needs.
    ///
    /// [`forward_measure_shift`]: Self::forward_measure_shift
    pub fn integrated_variance(&self, t0: f64, t1: f64) -> f64 {
        let segments = self.segments(t0, t1);
        let mut total = 0.0;
        let mut b_hi = 0.0; // B(hi, t1)
        for &(a, sigma, lo, hi) in segments.iter().rev() {
            let delta = hi - lo;
            // for s in the segment, w = hi - s: B(s, t1) = c1 + c2 e^{-a w}
            let c1 = 1.0 / a;
            let c2 = b_hi - c1;
            let single = (1.0 - (-a * delta).exp()) / a;
            let double = (1.0 - (-2.0 * a * delta).exp()) / (2.0 * a);
            total += sigma * sigma * (c1 * c1 * delta + 2.0 * c1 * c2 * single + c2 * c2 * double);
            b_hi = single + (-a * delta).exp() * b_hi;
        }
        total
    }

    /// `int_{t0}^{t1} alpha(u) du`: the deterministic part of the
    /// integrated short rate — `ln P(0,t0)/P(0,t1)` from the forwards
    /// plus the integrated convexity term by Simpson quadrature.
    pub fn integrated_alpha(&self, t0: f64, t1: f64) -> f64 {
        let forwards = (self.curve.df(t0) / self.curve.df(t1)).ln();
        let n = 32;
        let h = (t1 - t0) / n as f64;
        let mut convexity = 0.0;
        for k in 0..=n {
            let w = if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            convexity += w * self.forward_measure_shift(0.0, t0 + k as f64 * h);
        }
        forwards + convexity * h / 3.0
    }

    /// `alpha(t) = f(0,t) + M(0,t)` — the mean around which the short
    /// rate reverts under the fit (`sigma^2/(2a^2)(1 - e^{-at})^2`
    /// above the forward for constant coefficients).
    pub fn alpha(&self, t: f64) -> f64 {
        instantaneous_forward(&self.curve, t) + self.forward_measure_shift(0.0, t)
    }

    /// Unconditional mean of the short rate at `t` (from `r(0) = f(0,0)`).
    pub fn expected_short_rate(&self, t: f64) -> f64 {
        // E[r(t)] = alpha(t) + (r0 - alpha(0)) e^{-K(0,t)}; alpha(0) = r0
        self.alpha(t)
    }

    /// Conditional standard deviation of `r(t + dt)` given `r(t)`.
    pub fn short_rate_std(&self, t: f64, dt: f64) -> f64 {
        self.short_rate_variance(t, t + dt).sqrt()
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
        let vol_term = 0.5 * b_factor * b_factor * self.short_rate_variance(0.0, t);
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
        //   + (r(t) - alpha(t)) e^{-K(t,t+dt)} + std * z
        let mean = self.alpha(t + dt) + (short_rate - self.alpha(t)) * self.decay(t, t + dt);
        Ok(mean + self.short_rate_std(t, dt) * z)
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
        let p_settlement = self.curve.df(settlement);
        let p_bond = self.curve.df(bond_maturity);
        // the price ratio P(e, bond)/P(e, settlement) is lognormal with
        // variance (B(e, bond) - B(e, settlement))^2 V(0, e)
        let sigma_p = self.short_rate_variance(0.0, expiry).sqrt()
            * (self.b_factor(expiry, bond_maturity) - self.b_factor(expiry, settlement));
        Ok(gaussian_zero_bond_option(
            p_settlement,
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

    /// A sigma term structure: 120bp for the first year, 90bp to year
    /// three, 70bp after.
    fn piecewise() -> HullWhite {
        HullWhite::with_piecewise_sigma(0.1, &[1.0, 3.0], &[0.012, 0.009, 0.007], market_curve())
            .unwrap()
    }

    /// Both term structures: mean reversion 3% to year two then 12%,
    /// sigma as above on its own breakpoints.
    fn generalized() -> HullWhite {
        HullWhite::generalized(
            &[2.0],
            &[0.03, 0.12],
            &[1.0, 3.0],
            &[0.012, 0.009, 0.007],
            market_curve(),
        )
        .unwrap()
    }

    /// MC-discount 1 unit to `horizon` along exact-transition paths.
    fn simulated_df(m: &HullWhite, horizon: f64, steps: usize, paths: usize, seed: u64) -> f64 {
        use rand::{Rng, SeedableRng};
        let dt = horizon / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(seed);
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
        sum / paths as f64
    }

    #[test]
    fn fitted_model_reproduces_the_input_curve_exactly() {
        // the whole point of the extension: P(0,T) = market df(T), for
        // constant, piecewise-sigma and fully generalized coefficients
        for m in [model(), piecewise(), generalized()] {
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
    }

    #[test]
    fn constant_coefficients_reduce_to_the_textbook_closed_forms() {
        let m = model();
        let (a, sigma) = (m.a(), m.sigma());
        for t in [0.3, 1.0, 4.5, 20.0] {
            let b = (1.0 - (-a * t).exp()) / a;
            assert!((m.b_factor(0.0, t) - b).abs() < 1e-15, "B({t})");
            let variance = sigma * sigma * (1.0 - (-2.0 * a * t).exp()) / (2.0 * a);
            assert!(
                (m.short_rate_variance(0.0, t) - variance).abs() < 1e-18,
                "V({t})"
            );
            let convexity = 0.5 * sigma * sigma * b * b;
            assert!(
                (m.forward_measure_shift(0.0, t) - convexity).abs() < 1e-18,
                "M({t})"
            );
            assert!((m.decay(0.0, t) - (-a * t).exp()).abs() < 1e-16);
        }
        // breakpoints with equal values on both sides change nothing
        let split =
            HullWhite::generalized(&[1.5], &[a, a], &[2.0], &[sigma, sigma], market_curve())
                .unwrap();
        for t in [1.0, 2.0, 3.5, 10.0] {
            assert!((split.alpha(t) - m.alpha(t)).abs() < 1e-15);
            assert!((split.short_rate_std(0.5, t) - m.short_rate_std(0.5, t)).abs() < 1e-15);
            assert!((split.b_factor(0.7, t + 1.0) - m.b_factor(0.7, t + 1.0)).abs() < 1e-13);
            assert!(
                (split.zero_bond(1.0, t + 1.0, 0.03).unwrap()
                    - m.zero_bond(1.0, t + 1.0, 0.03).unwrap())
                .abs()
                    < 1e-15
            );
        }
        assert_eq!(m.sigma_at(100.0), sigma);
        assert_eq!(piecewise().sigma_at(0.5), 0.012);
        assert_eq!(piecewise().sigma_at(1.0), 0.009);
        assert_eq!(piecewise().sigma_at(3.0), 0.007);
        assert_eq!(generalized().a_at(1.9), 0.03);
        assert_eq!(generalized().a_at(2.0), 0.12);
    }

    #[test]
    fn integrals_are_additive_across_breakpoints() {
        for m in [piecewise(), generalized()] {
            let (s, t, u) = (0.7, 2.6, 4.2);
            // B(s,u) = B(s,t) + e^{-K(s,t)} B(t,u)
            let b = m.b_factor(s, t) + m.decay(s, t) * m.b_factor(t, u);
            assert!(
                (m.b_factor(s, u) - b).abs() < 1e-15,
                "B: {} vs {b}",
                m.b_factor(s, u)
            );
            // V(0,u) = V(0,t) e^{-2K(t,u)} + V(t,u)
            let v =
                m.short_rate_variance(0.0, t) * m.decay(t, u).powi(2) + m.short_rate_variance(t, u);
            assert!((m.short_rate_variance(0.0, u) - v).abs() < 1e-16);
            // decay composes
            assert!((m.decay(s, u) - m.decay(s, t) * m.decay(t, u)).abs() < 1e-16);
        }
        // the piecewise variance lies between the constant extremes
        let m = piecewise();
        let hi = HullWhite::new(m.a(), 0.012, market_curve()).unwrap();
        let lo = HullWhite::new(m.a(), 0.007, market_curve()).unwrap();
        let v = m.short_rate_variance(0.0, 4.2);
        assert!(lo.short_rate_variance(0.0, 4.2) < v && v < hi.short_rate_variance(0.0, 4.2));
    }

    #[test]
    fn zero_bond_options_satisfy_parity_against_the_market_curve() {
        for m in [model(), generalized()] {
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
        }
        // more volatility, more option value
        let m = model();
        let call = m.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Call).unwrap();
        let calm = HullWhite::new(0.1, 0.002, market_curve()).unwrap();
        assert!(
            calm.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Call)
                .unwrap()
                < call
        );
    }

    #[test]
    fn only_the_volatility_before_expiry_prices_a_bond_option() {
        // a 1y option sees V(0, 1): the 120bp first year of the piecewise
        // model, whatever comes after — so it prices as the constant
        // 120bp model, and a 2y option (which sees the 90bp year) does not
        let pw = piecewise();
        let flat = HullWhite::new(0.1, 0.012, market_curve()).unwrap();
        let one_year = |m: &HullWhite| m.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Put).unwrap();
        assert!((one_year(&pw) - one_year(&flat)).abs() < 1e-15);
        let two_year = |m: &HullWhite| m.zero_bond_option(2.0, 6.0, 0.83, PutOrCall::Put).unwrap();
        assert!(two_year(&pw) < two_year(&flat));
    }

    #[test]
    fn mean_path_follows_alpha_and_z_zero_evolution_tracks_it() {
        for m in [model(), piecewise(), generalized()] {
            // chaining z = 0 steps reproduces the unconditional mean
            // exactly (the transition is linear in the state)
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
        // does exactly this for its funding leg). With time-dependent
        // coefficients this checks alpha, the decay and the step
        // variance agree across every breakpoint
        for (m, seed) in [(model(), 7), (piecewise(), 11), (generalized(), 13)] {
            let mc_df = simulated_df(&m, 5.0, 100, 20_000, seed);
            let market_df = m.curve().df(5.0);
            assert!(
                (mc_df / market_df - 1.0).abs() < 0.005,
                "MC {mc_df} vs market {market_df}"
            );
        }
    }

    #[test]
    fn integrated_moments_match_the_closed_forms_and_the_bank_account() {
        // constant coefficients: Var(int x) = sigma^2/a^2 [dt - 2B + (1 - e^{-2a dt})/(2a)]
        let m = model();
        let (a, sigma) = (m.a(), m.sigma());
        for dt in [0.25, 1.0, 4.0] {
            let b = (1.0 - (-a * dt).exp()) / a;
            let closed = sigma * sigma / (a * a)
                * (dt - 2.0 * b + (1.0 - (-2.0 * a * dt).exp()) / (2.0 * a));
            assert!(
                (m.integrated_variance(2.0, 2.0 + dt) - closed).abs() < 1e-13 * closed,
                "W({dt})"
            );
        }
        // the integral of alpha is the log discount ratio plus convexity,
        // additive across breakpoints for the generalized model
        for m in [model(), generalized()] {
            let whole = m.integrated_alpha(0.5, 4.5);
            let split = m.integrated_alpha(0.5, 2.0) + m.integrated_alpha(2.0, 4.5);
            assert!((whole - split).abs() < 1e-12, "{whole} vs {split}");
            let forwards = (m.curve().df(0.5) / m.curve().df(4.5)).ln();
            assert!(whole > forwards);
        }
        // the integrated variance is positive and grows with the horizon
        let g = generalized();
        assert!(g.integrated_variance(0.0, 1.0) < g.integrated_variance(0.0, 3.0));
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(HullWhite::new(0.0, 0.01, market_curve()).is_err());
        assert!(HullWhite::new(0.1, -0.01, market_curve()).is_err());
        assert!(HullWhite::with_piecewise_sigma(0.1, &[1.0], &[0.01], market_curve()).is_err());
        assert!(HullWhite::with_piecewise_sigma(
            0.1,
            &[2.0, 1.0],
            &[0.01, 0.01, 0.01],
            market_curve()
        )
        .is_err());
        assert!(
            HullWhite::with_piecewise_sigma(0.1, &[0.0], &[0.01, 0.01], market_curve()).is_err()
        );
        assert!(HullWhite::generalized(&[1.0], &[0.1, 0.0], &[], &[0.01], market_curve()).is_err());
        let m = model();
        assert!(m.zero_bond(3.0, 2.0, 0.03).is_err());
        assert!(m.evolve(0.0, 0.03, -1.0, 0.0).is_err());
        assert!(m.zero_bond_option(0.0, 5.0, 0.8, PutOrCall::Call).is_err());
        // refit keeps both term structures
        let refit = generalized().refit(market_curve()).unwrap();
        assert_eq!(refit.sigmas(), generalized().sigmas());
        assert_eq!(refit.mean_reversions(), generalized().mean_reversions());
        assert_eq!(refit.a_times(), generalized().a_times());
    }
}
