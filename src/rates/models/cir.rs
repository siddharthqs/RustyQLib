//! Cox-Ingersoll-Ross (1985): `dr = a(b - r)dt + sigma sqrt(r) dW`, and
//! its curve-fitted extension **CIR++** (Brigo-Mercurio ch. 3.9):
//! `r(t) = x(t) + phi(t)` with `x` a CIR process and `phi` the
//! deterministic shift that reproduces the input curve exactly.
//!
//! Square-root dynamics keep the rate non-negative (strictly positive
//! under the Feller condition `2ab >= sigma^2`) at the cost of Gaussian
//! tractability: bond prices stay affine in closed form, and zero-bond
//! options are closed form too, through the **noncentral chi-square**
//! distribution of `r(T)` (Brigo-Mercurio eq. 3.26) — so Jamshidian
//! swaptions and caps price analytically under both models. The
//! transition is simulated with a full-truncation Euler step (bias
//! `O(dt)` — use small steps).
//!
//! CIR++ is QuantLib's `ExtendedCoxIngersollRoss`: every CIR++ bond
//! price is the CIR bond on `x` times a deterministic ratio of market
//! and CIR discount factors, and every CIR++ bond option is the CIR
//! option on `x` with the strike rescaled by that ratio, so the whole
//! analytic layer transfers.

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::special::noncentral_chi_square_cdf;
use crate::core::trade::PutOrCall;
use crate::rates::models::hull_white::instantaneous_forward;
use crate::rates::models::{
    gaussian_zero_bond_option, validate_bond_option_terms, OneFactorAffine, ShortRateModel,
};

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

    /// `h = sqrt(a^2 + 2 sigma^2)`.
    fn gamma(&self) -> f64 {
        (self.a * self.a + 2.0 * self.sigma * self.sigma).sqrt()
    }

    /// The affine coefficients `(A, B)` of `P(t,T) = A e^{-B r}` over `tau`.
    fn affine(&self, tau: f64) -> (f64, f64) {
        if self.sigma == 0.0 {
            // deterministic mean-reverting limit
            let b_det = (1.0 - (-self.a * tau).exp()) / self.a;
            return ((-self.b * (tau - b_det)).exp(), b_det);
        }
        let gamma = self.gamma();
        let e = (gamma * tau).exp();
        let denominator = (gamma + self.a) * (e - 1.0) + 2.0 * gamma;
        let b_factor = 2.0 * (e - 1.0) / denominator;
        let a_factor = (2.0 * gamma * ((self.a + gamma) * tau / 2.0).exp() / denominator)
            .powf(2.0 * self.a * self.b / (self.sigma * self.sigma));
        (a_factor, b_factor)
    }

    /// The bond-option call of Brigo-Mercurio eq. 3.26, at time 0 from
    /// state `x0`: a call expiring at `expiry` on the bond maturing at
    /// `bond_maturity`, struck at `strike`. Puts follow by parity.
    fn bond_option_from(
        &self,
        x0: f64,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        let p_expiry = self.zero_bond(0.0, expiry, x0)?;
        let p_bond = self.zero_bond(0.0, bond_maturity, x0)?;
        if self.sigma == 0.0 {
            return Ok(gaussian_zero_bond_option(
                p_expiry,
                p_bond,
                strike,
                0.0,
                put_or_call,
            ));
        }
        let h = self.gamma();
        let sigma2 = self.sigma * self.sigma;
        let rho = 2.0 * h / (sigma2 * ((h * expiry).exp() - 1.0));
        let psi = (self.a + h) / sigma2;
        let (a_ts, b_ts) = self.affine(bond_maturity - expiry);
        // the rate at expiry where the bond is worth the strike
        let r_bar = (a_ts / strike).ln() / b_ts;
        let df = 4.0 * self.a * self.b / sigma2;
        let scale = 2.0 * rho * rho * x0 * (h * expiry).exp();
        let call = if r_bar <= 0.0 {
            // the bond is worth less than the strike at every non-negative rate
            0.0
        } else {
            let first = noncentral_chi_square_cdf(
                2.0 * r_bar * (rho + psi + b_ts),
                df,
                scale / (rho + psi + b_ts),
            );
            let second =
                noncentral_chi_square_cdf(2.0 * r_bar * (rho + psi), df, scale / (rho + psi));
            p_bond * first - strike * p_expiry * second
        };
        Ok(match put_or_call {
            PutOrCall::Call => call.max(0.0),
            PutOrCall::Put => (call - p_bond + strike * p_expiry).max(0.0),
        })
    }
}

impl ShortRateModel for CoxIngersollRoss {
    fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn short_rate_floor(&self, _t: f64) -> f64 {
        0.0
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
        let (a_factor, b_factor) = self.affine(maturity - t);
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

impl OneFactorAffine for CoxIngersollRoss {
    fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        validate_bond_option_terms(expiry, expiry, bond_maturity, strike)?;
        self.bond_option_from(self.r0, expiry, bond_maturity, strike, put_or_call)
    }
}

/// CIR++: `r(t) = x(t) + phi(t)`, `x` a [`CoxIngersollRoss`] process
/// started at `x0`, `phi` the shift fitting the curve exactly.
#[derive(Debug, Clone)]
pub struct ExtendedCir {
    cir: CoxIngersollRoss,
    curve: YieldCurve,
    r0: f64,
}

impl ExtendedCir {
    /// The CIR process `(a, b, sigma)` on `x` from `x0`, shifted to fit
    /// `curve`. `x0` is a free parameter (it need not be the short rate:
    /// `phi(0)` absorbs the difference); with `x0` near the short rate
    /// the shift stays small and the rate positive.
    pub fn new(
        a: f64,
        b: f64,
        sigma: f64,
        x0: f64,
        curve: YieldCurve,
    ) -> Result<Self, RustyQLibError> {
        let cir = CoxIngersollRoss::new(a, b, sigma, x0)?;
        let mut model = ExtendedCir {
            cir,
            curve,
            r0: 0.0,
        };
        // r(0) = x0 + phi(0), with the same numerical phi the bond
        // formulas use, so the state maps back to x0 exactly at t = 0
        model.r0 = x0 + model.shift(0.0);
        if !model.r0.is_finite() {
            return Err(RustyQLibError::NumericalError(
                "the curve's short-end forward is not finite".to_string(),
            ));
        }
        Ok(model)
    }

    /// The underlying CIR process (its `r0` is `x0`).
    pub fn cir(&self) -> &CoxIngersollRoss {
        &self.cir
    }

    /// The fitted curve.
    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    /// `P^CIR(0, t)` from `x0`.
    fn cir_df(&self, t: f64) -> f64 {
        self.cir
            .zero_bond(0.0, t, self.cir.r0)
            .expect("non-negative x0 and t")
    }

    /// The CIR instantaneous forward `f^CIR(0,t)` by symmetric difference.
    fn cir_forward(&self, t: f64) -> f64 {
        let h = 1e-5;
        let lo = (t - h).max(0.0);
        let hi = t + h;
        (self.cir_df(lo).ln() - self.cir_df(hi).ln()) / (hi - lo)
    }

    /// The shift `phi(t) = f(0,t) - f^CIR(0,t)`.
    pub fn shift(&self, t: f64) -> f64 {
        instantaneous_forward(&self.curve, t) - self.cir_forward(t)
    }

    /// `e^{-int_t^T phi} = [P(0,T)/P(0,t)] / [P^CIR(0,T)/P^CIR(0,t)]`: the
    /// deterministic factor between CIR++ and CIR bond prices.
    fn shift_ratio(&self, t: f64, maturity: f64) -> f64 {
        (self.curve.df(maturity) / self.curve.df(t)) / (self.cir_df(maturity) / self.cir_df(t))
    }
}

impl ShortRateModel for ExtendedCir {
    fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn short_rate_floor(&self, t: f64) -> f64 {
        self.shift(t)
    }

    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError> {
        if !(maturity >= t && t >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "cir++",
                format!("need 0 <= t <= maturity, got t={t}, maturity={maturity}"),
            ));
        }
        let x = short_rate - self.shift(t);
        Ok(self.shift_ratio(t, maturity) * self.cir.zero_bond(t, maturity, x)?)
    }

    fn evolve(&self, t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError> {
        let x = short_rate - self.shift(t);
        Ok(self.cir.evolve(t, x, dt, z)? + self.shift(t + dt))
    }
}

impl OneFactorAffine for ExtendedCir {
    fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        validate_bond_option_terms(expiry, expiry, bond_maturity, strike)?;
        // P++(T,S) = ratio(T,S) P^CIR(T,S; x_T) and the CIR++ discount
        // factor to T is the CIR one times ratio(0,T), so the option is
        // the CIR option on x at the rescaled strike
        let ratio = self.shift_ratio(expiry, bond_maturity);
        Ok(self.shift_ratio(0.0, expiry)
            * ratio
            * self.cir.bond_option_from(
                self.cir.r0,
                expiry,
                bond_maturity,
                strike / ratio,
                put_or_call,
            )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::engines::jamshidian::european_swaption;
    use crate::rates::PayerReceiver;
    use chrono::NaiveDate;

    fn model() -> CoxIngersollRoss {
        // 2ab = 0.006 >= sigma^2 = 0.0036: Feller holds
        CoxIngersollRoss::new(0.3, 0.05, 0.06, 0.03).unwrap()
    }

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
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn extended() -> ExtendedCir {
        ExtendedCir::new(0.3, 0.045, 0.06, 0.04, market_curve()).unwrap()
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
    fn bond_options_satisfy_parity_grow_with_vol_and_match_monte_carlo() {
        let m = model();
        let (expiry, maturity) = (1.0, 4.0);
        let p_expiry = m.zero_bond(0.0, expiry, m.r0).unwrap();
        let p_bond = m.zero_bond(0.0, maturity, m.r0).unwrap();
        let strike = p_bond / p_expiry; // at the forward
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        let put = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        assert!(
            (call - put - (p_bond - strike * p_expiry)).abs() < 1e-12,
            "{call} vs {put}"
        );
        assert!(call > 0.0 && put > 0.0);
        // more volatility, more option value; zero vol is intrinsic
        let calm = CoxIngersollRoss::new(0.3, 0.05, 0.03, 0.03).unwrap();
        assert!(
            calm.zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
                .unwrap()
                < call
        );
        let still = CoxIngersollRoss::new(0.3, 0.05, 0.0, 0.03).unwrap();
        let ps = still.zero_bond(0.0, expiry, 0.03).unwrap();
        let pb = still.zero_bond(0.0, maturity, 0.03).unwrap();
        let intrinsic = (pb - 0.9 * ps).max(0.0);
        assert!(
            (still
                .zero_bond_option(expiry, maturity, 0.9, PutOrCall::Call)
                .unwrap()
                - intrinsic)
                .abs()
                < 1e-15
        );
        // Monte Carlo on the truncated Euler scheme, fine steps: the
        // noncentral chi-square formula is confirmed to a few percent
        use rand::{Rng, SeedableRng};
        let (steps, paths) = (400usize, 40_000usize);
        let dt = expiry / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(3);
        let mut sum = 0.0;
        for _ in 0..paths {
            let mut r = m.r0;
            let mut integral = 0.0;
            for _ in 0..steps {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                let next = m.evolve(0.0, r, dt, z).unwrap();
                integral += 0.5 * (r.max(0.0) + next.max(0.0)) * dt;
                r = next;
            }
            let bond = m.zero_bond(expiry, maturity, r.max(0.0)).unwrap();
            sum += (-integral).exp() * (bond - strike).max(0.0);
        }
        let mc = sum / paths as f64;
        assert!(
            (mc - call).abs() < 0.03 * call,
            "MC {mc} vs closed form {call}"
        );
    }

    #[test]
    fn extended_cir_reproduces_the_curve_and_keeps_the_analytic_layer() {
        let m = extended();
        let r0 = m.initial_short_rate();
        for t in [0.25, 1.0, 3.3, 7.0, 12.0, 25.0] {
            let model_df = m.zero_bond(0.0, t, r0).unwrap();
            let market_df = m.curve().df(t);
            assert!(
                (model_df / market_df - 1.0).abs() < 1e-9,
                "t={t}: {model_df} vs {market_df}"
            );
        }
        // the shift is small when x0 sits near the short rate
        for t in [0.5, 2.0, 5.0] {
            assert!(m.shift(t).abs() < 0.02, "phi({t}) = {}", m.shift(t));
        }
        // bond-option parity against the market curve
        let (expiry, maturity, strike) = (1.0, 5.0, 0.83);
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        let put = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let parity = m.curve().df(maturity) - strike * m.curve().df(expiry);
        assert!((call - put - parity).abs() < 1e-12, "{call} vs {put}");
        assert!(call > 0.0 && put > 0.0);
        // a Jamshidian swaption prices, and payer - receiver is the forward swap
        let leg: Vec<(f64, f64)> = (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect();
        let payer = european_swaption(&m, 1.0, &leg, 0.045, 1.0, PayerReceiver::Payer).unwrap();
        let receiver =
            european_swaption(&m, 1.0, &leg, 0.045, 1.0, PayerReceiver::Receiver).unwrap();
        let curve = m.curve();
        let fixed: f64 = leg
            .iter()
            .map(|&(t, tau)| 0.045 * tau * curve.df(t))
            .sum::<f64>()
            + curve.df(6.0);
        assert!((payer - receiver - (curve.df(1.0) - fixed)).abs() < 1e-9);
        assert!(payer > 0.0);
        // a settlement lag has no closed form under square-root dynamics
        assert!(m
            .zero_bond_exchange_option(1.0, 1.01, 5.0, 0.83, PutOrCall::Call)
            .is_err());
    }

    #[test]
    fn extended_cir_simulation_recovers_the_curve() {
        use rand::{Rng, SeedableRng};
        let m = extended();
        let (horizon, steps, paths) = (3.0_f64, 300usize, 20_000usize);
        let dt = horizon / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(5);
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
            (mc_df / market_df - 1.0).abs() < 0.01,
            "MC {mc_df} vs market {market_df}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(CoxIngersollRoss::new(0.0, 0.05, 0.06, 0.03).is_err());
        assert!(CoxIngersollRoss::new(0.3, 0.05, 0.06, -0.01).is_err());
        let m = model();
        assert!(m.zero_bond(2.0, 1.0, 0.03).is_err());
        assert!(m.zero_bond(0.0, 5.0, -0.01).is_err());
        assert!(m.evolve(0.0, 0.03, 0.0, 0.0).is_err());
        assert!(m.zero_bond_option(2.0, 1.0, 0.9, PutOrCall::Call).is_err());
        assert!(ExtendedCir::new(0.3, 0.05, 0.06, -0.1, market_curve()).is_err());
    }
}
