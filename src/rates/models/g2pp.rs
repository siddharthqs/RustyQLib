//! G2++: the two-additive-factor Gaussian model (Brigo-Mercurio ch. 4),
//! `r(t) = x(t) + y(t) + phi(t)` with
//!
//! ```text
//! dx = -a x dt + sigma dW1,   dy = -b y dt + eta dW2,   dW1 dW2 = rho dt
//! ```
//!
//! and `phi` the deterministic shift that reproduces the input curve
//! exactly. Two factors give the model what one cannot: a non-perfect
//! correlation between rates of different maturities, hence a
//! realistic fit to the whole swaption matrix rather than one column,
//! and decorrelation for products like CMS spread options.
//!
//! Everything Gaussian carries over from Hull-White:
//!
//! - bond prices are exponential-affine in `(x, y)` with a closed-form
//!   convexity term `V`;
//! - zero-bond (exchange) options are Black-style with the price-ratio
//!   variance built from the two loadings and their correlation;
//! - the transition is an exact bivariate Gaussian, so simulation is
//!   bias-free at any step;
//! - a European swaption is a one-dimensional integral (Brigo-Mercurio
//!   theorem 4.2.3): under the expiry-forward measure the fixed leg is
//!   monotone in `y` for each `x`, so the exercise boundary `y*(x)` is
//!   a root, the inner `y`-expectation is closed form, and only the
//!   outer `x`-integral is numerical.
//!
//! With `eta = 0` the model is Hull-White `(a, sigma)`, and with
//! `a = b`, `sigma = eta`, `rho = 0` it is Hull-White `(a, sigma
//! sqrt 2)` — the two identities the tests lean on.

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::core::utils::{norm_cdf, norm_pdf};
use crate::rates::models::hull_white::instantaneous_forward;
use crate::rates::models::{gaussian_zero_bond_option, validate_bond_option_terms};
use crate::rates::PayerReceiver;

const FIELD: &str = "g2++";
/// Outer `x`-integration nodes (odd, Simpson) and half-width in stds.
const X_NODES: usize = 401;
const X_SPAN: f64 = 8.0;

#[derive(Debug, Clone)]
pub struct G2pp {
    pub a: f64,
    pub b: f64,
    pub sigma: f64,
    pub eta: f64,
    pub rho: f64,
    curve: YieldCurve,
    r0: f64,
}

impl G2pp {
    pub fn new(
        a: f64,
        b: f64,
        sigma: f64,
        eta: f64,
        rho: f64,
        curve: YieldCurve,
    ) -> Result<Self, RustyQLibError> {
        for (name, v) in [("a", a), ("b", b)] {
            if !(v > 0.0 && v.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("{name} must be positive, got {v}"),
                ));
            }
        }
        for (name, v) in [("sigma", sigma), ("eta", eta)] {
            if !(v >= 0.0 && v.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("{name} must be non-negative, got {v}"),
                ));
            }
        }
        if !(-1.0..=1.0).contains(&rho) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("rho must lie in [-1, 1], got {rho}"),
            ));
        }
        let r0 = instantaneous_forward(&curve, 0.0);
        if !r0.is_finite() {
            return Err(RustyQLibError::NumericalError(
                "the curve's short-end forward is not finite".to_string(),
            ));
        }
        Ok(G2pp {
            a,
            b,
            sigma,
            eta,
            rho,
            curve,
            r0,
        })
    }

    /// The same dynamics fitted to another curve.
    pub fn refit(&self, curve: YieldCurve) -> Result<Self, RustyQLibError> {
        Self::new(self.a, self.b, self.sigma, self.eta, self.rho, curve)
    }

    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    /// `r(0) = f(0, 0)`.
    pub fn initial_short_rate(&self) -> f64 {
        self.r0
    }

    fn b_a(&self, tau: f64) -> f64 {
        (1.0 - (-self.a * tau).exp()) / self.a
    }

    fn b_b(&self, tau: f64) -> f64 {
        (1.0 - (-self.b * tau).exp()) / self.b
    }

    /// `V(t,T)`: the variance of `int_t^T (x + y) du` (eq. 4.10).
    fn v(&self, tau: f64) -> f64 {
        let (a, b, s, e, rho) = (self.a, self.b, self.sigma, self.eta, self.rho);
        let term_x = s * s / (a * a)
            * (tau + 2.0 / a * (-a * tau).exp() - 0.5 / a * (-2.0 * a * tau).exp() - 1.5 / a);
        let term_y = e * e / (b * b)
            * (tau + 2.0 / b * (-b * tau).exp() - 0.5 / b * (-2.0 * b * tau).exp() - 1.5 / b);
        let cross = 2.0 * rho * s * e / (a * b)
            * (tau + ((-a * tau).exp() - 1.0) / a + ((-b * tau).exp() - 1.0) / b
                - ((-(a + b) * tau).exp() - 1.0) / (a + b));
        term_x + term_y + cross
    }

    /// The shift `phi(t) = f(0,t) + convexity` (eq. 4.12).
    pub fn phi(&self, t: f64) -> f64 {
        let (a, b, s, e, rho) = (self.a, self.b, self.sigma, self.eta, self.rho);
        let ea = 1.0 - (-a * t).exp();
        let eb = 1.0 - (-b * t).exp();
        instantaneous_forward(&self.curve, t)
            + s * s / (2.0 * a * a) * ea * ea
            + e * e / (2.0 * b * b) * eb * eb
            + rho * s * e / (a * b) * ea * eb
    }

    /// The short rate at state `(x, y)`.
    pub fn short_rate(&self, t: f64, x: f64, y: f64) -> f64 {
        x + y + self.phi(t)
    }

    /// `P(t, T)` at state `(x, y)` (eq. 4.14).
    pub fn zero_bond(&self, t: f64, maturity: f64, x: f64, y: f64) -> Result<f64, RustyQLibError> {
        if !(maturity >= t && t >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("need 0 <= t <= maturity, got t={t}, maturity={maturity}"),
            ));
        }
        let tau = maturity - t;
        let ratio = self.curve.df(maturity) / self.curve.df(t);
        let exponent = 0.5 * (self.v(tau) - self.v(maturity) + self.v(t))
            - self.b_a(tau) * x
            - self.b_b(tau) * y;
        Ok(ratio * exponent.exp())
    }

    /// The conditional covariance of `(x, y)` over a step `dt`:
    /// `(var_x, var_y, cov)`.
    pub fn step_covariance(&self, dt: f64) -> (f64, f64, f64) {
        let var_x = self.sigma * self.sigma * (1.0 - (-2.0 * self.a * dt).exp()) / (2.0 * self.a);
        let var_y = self.eta * self.eta * (1.0 - (-2.0 * self.b * dt).exp()) / (2.0 * self.b);
        let cov = self.rho * self.sigma * self.eta * (1.0 - (-(self.a + self.b) * dt).exp())
            / (self.a + self.b);
        (var_x, var_y, cov)
    }

    /// One exact transition of `(x, y)` over `dt` from two independent
    /// standard normal draws.
    pub fn evolve(
        &self,
        x: f64,
        y: f64,
        dt: f64,
        z1: f64,
        z2: f64,
    ) -> Result<(f64, f64), RustyQLibError> {
        if !(dt > 0.0 && dt.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("dt must be positive, got {dt}"),
            ));
        }
        let (var_x, var_y, cov) = self.step_covariance(dt);
        let sx = var_x.sqrt();
        let sy = var_y.sqrt();
        let corr = if sx > 0.0 && sy > 0.0 {
            cov / (sx * sy)
        } else {
            0.0
        };
        let nx = x * (-self.a * dt).exp() + sx * z1;
        let ny =
            y * (-self.b * dt).exp() + sy * (corr * z1 + (1.0 - corr * corr).max(0.0).sqrt() * z2);
        Ok((nx, ny))
    }

    /// Mean and covariance of `(x_T, y_T)` under the `T`-forward
    /// measure (eq. 4.16-4.19): `(mu_x, mu_y, sigma_x, sigma_y, rho_xy)`.
    fn forward_measure_law(&self, expiry: f64) -> (f64, f64, f64, f64, f64) {
        let (a, b, s, e, rho) = (self.a, self.b, self.sigma, self.eta, self.rho);
        let t = expiry;
        let mu_x = -((s * s / (a * a) + rho * s * e / (a * b)) * (1.0 - (-a * t).exp())
            - s * s / (2.0 * a * a) * (1.0 - (-2.0 * a * t).exp())
            - rho * s * e / (b * (a + b)) * (1.0 - (-(a + b) * t).exp()));
        let mu_y = -((e * e / (b * b) + rho * s * e / (a * b)) * (1.0 - (-b * t).exp())
            - e * e / (2.0 * b * b) * (1.0 - (-2.0 * b * t).exp())
            - rho * s * e / (a * (a + b)) * (1.0 - (-(a + b) * t).exp()));
        let (var_x, var_y, cov) = self.step_covariance(t);
        let sx = var_x.sqrt();
        let sy = var_y.sqrt();
        let rho_xy = if sx > 0.0 && sy > 0.0 {
            cov / (sx * sy)
        } else {
            0.0
        };
        (mu_x, mu_y, sx, sy, rho_xy)
    }

    /// Variance of `ln P(e, bond) - ln P(e, settlement)` seen from 0.
    fn price_ratio_variance(&self, expiry: f64, settlement: f64, bond_maturity: f64) -> f64 {
        let ca = self.b_a(bond_maturity - expiry) - self.b_a(settlement - expiry);
        let cb = self.b_b(bond_maturity - expiry) - self.b_b(settlement - expiry);
        let (var_x, var_y, cov) = self.step_covariance(expiry);
        ca * ca * var_x + cb * cb * var_y + 2.0 * ca * cb * cov
    }

    /// European option on a zero-coupon bond (eq. 4.30, Black-style).
    pub fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        self.zero_bond_exchange_option(expiry, expiry, bond_maturity, strike, put_or_call)
    }

    /// Option at `expiry` to exchange `strike` settlement bonds for the
    /// `bond_maturity` bond (the plain option when `settlement == expiry`).
    pub fn zero_bond_exchange_option(
        &self,
        expiry: f64,
        settlement: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        validate_bond_option_terms(expiry, settlement, bond_maturity, strike)?;
        let sigma_p = self
            .price_ratio_variance(expiry, settlement, bond_maturity)
            .sqrt();
        Ok(gaussian_zero_bond_option(
            self.curve.df(settlement),
            self.curve.df(bond_maturity),
            strike,
            sigma_p,
            put_or_call,
        ))
    }

    /// European swaption on a swap starting at `swap_start >= expiry`:
    /// the fixed leg pays `notional * strike_rate * tau` at each
    /// `(payment_time, tau)` and the notional at the last payment
    /// (theorem 4.2.3, generalized to a settlement lag). The payer
    /// payoff at expiry is `N [P(e, start) - sum c_i P(e, t_i)]^+`.
    pub fn european_swaption(
        &self,
        expiry: f64,
        swap_start: f64,
        fixed_leg: &[(f64, f64)],
        strike_rate: f64,
        notional: f64,
        payer_receiver: PayerReceiver,
    ) -> Result<f64, RustyQLibError> {
        if !(expiry > 0.0 && swap_start >= expiry && notional > 0.0 && strike_rate > 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need expiry > 0, swap start >= expiry, positive notional and strike",
            ));
        }
        if fixed_leg.is_empty()
            || fixed_leg
                .iter()
                .any(|&(t, tau)| t <= swap_start || tau <= 0.0)
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "the fixed leg needs payments strictly after the swap start with positive accruals",
            ));
        }
        // the signed flows at expiry: +1 at the start, -c_i at the coupons
        let mut flows: Vec<(f64, f64)> = vec![(swap_start, 1.0)];
        for (i, &(t, tau)) in fixed_leg.iter().enumerate() {
            let mut amount = strike_rate * tau;
            if i + 1 == fixed_leg.len() {
                amount += 1.0;
            }
            flows.push((t, -amount));
        }
        // each flow's affine pieces at expiry: A_i (the state-free part)
        // and the loadings on x and y
        let pieces: Vec<(f64, f64, f64, f64)> = flows
            .iter()
            .map(|&(t, w)| {
                let tau = t - expiry;
                let a_i = self.curve.df(t) / self.curve.df(expiry)
                    * (0.5 * (self.v(tau) - self.v(t) + self.v(expiry))).exp();
                (w * a_i, self.b_a(tau), self.b_b(tau), w)
            })
            .collect();
        let (mu_x, mu_y, sx, sy, rho_xy) = self.forward_measure_law(expiry);
        let s_cond = sy * (1.0 - rho_xy * rho_xy).max(0.0).sqrt();
        let omega = match payer_receiver {
            PayerReceiver::Payer => 1.0,
            PayerReceiver::Receiver => -1.0,
        };
        // g(x, y) = sum_i w_i A_i e^{-Ba_i x - Bb_i y}; g / P(e, start) is
        // increasing in y, so the payer exercises for y above the root
        let g = |x: f64, y: f64| -> f64 {
            pieces
                .iter()
                .map(|&(wa, ba, bb, _)| wa * (-ba * x - bb * y).exp())
                .sum()
        };
        let mu_cond_at = |x: f64| mu_y + rho_xy * sy * (x - mu_x) / sx;
        // Simpson over [lo, hi] of f(x) times the x-density
        let simpson_x = |lo: f64, hi: f64, f: &dyn Fn(f64) -> Result<f64, RustyQLibError>| {
            let dx = (hi - lo) / (X_NODES - 1) as f64;
            let mut total = 0.0;
            for k in 0..X_NODES {
                let x = lo + k as f64 * dx;
                let simpson = if k == 0 || k == X_NODES - 1 {
                    1.0
                } else if k % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                total += simpson * dx / 3.0 * norm_pdf((x - mu_x) / sx) / sx * f(x)?;
            }
            Ok::<f64, RustyQLibError>(total)
        };
        let (x_lo, x_hi) = (mu_x - X_SPAN * sx, mu_x + X_SPAN * sx);
        let total = if s_cond <= 1e-14 {
            // y is deterministic given x: the exercise kink sits in the
            // x-integral, so locate it and integrate the exercise side
            // only, keeping the integrand smooth on each piece
            let h = |x: f64| g(x, mu_cond_at(x));
            let (g_lo, g_hi) = (h(x_lo), h(x_hi));
            let x_star = if g_lo >= 0.0 {
                x_lo
            } else if g_hi <= 0.0 {
                x_hi
            } else {
                Solver1d::new(1e-14, 200).bisection(h, x_lo, x_hi)?.x
            };
            let payoff = |x: f64| Ok((omega * h(x)).max(0.0));
            if omega > 0.0 {
                simpson_x(x_star, x_hi, &payoff)?
            } else {
                simpson_x(x_lo, x_star, &payoff)?
            }
        } else {
            let inner = |x: f64| -> Result<f64, RustyQLibError> {
                let mu_cond = mu_cond_at(x);
                // the exercise boundary in y: bracket on a wide range
                let (lo, hi) = (mu_cond - 40.0 * sy.max(1e-6), mu_cond + 40.0 * sy.max(1e-6));
                let (g_lo, g_hi) = (g(x, lo), g(x, hi));
                let y_star = if g_lo >= 0.0 {
                    lo
                } else if g_hi <= 0.0 {
                    hi
                } else {
                    Solver1d::new(1e-14, 200).bisection(|y| g(x, y), lo, hi)?.x
                };
                // E[(omega g)^+ | x] = omega sum_i w_i A_i e^{-Ba_i x}
                //   E[e^{-Bb_i y} 1{omega (y - y*) > 0}]
                let mut sum = 0.0;
                for &(wa, ba, bb, _) in &pieces {
                    let mean_term = (-bb * mu_cond + 0.5 * bb * bb * s_cond * s_cond).exp();
                    let threshold = (mu_cond - bb * s_cond * s_cond - y_star) / s_cond;
                    let prob = if omega > 0.0 {
                        norm_cdf(threshold)
                    } else {
                        norm_cdf(-threshold)
                    };
                    sum += wa * (-ba * x).exp() * mean_term * prob;
                }
                Ok((omega * sum).max(0.0))
            };
            simpson_x(x_lo, x_hi, &inner)?
        };
        Ok(notional * self.curve.df(expiry) * total)
    }
}

/// The result of a G2++ calibration.
#[derive(Debug, Clone)]
pub struct G2ppFit {
    pub model: G2pp,
    /// Root-mean-square relative price error over the quotes.
    pub price_rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Sum of squared relative price errors of the quotes under the model.
fn g2pp_objective(
    curve: &YieldCurve,
    quotes: &[crate::rates::models::calibration::SwaptionQuote],
    params: (f64, f64, f64, f64, f64),
) -> f64 {
    let (a, b, sigma, eta, rho) = params;
    let Ok(model) = G2pp::new(a, b, sigma, eta, rho, curve.clone()) else {
        return 1e10;
    };
    let mut sum = 0.0;
    for q in quotes {
        match model.european_swaption(
            q.expiry,
            q.swap_start,
            &q.fixed_leg,
            q.strike_rate,
            1.0,
            q.payer_receiver,
        ) {
            Ok(price) => {
                let error = (price - q.market_price) / q.market_price;
                sum += error * error;
            }
            Err(_) => return 1e10,
        }
    }
    sum
}

fn finish_g2pp_fit(
    curve: &YieldCurve,
    quotes: &[crate::rates::models::calibration::SwaptionQuote],
    params: (f64, f64, f64, f64, f64),
    value: f64,
    iterations: usize,
    converged: bool,
) -> Result<G2ppFit, RustyQLibError> {
    if !converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations,
            residual: value,
            reason: "G2++ swaption calibration did not converge".to_string(),
        });
    }
    let (a, b, sigma, eta, rho) = params;
    Ok(G2ppFit {
        model: G2pp::new(a, b, sigma, eta, rho, curve.clone())?,
        price_rmse: (value / quotes.len() as f64).sqrt(),
        iterations,
        converged,
    })
}

/// Calibrate the two volatilities `(sigma, eta)` to swaption quotes
/// with the mean reversions and the correlation fixed — the usual
/// setup, since `a`, `b` and `rho` are weakly identified by a grid of
/// ATM prices and are chosen for the shape of the correlation
/// structure. Nelder-Mead over the log-vols.
pub fn calibrate_g2pp_vols(
    curve: &YieldCurve,
    quotes: &[crate::rates::models::calibration::SwaptionQuote],
    a: f64,
    b: f64,
    rho: f64,
    sigma0: f64,
    eta0: f64,
) -> Result<G2ppFit, RustyQLibError> {
    use crate::core::optimization::{minimize, Method, OptimConfig, Problem};
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(FIELD, "no swaption quotes"));
    }
    if !(sigma0 > 0.0 && eta0 > 0.0) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "starting vols must be positive",
        ));
    }
    let f = |x: &[f64]| g2pp_objective(curve, quotes, (a, b, x[0].exp(), x[1].exp(), rho));
    let problem = Problem::scalar(&f, vec![sigma0.ln(), eta0.ln()]);
    let result = minimize(&OptimConfig::new(1e-14, 600), Method::NelderMead, &problem)?;
    finish_g2pp_fit(
        curve,
        quotes,
        (a, b, result.x[0].exp(), result.x[1].exp(), rho),
        result.value,
        result.iterations,
        result.converged,
    )
}

/// Calibrate all five parameters from `start = (a, b, sigma, eta, rho)`:
/// Nelder-Mead over the log mean reversions and vols and `atanh(rho)`,
/// so every iterate is a valid model.
pub fn calibrate_g2pp(
    curve: &YieldCurve,
    quotes: &[crate::rates::models::calibration::SwaptionQuote],
    start: (f64, f64, f64, f64, f64),
) -> Result<G2ppFit, RustyQLibError> {
    use crate::core::optimization::{minimize, Method, OptimConfig, Problem};
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(FIELD, "no swaption quotes"));
    }
    let (a0, b0, s0, e0, r0) = start;
    if !(a0 > 0.0 && b0 > 0.0 && s0 > 0.0 && e0 > 0.0 && r0.abs() < 1.0) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "starting values need positive a, b, sigma, eta and |rho| < 1",
        ));
    }
    let f = |x: &[f64]| {
        g2pp_objective(
            curve,
            quotes,
            (x[0].exp(), x[1].exp(), x[2].exp(), x[3].exp(), x[4].tanh()),
        )
    };
    let problem = Problem::scalar(&f, vec![a0.ln(), b0.ln(), s0.ln(), e0.ln(), r0.atanh()]);
    let result = minimize(&OptimConfig::new(1e-14, 1500), Method::NelderMead, &problem)?;
    finish_g2pp_fit(
        curve,
        quotes,
        (
            result.x[0].exp(),
            result.x[1].exp(),
            result.x[2].exp(),
            result.x[3].exp(),
            result.x[4].tanh(),
        ),
        result.value,
        result.iterations,
        result.converged,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::engines::jamshidian::european_swaption_settled;
    use crate::rates::models::calibration::{atm_swap_rate, SwaptionQuote};

    #[test]
    fn vol_calibration_recovers_the_generating_model() {
        let curve = market_curve();
        let truth = G2pp::new(0.1, 0.5, 0.008, 0.012, -0.7, curve.clone()).unwrap();
        let quotes: Vec<SwaptionQuote> =
            [(1.0, 5usize), (2.0, 4usize), (3.0, 3usize), (5.0, 5usize)]
                .iter()
                .map(|&(expiry, tenor)| {
                    let fixed_leg: Vec<(f64, f64)> =
                        (1..=tenor).map(|i| (expiry + i as f64, 1.0)).collect();
                    let strike = atm_swap_rate(&curve, expiry, &fixed_leg).unwrap();
                    let price = truth
                        .european_swaption(
                            expiry,
                            expiry,
                            &fixed_leg,
                            strike,
                            1.0,
                            PayerReceiver::Payer,
                        )
                        .unwrap();
                    SwaptionQuote {
                        expiry,
                        swap_start: expiry,
                        fixed_leg,
                        strike_rate: strike,
                        market_price: price,
                        payer_receiver: PayerReceiver::Payer,
                    }
                })
                .collect();
        let fit = calibrate_g2pp_vols(&curve, &quotes, 0.1, 0.5, -0.7, 0.005, 0.02).unwrap();
        assert!(fit.price_rmse < 1e-6, "rmse {}", fit.price_rmse);
        assert!(
            (fit.model.sigma - 0.008).abs() < 1e-4,
            "sigma {}",
            fit.model.sigma
        );
        assert!(
            (fit.model.eta - 0.012).abs() < 1e-4,
            "eta {}",
            fit.model.eta
        );
        // the full fit from a nearby start reprices the grid too
        let full = calibrate_g2pp(&curve, &quotes, (0.12, 0.4, 0.007, 0.010, -0.6)).unwrap();
        assert!(full.price_rmse < 1e-4, "full rmse {}", full.price_rmse);
        assert!(calibrate_g2pp_vols(&curve, &[], 0.1, 0.5, 0.0, 0.01, 0.01).is_err());
    }
    use crate::rates::models::{HullWhite, OneFactorAffine, ShortRateModel};
    use chrono::NaiveDate;

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

    fn model() -> G2pp {
        G2pp::new(0.1, 0.5, 0.008, 0.012, -0.7, market_curve()).unwrap()
    }

    fn fixed_leg() -> Vec<(f64, f64)> {
        (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect()
    }

    #[test]
    fn fitted_model_reproduces_the_input_curve_exactly() {
        let m = model();
        for t in [0.25, 1.0, 3.3, 7.0, 12.0, 25.0] {
            let model_df = m.zero_bond(0.0, t, 0.0, 0.0).unwrap();
            let market_df = m.curve().df(t);
            assert!(
                (model_df / market_df - 1.0).abs() < 1e-12,
                "t={t}: {model_df} vs {market_df}"
            );
        }
        assert!((m.short_rate(0.0, 0.0, 0.0) - m.initial_short_rate()).abs() < 1e-12);
    }

    #[test]
    fn the_forward_measure_law_prices_the_forward_bond() {
        // P(0,S)/P(0,T) = E^T[P(T,S)]: with (x,y) Gaussian under the
        // T-forward measure the expectation is closed form, which pins
        // the measure-change means mu_x, mu_y
        let m = model();
        for (expiry, maturity) in [(1.0, 3.0), (2.0, 7.0), (5.0, 6.0)] {
            let (mu_x, mu_y, sx, sy, rho_xy) = m.forward_measure_law(expiry);
            let tau = maturity - expiry;
            let (ba, bb) = (m.b_a(tau), m.b_b(tau));
            let state_free = m.zero_bond(expiry, maturity, 0.0, 0.0).unwrap();
            let expectation = state_free
                * (-ba * mu_x - bb * mu_y
                    + 0.5
                        * (ba * ba * sx * sx
                            + bb * bb * sy * sy
                            + 2.0 * ba * bb * rho_xy * sx * sy))
                    .exp();
            let forward = m.curve().df(maturity) / m.curve().df(expiry);
            assert!(
                (expectation / forward - 1.0).abs() < 1e-10,
                "{expiry}->{maturity}: {expectation} vs {forward}"
            );
        }
    }

    #[test]
    fn eta_zero_is_hull_white_and_equal_factors_are_hull_white_root_two() {
        let curve = market_curve();
        let leg = fixed_leg();
        let cases = [
            (
                G2pp::new(0.1, 0.5, 0.01, 0.0, 0.0, curve.clone()).unwrap(),
                HullWhite::new(0.1, 0.01, curve.clone()).unwrap(),
            ),
            (
                G2pp::new(0.1, 0.1, 0.007, 0.007, 0.0, curve.clone()).unwrap(),
                HullWhite::new(0.1, 0.007 * 2.0_f64.sqrt(), curve.clone()).unwrap(),
            ),
        ];
        for (g2, hw) in cases {
            // bonds, bond options, exchange options and swaptions agree
            let r0 = hw.initial_short_rate();
            for t in [1.0, 4.0] {
                let p_g2 = g2.zero_bond(0.5, t + 0.5, 0.01, 0.0).unwrap();
                let p_hw = hw
                    .zero_bond(0.5, t + 0.5, r0 + 0.01 - hw.alpha(0.5) + hw.alpha(0.5))
                    .unwrap();
                // the x-loading matches Hull-White's B; compare shapes
                // through the ratio of two states instead of levels
                let p_g2_shift = g2.zero_bond(0.5, t + 0.5, 0.02, 0.0).unwrap();
                let p_hw_shift = hw
                    .zero_bond(0.5, t + 0.5, r0 + 0.02 - hw.alpha(0.5) + hw.alpha(0.5))
                    .unwrap();
                assert!(((p_g2 / p_g2_shift) - (p_hw / p_hw_shift)).abs() < 1e-12);
            }
            let zbo_g2 = g2.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Put).unwrap();
            let zbo_hw = hw.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Put).unwrap();
            assert!((zbo_g2 - zbo_hw).abs() < 1e-14, "{zbo_g2} vs {zbo_hw}");
            let ex_g2 = g2
                .zero_bond_exchange_option(1.0, 1.01, 5.0, 0.83, PutOrCall::Put)
                .unwrap();
            let ex_hw = hw
                .zero_bond_exchange_option(1.0, 1.01, 5.0, 0.83, PutOrCall::Put)
                .unwrap();
            assert!((ex_g2 - ex_hw).abs() < 1e-14);
            for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
                let s_g2 = g2
                    .european_swaption(1.0, 1.0, &leg, 0.045, 1.0, side)
                    .unwrap();
                let s_hw =
                    european_swaption_settled(&hw, 1.0, 1.0, &leg, 0.045, 1.0, side).unwrap();
                assert!(
                    (s_g2 - s_hw).abs() < 1e-7 * s_hw.max(1e-3),
                    "{side:?}: {s_g2} vs {s_hw}"
                );
                let s_g2 = g2
                    .european_swaption(1.0, 1.01, &leg, 0.045, 1.0, side)
                    .unwrap();
                let s_hw =
                    european_swaption_settled(&hw, 1.0, 1.01, &leg, 0.045, 1.0, side).unwrap();
                assert!(
                    (s_g2 - s_hw).abs() < 1e-7 * s_hw.max(1e-3),
                    "lagged {side:?}: {s_g2} vs {s_hw}"
                );
            }
        }
    }

    #[test]
    fn two_factor_swaptions_keep_parity_and_price_the_decorrelation() {
        let m = model();
        let leg = fixed_leg();
        let payer = m
            .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Payer)
            .unwrap();
        let receiver = m
            .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Receiver)
            .unwrap();
        let curve = m.curve();
        let fixed: f64 = leg
            .iter()
            .map(|&(t, tau)| 0.045 * tau * curve.df(t))
            .sum::<f64>()
            + curve.df(6.0);
        let forward_swap = 1_000_000.0 * (curve.df(1.0) - fixed);
        assert!(
            (payer - receiver - forward_swap).abs() < 1e-3,
            "{payer} - {receiver} vs {forward_swap}"
        );
        assert!(payer > 0.0 && receiver > 0.0);
        // bond-option parity against the market curve
        let call = m.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Call).unwrap();
        let put = m.zero_bond_option(1.0, 5.0, 0.83, PutOrCall::Put).unwrap();
        assert!((call - put - (curve.df(5.0) - 0.83 * curve.df(1.0))).abs() < 1e-15);
        // a negative factor correlation lowers long-rate volatility: the
        // 1y-into-5y swaption is cheaper than with rho = 0
        let uncorrelated = G2pp::new(0.1, 0.5, 0.008, 0.012, 0.0, market_curve()).unwrap();
        let payer_0 = uncorrelated
            .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Payer)
            .unwrap();
        assert!(payer < payer_0, "{payer} vs {payer_0}");
    }

    #[test]
    fn exact_simulation_recovers_the_curve() {
        use rand::{Rng, SeedableRng};
        let m = model();
        let (horizon, steps, paths) = (5.0_f64, 50usize, 40_000usize);
        let dt = horizon / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(17);
        let mut sum = 0.0;
        for _ in 0..paths {
            let (mut x, mut y) = (0.0, 0.0);
            let mut r = m.initial_short_rate();
            let mut integral = 0.0;
            for step in 0..steps {
                let z1: f64 = rng.sample(rand_distr::StandardNormal);
                let z2: f64 = rng.sample(rand_distr::StandardNormal);
                let (nx, ny) = m.evolve(x, y, dt, z1, z2).unwrap();
                let next = m.short_rate((step + 1) as f64 * dt, nx, ny);
                integral += 0.5 * (r + next) * dt;
                (x, y, r) = (nx, ny, next);
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
        let c = market_curve();
        assert!(G2pp::new(0.0, 0.5, 0.01, 0.01, 0.0, c.clone()).is_err());
        assert!(G2pp::new(0.1, 0.5, -0.01, 0.01, 0.0, c.clone()).is_err());
        assert!(G2pp::new(0.1, 0.5, 0.01, 0.01, 1.5, c.clone()).is_err());
        let m = model();
        assert!(m.zero_bond(2.0, 1.0, 0.0, 0.0).is_err());
        assert!(m.evolve(0.0, 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(m
            .european_swaption(1.0, 0.5, &fixed_leg(), 0.04, 1.0, PayerReceiver::Payer)
            .is_err());
        assert!(m.zero_bond_option(0.0, 5.0, 0.8, PutOrCall::Call).is_err());
    }
}
