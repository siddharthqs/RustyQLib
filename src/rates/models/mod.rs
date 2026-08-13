//! Stochastic short-rate models — the cross-asset foundation.
//!
//! Design follows the classic QuantLib / OpenGamma split:
//!
//! - [`ShortRateModel`] is the **simulation contract**: an initial state,
//!   an exact (or best-available) one-step transition driven by a
//!   standard normal draw, and the zero-coupon bond reconstitution
//!   `P(t, T) = f(state)`. This is what a Monte Carlo engine needs —
//!   whether it is pricing an IR exotic or discounting a very long-dated
//!   equity payoff under stochastic rates (a Black-Scholes-Hull-White
//!   hybrid simulates the equity and the short rate together and
//!   reconstitutes bonds from the rate state at every node).
//! - [`OneFactorAffine`] adds the **analytic layer** of affine models:
//!   closed-form European options on zero-coupon bonds. Everything else
//!   in [`pricers`] is generic on top of it — coupon-bond options by
//!   Jamshidian's decomposition, European swaptions via the
//!   bond-option equivalence, caplets and floorlets via zero-bond puts
//!   and calls.
//!
//! Models:
//!
//! - [`Vasicek`] — `dr = a(b - r)dt + sigma dW`. The pedagogical
//!   Gaussian model with its own endogenous term structure.
//! - [`HullWhite`] — `dr = (theta(t) - a r)dt + sigma dW`, the extended
//!   Vasicek fitted **exactly** to an input [`YieldCurve`]
//!   (term-structure consistent: `P(0,T)` reproduces the curve's
//!   discount factors by construction).
//! - [`CoxIngersollRoss`] — `dr = a(b - r)dt + sigma sqrt(r) dW`,
//!   square-root dynamics keeping rates non-negative under the Feller
//!   condition.
//!
//! Time is measured in year fractions from the model's anchor (for
//! Hull-White, the curve's reference date and day count), which keeps
//! the API asset-class agnostic: date handling stays with the caller.
//!
//! [`YieldCurve`]: crate::core::curves::YieldCurve

pub mod calibration;
pub mod cir;
pub mod hull_white;
pub mod pricers;
pub mod vasicek;

pub use calibration::{
    atm_swap_rate, calibrate_hull_white, calibrate_hull_white_sigma, HullWhiteFit, SwaptionQuote,
};
pub use cir::CoxIngersollRoss;
pub use hull_white::HullWhite;
pub use pricers::{caplet, coupon_bond_option, european_swaption, floorlet};
pub use vasicek::Vasicek;

use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;

/// A one-factor short-rate model as a simulation engine sees it.
pub trait ShortRateModel {
    /// The short rate at the anchor time `t = 0`.
    fn initial_short_rate(&self) -> f64;

    /// Zero-coupon bond price `P(t, maturity)` given the short rate at
    /// `t` — the reconstitution formula that turns a simulated rate
    /// state into a full discount curve at that node.
    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError>;

    /// One transition step `r(t) -> r(t + dt)` driven by a standard
    /// normal draw `z`. Exact for the Gaussian models; a full-truncation
    /// Euler step for CIR.
    fn evolve(&self, t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError>;
}

/// The analytic layer of affine one-factor models: closed-form European
/// options on zero-coupon bonds, valued at the anchor time `t = 0`.
/// [`pricers`] builds coupon-bond options, swaptions and caps from this
/// single primitive.
pub trait OneFactorAffine: ShortRateModel {
    /// Value today of a European option, expiring at `expiry`, on the
    /// zero-coupon bond maturing at `bond_maturity`, struck at `strike`
    /// (a price per unit face).
    fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError>;
}

/// Shared validation for `(expiry, bond_maturity, strike)` option terms.
pub(crate) fn validate_bond_option_terms(
    expiry: f64,
    bond_maturity: f64,
    strike: f64,
) -> Result<(), RustyQLibError> {
    if !(expiry > 0.0 && expiry.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("expiry must be positive, got {expiry}"),
        ));
    }
    if !(bond_maturity > expiry && bond_maturity.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("bond maturity {bond_maturity} must exceed the expiry {expiry}"),
        ));
    }
    if !(strike > 0.0 && strike.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("strike must be positive, got {strike}"),
        ));
    }
    Ok(())
}

/// The Gaussian zero-bond option formula shared by Vasicek and
/// Hull-White (Jamshidian 1989): a Black-style exchange option between
/// the `bond_maturity` bond and `strike` units of the `expiry` bond,
/// with price volatility `sigma_p`.
pub(crate) fn gaussian_zero_bond_option(
    p_expiry: f64,
    p_bond: f64,
    strike: f64,
    sigma_p: f64,
    put_or_call: PutOrCall,
) -> f64 {
    use crate::core::utils::norm_cdf;
    if sigma_p <= 0.0 {
        // deterministic limit: discounted intrinsic
        let forward_intrinsic = match put_or_call {
            PutOrCall::Call => (p_bond - strike * p_expiry).max(0.0),
            PutOrCall::Put => (strike * p_expiry - p_bond).max(0.0),
        };
        return forward_intrinsic;
    }
    let h = (p_bond / (p_expiry * strike)).ln() / sigma_p + 0.5 * sigma_p;
    match put_or_call {
        PutOrCall::Call => p_bond * norm_cdf(h) - strike * p_expiry * norm_cdf(h - sigma_p),
        PutOrCall::Put => strike * p_expiry * norm_cdf(sigma_p - h) - p_bond * norm_cdf(-h),
    }
}
