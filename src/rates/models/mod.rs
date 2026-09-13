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
//!   closed-form European options on zero-coupon bonds. Everything in
//!   [`jamshidian`](crate::rates::engines::jamshidian) is generic on
//!   top of it — coupon-bond options by Jamshidian's decomposition,
//!   European swaptions via the bond-option equivalence, caplets and
//!   floorlets via zero-bond puts and calls — and the date-aware
//!   [`Swaption`](crate::rates::contracts::swaption::Swaption) and
//!   [`CapFloor`](crate::rates::contracts::cap_floor::CapFloor)
//!   products sit on that.
//!
//! Models:
//!
//! - [`Vasicek`] — `dr = a(b - r)dt + sigma dW`. The pedagogical
//!   Gaussian model with its own endogenous term structure.
//! - [`HullWhite`] — `dr = (theta(t) - a(t) r)dt + sigma(t) dW`, the
//!   extended Vasicek fitted **exactly** to an input [`YieldCurve`]
//!   (term-structure consistent: `P(0,T)` reproduces the curve's
//!   discount factors by construction), with constant or piecewise
//!   coefficients — QuantLib's Hull-White, GSR ([`Gsr`]) and
//!   GeneralizedHullWhite ([`GeneralizedHullWhite`]) in one type.
//! - [`CoxIngersollRoss`] — `dr = a(b - r)dt + sigma sqrt(r) dW`,
//!   square-root dynamics keeping rates non-negative under the Feller
//!   condition, with noncentral chi-square bond options; and
//!   [`ExtendedCir`] (CIR++), the same fitted to the curve by a shift.
//! - [`BlackKarasinski`] — the lognormal short rate on a curve-fitted
//!   trinomial tree.
//! - [`G2pp`] — the two-additive-factor Gaussian model, its own
//!   two-state API (bonds, options, an integral swaption formula and an
//!   exact bivariate transition).
//! - [`MarkovFunctional`] — the Hunt-Kennedy-Pelsser model with a
//!   terminal-bond numeraire, calibrated backward to a coterminal
//!   column so every calibrating swaption reprices at every strike.
//! - [`Gaussian1dModel`] — the framework the last one and Hull-White
//!   share: a Gaussian Markov driver under the numeraire measure, with
//!   the [`gaussian1d`](crate::rates::engines::gaussian1d) engines
//!   pricing Europeans and Bermudans on any implementation.
//!
//! Time is measured in year fractions from the model's anchor (for
//! Hull-White, the curve's reference date and day count), which keeps
//! the API asset-class agnostic: date handling stays with the caller.
//!
//! [`YieldCurve`]: crate::core::curves::YieldCurve

pub mod black_karasinski;
pub mod calibration;
pub mod caplet_vol;
pub mod cir;
pub mod g2pp;
pub mod gaussian1d;
pub mod hull_white;
pub mod markov_functional;
pub mod sabr;
pub mod vasicek;
pub mod vol_surface;
pub mod zabr;

/// Flat-path compatibility: the Jamshidian pricers used to live at
/// `rates::models::pricers`; they are now
/// [`rates::engines::jamshidian`](crate::rates::engines::jamshidian).
pub use crate::rates::engines::jamshidian as pricers;

pub use black_karasinski::{BlackKarasinski, TailSwap};
pub use calibration::{
    atm_swap_rate, calibrate_hull_white, calibrate_hull_white_piecewise,
    calibrate_hull_white_sigma, HullWhiteFit, SwaptionQuote,
};
pub use caplet_vol::{strip_caplet_vols, CapQuote, CapletVolCurve};
pub use cir::{CoxIngersollRoss, ExtendedCir};
pub use g2pp::{calibrate_g2pp, calibrate_g2pp_vols, G2pp, G2ppFit};
pub use gaussian1d::{Gaussian1dModel, HullWhite1d};
pub use hull_white::HullWhite;
pub use markov_functional::MarkovFunctional;
pub use pricers::{caplet, coupon_bond_option, european_swaption, floorlet};
pub use sabr::{RateSabr, RateSabrFit, SabrSwaptionCube};
pub use vasicek::Vasicek;
pub use vol_surface::SwaptionVolSurface;
pub use zabr::{ZabrConfig, ZabrFit, ZabrParams, ZabrSmile};

/// QuantLib's name for Hull-White with a piecewise-constant sigma —
/// [`HullWhite::with_piecewise_sigma`]; the same type.
pub type Gsr = HullWhite;
/// QuantLib's name for Hull-White with piecewise-constant mean
/// reversion and sigma — [`HullWhite::generalized`]; the same type.
pub type GeneralizedHullWhite = HullWhite;

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

    /// The lowest short rate the model admits at `t` — unbounded for
    /// the Gaussian models, zero for square-root dynamics (and the
    /// shift for CIR++). Root searches over the rate stay above it.
    fn short_rate_floor(&self, _t: f64) -> f64 {
        f64::NEG_INFINITY
    }
}

/// The analytic layer of affine one-factor models: closed-form European
/// options on zero-coupon bonds, valued at the anchor time `t = 0`.
/// [`jamshidian`](crate::rates::engines::jamshidian) builds coupon-bond
/// options, swaptions and caps from this single primitive.
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

    /// Value today of a European option, expiring at `expiry`, to
    /// exchange `strike` units of the zero-coupon bond maturing at
    /// `settlement` for the bond maturing at `bond_maturity`: a call
    /// pays `(P(expiry, bond_maturity) - strike * P(expiry, settlement))^+`
    /// (reversed for a put). With `settlement == expiry` this is
    /// [`zero_bond_option`](Self::zero_bond_option); the general case
    /// is what a swaption needs when the swap starts a settlement lag
    /// after exercise. The default falls back to the plain option when
    /// the two dates coincide and refuses otherwise; the Gaussian
    /// models supply the closed form.
    fn zero_bond_exchange_option(
        &self,
        expiry: f64,
        settlement: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        if settlement == expiry {
            return self.zero_bond_option(expiry, bond_maturity, strike, put_or_call);
        }
        Err(RustyQLibError::invalid_input(
            "bond exchange option",
            "this model has no exchange-option formula for a settlement after the expiry",
        ))
    }
}

/// Shared validation for `(expiry, settlement, bond_maturity, strike)`
/// exchange-option terms: `0 < expiry <= settlement < bond_maturity`.
pub(crate) fn validate_bond_option_terms(
    expiry: f64,
    settlement: f64,
    bond_maturity: f64,
    strike: f64,
) -> Result<(), RustyQLibError> {
    if !(expiry > 0.0 && expiry.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("expiry must be positive, got {expiry}"),
        ));
    }
    if !(settlement >= expiry && settlement.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("settlement {settlement} must not precede the expiry {expiry}"),
        ));
    }
    if !(bond_maturity > settlement && bond_maturity.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "bond option",
            format!("bond maturity {bond_maturity} must exceed the settlement {settlement}"),
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

/// `B(t,T) = (1 - e^{-a(T-t)}) / a` — the affine loading of the bond
/// price on the short rate, shared by the mean-reverting Gaussian
/// models (Vasicek and Hull-White).
pub(crate) fn b_factor(a: f64, t: f64, maturity: f64) -> f64 {
    (1.0 - (-a * (maturity - t)).exp()) / a
}

/// Conditional standard deviation of `r(t + dt)` given `r(t)` for a
/// mean-reverting Gaussian short rate — identical for Vasicek and
/// Hull-White, since `theta(t)` shifts only the mean.
pub(crate) fn gaussian_short_rate_std(a: f64, sigma: f64, dt: f64) -> f64 {
    (sigma * sigma * (1.0 - (-2.0 * a * dt).exp()) / (2.0 * a)).sqrt()
}

/// Volatility at `expiry` of the price ratio
/// `P(expiry, bond_maturity) / P(expiry, settlement)` — the `sigma_p`
/// in [`gaussian_zero_bond_option`]. With `settlement == expiry` it is
/// the plain bond price volatility `std(r) * B(expiry, bond_maturity)`.
pub(crate) fn gaussian_bond_price_vol(
    a: f64,
    sigma: f64,
    expiry: f64,
    settlement: f64,
    bond_maturity: f64,
) -> f64 {
    gaussian_short_rate_std(a, sigma, expiry)
        * (b_factor(a, expiry, bond_maturity) - b_factor(a, expiry, settlement))
}

/// The Gaussian zero-bond (exchange) option formula shared by Vasicek
/// and Hull-White (Jamshidian 1989): a Black-style exchange option
/// between the `bond_maturity` bond and `strike` units of the
/// settlement bond (`p_settlement`, the `expiry` bond itself for a
/// plain option), with price-ratio volatility `sigma_p`.
pub(crate) fn gaussian_zero_bond_option(
    p_settlement: f64,
    p_bond: f64,
    strike: f64,
    sigma_p: f64,
    put_or_call: PutOrCall,
) -> f64 {
    use crate::core::utils::norm_cdf;
    if sigma_p <= 0.0 {
        // deterministic limit: discounted intrinsic
        let forward_intrinsic = match put_or_call {
            PutOrCall::Call => (p_bond - strike * p_settlement).max(0.0),
            PutOrCall::Put => (strike * p_settlement - p_bond).max(0.0),
        };
        return forward_intrinsic;
    }
    let h = (p_bond / (p_settlement * strike)).ln() / sigma_p + 0.5 * sigma_p;
    match put_or_call {
        PutOrCall::Call => p_bond * norm_cdf(h) - strike * p_settlement * norm_cdf(h - sigma_p),
        PutOrCall::Put => strike * p_settlement * norm_cdf(sigma_p - h) - p_bond * norm_cdf(-h),
    }
}
