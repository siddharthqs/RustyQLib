//! Commodity derivatives.
//!
//! - [`CommoditySwap`] — fixed-for-floating commodity swap: each
//!   calculation period cash-settles the arithmetic average of the
//!   daily index price over the period's business days against a fixed
//!   price, times the period's notional quantity
//! - [`CommodityBasisSwap`] — floating-for-floating differential
//!   between two indexes (WTI–Brent, location or quality basis), each
//!   leg averaging over its own pricing calendar, with a fixed spread
//!   on the received leg
//! - [`CommodityOption`] — European option on a commodity future,
//!   priced with Black-76 (premium up front or futures-style margined)
//! - [`AveragePriceOption`] — Asian-style APO on the same averaged
//!   index as the swap, priced by discrete moment matching (Levy) with
//!   realized fixings folded into an adjusted strike
//! - [`CommoditySwaption`] — European option to enter a
//!   [`CommoditySwap`]: Black-on-par times the settlement annuity,
//!   exact under the one-factor flat-vol dynamics
//! - [`CommoditySpreadOption`] — European option on the spread of two
//!   futures (crack, spark, location, calendar), Kirk's approximation
//!   for lognormal legs and exact Bachelier for normal legs
//! - [`CommodityForwardCurve`] — the strip of forward prices both
//!   products are projected from, linearly interpolated between pillar
//!   dates
//!
//! Discounting comes from a [`crate::core::curves::YieldCurve`];
//! partially realized swap periods blend published [`PriceFixings`]
//! with curve forwards, mirroring the money-market futures in
//! [`crate::rates`].
//!
//! # Conventions
//!
//! **Valuation and settled periods.** The valuation date is the
//! discount curve's reference date. The no-fixings entry points
//! (`pv`, `par_price`, `delta`, ...) value as of that date, so a
//! seasoned swap — one whose pricing days have started — must be
//! valued through the `*_with_fixings` variants, whose `asof` may not
//! precede the reference date. A calculation period whose payment
//! falls **on** the valuation date is settled and contributes nothing
//! (the [`crate::rates`] convention): only periods paying strictly
//! after `asof` remain in a PV or annuity.
//!
//! **Vol time.** Option expiries, averaging observations and swaption
//! exercise dates are converted to the year fractions the volatility
//! multiplies on **Act/365F from the valuation date**
//! ([`crate::core::daycount::DayCountConvention::Act365`]), whatever
//! day count the discount curve quotes on — an Act/360 curve must not
//! inflate variance by 365/360. Discount factors still come from the
//! curve at the exact date, and the rate handed to the Black-76 and
//! Bachelier kernels is the continuous rate `r = -ln(df) / t` on that
//! same Act/365 time, so `exp(-r t)` reproduces the curve's discount
//! factor. The smile and term-structure generators ([`ShiftedSabr`],
//! [`ClewlowStrickland`]) resolve their `quote_for` times on the same
//! basis, so quotes and pricing agree.
//!
//! Options take their distribution model from the vol quote
//! ([`CommodityVol`]): Black-76 lognormal (the default — a bare `f64`
//! vol), shifted lognormal, or Bachelier normal ([`bachelier`]) for
//! underlyings that can print negative (Waha/AECO basis, WTI in an
//! April-2020 dislocation). A [`ShiftedSabr`] smile generates the
//! shifted-lognormal quote per strike, so whole smiles price
//! consistently through the same dispatch; a [`ClewlowStrickland`]
//! model does the same across maturities (the Samuelson effect), and
//! additionally supplies the cross-maturity covariances the APO and
//! swaption use in their `price_cs` variants.

pub mod apo;
pub mod bachelier;
pub mod basis_swap;
pub mod clewlow_strickland;
pub mod forward_curve;
pub mod option;
pub mod sabr;
pub mod spread_option;
pub mod swap;
pub mod swaption;
pub mod vol;

pub use apo::AveragePriceOption;
pub use basis_swap::CommodityBasisSwap;
pub use clewlow_strickland::{ClewlowStrickland, ClewlowStricklandFit};
pub use forward_curve::CommodityForwardCurve;
pub use option::{CommodityOption, FuturesSettlement};
pub use sabr::{ShiftedSabr, ShiftedSabrFit};
pub use spread_option::CommoditySpreadOption;
pub use swap::{CommoditySwap, PriceFixings};
pub use swaption::CommoditySwaption;
pub use vol::CommodityVol;

use chrono::NaiveDate;

use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

/// The year fraction volatility multiplies: Act/365F from `from` to
/// `to`, independent of the discount curve's day count (see the module
/// docs).
pub(crate) fn vol_time(from: NaiveDate, to: NaiveDate) -> f64 {
    DayCountConvention::Act365.year_fraction(from, to)
}

/// The time and discount inputs of one European expiry, resolved on
/// one basis for every option in this module.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpiryInputs {
    /// Vol time to expiry: Act/365F from the valuation date.
    pub t: f64,
    /// The discount curve's discount factor to the expiry date.
    pub df: f64,
    /// Continuous rate on `t` reproducing `df`: `-ln(df) / t`, zero at
    /// `t = 0`. Hand this to the kernels so `exp(-r t) == df`.
    pub r: f64,
}

/// Resolve `(t, df, r)` for an option expiring on `expiry`, rejecting
/// an expiry before the discount curve's reference date (the valuation
/// date). `field` names the instrument in the error.
pub(crate) fn expiry_inputs(
    field: &str,
    expiry: NaiveDate,
    discount: &YieldCurve,
) -> Result<ExpiryInputs, RustyQLibError> {
    let valuation = discount.reference_date();
    if expiry < valuation {
        return Err(RustyQLibError::invalid_input(
            field,
            format!("expired {expiry} (valuing {valuation})"),
        ));
    }
    let t = vol_time(valuation, expiry);
    let df = discount.df_date(expiry);
    let r = if t > 0.0 { -df.ln() / t } else { 0.0 };
    Ok(ExpiryInputs { t, df, r })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn vol_time_is_act365_whatever_the_curve_quotes_on() {
        let reference = d(2026, 9, 1);
        let expiry = d(2027, 9, 1);
        for dc in [DayCountConvention::Act365, DayCountConvention::Act360] {
            let curve = YieldCurve::flat(0.04, reference, dc, Compounding::Continuous).unwrap();
            let inputs = expiry_inputs("x", expiry, &curve).unwrap();
            // one Act/365 year, never 365/360 of one
            assert_eq!(inputs.t, 1.0, "{dc:?}");
            // the discount factor is the curve's at the exact date
            assert_eq!(inputs.df, curve.df_date(expiry));
            // and the rate reproduces it on the vol time
            assert!(((-inputs.r * inputs.t).exp() - inputs.df).abs() < 1e-15);
        }
        // an Act/360 curve discounts one calendar year at 365/360 of the rate
        let act360 = YieldCurve::flat(
            0.04,
            reference,
            DayCountConvention::Act360,
            Compounding::Continuous,
        )
        .unwrap();
        let r = expiry_inputs("x", expiry, &act360).unwrap().r;
        assert!((r - 0.04 * 365.0 / 360.0).abs() < 1e-12, "{r}");
    }

    #[test]
    fn expiry_on_the_valuation_date_has_zero_time_and_unit_discount() {
        let reference = d(2026, 9, 1);
        let curve = YieldCurve::flat(
            0.04,
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let inputs = expiry_inputs("x", reference, &curve).unwrap();
        assert_eq!((inputs.t, inputs.df, inputs.r), (0.0, 1.0, 0.0));
        assert!(expiry_inputs("x", d(2026, 8, 31), &curve).is_err());
    }
}
