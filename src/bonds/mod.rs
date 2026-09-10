//! Fixed-income instruments and discount-curve construction.
//!
//! This module replaces the legacy `rates` module. It holds:
//!
//! - the money-market instruments that pin the short end of a discount
//!   curve ([`Deposit`], [`Fra`]), the [`CurveInstrument`] abstraction
//!   they share, and a sequential bootstrapper ([`bootstrap_curve`])
//!   producing the library-wide [`YieldCurve`];
//! - bond instruments: [`FixedRateBond`] with US-Treasury street
//!   analytics (accrued interest, price/yield, duration, convexity,
//!   DV01, curve pricing) and the discount-quoted [`TreasuryBill`];
//! - quoted-instrument pillars ([`BillQuote`], [`BondQuote`]) that
//!   bootstrap a Treasury discount curve which exactly reprices its
//!   input quotes;
//! - Treasury bond futures ([`BondFuture`]): CME conversion factors,
//!   invoice prices, gross/net basis, implied repo and the
//!   cheapest-to-deliver;
//! - corporate bond analytics: 30/360 T+2 conventions
//!   ([`FixedRateBond::us_corporate`]), z-spread / spread DV01 / G-spread
//!   / asset-swap spread ([`spreads`]), yield-to-call/put/worst
//!   ([`CallOption`], [`PutOption`]), step-up coupons and sinking funds,
//!   floating rate notes with discount margins ([`FloatingRateNote`]),
//!   hazard-rate credit pricing and curve bootstrapping ([`credit`]),
//!   and the Hull-White option model for calls, puts and make-wholes
//!   ([`callable`]);
//! - convertible bonds ([`ConvertibleBond`]) and preferreds
//!   ([`ConvertiblePreferred`]), both [`ConvertibleInstrument`]s priced
//!   through [`ConvertiblePricing`]: Tsiveriotis-Fernandes and
//!   jump-to-default pricing on an equity tree or by finite differences
//!   ([`convertible::fd`], value plus delta and gamma from one solve, and
//!   vega, theta, parallel/key-rate and credit DV01s by bumping; flat,
//!   Dupire local or custom volatility), with
//!   soft calls, puts, contingent conversion, coupon and
//!   fundamental-change make-wholes, mandatory conversion,
//!   parity/premium analytics, and
//!   implied credit spreads, hazard rates and volatilities.
//!
//! Conventions follow the rest of the library: instruments carry their own
//! [`DayCountConvention`](crate::core::daycount::DayCountConvention) for
//! accrual, the curve carries its own for date-to-time conversion, and all
//! valuation dates are explicit inputs — nothing here reads the wall clock.

pub mod bills;
pub mod bootstrap;
pub mod build_contracts;
pub mod callable;
pub mod convertible;
pub mod credit;
pub mod deposit;
pub mod fixed_rate_bond;
pub mod fra;
pub mod frn;
pub mod futures;
pub mod preferred;
pub mod quotes;
pub mod schedule;
pub mod service;
pub mod spreads;

pub use bills::TreasuryBill;
pub use bootstrap::bootstrap_curve;
pub use callable::{BondOptionality, MakeWholeCall};
pub use convertible::{
    dejump_implied_vol, dejump_surface, CashDividend, ContingentConversion, ConvertibleBond,
    ConvertibleFdGreeks, ConvertibleFdGrid, ConvertibleFdValuation, ConvertibleInstrument,
    ConvertibleMarket, ConvertiblePricing, CouponMakeWhole, CreditModel, FdVolModel,
    FundamentalChangeMakeWhole, JumpToDefaultMarket, MandatoryConversion,
};
pub use credit::{bootstrap_credit_curve, CreditCurve};
pub use deposit::Deposit;
pub use fixed_rate_bond::{CallOption, Cashflow, FixedRateBond, PutOption};
pub use fra::Fra;
pub use frn::FloatingRateNote;
pub use futures::{conversion_factor, BondFuture, DeliverableBond, FactorRounding};
pub use preferred::ConvertiblePreferred;
pub use quotes::{BillQuote, BondQuote};
pub use schedule::CouponSchedule;
pub use spreads::{g_spread, interpolated_benchmark_yield};

// Frequency moved to `core::calendar` (shared with the swaps in
// [`crate::rates`]); re-exported here so existing paths keep working.
pub use crate::core::calendar::Frequency;

use chrono::NaiveDate;

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// An instrument that pins one pillar of a discount curve.
///
/// During a sequential bootstrap the instruments are processed in maturity
/// order; each one receives the curve built from the instruments before it
/// (`None` for the first) and returns the discount factor from
/// `reference_date` to its own maturity implied by its quote.
pub trait CurveInstrument {
    /// The pillar date this instrument determines.
    fn maturity_date(&self) -> NaiveDate;

    /// Discount factor from `reference_date` to
    /// [`maturity_date`](Self::maturity_date) implied by this instrument,
    /// given the shorter part of the curve built so far.
    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError>;
}

/// Discount factor from `reference_date` to `start`: 1.0 for a spot start,
/// read off the curve for a forward start. Shared by [`Deposit`] and
/// [`Fra`] during bootstrapping.
fn df_to_start(
    instrument: &str,
    start: NaiveDate,
    reference_date: NaiveDate,
    curve_so_far: Option<&YieldCurve>,
) -> Result<f64, RustyQLibError> {
    if start < reference_date {
        return Err(RustyQLibError::invalid_input(
            instrument,
            format!("start date {start} is before the curve reference date {reference_date}"),
        ));
    }
    if start == reference_date {
        return Ok(1.0);
    }
    match curve_so_far {
        Some(curve) => Ok(curve.df_date(start)),
        None => Err(RustyQLibError::invalid_input(
            instrument,
            format!(
                "forward start {start} needs the shorter part of the curve; \
                 the first bootstrap instrument must start on the reference date"
            ),
        )),
    }
}
