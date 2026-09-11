//! Interest rates: contracts, pricing engines, short-rate models, and
//! the shared leg machinery that binds them — the same layout as
//! [`equity`](crate::equity).
//!
//! - [`contracts`] — what you can trade: swaps ([`VanillaSwap`],
//!   [`OvernightIndexSwap`], [`BasisSwap`]), money-market futures
//!   ([`FedFundsFuture`], [`SofrFuture`]), and the options on rates —
//!   [`Swaption`] and [`CapFloor`]
//! - [`engines`] — how the options get priced: Jamshidian's analytic
//!   engine on one-factor affine models
//! - [`models`] — stochastic short-rate dynamics (Vasicek, Hull-White,
//!   CIR) with exact simulation transitions, plus Hull-White
//!   calibration to swaptions
//! - top level — the spine: [`schedule`] (stub and roll conventions),
//!   [`leg`] (leg PVs, reset compounding), [`overnight`] (fixings,
//!   overnight forwards and compounded-in-arrears conventions),
//!   [`PayerReceiver`]
//!
//! Linear pricing is discounting off [`YieldCurve`]s: each product
//! takes an explicit **discount** curve and one **forecast** curve per
//! floating leg, so single-curve (pass the same curve) and dual-curve
//! setups both work. With no fixings modelled, a floating accrual over
//! `[s, e]` is forecast as the curve ratio `df(s)/df(e) - 1` — the
//! simple forward for an IBOR-style leg and the compounded overnight
//! rate for an OIS leg alike.
//!
//! Swap accrual dates are business-day adjusted (unlike bond accrual,
//! which runs on scheduled dates). By default schedules roll backward
//! from maturity, so a stub lands at the front; [`StubConvention`] and
//! [`RollConvention`] select the other market conventions.
//!
//! Every submodule is also re-exported at this level, so the historical
//! flat paths (`rates::vanilla_swap`, `rates::models::pricers`, …) keep
//! working unchanged.

pub mod leg;
pub mod overnight;
pub mod schedule;

pub mod contracts;
pub mod engines;
pub mod models;
pub mod multicurve;

// Flat-path compatibility re-exports.
pub use contracts::{
    basis_swap, cap_floor, fed_funds_future, ois, sofr_future, swaption, vanilla_swap,
};
pub use engines::RateVol;
pub use engines::{black, jamshidian};

pub use contracts::{
    hull_convexity_adjustment, BasisSwap, BasisSwapLeg, CapFloor, CapOrFloor, CapletValue,
    FedFundsFuture, OvernightIndexSwap, SofrContract, SofrFuture, Swaption, VanillaSwap,
};
pub use leg::{AccrualPeriod, CompoundingMethod, FloatPeriod};
pub use models::{CoxIngersollRoss, HullWhite, OneFactorAffine, ShortRateModel, Vasicek};
pub use multicurve::{MultiCurve, MultiCurveBuilder, Pillar, QuoteSensitivity, RateInstrument};
pub use overnight::{overnight_forward, simple_forward, OvernightConvention, RateFixings};
pub use schedule::{LegSchedule, RollConvention, StubConvention};

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// Which side of the fixed leg the position is on. `Payer` pays fixed
/// and receives floating; `Receiver` the reverse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayerReceiver {
    Payer,
    Receiver,
}

impl PayerReceiver {
    /// Sign applied to `float - fixed`: +1 for a payer, -1 for a receiver.
    pub(crate) fn sign(self) -> f64 {
        match self {
            PayerReceiver::Payer => 1.0,
            PayerReceiver::Receiver => -1.0,
        }
    }
}

/// Reused across swap types: validate the terms every swap constructor
/// shares. `instrument` names the swap type in errors and `rate_label`
/// its rate-like input ("fixed rate" or "spread"), so each
/// constructor's messages read exactly as before.
pub(crate) fn validate_swap_terms(
    instrument: &str,
    notional: f64,
    rate: f64,
    rate_label: &str,
    effective_date: chrono::NaiveDate,
    maturity_date: chrono::NaiveDate,
) -> Result<(), RustyQLibError> {
    if !notional.is_finite() || notional <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            instrument,
            format!("notional must be positive, got {notional}"),
        ));
    }
    if !rate.is_finite() {
        return Err(RustyQLibError::invalid_input(
            instrument,
            format!("{rate_label} must be finite, got {rate}"),
        ));
    }
    if maturity_date <= effective_date {
        return Err(RustyQLibError::invalid_input(
            instrument,
            format!("maturity {maturity_date} must be after effective {effective_date}"),
        ));
    }
    Ok(())
}

/// Reused across swap types: reject a non-positive discount factor from
/// a forecast curve before dividing by it.
pub(crate) fn checked_df(
    curve: &YieldCurve,
    date: chrono::NaiveDate,
) -> Result<f64, crate::core::errors::RustyQLibError> {
    let df = curve.df_date(date);
    if !df.is_finite() || df <= 0.0 {
        return Err(crate::core::errors::RustyQLibError::NumericalError(
            format!("non-positive discount factor {df} at {date}"),
        ));
    }
    Ok(df)
}
