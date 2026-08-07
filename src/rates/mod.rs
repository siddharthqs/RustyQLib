//! Linear interest-rate products: swaps and money-market futures.
//!
//! - [`VanillaSwap`] — fixed-for-floating interest rate swap (IRS)
//! - [`OvernightIndexSwap`] — fixed versus daily-compounded overnight
//!   (SOFR-style OIS), with an optional payment lag
//! - [`BasisSwap`] — floating-for-floating with a spread on one leg
//! - [`FedFundsFuture`] — CME 30-day fed funds futures (ZQ): the
//!   arithmetic average of daily EFFR over the contract month, with
//!   realized-fixings blending and FOMC step analytics
//! - [`SofrFuture`] — CME SOFR futures: 1-month (SR1, arithmetic average
//!   over the calendar month) and 3-month (SR3, daily compounding over
//!   an IMM quarter), with a convexity helper for the futures/forward
//!   bias
//!
//! All pricing is linear discounting off [`YieldCurve`]s: each product
//! takes an explicit **discount** curve and one **forecast** curve per
//! floating leg, so single-curve (pass the same curve) and dual-curve
//! setups both work. With no fixings modelled, a floating accrual over
//! `[s, e]` is forecast as the curve ratio `df(s)/df(e) - 1` — the
//! simple forward for an IBOR-style leg and the compounded overnight
//! rate for an OIS leg alike.
//!
//! Swap accrual dates are business-day adjusted (unlike bond accrual,
//! which runs on scheduled dates); schedules roll backward from
//! maturity, so a stub lands at the front.

pub mod basis_swap;
pub mod fed_funds_future;
pub mod leg;
pub mod ois;
pub mod overnight;
pub mod sofr_future;
pub mod vanilla_swap;

pub use basis_swap::{BasisSwap, BasisSwapLeg};
pub use fed_funds_future::FedFundsFuture;
pub use leg::AccrualPeriod;
pub use ois::OvernightIndexSwap;
pub use overnight::{overnight_forward, simple_forward, RateFixings};
pub use sofr_future::{hull_convexity_adjustment, SofrContract, SofrFuture};
pub use vanilla_swap::VanillaSwap;

use crate::core::curves::YieldCurve;

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
