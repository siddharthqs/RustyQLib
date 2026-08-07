//! Linear interest-rate products: swaps.
//!
//! - [`VanillaSwap`] — fixed-for-floating interest rate swap (IRS)
//! - [`OvernightIndexSwap`] — fixed versus daily-compounded overnight
//!   (SOFR-style OIS), with an optional payment lag
//! - [`BasisSwap`] — floating-for-floating with a spread on one leg
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
pub mod leg;
pub mod ois;
pub mod vanilla_swap;

pub use basis_swap::{BasisSwap, BasisSwapLeg};
pub use leg::AccrualPeriod;
pub use ois::OvernightIndexSwap;
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
