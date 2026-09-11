//! The instrument abstraction the engines price: what a convertible
//! *is* (its conversion terms, its schedule of events, its floor),
//! separated from the credit model ([`CreditModel`]) and the numerical
//! engine ([`tree`](super::tree), [`fd`](super::fd)).
//!
//! [`ConvertibleBond`](super::ConvertibleBond) and
//! [`ConvertiblePreferred`](crate::hybrid::ConvertiblePreferred) implement
//! it, each mapping its own conventions (discrete versus continuous
//! calls, day-count accrued versus dividend-cycle accrued, a maturity
//! versus a perpetuity tail) onto the shared [`EventGrid`]. The whole
//! pricing API then comes for free through the blanket
//! [`ConvertiblePricing`](super::ConvertiblePricing) trait.

use chrono::NaiveDate;

use super::events::EventGrid;
use super::models::CreditModel;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// A discrete cash dividend on the underlying share.
///
/// On the ex-date the share drops by the amount and the holder's claim
/// jumps with it: `V(S) = V(S - D)`, applied by interpolation on the
/// engine's own spot ladder (Vellekoop-Nieuwenhuis), which keeps the
/// tree recombining and is exact on the grid. Cash dividends come on
/// top of the market's continuous yield; use one or the other for the
/// same payments, not both. A dividend larger than the share price
/// takes the price to the ladder's floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CashDividend {
    pub ex_date: NaiveDate,
    /// Per share.
    pub amount: f64,
}

pub(crate) fn validate_cash_dividends(dividends: &[CashDividend]) -> Result<(), RustyQLibError> {
    if dividends
        .iter()
        .any(|d| !(d.amount.is_finite() && d.amount >= 0.0))
    {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            "cash dividends must be non-negative",
        ));
    }
    Ok(())
}

/// A convertible instrument: the contract terms the engines need and
/// the mapping of its schedule onto the event grid.
///
/// Prices are in the instrument's own units — per 100 face for a bond,
/// per share for a preferred — which the event grid's `outstanding`
/// scale sets.
pub trait ConvertibleInstrument {
    /// Shares received per unit on conversion (the minimum ratio for a
    /// mandatory).
    fn conversion_ratio(&self) -> f64;

    /// Shares delivered at maturity at a terminal share price: the
    /// mandatory schedule, or the plain ratio.
    fn maturity_shares(&self, spot: f64) -> f64;

    /// Whether conversion at maturity is mandatory (no cash principal).
    fn is_mandatory(&self) -> bool;

    /// The issuer's calls are exercisable only at or above this share
    /// price, if set.
    fn soft_call_trigger(&self) -> Option<f64>;

    /// The share price at which conversion breaks even against the
    /// face or preference.
    fn conversion_price(&self) -> f64;

    /// Accrued interest or dividend at `settlement`, in price units.
    fn accrued(&self, settlement: NaiveDate) -> Result<f64, RustyQLibError>;

    /// The last payment date of the schedule after `settlement` (for a
    /// perpetual, of its truncated schedule).
    fn final_payment_date(&self, settlement: NaiveDate) -> Result<NaiveDate, RustyQLibError>;

    /// Validates the contract terms.
    fn validate(&self) -> Result<(), RustyQLibError>;

    /// The event grid for `steps` time steps from `settlement` to the
    /// final payment. `credit_rate` is the rate over the risk-free
    /// forward at which at-stake coupons discount to their step edge
    /// (and, for a perpetual, which prices the tail).
    fn event_grid(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
        credit_rate: f64,
    ) -> Result<EventGrid, RustyQLibError>;

    /// The straight floor per unit under a credit model: the schedule
    /// ignoring the conversion right.
    fn floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError>;
}
