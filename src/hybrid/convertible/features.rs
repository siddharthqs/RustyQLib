//! The contractual extras of a convertible bond: contingent
//! conversion, the coupon make-whole on calls, the fundamental-change
//! make-whole table, and mandatory conversion. Each is an optional
//! field on [`ConvertibleBond`](super::ConvertibleBond); the pricing
//! engines read them through the shared event grid.

use chrono::NaiveDate;

use crate::core::errors::RustyQLibError;

/// Contingent conversion ("CoCo"): the holder may convert only while
/// the share price is at or above `trigger`, on dates before `until`;
/// from `until` on, conversion is unconditional inside the window, and
/// the terminal conversion at the final payment is always allowed.
///
/// The contractual test is usually a closing price above the trigger
/// on 20 of the last 30 trading days of a quarter; the standard
/// single-factor simplification checks the current spot only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContingentConversion {
    /// Share-price trigger, e.g. 130% of the conversion price.
    pub trigger: f64,
    /// First date on which conversion is unconditional.
    pub until: NaiveDate,
}

/// Coupon make-whole on an issuer call: on any call dated before
/// `until` the holder also receives the present value, at the call
/// date, of the coupons scheduled after the call date up to and
/// including `until`, discounted on the curve plus `spread`. It is
/// paid whether the holder takes the cash or answers the call by
/// converting — that is the point of the provision on a provisional
/// call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouponMakeWhole {
    /// Last coupon date made whole, typically the first hard-call date.
    pub until: NaiveDate,
    /// Discounting spread over the curve, e.g. `0.005` for T+50.
    pub spread: f64,
}

/// Fundamental-change make-whole: on a takeover or similar event the
/// holder may convert at the ratio plus the additional shares in the
/// indenture's table, put the bond at par plus accrued, or carry on,
/// whichever is worth most.
///
/// The table has one row per effective date and one column per share
/// price, as printed in the indenture, in additional shares per bond
/// of `face_value`. It is interpolated linearly in both directions;
/// outside the price range, or after the last date, no additional
/// shares are due. The event itself arrives at `event_intensity` per
/// year — a modelling input (a takeover probability), not a contract
/// term.
#[derive(Debug, Clone, PartialEq)]
pub struct FundamentalChangeMakeWhole {
    /// Annual intensity of a fundamental change.
    pub event_intensity: f64,
    /// Share-price columns, strictly increasing.
    pub stock_prices: Vec<f64>,
    /// Effective-date rows, strictly increasing.
    pub effective_dates: Vec<NaiveDate>,
    /// `additional_shares[date_row][price_column]`.
    pub additional_shares: Vec<Vec<f64>>,
}

impl FundamentalChangeMakeWhole {
    pub(crate) fn validate(&self) -> Result<(), RustyQLibError> {
        let invalid = |what: &str| Err(RustyQLibError::invalid_input("convertible", what));
        if !(self.event_intensity >= 0.0 && self.event_intensity.is_finite()) {
            return invalid("the fundamental-change intensity must be non-negative");
        }
        if self.stock_prices.is_empty() || self.effective_dates.is_empty() {
            return invalid("the make-whole table needs at least one price and one date");
        }
        if self
            .stock_prices
            .iter()
            .any(|p| !(p.is_finite() && *p > 0.0))
            || self.stock_prices.windows(2).any(|w| w[0] >= w[1])
        {
            return invalid("the make-whole share prices must be positive and increasing");
        }
        if self.effective_dates.windows(2).any(|w| w[0] >= w[1]) {
            return invalid("the make-whole effective dates must be increasing");
        }
        if self.additional_shares.len() != self.effective_dates.len()
            || self
                .additional_shares
                .iter()
                .any(|row| row.len() != self.stock_prices.len())
        {
            return invalid("the make-whole table must be dates x prices");
        }
        if self
            .additional_shares
            .iter()
            .flatten()
            .any(|x| !(x.is_finite() && *x >= 0.0))
        {
            return invalid("the make-whole additional shares must be non-negative");
        }
        Ok(())
    }

    /// The table row at `time`, interpolated between the rows' times:
    /// the first row before the first date, zeros after the last.
    pub(crate) fn row_at(&self, time: f64, row_times: &[f64]) -> Vec<f64> {
        let last = row_times.len() - 1;
        if time > row_times[last] + 1e-9 {
            return vec![0.0; self.stock_prices.len()];
        }
        if time <= row_times[0] {
            return self.additional_shares[0].clone();
        }
        let k = row_times.partition_point(|&t| t < time).clamp(1, last);
        let (t0, t1) = (row_times[k - 1], row_times[k]);
        let w = ((time - t0) / (t1 - t0)).clamp(0.0, 1.0);
        self.additional_shares[k - 1]
            .iter()
            .zip(&self.additional_shares[k])
            .map(|(a, b)| a + w * (b - a))
            .collect()
    }
}

/// Dividend protection: when a cash dividend exceeds `threshold` per
/// share, the conversion ratio is adjusted by `S / (S - (D - threshold))`
/// at the ex-date so that parity is preserved for the excess. A zero
/// threshold is full protection.
///
/// On the engines the adjustment is applied through the ex-date jump:
/// the protected part of a dividend no longer moves the holder's claim,
/// so only `min(D, threshold)` jumps, and full protection means no
/// jump at all (the holder effectively owns the total-return share).
/// This neglects the scaling of the unprotected amount by the ratio
/// factors accumulated so far, an error of order `threshold * (ratio
/// growth - 1)`. Protection applies to [`CashDividend`]s only; a
/// continuous yield in the market struct is never protected, so under
/// full protection express the dividends as cash entries or set the
/// yield to zero.
///
/// [`CashDividend`]: super::CashDividend
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DividendProtection {
    /// Per share, per dividend; dividends up to this amount are not
    /// protected. Zero is full protection.
    pub threshold: f64,
}

pub(crate) fn validate_dividend_protection(
    protection: Option<DividendProtection>,
) -> Result<(), RustyQLibError> {
    if let Some(p) = protection {
        if !(p.threshold.is_finite() && p.threshold >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!(
                    "the dividend protection threshold must be non-negative, got {}",
                    p.threshold
                ),
            ));
        }
    }
    Ok(())
}

/// Mandatory conversion (PEPS / DECS / ACES): at maturity the bond
/// converts into shares whatever the share price, with the number of
/// shares set by the terminal price between two ratios:
///
/// - at or below the lower threshold `face / max_ratio`, `max_ratio`
///   shares (the holder takes the full downside);
/// - between the thresholds, shares worth exactly the face (the
///   "dead zone": no participation);
/// - at or above the upper threshold `face / conversion_ratio`,
///   `conversion_ratio` shares (the minimum ratio; the holder takes
///   the upside above the conversion premium).
///
/// No principal is paid in cash, so the bond floor is the coupons
/// alone; early conversion, if the window allows it, is at the
/// minimum ratio, as the indentures provide. Calls and puts on the
/// chassis keep their ordinary meaning (a mandatory's issuer early
/// settlement, which delivers the maximum ratio plus the remaining
/// coupons, is not modelled).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MandatoryConversion {
    /// Shares per bond at or below the lower threshold; must exceed the
    /// bond's `conversion_ratio`, the minimum.
    pub max_ratio: f64,
}
