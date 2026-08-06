//! Unadjusted coupon-date generation for bonds.
//!
//! Bond accrual runs between *scheduled* (unadjusted) coupon dates — a
//! payment that slips to the next business day does not change accrued
//! interest — so this roll is calendar-free: anchors are generated
//! backward from maturity by whole months, with an optional end-of-month
//! rule. Payment-date adjustment is applied later by the bond itself.

use chrono::{Datelike, Months, NaiveDate};

use crate::core::errors::RustyQLibError;

/// The unadjusted coupon grid of a bond.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CouponSchedule {
    /// The anchor at or before the dated date. Equal to the dated date
    /// for a regular schedule; earlier when the first coupon is a short
    /// front stub, in which case it is the notional (quasi) period start
    /// used as the Act/Act ICMA reference.
    pub prev_anchor: NaiveDate,
    /// Scheduled coupon dates, strictly increasing, ending at maturity.
    /// The dated date itself is not included.
    pub dates: Vec<NaiveDate>,
}

/// Last calendar day of `date`'s month.
pub fn end_of_month(date: NaiveDate) -> NaiveDate {
    let first_next = if date.month() == 12 {
        NaiveDate::from_ymd_opt(date.year() + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(date.year(), date.month() + 1, 1)
    }
    .expect("valid month start");
    first_next - chrono::Duration::days(1)
}

/// Whether `date` is the last day of its month.
pub fn is_end_of_month(date: NaiveDate) -> bool {
    date == end_of_month(date)
}

/// Generate the coupon grid: anchors rolled backward from `maturity`
/// every `months` months until at or before `dated`. Each anchor is
/// computed directly from maturity (no cumulative clamping). With
/// `eom = true` every anchor snaps to its month-end — the rule for bonds
/// maturing on a month-end, where all coupons fall on month-ends.
pub fn coupon_dates(
    dated: NaiveDate,
    maturity: NaiveDate,
    months: u32,
    eom: bool,
) -> Result<CouponSchedule, RustyQLibError> {
    if months == 0 {
        return Err(RustyQLibError::invalid_input(
            "coupon schedule",
            "the coupon period must be at least one month",
        ));
    }
    if maturity <= dated {
        return Err(RustyQLibError::invalid_input(
            "coupon schedule",
            format!("maturity {maturity} must be after the dated date {dated}"),
        ));
    }

    let mut dates = vec![maturity];
    let mut prev_anchor = None;
    for k in 1.. {
        let mut anchor = maturity - Months::new(k * months);
        if eom {
            anchor = end_of_month(anchor);
        }
        if anchor <= dated {
            prev_anchor = Some(anchor);
            break;
        }
        dates.push(anchor);
    }
    dates.reverse();
    Ok(CouponSchedule {
        prev_anchor: prev_anchor.expect("the backward roll terminates at or before the dated date"),
        dates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn regular_semiannual_grid_on_the_fifteenth() {
        // classic UST May 15 / Nov 15 cycle
        let s = coupon_dates(d(2026, 5, 15), d(2028, 5, 15), 6, false).unwrap();
        assert_eq!(s.prev_anchor, d(2026, 5, 15)); // regular: dated on-cycle
        assert_eq!(
            s.dates,
            vec![
                d(2026, 11, 15),
                d(2027, 5, 15),
                d(2027, 11, 15),
                d(2028, 5, 15)
            ]
        );
    }

    #[test]
    fn end_of_month_rule_snaps_every_anchor() {
        // maturing Jun 30: EOM coupons must be Dec 31, not Dec 30
        let s = coupon_dates(d(2026, 6, 30), d(2028, 6, 30), 6, true).unwrap();
        assert_eq!(s.prev_anchor, d(2026, 6, 30));
        assert_eq!(
            s.dates,
            vec![
                d(2026, 12, 31),
                d(2027, 6, 30),
                d(2027, 12, 31),
                d(2028, 6, 30)
            ]
        );
    }

    #[test]
    fn without_eom_the_day_clamps_but_does_not_snap() {
        // maturity Jun 30, no EOM rule: December anchors stay on the 30th
        let s = coupon_dates(d(2026, 6, 30), d(2027, 6, 30), 6, false).unwrap();
        assert_eq!(s.dates, vec![d(2026, 12, 30), d(2027, 6, 30)]);
    }

    #[test]
    fn no_cumulative_clamping_through_short_months() {
        // maturity Aug 31, quarterly, no EOM: anchors from maturity directly
        // (May 31, Feb 28/29, Nov 30), not Feb-clamped copies
        let s = coupon_dates(d(2026, 11, 30), d(2027, 8, 31), 3, false).unwrap();
        assert_eq!(
            s.dates,
            vec![d(2027, 2, 28), d(2027, 5, 31), d(2027, 8, 31)]
        );
        assert_eq!(s.prev_anchor, d(2026, 11, 30));
    }

    #[test]
    fn short_front_stub_reports_the_quasi_anchor() {
        // dated Jul 1 sits inside the May 15 / Nov 15 cycle
        let s = coupon_dates(d(2026, 7, 1), d(2027, 5, 15), 6, false).unwrap();
        assert_eq!(s.prev_anchor, d(2026, 5, 15));
        assert_eq!(s.dates, vec![d(2026, 11, 15), d(2027, 5, 15)]);
    }

    #[test]
    fn single_period_bond() {
        let s = coupon_dates(d(2026, 5, 15), d(2026, 11, 15), 6, false).unwrap();
        assert_eq!(s.dates, vec![d(2026, 11, 15)]);
        assert_eq!(s.prev_anchor, d(2026, 5, 15));
    }

    #[test]
    fn end_of_month_helpers() {
        assert_eq!(end_of_month(d(2028, 2, 1)), d(2028, 2, 29)); // leap
        assert_eq!(end_of_month(d(2026, 12, 5)), d(2026, 12, 31));
        assert!(is_end_of_month(d(2026, 6, 30)));
        assert!(!is_end_of_month(d(2026, 6, 29)));
    }

    #[test]
    fn rejects_bad_inputs() {
        assert!(coupon_dates(d(2026, 5, 15), d(2026, 5, 15), 6, false).is_err());
        assert!(coupon_dates(d(2026, 5, 15), d(2027, 5, 15), 0, false).is_err());
    }
}
