//! Shared leg machinery for swaps: schedule construction and the fixed
//! and floating leg present values.

use chrono::NaiveDate;

use crate::bonds::schedule::coupon_dates;
use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::checked_df;

/// One swap accrual period. Unlike bonds, swap accrual runs between
/// business-day **adjusted** dates; payment can lag the accrual end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccrualPeriod {
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub payment: NaiveDate,
}

/// Build the accrual periods of one leg: unadjusted anchors rolled
/// backward from `maturity` (stub at the front), each anchor adjusted on
/// `calendar` under `convention`, payments lagged `payment_lag` business
/// days after the accrual end. Anchors that collapse after adjustment
/// are dropped.
pub fn accrual_periods(
    effective: NaiveDate,
    maturity: NaiveDate,
    frequency: Frequency,
    calendar: &Calendar,
    convention: BusinessDayConvention,
    payment_lag: i64,
) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
    if payment_lag < 0 {
        return Err(RustyQLibError::invalid_input(
            "payment_lag",
            format!("must be non-negative, got {payment_lag}"),
        ));
    }
    let schedule = coupon_dates(effective, maturity, frequency.months(), false)?;
    let mut periods = Vec::with_capacity(schedule.dates.len());
    let mut start = calendar.adjust(effective, convention);
    for &anchor in &schedule.dates {
        let end = calendar.adjust(anchor, convention);
        if end <= start {
            continue; // anchors collapsed by adjustment
        }
        periods.push(AccrualPeriod {
            start,
            end,
            payment: calendar.add_business_days(end, payment_lag),
        });
        start = end;
    }
    if periods.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "swap schedule",
            format!("no accrual periods between {effective} and {maturity}"),
        ));
    }
    Ok(periods)
}

/// PV of a fixed leg: `sum rate * tau_i * df(pay_i)` on unit notional.
pub fn fixed_leg_pv(
    periods: &[AccrualPeriod],
    rate: f64,
    day_count: DayCountConvention,
    discount: &YieldCurve,
) -> f64 {
    rate * annuity(periods, day_count, discount)
}

/// The fixed-leg annuity `sum tau_i * df(pay_i)` on unit notional — the
/// PV of 1 unit of rate, and the sensitivity of the swap to its fixed
/// rate.
pub fn annuity(
    periods: &[AccrualPeriod],
    day_count: DayCountConvention,
    discount: &YieldCurve,
) -> f64 {
    periods
        .iter()
        .map(|p| day_count.year_fraction(p.start, p.end) * discount.df_date(p.payment))
        .sum()
}

/// PV of a floating leg plus `spread` on unit notional. Each period's
/// floating accrual is forecast from the curve ratio
/// `df(start)/df(end) - 1` (the simple forward, and equally the
/// compounded overnight accrual); the spread accrues on `day_count`.
pub fn float_leg_pv(
    periods: &[AccrualPeriod],
    spread: f64,
    day_count: DayCountConvention,
    discount: &YieldCurve,
    forecast: &YieldCurve,
) -> Result<f64, RustyQLibError> {
    let mut pv = 0.0;
    for p in periods {
        let ratio = checked_df(forecast, p.start)? / checked_df(forecast, p.end)?;
        let tau = day_count.year_fraction(p.start, p.end);
        pv += (ratio - 1.0 + spread * tau) * discount.df_date(p.payment);
    }
    Ok(pv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat(rate: f64, reference: NaiveDate) -> YieldCurve {
        YieldCurve::flat(
            rate,
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn quarterly_schedule_rolls_backward_and_adjusts() {
        // effective Thu 2026-08-06, 1y quarterly, modified following
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            0,
        )
        .unwrap();
        assert_eq!(periods.len(), 4);
        // contiguous: each period starts where the previous ended
        for pair in periods.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        // Nov 6 2026 is a Friday, stays; Feb 6 2027 is a Saturday -> Mon Feb 8
        assert_eq!(periods[0].end, d(2026, 11, 6));
        assert_eq!(periods[1].end, d(2027, 2, 8));
        // Aug 6 2027 is a Friday
        assert_eq!(periods[3].end, d(2027, 8, 6));
        // no lag: payment = accrual end
        assert!(periods.iter().all(|p| p.payment == p.end));
    }

    #[test]
    fn payment_lag_shifts_payments_by_business_days() {
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            2,
        )
        .unwrap();
        // first period ends Fri Nov 6 2026 -> T+2 is Tue Nov 10
        assert_eq!(periods[0].payment, d(2026, 11, 10));
        for p in &periods {
            assert!(p.payment > p.end);
        }
    }

    #[test]
    fn float_leg_telescopes_on_a_single_curve() {
        // unadjusted periods and zero lag: the df ratios telescope so the
        // float leg is exactly df(effective) - df(maturity) = 1 - df(T)
        let reference = d(2026, 8, 6);
        let curve = flat(0.045, reference);
        let periods = accrual_periods(
            reference,
            d(2028, 8, 6),
            Frequency::Quarterly,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
            0,
        )
        .unwrap();
        let pv = float_leg_pv(&periods, 0.0, DayCountConvention::Act360, &curve, &curve).unwrap();
        let expected = 1.0 - curve.df_date(d(2028, 8, 6));
        assert!((pv - expected).abs() < 1e-14, "{pv} vs {expected}");
    }

    #[test]
    fn annuity_is_the_fixed_rate_sensitivity() {
        let reference = d(2026, 8, 6);
        let curve = flat(0.04, reference);
        let periods = accrual_periods(
            reference,
            d(2028, 8, 6),
            Frequency::Semiannual,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            0,
        )
        .unwrap();
        let dc = DayCountConvention::Thirty360;
        let a = annuity(&periods, dc, &curve);
        assert!(a > 0.0);
        // fixed_leg_pv is linear in the rate with slope = annuity
        let pv1 = fixed_leg_pv(&periods, 0.04, dc, &curve);
        let pv2 = fixed_leg_pv(&periods, 0.05, dc, &curve);
        assert!(((pv2 - pv1) - 0.01 * a).abs() < 1e-15);
    }

    #[test]
    fn rejects_bad_inputs() {
        let e = d(2026, 8, 6);
        let cal = Calendar::WeekendsOnly;
        let conv = BusinessDayConvention::Following;
        assert!(accrual_periods(e, e, Frequency::Quarterly, &cal, conv, 0).is_err());
        assert!(accrual_periods(e, d(2027, 8, 6), Frequency::Quarterly, &cal, conv, -1).is_err());
    }
}
