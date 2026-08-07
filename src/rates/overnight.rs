//! Shared overnight-rate machinery for money-market futures.
//!
//! Both fed funds and SOFR futures settle on an average of a daily
//! overnight rate: published fixings for days already realized, curve
//! forwards for the rest. The pieces they share live here.

use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// Published daily fixings, keyed by the date the rate applies to.
/// Gaps (weekends, holidays) are filled by carrying the previous
/// business day's rate forward.
pub type RateFixings = BTreeMap<NaiveDate, f64>;

/// The fixing applying to `day`: its own, or the most recent earlier one
/// (weekend and holiday carry-forward).
pub fn fixing_on_or_before(fixings: &RateFixings, day: NaiveDate) -> Result<f64, RustyQLibError> {
    fixings
        .range(..=day)
        .next_back()
        .map(|(_, &rate)| rate)
        .ok_or_else(|| {
            RustyQLibError::invalid_input(
                "fixings",
                format!("no published fixing on or before {day}"),
            )
        })
}

/// Simple overnight forward rate from `day` to the next calendar day,
/// quoted Act/360 off `curve`.
pub fn overnight_forward(curve: &YieldCurve, day: NaiveDate) -> Result<f64, RustyQLibError> {
    simple_forward(curve, day, 1)
}

/// Simple forward rate over `days` calendar days from `day`, quoted
/// Act/360 — the rate `r` with `1 + r * days/360 = df(day)/df(day+days)`.
pub fn simple_forward(
    curve: &YieldCurve,
    day: NaiveDate,
    days: i64,
) -> Result<f64, RustyQLibError> {
    if days <= 0 {
        return Err(RustyQLibError::invalid_input(
            "simple_forward",
            format!("the period must be at least one day, got {days}"),
        ));
    }
    let df0 = curve.df_date(day);
    let df1 = curve.df_date(day + Duration::days(days));
    if !df0.is_finite() || df0 <= 0.0 || !df1.is_finite() || df1 <= 0.0 {
        return Err(RustyQLibError::NumericalError(format!(
            "non-positive discount factor around {day}"
        )));
    }
    Ok((df0 / df1 - 1.0) * 360.0 / days as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn carry_forward_picks_the_latest_earlier_fixing() {
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 9, 4), 0.05);
        fixings.insert(d(2026, 9, 8), 0.04);
        // exact hit, and the weekend days after the Friday fixing
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 4)).unwrap(), 0.05);
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 6)).unwrap(), 0.05);
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 8)).unwrap(), 0.04);
        // before the first fixing there is nothing to carry
        assert!(fixing_on_or_before(&fixings, d(2026, 9, 1)).is_err());
    }

    #[test]
    fn simple_forward_inverts_the_discount_ratio() {
        let curve = YieldCurve::flat(
            0.04,
            d(2026, 8, 6),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        for days in [1, 3, 7, 91] {
            let day = d(2026, 9, 16);
            let r = simple_forward(&curve, day, days).unwrap();
            let ratio = curve.df_date(day) / curve.df_date(day + Duration::days(days));
            assert!(
                (1.0 + r * days as f64 / 360.0 - ratio).abs() < 1e-14,
                "{days}d"
            );
        }
        // the one-day case is the overnight forward
        let day = d(2026, 9, 16);
        assert_eq!(
            overnight_forward(&curve, day).unwrap(),
            simple_forward(&curve, day, 1).unwrap()
        );
        assert!(simple_forward(&curve, day, 0).is_err());
    }
}
