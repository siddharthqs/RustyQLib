use chrono::{Datelike, NaiveDate};
use serde::{Deserialize, Serialize};

/// Day count conventions used to convert a pair of dates into a year fraction.
///
/// This is the single bridge between calendar dates and the year-fraction
/// times used by [`crate::core::curves::YieldCurve`]; instruments carry their
/// own convention and curves carry theirs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DayCountConvention {
    /// Actual days / 365 (fixed)
    #[default]
    #[serde(alias = "Act/365", alias = "ACT/365", alias = "act365", alias = "A365")]
    Act365,
    /// Actual days / 360
    #[serde(alias = "Act/360", alias = "ACT/360", alias = "act360", alias = "A360")]
    Act360,
    /// 30/360 US (bond basis)
    #[serde(alias = "30/360", alias = "thirty360")]
    Thirty360,
    /// Actual/Actual (ISDA): actual days in each calendar year divided by
    /// that year's length (365 or 366).
    #[serde(
        alias = "Act/Act",
        alias = "ACT/ACT",
        alias = "act_act",
        alias = "ActActISDA",
        alias = "Act/Act ISDA"
    )]
    ActActIsda,
    /// Actual/Actual (ICMA): the bond-market convention (US Treasury
    /// street convention among others). Proper use requires the coupon
    /// reference period — see
    /// [`year_fraction_icma`](DayCountConvention::year_fraction_icma).
    /// The two-date [`year_fraction`](DayCountConvention::year_fraction)
    /// falls back to the ISDA rule.
    #[serde(alias = "ActActICMA", alias = "Act/Act ICMA", alias = "act_act_icma")]
    ActActIcma,
}

impl DayCountConvention {
    /// Year fraction from `start` to `end` under this convention.
    /// Negative if `end` is before `start`.
    ///
    /// [`ActActIcma`](Self::ActActIcma) has no meaningful two-date form;
    /// it falls back to the ISDA rule here and should be queried through
    /// [`year_fraction_icma`](Self::year_fraction_icma) when the coupon
    /// period is known.
    pub fn year_fraction(&self, start: NaiveDate, end: NaiveDate) -> f64 {
        match self {
            DayCountConvention::Act365 => (end - start).num_days() as f64 / 365.0,
            DayCountConvention::Act360 => (end - start).num_days() as f64 / 360.0,
            DayCountConvention::Thirty360 => {
                let (y1, m1, mut d1) = (start.year(), start.month() as i64, start.day() as i64);
                let (y2, m2, mut d2) = (end.year(), end.month() as i64, end.day() as i64);
                if d1 == 31 {
                    d1 = 30;
                }
                if d2 == 31 && d1 == 30 {
                    d2 = 30;
                }
                (360 * (y2 - y1) as i64 + 30 * (m2 - m1) + (d2 - d1)) as f64 / 360.0
            }
            DayCountConvention::ActActIsda | DayCountConvention::ActActIcma => {
                act_act_isda(start, end)
            }
        }
    }

    /// Actual/Actual (ICMA) year fraction of `[start, end]` within one
    /// coupon reference period `[ref_start, ref_end]` paying `frequency`
    /// times per year:
    /// `days(start, end) / (frequency * days(ref_start, ref_end))`.
    ///
    /// For a full regular period this is exactly `1 / frequency`; for the
    /// accrued portion of a period it prorates the coupon by actual days —
    /// the US Treasury street convention. Available on every convention so
    /// coupon code does not need to branch; only the ICMA variant is
    /// expected to use it.
    pub fn year_fraction_icma(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        ref_start: NaiveDate,
        ref_end: NaiveDate,
        frequency: u32,
    ) -> f64 {
        match self {
            DayCountConvention::ActActIcma => {
                let period_days = (ref_end - ref_start).num_days();
                if period_days <= 0 || frequency == 0 {
                    return 0.0;
                }
                (end - start).num_days() as f64 / (frequency as f64 * period_days as f64)
            }
            _ => self.year_fraction(start, end),
        }
    }
}

/// Actual/Actual (ISDA): split `[start, end]` at calendar-year boundaries
/// and weight each piece by its own year length.
fn act_act_isda(start: NaiveDate, end: NaiveDate) -> f64 {
    if start == end {
        return 0.0;
    }
    if end < start {
        return -act_act_isda(end, start);
    }
    let days_in_year = |year: i32| -> f64 {
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        if leap {
            366.0
        } else {
            365.0
        }
    };
    let (y1, y2) = (start.year(), end.year());
    if y1 == y2 {
        return (end - start).num_days() as f64 / days_in_year(y1);
    }
    // days remaining in the first year, whole years between, days into the last
    let first_year_end = NaiveDate::from_ymd_opt(y1 + 1, 1, 1).expect("valid date");
    let last_year_start = NaiveDate::from_ymd_opt(y2, 1, 1).expect("valid date");
    (first_year_end - start).num_days() as f64 / days_in_year(y1)
        + (y2 - y1 - 1) as f64
        + (end - last_year_start).num_days() as f64 / days_in_year(y2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn act365_one_year() {
        let yf = DayCountConvention::Act365.year_fraction(d(2026, 1, 1), d(2027, 1, 1));
        assert!((yf - 1.0).abs() < 1e-12);
    }

    #[test]
    fn act360_ninety_days() {
        let yf = DayCountConvention::Act360.year_fraction(d(2026, 1, 1), d(2026, 4, 1));
        assert!((yf - 90.0 / 360.0).abs() < 1e-12);
    }

    #[test]
    fn thirty360_half_year_month_ends() {
        // Jan 31 -> Jul 31 is exactly 0.5 under 30/360 US
        let yf = DayCountConvention::Thirty360.year_fraction(d(2026, 1, 31), d(2026, 7, 31));
        assert!((yf - 0.5).abs() < 1e-12);
    }

    #[test]
    fn act_act_isda_within_one_year() {
        // 2028 is a leap year: 90 days / 366
        let yf = DayCountConvention::ActActIsda.year_fraction(d(2028, 1, 1), d(2028, 3, 31));
        assert!((yf - 90.0 / 366.0).abs() < 1e-12);
        // non-leap 2026: 90 days / 365
        let yf = DayCountConvention::ActActIsda.year_fraction(d(2026, 1, 1), d(2026, 4, 1));
        assert!((yf - 90.0 / 365.0).abs() < 1e-12);
    }

    #[test]
    fn act_act_isda_across_years() {
        // Nov 1 2027 -> Mar 1 2028: 61/365 + 60/366
        let yf = DayCountConvention::ActActIsda.year_fraction(d(2027, 11, 1), d(2028, 3, 1));
        assert!((yf - (61.0 / 365.0 + 60.0 / 366.0)).abs() < 1e-12);
        // exactly one calendar year spanning a leap boundary
        let yf = DayCountConvention::ActActIsda.year_fraction(d(2027, 7, 1), d(2028, 7, 1));
        assert!((yf - (184.0 / 365.0 + 182.0 / 366.0)).abs() < 1e-12);
        // antisymmetric
        let fwd = DayCountConvention::ActActIsda.year_fraction(d(2027, 11, 1), d(2028, 3, 1));
        let back = DayCountConvention::ActActIsda.year_fraction(d(2028, 3, 1), d(2027, 11, 1));
        assert!((fwd + back).abs() < 1e-12);
    }

    #[test]
    fn act_act_icma_prorates_the_coupon_period() {
        let dc = DayCountConvention::ActActIcma;
        // semiannual period May 15 -> Nov 15 2026 (184 days)
        let (ref_start, ref_end) = (d(2026, 5, 15), d(2026, 11, 15));
        // full period = 1/frequency exactly
        let full = dc.year_fraction_icma(ref_start, ref_end, ref_start, ref_end, 2);
        assert!((full - 0.5).abs() < 1e-15);
        // 61 accrued days (May 15 -> Jul 15): 61 / (2 * 184)
        let part = dc.year_fraction_icma(ref_start, d(2026, 7, 15), ref_start, ref_end, 2);
        assert!((part - 61.0 / 368.0).abs() < 1e-15);
        // degenerate reference period yields zero rather than dividing by zero
        assert_eq!(
            dc.year_fraction_icma(ref_start, ref_end, ref_start, ref_start, 2),
            0.0
        );
    }

    #[test]
    fn act_act_icma_two_date_form_falls_back_to_isda() {
        let icma = DayCountConvention::ActActIcma.year_fraction(d(2026, 1, 1), d(2026, 7, 1));
        let isda = DayCountConvention::ActActIsda.year_fraction(d(2026, 1, 1), d(2026, 7, 1));
        assert_eq!(icma, isda);
    }

    #[test]
    fn act_act_aliases_deserialize() {
        let dc: DayCountConvention = serde_json::from_str(r#""Act/Act ICMA""#).unwrap();
        assert_eq!(dc, DayCountConvention::ActActIcma);
        let dc: DayCountConvention = serde_json::from_str(r#""ACT/ACT""#).unwrap();
        assert_eq!(dc, DayCountConvention::ActActIsda);
    }
}
