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
}

impl DayCountConvention {
    /// Year fraction from `start` to `end` under this convention.
    /// Negative if `end` is before `start`.
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
        }
    }
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
}
