//! Holiday calendars, business-day conventions and schedule generation.
//!
//! Holidays are computed from rules (Easter algorithm, nth-weekday-of-month,
//! observance shifts), not stored as date lists, so any year works. The
//! named calendars cover the markets an equity derivatives book usually
//! needs — weekends-only, TARGET (EUR), NYSE (US equities), UK bank
//! holidays — and [`Calendar::Custom`] takes an explicit holiday list for
//! anything else. One-off closures (mourning days, exchange incidents) are
//! not modelled; add them through `Custom`.
//!
//! ```
//! use chrono::NaiveDate;
//! use rustyqlib::core::calendar::{BusinessDayConvention, Calendar};
//!
//! let nyse = Calendar::UsNyse;
//! let good_friday = NaiveDate::from_ymd_opt(2026, 4, 3).unwrap();
//! assert!(!nyse.is_business_day(good_friday));
//! // settle T+2 over a holiday weekend
//! let trade = NaiveDate::from_ymd_opt(2026, 4, 1).unwrap();
//! assert_eq!(
//!     nyse.add_business_days(trade, 2),
//!     NaiveDate::from_ymd_opt(2026, 4, 6).unwrap()
//! );
//! let _ = BusinessDayConvention::ModifiedFollowing;
//! ```

use chrono::{Datelike, Duration, Months, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::core::errors::RustyQLibError;

// ── Business-day conventions ────────────────────────────────────────────

/// How a date falling on a non-business day is adjusted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BusinessDayConvention {
    /// Leave the date as it is.
    Unadjusted,
    /// Move to the next business day.
    #[default]
    Following,
    /// Move to the next business day unless that crosses into the next
    /// calendar month, in which case move to the preceding business day.
    ModifiedFollowing,
    /// Move to the previous business day.
    Preceding,
    /// Move to the previous business day unless that crosses into the
    /// previous calendar month, in which case move to the following one.
    ModifiedPreceding,
}

// ── Calendars ───────────────────────────────────────────────────────────

/// A holiday calendar: weekends plus market-specific holidays.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Calendar {
    /// Saturdays and Sundays only.
    WeekendsOnly,
    /// TARGET (Trans-European Automated Real-time Gross settlement):
    /// the euro-area settlement calendar.
    Target,
    /// New York Stock Exchange trading calendar.
    UsNyse,
    /// US government bond market (SIFMA recommended full-close holidays):
    /// the NYSE set plus Columbus Day and Veterans Day. SIFMA early
    /// closes and its occasional Good Friday exceptions are not modelled.
    UsGovernmentBond,
    /// England-and-Wales bank holidays (regular rules; one-off royal or
    /// millennium holidays are not modelled).
    UkSettlement,
    /// Weekends plus an explicit list of extra holidays.
    Custom { holidays: BTreeSet<NaiveDate> },
}

impl Calendar {
    /// Saturday or Sunday.
    pub fn is_weekend(date: NaiveDate) -> bool {
        matches!(date.weekday(), Weekday::Sat | Weekday::Sun)
    }

    /// A non-weekend holiday on this calendar.
    pub fn is_holiday(&self, date: NaiveDate) -> bool {
        if Self::is_weekend(date) {
            return false;
        }
        match self {
            Calendar::WeekendsOnly => false,
            Calendar::Target => is_target_holiday(date),
            Calendar::UsNyse => is_nyse_holiday(date),
            Calendar::UsGovernmentBond => is_us_government_bond_holiday(date),
            Calendar::UkSettlement => is_uk_holiday(date),
            Calendar::Custom { holidays } => holidays.contains(&date),
        }
    }

    /// Neither a weekend nor a holiday.
    pub fn is_business_day(&self, date: NaiveDate) -> bool {
        !Self::is_weekend(date) && !self.is_holiday(date)
    }

    /// Adjust a date to a business day under the given convention.
    pub fn adjust(&self, date: NaiveDate, convention: BusinessDayConvention) -> NaiveDate {
        use BusinessDayConvention::*;
        if convention == Unadjusted || self.is_business_day(date) {
            return date;
        }
        match convention {
            Following => self.next_business_day(date),
            Preceding => self.previous_business_day(date),
            ModifiedFollowing => {
                let next = self.next_business_day(date);
                if next.month() != date.month() {
                    self.previous_business_day(date)
                } else {
                    next
                }
            }
            ModifiedPreceding => {
                let prev = self.previous_business_day(date);
                if prev.month() != date.month() {
                    self.next_business_day(date)
                } else {
                    prev
                }
            }
            Unadjusted => date,
        }
    }

    /// The first business day strictly after weekends/holidays from `date`
    /// (returns `date` itself when it is already a business day).
    fn next_business_day(&self, mut date: NaiveDate) -> NaiveDate {
        while !self.is_business_day(date) {
            date += Duration::days(1);
        }
        date
    }

    fn previous_business_day(&self, mut date: NaiveDate) -> NaiveDate {
        while !self.is_business_day(date) {
            date -= Duration::days(1);
        }
        date
    }

    /// Move `n` business days (n may be negative). `n = 0` returns the
    /// date unchanged, even on a holiday. The classic settlement-lag
    /// helper: `calendar.add_business_days(trade_date, 2)` is T+2.
    pub fn add_business_days(&self, date: NaiveDate, n: i64) -> NaiveDate {
        let mut d = date;
        let step = if n >= 0 { 1 } else { -1 };
        let mut remaining = n.abs();
        while remaining > 0 {
            d += Duration::days(step);
            if self.is_business_day(d) {
                remaining -= 1;
            }
        }
        d
    }

    /// Advance by a calendar period and adjust the result. Month/year
    /// arithmetic clamps to the end of month (Jan 31 + 1M = Feb 28/29)
    /// before adjustment.
    pub fn advance(
        &self,
        date: NaiveDate,
        period: Period,
        convention: BusinessDayConvention,
    ) -> NaiveDate {
        let moved = match period {
            Period::Days(n) => return self.add_business_days(date, n),
            Period::Weeks(n) => date + Duration::weeks(n),
            Period::Months(n) => add_months_signed(date, n),
            Period::Years(n) => add_months_signed(date, 12 * n),
        };
        self.adjust(moved, convention)
    }

    /// Business days in the half-open interval `(from, to]`; negative when
    /// `to < from`.
    pub fn business_days_between(&self, from: NaiveDate, to: NaiveDate) -> i64 {
        if to < from {
            return -self.business_days_between(to, from);
        }
        let mut count = 0;
        let mut d = from;
        while d < to {
            d += Duration::days(1);
            if self.is_business_day(d) {
                count += 1;
            }
        }
        count
    }
}

/// A calendar period for [`Calendar::advance`]. `Days` counts **business**
/// days; the others move in calendar time and then adjust.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Period {
    Days(i64),
    Weeks(i64),
    Months(i32),
    Years(i32),
}

fn add_months_signed(date: NaiveDate, n: i32) -> NaiveDate {
    if n >= 0 {
        date + Months::new(n as u32)
    } else {
        date - Months::new((-n) as u32)
    }
}

// ── Holiday rules ───────────────────────────────────────────────────────

/// Easter Sunday by the Meeus/Jones/Butcher Gregorian algorithm.
pub fn easter_sunday(year: i32) -> NaiveDate {
    let a = year % 19;
    let b = year / 100;
    let c = year % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = ((h + l - 7 * m + 114) % 31) + 1;
    NaiveDate::from_ymd_opt(year, month as u32, day as u32).expect("valid Easter date")
}

/// The `n`-th (1-based) given weekday of a month.
fn nth_weekday(year: i32, month: u32, weekday: Weekday, n: u32) -> NaiveDate {
    let first = NaiveDate::from_ymd_opt(year, month, 1).expect("valid month start");
    let offset = (7 + weekday.num_days_from_monday() as i64
        - first.weekday().num_days_from_monday() as i64)
        % 7;
    first + Duration::days(offset + 7 * (n as i64 - 1))
}

/// The last given weekday of a month.
fn last_weekday(year: i32, month: u32, weekday: Weekday) -> NaiveDate {
    let first_next = if month == 12 {
        NaiveDate::from_ymd_opt(year + 1, 1, 1)
    } else {
        NaiveDate::from_ymd_opt(year, month + 1, 1)
    }
    .expect("valid month start");
    let last = first_next - Duration::days(1);
    let offset = (7 + last.weekday().num_days_from_monday() as i64
        - weekday.num_days_from_monday() as i64)
        % 7;
    last - Duration::days(offset)
}

/// TARGET holidays: New Year, Good Friday, Easter Monday, Labour Day,
/// Christmas, Boxing Day (Dec 26, since 2000).
fn is_target_holiday(date: NaiveDate) -> bool {
    let (y, m, d) = (date.year(), date.month(), date.day());
    if matches!((m, d), (1, 1) | (5, 1) | (12, 25)) {
        return true;
    }
    if m == 12 && d == 26 && y >= 2000 {
        return true;
    }
    let easter = easter_sunday(y);
    date == easter - Duration::days(2) || date == easter + Duration::days(1)
}

/// NYSE trading holidays (regular rules): New Year (Sunday observed on
/// Monday), Martin Luther King Jr. Day (since 1998), Washington's
/// Birthday, Good Friday, Memorial Day, Juneteenth (since 2022),
/// Independence Day, Labor Day, Thanksgiving, Christmas. Saturday
/// holidays for New Year are not observed (exchange convention);
/// Saturday Independence Day / Christmas are observed on Friday.
fn is_nyse_holiday(date: NaiveDate) -> bool {
    let (y, m, d) = (date.year(), date.month(), date.day());
    let wd = date.weekday();

    // New Year's Day: Jan 1, or Jan 2 when the 1st is a Sunday
    if m == 1 && (d == 1 || (d == 2 && wd == Weekday::Mon)) {
        return true;
    }
    // MLK Day: third Monday of January, since 1998
    if y >= 1998 && m == 1 && date == nth_weekday(y, 1, Weekday::Mon, 3) {
        return true;
    }
    // Washington's Birthday: third Monday of February
    if m == 2 && date == nth_weekday(y, 2, Weekday::Mon, 3) {
        return true;
    }
    // Good Friday
    if date == easter_sunday(y) - Duration::days(2) {
        return true;
    }
    // Memorial Day: last Monday of May
    if m == 5 && date == last_weekday(y, 5, Weekday::Mon) {
        return true;
    }
    // Juneteenth: June 19 (Fri if Sat, Mon if Sun), since 2022
    if y >= 2022 && observed_on(date, 6, 19) {
        return true;
    }
    // Independence Day: July 4 (Fri if Sat, Mon if Sun)
    if observed_on(date, 7, 4) {
        return true;
    }
    // Labor Day: first Monday of September
    if m == 9 && date == nth_weekday(y, 9, Weekday::Mon, 1) {
        return true;
    }
    // Thanksgiving: fourth Thursday of November
    if m == 11 && date == nth_weekday(y, 11, Weekday::Thu, 4) {
        return true;
    }
    // Christmas: Dec 25 (Fri if Sat, Mon if Sun)
    if observed_on(date, 12, 25) {
        return true;
    }
    false
}

/// US bond-market holidays (SIFMA recommended full closes): the NYSE set
/// plus Columbus Day (second Monday of October) and Veterans Day
/// (November 11, observed). The bond market closes on both while the
/// NYSE trades.
fn is_us_government_bond_holiday(date: NaiveDate) -> bool {
    if is_nyse_holiday(date) {
        return true;
    }
    let (y, m) = (date.year(), date.month());
    // Columbus Day: second Monday of October
    if m == 10 && date == nth_weekday(y, 10, Weekday::Mon, 2) {
        return true;
    }
    // Veterans Day: November 11 (Fri if Sat, Mon if Sun)
    if observed_on(date, 11, 11) {
        return true;
    }
    false
}

/// Whether `date` is the observed weekday for the fixed holiday
/// `month`/`day`: the day itself on a weekday, the preceding Friday when
/// it falls on Saturday, the following Monday when it falls on Sunday.
fn observed_on(date: NaiveDate, month: u32, day: u32) -> bool {
    let holiday = match NaiveDate::from_ymd_opt(date.year(), month, day) {
        Some(d) => d,
        None => return false,
    };
    let observed = match holiday.weekday() {
        Weekday::Sat => holiday - Duration::days(1),
        Weekday::Sun => holiday + Duration::days(1),
        _ => holiday,
    };
    date == observed
}

/// England-and-Wales bank holidays (regular rules): New Year (observed),
/// Good Friday, Easter Monday, early-May bank holiday, spring bank
/// holiday, summer bank holiday, Christmas and Boxing Day (both observed
/// past the weekend).
fn is_uk_holiday(date: NaiveDate) -> bool {
    let (y, m, d) = (date.year(), date.month(), date.day());
    let wd = date.weekday();

    // New Year's Day, observed on Monday when Jan 1 is a weekend
    if m == 1 && (d == 1 || (d == 2 && wd == Weekday::Mon) || (d == 3 && wd == Weekday::Mon)) {
        return true;
    }
    let easter = easter_sunday(y);
    if date == easter - Duration::days(2) || date == easter + Duration::days(1) {
        return true;
    }
    // early-May bank holiday: first Monday of May
    if m == 5 && date == nth_weekday(y, 5, Weekday::Mon, 1) {
        return true;
    }
    // spring bank holiday: last Monday of May
    if m == 5 && date == last_weekday(y, 5, Weekday::Mon) {
        return true;
    }
    // summer bank holiday: last Monday of August
    if m == 8 && date == last_weekday(y, 8, Weekday::Mon) {
        return true;
    }
    // Christmas and Boxing Day: Dec 25/26, shifted past a weekend so two
    // weekdays are always taken (25th Sat -> 27th/28th, 25th Sun -> 27th/28th, ...)
    if m == 12 {
        let christmas = NaiveDate::from_ymd_opt(y, 12, 25).expect("valid date");
        let (obs_christmas, obs_boxing) = match christmas.weekday() {
            Weekday::Fri => (25, 28), // Boxing Day Sat -> Mon 28
            Weekday::Sat => (27, 28), // Mon 27 and Tue 28
            Weekday::Sun => (27, 28), // Boxing Mon 26? convention: Mon 26 is Boxing observed, Tue 27 Christmas observed; use 26/27
            _ => (25, 26),
        };
        // Sunday Christmas: Boxing Day (Mon 26) and substitute Christmas (Tue 27)
        if christmas.weekday() == Weekday::Sun {
            return d == 26 || d == 27;
        }
        return d == obs_christmas || d == obs_boxing;
    }
    false
}

// ── Schedules ───────────────────────────────────────────────────────────

/// Direction of periodic date generation. Backward (from termination) is
/// the market default: the stub, if any, lands at the front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DateGeneration {
    #[default]
    Backward,
    Forward,
}

/// A periodic date schedule: unadjusted anchor dates rolled from a period,
/// then business-day adjusted. Used for autocallable observation dates,
/// coupon schedules and averaging fixings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    /// Adjusted, strictly increasing dates, ending at the (adjusted)
    /// termination date. The effective date itself is not included.
    pub dates: Vec<NaiveDate>,
}

impl Schedule {
    /// Generate a schedule from `effective` (exclusive) to `termination`
    /// (inclusive) every `months` months.
    ///
    /// Backward generation rolls anchor dates back from termination, so a
    /// remainder shorter than a full period becomes a short **front** stub;
    /// forward generation rolls from the effective date and leaves a short
    /// **back** stub. Anchor dates are adjusted with `convention`;
    /// duplicates after adjustment collapse.
    pub fn generate(
        effective: NaiveDate,
        termination: NaiveDate,
        months: u32,
        calendar: &Calendar,
        convention: BusinessDayConvention,
        generation: DateGeneration,
    ) -> Result<Schedule, RustyQLibError> {
        if months == 0 {
            return Err(RustyQLibError::invalid_input(
                "schedule",
                "the period must be at least one month",
            ));
        }
        if termination <= effective {
            return Err(RustyQLibError::invalid_input(
                "schedule",
                format!("termination {termination} must be after the effective date {effective}"),
            ));
        }

        let mut anchors: Vec<NaiveDate> = Vec::new();
        match generation {
            DateGeneration::Backward => {
                let mut k = 0u32;
                loop {
                    k += months;
                    let date = termination - Months::new(k);
                    if date <= effective {
                        break;
                    }
                    anchors.push(date);
                }
                anchors.reverse();
                anchors.push(termination);
            }
            DateGeneration::Forward => {
                let mut k = 0u32;
                loop {
                    k += months;
                    let date = effective + Months::new(k);
                    if date >= termination {
                        break;
                    }
                    anchors.push(date);
                }
                anchors.push(termination);
            }
        }

        let mut dates: Vec<NaiveDate> = anchors
            .into_iter()
            .map(|d| calendar.adjust(d, convention))
            .collect();
        dates.dedup();
        // adjustment must not push a date past the adjusted termination
        let last = *dates.last().expect("schedule has at least one date");
        dates.retain(|d| *d <= last);
        dates.dedup();

        Ok(Schedule { dates })
    }

    /// Year fractions of every schedule date from `valuation` under the
    /// given day count.
    pub fn year_fractions(
        &self,
        valuation: NaiveDate,
        day_count: crate::core::daycount::DayCountConvention,
    ) -> Vec<f64> {
        self.dates
            .iter()
            .map(|d| day_count.year_fraction(valuation, *d))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.dates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn easter_matches_known_years() {
        assert_eq!(easter_sunday(2024), d(2024, 3, 31));
        assert_eq!(easter_sunday(2025), d(2025, 4, 20));
        assert_eq!(easter_sunday(2026), d(2026, 4, 5));
        assert_eq!(easter_sunday(2027), d(2027, 3, 28));
        assert_eq!(easter_sunday(2030), d(2030, 4, 21));
    }

    #[test]
    fn nyse_holidays_2026() {
        let c = Calendar::UsNyse;
        for holiday in [
            d(2026, 1, 1),   // New Year
            d(2026, 1, 19),  // MLK (3rd Monday)
            d(2026, 2, 16),  // Washington
            d(2026, 4, 3),   // Good Friday
            d(2026, 5, 25),  // Memorial Day
            d(2026, 6, 19),  // Juneteenth (Friday)
            d(2026, 7, 3),   // Independence Day observed (July 4 is Saturday)
            d(2026, 9, 7),   // Labor Day
            d(2026, 11, 26), // Thanksgiving
            d(2026, 12, 25), // Christmas (Friday)
        ] {
            assert!(!c.is_business_day(holiday), "{holiday} must be a holiday");
        }
        // regular trading days around them
        for business in [d(2026, 1, 2), d(2026, 4, 6), d(2026, 7, 6), d(2026, 11, 27)] {
            assert!(
                c.is_business_day(business),
                "{business} must be a business day"
            );
        }
    }

    #[test]
    fn nyse_sunday_new_year_observed_on_monday() {
        // Jan 1 2023 was a Sunday -> observed Monday Jan 2
        assert!(!Calendar::UsNyse.is_business_day(d(2023, 1, 2)));
        // Jan 1 2022 was a Saturday -> NOT observed (NYSE convention)
        assert!(Calendar::UsNyse.is_business_day(d(2021, 12, 31)));
    }

    #[test]
    fn target_holidays() {
        let c = Calendar::Target;
        for holiday in [
            d(2026, 1, 1),
            d(2026, 4, 3), // Good Friday
            d(2026, 4, 6), // Easter Monday
            d(2026, 5, 1), // Labour Day
            d(2026, 12, 25),
            // Dec 26 2026 is a Saturday: weekend, not counted as holiday
            d(2025, 12, 26),
        ] {
            assert!(!c.is_business_day(holiday), "{holiday} must be a holiday");
        }
        assert!(
            c.is_business_day(d(2026, 5, 25)),
            "no TARGET holiday on UK spring bank"
        );
    }

    #[test]
    fn uk_holidays_2026() {
        let c = Calendar::UkSettlement;
        for holiday in [
            d(2026, 1, 1),
            d(2026, 4, 3),  // Good Friday
            d(2026, 4, 6),  // Easter Monday
            d(2026, 5, 4),  // early-May bank holiday
            d(2026, 5, 25), // spring bank holiday
            d(2026, 8, 31), // summer bank holiday
            d(2026, 12, 25),
            d(2026, 12, 28), // Boxing Day observed (26th is a Saturday)
        ] {
            assert!(!c.is_business_day(holiday), "{holiday} must be a holiday");
        }
        // Christmas 2021: Sat 25th, Sun 26th -> observed Mon 27, Tue 28
        assert!(!c.is_business_day(d(2021, 12, 27)));
        assert!(!c.is_business_day(d(2021, 12, 28)));
        assert!(c.is_business_day(d(2021, 12, 29)));
    }

    #[test]
    fn adjust_conventions() {
        use BusinessDayConvention::*;
        let c = Calendar::WeekendsOnly;
        let saturday = d(2026, 5, 30);
        assert_eq!(c.adjust(saturday, Unadjusted), saturday);
        assert_eq!(c.adjust(saturday, Following), d(2026, 6, 1));
        assert_eq!(c.adjust(saturday, Preceding), d(2026, 5, 29));
        // month-end rollover: Sunday May 31 -> Following crosses into June,
        // ModifiedFollowing rolls back to Friday May 29
        let sunday_eom = d(2026, 5, 31);
        assert_eq!(c.adjust(sunday_eom, Following), d(2026, 6, 1));
        assert_eq!(c.adjust(sunday_eom, ModifiedFollowing), d(2026, 5, 29));
        // month-start mirror: ModifiedPreceding on Sunday Nov 1 rolls forward
        let sunday_som = d(2026, 11, 1);
        assert_eq!(c.adjust(sunday_som, Preceding), d(2026, 10, 30));
        assert_eq!(c.adjust(sunday_som, ModifiedPreceding), d(2026, 11, 2));
    }

    #[test]
    fn business_day_arithmetic_and_settlement_lag() {
        let c = Calendar::UsNyse;
        // T+2 from Wed Apr 1 2026 over Good Friday (Apr 3): Thu, then Mon
        assert_eq!(c.add_business_days(d(2026, 4, 1), 2), d(2026, 4, 6));
        // negative movement
        assert_eq!(c.add_business_days(d(2026, 4, 6), -1), d(2026, 4, 2));
        // count over the same stretch
        assert_eq!(c.business_days_between(d(2026, 4, 1), d(2026, 4, 6)), 2);
        assert_eq!(c.business_days_between(d(2026, 4, 6), d(2026, 4, 1)), -2);
    }

    #[test]
    fn advance_periods_clamp_month_ends() {
        let c = Calendar::WeekendsOnly;
        // Jan 31 + 1M clamps to Feb 28 (2026 is not a leap year), a Saturday
        // in 2026 -> Following moves to Mar 2
        assert_eq!(
            c.advance(
                d(2026, 1, 31),
                Period::Months(1),
                BusinessDayConvention::Following
            ),
            d(2026, 3, 2)
        );
        assert_eq!(
            c.advance(
                d(2026, 1, 31),
                Period::Months(1),
                BusinessDayConvention::ModifiedFollowing
            ),
            d(2026, 2, 27)
        );
        assert_eq!(
            c.advance(
                d(2026, 3, 15),
                Period::Years(1),
                BusinessDayConvention::Following
            ),
            d(2027, 3, 15)
        );
    }

    #[test]
    fn custom_calendar_takes_explicit_holidays() {
        let holidays: BTreeSet<NaiveDate> = [d(2026, 3, 17)].into();
        let c = Calendar::Custom { holidays };
        assert!(!c.is_business_day(d(2026, 3, 17)));
        assert!(c.is_business_day(d(2026, 3, 18)));
    }

    #[test]
    fn quarterly_backward_schedule_with_front_stub() {
        // 10 months of quarterly observations, backward: stub at the front
        let s = Schedule::generate(
            d(2026, 1, 15),
            d(2026, 11, 16),
            3,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
            DateGeneration::Backward,
        )
        .unwrap();
        assert_eq!(
            s.dates,
            vec![
                d(2026, 2, 16),
                d(2026, 5, 18),
                d(2026, 8, 17),
                d(2026, 11, 16)
            ]
        );
        // (Feb 16 anchor = Nov 16 - 9M; Feb 16 2026 is a Monday. May 16 is
        // a Saturday -> May 18; Aug 16 is a Sunday -> Aug 17.)
    }

    #[test]
    fn forward_schedule_puts_stub_at_the_back() {
        let s = Schedule::generate(
            d(2026, 1, 15),
            d(2026, 11, 16),
            3,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
            DateGeneration::Forward,
        )
        .unwrap();
        assert_eq!(
            s.dates,
            vec![
                d(2026, 4, 15),
                d(2026, 7, 15),
                d(2026, 10, 15),
                d(2026, 11, 16)
            ]
        );
    }

    #[test]
    fn schedule_dates_avoid_holidays() {
        // monthly observations across Good Friday 2026 (Apr 3) on NYSE
        let s = Schedule::generate(
            d(2026, 1, 5),
            d(2026, 6, 3),
            1,
            &Calendar::UsNyse,
            BusinessDayConvention::Following,
            DateGeneration::Backward,
        )
        .unwrap();
        for date in &s.dates {
            assert!(
                Calendar::UsNyse.is_business_day(*date),
                "{date} is not a business day"
            );
        }
        // Apr 3 anchor (Jun 3 - 2M) is Good Friday -> moved to Apr 6
        assert!(s.dates.contains(&d(2026, 4, 6)));
    }

    #[test]
    fn us_government_bond_calendar_extends_nyse() {
        let bond = Calendar::UsGovernmentBond;
        // Columbus Day 2026: Mon Oct 12 — bond market closed, NYSE open
        let columbus = d(2026, 10, 12);
        assert!(!bond.is_business_day(columbus));
        assert!(Calendar::UsNyse.is_business_day(columbus));
        // Veterans Day 2026: Wed Nov 11 — bond market closed, NYSE open
        let veterans = d(2026, 11, 11);
        assert!(!bond.is_business_day(veterans));
        assert!(Calendar::UsNyse.is_business_day(veterans));
        // Veterans Day 2028 falls on Saturday -> observed Friday Nov 10
        assert!(!bond.is_business_day(d(2028, 11, 10)));
        // NYSE holidays carry over (Good Friday 2026: Apr 3)
        assert!(!bond.is_business_day(d(2026, 4, 3)));
        // an ordinary Tuesday is open
        assert!(bond.is_business_day(d(2026, 10, 13)));
        // T+1 settlement over Columbus Day weekend: Fri Oct 9 -> Tue Oct 13
        assert_eq!(bond.add_business_days(d(2026, 10, 9), 1), d(2026, 10, 13));
    }

    #[test]
    fn schedule_rejects_bad_inputs() {
        let r = Schedule::generate(
            d(2026, 5, 1),
            d(2026, 1, 1),
            3,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
            DateGeneration::Backward,
        );
        assert!(r.is_err());
        let r = Schedule::generate(
            d(2026, 1, 1),
            d(2026, 5, 1),
            0,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
            DateGeneration::Backward,
        );
        assert!(r.is_err());
    }
}
