//! Leg schedule generation with the market's stub and roll conventions.
//!
//! A leg's accrual dates are unadjusted **anchors** rolled from one end
//! of the leg every `frequency`, snapped by a [`RollConvention`], then
//! business-day adjusted. Where the leg's length is not a whole number
//! of periods a **stub** appears, and [`StubConvention`] says where and
//! how long:
//!
//! - `ShortFront` (the market default): anchors roll backward from
//!   maturity, the remainder is a short first period.
//! - `LongFront`: as above, but the short first period is merged into
//!   the second.
//! - `ShortBack` / `LongBack`: anchors roll forward from the effective
//!   date, the remainder is a short (or merged, long) last period.
//!
//! The roll convention pins the anchor day within the month: the roll
//! origin's own day (`None`, with month-end clamping), the end of
//! month, a fixed day of month, or the IMM date — the third Wednesday,
//! the cycle IMM-dated swaps and futures strips run on.
//!
//! [`LegSchedule`] bundles the terms and produces [`AccrualPeriod`]s;
//! every swap in [`contracts`](crate::rates::contracts) builds its legs
//! through it.

use chrono::{Datelike, Months, NaiveDate};

use crate::core::calendar::{imm_date, BusinessDayConvention, Calendar, Frequency};
use crate::core::errors::RustyQLibError;
use crate::rates::leg::AccrualPeriod;

const FIELD: &str = "leg schedule";

/// Where a leg's odd period sits and whether it is merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StubConvention {
    /// Roll backward from maturity; a short first period. The default.
    #[default]
    ShortFront,
    /// Roll backward from maturity; a short first period is merged into
    /// the next, giving a long first period.
    LongFront,
    /// Roll forward from the effective date; a short last period.
    ShortBack,
    /// Roll forward from the effective date; a short last period is
    /// merged into the previous, giving a long last period.
    LongBack,
}

impl StubConvention {
    fn rolls_backward(self) -> bool {
        matches!(self, StubConvention::ShortFront | StubConvention::LongFront)
    }
}

/// How anchor dates sit within their month.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RollConvention {
    /// The roll origin's day of month, clamped to shorter months
    /// (Aug 31 - 6M = Feb 28). The default.
    #[default]
    None,
    /// Every anchor on its month-end.
    EndOfMonth,
    /// Every anchor on this day of month, clamped to shorter months.
    DayOfMonth(u32),
    /// Every anchor on its month's IMM date (third Wednesday).
    Imm,
}

impl RollConvention {
    /// Snap an anchor to the convention.
    pub fn apply(self, date: NaiveDate) -> NaiveDate {
        match self {
            RollConvention::None => date,
            RollConvention::EndOfMonth => end_of_month(date),
            RollConvention::DayOfMonth(day) => {
                let last = end_of_month(date).day();
                date.with_day(day.clamp(1, last))
                    .expect("clamped day is valid")
            }
            RollConvention::Imm => imm_date(date.year(), date.month()),
        }
    }

    fn validate(self) -> Result<(), RustyQLibError> {
        if let RollConvention::DayOfMonth(day) = self {
            if !(1..=31).contains(&day) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("day of month must be 1..=31, got {day}"),
                ));
            }
        }
        Ok(())
    }
}

/// Last day of the month `date` sits in.
pub fn end_of_month(date: NaiveDate) -> NaiveDate {
    let first = date.with_day(1).expect("day 1 exists");
    (first + Months::new(1)).pred_opt().expect("valid date")
}

/// The unadjusted period boundaries of a leg, from `effective` to
/// `maturity` inclusive, every `frequency` under the stub and roll
/// conventions. Both end dates are kept as given; only the interior
/// anchors are rolled and snapped.
pub fn unadjusted_dates(
    effective: NaiveDate,
    maturity: NaiveDate,
    frequency: Frequency,
    stub: StubConvention,
    roll: RollConvention,
) -> Result<Vec<NaiveDate>, RustyQLibError> {
    if maturity <= effective {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("maturity {maturity} must be after the effective date {effective}"),
        ));
    }
    roll.validate()?;
    let months = frequency.months();
    if months == 0 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "the period must be at least one month",
        ));
    }

    let mut interior: Vec<NaiveDate> = Vec::new();
    let has_stub;
    if stub.rolls_backward() {
        // anchors back from maturity, strictly inside (effective, maturity)
        let mut k = 0u32;
        loop {
            k += months;
            let anchor = roll.apply(maturity - Months::new(k));
            if anchor <= effective {
                // a stub exists when the roll lands short of the effective
                // date rather than on it
                has_stub = anchor < effective;
                break;
            }
            interior.push(anchor);
        }
        interior.reverse();
        if has_stub && stub == StubConvention::LongFront && !interior.is_empty() {
            interior.remove(0);
        }
    } else {
        let mut k = 0u32;
        loop {
            k += months;
            let anchor = roll.apply(effective + Months::new(k));
            if anchor >= maturity {
                has_stub = anchor > maturity;
                break;
            }
            interior.push(anchor);
        }
        if has_stub && stub == StubConvention::LongBack && !interior.is_empty() {
            interior.pop();
        }
    }

    let mut dates = Vec::with_capacity(interior.len() + 2);
    dates.push(effective);
    dates.extend(interior);
    dates.push(maturity);
    dates.dedup();
    if dates.windows(2).any(|w| w[1] <= w[0]) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("the roll convention {roll:?} produced non-increasing anchors"),
        ));
    }
    Ok(dates)
}

/// The terms that define one leg's accrual periods.
#[derive(Debug, Clone)]
pub struct LegSchedule {
    pub effective: NaiveDate,
    pub maturity: NaiveDate,
    pub frequency: Frequency,
    pub stub: StubConvention,
    pub roll: RollConvention,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
    /// Business days from accrual end to payment.
    pub payment_lag: i64,
}

impl LegSchedule {
    /// A schedule with the default conventions: short front stub, the
    /// maturity's day of month, no payment lag.
    pub fn new(
        effective: NaiveDate,
        maturity: NaiveDate,
        frequency: Frequency,
        calendar: Calendar,
        convention: BusinessDayConvention,
    ) -> Self {
        LegSchedule {
            effective,
            maturity,
            frequency,
            stub: StubConvention::default(),
            roll: RollConvention::default(),
            calendar,
            convention,
            payment_lag: 0,
        }
    }

    pub fn with_stub(mut self, stub: StubConvention) -> Self {
        self.stub = stub;
        self
    }

    pub fn with_roll(mut self, roll: RollConvention) -> Self {
        self.roll = roll;
        self
    }

    pub fn with_payment_lag(mut self, payment_lag: i64) -> Self {
        self.payment_lag = payment_lag;
        self
    }

    /// The same terms at another frequency — a floating leg's reset
    /// schedule inside its payment schedule.
    pub fn at_frequency(&self, frequency: Frequency) -> Self {
        LegSchedule {
            frequency,
            calendar: self.calendar.clone(),
            ..*self
        }
    }

    /// The unadjusted period boundaries.
    pub fn unadjusted_dates(&self) -> Result<Vec<NaiveDate>, RustyQLibError> {
        unadjusted_dates(
            self.effective,
            self.maturity,
            self.frequency,
            self.stub,
            self.roll,
        )
    }

    /// The business-day-adjusted accrual periods, payments lagged.
    /// Anchors that collapse after adjustment are dropped.
    pub fn accrual_periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        if self.payment_lag < 0 {
            return Err(RustyQLibError::invalid_input(
                "payment_lag",
                format!("must be non-negative, got {}", self.payment_lag),
            ));
        }
        let dates = self.unadjusted_dates()?;
        let mut periods = Vec::with_capacity(dates.len());
        let mut start = self.calendar.adjust(dates[0], self.convention);
        for &anchor in &dates[1..] {
            let end = self.calendar.adjust(anchor, self.convention);
            if end <= start {
                continue;
            }
            periods.push(AccrualPeriod {
                start,
                end,
                payment: self.calendar.add_business_days(end, self.payment_lag),
            });
            start = end;
        }
        if periods.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "swap schedule",
                format!(
                    "no accrual periods between {} and {}",
                    self.effective, self.maturity
                ),
            ));
        }
        Ok(periods)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn dates(stub: StubConvention, roll: RollConvention) -> Vec<NaiveDate> {
        // 14 months quarterly: an odd two-month remainder somewhere
        unadjusted_dates(
            d(2026, 8, 20),
            d(2027, 10, 20),
            Frequency::Quarterly,
            stub,
            roll,
        )
        .unwrap()
    }

    #[test]
    fn front_stubs_roll_backward_short_or_merged() {
        let short = dates(StubConvention::ShortFront, RollConvention::None);
        assert_eq!(
            short,
            vec![
                d(2026, 8, 20),
                d(2026, 10, 20),
                d(2027, 1, 20),
                d(2027, 4, 20),
                d(2027, 7, 20),
                d(2027, 10, 20)
            ]
        );
        let long = dates(StubConvention::LongFront, RollConvention::None);
        assert_eq!(
            long,
            vec![
                d(2026, 8, 20),
                d(2027, 1, 20),
                d(2027, 4, 20),
                d(2027, 7, 20),
                d(2027, 10, 20)
            ]
        );
    }

    #[test]
    fn back_stubs_roll_forward_short_or_merged() {
        let short = dates(StubConvention::ShortBack, RollConvention::None);
        assert_eq!(
            short,
            vec![
                d(2026, 8, 20),
                d(2026, 11, 20),
                d(2027, 2, 20),
                d(2027, 5, 20),
                d(2027, 8, 20),
                d(2027, 10, 20)
            ]
        );
        let long = dates(StubConvention::LongBack, RollConvention::None);
        assert_eq!(
            long,
            vec![
                d(2026, 8, 20),
                d(2026, 11, 20),
                d(2027, 2, 20),
                d(2027, 5, 20),
                d(2027, 10, 20)
            ]
        );
    }

    #[test]
    fn a_regular_leg_has_no_stub_under_any_convention() {
        for stub in [
            StubConvention::ShortFront,
            StubConvention::LongFront,
            StubConvention::ShortBack,
            StubConvention::LongBack,
        ] {
            let dates = unadjusted_dates(
                d(2026, 8, 20),
                d(2028, 8, 20),
                Frequency::Semiannual,
                stub,
                RollConvention::None,
            )
            .unwrap();
            assert_eq!(dates.len(), 5, "{stub:?}");
            assert_eq!(dates[1], d(2027, 2, 20), "{stub:?}");
        }
    }

    #[test]
    fn roll_conventions_pin_the_anchor_day() {
        // month-end: Aug 31 maturity, semiannual — Feb anchors on the 28th
        // by clamping, on the 28th/29th too under EOM, but a Nov 30 origin
        // shows the difference: clamped roll gives May 30, EOM gives May 31
        let clamped = unadjusted_dates(
            d(2026, 5, 30),
            d(2027, 11, 30),
            Frequency::Semiannual,
            StubConvention::ShortFront,
            RollConvention::None,
        )
        .unwrap();
        assert_eq!(clamped[1], d(2026, 11, 30));
        assert_eq!(clamped[2], d(2027, 5, 30));
        let eom = unadjusted_dates(
            d(2026, 5, 31),
            d(2027, 11, 30),
            Frequency::Semiannual,
            StubConvention::ShortFront,
            RollConvention::EndOfMonth,
        )
        .unwrap();
        assert_eq!(eom[2], d(2027, 5, 31));
        // fixed day of month, clamped in February
        let dom = unadjusted_dates(
            d(2026, 8, 30),
            d(2027, 8, 30),
            Frequency::Quarterly,
            StubConvention::ShortFront,
            RollConvention::DayOfMonth(30),
        )
        .unwrap();
        assert_eq!(dom[2], d(2027, 2, 28));
        // IMM: quarterly from an IMM maturity, every anchor a third Wednesday
        let imm = unadjusted_dates(
            d(2026, 9, 16),
            d(2027, 9, 15),
            Frequency::Quarterly,
            StubConvention::ShortFront,
            RollConvention::Imm,
        )
        .unwrap();
        assert_eq!(
            imm,
            vec![
                d(2026, 9, 16),
                d(2026, 12, 16),
                d(2027, 3, 17),
                d(2027, 6, 16),
                d(2027, 9, 15)
            ]
        );
    }

    #[test]
    fn schedule_adjusts_and_lags_payments() {
        let s = LegSchedule::new(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
        .with_payment_lag(2);
        let periods = s.accrual_periods().unwrap();
        assert_eq!(periods.len(), 4);
        // Feb 6 2027 is a Saturday -> Mon Feb 8; payment T+2 -> Wed Feb 10
        assert_eq!(periods[1].end, d(2027, 2, 8));
        assert_eq!(periods[1].payment, d(2027, 2, 10));
        // the reset schedule inside a payment schedule shares the terms
        let monthly = s
            .at_frequency(Frequency::Monthly)
            .accrual_periods()
            .unwrap();
        assert_eq!(monthly.len(), 12);
        assert_eq!(monthly[2].end, periods[0].end);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(unadjusted_dates(
            d(2027, 1, 1),
            d(2026, 1, 1),
            Frequency::Annual,
            StubConvention::ShortFront,
            RollConvention::None
        )
        .is_err());
        assert!(unadjusted_dates(
            d(2026, 1, 1),
            d(2027, 1, 1),
            Frequency::Annual,
            StubConvention::ShortFront,
            RollConvention::DayOfMonth(0)
        )
        .is_err());
        let s = LegSchedule::new(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
        )
        .with_payment_lag(-1);
        assert!(s.accrual_periods().is_err());
    }
}
