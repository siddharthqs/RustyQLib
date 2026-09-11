//! CME SOFR futures: 1-month (SR1) and 3-month (SR3).
//!
//! Both settle on `100 - R` over a reference period, but they average
//! the daily SOFR fixings differently, and that difference is the whole
//! point of the pair:
//!
//! - **SR1** — the **arithmetic** average of daily SOFR over every
//!   calendar day of the contract month, $5,000,000 notional, one basis
//!   point worth $41.67 (`30/360`). Structurally identical to a fed
//!   funds future ([`FedFundsFuture`](crate::rates::FedFundsFuture)).
//! - **SR3** — the **compounded** daily SOFR over an IMM quarter (the
//!   third Wednesday of the contract month to the third Wednesday three
//!   months later), $1,000,000 notional, one basis point worth $25.00
//!   (`90/360`).
//!
//! Compounding follows the published convention: each business day's
//! rate accrues over the calendar days until the next business day, so a
//! Friday fixing earns three days. Projected on a curve the daily
//! factors telescope, so the compounded rate collapses to
//! `(df(start)/df(end) - 1) * 360/D` exactly.
//!
//! SR3 carries a genuine futures/forward convexity bias (daily margining
//! plus compounding); [`hull_convexity_adjustment`] gives the standard
//! Ho-Lee/Hull estimate to convert a futures rate into a forward rate.
//! It is never applied implicitly — the pricing here is the contract's
//! own definition.

use chrono::{Datelike, Duration, NaiveDate};

use crate::core::calendar::{imm_date, Calendar};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::overnight::{fixing_on_or_before, simple_forward, RateFixings};

/// SR1 contract size: $5,000,000.
pub const ONE_MONTH_NOTIONAL: f64 = 5_000_000.0;
/// SR3 contract size: $1,000,000.
pub const THREE_MONTH_NOTIONAL: f64 = 1_000_000.0;

/// Which SOFR contract, and therefore how the daily fixings are averaged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SofrContract {
    /// SR1: arithmetic average over the calendar month.
    OneMonth,
    /// SR3: daily compounding over an IMM quarter.
    ThreeMonth,
}

impl SofrContract {
    /// The `x/360` factor fixing the contract's basis-point value.
    fn valuation_days(self) -> f64 {
        match self {
            SofrContract::OneMonth => 30.0,
            SofrContract::ThreeMonth => 90.0,
        }
    }
}

/// A SOFR future over its reference period `[start, end)`.
#[derive(Debug, Clone)]
pub struct SofrFuture {
    pub contract: SofrContract,
    /// First day of the reference period (inclusive).
    pub start: NaiveDate,
    /// First day after the reference period (exclusive).
    pub end: NaiveDate,
    pub notional: f64,
    /// Publication calendar for the daily fixings.
    pub calendar: Calendar,
}

impl SofrFuture {
    /// The SR1 contract for `year`/`month`.
    pub fn one_month(year: i32, month: u32) -> Result<Self, RustyQLibError> {
        let start = NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
            RustyQLibError::invalid_input(
                "sofr future",
                format!("invalid contract month {year}-{month}"),
            )
        })?;
        let end = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1)
        }
        .expect("valid month start");
        Ok(SofrFuture {
            contract: SofrContract::OneMonth,
            start,
            end,
            notional: ONE_MONTH_NOTIONAL,
            calendar: Calendar::UsGovernmentBond,
        })
    }

    /// The SR3 contract for `year`/`month`: the IMM quarter running from
    /// that month's third Wednesday to the third Wednesday three months
    /// later. Quarterly contracts use March, June, September and
    /// December, but any month is accepted (CME also lists serials).
    pub fn three_month(year: i32, month: u32) -> Result<Self, RustyQLibError> {
        if !(1..=12).contains(&month) {
            return Err(RustyQLibError::invalid_input(
                "sofr future",
                format!("invalid contract month {year}-{month}"),
            ));
        }
        let start = imm_date(year, month);
        let (end_year, end_month) = if month > 9 {
            (year + 1, month - 9)
        } else {
            (year, month + 3)
        };
        let end = imm_date(end_year, end_month);
        Ok(SofrFuture {
            contract: SofrContract::ThreeMonth,
            start,
            end,
            notional: THREE_MONTH_NOTIONAL,
            calendar: Calendar::UsGovernmentBond,
        })
    }

    /// The SR3 contract whose IMM quarter contains `date`.
    pub fn three_month_containing(date: NaiveDate) -> Result<Self, RustyQLibError> {
        // step back to the most recent quarterly IMM start
        let mut year = date.year();
        let mut month = ((date.month() - 1) / 3) * 3 + 3; // 3, 6, 9 or 12
        if imm_date(year, month) > date {
            if month == 3 {
                year -= 1;
                month = 12;
            } else {
                month -= 3;
            }
        }
        Self::three_month(year, month)
    }

    /// Calendar days in the reference period.
    pub fn reference_days(&self) -> i64 {
        (self.end - self.start).num_days()
    }

    /// Every calendar day of the reference period.
    pub fn calendar_days(&self) -> Vec<NaiveDate> {
        let mut days = Vec::with_capacity(self.reference_days() as usize);
        let mut day = self.start;
        while day < self.end {
            days.push(day);
            day += Duration::days(1);
        }
        days
    }

    /// Compounding segments `(segment_start, calendar_days)`: each
    /// business day's rate accrues until the next business day, with the
    /// last segment truncated at the period end.
    fn segments(&self) -> Vec<(NaiveDate, i64)> {
        let mut segments = Vec::new();
        let mut day = self.start;
        while day < self.end {
            let next = self.calendar.add_business_days(day, 1).min(self.end);
            segments.push((day, (next - day).num_days()));
            day = next;
        }
        segments
    }

    // ── Settlement rate ─────────────────────────────────────────────────

    /// Rate for the period given a per-day rate source: arithmetic for
    /// SR1, compounded for SR3.
    fn average(
        &self,
        mut daily: impl FnMut(NaiveDate, i64) -> Result<f64, RustyQLibError>,
    ) -> Result<f64, RustyQLibError> {
        let total_days = self.reference_days();
        if total_days <= 0 {
            return Err(RustyQLibError::invalid_input(
                "sofr future",
                format!("empty reference period {}..{}", self.start, self.end),
            ));
        }
        match self.contract {
            SofrContract::OneMonth => {
                let mut total = 0.0;
                for day in self.calendar_days() {
                    total += daily(day, 1)?;
                }
                Ok(total / total_days as f64)
            }
            SofrContract::ThreeMonth => {
                let mut factor = 1.0;
                for (day, days) in self.segments() {
                    factor *= 1.0 + daily(day, days)? * days as f64 / 360.0;
                }
                Ok((factor - 1.0) * 360.0 / total_days as f64)
            }
        }
    }

    /// Settlement rate from published fixings alone.
    pub fn settlement_rate(&self, fixings: &RateFixings) -> Result<f64, RustyQLibError> {
        self.average(|day, _| fixing_on_or_before(fixings, day))
    }

    /// Fair rate projected entirely off `curve`.
    pub fn fair_rate(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        self.average(|day, days| simple_forward(curve, day, days))
    }

    /// Fair rate for a partially realized contract: days before `asof`
    /// come from `fixings`, the rest are projected on `curve`. For SR3 a
    /// compounding segment is treated as realized when it starts before
    /// `asof`.
    pub fn fair_rate_with_fixings(
        &self,
        curve: &YieldCurve,
        fixings: &RateFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.average(|day, days| {
            if day < asof {
                fixing_on_or_before(fixings, day)
            } else {
                simple_forward(curve, day, days)
            }
        })
    }

    // ── Price / rate conversions ────────────────────────────────────────

    /// Futures price from a rate in decimal (`0.0433` → `95.67`).
    pub fn price_from_rate(rate: f64) -> f64 {
        100.0 - 100.0 * rate
    }

    /// Rate in decimal implied by a futures price.
    pub fn rate_from_price(price: f64) -> f64 {
        (100.0 - price) / 100.0
    }

    /// Fair price projected off `curve`.
    pub fn fair_price(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(Self::price_from_rate(self.fair_rate(curve)?))
    }

    /// Fair price for a partially realized contract.
    pub fn fair_price_with_fixings(
        &self,
        curve: &YieldCurve,
        fixings: &RateFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(Self::price_from_rate(
            self.fair_rate_with_fixings(curve, fixings, asof)?,
        ))
    }

    // ── Position analytics ──────────────────────────────────────────────

    /// Value of one basis point: `notional * 0.0001 * x/360` — $41.67 for
    /// SR1 (`30/360`), $25.00 for SR3 (`90/360`), fixed by the contract
    /// regardless of the reference period's actual length.
    pub fn dv01(&self) -> f64 {
        self.notional * 0.0001 * self.contract.valuation_days() / 360.0
    }

    /// P&L of a long position of `contracts` bought at `entry_price` and
    /// marked at `exit_price`.
    pub fn pnl(&self, entry_price: f64, exit_price: f64, contracts: f64) -> f64 {
        (exit_price - entry_price) * 100.0 * self.dv01() * contracts
    }

    /// The forward rate implied by a quoted futures rate, correcting the
    /// margining/compounding bias with [`hull_convexity_adjustment`].
    /// `sigma` is the annualized basis-point volatility of the short
    /// rate in decimal (e.g. `0.011` for 110bp), `day_count` measures the
    /// times to the period start and end.
    pub fn forward_rate_from_futures(
        &self,
        futures_rate: f64,
        sigma: f64,
        asof: NaiveDate,
        day_count: crate::core::daycount::DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        let t1 = day_count.year_fraction(asof, self.start);
        let t2 = day_count.year_fraction(asof, self.end);
        if t1 < 0.0 || t2 <= t1 {
            return Err(RustyQLibError::invalid_input(
                "convexity",
                format!("need 0 <= t1 < t2, got t1={t1}, t2={t2} from {asof}"),
            ));
        }
        Ok(futures_rate - hull_convexity_adjustment(sigma, t1, t2))
    }
}

/// Futures/forward convexity adjustment `0.5 * sigma^2 * t1 * t2` under
/// the Ho-Lee (normal short-rate) model — the standard Hull estimate.
/// Subtract it from a futures rate to get the forward rate; the futures
/// rate is always the higher of the two.
pub fn hull_convexity_adjustment(sigma: f64, t1: f64, t2: f64) -> f64 {
    0.5 * sigma * sigma * t1 * t2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;
    use crate::rates::FedFundsFuture;

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
    fn contract_periods_follow_the_calendar_and_imm_cycle() {
        let sr1 = SofrFuture::one_month(2026, 9).unwrap();
        assert_eq!((sr1.start, sr1.end), (d(2026, 9, 1), d(2026, 10, 1)));
        assert_eq!(sr1.reference_days(), 30);

        // Sep-26 IMM quarter: Wed Sep 16 to Wed Dec 16 = 91 days
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        assert_eq!((sr3.start, sr3.end), (d(2026, 9, 16), d(2026, 12, 16)));
        assert_eq!(sr3.reference_days(), 91);
        // December rolls into the next year
        let dec = SofrFuture::three_month(2026, 12).unwrap();
        assert_eq!((dec.start, dec.end), (d(2026, 12, 16), d(2027, 3, 17)));
        // the contract containing a date inside the quarter
        let containing = SofrFuture::three_month_containing(d(2026, 11, 2)).unwrap();
        assert_eq!(containing.start, d(2026, 9, 16));
        // a date just before an IMM start belongs to the previous quarter
        let before_imm = SofrFuture::three_month_containing(d(2026, 9, 15)).unwrap();
        assert_eq!(before_imm.start, d(2026, 6, 17));
        assert!(SofrFuture::one_month(2026, 13).is_err());
        assert!(SofrFuture::three_month(2026, 0).is_err());
    }

    #[test]
    fn sr1_matches_the_fed_funds_arithmetic_average() {
        // SR1 and ZQ average identically; only the underlying differs
        let sr1 = SofrFuture::one_month(2026, 9).unwrap();
        let zq = FedFundsFuture::for_month(2026, 9).unwrap();
        let curve = flat(0.0435, d(2026, 8, 6));
        assert!((sr1.fair_rate(&curve).unwrap() - zq.fair_rate(&curve).unwrap()).abs() < 1e-15);
        // and the same $41.67 basis-point value
        assert!((sr1.dv01() - zq.dv01()).abs() < 1e-12);
        assert!((sr1.dv01() - 41.666_666_666_666_664).abs() < 1e-9);
    }

    #[test]
    fn sr3_compounding_telescopes_to_the_discount_ratio() {
        // projected on a curve, the daily factors telescope exactly:
        // R = (df(start)/df(end) - 1) * 360/D
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        let curve = flat(0.0435, d(2026, 8, 6));
        let rate = sr3.fair_rate(&curve).unwrap();
        let ratio = curve.df_date(sr3.start) / curve.df_date(sr3.end);
        let expected = (ratio - 1.0) * 360.0 / sr3.reference_days() as f64;
        assert!((rate - expected).abs() < 1e-14, "{rate} vs {expected}");
        // compounding over the quarter lifts it above the overnight
        // money-market equivalent, by a couple of basis points
        let overnight_equivalent = 0.0435 * 360.0 / 365.0;
        let pickup = rate - overnight_equivalent;
        assert!(
            pickup > 0.0 && pickup < 0.0005,
            "compounding pickup {pickup}"
        );
    }

    #[test]
    fn compounding_exceeds_the_arithmetic_average_of_the_same_fixings() {
        // one flat rate everywhere: compounded > arithmetic by the
        // interest-on-interest term
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 1, 1), 0.05);
        let compounded = sr3.settlement_rate(&fixings).unwrap();
        assert!(compounded > 0.05, "compounded {compounded}");
        // over a quarter at 5% the pickup is a few basis points
        assert!(compounded - 0.05 < 0.001, "compounded {compounded}");
        // an SR1 over flat fixings returns the rate itself
        let sr1 = SofrFuture::one_month(2026, 9).unwrap();
        assert!((sr1.settlement_rate(&fixings).unwrap() - 0.05).abs() < 1e-15);
    }

    #[test]
    fn segments_cover_the_period_and_weight_weekends() {
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        let segments = sr3.segments();
        // segments tile the period exactly
        let total: i64 = segments.iter().map(|(_, days)| days).sum();
        assert_eq!(total, sr3.reference_days());
        assert_eq!(segments[0].0, sr3.start);
        // Friday segments carry three days over the weekend
        let friday = segments
            .iter()
            .find(|(day, _)| day.weekday() == chrono::Weekday::Fri)
            .expect("a Friday in the quarter");
        assert_eq!(friday.1, 3, "Friday segment {friday:?}");
        // Columbus Day 2026 (Mon Oct 12) is a SOFR holiday: the Friday
        // before it accrues four days
        let pre_holiday = segments
            .iter()
            .find(|(day, _)| *day == d(2026, 10, 9))
            .expect("Oct 9 segment");
        assert_eq!(pre_holiday.1, 4);
    }

    #[test]
    fn partial_realization_reproduces_the_curve_when_fixings_match_it() {
        for future in [
            SofrFuture::one_month(2026, 9).unwrap(),
            SofrFuture::three_month(2026, 9).unwrap(),
        ] {
            let curve = flat(0.0435, d(2026, 8, 6));
            let pure = future.fair_rate(&curve).unwrap();
            // fixings equal to the curve's own forwards, on whichever
            // grid the contract reads: per calendar day for the
            // arithmetic SR1, per compounding segment for SR3
            let mut matching = RateFixings::new();
            match future.contract {
                SofrContract::OneMonth => {
                    for day in future.calendar_days() {
                        matching.insert(day, simple_forward(&curve, day, 1).unwrap());
                    }
                }
                SofrContract::ThreeMonth => {
                    for (day, days) in future.segments() {
                        matching.insert(day, simple_forward(&curve, day, days).unwrap());
                    }
                }
            }
            let asof = future.start + Duration::days(20);
            let blended = future
                .fair_rate_with_fixings(&curve, &matching, asof)
                .unwrap();
            assert!(
                (blended - pure).abs() < 1e-12,
                "{:?}: {blended} vs {pure}",
                future.contract
            );
            // asof before the period start is the pure curve
            let early = future
                .fair_rate_with_fixings(&curve, &matching, future.start)
                .unwrap();
            assert!((early - pure).abs() < 1e-15);
        }
    }

    #[test]
    fn realized_fixings_above_the_curve_lift_the_rate() {
        let sr1 = SofrFuture::one_month(2026, 9).unwrap();
        let curve = flat(0.0435, d(2026, 8, 6));
        let pure = sr1.fair_rate(&curve).unwrap();
        let mut rich = RateFixings::new();
        for day in sr1.calendar_days() {
            rich.insert(day, simple_forward(&curve, day, 1).unwrap() + 0.01);
        }
        // 10 of 30 days realized 100bp high -> +100bp * 10/30
        let lifted = sr1
            .fair_rate_with_fixings(&curve, &rich, d(2026, 9, 11))
            .unwrap();
        assert!((lifted - pure - 0.01 * 10.0 / 30.0).abs() < 1e-15);
    }

    #[test]
    fn sr3_basis_point_value_is_twenty_five_dollars() {
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        assert!((sr3.dv01() - 25.0).abs() < 1e-12);
        // long 10 lots, price up 1bp
        assert!((sr3.pnl(95.65, 95.66, 10.0) - 250.0).abs() < 1e-9);
        assert!((sr3.pnl(95.65, 95.66, -10.0) + 250.0).abs() < 1e-9);
        // every quarterly contract carries the same fixed 90/360 bpv,
        // whatever its reference period actually spans
        for (year, month) in [(2026, 12), (2027, 3), (2027, 6), (2028, 9)] {
            let quarterly = SofrFuture::three_month(year, month).unwrap();
            assert!((quarterly.dv01() - 25.0).abs() < 1e-12, "{year}-{month}");
        }
        // SR1 values a basis point at $41.67 against SR3's $25.00
        let sr1 = SofrFuture::one_month(2026, 9).unwrap();
        assert!(sr1.dv01() > sr3.dv01());
    }

    #[test]
    fn convexity_adjustment_pushes_the_forward_below_the_futures_rate() {
        let sr3 = SofrFuture::three_month(2027, 9).unwrap();
        let asof = d(2026, 8, 6);
        let futures_rate = 0.0435;
        let forward = sr3
            .forward_rate_from_futures(futures_rate, 0.011, asof, DayCountConvention::Act365)
            .unwrap();
        assert!(forward < futures_rate, "forward {forward}");
        // ~1y out at 110bp vol the bias is a couple of basis points
        let bias = futures_rate - forward;
        assert!(bias > 0.0 && bias < 0.001, "bias {bias}");
        // the raw formula is symmetric in t1, t2 and grows with vol
        assert!(
            hull_convexity_adjustment(0.02, 1.0, 1.25) > hull_convexity_adjustment(0.01, 1.0, 1.25)
        );
        assert_eq!(hull_convexity_adjustment(0.011, 0.0, 1.0), 0.0);
        // a period already in the past cannot be adjusted
        assert!(sr3
            .forward_rate_from_futures(
                futures_rate,
                0.011,
                d(2030, 1, 1),
                DayCountConvention::Act365
            )
            .is_err());
    }

    #[test]
    fn missing_fixings_are_reported() {
        let sr3 = SofrFuture::three_month(2026, 9).unwrap();
        let mut late = RateFixings::new();
        late.insert(d(2026, 11, 1), 0.043);
        assert!(sr3.settlement_rate(&late).is_err());
    }
}
