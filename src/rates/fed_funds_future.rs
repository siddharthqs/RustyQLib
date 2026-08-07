//! CME 30-day federal funds futures (ZQ).
//!
//! The contract settles on `100 - R`, where `R` is the **arithmetic
//! average of the daily effective federal funds rate over every calendar
//! day of the contract month** (in percent). Days without a published
//! fixing — weekends and holidays — carry the previous business day's
//! rate forward, which is what the exchange's averaging does.
//!
//! Because the average is arithmetic rather than compounded, the whole
//! product is linear in the daily rates. That makes three things easy
//! and exact:
//!
//! - **Partially realized contracts**: past days come from published
//!   fixings, the rest from a projection curve
//!   ([`fair_rate_with_fixings`](FedFundsFuture::fair_rate_with_fixings)).
//! - **A fixed DV01**: the contract values `notional * 30/360`, so one
//!   basis point is worth exactly `$41.67` on the standard $5mm size,
//!   whatever the month length.
//! - **FOMC step analytics**: with one meeting in the month the average
//!   splits into two flat pieces, so a quoted price implies the
//!   post-meeting rate — and, against an assumed move size, the
//!   market-implied probability of that move.
//!
//! Futures margining makes the futures rate differ slightly from the
//! forward rate; that convexity bias is second-order for a one-month
//! contract and is not modelled here.

use chrono::{Datelike, Duration, NaiveDate};

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::overnight::fixing_on_or_before;

// The overnight machinery is shared with SOFR futures; re-exported here
// so `rates::fed_funds_future::{overnight_forward, RateFixings}` keeps
// working.
pub use crate::rates::overnight::{overnight_forward, RateFixings};

/// Standard ZQ contract size: $5,000,000.
pub const CONTRACT_SIZE: f64 = 5_000_000.0;

/// A 30-day federal funds future on one contract month.
#[derive(Debug, Clone)]
pub struct FedFundsFuture {
    /// First calendar day of the contract month.
    pub month_start: NaiveDate,
    /// Contract notional (default [`CONTRACT_SIZE`]).
    pub notional: f64,
}

impl FedFundsFuture {
    /// The contract for `year`/`month` at the standard $5mm size.
    pub fn for_month(year: i32, month: u32) -> Result<Self, RustyQLibError> {
        let month_start = NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
            RustyQLibError::invalid_input(
                "fed funds future",
                format!("invalid contract month {year}-{month}"),
            )
        })?;
        Ok(FedFundsFuture {
            month_start,
            notional: CONTRACT_SIZE,
        })
    }

    /// The contract covering `date`'s month.
    pub fn containing(date: NaiveDate) -> Result<Self, RustyQLibError> {
        Self::for_month(date.year(), date.month())
    }

    /// Last calendar day of the contract month.
    pub fn month_end(&self) -> NaiveDate {
        let next_month = if self.month_start.month() == 12 {
            NaiveDate::from_ymd_opt(self.month_start.year() + 1, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(self.month_start.year(), self.month_start.month() + 1, 1)
        }
        .expect("valid month start");
        next_month - Duration::days(1)
    }

    /// Every calendar day of the contract month — the averaging set.
    pub fn calendar_days(&self) -> Vec<NaiveDate> {
        let end = self.month_end();
        let mut days = Vec::with_capacity(31);
        let mut day = self.month_start;
        while day <= end {
            days.push(day);
            day += Duration::days(1);
        }
        days
    }

    /// Calendar days in the contract month.
    pub fn days_in_month(&self) -> usize {
        (self.month_end() - self.month_start).num_days() as usize + 1
    }

    // ── Settlement rate ─────────────────────────────────────────────────

    /// Settlement rate from published fixings alone (a fully realized
    /// contract): the arithmetic average over every calendar day, with
    /// gaps carried forward from the previous available fixing.
    pub fn settlement_rate(&self, fixings: &RateFixings) -> Result<f64, RustyQLibError> {
        let days = self.calendar_days();
        let mut total = 0.0;
        for day in &days {
            total += fixing_on_or_before(fixings, *day)?;
        }
        Ok(total / days.len() as f64)
    }

    /// Fair rate from a projection curve alone (a contract whose month
    /// has not started): the average of the daily Act/360 overnight
    /// forwards.
    pub fn fair_rate(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        let days = self.calendar_days();
        let mut total = 0.0;
        for day in &days {
            total += overnight_forward(curve, *day)?;
        }
        Ok(total / days.len() as f64)
    }

    /// Fair rate for a partially realized contract: calendar days
    /// strictly before `asof` come from `fixings` (carried forward over
    /// weekends and holidays), days from `asof` onward are projected on
    /// `curve`.
    pub fn fair_rate_with_fixings(
        &self,
        curve: &YieldCurve,
        fixings: &RateFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let days = self.calendar_days();
        let mut total = 0.0;
        for day in &days {
            total += if *day < asof {
                fixing_on_or_before(fixings, *day)?
            } else {
                overnight_forward(curve, *day)?
            };
        }
        Ok(total / days.len() as f64)
    }

    // ── Price / rate conversions ────────────────────────────────────────

    /// Futures price from an average rate in decimal (`0.0433` → `95.67`).
    pub fn price_from_rate(rate: f64) -> f64 {
        100.0 - 100.0 * rate
    }

    /// Average rate in decimal implied by a futures price.
    pub fn rate_from_price(price: f64) -> f64 {
        (100.0 - price) / 100.0
    }

    /// Fair price off a projection curve.
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

    /// Round a price to the nearest one-tenth of a basis point (`0.001`
    /// price points) — the conventional final-settlement rounding.
    pub fn round_price(price: f64) -> f64 {
        (price * 1000.0).round() / 1000.0
    }

    // ── Position analytics ──────────────────────────────────────────────

    /// Value of one basis point: `notional * 0.0001 * 30/360`, i.e.
    /// $41.67 on the standard contract. The 30/360 factor is fixed by
    /// the contract, independent of the month's actual length.
    pub fn dv01(&self) -> f64 {
        self.notional * 0.0001 * 30.0 / 360.0
    }

    /// P&L of a long position of `contracts` bought at `entry_price` and
    /// marked at `exit_price`. Long futures gain when the price rises,
    /// i.e. when the average rate falls.
    pub fn pnl(&self, entry_price: f64, exit_price: f64, contracts: f64) -> f64 {
        // price points -> basis points of rate: 1 price point = 100bp
        (exit_price - entry_price) * 100.0 * self.dv01() * contracts
    }

    // ── FOMC step analytics ─────────────────────────────────────────────

    /// Split the month at `effective_date` — the first day the new
    /// target applies, i.e. the day after an FOMC decision. Returns
    /// `(days_before, days_from)`, both positive.
    pub fn split_at(&self, effective_date: NaiveDate) -> Result<(usize, usize), RustyQLibError> {
        let (start, end) = (self.month_start, self.month_end());
        if effective_date <= start || effective_date > end {
            return Err(RustyQLibError::invalid_input(
                "fed funds future",
                format!(
                    "effective date {effective_date} must fall strictly inside the \
                     contract month {start}..={end}"
                ),
            ));
        }
        let before = (effective_date - start).num_days() as usize;
        Ok((before, self.days_in_month() - before))
    }

    /// The post-change rate implied by a settlement rate, given the rate
    /// prevailing before `effective_date`:
    /// `avg = (n1 * r_pre + n2 * r_post) / D`.
    pub fn implied_rate_after(
        &self,
        settlement_rate: f64,
        pre_rate: f64,
        effective_date: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (before, from) = self.split_at(effective_date)?;
        let days = self.days_in_month() as f64;
        Ok((settlement_rate * days - pre_rate * before as f64) / from as f64)
    }

    /// The post-change rate implied by a quoted futures price.
    pub fn implied_rate_after_from_price(
        &self,
        price: f64,
        pre_rate: f64,
        effective_date: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.implied_rate_after(Self::rate_from_price(price), pre_rate, effective_date)
    }

    /// Market-implied probability of a move of `move_size` (signed, e.g.
    /// `0.0025` for a 25bp hike, `-0.0025` for a cut) at the meeting
    /// effective on `effective_date`, from a quoted price.
    ///
    /// This is the usual two-outcome reading — move or no move — so the
    /// result is the implied post-meeting rate expressed as a fraction
    /// of the full move. Values outside `[0, 1]` mean the market prices
    /// something other than the assumed pair of outcomes (a larger move,
    /// or a move the other way); they are returned as computed rather
    /// than clamped.
    pub fn implied_move_probability(
        &self,
        price: f64,
        pre_rate: f64,
        move_size: f64,
        effective_date: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !move_size.is_finite() || move_size == 0.0 {
            return Err(RustyQLibError::invalid_input(
                "move_size",
                format!("must be a non-zero finite rate move, got {move_size}"),
            ));
        }
        let post = self.implied_rate_after_from_price(price, pre_rate, effective_date)?;
        Ok((post - pre_rate) / move_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

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
    fn contract_month_geometry() {
        let sep = FedFundsFuture::for_month(2026, 9).unwrap();
        assert_eq!(sep.month_start, d(2026, 9, 1));
        assert_eq!(sep.month_end(), d(2026, 9, 30));
        assert_eq!(sep.days_in_month(), 30);
        assert_eq!(sep.calendar_days().len(), 30);
        // December rolls the year
        let dec = FedFundsFuture::for_month(2026, 12).unwrap();
        assert_eq!(dec.month_end(), d(2026, 12, 31));
        // February in a leap year
        let feb = FedFundsFuture::for_month(2028, 2).unwrap();
        assert_eq!(feb.days_in_month(), 29);
        assert_eq!(
            FedFundsFuture::containing(d(2026, 9, 17))
                .unwrap()
                .month_start,
            d(2026, 9, 1)
        );
        assert!(FedFundsFuture::for_month(2026, 13).is_err());
    }

    #[test]
    fn flat_curve_gives_the_money_market_equivalent_rate() {
        // a 4% Act/365 continuous curve quotes ~4% * 360/365 on Act/360
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        let curve = flat(0.04, d(2026, 8, 6));
        let rate = future.fair_rate(&curve).unwrap();
        let expected = 0.04 * 360.0 / 365.0;
        assert!((rate - expected).abs() < 1e-5, "{rate} vs {expected}");
        // price is 100 - 100 * rate, and round trips
        let price = future.fair_price(&curve).unwrap();
        assert!((price - (100.0 - 100.0 * rate)).abs() < 1e-12);
        assert!((FedFundsFuture::rate_from_price(price) - rate).abs() < 1e-14);
    }

    #[test]
    fn settlement_rate_is_the_plain_arithmetic_average() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        // 4.33% for the first 10 days, 4.08% for the remaining 20
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 9, 1), 0.0433);
        fixings.insert(d(2026, 9, 11), 0.0408);
        let rate = future.settlement_rate(&fixings).unwrap();
        let expected = (10.0 * 0.0433 + 20.0 * 0.0408) / 30.0;
        assert!((rate - expected).abs() < 1e-15, "{rate} vs {expected}");
    }

    #[test]
    fn weekend_gaps_carry_the_previous_business_day_forward() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        // one fixing for the whole month: every day inherits it
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 8, 31), 0.0425);
        assert!((future.settlement_rate(&fixings).unwrap() - 0.0425).abs() < 1e-15);
        // a Friday fixing covers Saturday and Sunday
        let mut sparse = RateFixings::new();
        sparse.insert(d(2026, 9, 1), 0.04);
        sparse.insert(d(2026, 9, 4), 0.05); // Friday
        sparse.insert(d(2026, 9, 7), 0.04); // Monday
                                            // days 4,5,6 all take 5% -> 3 days at 5%, 27 at 4%
        let expected = (3.0 * 0.05 + 27.0 * 0.04) / 30.0;
        assert!((future.settlement_rate(&sparse).unwrap() - expected).abs() < 1e-15);
        // a month starting before the first fixing is an error
        let mut late = RateFixings::new();
        late.insert(d(2026, 9, 15), 0.04);
        assert!(future.settlement_rate(&late).is_err());
    }

    #[test]
    fn partial_realization_blends_fixings_and_forwards() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        let curve = flat(0.04, d(2026, 8, 6));
        let asof = d(2026, 9, 11); // 10 days realized, 20 projected
        let forward = future.fair_rate(&curve).unwrap();

        // fixings equal to the curve's own forwards must reproduce the
        // pure-curve rate exactly
        let mut matching = RateFixings::new();
        for day in future.calendar_days() {
            matching.insert(day, overnight_forward(&curve, day).unwrap());
        }
        let blended = future
            .fair_rate_with_fixings(&curve, &matching, asof)
            .unwrap();
        assert!((blended - forward).abs() < 1e-15, "{blended} vs {forward}");

        // realized 100bp above the forwards moves the average by
        // exactly 10/30 of that
        let mut rich = RateFixings::new();
        for day in future.calendar_days() {
            rich.insert(day, overnight_forward(&curve, day).unwrap() + 0.01);
        }
        let lifted = future.fair_rate_with_fixings(&curve, &rich, asof).unwrap();
        assert!((lifted - forward - 0.01 * 10.0 / 30.0).abs() < 1e-15);

        // asof before the month = pure curve; after it = pure fixings
        let pure_curve = future
            .fair_rate_with_fixings(&curve, &matching, d(2026, 8, 1))
            .unwrap();
        assert!((pure_curve - forward).abs() < 1e-15);
        let pure_fixings = future
            .fair_rate_with_fixings(&curve, &rich, d(2026, 10, 5))
            .unwrap();
        assert!((pure_fixings - future.settlement_rate(&rich).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn dv01_is_forty_one_sixty_seven_and_pnl_scales() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        assert!((future.dv01() - 41.666_666_666_666_664).abs() < 1e-9);
        // month length does not change it (February is shorter)
        let feb = FedFundsFuture::for_month(2026, 2).unwrap();
        assert!((feb.dv01() - future.dv01()).abs() < 1e-12);
        // long 10 contracts, price up 1bp (0.01 price points)
        let pnl = future.pnl(95.67, 95.68, 10.0);
        assert!((pnl - 10.0 * future.dv01()).abs() < 1e-9, "pnl {pnl}");
        // short position loses on the same move
        assert!((future.pnl(95.67, 95.68, -10.0) + 10.0 * future.dv01()).abs() < 1e-9);
    }

    #[test]
    fn fomc_step_recovers_the_post_meeting_rate_exactly() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        // hike effective Sep 17: 16 days at 4.33%, 14 days at 4.58%
        let effective = d(2026, 9, 17);
        let (before, from) = future.split_at(effective).unwrap();
        assert_eq!((before, from), (16, 14));
        let (pre, post) = (0.0433, 0.0458);
        let average = (before as f64 * pre + from as f64 * post) / 30.0;
        let recovered = future.implied_rate_after(average, pre, effective).unwrap();
        assert!((recovered - post).abs() < 1e-15, "{recovered} vs {post}");
        // and through the quoted price
        let price = FedFundsFuture::price_from_rate(average);
        let from_price = future
            .implied_rate_after_from_price(price, pre, effective)
            .unwrap();
        assert!((from_price - post).abs() < 1e-14);
    }

    #[test]
    fn implied_probability_reads_a_priced_hike() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        let effective = d(2026, 9, 17);
        let (before, from) = future.split_at(effective).unwrap();
        let pre = 0.0433;
        let hike = 0.0025;
        let priced = |probability: f64| {
            let post = pre + probability * hike;
            let average = (before as f64 * pre + from as f64 * post) / 30.0;
            FedFundsFuture::price_from_rate(average)
        };
        for expected in [0.0, 0.25, 0.5, 1.0] {
            let p = future
                .implied_move_probability(priced(expected), pre, hike, effective)
                .unwrap();
            assert!((p - expected).abs() < 1e-12, "{p} vs {expected}");
        }
        // a cut priced against an assumed hike reads negative — reported,
        // not clamped
        let cut = future
            .implied_move_probability(priced(-1.0), pre, hike, effective)
            .unwrap();
        assert!((cut + 1.0).abs() < 1e-12, "cut {cut}");
        assert!(future
            .implied_move_probability(priced(0.5), pre, 0.0, effective)
            .is_err());
    }

    #[test]
    fn split_rejects_meetings_outside_the_month() {
        let future = FedFundsFuture::for_month(2026, 9).unwrap();
        // the first day cannot split the month (nothing before it)
        assert!(future.split_at(d(2026, 9, 1)).is_err());
        assert!(future.split_at(d(2026, 10, 1)).is_err());
        assert!(future.split_at(d(2026, 8, 20)).is_err());
        // the last day is a valid one-day tail
        assert_eq!(future.split_at(d(2026, 9, 30)).unwrap(), (29, 1));
    }
}
