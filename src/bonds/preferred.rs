//! Convertible preferred stock (CPS).
//!
//! A convertible preferred is the convertible bond's equity-capital
//! sibling: a fixed dividend on a liquidation preference instead of a
//! coupon on face, **perpetual** unless a mandatory redemption date is
//! set, convertible into common at a fixed ratio, and callable by the
//! issuer (typically to force conversion). It is a
//! [`ConvertibleInstrument`], so it prices on the same engines and
//! credit models as [`ConvertibleBond`](crate::bonds::ConvertibleBond)
//! — Tsiveriotis-Fernandes or jump to default, tree or finite
//! differences, with the greeks, implied solves and pluggable
//! volatility of [`ConvertiblePricing`](crate::bonds::ConvertiblePricing)
//! — mapping its own conventions
//! onto the shared event grid:
//!
//! - **the perpetuity tail**: with no maturity, the schedule truncates
//!   at [`PERPETUAL_HORIZON_YEARS`] and the terminal value is the
//!   remaining dividend perpetuity discounted at the curve's own long
//!   forward plus the credit rate (the same tail the analytic
//!   [`preferred_floor`](ConvertiblePreferred::preferred_floor) uses,
//!   so the busted limit matches it); a standing call still caps that
//!   continuing value at the horizon;
//! - **preferred dividends** are fixed periodic amounts
//!   (`rate * preference / frequency`), not day-count accruals, and a
//!   **non-cumulative** preferred trades flat (no accrued, none in the
//!   call strike); a cumulative one accrues linearly within the period;
//! - **calls are American**: unlike the discrete-date convertible-bond
//!   calls, each schedule entry applies at every step from its date
//!   onward (superseded by the next entry), matching how preferred
//!   call schedules actually work, and the soft-call trigger is
//!   smoothed across the one grid cell it falls in (a hard indicator
//!   makes a lattice price sawtooth in spot as the node ladder slides
//!   past the fixed trigger).
//!
//! Dividend deferral risk is folded into the credit input — the
//! standard practical treatment (a preferred's spread trades well wide
//! of the same issuer's senior debt for exactly this reason). Prices
//! are per preferred share.

use chrono::{Months, NaiveDate};

use crate::bonds::convertible::events::cash_dividends_at_steps;
use crate::bonds::convertible::instrument::validate_cash_dividends;
use crate::bonds::convertible::{CashDividend, ConvertibleInstrument, CreditModel, EventGrid};
use crate::bonds::schedule::coupon_dates;
use crate::bonds::CallOption;
use crate::core::calendar::Frequency;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// Tree truncation horizon for perpetual preferreds, in years; the
/// dividend stream beyond it is carried as an exact perpetuity tail.
pub const PERPETUAL_HORIZON_YEARS: u32 = 40;

/// A convertible preferred share.
#[derive(Debug, Clone)]
pub struct ConvertiblePreferred {
    /// Liquidation preference (par) per preferred share.
    pub preference: f64,
    /// Annual dividend rate on the preference (e.g. `0.055`).
    pub dividend_rate: f64,
    /// Dividend payment frequency (quarterly is standard).
    pub frequency: Frequency,
    /// Cumulative dividends accrue within the period (and into call
    /// strikes); non-cumulative preferreds trade flat.
    pub cumulative: bool,
    /// Dividend accrual start.
    pub dated_date: NaiveDate,
    /// Mandatory redemption at the preference on this date; `None` for
    /// a perpetual preferred.
    pub mandatory_redemption: Option<NaiveDate>,
    /// Common shares received per preferred share on conversion.
    pub conversion_ratio: f64,
    /// First date conversion is allowed (default: the dated date).
    pub convert_from: Option<NaiveDate>,
    /// Issuer call schedule: redeemable at `call_price` (a percentage
    /// of the preference) plus accrued at any time **from** `call_date`
    /// onward — preferreds are continuously callable once the first
    /// call date passes — with a later entry superseding an earlier one.
    pub calls: Vec<CallOption>,
    /// Calls exercisable only at or above this common-share price.
    pub soft_call_trigger: Option<f64>,
    /// Discrete cash dividends on the common share, on top of the
    /// market's continuous yield.
    pub cash_dividends: Vec<CashDividend>,
}

impl ConvertiblePreferred {
    pub fn new(
        preference: f64,
        dividend_rate: f64,
        dated_date: NaiveDate,
        conversion_ratio: f64,
    ) -> Result<Self, RustyQLibError> {
        if !(preference > 0.0 && preference.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("liquidation preference must be positive, got {preference}"),
            ));
        }
        if !dividend_rate.is_finite() || dividend_rate < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("dividend rate must be non-negative, got {dividend_rate}"),
            ));
        }
        if !(conversion_ratio > 0.0 && conversion_ratio.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("conversion ratio must be positive, got {conversion_ratio}"),
            ));
        }
        Ok(ConvertiblePreferred {
            preference,
            dividend_rate,
            frequency: Frequency::Quarterly,
            cumulative: true,
            dated_date,
            mandatory_redemption: None,
            conversion_ratio,
            convert_from: None,
            calls: Vec::new(),
            soft_call_trigger: None,
            cash_dividends: Vec::new(),
        })
    }

    /// Fixed dividend paid each period.
    pub fn periodic_dividend(&self) -> f64 {
        self.preference * self.dividend_rate / self.frequency.per_year() as f64
    }

    /// The common price at which conversion matches the preference.
    pub fn conversion_price(&self) -> f64 {
        self.preference / self.conversion_ratio
    }

    /// Conversion value per preferred share.
    pub fn parity(&self, spot: f64) -> f64 {
        self.conversion_ratio * spot
    }

    /// Premium of a price over parity, as a fraction.
    pub fn conversion_premium(&self, price: f64, spot: f64) -> Result<f64, RustyQLibError> {
        let parity = self.parity(spot);
        if parity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive parity {parity}"
            )));
        }
        Ok(price / parity - 1.0)
    }

    /// Current yield of a price: annual dividend over price.
    pub fn current_yield(&self, price: f64) -> Result<f64, RustyQLibError> {
        if !(price > 0.0 && price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("price must be positive, got {price}"),
            ));
        }
        Ok(self.preference * self.dividend_rate / price)
    }

    /// The truncation date of the dividend schedule: the mandatory
    /// redemption, or — for a perpetual — a whole number of dividend
    /// periods from the dated date spanning the horizon past
    /// `settlement`, so the truncated schedule stays on the dividend
    /// cycle.
    fn horizon_date(&self, settlement: NaiveDate) -> NaiveDate {
        if let Some(redemption) = self.mandatory_redemption {
            return redemption;
        }
        let months_per_period = self.frequency.months();
        let target = settlement + Months::new(12 * PERPETUAL_HORIZON_YEARS);
        let mut months = months_per_period;
        let mut date = self.dated_date + Months::new(months);
        while date < target {
            months += months_per_period;
            date = self.dated_date + Months::new(months);
        }
        date
    }

    /// Scheduled dividend dates strictly after `settlement`, up to and
    /// including the horizon.
    fn dividend_dates(&self, settlement: NaiveDate) -> Result<Vec<NaiveDate>, RustyQLibError> {
        let horizon = self.horizon_date(settlement);
        if horizon <= settlement {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("redemption {horizon} is not after settlement {settlement}"),
            ));
        }
        let schedule = coupon_dates(self.dated_date, horizon, self.frequency.months(), false)?;
        Ok(schedule
            .dates
            .iter()
            .copied()
            .filter(|&date| date > settlement)
            .collect())
    }

    /// Accrued dividend per share at `settlement` (zero for a
    /// non-cumulative preferred, which trades flat).
    pub fn accrued_dividend(&self, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        if !self.cumulative || settlement <= self.dated_date {
            return Ok(0.0);
        }
        let horizon = self.horizon_date(settlement);
        if settlement >= horizon {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("settlement {settlement} is at or after redemption {horizon}"),
            ));
        }
        let schedule = coupon_dates(self.dated_date, horizon, self.frequency.months(), false)?;
        let mut period_start = self.dated_date;
        for &date in &schedule.dates {
            if settlement < date {
                let elapsed = (settlement - period_start).num_days() as f64;
                let full = (date - period_start).num_days() as f64;
                return Ok(self.periodic_dividend() * elapsed / full.max(1.0));
            }
            period_start = date;
        }
        Ok(0.0)
    }

    /// The straight-preferred floor per share: the dividend stream (and
    /// mandatory redemption, or the perpetuity tail) under the market's
    /// credit model — the value ignoring conversion. Under
    /// Tsiveriotis-Fernandes that is the stream discounted at the curve
    /// plus the spread; under jump to default the survival-weighted
    /// stream plus the recovery on the preference.
    pub fn preferred_floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let day_count = curve.day_count();
        let year_fraction = |date: NaiveDate| day_count.year_fraction(curve.reference_date(), date);
        let t0 = year_fraction(settlement);
        let dividend = self.periodic_dividend();
        let mut flows: Vec<(f64, f64)> = self
            .dividend_dates(settlement)?
            .iter()
            .map(|&date| (year_fraction(date) - t0, dividend))
            .collect();
        let horizon = year_fraction(self.horizon_date(settlement));
        let terminal = if self.mandatory_redemption.is_some() {
            self.preference
        } else {
            self.perpetuity_tail(market.credit_rate(), curve, horizon)
        };
        flows.push((horizon - t0, terminal));
        Ok(market.value_of_flows(curve, settlement, &flows, self.preference))
    }

    /// Continuing value at the horizon of the perpetual dividend
    /// stream: a discrete perpetuity at the curve's one-year forward
    /// plus the credit rate.
    fn perpetuity_tail(&self, credit_rate: f64, curve: &YieldCurve, horizon: f64) -> f64 {
        let tail_yield = (curve.df(horizon) / curve.df(horizon + 1.0)).ln() + credit_rate;
        let per_period = (-tail_yield / self.frequency.per_year() as f64).exp();
        self.periodic_dividend() * per_period / (1.0 - per_period)
    }
}

impl ConvertibleInstrument for ConvertiblePreferred {
    fn conversion_ratio(&self) -> f64 {
        self.conversion_ratio
    }

    fn maturity_shares(&self, _spot: f64) -> f64 {
        self.conversion_ratio
    }

    fn is_mandatory(&self) -> bool {
        false
    }

    fn soft_call_trigger(&self) -> Option<f64> {
        self.soft_call_trigger
    }

    fn conversion_price(&self) -> f64 {
        ConvertiblePreferred::conversion_price(self)
    }

    fn accrued(&self, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        self.accrued_dividend(settlement)
    }

    fn final_payment_date(&self, settlement: NaiveDate) -> Result<NaiveDate, RustyQLibError> {
        let horizon = self.horizon_date(settlement);
        if horizon <= settlement {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("redemption {horizon} is not after settlement {settlement}"),
            ));
        }
        Ok(horizon)
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        validate_cash_dividends(&self.cash_dividends)?;
        if let Some(trigger) = self.soft_call_trigger {
            if !(trigger > 0.0 && trigger.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "preferred",
                    format!("soft call trigger must be positive, got {trigger}"),
                ));
            }
        }
        for call in &self.calls {
            if !(call.call_price > 0.0 && call.call_price.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "preferred",
                    format!("call price must be positive, got {}", call.call_price),
                ));
            }
        }
        Ok(())
    }

    /// The preferred's schedule on the shared grid: dividends at stake
    /// in their step, a standing (American) call from each schedule
    /// entry with the dividend-cycle accrued in its strike, the
    /// mandatory redemption or the perpetuity tail as the terminal
    /// value (with the standing call applied there too), the
    /// preference as the recovery claim, and a smoothed soft trigger.
    fn event_grid(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
        credit_rate: f64,
    ) -> Result<EventGrid, RustyQLibError> {
        let year_fraction = |date: NaiveDate| {
            curve
                .day_count()
                .year_fraction(curve.reference_date(), date)
        };
        let t0 = year_fraction(settlement);
        let horizon_date = self.horizon_date(settlement);
        let horizon = year_fraction(horizon_date);
        if horizon <= t0 {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                "the horizon is not after settlement",
            ));
        }
        let dividend_dates = self.dividend_dates(settlement)?;
        let dt = (horizon - t0) / steps as f64;
        let times: Vec<f64> = (0..=steps).map(|i| t0 + i as f64 * dt).collect();
        let riskfree_df: Vec<f64> = (0..steps)
            .map(|i| curve.df(times[i + 1]) / curve.df(times[i]))
            .collect();

        // dividends assigned to their step interval, discounted to its
        // edge; all at stake, since a standing call sits at every step
        let dividend = self.periodic_dividend();
        let mut coupon_at_step = vec![0.0_f64; steps];
        for &date in &dividend_dates {
            let time = year_fraction(date).clamp(t0, horizon);
            let index = (((time - t0) / dt).ceil() as usize).clamp(1, steps) - 1;
            let forward = -(riskfree_df[index].ln()) / dt;
            coupon_at_step[index] +=
                dividend * (-(forward + credit_rate) * (time - times[index])).exp();
        }

        // the recovery claim per step: the preference, paid on the
        // dividend date that ends the period default is observed in
        let dividend_times: Vec<f64> = dividend_dates
            .iter()
            .map(|&date| year_fraction(date))
            .collect();
        let default_claim_at_step: Vec<f64> = times[1..]
            .iter()
            .map(|&time| {
                let paid = dividend_times
                    .iter()
                    .copied()
                    .find(|&paid| paid >= time - 1e-9)
                    .unwrap_or(horizon);
                self.preference * curve.df(paid) / curve.df(time)
            })
            .collect();

        // calls apply from their date onward — a preferred is
        // continuously callable once its first call date passes — with
        // a later schedule entry superseding the earlier one; the strike
        // carries the accrued of the step's position in the dividend
        // cycle
        let mut call_at_step: Vec<Option<f64>> = vec![None; steps + 1];
        if !self.calls.is_empty() {
            let mut schedule: Vec<(f64, f64)> = self
                .calls
                .iter()
                .filter(|call| call.call_date < horizon_date)
                .map(|call| {
                    (
                        year_fraction(call.call_date),
                        self.preference * call.call_price / 100.0,
                    )
                })
                .collect();
            schedule.sort_by(|a, b| a.0.total_cmp(&b.0));
            let period = 1.0 / self.frequency.per_year() as f64;
            let accrued_at = |t: f64| -> f64 {
                if !self.cumulative {
                    return 0.0;
                }
                match dividend_times
                    .iter()
                    .copied()
                    .find(|&paid| paid > t + 1e-12)
                {
                    Some(next) => dividend * (1.0 - ((next - t) / period).clamp(0.0, 1.0)),
                    None => 0.0,
                }
            };
            for (step, &t) in times.iter().enumerate() {
                let redemption = schedule
                    .iter()
                    .rev()
                    .find(|(from, _)| *from <= t + 1e-9)
                    .map(|(_, k)| *k);
                if let Some(k) = redemption {
                    call_at_step[step] = Some(k + accrued_at(t));
                }
            }
        }

        // terminal: mandatory redemption at the preference, or the
        // perpetuity continuing value of the remaining dividends
        let redemption = if self.mandatory_redemption.is_some() {
            self.preference
        } else {
            self.perpetuity_tail(credit_rate, curve, horizon)
        };

        let cash_dividend_at_step =
            cash_dividends_at_steps(&self.cash_dividends, &times, year_fraction);
        Ok(EventGrid {
            dt,
            times,
            riskfree_df,
            coupon_at_step,
            coupon_kept_at_step: vec![0.0; steps],
            default_claim_at_step,
            call_at_step,
            put_at_step: vec![None; steps + 1],
            make_whole_at_step: vec![0.0; steps + 1],
            conversion_trigger_at_step: vec![None; steps + 1],
            event_probability_at_step: vec![0.0; steps + 1],
            par_put_at_step: vec![0.0; steps + 1],
            additional_shares_at_step: vec![Vec::new(); steps + 1],
            additional_share_prices: Vec::new(),
            convert_from: year_fraction(self.convert_from.unwrap_or(self.dated_date)),
            convert_until: horizon,
            redemption,
            final_principal: redemption,
            // prices are per share
            outstanding: 100.0,
            smooth_trigger: true,
            exercise_at_horizon: true,
            cash_dividend_at_step,
        })
    }

    fn floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.preferred_floor(market, curve, settlement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::convertible::{
        ConvertibleFdGrid, ConvertibleMarket, ConvertiblePricing, JumpToDefaultMarket,
    };
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// $100 preference, 5.5% quarterly, 1.6 common shares per preferred
    /// (conversion price 62.50), perpetual.
    fn preferred() -> ConvertiblePreferred {
        ConvertiblePreferred::new(100.0, 0.055, d(2026, 5, 15), 1.6).unwrap()
    }

    fn market(spot: f64) -> ConvertibleMarket {
        ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.03,
        }
    }

    fn jtd_market(spot: f64) -> JumpToDefaultMarket {
        JumpToDefaultMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            borrow_cost: 0.0,
            hazard_rate: 0.03,
            recovery_rate: 0.10,
        }
    }

    #[test]
    fn busted_preferred_collapses_to_the_analytic_floor() {
        // near-worthless common: the cash part is deterministic, so the
        // tree must reproduce the perpetuity-tailed floor
        let cps = preferred();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(0.01);
        let tree = cps.dirty_price(&m, &curve, settlement).unwrap();
        let floor = cps.preferred_floor(&m, &curve, settlement).unwrap();
        // the residual above the floor is the (tiny but real) conversion
        // option at spot 0.01, bounded by ratio * spot
        assert!(tree >= floor - 1e-9, "{tree} vs {floor}");
        assert!(
            tree - floor <= cps.conversion_ratio * m.spot,
            "{tree} vs {floor}"
        );
        // sanity: a 5.5% perpetuity at ~7% risky yield sits below par
        assert!(floor > 50.0 && floor < 100.0, "floor {floor}");
        // wider spread, lower floor
        let wide = ConvertibleMarket {
            credit_spread: 0.05,
            ..m
        };
        assert!(cps.preferred_floor(&wide, &curve, settlement).unwrap() < floor);
        // and under jump to default the survival-weighted floor with
        // the recovery on the preference
        let jm = jtd_market(0.01);
        let tree = cps.dirty_price(&jm, &curve, settlement).unwrap();
        let floor = cps.preferred_floor(&jm, &curve, settlement).unwrap();
        assert!((tree - floor).abs() < 0.05, "{tree} vs {floor}");
        let generous = JumpToDefaultMarket {
            recovery_rate: 0.5,
            ..jm
        };
        assert!(cps.preferred_floor(&generous, &curve, settlement).unwrap() > floor);
    }

    #[test]
    fn mandatory_redemption_floor_matches_a_bullet_stream() {
        // finite CPS: dividends plus preference at redemption, hand-
        // discounted at curve + spread
        let mut cps = preferred();
        cps.mandatory_redemption = Some(d(2031, 5, 15));
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(0.01);
        let floor = cps.preferred_floor(&m, &curve, settlement).unwrap();
        let dc = curve.day_count();
        let t0 = dc.year_fraction(curve.reference_date(), settlement);
        let df = |date: NaiveDate| {
            let t = dc.year_fraction(curve.reference_date(), date);
            curve.df(t) / curve.df(t0) * (-m.credit_spread * (t - t0)).exp()
        };
        let mut manual = 0.0;
        for &date in &cps.dividend_dates(settlement).unwrap() {
            manual += cps.periodic_dividend() * df(date);
        }
        manual += 100.0 * df(d(2031, 5, 15));
        assert!((floor - manual).abs() < 1e-9, "{floor} vs {manual}");
        // and the tree agrees in the busted limit
        let tree = cps.dirty_price(&m, &curve, settlement).unwrap();
        assert!((tree - floor).abs() < 1e-6, "{tree} vs {floor}");
    }

    #[test]
    fn deep_in_the_money_trades_at_parity_with_full_delta() {
        let cps = preferred();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let spot = 300.0; // parity 480 vs floor ~80
        let price = cps.dirty_price(&market(spot), &curve, settlement).unwrap();
        let parity = cps.parity(spot);
        assert!(price >= parity - 1e-9, "conversion floor violated");
        // a perpetual carries a real premium even deep in the money: the
        // preferred dividend exceeds the forgone common dividend, so the
        // holder rationally delays conversion — but the premium is small
        assert!(
            (price - parity) / parity < 0.05,
            "{price} vs parity {parity}"
        );
        let delta = cps.delta(&market(spot), &curve, settlement).unwrap();
        assert!(
            (delta - cps.conversion_ratio).abs() < 0.05 * cps.conversion_ratio,
            "delta {delta}"
        );
    }

    #[test]
    fn price_sits_above_floor_and_parity_and_orders_in_vol_and_spread() {
        let cps = preferred();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        for spot in [30.0, 50.0, 62.5, 80.0] {
            let m = market(spot);
            let price = cps.dirty_price(&m, &curve, settlement).unwrap();
            let floor = cps.preferred_floor(&m, &curve, settlement).unwrap();
            assert!(
                price >= floor - 0.05,
                "spot {spot}: {price} vs floor {floor}"
            );
            assert!(
                price >= cps.parity(spot) - 0.05,
                "spot {spot}: {price} vs parity"
            );
        }
        let base = cps.dirty_price(&market(60.0), &curve, settlement).unwrap();
        let hot = ConvertibleMarket {
            volatility: 0.45,
            ..market(60.0)
        };
        assert!(cps.dirty_price(&hot, &curve, settlement).unwrap() > base);
        let wide = ConvertibleMarket {
            credit_spread: 0.06,
            ..market(60.0)
        };
        assert!(cps.dirty_price(&wide, &curve, settlement).unwrap() < base);
    }

    #[test]
    fn forced_conversion_call_caps_the_price_and_soft_trigger_softens() {
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(60.0);
        let free = preferred();
        let unconstrained = free.dirty_price(&m, &curve, settlement).unwrap();

        let mut hard = free.clone();
        hard.calls = vec![CallOption {
            call_date: d(2029, 5, 15),
            call_price: 100.0,
        }];
        let called = hard.dirty_price(&m, &curve, settlement).unwrap();
        assert!(called < unconstrained, "{called} vs {unconstrained}");
        // when parity is above the strike the holder converts: the
        // called preferred still holds the conversion floor
        assert!(called >= free.parity(60.0) - 0.05);

        let mut soft = hard.clone();
        soft.soft_call_trigger = Some(81.25); // 130% of conversion price
        let softened = soft.dirty_price(&m, &curve, settlement).unwrap();
        assert!(
            called < softened && softened <= unconstrained + 1e-9,
            "{called} < {softened} <= {unconstrained}"
        );
    }

    #[test]
    fn cumulative_accrues_and_non_cumulative_trades_flat() {
        let cps = preferred();
        let settlement = d(2026, 8, 14);
        // May 15 -> Aug 14 is 91 of the 92 days to Aug 15
        let accrued = cps.accrued_dividend(settlement).unwrap();
        let expected = cps.periodic_dividend() * 91.0 / 92.0;
        assert!(
            (accrued - expected).abs() < 1e-12,
            "{accrued} vs {expected}"
        );
        let mut flat_pref = cps.clone();
        flat_pref.cumulative = false;
        assert_eq!(flat_pref.accrued_dividend(settlement).unwrap(), 0.0);
        let curve = flat(0.04);
        let m = market(60.0);
        // clean == dirty for the non-cumulative
        let dirty = flat_pref.dirty_price(&m, &curve, settlement).unwrap();
        let clean = flat_pref.clean_price(&m, &curve, settlement).unwrap();
        assert_eq!(dirty, clean);
    }

    #[test]
    fn implied_spread_round_trips_and_quote_helpers_work() {
        let cps = preferred();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(60.0);
        let clean = cps.clean_price(&m, &curve, settlement).unwrap();
        let implied = cps
            .implied_credit_spread(clean, &m, &curve, settlement)
            .unwrap();
        assert!(
            (implied - m.credit_spread).abs() < 1e-5,
            "implied {implied}"
        );
        let implied_vol = cps
            .implied_volatility(clean, &m, &curve, settlement)
            .unwrap();
        assert!(
            (implied_vol - m.volatility).abs() < 1e-5,
            "vol {implied_vol}"
        );
        assert!((cps.conversion_price() - 62.5).abs() < 1e-12);
        assert!((cps.parity(60.0) - 96.0).abs() < 1e-12);
        let premium = cps.conversion_premium(clean, 60.0).unwrap();
        assert!(premium > 0.0);
        let yield_now = cps.current_yield(clean).unwrap();
        assert!((yield_now - 5.5 / clean).abs() < 1e-12);
    }

    #[test]
    fn finite_differences_agree_with_the_tree() {
        // the shared grid engine on the preferred, under both credit
        // models, with a standing soft call in play
        let mut cps = preferred();
        cps.calls = vec![CallOption {
            call_date: d(2029, 5, 15),
            call_price: 101.0,
        }];
        cps.soft_call_trigger = Some(81.25);
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        // a 40-year perpetual with a standing call converges slowly on
        // both engines (each smooths the trigger over its own cell), so
        // the comparison uses refined grids and a loose tolerance
        let grid = ConvertibleFdGrid {
            time_steps: 1600,
            space_steps: 1600,
            grid_stdevs: 5.0,
        };
        for spot in [30.0, 62.5, 100.0] {
            let m = market(spot);
            let tree = cps
                .dirty_price_with_steps(&m, &curve, settlement, 1600)
                .unwrap();
            let fd = cps.fd_valuation(&m, &curve, settlement, grid).unwrap();
            assert!(
                (fd.dirty_price - tree).abs() < 0.6,
                "TF spot {spot}: fd {} vs tree {tree}",
                fd.dirty_price
            );
            assert!(fd.delta > 0.0 && fd.delta <= cps.conversion_ratio + 0.05);
            let jm = jtd_market(spot);
            let tree = cps
                .dirty_price_with_steps(&jm, &curve, settlement, 1600)
                .unwrap();
            let fd = cps.fd_valuation(&jm, &curve, settlement, grid).unwrap();
            assert!(
                (fd.dirty_price - tree).abs() < 0.6,
                "JTD spot {spot}: fd {} vs tree {tree}",
                fd.dirty_price
            );
        }
        // and the bump greeks run
        let greeks = cps
            .fd_greeks(&market(62.5), &curve, settlement, grid, &[5.0, 10.0, 30.0])
            .unwrap();
        assert!(greeks.vega > 0.0 && greeks.credit_dv01 > 0.0, "{greeks:?}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(ConvertiblePreferred::new(0.0, 0.055, d(2026, 5, 15), 1.6).is_err());
        assert!(ConvertiblePreferred::new(100.0, -0.01, d(2026, 5, 15), 1.6).is_err());
        assert!(ConvertiblePreferred::new(100.0, 0.055, d(2026, 5, 15), 0.0).is_err());
        let cps = preferred();
        let curve = flat(0.04);
        let bad = ConvertibleMarket {
            spot: -1.0,
            volatility: 0.3,
            dividend_yield: 0.0,
            credit_spread: 0.0,
        };
        assert!(cps.dirty_price(&bad, &curve, d(2026, 8, 14)).is_err());
        let m = market(60.0);
        assert!(cps
            .dirty_price_with_steps(&m, &curve, d(2026, 8, 14), 5)
            .is_err());
        let mut bad_trigger = cps.clone();
        bad_trigger.soft_call_trigger = Some(-1.0);
        assert!(bad_trigger.dirty_price(&m, &curve, d(2026, 8, 14)).is_err());
        // settlement past a mandatory redemption
        let mut finite = cps.clone();
        finite.mandatory_redemption = Some(d(2031, 5, 15));
        assert!(finite.dirty_price(&m, &curve, d(2032, 1, 1)).is_err());
    }
}
