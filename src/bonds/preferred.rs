//! Convertible preferred stock (CPS) under the Tsiveriotis-Fernandes
//! split.
//!
//! A convertible preferred is the convertible bond's equity-capital
//! sibling: a fixed dividend on a liquidation preference instead of a
//! coupon on face, **perpetual** unless a mandatory redemption date is
//! set, convertible into common at a fixed ratio, and callable by the
//! issuer (typically to force conversion). It prices on the same
//! equity-tree split as [`ConvertibleBond`](crate::bonds::ConvertibleBond)
//! — the dividend/cash part discounted at risk-free plus the credit
//! spread, the conversion part risk-free — with two structural changes:
//!
//! - **the perpetuity tail**: with no maturity, the tree truncates at
//!   [`PERPETUAL_HORIZON_YEARS`] and the cash part carries a terminal
//!   continuing value — the remaining dividend perpetuity discounted at
//!   the curve's own long forward plus the spread (the same tail the
//!   analytic [`preferred_floor`](ConvertiblePreferred::preferred_floor)
//!   uses, so the busted limit matches it);
//! - **preferred dividends** are fixed periodic amounts
//!   (`rate * preference / frequency`), not day-count accruals, and a
//!   **non-cumulative** preferred trades flat (no accrued, none in the
//!   call strike); a cumulative one accrues within the period;
//! - **calls are American**: unlike the discrete-date convertible-bond
//!   calls, each schedule entry applies at every tree step from its
//!   date onward (superseded by the next entry), matching how preferred
//!   call schedules actually work.
//!
//! Dividend deferral risk is folded into the credit spread — the
//! standard practical treatment (a preferred's spread trades well wide
//! of the same issuer's senior debt for exactly this reason).

use chrono::{Months, NaiveDate};

use crate::bonds::convertible::{solve_implied_credit_spread, spot_bump_delta, ConvertibleMarket};
use crate::bonds::schedule::coupon_dates;
use crate::bonds::CallOption;
use crate::core::calendar::Frequency;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// Tree truncation horizon for perpetual preferreds, in years; the
/// dividend stream beyond it is carried as an exact perpetuity tail.
pub const PERPETUAL_HORIZON_YEARS: u32 = 40;

/// Default number of tree steps.
pub const DEFAULT_TREE_STEPS: usize = 800;

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
    /// mandatory redemption, or the perpetuity tail) discounted at the
    /// curve plus the credit spread — the value ignoring conversion.
    pub fn preferred_floor(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let year_fraction = |date: NaiveDate| {
            curve
                .day_count()
                .year_fraction(curve.reference_date(), date)
        };
        let t0 = year_fraction(settlement);
        let df_risky =
            |t: f64| curve.df(t) / curve.df(t0) * (-market.credit_spread * (t - t0)).exp();
        let dividend = self.periodic_dividend();
        let dates = self.dividend_dates(settlement)?;
        let mut value = 0.0;
        for &date in &dates {
            value += dividend * df_risky(year_fraction(date));
        }
        let horizon = year_fraction(self.horizon_date(settlement));
        if self.mandatory_redemption.is_some() {
            value += self.preference * df_risky(horizon);
        } else {
            value += self.perpetuity_tail(market, curve, horizon) * df_risky(horizon);
        }
        Ok(value)
    }

    /// Continuing value at the horizon of the perpetual dividend
    /// stream: a discrete perpetuity at the curve's one-year forward
    /// plus the credit spread.
    fn perpetuity_tail(&self, market: &ConvertibleMarket, curve: &YieldCurve, horizon: f64) -> f64 {
        let tail_yield = (curve.df(horizon) / curve.df(horizon + 1.0)).ln() + market.credit_spread;
        let per_period = (-tail_yield / self.frequency.per_year() as f64).exp();
        self.periodic_dividend() * per_period / (1.0 - per_period)
    }

    /// Price per preferred share (dividend-accrual inclusive) on a
    /// Tsiveriotis-Fernandes tree with `steps` time steps.
    pub fn dirty_price_with_steps(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
    ) -> Result<f64, RustyQLibError> {
        if steps < 10 {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("the tree needs at least 10 steps, got {steps}"),
            ));
        }
        if !(market.spot > 0.0
            && market.spot.is_finite()
            && market.volatility > 0.0
            && market.volatility.is_finite()
            && market.dividend_yield.is_finite()
            && market.credit_spread.is_finite())
        {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                "market inputs must be finite with positive spot and volatility",
            ));
        }
        if let Some(trigger) = self.soft_call_trigger {
            if !(trigger > 0.0 && trigger.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "preferred",
                    format!("soft call trigger must be positive, got {trigger}"),
                ));
            }
        }
        tf_tree_value(self, market, curve, settlement, steps)
    }

    /// Price per preferred share with [`DEFAULT_TREE_STEPS`].
    pub fn dirty_price(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.dirty_price_with_steps(market, curve, settlement, DEFAULT_TREE_STEPS)
    }

    /// Price net of the accrued dividend (equal to the dirty price for
    /// a non-cumulative preferred).
    pub fn clean_price(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.dirty_price(market, curve, settlement)? - self.accrued_dividend(settlement)?)
    }

    /// Equity delta per preferred share from a symmetric 1% spot bump.
    pub fn delta(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        spot_bump_delta(market, |m| self.dirty_price(m, curve, settlement))
    }

    /// The credit spread implied by a market price, holding the equity
    /// inputs fixed.
    pub fn implied_credit_spread(
        &self,
        dirty_price: f64,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !(dirty_price > 0.0 && dirty_price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "preferred",
                format!("price must be positive, got {dirty_price}"),
            ));
        }
        solve_implied_credit_spread(dirty_price, market, |m| {
            self.dirty_price(m, curve, settlement)
        })
    }
}

/// The Tsiveriotis-Fernandes backward induction for the preferred.
fn tf_tree_value(
    preferred: &ConvertiblePreferred,
    market: &ConvertibleMarket,
    curve: &YieldCurve,
    settlement: NaiveDate,
    steps: usize,
) -> Result<f64, RustyQLibError> {
    let year_fraction = |date: NaiveDate| {
        curve
            .day_count()
            .year_fraction(curve.reference_date(), date)
    };
    let t0 = year_fraction(settlement);
    let horizon_date = preferred.horizon_date(settlement);
    let horizon = year_fraction(horizon_date);
    if horizon <= t0 {
        return Err(RustyQLibError::invalid_input(
            "preferred",
            "the horizon is not after settlement",
        ));
    }
    let dividend_dates = preferred.dividend_dates(settlement)?;
    let dt = (horizon - t0) / steps as f64;
    let up = (market.volatility * dt.sqrt()).exp();
    let down = 1.0 / up;

    let times: Vec<f64> = (0..=steps).map(|i| t0 + i as f64 * dt).collect();
    let spread_df = (-market.credit_spread * dt).exp();
    let mut riskfree_df = Vec::with_capacity(steps);
    let mut risky_df = Vec::with_capacity(steps);
    let mut probability = Vec::with_capacity(steps);
    for i in 0..steps {
        let df_step = curve.df(times[i + 1]) / curve.df(times[i]);
        let growth = (-market.dividend_yield * dt).exp() / df_step;
        let p = (growth - down) / (up - down);
        if !(0.0..=1.0).contains(&p) {
            return Err(RustyQLibError::NumericalError(format!(
                "risk-neutral probability {p} outside [0, 1] at step {i}; \
                 increase the tree steps or check the inputs"
            )));
        }
        riskfree_df.push(df_step);
        risky_df.push(df_step * spread_df);
        probability.push(p);
    }

    // dividends assigned to their step interval, discounted to its edge
    let dividend = preferred.periodic_dividend();
    let mut dividend_at_step = vec![0.0_f64; steps];
    for &date in &dividend_dates {
        let time = year_fraction(date).clamp(t0, horizon);
        let index = (((time - t0) / dt).ceil() as usize).clamp(1, steps) - 1;
        let forward = -(riskfree_df[index].ln()) / dt;
        dividend_at_step[index] +=
            dividend * (-(forward + market.credit_spread) * (time - times[index])).exp();
    }

    // calls apply from their date onward — a preferred is continuously
    // callable once its first call date passes — with a later schedule
    // entry superseding the earlier one; the strike carries the accrued
    // of the step's position in the dividend cycle
    let mut call_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    if !preferred.calls.is_empty() {
        let mut schedule: Vec<(f64, f64)> = Vec::new();
        for call in &preferred.calls {
            if !(call.call_price > 0.0 && call.call_price.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "preferred",
                    format!("call price must be positive, got {}", call.call_price),
                ));
            }
            if call.call_date >= horizon_date {
                continue;
            }
            let redemption = preferred.preference * call.call_price / 100.0;
            schedule.push((year_fraction(call.call_date), redemption));
        }
        schedule.sort_by(|a, b| a.0.total_cmp(&b.0));
        let dividend_times: Vec<f64> = dividend_dates
            .iter()
            .map(|&date| year_fraction(date))
            .collect();
        let period = 1.0 / preferred.frequency.per_year() as f64;
        let accrued_at = |t: f64| -> f64 {
            if !preferred.cumulative {
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

    let convert_from = year_fraction(preferred.convert_from.unwrap_or(preferred.dated_date));
    let ratio = preferred.conversion_ratio;
    let spot_at =
        |step: usize, j: usize| market.spot * up.powi(j as i32) * down.powi((step - j) as i32);

    // terminal: mandatory redemption at the preference, or the
    // perpetuity continuing value of the remaining dividends
    let terminal_cash = if preferred.mandatory_redemption.is_some() {
        preferred.preference
    } else {
        preferred.perpetuity_tail(market, curve, horizon)
    };
    let mut equity: Vec<f64> = Vec::with_capacity(steps + 1);
    let mut cash: Vec<f64> = Vec::with_capacity(steps + 1);
    for j in 0..=steps {
        let shares = ratio * spot_at(steps, j);
        if shares > terminal_cash {
            equity.push(shares);
            cash.push(0.0);
        } else {
            equity.push(0.0);
            cash.push(terminal_cash);
        }
    }
    // the soft trigger is smoothed across the one-node cell it falls in
    // (linear in log-price): a hard indicator makes the lattice price
    // sawtooth in spot as the node ladder slides past the fixed trigger
    let call_weight = |spot: f64| -> f64 {
        match preferred.soft_call_trigger {
            None => 1.0,
            Some(trigger) => ((spot / trigger).ln() / (2.0 * up.ln()) + 0.5).clamp(0.0, 1.0),
        }
    };
    // the truncation horizon inherits a standing call: the continuing
    // value is capped at the strike (or forced into conversion)
    // wherever the trigger is met
    if let Some(strike) = call_at_step[steps] {
        let in_window = times[steps] >= convert_from - 1e-9;
        for j in 0..=steps {
            let spot = spot_at(steps, j);
            let weight = call_weight(spot);
            if weight <= 0.0 {
                continue;
            }
            let shares = ratio * spot;
            let forced = if in_window {
                strike.max(shares)
            } else {
                strike
            };
            if equity[j] + cash[j] > forced {
                let (forced_e, forced_b) = if in_window && shares > strike {
                    (shares, 0.0)
                } else {
                    (0.0, strike)
                };
                equity[j] += weight * (forced_e - equity[j]);
                cash[j] += weight * (forced_b - cash[j]);
            }
        }
    }

    for step in (0..steps).rev() {
        let p = probability[step];
        let in_window = times[step] >= convert_from - 1e-9;
        for j in 0..=step {
            let mut e = riskfree_df[step] * (p * equity[j + 1] + (1.0 - p) * equity[j]);
            let mut b = risky_df[step] * (p * cash[j + 1] + (1.0 - p) * cash[j]);
            b += dividend_at_step[step];
            let spot = spot_at(step, j);
            let shares = ratio * spot;

            if let Some(strike) = call_at_step[step] {
                let weight = call_weight(spot);
                if weight > 0.0 {
                    let forced = if in_window {
                        strike.max(shares)
                    } else {
                        strike
                    };
                    if e + b > forced {
                        let (forced_e, forced_b) = if in_window && shares > strike {
                            (shares, 0.0)
                        } else {
                            (0.0, strike)
                        };
                        e += weight * (forced_e - e);
                        b += weight * (forced_b - b);
                    }
                }
            }
            if in_window && shares > e + b {
                e = shares;
                b = 0.0;
            }
            equity[j] = e;
            cash[j] = b;
        }
    }
    Ok(equity[0] + cash[0])
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!((floor - manual).abs() < 1e-12, "{floor} vs {manual}");
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
        let price = cps.dirty_price(&m, &curve, settlement).unwrap();
        let implied = cps
            .implied_credit_spread(price, &m, &curve, settlement)
            .unwrap();
        assert!(
            (implied - m.credit_spread).abs() < 1e-5,
            "implied {implied}"
        );
        assert!((cps.conversion_price() - 62.5).abs() < 1e-12);
        assert!((cps.parity(60.0) - 96.0).abs() < 1e-12);
        let premium = cps.conversion_premium(price, 60.0).unwrap();
        assert!(premium > 0.0);
        let yield_now = cps.current_yield(price).unwrap();
        assert!((yield_now - 5.5 / price).abs() < 1e-12);
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
        // settlement past a mandatory redemption
        let mut finite = cps.clone();
        finite.mandatory_redemption = Some(d(2031, 5, 15));
        assert!(finite.dirty_price(&m, &curve, d(2032, 1, 1)).is_err());
    }
}
