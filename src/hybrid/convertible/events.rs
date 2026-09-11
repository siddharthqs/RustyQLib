//! The time grid and the instrument's events mapped onto it, shared
//! by the tree and finite-difference engines, plus the per-node
//! exercise logic and terminal payoff, generic over the instrument and
//! the credit model's node type. Each instrument builds its own grid
//! ([`ConvertibleInstrument::event_grid`]); the bond's builder lives
//! here.

use chrono::NaiveDate;

use super::instrument::{CashDividend, ConvertibleInstrument};
use super::models::NodeValue;
use super::ConvertibleBond;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// The time grid and the instrument's events mapped onto it, shared by
/// the tree engines and the finite-difference engine. Built by each
/// [`ConvertibleInstrument`]; its fields are crate-private, so the trait
/// is implementable inside the crate only for now. Cash amounts are absolute (on `face_value`);
/// times are on the curve's day count from its reference date.
pub struct EventGrid {
    pub(crate) dt: f64,
    /// `steps + 1` node times.
    pub(crate) times: Vec<f64>,
    /// Risk-free discount factor over each step.
    pub(crate) riskfree_df: Vec<f64>,
    /// Coupons (ex the final flow) valued at the left edge of the step
    /// interval containing their payment, discounted at the step's
    /// forward rate plus `extra_rate`. These are **at stake**: added to
    /// the continuation before any exercise at the step, so a holder
    /// who puts or converts, or an issuer who calls, forfeits them.
    pub(crate) coupon_at_step: Vec<f64>,
    /// Coupons paid on or before the call/put date mapped to the same
    /// step: they belong to the holder whatever happens, so they are
    /// added after the exercise decisions. (A put on a coupon date pays
    /// par plus that coupon; without this split the coupon would be
    /// lost to the exercise comparison whenever both land on one step.)
    pub(crate) coupon_kept_at_step: Vec<f64>,
    /// Face-recovery claim (per unit recovery rate) valued at the end
    /// of each step, should default be observed there: the face
    /// outstanding in that coupon period, paid at the period's payment
    /// date. Used by the jump-to-default engine only.
    pub(crate) default_claim_at_step: Vec<f64>,
    /// Dirty call / put strikes at the step nearest each option date.
    pub(crate) call_at_step: Vec<Option<f64>>,
    pub(crate) put_at_step: Vec<Option<f64>>,
    /// Coupon make-whole owed with the call at each step.
    pub(crate) make_whole_at_step: Vec<f64>,
    /// Contingent-conversion trigger in force at each step.
    pub(crate) conversion_trigger_at_step: Vec<Option<f64>>,
    /// Probability of a fundamental change over each step (`steps + 1`
    /// entries, the last zero: no event past the horizon).
    pub(crate) event_probability_at_step: Vec<f64>,
    /// Par plus accrued at each step: the fundamental-change put.
    pub(crate) par_put_at_step: Vec<f64>,
    /// Additional shares on a fundamental change at each step, over
    /// `additional_share_prices`.
    pub(crate) additional_shares_at_step: Vec<Vec<f64>>,
    pub(crate) additional_share_prices: Vec<f64>,
    pub(crate) convert_from: f64,
    pub(crate) convert_until: f64,
    /// The final flow (face plus last coupon).
    pub(crate) redemption: f64,
    /// The principal inside the final flow.
    pub(crate) final_principal: f64,
    /// The price-unit base: values scale by `100 / outstanding`, so
    /// the face outstanding at settlement for a per-100 bond price,
    /// or 100 for a per-share price.
    pub(crate) outstanding: f64,
    /// Smooth the soft-call trigger across the one grid cell it falls
    /// in (linear in log-price) instead of a hard indicator; a hard
    /// trigger makes a lattice price sawtooth in spot.
    pub(crate) smooth_trigger: bool,
    /// Apply a standing call at the final step too (a truncated
    /// perpetual's continuing value is capped by it).
    pub(crate) exercise_at_horizon: bool,
    /// Cash dividend per share going ex at each step (`steps + 1`).
    pub(crate) cash_dividend_at_step: Vec<f64>,
}

/// Cash dividends mapped to the step nearest their ex-date; dividends
/// outside `(settlement, horizon)` do not touch the holder. With a
/// protection threshold only the unprotected part, `min(D, threshold)`,
/// jumps (see [`DividendProtection`](super::DividendProtection)).
pub(crate) fn cash_dividends_at_steps(
    dividends: &[CashDividend],
    protection_threshold: Option<f64>,
    times: &[f64],
    year_fraction: impl Fn(NaiveDate) -> f64,
) -> Vec<f64> {
    let steps = times.len() - 1;
    let (t0, horizon) = (times[0], times[steps]);
    let dt = (horizon - t0) / steps as f64;
    let mut at_step = vec![0.0; steps + 1];
    for dividend in dividends {
        let time = year_fraction(dividend.ex_date);
        if time <= t0 + 1e-9 || time >= horizon - 1e-9 {
            continue;
        }
        let index = ((time - t0) / dt).round() as usize;
        let unprotected = match protection_threshold {
            Some(threshold) => dividend.amount.min(threshold),
            None => dividend.amount,
        };
        at_step[index.min(steps)] += unprotected;
    }
    at_step
}

/// The ex-dividend jump: every node takes the value the ladder holds
/// at its spot less the dividend, linearly interpolated between the
/// two nodes around it, and the ladder's lowest value below the
/// bottom. `spots` is ascending and aligned with `nodes`.
pub(crate) fn apply_cash_dividend<N: NodeValue>(nodes: &mut [N], spots: &[f64], dividend: f64) {
    if dividend <= 0.0 {
        return;
    }
    let before: Vec<N> = nodes.to_vec();
    for (node, &spot) in nodes.iter_mut().zip(spots) {
        let target = spot - dividend;
        if target <= spots[0] {
            *node = before[0];
            continue;
        }
        let k = spots
            .partition_point(|&s| s <= target)
            .clamp(1, spots.len() - 1);
        let (lo, hi) = (spots[k - 1], spots[k]);
        let w = ((target - lo) / (hi - lo)).clamp(0.0, 1.0);
        *node = N::blend(before[k - 1], before[k], w);
    }
}

impl EventGrid {
    pub(crate) fn can_convert_at_maturity(&self) -> bool {
        self.convert_until >= self.times[self.times.len() - 1] - 1e-9
    }

    pub(crate) fn in_window(&self, step: usize) -> bool {
        let time = self.times[step];
        time >= self.convert_from - 1e-9 && time <= self.convert_until + 1e-9
    }

    /// The terminal node at a terminal share price: a mandatory
    /// delivers its share schedule (equity) plus the final coupon
    /// (cash); otherwise the holder converts against the full
    /// redemption package when parity beats it and the window allows.
    /// A grid that exercises at the horizon then applies the final
    /// step's standing events (a truncated perpetual's call).
    pub(crate) fn terminal<N: NodeValue, I: ConvertibleInstrument + ?Sized>(
        &self,
        instrument: &I,
        spot: f64,
        cell_width: f64,
    ) -> N {
        let shares = instrument.maturity_shares(spot) * spot;
        let node = if instrument.is_mandatory() {
            N::equity(shares).plus_cash(self.redemption - self.final_principal)
        } else if self.can_convert_at_maturity() && shares > self.redemption {
            N::equity(shares)
        } else {
            N::cash(self.redemption)
        };
        if self.exercise_at_horizon {
            let last = self.times.len() - 1;
            self.exercise(instrument, last, spot, node, cell_width)
        } else {
            node
        }
    }

    /// Whether the holder may convert at this step and share price:
    /// inside the window and, if contingent, at or above the trigger.
    fn can_convert(&self, step: usize, spot: f64) -> bool {
        self.in_window(step)
            && self.conversion_trigger_at_step[step].is_none_or(|trigger| spot >= trigger)
    }

    /// The issuer's dirty call strike at this step with the weight of
    /// its soft trigger at this share price: one without a trigger or
    /// with a hard trigger met, zero with a hard trigger missed, and
    /// linear in log-price across one grid cell when smoothed. `None`
    /// when there is no call or no weight.
    fn call_strike<I: ConvertibleInstrument + ?Sized>(
        &self,
        instrument: &I,
        step: usize,
        spot: f64,
        cell_width: f64,
    ) -> Option<(f64, f64)> {
        let strike = self.call_at_step[step]?;
        let weight = match instrument.soft_call_trigger() {
            None => 1.0,
            Some(trigger) if self.smooth_trigger => {
                ((spot / trigger).ln() / cell_width + 0.5).clamp(0.0, 1.0)
            }
            Some(trigger) => {
                if spot >= trigger {
                    1.0
                } else {
                    0.0
                }
            }
        };
        (weight > 0.0).then_some((strike, weight))
    }

    /// Additional shares on a fundamental change at this step and
    /// share price, from the table interpolated in price.
    fn additional_shares(&self, step: usize, spot: f64) -> f64 {
        let prices = &self.additional_share_prices;
        if prices.is_empty() || spot < prices[0] || spot > prices[prices.len() - 1] {
            return 0.0;
        }
        let row = &self.additional_shares_at_step[step];
        let k = prices
            .partition_point(|&p| p <= spot)
            .clamp(1, prices.len() - 1);
        let (p0, p1) = (prices[k - 1], prices[k]);
        row[k - 1] + (row[k] - row[k - 1]) * (spot - p0) / (p1 - p0)
    }

    /// The node after this step's events, from the continuation `node`
    /// at `spot` (already carrying the at-stake coupon). In order: the
    /// fundamental change (the holder takes the best of carrying on,
    /// the make-whole conversion and the par put), the issuer's call
    /// with its coupon make-whole (the holder answers with conversion
    /// when parity beats the strike; a smoothed trigger blends the
    /// called and uncalled nodes), the holder's put, voluntary
    /// conversion. Generic over the node: under the Tsiveriotis-
    /// Fernandes split a conversion moves the value to the equity part
    /// and a cash settlement to the cash part; a scalar node just takes
    /// the values.
    pub(crate) fn exercise<N: NodeValue, I: ConvertibleInstrument + ?Sized>(
        &self,
        instrument: &I,
        step: usize,
        spot: f64,
        mut node: N,
        cell_width: f64,
    ) -> N {
        let ratio = instrument.conversion_ratio();
        let shares = ratio * spot;
        let can_convert = self.can_convert(step, spot);

        let p = self.event_probability_at_step[step];
        if p > 0.0 {
            let make_whole_shares = (ratio + self.additional_shares(step, spot)) * spot;
            let par = self.par_put_at_step[step];
            let on_event = if make_whole_shares >= node.total().max(par) {
                N::equity(make_whole_shares)
            } else if par > node.total() {
                N::cash(par)
            } else {
                node
            };
            node = N::blend(node, on_event, p);
        }
        if let Some((strike, weight)) = self.call_strike(instrument, step, spot, cell_width) {
            let make_whole = self.make_whole_at_step[step];
            let forced = if can_convert {
                strike.max(shares)
            } else {
                strike
            } + make_whole;
            if node.total() > forced {
                let called = if can_convert && shares > strike {
                    N::equity(shares).plus_cash(make_whole)
                } else {
                    N::cash(strike + make_whole)
                };
                node = N::blend(node, called, weight);
            }
        }
        if let Some(strike) = self.put_at_step[step] {
            if strike > node.total() {
                node = N::cash(strike);
            }
        }
        if can_convert && shares > node.total() {
            node = N::equity(shares);
        }
        node
    }
}

/// Builds the bond's [`EventGrid`] for `steps` steps from settlement to the
/// final payment. `extra_rate` is the rate on top of the risk-free
/// forward at which intra-step coupons are discounted to the step
/// edge: the credit spread under Tsiveriotis-Fernandes, the hazard
/// rate under jump to default (survival to the payment).
pub(crate) fn event_grid(
    convertible: &ConvertibleBond,
    curve: &YieldCurve,
    settlement: NaiveDate,
    steps: usize,
    extra_rate: f64,
) -> Result<EventGrid, RustyQLibError> {
    let bond = &convertible.bond;
    // settlement validity as for every other bond pricer
    bond.accrued_interest(settlement)?;
    convertible.validate_features()?;

    let year_fraction = |date: NaiveDate| {
        curve
            .day_count()
            .year_fraction(curve.reference_date(), date)
    };
    let cashflows: Vec<_> = bond
        .cashflows()
        .iter()
        .filter(|cf| cf.accrual_end > settlement)
        .cloned()
        .collect();
    let last = cashflows.last().ok_or_else(|| {
        RustyQLibError::invalid_input("convertible", "no cash flows after settlement")
    })?;

    let t0 = year_fraction(settlement);
    let horizon = year_fraction(last.payment_date);
    if horizon <= t0 {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            "the bond matures at settlement",
        ));
    }
    let dt = (horizon - t0) / steps as f64;
    let times: Vec<f64> = (0..=steps).map(|i| t0 + i as f64 * dt).collect();
    let riskfree_df: Vec<f64> = (0..steps)
        .map(|i| curve.df(times[i + 1]) / curve.df(times[i]))
        .collect();

    // the recovery claim per step: default observed at the step's end
    // falls in the coupon period whose accrual end is the first at or
    // after it, and pays on that period's payment date
    let periods: Vec<(f64, f64, f64)> = cashflows
        .iter()
        .map(|cf| {
            (
                year_fraction(cf.accrual_end),
                curve.df(year_fraction(cf.payment_date)),
                bond.outstanding_face(cf.accrual_start),
            )
        })
        .collect();
    let mut period = 0;
    let mut default_claim_at_step: Vec<f64> = Vec::with_capacity(steps);
    for &time in &times[1..] {
        while period + 1 < periods.len() && periods[period].0 < time - 1e-9 {
            period += 1;
        }
        let (_, payment_df, face) = periods[period];
        default_claim_at_step.push(face * payment_df / curve.df(time));
    }

    // decision windows in step time
    let convert_from = year_fraction(convertible.convert_from.unwrap_or(bond.dated_date));
    let convert_until = year_fraction(convertible.convert_until.unwrap_or(bond.maturity_date));
    // calls and puts at the step nearest their date; the earliest
    // option date at each step decides which coupons are at stake there
    let mut option_time_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    let mut note_option = |index: usize, time: f64| {
        option_time_at_step[index] = Some(option_time_at_step[index].map_or(time, |t| t.min(time)));
    };
    let mut call_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    for call in &convertible.calls {
        if call.call_date <= settlement || call.call_date >= bond.maturity_date {
            continue;
        }
        let strike = bond.dirty_redemption_amount(call.call_date, call.call_price)?;
        let time = year_fraction(call.call_date);
        let index = (((time - t0) / dt).round() as usize).min(steps);
        call_at_step[index] = Some(strike);
        note_option(index, time);
    }
    let mut put_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    for put in &convertible.puts {
        if put.put_date <= settlement || put.put_date >= bond.maturity_date {
            continue;
        }
        let strike = bond.dirty_redemption_amount(put.put_date, put.put_price)?;
        let time = year_fraction(put.put_date);
        let index = (((time - t0) / dt).round() as usize).min(steps);
        put_at_step[index] = Some(strike);
        note_option(index, time);
    }

    // the coupon make-whole owed with each call: the coupons scheduled
    // after the call date up to `until`, discounted to the call date on
    // the curve plus the make-whole spread
    let mut make_whole_at_step: Vec<f64> = vec![0.0; steps + 1];
    if let Some(make_whole) = &convertible.coupon_make_whole {
        for call in &convertible.calls {
            if call.call_date <= settlement
                || call.call_date >= bond.maturity_date
                || call.call_date >= make_whole.until
            {
                continue;
            }
            let call_time = year_fraction(call.call_date);
            let index = (((call_time - t0) / dt).round() as usize).min(steps);
            let df_call = curve.df(call_time);
            let pv: f64 = cashflows
                .iter()
                .filter(|cf| cf.payment_date > call.call_date && cf.accrual_end <= make_whole.until)
                .map(|cf| {
                    let coupon = cf.amount - bond.principal_at(cf.accrual_end);
                    let pay_time = year_fraction(cf.payment_date);
                    coupon * curve.df(pay_time) / df_call
                        * (-make_whole.spread * (pay_time - call_time)).exp()
                })
                .sum();
            make_whole_at_step[index] = pv;
        }
    }

    // contingent conversion: the trigger applies before its end date
    let conversion_trigger_at_step: Vec<Option<f64>> = match &convertible.contingent_conversion {
        Some(coco) => {
            let until = year_fraction(coco.until);
            times
                .iter()
                .map(|&t| (t < until - 1e-9).then_some(coco.trigger))
                .collect()
        }
        None => vec![None; steps + 1],
    };

    // the fundamental change: event probability per step, the par put
    // (outstanding plus accrued at the step's date) and the make-whole
    // table's row at each step
    let mut event_probability_at_step = vec![0.0; steps + 1];
    let mut par_put_at_step = vec![0.0; steps + 1];
    let mut additional_shares_at_step: Vec<Vec<f64>> = vec![Vec::new(); steps + 1];
    let mut additional_share_prices = Vec::new();
    if let Some(fc) = &convertible.fundamental_change {
        event_probability_at_step = vec![1.0 - (-fc.event_intensity * dt).exp(); steps + 1];
        event_probability_at_step[steps] = 0.0;
        additional_share_prices = fc.stock_prices.clone();
        let row_times: Vec<f64> = fc
            .effective_dates
            .iter()
            .map(|&d| year_fraction(d))
            .collect();
        // the calendar date of each step, walked forward from settlement
        let last_accrual_date = bond.maturity_date.pred_opt().unwrap_or(bond.maturity_date);
        let mut date = settlement;
        for (step, &time) in times.iter().enumerate() {
            while year_fraction(date) < time - 1e-9 && date < last_accrual_date {
                date = date.succ_opt().unwrap_or(date);
            }
            par_put_at_step[step] = bond.dirty_redemption_amount(date, 100.0)?;
            additional_shares_at_step[step] = fc.row_at(time, &row_times);
        }
    }

    // coupons (ex the final flow) assigned to the step interval that
    // contains their payment time; forward rates discount them back to
    // the step's left edge. A coupon paid on or before an option date
    // at the same step is the holder's regardless of exercise (the dirty
    // strike carries no accrued for it), so it is kept aside; any other
    // coupon is at stake in the exercise comparison.
    let mut coupon_at_step: Vec<f64> = vec![0.0; steps];
    let mut coupon_kept_at_step: Vec<f64> = vec![0.0; steps];
    for cf in cashflows.iter().take(cashflows.len() - 1) {
        let time = year_fraction(cf.payment_date).clamp(t0, horizon);
        let index = (((time - t0) / dt).ceil() as usize).clamp(1, steps) - 1;
        let forward = -(riskfree_df[index].ln()) / dt;
        let discount_to_edge = (-(forward + extra_rate) * (time - times[index])).exp();
        let kept = option_time_at_step[index].is_some_and(|option_time| time <= option_time + 1e-9);
        if kept {
            coupon_kept_at_step[index] += cf.amount * discount_to_edge;
        } else {
            coupon_at_step[index] += cf.amount * discount_to_edge;
        }
    }

    let cash_dividend_at_step = cash_dividends_at_steps(
        &convertible.cash_dividends,
        convertible.dividend_protection.map(|p| p.threshold),
        &times,
        year_fraction,
    );
    Ok(EventGrid {
        dt,
        times,
        riskfree_df,
        coupon_at_step,
        coupon_kept_at_step,
        default_claim_at_step,
        call_at_step,
        put_at_step,
        make_whole_at_step,
        conversion_trigger_at_step,
        event_probability_at_step,
        par_put_at_step,
        additional_shares_at_step,
        additional_share_prices,
        convert_from,
        convert_until,
        redemption: last.amount,
        final_principal: bond.principal_at(last.accrual_end),
        outstanding: bond.outstanding_face(settlement),
        smooth_trigger: false,
        exercise_at_horizon: false,
        cash_dividend_at_step,
    })
}
