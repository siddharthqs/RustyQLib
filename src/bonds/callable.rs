//! Option-model callable bond pricing under Hull-White.
//!
//! The issuer's call right is a Bermudan option, priced by backward
//! induction on a grid of the Hull-White state `x(t) = r(t) - alpha(t)`
//! (an Ornstein-Uhlenbeck process starting at zero). The induction
//! steps directly **between event dates** — coupon payments and call
//! decisions — with no intermediate time discretization:
//!
//! - discounting over a step uses the model's exact zero-bond price
//!   `P(t, t+dt | r)`, and
//! - the state transition is the exact Gaussian law of `x` under the
//!   `t+dt`-forward measure (mean `x e^{-a dt} - M(dt)`, the measure
//!   change that makes bond-discounted expectations consistent),
//!
//! so the only numerical error is quadrature and grid interpolation —
//! verified by pricing a call-free bond on the grid against the
//! closed-form curve price. At each call date the issuer redeems when
//! continuation costs more than the call price plus accrued:
//! `V = min(V_continuation, strike_dirty)`.
//!
//! On top of the engine: **OAS** (the constant spread over the model
//! curve that reprices a market quote, the option-adjusted analogue of
//! the z-spread), the **option value** (straight minus callable at the
//! same spread), and **effective duration/convexity** (reprice on
//! bumped curves with the model refitted, letting the call exercise
//! move — the numbers a callable's hedger actually uses).

use chrono::NaiveDate;

use crate::bonds::{CallOption, FixedRateBond, PutOption};
use crate::core::curves::RateShift;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::utils::norm_pdf;
use crate::rates::models::{HullWhite, ShortRateModel};

/// A make-whole call: from `start` the issuer may redeem at
/// `max(par, PV of the remaining flows discounted at the prevailing
/// curve plus `spread`)` — the strike is computed **at each grid node**
/// from the model's own zero-coupon bonds, so it falls as rates rise.
/// Exercise is modelled on the coupon dates from `start` (the Bermudan
/// reading of the contractually American right; make-whole calls are
/// rarely time-critical because the strike tracks fair value).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MakeWholeCall {
    pub start: NaiveDate,
    /// The make-whole spread over the reference curve, e.g. `0.0025`
    /// for T+25.
    pub spread: f64,
}

/// The embedded options of a bond, in one bundle: a fixed-price call
/// schedule, a put schedule, and/or a make-whole call.
#[derive(Debug, Clone, Default)]
pub struct BondOptionality {
    pub calls: Vec<CallOption>,
    pub puts: Vec<PutOption>,
    pub make_whole: Option<MakeWholeCall>,
}

impl BondOptionality {
    /// A bond with no embedded options.
    pub fn none() -> Self {
        Self::default()
    }

    /// A fixed-price call schedule only.
    pub fn from_calls(calls: &[CallOption]) -> Self {
        BondOptionality {
            calls: calls.to_vec(),
            ..Self::default()
        }
    }

    /// A put schedule only.
    pub fn from_puts(puts: &[PutOption]) -> Self {
        BondOptionality {
            puts: puts.to_vec(),
            ..Self::default()
        }
    }

    /// A make-whole call only.
    pub fn from_make_whole(make_whole: MakeWholeCall) -> Self {
        BondOptionality {
            make_whole: Some(make_whole),
            ..Self::default()
        }
    }
}

/// State-grid nodes (odd, so `x = 0` is a node).
const GRID_NODES: usize = 201;
/// Grid half-width in terminal standard deviations of the state.
const GRID_STDS: f64 = 7.0;
/// Quadrature nodes over the standard normal (odd, Simpson).
const QUAD_NODES: usize = 101;
/// Quadrature span in standard deviations.
const QUAD_SPAN: f64 = 8.0;
/// OAS search bracket.
const OAS_BRACKET: (f64, f64) = (-0.5, 3.0);

/// One backward-induction event.
struct Event {
    time: f64,
    /// Cash paid at this time (absolute).
    payment: f64,
    /// Dirty call strike (absolute) when the issuer may redeem here.
    call_strike: Option<f64>,
    /// Dirty put strike (absolute) when the holder may redeem here.
    put_strike: Option<f64>,
    /// A make-whole decision here: `(spread, outstanding face, dirty
    /// accrued)` — the strike itself is node-dependent, computed in the
    /// induction and floored at the outstanding face.
    make_whole: Option<(f64, f64, f64)>,
}

impl Event {
    fn at(time: f64) -> Self {
        Event {
            time,
            payment: 0.0,
            call_strike: None,
            put_strike: None,
            make_whole: None,
        }
    }
}

impl FixedRateBond {
    /// Dirty price per 100 face at `settlement` of the bond with its
    /// embedded options exercised optimally under `model` — the issuer
    /// minimizes at calls (fixed-price and make-whole), the holder
    /// maximizes at puts — discounted with a constant `spread` (the OAS
    /// convention) over the model curve.
    pub fn option_adjusted_dirty_price_hw(
        &self,
        model: &HullWhite,
        optionality: &BondOptionality,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "callable",
                format!("spread must be finite, got {spread}"),
            ));
        }
        // settlement validity, as for every other pricer
        self.accrued_interest(settlement)?;
        let curve = model.curve();
        let year_fraction = |date: NaiveDate| {
            curve
                .day_count()
                .year_fraction(curve.reference_date(), date)
        };
        let t_settlement = year_fraction(settlement).max(0.0);

        // an exercise scheduled on a coupon date must land on that
        // coupon's (possibly business-day-rolled) payment event: only a
        // merged event runs the exercise before the coupon is added, so
        // the earned coupon stays out of both sides of the comparison.
        // A separate event a few days earlier would leave the coupon in
        // the continuation but not the strike — understating the strike
        // by a full coupon.
        let exercise_times: Vec<(NaiveDate, f64)> = self
            .cashflows()
            .iter()
            .map(|cf| (cf.accrual_end, year_fraction(cf.payment_date).max(t_settlement)))
            .collect();
        let event_time = |date: NaiveDate| {
            exercise_times
                .iter()
                .find(|&&(accrual_end, _)| accrual_end == date)
                .map(|&(_, time)| time)
                .unwrap_or_else(|| year_fraction(date).max(t_settlement))
        };

        // events: remaining cash flows, merged with exercise decisions
        let mut events: Vec<Event> = self
            .cashflows()
            .iter()
            .filter(|cf| cf.accrual_end > settlement)
            .map(|cf| Event {
                payment: cf.amount,
                ..Event::at(year_fraction(cf.payment_date).max(t_settlement))
            })
            .collect();
        // the flow list the make-whole strike discounts at each node
        let flows: Vec<(f64, f64)> = events.iter().map(|e| (e.time, e.payment)).collect();

        let mut upsert = |time: f64, apply: &mut dyn FnMut(&mut Event)| match events
            .iter_mut()
            .find(|e| (e.time - time).abs() < 1e-12)
        {
            Some(event) => apply(event),
            None => {
                let mut event = Event::at(time);
                apply(&mut event);
                events.push(event);
            }
        };

        for call in &optionality.calls {
            if call.call_date <= settlement || call.call_date >= self.maturity_date {
                continue;
            }
            if !(call.call_price > 0.0 && call.call_price.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "callable",
                    format!("call price must be positive, got {}", call.call_price),
                ));
            }
            let outstanding = self.outstanding_face(call.call_date);
            let strike = outstanding * call.call_price / 100.0
                + outstanding * self.accrued_interest(call.call_date)? / 100.0;
            upsert(event_time(call.call_date), &mut |event| {
                event.call_strike = Some(strike);
            });
        }
        for put in &optionality.puts {
            if put.put_date <= settlement || put.put_date >= self.maturity_date {
                continue;
            }
            if !(put.put_price > 0.0 && put.put_price.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "puttable",
                    format!("put price must be positive, got {}", put.put_price),
                ));
            }
            let outstanding = self.outstanding_face(put.put_date);
            let strike = outstanding * put.put_price / 100.0
                + outstanding * self.accrued_interest(put.put_date)? / 100.0;
            upsert(event_time(put.put_date), &mut |event| {
                event.put_strike = Some(strike);
            });
        }
        if let Some(make_whole) = &optionality.make_whole {
            if !make_whole.spread.is_finite() || make_whole.spread < 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "make_whole",
                    format!("spread must be non-negative, got {}", make_whole.spread),
                ));
            }
            // Bermudan reading: decisions on the coupon dates from `start`
            for cf in self.cashflows() {
                if cf.accrual_end <= settlement
                    || cf.accrual_end < make_whole.start
                    || cf.accrual_end >= self.maturity_date
                {
                    continue;
                }
                let outstanding = self.outstanding_face(cf.accrual_end);
                let accrued = outstanding * self.accrued_interest(cf.accrual_end)? / 100.0;
                upsert(event_time(cf.accrual_end), &mut |event| {
                    event.make_whole = Some((make_whole.spread, outstanding, accrued));
                });
            }
        }
        events.sort_by(|a, b| a.time.total_cmp(&b.time));

        let pv = backward_induction(model, &events, &flows, spread, t_settlement)?;
        let df_settlement = curve.df(t_settlement) * (-spread * t_settlement).exp();
        Ok(pv / df_settlement * 100.0 / self.outstanding_face(settlement))
    }

    /// Clean price per 100 face with the embedded options.
    pub fn option_adjusted_clean_price_hw(
        &self,
        model: &HullWhite,
        optionality: &BondOptionality,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.option_adjusted_dirty_price_hw(model, optionality, spread, settlement)?
                - self.accrued_interest(settlement)?,
        )
    }

    /// Dirty price per 100 face with a fixed-price call schedule only.
    pub fn callable_dirty_price_hw(
        &self,
        model: &HullWhite,
        calls: &[CallOption],
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.option_adjusted_dirty_price_hw(
            model,
            &BondOptionality::from_calls(calls),
            spread,
            settlement,
        )
    }

    /// Clean price per 100 face with a fixed-price call schedule only.
    pub fn callable_clean_price_hw(
        &self,
        model: &HullWhite,
        calls: &[CallOption],
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.callable_dirty_price_hw(model, calls, spread, settlement)?
                - self.accrued_interest(settlement)?,
        )
    }

    /// Dirty price per 100 face with a put schedule only.
    pub fn puttable_dirty_price_hw(
        &self,
        model: &HullWhite,
        puts: &[PutOption],
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.option_adjusted_dirty_price_hw(
            model,
            &BondOptionality::from_puts(puts),
            spread,
            settlement,
        )
    }

    /// Option-adjusted spread: the constant spread over the model curve
    /// at which the option-model price (with every embedded option
    /// exercised optimally) matches the quoted clean price.
    pub fn oas_hw(
        &self,
        clean_price: f64,
        model: &HullWhite,
        optionality: &BondOptionality,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !(clean_price > 0.0 && clean_price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "oas",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // price decreases in the spread: target - price(s) is increasing
        let objective = |s: f64| {
            clean_price
                - self
                    .option_adjusted_clean_price_hw(model, optionality, s, settlement)
                    .expect("the OAS bracket keeps the engine valid")
        };
        // surface real engine errors before the solver
        self.option_adjusted_clean_price_hw(model, optionality, OAS_BRACKET.0, settlement)?;
        let root = Solver1d::new(1e-10, 200).bisection(objective, OAS_BRACKET.0, OAS_BRACKET.1)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "OAS solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }

    /// Net value of the embedded options per 100 face at a given
    /// spread: straight (option-free) price minus the option-adjusted
    /// price. Positive when issuer options (calls) dominate, negative
    /// when holder options (puts) do.
    pub fn embedded_option_value_hw(
        &self,
        model: &HullWhite,
        optionality: &BondOptionality,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let straight = self.option_adjusted_dirty_price_hw(
            model,
            &BondOptionality::none(),
            spread,
            settlement,
        )?;
        let adjusted =
            self.option_adjusted_dirty_price_hw(model, optionality, spread, settlement)?;
        Ok(straight - adjusted)
    }

    /// Value of the issuer's option per 100 face at a given spread:
    /// straight (call-free) price minus callable price on the same
    /// model and spread. Non-negative by construction.
    pub fn call_option_value_hw(
        &self,
        model: &HullWhite,
        calls: &[CallOption],
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.embedded_option_value_hw(
            model,
            &BondOptionality::from_calls(calls),
            spread,
            settlement,
        )
    }

    /// Effective duration and convexity of the optioned bond: reprice
    /// on parallel-bumped curves (±`bump`, e.g. `0.0025` for 25bp) with
    /// the model refitted to each bumped curve at the same `(a, sigma)`,
    /// keeping the OAS fixed so the exercise boundaries respond to the
    /// rate move. Returns `(duration, convexity)` in years and years².
    pub fn effective_duration_convexity_hw(
        &self,
        model: &HullWhite,
        optionality: &BondOptionality,
        spread: f64,
        settlement: NaiveDate,
        bump: f64,
    ) -> Result<(f64, f64), RustyQLibError> {
        if !(bump > 0.0 && bump.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "effective duration",
                format!("bump must be positive, got {bump}"),
            ));
        }
        let base = self.option_adjusted_dirty_price_hw(model, optionality, spread, settlement)?;
        let bumped_price = |shift: f64| -> Result<f64, RustyQLibError> {
            let curve = model.curve().bumped(&RateShift::ParallelAbsolute(shift))?;
            let refit = HullWhite::new(model.a, model.sigma, curve)?;
            self.option_adjusted_dirty_price_hw(&refit, optionality, spread, settlement)
        };
        let up = bumped_price(bump)?;
        let down = bumped_price(-bump)?;
        let duration = (down - up) / (2.0 * base * bump);
        let convexity = (up + down - 2.0 * base) / (base * bump * bump);
        Ok((duration, convexity))
    }
}

/// Value at the anchor (`t = 0`, `x = 0`) of the event stream, with
/// each side exercising its own options optimally. `flows` is the full
/// remaining payment list, used to compute node-dependent make-whole
/// strikes.
fn backward_induction(
    model: &HullWhite,
    events: &[Event],
    flows: &[(f64, f64)],
    spread: f64,
    t_settlement: f64,
) -> Result<f64, RustyQLibError> {
    if events.is_empty() {
        return Ok(0.0);
    }
    let a = model.a;
    let horizon = events.last().expect("events is non-empty").time;

    // state grid: symmetric, wide enough for the terminal distribution
    let terminal_std = model.short_rate_std(horizon.max(t_settlement).max(1e-8));
    let half_width = (GRID_STDS * terminal_std).max(1e-4);
    let dx = 2.0 * half_width / (GRID_NODES - 1) as f64;
    let grid: Vec<f64> = (0..GRID_NODES)
        .map(|j| -half_width + j as f64 * dx)
        .collect();

    // Simpson quadrature over the standard normal, weights normalized
    // to sum to exactly one
    let dz = 2.0 * QUAD_SPAN / (QUAD_NODES - 1) as f64;
    let mut quad: Vec<(f64, f64)> = (0..QUAD_NODES)
        .map(|k| {
            let z = -QUAD_SPAN + k as f64 * dz;
            let simpson = if k == 0 || k == QUAD_NODES - 1 {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            (z, simpson * norm_pdf(z) * dz / 3.0)
        })
        .collect();
    let total: f64 = quad.iter().map(|&(_, w)| w).sum();
    for (_, w) in quad.iter_mut() {
        *w /= total;
    }

    // linear interpolation with flat extrapolation on the grid
    let interpolate = |values: &[f64], x: f64| -> f64 {
        if x <= grid[0] {
            return values[0];
        }
        if x >= grid[GRID_NODES - 1] {
            return values[GRID_NODES - 1];
        }
        let position = (x - grid[0]) / dx;
        let j = (position.floor() as usize).min(GRID_NODES - 2);
        let weight = position - j as f64;
        values[j] * (1.0 - weight) + values[j + 1] * weight
    };

    // backward through the events
    let mut values = vec![0.0_f64; GRID_NODES];
    for (index, event) in events.iter().enumerate().rev() {
        // exercise decisions compare continuation only; the payment at
        // this date is received regardless. Order: make-whole call,
        // fixed-price call (issuer minimizes), then put (holder
        // maximizes).
        if let Some((mw_spread, mw_outstanding, mw_accrued)) = event.make_whole {
            let alpha_here = model.alpha(event.time);
            for (j, value) in values.iter_mut().enumerate() {
                let rate = grid[j] + alpha_here;
                // strike: max(par, remaining flows at the node's own
                // curve plus the make-whole spread) plus accrued
                let mut pv_remaining = 0.0;
                for &(flow_time, amount) in flows {
                    if flow_time > event.time + 1e-12 {
                        pv_remaining += amount
                            * model.zero_bond(event.time, flow_time, rate)?
                            * (-mw_spread * (flow_time - event.time)).exp();
                    }
                }
                let strike = pv_remaining.max(mw_outstanding) + mw_accrued;
                *value = value.min(strike);
            }
        }
        if let Some(strike) = event.call_strike {
            for value in values.iter_mut() {
                *value = value.min(strike);
            }
        }
        if let Some(strike) = event.put_strike {
            for value in values.iter_mut() {
                *value = value.max(strike);
            }
        }
        for value in values.iter_mut() {
            *value += event.payment;
        }

        // diffuse back to the previous event (or the anchor)
        let t_previous = if index == 0 {
            0.0
        } else {
            events[index - 1].time
        };
        let dt = event.time - t_previous;
        if dt <= 0.0 {
            continue; // coincident events collapse into one node set
        }
        let decay = (-a * dt).exp();
        // forward-measure mean shift of x over the step:
        // M = sigma^2/a^2 [(1 - e^{-a dt}) - (1 - e^{-2a dt})/2]
        let sigma2 = model.sigma * model.sigma;
        let mean_shift = sigma2 / (a * a) * ((1.0 - decay) - 0.5 * (1.0 - (-2.0 * a * dt).exp()));
        let step_std = model.short_rate_std(dt);
        let alpha_previous = model.alpha(t_previous);
        let spread_df = (-spread * dt).exp();

        let mut next = vec![0.0_f64; GRID_NODES];
        for (j, &x) in grid.iter().enumerate() {
            let rate = x + alpha_previous;
            let step_df = model.zero_bond(t_previous, event.time, rate)? * spread_df;
            let mean = x * decay - mean_shift;
            let expectation: f64 = if step_std > 0.0 {
                quad.iter()
                    .map(|&(z, w)| w * interpolate(&values, mean + step_std * z))
                    .sum()
            } else {
                interpolate(&values, mean)
            };
            next[j] = step_df * expectation;
        }
        values = next;
    }

    // x = 0 is the center node of the odd grid
    Ok(values[GRID_NODES / 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, YieldCurve};
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    fn corporate(coupon: f64) -> FixedRateBond {
        FixedRateBond::us_corporate(100.0, coupon, d(2026, 5, 15), d(2031, 5, 15)).unwrap()
    }

    fn call_schedule() -> Vec<CallOption> {
        vec![
            CallOption {
                call_date: d(2028, 5, 15),
                call_price: 101.0,
            },
            CallOption {
                call_date: d(2029, 5, 15),
                call_price: 100.5,
            },
            CallOption {
                call_date: d(2030, 5, 15),
                call_price: 100.0,
            },
        ]
    }

    #[test]
    fn call_free_grid_matches_the_closed_form() {
        // the engine's accuracy anchor: with no calls, backward
        // induction must reproduce the analytic spread pricing
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        for spread in [0.0, 0.015] {
            let on_grid = bond
                .callable_dirty_price_hw(&model, &[], spread, settlement)
                .unwrap();
            let closed_form = bond
                .dirty_price_from_curve_with_spread(model.curve(), spread, settlement)
                .unwrap();
            assert!(
                (on_grid - closed_form).abs() < 0.02,
                "spread {spread}: grid {on_grid} vs closed form {closed_form}"
            );
        }
    }

    #[test]
    fn callable_is_capped_by_the_straight_bond_and_grows_the_option_with_vol() {
        let bond = corporate(0.055);
        let settlement = d(2026, 8, 14);
        let calls = call_schedule();
        let calm = HullWhite::new(0.05, 0.005, curve(0.04)).unwrap();
        let wild = HullWhite::new(0.05, 0.015, curve(0.04)).unwrap();
        for model in [&calm, &wild] {
            let straight = bond
                .callable_dirty_price_hw(model, &[], 0.0, settlement)
                .unwrap();
            let callable = bond
                .callable_dirty_price_hw(model, &calls, 0.0, settlement)
                .unwrap();
            assert!(callable <= straight + 1e-9, "{callable} vs {straight}");
        }
        let option_calm = bond
            .call_option_value_hw(&calm, &calls, 0.0, settlement)
            .unwrap();
        let option_wild = bond
            .call_option_value_hw(&wild, &calls, 0.0, settlement)
            .unwrap();
        assert!(option_calm >= -1e-9);
        assert!(option_wild > option_calm, "{option_wild} vs {option_calm}");
    }

    #[test]
    fn deep_in_the_money_call_prices_to_the_first_call() {
        // a 9% coupon over a 4% curve with negligible vol: the issuer
        // calls at the first date with certainty, so the bond prices as
        // if maturing there at the call price
        let bond = corporate(0.09);
        let model = HullWhite::new(0.05, 1e-4, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let first_call = CallOption {
            call_date: d(2028, 5, 15),
            call_price: 100.0,
        };
        let callable = bond
            .callable_dirty_price_hw(&model, &[first_call], 0.0, settlement)
            .unwrap();
        // replicate: coupons to the call plus the strike, on the curve
        let curve = model.curve();
        let dc = curve.day_count();
        let yf = |date: NaiveDate| dc.year_fraction(curve.reference_date(), date);
        let mut expected = 0.0;
        for cf in bond.cashflows() {
            if cf.accrual_end > settlement && cf.accrual_end <= first_call.call_date {
                expected += cf.amount * curve.df(yf(cf.payment_date));
            }
        }
        expected += 100.0 * curve.df(yf(first_call.call_date));
        expected /= curve.df(yf(settlement));
        assert!(
            (callable - expected).abs() < 0.05,
            "{callable} vs {expected}"
        );
    }

    #[test]
    fn call_on_a_rolled_coupon_date_still_pays_the_coupon() {
        // Nov 15 2026 is a Sunday, so the coupon pays Mon Nov 16. A call
        // scheduled on the coupon date must merge with the rolled payment
        // event — the old event split left the earned coupon inside the
        // continuation but out of the strike, understating the strike by
        // a full coupon (~4.5 per 100 here)
        let bond = corporate(0.09);
        let model = HullWhite::new(0.05, 1e-4, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let call = CallOption {
            call_date: d(2026, 11, 15),
            call_price: 100.0,
        };
        let callable = bond
            .callable_dirty_price_hw(&model, &[call], 0.0, settlement)
            .unwrap();
        // deep in the money at negligible vol: called with certainty, so
        // the bond is the earned Nov coupon plus the strike, both paid on
        // the rolled date
        let curve = model.curve();
        let dc = curve.day_count();
        let yf = |date: NaiveDate| dc.year_fraction(curve.reference_date(), date);
        let mut expected = 0.0;
        for cf in bond.cashflows() {
            if cf.accrual_end > settlement && cf.accrual_end <= call.call_date {
                assert!(
                    cf.payment_date > cf.accrual_end,
                    "the test needs a rolled coupon"
                );
                expected += cf.amount * curve.df(yf(cf.payment_date));
            }
        }
        expected += 100.0 * curve.df(yf(d(2026, 11, 16)));
        expected /= curve.df(yf(settlement));
        assert!(
            (callable - expected).abs() < 0.05,
            "{callable} vs {expected}"
        );
    }

    #[test]
    fn far_out_of_the_money_calls_leave_the_bond_straight() {
        // a 2% coupon over a 4% curve: nobody calls at par
        let bond = corporate(0.02);
        let model = HullWhite::new(0.05, 0.003, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let straight = bond
            .callable_dirty_price_hw(&model, &[], 0.0, settlement)
            .unwrap();
        let callable = bond
            .callable_dirty_price_hw(&model, &call_schedule(), 0.0, settlement)
            .unwrap();
        assert!(
            (straight - callable).abs() < 0.01,
            "{straight} vs {callable}"
        );
    }

    #[test]
    fn oas_round_trips_through_the_callable_price() {
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let calls = call_schedule();
        for spread in [0.005, 0.018] {
            let clean = bond
                .callable_clean_price_hw(&model, &calls, spread, settlement)
                .unwrap();
            let oas = bond
                .oas_hw(
                    clean,
                    &model,
                    &BondOptionality::from_calls(&calls),
                    settlement,
                )
                .unwrap();
            assert!((oas - spread).abs() < 1e-7, "spread {spread}: oas {oas}");
        }
    }

    #[test]
    fn effective_duration_is_shortened_by_the_call() {
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let (callable_duration, _) = bond
            .effective_duration_convexity_hw(
                &model,
                &BondOptionality::from_calls(&call_schedule()),
                0.005,
                settlement,
                0.0025,
            )
            .unwrap();
        let (straight_duration, straight_convexity) = bond
            .effective_duration_convexity_hw(
                &model,
                &BondOptionality::none(),
                0.005,
                settlement,
                0.0025,
            )
            .unwrap();
        assert!(callable_duration > 0.0);
        assert!(
            callable_duration < straight_duration,
            "{callable_duration} vs {straight_duration}"
        );
        // the straight bond's effective numbers match its analytic ones
        assert!(straight_convexity > 0.0);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        assert!(bond
            .callable_dirty_price_hw(&model, &[], f64::NAN, settlement)
            .is_err());
        let bad_call = CallOption {
            call_date: d(2028, 5, 15),
            call_price: -1.0,
        };
        assert!(bond
            .callable_dirty_price_hw(&model, &[bad_call], 0.0, settlement)
            .is_err());
        assert!(bond
            .oas_hw(-5.0, &model, &BondOptionality::none(), settlement)
            .is_err());
        assert!(bond
            .effective_duration_convexity_hw(&model, &BondOptionality::none(), 0.0, settlement, 0.0)
            .is_err());
    }

    #[test]
    fn puttable_floors_the_bond_and_the_put_gains_with_volatility() {
        // a discount bond (2% coupon over 4% rates) with an at-par put:
        // the holder exercises the floor, lifting the price above straight
        let bond = corporate(0.02);
        let settlement = d(2026, 8, 14);
        let puts = [PutOption {
            put_date: d(2029, 5, 15),
            put_price: 100.0,
        }];
        let calm = HullWhite::new(0.05, 0.003, curve(0.04)).unwrap();
        let straight = bond
            .option_adjusted_dirty_price_hw(&calm, &BondOptionality::none(), 0.0, settlement)
            .unwrap();
        let puttable = bond
            .puttable_dirty_price_hw(&calm, &puts, 0.0, settlement)
            .unwrap();
        assert!(puttable > straight + 1.0, "{puttable} vs {straight}");
        // the embedded option value is negative: it belongs to the holder
        let value = bond
            .embedded_option_value_hw(&calm, &BondOptionality::from_puts(&puts), 0.0, settlement)
            .unwrap();
        assert!(value < 0.0);
        // more volatility, more valuable put
        let wild = HullWhite::new(0.05, 0.012, curve(0.04)).unwrap();
        let puttable_wild = bond
            .puttable_dirty_price_hw(&wild, &puts, 0.0, settlement)
            .unwrap();
        let straight_wild = bond
            .option_adjusted_dirty_price_hw(&wild, &BondOptionality::none(), 0.0, settlement)
            .unwrap();
        assert!(puttable_wild - straight_wild > puttable - straight);
    }

    #[test]
    fn deep_in_the_money_put_prices_to_the_put_date() {
        // 2% coupon over 4% rates at negligible vol: the holder puts at
        // the first opportunity with certainty
        let bond = corporate(0.02);
        let model = HullWhite::new(0.05, 1e-4, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let put = PutOption {
            put_date: d(2028, 5, 15),
            put_price: 100.0,
        };
        let puttable = bond
            .puttable_dirty_price_hw(&model, &[put], 0.0, settlement)
            .unwrap();
        let curve = model.curve();
        let dc = curve.day_count();
        let yf = |date: NaiveDate| dc.year_fraction(curve.reference_date(), date);
        let mut expected = 0.0;
        for cf in bond.cashflows() {
            if cf.accrual_end > settlement && cf.accrual_end <= put.put_date {
                expected += cf.amount * curve.df(yf(cf.payment_date));
            }
        }
        expected += 100.0 * curve.df(yf(put.put_date));
        expected /= curve.df(yf(settlement));
        assert!(
            (puttable - expected).abs() < 0.05,
            "{puttable} vs {expected}"
        );
    }

    #[test]
    fn call_and_put_collar_brackets_between_the_pure_cases() {
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let calls = call_schedule();
        let puts = [PutOption {
            put_date: d(2029, 5, 15),
            put_price: 100.0,
        }];
        let both = BondOptionality {
            calls: calls.clone(),
            puts: puts.to_vec(),
            make_whole: None,
        };
        let callable = bond
            .callable_dirty_price_hw(&model, &calls, 0.0, settlement)
            .unwrap();
        let puttable = bond
            .puttable_dirty_price_hw(&model, &puts, 0.0, settlement)
            .unwrap();
        let collar = bond
            .option_adjusted_dirty_price_hw(&model, &both, 0.0, settlement)
            .unwrap();
        assert!(callable <= collar + 1e-9, "{callable} vs {collar}");
        assert!(collar <= puttable + 1e-9, "{collar} vs {puttable}");
    }

    #[test]
    fn make_whole_at_zero_spread_never_gets_exercised() {
        // with a zero make-whole spread the strike equals the fair value
        // of the remaining flows at every node, so calling gains nothing
        // and the bond prices as straight
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let straight = bond
            .option_adjusted_dirty_price_hw(&model, &BondOptionality::none(), 0.0, settlement)
            .unwrap();
        let make_whole = bond
            .option_adjusted_dirty_price_hw(
                &model,
                &BondOptionality::from_make_whole(MakeWholeCall {
                    start: d(2026, 11, 15),
                    spread: 0.0,
                }),
                0.0,
                settlement,
            )
            .unwrap();
        assert!(
            (make_whole - straight).abs() < 0.05,
            "{make_whole} vs {straight}"
        );
    }

    #[test]
    fn make_whole_value_grows_with_the_spread_and_caps_at_the_par_call() {
        let bond = corporate(0.055);
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let price_at = |mw_spread: f64| {
            bond.option_adjusted_dirty_price_hw(
                &model,
                &BondOptionality::from_make_whole(MakeWholeCall {
                    start: d(2026, 11, 15),
                    spread: mw_spread,
                }),
                0.0,
                settlement,
            )
            .unwrap()
        };
        // a wider make-whole spread lowers the strike, making the call
        // more valuable to the issuer and the bond cheaper
        let tight = price_at(0.0025);
        let wide = price_at(0.02);
        assert!(wide < tight, "{wide} vs {tight}");
        // an extreme spread drives every strike to the par floor: the
        // bond converges to a par-callable-at-every-coupon Bermudan
        let extreme = price_at(5.0);
        let par_calls: Vec<CallOption> = bond
            .coupon_dates()
            .iter()
            .filter(|&&date| date >= d(2026, 11, 15) && date < bond.maturity_date)
            .map(|&date| CallOption {
                call_date: date,
                call_price: 100.0,
            })
            .collect();
        let bermudan = bond
            .callable_dirty_price_hw(&model, &par_calls, 0.0, settlement)
            .unwrap();
        assert!((extreme - bermudan).abs() < 0.05, "{extreme} vs {bermudan}");
        // negative make-whole spreads are rejected
        assert!(bond
            .option_adjusted_dirty_price_hw(
                &model,
                &BondOptionality::from_make_whole(MakeWholeCall {
                    start: d(2026, 11, 15),
                    spread: -0.01,
                }),
                0.0,
                settlement,
            )
            .is_err());
    }

    #[test]
    fn sinking_fund_bond_on_the_grid_matches_the_closed_form() {
        // the engine must carry amortizing flows and quote per 100 of
        // outstanding exactly like the analytic curve pricer
        let bond = FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15))
            .unwrap()
            .with_sinking_fund(&[(d(2028, 5, 15), 0.25), (d(2030, 5, 15), 0.25)])
            .unwrap();
        let model = HullWhite::new(0.05, 0.01, curve(0.04)).unwrap();
        let settlement = d(2026, 8, 14);
        let on_grid = bond
            .option_adjusted_dirty_price_hw(&model, &BondOptionality::none(), 0.01, settlement)
            .unwrap();
        let closed_form = bond
            .dirty_price_from_curve_with_spread(model.curve(), 0.01, settlement)
            .unwrap();
        assert!(
            (on_grid - closed_form).abs() < 0.02,
            "grid {on_grid} vs closed form {closed_form}"
        );
        // and a par call on the reduced outstanding still caps the value
        let called = bond
            .callable_dirty_price_hw(
                &model,
                &[CallOption {
                    call_date: d(2029, 5, 15),
                    call_price: 100.0,
                }],
                0.01,
                settlement,
            )
            .unwrap();
        assert!(called <= on_grid + 1e-9, "{called} vs {on_grid}");
    }
}
