//! The CRR binomial tree engine, generic over the instrument and the
//! credit model: CRR spacing in the share price, the risk-neutral
//! drift from the discount curve's own forward factors per step (plus
//! the model's survival drift), and the model's discounting of each
//! node.

use chrono::NaiveDate;

use super::credit::{CreditModel, NodeValue};
use super::events::apply_cash_dividend;
use super::instrument::ConvertibleInstrument;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// CRR risk-neutral up probability for a step with the given growth
/// factor (conditional on no default), or an error when the step is
/// too coarse for the drift.
fn crr_probability(growth: f64, up: f64, down: f64, step: usize) -> Result<f64, RustyQLibError> {
    let p = (growth - down) / (up - down);
    if !(0.0..=1.0).contains(&p) {
        return Err(RustyQLibError::NumericalError(format!(
            "risk-neutral probability {p} outside [0, 1] at step {step}; \
             increase the tree steps or check the inputs"
        )));
    }
    Ok(p)
}

/// The backward induction. Returns the dirty value in the instrument's
/// price units at `settlement`.
pub(crate) fn tree_value<I: ConvertibleInstrument + ?Sized, M: CreditModel>(
    instrument: &I,
    market: &M,
    curve: &YieldCurve,
    settlement: NaiveDate,
    steps: usize,
) -> Result<f64, RustyQLibError> {
    instrument.validate()?;
    let grid = instrument.event_grid(curve, settlement, steps, market.credit_rate())?;
    let equity = market.equity();
    let dt = grid.dt;
    let up = (equity.volatility * dt.sqrt()).exp();
    let down = 1.0 / up;
    // the log-price width of one node cell, for a smoothed soft trigger
    let cell_width = 2.0 * up.ln();
    let credit_df = (-market.credit_rate() * dt).exp();
    let drift = (market.survival_drift() * dt).exp();

    // per-step risk-neutral probabilities from the curve's forwards
    let probability = (0..steps)
        .map(|i| crr_probability(drift / grid.riskfree_df[i], up, down, i))
        .collect::<Result<Vec<f64>, _>>()?;

    let spot_at =
        |step: usize, j: usize| equity.spot * up.powi(j as i32) * down.powi((step - j) as i32);

    let mut nodes: Vec<M::Node> = (0..=steps)
        .map(|j| grid.terminal(instrument, spot_at(steps, j), cell_width))
        .collect();

    // backward induction: expectation and discounting, the ex-dividend
    // jump, then the step's events
    for step in (0..steps).rev() {
        let p = probability[step];
        let riskfree_df = grid.riskfree_df[step];
        let default_claim = grid.default_claim_at_step[step];
        for j in 0..=step {
            let expected = M::Node::blend(nodes[j], nodes[j + 1], p);
            nodes[j] = market.discount_step(expected, riskfree_df, credit_df, default_claim);
        }
        nodes.truncate(step + 1);
        let dividend = grid.cash_dividend_at_step[step];
        if dividend > 0.0 {
            let spots: Vec<f64> = (0..=step).map(|j| spot_at(step, j)).collect();
            apply_cash_dividend(&mut nodes, &spots, dividend);
        }
        for (j, slot) in nodes.iter_mut().enumerate() {
            let node = slot.plus_cash(grid.coupon_at_step[step]);
            *slot = grid
                .exercise(instrument, step, spot_at(step, j), node, cell_width)
                .plus_cash(grid.coupon_kept_at_step[step]);
        }
    }

    Ok(nodes[0].total() * 100.0 / grid.outstanding)
}
