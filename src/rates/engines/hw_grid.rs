//! Backward induction on the Hull-White state grid — the engine behind
//! Bermudan swaptions and callable bonds.
//!
//! The state is `x(t) = r(t) - alpha(t)`, an Ornstein-Uhlenbeck process
//! starting at zero, on a fixed symmetric grid wide enough for the
//! terminal distribution. The induction steps directly **between event
//! dates** — exercise decisions, coupon payments — with no intermediate
//! time discretization:
//!
//! - discounting over a step uses the model's exact zero-bond price
//!   `P(t, t+dt | r)` (times `e^{-spread dt}` for an OAS-style spread),
//! - the state transition is the exact Gaussian law of `x` under the
//!   `t+dt`-forward measure — mean `x e^{-a dt} - M(t, t+dt)`, the
//!   measure change that makes bond-discounted expectations consistent,
//!   variance `V(t, t+dt)` — integrated by Simpson quadrature over the
//!   standard normal with linear interpolation on the grid,
//!
//! so the only numerical error is quadrature and grid interpolation.
//! The product supplies what happens **at** each event through a
//! closure `(event, short rate, continuation) -> value`: a Bermudan
//! takes `max(continuation, exercise value)`, a callable bond floors
//! and caps at its strikes and adds the coupon.

use crate::core::errors::RustyQLibError;
use crate::core::utils::norm_pdf;
use crate::rates::models::{HullWhite, ShortRateModel};

/// Grid and quadrature resolution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridConfig {
    /// State nodes (odd, so `x = 0` is a node).
    pub nodes: usize,
    /// Grid half-width in terminal standard deviations of the short rate.
    pub stds: f64,
    /// Quadrature nodes over the standard normal (odd, Simpson).
    pub quad_nodes: usize,
    /// Quadrature span in standard deviations.
    pub quad_span: f64,
    /// Lower bound on the horizon used to size the grid (year
    /// fractions) — for event streams that start late.
    pub min_horizon: f64,
}

impl Default for GridConfig {
    fn default() -> Self {
        GridConfig {
            nodes: 201,
            stds: 7.0,
            quad_nodes: 101,
            quad_span: 8.0,
            min_horizon: 0.0,
        }
    }
}

impl GridConfig {
    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.nodes < 3 || self.nodes % 2 == 0 {
            return Err(RustyQLibError::invalid_input(
                "grid",
                format!("nodes must be odd and at least 3, got {}", self.nodes),
            ));
        }
        if self.quad_nodes < 3 || self.quad_nodes % 2 == 0 {
            return Err(RustyQLibError::invalid_input(
                "grid",
                format!(
                    "quadrature nodes must be odd and at least 3, got {}",
                    self.quad_nodes
                ),
            ));
        }
        if !(self.stds > 0.0 && self.quad_span > 0.0 && self.min_horizon >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "grid",
                "stds and quad_span must be positive, min_horizon non-negative",
            ));
        }
        Ok(())
    }
}

/// Value at the anchor (`t = 0`, `x = 0`) of an event stream under
/// `model`. `times` are the event times, ascending (coincident events
/// share a node set); `at_event(k, r, continuation)` returns the value
/// at event `k` given the short rate `r` at the node and the value of
/// continuing (zero after the last event). `spread` is a continuous
/// spread over the model curve applied in discounting.
pub fn backward_induction(
    model: &HullWhite,
    times: &[f64],
    spread: f64,
    mut at_event: impl FnMut(usize, f64, f64) -> Result<f64, RustyQLibError>,
    config: &GridConfig,
) -> Result<f64, RustyQLibError> {
    config.validate()?;
    if times.is_empty() {
        return Ok(0.0);
    }
    for pair in times.windows(2) {
        if pair[1] < pair[0] {
            return Err(RustyQLibError::invalid_input(
                "grid",
                format!(
                    "event times must be ascending, got {} after {}",
                    pair[1], pair[0]
                ),
            ));
        }
    }
    if times[0] < 0.0 {
        return Err(RustyQLibError::invalid_input(
            "grid",
            format!("event times must be non-negative, got {}", times[0]),
        ));
    }
    let nodes = config.nodes;
    let horizon = times
        .last()
        .copied()
        .unwrap_or(0.0)
        .max(config.min_horizon)
        .max(1e-8);

    // state grid: symmetric, wide enough for the terminal distribution
    let terminal_std = model.short_rate_std(0.0, horizon);
    let half_width = (config.stds * terminal_std).max(1e-4);
    let dx = 2.0 * half_width / (nodes - 1) as f64;
    let grid: Vec<f64> = (0..nodes).map(|j| -half_width + j as f64 * dx).collect();

    // Simpson quadrature over the standard normal, weights normalized
    // to sum to exactly one
    let quad_nodes = config.quad_nodes;
    let dz = 2.0 * config.quad_span / (quad_nodes - 1) as f64;
    let mut quad: Vec<(f64, f64)> = (0..quad_nodes)
        .map(|k| {
            let z = -config.quad_span + k as f64 * dz;
            let simpson = if k == 0 || k == quad_nodes - 1 {
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
        if x >= grid[nodes - 1] {
            return values[nodes - 1];
        }
        let position = (x - grid[0]) / dx;
        let j = (position.floor() as usize).min(nodes - 2);
        let weight = position - j as f64;
        values[j] * (1.0 - weight) + values[j + 1] * weight
    };

    let mut values = vec![0.0_f64; nodes];
    for (index, &time) in times.iter().enumerate().rev() {
        let alpha_here = model.alpha(time);
        for (j, value) in values.iter_mut().enumerate() {
            *value = at_event(index, grid[j] + alpha_here, *value)?;
        }

        // diffuse back to the previous event (or the anchor)
        let t_previous = if index == 0 { 0.0 } else { times[index - 1] };
        let dt = time - t_previous;
        if dt <= 0.0 {
            continue; // coincident events collapse into one node set
        }
        let decay = model.decay(t_previous, time);
        let mean_shift = model.forward_measure_shift(t_previous, time);
        let step_std = model.short_rate_std(t_previous, dt);
        let alpha_previous = model.alpha(t_previous);
        let spread_df = (-spread * dt).exp();

        let mut next = vec![0.0_f64; nodes];
        for (j, &x) in grid.iter().enumerate() {
            let rate = x + alpha_previous;
            let step_df = model.zero_bond(t_previous, time, rate)? * spread_df;
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
    Ok(values[nodes / 2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::core::trade::PutOrCall;
    use crate::rates::models::OneFactorAffine;
    use chrono::NaiveDate;

    fn model(sigma: f64) -> HullWhite {
        let curve = YieldCurve::flat(
            0.04,
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        HullWhite::new(0.05, sigma, curve).unwrap()
    }

    #[test]
    fn a_zero_bond_paid_at_the_last_event_reprices_the_curve() {
        // pay 1 at 7y, roll back through three intermediate events
        for m in [model(0.01), model(0.0)] {
            let value = backward_induction(
                &m,
                &[1.0, 3.0, 5.0, 7.0],
                0.0,
                |k, _, continuation| Ok(if k == 3 { 1.0 } else { continuation }),
                &GridConfig::default(),
            )
            .unwrap();
            let df = m.curve().df(7.0);
            assert!((value - df).abs() < 2e-5, "{value} vs {df}");
        }
    }

    #[test]
    fn a_single_exercise_reproduces_the_analytic_zero_bond_option() {
        // European put on the 6y bond, exercised at 2y: the grid matches
        // Jamshidian's closed form
        let m = model(0.012);
        let (expiry, maturity, strike) = (2.0, 6.0, 0.85);
        let analytic = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let on_grid = backward_induction(
            &m,
            &[expiry],
            0.0,
            |_, rate, _| Ok((strike - m.zero_bond(expiry, maturity, rate)?).max(0.0)),
            &GridConfig::default(),
        )
        .unwrap();
        assert!(
            (on_grid - analytic).abs() < 2e-3 * analytic,
            "grid {on_grid} vs analytic {analytic}"
        );
        // and a piecewise sigma flows through the step variance and shift
        let pw = HullWhite::with_piecewise_sigma(0.05, &[1.0], &[0.015, 0.009], m.curve().clone())
            .unwrap();
        let analytic = pw
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let on_grid = backward_induction(
            &pw,
            &[1.0, expiry],
            0.0,
            |k, rate, continuation| {
                Ok(if k == 1 {
                    (strike - pw.zero_bond(expiry, maturity, rate)?).max(0.0)
                } else {
                    continuation
                })
            },
            &GridConfig::default(),
        )
        .unwrap();
        assert!(
            (on_grid - analytic).abs() < 2e-3 * analytic,
            "piecewise grid {on_grid} vs analytic {analytic}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let m = model(0.01);
        let ok = |_: usize, _: f64, c: f64| Ok(c);
        assert!(backward_induction(&m, &[2.0, 1.0], 0.0, ok, &GridConfig::default()).is_err());
        assert!(backward_induction(&m, &[-1.0], 0.0, ok, &GridConfig::default()).is_err());
        let even = GridConfig {
            nodes: 200,
            ..Default::default()
        };
        assert!(backward_induction(&m, &[1.0], 0.0, ok, &even).is_err());
        assert_eq!(
            backward_induction(&m, &[], 0.0, ok, &GridConfig::default()).unwrap(),
            0.0
        );
    }
}
