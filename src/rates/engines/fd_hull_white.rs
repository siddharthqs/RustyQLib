//! Finite-difference engine under Hull-White (QuantLib's
//! `FdHullWhiteSwaptionEngine`): the pricing PDE in the state
//! `x = r - alpha(t)`, solved backward between event dates.
//!
//! ```text
//! V_t + 1/2 sigma(t)^2 V_xx - a(t) x V_x - (x + alpha(t) + spread) V = 0
//! ```
//!
//! A theta scheme in time (Crank-Nicolson, with fully implicit
//! Rannacher steps after every event to damp the kinks that exercise
//! and coupons introduce), central differences inside, one-sided drift
//! and no diffusion on the two far boundaries. Events enter through the
//! same closure as the [`hw_grid`] engine — `(event, short rate,
//! continuation) -> value` — so a European, a Bermudan and a callable
//! bond are the same call with different closures, and the two engines
//! cross-check each other: the grid integrates the exact transition
//! between events, the PDE marches through them.
//!
//! [`hw_grid`]: crate::rates::engines::hw_grid

use crate::core::errors::RustyQLibError;
use crate::core::fd_solvers::adi::douglas_step;
use crate::core::fd_solvers::axis_operator::{AxisOperator, TensorGrid};
use crate::rates::engines::hw_grid::GridConfig;
use crate::rates::models::black_karasinski::TailSwap;
use crate::rates::models::{HullWhite, ShortRateModel};
use crate::rates::PayerReceiver;

const FIELD: &str = "fd hull-white";

/// Resolution of the finite-difference solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdConfig {
    /// State nodes (odd, so `x = 0` is a node).
    pub nodes: usize,
    /// Grid half-width in terminal standard deviations of the state.
    pub stds: f64,
    /// Time steps per year between events (at least one per interval).
    pub steps_per_year: usize,
    /// Fully implicit steps taken right after each event (Rannacher).
    pub rannacher_steps: usize,
    /// Lower bound on the horizon used to size the grid.
    pub min_horizon: f64,
}

impl Default for FdConfig {
    fn default() -> Self {
        FdConfig {
            nodes: 401,
            stds: 7.0,
            steps_per_year: 48,
            rannacher_steps: 2,
            min_horizon: 0.0,
        }
    }
}

impl From<GridConfig> for FdConfig {
    fn from(g: GridConfig) -> Self {
        FdConfig {
            nodes: g.nodes,
            stds: g.stds,
            min_horizon: g.min_horizon,
            ..FdConfig::default()
        }
    }
}

impl FdConfig {
    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.nodes < 5 || self.nodes % 2 == 0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("nodes must be odd and at least 5, got {}", self.nodes),
            ));
        }
        if !(self.stds > 0.0) || self.steps_per_year == 0 || !(self.min_horizon >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "stds must be positive, steps_per_year at least one, min_horizon non-negative",
            ));
        }
        Ok(())
    }
}

/// Value at the anchor of an event stream under `model` by finite
/// differences; see the module docs. `times` ascending, `spread` a
/// continuous spread over the curve in discounting, `at_event(k, r,
/// continuation)` the value at event `k` at short rate `r`.
pub fn backward_induction(
    model: &HullWhite,
    times: &[f64],
    spread: f64,
    mut at_event: impl FnMut(usize, f64, f64) -> Result<f64, RustyQLibError>,
    config: &FdConfig,
) -> Result<f64, RustyQLibError> {
    config.validate()?;
    if times.is_empty() {
        return Ok(0.0);
    }
    for pair in times.windows(2) {
        if pair[1] < pair[0] {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "event times must be ascending",
            ));
        }
    }
    if times[0] < 0.0 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "event times must be non-negative",
        ));
    }
    let nodes = config.nodes;
    let horizon = times
        .last()
        .copied()
        .unwrap_or(0.0)
        .max(config.min_horizon)
        .max(1e-8);
    let half_width = (config.stds * model.short_rate_std(0.0, horizon)).max(1e-4);
    let dx = 2.0 * half_width / (nodes - 1) as f64;
    let x_grid: Vec<f64> = (0..nodes).map(|j| -half_width + j as f64 * dx).collect();
    let grid = TensorGrid::new(&[nodes]);

    // the spatial operator at time t: 1/2 sigma^2 u_xx - a x u_x - r u
    let operator = |t: f64| -> AxisOperator {
        let sigma = model.sigma_at(t);
        let a = model.a_at(t);
        let alpha = model.alpha(t);
        let diffusion = 0.5 * sigma * sigma / (dx * dx);
        let mut op = AxisOperator::zero(&grid, 0);
        for (j, &x) in x_grid.iter().enumerate() {
            let drift = -a * x;
            let rate = x + alpha + spread;
            if j == 0 {
                // far low boundary: no diffusion, upwind (forward) drift
                op.diag[j] = -drift / dx - rate;
                op.sup[j] = drift / dx;
            } else if j == nodes - 1 {
                op.diag[j] = drift / dx - rate;
                op.sub[j] = -drift / dx;
            } else {
                op.sub[j] = diffusion - drift / (2.0 * dx);
                op.diag[j] = -2.0 * diffusion - rate;
                op.sup[j] = diffusion + drift / (2.0 * dx);
            }
        }
        op
    };

    let mut values = vec![0.0_f64; nodes];
    for (index, &time) in times.iter().enumerate().rev() {
        let alpha_here = model.alpha(time);
        for (j, value) in values.iter_mut().enumerate() {
            *value = at_event(index, x_grid[j] + alpha_here, *value)?;
        }
        let t_previous = if index == 0 { 0.0 } else { times[index - 1] };
        let span = time - t_previous;
        if span <= 0.0 {
            continue;
        }
        let steps = ((span * config.steps_per_year as f64).ceil() as usize).max(1);
        let dt = span / steps as f64;
        for step in 0..steps {
            // coefficients at the midpoint of the step
            let t_mid = time - (step as f64 + 0.5) * dt;
            let op = operator(t_mid);
            let theta = if step < config.rannacher_steps {
                1.0
            } else {
                0.5
            };
            values = douglas_step(&grid, &[op], None, &values, dt, theta);
        }
    }
    Ok(values[nodes / 2])
}

/// European swaption by finite differences (the same inputs as the
/// analytic engines).
pub fn european_swaption(
    model: &HullWhite,
    expiry: f64,
    swap_start: f64,
    fixed_leg: &[(f64, f64)],
    strike_rate: f64,
    notional: f64,
    payer_receiver: PayerReceiver,
    config: &FdConfig,
) -> Result<f64, RustyQLibError> {
    let Some(&(last, _)) = fixed_leg.last() else {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "the fixed leg has no payments",
        ));
    };
    let tail = TailSwap {
        expiry,
        start: swap_start,
        coupons: fixed_leg
            .iter()
            .map(|&(t, tau)| (t, notional * strike_rate * tau))
            .collect(),
        last,
    };
    bermudan_swaption(model, &[tail], notional, payer_receiver, config)
}

/// Bermudan swaption by finite differences: on each `tails[k].expiry`
/// the holder may enter that tail.
pub fn bermudan_swaption(
    model: &HullWhite,
    tails: &[TailSwap],
    notional: f64,
    payer_receiver: PayerReceiver,
    config: &FdConfig,
) -> Result<f64, RustyQLibError> {
    if tails.is_empty() {
        return Err(RustyQLibError::invalid_input(FIELD, "no exercise dates"));
    }
    if !(notional > 0.0) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "notional must be positive",
        ));
    }
    let times: Vec<f64> = tails.iter().map(|t| t.expiry).collect();
    let sign = match payer_receiver {
        PayerReceiver::Payer => 1.0,
        PayerReceiver::Receiver => -1.0,
    };
    backward_induction(
        model,
        &times,
        0.0,
        |k, rate, continuation| {
            let e = &tails[k];
            let float = notional
                * (model.zero_bond(e.expiry, e.start, rate)?
                    - model.zero_bond(e.expiry, e.last, rate)?);
            let mut fixed = 0.0;
            for &(pay, amount) in &e.coupons {
                fixed += amount * model.zero_bond(e.expiry, pay, rate)?;
            }
            Ok(continuation.max(sign * (float - fixed)))
        },
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::contracts::{BermudanSwaption, VanillaSwap};
    use crate::rates::engines::jamshidian::european_swaption_settled;
    use crate::rates::models::OneFactorAffine;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    #[test]
    fn a_zero_bond_paid_at_the_last_event_reprices_the_curve() {
        for m in [
            HullWhite::new(0.05, 0.011, market_curve()).unwrap(),
            HullWhite::generalized(
                &[2.0],
                &[0.03, 0.12],
                &[1.0],
                &[0.012, 0.008],
                market_curve(),
            )
            .unwrap(),
        ] {
            let value = backward_induction(
                &m,
                &[1.0, 3.0, 6.0],
                0.0,
                |k, _, continuation| Ok(if k == 2 { 1.0 } else { continuation }),
                &FdConfig::default(),
            )
            .unwrap();
            let df = m.curve().df(6.0);
            assert!((value / df - 1.0).abs() < 2e-4, "{value} vs {df}");
        }
    }

    #[test]
    fn european_swaptions_match_jamshidian() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let leg: Vec<(f64, f64)> = (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect();
        for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
            let fd = european_swaption(
                &m,
                1.0,
                1.0,
                &leg,
                0.045,
                1_000_000.0,
                side,
                &FdConfig::default(),
            )
            .unwrap();
            let analytic =
                european_swaption_settled(&m, 1.0, 1.0, &leg, 0.045, 1_000_000.0, side).unwrap();
            assert!(
                (fd - analytic).abs() < 2e-3 * analytic,
                "{side:?}: FD {fd} vs {analytic}"
            );
        }
        // a zero-bond put through the same machinery
        let put = backward_induction(
            &m,
            &[2.0],
            0.0,
            |_, rate, _| Ok((0.85 - m.zero_bond(2.0, 6.0, rate)?).max(0.0)),
            &FdConfig::default(),
        )
        .unwrap();
        let analytic = m
            .zero_bond_option(2.0, 6.0, 0.85, crate::core::trade::PutOrCall::Put)
            .unwrap();
        assert!(
            (put - analytic).abs() < 2e-3 * analytic,
            "{put} vs {analytic}"
        );
    }

    #[test]
    fn bermudans_match_the_integration_grid() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let swap = VanillaSwap::usd_standard(
            10_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 8, 17),
            d(2032, 8, 17),
        )
        .unwrap();
        let b = BermudanSwaption::on_fixed_period_starts(swap, d(2027, 8, 1)).unwrap();
        let on_grid = b.npv_hull_white(&m, &GridConfig::default()).unwrap();
        let by_fd = b.npv_fd_hull_white(&m, &FdConfig::default()).unwrap();
        assert!(
            (on_grid - by_fd).abs() < 3e-3 * on_grid,
            "grid {on_grid} vs FD {by_fd}"
        );
        let best = b
            .european_values_hull_white(&m)
            .unwrap()
            .into_iter()
            .fold(f64::MIN, f64::max);
        assert!(by_fd > best);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let ok = |_: usize, _: f64, c: f64| Ok(c);
        assert!(backward_induction(&m, &[2.0, 1.0], 0.0, ok, &FdConfig::default()).is_err());
        let even = FdConfig {
            nodes: 400,
            ..FdConfig::default()
        };
        assert!(backward_induction(&m, &[1.0], 0.0, ok, &even).is_err());
        assert!(
            bermudan_swaption(&m, &[], 1.0, PayerReceiver::Payer, &FdConfig::default()).is_err()
        );
    }
}
