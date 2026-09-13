//! Finite-difference engine under G2++ (QuantLib's
//! `FdG2SwaptionEngine`): the two-factor pricing PDE solved by ADI,
//! backward between event dates.
//!
//! ```text
//! V_t + 1/2 sigma^2 V_xx + 1/2 eta^2 V_yy + rho sigma eta V_xy
//!     - a x V_x - b y V_y - (x + y + phi(t)) V = 0
//! ```
//!
//! Each factor gets a tridiagonal operator carrying half the discount;
//! the mixed derivative is treated explicitly. Hundsdorfer-Verwer
//! steps (second order with the mixed term) after fully implicit
//! Douglas steps following every event (Rannacher), central
//! differences inside, one-sided drift and no diffusion on the far
//! boundaries. This is the only Bermudan engine the two-factor model
//! has — Jamshidian and the one-dimensional grids need a single state —
//! so it is what prices a G2++ Bermudan against the analytic
//! Europeans. With `eta = 0` the `y` axis collapses to one node and the
//! solve is the Hull-White PDE, the cross-check used in the tests.

use crate::core::errors::RustyQLibError;
use crate::core::fd_solvers::adi::{douglas_step, hundsdorfer_verwer_step};
use crate::core::fd_solvers::axis_operator::{AxisOperator, TensorGrid};
use crate::rates::models::black_karasinski::TailSwap;
use crate::rates::models::g2pp::G2pp;
use crate::rates::PayerReceiver;

const FIELD: &str = "fd g2++";
/// Hundsdorfer-Verwer theta, `1/2 + sqrt(3)/6`, unconditionally stable
/// with the explicit mixed term.
const HV_THETA: f64 = 0.788_675_134_594_812_9;
const HV_MU: f64 = 0.5;

/// Resolution of the two-dimensional solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdG2Config {
    /// Nodes on the `x` axis (odd).
    pub nodes_x: usize,
    /// Nodes on the `y` axis (odd).
    pub nodes_y: usize,
    /// Half-width of each axis in terminal standard deviations.
    pub stds: f64,
    /// Time steps per year between events.
    pub steps_per_year: usize,
    /// Fully implicit steps after each event.
    pub rannacher_steps: usize,
}

impl Default for FdG2Config {
    fn default() -> Self {
        FdG2Config {
            nodes_x: 101,
            nodes_y: 101,
            stds: 6.0,
            steps_per_year: 24,
            rannacher_steps: 2,
        }
    }
}

impl FdG2Config {
    fn validate(&self) -> Result<(), RustyQLibError> {
        for (name, n) in [("nodes_x", self.nodes_x), ("nodes_y", self.nodes_y)] {
            if n < 3 || n % 2 == 0 {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("{name} must be odd and at least 3, got {n}"),
                ));
            }
        }
        if !(self.stds > 0.0) || self.steps_per_year == 0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "stds must be positive and steps_per_year at least one",
            ));
        }
        Ok(())
    }
}

/// Value at the anchor of an event stream under `model`. `times`
/// ascending; `at_event(k, x, y, continuation)` the value at event `k`
/// at factor state `(x, y)`.
pub fn backward_induction(
    model: &G2pp,
    times: &[f64],
    mut at_event: impl FnMut(usize, f64, f64, f64) -> Result<f64, RustyQLibError>,
    config: &FdG2Config,
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
    let horizon = times.last().copied().unwrap_or(0.0).max(1e-8);
    let (var_x, var_y, _) = model.step_covariance(horizon);
    // an axis with no variance collapses to its single node at zero
    let axis = |nodes: usize, std: f64| -> (Vec<f64>, f64) {
        if std < 1e-12 {
            return (vec![0.0], 1.0);
        }
        let half = config.stds * std;
        let d = 2.0 * half / (nodes - 1) as f64;
        ((0..nodes).map(|k| -half + k as f64 * d).collect(), d)
    };
    let (x_grid, dx) = axis(config.nodes_x, var_x.sqrt());
    let (y_grid, dy) = axis(config.nodes_y, var_y.sqrt());
    let (nx, ny) = (x_grid.len(), y_grid.len());
    let grid = TensorGrid::new(&[nx, ny]);
    let at = |i: usize, j: usize| i * ny + j;

    let (a, b, sigma, eta, rho) = (model.a, model.b, model.sigma, model.eta, model.rho);
    let build = |t: f64| -> (AxisOperator, AxisOperator) {
        let phi = model.phi(t);
        let mut a_x = AxisOperator::zero(&grid, 0);
        let mut a_y = AxisOperator::zero(&grid, 1);
        let d_x = 0.5 * sigma * sigma / (dx * dx);
        let d_y = 0.5 * eta * eta / (dy * dy);
        for (i, &x) in x_grid.iter().enumerate() {
            for (j, &y) in y_grid.iter().enumerate() {
                let idx = at(i, j);
                let half_rate = 0.5 * (x + y + phi);
                let drift_x = -a * x;
                let drift_y = -b * y;
                if nx == 1 {
                    a_x.diag[idx] = -half_rate;
                } else if i == 0 {
                    a_x.diag[idx] = -drift_x / dx - half_rate;
                    a_x.sup[idx] = drift_x / dx;
                } else if i == nx - 1 {
                    a_x.diag[idx] = drift_x / dx - half_rate;
                    a_x.sub[idx] = -drift_x / dx;
                } else {
                    a_x.sub[idx] = d_x - drift_x / (2.0 * dx);
                    a_x.diag[idx] = -2.0 * d_x - half_rate;
                    a_x.sup[idx] = d_x + drift_x / (2.0 * dx);
                }
                if ny == 1 {
                    a_y.diag[idx] = -half_rate;
                } else if j == 0 {
                    a_y.diag[idx] = -drift_y / dy - half_rate;
                    a_y.sup[idx] = drift_y / dy;
                } else if j == ny - 1 {
                    a_y.diag[idx] = drift_y / dy - half_rate;
                    a_y.sub[idx] = -drift_y / dy;
                } else {
                    a_y.sub[idx] = d_y - drift_y / (2.0 * dy);
                    a_y.diag[idx] = -2.0 * d_y - half_rate;
                    a_y.sup[idx] = d_y + drift_y / (2.0 * dy);
                }
            }
        }
        (a_x, a_y)
    };
    let cross = rho * sigma * eta;
    let mixed = |u: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; u.len()];
        if cross == 0.0 || nx < 3 || ny < 3 {
            return out;
        }
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let idx = at(i, j);
                let d2 = (u[idx + ny + 1] - u[idx + ny - 1] - u[idx - ny + 1] + u[idx - ny - 1])
                    / (4.0 * dx * dy);
                out[idx] = cross * d2;
            }
        }
        out
    };

    let mut values = vec![0.0_f64; grid.len()];
    for (index, &time) in times.iter().enumerate().rev() {
        for (i, &x) in x_grid.iter().enumerate() {
            for (j, &y) in y_grid.iter().enumerate() {
                let idx = at(i, j);
                values[idx] = at_event(index, x, y, values[idx])?;
            }
        }
        let t_previous = if index == 0 { 0.0 } else { times[index - 1] };
        let span = time - t_previous;
        if span <= 0.0 {
            continue;
        }
        let steps = ((span * config.steps_per_year as f64).ceil() as usize).max(1);
        let dt = span / steps as f64;
        for step in 0..steps {
            let t_mid = time - (step as f64 + 0.5) * dt;
            let (a_x, a_y) = build(t_mid);
            let ops = [a_x, a_y];
            values = if step < config.rannacher_steps {
                douglas_step(&grid, &ops, Some(&mixed), &values, dt, 1.0)
            } else {
                hundsdorfer_verwer_step(&grid, &ops, Some(&mixed), &values, dt, HV_THETA, HV_MU)
            };
        }
    }
    Ok(values[at(nx / 2, ny / 2)])
}

fn side_sign(payer_receiver: PayerReceiver) -> f64 {
    match payer_receiver {
        PayerReceiver::Payer => 1.0,
        PayerReceiver::Receiver => -1.0,
    }
}

/// Bermudan swaption by ADI finite differences: on each
/// `tails[k].expiry` the holder may enter that tail.
pub fn bermudan_swaption(
    model: &G2pp,
    tails: &[TailSwap],
    notional: f64,
    payer_receiver: PayerReceiver,
    config: &FdG2Config,
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
    let sign = side_sign(payer_receiver);
    backward_induction(
        model,
        &times,
        |k, x, y, continuation| {
            let e = &tails[k];
            let float = notional
                * (model.zero_bond(e.expiry, e.start, x, y)?
                    - model.zero_bond(e.expiry, e.last, x, y)?);
            let mut fixed = 0.0;
            for &(pay, amount) in &e.coupons {
                fixed += amount * model.zero_bond(e.expiry, pay, x, y)?;
            }
            Ok(continuation.max(sign * (float - fixed)))
        },
        config,
    )
}

/// European swaption by ADI finite differences.
pub fn european_swaption(
    model: &G2pp,
    expiry: f64,
    swap_start: f64,
    fixed_leg: &[(f64, f64)],
    strike_rate: f64,
    notional: f64,
    payer_receiver: PayerReceiver,
    config: &FdG2Config,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::contracts::{BermudanSwaption, VanillaSwap};
    use crate::rates::engines::hw_grid::GridConfig;
    use crate::rates::models::HullWhite;
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

    fn model() -> G2pp {
        G2pp::new(0.1, 0.5, 0.008, 0.012, -0.7, market_curve()).unwrap()
    }

    fn fixed_leg() -> Vec<(f64, f64)> {
        (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect()
    }

    #[test]
    fn a_zero_bond_paid_at_the_last_event_reprices_the_curve() {
        let m = model();
        let value = backward_induction(
            &m,
            &[1.0, 3.0, 6.0],
            |k, _, _, continuation| Ok(if k == 2 { 1.0 } else { continuation }),
            &FdG2Config::default(),
        )
        .unwrap();
        let df = m.curve().df(6.0);
        assert!((value / df - 1.0).abs() < 5e-4, "{value} vs {df}");
    }

    #[test]
    fn european_swaptions_match_the_analytic_integral() {
        let m = model();
        let leg = fixed_leg();
        for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
            let fd = european_swaption(
                &m,
                1.0,
                1.0,
                &leg,
                0.045,
                1_000_000.0,
                side,
                &FdG2Config::default(),
            )
            .unwrap();
            let analytic = m
                .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, side)
                .unwrap();
            assert!(
                (fd - analytic).abs() < 5e-3 * analytic,
                "{side:?}: FD {fd} vs analytic {analytic}"
            );
        }
    }

    #[test]
    fn with_one_factor_the_bermudan_matches_the_hull_white_grid() {
        let curve = market_curve();
        let g2 = G2pp::new(0.05, 0.5, 0.011, 0.0, 0.0, curve.clone()).unwrap();
        let hw = HullWhite::new(0.05, 0.011, curve).unwrap();
        let swap = VanillaSwap::usd_standard(
            10_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 8, 17),
            d(2032, 8, 17),
        )
        .unwrap();
        let b = BermudanSwaption::on_fixed_period_starts(swap, d(2027, 8, 1)).unwrap();
        let on_grid = b.npv_hull_white(&hw, &GridConfig::default()).unwrap();
        let by_fd = b.npv_fd_g2pp(&g2, &FdG2Config::default()).unwrap();
        assert!(
            (on_grid - by_fd).abs() < 5e-3 * on_grid,
            "grid {on_grid} vs FD G2 {by_fd}"
        );
    }

    #[test]
    fn two_factor_bermudan_dominates_its_europeans() {
        let m = model();
        let swap = VanillaSwap::usd_standard(
            10_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 8, 17),
            d(2032, 8, 17),
        )
        .unwrap();
        let b = BermudanSwaption::on_fixed_period_starts(swap, d(2027, 8, 1)).unwrap();
        let bermudan = b.npv_fd_g2pp(&m, &FdG2Config::default()).unwrap();
        let curve = m.curve();
        let periods = b.swap.fixed_periods().unwrap();
        let best = b
            .exercise_dates
            .iter()
            .map(|&date| {
                let start = periods.iter().find(|p| p.start >= date).unwrap().start;
                let tail = VanillaSwap {
                    effective_date: start,
                    ..b.swap.clone()
                };
                crate::rates::contracts::Swaption::new(tail, date)
                    .unwrap()
                    .npv_g2pp(&m)
                    .unwrap()
            })
            .fold(f64::MIN, f64::max);
        let _ = curve;
        assert!(bermudan > best, "{bermudan} vs {best}");
        assert!(bermudan < 2.0 * best);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let m = model();
        let ok = |_: usize, _: f64, _: f64, c: f64| Ok(c);
        assert!(backward_induction(&m, &[2.0, 1.0], ok, &FdG2Config::default()).is_err());
        let even = FdG2Config {
            nodes_x: 100,
            ..FdG2Config::default()
        };
        assert!(backward_induction(&m, &[1.0], ok, &even).is_err());
        assert!(
            bermudan_swaption(&m, &[], 1.0, PayerReceiver::Payer, &FdG2Config::default()).is_err()
        );
    }
}
