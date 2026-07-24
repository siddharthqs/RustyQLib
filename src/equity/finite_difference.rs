//! Finite difference pricer for the backward pricing PDE in log-spot.
//!
//! Features:
//! - theta-scheme (Crank-Nicolson with a Rannacher fully-implicit start),
//!   cell-averaged terminal conditions (kinks and digital jumps), generic
//!   Dirichlet boundaries.
//! - **Per-node, per-step coefficient assembly**: supports the Dupire local
//!   vol model (`mc_model: "local_vol"` applies to this engine too) and
//!   term-structure-consistent rates (each time step discounts and drifts
//!   at the curve's forward rate for its own calendar interval). This
//!   assembly structure is the 1-D basis a stochastic vol (ADI) solver
//!   will extend.
//! - **American exercise via Brennan-Schwartz** (projection inside the
//!   tridiagonal solve, swept from the out-of-the-money side).
//! - **Barrier options**: knock-out via an absorbing boundary with the grid
//!   edge placed exactly at the barrier; knock-in by parity (European).
//! - **Greeks from the grid**: delta/gamma from a local quadratic fit at
//!   the spot, theta from the last two time layers — one solve yields
//!   npv/delta/gamma/theta; vega and rho are bump-and-resolve.
//!
//! Grid sizes are configurable per contract (`fd_spot_steps`,
//! `fd_time_steps` in JSON).

use crate::core::curves::Compounding;
use crate::core::data_models::EquityOptionData;
// linear kernels live in core::fd_solvers; thomas_algorithm is re-exported
// because it was previously public from this module
use crate::core::fd_solvers::brennan_schwartz;
pub use crate::core::fd_solvers::thomas_algorithm;
use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::local_vol::LocalVol;
use crate::equity::montecarlo::McModel;
use crate::equity::utils::Payoff;
use crate::equity::vanila_option::{BarrierPayoff, EquityOption};

const RANNACHER_STEPS: usize = 4;
const CELL_AVG_POINTS: usize = 16;

#[derive(Debug, Clone, Copy)]
pub struct FdConfig {
    pub spot_steps: usize,
    pub time_steps: usize,
    pub grid_stdevs: f64,
}

impl Default for FdConfig {
    fn default() -> Self {
        FdConfig { spot_steps: 400, time_steps: 400, grid_stdevs: 5.0 }
    }
}

impl FdConfig {
    pub fn from_data(data: &EquityOptionData) -> Self {
        let defaults = FdConfig::default();
        FdConfig {
            spot_steps: data.fd_spot_steps.unwrap_or(defaults.spot_steps).max(10),
            time_steps: data.fd_time_steps.unwrap_or(defaults.time_steps).max(10),
            grid_stdevs: defaults.grid_stdevs,
        }
    }
}

/// One solve returns the value and the grid Greeks.
#[derive(Debug, Clone, Copy)]
pub struct FdSolution {
    pub npv: f64,
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
}

impl FdSolution {
    fn zero() -> Self {
        FdSolution { npv: 0.0, delta: 0.0, gamma: 0.0, theta: 0.0 }
    }
    fn minus(self, other: FdSolution) -> Self {
        FdSolution {
            npv: self.npv - other.npv,
            delta: self.delta - other.delta,
            gamma: self.gamma - other.gamma,
            theta: self.theta - other.theta,
        }
    }
}

pub fn npv(option: &EquityOption) -> f64 {
    solution(option).npv
}
pub fn delta(option: &EquityOption) -> f64 {
    solution(option).delta
}
pub fn gamma(option: &EquityOption) -> f64 {
    solution(option).gamma
}
pub fn theta(option: &EquityOption) -> f64 {
    solution(option).theta
}
pub fn vega(option: &EquityOption) -> f64 {
    // parallel vol bump: constant-vol solves shift sigma, local vol solves
    // shift the implied surface before the Dupire transform
    let h = 1e-3;
    (solve_dispatch(option, h, 0.0, 0.0).npv - solve_dispatch(option, -h, 0.0, 0.0).npv) / (2.0 * h)
}
pub fn rho(option: &EquityOption) -> f64 {
    let h = 1e-4;
    (solve_dispatch(option, 0.0, h, 0.0).npv - solve_dispatch(option, 0.0, -h, 0.0).npv) / (2.0 * h)
}

/// Vanna from the change in the grid delta under a parallel vol bump.
pub fn vanna(option: &EquityOption) -> f64 {
    let h = 1e-3;
    (solve_dispatch(option, h, 0.0, 0.0).delta - solve_dispatch(option, -h, 0.0, 0.0).delta)
        / (2.0 * h)
}

/// Charm from the spot derivative of the grid's calendar theta.
pub fn charm(option: &EquityOption) -> f64 {
    let h = option.base.underlying_price.value() * 1e-3;
    (solve_dispatch(option, 0.0, 0.0, h).theta - solve_dispatch(option, 0.0, 0.0, -h).theta)
        / (2.0 * h)
}

/// Zomma from the change in the grid gamma under a parallel vol bump.
pub fn zomma(option: &EquityOption) -> f64 {
    let h = 1e-3;
    (solve_dispatch(option, h, 0.0, 0.0).gamma - solve_dispatch(option, -h, 0.0, 0.0).gamma)
        / (2.0 * h)
}

/// Volga as the second price derivative under a parallel vol bump. A larger
/// step than the first-order Greeks tempers the roundoff amplification of a
/// second difference against the grid's own discretization error.
pub fn volga(option: &EquityOption) -> f64 {
    let h = 1e-2;
    (solve_dispatch(option, h, 0.0, 0.0).npv - 2.0 * solve_dispatch(option, 0.0, 0.0, 0.0).npv
        + solve_dispatch(option, -h, 0.0, 0.0).npv)
        / (h * h)
}

/// Value and grid Greeks in a single solve (two for knock-ins).
pub fn solution(option: &EquityOption) -> FdSolution {
    solve_dispatch(option, 0.0, 0.0, 0.0)
}

/// Reprice under a shifted market for PnL attribution. The grid has no
/// calendar-time shift, so the elapsed-time effect is rolled forward with the
/// bumped grid's own theta (exact to first order in `d_time`, which is small
/// for the daily PnL horizon this supports).
pub(crate) fn npv_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> f64 {
    let sol = solve_dispatch(option, d_vol, d_rate, d_spot);
    sol.npv + sol.theta * d_time
}

fn solve_dispatch(
    option: &EquityOption,
    sigma_bump: f64,
    r_bump: f64,
    spot_bump: f64,
) -> FdSolution {
    let t = option.time_to_maturity();
    assert!(t >= 0.0, "Option is expired or negative time");
    let s0 = option.base.underlying_price.value() + spot_bump;
    assert!(s0 > 0.0, "underlying price must be positive");
    if t == 0.0 {
        let mut sol = FdSolution::zero();
        sol.npv = option.payoff.payoff(s0, option.base.strike_price);
        return sol;
    }

    if let Some(barrier) = option.payoff.as_any().downcast_ref::<BarrierPayoff>() {
        let down = barrier.direction == BarrierDirection::Down;
        let knocked = if down { s0 <= barrier.barrier } else { s0 >= barrier.barrier };
        return match barrier.knock {
            KnockType::Out => {
                if knocked {
                    FdSolution::zero()
                } else {
                    solve(option, sigma_bump, r_bump, spot_bump, Some(barrier))
                }
            }
            KnockType::In => {
                // knock-in by parity (European only; guarded upstream):
                // KI = vanilla leg - KO, which is linear in all Greeks
                let vanilla = solve(option, sigma_bump, r_bump, spot_bump, None);
                if knocked {
                    vanilla
                } else {
                    vanilla.minus(solve(option, sigma_bump, r_bump, spot_bump, Some(barrier)))
                }
            }
        };
    }
    solve(option, sigma_bump, r_bump, spot_bump, None)
}

/// Volatility field used to assemble the PDE coefficients.
enum FdVol<'a> {
    Const(f64),
    Local(LocalVol<'a>),
}

impl FdVol<'_> {
    fn vol(&self, s: f64, calendar_t: f64) -> f64 {
        match self {
            FdVol::Const(v) => *v,
            FdVol::Local(lv) => lv.vol(s, calendar_t),
        }
    }
}

fn solve(
    option: &EquityOption,
    sigma_bump: f64,
    r_bump: f64,
    spot_bump: f64,
    knock_out: Option<&BarrierPayoff>,
) -> FdSolution {
    let cfg = &option.fd;
    let payoff = option.payoff.as_ref();
    let strike = option.base.strike_price;
    let s0 = option.base.underlying_price.value() + spot_bump;
    let q = option.base.carry_yield();
    let t = option.time_to_maturity();
    let sigma_ref = option.base.volatility() + sigma_bump;
    assert!(sigma_ref > 0.0, "volatility must be positive");
    let american = matches!(payoff.exercise_style(), ContractStyle::American);
    let put = matches!(payoff.put_or_call(), PutOrCall::Put);

    let vol_field = match option.mc.model {
        McModel::Gbm => FdVol::Const(sigma_ref),
        McModel::LocalVol => FdVol::Local(LocalVol::new(
            &option.base.vol_surface,
            &option.base.discount_curve,
            // A spot bump moves the valuation point, not the calibrated
            // local-vol surface reference spot.
            option.base.underlying_price.value(),
            q,
            sigma_bump,
        )),
        McModel::Heston => panic!(
            "The Heston model needs a 2-D (ADI) FD solver, which is future \
             work; use the Analytical or MonteCarlo engines"
        ),
    };

    // ── Grid geometry (log-spot). A knock-out barrier becomes the exact
    // grid edge (absorbing boundary); otherwise the grid centers on x0.
    let x0 = s0.ln();
    let r_flat = option.base.risk_free_rate() + r_bump;
    let drift_width = ((r_flat - q - 0.5 * sigma_ref * sigma_ref) * t).abs();
    let half_width =
        cfg.grid_stdevs * sigma_ref * t.sqrt() + drift_width + (strike / s0).ln().abs().max(1e-2);
    let (x_min, x_max, barrier_low, barrier_high) = match knock_out {
        Some(b) if b.direction == BarrierDirection::Down => {
            (b.barrier.ln(), x0 + half_width, true, false)
        }
        Some(b) => (x0 - half_width, b.barrier.ln(), false, true),
        None => (x0 - half_width, x0 + half_width, false, false),
    };
    let n = cfg.spot_steps;
    let dx = (x_max - x_min) / n as f64;
    let x_at = |i: usize| x_min + i as f64 * dx;
    let s_grid: Vec<f64> = (0..=n).map(|i| x_at(i).exp()).collect();
    let exercise: Vec<f64> = s_grid.iter().map(|&s| payoff.payoff(s, strike)).collect();

    // ── Per-step forward rates from the discount curve (term-structure
    // consistent drift and discounting), plus any rho bump.
    let steps = cfg.time_steps;
    let dt = t / steps as f64;
    let curve = &option.base.discount_curve;
    let step_rates: Vec<f64> = (0..steps)
        .map(|k| {
            // step k advances time-to-expiry tau from k*dt to (k+1)*dt,
            // i.e. calendar time from t - k*dt back to t - (k+1)*dt
            let t2 = t - k as f64 * dt;
            let t1 = t - (k + 1) as f64 * dt;
            let fwd = if t1 <= 0.0 {
                curve.zero_rate_with(t2.max(1e-8), Compounding::Continuous)
            } else {
                curve
                    .forward_rate_with(t1, t2, Compounding::Continuous)
                    .unwrap_or_else(|_| curve.zero_rate_with(t2, Compounding::Continuous))
            };
            fwd + r_bump
        })
        .collect();

    // cash dividend ex-dates as year fractions inside the option's life
    let cash_divs: Vec<(f64, f64)> = option
        .base
        .cash_dividends
        .iter()
        .filter_map(|(date, amount)| {
            let td = (*date - option.base.valuation_date).num_days() as f64 / 365.0;
            (td > 0.0 && td <= t).then_some((td, *amount))
        })
        .collect();

    // terminal condition: cell-averaged payoff
    let mut v: Vec<f64> = (0..=n)
        .map(|i| cell_average_payoff(payoff, strike, x_at(i), dx))
        .collect();
    if barrier_low {
        v[0] = 0.0;
    }
    if barrier_high {
        v[n] = 0.0;
    }

    let m = n - 1; // interior unknowns
    let mut sub = vec![0.0; m - 1];
    let mut dia = vec![0.0; m];
    let mut sup = vec![0.0; m - 1];
    let mut rhs = vec![0.0; m];
    let mut lower = vec![0.0; n + 1];
    let mut diag = vec![0.0; n + 1];
    let mut upper = vec![0.0; n + 1];

    // cumulative discount and forward growth to the current time layer,
    // for the generic Dirichlet boundary V(S_b, tau) = D * payoff(S_b * G)
    let mut cum_df = 1.0;
    let mut cum_growth = 1.0;
    let mut theta_layer_value = 0.0; // value at spot one step before the end

    for step in 0..steps {
        let theta_w = if step < RANNACHER_STEPS { 1.0 } else { 0.5 };
        let r_step = step_rates[step];
        let calendar_mid = (t - (step as f64 + 0.5) * dt).max(0.0);
        cum_df *= (-r_step * dt).exp();
        cum_growth *= ((r_step - q) * dt).exp();

        // per-node coefficients at this time layer
        for i in 0..=n {
            let sigma = vol_field.vol(s_grid[i], calendar_mid);
            let s2 = 0.5 * sigma * sigma;
            let mu = r_step - q - s2;
            lower[i] = s2 / (dx * dx) - mu / (2.0 * dx);
            diag[i] = -2.0 * s2 / (dx * dx) - r_step;
            upper[i] = s2 / (dx * dx) + mu / (2.0 * dx);
        }

        // boundary values at the new time layer
        let boundary = |i: usize, is_barrier: bool| -> f64 {
            if is_barrier {
                return 0.0;
            }
            let mut val = cum_df * payoff.payoff(s_grid[i] * cum_growth, strike);
            if american {
                val = val.max(exercise[i]);
            }
            val
        };
        let v_low = boundary(0, barrier_low);
        let v_high = boundary(n, barrier_high);

        for i in 1..n {
            let av = lower[i] * v[i - 1] + diag[i] * v[i] + upper[i] * v[i + 1];
            rhs[i - 1] = v[i] + (1.0 - theta_w) * dt * av;
        }
        rhs[0] += theta_w * dt * lower[1] * v_low;
        rhs[m - 1] += theta_w * dt * upper[n - 1] * v_high;
        for i in 1..n {
            dia[i - 1] = 1.0 - theta_w * dt * diag[i];
        }
        for i in 1..n - 1 {
            sub[i - 1] = -theta_w * dt * lower[i + 1];
            sup[i - 1] = -theta_w * dt * upper[i];
        }

        let interior = if american {
            // Brennan-Schwartz: apply the exercise constraint inside the
            // back-substitution, sweeping from the out-of-the-money side
            // toward the exercise region (low spot for puts, high for calls)
            brennan_schwartz(&sub, &dia, &sup, &rhs, &exercise[1..n], put)
        } else {
            thomas_algorithm(&sub, &dia, &sup, &rhs)
        };
        v[0] = v_low;
        v[n] = v_high;
        v[1..n].copy_from_slice(&interior);

        // cash dividend jump condition: when the backward induction crosses
        // an ex-date, V(S, t_ex^-) = V(S - D, t_ex^+)
        if !cash_divs.is_empty() {
            let cal_old = t - step as f64 * dt;
            let cal_new = t - (step + 1) as f64 * dt;
            let crossing: f64 = cash_divs
                .iter()
                .filter(|(td, _)| *td < cal_old && *td >= cal_new)
                .map(|(_, amount)| *amount)
                .sum();
            if crossing > 0.0 {
                let shifted: Vec<f64> = (0..=n)
                    .map(|i| {
                        let s_target = s_grid[i] - crossing;
                        if s_target <= s_grid[0] {
                            v[0]
                        } else {
                            let x_target = s_target.ln();
                            let j =
                                (((x_target - x_min) / dx).floor() as usize).min(n - 1);
                            let w = ((x_target - x_at(j)) / dx).clamp(0.0, 1.0);
                            v[j] * (1.0 - w) + v[j + 1] * w
                        }
                    })
                    .collect();
                v = shifted;
                if american {
                    for i in 0..=n {
                        if v[i] < exercise[i] {
                            v[i] = exercise[i];
                        }
                    }
                }
            }
        }

        if step + 1 == steps.saturating_sub(1) {
            theta_layer_value = read_grid(&v, x_min, dx, x0).0;
        }
    }

    let (npv, delta_x, gamma_x) = read_grid(&v, x_min, dx, x0);
    // chain rule from log-spot: V_S = V_x / S, V_SS = (V_xx - V_x) / S^2
    let delta = delta_x / s0;
    let gamma = (gamma_x - delta_x) / (s0 * s0);
    let theta = if steps >= 2 { (theta_layer_value - npv) / dt } else { 0.0 };
    FdSolution { npv, delta, gamma, theta }
}

/// Quadratic fit through the three nodes nearest `x0`:
/// returns (value, dV/dx, d2V/dx2) at x0.
fn read_grid(v: &[f64], x_min: f64, dx: f64, x0: f64) -> (f64, f64, f64) {
    let n = v.len() - 1;
    let i = (((x0 - x_min) / dx).round() as usize).clamp(1, n - 1);
    let e = x0 - (x_min + i as f64 * dx);
    let b = (v[i + 1] - v[i - 1]) / (2.0 * dx);
    let c = (v[i + 1] - 2.0 * v[i] + v[i - 1]) / (2.0 * dx * dx);
    (v[i] + b * e + c * e * e, b + 2.0 * c * e, 2.0 * c)
}

/// Average of the payoff over the grid cell `[x - dx/2, x + dx/2]`.
fn cell_average_payoff(payoff: &dyn Payoff, strike: f64, x: f64, dx: f64) -> f64 {
    let k = CELL_AVG_POINTS;
    let mut sum = 0.0;
    for j in 0..k {
        let xi = x - 0.5 * dx + (j as f64 + 0.5) * dx / k as f64;
        sum += payoff.payoff(xi.exp(), strike);
    }
    sum / k as f64
}
