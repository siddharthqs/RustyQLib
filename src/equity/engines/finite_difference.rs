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
use crate::core::errors::RustyQLibError;
// linear kernels live in core::fd_solvers; thomas_algorithm is re-exported
// because it was previously public from this module
use crate::core::fd_solvers::brennan_schwartz;
pub use crate::core::fd_solvers::thomas_algorithm;
use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::bump::BumpedMarket;
use crate::equity::conventions::{
    MIN_BUMPED_SPOT_FRAC, MIN_BUMPED_VOL, RATE_BUMP, SPOT_REL_BUMP, VOLGA_BUMP, VOL_BUMP,
};
use crate::equity::local_vol::{LocalVol, LocalVolGrid};
use crate::equity::utils::Model;
use crate::equity::utils::Payoff;
use crate::equity::vanilla_option::{BarrierPayoff, EquityOption};

/// Fully-implicit starting layers (kink damping); shared with the 2-D
/// Heston ADI engine.
pub(crate) const RANNACHER_STEPS: usize = 4;
/// Sub-samples for the cell-averaged terminal condition.
const CELL_AVG_POINTS: usize = 16;

/// Time-stepping weight of backward step `step`: fully implicit inside
/// the Rannacher start, Crank-Nicolson afterwards.
pub(crate) fn rannacher_theta(step: usize) -> f64 {
    if step < RANNACHER_STEPS {
        1.0
    } else {
        0.5
    }
}

/// Bermudan exercise mask over the backward steps: backward step `s`
/// covers calendar time `t - (s+1) dt`, so an exercise time (forward,
/// 1-based grid index `g`) maps to `s = steps - g - 1`. `None` for the
/// other exercise styles.
pub(crate) fn bermudan_backward_mask(
    style: &ContractStyle,
    t: f64,
    steps: usize,
) -> Option<Vec<bool>> {
    match style {
        ContractStyle::Bermudan(times) => {
            let mut mask = vec![false; steps];
            for g in crate::core::utils::times_to_grid_steps(times, t, steps) {
                if g < steps {
                    mask[steps - g - 1] = true;
                }
            }
            Some(mask)
        }
        _ => None,
    }
}

/// Half-width of the log-spot grid around `ln S0`: `grid_stdevs` standard
/// deviations plus the drift over the horizon plus the log-distance to
/// the strike.
pub(crate) fn grid_half_width(
    grid_stdevs: f64,
    sigma_ref: f64,
    t: f64,
    r: f64,
    q: f64,
    strike: f64,
    s0: f64,
) -> f64 {
    let drift_width = ((r - q - 0.5 * sigma_ref * sigma_ref) * t).abs();
    grid_stdevs * sigma_ref * t.sqrt() + drift_width + (strike / s0).ln().abs().max(1e-2)
}

/// Value and grid Greeks from the log-spot reads: chain rule from
/// log-spot (`V_S = V_x / S`, `V_SS = (V_xx - V_x) / S^2`) and calendar
/// theta from the last two time layers.
pub(crate) fn grid_solution(
    npv: f64,
    delta_x: f64,
    gamma_x: f64,
    s0: f64,
    theta_layer_value: f64,
    steps: usize,
    dt: f64,
) -> FdSolution {
    let delta = delta_x / s0;
    let gamma = (gamma_x - delta_x) / (s0 * s0);
    let theta = if steps >= 2 {
        (theta_layer_value - npv) / dt
    } else {
        0.0
    };
    FdSolution {
        npv,
        delta,
        gamma,
        theta,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FdConfig {
    pub spot_steps: usize,
    pub time_steps: usize,
    pub grid_stdevs: f64,
}

impl Default for FdConfig {
    fn default() -> Self {
        FdConfig {
            spot_steps: 400,
            time_steps: 400,
            grid_stdevs: 5.0,
        }
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

    /// Domain checks on the grid dimensions.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.spot_steps < 3 || self.time_steps < 1 {
            return Err(RustyQLibError::invalid_input(
                "fd_grid",
                format!(
                    "the FD grid needs at least 3 spot steps and 1 time step, got {} x {}",
                    self.spot_steps, self.time_steps
                ),
            ));
        }
        Ok(())
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
        FdSolution {
            npv: 0.0,
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
        }
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

/// Price under a market view; `None` prices the base market. The grid
/// has no calendar-time shift, so the view's `d_time` is rolled forward
/// with the bumped grid's own theta (exact to first order in `d_time`,
/// which is small for the daily PnL horizon this supports); the other
/// shifts solve the bumped grid directly.
pub fn npv(option: &EquityOption, bumped_market: Option<&BumpedMarket>) -> f64 {
    let base = BumpedMarket::base(&option.market);
    let b = bumped_market.unwrap_or(&base).bump();
    let sol = solve_dispatch(option, b.d_vol, b.d_rate, b.d_spot);
    sol.npv + sol.theta * b.d_time
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
    let h = VOL_BUMP;
    (solve_dispatch(option, h, 0.0, 0.0).npv - solve_dispatch(option, -h, 0.0, 0.0).npv) / (2.0 * h)
}
pub fn rho(option: &EquityOption) -> f64 {
    let h = RATE_BUMP;
    (solve_dispatch(option, 0.0, h, 0.0).npv - solve_dispatch(option, 0.0, -h, 0.0).npv) / (2.0 * h)
}

/// Vanna from the change in the grid delta under a parallel vol bump.
pub fn vanna(option: &EquityOption) -> f64 {
    let h = VOL_BUMP;
    (solve_dispatch(option, h, 0.0, 0.0).delta - solve_dispatch(option, -h, 0.0, 0.0).delta)
        / (2.0 * h)
}

/// Charm from the spot derivative of the grid's calendar theta.
pub fn charm(option: &EquityOption) -> f64 {
    let h = option.market.spot.value() * SPOT_REL_BUMP;
    (solve_dispatch(option, 0.0, 0.0, h).theta - solve_dispatch(option, 0.0, 0.0, -h).theta)
        / (2.0 * h)
}

/// Zomma from the change in the grid gamma under a parallel vol bump.
pub fn zomma(option: &EquityOption) -> f64 {
    let h = VOL_BUMP;
    (solve_dispatch(option, h, 0.0, 0.0).gamma - solve_dispatch(option, -h, 0.0, 0.0).gamma)
        / (2.0 * h)
}

/// Volga as the second price derivative under a parallel vol bump. A larger
/// step than the first-order Greeks tempers the roundoff amplification of a
/// second difference against the grid's own discretization error.
pub fn volga(option: &EquityOption) -> f64 {
    let h = VOLGA_BUMP;
    (solve_dispatch(option, h, 0.0, 0.0).npv - 2.0 * solve_dispatch(option, 0.0, 0.0, 0.0).npv
        + solve_dispatch(option, -h, 0.0, 0.0).npv)
        / (h * h)
}

/// Value and grid Greeks in a single solve (two for knock-ins).
pub fn solution(option: &EquityOption) -> FdSolution {
    solve_dispatch(option, 0.0, 0.0, 0.0)
}

/// Value and all nine reported Greeks from a **shared** set of grid solves
/// instead of a re-solve per Greek: the base solve yields the price plus
/// delta/gamma/theta for free, the two vol-bumped solves yield vega, vanna
/// and zomma together, two rate bumps yield rho, and two spot-bumped
/// solves yield charm — nine solves in total. Each number is produced by
/// exactly the same solves and arithmetic as its accessor above.
pub fn pricing_result(option: &EquityOption) -> crate::core::results::PricingResult {
    use crate::core::results::{Greeks, PricingResult};
    let hv = VOL_BUMP;
    let hr = RATE_BUMP;
    let hs = option.market.spot.value() * SPOT_REL_BUMP;
    // the nine grid solves are independent: run them on the rayon pool
    // (values are the same solves as the sequential code, bit for bit)
    let ((base, (vol_up, vol_down)), ((rate_up, rate_down), (spot_up, spot_down))) = rayon::join(
        || {
            rayon::join(
                || solution(option),
                || {
                    rayon::join(
                        || solve_dispatch(option, hv, 0.0, 0.0),
                        || solve_dispatch(option, -hv, 0.0, 0.0),
                    )
                },
            )
        },
        || {
            rayon::join(
                || {
                    rayon::join(
                        || solve_dispatch(option, 0.0, hr, 0.0),
                        || solve_dispatch(option, 0.0, -hr, 0.0),
                    )
                },
                || {
                    rayon::join(
                        || solve_dispatch(option, 0.0, 0.0, hs),
                        || solve_dispatch(option, 0.0, 0.0, -hs),
                    )
                },
            )
        },
    );
    let rho = (rate_up.npv - rate_down.npv) / (2.0 * hr);
    let charm = (spot_up.theta - spot_down.theta) / (2.0 * hs);
    let gamma_p = crate::equity::greeks::gamma_p_from(option.market.spot.value(), base.gamma);
    PricingResult {
        pv: base.npv,
        greeks: Greeks {
            delta: base.delta,
            gamma: base.gamma,
            vega: (vol_up.npv - vol_down.npv) / (2.0 * hv),
            theta: base.theta,
            rho,
            vanna: (vol_up.delta - vol_down.delta) / (2.0 * hv),
            charm,
            gamma_p,
            zomma: (vol_up.gamma - vol_down.gamma) / (2.0 * hv),
        },
        std_err: None,
        asset_greeks: None,
    }
}

fn solve_dispatch(
    option: &EquityOption,
    sigma_bump: f64,
    r_bump: f64,
    spot_bump: f64,
) -> FdSolution {
    let t = option.time_to_maturity();
    assert!(t >= 0.0, "Option is expired or negative time");
    // The bumped spot and vol are floored exactly like a `BumpedMarket`
    // read (a deep spot-down stress, a vega down-bump on a tiny-vol
    // option) so the stencil legs price at the floor instead of tripping
    // the positivity asserts. The *effective* shift is what reaches the
    // solvers, so their own recomputation sees the floored value; an
    // unfloored shift passes through untouched (bit-identical).
    let spot = option.market.spot.value();
    let s0 = (spot + spot_bump).max(spot * MIN_BUMPED_SPOT_FRAC);
    let spot_bump = if s0 == spot + spot_bump {
        spot_bump
    } else {
        s0 - spot
    };
    assert!(s0 > 0.0, "underlying price must be positive");
    if t == 0.0 {
        let mut sol = FdSolution::zero();
        sol.npv = option.payoff.payoff(s0, option.base.strike_price);
        return sol;
    }

    if option.model.is_heston() {
        // the ADI engine floors its own vol-parameter shift
        return crate::equity::heston_adi::solve(option, sigma_bump, r_bump, spot_bump);
    }

    let vol = option.volatility();
    let sigma_ref = (vol + sigma_bump).max(MIN_BUMPED_VOL);
    let sigma_bump = if sigma_ref == vol + sigma_bump {
        sigma_bump
    } else {
        sigma_ref - vol
    };

    if let Some(barrier) = option.payoff.as_any().downcast_ref::<BarrierPayoff>() {
        assert!(
            barrier.barrier2.is_none() && barrier.rebate == 0.0,
            "double barriers and rebates are not supported on the FD engine;              use the Analytical or MonteCarlo engine"
        );
        let down = barrier.direction == BarrierDirection::Down;
        let knocked = if down {
            s0 <= barrier.barrier
        } else {
            s0 >= barrier.barrier
        };
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
enum FdVol {
    Const(f64),
    Grid(LocalVolGrid),
}

impl FdVol {
    fn vol(&self, s: f64, calendar_t: f64) -> f64 {
        match self {
            FdVol::Const(v) => *v,
            FdVol::Grid(grid) => grid.vol(s, calendar_t),
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
    let cfg = option.fd_cfg();
    let payoff = option.payoff.as_ref();
    let strike = option.base.strike_price;
    let s0 = option.market.spot.value() + spot_bump;
    let q = option.carry_yield();
    let t = option.time_to_maturity();
    let sigma_ref = option.volatility() + sigma_bump;
    assert!(sigma_ref > 0.0, "volatility must be positive");
    let american = matches!(payoff.exercise_style(), ContractStyle::American);
    let bermudan_backward = bermudan_backward_mask(payoff.exercise_style(), t, cfg.time_steps);
    let put = matches!(payoff.put_or_call(), PutOrCall::Put);

    let vol_field = match option.model {
        Model::Gbm => FdVol::Const(sigma_ref),
        Model::LocalVol => FdVol::Grid(
            LocalVol::new(
                &option.market.vol_surface,
                &option.market.discount_curve,
                // A spot bump moves the valuation point, not the calibrated
                // local-vol surface reference spot.
                option.market.spot.value(),
                q,
                sigma_bump,
            )
            // sampled once per solve; the backward march queries t in (0, t]
            .to_grid(t),
        ),
        // routed to the 2-D ADI solver in solve_dispatch
        Model::Heston(_) => unreachable!("Heston is dispatched to heston_adi::solve"),
        // rejected by check_engine_support (Monte Carlo only)
        Model::RBergomi(_) => unreachable!("rBergomi never reaches the FD engine"),
        // rejected by check_engine_support (Analytical / Monte Carlo only)
        Model::Sabr(_) => unreachable!("SABR never reaches the FD engine"),
    };

    // ── Grid geometry (log-spot). A knock-out barrier becomes the exact
    // grid edge (absorbing boundary); otherwise the grid centers on x0.
    let x0 = s0.ln();
    let r_flat = option.risk_free_rate() + r_bump;
    let half_width = grid_half_width(cfg.grid_stdevs, sigma_ref, t, r_flat, q, strike, s0);
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
    let curve = &option.market.discount_curve;
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

    // cash dividend ex-dates as year fractions inside the option's life;
    // a dividend going ex on the maturity date is applied to the terminal
    // condition (the backward march only crosses ex-dates strictly inside
    // its steps), so it is kept out of the in-loop list
    let (terminal_divs, cash_divs): (Vec<(f64, f64)>, Vec<(f64, f64)>) = option
        .market
        .cash_dividends
        .iter()
        .filter_map(|(date, amount)| {
            let td = crate::equity::conventions::year_fraction(option.market.valuation_date, *date);
            (td > 0.0 && td <= t).then_some((td, *amount))
        })
        .partition(|(td, _)| *td >= t - 1e-12);

    // terminal condition: cell-averaged payoff
    let mut v: Vec<f64> = (0..=n)
        .map(|i| cell_average_payoff(payoff, strike, x_at(i), dx))
        .collect();
    // a dividend on the maturity date: the payoff is observed on the
    // ex-dividend price, V(S, T^-) = payoff(S - D)
    let terminal_div: f64 = terminal_divs.iter().map(|(_, amount)| *amount).sum();
    if terminal_div > 0.0 {
        v = shift_for_dividend(&v, &s_grid, x_min, dx, terminal_div);
        if american {
            for i in 0..=n {
                if v[i] < exercise[i] {
                    v[i] = exercise[i];
                }
            }
        }
    }
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

    // `step` drives the Rannacher switch, calendar time and the bermudan
    // mask as well as `step_rates`; enumerate() would misplace the emphasis
    #[allow(clippy::needless_range_loop)]
    for step in 0..steps {
        let exercise_now = american
            || bermudan_backward
                .as_ref()
                .is_some_and(|m| m.get(step).copied().unwrap_or(false));
        let theta_w = rannacher_theta(step);
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
            if exercise_now {
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

        let interior = if exercise_now {
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
                v = shift_for_dividend(&v, &s_grid, x_min, dx, crossing);
                // the re-sampling reads interior values into the edge
                // nodes: an absorbing barrier edge must stay at zero
                if barrier_low {
                    v[0] = 0.0;
                }
                if barrier_high {
                    v[n] = 0.0;
                }
                if exercise_now {
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
    grid_solution(npv, delta_x, gamma_x, s0, theta_layer_value, steps, dt)
}

/// Cash dividend jump condition `V(S, t_ex^-) = V(S - D, t_ex^+)`: the
/// layer re-sampled at the ex-dividend spot by linear interpolation in
/// log-spot (nodes shifted below the grid take the lowest node's value).
fn shift_for_dividend(v: &[f64], s_grid: &[f64], x_min: f64, dx: f64, crossing: f64) -> Vec<f64> {
    let n = v.len() - 1;
    (0..=n)
        .map(|i| {
            let s_target = s_grid[i] - crossing;
            if s_target <= s_grid[0] {
                v[0]
            } else {
                let x_target = s_target.ln();
                let j = (((x_target - x_min) / dx).floor() as usize).min(n - 1);
                let w = ((x_target - (x_min + j as f64 * dx)) / dx).clamp(0.0, 1.0);
                v[j] * (1.0 - w) + v[j + 1] * w
            }
        })
        .collect()
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

/// Average of the payoff over the grid cell `[x - dx/2, x + dx/2]`
/// (log-spot `x`); shared with the Heston ADI engine.
pub(crate) fn cell_average_payoff(payoff: &dyn Payoff, strike: f64, x: f64, dx: f64) -> f64 {
    let k = CELL_AVG_POINTS;
    let mut sum = 0.0;
    for j in 0..k {
        let xi = x - 0.5 * dx + (j as f64 + 0.5) * dx / k as f64;
        sum += payoff.payoff(xi.exp(), strike);
    }
    sum / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::bump::Bump;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    /// A one-year FD call at `vol`.
    fn call_at_vol(vol: f64) -> EquityOption {
        EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(vol)
            .flat_rate(0.03)
            .valuation_date(date(2026, 1, 1))
            .maturity_date(date(2027, 1, 1))
            .vanilla(PutOrCall::Call)
            .engine(Engine::FiniteDifference)
            .fd_grid(200, 200)
            .build()
            .expect("option must build")
    }

    #[test]
    fn degenerate_vol_bumps_price_at_the_floor_instead_of_panicking() {
        // the FD engine took its bumps as raw scalars, bypassing the
        // BumpedMarket floors: a vega/volga down-bump larger than a
        // tiny-vol option's own vol tripped `sigma_ref > 0`
        let option = call_at_vol(0.005);
        // VOLGA_BUMP is 1e-2, twice this option's volatility
        let volga = volga(&option);
        assert!(volga.is_finite(), "volga must be finite, got {volga}");
        for greek in [vega(&option), vanna(&option), zomma(&option)] {
            assert!(greek.is_finite(), "every vol Greek must be finite");
        }

        // past the floor every bump prices the same solve, bit for bit
        let crushed =
            |d_vol: f64| option.price_bumped(&BumpedMarket::new(&option.market, Bump::vol(d_vol)));
        assert_eq!(crushed(-0.05), crushed(-0.10));
        // and that solve is the option quoted at the floor volatility
        let at_floor = call_at_vol(MIN_BUMPED_VOL).npv();
        assert!(
            (crushed(-0.05) - at_floor).abs() < 1e-9,
            "the floored bump must price at MIN_BUMPED_VOL: {} vs {at_floor}",
            crushed(-0.05)
        );
    }

    #[test]
    fn a_deep_spot_down_stress_prices_at_the_spot_floor() {
        // the same bypass on the spot axis: a stress below zero used to
        // trip `s0 > 0` instead of pricing at the floored spot
        let put = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(0.03)
            .valuation_date(date(2026, 1, 1))
            .maturity_date(date(2027, 1, 1))
            .vanilla(PutOrCall::Put)
            .engine(Engine::FiniteDifference)
            .fd_grid(200, 200)
            .build()
            .expect("option must build");
        let crushed = put.price_bumped(&BumpedMarket::new(&put.market, Bump::spot(-150.0)));
        assert!(crushed.is_finite(), "the stress must value, got {crushed}");
        // a put on a spot pinned just above zero is worth ~ the
        // discounted strike
        assert!(crushed > 90.0, "deep-crash put must be near max: {crushed}");
    }

    #[test]
    fn a_dividend_going_ex_on_the_maturity_date_reaches_the_terminal_condition() {
        // The backward march only crosses ex-dates strictly inside its
        // steps, so a dividend dated on the maturity date was filtered in
        // but never applied — the option priced dividend-free. It belongs
        // in the terminal condition: payoff(S_T - D), which for a call is
        // exactly the payoff of a call struck at K + D.
        let build = |strike: f64, dividend: Option<f64>, engine: Engine| {
            let mut b = EquityOptionBuilder::new()
                .spot(100.0)
                .strike(strike)
                .flat_vol(0.20)
                .flat_rate(0.05)
                .valuation_date(date(2026, 1, 1))
                .maturity_date(date(2027, 1, 1))
                .vanilla(PutOrCall::Call)
                .engine(engine)
                .fd_grid(800, 400);
            if let Some(amount) = dividend {
                b = b.cash_dividend(date(2027, 1, 1), amount);
            }
            b.build().expect("option must build")
        };
        let with_dividend = build(100.0, Some(5.0), Engine::FiniteDifference).npv();
        let shifted_strike = build(105.0, None, Engine::FiniteDifference).npv();
        let analytic = build(105.0, None, Engine::BlackScholes).npv();
        assert!(
            (with_dividend - shifted_strike).abs() < 0.05,
            "K=100 with a 5.00 terminal dividend must price as K=105: \
             {with_dividend} vs {shifted_strike}"
        );
        assert!(
            (with_dividend - analytic).abs() < 0.05,
            "and match the closed form: {with_dividend} vs {analytic}"
        );
        // the regression itself: the dividend must not be dropped
        let no_dividend = build(100.0, None, Engine::FiniteDifference).npv();
        assert!(
            no_dividend - with_dividend > 2.0,
            "a 5.00 dividend at expiry must cost ~2.4 of premium: \
             {no_dividend} vs {with_dividend}"
        );
    }

    #[test]
    fn a_dividend_jump_cannot_revive_the_absorbing_barrier_node() {
        // The dividend re-samples every node at S - D, which pulls an
        // interior value into the knocked-out edge node. With the ex-date
        // inside the last backward step nothing re-imposes the boundary
        // afterwards, so a spot hugging the barrier was valued as if it
        // had already jumped D below it.
        let up_and_out = |spot: f64, dividend: Option<(NaiveDate, f64)>| {
            let mut b = EquityOptionBuilder::new()
                .spot(spot)
                .strike(100.0)
                .flat_vol(0.25)
                .flat_rate(0.03)
                .valuation_date(date(2026, 1, 1))
                .maturity_date(date(2027, 1, 1))
                .barrier(PutOrCall::Call, BarrierDirection::Up, KnockType::Out, 120.0)
                .engine(Engine::FiniteDifference)
                .fd_grid(200, 200);
            if let Some((ex_date, amount)) = dividend {
                b = b.cash_dividend(ex_date, amount);
            }
            b.build().expect("option must build").npv()
        };
        // the value the leak would import: the same contract valued at
        // the post-dividend spot, far from the barrier
        let below = up_and_out(79.9, None);
        assert!(below > 0.1, "the reference leg must be worth something");
        // a spot 0.1 below the barrier with a 40.00 dividend going ex
        // tomorrow: knock-out is all but certain before the jump
        let hugging = up_and_out(119.9, Some((date(2026, 1, 2), 40.0)));
        assert!(hugging >= 0.0, "a knock-out value cannot be negative");
        assert!(
            hugging < 0.7 * below,
            "the absorbing edge must survive the dividend jump: {hugging} \
             against the post-jump value {below}"
        );
    }
}
