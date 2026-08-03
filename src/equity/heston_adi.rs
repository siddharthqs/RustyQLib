//! 2-D ADI finite-difference engine for the Heston PDE — the grid
//! counterpart of the Monte Carlo QE engine, and the independent
//! cross-check for early exercise under stochastic volatility.
//!
//! The backward pricing PDE in `x = ln S` and variance `v`:
//!
//! ```text
//! V_tau = 1/2 v V_xx + (r - q - v/2) V_x            (A_x)
//!       + 1/2 xi^2 v V_vv + kappa (theta - v) V_v   (A_v)
//!       + rho xi v V_xv                             (A_0, mixed)
//!       - r V                                       (split A_x/A_v)
//! ```
//!
//! is split into one tridiagonal operator per axis plus the explicitly
//! treated mixed term, and stepped with the core ADI machinery
//! ([`core::fd_solvers::adi`](crate::core::fd_solvers)): the
//! Hundsdorfer-Verwer scheme for European exercise (second order in time
//! including the mixed term), Douglas with a per-step exercise projection
//! for American/Bermudan (the projection is an operator-splitting
//! approximation, O(dt) at the exercise boundary), both behind a
//! Rannacher fully-implicit start to damp the payoff kink.
//!
//! Boundaries: Dirichlet in `x` refreshed each layer with the discounted
//! forward payoff (the 1-D engine's convention); at `v = 0` the PDE
//! degenerates to its own outflow equation (one-sided `kappa theta V_v`);
//! at `v = v_max` the standard `V_v = 0` closure reduces the row to a
//! 1-D equation at variance `v_max`.
//!
//! Vega bumps follow the library's Heston convention — a parallel shift
//! of `sqrt(v0)` and `sqrt(theta)` — so the whole FD Greek suite
//! (vega/vanna/zomma/volga through [`finite_difference`]'s shared
//! solve dispatch) works unchanged.

use crate::core::fd_solvers::adi::{douglas_step, hundsdorfer_verwer_step};
use crate::core::fd_solvers::axis_operator::{AxisOperator, TensorGrid};
use crate::core::utils::ContractStyle;
use crate::equity::utils::PayoffType;

use super::finite_difference::FdSolution;
use super::vanilla_option::EquityOption;

/// Variance-axis nodes, derived from the spot resolution.
fn var_nodes(spot_steps: usize) -> usize {
    (spot_steps / 8).clamp(30, 80)
}

/// Fully-implicit starting layers (kink damping), matching the 1-D engine.
const RANNACHER_STEPS: usize = 4;

/// Sub-samples for the cell-averaged terminal condition.
const CELL_AVG_POINTS: usize = 16;

/// Solve the Heston PDE and read value, delta, gamma and calendar theta
/// at `(S0, v0)`. Bumps mirror [`finite_difference::solve_dispatch`]:
/// `sigma_bump` shifts `sqrt(v0)`/`sqrt(theta)`, `r_bump` the flat rate,
/// `spot_bump` the valuation spot.
pub(crate) fn solve(
    option: &EquityOption,
    sigma_bump: f64,
    r_bump: f64,
    spot_bump: f64,
) -> FdSolution {
    assert!(
        matches!(option.payoff.payoff_kind(), PayoffType::Vanilla | PayoffType::Binary),
        "the Heston ADI engine prices vanilla and binary payoffs; \
         use MonteCarlo for path-dependent payoffs"
    );
    assert!(
        option.market.cash_dividends.is_empty(),
        "cash dividends are not supported on the Heston ADI engine; use MonteCarlo"
    );
    let hp = option.heston_params().with_vol_shift(sigma_bump);
    let cfg = option.fd_cfg();
    let payoff = option.payoff.as_ref();
    let strike = option.base.strike_price;
    let s0 = option.market.spot.value() + spot_bump;
    let t = option.time_to_maturity();
    let r = option.risk_free_rate() + r_bump;
    let q = option.carry_yield();
    let american = matches!(payoff.exercise_style(), ContractStyle::American);
    let steps = cfg.time_steps;
    let dt = t / steps as f64;
    // Bermudan: backward step `s` covers calendar time t-(s+1)dt
    let bermudan_backward: Option<Vec<bool>> = match payoff.exercise_style() {
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
    };

    // ── Grid geometry: x centered so S0 is exactly a node; v on [0, v_max]
    let sigma_ref = hp.v0.max(hp.theta).sqrt();
    let x0 = s0.ln();
    let drift_width = ((r - q - 0.5 * sigma_ref * sigma_ref) * t).abs();
    let half_width = cfg.grid_stdevs * sigma_ref * t.sqrt()
        + drift_width
        + (strike / s0).ln().abs().max(1e-2);
    let nx = if cfg.spot_steps % 2 == 0 { cfg.spot_steps } else { cfg.spot_steps + 1 };
    let dx = 2.0 * half_width / nx as f64;
    let x_min = x0 - half_width;
    let ix0 = nx / 2;
    let s_grid: Vec<f64> = (0..=nx).map(|i| (x_min + i as f64 * dx).exp()).collect();

    let nv = var_nodes(cfg.spot_steps);
    let v_max = 5.0 * hp.v0.max(hp.theta);
    let dv = v_max / nv as f64;
    let v_grid: Vec<f64> = (0..=nv).map(|j| j as f64 * dv).collect();

    let grid = TensorGrid::new(&[nx + 1, nv + 1]);
    let sx = grid.strides()[0]; // = nv + 1
    let at = |i: usize, j: usize| i * sx + j;

    // ── Axis operators (boundary rows zero => values held by the solver,
    // refreshed manually below). The -rV discounting splits evenly.
    let mut a_x = AxisOperator::zero(&grid, 0);
    let mut a_v = AxisOperator::zero(&grid, 1);
    for i in 1..nx {
        for j in 0..=nv {
            let idx = at(i, j);
            let vj = v_grid[j];
            let s2 = 0.5 * vj;
            let mu = r - q - 0.5 * vj;
            a_x.sub[idx] = s2 / (dx * dx) - mu / (2.0 * dx);
            a_x.diag[idx] = -2.0 * s2 / (dx * dx) - 0.5 * r;
            a_x.sup[idx] = s2 / (dx * dx) + mu / (2.0 * dx);
        }
    }
    for i in 1..nx {
        for j in 0..=nv {
            let idx = at(i, j);
            let vj = v_grid[j];
            if j == 0 {
                // v = 0: the PDE degenerates to outflow, kappa*theta V_v
                // with a one-sided (forward) difference
                let c = hp.kappa * hp.theta / dv;
                a_v.diag[idx] = -c - 0.5 * r;
                a_v.sup[idx] = c;
            } else if j == nv {
                // v = v_max: V_v = 0 closure — no v-coupling, keep the
                // discounting half so the row stays a 1-D equation
                a_v.diag[idx] = -0.5 * r;
            } else {
                let d2 = 0.5 * hp.vol_of_vol * hp.vol_of_vol * vj;
                let mu_v = hp.kappa * (hp.theta - vj);
                a_v.sub[idx] = d2 / (dv * dv) - mu_v / (2.0 * dv);
                a_v.diag[idx] = -2.0 * d2 / (dv * dv) - 0.5 * r;
                a_v.sup[idx] = d2 / (dv * dv) + mu_v / (2.0 * dv);
            }
        }
    }

    // mixed derivative rho xi v V_xv, explicit, zero at every boundary
    let rho_xi = hp.rho * hp.vol_of_vol;
    let v_for_mixed = v_grid.clone();
    let mixed = move |u: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0; u.len()];
        if rho_xi == 0.0 {
            return out;
        }
        for i in 1..nx {
            for j in 1..nv {
                let idx = i * sx + j;
                let cross = (u[idx + sx + 1] - u[idx + sx - 1] - u[idx - sx + 1]
                    + u[idx - sx - 1])
                    / (4.0 * dx * dv);
                out[idx] = rho_xi * v_for_mixed[j] * cross;
            }
        }
        out
    };

    // ── Terminal condition: cell-averaged payoff, constant across v
    let exercise: Vec<f64> = s_grid.iter().map(|&s| payoff.payoff(s, strike)).collect();
    let mut u = vec![0.0; grid.len()];
    for i in 0..=nx {
        let x = x_min + i as f64 * dx;
        let mut avg = 0.0;
        for p in 0..CELL_AVG_POINTS {
            let xi = x - 0.5 * dx + (p as f64 + 0.5) * dx / CELL_AVG_POINTS as f64;
            avg += payoff.payoff(xi.exp(), strike);
        }
        avg /= CELL_AVG_POINTS as f64;
        for j in 0..=nv {
            u[at(i, j)] = avg;
        }
    }

    // ── Backward induction in tau
    let ops = [a_x, a_v];
    let mut theta_layer_value = 0.0;
    for step in 0..steps {
        let exercise_now = american
            || bermudan_backward
                .as_ref()
                .is_some_and(|m| m.get(step).copied().unwrap_or(false));
        let theta_w = if step < RANNACHER_STEPS { 1.0 } else { 0.5 };
        u = if exercise_now || step < RANNACHER_STEPS {
            douglas_step(&grid, &ops, Some(&mixed), &u, dt, theta_w)
        } else {
            hundsdorfer_verwer_step(&grid, &ops, Some(&mixed), &u, dt, 0.5, 0.5)
        };

        // Dirichlet x-boundaries: discounted forward payoff at the new
        // layer (identical convention to the 1-D engine)
        let tau = (step + 1) as f64 * dt;
        let df = (-r * tau).exp();
        let growth = ((r - q) * tau).exp();
        for &i in &[0usize, nx] {
            let mut val = df * payoff.payoff(s_grid[i] * growth, strike);
            if exercise_now {
                val = val.max(exercise[i]);
            }
            for j in 0..=nv {
                u[at(i, j)] = val;
            }
        }

        // exercise projection (operator splitting)
        if exercise_now {
            for i in 0..=nx {
                for j in 0..=nv {
                    let idx = at(i, j);
                    if u[idx] < exercise[i] {
                        u[idx] = exercise[i];
                    }
                }
            }
        }

        if step + 1 == steps.saturating_sub(1) {
            theta_layer_value = read_at(&u, ix0, sx, dx, nv, dv, hp.v0).0;
        }
    }

    let (npv, delta_x, gamma_x) = read_at(&u, ix0, sx, dx, nv, dv, hp.v0);
    let delta = delta_x / s0;
    let gamma = (gamma_x - delta_x) / (s0 * s0);
    let theta = if steps >= 2 { (theta_layer_value - npv) / dt } else { 0.0 };
    FdSolution { npv, delta, gamma, theta }
}

/// Value and x-derivatives at the spot node, linearly interpolated in `v`
/// to the model's `v0`.
fn read_at(
    u: &[f64],
    ix0: usize,
    sx: usize,
    dx: f64,
    nv: usize,
    dv: f64,
    v0: f64,
) -> (f64, f64, f64) {
    let jf = (v0 / dv).min(nv as f64 - 1e-9);
    let j = (jf.floor() as usize).min(nv - 1);
    let w = jf - j as f64;
    let read_row = |jj: usize| -> (f64, f64, f64) {
        let idx = ix0 * sx + jj;
        let val = u[idx];
        let b = (u[idx + sx] - u[idx - sx]) / (2.0 * dx);
        let c = (u[idx + sx] - 2.0 * u[idx] + u[idx - sx]) / (dx * dx);
        (val, b, c)
    };
    let (v_lo, b_lo, c_lo) = read_row(j);
    let (v_hi, b_hi, c_hi) = read_row(j + 1);
    (
        v_lo * (1.0 - w) + v_hi * w,
        b_lo * (1.0 - w) + b_hi * w,
        c_lo * (1.0 - w) + c_hi * w,
    )
}

#[cfg(test)]
mod tests {
    use crate::core::trade::PutOrCall;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::heston::{heston_price, HestonParams};
    use crate::equity::utils::Engine;
    use crate::Instrument;

    fn hp() -> HestonParams {
        HestonParams { v0: 0.09, kappa: 2.0, theta: 0.09, vol_of_vol: 0.4, rho: -0.7 }
    }

    fn option(pc: PutOrCall, strike: f64, engine: Engine, american: bool) -> crate::equity::vanilla_option::EquityOption {
        let mut b = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(strike)
            .flat_vol(0.30)
            .flat_rate(0.05)
            .dividend_yield(0.02)
            .years_to_maturity(1.0)
            .vanilla(pc)
            .heston(hp())
            .engine(engine);
        if american {
            b = b.american();
        }
        b.build().expect("option must build")
    }

    #[test]
    fn european_prices_match_the_characteristic_function() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            for strike in [80.0, 100.0, 120.0] {
                let fd = option(pc, strike, Engine::FiniteDifference, false).npv();
                let cf = heston_price(100.0, strike, 0.05, 0.02, 1.0, &hp(), pc);
                assert!(
                    (fd - cf).abs() < 0.05,
                    "{pc:?} K={strike}: fd {fd:.4} vs cf {cf:.4}"
                );
            }
        }
    }

    #[test]
    fn grid_greeks_match_the_characteristic_function_bumps() {
        let opt = option(PutOrCall::Call, 100.0, Engine::FiniteDifference, false);
        let sol = crate::equity::finite_difference::solution(&opt);
        let h = 0.5;
        let bump = |ds: f64| heston_price(100.0 + ds, 100.0, 0.05, 0.02, 1.0, &hp(), PutOrCall::Call);
        let delta_ref = (bump(h) - bump(-h)) / (2.0 * h);
        let gamma_ref = (bump(h) - 2.0 * bump(0.0) + bump(-h)) / (h * h);
        assert!((sol.delta - delta_ref).abs() < 5e-3, "delta {} vs {delta_ref}", sol.delta);
        assert!((sol.gamma - gamma_ref).abs() < 5e-4, "gamma {} vs {gamma_ref}", sol.gamma);
    }

    #[test]
    fn american_put_is_corroborated_by_the_lsmc() {
        // two independent engines — grid vs regression Monte Carlo — must
        // agree on early exercise under full-strength Heston dynamics
        let fd = option(PutOrCall::Put, 100.0, Engine::FiniteDifference, true).npv();
        let mut mc = option(PutOrCall::Put, 100.0, Engine::MonteCarlo, true);
        mc.mc_cfg_mut().paths = 100_000;
        let lsmc = mc.npv();
        let european = heston_price(100.0, 100.0, 0.05, 0.02, 1.0, &hp(), PutOrCall::Put);
        assert!(fd > european + 0.05, "american fd {fd} vs european {european}");
        // LSMC is biased slightly low (suboptimal fitted rule): allow an
        // asymmetric band around the grid value
        assert!(
            lsmc - fd < 0.10 && fd - lsmc < 0.25,
            "fd {fd:.4} vs lsmc {lsmc:.4}"
        );
    }

    #[test]
    fn tiny_vol_of_vol_degenerates_to_the_lattice() {
        let degenerate = HestonParams { v0: 0.09, kappa: 1.0, theta: 0.09, vol_of_vol: 1e-3, rho: 0.0 };
        let fd = {
            let b = EquityOptionBuilder::new()
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.30)
                .flat_rate(0.05)
                .dividend_yield(0.02)
                .years_to_maturity(1.0)
                .vanilla(PutOrCall::Put)
                .heston(degenerate)
                .engine(Engine::FiniteDifference)
                .american();
            b.build().unwrap().npv()
        };
        let lattice = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(0.05)
            .dividend_yield(0.02)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Put)
            .american()
            .engine(Engine::Binomial)
            .build()
            .unwrap()
            .npv();
        assert!((fd - lattice).abs() < 0.05, "adi {fd:.4} vs lattice {lattice:.4}");
    }
}
