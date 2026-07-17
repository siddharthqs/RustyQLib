//! Finite difference pricer for the Black-Scholes PDE.
//!
//! Solves the log-spot PDE
//! `V_t + (r - q - sigma^2/2) V_x + (sigma^2/2) V_xx - r V = 0`, `x = ln S`,
//! backwards from the terminal payoff on a uniform log-spot grid with a
//! theta-scheme: Crank-Nicolson (`theta = 1/2`) with the first four steps
//! fully implicit (Rannacher smoothing), which damps the oscillations CN
//! produces on kinked/discontinuous payoffs such as binaries.
//!
//! The payoff enters only through the [`Payoff`](crate::equity::utils::Payoff)
//! trait, so any terminal payoff prices without changes here:
//! - terminal condition: cell average of the payoff (integrates kinks and
//!   digital jumps correctly instead of sampling them at a node),
//! - Dirichlet boundaries: `V(S_b, tau) = e^{-r tau} * payoff(S_b e^{(r-q) tau})`,
//!   which reduces to the familiar conditions for calls, puts and binaries,
//! - American exercise: projection `V = max(V, payoff)` after each step.

use super::vanila_option::EquityOption;
use crate::core::utils::ContractStyle;
use crate::equity::utils::Payoff;

const SPOT_STEPS: usize = 400; // grid nodes in x (even, so ln(S0) is a node)
const TIME_STEPS: usize = 400;
const RANNACHER_STEPS: usize = 4;
const GRID_STDEVS: f64 = 5.0;
const CELL_AVG_POINTS: usize = 16;

pub fn npv(option: &EquityOption) -> f64 {
    let s0 = option.base.underlying_price.value();
    let strike = option.base.strike_price;
    let t = option.time_to_maturity();
    let sigma = option.base.volatility();
    let r = option.base.risk_free_rate();
    let q = option.base.dividend_yield;
    assert!(sigma > 0.0, "volatility must be positive");
    assert!(t >= 0.0, "Option is expired or negative time");
    assert!(s0 > 0.0, "underlying price must be positive");
    if t == 0.0 {
        return option.payoff.payoff(s0, strike);
    }
    solve(option.payoff.as_ref(), s0, strike, r, q, sigma, t)
}

fn solve(
    payoff: &dyn Payoff,
    s0: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
) -> f64 {
    let american = matches!(payoff.exercise_style(), ContractStyle::American);
    let x0 = s0.ln();
    // grid wide enough that the boundaries are asymptotic for both the spot
    // and the strike region
    let drift = (r - q - 0.5 * sigma * sigma) * t;
    let half_width =
        GRID_STDEVS * sigma * t.sqrt() + drift.abs() + (strike / s0).ln().abs().max(1e-2);
    let n = SPOT_STEPS; // nodes 0..=n, x0 at node n/2
    let dx = 2.0 * half_width / n as f64;
    let x_min = x0 - half_width;
    let x_at = |i: usize| x_min + i as f64 * dx;

    // terminal condition: cell-averaged payoff (handles kinks and digital
    // jumps at nodes without O(dx) sampling error)
    let mut v: Vec<f64> = (0..=n)
        .map(|i| cell_average_payoff(payoff, strike, x_at(i), dx))
        .collect();

    // constant PDE coefficients in log-spot
    let mu = r - q - 0.5 * sigma * sigma;
    let s2 = 0.5 * sigma * sigma;
    let lower = s2 / (dx * dx) - mu / (2.0 * dx);
    let diag = -2.0 * s2 / (dx * dx) - r;
    let upper = s2 / (dx * dx) + mu / (2.0 * dx);

    let dt = t / TIME_STEPS as f64;
    // interior unknowns are nodes 1..=n-1
    let m = n - 1;

    let mut rhs = vec![0.0; m];
    let mut sub = vec![0.0; m - 1];
    let mut dia = vec![0.0; m];
    let mut sup = vec![0.0; m - 1];

    for step in 0..TIME_STEPS {
        // Rannacher: first steps after expiry fully implicit, then CN
        let theta = if step < RANNACHER_STEPS { 1.0 } else { 0.5 };
        let tau_new = (step + 1) as f64 * dt; // time to expiry after this step

        // boundary values at the new time level
        let fwd_growth = ((r - q) * tau_new).exp();
        let disc = (-r * tau_new).exp();
        let boundary = |x: f64| -> f64 {
            let s_b = x.exp();
            let mut val = disc * payoff.payoff(s_b * fwd_growth, strike);
            if american {
                val = val.max(payoff.payoff(s_b, strike));
            }
            val
        };
        let v_low = boundary(x_at(0));
        let v_high = boundary(x_at(n));

        // rhs = (I + (1-theta) dt A) v  restricted to interior nodes
        for i in 1..n {
            let av = lower * v[i - 1] + diag * v[i] + upper * v[i + 1];
            rhs[i - 1] = v[i] + (1.0 - theta) * dt * av;
        }
        // move the implicit boundary terms to the rhs
        rhs[0] += theta * dt * lower * v_low;
        rhs[m - 1] += theta * dt * upper * v_high;

        // lhs = (I - theta dt A)
        for i in 0..m {
            dia[i] = 1.0 - theta * dt * diag;
        }
        for i in 0..m - 1 {
            sub[i] = -theta * dt * lower;
            sup[i] = -theta * dt * upper;
        }

        let interior = thomas_algorithm(&sub, &dia, &sup, &rhs);
        v[0] = v_low;
        v[n] = v_high;
        v[1..n].copy_from_slice(&interior);

        if american {
            for i in 0..=n {
                let exercise = payoff.payoff(x_at(i).exp(), strike);
                if v[i] < exercise {
                    v[i] = exercise;
                }
            }
        }
    }

    v[n / 2]
}

/// Average of the payoff over the grid cell `[x - dx/2, x + dx/2]` (midpoint
/// sampling in log-spot). For smooth regions this is second-order accurate;
/// at a digital jump it assigns the node the correct cell fraction.
fn cell_average_payoff(payoff: &dyn Payoff, strike: f64, x: f64, dx: f64) -> f64 {
    let k = CELL_AVG_POINTS;
    let mut sum = 0.0;
    for j in 0..k {
        let xi = x - 0.5 * dx + (j as f64 + 0.5) * dx / k as f64;
        sum += payoff.payoff(xi.exp(), strike);
    }
    sum / k as f64
}

/// Solves a tridiagonal system `A x = d` where `a` is the sub-diagonal
/// (`a[i-1]` multiplies `x[i-1]` in row `i`), `b` the diagonal and `c` the
/// super-diagonal (`c[i]` multiplies `x[i+1]` in row `i`).
/// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
pub fn thomas_algorithm(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    assert!(b.len() == n && a.len() == n - 1 && c.len() == n - 1);
    if n == 1 {
        return vec![d[0] / b[0]];
    }
    let mut c_ = c.to_vec();
    let mut d_ = d.to_vec();
    let mut x: Vec<f64> = vec![0.0; n];

    c_[0] = c_[0] / b[0];
    d_[0] = d_[0] / b[0];
    for i in 1..n - 1 {
        let id = 1.0 / (b[i] - a[i - 1] * c_[i - 1]);
        c_[i] = c_[i] * id;
        d_[i] = (d_[i] - a[i - 1] * d_[i - 1]) * id;
    }
    d_[n - 1] = (d_[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2]);

    x[n - 1] = d_[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_[i] - c_[i] * x[i + 1];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thomas_solves_small_system() {
        // [2 1 0; 1 2 1; 0 1 2] x = [4; 8; 8] -> x = [1; 2; 3]
        let x = thomas_algorithm(&[1.0, 1.0], &[2.0, 2.0, 2.0], &[1.0, 1.0], &[4.0, 8.0, 8.0]);
        for (got, want) in x.iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "{x:?}");
        }
    }
}
