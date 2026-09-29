//! The log-spot mesh shared by every quote of a capture, the per-expiry
//! time grid, and the per-step market slice (rates, carry, cash
//! dividends) read by the solver and the adjoint.
//!
//! Conventions (the adjoint depends on every one of them):
//!
//! ```text
//! Space:  x = ln S, uniform nodes x_j = x_min + j dx, j = 0..=n_x (n_x + 1 nodes),
//!         centred on x0 = ln S0 (x0 is a node when n_x is even).
//! Time:   0 = t_0 < t_1 < ... < t_N = T; uniform dt = T / n_t with every
//!         ex-dividend date in (0, T) inserted as a node (nodes within
//!         NODE_MERGE_TOL are merged, so no step has zero length), then the
//!         RANNACHER_SPLIT = 2 intervals adjacent to T, adjacent to t = 0
//!         and immediately before each ex-date are halved (Rannacher's
//!         start-up: N = n_t + 4 + 2 n_jumps steps for on-node ex-dates);
//!         backward step n advances the layer from level n+1 (time t_{n+1})
//!         to level n (time t_n) with dt_n = t_{n+1} - t_n and operator
//!         evaluated at t_mid[n] = (t_n + t_{n+1}) / 2.
//! Theta:  theta_n = 1 (fully implicit) for the RANNACHER_STEPS = 4
//!         half-steps adjacent to T, the 4 adjacent to t = 0, and the 4
//!         executed after each dividend jump (t < t_ex); 0.5 (Crank-Nicolson)
//!         otherwise. Half-steps quarter the first-order error of the
//!         implicit start-up at the same step count (measured at 1w ATM:
//!         4 full implicit steps at each end cost 6.5e-4 relative, 4
//!         half-steps 2.5e-4); the implicit steps at t = 0 damp the
//!         adjoint's delta seed so the kernel fields carry no checkerboard.
//! Jumps:  div_steps[k] = (m, delta): the jump S -> S - delta is applied to
//!         the layer at level m (t_m = ex-date), after backward step m has
//!         produced the pre-jump layer u^{m,+}.
//! Rates:  rate[n], carry[n] are the continuous rate and carry over step n
//!         (piecewise constant per step).
//! ```
//!
//! The mesh carries no market data except the centre `x0`; the
//! [`MarketSlice`] carries rates and dividends and is validated against
//! the mesh by the solver.

use super::vol_field::{NodeField, VolField};
use crate::core::errors::RustyQLibError;

/// Default number of spatial intervals (401 nodes).
pub const N_X_DEFAULT: usize = 400;
/// Spatial intervals of the fine reference mesh used for synthetic truth.
pub const N_X_FINE: usize = 1600;
/// Lower clamp of the per-expiry time-step count.
pub const N_T_MIN: usize = 40;
/// Upper clamp of the per-expiry time-step count (working mesh).
pub const N_T_MAX: usize = 250;
/// Upper clamp of the time-step count on the fine reference mesh.
pub const N_T_FINE_MAX: usize = 1000;
/// Time steps per year before clamping (two per calendar day).
pub const STEPS_PER_YEAR: f64 = 730.0;
/// Fully implicit (half-)steps at the terminal layer, at `t = 0` and after
/// each dividend jump (Rannacher 1984; Windcliff, Forsyth & Vetzal 2004).
pub const RANNACHER_STEPS: usize = 4;
/// Time intervals halved at each of those locations (`RANNACHER_STEPS / 2`),
/// so that the implicit steps are half-steps.
pub const RANNACHER_SPLIT: usize = 2;
/// Two time nodes closer than this (in years) are merged.
pub const NODE_MERGE_TOL: f64 = 1e-12;

/// Default number of uniform time steps for an expiry before the
/// Rannacher refinement: `clamp(ceil(730 T), 40, 250)` (the mesh then has
/// `n_t + 4 + 2 n_jumps` steps, plus one per off-node ex-date).
pub fn n_t_default(t_expiry: f64) -> usize {
    let raw = (STEPS_PER_YEAR * t_expiry).ceil();
    let raw = if raw.is_finite() && raw > 0.0 {
        raw as usize
    } else {
        N_T_MIN
    };
    raw.clamp(N_T_MIN, N_T_MAX)
}

/// Time steps of the fine reference mesh: four times the default, capped
/// at [`N_T_FINE_MAX`].
pub fn n_t_fine(t_expiry: f64) -> usize {
    (4 * n_t_default(t_expiry)).min(N_T_FINE_MAX)
}

/// Half-width `L` of the log-spot grid for a capture:
/// `max(4 sigma_ref sqrt(T_max) + drift_abs_max T_max, ln(S0/K_min) + 0.5,
/// ln(K_max/S0) + 0.5)`.
pub fn half_width(
    sigma_ref: f64,
    t_max: f64,
    drift_abs_max: f64,
    s0: f64,
    k_min: f64,
    k_max: f64,
) -> f64 {
    let diffusive = 4.0 * sigma_ref * t_max.sqrt() + drift_abs_max * t_max;
    let low = (s0 / k_min).ln() + 0.5;
    let high = (k_max / s0).ln() + 0.5;
    diffusive.max(low).max(high)
}

/// Whether the central first-difference keeps the row an M-matrix row:
/// `|mu| dx <= sigma^2` with `mu = r - q - sigma^2/2` the log drift
/// (equivalently `|r - q - sigma^2/2| dx <= sigma^2`). Rows that fail
/// are upwinded by the solver and the adjoint alike.
#[inline]
pub fn peclet_central_ok(mu: f64, sigma: f64, dx: f64) -> bool {
    mu.abs() * dx <= sigma * sigma
}

/// Linear-interpolation stencil of one cash-dividend jump on the mesh:
/// `(D u)_j = (1 - w[j]) u[idx[j]] + w[j] u[idx[j] + 1]` samples the
/// pre-jump layer at `ln(e^{x_j} - amount)`; nodes whose shifted spot
/// falls at or below the lowest node are clamped to it (`idx = 0, w = 0`).
#[derive(Debug, Clone, PartialEq)]
pub struct JumpStencil {
    /// Time level (index into `Mesh::t`) at which the jump is applied.
    pub level: usize,
    /// Cash amount `delta` (spot currency).
    pub amount: f64,
    /// Left node of the interpolation cell per target node.
    pub idx: Vec<usize>,
    /// Weight of the right node per target node, in `[0, 1]`.
    pub w: Vec<f64>,
}

impl JumpStencil {
    /// Build the stencil for a jump of `amount` on the nodes `x`.
    pub fn new(level: usize, amount: f64, x: &[f64]) -> Self {
        let n = x.len();
        let x_min = x[0];
        let dx = if n > 1 {
            (x[n - 1] - x_min) / (n - 1) as f64
        } else {
            1.0
        };
        let s_min = x_min.exp();
        let mut idx = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        for &xj in x {
            let s = xj.exp() - amount;
            if s <= s_min || n < 2 {
                idx.push(0);
                w.push(0.0);
                continue;
            }
            let pos = (s.ln() - x_min) / dx;
            let mut i = pos.floor();
            if i < 0.0 {
                i = 0.0;
            }
            let mut i = i as usize;
            if i >= n - 1 {
                i = n - 2;
            }
            let wj = (pos - i as f64).clamp(0.0, 1.0);
            idx.push(i);
            w.push(wj);
        }
        JumpStencil {
            level,
            amount,
            idx,
            w,
        }
    }

    /// Apply the interpolation `out = D u`.
    #[inline]
    pub fn apply(&self, u: &[f64], out: &mut [f64]) {
        for ((o, &i), &w) in out.iter_mut().zip(&self.idx).zip(&self.w) {
            *o = (1.0 - w) * u[i] + w * u[i + 1];
        }
    }

    /// Apply the transpose `out = D^T v` (overwrites `out`).
    #[inline]
    pub fn apply_transpose(&self, v: &[f64], out: &mut [f64]) {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        for ((&vj, &i), &w) in v.iter().zip(&self.idx).zip(&self.w) {
            out[i] += (1.0 - w) * vj;
            out[i + 1] += w * vj;
        }
    }
}

/// One expiry's discretization: the shared log-spot nodes and this
/// expiry's time grid with its theta pattern and dividend jumps.
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    /// Log-spot nodes `x_j`, `n_x + 1` of them, uniform.
    pub x: Vec<f64>,
    /// Node spacing.
    pub dx: f64,
    /// Number of spatial intervals (`x.len() - 1`).
    pub n_x: usize,
    /// Centre of the grid, `ln S0`.
    pub x0: f64,
    /// Linear interpolation of `x0`: `(i0, w0, w1)` with
    /// `f(x0) = w0 f(x_{i0}) + w1 f(x_{i0 + 1})`.
    pub x0_weights: (usize, f64, f64),
    /// Time nodes `0 = t_0 < ... < t_N = T`.
    pub t: Vec<f64>,
    /// Step lengths `dt_n = t_{n+1} - t_n`, `N` entries.
    pub dt: Vec<f64>,
    /// Step mid-times `(t_n + t_{n+1}) / 2`, where the operator is built.
    pub t_mid: Vec<f64>,
    /// Theta of each backward step (1 = implicit, 0.5 = Crank-Nicolson).
    pub theta: Vec<f64>,
    /// `(level, amount)` of each cash dividend applied on this mesh, in
    /// increasing time order.
    pub div_steps: Vec<(usize, f64)>,
    /// Interpolation stencils of the jumps, aligned with `div_steps`.
    pub div_stencils: Vec<JumpStencil>,
    /// Expiry `T = t_N`.
    pub t_expiry: f64,
}

impl Mesh {
    /// Build a mesh centred on `ln s0` with `n_x` intervals over
    /// `[x0 - half_width, x0 + half_width]`, `n_t` uniform time steps to
    /// `t_expiry`, the cash dividends `(t_j, delta_j)` with `0 < t_j < T`
    /// inserted as time nodes (those at or before the valuation date or
    /// at/after expiry are ignored with a debug log), and the Rannacher
    /// half-steps: the two intervals adjacent to `t = 0`, to `T` and to
    /// each ex-date (on the `t < t_j` side) are halved and the resulting
    /// four steps are fully implicit.
    pub fn new(
        s0: f64,
        half_width: f64,
        n_x: usize,
        t_expiry: f64,
        n_t: usize,
        dividends: &[(f64, f64)],
    ) -> Result<Mesh, RustyQLibError> {
        if !(s0.is_finite() && s0 > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "s0",
                format!("spot must be positive and finite, got {s0}"),
            ));
        }
        if !(half_width.is_finite() && half_width > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "half_width",
                format!("grid half-width must be positive and finite, got {half_width}"),
            ));
        }
        if n_x < 3 {
            return Err(RustyQLibError::invalid_input(
                "n_x",
                format!("at least 3 spatial intervals are required, got {n_x}"),
            ));
        }
        if !(t_expiry.is_finite() && t_expiry > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "t_expiry",
                format!("expiry must be positive and finite, got {t_expiry}"),
            ));
        }
        if n_t == 0 {
            return Err(RustyQLibError::invalid_input(
                "n_t",
                "at least one time step is required",
            ));
        }
        for &(tj, amount) in dividends {
            if !(tj.is_finite() && amount.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "dividends",
                    format!("non-finite dividend ({tj}, {amount})"),
                ));
            }
            if amount < 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "dividends",
                    format!("negative cash dividend {amount} at t = {tj}"),
                ));
            }
        }

        // ── space ────────────────────────────────────────────────────
        let x0 = s0.ln();
        let x_min = x0 - half_width;
        let dx = 2.0 * half_width / n_x as f64;
        let x: Vec<f64> = (0..=n_x).map(|j| x_min + j as f64 * dx).collect();
        let pos = (x0 - x_min) / dx;
        let mut i0 = pos.floor().max(0.0) as usize;
        if i0 > n_x - 1 {
            i0 = n_x - 1;
        }
        let mut w1 = (pos - i0 as f64).clamp(0.0, 1.0);
        // snap to the node when x0 sits on it up to rounding, so that an
        // even n_x reads the price at one node exactly
        if w1 < 1e-9 {
            w1 = 0.0;
        } else if w1 > 1.0 - 1e-9 {
            w1 = 0.0;
            i0 = (i0 + 1).min(n_x - 1);
            if i0 == n_x - 1 && (pos - (n_x - 1) as f64) > 0.5 {
                // x0 is the top node: interpolate from the last cell's right end
                w1 = 1.0;
            }
        }
        let x0_weights = (i0, 1.0 - w1, w1);

        // ── time ─────────────────────────────────────────────────────
        let mut t: Vec<f64> = (0..=n_t)
            .map(|i| i as f64 * t_expiry / n_t as f64)
            .collect();
        t[n_t] = t_expiry;
        let mut divs: Vec<(f64, f64)> = dividends
            .iter()
            .copied()
            .filter(|&(tj, amount)| {
                let keep = tj > NODE_MERGE_TOL && tj < t_expiry - NODE_MERGE_TOL && amount > 0.0;
                if !keep {
                    log::debug!(
                        "mesh: dividend ({tj}, {amount}) outside (0, T = {t_expiry}) or zero, not applied"
                    );
                }
                keep
            })
            .collect();
        divs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // merge dividends on the same date
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(divs.len());
        for (tj, amount) in divs {
            match merged.last_mut() {
                Some(last) if (last.0 - tj).abs() <= NODE_MERGE_TOL => last.1 += amount,
                _ => merged.push((tj, amount)),
            }
        }
        for &(tj, _) in &merged {
            let near = t.iter().any(|&tn| (tn - tj).abs() <= NODE_MERGE_TOL);
            if !near {
                let pos = t.partition_point(|&tn| tn < tj);
                t.insert(pos, tj);
            }
        }
        // Rannacher half-steps: halve the RANNACHER_SPLIT intervals adjacent
        // to t = 0, adjacent to T, and immediately before each ex-date
        let n0 = t.len() - 1;
        let mut split = vec![false; n0];
        for k in 0..RANNACHER_SPLIT.min(n0) {
            split[k] = true;
            split[n0 - 1 - k] = true;
        }
        for &(tj, _) in &merged {
            let m = t
                .iter()
                .position(|&tn| (tn - tj).abs() <= NODE_MERGE_TOL)
                .expect("inserted ex-date node is present");
            for k in 1..=RANNACHER_SPLIT {
                if m >= k {
                    split[m - k] = true;
                }
            }
        }
        let mut refined = Vec::with_capacity(n0 + 1 + split.iter().filter(|&&s| s).count());
        for i in 0..n0 {
            refined.push(t[i]);
            if split[i] {
                refined.push(0.5 * (t[i] + t[i + 1]));
            }
        }
        refined.push(t[n0]);
        let t = refined;
        let n_steps = t.len() - 1;
        let mut div_steps = Vec::with_capacity(merged.len());
        for &(tj, amount) in &merged {
            let level = t
                .iter()
                .position(|&tn| (tn - tj).abs() <= NODE_MERGE_TOL)
                .expect("inserted ex-date node is present");
            div_steps.push((level, amount));
        }
        let dt: Vec<f64> = (0..n_steps).map(|n| t[n + 1] - t[n]).collect();
        let t_mid: Vec<f64> = (0..n_steps).map(|n| 0.5 * (t[n] + t[n + 1])).collect();
        debug_assert!(dt.iter().all(|&d| d > 0.0), "zero-length time step");

        let mut theta = vec![0.5; n_steps];
        for k in 0..RANNACHER_STEPS {
            if k < n_steps {
                theta[n_steps - 1 - k] = 1.0;
                theta[k] = 1.0;
            }
        }
        for &(m, _) in &div_steps {
            for k in 1..=RANNACHER_STEPS {
                if m >= k {
                    theta[m - k] = 1.0;
                }
            }
        }
        let div_stencils = div_steps
            .iter()
            .map(|&(level, amount)| JumpStencil::new(level, amount, &x))
            .collect();

        Ok(Mesh {
            x,
            dx,
            n_x,
            x0,
            x0_weights,
            t,
            dt,
            t_mid,
            theta,
            div_steps,
            div_stencils,
            t_expiry,
        })
    }

    /// The working mesh: [`N_X_DEFAULT`] intervals and [`n_t_default`]
    /// steps.
    pub fn standard(
        s0: f64,
        half_width: f64,
        t_expiry: f64,
        dividends: &[(f64, f64)],
    ) -> Result<Mesh, RustyQLibError> {
        Mesh::new(
            s0,
            half_width,
            N_X_DEFAULT,
            t_expiry,
            n_t_default(t_expiry),
            dividends,
        )
    }

    /// The fine reference mesh for synthetic truth: [`N_X_FINE`]
    /// intervals and [`n_t_fine`] steps.
    pub fn fine_reference(
        s0: f64,
        half_width: f64,
        t_expiry: f64,
        dividends: &[(f64, f64)],
    ) -> Result<Mesh, RustyQLibError> {
        Mesh::new(
            s0,
            half_width,
            N_X_FINE,
            t_expiry,
            n_t_fine(t_expiry),
            dividends,
        )
    }

    /// Number of backward steps `N` (uniform steps plus inserted ex-date
    /// nodes plus the Rannacher half-steps).
    #[inline]
    pub fn n_steps(&self) -> usize {
        self.dt.len()
    }

    /// Number of nodes per layer (`n_x + 1`), the stride of flat fields.
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.x.len()
    }

    /// Spot at every node, `exp(x_j)`.
    pub fn s_nodes(&self) -> Vec<f64> {
        self.x.iter().map(|x| x.exp()).collect()
    }

    /// Index into `div_steps` of the jump applied at `level`, if any.
    #[inline]
    pub fn jump_at(&self, level: usize) -> Option<usize> {
        self.div_steps.iter().position(|&(m, _)| m == level)
    }

    /// The time node nearest `t`.
    pub fn level_of(&self, t: f64) -> usize {
        let pos = self.t.partition_point(|&tn| tn < t);
        if pos == 0 {
            0
        } else if pos >= self.t.len() {
            self.t.len() - 1
        } else if (self.t[pos] - t).abs() < (t - self.t[pos - 1]).abs() {
            pos
        } else {
            pos - 1
        }
    }

    /// Read a layer at `x0` with the stored interpolation weights.
    #[inline]
    pub fn read_x0(&self, layer: &[f64]) -> f64 {
        let (i0, w0, w1) = self.x0_weights;
        if w1 == 0.0 {
            w0 * layer[i0]
        } else {
            w0 * layer[i0] + w1 * layer[i0 + 1]
        }
    }

    /// Sample a volatility field at this mesh's nodes and step mid-times.
    pub fn node_field(&self, field: &dyn VolField) -> NodeField {
        NodeField::sample(&self.x, &self.t_mid, field)
    }

    /// Number of fully implicit steps (diagnostic).
    pub fn implicit_steps(&self) -> usize {
        self.theta.iter().filter(|&&th| th == 1.0).count()
    }
}

/// Per-step market data for one expiry mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketSlice {
    /// Spot at the valuation date.
    pub s0: f64,
    /// Continuous risk-free rate over each backward step (`N` entries).
    pub rate: Vec<f64>,
    /// Continuous carry (dividend yield / borrow) over each step.
    pub carry: Vec<f64>,
    /// Cash dividends `(t, amount)` the mesh was built from (informational;
    /// the solver applies `Mesh::div_steps`).
    pub dividends: Vec<(f64, f64)>,
}

impl MarketSlice {
    /// Flat rate and carry on every step of `mesh`.
    pub fn flat(mesh: &Mesh, s0: f64, r: f64, q: f64, dividends: &[(f64, f64)]) -> MarketSlice {
        let n = mesh.n_steps();
        MarketSlice {
            s0,
            rate: vec![r; n],
            carry: vec![q; n],
            dividends: dividends.to_vec(),
        }
    }

    /// Per-step rates from instantaneous forward-rate closures `r(t)`,
    /// `q(t)`: each step takes the Simpson average
    /// `(f(t_n) + 4 f(t_mid) + f(t_{n+1})) / 6` of the closure over the
    /// step (exact for a piecewise-constant forward that does not switch
    /// inside the step). Prefer [`MarketSlice::from_discount_factors`]
    /// when a discount curve is available.
    pub fn from_curves(
        mesh: &Mesh,
        s0: f64,
        r_fn: &dyn Fn(f64) -> f64,
        q_fn: &dyn Fn(f64) -> f64,
        dividends: &[(f64, f64)],
    ) -> MarketSlice {
        let avg = |f: &dyn Fn(f64) -> f64, n: usize| {
            (f(mesh.t[n]) + 4.0 * f(mesh.t_mid[n]) + f(mesh.t[n + 1])) / 6.0
        };
        let n = mesh.n_steps();
        MarketSlice {
            s0,
            rate: (0..n).map(|k| avg(r_fn, k)).collect(),
            carry: (0..n).map(|k| avg(q_fn, k)).collect(),
            dividends: dividends.to_vec(),
        }
    }

    /// Per-step rates from a discount-factor function `df(t)` (exact
    /// continuous forward `ln(df(t_n)/df(t_{n+1})) / dt_n`) and a carry
    /// closure averaged as in [`MarketSlice::from_curves`].
    pub fn from_discount_factors(
        mesh: &Mesh,
        s0: f64,
        df_fn: &dyn Fn(f64) -> f64,
        q_fn: &dyn Fn(f64) -> f64,
        dividends: &[(f64, f64)],
    ) -> MarketSlice {
        let n = mesh.n_steps();
        let rate = (0..n)
            .map(|k| (df_fn(mesh.t[k]) / df_fn(mesh.t[k + 1])).ln() / mesh.dt[k])
            .collect();
        let carry = (0..n)
            .map(|k| (q_fn(mesh.t[k]) + 4.0 * q_fn(mesh.t_mid[k]) + q_fn(mesh.t[k + 1])) / 6.0)
            .collect();
        MarketSlice {
            s0,
            rate,
            carry,
            dividends: dividends.to_vec(),
        }
    }

    /// From precomputed per-step slices (validated finite).
    pub fn from_steps(
        s0: f64,
        rate: Vec<f64>,
        carry: Vec<f64>,
        dividends: Vec<(f64, f64)>,
    ) -> Result<MarketSlice, RustyQLibError> {
        if !(s0.is_finite() && s0 > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "s0",
                format!("spot must be positive and finite, got {s0}"),
            ));
        }
        if rate.len() != carry.len() {
            return Err(RustyQLibError::invalid_input(
                "carry",
                format!(
                    "rate and carry slices differ in length ({} vs {})",
                    rate.len(),
                    carry.len()
                ),
            ));
        }
        if rate.iter().chain(carry.iter()).any(|v| !v.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "rate",
                "non-finite per-step rate or carry",
            ));
        }
        Ok(MarketSlice {
            s0,
            rate,
            carry,
            dividends,
        })
    }

    /// Check that the slice matches `mesh` (one rate and carry per step,
    /// spot consistent with the mesh centre).
    pub fn validate(&self, mesh: &Mesh) -> Result<(), RustyQLibError> {
        let n = mesh.n_steps();
        if self.rate.len() != n || self.carry.len() != n {
            return Err(RustyQLibError::invalid_input(
                "market",
                format!(
                    "market slice has {} rates / {} carries for a mesh with {} steps",
                    self.rate.len(),
                    self.carry.len(),
                    n
                ),
            ));
        }
        if !(self.s0.is_finite() && self.s0 > 0.0) || (self.s0.ln() - mesh.x0).abs() > 1e-9 {
            return Err(RustyQLibError::invalid_input(
                "market.s0",
                format!(
                    "spot {} does not match the mesh centre exp({}) = {}",
                    self.s0,
                    mesh.x0,
                    mesh.x0.exp()
                ),
            ));
        }
        Ok(())
    }
}

/// Model forward `F(t_n)` at every time node, cash dividends included:
/// `F(t) = S0 e^{int_0^t (r - q)} - sum_{t_j <= t} delta_j e^{int_{t_j}^t (r - q)}`.
pub fn forward_at_levels(mesh: &Mesh, market: &MarketSlice) -> Vec<f64> {
    let n = mesh.n_steps();
    let mut f = Vec::with_capacity(n + 1);
    let mut cur = market.s0;
    f.push(cur);
    for step in 0..n {
        cur *= ((market.rate[step] - market.carry[step]) * mesh.dt[step]).exp();
        if let Some(k) = mesh.jump_at(step + 1) {
            cur -= mesh.div_steps[k].1;
        }
        f.push(cur);
    }
    f
}

/// Effective flat rates of the expiry for implied-vol inversions:
/// `(r_eff, q_eff, forward, df)` with `df = D(T) = e^{-int r}`,
/// `r_eff = -ln D(T) / T`, `forward = F(T)` including cash dividends and
/// `q_eff = r_eff - ln(F(T)/S0) / T`. Inverting a model price with
/// `(r_eff, q_eff)` reproduces the model's forward and discount exactly.
/// A forward that is not positive (dividends exceed the grown spot)
/// yields `q_eff = NaN` and a warning.
pub fn effective_rates(mesh: &Mesh, market: &MarketSlice) -> (f64, f64, f64, f64) {
    let t = mesh.t_expiry;
    let int_r: f64 = market.rate.iter().zip(&mesh.dt).map(|(r, dt)| r * dt).sum();
    let df = (-int_r).exp();
    let r_eff = int_r / t;
    let forward = *forward_at_levels(mesh, market)
        .last()
        .expect("mesh has at least one level");
    let q_eff = if forward > 0.0 {
        r_eff - (forward / market.s0).ln() / t
    } else {
        log::warn!("effective_rates: non-positive forward {forward} (dividends exceed the spot)");
        f64::NAN
    };
    (r_eff, q_eff, forward, df)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_step_counts_clamp_and_scale() {
        assert_eq!(
            n_t_default(1.0 / 52.0),
            N_T_MIN,
            "1 week clamps to the floor"
        );
        assert_eq!(n_t_default(0.25), 183, "3 months: ceil(730 * 0.25) = 183");
        assert_eq!(n_t_default(1.0), N_T_MAX, "1 year clamps to the cap");
        assert_eq!(n_t_fine(0.25), 732, "fine mesh: 4x");
        assert_eq!(n_t_fine(1.0), N_T_FINE_MAX, "fine mesh cap");
        let l = half_width(0.25, 1.0, 0.05, 100.0, 70.0, 130.0);
        let expect = (4.0 * 0.25 + 0.05f64)
            .max((100.0f64 / 70.0).ln() + 0.5)
            .max((130.0f64 / 100.0).ln() + 0.5);
        assert!((l - expect).abs() < 1e-14, "half width {l} vs {expect}");
    }

    #[test]
    fn uniform_space_grid_puts_x0_on_a_node_for_even_n_x() {
        let mesh = Mesh::new(100.0, 1.0, 400, 0.5, 50, &[]).unwrap();
        assert_eq!(mesh.n_nodes(), 401);
        assert!((mesh.dx - 2.0 / 400.0).abs() < 1e-15, "dx {}", mesh.dx);
        let (i0, w0, w1) = mesh.x0_weights;
        assert_eq!(i0, 200, "x0 is the middle node");
        assert_eq!(w0, 1.0);
        assert_eq!(w1, 0.0);
        assert!((mesh.x[200] - 100f64.ln()).abs() < 1e-12, "x0 node value");
        let layer: Vec<f64> = mesh.x.iter().map(|x| 3.0 * x + 1.0).collect();
        assert!(
            (mesh.read_x0(&layer) - (3.0 * mesh.x0 + 1.0)).abs() < 1e-12,
            "x0 read of a linear layer"
        );
        // odd n_x: weights interpolate linearly and reproduce a linear layer
        let mesh = Mesh::new(100.0, 1.0, 401, 0.5, 50, &[]).unwrap();
        let (i0, w0, w1) = mesh.x0_weights;
        assert!(
            (w0 + w1 - 1.0).abs() < 1e-15 && w1 > 0.0 && w1 < 1.0,
            "weights {w0} {w1}"
        );
        assert!(mesh.x[i0] <= mesh.x0 + 1e-12 && mesh.x[i0 + 1] >= mesh.x0 - 1e-12);
        let layer: Vec<f64> = mesh.x.iter().map(|x| 3.0 * x + 1.0).collect();
        assert!(
            (mesh.read_x0(&layer) - (3.0 * mesh.x0 + 1.0)).abs() < 1e-12,
            "x0 read of a linear layer (odd)"
        );
    }

    #[test]
    fn time_grid_inserts_ex_dates_and_deduplicates_coincident_nodes() {
        // 0.2 is not a node of the 50-step grid on [0, 0.5]? dt = 0.01, so it is:
        // 20 * 0.01 = 0.2 up to rounding -> merged, no extra node
        // (+2 half-steps at t = 0, +2 at T, +2 before the ex-date)
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 50, &[(0.2, 1.5)]).unwrap();
        assert_eq!(
            mesh.n_steps(),
            56,
            "ex-date on a uniform node adds no step beyond the 6 half-steps"
        );
        assert_eq!(mesh.div_steps.len(), 1);
        assert_eq!(
            mesh.div_steps[0].0, 24,
            "jump level: node 20 shifted by the 4 inserted midpoints before it"
        );
        assert!((mesh.t[24] - 0.2).abs() < 1e-12);
        let m = 24;
        for k in 1..=2 {
            assert!(
                (mesh.dt[m - k] - 0.005).abs() < 1e-12,
                "half-step before the ex-date"
            );
        }
        assert!(
            (mesh.dt[0] - 0.005).abs() < 1e-12 && (mesh.dt[1] - 0.005).abs() < 1e-12,
            "half-steps at t = 0"
        );
        assert!(
            (mesh.dt[55] - 0.005).abs() < 1e-12 && (mesh.dt[54] - 0.005).abs() < 1e-12,
            "half-steps at T"
        );
        assert!((mesh.dt[10] - 0.01).abs() < 1e-12, "full step elsewhere");
        // an ex-date strictly between nodes is inserted
        let mesh = Mesh::new(
            100.0,
            1.0,
            10,
            0.5,
            50,
            &[(0.2037, 1.5), (0.0, 9.0), (0.5, 9.0)],
        )
        .unwrap();
        assert_eq!(
            mesh.n_steps(),
            57,
            "one inserted node + 6 half-steps; t = 0 and t = T dividends ignored"
        );
        assert_eq!(mesh.div_steps.len(), 1);
        let m = mesh.div_steps[0].0;
        assert!(
            (mesh.t[m] - 0.2037).abs() < 1e-12,
            "inserted node holds the ex-date"
        );
        for n in 0..mesh.n_steps() {
            assert!(mesh.dt[n] > 0.0, "positive step {n}");
            assert!((mesh.t_mid[n] - 0.5 * (mesh.t[n] + mesh.t[n + 1])).abs() < 1e-15);
        }
        let total: f64 = mesh.dt.iter().sum();
        assert!((total - 0.5).abs() < 1e-12, "steps sum to T: {total}");
        assert!(
            mesh.t.windows(2).all(|w| w[1] > w[0]),
            "strictly increasing nodes"
        );
        // two dividends on the same date merge
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 50, &[(0.2037, 1.0), (0.2037, 0.5)]).unwrap();
        assert_eq!(mesh.div_steps.len(), 1);
        assert!((mesh.div_steps[0].1 - 1.5).abs() < 1e-15, "merged amount");
    }

    #[test]
    fn theta_pattern_is_implicit_at_both_ends_and_after_each_jump() {
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 50, &[(0.2037, 1.5)]).unwrap();
        let n = mesh.n_steps();
        let m = mesh.div_steps[0].0;
        for k in 0..n {
            let expect = if k < 4 || k + 4 >= n || (k + 4 >= m && k < m) {
                1.0
            } else {
                0.5
            };
            assert_eq!(mesh.theta[k], expect, "theta at step {k} (jump level {m})");
        }
        assert_eq!(mesh.implicit_steps(), 12, "4 + 4 + 4 implicit steps");
        // the implicit steps are half-steps
        for k in 0..4 {
            assert!(
                (mesh.dt[k] - 0.5 * 0.01).abs() < 1e-12
                    && (mesh.dt[n - 1 - k] - 0.5 * 0.01).abs() < 1e-12
            );
        }
        // a short expiry with overlapping patterns never exceeds the step count
        let mesh = Mesh::new(100.0, 1.0, 10, 0.01, 4, &[]).unwrap();
        assert_eq!(mesh.n_steps(), 8, "4 uniform steps all halved");
        assert!(
            mesh.theta.iter().all(|&t| t == 1.0),
            "all implicit when N <= 8"
        );
        let mesh = Mesh::new(100.0, 1.0, 10, 0.01, 1, &[]).unwrap();
        assert_eq!(mesh.n_steps(), 2, "a single interval is halved once");
    }

    #[test]
    fn jump_stencil_interpolates_linearly_and_clamps_below_the_grid() {
        let mesh = Mesh::new(100.0, 1.0, 20, 0.5, 10, &[(0.25, 5.0)]).unwrap();
        let st = &mesh.div_stencils[0];
        let layer: Vec<f64> = mesh.x.iter().map(|x| 2.0 * x.exp() + 7.0).collect();
        let mut out = vec![0.0; mesh.n_nodes()];
        st.apply(&layer, &mut out);
        let s_min = mesh.x[0].exp();
        for (j, &xj) in mesh.x.iter().enumerate() {
            let s = xj.exp() - 5.0;
            if s <= s_min {
                assert!((out[j] - layer[0]).abs() < 1e-12, "clamped node {j}");
            } else {
                // linear-in-x interpolation of a function linear in S has
                // O(dx^2) error; check the exact interpolant instead
                let (i, w) = (st.idx[j], st.w[j]);
                let expect = (1.0 - w) * layer[i] + w * layer[i + 1];
                assert!((out[j] - expect).abs() < 1e-12, "node {j}");
                assert!(mesh.x[i] <= s.ln() + 1e-12 && s.ln() <= mesh.x[i + 1] + 1e-12);
            }
        }
        // transpose identity <D u, v> = <u, D^T v>
        let v: Vec<f64> = (0..mesh.n_nodes())
            .map(|j| (j as f64 * 0.37).sin())
            .collect();
        let mut dtv = vec![0.0; mesh.n_nodes()];
        st.apply_transpose(&v, &mut dtv);
        let lhs: f64 = out.iter().zip(&v).map(|(a, b)| a * b).sum();
        let rhs: f64 = layer.iter().zip(&dtv).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
            "transpose {lhs} vs {rhs}"
        );
    }

    #[test]
    fn effective_rates_reproduce_flat_inputs_and_fold_dividends_into_the_forward() {
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 50, &[]).unwrap();
        let market = MarketSlice::flat(&mesh, 100.0, 0.04, 0.02, &[]);
        let (r, q, f, df) = effective_rates(&mesh, &market);
        assert!((r - 0.04).abs() < 1e-14, "r_eff {r}");
        assert!((q - 0.02).abs() < 1e-13, "q_eff {q}");
        assert!(
            (f - 100.0 * (0.02f64 * 0.5).exp()).abs() < 1e-10,
            "forward {f}"
        );
        assert!((df - (-0.02f64).exp()).abs() < 1e-14, "df {df}");
        let divs = [(0.2, 1.5)];
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 50, &divs).unwrap();
        let market = MarketSlice::flat(&mesh, 100.0, 0.04, 0.0, &divs);
        let (r, q, f, _) = effective_rates(&mesh, &market);
        let expect_f = 100.0 * (0.04f64 * 0.5).exp() - 1.5 * (0.04f64 * 0.3).exp();
        assert!(
            (f - expect_f).abs() < 1e-10,
            "forward with dividend {f} vs {expect_f}"
        );
        assert!((r - 0.04).abs() < 1e-14);
        assert!(
            (q - (r - (expect_f / 100.0).ln() / 0.5)).abs() < 1e-12,
            "q_eff {q}"
        );
        let levels = forward_at_levels(&mesh, &market);
        assert_eq!(levels.len(), mesh.n_steps() + 1);
        let m = mesh.div_steps[0].0;
        assert!(
            (levels[m] - (100.0 * (0.04f64 * 0.2).exp() - 1.5)).abs() < 1e-10,
            "F(t_ex)"
        );
    }

    #[test]
    fn per_step_rates_from_curves_match_closed_forms() {
        let mesh = Mesh::new(100.0, 1.0, 10, 1.0, 100, &[]).unwrap();
        // linear instantaneous forward: Simpson is exact
        let r_fn = |t: f64| 0.03 + 0.02 * t;
        let market = MarketSlice::from_curves(&mesh, 100.0, &r_fn, &|_| 0.01, &[]);
        for n in 0..mesh.n_steps() {
            let exact = 0.03 + 0.02 * mesh.t_mid[n];
            assert!((market.rate[n] - exact).abs() < 1e-14, "step {n}");
            assert!((market.carry[n] - 0.01).abs() < 1e-15);
        }
        // discount factors: exact per-step forward of a quadratic zero curve
        let df_fn = |t: f64| (-(0.03 * t + 0.01 * t * t)).exp();
        let market = MarketSlice::from_discount_factors(&mesh, 100.0, &df_fn, &|_| 0.0, &[]);
        let (r_eff, _, _, df) = effective_rates(&mesh, &market);
        assert!((df - df_fn(1.0)).abs() < 1e-14, "df {df}");
        assert!((r_eff - 0.04).abs() < 1e-13, "r_eff {r_eff}");
        assert!(market.validate(&mesh).is_ok());
        let bad = MarketSlice::from_steps(100.0, vec![0.01; 3], vec![0.0; 3], vec![]).unwrap();
        assert!(bad.validate(&mesh).is_err(), "length mismatch is rejected");
        assert!(
            MarketSlice::from_steps(100.0, vec![f64::NAN], vec![0.0], vec![]).is_err(),
            "non-finite rate is rejected"
        );
        assert!(
            Mesh::new(100.0, 1.0, 2, 0.5, 10, &[]).is_err(),
            "n_x < 3 is rejected"
        );
        assert!(
            Mesh::new(100.0, 1.0, 10, 0.0, 10, &[]).is_err(),
            "T = 0 is rejected"
        );
    }

    #[test]
    fn peclet_condition_matches_the_m_matrix_bound() {
        // interior rows: lower = s2/dx^2 - mu/(2dx) >= 0 iff mu dx <= sigma^2
        let (sigma, dx) = (0.02f64, 0.005f64);
        assert!(
            peclet_central_ok(0.05, sigma, dx),
            "holds at n_x = 400 scale"
        );
        assert!(!peclet_central_ok(0.05, sigma, 0.01), "fails at dx = 0.01");
        // nodes 0, .05, .1, .15, .2, .3, .35, .4, .45, .5
        let mesh = Mesh::new(100.0, 1.0, 10, 0.5, 5, &[]).unwrap();
        assert_eq!(mesh.n_steps(), 9);
        assert_eq!(mesh.level_of(0.21), 4);
        assert_eq!(mesh.level_of(-1.0), 0);
        assert_eq!(mesh.level_of(9.0), 9);
        assert_eq!(mesh.jump_at(3), None);
    }
}
