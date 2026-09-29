//! Penalized American / European backward solver on the log-spot mesh.
//!
//! One quote, one expiry mesh, one march from `T` back to `0`. The
//! discrete map is written so that [`super::adjoint`] can be its exact
//! transpose; every convention below is therefore load-bearing.
//!
//! ```text
//! Operator (per node j, step n, ONE operator per step built at t_mid[n] and
//! used for both theta halves), with s2 = sigma_{j,n}^2 / 2 and
//! mu = r_n - q_n - s2:
//!   central:   lower = s2/dx^2 - mu/(2dx),  diag = -2 s2/dx^2 - r_n,  upper = s2/dx^2 + mu/(2dx)
//!   upwinded (only where |mu| dx > sigma^2, see grid::peclet_central_ok):
//!     mu >= 0:  lower = s2/dx^2,            diag = -2 s2/dx^2 - mu/dx - r_n,  upper = s2/dx^2 + mu/dx
//!     mu <  0:  lower = s2/dx^2 - mu/dx,    diag = -2 s2/dx^2 + mu/dx - r_n,  upper = s2/dx^2
//!   edge rows (linear in S: u_SS = 0 <=> u_xx = u_x, the sigma terms cancel;
//!   Windcliff, Forsyth & Vetzal 2004), with c = r_n - q_n:
//!     row 0:    diag = -c/dx - r_n,  upper = c/dx
//!     row n_x:  lower = -c/dx,       diag = c/dx - r_n
//!   The edge nodes are unknowns: all bands have n_x + 1 rows.
//!
//! Step n (level n+1 -> level n), theta = theta_n, dt = dt_n, P_n = diag(1{u < psi}):
//!   A_n = I - theta dt L_n + rho dt P_n,   B_n = I + (1 - theta) dt L_n,   c_n = rho dt P_n psi
//!   A_n u^{n,+} = B_n u^{n+1} + c_n
//!   penalty iteration: P from the previous step's final active set, solve,
//!   recompute P = 1{u < psi}, repeat until P is unchanged (cap 20); on exit
//!   P(u_final) == P_used holds whenever the cap was not hit (counted).
//!   European mode: P == 0 (a single solve per step).
//! Dividend jump at level n (t_n an ex-date): u^n = M_n D u^{n,+} + (I - M_n) psi,
//!   D = linear interpolation in x at ln(e^x - delta) (clamped at the low end),
//!   M_n = diag(1{(D u^{n,+})_j > psi_j}) in American mode, I in European mode.
//!   The pre-jump layer u^{n,+} is what the retained grid stores at level n;
//!   the post-jump layer u^n and the mask M_n are stored separately.
//! Terminal layer: cell-averaged payoff (exact integral over
//!   [x - dx/2, x + dx/2]); obstacle psi_j = plain node payoff.
//! Price: u^0 read at x0 with the mesh's linear interpolation weights.
//! ```
//!
//! `rho` is a rate (1/year), default [`RHO_DEFAULT`]: the penalty error is
//! `u - psi ~ (L psi)/rho`, scale-free in the price.
//!
//! Entry points: [`solve_backward`] / [`solve_into`] on a [`NodeField`]
//! (the hot path: one memcpy per step for `sigma`), the `_dyn` variants on
//! any [`VolField`], [`price_only`] without retention, and
//! [`price_frozen`] which re-solves with the active sets and jump masks of
//! a previous solution held fixed (the frozen-active-set finite
//! differences that verify the adjoint).

// Stencil code indexes neighbours j-1, j+1 of the same row; index loops
// are clearer than zipped iterators here.
#![allow(clippy::needless_range_loop)]

use super::grid::{peclet_central_ok, MarketSlice, Mesh};
use super::vol_field::{NodeField, VolField};
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;

/// Default penalty rate (1/year).
pub const RHO_DEFAULT: f64 = 1e7;
/// Cap on penalty iterations per backward step. The iteration activates
/// nodes in one shot but releases them one node per iteration, so the
/// steps nearest expiry (where the free boundary moves several nodes per
/// step) take 5-15 iterations; elsewhere 1-3. The cap only guards against
/// cycling.
pub const MAX_PENALTY_ITERATIONS: usize = 50;
/// Half-width, in units of `sigma sqrt(dt)`, of the strike neighbourhood
/// excluded from the first step's seed active set.
pub const SEED_MARGIN_SIGMAS: f64 = 4.0;

// ── Quote and mode ───────────────────────────────────────────────────────

/// A vanilla quote to price on a mesh: strike, expiry (years, must equal
/// the mesh's `t_expiry`) and right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteSpec {
    /// Strike in spot currency.
    pub strike: f64,
    /// Expiry in years.
    pub t_expiry: f64,
    /// Put or call.
    pub right: PutOrCall,
}

impl QuoteSpec {
    /// Build a quote spec.
    pub fn new(strike: f64, t_expiry: f64, right: PutOrCall) -> Self {
        QuoteSpec {
            strike,
            t_expiry,
            right,
        }
    }

    /// Plain payoff `psi(S)` at spot `s`.
    #[inline]
    pub fn payoff(&self, s: f64) -> f64 {
        match self.right {
            PutOrCall::Call => (s - self.strike).max(0.0),
            PutOrCall::Put => (self.strike - s).max(0.0),
        }
    }

    /// Validate strike and expiry against a mesh.
    pub fn validate(&self, mesh: &Mesh) -> Result<(), RustyQLibError> {
        if !(self.strike.is_finite() && self.strike > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "strike",
                format!("strike must be positive and finite, got {}", self.strike),
            ));
        }
        let tol = 1e-10 * mesh.t_expiry.max(1.0);
        if !self.t_expiry.is_finite() || (self.t_expiry - mesh.t_expiry).abs() > tol {
            return Err(RustyQLibError::invalid_input(
                "t_expiry",
                format!(
                    "quote expiry {} does not match the mesh expiry {}",
                    self.t_expiry, mesh.t_expiry
                ),
            ));
        }
        Ok(())
    }
}

/// Exercise style of the backward solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// No early exercise (penalty off).
    European,
    /// Early exercise through the penalty term `rho (psi - u)^+`, `rho` in
    /// 1/year.
    American {
        /// Penalty rate (1/year).
        rho: f64,
    },
}

impl Mode {
    /// American mode at the default penalty rate.
    pub fn american() -> Mode {
        Mode::American { rho: RHO_DEFAULT }
    }

    /// Whether early exercise is on.
    #[inline]
    pub fn is_american(&self) -> bool {
        matches!(self, Mode::American { .. })
    }

    /// Penalty rate (`0` in European mode).
    #[inline]
    pub fn rho(&self) -> f64 {
        match self {
            Mode::European => 0.0,
            Mode::American { rho } => *rho,
        }
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        if let Mode::American { rho } = self {
            if !(rho.is_finite() && *rho > 0.0) {
                return Err(RustyQLibError::invalid_input(
                    "rho",
                    format!("penalty rate must be positive and finite, got {rho}"),
                ));
            }
        }
        Ok(())
    }
}

// ── Volatility access ────────────────────────────────────────────────────

/// How the march reads `sigma` at (step, node). Implemented by
/// [`NodeField`] (the hot path) and by a `dyn VolField` adapter.
pub(super) trait StepVol {
    /// `sigma_{j,n}` at backward step `step`, node `node`.
    fn sigma(&self, step: usize, node: usize) -> f64;
    /// Fill `out[j] = sigma_{j, step}` for every node.
    fn fill_row(&self, step: usize, out: &mut [f64]);
    /// Dimension check against the mesh.
    fn check(&self, mesh: &Mesh) -> Result<(), RustyQLibError>;
}

impl StepVol for NodeField {
    #[inline]
    fn sigma(&self, step: usize, node: usize) -> f64 {
        self.at(step, node)
    }

    #[inline]
    fn fill_row(&self, step: usize, out: &mut [f64]) {
        let s = self.stride;
        out[..s].copy_from_slice(&self.values[step * s..(step + 1) * s]);
    }

    fn check(&self, mesh: &Mesh) -> Result<(), RustyQLibError> {
        if self.stride != mesh.n_nodes()
            || self.steps() != mesh.n_steps()
            || self.values.len() != self.stride * self.steps()
        {
            return Err(RustyQLibError::invalid_input(
                "vol",
                format!(
                    "node field is {} steps x {} nodes ({} values) for a mesh with {} steps x {} nodes",
                    self.steps(),
                    self.stride,
                    self.values.len(),
                    mesh.n_steps(),
                    mesh.n_nodes()
                ),
            ));
        }
        if self.values.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(RustyQLibError::invalid_input(
                "vol",
                "node field has a non-positive or non-finite volatility",
            ));
        }
        Ok(())
    }
}

/// A `dyn VolField` read at the mesh nodes and step mid-times.
pub(super) struct DynVol<'a> {
    pub(super) field: &'a dyn VolField,
    pub(super) mesh: &'a Mesh,
}

impl StepVol for DynVol<'_> {
    #[inline]
    fn sigma(&self, step: usize, node: usize) -> f64 {
        self.field.vol(self.mesh.x[node], self.mesh.t_mid[step])
    }

    fn fill_row(&self, step: usize, out: &mut [f64]) {
        let t = self.mesh.t_mid[step];
        for (o, &x) in out.iter_mut().zip(&self.mesh.x) {
            *o = self.field.vol(x, t);
        }
    }

    fn check(&self, _mesh: &Mesh) -> Result<(), RustyQLibError> {
        Ok(())
    }
}

// ── Shared kernels (forward and adjoint) ─────────────────────────────────

/// Allocation-free Thomas solve of the tridiagonal system with sub-diagonal
/// `a` (`n - 1`), diagonal `b` (`n`), super-diagonal `c` (`n - 1`) and
/// right-hand side `d` (`n`), writing the solution into `x`; `cw`, `dw`
/// are scratch of length `>= n`. Same elimination order as
/// `crate::core::fd_solvers::thomas_algorithm` (bit-identical results).
/// The transpose system is solved by swapping `a` and `c`.
///
/// Panics on a near-zero pivot (the system is not diagonally dominant),
/// a programmer-error invariant as in the crate kernel.
pub(super) fn thomas_inplace(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    cw: &mut [f64],
    dw: &mut [f64],
    x: &mut [f64],
) {
    let n = d.len();
    debug_assert!(b.len() == n && a.len() + 1 == n && c.len() + 1 == n);
    debug_assert!(cw.len() >= n && dw.len() >= n && x.len() >= n);
    #[inline(always)]
    fn check(piv: f64, b_i: f64, i: usize) -> f64 {
        assert!(
            piv.abs() > 1e-14 * (1.0 + b_i.abs()),
            "tridiagonal solve broke down at row {i} (near-zero pivot)"
        );
        piv
    }
    if n == 1 {
        x[0] = d[0] / check(b[0], b[0], 0);
        return;
    }
    let b0 = check(b[0], b[0], 0);
    cw[0] = c[0] / b0;
    dw[0] = d[0] / b0;
    for i in 1..n - 1 {
        let id = 1.0 / check(b[i] - a[i - 1] * cw[i - 1], b[i], i);
        cw[i] = c[i] * id;
        dw[i] = (d[i] - a[i - 1] * dw[i - 1]) * id;
    }
    dw[n - 1] =
        (d[n - 1] - a[n - 2] * dw[n - 2]) / check(b[n - 1] - a[n - 2] * cw[n - 2], b[n - 1], n - 1);
    x[n - 1] = dw[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dw[i] - cw[i] * x[i + 1];
    }
}

/// Assemble the step operator `L` (bands indexed by ROW: `lower[j]` is the
/// coefficient of `u_{j-1}` in row `j`, `upper[j]` that of `u_{j+1}`;
/// `lower[0]` and `upper[n_x]` are unused and set to 0). `upwind[j] = 1`
/// marks interior rows that fell back to the one-sided drift stencil.
/// Returns the number of upwinded rows. Used identically by the forward
/// march and the adjoint.
#[allow(clippy::too_many_arguments)]
pub(super) fn assemble_operator(
    sigma: &[f64],
    r: f64,
    q: f64,
    dx: f64,
    lower: &mut [f64],
    diag: &mut [f64],
    upper: &mut [f64],
    upwind: &mut [u8],
) -> usize {
    let n = sigma.len();
    let n_x = n - 1;
    let inv_dx2 = 1.0 / (dx * dx);
    let inv_2dx = 0.5 / dx;
    let inv_dx = 1.0 / dx;
    let mut count = 0usize;
    for j in 1..n_x {
        let s = sigma[j];
        let s2 = 0.5 * s * s;
        let mu = r - q - s2;
        if peclet_central_ok(mu, s, dx) {
            lower[j] = s2 * inv_dx2 - mu * inv_2dx;
            diag[j] = -2.0 * s2 * inv_dx2 - r;
            upper[j] = s2 * inv_dx2 + mu * inv_2dx;
            upwind[j] = 0;
        } else {
            count += 1;
            upwind[j] = 1;
            if mu >= 0.0 {
                lower[j] = s2 * inv_dx2;
                diag[j] = -2.0 * s2 * inv_dx2 - mu * inv_dx - r;
                upper[j] = s2 * inv_dx2 + mu * inv_dx;
            } else {
                lower[j] = s2 * inv_dx2 - mu * inv_dx;
                diag[j] = -2.0 * s2 * inv_dx2 + mu * inv_dx - r;
                upper[j] = s2 * inv_dx2;
            }
        }
    }
    let c = r - q;
    lower[0] = 0.0;
    diag[0] = -c * inv_dx - r;
    upper[0] = c * inv_dx;
    upwind[0] = 0;
    lower[n_x] = -c * inv_dx;
    diag[n_x] = c * inv_dx - r;
    upper[n_x] = 0.0;
    upwind[n_x] = 0;
    count
}

/// `out_j = u_j + scale (L u)_j` for every row (edge rows included).
#[inline]
pub(super) fn apply_operator(
    u: &[f64],
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    scale: f64,
    out: &mut [f64],
) {
    let n = u.len();
    let n_x = n - 1;
    out[0] = u[0] + scale * (diag[0] * u[0] + upper[0] * u[1]);
    for j in 1..n_x {
        out[j] = u[j] + scale * (lower[j] * u[j - 1] + diag[j] * u[j] + upper[j] * u[j + 1]);
    }
    out[n_x] = u[n_x] + scale * (lower[n_x] * u[n_x - 1] + diag[n_x] * u[n_x]);
}

/// `out_j = v_j + scale (L^T v)_j`: the transpose of [`apply_operator`]
/// with the same row-indexed bands (`(L^T v)_j = upper_{j-1} v_{j-1} +
/// diag_j v_j + lower_{j+1} v_{j+1}`).
#[inline]
pub(super) fn apply_operator_transpose(
    v: &[f64],
    lower: &[f64],
    diag: &[f64],
    upper: &[f64],
    scale: f64,
    out: &mut [f64],
) {
    let n = v.len();
    let n_x = n - 1;
    out[0] = v[0] + scale * (diag[0] * v[0] + lower[1] * v[1]);
    for j in 1..n_x {
        out[j] =
            v[j] + scale * (upper[j - 1] * v[j - 1] + diag[j] * v[j] + lower[j + 1] * v[j + 1]);
    }
    out[n_x] = v[n_x] + scale * (upper[n_x - 1] * v[n_x - 1] + diag[n_x] * v[n_x]);
}

/// `d/d sigma_j` of row `j` of `L u` divided by `sigma_j`: the discrete
/// `S^2 u_SS` the row actually uses, `delta_xx u - delta_x u` with the
/// central first difference on central rows and the same one-sided
/// difference the operator used on upwinded rows (`upwind = 1`, sign of
/// `mu`). Interior rows only; edge rows carry no `sigma`.
#[inline]
pub(super) fn d2_at(u: &[f64], j: usize, dx: f64, upwind: u8, mu: f64) -> f64 {
    let second = (u[j + 1] - 2.0 * u[j] + u[j - 1]) / (dx * dx);
    let first = if upwind == 0 {
        (u[j + 1] - u[j - 1]) / (2.0 * dx)
    } else if mu >= 0.0 {
        (u[j + 1] - u[j]) / dx
    } else {
        (u[j] - u[j - 1]) / dx
    };
    second - first
}

/// Discrete `S u_S = u_x` at node `j` as the operator uses it: central on
/// interior rows, one-sided (`upwind` = 1 with the sign of `mu`) where the
/// row was upwinded, one-sided at the two edge rows. `d/d(r - q)` of row
/// `j` of `L u` at fixed discounting.
#[inline]
pub(super) fn d1_at(u: &[f64], j: usize, dx: f64, upwind: u8, mu: f64) -> f64 {
    let n_x = u.len() - 1;
    if j == 0 {
        (u[1] - u[0]) / dx
    } else if j == n_x {
        (u[n_x] - u[n_x - 1]) / dx
    } else if upwind == 0 {
        (u[j + 1] - u[j - 1]) / (2.0 * dx)
    } else if mu >= 0.0 {
        (u[j + 1] - u[j]) / dx
    } else {
        (u[j] - u[j - 1]) / dx
    }
}

/// Cell-averaged payoff over `[x - dx/2, x + dx/2]`, the terminal layer:
/// the exact integral of the vanilla payoff over the cell,
/// `(1/dx) int_{x-dx/2}^{x+dx/2} (e^y - K)^+ dy` (call) and its put
/// counterpart. (A 16-point midpoint rule leaves an `O(K dx / 2048)` error
/// on the cell containing the kink, which was the dominant price error of
/// OTM quotes on the working mesh.)
pub fn cell_average_payoff(spec: &QuoteSpec, x: f64, dx: f64) -> f64 {
    let a = x - 0.5 * dx;
    let b = x + 0.5 * dx;
    let k = spec.strike;
    let c = k.ln();
    // int_a^b (e^y - K) dy over the part of the cell where the payoff is live
    let call_part = |lo: f64, hi: f64| (hi.exp() - lo.exp()) - k * (hi - lo);
    match spec.right {
        PutOrCall::Call => {
            if c >= b {
                0.0
            } else if c <= a {
                call_part(a, b) / dx
            } else {
                call_part(c, b) / dx
            }
        }
        PutOrCall::Put => {
            if c <= a {
                0.0
            } else if c >= b {
                -call_part(a, b) / dx
            } else {
                -call_part(a, c) / dx
            }
        }
    }
}

// ── Workspace and solution ───────────────────────────────────────────────

/// Scratch buffers of one backward march, reused across quotes and
/// Jacobians (`par_iter().map_init(Workspace::new, ...)`). Grows to the
/// mesh's node count on first use and never shrinks.
#[derive(Debug, Clone, Default)]
pub struct Workspace {
    n: usize,
    sig: Vec<f64>,
    psi: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    rhs: Vec<f64>,
    rhs_pen: Vec<f64>,
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
    upwind: Vec<u8>,
    sub: Vec<f64>,
    sup: Vec<f64>,
    dia0: Vec<f64>,
    dia: Vec<f64>,
    cw: Vec<f64>,
    dw: Vec<f64>,
    active: Vec<u8>,
    active_new: Vec<u8>,
    tmp: Vec<f64>,
}

impl Workspace {
    /// An empty workspace (allocates on first solve).
    pub fn new() -> Self {
        Workspace::default()
    }

    fn ensure(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        for v in [
            &mut self.sig,
            &mut self.psi,
            &mut self.u,
            &mut self.v,
            &mut self.rhs,
            &mut self.rhs_pen,
            &mut self.lower,
            &mut self.diag,
            &mut self.upper,
            &mut self.dia0,
            &mut self.dia,
            &mut self.cw,
            &mut self.dw,
            &mut self.tmp,
        ] {
            v.clear();
            v.resize(n, 0.0);
        }
        self.sub.clear();
        self.sub.resize(n - 1, 0.0);
        self.sup.clear();
        self.sup.resize(n - 1, 0.0);
        self.upwind.clear();
        self.upwind.resize(n, 0);
        self.active.clear();
        self.active.resize(n, 0);
        self.active_new.clear();
        self.active_new.resize(n, 0);
    }
}

/// Result of one backward march. With `retain`, the whole grid is kept in
/// flat storage (level-major, stride `n_x + 1`) for the adjoint; the
/// buffers are reused when the struct is passed back to [`solve_into`].
#[derive(Debug, Clone)]
pub struct Solution {
    /// `u^0` read at `x0`.
    pub price: f64,
    /// Retained layers, `(n_steps + 1) * stride` entries, level-major:
    /// level `n` holds the PRE-jump layer `u^{n,+}` (the terminal layer at
    /// level `n_steps`). `None` when not retained.
    pub u: Option<Vec<f64>>,
    /// Active set `P_n` used at each step, `n_steps * stride` entries
    /// (step-major); all zero in European mode; empty when not retained.
    pub active: Vec<u8>,
    /// Post-jump layers `u^m` at each dividend level, `n_jumps * stride`,
    /// aligned with `Mesh::div_steps`; empty when not retained.
    pub jump_post: Vec<f64>,
    /// Masks `M_m` at each dividend level (`1` = keep the interpolated
    /// value, `0` = exercised into `psi`), `n_jumps * stride`; all one in
    /// European mode; empty when not retained.
    pub jump_masks: Vec<u8>,
    /// Backward steps of the mesh the solution was computed on.
    pub n_steps: usize,
    /// Nodes per layer.
    pub stride: usize,
    /// Number of dividend jumps.
    pub n_jumps: usize,
    /// Exercise mode of the solve.
    pub mode: Mode,
    /// Total tridiagonal solves in the penalty iterations (one per step in
    /// European mode).
    pub penalty_iterations: usize,
    /// Largest penalty-iteration count of any step.
    pub max_penalty_iterations_in_a_step: usize,
    /// Penalty iterations (tridiagonal solves) of each step, `n_steps`
    /// entries, step-major.
    pub iterations_per_step: Vec<u8>,
    /// Steps whose penalty iteration hit the cap with `P(u) != P_used`.
    pub penalty_inconsistent_steps: usize,
    /// Rows (over all steps) that used the upwinded drift stencil.
    pub upwinded_rows: usize,
    /// Largest `rho dt_n` on the mesh (0 in European mode).
    pub rho_dt_max: f64,
    /// Relative deviation of the discrete boundary dollar gamma from
    /// `2 (r K - q b) / sigma^2` at the middle time level (retained
    /// American solves with a free boundary inside the grid; see
    /// [`boundary_dollar_gamma`]).
    pub boundary_gamma_check: Option<f64>,
    /// Allocation kept when the last solve did not retain.
    spare: Vec<f64>,
}

impl Default for Solution {
    fn default() -> Self {
        Solution::new()
    }
}

impl Solution {
    /// An empty solution (buffers allocate on first use).
    pub fn new() -> Self {
        Solution {
            price: f64::NAN,
            u: None,
            active: Vec::new(),
            jump_post: Vec::new(),
            jump_masks: Vec::new(),
            n_steps: 0,
            stride: 0,
            n_jumps: 0,
            mode: Mode::European,
            penalty_iterations: 0,
            max_penalty_iterations_in_a_step: 0,
            iterations_per_step: Vec::new(),
            penalty_inconsistent_steps: 0,
            upwinded_rows: 0,
            rho_dt_max: 0.0,
            boundary_gamma_check: None,
            spare: Vec::new(),
        }
    }

    /// Whether the grid was retained.
    #[inline]
    pub fn retained(&self) -> bool {
        self.u.is_some()
    }

    /// Pre-jump layer at `level` (panics if not retained).
    #[inline]
    pub fn layer(&self, level: usize) -> &[f64] {
        let u = self.u.as_ref().expect("solution was not retained");
        &u[level * self.stride..(level + 1) * self.stride]
    }

    /// The layer at `t = 0`.
    #[inline]
    pub fn final_layer(&self) -> &[f64] {
        self.layer(0)
    }

    /// Active set used at `step` (all zero in European mode).
    #[inline]
    pub fn active_row(&self, step: usize) -> &[u8] {
        &self.active[step * self.stride..(step + 1) * self.stride]
    }

    /// Post-jump layer of dividend `k` (index into `Mesh::div_steps`).
    #[inline]
    pub fn jump_post_layer(&self, k: usize) -> &[f64] {
        &self.jump_post[k * self.stride..(k + 1) * self.stride]
    }

    /// Mask of dividend `k`.
    #[inline]
    pub fn jump_mask(&self, k: usize) -> &[u8] {
        &self.jump_masks[k * self.stride..(k + 1) * self.stride]
    }

    /// The layer `u^{n+1}` that backward step `step` started from: the
    /// post-jump layer when level `step + 1` is an ex-date, the plain
    /// layer otherwise.
    #[inline]
    pub fn layer_into_step(&self, mesh: &Mesh, step: usize) -> &[f64] {
        match mesh.jump_at(step + 1) {
            Some(k) => self.jump_post_layer(k),
            None => self.layer(step + 1),
        }
    }

    /// Number of active nodes at `step`.
    pub fn active_count(&self, step: usize) -> usize {
        self.active_row(step).iter().filter(|&&p| p == 1).count()
    }

    /// Check that the retained grid matches `mesh` (for the adjoint and the
    /// frozen re-solve).
    pub fn check_against(&self, mesh: &Mesh) -> Result<(), RustyQLibError> {
        if !self.retained() {
            return Err(RustyQLibError::invalid_input(
                "solution",
                "the solution was not retained (solve with retain = true)",
            ));
        }
        if self.n_steps != mesh.n_steps()
            || self.stride != mesh.n_nodes()
            || self.n_jumps != mesh.div_steps.len()
        {
            return Err(RustyQLibError::invalid_input(
                "solution",
                format!(
                    "solution ({} steps x {} nodes, {} jumps) does not match the mesh ({} x {}, {})",
                    self.n_steps,
                    self.stride,
                    self.n_jumps,
                    mesh.n_steps(),
                    mesh.n_nodes(),
                    mesh.div_steps.len()
                ),
            ));
        }
        Ok(())
    }

    fn reset(&mut self, mesh: &Mesh, mode: Mode, retain: bool) {
        let n = mesh.n_nodes();
        let steps = mesh.n_steps();
        self.n_steps = steps;
        self.stride = n;
        self.n_jumps = mesh.div_steps.len();
        self.mode = mode;
        self.price = f64::NAN;
        self.penalty_iterations = 0;
        self.max_penalty_iterations_in_a_step = 0;
        self.iterations_per_step.clear();
        self.iterations_per_step.resize(steps, 0);
        self.penalty_inconsistent_steps = 0;
        self.upwinded_rows = 0;
        self.rho_dt_max = 0.0;
        self.boundary_gamma_check = None;
        if retain {
            let mut u = self
                .u
                .take()
                .unwrap_or_else(|| std::mem::take(&mut self.spare));
            u.clear();
            u.resize((steps + 1) * n, 0.0);
            self.u = Some(u);
            self.active.clear();
            self.active.resize(steps * n, 0);
            self.jump_post.clear();
            self.jump_post.resize(self.n_jumps * n, 0.0);
            self.jump_masks.clear();
            self.jump_masks.resize(self.n_jumps * n, 1);
        } else {
            if let Some(mut u) = self.u.take() {
                u.clear();
                self.spare = u;
            }
            self.active.clear();
            self.jump_post.clear();
            self.jump_masks.clear();
        }
    }
}

// ── The march ────────────────────────────────────────────────────────────

/// `dia = dia0 + rho dt P`, `rhs_pen = rhs + rho dt P psi`.
#[inline]
fn penalize(
    dia0: &[f64],
    rhs: &[f64],
    psi: &[f64],
    active: &[u8],
    rdt: f64,
    dia: &mut [f64],
    rhs_pen: &mut [f64],
) {
    for j in 0..dia0.len() {
        if active[j] == 1 {
            dia[j] = dia0[j] + rdt;
            rhs_pen[j] = rhs[j] + rdt * psi[j];
        } else {
            dia[j] = dia0[j];
            rhs_pen[j] = rhs[j];
        }
    }
}

/// The backward march shared by every entry point. `frozen` supplies the
/// active sets and jump masks to hold fixed (no penalty iteration).
#[allow(clippy::too_many_arguments)]
fn march<V: StepVol + ?Sized>(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &V,
    market: &MarketSlice,
    mode: Mode,
    retain: bool,
    frozen: Option<&Solution>,
    ws: &mut Workspace,
    out: &mut Solution,
) -> Result<(), RustyQLibError> {
    spec.validate(mesh)?;
    market.validate(mesh)?;
    vol.check(mesh)?;
    mode.validate()?;
    if let Some(fz) = frozen {
        fz.check_against(mesh)?;
        if fz.mode.is_american() != mode.is_american() {
            return Err(RustyQLibError::invalid_input(
                "frozen",
                "the frozen solution's exercise mode differs from the requested mode",
            ));
        }
    }
    let n = mesh.n_nodes();
    let n_x = n - 1;
    let steps = mesh.n_steps();
    let dx = mesh.dx;
    ws.ensure(n);
    out.reset(mesh, mode, retain);
    let rho = mode.rho();
    let american = mode.is_american();

    // terminal layer (cell-averaged) and obstacle (plain payoff). The first
    // step's penalty iteration is seeded with a SUBSET of the exercise set:
    // {psi > 0, L psi < 0} = {q S < r K} (put) / {q S > r K} (call), minus
    // a margin of SEED_MARGIN_SIGMAS sigma sqrt(dt) + 2 dx around the
    // strike (near expiry the boundary sits ~3 sigma sqrt(tau) from the
    // strike). The iteration grows an under-estimated set in one shot but
    // releases an over-estimated one a node at a time.
    let (r_last, q_last) = (market.rate[steps - 1], market.carry[steps - 1]);
    let dt_last = mesh.dt[steps - 1];
    let ln_k = spec.strike.ln();
    vol.fill_row(steps - 1, &mut ws.sig);
    for j in 0..n {
        let s = mesh.x[j].exp();
        ws.psi[j] = spec.payoff(s);
        ws.u[j] = cell_average_payoff(spec, mesh.x[j], dx);
        let decays = match spec.right {
            PutOrCall::Put => q_last * s < r_last * spec.strike,
            PutOrCall::Call => q_last * s > r_last * spec.strike,
        };
        let margin = SEED_MARGIN_SIGMAS * ws.sig[j] * dt_last.sqrt() + 2.0 * dx;
        let far = (mesh.x[j] - ln_k).abs() > margin;
        ws.active[j] = u8::from(american && ws.psi[j] > 0.0 && decays && far);
    }
    if retain {
        let grid = out.u.as_mut().expect("retained buffer");
        grid[steps * n..(steps + 1) * n].copy_from_slice(&ws.u);
    }

    // ws.u: the layer entering the step (u^{n+1}); ws.v receives u^{n,+}
    for step in (0..steps).rev() {
        let r = market.rate[step];
        let q = market.carry[step];
        let dt = mesh.dt[step];
        let theta = mesh.theta[step];
        vol.fill_row(step, &mut ws.sig);
        out.upwinded_rows += assemble_operator(
            &ws.sig,
            r,
            q,
            dx,
            &mut ws.lower,
            &mut ws.diag,
            &mut ws.upper,
            &mut ws.upwind,
        );
        // rhs = B_n u^{n+1}
        apply_operator(
            &ws.u,
            &ws.lower,
            &ws.diag,
            &ws.upper,
            (1.0 - theta) * dt,
            &mut ws.rhs,
        );
        // bands of A_n = I - theta dt L_n (row-indexed)
        let tdt = theta * dt;
        for j in 0..n {
            ws.dia0[j] = 1.0 - tdt * ws.diag[j];
        }
        for j in 1..n {
            ws.sub[j - 1] = -tdt * ws.lower[j];
        }
        for j in 0..n_x {
            ws.sup[j] = -tdt * ws.upper[j];
        }

        if !american {
            thomas_inplace(
                &ws.sub, &ws.dia0, &ws.sup, &ws.rhs, &mut ws.cw, &mut ws.dw, &mut ws.v,
            );
            out.penalty_iterations += 1;
            out.max_penalty_iterations_in_a_step = out.max_penalty_iterations_in_a_step.max(1);
            out.iterations_per_step[step] = 1;
        } else {
            let rdt = rho * dt;
            if rdt > out.rho_dt_max {
                out.rho_dt_max = rdt;
            }
            if let Some(fz) = frozen {
                ws.active.copy_from_slice(fz.active_row(step));
                penalize(
                    &ws.dia0,
                    &ws.rhs,
                    &ws.psi,
                    &ws.active,
                    rdt,
                    &mut ws.dia,
                    &mut ws.rhs_pen,
                );
                thomas_inplace(
                    &ws.sub,
                    &ws.dia,
                    &ws.sup,
                    &ws.rhs_pen,
                    &mut ws.cw,
                    &mut ws.dw,
                    &mut ws.v,
                );
                out.penalty_iterations += 1;
                out.max_penalty_iterations_in_a_step = out.max_penalty_iterations_in_a_step.max(1);
                out.iterations_per_step[step] = 1;
            } else {
                // penalty iteration warm-started from the previous step's set
                let mut iters = 0usize;
                let consistent = loop {
                    penalize(
                        &ws.dia0,
                        &ws.rhs,
                        &ws.psi,
                        &ws.active,
                        rdt,
                        &mut ws.dia,
                        &mut ws.rhs_pen,
                    );
                    thomas_inplace(
                        &ws.sub,
                        &ws.dia,
                        &ws.sup,
                        &ws.rhs_pen,
                        &mut ws.cw,
                        &mut ws.dw,
                        &mut ws.v,
                    );
                    iters += 1;
                    let mut changed = false;
                    for j in 0..n {
                        let p = u8::from(ws.v[j] < ws.psi[j]);
                        ws.active_new[j] = p;
                        changed |= p != ws.active[j];
                    }
                    if !changed {
                        break true;
                    }
                    if iters >= MAX_PENALTY_ITERATIONS {
                        break false;
                    }
                    std::mem::swap(&mut ws.active, &mut ws.active_new);
                };
                out.penalty_iterations += iters;
                out.max_penalty_iterations_in_a_step =
                    out.max_penalty_iterations_in_a_step.max(iters);
                out.iterations_per_step[step] = iters.min(u8::MAX as usize) as u8;
                if !consistent {
                    // P(u_final) != P_used: the frozen-set derivative is not
                    // exact at this step; report rather than hide it
                    out.penalty_inconsistent_steps += 1;
                    log::warn!(
                        "penalty iteration hit the cap of {MAX_PENALTY_ITERATIONS} at step {step} \
                         (K = {}, T = {}) with an inconsistent active set",
                        spec.strike,
                        spec.t_expiry
                    );
                }
            }
        }

        if retain {
            let grid = out.u.as_mut().expect("retained buffer");
            grid[step * n..(step + 1) * n].copy_from_slice(&ws.v);
            out.active[step * n..(step + 1) * n].copy_from_slice(&ws.active);
        }

        // dividend jump landing on level `step`
        if let Some(k) = mesh.jump_at(step) {
            let stencil = &mesh.div_stencils[k];
            stencil.apply(&ws.v, &mut ws.tmp);
            if american {
                if let Some(fz) = frozen {
                    let mask = fz.jump_mask(k);
                    for j in 0..n {
                        if mask[j] == 0 {
                            ws.tmp[j] = ws.psi[j];
                        }
                    }
                    if retain {
                        out.jump_masks[k * n..(k + 1) * n].copy_from_slice(mask);
                    }
                } else {
                    for j in 0..n {
                        let keep = ws.tmp[j] > ws.psi[j];
                        if !keep {
                            ws.tmp[j] = ws.psi[j];
                        }
                        if retain {
                            out.jump_masks[k * n + j] = u8::from(keep);
                        }
                    }
                }
            }
            if retain {
                out.jump_post[k * n..(k + 1) * n].copy_from_slice(&ws.tmp);
            }
            std::mem::swap(&mut ws.v, &mut ws.tmp);
        }
        std::mem::swap(&mut ws.u, &mut ws.v);
    }

    out.price = mesh.read_x0(&ws.u);
    if !out.price.is_finite() {
        return Err(RustyQLibError::NumericalError(format!(
            "backward solve produced a non-finite price (K = {}, T = {})",
            spec.strike, spec.t_expiry
        )));
    }
    if retain && american && steps >= 2 {
        let level = steps / 2;
        let step = level.min(steps - 1);
        if let Some(bg) =
            boundary_gamma_generic(mesh, spec, out, market, level, |j| vol.sigma(step, j))
        {
            out.boundary_gamma_check = Some(bg.relative_deviation());
        }
    }
    log::debug!(
        "solve K = {} T = {} {:?}: price {:.8}, {} solves (max {} per step, {} inconsistent), \
         {} upwinded rows",
        spec.strike,
        spec.t_expiry,
        mode,
        out.price,
        out.penalty_iterations,
        out.max_penalty_iterations_in_a_step,
        out.penalty_inconsistent_steps,
        out.upwinded_rows
    );
    Ok(())
}

// ── Entry points ─────────────────────────────────────────────────────────

/// Backward solve on a precomputed [`NodeField`] (the hot path). With
/// `retain` the whole grid, the active sets and the jump layers/masks are
/// kept for the adjoint.
pub fn solve_backward(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &NodeField,
    market: &MarketSlice,
    mode: Mode,
    retain: bool,
    ws: &mut Workspace,
) -> Result<Solution, RustyQLibError> {
    let mut out = Solution::new();
    march(mesh, spec, vol, market, mode, retain, None, ws, &mut out)?;
    Ok(out)
}

/// [`solve_backward`] writing into an existing [`Solution`], whose buffers
/// are reused (no allocation when the mesh size is unchanged).
#[allow(clippy::too_many_arguments)]
pub fn solve_into(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &NodeField,
    market: &MarketSlice,
    mode: Mode,
    retain: bool,
    ws: &mut Workspace,
    out: &mut Solution,
) -> Result<(), RustyQLibError> {
    march(mesh, spec, vol, market, mode, retain, None, ws, out)
}

/// Backward solve reading `sigma(x_j, t_mid[n])` from any [`VolField`].
pub fn solve_backward_dyn(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &dyn VolField,
    market: &MarketSlice,
    mode: Mode,
    retain: bool,
    ws: &mut Workspace,
) -> Result<Solution, RustyQLibError> {
    let dv = DynVol { field: vol, mesh };
    let mut out = Solution::new();
    march(mesh, spec, &dv, market, mode, retain, None, ws, &mut out)?;
    Ok(out)
}

/// [`solve_backward_dyn`] writing into an existing [`Solution`].
#[allow(clippy::too_many_arguments)]
pub fn solve_into_dyn(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &dyn VolField,
    market: &MarketSlice,
    mode: Mode,
    retain: bool,
    ws: &mut Workspace,
    out: &mut Solution,
) -> Result<(), RustyQLibError> {
    let dv = DynVol { field: vol, mesh };
    march(mesh, spec, &dv, market, mode, retain, None, ws, out)
}

/// Price without retention (no grid storage; the workspace is the only
/// memory touched).
pub fn price_only(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &NodeField,
    market: &MarketSlice,
    mode: Mode,
    ws: &mut Workspace,
) -> Result<f64, RustyQLibError> {
    let mut out = Solution::new();
    march(mesh, spec, vol, market, mode, false, None, ws, &mut out)?;
    Ok(out.price)
}

/// [`price_only`] on any [`VolField`].
pub fn price_only_dyn(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &dyn VolField,
    market: &MarketSlice,
    mode: Mode,
    ws: &mut Workspace,
) -> Result<f64, RustyQLibError> {
    let dv = DynVol { field: vol, mesh };
    let mut out = Solution::new();
    march(mesh, spec, &dv, market, mode, false, None, ws, &mut out)?;
    Ok(out.price)
}

/// Price with the active sets `P_n` and jump masks `M_n` of `frozen` held
/// fixed (one tridiagonal solve per step, no penalty iteration): the map
/// whose exact derivative the adjoint computes. `frozen` must be a
/// retained solution on the same mesh; its mode (and `rho`) is reused.
pub fn price_frozen(
    mesh: &Mesh,
    spec: &QuoteSpec,
    vol: &NodeField,
    market: &MarketSlice,
    frozen: &Solution,
    ws: &mut Workspace,
) -> Result<f64, RustyQLibError> {
    let mut out = Solution::new();
    march(
        mesh,
        spec,
        vol,
        market,
        frozen.mode,
        false,
        Some(frozen),
        ws,
        &mut out,
    )?;
    Ok(out.price)
}

// ── Boundary dollar gamma ────────────────────────────────────────────────

/// The discrete free boundary at one time level and the dollar gamma on
/// its continuation side, against the smooth-fit identity
/// `S^2 u_SS(b+, t) = 2 (r K - q b) / sigma^2` (put; `2 (q b - r K) /
/// sigma^2` for a call).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryGamma {
    /// Exercise boundary `b` (spot), from the zero crossing of `u - psi`.
    pub boundary: f64,
    /// Discrete `S^2 u_SS` at `b`: pair-averaged (checkerboard-free)
    /// stencils on the continuation side, least-squares line evaluated at
    /// the boundary.
    pub discrete: f64,
    /// `2 (r K - q b) / sigma^2` (put) or `2 (q b - r K) / sigma^2` (call).
    pub analytic: f64,
}

impl BoundaryGamma {
    /// `|discrete - analytic| / |analytic|`.
    pub fn relative_deviation(&self) -> f64 {
        (self.discrete - self.analytic).abs() / self.analytic.abs().max(1e-300)
    }
}

fn boundary_gamma_generic<F: Fn(usize) -> f64>(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    level: usize,
    sigma_at: F,
) -> Option<BoundaryGamma> {
    if !sol.retained() || !sol.mode.is_american() || level > sol.n_steps {
        return None;
    }
    let n = sol.stride;
    let n_x = n - 1;
    let step = level.min(sol.n_steps - 1);
    let active = sol.active_row(step);
    let u = sol.layer(level);
    let dx = mesh.dx;
    let r = market.rate[step];
    let q = market.carry[step];
    let k = spec.strike;
    let psi = |j: usize| spec.payoff(mesh.x[j].exp());
    // j_a: last active node (put) / first active node (call) inside the
    // payoff-positive region (a far-OTM node can be "active" at u = -1e-6
    // < psi = 0 through the linear boundary row; that is not the free
    // boundary); the boundary lies between j_a and its continuation-side
    // neighbour
    let k_pos = mesh.x.partition_point(|&x| x.exp() < k);
    let (j_a, dir): (usize, isize) = match spec.right {
        PutOrCall::Put => {
            let j = active[..k_pos.min(n)].iter().rposition(|&p| p == 1)?;
            (j, 1)
        }
        PutOrCall::Call => {
            let j = k_pos + active[k_pos.min(n)..].iter().position(|&p| p == 1)?;
            (j, -1)
        }
    };
    let at = |k: isize| -> Option<usize> {
        let idx = j_a as isize + dir * k;
        if idx >= 1 && (idx as usize) < n_x {
            Some(idx as usize)
        } else {
            None
        }
    };
    let j1 = at(1)?;
    // nine clean continuation-side stencils (none straddles the boundary)
    let fit: Vec<usize> = (2..11).map(at).collect::<Option<Vec<_>>>()?;
    if active[j1] == 1 || fit.iter().any(|&j| active[j] == 1) {
        return None; // not a single connected boundary
    }
    let f_a = u[j_a] - psi(j_a);
    let f_b = u[j1] - psi(j1);
    if f_b.partial_cmp(&f_a) != Some(std::cmp::Ordering::Greater) {
        return None;
    }
    let frac = (-f_a / (f_b - f_a)).clamp(0.0, 1.0);
    let x_b = mesh.x[j_a] + dir as f64 * frac * dx;
    let b = x_b.exp();
    // the free boundary excites a Crank-Nicolson checkerboard in the
    // discrete dollar gamma: average adjacent pairs (which removes the
    // alternating mode exactly), then fit a least-squares line through the
    // eight pair means and evaluate it at x_b
    let mut s_pts = [0.0f64; 8];
    let mut g_pts = [0.0f64; 8];
    for i in 0..8 {
        let (ja, jb) = (fit[i], fit[i + 1]);
        s_pts[i] = 0.5 * (mesh.x[ja] + mesh.x[jb]) - x_b;
        g_pts[i] = 0.5 * (d2_at(u, ja, dx, 0, 0.0) + d2_at(u, jb, dx, 0, 0.0));
    }
    let s_mean = s_pts.iter().sum::<f64>() / 8.0;
    let g_mean = g_pts.iter().sum::<f64>() / 8.0;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for i in 0..8 {
        let ds = s_pts[i] - s_mean;
        sxy += ds * (g_pts[i] - g_mean);
        sxx += ds * ds;
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let discrete = g_mean - slope * s_mean;
    let sigma = sigma_at(j1);
    let analytic = match spec.right {
        PutOrCall::Put => 2.0 * (r * k - q * b) / (sigma * sigma),
        PutOrCall::Call => 2.0 * (q * b - r * k) / (sigma * sigma),
    };
    Some(BoundaryGamma {
        boundary: b,
        discrete,
        analytic,
    })
}

/// Boundary dollar gamma of a retained American solution at time level
/// `level`, with `sigma` the local volatility at the boundary (the flat
/// value for a flat solve). `None` when the level has no free boundary
/// inside the grid (no active node, or the boundary within three nodes of
/// an edge).
pub fn boundary_dollar_gamma(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    level: usize,
    sigma: f64,
) -> Option<BoundaryGamma> {
    boundary_gamma_generic(mesh, spec, sol, market, level, |_| sigma)
}

/// Discrete dollar gamma `S^2 u_SS = delta_xx u - delta_x u` of a layer
/// read at `x0` with the mesh's interpolation weights (central stencils).
pub fn dollar_gamma_at_x0(mesh: &Mesh, layer: &[f64]) -> f64 {
    let (i0, w0, w1) = mesh.x0_weights;
    let n_x = mesh.n_x;
    let g = |j: usize| {
        let j = j.clamp(1, n_x - 1);
        d2_at(layer, j, mesh.dx, 0, 0.0)
    };
    if w1 == 0.0 {
        w0 * g(i0)
    } else {
        w0 * g(i0) + w1 * g(i0 + 1)
    }
}

#[cfg(test)]
#[path = "solver_tests.rs"]
mod solver_tests;
