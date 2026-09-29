//! The discrete adjoint of the backward march: the exact transpose of the
//! frozen-active-set map of [`super::solver`], the gradient field
//! `dF/dSigma_{j,n}` on the mesh, the drift (carry) sensitivities, and the
//! normalized American / European kernels at a flat volatility.
//!
//! ```text
//! Forward step n (level n+1 -> n), P_n and M_n frozen, L = L_n at t_mid[n]:
//!   A_n u^{n,+} = B_n u^{n+1} + c_n,   A_n = I - theta dt L + rho dt P_n,  B_n = I + (1-theta) dt L
//!   u^n = M_n D u^{n,+} + (I - M_n) psi at an ex-date level, u^n = u^{n,+} otherwise
//!   F = w . u^0                          (x0 interpolation weights)
//! Chain rule (dL = perturbation of the operator):
//!   delta u^{n,+} = A_n^{-1} [ B_n delta u^{n+1} + dt (dL)(theta u^{n,+} + (1-theta) u^{n+1}) ]
//!   delta u^n     = M_n D delta u^{n,+}
//! Adjoint recursion, n = 0, 1, ..., N-1:
//!   nu^0 = w
//!   at an ex-date level n:  nu^{n,+} = D^T (M_n nu^n)        (MASK FIRST, THEN D^T)
//!   lambda^n = A_n^{-T} nu^{n,+}   (Thomas with sub/super bands swapped, same frozen bands)
//!   nu^{n+1} = B_n^T lambda^n
//! Gradient field (row j of L depends on sigma_j only):
//!   g_{j,n} = dt_n lambda^n_j sigma_{j,n} [ theta (D2 u^{n,+})_j + (1-theta) (D2 u^{n+1})_j ]
//!   D2 = delta_xx - delta_x, the discrete S^2 u_SS the row uses (the same
//!   one-sided delta_x on upwinded rows); u^{n,+} is the PRE-jump layer.
//!   g = 0 on the two edge rows (the linear boundary rows carry no sigma).
//! Drift sensitivity (d/d(r_n - q_n) of L at fixed discounting):
//!   h_n = dt_n lambda^n . [ theta D1 u^{n,+} + (1-theta) D1 u^{n+1} ],  dF/dq_k = - sum_{n in I_k} h_n
//! Vega of a flat field:  dF/dsigma = sum_{j,n} g_{j,n}
//! ```
//!
//! Conventions: `g` is stored step-major with stride `n_x + 1`, exactly
//! the layout of [`NodeField::values`], so `g[n * stride + j]` is the
//! derivative with respect to `Sigma(x_j, t_mid[n])`. `g_{j,n} / (dx dt_n)`
//! is a density per unit `x` and `t`; for an `(S, t)` plot divide by `S`;
//! the normalized kernel `kappa = g / sum g` is invariant to these
//! conventions. The exercise mask and the active set are frozen, so the
//! field is the derivative of the frozen-active-set map (the Clarke
//! generalized Jacobian of the penalized price; Hintermüller, Ito & Kunisch
//! 2002). On upwinded rows the row's own one-sided stencil is
//! differentiated: `d/dsigma_j` of the upwinded row is
//! `sigma_j (delta_xx u - delta_x^{one-sided} u)`, not zero, because the
//! diffusion term and the `-sigma^2/2` part of the drift keep their
//! `sigma`-dependence; the field is therefore the exact derivative of the
//! frozen stencil there too, and only the Peclet switch itself is frozen.

// Stencil code indexes neighbours j-1, j+1 of the same row.
#![allow(clippy::needless_range_loop)]

use super::grid::{MarketSlice, Mesh};
use super::solver::{
    apply_operator_transpose, assemble_operator, d1_at, d2_at, dollar_gamma_at_x0, solve_into,
    thomas_inplace, DynVol, Mode, QuoteSpec, Solution, StepVol, Workspace,
};
use super::vol_field::{FlatVol, NodeField, VolField};
use crate::core::errors::RustyQLibError;

// ── Workspace ────────────────────────────────────────────────────────────

/// Scratch buffers of one adjoint march, reused across quotes and
/// Jacobians (thread-local through `map_init`). Grows to the mesh's node
/// count on first use and never shrinks.
#[derive(Debug, Clone, Default)]
pub struct AdjointWorkspace {
    n: usize,
    nu: Vec<f64>,
    nu_next: Vec<f64>,
    lambda: Vec<f64>,
    sig: Vec<f64>,
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
    upwind: Vec<u8>,
    sub: Vec<f64>,
    sup: Vec<f64>,
    dia: Vec<f64>,
    cw: Vec<f64>,
    dw: Vec<f64>,
    g_row: Vec<f64>,
    tmp: Vec<f64>,
}

impl AdjointWorkspace {
    /// An empty workspace (allocates on first use).
    pub fn new() -> Self {
        AdjointWorkspace::default()
    }

    fn ensure(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        for v in [
            &mut self.nu,
            &mut self.nu_next,
            &mut self.lambda,
            &mut self.sig,
            &mut self.lower,
            &mut self.diag,
            &mut self.upper,
            &mut self.dia,
            &mut self.cw,
            &mut self.dw,
            &mut self.g_row,
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
    }
}

// ── The recursion ────────────────────────────────────────────────────────

/// The adjoint march. `sink(step, g_row, h_step)` receives the gradient
/// row and the drift sensitivity of every step in increasing time order;
/// returns `sum g` (the flat vega).
#[allow(clippy::too_many_arguments)]
fn adjoint_core<V: StepVol + ?Sized, F: FnMut(usize, &[f64], f64)>(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &V,
    ws: &mut AdjointWorkspace,
    mut sink: F,
) -> Result<f64, RustyQLibError> {
    spec.validate(mesh)?;
    market.validate(mesh)?;
    vol.check(mesh)?;
    sol.check_against(mesh)?;
    let n = mesh.n_nodes();
    let n_x = n - 1;
    let steps = mesh.n_steps();
    let dx = mesh.dx;
    ws.ensure(n);
    let rho = sol.mode.rho();

    // seed: the x0 interpolation weights
    for v in ws.nu.iter_mut() {
        *v = 0.0;
    }
    let (i0, w0, w1) = mesh.x0_weights;
    ws.nu[i0] = w0;
    if w1 != 0.0 {
        ws.nu[i0 + 1] = w1;
    }

    let mut vega = 0.0;
    for step in 0..steps {
        // ex-date level: mask first, then D^T
        if let Some(k) = mesh.jump_at(step) {
            let mask = sol.jump_mask(k);
            for j in 0..n {
                ws.tmp[j] = if mask[j] == 1 { ws.nu[j] } else { 0.0 };
            }
            mesh.div_stencils[k].apply_transpose(&ws.tmp, &mut ws.nu);
        }
        let r = market.rate[step];
        let q = market.carry[step];
        let dt = mesh.dt[step];
        let theta = mesh.theta[step];
        vol.fill_row(step, &mut ws.sig);
        assemble_operator(
            &ws.sig,
            r,
            q,
            dx,
            &mut ws.lower,
            &mut ws.diag,
            &mut ws.upper,
            &mut ws.upwind,
        );
        // frozen bands of A_n (row-indexed), including rho dt P_n
        let tdt = theta * dt;
        let rdt = rho * dt;
        let active = sol.active_row(step);
        for j in 0..n {
            ws.dia[j] = 1.0 - tdt * ws.diag[j] + if active[j] == 1 { rdt } else { 0.0 };
        }
        for j in 1..n {
            ws.sub[j - 1] = -tdt * ws.lower[j];
        }
        for j in 0..n_x {
            ws.sup[j] = -tdt * ws.upper[j];
        }
        // lambda = A^{-T} nu: swap the sub and super bands
        thomas_inplace(
            &ws.sup,
            &ws.dia,
            &ws.sub,
            &ws.nu,
            &mut ws.cw,
            &mut ws.dw,
            &mut ws.lambda,
        );
        // gradient row and drift sensitivity from the PRE-jump layer at
        // level `step` and the layer the step started from
        let u_plus = sol.layer(step);
        let u_next = sol.layer_into_step(mesh, step);
        let omt = 1.0 - theta;
        ws.g_row[0] = 0.0;
        ws.g_row[n_x] = 0.0;
        let mut row_sum = 0.0;
        for j in 1..n_x {
            let s = ws.sig[j];
            let mu = r - q - 0.5 * s * s;
            let uw = ws.upwind[j];
            let d2 = theta * d2_at(u_plus, j, dx, uw, mu) + omt * d2_at(u_next, j, dx, uw, mu);
            let g = dt * ws.lambda[j] * s * d2;
            ws.g_row[j] = g;
            row_sum += g;
        }
        vega += row_sum;
        let mut h = 0.0;
        for j in 0..n {
            let s = ws.sig[j];
            let mu = r - q - 0.5 * s * s;
            let uw = ws.upwind[j];
            h += ws.lambda[j]
                * (theta * d1_at(u_plus, j, dx, uw, mu) + omt * d1_at(u_next, j, dx, uw, mu));
        }
        h *= dt;
        sink(step, &ws.g_row, h);
        // nu^{n+1} = B_n^T lambda^n
        apply_operator_transpose(
            &ws.lambda,
            &ws.lower,
            &ws.diag,
            &ws.upper,
            omt * dt,
            &mut ws.nu_next,
        );
        std::mem::swap(&mut ws.nu, &mut ws.nu_next);
    }
    if !vega.is_finite() {
        return Err(RustyQLibError::NumericalError(format!(
            "adjoint produced a non-finite gradient (K = {}, T = {})",
            spec.strike, spec.t_expiry
        )));
    }
    Ok(vega)
}

// ── Stored result ────────────────────────────────────────────────────────

/// The gradient field of one quote, its drift sensitivities and the flat
/// vega. Buffers are reused by [`adjoint_into`].
#[derive(Debug, Clone, Default)]
pub struct Adjoint {
    /// `dF/dSigma_{j,n}`, `n_steps * stride` entries, step-major
    /// (`g[n * stride + j]`), the layout of [`NodeField::values`].
    pub g: Vec<f64>,
    /// `h_n = dF/d(r_n - q_n)` at fixed discounting, `n_steps` entries;
    /// `dF/dq_n = -h_n`.
    pub h: Vec<f64>,
    /// `sum g`: the derivative of the price with respect to a parallel
    /// shift of the whole field (the flat vega when the field is flat).
    pub vega_from_field: f64,
    /// Backward steps.
    pub n_steps: usize,
    /// Nodes per level.
    pub stride: usize,
}

impl Adjoint {
    /// An empty result (allocates on first use).
    pub fn new() -> Self {
        Adjoint::default()
    }

    fn reset(&mut self, mesh: &Mesh) {
        self.n_steps = mesh.n_steps();
        self.stride = mesh.n_nodes();
        self.g.clear();
        self.g.resize(self.n_steps * self.stride, 0.0);
        self.h.clear();
        self.h.resize(self.n_steps, 0.0);
        self.vega_from_field = 0.0;
    }

    /// Gradient row of `step`.
    #[inline]
    pub fn g_row(&self, step: usize) -> &[f64] {
        &self.g[step * self.stride..(step + 1) * self.stride]
    }

    /// Directional derivative `sum g eta` along a field perturbation in
    /// the same layout (e.g. `eta = NodeField::values` of a bump).
    pub fn directional(&self, eta: &[f64]) -> f64 {
        directional_derivative(&self.g, eta)
    }

    /// `dF/dq_k` for piecewise-constant carry on the half-open intervals
    /// `[a_k, b_k)` (see [`carry_sensitivities`]).
    pub fn carry_sensitivities(&self, mesh: &Mesh, intervals: &[(f64, f64)]) -> Vec<f64> {
        carry_sensitivities(mesh, &self.h, intervals)
    }
}

/// `sum g eta` over a field in the layout of [`Adjoint::g`].
pub fn directional_derivative(g: &[f64], eta: &[f64]) -> f64 {
    assert_eq!(
        g.len(),
        eta.len(),
        "gradient field and perturbation differ in size"
    );
    g.iter().zip(eta).map(|(a, b)| a * b).sum()
}

/// `dF/dq_k = - sum_{n : a_k <= t_mid[n] < b_k} h_n` for the carry
/// intervals `[a_k, b_k)` (a step belongs to the interval containing its
/// mid-time; steps outside every interval contribute to none).
pub fn carry_sensitivities(mesh: &Mesh, h: &[f64], intervals: &[(f64, f64)]) -> Vec<f64> {
    let mut out = vec![0.0; intervals.len()];
    carry_sensitivities_into(mesh, h, intervals, &mut out);
    out
}

/// [`carry_sensitivities`] written into `out` (one entry per interval; no
/// allocation, for Jacobian rows streamed per quote).
pub fn carry_sensitivities_into(mesh: &Mesh, h: &[f64], intervals: &[(f64, f64)], out: &mut [f64]) {
    assert_eq!(
        out.len(),
        intervals.len(),
        "carry sensitivities: output length differs from the interval count"
    );
    for (o, &(a, b)) in out.iter_mut().zip(intervals) {
        *o = -mesh
            .t_mid
            .iter()
            .zip(h)
            .filter(|(&tm, _)| tm >= a && tm < b)
            .map(|(_, &hn)| hn)
            .sum::<f64>();
    }
}

// ── Entry points ─────────────────────────────────────────────────────────

/// Gradient field, drift sensitivities and vega of a retained solution on
/// a [`NodeField`] (the hot path).
pub fn adjoint(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &NodeField,
    ws: &mut AdjointWorkspace,
) -> Result<Adjoint, RustyQLibError> {
    let mut out = Adjoint::new();
    adjoint_into(mesh, spec, sol, market, vol, ws, &mut out)?;
    Ok(out)
}

/// [`adjoint`] writing into an existing [`Adjoint`] (no allocation when
/// the mesh size is unchanged).
pub fn adjoint_into(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &NodeField,
    ws: &mut AdjointWorkspace,
    out: &mut Adjoint,
) -> Result<(), RustyQLibError> {
    out.reset(mesh);
    let stride = out.stride;
    let (g, h) = (&mut out.g, &mut out.h);
    let vega = adjoint_core(mesh, spec, sol, market, vol, ws, |step, row, h_n| {
        g[step * stride..(step + 1) * stride].copy_from_slice(row);
        h[step] = h_n;
    })?;
    out.vega_from_field = vega;
    Ok(())
}

/// [`adjoint`] reading `sigma(x_j, t_mid[n])` from any [`VolField`].
pub fn adjoint_dyn(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &dyn VolField,
    ws: &mut AdjointWorkspace,
) -> Result<Adjoint, RustyQLibError> {
    let mut out = Adjoint::new();
    adjoint_into_dyn(mesh, spec, sol, market, vol, ws, &mut out)?;
    Ok(out)
}

/// [`adjoint_dyn`] writing into an existing [`Adjoint`].
pub fn adjoint_into_dyn(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &dyn VolField,
    ws: &mut AdjointWorkspace,
    out: &mut Adjoint,
) -> Result<(), RustyQLibError> {
    let dv = DynVol { field: vol, mesh };
    out.reset(mesh);
    let stride = out.stride;
    let (g, h) = (&mut out.g, &mut out.h);
    let vega = adjoint_core(mesh, spec, sol, market, &dv, ws, |step, row, h_n| {
        g[step * stride..(step + 1) * stride].copy_from_slice(row);
        h[step] = h_n;
    })?;
    out.vega_from_field = vega;
    Ok(())
}

/// Streaming adjoint: `sink(step, g_row, h_step)` is called once per
/// backward step in increasing time order with the gradient row (valid
/// only during the call) and the drift sensitivity; nothing is allocated
/// per call beyond the workspace. Returns `sum g`.
pub fn adjoint_stream<F: FnMut(usize, &[f64], f64)>(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &NodeField,
    ws: &mut AdjointWorkspace,
    sink: F,
) -> Result<f64, RustyQLibError> {
    adjoint_core(mesh, spec, sol, market, vol, ws, sink)
}

/// `sum g` alone (the flat vega of a retained solution) without storing
/// the field.
pub fn vega_from_adjoint(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sol: &Solution,
    market: &MarketSlice,
    vol: &NodeField,
    ws: &mut AdjointWorkspace,
) -> Result<f64, RustyQLibError> {
    adjoint_core(mesh, spec, sol, market, vol, ws, |_, _, _| {})
}

// ── Kernels at a flat volatility ─────────────────────────────────────────

/// The two local-vega kernels of one quote at a flat volatility: the
/// normalized gradient fields of the American and the European price,
/// their vegas, time marginals and total-variation distance.
#[derive(Debug, Clone, PartialEq)]
pub struct Kernels {
    /// American kernel `kappa^A = g^A / sum g^A`, step-major, sums to 1.
    pub kappa_a: Vec<f64>,
    /// European kernel `kappa^E = g^E / sum g^E`, step-major, sums to 1.
    pub kappa_e: Vec<f64>,
    /// `nu^A = sum g^A`.
    pub vega_a: f64,
    /// `nu^E = sum g^E`.
    pub vega_e: f64,
    /// American price on the mesh.
    pub price_a: f64,
    /// European price on the mesh.
    pub price_e: f64,
    /// Time marginal of the American kernel, `w^A(t_mid[n]) = sum_j
    /// kappa^A_{j,n} / dt_n` (a density in `t`: `sum_n w^A_n dt_n = 1`).
    pub w_a: Vec<f64>,
    /// Time marginal of the European kernel (`1/T` in the continuum).
    pub w_e: Vec<f64>,
    /// `d_TV = 1/2 sum |kappa^A - kappa^E|`, the identification capacity.
    pub tv_distance: f64,
    /// Discrete dollar gamma `S0^2 Gamma^A(S0, 0)` of the American price.
    pub dollar_gamma_a0: f64,
    /// Discrete dollar gamma `S0^2 Gamma^E(S0, 0)` of the European price.
    pub dollar_gamma_e0: f64,
    /// Active sets of the American solve, step-major (`1` = exercised).
    pub active_a: Vec<u8>,
    /// Backward steps.
    pub n_steps: usize,
    /// Nodes per level.
    pub stride: usize,
}

impl Kernels {
    /// Row `step` of a field in this layout.
    #[inline]
    pub fn row<'a>(&self, field: &'a [f64], step: usize) -> &'a [f64] {
        &field[step * self.stride..(step + 1) * self.stride]
    }
}

/// `kappa / (dx dt_n)`: a normalized field as a density per unit `x` and
/// `t` on the mesh.
pub fn field_density(mesh: &Mesh, kappa: &[f64]) -> Vec<f64> {
    let stride = mesh.n_nodes();
    let mut out = Vec::with_capacity(kappa.len());
    for (n, row) in kappa.chunks(stride).enumerate() {
        let scale = 1.0 / (mesh.dx * mesh.dt[n]);
        out.extend(row.iter().map(|v| v * scale));
    }
    out
}

/// Time marginal of a normalized field: `w_n = sum_j kappa_{j,n} / dt_n`.
pub fn time_marginal(mesh: &Mesh, kappa: &[f64]) -> Vec<f64> {
    let stride = mesh.n_nodes();
    kappa
        .chunks(stride)
        .enumerate()
        .map(|(n, row)| row.iter().sum::<f64>() / mesh.dt[n])
        .collect()
}

/// `1/2 sum |a - b|` of two normalized fields.
pub fn total_variation(a: &[f64], b: &[f64]) -> f64 {
    0.5 * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()
}

/// `g / sum g`; `None` when `sum g <= 0` (no vega: at intrinsic).
pub fn normalize_field(g: &[f64]) -> Option<Vec<f64>> {
    let total: f64 = g.iter().sum();
    // a NaN total fails `total > 0.0` as well and yields None
    let positive = total.is_finite() && total > 0.0;
    if !positive {
        return None;
    }
    Some(g.iter().map(|v| v / total).collect())
}

/// The closed-form European kernel as a density per unit `x` and `t`,
/// sampled at `(x_j, t_mid[n])`:
/// `kappa^E(x, t) = (1/T) phi(x; x0 + (ln K - x0) t/T, sigma0^2 t (T - t)/T)`
/// (uniform in time times the Brownian-bridge marginal from `ln S0` at
/// `t = 0` to `ln K` at `T`; independent of `r` and `q`). Same layout as
/// [`Adjoint::g`].
pub fn brownian_bridge_kernel(mesh: &Mesh, spec: &QuoteSpec, sigma0: f64) -> Vec<f64> {
    let t_exp = mesh.t_expiry;
    let x0 = mesh.x0;
    let ln_k = spec.strike.ln();
    let mut out = Vec::with_capacity(mesh.n_steps() * mesh.n_nodes());
    for &t in &mesh.t_mid {
        let m = x0 + (ln_k - x0) * t / t_exp;
        let v = sigma0 * sigma0 * t * (t_exp - t) / t_exp;
        let norm = 1.0 / (t_exp * (2.0 * std::f64::consts::PI * v).sqrt());
        for &x in &mesh.x {
            out.push(norm * (-(x - m) * (x - m) / (2.0 * v)).exp());
        }
    }
    out
}

/// Both kernels of `spec` at the flat volatility `sigma0`: an American
/// (penalty `rho`) and a European retained solve, one adjoint each.
/// Errors when either vega is not positive (the quote is at intrinsic).
pub fn kernels_flat(
    mesh: &Mesh,
    spec: &QuoteSpec,
    sigma0: f64,
    market: &MarketSlice,
    rho: f64,
) -> Result<Kernels, RustyQLibError> {
    if !(sigma0.is_finite() && sigma0 > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "sigma0",
            format!("flat volatility must be positive and finite, got {sigma0}"),
        ));
    }
    let vol = mesh.node_field(&FlatVol(sigma0));
    let mut ws = Workspace::new();
    let mut aws = AdjointWorkspace::new();
    let mut sol = Solution::new();
    solve_into(
        mesh,
        spec,
        &vol,
        market,
        Mode::American { rho },
        true,
        &mut ws,
        &mut sol,
    )?;
    let adj_a = adjoint(mesh, spec, &sol, market, &vol, &mut aws)?;
    let price_a = sol.price;
    let active_a = sol.active.clone();
    let gamma_a0 = dollar_gamma_at_x0(mesh, sol.final_layer());
    solve_into(
        mesh,
        spec,
        &vol,
        market,
        Mode::European,
        true,
        &mut ws,
        &mut sol,
    )?;
    let adj_e = adjoint(mesh, spec, &sol, market, &vol, &mut aws)?;
    let price_e = sol.price;
    let gamma_e0 = dollar_gamma_at_x0(mesh, sol.final_layer());
    let vega_a = adj_a.vega_from_field;
    let vega_e = adj_e.vega_from_field;
    let kappa_a = normalize_field(&adj_a.g).ok_or_else(|| {
        RustyQLibError::NumericalError(format!(
            "American vega {vega_a} is not positive (K = {}, T = {}): no kernel",
            spec.strike, spec.t_expiry
        ))
    })?;
    let kappa_e = normalize_field(&adj_e.g).ok_or_else(|| {
        RustyQLibError::NumericalError(format!(
            "European vega {vega_e} is not positive (K = {}, T = {}): no kernel",
            spec.strike, spec.t_expiry
        ))
    })?;
    let w_a = time_marginal(mesh, &kappa_a);
    let w_e = time_marginal(mesh, &kappa_e);
    let tv_distance = total_variation(&kappa_a, &kappa_e);
    Ok(Kernels {
        kappa_a,
        kappa_e,
        vega_a,
        vega_e,
        price_a,
        price_e,
        w_a,
        w_e,
        tv_distance,
        dollar_gamma_a0: gamma_a0,
        dollar_gamma_e0: gamma_e0,
        active_a,
        n_steps: mesh.n_steps(),
        stride: mesh.n_nodes(),
    })
}

#[cfg(test)]
#[path = "adjoint_tests.rs"]
mod adjoint_tests;
