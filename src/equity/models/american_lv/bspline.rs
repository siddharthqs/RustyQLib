//! Tensor-product clamped cubic B-spline local-volatility surface in
//! forward log-moneyness and time: the calibration unknown of the
//! American local-vol problem.
//!
//! ```text
//! Coordinates:  y(x, t) = x - ln F_ref(t)   clamped to [y_min, y_max]
//!               t                          clamped to [0, t_max]
//! Surface:      Sigma(x, t) = sum_{a,b} theta[a * n_t_basis + b] B_a(y) C_b(t)
//! Index:        j = j_y * n_t_basis + j_t   (row-major, y outer, t inner)
//! Regularizer:  Q = lambda_y (G_y kron M_t) + lambda_t (M_y kron G_t)
//!               theta^T Q theta = int int lambda_y (d_y Sigma)^2 + lambda_t (d_t Sigma)^2
//! ```
//!
//! `F_ref(t)` is the *prior* forward, fixed for the whole calibration:
//! the surface depends on the carry only through the drift of the
//! pricing PDE, not through its parameterisation. `ln F_ref` is stored
//! as a piecewise-linear table in `t`.
//!
//! Conventions that other files rely on:
//!
//! - The coefficient index is **`j = j_y * n_t_basis + j_t`** everywhere:
//!   in `theta`, in the gradient returned by [`BasisCache::project`], and
//!   in both Kronecker factors of [`BSplineLocalVol::regularizer_matrix`].
//!   Swapping the order does not change any dimension, so it is asserted
//!   by the exact Greville identities in the tests rather than by types.
//! - A [`NodeField`] built here holds `Sigma(x_j, t_mid[n])` at
//!   `values[n * stride + j]`, matching [`super::vol_field`].
//! - A gradient field `g` passed to [`BasisCache::project`] is flat with
//!   the same layout (`g[n * stride + j]`), and the projection is the
//!   chain rule `dF/dtheta_m = sum_{j,n} g_{j,n} B_m(y_{j,n}, t_n)`.
//! - Coefficient bounds are enforced by projection (clamping); because
//!   the basis is nonnegative and sums to one, a coefficient vector inside
//!   `[sigma_min, sigma_max]` gives a surface inside the same interval,
//!   so `vol` never clamps its output.
//!
//! The 1-D mass and stiffness matrices are integrated exactly with a
//! 4-point Gauss–Legendre rule per knot interval (cubic times cubic is
//! degree 6; the rule is exact to degree 7). No Cholesky of `Q` is taken
//! here: the calibration adds `alpha Q` to its normal matrix directly.

use super::vol_field::{NodeField, VolField};
use crate::core::errors::RustyQLibError;
use crate::core::linalg::decomp::cholesky::{cholesky_factor, cholesky_solve};

// ── Defaults ────────────────────────────────────────────────────────────

/// Default number of y basis functions (`N_Y`).
pub const DEFAULT_N_Y: usize = 16;
/// Default lower end of the forward log-moneyness domain.
pub const DEFAULT_Y_MIN: f64 = -1.0;
/// Default upper end of the forward log-moneyness domain.
pub const DEFAULT_Y_MAX: f64 = 0.8;
/// Default interior time breakpoints (years): 1w, 1m, 2m, 3m, 6m, 9m, 1y.
pub const DEFAULT_T_BREAKPOINTS: [f64; 7] =
    [1.0 / 52.0, 1.0 / 12.0, 1.0 / 6.0, 0.25, 0.5, 0.75, 1.0];
/// Default end of the time domain (`T_MAX`).
pub const DEFAULT_T_MAX: f64 = 1.1;
/// Default lower coefficient bound (annualised vol).
pub const SIGMA_MIN: f64 = 0.02;
/// Default upper coefficient bound (annualised vol).
pub const SIGMA_MAX: f64 = 3.0;
/// Subdivisions of each knot interval (per axis) in the composite Gauss
/// rule used by [`BSplineLocalVol::projection_of`].
pub const PROJECTION_SUBDIVISIONS: usize = 4;

/// Polynomial degree of the basis.
const DEGREE: usize = 3;

// ── Gauss–Legendre ──────────────────────────────────────────────────────

/// The 4-point Gauss–Legendre rule on `[-1, 1]` as `(node, weight)`
/// pairs; exact for polynomials of degree `<= 7`.
pub fn gauss_legendre_4() -> [(f64, f64); 4] {
    const X1: f64 = 0.339_981_043_584_856_3;
    const X2: f64 = 0.861_136_311_594_052_6;
    const W1: f64 = 0.652_145_154_862_546_1;
    const W2: f64 = 0.347_854_845_137_453_9;
    [(-X2, W2), (-X1, W1), (X1, W1), (X2, W2)]
}

// ── Knot vectors and Cox–de Boor ────────────────────────────────────────

/// Clamped cubic knot vector with `n` basis functions on `[a, b]` and
/// the given interior knots (strictly increasing, inside `(a, b)`):
/// `a` four times, the interior knots, `b` four times; `n = interior + 4`.
fn clamped_knots(a: f64, b: f64, interior: &[f64]) -> Vec<f64> {
    let mut knots = Vec::with_capacity(interior.len() + 8);
    knots.extend(std::iter::repeat_n(a, DEGREE + 1));
    knots.extend_from_slice(interior);
    knots.extend(std::iter::repeat_n(b, DEGREE + 1));
    knots
}

/// Uniform interior knots for `n` clamped cubic functions on `[a, b]`
/// (`n - 3` intervals, `n - 4` interior knots).
fn uniform_interior(a: f64, b: f64, n: usize) -> Vec<f64> {
    let intervals = n - DEGREE;
    let h = (b - a) / intervals as f64;
    (1..intervals).map(|i| a + h * i as f64).collect()
}

/// The knot span `k` with `knots[k] <= u < knots[k + 1]`, restricted to
/// the nondegenerate spans `3..=n-1` (`u >= b` maps to the last span).
/// `u` must already be clamped to `[a, b]`.
#[inline]
fn find_span(knots: &[f64], n: usize, u: f64) -> usize {
    // knots[4..=n] are the interior knots followed by b; count those <= u.
    let c = knots[DEGREE + 1..=n].partition_point(|&k| k <= u);
    (DEGREE + c).min(n - 1)
}

/// The four nonzero cubic basis functions at `u` in span `span`
/// (`N_{span-3..=span, 3}`), by the triangular Cox–de Boor recursion.
#[inline]
fn basis_funs(knots: &[f64], span: usize, u: f64) -> [f64; 4] {
    let mut n = [1.0, 0.0, 0.0, 0.0];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    for j in 1..=DEGREE {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
    }
    n
}

/// The four nonzero cubic basis functions and their first derivatives at
/// `u` in span `span`.
///
/// The derivative uses `N'_{i,3} = 3 (N_{i,2}/(u_{i+3}-u_i) -
/// N_{i+1,2}/(u_{i+4}-u_{i+1}))` with the `0/0 = 0` convention at the
/// clamped ends (there the numerator vanishes as well).
fn basis_funs_with_derivs(knots: &[f64], span: usize, u: f64) -> ([f64; 4], [f64; 4]) {
    let mut n = [1.0, 0.0, 0.0, 0.0];
    let mut left = [0.0; 4];
    let mut right = [0.0; 4];
    let mut n2 = [0.0; 3];
    for j in 1..=DEGREE {
        left[j] = u - knots[span + 1 - j];
        right[j] = knots[span + j] - u;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = n[r] / (right[r + 1] + left[j - r]);
            n[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        n[j] = saved;
        if j == 2 {
            n2.copy_from_slice(&n[..3]);
        }
    }
    // n2[s] = N_{span-2+s, 2}.
    let mut d = [0.0; 4];
    for r in 0..4 {
        let i = span - DEGREE + r;
        let term_a = if r >= 1 {
            let den = knots[i + 3] - knots[i];
            if den > 0.0 {
                n2[r - 1] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        let term_b = if r <= 2 {
            let den = knots[i + 4] - knots[i + 1];
            if den > 0.0 {
                n2[r] / den
            } else {
                0.0
            }
        } else {
            0.0
        };
        d[r] = DEGREE as f64 * (term_a - term_b);
    }
    (n, d)
}

/// Greville abscissae `xi_j = (u_{j+1} + u_{j+2} + u_{j+3}) / 3` of a
/// clamped cubic knot vector with `n` functions. Coefficients
/// `theta_j = xi_j` reproduce the identity exactly.
pub fn greville_abscissae(knots: &[f64], n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| (knots[j + 1] + knots[j + 2] + knots[j + 3]) / 3.0)
        .collect()
}

/// Exact 1-D mass matrix `M[i][j] = int B_i B_j` and stiffness matrix
/// `G[i][j] = int B_i' B_j'` of the clamped cubic basis with `n`
/// functions on `knots`, by 4-point Gauss–Legendre per knot interval.
/// Both are returned row-major and are exactly symmetric.
pub fn mass_and_stiffness(knots: &[f64], n: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut mass = vec![vec![0.0; n]; n];
    let mut stiff = vec![vec![0.0; n]; n];
    let rule = gauss_legendre_4();
    for span in DEGREE..n {
        let (lo, hi) = (knots[span], knots[span + 1]);
        if hi <= lo {
            continue;
        }
        let half = 0.5 * (hi - lo);
        let mid = 0.5 * (hi + lo);
        for &(node, weight) in &rule {
            let u = mid + half * node;
            let w = weight * half;
            let (b, db) = basis_funs_with_derivs(knots, span, u);
            // `w * (p * q)` rather than `w * p * q`: multiplication is
            // commutative, so the (i, j) and (j, i) entries are bit-equal.
            for r in 0..4 {
                let i = span - DEGREE + r;
                for s in 0..4 {
                    let j = span - DEGREE + s;
                    mass[i][j] += w * (b[r] * b[s]);
                    stiff[i][j] += w * (db[r] * db[s]);
                }
            }
        }
    }
    (mass, stiff)
}

/// Clamp every coefficient into `[lo, hi]` (the projection onto the
/// box constraint).
pub fn project_onto_bounds(theta: &mut [f64], lo: f64, hi: f64) {
    for v in theta.iter_mut() {
        *v = v.clamp(lo, hi);
    }
}

// ── The surface ─────────────────────────────────────────────────────────

/// Tensor-product clamped cubic B-spline local-volatility surface
/// `Sigma(x, t) = sum_m theta[m] B_m(y(x, t), t)`.
///
/// **Coefficient index: `m = j_y * n_t_basis + j_t`** (row-major, `y`
/// outer, `t` inner); `theta.len() == n_y * n_t_basis`. The same order
/// is used by [`BasisCache::project`] and by the Kronecker products in
/// [`BSplineLocalVol::regularizer_matrix`].
///
/// `y = x - ln F_ref(t)` is clamped to `[y_min, y_max]` and `t` to
/// `[0, t_max]`, so the surface extends constantly (in the clamped
/// coordinate) outside its domain. The clamp is silent: meshes wider
/// than the domain are expected at the far wings.
#[derive(Debug, Clone)]
pub struct BSplineLocalVol {
    /// Clamped cubic knot vector in `y` (`n_y + 4` entries).
    y_knots: Vec<f64>,
    /// Clamped cubic knot vector in `t` (`n_t_basis + 4` entries).
    t_knots: Vec<f64>,
    /// Number of `y` basis functions.
    n_y: usize,
    /// Number of `t` basis functions (`interior breakpoints + 4`).
    n_t_basis: usize,
    /// Coefficients, `theta[j_y * n_t_basis + j_t]`.
    pub theta: Vec<f64>,
    /// `(t, ln F_ref(t))` table, increasing in `t`, piecewise linear.
    f_ref: Vec<(f64, f64)>,
    /// Coefficient bounds `(sigma_min, sigma_max)` used by
    /// [`BSplineLocalVol::clamp_theta`].
    pub bounds: (f64, f64),
}

impl BSplineLocalVol {
    /// Build a surface with all coefficients zero.
    ///
    /// - `n_y >= 4` basis functions, uniform interior knots on
    ///   `[y_min, y_max]`;
    /// - `t_breakpoints`: strictly increasing interior time knots in
    ///   `(0, t_max)`; the basis then has `t_breakpoints.len() + 4`
    ///   functions;
    /// - `f_ref`: `(t, ln F_ref(t))` pairs, strictly increasing in `t`,
    ///   at least one entry; interpolated linearly, extrapolated linearly
    ///   from the end segments (constant with a single entry).
    ///
    /// Coefficient order `theta[j_y * n_t_basis + j_t]`.
    pub fn new(
        n_y: usize,
        y_min: f64,
        y_max: f64,
        t_breakpoints: &[f64],
        t_max: f64,
        f_ref: Vec<(f64, f64)>,
    ) -> Result<Self, RustyQLibError> {
        if n_y < DEGREE + 1 {
            return Err(RustyQLibError::invalid_input(
                "n_y",
                "at least 4 cubic basis functions are required",
            ));
        }
        if !(y_min.is_finite() && y_max.is_finite() && y_min < y_max) {
            return Err(RustyQLibError::invalid_input(
                "y_range",
                "y_min < y_max must be finite",
            ));
        }
        if !(t_max.is_finite() && t_max > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "t_max",
                "must be finite and positive",
            ));
        }
        let mut prev = 0.0;
        for &b in t_breakpoints {
            if !(b.is_finite() && b > prev && b < t_max) {
                return Err(RustyQLibError::invalid_input(
                    "t_breakpoints",
                    "must be finite, strictly increasing and inside (0, t_max)",
                ));
            }
            prev = b;
        }
        if f_ref.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "f_ref",
                "at least one (t, ln F_ref) entry is required",
            ));
        }
        for (i, &(t, lf)) in f_ref.iter().enumerate() {
            if !(t.is_finite() && lf.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "f_ref",
                    "entries must be finite",
                ));
            }
            if i > 0 && t <= f_ref[i - 1].0 {
                return Err(RustyQLibError::invalid_input(
                    "f_ref",
                    "times must be strictly increasing",
                ));
            }
        }
        let y_knots = clamped_knots(y_min, y_max, &uniform_interior(y_min, y_max, n_y));
        let t_knots = clamped_knots(0.0, t_max, t_breakpoints);
        let n_t_basis = t_breakpoints.len() + DEGREE + 1;
        Ok(BSplineLocalVol {
            y_knots,
            t_knots,
            n_y,
            n_t_basis,
            theta: vec![0.0; n_y * n_t_basis],
            f_ref,
            bounds: (SIGMA_MIN, SIGMA_MAX),
        })
    }

    /// Like [`BSplineLocalVol::new`] with every coefficient equal to
    /// `sigma0` (the flat calibration start; the surface is then exactly
    /// `sigma0` everywhere by partition of unity).
    pub fn flat(
        sigma0: f64,
        n_y: usize,
        y_min: f64,
        y_max: f64,
        t_breakpoints: &[f64],
        t_max: f64,
        f_ref: Vec<(f64, f64)>,
    ) -> Result<Self, RustyQLibError> {
        if !(sigma0.is_finite() && sigma0 > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "sigma0",
                "must be finite and positive",
            ));
        }
        let mut s = Self::new(n_y, y_min, y_max, t_breakpoints, t_max, f_ref)?;
        s.theta.iter_mut().for_each(|v| *v = sigma0);
        Ok(s)
    }

    /// The paper's default basis (`N_Y = 16` on `[-1.0, 0.8]`, time
    /// breakpoints [`DEFAULT_T_BREAKPOINTS`] on `[0, 1.1]`, hence 11 time
    /// functions) at the flat level `sigma0`.
    pub fn with_defaults(sigma0: f64, f_ref: Vec<(f64, f64)>) -> Result<Self, RustyQLibError> {
        Self::flat(
            sigma0,
            DEFAULT_N_Y,
            DEFAULT_Y_MIN,
            DEFAULT_Y_MAX,
            &DEFAULT_T_BREAKPOINTS,
            DEFAULT_T_MAX,
            f_ref,
        )
    }

    /// Replace the coefficients (`theta[j_y * n_t_basis + j_t]`); the
    /// length must equal [`BSplineLocalVol::len`].
    pub fn set_theta(&mut self, theta: Vec<f64>) -> Result<(), RustyQLibError> {
        if theta.len() != self.len() {
            return Err(RustyQLibError::invalid_input(
                "theta",
                format!("expected {} coefficients, got {}", self.len(), theta.len()),
            ));
        }
        self.theta = theta;
        Ok(())
    }

    /// Builder form of [`BSplineLocalVol::set_theta`].
    pub fn with_theta(mut self, theta: Vec<f64>) -> Result<Self, RustyQLibError> {
        self.set_theta(theta)?;
        Ok(self)
    }

    /// Number of coefficients `M = n_y * n_t_basis`.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_y * self.n_t_basis
    }

    /// `true` when there are no coefficients (never, for a valid surface;
    /// provided for the `len`/`is_empty` lint pair).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of `y` basis functions.
    #[inline]
    pub fn n_y(&self) -> usize {
        self.n_y
    }

    /// Number of `t` basis functions.
    #[inline]
    pub fn n_t_basis(&self) -> usize {
        self.n_t_basis
    }

    /// Flat coefficient index of `(j_y, j_t)`: `j_y * n_t_basis + j_t`.
    #[inline]
    pub fn index(&self, j_y: usize, j_t: usize) -> usize {
        j_y * self.n_t_basis + j_t
    }

    /// Clamped cubic knot vector in `y` (for the manifest).
    pub fn y_knots(&self) -> &[f64] {
        &self.y_knots
    }

    /// Clamped cubic knot vector in `t` (for the manifest).
    pub fn t_knots(&self) -> &[f64] {
        &self.t_knots
    }

    /// `(y_min, y_max)`.
    #[inline]
    pub fn y_range(&self) -> (f64, f64) {
        (self.y_knots[0], self.y_knots[self.n_y])
    }

    /// `t_max`.
    #[inline]
    pub fn t_max(&self) -> f64 {
        self.t_knots[self.n_t_basis]
    }

    /// The `(t, ln F_ref(t))` table.
    pub fn f_ref(&self) -> &[(f64, f64)] {
        &self.f_ref
    }

    /// Greville abscissae of the `y` basis (length `n_y`).
    pub fn greville_y(&self) -> Vec<f64> {
        greville_abscissae(&self.y_knots, self.n_y)
    }

    /// Greville abscissae of the `t` basis (length `n_t_basis`).
    pub fn greville_t(&self) -> Vec<f64> {
        greville_abscissae(&self.t_knots, self.n_t_basis)
    }

    /// Default regularizer weights `(lambda_y, lambda_t) =
    /// (1/(y_max - y_min)^2, 1/t_max^2)`, dimensionless per domain.
    pub fn default_lambdas(&self) -> (f64, f64) {
        let (y_min, y_max) = self.y_range();
        let t_max = self.t_max();
        (
            1.0 / ((y_max - y_min) * (y_max - y_min)),
            1.0 / (t_max * t_max),
        )
    }

    /// `ln F_ref(t)`: linear interpolation of the table, linear
    /// extrapolation from the end segments, constant for a single entry.
    pub fn ln_f_ref(&self, t: f64) -> f64 {
        let f = &self.f_ref;
        let n = f.len();
        if n == 1 {
            return f[0].1;
        }
        let i = f.partition_point(|p| p.0 <= t).clamp(1, n - 1);
        let (t0, v0) = f[i - 1];
        let (t1, v1) = f[i];
        v0 + (v1 - v0) * (t - t0) / (t1 - t0)
    }

    /// Forward log-moneyness `y = x - ln F_ref(t)` clamped to
    /// `[y_min, y_max]`.
    #[inline]
    pub fn y_of(&self, x: f64, t: f64) -> f64 {
        let (y_min, y_max) = self.y_range();
        (x - self.ln_f_ref(t)).clamp(y_min, y_max)
    }

    /// The four nonzero `y` basis functions at `y` (clamped to the
    /// domain): consecutive indices `first..first + 4` and their values.
    /// Values are nonnegative and sum to one.
    #[inline]
    pub fn basis_y(&self, y: f64) -> ([usize; 4], [f64; 4]) {
        let (y_min, y_max) = self.y_range();
        let u = y.clamp(y_min, y_max);
        let span = find_span(&self.y_knots, self.n_y, u);
        let first = span - DEGREE;
        (
            [first, first + 1, first + 2, first + 3],
            basis_funs(&self.y_knots, span, u),
        )
    }

    /// The four nonzero `t` basis functions at `t` (clamped to
    /// `[0, t_max]`), as for [`BSplineLocalVol::basis_y`].
    #[inline]
    pub fn basis_t(&self, t: f64) -> ([usize; 4], [f64; 4]) {
        let u = t.clamp(0.0, self.t_max());
        let span = find_span(&self.t_knots, self.n_t_basis, u);
        let first = span - DEGREE;
        (
            [first, first + 1, first + 2, first + 3],
            basis_funs(&self.t_knots, span, u),
        )
    }

    /// The surface in its own coordinates, `sum theta[a * n_t_basis + b]
    /// B_a(y) C_b(t)` (no `F_ref` shift; inputs clamped to the domain).
    pub fn evaluate_yt(&self, y: f64, t: f64) -> f64 {
        let (iy, by) = self.basis_y(y);
        let (it, bt) = self.basis_t(t);
        let nt = self.n_t_basis;
        let mut acc = 0.0;
        for (a, &bya) in by.iter().enumerate() {
            let row = iy[a] * nt + it[0];
            let mut partial = 0.0;
            for (b, &btb) in bt.iter().enumerate() {
                partial += self.theta[row + b] * btb;
            }
            acc += bya * partial;
        }
        acc
    }

    /// Clamp the coefficients into `self.bounds`.
    pub fn clamp_theta(&mut self) {
        project_onto_bounds(&mut self.theta, self.bounds.0, self.bounds.1);
    }

    /// Basis cache for one expiry mesh (`x_nodes` nodes, `t_mid` step
    /// mid-times); depends on the knots and `F_ref` only, not on `theta`.
    pub fn basis_cache(&self, x_nodes: &[f64], t_mid: &[f64]) -> BasisCache {
        BasisCache::build(self, x_nodes, t_mid)
    }

    /// The surface sampled at every `(x_nodes[j], t_mid[n])`, stored at
    /// `values[n * stride + j]` (the solver's read layout).
    pub fn node_field(&self, x_nodes: &[f64], t_mid: &[f64]) -> NodeField {
        let stride = x_nodes.len();
        let nt = self.n_t_basis;
        let mut values = Vec::with_capacity(stride * t_mid.len());
        for &t in t_mid {
            let (it, bt) = self.basis_t(t);
            let lf = self.ln_f_ref(t);
            let (y_min, y_max) = self.y_range();
            for &x in x_nodes {
                let (iy, by) = self.basis_y((x - lf).clamp(y_min, y_max));
                let mut acc = 0.0;
                for (a, &bya) in by.iter().enumerate() {
                    let row = iy[a] * nt + it[0];
                    let mut partial = 0.0;
                    for (b, &btb) in bt.iter().enumerate() {
                        partial += self.theta[row + b] * btb;
                    }
                    acc += bya * partial;
                }
                values.push(acc);
            }
        }
        NodeField {
            values,
            stride,
            t_mid: t_mid.to_vec(),
            x_min: x_nodes.first().copied().unwrap_or(0.0),
            dx: node_spacing(x_nodes),
        }
    }

    /// The node field of the current `theta` through a prebuilt cache
    /// (16 multiply-adds per node; the per-theta path of the calibration).
    pub fn node_field_from_cache(&self, cache: &BasisCache) -> NodeField {
        cache.node_field(&self.theta)
    }

    /// Project a gradient field `g[n * stride + j] = dF/dSigma_{j,n}` onto
    /// the coefficients: `out[m] = sum_{j,n} g_{j,n} B_m(y_{j,n}, t_n)`
    /// with `m = j_y * n_t_basis + j_t`. Delegates to
    /// [`BasisCache::project`].
    pub fn project(&self, gradient_field: &[f64], cache: &BasisCache) -> Vec<f64> {
        cache.project(gradient_field)
    }

    /// The exact anisotropic `H^1` Gram matrix
    /// `Q = lambda_y (G_y kron M_t) + lambda_t (M_y kron G_t)`, `M x M`
    /// row-major, so that `theta^T Q theta = int int lambda_y (d_y
    /// Sigma)^2 + lambda_t (d_t Sigma)^2` over `[y_min, y_max] x [0,
    /// t_max]`. Entry `Q[a * n_t + b][c * n_t + d] = lambda_y G_y[a][c]
    /// M_t[b][d] + lambda_t M_y[a][c] G_t[b][d]` (coefficient index
    /// `j_y * n_t_basis + j_t`). Symmetric positive semidefinite; its
    /// kernel is the constant surface, so callers add it to an already
    /// positive matrix rather than factorising it alone.
    pub fn regularizer_matrix(&self, lambda_y: f64, lambda_t: f64) -> Vec<Vec<f64>> {
        let (m_y, g_y) = mass_and_stiffness(&self.y_knots, self.n_y);
        let (m_t, g_t) = mass_and_stiffness(&self.t_knots, self.n_t_basis);
        let nt = self.n_t_basis;
        let m = self.len();
        let mut q = vec![vec![0.0; m]; m];
        for a in 0..self.n_y {
            for c in 0..self.n_y {
                let (gy, my) = (g_y[a][c], m_y[a][c]);
                if gy == 0.0 && my == 0.0 {
                    continue;
                }
                for b in 0..nt {
                    let row = &mut q[a * nt + b];
                    for d in 0..nt {
                        row[c * nt + d] = lambda_y * gy * m_t[b][d] + lambda_t * my * g_t[b][d];
                    }
                }
            }
        }
        q
    }

    /// [`BSplineLocalVol::regularizer_matrix`] at
    /// [`BSplineLocalVol::default_lambdas`].
    pub fn regularizer_matrix_default(&self) -> Vec<Vec<f64>> {
        let (ly, lt) = self.default_lambdas();
        self.regularizer_matrix(ly, lt)
    }

    /// Least-squares projection of an arbitrary field onto the basis:
    /// the coefficients (index `j_y * n_t_basis + j_t`) minimising
    /// `int int (Sigma_theta(y, t) - field(y + ln F(t), t))^2` over the
    /// domain, with the integral taken by a composite 4-point Gauss rule
    /// ([`PROJECTION_SUBDIVISIONS`] subintervals per knot interval on each
    /// axis). `f_ref` is the forward table used to map `y` back to the
    /// field's `x` (`None`: this surface's own table); pass the truth's
    /// forward when the field is a synthetic truth defined in its own
    /// forward moneyness. This is the "basis floor" of the synthetic
    /// study. The result is not clamped to the bounds.
    pub fn projection_of(
        &self,
        field: &dyn VolField,
        f_ref: Option<&[(f64, f64)]>,
    ) -> Result<Vec<f64>, RustyQLibError> {
        let own;
        let table: &[(f64, f64)] = match f_ref {
            Some(t) => {
                if t.is_empty() {
                    return Err(RustyQLibError::invalid_input(
                        "f_ref",
                        "at least one (t, ln F_ref) entry is required",
                    ));
                }
                t
            }
            None => {
                own = &self.f_ref;
                own
            }
        };
        let ln_f = |t: f64| interp_table(table, t);

        // Composite quadrature points (node, weight) on each axis.
        let y_pts = composite_gauss(&self.y_knots, self.n_y, PROJECTION_SUBDIVISIONS);
        let t_pts = composite_gauss(&self.t_knots, self.n_t_basis, PROJECTION_SUBDIVISIONS);

        // 1-D discrete Gram matrices A_y, A_t (equal to the exact mass
        // matrices for this rule) and the basis values at every point.
        let ny = self.n_y;
        let nt = self.n_t_basis;
        let mut a_y = vec![vec![0.0; ny]; ny];
        let mut a_t = vec![vec![0.0; nt]; nt];
        let y_basis: Vec<([usize; 4], [f64; 4])> =
            y_pts.iter().map(|p| self.basis_y(p.0)).collect();
        let t_basis: Vec<([usize; 4], [f64; 4])> =
            t_pts.iter().map(|p| self.basis_t(p.0)).collect();
        // `w * (p * q)` keeps the Gram matrices bit-symmetric (Cholesky
        // rejects asymmetry beyond 1e-10 relative).
        for (p, (idx, val)) in y_pts.iter().zip(&y_basis) {
            for r in 0..4 {
                for s in 0..4 {
                    a_y[idx[r]][idx[s]] += p.1 * (val[r] * val[s]);
                }
            }
        }
        for (p, (idx, val)) in t_pts.iter().zip(&t_basis) {
            for r in 0..4 {
                for s in 0..4 {
                    a_t[idx[r]][idx[s]] += p.1 * (val[r] * val[s]);
                }
            }
        }

        // Right-hand side sum_q w_q B_m(q) f(q), index j_y * n_t + j_t.
        let m = self.len();
        let mut rhs = vec![0.0; m];
        for (pt, (it, bt)) in t_pts.iter().zip(&t_basis) {
            let t = pt.0;
            let lf = ln_f(t);
            for (py, (iy, by)) in y_pts.iter().zip(&y_basis) {
                let f = field.vol(py.0 + lf, t);
                if !f.is_finite() {
                    return Err(RustyQLibError::NumericalError(format!(
                        "projection_of: field is not finite at y = {}, t = {}",
                        py.0, t
                    )));
                }
                let w = pt.1 * py.1 * f;
                for (a, &bya) in by.iter().enumerate() {
                    let row = iy[a] * nt;
                    let wa = w * bya;
                    for (b, &btb) in bt.iter().enumerate() {
                        rhs[row + it[b]] += wa * btb;
                    }
                }
            }
        }

        // Normal matrix A_y kron A_t (same Kronecker order as Q).
        let mut normal = vec![vec![0.0; m]; m];
        for a in 0..ny {
            for c in 0..ny {
                let ay = a_y[a][c];
                if ay == 0.0 {
                    continue;
                }
                for b in 0..nt {
                    let row = &mut normal[a * nt + b];
                    for d in 0..nt {
                        row[c * nt + d] = ay * a_t[b][d];
                    }
                }
            }
        }
        let l = cholesky_factor(&normal).map_err(|e| {
            RustyQLibError::NumericalError(format!("projection_of: Gram matrix not SPD ({e})"))
        })?;
        cholesky_solve(&l, &rhs)
    }
}

impl VolField for BSplineLocalVol {
    /// `Sigma(x, t)` with `y = x - ln F_ref(t)` clamped to the domain and
    /// `t` clamped to `[0, t_max]`.
    #[inline]
    fn vol(&self, x: f64, t: f64) -> f64 {
        self.evaluate_yt(self.y_of(x, t), t)
    }
}

/// Uniform node spacing of a mesh (`0` for a single node), matching
/// [`NodeField::sample`].
fn node_spacing(x_nodes: &[f64]) -> f64 {
    let n = x_nodes.len();
    if n > 1 {
        (x_nodes[n - 1] - x_nodes[0]) / (n - 1) as f64
    } else {
        0.0
    }
}

/// Linear interpolation / end-segment extrapolation of a `(t, v)` table
/// (non-empty, increasing in `t`).
fn interp_table(table: &[(f64, f64)], t: f64) -> f64 {
    let n = table.len();
    if n == 1 {
        return table[0].1;
    }
    let i = table.partition_point(|p| p.0 <= t).clamp(1, n - 1);
    let (t0, v0) = table[i - 1];
    let (t1, v1) = table[i];
    v0 + (v1 - v0) * (t - t0) / (t1 - t0)
}

/// Composite 4-point Gauss rule on the nondegenerate knot intervals of a
/// clamped knot vector, `sub` subintervals per knot interval, as
/// `(node, weight)` pairs.
fn composite_gauss(knots: &[f64], n: usize, sub: usize) -> Vec<(f64, f64)> {
    let rule = gauss_legendre_4();
    let mut pts = Vec::with_capacity((n - DEGREE) * sub * 4);
    for span in DEGREE..n {
        let (lo, hi) = (knots[span], knots[span + 1]);
        if hi <= lo {
            continue;
        }
        let h = (hi - lo) / sub as f64;
        for s in 0..sub {
            let a = lo + h * s as f64;
            let half = 0.5 * h;
            let mid = a + half;
            for &(node, weight) in &rule {
                pts.push((mid + half * node, weight * half));
            }
        }
    }
    pts
}

// ── Basis cache ─────────────────────────────────────────────────────────

/// Per-expiry-mesh cache of the nonzero basis functions at every solver
/// evaluation point `(x_j, t_mid[n])`.
///
/// Separable storage: per `(step, node)` the first `y` index and the four
/// `y` basis values (the four nonzero functions have consecutive indices,
/// so `(index, value)` pairs are `(y_first + r, y_vals[r])`), and per step
/// the first `t` index and four `t` values. The `y` entries depend on the
/// step because `y = x - ln F_ref(t)` shifts with `t`. About 40 bytes per
/// `(step, node)`: 4 MB at 400 x 250. Independent of `theta`; rebuild
/// only when the knots or `F_ref` change.
///
/// Layouts: `y_first[n * stride + j]`, `y_vals[n * stride + j]`,
/// `t_first[n]`, `t_vals[n]`; coefficient index `j_y * n_t_basis + j_t`.
#[derive(Debug, Clone)]
pub struct BasisCache {
    /// Number of `y` basis functions.
    pub n_y: usize,
    /// Number of `t` basis functions.
    pub n_t_basis: usize,
    /// Nodes per level.
    pub stride: usize,
    /// Number of steps (levels).
    pub n_steps: usize,
    /// First nonzero `y` index per `(step, node)`.
    pub y_first: Vec<usize>,
    /// The four nonzero `y` basis values per `(step, node)`.
    pub y_vals: Vec<[f64; 4]>,
    /// First nonzero `t` index per step.
    pub t_first: Vec<usize>,
    /// The four nonzero `t` basis values per step.
    pub t_vals: Vec<[f64; 4]>,
    /// First mesh node (for building a [`NodeField`]).
    pub x_min: f64,
    /// Mesh node spacing (for building a [`NodeField`]).
    pub dx: f64,
    /// Step mid-times (for building a [`NodeField`]).
    pub t_mid: Vec<f64>,
}

impl BasisCache {
    /// Evaluate the basis of `surface` at every `(x_nodes[j], t_mid[n])`.
    pub fn build(surface: &BSplineLocalVol, x_nodes: &[f64], t_mid: &[f64]) -> Self {
        let stride = x_nodes.len();
        let n_steps = t_mid.len();
        let mut y_first = Vec::with_capacity(stride * n_steps);
        let mut y_vals = Vec::with_capacity(stride * n_steps);
        let mut t_first = Vec::with_capacity(n_steps);
        let mut t_vals = Vec::with_capacity(n_steps);
        let (y_min, y_max) = surface.y_range();
        for &t in t_mid {
            let (it, bt) = surface.basis_t(t);
            t_first.push(it[0]);
            t_vals.push(bt);
            let lf = surface.ln_f_ref(t);
            for &x in x_nodes {
                let (iy, by) = surface.basis_y((x - lf).clamp(y_min, y_max));
                y_first.push(iy[0]);
                y_vals.push(by);
            }
        }
        BasisCache {
            n_y: surface.n_y,
            n_t_basis: surface.n_t_basis,
            stride,
            n_steps,
            y_first,
            y_vals,
            t_first,
            t_vals,
            x_min: x_nodes.first().copied().unwrap_or(0.0),
            dx: node_spacing(x_nodes),
            t_mid: t_mid.to_vec(),
        }
    }

    /// Number of coefficients `M = n_y * n_t_basis`.
    #[inline]
    pub fn len(&self) -> usize {
        self.n_y * self.n_t_basis
    }

    /// `true` when the cache covers no evaluation points.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.y_vals.is_empty()
    }

    /// Approximate heap size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.y_first.len() * std::mem::size_of::<usize>()
            + self.y_vals.len() * std::mem::size_of::<[f64; 4]>()
            + self.t_first.len() * std::mem::size_of::<usize>()
            + self.t_vals.len() * std::mem::size_of::<[f64; 4]>()
            + self.t_mid.len() * std::mem::size_of::<f64>()
    }

    /// The four `(y index, value)` pairs at `(step, node)`.
    #[inline]
    pub fn y_pairs(&self, step: usize, node: usize) -> [(usize, f64); 4] {
        let k = step * self.stride + node;
        let f = self.y_first[k];
        let v = self.y_vals[k];
        [(f, v[0]), (f + 1, v[1]), (f + 2, v[2]), (f + 3, v[3])]
    }

    /// The four `(t index, value)` pairs at `step`.
    #[inline]
    pub fn t_pairs(&self, step: usize) -> [(usize, f64); 4] {
        let f = self.t_first[step];
        let v = self.t_vals[step];
        [(f, v[0]), (f + 1, v[1]), (f + 2, v[2]), (f + 3, v[3])]
    }

    /// `Sigma(x_node, t_mid[step])` for the coefficients `theta`
    /// (index `j_y * n_t_basis + j_t`).
    #[inline]
    pub fn evaluate(&self, theta: &[f64], step: usize, node: usize) -> f64 {
        let k = step * self.stride + node;
        let nt = self.n_t_basis;
        let by = &self.y_vals[k];
        let bt = &self.t_vals[step];
        let t0 = self.t_first[step];
        let mut acc = 0.0;
        for (a, &bya) in by.iter().enumerate() {
            let row = (self.y_first[k] + a) * nt + t0;
            let mut partial = 0.0;
            for (b, &btb) in bt.iter().enumerate() {
                partial += theta[row + b] * btb;
            }
            acc += bya * partial;
        }
        acc
    }

    /// Fill `out[n * stride + j] = Sigma(x_j, t_mid[n])` for `theta`
    /// without allocating; `out.len()` must be `n_steps * stride`.
    pub fn fill_node_values(&self, theta: &[f64], out: &mut [f64]) {
        assert_eq!(
            theta.len(),
            self.len(),
            "BasisCache::fill_node_values: theta length mismatch"
        );
        assert_eq!(
            out.len(),
            self.n_steps * self.stride,
            "BasisCache::fill_node_values: output length mismatch"
        );
        let nt = self.n_t_basis;
        for n in 0..self.n_steps {
            let bt = self.t_vals[n];
            let t0 = self.t_first[n];
            let base = n * self.stride;
            for j in 0..self.stride {
                let k = base + j;
                let by = &self.y_vals[k];
                let y0 = self.y_first[k];
                let mut acc = 0.0;
                for (a, &bya) in by.iter().enumerate() {
                    let row = (y0 + a) * nt + t0;
                    let partial = theta[row] * bt[0]
                        + theta[row + 1] * bt[1]
                        + theta[row + 2] * bt[2]
                        + theta[row + 3] * bt[3];
                    acc += bya * partial;
                }
                out[k] = acc;
            }
        }
    }

    /// A [`NodeField`] of `theta` on the cached mesh.
    pub fn node_field(&self, theta: &[f64]) -> NodeField {
        let mut values = vec![0.0; self.n_steps * self.stride];
        self.fill_node_values(theta, &mut values);
        NodeField {
            values,
            stride: self.stride,
            t_mid: self.t_mid.clone(),
            x_min: self.x_min,
            dx: self.dx,
        }
    }

    /// Projection of a gradient field onto the coefficients:
    /// `out[j_y * n_t_basis + j_t] = sum_{j,n} g[n * stride + j]
    /// B_{j_y}(y_{j,n}) C_{j_t}(t_n)`; each row is contracted with the
    /// `y` basis first (`n_y` partials per step), then with the `t`
    /// basis. Allocates the output and an `n_y` scratch.
    pub fn project(&self, gradient_field: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        let mut scratch = vec![0.0; self.n_y];
        self.project_into(gradient_field, &mut out, &mut scratch);
        out
    }

    /// Allocation-free form of [`BasisCache::project`]: `out` (length
    /// `M`) is overwritten, `scratch` needs `n_y` entries.
    pub fn project_into(&self, gradient_field: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        assert_eq!(
            gradient_field.len(),
            self.n_steps * self.stride,
            "BasisCache::project_into: gradient field length mismatch"
        );
        assert_eq!(
            out.len(),
            self.len(),
            "BasisCache::project_into: output length mismatch"
        );
        assert!(
            scratch.len() >= self.n_y,
            "BasisCache::project_into: scratch needs n_y entries"
        );
        out.iter_mut().for_each(|v| *v = 0.0);
        let nt = self.n_t_basis;
        let partial = &mut scratch[..self.n_y];
        for n in 0..self.n_steps {
            partial.iter_mut().for_each(|v| *v = 0.0);
            let base = n * self.stride;
            for j in 0..self.stride {
                let k = base + j;
                let g = gradient_field[k];
                if g == 0.0 {
                    continue;
                }
                let by = &self.y_vals[k];
                let y0 = self.y_first[k];
                partial[y0] += g * by[0];
                partial[y0 + 1] += g * by[1];
                partial[y0 + 2] += g * by[2];
                partial[y0 + 3] += g * by[3];
            }
            let bt = self.t_vals[n];
            let t0 = self.t_first[n];
            for (a, &p) in partial.iter().enumerate() {
                if p == 0.0 {
                    continue;
                }
                let row = a * nt + t0;
                out[row] += p * bt[0];
                out[row + 1] += p * bt[1];
                out[row + 2] += p * bt[2];
                out[row + 3] += p * bt[3];
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::models::american_lv::vol_field::CallbackVol;

    /// Small deterministic LCG (no rand dependency).
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    /// The paper's default basis with `ln F_ref(t) = 4.6 + 0.02 t`.
    fn surface() -> BSplineLocalVol {
        let f_ref = vec![(0.0, 4.6), (0.5, 4.61), (1.1, 4.622)];
        BSplineLocalVol::with_defaults(0.2, f_ref).unwrap()
    }

    /// A basis with zero forward shift, so `x == y`.
    fn surface_zero_fref() -> BSplineLocalVol {
        BSplineLocalVol::new(
            DEFAULT_N_Y,
            DEFAULT_Y_MIN,
            DEFAULT_Y_MAX,
            &DEFAULT_T_BREAKPOINTS,
            DEFAULT_T_MAX,
            vec![(0.0, 0.0)],
        )
        .unwrap()
    }

    fn max_abs(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |m, x| m.max(x.abs()))
    }

    #[test]
    fn gauss_rule_integrates_degree_seven_exactly() {
        // int_{-1}^{1} (x^7 + 3x^6 - x^4 + 2) dx = 0 + 6/7 - 2/5 + 4
        let exact = 6.0 / 7.0 - 2.0 / 5.0 + 4.0;
        let got: f64 = gauss_legendre_4()
            .iter()
            .map(|&(x, w)| w * (x.powi(7) + 3.0 * x.powi(6) - x.powi(4) + 2.0))
            .sum();
        assert!(
            (got - exact).abs() < 1e-14,
            "gauss 4-point: {got} vs {exact}"
        );
    }

    #[test]
    fn default_basis_has_the_documented_dimensions_and_knots() {
        let s = surface();
        assert_eq!(s.n_y(), 16);
        assert_eq!(s.n_t_basis(), 11, "7 interior breakpoints + 4");
        assert_eq!(s.len(), 176);
        assert_eq!(s.y_knots().len(), 20);
        assert_eq!(s.t_knots().len(), 15);
        assert_eq!(&s.y_knots()[..4], &[-1.0; 4]);
        assert_eq!(&s.y_knots()[16..], &[0.8; 4]);
        assert_eq!(&s.t_knots()[..4], &[0.0; 4]);
        assert_eq!(&s.t_knots()[11..], &[1.1; 4]);
        assert!(
            (s.t_knots()[4] - 1.0 / 52.0).abs() < 1e-15,
            "first interior t knot"
        );
        let h = 1.8 / 13.0;
        assert!(
            (s.y_knots()[4] - (-1.0 + h)).abs() < 1e-14,
            "first interior y knot"
        );
        assert!(
            (s.y_knots()[5] - (-1.0 + 2.0 * h)).abs() < 1e-14,
            "uniform y knots"
        );
        let (ly, lt) = s.default_lambdas();
        assert!((ly - 1.0 / 3.24).abs() < 1e-14 && (lt - 1.0 / 1.21).abs() < 1e-14);
    }

    #[test]
    fn constructors_reject_bad_inputs() {
        let ok = vec![(0.0, 0.0)];
        assert!(
            BSplineLocalVol::new(3, -1.0, 0.8, &[0.5], 1.0, ok.clone()).is_err(),
            "n_y < 4"
        );
        assert!(
            BSplineLocalVol::new(8, 0.8, -1.0, &[0.5], 1.0, ok.clone()).is_err(),
            "y range"
        );
        assert!(
            BSplineLocalVol::new(8, -1.0, 0.8, &[0.5], 0.0, ok.clone()).is_err(),
            "t_max"
        );
        assert!(
            BSplineLocalVol::new(8, -1.0, 0.8, &[0.5, 0.5], 1.0, ok.clone()).is_err(),
            "dup"
        );
        assert!(
            BSplineLocalVol::new(8, -1.0, 0.8, &[1.5], 1.0, ok.clone()).is_err(),
            "outside"
        );
        assert!(
            BSplineLocalVol::new(8, -1.0, 0.8, &[0.5], 1.0, vec![]).is_err(),
            "empty f_ref"
        );
        assert!(
            BSplineLocalVol::new(8, -1.0, 0.8, &[0.5], 1.0, vec![(0.0, 0.0), (0.0, 1.0)]).is_err(),
            "non-increasing f_ref"
        );
        assert!(BSplineLocalVol::flat(-0.2, 8, -1.0, 0.8, &[0.5], 1.0, ok.clone()).is_err());
        let s = BSplineLocalVol::new(8, -1.0, 0.8, &[0.5], 1.0, ok).unwrap();
        assert_eq!(s.len(), 8 * 5);
        assert!(
            s.clone().with_theta(vec![0.1; 39]).is_err(),
            "wrong theta length"
        );
        assert!(s.with_theta(vec![0.1; 40]).is_ok());
    }

    #[test]
    fn basis_is_a_nonnegative_partition_of_unity_on_and_off_the_domain() {
        let s = surface();
        let (y_min, y_max) = s.y_range();
        let mut rng = Lcg(7);
        for _ in 0..500 {
            let y = y_min - 0.5 + (y_max - y_min + 1.0) * rng.next_f64();
            let (idx, vals) = s.basis_y(y);
            let sum: f64 = vals.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "y partition of unity at {y}: {sum}"
            );
            assert!(
                vals.iter().all(|&v| v >= -1e-16),
                "nonnegative y basis at {y}: {vals:?}"
            );
            assert!(idx[3] < s.n_y(), "index range at {y}: {idx:?}");
            assert!(
                idx.windows(2).all(|w| w[1] == w[0] + 1),
                "consecutive indices"
            );
            let t = -0.1 + 1.4 * rng.next_f64();
            let (it, tv) = s.basis_t(t);
            let sum_t: f64 = tv.iter().sum();
            assert!(
                (sum_t - 1.0).abs() < 1e-14,
                "t partition of unity at {t}: {sum_t}"
            );
            assert!(
                tv.iter().all(|&v| v >= -1e-16),
                "nonnegative t basis at {t}"
            );
            assert!(it[3] < s.n_t_basis(), "t index range at {t}: {it:?}");
        }
        // Knot points, ends, and the exact end values (clamped: B_0(a) = 1).
        for &y in s.y_knots() {
            let (_, v) = s.basis_y(y);
            let sum: f64 = v.iter().sum();
            assert!((sum - 1.0).abs() < 1e-14, "at knot {y}");
        }
        let (i0, v0) = s.basis_y(y_min);
        assert_eq!(i0[0], 0);
        assert!((v0[0] - 1.0).abs() < 1e-15, "B_0(y_min) = 1: {v0:?}");
        let (i1, v1) = s.basis_y(y_max);
        assert_eq!(i1[3], s.n_y() - 1);
        assert!((v1[3] - 1.0).abs() < 1e-15, "B_last(y_max) = 1: {v1:?}");
        // Flat coefficients give a flat surface everywhere.
        for &(x, t) in &[(3.0, 0.0), (4.6, 0.5), (5.9, 1.1), (4.0, 3.0), (4.7, -1.0)] {
            assert!(
                (s.vol(x, t) - 0.2).abs() < 1e-15,
                "flat surface at ({x}, {t})"
            );
        }
    }

    #[test]
    fn basis_derivatives_match_finite_differences_and_sum_to_zero() {
        let s = surface();
        let mut rng = Lcg(11);
        for knots_n in [(s.y_knots(), s.n_y()), (s.t_knots(), s.n_t_basis())] {
            let (knots, n) = knots_n;
            let (a, b) = (knots[0], knots[n]);
            for _ in 0..200 {
                let u = a + (b - a) * (0.01 + 0.98 * rng.next_f64());
                let span = find_span(knots, n, u);
                let (_, d) = basis_funs_with_derivs(knots, span, u);
                let dsum: f64 = d.iter().sum();
                assert!(dsum.abs() < 1e-11, "derivatives sum to zero at {u}: {dsum}");
                let h = 1e-6;
                // Stay inside the same span for the central difference.
                if u - h < knots[span] || u + h > knots[span + 1] {
                    continue;
                }
                let bp = basis_funs(knots, span, u + h);
                let bm = basis_funs(knots, span, u - h);
                for r in 0..4 {
                    let fd = (bp[r] - bm[r]) / (2.0 * h);
                    assert!(
                        (fd - d[r]).abs() < 1e-7,
                        "derivative {r} at {u}: {} vs {fd}",
                        d[r]
                    );
                }
            }
        }
    }

    #[test]
    fn greville_coefficients_reproduce_linear_functions_exactly() {
        let mut s = surface_zero_fref();
        let xi = s.greville_y();
        let tau = s.greville_t();
        assert_eq!(xi.len(), s.n_y());
        assert_eq!(tau.len(), s.n_t_basis());
        let (y_min, y_max) = s.y_range();
        assert!((xi[0] - y_min).abs() < 1e-15 && (xi[s.n_y() - 1] - y_max).abs() < 1e-15);
        assert!((tau[0]).abs() < 1e-15 && (tau[s.n_t_basis() - 1] - s.t_max()).abs() < 1e-15);
        // theta_{(j,k)} = 2 xi_j - 0.3  ->  Sigma = 2 y - 0.3.
        for a in 0..s.n_y() {
            for b in 0..s.n_t_basis() {
                let m = s.index(a, b);
                s.theta[m] = 2.0 * xi[a] - 0.3;
            }
        }
        let mut rng = Lcg(3);
        for _ in 0..300 {
            let y = y_min + (y_max - y_min) * rng.next_f64();
            let t = s.t_max() * rng.next_f64();
            let got = s.vol(y, t);
            assert!(
                (got - (2.0 * y - 0.3)).abs() < 1e-13,
                "linear in y at ({y}, {t}): {got}"
            );
        }
        // theta_{(j,k)} = 0.5 tau_k + 0.1  ->  Sigma = 0.5 t + 0.1.
        for a in 0..s.n_y() {
            for b in 0..s.n_t_basis() {
                let m = s.index(a, b);
                s.theta[m] = 0.5 * tau[b] + 0.1;
            }
        }
        for _ in 0..300 {
            let y = y_min + (y_max - y_min) * rng.next_f64();
            let t = s.t_max() * rng.next_f64();
            let got = s.vol(y, t);
            assert!(
                (got - (0.5 * t + 0.1)).abs() < 1e-13,
                "linear in t at ({y}, {t}): {got}"
            );
        }
    }

    #[test]
    fn one_dimensional_mass_and_stiffness_have_the_exact_sums() {
        let s = surface();
        for (knots, n) in [(s.y_knots(), s.n_y()), (s.t_knots(), s.n_t_basis())] {
            let (m, g) = mass_and_stiffness(knots, n);
            let length = knots[n] - knots[0];
            let total: f64 = m.iter().flatten().sum();
            assert!(
                (total - length).abs() < 1e-13,
                "sum of mass = domain length: {total}"
            );
            for i in 0..n {
                let row: f64 = g[i].iter().sum();
                assert!(row.abs() < 1e-12, "stiffness row {i} sums to zero: {row}");
                for j in 0..n {
                    assert_eq!(m[i][j], m[j][i], "mass symmetry");
                    assert_eq!(g[i][j], g[j][i], "stiffness symmetry");
                    if j + 4 <= i || i + 4 <= j {
                        assert_eq!(m[i][j], 0.0, "mass bandwidth");
                        assert_eq!(g[i][j], 0.0, "stiffness bandwidth");
                    }
                }
                assert!(m[i][i] > 0.0, "positive diagonal");
            }
        }
    }

    /// Quadratic form `theta^T Q theta`.
    fn quad(q: &[Vec<f64>], theta: &[f64]) -> f64 {
        q.iter()
            .zip(theta)
            .map(|(row, &ti)| ti * row.iter().zip(theta).map(|(a, b)| a * b).sum::<f64>())
            .sum()
    }

    #[test]
    fn regularizer_satisfies_the_greville_identities_and_detects_a_swapped_kronecker_order() {
        let s = surface();
        let (lambda_y, lambda_t) = (0.7, 0.3);
        let q = s.regularizer_matrix(lambda_y, lambda_t);
        let m = s.len();
        assert_eq!(q.len(), m);
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (q[i][j] - q[j][i]).abs() < 1e-15,
                    "Q symmetric at ({i}, {j})"
                );
            }
        }
        let (y_min, y_max) = s.y_range();
        let area = (y_max - y_min) * s.t_max();
        let xi = s.greville_y();
        let tau = s.greville_t();
        let (c1, c2) = (1.7, -0.9);
        let mut theta_y = vec![0.0; m];
        let mut theta_t = vec![0.0; m];
        let mut theta_c = vec![0.37; m];
        for a in 0..s.n_y() {
            for b in 0..s.n_t_basis() {
                theta_y[s.index(a, b)] = c1 * xi[a];
                theta_t[s.index(a, b)] = c2 * tau[b];
            }
        }
        let ry = quad(&q, &theta_y);
        let expect_y = lambda_y * c1 * c1 * area;
        assert!(
            ((ry - expect_y) / expect_y).abs() < 1e-11,
            "theta = c1 xi: {ry} vs lambda_y c1^2 area = {expect_y}"
        );
        let rt = quad(&q, &theta_t);
        let expect_t = lambda_t * c2 * c2 * area;
        assert!(
            ((rt - expect_t) / expect_t).abs() < 1e-11,
            "theta = c2 tau: {rt} vs lambda_t c2^2 area = {expect_t}"
        );
        let rc = quad(&q, &theta_c);
        assert!(
            rc.abs() < 1e-12,
            "constant surface has zero H^1 seminorm: {rc}"
        );
        theta_c.iter_mut().zip(&theta_y).for_each(|(c, y)| *c += y);
        let rcy = quad(&q, &theta_c);
        assert!(
            ((rcy - expect_y) / expect_y).abs() < 1e-11,
            "seminorm ignores the constant"
        );

        // A swapped Kronecker order (theta index j_t * n_y + j_y with the
        // same Q) fails both identities when lambda_y != lambda_t.
        let (ny, nt) = (s.n_y(), s.n_t_basis());
        let mut swapped_y = vec![0.0; m];
        let mut swapped_t = vec![0.0; m];
        for a in 0..ny {
            for b in 0..nt {
                swapped_y[b * ny + a] = c1 * xi[a];
                swapped_t[b * ny + a] = c2 * tau[b];
            }
        }
        let sy = quad(&q, &swapped_y);
        let st = quad(&q, &swapped_t);
        assert!(
            ((sy - expect_y) / expect_y).abs() > 1e-2 && ((st - expect_t) / expect_t).abs() > 1e-2,
            "the identities must discriminate the ordering: {sy} vs {expect_y}, {st} vs {expect_t}"
        );
    }

    #[test]
    fn node_field_equals_vol_at_the_nodes_directly_and_through_the_cache() {
        let mut s = surface();
        let mut rng = Lcg(19);
        for v in s.theta.iter_mut() {
            *v = 0.1 + 0.4 * rng.next_f64();
        }
        let x_nodes: Vec<f64> = (0..=80).map(|i| 3.4 + 0.03 * i as f64).collect();
        let t_mid: Vec<f64> = (0..37).map(|n| (n as f64 + 0.5) * 0.9 / 37.0).collect();
        let nf = s.node_field(&x_nodes, &t_mid);
        assert_eq!(nf.stride, 81);
        assert_eq!(nf.steps(), 37);
        assert!((nf.x_min - 3.4).abs() < 1e-15 && (nf.dx - 0.03).abs() < 1e-14);
        let cache = s.basis_cache(&x_nodes, &t_mid);
        assert_eq!(cache.memory_bytes(), 81 * 37 * 40 + 37 * 48);
        let nf2 = s.node_field_from_cache(&cache);
        let mut filled = vec![0.0; 81 * 37];
        cache.fill_node_values(&s.theta, &mut filled);
        for (n, &t) in t_mid.iter().enumerate() {
            for (j, &x) in x_nodes.iter().enumerate() {
                let direct = s.vol(x, t);
                assert!(
                    (nf.at(n, j) - direct).abs() < 1e-14,
                    "node_field at ({n}, {j})"
                );
                assert!(
                    (nf2.at(n, j) - direct).abs() < 1e-14,
                    "cached node field at ({n}, {j})"
                );
                assert!(
                    (filled[n * 81 + j] - direct).abs() < 1e-14,
                    "fill_node_values"
                );
                assert!(
                    (cache.evaluate(&s.theta, n, j) - direct).abs() < 1e-14,
                    "evaluate"
                );
                // The cached pairs are the basis at the shifted, clamped y.
                let y = s.y_of(x, t);
                let (iy, by) = s.basis_y(y);
                let pairs = cache.y_pairs(n, j);
                for r in 0..4 {
                    assert_eq!(pairs[r].0, iy[r]);
                    assert_eq!(pairs[r].1, by[r]);
                }
            }
            let (it, bt) = s.basis_t(t);
            let tp = cache.t_pairs(n);
            for r in 0..4 {
                assert_eq!(tp[r].0, it[r]);
                assert_eq!(tp[r].1, bt[r]);
            }
        }
        // Nodes outside the y domain are clamped, not extrapolated.
        let far = s.vol(9.0, 0.3);
        assert!(
            (far - s.evaluate_yt(s.y_range().1, 0.3)).abs() < 1e-15,
            "clamped wing"
        );
    }

    #[test]
    fn project_equals_the_direct_double_sum_and_is_the_gradient_of_the_pairing() {
        let mut s = surface();
        let mut rng = Lcg(23);
        for v in s.theta.iter_mut() {
            *v = 0.1 + 0.4 * rng.next_f64();
        }
        let x_nodes: Vec<f64> = (0..=60).map(|i| 3.5 + 0.04 * i as f64).collect();
        let t_mid: Vec<f64> = (0..29).map(|n| (n as f64 + 0.5) * 0.7 / 29.0).collect();
        let cache = s.basis_cache(&x_nodes, &t_mid);
        let g: Vec<f64> = (0..61 * 29).map(|_| rng.next_f64() - 0.5).collect();
        let proj = s.project(&g, &cache);
        assert_eq!(proj.len(), s.len());
        // Direct sum over (node, step) of g B_m(y_{j,n}, t_n).
        let mut direct = vec![0.0; s.len()];
        for (n, &t) in t_mid.iter().enumerate() {
            let (it, bt) = s.basis_t(t);
            for (j, &x) in x_nodes.iter().enumerate() {
                let (iy, by) = s.basis_y(s.y_of(x, t));
                for a in 0..4 {
                    for b in 0..4 {
                        direct[s.index(iy[a], it[b])] += g[n * 61 + j] * by[a] * bt[b];
                    }
                }
            }
        }
        let scale = max_abs(&direct).max(1.0);
        for m in 0..s.len() {
            assert!(
                (proj[m] - direct[m]).abs() < 1e-12 * scale,
                "project vs direct sum at {m}: {} vs {}",
                proj[m],
                direct[m]
            );
        }
        // <g, Sigma_theta at nodes> is linear in theta with coefficients proj.
        let nf = cache.node_field(&s.theta);
        let pairing: f64 = g.iter().zip(&nf.values).map(|(a, b)| a * b).sum();
        let via_proj: f64 = proj.iter().zip(&s.theta).map(|(a, b)| a * b).sum();
        assert!(
            (pairing - via_proj).abs() < 1e-11,
            "chain rule: {pairing} vs {via_proj}"
        );
        // Allocation-free form agrees and tolerates a longer scratch.
        let mut out = vec![1.0; s.len()];
        let mut scratch = vec![0.0; s.n_y() + 3];
        cache.project_into(&g, &mut out, &mut scratch);
        assert!(
            out.iter().zip(&proj).all(|(a, b)| a == b),
            "project_into == project"
        );
    }

    #[test]
    fn projection_of_recovers_a_surface_inside_the_span() {
        let mut truth = surface();
        let mut rng = Lcg(31);
        for v in truth.theta.iter_mut() {
            *v = 0.15 + 0.3 * rng.next_f64();
        }
        let basis = surface();
        let theta = basis.projection_of(&truth, None).unwrap();
        let err = theta
            .iter()
            .zip(&truth.theta)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err < 1e-10,
            "projection recovers the coefficients: max error {err}"
        );
        // A linear-in-y truth expressed as a callback in x with the same
        // forward: recovered to the Greville coefficients.
        let f_ref = basis.f_ref().to_vec();
        let cb = CallbackVol::new(move |x, t| 0.2 + 0.1 * (x - interp_table(&f_ref, t)));
        let theta_lin = basis.projection_of(&cb, None).unwrap();
        let xi = basis.greville_y();
        for a in 0..basis.n_y() {
            for b in 0..basis.n_t_basis() {
                let got = theta_lin[basis.index(a, b)];
                let want = 0.2 + 0.1 * xi[a];
                assert!(
                    (got - want).abs() < 1e-10,
                    "linear truth at ({a}, {b}): {got} vs {want}"
                );
            }
        }
        // A truth defined in a different forward coordinate: pass its table.
        let other_fref = vec![(0.0, 1.0), (1.1, 1.05)];
        let of = other_fref.clone();
        let cb2 = CallbackVol::new(move |x, t| 0.3 - 0.05 * (x - interp_table(&of, t)));
        let theta2 = basis.projection_of(&cb2, Some(&other_fref)).unwrap();
        for a in 0..basis.n_y() {
            let got = theta2[basis.index(a, 2)];
            let want = 0.3 - 0.05 * xi[a];
            assert!(
                (got - want).abs() < 1e-10,
                "foreign forward at {a}: {got} vs {want}"
            );
        }
        // Outside the span the projection is the least-squares fit: its
        // error is not zero but small for a smooth surface.
        let bump = CallbackVol::new(|x, t| 0.2 + 0.05 * (-(x * x) * 4.0).exp() * (1.0 + t));
        let zero = surface_zero_fref();
        let theta_b = zero.projection_of(&bump, None).unwrap();
        let fit = zero.clone().with_theta(theta_b).unwrap();
        let mut worst = 0.0_f64;
        for i in 0..=60 {
            let y = -1.0 + 1.8 * i as f64 / 60.0;
            for k in 0..=20 {
                let t = 1.1 * k as f64 / 20.0;
                worst = worst.max((fit.vol(y, t) - bump.vol(y, t)).abs());
            }
        }
        assert!(worst < 2e-3, "basis floor of a smooth bump: {worst}");
        assert!(
            worst > 1e-9,
            "the bump is not in the span (floor is a real number): {worst}"
        );
        // Non-finite fields are rejected.
        let bad = CallbackVol::new(|_, _| f64::NAN);
        assert!(
            basis.projection_of(&bad, None).is_err(),
            "NaN field rejected"
        );
        assert!(
            basis.projection_of(&truth, Some(&[])).is_err(),
            "empty f_ref rejected"
        );
    }

    #[test]
    fn bounds_projection_clamps_coefficients() {
        let mut s = surface();
        assert_eq!(s.bounds, (SIGMA_MIN, SIGMA_MAX));
        s.theta[0] = -1.0;
        s.theta[1] = 10.0;
        s.theta[2] = 0.5;
        s.clamp_theta();
        assert_eq!(s.theta[0], SIGMA_MIN);
        assert_eq!(s.theta[1], SIGMA_MAX);
        assert_eq!(s.theta[2], 0.5);
        let mut v = vec![0.0, 0.5, 1.5];
        project_onto_bounds(&mut v, 0.1, 1.0);
        assert_eq!(v, vec![0.1, 0.5, 1.0]);
        // Bounded coefficients give a bounded surface (convex hull).
        let mut rng = Lcg(5);
        for x in s.theta.iter_mut() {
            *x = if rng.next_f64() < 0.5 {
                SIGMA_MIN
            } else {
                SIGMA_MAX
            };
        }
        for i in 0..50 {
            let v = s.vol(3.0 + 0.06 * i as f64, 0.02 * i as f64);
            assert!(
                v >= SIGMA_MIN - 1e-12 && v <= SIGMA_MAX + 1e-12,
                "surface inside the coefficient box (to round-off): {v}"
            );
        }
    }

    #[test]
    fn forward_table_interpolates_and_extrapolates_linearly() {
        let s = surface();
        assert!((s.ln_f_ref(0.0) - 4.6).abs() < 1e-15);
        assert!(
            (s.ln_f_ref(0.25) - 4.605).abs() < 1e-15,
            "interior interpolation"
        );
        assert!((s.ln_f_ref(0.8) - 4.616).abs() < 1e-14, "second segment");
        assert!(
            (s.ln_f_ref(1.6) - 4.632).abs() < 1e-14,
            "linear extrapolation beyond the end"
        );
        assert!(
            (s.ln_f_ref(-0.5) - 4.59).abs() < 1e-14,
            "linear extrapolation before the start"
        );
        assert!((s.y_of(4.6 + 0.3, 0.0) - 0.3).abs() < 1e-15);
        assert_eq!(s.y_of(-10.0, 0.0), -1.0, "clamped below");
        assert_eq!(s.y_of(10.0, 0.0), 0.8, "clamped above");
        let single = BSplineLocalVol::new(8, -1.0, 1.0, &[0.5], 1.0, vec![(0.3, 2.0)]).unwrap();
        assert_eq!(single.ln_f_ref(-1.0), 2.0);
        assert_eq!(single.ln_f_ref(5.0), 2.0);
    }
}
