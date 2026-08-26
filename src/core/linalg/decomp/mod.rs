//! Matrix decompositions, one per file — general-purpose kernels for
//! future use across the library (regression, calibration, PCA, factor
//! models), independent of any finance semantics.
//!
//! - [`cholesky`]: `A = L L^T` for symmetric PSD matrices, plus the SPD
//!   linear solve through the factor — the fast path for normal
//!   equations and covariance sampling;
//! - [`qr`]: Householder QR (`A = Q R`, thin form), plus numerically
//!   stable linear least squares — the right tool for regression
//!   (e.g. Longstaff-Schwartz bases) without forming `A^T A`;
//! - [`svd`]: one-sided Jacobi singular value decomposition
//!   (`A = U S V^T`), plus the minimum-norm pseudo-inverse solve for
//!   rank-deficient problems;
//! - [`eigen`]: cyclic Jacobi eigendecomposition of symmetric matrices —
//!   the engine behind the PSD projection in
//!   [`nearest_correlation`](super::nearest_correlation).
//!
//! All matrices are `Vec<Vec<f64>>` row-major, matching the rest of the
//! crate; the implementations favor clarity and robustness on the
//! small-to-moderate sizes quant workflows use.

pub mod cholesky;
pub mod eigen;
pub mod qr;
pub mod svd;

pub use cholesky::{cholesky_factor, cholesky_solve};
pub use eigen::symmetric_eigen;
pub use qr::{least_squares, qr};
pub use svd::{pseudo_solve, svd};

/// Jacobi rotation parameters `(c, s)` annihilating the off-diagonal
/// entry of the symmetric 2x2 pivot `[[a_pp, a_pq], [a_pq, a_qq]]` —
/// the standard stable formulas, taking the smaller root for `t` so the
/// rotation angle stays within `[-pi/4, pi/4]`. Shared by the two-sided
/// Jacobi sweep in [`eigen`] (pivot entries of the matrix itself) and
/// the one-sided sweep in [`svd`] (entries of the implicit Gram matrix
/// of columns). Callers guard against `a_pq == 0`.
pub(crate) fn jacobi_rotation(a_pp: f64, a_qq: f64, a_pq: f64) -> (f64, f64) {
    let tau = (a_qq - a_pp) / (2.0 * a_pq);
    let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
    let c = 1.0 / (1.0 + t * t).sqrt();
    (c, c * t)
}
