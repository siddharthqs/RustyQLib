//! Linear algebra utilities: matrix decompositions and correlation
//! handling.
//!
//! [`decomp`] holds the general-purpose factorizations (Cholesky, QR,
//! SVD, symmetric eigen). On top of them sit the correlation tools:
//! empirical correlation matrices (estimated pairwise, stressed by hand,
//! or copied from a term sheet) frequently fail positive
//! semi-definiteness; [`nearest_correlation`](nearest_correlation::nearest_correlation)
//! repairs them with the alternating-projections algorithm of Higham
//! (2002), and [`cholesky`](cholesky::cholesky) factorizes the result for
//! correlated simulation.

pub mod cholesky;
pub mod decomp;
pub mod nearest_correlation;

pub use cholesky::cholesky;
pub use decomp::{
    cholesky_factor, cholesky_solve, least_squares, pseudo_solve, qr, svd, symmetric_eigen,
};
pub use nearest_correlation::nearest_correlation;

/// Cholesky factor of a correlation matrix, repairing a non-PSD input
/// with Higham's nearest-correlation projection (logged) — the standard
/// treatment for empirical or hand-stressed matrices. Asymmetry or a
/// non-unit diagonal is a data error and still rejected.
pub fn cholesky_with_repair(
    correlations: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, crate::core::errors::RustyQLibError> {
    match cholesky(correlations) {
        Ok(l) => Ok(l),
        Err(crate::core::errors::RustyQLibError::NumericalError(ref msg))
            if msg.contains("positive semi-definite") =>
        {
            log::warn!(
                "correlation matrix is not PSD; \
                 projecting to the nearest correlation matrix (Higham)"
            );
            let repaired = nearest_correlation(correlations, 1e-12, 200)?;
            cholesky(&repaired)
        }
        Err(e) => Err(e),
    }
}
