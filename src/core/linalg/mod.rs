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
