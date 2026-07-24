//! Matrix utilities for correlation handling: Cholesky factorization and
//! Higham's nearest-correlation-matrix projection.
//!
//! Empirical correlation matrices (estimated pairwise, stressed by hand,
//! or copied from a term sheet) frequently fail positive
//! semi-definiteness; [`nearest_correlation`](nearest_correlation::nearest_correlation)
//! repairs them with the alternating-projections algorithm of Higham
//! (2002), and [`cholesky`](cholesky::cholesky) factorizes the result for
//! correlated simulation.

pub mod cholesky;
pub mod nearest_correlation;

pub use cholesky::cholesky;
pub use nearest_correlation::nearest_correlation;
