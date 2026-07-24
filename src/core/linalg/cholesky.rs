//! Cholesky factorization of correlation matrices (PSD-tolerant): the
//! unit-diagonal validation layer over the general factorization in
//! [`decomp::cholesky`](super::decomp::cholesky).

use super::decomp::cholesky::cholesky_factor;
use crate::core::errors::RustyQLibError;

/// Cholesky decomposition of a correlation matrix; Err if the matrix is
/// not symmetric positive **semi**-definite with a unit diagonal
/// (perfectly correlated assets are allowed).
pub fn cholesky(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, RustyQLibError> {
    for (i, row) in m.iter().enumerate() {
        if (row[i] - 1.0).abs() > 1e-10 {
            return Err(RustyQLibError::NumericalError(format!("diagonal element [{i}][{i}] must be 1")));
        }
    }
    cholesky_factor(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorizes_and_rejects_correctly() {
        // valid 2x2
        let l = cholesky(&[vec![1.0, 0.5], vec![0.5, 1.0]]).unwrap();
        assert!((l[1][0] - 0.5).abs() < 1e-12 && (l[1][1] - 0.75_f64.sqrt()).abs() < 1e-12);
        // asymmetric / bad diagonal / non-PSD all rejected
        assert!(cholesky(&[vec![1.0, 0.5], vec![0.4, 1.0]]).is_err());
        assert!(cholesky(&[vec![2.0, 0.0], vec![0.0, 1.0]]).is_err());
        assert!(cholesky(&[
            vec![1.0, 0.9, -0.9],
            vec![0.9, 1.0, 0.9],
            vec![-0.9, 0.9, 1.0]
        ])
        .is_err());
    }
}
