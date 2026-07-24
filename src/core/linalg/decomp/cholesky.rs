//! Cholesky factorization `A = L L^T` of a symmetric positive
//! (semi-)definite matrix, and the SPD linear solve through the factor.

/// Lower-triangular Cholesky factor of a symmetric PSD matrix.
///
/// Semi-definite inputs are tolerated (zero pivots produce zero
/// columns); an indefinite matrix returns `Err`. Symmetry is the
/// caller's contract and is checked.
pub fn cholesky_factor(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    for i in 0..n {
        if a[i].len() != n {
            return Err("matrix must be square".to_string());
        }
        for j in 0..i {
            if (a[i][j] - a[j][i]).abs() > 1e-10 * (1.0 + a[i][j].abs()) {
                return Err("matrix must be symmetric".to_string());
            }
        }
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = a[i][i] - s;
                if d < -1e-10 * (1.0 + a[i][i].abs()) {
                    return Err("matrix is not positive semi-definite".to_string());
                }
                l[i][j] = d.max(0.0).sqrt();
            } else if l[j][j] > 1e-12 {
                l[i][j] = (a[i][j] - s) / l[j][j];
            } else {
                l[i][j] = 0.0;
            }
        }
    }
    Ok(l)
}

/// Solve `A x = b` given the Cholesky factor `L` of `A` (forward then
/// backward substitution). Errs when the factor is singular (a zero
/// pivot from a semi-definite matrix).
pub fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, String> {
    let n = b.len();
    if l.len() != n {
        return Err("dimension mismatch".to_string());
    }
    if (0..n).any(|i| l[i][i].abs() < 1e-300) {
        return Err("factor is singular".to_string());
    }
    // L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i][k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i][i];
    }
    // L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k][i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factor_reconstructs_the_matrix() {
        let a = vec![
            vec![4.0, 2.0, 0.6],
            vec![2.0, 5.0, 1.5],
            vec![0.6, 1.5, 2.0],
        ];
        let l = cholesky_factor(&a).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let recon: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
                assert!((recon - a[i][j]).abs() < 1e-12, "[{i}][{j}]");
            }
            // lower triangular
            for j in i + 1..3 {
                assert_eq!(l[i][j], 0.0);
            }
        }
    }

    #[test]
    fn solve_inverts_the_system() {
        let a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let l = cholesky_factor(&a).unwrap();
        let x = cholesky_solve(&l, &[1.0, 2.0]).unwrap();
        // matches the known solution [1/11, 7/11]
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-13);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-13);
    }

    #[test]
    fn rejects_indefinite_and_tolerates_semi_definite() {
        // indefinite
        assert!(cholesky_factor(&[vec![1.0, 2.0], vec![2.0, 1.0]]).is_err());
        // rank-1 PSD factors, but cannot be solved through
        let semi = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let l = cholesky_factor(&semi).unwrap();
        assert!(cholesky_solve(&l, &[1.0, 1.0]).is_err());
        // asymmetry rejected
        assert!(cholesky_factor(&[vec![1.0, 0.5], vec![0.4, 1.0]]).is_err());
    }
}
