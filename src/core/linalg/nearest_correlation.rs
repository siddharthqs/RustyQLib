//! Higham's alternating-projections algorithm for the nearest
//! correlation matrix (Higham, *IMA J. Numer. Anal.* 2002).
//!
//! Empirical or hand-stressed correlation matrices are often not
//! positive semi-definite. This computes the nearest true correlation
//! matrix in the Frobenius norm by alternating projections onto
//!
//! - the PSD cone (eigendecomposition, negative eigenvalues clipped),
//! - the unit-diagonal affine subspace,
//!
//! with Dykstra's correction so the iteration converges to the actual
//! nearest point of the intersection rather than just any point in it.
//! The eigendecompositions use a cyclic Jacobi method — exactly the
//! right tool for the small symmetric matrices of basket products.

/// The nearest correlation matrix to symmetric `a` (unit diagonal, PSD).
///
/// `tol` bounds the change between successive iterates (1e-10 is plenty)
/// and `max_iter` caps the projection rounds (typically converges in
/// tens). The result has an exact unit diagonal and eigenvalues
/// `>= -1e-12`.
pub fn nearest_correlation(a: &[Vec<f64>], tol: f64, max_iter: usize) -> Result<Vec<Vec<f64>>, String> {
    let n = a.len();
    if a.iter().any(|row| row.len() != n) {
        return Err("matrix must be square".into());
    }
    for i in 0..n {
        for j in 0..n {
            if (a[i][j] - a[j][i]).abs() > 1e-8 {
                return Err("matrix must be symmetric".into());
            }
        }
    }

    let mut y = a.to_vec();
    // symmetrize exactly to kill representational asymmetry
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (y[i][j] + y[j][i]);
            y[i][j] = s;
            y[j][i] = s;
        }
    }
    let mut dykstra = vec![vec![0.0; n]; n];
    for _ in 0..max_iter {
        // PSD projection of the Dykstra-corrected iterate
        let mut r = y.clone();
        for i in 0..n {
            for j in 0..n {
                r[i][j] -= dykstra[i][j];
            }
        }
        let x = psd_projection(&r);
        for i in 0..n {
            for j in 0..n {
                dykstra[i][j] = x[i][j] - r[i][j];
            }
        }
        // unit-diagonal projection
        let mut y_next = x.clone();
        for (i, row) in y_next.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let delta = max_abs_diff(&y_next, &y);
        y = y_next;
        if delta <= tol {
            break;
        }
    }
    // final polish: guarantee PSD-ness of the returned matrix, then pin
    // the diagonal and clamp into [-1, 1]
    let mut out = psd_projection(&y);
    for i in 0..n {
        out[i][i] = 1.0;
        for j in 0..n {
            if i != j {
                out[i][j] = out[i][j].clamp(-1.0, 1.0);
            }
        }
    }
    Ok(out)
}

fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

/// Projection onto the PSD cone: eigendecompose, clip negative
/// eigenvalues to zero, recompose.
fn psd_projection(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let (mut vals, vecs) = jacobi_eigen(a);
    for v in vals.iter_mut() {
        *v = v.max(0.0);
    }
    // A+ = V diag(vals+) V^T
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for (k, &val) in vals.iter().enumerate() {
                s += vecs[i][k] * val * vecs[j][k];
            }
            out[i][j] = s;
            out[j][i] = s;
        }
    }
    out
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix: returns
/// `(eigenvalues, eigenvectors)` with eigenvectors in the columns.
fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
            .map(|(i, j)| m[i][j] * m[i][j])
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if m[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k][p], m[k][q]);
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p][k], m[q][k]);
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                for row in v.iter_mut() {
                    let (vp, vq) = (row[p], row[q]);
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i][i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::super::cholesky::cholesky;
    use super::*;

    fn eigenvalues(a: &[Vec<f64>]) -> Vec<f64> {
        jacobi_eigen(a).0
    }

    #[test]
    fn higham_2002_example_matches_the_published_answer() {
        // the worked example from Higham's paper: nearest correlation
        // matrix to [[1,1,0],[1,1,1],[0,1,1]] has off-diagonals
        // 0.7607 and 0.1573
        let a = vec![vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 1.0], vec![0.0, 1.0, 1.0]];
        let x = nearest_correlation(&a, 1e-12, 200).unwrap();
        assert!((x[0][1] - 0.7607).abs() < 1e-3, "x01 {}", x[0][1]);
        assert!((x[1][2] - 0.7607).abs() < 1e-3, "x12 {}", x[1][2]);
        assert!((x[0][2] - 0.1573).abs() < 1e-3, "x02 {}", x[0][2]);
    }

    #[test]
    fn output_is_a_valid_correlation_matrix() {
        let a = vec![
            vec![1.0, 0.9, -0.9],
            vec![0.9, 1.0, 0.9],
            vec![-0.9, 0.9, 1.0],
        ];
        assert!(eigenvalues(&a).iter().any(|&v| v < -1e-6), "test input should be indefinite");
        let x = nearest_correlation(&a, 1e-12, 200).unwrap();
        for i in 0..3 {
            assert_eq!(x[i][i], 1.0);
            for j in 0..3 {
                assert!((x[i][j] - x[j][i]).abs() < 1e-12);
                assert!(x[i][j].abs() <= 1.0 + 1e-12);
            }
        }
        assert!(eigenvalues(&x).iter().all(|&v| v >= -1e-10), "not PSD: {:?}", eigenvalues(&x));
        // and it now factorizes for simulation
        assert!(cholesky(&x).is_ok());
    }

    #[test]
    fn valid_matrices_pass_through_unchanged() {
        let a = vec![vec![1.0, 0.3, 0.1], vec![0.3, 1.0, 0.2], vec![0.1, 0.2, 1.0]];
        let x = nearest_correlation(&a, 1e-12, 200).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!((x[i][j] - a[i][j]).abs() < 1e-8, "[{i}][{j}]");
            }
        }
    }

    #[test]
    fn jacobi_reproduces_known_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3
        let mut vals = eigenvalues(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        vals.sort_by(f64::total_cmp);
        assert!((vals[0] - 1.0).abs() < 1e-10 && (vals[1] - 3.0).abs() < 1e-10);
    }
}
