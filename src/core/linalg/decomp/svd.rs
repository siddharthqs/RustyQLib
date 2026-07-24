//! Singular value decomposition by one-sided Jacobi rotations — simple,
//! accurate for the small-to-moderate matrices of quant workflows, and
//! rank-revealing. `A = U diag(S) V^T` with orthonormal `U` (m x n,
//! columns for nonzero singular values), non-negative `S` sorted
//! descending, and orthogonal `V` (n x n).

/// SVD of an `m x n` matrix (any shape; internally transposes when
/// `m < n`). Returns `(u, s, v)` with `A = U diag(S) V^T`.
pub fn svd(a: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
    let m = a.len();
    let n = if m == 0 { 0 } else { a[0].len() };
    assert!(m > 0 && n > 0, "empty matrix");
    assert!(a.iter().all(|row| row.len() == n), "ragged matrix");
    if m < n {
        // A^T = U' S V'^T  =>  A = V' S U'^T
        let at: Vec<Vec<f64>> = (0..n).map(|j| (0..m).map(|i| a[i][j]).collect()).collect();
        let (u_t, s, v_t) = svd(&at);
        return (v_t, s, u_t);
    }

    // one-sided Jacobi: orthogonalize the columns of B = A V
    let mut b = a.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..60 {
        let mut off = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let mut alpha = 0.0;
                let mut beta = 0.0;
                let mut g = 0.0;
                for row in &b {
                    alpha += row[i] * row[i];
                    beta += row[j] * row[j];
                    g += row[i] * row[j];
                }
                if g.abs() <= 1e-14 * (alpha * beta).sqrt().max(1e-300) {
                    continue;
                }
                off = off.max(g.abs());
                let zeta = (beta - alpha) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for row in b.iter_mut() {
                    let (bi, bj) = (row[i], row[j]);
                    row[i] = c * bi - s * bj;
                    row[j] = s * bi + c * bj;
                }
                for row in v.iter_mut() {
                    let (vi, vj) = (row[i], row[j]);
                    row[i] = c * vi - s * vj;
                    row[j] = s * vi + c * vj;
                }
            }
        }
        if off == 0.0 {
            break;
        }
    }

    // singular values = column norms; U = normalized columns
    let mut s: Vec<f64> = (0..n)
        .map(|j| b.iter().map(|row| row[j] * row[j]).sum::<f64>().sqrt())
        .collect();
    let mut u = vec![vec![0.0; n]; m];
    for j in 0..n {
        if s[j] > 1e-300 {
            for i in 0..m {
                u[i][j] = b[i][j] / s[j];
            }
        }
    }
    // sort descending, permuting U and V columns alongside
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[j].total_cmp(&s[i]));
    let s_sorted: Vec<f64> = order.iter().map(|&k| s[k]).collect();
    let permute = |mat: &[Vec<f64>]| -> Vec<Vec<f64>> {
        mat.iter().map(|row| order.iter().map(|&k| row[k]).collect()).collect()
    };
    let (u, v) = (permute(&u), permute(&v));
    s = s_sorted;
    (u, s, v)
}

/// Minimum-norm least-squares solve `A x ~ b` through the SVD
/// pseudo-inverse, dropping singular values below `tol * s_max` — the
/// robust choice for rank-deficient or ill-conditioned systems.
pub fn pseudo_solve(a: &[Vec<f64>], b: &[f64], tol: f64) -> Vec<f64> {
    let (u, s, v) = svd(a);
    let m = a.len();
    let n = s.len();
    assert_eq!(b.len(), m, "dimension mismatch");
    let cutoff = tol * s.first().copied().unwrap_or(0.0);
    let mut x = vec![0.0; n];
    for k in 0..n {
        if s[k] <= cutoff || s[k] == 0.0 {
            continue;
        }
        let utb: f64 = (0..m).map(|i| u[i][k] * b[i]).sum();
        let coeff = utb / s[k];
        for (j, xj) in x.iter_mut().enumerate() {
            *xj += coeff * v[j][k];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::super::qr::least_squares;
    use super::*;

    fn fixture() -> Vec<Vec<f64>> {
        (0..5)
            .map(|i| (0..3).map(|j| ((2 * i + 3 * j) as f64).cos() + if i == j { 1.5 } else { 0.0 }).collect())
            .collect()
    }

    fn reconstruct(u: &[Vec<f64>], s: &[f64], v: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (m, n) = (u.len(), s.len());
        (0..m)
            .map(|i| {
                (0..v.len())
                    .map(|j| (0..n).map(|k| u[i][k] * s[k] * v[j][k]).sum())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn decomposition_reconstructs_and_is_orthogonal() {
        let a = fixture();
        let (u, s, v) = svd(&a);
        let recon = reconstruct(&u, &s, &v);
        for i in 0..5 {
            for j in 0..3 {
                assert!((recon[i][j] - a[i][j]).abs() < 1e-10, "[{i}][{j}]");
            }
        }
        // descending non-negative singular values
        assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0));
        // orthonormal columns
        for j1 in 0..3 {
            for j2 in 0..3 {
                let want = if j1 == j2 { 1.0 } else { 0.0 };
                let uu: f64 = (0..5).map(|i| u[i][j1] * u[i][j2]).sum();
                let vv: f64 = (0..3).map(|i| v[i][j1] * v[i][j2]).sum();
                assert!((uu - want).abs() < 1e-11, "U [{j1}][{j2}]");
                assert!((vv - want).abs() < 1e-11, "V [{j1}][{j2}]");
            }
        }
    }

    #[test]
    fn known_singular_values_of_a_diagonal_matrix() {
        let (_, s, _) = svd(&[vec![2.0, 0.0], vec![0.0, -3.0]]);
        assert!((s[0] - 3.0).abs() < 1e-12 && (s[1] - 2.0).abs() < 1e-12, "{s:?}");
    }

    #[test]
    fn rank_deficiency_is_revealed_and_wide_matrices_work() {
        // rank-1: second row is a multiple of the first; also test m < n
        let a = vec![vec![1.0, 2.0, 3.0], vec![2.0, 4.0, 6.0]];
        let (u, s, v) = svd(&a);
        assert!(s[0] > 1.0 && s[1].abs() < 1e-10, "{s:?}");
        let recon = reconstruct(&u, &s, &v);
        for i in 0..2 {
            for j in 0..3 {
                assert!((recon[i][j] - a[i][j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn pseudo_solve_matches_qr_on_full_rank_and_handles_deficiency() {
        let a = fixture();
        let b = [1.0, -2.0, 0.5, 3.0, -1.0];
        let via_qr = least_squares(&a, &b).unwrap();
        let via_svd = pseudo_solve(&a, &b, 1e-12);
        for (x, y) in via_qr.iter().zip(&via_svd) {
            assert!((x - y).abs() < 1e-10, "{via_qr:?} vs {via_svd:?}");
        }
        // rank-deficient: QR errors, the pseudo-inverse returns the
        // minimum-norm solution
        let deficient: Vec<Vec<f64>> =
            (0..4).map(|i| vec![i as f64 + 1.0, 2.0 * (i as f64 + 1.0)]).collect();
        let rhs = [1.0, 2.0, 3.0, 4.0];
        assert!(least_squares(&deficient, &rhs).is_err());
        let x = pseudo_solve(&deficient, &rhs, 1e-12);
        // x must satisfy the normal equations projected on the range:
        // residual orthogonal to the columns
        for j in 0..2 {
            let r: f64 = (0..4)
                .map(|i| {
                    let ax: f64 = (0..2).map(|k| deficient[i][k] * x[k]).sum();
                    deficient[i][j] * (ax - rhs[i])
                })
                .sum();
            assert!(r.abs() < 1e-10, "column {j} residual {r}");
        }
        // and among solutions it is minimum norm: x parallel to (1, 2)
        assert!((x[1] - 2.0 * x[0]).abs() < 1e-10, "{x:?}");
    }
}
