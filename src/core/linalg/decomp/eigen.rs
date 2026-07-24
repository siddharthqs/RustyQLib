//! Eigendecomposition of symmetric matrices by the cyclic Jacobi
//! method — robust and accurate for the small matrices of correlation
//! and covariance work (the PSD projection inside Higham's
//! nearest-correlation algorithm runs on this).

/// Eigendecomposition of a symmetric matrix: returns
/// `(eigenvalues, eigenvectors)` with the eigenvectors in the columns
/// (`A = V diag(vals) V^T`). Order is unspecified.
pub fn symmetric_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
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
    use super::*;

    #[test]
    fn reproduces_known_eigenvalues() {
        // [[2,1],[1,2]] has eigenvalues 1 and 3
        let (mut vals, _) = symmetric_eigen(&[vec![2.0, 1.0], vec![1.0, 2.0]]);
        vals.sort_by(f64::total_cmp);
        assert!((vals[0] - 1.0).abs() < 1e-10 && (vals[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn reconstructs_the_matrix() {
        let a = vec![
            vec![3.0, 1.0, 0.5],
            vec![1.0, 2.0, -0.4],
            vec![0.5, -0.4, 1.5],
        ];
        let (vals, vecs) = symmetric_eigen(&a);
        for i in 0..3 {
            for j in 0..3 {
                let recon: f64 = (0..3).map(|k| vecs[i][k] * vals[k] * vecs[j][k]).sum();
                assert!((recon - a[i][j]).abs() < 1e-10, "[{i}][{j}]");
            }
        }
    }
}
