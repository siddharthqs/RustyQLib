//! Householder QR decomposition and numerically stable linear least
//! squares (no `A^T A` squaring of the condition number).
use crate::core::errors::RustyQLibError;

/// Thin QR of an `m x n` matrix with `m >= n`: returns `(Q, R)` with `Q`
/// an `m x n` orthonormal-column matrix and `R` upper triangular `n x n`
/// such that `A = Q R`.
pub fn qr(a: &[Vec<f64>]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), RustyQLibError> {
    let m = a.len();
    if m == 0 {
        return Err(RustyQLibError::NumericalError("empty matrix".to_string()));
    }
    let n = a[0].len();
    if a.iter().any(|row| row.len() != n) {
        return Err(RustyQLibError::NumericalError("ragged matrix".to_string()));
    }
    if m < n {
        return Err(RustyQLibError::NumericalError("QR needs at least as many rows as columns".to_string()));
    }
    let mut r = a.to_vec();
    // householder vectors, v[k] has zeros above row k
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(n);
    for k in 0..n {
        let norm: f64 = (k..m).map(|i| r[i][k] * r[i][k]).sum::<f64>().sqrt();
        let mut v = vec![0.0; m];
        if norm < 1e-300 {
            vs.push(v);
            continue;
        }
        let alpha = if r[k][k] >= 0.0 { -norm } else { norm };
        for (i, vi) in v.iter_mut().enumerate().take(m).skip(k) {
            *vi = r[i][k];
        }
        v[k] -= alpha;
        let v_norm_sq: f64 = v[k..].iter().map(|x| x * x).sum();
        if v_norm_sq < 1e-300 {
            vs.push(vec![0.0; m]);
            continue;
        }
        for j in k..n {
            let dot: f64 = (k..m).map(|i| v[i] * r[i][j]).sum();
            let f = 2.0 * dot / v_norm_sq;
            for i in k..m {
                r[i][j] -= f * v[i];
            }
        }
        vs.push(v);
    }
    // thin Q: reflectors applied in reverse to the first n identity columns
    let mut q = vec![vec![0.0; n]; m];
    for j in 0..n {
        let mut e = vec![0.0; m];
        e[j] = 1.0;
        for v in vs.iter().rev() {
            let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
            if v_norm_sq < 1e-300 {
                continue;
            }
            let dot: f64 = v.iter().zip(&e).map(|(a, b)| a * b).sum();
            let f = 2.0 * dot / v_norm_sq;
            for i in 0..m {
                e[i] -= f * v[i];
            }
        }
        for (i, row) in q.iter_mut().enumerate() {
            row[j] = e[i];
        }
    }
    let r_top: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if j >= i { r[i][j] } else { 0.0 }).collect())
        .collect();
    Ok((q, r_top))
}

/// Least-squares solution of the overdetermined system `A x ~ b` via QR:
/// `x = R^{-1} Q^T b`. Errs on rank deficiency (use
/// [`pseudo_solve`](super::svd::pseudo_solve) there).
pub fn least_squares(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, RustyQLibError> {
    let (q, r) = qr(a)?;
    let m = a.len();
    let n = r.len();
    if b.len() != m {
        return Err(RustyQLibError::NumericalError("dimension mismatch".to_string()));
    }
    // Q^T b
    let qtb: Vec<f64> = (0..n).map(|j| (0..m).map(|i| q[i][j] * b[i]).sum()).collect();
    // back-substitute R x = Q^T b
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        if r[i][i].abs() < 1e-12 {
            return Err(RustyQLibError::NumericalError("matrix is rank deficient".to_string()));
        }
        let s: f64 = (i + 1..n).map(|k| r[i][k] * x[k]).sum();
        x[i] = (qtb[i] - s) / r[i][i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Vec<Vec<f64>> {
        // deterministic 5 x 3 full-rank matrix
        (0..5)
            .map(|i| (0..3).map(|j| ((i * 3 + j) as f64).sin() + if i == j { 2.0 } else { 0.0 }).collect())
            .collect()
    }

    #[test]
    fn q_is_orthonormal_and_reconstructs_a() {
        let a = fixture();
        let (q, r) = qr(&a).unwrap();
        for j1 in 0..3 {
            for j2 in 0..3 {
                let dot: f64 = (0..5).map(|i| q[i][j1] * q[i][j2]).sum();
                let want = if j1 == j2 { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-12, "Q^T Q [{j1}][{j2}] = {dot}");
            }
        }
        for i in 0..5 {
            for j in 0..3 {
                let recon: f64 = (0..3).map(|k| q[i][k] * r[k][j]).sum();
                assert!((recon - a[i][j]).abs() < 1e-12, "QR [{i}][{j}]");
            }
        }
    }

    #[test]
    fn least_squares_matches_the_closed_form_line_fit() {
        // same fixture the Levenberg-Marquardt test uses: intercept 1.04,
        // slope 0.99
        let ts = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.1, 1.9, 3.2, 3.8, 5.1];
        let a: Vec<Vec<f64>> = ts.iter().map(|&t| vec![1.0, t]).collect();
        let x = least_squares(&a, &ys).unwrap();
        assert!((x[0] - 1.04).abs() < 1e-12 && (x[1] - 0.99).abs() < 1e-12, "{x:?}");
    }

    #[test]
    fn rank_deficiency_is_reported() {
        // second column is twice the first
        let a: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64 + 1.0, 2.0 * (i as f64 + 1.0)]).collect();
        assert!(least_squares(&a, &[1.0, 2.0, 3.0, 4.0]).is_err());
    }
}
