//! Cholesky factorization of correlation matrices (PSD-tolerant).

/// Cholesky decomposition of a correlation matrix; Err if the matrix is
/// not symmetric positive **semi**-definite with a unit diagonal
/// (perfectly correlated assets are allowed).
pub fn cholesky(m: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
    let n = m.len();
    for i in 0..n {
        if (m[i][i] - 1.0).abs() > 1e-10 {
            return Err(format!("diagonal element [{i}][{i}] must be 1"));
        }
        for j in 0..n {
            if (m[i][j] - m[j][i]).abs() > 1e-10 {
                return Err("matrix must be symmetric".to_string());
            }
        }
    }
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                let d = m[i][i] - s;
                if d < -1e-10 {
                    return Err("matrix is not positive semi-definite".to_string());
                }
                l[i][j] = d.max(0.0).sqrt();
            } else if l[j][j] > 1e-12 {
                l[i][j] = (m[i][j] - s) / l[j][j];
            } else {
                l[i][j] = 0.0;
            }
        }
    }
    Ok(l)
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
