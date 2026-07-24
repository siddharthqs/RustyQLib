//! The Thomas algorithm: direct O(n) solve of a tridiagonal system.

/// Solves a tridiagonal system `A x = d` where `a` is the sub-diagonal
/// (`a[i-1]` multiplies `x[i-1]` in row `i`), `b` the diagonal and `c` the
/// super-diagonal (`c[i]` multiplies `x[i+1]` in row `i`).
/// https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
pub fn thomas_algorithm(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = d.len();
    assert!(b.len() == n && a.len() == n - 1 && c.len() == n - 1);
    if n == 1 {
        return vec![d[0] / b[0]];
    }
    let mut c_ = c.to_vec();
    let mut d_ = d.to_vec();
    let mut x: Vec<f64> = vec![0.0; n];

    c_[0] = c_[0] / b[0];
    d_[0] = d_[0] / b[0];
    for i in 1..n - 1 {
        let id = 1.0 / (b[i] - a[i - 1] * c_[i - 1]);
        c_[i] = c_[i] * id;
        d_[i] = (d_[i] - a[i - 1] * d_[i - 1]) * id;
    }
    d_[n - 1] = (d_[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2]);

    x[n - 1] = d_[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_[i] - c_[i] * x[i + 1];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thomas_solves_small_system() {
        // [2 1 0; 1 2 1; 0 1 2] x = [4; 8; 8] -> x = [1; 2; 3]
        let x = thomas_algorithm(&[1.0, 1.0], &[2.0, 2.0, 2.0], &[1.0, 1.0], &[4.0, 8.0, 8.0]);
        for (got, want) in x.iter().zip(&[1.0, 2.0, 3.0]) {
            assert!((got - want).abs() < 1e-12, "{x:?}");
        }
    }
}
