//! Shared numerical helpers: finite-difference derivatives, small dense
//! linear solves, and vector arithmetic used by the optimizers.

/// Central-difference gradient.
pub(crate) fn numeric_gradient(f: &dyn Fn(&[f64]) -> f64, x: &[f64]) -> Vec<f64> {
    let mut g = vec![0.0; x.len()];
    let mut xp = x.to_vec();
    for i in 0..x.len() {
        let h = 1e-6 * (1.0 + x[i].abs());
        xp[i] = x[i] + h;
        let up = f(&xp);
        xp[i] = x[i] - h;
        let dn = f(&xp);
        xp[i] = x[i];
        g[i] = (up - dn) / (2.0 * h);
    }
    g
}

/// Forward-difference Jacobian of a residual vector (rows = residuals).
pub(crate) fn numeric_jacobian(r: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
    let r0 = r(x);
    let m = r0.len();
    let n = x.len();
    let mut jac = vec![vec![0.0; n]; m];
    let mut xp = x.to_vec();
    for j in 0..n {
        let h = 1e-6 * (1.0 + x[j].abs());
        xp[j] = x[j] + h;
        let rp = r(&xp);
        xp[j] = x[j];
        for i in 0..m {
            jac[i][j] = (rp[i] - r0[i]) / h;
        }
    }
    jac
}

/// Solve the small dense system `A x = b` by Gaussian elimination with
/// partial pivoting (calibrations have a handful of parameters).
pub(crate) fn solve_dense(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = b.len();
    for k in 0..n {
        // pivot
        let piv = (k..n).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[piv][k].abs() < 1e-300 {
            return None;
        }
        a.swap(k, piv);
        b.swap(k, piv);
        for i in k + 1..n {
            let m = a[i][k] / a[k][k];
            for j in k..n {
                a[i][j] -= m * a[k][j];
            }
            b[i] -= m * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm_inf(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_and_jacobian_match_analytic() {
        let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1].sin();
        let g = numeric_gradient(&f, &[1.5, 0.7]);
        assert!((g[0] - 2.0 * 1.5 * 0.7).abs() < 1e-6);
        assert!((g[1] - (1.5f64 * 1.5 + 0.7f64.cos())).abs() < 1e-6);

        let r = |x: &[f64]| vec![x[0] * x[1], x[0] - x[1]];
        let j = numeric_jacobian(&r, &[2.0, 3.0]);
        assert!((j[0][0] - 3.0).abs() < 1e-5 && (j[0][1] - 2.0).abs() < 1e-5);
        assert!((j[1][0] - 1.0).abs() < 1e-5 && (j[1][1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn dense_solve_inverts_a_small_system() {
        let mut a = vec![vec![4.0, 1.0], vec![1.0, 3.0]];
        let mut b = vec![1.0, 2.0];
        let x = solve_dense(&mut a, &mut b).unwrap();
        // A x = b with A = [4 1; 1 3], b = [1; 2] -> x = [1/11, 7/11]
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-12);
        assert!((x[1] - 7.0 / 11.0).abs() < 1e-12);
    }
}
