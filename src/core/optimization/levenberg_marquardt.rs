//! Levenberg-Marquardt: the standard damped Gauss-Newton method for
//! nonlinear least squares `min sum r_i(x)^2` — the calibration
//! workhorse (Heston, SABR, Nelson-Siegel, curve and surface fits).

use super::numerics::{norm_inf, numeric_jacobian, solve_dense};
use super::{OptimConfig, OptimResult};

/// Minimize the sum of squared residuals from `x0`.
///
/// Each iteration solves the damped normal equations
/// `(J^T J + lambda diag(J^T J)) step = -J^T r` (Marquardt scaling) and
/// adapts `lambda`: accepted steps shrink it toward Gauss-Newton,
/// rejected steps grow it toward small-step gradient descent.
/// `jacobian` (rows = residuals) falls back to forward finite
/// differences when absent. `OptimResult::value` is the sum of squares.
pub fn levenberg_marquardt(
    cfg: &OptimConfig,
    residuals: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian: Option<&dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    x0: &[f64],
) -> OptimResult {
    let jac_of = |x: &[f64]| match jacobian {
        Some(j) => j(x),
        None => numeric_jacobian(residuals, x),
    };
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut r = residuals(&x);
    let mut cost: f64 = r.iter().map(|e| e * e).sum();
    let mut lambda = 1e-3;

    for it in 0..cfg.max_iter {
        let jac = jac_of(&x);
        let m = r.len();
        // g = J^T r  and  a = J^T J
        let mut g = vec![0.0; n];
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..m {
            for j in 0..n {
                g[j] += jac[i][j] * r[i];
                for k in j..n {
                    a[j][k] += jac[i][j] * jac[i][k];
                }
            }
        }
        for j in 0..n {
            for k in 0..j {
                a[j][k] = a[k][j];
            }
        }
        if norm_inf(&g) <= cfg.tol {
            return OptimResult { x, value: cost, iterations: it, converged: true };
        }

        // try damped steps, growing lambda until one reduces the cost
        let mut accepted = false;
        for _ in 0..40 {
            let mut damped = a.clone();
            for j in 0..n {
                damped[j][j] += lambda * a[j][j].max(1e-12);
            }
            let mut rhs: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let step = match solve_dense(&mut damped, &mut rhs) {
                Some(s) => s,
                None => break,
            };
            let x_try: Vec<f64> = x.iter().zip(&step).map(|(xi, si)| xi + si).collect();
            let r_try = residuals(&x_try);
            let cost_try: f64 = r_try.iter().map(|e| e * e).sum();
            if cost_try < cost {
                let improvement = cost - cost_try;
                x = x_try;
                r = r_try;
                cost = cost_try;
                lambda = (lambda / 10.0).max(1e-12);
                accepted = true;
                // converged when the improvement stalls
                if improvement <= cfg.tol * (1.0 + cost) {
                    return OptimResult { x, value: cost, iterations: it + 1, converged: true };
                }
                break;
            }
            lambda = (lambda * 10.0).min(1e12);
        }
        if !accepted {
            return OptimResult { x, value: cost, iterations: it, converged: false };
        }
    }
    OptimResult { x, value: cost, iterations: cfg.max_iter, converged: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_exponential_decay_parameters() {
        // data from y = 2 e^{-1.3 t}: fit (a, b) from a distant start
        let ts: Vec<f64> = (0..12).map(|i| i as f64 * 0.25).collect();
        let data: Vec<f64> = ts.iter().map(|t| 2.0 * (-1.3 * t).exp()).collect();
        let residuals = move |p: &[f64]| -> Vec<f64> {
            ts.iter().zip(&data).map(|(t, y)| p[0] * (p[1] * t).exp() - y).collect()
        };
        let r = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &[1.0, -0.5]);
        assert!(r.converged, "{r:?}");
        assert!((r.x[0] - 2.0).abs() < 1e-6 && (r.x[1] + 1.3).abs() < 1e-6, "{:?}", r.x);
    }

    #[test]
    fn analytic_jacobian_agrees_with_numeric() {
        let ts: Vec<f64> = (0..8).map(|i| i as f64 * 0.3).collect();
        let data: Vec<f64> = ts.iter().map(|t| 1.5 * (-0.8 * t).exp()).collect();
        let res = {
            let (ts, data) = (ts.clone(), data.clone());
            move |p: &[f64]| -> Vec<f64> {
                ts.iter().zip(&data).map(|(t, y)| p[0] * (p[1] * t).exp() - y).collect()
            }
        };
        let jac = move |p: &[f64]| -> Vec<Vec<f64>> {
            ts.iter().map(|t| vec![(p[1] * t).exp(), p[0] * t * (p[1] * t).exp()]).collect()
        };
        let with = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &res, Some(&jac), &[1.0, -0.4]);
        let without = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &res, None, &[1.0, -0.4]);
        assert!(with.converged && without.converged);
        assert!((with.x[0] - without.x[0]).abs() < 1e-6);
        assert!((with.x[1] - without.x[1]).abs() < 1e-6);
    }

    #[test]
    fn overdetermined_linear_fit_matches_normal_equations() {
        // fit y = c0 + c1 t to 5 points: LM must find the least-squares line
        let ts = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.1, 1.9, 3.2, 3.8, 5.1];
        let residuals =
            move |p: &[f64]| -> Vec<f64> { ts.iter().zip(&ys).map(|(t, y)| p[0] + p[1] * t - y).collect() };
        let r = levenberg_marquardt(&OptimConfig::new(1e-14, 100), &residuals, None, &[0.0, 0.0]);
        // closed form: slope 0.99, intercept 1.04
        assert!((r.x[0] - 1.04).abs() < 1e-6 && (r.x[1] - 0.99).abs() < 1e-6, "{:?}", r.x);
    }
}
