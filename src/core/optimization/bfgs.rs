//! BFGS quasi-Newton: builds an inverse-Hessian approximation from
//! gradient differences, giving superlinear convergence on smooth
//! problems. The default choice for unconstrained smooth calibration.

use super::line_search::backtracking;
use super::numerics::{dot, norm_inf, numeric_gradient};
use super::{OptimConfig, OptimResult};

/// Minimize `f` from `x0` by BFGS. `grad` falls back to central finite
/// differences when absent. The inverse-Hessian update is skipped
/// whenever the curvature condition `s.y > 0` fails, which keeps the
/// approximation positive definite under the Armijo-only line search.
pub fn bfgs(
    cfg: &OptimConfig,
    f: &dyn Fn(&[f64]) -> f64,
    grad: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    x0: &[f64],
) -> OptimResult {
    let g_of = |x: &[f64]| match grad {
        Some(g) => g(x),
        None => numeric_gradient(f, x),
    };
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut g = g_of(&x);
    // h = approximate inverse Hessian, dense n x n, initialized to I
    let mut h: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    for it in 0..cfg.max_iter {
        if norm_inf(&g) <= cfg.tol {
            return OptimResult { x, value: fx, iterations: it, converged: true };
        }
        // d = -H g
        let mut dir: Vec<f64> = (0..n).map(|i| -dot(&h[i], &g)).collect();
        let mut slope = dot(&g, &dir);
        if slope >= 0.0 {
            // numerical breakdown: reset the approximation
            for (i, row) in h.iter_mut().enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = if i == j { 1.0 } else { 0.0 };
                }
            }
            dir = g.iter().map(|gi| -gi).collect();
            slope = dot(&g, &dir);
        }
        let (x_new, f_new) = match backtracking(f, &x, fx, &dir, slope, 1.0) {
            Some(step) => step,
            None => return OptimResult { x, value: fx, iterations: it, converged: false },
        };
        let g_new = g_of(&x_new);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            // H <- (I - s y^T / sy) H (I - y s^T / sy) + s s^T / sy
            let rho = 1.0 / sy;
            let hy: Vec<f64> = (0..n).map(|i| dot(&h[i], &y)).collect();
            let yhy = dot(&y, &hy);
            for i in 0..n {
                for j in 0..n {
                    h[i][j] += -rho * (s[i] * hy[j] + hy[i] * s[j])
                        + rho * rho * yhy * s[i] * s[j]
                        + rho * s[i] * s[j];
                }
            }
        }
        x = x_new;
        fx = f_new;
        g = g_new;
    }
    OptimResult { x, value: fx, iterations: cfg.max_iter, converged: norm_inf(&g) <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    #[test]
    fn minimizes_rosenbrock_quickly() {
        let f = |x: &[f64]| rosenbrock(x);
        let r = bfgs(&OptimConfig::new(1e-8, 500), &f, None, &[-1.2, 1.0]);
        assert!((r.x[0] - 1.0).abs() < 1e-5 && (r.x[1] - 1.0).abs() < 1e-5, "{r:?}");
        assert!(r.iterations < 200, "took {} iterations", r.iterations);
    }

    #[test]
    fn superlinear_beats_conjugate_gradient_on_rosenbrock() {
        let f = |x: &[f64]| rosenbrock(x);
        let cfg = OptimConfig::new(1e-6, 20_000);
        let b = bfgs(&cfg, &f, None, &[-1.2, 1.0]);
        let cg = super::super::conjugate_gradient::conjugate_gradient(&cfg, &f, None, &[-1.2, 1.0]);
        assert!(b.iterations < cg.iterations, "bfgs {} vs cg {}", b.iterations, cg.iterations);
    }

    #[test]
    fn four_dimensional_quadratic_converges() {
        let f = |x: &[f64]| {
            x.iter().enumerate().map(|(i, xi)| (i + 1) as f64 * (xi - i as f64).powi(2)).sum()
        };
        let r = bfgs(&OptimConfig::default(), &f, None, &[5.0; 4]);
        assert!(r.converged, "{r:?}");
        for (i, xi) in r.x.iter().enumerate() {
            assert!((xi - i as f64).abs() < 1e-6, "{:?}", r.x);
        }
    }
}
