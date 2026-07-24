//! Nonlinear conjugate gradient (Polak-Ribiere+): steepest descent's
//! cost per iteration with far better search directions on ill-
//! conditioned valleys, no matrix storage.

use super::line_search::backtracking;
use super::numerics::{dot, norm_inf, numeric_gradient};
use super::{OptimConfig, OptimResult};

/// Minimize `f` from `x0` by Polak-Ribiere conjugate gradient with the
/// `beta >= 0` safeguard (automatic restart to steepest descent).
/// `grad` falls back to central finite differences when absent.
pub fn conjugate_gradient(
    cfg: &OptimConfig,
    f: &dyn Fn(&[f64]) -> f64,
    grad: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    x0: &[f64],
) -> OptimResult {
    let g_of = |x: &[f64]| match grad {
        Some(g) => g(x),
        None => numeric_gradient(f, x),
    };
    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut g = g_of(&x);
    let mut dir: Vec<f64> = g.iter().map(|gi| -gi).collect();
    for it in 0..cfg.max_iter {
        if norm_inf(&g) <= cfg.tol {
            return OptimResult { x, value: fx, iterations: it, converged: true };
        }
        let mut slope = dot(&g, &dir);
        if slope >= 0.0 {
            // not a descent direction: restart with steepest descent
            dir = g.iter().map(|gi| -gi).collect();
            slope = dot(&g, &dir);
        }
        let alpha0 = 1.0_f64.min(1.0 / norm_inf(&g).max(1e-12));
        let (x_new, f_new) = match backtracking(f, &x, fx, &dir, slope, alpha0) {
            Some(step) => step,
            None => return OptimResult { x, value: fx, iterations: it, converged: false },
        };
        let g_new = g_of(&x_new);
        // Polak-Ribiere+ conjugacy factor
        let beta = (dot(&g_new, &g_new) - dot(&g_new, &g)) / dot(&g, &g).max(1e-300);
        let beta = beta.max(0.0);
        dir = g_new.iter().zip(&dir).map(|(gi, di)| -gi + beta * di).collect();
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
    fn minimizes_the_rosenbrock_valley() {
        let f = |x: &[f64]| rosenbrock(x);
        let r = conjugate_gradient(&OptimConfig::new(1e-6, 20_000), &f, None, &[-1.2, 1.0]);
        assert!((r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] - 1.0).abs() < 1e-3, "{r:?}");
        assert!(r.value < 1e-6, "{r:?}");
    }

    #[test]
    fn converges_on_an_ill_conditioned_quadratic() {
        // condition number 10^6 across four dimensions
        let f = |x: &[f64]| {
            x.iter()
                .enumerate()
                .map(|(i, xi)| 10.0_f64.powi(2 * i as i32) * xi * xi)
                .sum::<f64>()
        };
        let cfg = OptimConfig::new(1e-8, 50_000);
        let cg = conjugate_gradient(&cfg, &f, None, &[1.0, 1.0, 1.0, 1.0]);
        assert!(cg.converged, "{cg:?}");
        assert!(cg.value < 1e-10, "{cg:?}");
    }
}
