//! Steepest descent: follow `-gradient` with a backtracking line
//! search. Linear convergence — the robust baseline, and the reference
//! the fancier methods are tested against.

use super::line_search::backtracking;
use super::numerics::{dot, norm_inf, numeric_gradient};
use super::{OptimConfig, OptimResult};

/// Minimize `f` from `x0` by steepest descent. `grad` falls back to
/// central finite differences when absent.
pub fn steepest_descent(
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
    for it in 0..cfg.max_iter {
        let g = g_of(&x);
        if norm_inf(&g) <= cfg.tol {
            return OptimResult { x, value: fx, iterations: it, converged: true };
        }
        let dir: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let slope = dot(&g, &dir);
        // scale the first trial step to a unit-size move
        let alpha0 = 1.0_f64.min(1.0 / norm_inf(&g).max(1e-12));
        match backtracking(f, &x, fx, &dir, slope, alpha0) {
            Some((x_new, f_new)) => {
                x = x_new;
                fx = f_new;
            }
            None => return OptimResult { x, value: fx, iterations: it, converged: false },
        }
    }
    let g = g_of(&x);
    OptimResult { x, value: fx, iterations: cfg.max_iter, converged: norm_inf(&g) <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_a_convex_quadratic() {
        let f = |x: &[f64]| 2.0 * (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let r = steepest_descent(&OptimConfig::new(1e-8, 5000), &f, None, &[10.0, -10.0]);
        assert!(r.converged, "{r:?}");
        assert!((r.x[0] - 3.0).abs() < 1e-6 && (r.x[1] + 1.0).abs() < 1e-6, "{:?}", r.x);
    }

    #[test]
    fn analytic_gradient_is_used_when_given() {
        let f = |x: &[f64]| (x[0] - 1.0).powi(2);
        let g = |x: &[f64]| vec![2.0 * (x[0] - 1.0)];
        let r = steepest_descent(&OptimConfig::default(), &f, Some(&g), &[7.0]);
        assert!(r.converged && (r.x[0] - 1.0).abs() < 1e-7, "{r:?}");
    }
}
