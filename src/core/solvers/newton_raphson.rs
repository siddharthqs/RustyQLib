//! Newton-Raphson: quadratic convergence near the root, given an analytic
//! derivative. Used by the Barone-Adesi-Whaley critical-boundary solve.

use super::solver_1d::{Root, Solver1d};

/// Newton-Raphson with an analytic derivative. A vanishing or non-finite
/// derivative stops the iteration with `converged: false`.
pub fn newton_raphson(
    cfg: &Solver1d,
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    x0: f64,
) -> Root {
    let mut x = x0;
    for i in 0..cfg.max_iter {
        let fx = f(x);
        if fx.abs() <= cfg.tol {
            return Root { x, iterations: i, converged: true };
        }
        let dfx = df(x);
        if dfx == 0.0 || !dfx.is_finite() {
            return Root { x, iterations: i, converged: false };
        }
        x -= fx / dfx;
    }
    Root { x, iterations: cfg.max_iter, converged: f(x).abs() <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sqrt_two() {
        let r = newton_raphson(&Solver1d::default(), |x| x * x - 2.0, |x| 2.0 * x, 1.0);
        assert!(r.converged && (r.x - std::f64::consts::SQRT_2).abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn reports_failure_on_zero_derivative() {
        // f'(0) = 0: Newton cannot move
        let r = newton_raphson(&Solver1d::default(), |x| x * x - 2.0, |x| 2.0 * x, 0.0);
        assert!(!r.converged);
    }
}
