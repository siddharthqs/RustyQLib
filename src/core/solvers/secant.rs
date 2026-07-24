//! Secant method: Newton's convergence class without a derivative,
//! approximating the slope from the last two iterates.

use super::solver_1d::{Root, Solver1d};

/// Secant iteration from two starting points.
pub fn secant(cfg: &Solver1d, f: impl Fn(f64) -> f64, x0: f64, x1: f64) -> Root {
    let (mut x_prev, mut x) = (x0, x1);
    let mut f_prev = f(x_prev);
    if f_prev.abs() <= cfg.tol {
        return Root { x: x_prev, iterations: 0, converged: true };
    }
    for i in 0..cfg.max_iter {
        let fx = f(x);
        if fx.abs() <= cfg.tol {
            return Root { x, iterations: i, converged: true };
        }
        let slope = (fx - f_prev) / (x - x_prev);
        if slope == 0.0 || !slope.is_finite() {
            return Root { x, iterations: i, converged: false };
        }
        (x_prev, f_prev) = (x, fx);
        x -= fx / slope;
    }
    Root { x, iterations: cfg.max_iter, converged: f(x).abs() <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sqrt_two() {
        let r = secant(&Solver1d::default(), |x| x * x - 2.0, 1.0, 2.0);
        assert!(r.converged && (r.x - std::f64::consts::SQRT_2).abs() < 1e-12, "{r:?}");
    }
}
