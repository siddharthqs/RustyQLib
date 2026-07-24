//! Halley's method: cubic convergence using the first and second
//! derivatives. Used to polish the inverse normal CDF
//! ([`inv_norm_cdf`](crate::core::utils::inv_norm_cdf)).

use super::solver_1d::{Root, Solver1d};

/// Halley iteration `x - 2 f f' / (2 f'^2 - f f'')`.
pub fn halley(
    cfg: &Solver1d,
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    d2f: impl Fn(f64) -> f64,
    x0: f64,
) -> Root {
    let mut x = x0;
    for i in 0..cfg.max_iter {
        let fx = f(x);
        if fx.abs() <= cfg.tol {
            return Root { x, iterations: i, converged: true };
        }
        let dfx = df(x);
        let denom = 2.0 * dfx * dfx - fx * d2f(x);
        if denom == 0.0 || !denom.is_finite() {
            return Root { x, iterations: i, converged: false };
        }
        x -= 2.0 * fx * dfx / denom;
    }
    Root { x, iterations: cfg.max_iter, converged: f(x).abs() <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f64) -> f64 {
        x * x - 2.0
    }
    fn df(x: f64) -> f64 {
        2.0 * x
    }
    fn d2f(_: f64) -> f64 {
        2.0
    }

    #[test]
    fn finds_sqrt_two() {
        let r = halley(&Solver1d::default(), f, df, d2f, 1.0);
        assert!(r.converged && (r.x - std::f64::consts::SQRT_2).abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn converges_in_fewer_iterations_than_newton() {
        let cfg = Solver1d::new(1e-14, 200);
        let n = super::super::newton_raphson::newton_raphson(&cfg, f, df, 100.0);
        let h = halley(&cfg, f, df, d2f, 100.0);
        assert!(n.converged && h.converged);
        assert!(h.iterations <= n.iterations, "halley {} vs newton {}", h.iterations, n.iterations);
    }
}
