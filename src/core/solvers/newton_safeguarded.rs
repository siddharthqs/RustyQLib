//! Newton safeguarded by a bisection bracket: globally convergent where
//! raw Newton diverges (flat tails, distant starts). Used by the implied
//! volatility solver.

use super::solver_1d::{Root, Solver1d};

/// Full Newton steps while they land strictly inside the current
/// `[lo, hi]`, bisection otherwise, with the bracket shrunk every
/// iteration.
///
/// Requires `f(lo) <= 0 <= f(hi)` — i.e. a bracket in which `f` crosses
/// from non-positive to non-negative, as with monotonically increasing
/// objectives like an option price in volatility.
pub fn newton_safeguarded(
    cfg: &Solver1d,
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    x0: f64,
) -> Root {
    let (mut lo, mut hi) = (lo, hi);
    let mut x = x0.clamp(lo, hi);
    for i in 0..cfg.max_iter {
        let fx = f(x);
        if fx.abs() <= cfg.tol {
            return Root { x, iterations: i, converged: true };
        }
        if fx > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let dfx = df(x);
        let newton = x - fx / dfx;
        x = if dfx > 1e-12 && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if hi - lo <= 1e-14 {
            return Root { x, iterations: i + 1, converged: f(x).abs() <= cfg.tol };
        }
    }
    Root { x, iterations: cfg.max_iter, converged: f(x).abs() <= cfg.tol }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sqrt_two() {
        let r = newton_safeguarded(
            &Solver1d::default(),
            |x| x * x - 2.0,
            |x| 2.0 * x,
            0.0,
            2.0,
            1.0,
        );
        assert!(r.converged && (r.x - std::f64::consts::SQRT_2).abs() < 1e-12, "{r:?}");
    }

    #[test]
    fn survives_where_raw_newton_diverges() {
        // atan has flat tails: raw Newton from x0 = 30 overshoots to
        // +/- infinity, the safeguarded variant just bisects its way in
        let g = |x: f64| (x - 1.0).atan();
        let dg = |x: f64| 1.0 / (1.0 + (x - 1.0) * (x - 1.0));
        let cfg = Solver1d::new(1e-12, 100);
        let raw = super::super::newton_raphson::newton_raphson(&cfg, g, dg, 30.0);
        assert!(!raw.converged || (raw.x - 1.0).abs() > 1e-6, "raw newton unexpectedly fine");
        let safe = newton_safeguarded(&cfg, g, dg, -50.0, 50.0, 30.0);
        assert!(safe.converged && (safe.x - 1.0).abs() < 1e-9, "{safe:?}");
    }
}
