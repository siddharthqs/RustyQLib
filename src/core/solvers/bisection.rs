//! Bisection: derivative-free and unconditionally convergent on a
//! sign-changing bracket, at a linear rate.

use super::solver_1d::{Root, Solver1d};
use crate::core::errors::RustyQLibError;

/// Bisection on a sign-changing bracket `[lo, hi]`. Errors when the
/// bracket does not straddle a root.
pub fn bisection(
    cfg: &Solver1d,
    f: impl Fn(f64) -> f64,
    lo: f64,
    hi: f64,
) -> Result<Root, RustyQLibError> {
    let (mut lo, mut hi) = (lo.min(hi), lo.max(hi));
    let f_lo = f(lo);
    if f_lo.abs() <= cfg.tol {
        return Ok(Root { x: lo, iterations: 0, converged: true });
    }
    let f_hi = f(hi);
    if f_hi.abs() <= cfg.tol {
        return Ok(Root { x: hi, iterations: 0, converged: true });
    }
    if f_lo * f_hi > 0.0 {
        return Err(RustyQLibError::NumericalError(format!(
            "bisection needs a sign change: f({lo}) = {f_lo}, f({hi}) = {f_hi}"
        )));
    }
    let mut x = 0.5 * (lo + hi);
    for i in 1..=cfg.max_iter {
        let fx = f(x);
        if fx.abs() <= cfg.tol {
            return Ok(Root { x, iterations: i, converged: true });
        }
        if fx * f_lo < 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        x = 0.5 * (lo + hi);
        if hi - lo <= f64::EPSILON * (1.0 + x.abs()) {
            return Ok(Root { x, iterations: i, converged: f(x).abs() <= cfg.tol });
        }
    }
    Ok(Root { x, iterations: cfg.max_iter, converged: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sqrt_two() {
        let r = bisection(&Solver1d::default(), |x| x * x - 2.0, 0.0, 2.0).unwrap();
        assert!(r.converged && (r.x - std::f64::consts::SQRT_2).abs() < 1e-9, "{r:?}");
    }

    #[test]
    fn rejects_a_bracket_without_sign_change() {
        assert!(bisection(&Solver1d::default(), |x| x * x - 2.0, 2.0, 3.0).is_err());
    }
}
