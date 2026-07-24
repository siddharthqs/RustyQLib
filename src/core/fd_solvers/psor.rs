//! Projected SOR (PSOR): iterative solve of the tridiagonal linear
//! complementarity problem with one- or two-sided obstacles.
//!
//! Brennan-Schwartz is exact and O(n) but only handles a one-sided
//! obstacle reachable by a directional sweep; PSOR is the general tool —
//! two-sided constraints (e.g. callable/putable structures), and the
//! smoother of choice inside multi-dimensional splitting schemes.

/// Result of a PSOR solve.
#[derive(Debug, Clone)]
pub struct PsorResult {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Solve `A x = d` projected onto `floor <= x <= cap` by projected SOR,
/// where `A` is tridiagonal in the same layout as
/// [`thomas_algorithm`](super::tridiagonal::thomas_algorithm): `a` is the
/// sub-diagonal (`a[i-1]` multiplies `x[i-1]` in row `i`), `b` the
/// diagonal and `c` the super-diagonal.
///
/// `omega` in `(0, 2)` is the relaxation factor (1 = projected
/// Gauss-Seidel; ~1.2-1.6 typically accelerates diffusion operators).
/// Convergence requires the usual SOR conditions (diagonally dominant or
/// symmetric positive definite `A`), which theta-scheme matrices satisfy.
/// The iteration stops when the largest update falls below `tol`.
pub fn psor(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    floor: Option<&[f64]>,
    cap: Option<&[f64]>,
    omega: f64,
    tol: f64,
    max_iter: usize,
) -> PsorResult {
    let n = d.len();
    assert!(b.len() == n && a.len() == n - 1 && c.len() == n - 1);
    assert!(omega > 0.0 && omega < 2.0, "SOR needs omega in (0, 2)");
    let project = |i: usize, v: f64| -> f64 {
        let mut v = v;
        if let Some(f) = floor {
            v = v.max(f[i]);
        }
        if let Some(cp) = cap {
            v = v.min(cp[i]);
        }
        v
    };

    // start from the projected diagonal solve
    let mut x: Vec<f64> = (0..n).map(|i| project(i, d[i] / b[i])).collect();
    for it in 1..=max_iter {
        let mut max_update: f64 = 0.0;
        for i in 0..n {
            let mut gs = d[i];
            if i > 0 {
                gs -= a[i - 1] * x[i - 1];
            }
            if i < n - 1 {
                gs -= c[i] * x[i + 1];
            }
            gs /= b[i];
            let xi = project(i, (1.0 - omega) * x[i] + omega * gs);
            max_update = max_update.max((xi - x[i]).abs());
            x[i] = xi;
        }
        if max_update <= tol {
            return PsorResult { x, iterations: it, converged: true };
        }
    }
    PsorResult { x, iterations: max_iter, converged: false }
}

#[cfg(test)]
mod tests {
    use super::super::brennan_schwartz::brennan_schwartz;
    use super::super::tridiagonal::thomas_algorithm;
    use super::*;

    const A: [f64; 2] = [1.0, 1.0];
    const B: [f64; 3] = [3.0, 3.0, 3.0];
    const C: [f64; 2] = [1.0, 1.0];
    const D: [f64; 3] = [5.0, 10.0, 11.0];

    #[test]
    fn unconstrained_psor_matches_thomas() {
        let free = thomas_algorithm(&A, &B, &C, &D);
        let r = psor(&A, &B, &C, &D, None, None, 1.2, 1e-13, 10_000);
        assert!(r.converged);
        for (x, y) in r.x.iter().zip(&free) {
            assert!((x - y).abs() < 1e-10, "{:?} vs {free:?}", r.x);
        }
    }

    #[test]
    fn floored_psor_matches_brennan_schwartz_on_a_put_lcp() {
        // one implicit step of an American-put discretization: obstacle =
        // convex put payoff, the setting in which Brennan-Schwartz is exact
        // (Jaillet-Lamberton-Lapeyre), so both solvers must agree. (On
        // arbitrary obstacles only PSOR solves the true LCP.)
        let n = 60;
        let lam = 0.45;
        let strike = 30.0;
        let a = vec![-lam; n - 1];
        let b = vec![1.0 + 2.0 * lam; n];
        let c = vec![-lam; n - 1];
        let payoff: Vec<f64> = (0..n).map(|i| (strike - i as f64).max(0.0)).collect();

        let bs = brennan_schwartz(&a, &b, &c, &payoff, &payoff, true);
        let r = psor(&a, &b, &c, &payoff, Some(&payoff), None, 1.4, 1e-13, 20_000);
        assert!(r.converged, "psor did not converge");
        for (x, y) in r.x.iter().zip(&bs) {
            assert!((x - y).abs() < 1e-7, "psor vs brennan-schwartz mismatch");
        }
    }

    #[test]
    fn two_sided_constraints_are_enforced() {
        let floor = [1.0, 1.0, 1.0];
        let cap = [2.0, 2.0, 2.0];
        let r = psor(&A, &B, &C, &D, Some(&floor), Some(&cap), 1.2, 1e-13, 10_000);
        assert!(r.converged);
        assert!(r.x.iter().all(|&v| v >= 1.0 - 1e-12 && v <= 2.0 + 1e-12), "{:?}", r.x);
        // the cap actually binds somewhere (the free solution exceeds 2)
        let free = thomas_algorithm(&A, &B, &C, &D);
        assert!(free.iter().any(|&v| v > 2.0));
        assert!(r.x.iter().any(|&v| (v - 2.0).abs() < 1e-10));
    }
}
