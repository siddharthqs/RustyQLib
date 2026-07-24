//! The shared solver types and the pluggable method dispatch.
//!
//! [`Solver1d`] is the configuration (tolerance + iteration cap) every
//! algorithm takes; its inherent methods delegate to the per-algorithm
//! files for direct, statically-chosen calls. [`Problem`] + [`Method`] are
//! the pluggable layer: describe the objective once, then swap algorithms
//! with an enum value.

use super::{bisection, halley, newton_raphson, newton_safeguarded, secant};

/// Result of a 1-D root search.
#[derive(Debug, Clone, Copy)]
pub struct Root {
    /// The best estimate of the root.
    pub x: f64,
    /// Iterations actually taken.
    pub iterations: usize,
    /// True when `|f(x)| <= tol` was reached.
    pub converged: bool,
}

/// A 1-D root finder: residual tolerance plus an iteration cap.
#[derive(Debug, Clone, Copy)]
pub struct Solver1d {
    /// Convergence tolerance on `|f(x)|`.
    pub tol: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
}

impl Default for Solver1d {
    fn default() -> Self {
        Self { tol: 1e-12, max_iter: 100 }
    }
}

impl Solver1d {
    pub fn new(tol: f64, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }

    // ── direct calls (method fixed at the call site) ────────────────────

    /// [Bisection](bisection::bisection) on a sign-changing bracket.
    pub fn bisection(&self, f: impl Fn(f64) -> f64, lo: f64, hi: f64) -> Result<Root, String> {
        bisection::bisection(self, f, lo, hi)
    }

    /// [Newton-Raphson](newton_raphson::newton_raphson) with an analytic
    /// derivative.
    pub fn newton_raphson(
        &self,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        x0: f64,
    ) -> Root {
        newton_raphson::newton_raphson(self, f, df, x0)
    }

    /// [Secant](secant::secant) from two starting points.
    pub fn secant(&self, f: impl Fn(f64) -> f64, x0: f64, x1: f64) -> Root {
        secant::secant(self, f, x0, x1)
    }

    /// [Halley](halley::halley) with analytic first and second derivatives.
    pub fn halley(
        &self,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        d2f: impl Fn(f64) -> f64,
        x0: f64,
    ) -> Root {
        halley::halley(self, f, df, d2f, x0)
    }

    /// [Bracket-safeguarded Newton](newton_safeguarded::newton_safeguarded).
    pub fn newton_safeguarded(
        &self,
        f: impl Fn(f64) -> f64,
        df: impl Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        x0: f64,
    ) -> Root {
        newton_safeguarded::newton_safeguarded(self, f, df, lo, hi, x0)
    }

    // ── pluggable dispatch (method chosen at run time) ──────────────────

    /// Solve `problem` with the chosen [`Method`].
    ///
    /// Derivatives the problem does not carry are approximated by central
    /// finite differences, so Newton/Halley run on derivative-free
    /// problems too. `Bisection` and `NewtonSafeguarded` require a
    /// bracket and error without one.
    pub fn solve(&self, method: Method, problem: &Problem) -> Result<Root, String> {
        let f = |x: f64| (problem.f)(x);
        let df = |x: f64| match problem.df {
            Some(df) => df(x),
            None => numeric_derivative(problem.f, x),
        };
        let d2f = |x: f64| match problem.d2f {
            Some(d2f) => d2f(x),
            None => numeric_second_derivative(problem.f, x),
        };
        let bracket = |name: &str| {
            problem
                .bracket
                .ok_or_else(|| format!("{name} needs a bracket: use Problem::with_bracket"))
        };
        match method {
            Method::Bisection => {
                let (lo, hi) = bracket("bisection")?;
                self.bisection(f, lo, hi)
            }
            Method::NewtonRaphson => Ok(self.newton_raphson(f, df, problem.x0)),
            Method::Secant => {
                // second start: the far bracket end when one is given,
                // otherwise a small relative step from x0
                let x1 = match problem.bracket {
                    Some((lo, hi)) => {
                        if (problem.x0 - lo).abs() > (problem.x0 - hi).abs() { lo } else { hi }
                    }
                    None => problem.x0 + 1e-4 * (1.0 + problem.x0.abs()),
                };
                Ok(self.secant(f, problem.x0, x1))
            }
            Method::Halley => Ok(self.halley(f, df, d2f, problem.x0)),
            Method::NewtonSafeguarded => {
                let (lo, hi) = bracket("newton_safeguarded")?;
                Ok(self.newton_safeguarded(f, df, lo, hi, problem.x0))
            }
        }
    }
}

/// The pluggable algorithm choice for [`Solver1d::solve`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Bisection,
    NewtonRaphson,
    Secant,
    Halley,
    NewtonSafeguarded,
}

/// A 1-D root-finding problem: the objective, whatever derivatives are
/// available, a starting point, and an optional bracket.
pub struct Problem<'a> {
    pub f: &'a dyn Fn(f64) -> f64,
    pub df: Option<&'a dyn Fn(f64) -> f64>,
    pub d2f: Option<&'a dyn Fn(f64) -> f64>,
    pub x0: f64,
    pub bracket: Option<(f64, f64)>,
}

impl<'a> Problem<'a> {
    pub fn new(f: &'a dyn Fn(f64) -> f64, x0: f64) -> Self {
        Self { f, df: None, d2f: None, x0, bracket: None }
    }

    pub fn with_derivative(mut self, df: &'a dyn Fn(f64) -> f64) -> Self {
        self.df = Some(df);
        self
    }

    pub fn with_second_derivative(mut self, d2f: &'a dyn Fn(f64) -> f64) -> Self {
        self.d2f = Some(d2f);
        self
    }

    pub fn with_bracket(mut self, lo: f64, hi: f64) -> Self {
        self.bracket = Some((lo.min(hi), lo.max(hi)));
        self
    }
}

fn numeric_derivative(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-6 * (1.0 + x.abs());
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn numeric_second_derivative(f: &dyn Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-4 * (1.0 + x.abs());
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQRT2: f64 = std::f64::consts::SQRT_2;

    fn f(x: f64) -> f64 {
        x * x - 2.0
    }
    fn df(x: f64) -> f64 {
        2.0 * x
    }

    #[test]
    fn every_method_is_pluggable_on_one_problem() {
        let obj = |x: f64| f(x);
        let problem = Problem::new(&obj, 1.0).with_bracket(0.0, 2.0);
        let solver = Solver1d::default();
        for method in [
            Method::Bisection,
            Method::NewtonRaphson,
            Method::Secant,
            Method::Halley,
            Method::NewtonSafeguarded,
        ] {
            let root = solver.solve(method, &problem).unwrap();
            assert!(
                root.converged && (root.x - SQRT2).abs() < 1e-9,
                "{method:?}: {root:?}"
            );
        }
    }

    #[test]
    fn numeric_derivative_fallback_matches_analytic() {
        // same problem with and without the analytic derivative: Newton
        // must land on the same root either way
        let obj = |x: f64| f(x);
        let d = |x: f64| df(x);
        let with = Problem::new(&obj, 1.0).with_derivative(&d);
        let without = Problem::new(&obj, 1.0);
        let solver = Solver1d::default();
        let ra = solver.solve(Method::NewtonRaphson, &with).unwrap();
        let rn = solver.solve(Method::NewtonRaphson, &without).unwrap();
        assert!(ra.converged && rn.converged);
        assert!((ra.x - rn.x).abs() < 1e-9, "{} vs {}", ra.x, rn.x);
    }

    #[test]
    fn bracketed_methods_error_without_a_bracket() {
        let obj = |x: f64| f(x);
        let problem = Problem::new(&obj, 1.0);
        let solver = Solver1d::default();
        assert!(solver.solve(Method::Bisection, &problem).is_err());
        assert!(solver.solve(Method::NewtonSafeguarded, &problem).is_err());
        // non-bracketed methods still work
        assert!(solver.solve(Method::Secant, &problem).unwrap().converged);
    }

    #[test]
    fn iteration_counts_are_reported() {
        let r = Solver1d::default().newton_raphson(f, df, 1.0);
        assert!(r.iterations > 0 && r.iterations < 10);
    }
}
