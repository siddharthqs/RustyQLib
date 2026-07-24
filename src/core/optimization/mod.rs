//! Multi-dimensional optimization for model calibration, one algorithm
//! per file. This is the fitting layer for every parametric model —
//! Heston today ([`equity::heston::calibrate`](crate::equity::heston)),
//! SABR / Nelson-Siegel or any other least-squares fit tomorrow — so the
//! machinery lives in one place, like [`solvers`](crate::core::solvers)
//! does for 1-D root finding.
//!
//! **Gradient-based** (analytic gradient optional — central finite
//! differences fill in when absent):
//!
//! - [`steepest_descent`]: robust baseline, linear convergence;
//! - [`conjugate_gradient`]: nonlinear CG (Polak-Ribiere with restarts);
//! - [`bfgs`]: quasi-Newton with the inverse-Hessian update — the
//!   default choice for smooth unconstrained problems;
//! - [`levenberg_marquardt`]: for least squares `min sum r_i(x)^2`
//!   specifically — the standard calibration workhorse.
//!
//! **Gradient-free**:
//!
//! - [`nelder_mead`]: the simplex method, for noisy or non-smooth
//!   objectives;
//! - [`differential_evolution`]: seeded, bounded global search for
//!   multimodal landscapes (e.g. a cold-start calibration before a
//!   gradient polish).
//!
//! Every algorithm can be called directly, or through the pluggable
//! [`minimize`] with a [`Method`] enum and a [`Problem`] description:
//!
//! ```
//! use rustyqlib::core::optimization::{minimize, Method, OptimConfig, Problem};
//! let rosenbrock = |x: &[f64]| {
//!     (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
//! };
//! let problem = Problem::scalar(&rosenbrock, vec![-1.2, 1.0]);
//! let fit = minimize(&OptimConfig::default(), Method::Bfgs, &problem).unwrap();
//! assert!((fit.x[0] - 1.0).abs() < 1e-5 && (fit.x[1] - 1.0).abs() < 1e-5);
//! ```

pub mod bfgs;
pub mod conjugate_gradient;
pub mod differential_evolution;
pub mod levenberg_marquardt;
pub(crate) mod line_search;
pub mod nelder_mead;
pub(crate) mod numerics;
pub mod steepest_descent;

pub use bfgs::bfgs;
pub use conjugate_gradient::conjugate_gradient;
pub use differential_evolution::differential_evolution;
pub use levenberg_marquardt::levenberg_marquardt;
pub use nelder_mead::nelder_mead;
pub use steepest_descent::steepest_descent;

/// Result of an optimization run.
#[derive(Debug, Clone)]
pub struct OptimResult {
    /// The best parameter vector found.
    pub x: Vec<f64>,
    /// Objective value at `x` (for least squares: the sum of squared
    /// residuals).
    pub value: f64,
    /// Iterations (gradient methods), simplex steps, or generations.
    pub iterations: usize,
    /// True when the method's convergence criterion was met.
    pub converged: bool,
}

/// Optimizer configuration: convergence tolerance and iteration cap.
///
/// The tolerance applies to each method's natural criterion: the
/// gradient infinity norm (gradient-based), the simplex value spread
/// (Nelder-Mead), the population value spread (differential evolution),
/// or the cost decrease (Levenberg-Marquardt).
#[derive(Debug, Clone, Copy)]
pub struct OptimConfig {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for OptimConfig {
    fn default() -> Self {
        Self { tol: 1e-8, max_iter: 500 }
    }
}

impl OptimConfig {
    pub fn new(tol: f64, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }
}

/// The pluggable algorithm choice for [`minimize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    SteepestDescent,
    ConjugateGradient,
    Bfgs,
    LevenbergMarquardt,
    NelderMead,
    DifferentialEvolution,
}

/// An optimization problem: a scalar objective or a residual vector
/// (least squares), whatever derivatives are available, a start, and
/// optional bounds.
pub struct Problem<'a> {
    pub f: Option<&'a dyn Fn(&[f64]) -> f64>,
    pub residuals: Option<&'a dyn Fn(&[f64]) -> Vec<f64>>,
    pub gradient: Option<&'a dyn Fn(&[f64]) -> Vec<f64>>,
    pub jacobian: Option<&'a dyn Fn(&[f64]) -> Vec<Vec<f64>>>,
    pub x0: Vec<f64>,
    /// Per-parameter `(lo, hi)` box, required by differential evolution.
    pub bounds: Option<Vec<(f64, f64)>>,
    /// RNG seed for stochastic methods (differential evolution).
    pub seed: u64,
}

impl<'a> Problem<'a> {
    /// A scalar-objective problem.
    pub fn scalar(f: &'a dyn Fn(&[f64]) -> f64, x0: Vec<f64>) -> Self {
        Self { f: Some(f), residuals: None, gradient: None, jacobian: None, x0, bounds: None, seed: 42 }
    }

    /// A least-squares problem `min sum r_i(x)^2`.
    pub fn least_squares(residuals: &'a dyn Fn(&[f64]) -> Vec<f64>, x0: Vec<f64>) -> Self {
        Self { f: None, residuals: Some(residuals), gradient: None, jacobian: None, x0, bounds: None, seed: 42 }
    }

    pub fn with_gradient(mut self, gradient: &'a dyn Fn(&[f64]) -> Vec<f64>) -> Self {
        self.gradient = Some(gradient);
        self
    }

    pub fn with_jacobian(mut self, jacobian: &'a dyn Fn(&[f64]) -> Vec<Vec<f64>>) -> Self {
        self.jacobian = Some(jacobian);
        self
    }

    pub fn with_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.bounds = Some(bounds);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Minimize `problem` with the chosen [`Method`].
///
/// Scalar methods accept either problem form (a residual problem is
/// minimized as its sum of squares); `LevenbergMarquardt` requires
/// residuals and `DifferentialEvolution` requires bounds.
pub fn minimize(
    cfg: &OptimConfig,
    method: Method,
    problem: &Problem,
) -> Result<OptimResult, String> {
    // the scalar view of the problem, however it was posed
    let sum_sq;
    let f: &dyn Fn(&[f64]) -> f64 = match (problem.f, problem.residuals) {
        (Some(f), _) => f,
        (None, Some(r)) => {
            sum_sq = move |x: &[f64]| r(x).iter().map(|e| e * e).sum::<f64>();
            &sum_sq
        }
        (None, None) => return Err("Problem has neither a scalar objective nor residuals".into()),
    };
    match method {
        Method::SteepestDescent => Ok(steepest_descent(cfg, f, problem.gradient, &problem.x0)),
        Method::ConjugateGradient => Ok(conjugate_gradient(cfg, f, problem.gradient, &problem.x0)),
        Method::Bfgs => Ok(bfgs(cfg, f, problem.gradient, &problem.x0)),
        Method::LevenbergMarquardt => {
            let r = problem
                .residuals
                .ok_or("Levenberg-Marquardt needs residuals: use Problem::least_squares")?;
            Ok(levenberg_marquardt(cfg, r, problem.jacobian, &problem.x0))
        }
        Method::NelderMead => Ok(nelder_mead(cfg, f, &problem.x0)),
        Method::DifferentialEvolution => {
            let bounds = problem
                .bounds
                .as_deref()
                .ok_or("differential evolution needs bounds: use Problem::with_bounds")?;
            Ok(differential_evolution(cfg, f, bounds, problem.seed))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] + 2.0).powi(2)
    }

    #[test]
    fn every_scalar_method_is_pluggable_on_one_problem() {
        let f = |x: &[f64]| sphere(x);
        let cfg = OptimConfig::new(1e-10, 2000);
        for method in [
            Method::SteepestDescent,
            Method::ConjugateGradient,
            Method::Bfgs,
            Method::NelderMead,
            Method::DifferentialEvolution,
        ] {
            let problem = Problem::scalar(&f, vec![4.0, 4.0])
                .with_bounds(vec![(-10.0, 10.0), (-10.0, 10.0)]);
            let r = minimize(&cfg, method, &problem).unwrap();
            assert!(
                (r.x[0] - 1.0).abs() < 1e-3 && (r.x[1] + 2.0).abs() < 1e-3,
                "{method:?}: {:?}",
                r.x
            );
        }
    }

    #[test]
    fn least_squares_problems_feed_scalar_methods_too() {
        // residuals of the sphere: r = (x0 - 1, x1 + 2)
        let r = |x: &[f64]| vec![x[0] - 1.0, x[1] + 2.0];
        let problem = Problem::least_squares(&r, vec![5.0, 5.0]);
        let fit = minimize(&OptimConfig::default(), Method::Bfgs, &problem).unwrap();
        assert!(fit.value < 1e-10, "{fit:?}");
    }

    #[test]
    fn missing_requirements_error_clearly() {
        let f = |x: &[f64]| sphere(x);
        let no_bounds = Problem::scalar(&f, vec![0.0, 0.0]);
        assert!(minimize(&OptimConfig::default(), Method::DifferentialEvolution, &no_bounds).is_err());
        assert!(minimize(&OptimConfig::default(), Method::LevenbergMarquardt, &no_bounds).is_err());
    }
}
