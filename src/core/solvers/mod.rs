//! Shared 1-D root-finding for the whole library, one algorithm per file.
//!
//! Every iterative solve in the pricers routes through here — the BAW
//! critical-exercise boundary (Newton-Raphson), implied volatility
//! (safeguarded Newton), the inverse normal CDF polish (Halley) — so the
//! numerical machinery lives in one maintainable place. Companion
//! modules: [`fd_solvers`](crate::core::fd_solvers) for the finite-
//! difference linear kernels and
//! [`optimization`](crate::core::optimization) for multi-dimensional
//! model calibration.
//!
//! Two ways to call a solver:
//!
//! **Direct**, when the method is fixed at the call site — [`Solver1d`]
//! carries the convergence settings and each method is a call on it:
//!
//! ```
//! use rustyqlib::core::solvers::Solver1d;
//! let root = Solver1d::default()
//!     .newton_raphson(|x| x * x - 2.0, |x| 2.0 * x, 1.0);
//! assert!(root.converged && (root.x - 2.0_f64.sqrt()).abs() < 1e-12);
//! ```
//!
//! **Pluggable**, when the method should be swappable — describe the
//! problem once and pick the algorithm with [`Method`]; derivatives the
//! problem does not supply are replaced by central finite differences, so
//! every method runs on every problem (bracketed methods do require a
//! bracket):
//!
//! ```
//! use rustyqlib::core::solvers::{Method, Problem, Solver1d};
//! let f = |x: f64| x * x - 2.0;
//! let problem = Problem::new(&f, 1.0).with_bracket(0.0, 2.0);
//! for method in [
//!     Method::Bisection,
//!     Method::NewtonRaphson,
//!     Method::Secant,
//!     Method::Halley,
//!     Method::NewtonSafeguarded,
//! ] {
//!     let root = Solver1d::default().solve(method, &problem).unwrap();
//!     assert!((root.x - 2.0_f64.sqrt()).abs() < 1e-9, "{method:?}");
//! }
//! ```
//!
//! Convergence is on the residual: a solve is converged when
//! `|f(x)| <= tol`. The bracketed methods additionally stop when the
//! bracket collapses to machine width. Non-bracketed methods never error:
//! they return a [`Root`] whose `converged` flag says whether the tolerance
//! was met, leaving the retry/fallback policy to the caller.

pub mod bisection;
pub mod halley;
pub mod newton_raphson;
pub mod newton_safeguarded;
pub mod secant;
pub mod solver_1d;

pub use solver_1d::{Method, Problem, Root, Solver1d};
