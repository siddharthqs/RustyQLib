//! Finite-difference (PDE) solvers for 1-D, 2-D and 3-D problems, one
//! scheme per file — numerical kernels only, independent of any payoff or
//! grid, so they are usable as a standalone FD toolkit.
//!
//! **Linear kernels** (consumed by the equity FD engine,
//! [`equity::finite_difference`](crate::equity::finite_difference)):
//!
//! - [`tridiagonal`]: the Thomas algorithm for `A x = d` with a
//!   tridiagonal `A` — the workhorse of every implicit 1-D step;
//! - [`brennan_schwartz`]: the Brennan-Schwartz sweep for the linear
//!   complementarity problem `A x = d, x >= exercise` of American
//!   exercise (one-sided obstacle, exact, O(n));
//! - [`psor`]: projected SOR for the general LCP — two-sided obstacles
//!   (callable/putable structures) and the smoother inside splitting
//!   schemes.
//!
//! **Multi-dimensional machinery** (for two/three-factor models such as
//! Heston or hybrid equity-rates):
//!
//! - [`axis_operator`]: [`TensorGrid`](axis_operator::TensorGrid) +
//!   [`AxisOperator`](axis_operator::AxisOperator) — per-axis tridiagonal
//!   operators with node-varying coefficients, with explicit application
//!   and line-by-line implicit solves;
//! - [`adi`]: the Douglas and Hundsdorfer-Verwer ADI time steppers over
//!   those operators, with mixed-derivative terms (correlation) handled
//!   explicitly. One axis with no mixed term reduces exactly to the 1-D
//!   theta scheme.
//!
//! Craig-Sneyd / Modified Craig-Sneyd steppers would slot into [`adi`]
//! alongside the existing two if ever needed.

pub mod adi;
pub mod axis_operator;
pub mod brennan_schwartz;
pub mod psor;
pub mod tridiagonal;

pub use adi::{douglas_step, hundsdorfer_verwer_step};
pub use axis_operator::{AxisOperator, TensorGrid};
pub use brennan_schwartz::brennan_schwartz;
pub use psor::{psor, PsorResult};
pub use tridiagonal::thomas_algorithm;
