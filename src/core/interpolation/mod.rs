//! Interpolation toolkit, one scheme per file — the single home for
//! every interpolation in the library (curves, smiles, surfaces) and a
//! standalone toolkit in its own right.
//!
//! **1-D:**
//! - [`linear`]: piecewise linear with flat extrapolation — the pillar
//!   bracketing used by discount curves and smile interpolation;
//! - [`cubic_spline`]: C2 cubic splines with **Natural**, **Clamped**
//!   and **Not-a-Knot** boundary conditions;
//! - [`pchip`]: Fritsch-Carlson monotone cubic Hermite (PCHIP) —
//!   shape-preserving, no overshoot on monotone data (the safe choice
//!   for zero curves and CDF-like data);
//! - [`akima`]: Akima's spline — local, outlier-robust slopes with far
//!   less oscillation than a global cubic fit.
//!
//! **2-D (volatility grids):**
//! - [`bilinear`]: rectangular-grid bilinear interpolation with flat
//!   extrapolation;
//! - [`thin_plate`]: bivariate thin-plate spline (RBF) on scattered
//!   points, with optional smoothing — for irregular vol quote grids.
//!
//! Related: [`linalg`](crate::core::linalg) holds the nearest-correlation
//! projection (Higham) used to repair empirical correlation surfaces.

pub mod akima;
pub mod bilinear;
pub mod cubic_spline;
pub mod linear;
pub mod pchip;
pub mod thin_plate;

pub use akima::Akima;
pub use bilinear::BilinearGrid;
pub use cubic_spline::{BoundaryCondition, CubicSpline};
pub use linear::{bracket, interp_pairs, lerp, linear_interp};
pub use pchip::Pchip;
pub use thin_plate::ThinPlateSpline;
