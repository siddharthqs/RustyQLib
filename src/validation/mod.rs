//! Model validation: runtime checks that measure model *quality on
//! given data*, complementing the unit tests that pin implementation
//! correctness.
//!
//! A unit test asks "does this code compute what it claims, exactly,
//! forever" — binary, fixed inputs, run in CI. A validation check asks
//! "is this calibration usable, does this simulation respect its
//! martingale property, at what rate does this engine converge — on
//! *these* inputs, *today*" — numbers with tolerances and context,
//! computed at runtime, shipped as serializable reports so a validation
//! run is an auditable artifact (the same document philosophy as the
//! rest of the library). Statistical checks report z-scores rather than
//! naked pass/fail: they say how surprised you should be.
//!
//! Families:
//!
//! - [`martingale`] — forward recovery under simulated dynamics against
//!   externally supplied target forwards, in z-scores.
//! - Planned: `convergence` (empirical convergence order of the
//!   engines), `stability` (input-bump amplification ratios),
//!   `sensitivity` (Greeks and parity consistency across engines).
//!
//! Checks that grew up elsewhere in the library and belong to this
//! family are re-exported here as the single index of "how do I know
//! these numbers are right":
//!
//! - [`SurfaceDiagnostics`] — static-arbitrage findings on an implied
//!   vol surface (butterfly / calendar), from
//!   [`VolSurface::diagnostics`](crate::core::vols::VolSurface::diagnostics);
//! - [`RepairReport`] — what
//!   [`repair_arbitrage`](crate::equity::surface_repair::repair_arbitrage)
//!   changed;
//! - [`UsabilityReport`] — local-vol round-trip repricing, clamp and
//!   guard-fallback fractions, trusted region;
//! - VaR backtesting lives in [`risk`](crate::risk).

pub mod martingale;

pub use crate::core::vols::{ButterflyViolation, CalendarViolation, SurfaceDiagnostics};
pub use crate::equity::models::surface_repair::RepairReport;
pub use crate::equity::models::usability::{usability_report, UsabilityConfig, UsabilityReport};
pub use martingale::{martingale_report, MartingaleCheck, MartingaleConfig, MartingaleReport};
