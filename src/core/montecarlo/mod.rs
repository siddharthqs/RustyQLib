//! Asset-agnostic Monte Carlo machinery, one concern per file — usable
//! as a standalone simulation toolkit and consumed by the equity pricers.
//!
//! Nothing in this module knows about options, spots or curves: it deals
//! in uniforms, normals, Brownian increments and estimator statistics,
//! so rates, FX, commodity or credit simulations plug in the same way
//! equities do.
//!
//! - [`rng`]: deterministic pseudo-random generation — SplitMix64 stream
//!   derivation and per-path PCG64 streams (bit-reproducible under any
//!   thread scheduling), plus seeded standard-normal draws;
//! - [`sobol`]: multi-dimensional **Sobol** low-discrepancy sequences
//!   (Gray-code, direction numbers, optional seeded digital-shift
//!   scrambling), and the 1-D van der Corput normals;
//! - [`halton`]: Halton sequences with Cranley-Patterson rotation — the
//!   arbitrary-dimension quasi-random fallback;
//! - [`brownian_bridge`]: Brownian-bridge path construction, so the
//!   best low-discrepancy coordinates carry each path's coarse
//!   structure;
//! - [`variance_reduction`]: antithetic pairing, moment matching, and
//!   the generic regression-based control-variate estimator;
//! - [`sampling`]: stratified sampling and Latin hypercube designs;
//! - [`stats`]: simulation statistics — mean / standard error from
//!   accumulated sums and a Welford running accumulator.

pub mod brownian_bridge;
pub mod halton;
pub mod rng;
pub mod sampling;
pub mod sobol;
pub mod stats;
pub mod variance_reduction;

pub use brownian_bridge::BrownianBridge;
pub use halton::QmcSequence;
pub use rng::{path_normals, path_rng, pseudo_normal_matrix, pseudo_normals, splitmix64};
pub use sampling::{latin_hypercube, stratified_normals, stratified_uniforms};
pub use sobol::{sobol_normals, SobolSequence};
pub use stats::{mean_std_err, RunningStats, SimStats};
pub use variance_reduction::{control_variate_estimate, moment_match};
