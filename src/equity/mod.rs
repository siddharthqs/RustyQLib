//! Equity derivatives: contracts, pricing engines, models, and the
//! shared spine that binds them.
//!
//! - [`contracts`] — what you can trade (payoffs and product logic)
//! - [`engines`] — how it gets priced (numerical and analytic methods)
//! - [`models`] — dynamics and volatility structure
//! - [`service`] — JSON batch entry points
//! - top level — the cross-cutting spine: [`market`], [`builder`],
//!   [`greeks`], [`portfolio`], [`utils`]
//!
//! Every submodule is also re-exported at this level, so the historical
//! flat paths (`equity::vanilla_option`, `equity::montecarlo`, …) keep
//! working unchanged.

pub mod builder;
pub mod bump;
pub mod greeks;
pub mod market;
pub mod portfolio;
pub mod utils;

pub mod contracts;
pub mod engines;
pub mod models;
pub mod service;

// Flat-path compatibility re-exports.
pub use contracts::{
    accumulator, asian, autocallable, barrier, binary_option, chooser, cliquet, equity_forward,
    equity_future, equity_option, forward_start_option, lookback, perpetual, rainbow,
    vanilla_option, variance_swap, worst_of,
};
pub use engines::{
    baw, binomial, bjerksund_stensland, black76, blackscholes, carr_madan, cos, finite_difference,
    heston_adi, montecarlo,
};
pub use models::{bates, heston, local_vol, processes, rbergomi, slv, svi, vol_surface};
pub use service::{build_contracts, handle_equity_contracts};
