//! # RustyQLib
//!
//! A lightweight quantitative finance library for pricing derivatives and
//! performing risk analysis.
//!
//! The crate is organised into asset-class modules:
//!
//! - [`core`] — shared building blocks: traits ([`core::traits::Instrument`]),
//!   quotes, term structures, interpolation and data models
//! - [`equity`] — equity options, forwards and futures with Black-Scholes,
//!   binomial, Monte Carlo and finite-difference engines
//! - [`rates`] — interest-rate instruments (deposits, FRAs) and curve building
//! - [`risk`] — VaR / Expected Shortfall, portfolio scenario risk, volatility
//!   estimation, performance statistics and VaR backtesting
//! - [`cmdty`] — commodity options
//! - [`utils`] — random number generation, stochastic processes and the
//!   JSON/CLI plumbing used by the `rustyqlib` binary
//!
//! # Example
//!
//! Pricing contracts from JSON is the primary workflow (see the `examples/`
//! directory in the repository); the same types can be constructed directly
//! and priced through the [`core::traits::Instrument`] trait.

pub mod cmdty;
pub mod core;
pub mod equity;
pub mod rates;
pub mod risk;
pub mod utils;

pub use crate::core::curves::{Compounding, CurveInput, InterpolationMethod, Tenor, YieldCurve};
pub use crate::core::errors::RustyQLibError;
pub use crate::equity::black76::FuturesSettlement;
pub use crate::equity::builder::EquityOptionBuilder;
pub use crate::core::daycount::DayCountConvention;
pub use crate::core::traits::Instrument;
pub use crate::core::vols::{VolInput, VolSurface};
