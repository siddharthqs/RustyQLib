//! # RustyQLib
//!
//! A lightweight quantitative finance library for pricing derivatives and
//! performing risk analysis.
//!
//! The crate is organised into asset-class modules:
//!
//! - [`core`] — shared building blocks: traits ([`core::traits::Instrument`]),
//!   quotes, discount curves, calendars, interpolation and data models
//! - [`equity`] — equity options, forwards and futures with Black-Scholes,
//!   binomial, Monte Carlo and finite-difference engines
//! - [`bonds`] — fixed income: US Treasury notes/bonds and bills with
//!   street-convention analytics (accrued interest, price/yield,
//!   duration, convexity, DV01), money-market instruments (deposits,
//!   FRAs) and discount-curve bootstrapping
//! - [`rates`] — linear interest-rate products: vanilla swaps (IRS),
//!   overnight indexed swaps (OIS) and basis swaps, priced by dual-curve
//!   discounting
//! - [`risk`] — VaR / Expected Shortfall, portfolio scenario risk, volatility
//!   estimation, performance statistics and VaR backtesting
//! - [`cmdty`] — commodity options
//! - `data` *(feature `fetch`)* — free official end-of-day market data:
//!   the US Treasury daily par yield curve, passed through as published
//!   with provenance metadata
//! - [`utils`] — random number generation, stochastic processes and the
//!   JSON/CLI plumbing used by the `rustyqlib` binary
//!
//! # Example
//!
//! Pricing contracts from JSON is the primary workflow (see the `examples/`
//! directory in the repository); the same types can be constructed directly
//! and priced through the [`core::traits::Instrument`] trait.

pub mod bonds;
pub mod cmdty;
pub mod core;
#[cfg(feature = "fetch")]
pub mod data;
pub mod equity;
pub mod rates;
pub mod risk;
pub mod utils;

pub use crate::bonds::{
    bootstrap_curve, conversion_factor, BillQuote, BondFuture, BondQuote, CurveInstrument,
    DeliverableBond, Deposit, FactorRounding, FixedRateBond, Fra, Frequency, TreasuryBill,
};
pub use crate::core::calendar::{
    BusinessDayConvention, Calendar, DateGeneration, Period, Schedule,
};
pub use crate::core::curves::{
    Compounding, CurveInput, InterpolationMethod, RateShift, Tenor, YieldCurve,
};
pub use crate::core::daycount::DayCountConvention;
pub use crate::core::depth::{DepthLevel, MarketDepth};
pub use crate::core::errors::RustyQLibError;
pub use crate::core::market::{
    BumpMode, Depth, Discount, Market, MarketKey, RiskFactor, Shock, Spot, Vol,
};
pub use crate::core::quotes::Quote;
pub use crate::core::results::{Greeks, PricingResult};
pub use crate::core::traits::Instrument;
pub use crate::core::vols::{VolInput, VolSurface};
pub use crate::equity::black76::FuturesSettlement;
pub use crate::equity::builder::EquityOptionBuilder;
pub use crate::rates::{
    BasisSwap, BasisSwapLeg, FedFundsFuture, OvernightIndexSwap, PayerReceiver, RateFixings,
    SofrContract, SofrFuture, VanillaSwap,
};
