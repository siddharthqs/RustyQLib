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
//! - [`cmdty`] — commodity swaps (fixed versus the averaged daily index
//!   price) and Black-76 options on commodity futures, projected on
//!   commodity forward curves
//! - `data` *(feature `fetch`)* — free official end-of-day market data:
//!   the US Treasury daily par yield curve, passed through as published
//!   with provenance metadata
//! - [`validation`] — runtime model-validation checks (martingale
//!   forward recovery, surface diagnostics, local-vol usability) that
//!   measure model quality on given data and ship as reports
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
pub mod validation;

pub use crate::bonds::{
    bootstrap_credit_curve, bootstrap_curve, conversion_factor, g_spread, BillQuote, BondFuture,
    BondOptionality, BondQuote, CallOption, ConvertibleBond, ConvertibleMarket, CreditCurve,
    CurveInstrument, DeliverableBond, Deposit, FactorRounding, FixedRateBond, FloatingRateNote,
    Fra, Frequency, MakeWholeCall, PutOption, TreasuryBill,
};
pub use crate::cmdty::{
    AveragePriceOption, ClewlowStrickland, ClewlowStricklandFit, CommodityBasisSwap,
    CommodityForwardCurve, CommodityOption, CommoditySpreadOption, CommoditySwap,
    CommoditySwaption, CommodityVol, PriceFixings, ShiftedSabr, ShiftedSabrFit,
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
pub use crate::core::vols::{SmileCoordinate, VolInput, VolSurface, VolSurfaceDocument};
pub use crate::equity::black76::FuturesSettlement;
pub use crate::equity::builder::EquityOptionBuilder;
pub use crate::equity::multi_asset::{
    AssetLeg, MultiAssetEquityOption, MultiAssetEquityOptionBuilder, MultiAssetMarketData,
};
pub use crate::equity::option_chain::{
    implied_vol_surface_from_chain, FilterConfig, OptionChain, OptionQuote, SurfaceBuildReport,
};
pub use crate::equity::surface_repair::{repair_arbitrage, RepairReport};
pub use crate::rates::{
    BasisSwap, BasisSwapLeg, CoxIngersollRoss, FedFundsFuture, HullWhite, OneFactorAffine,
    OvernightIndexSwap, PayerReceiver, RateFixings, ShortRateModel, SofrContract, SofrFuture,
    VanillaSwap, Vasicek,
};
pub use crate::validation::martingale::{martingale_report, MartingaleConfig, MartingaleReport};
