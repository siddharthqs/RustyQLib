//! Commodity derivatives.
//!
//! - [`CommoditySwap`] — fixed-for-floating commodity swap: each
//!   calculation period cash-settles the arithmetic average of the
//!   daily index price over the period's business days against a fixed
//!   price, times the period's notional quantity
//! - [`CommodityBasisSwap`] — floating-for-floating differential
//!   between two indexes (WTI–Brent, location or quality basis), each
//!   leg averaging over its own pricing calendar, with a fixed spread
//!   on the received leg
//! - [`CommodityOption`] — European option on a commodity future,
//!   priced with Black-76 (premium up front or futures-style margined)
//! - [`AveragePriceOption`] — Asian-style APO on the same averaged
//!   index as the swap, priced by discrete moment matching (Levy) with
//!   realized fixings folded into an adjusted strike
//! - [`CommoditySwaption`] — European option to enter a
//!   [`CommoditySwap`]: Black-on-par times the settlement annuity,
//!   exact under the one-factor flat-vol dynamics
//! - [`CommoditySpreadOption`] — European option on the spread of two
//!   futures (crack, spark, location, calendar), Kirk's approximation
//!   for lognormal legs and exact Bachelier for normal legs
//! - [`CommodityForwardCurve`] — the strip of forward prices both
//!   products are projected from, linearly interpolated between pillar
//!   dates
//!
//! Discounting comes from a [`crate::core::curves::YieldCurve`];
//! partially realized swap periods blend published [`PriceFixings`]
//! with curve forwards, mirroring the money-market futures in
//! [`crate::rates`].
//!
//! Options take their distribution model from the vol quote
//! ([`CommodityVol`]): Black-76 lognormal (the default — a bare `f64`
//! vol), shifted lognormal, or Bachelier normal ([`bachelier`]) for
//! underlyings that can print negative (Waha/AECO basis, WTI in an
//! April-2020 dislocation). A [`ShiftedSabr`] smile generates the
//! shifted-lognormal quote per strike, so whole smiles price
//! consistently through the same dispatch; a [`ClewlowStrickland`]
//! model does the same across maturities (the Samuelson effect), and
//! additionally supplies the cross-maturity covariances the APO and
//! swaption use in their `price_cs` variants.

pub mod apo;
pub mod bachelier;
pub mod basis_swap;
pub mod clewlow_strickland;
pub mod forward_curve;
pub mod option;
pub mod sabr;
pub mod spread_option;
pub mod swap;
pub mod swaption;
pub mod vol;

pub use apo::AveragePriceOption;
pub use basis_swap::CommodityBasisSwap;
pub use clewlow_strickland::{ClewlowStrickland, ClewlowStricklandFit};
pub use forward_curve::CommodityForwardCurve;
pub use option::{CommodityOption, FuturesSettlement};
pub use sabr::{ShiftedSabr, ShiftedSabrFit};
pub use spread_option::CommoditySpreadOption;
pub use swap::{CommoditySwap, PriceFixings};
pub use swaption::CommoditySwaption;
pub use vol::CommodityVol;
