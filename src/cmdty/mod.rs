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
//! April-2020 dislocation).

pub mod apo;
pub mod bachelier;
pub mod basis_swap;
pub mod forward_curve;
pub mod option;
pub mod swap;
pub mod vol;

pub use apo::AveragePriceOption;
pub use basis_swap::CommodityBasisSwap;
pub use forward_curve::CommodityForwardCurve;
pub use option::{CommodityOption, FuturesSettlement};
pub use swap::{CommoditySwap, PriceFixings};
pub use vol::CommodityVol;
