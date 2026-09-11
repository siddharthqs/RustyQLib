//! How it gets priced: the pricing methods for rate options.
//!
//! - [`jamshidian`] — the analytic engine for one-factor affine
//!   short-rate models: zero-bond options composed into coupon-bond
//!   options, European swaptions, caplets and floorlets.
//! - [`black`] — the market-quote formulas: Black-76 and Bachelier on
//!   the forward rate times the annuity, and the implied-vol
//!   inversions that read a premium back as a quote.
//!
//! Linear products (swaps, futures) need no engine: they discount off
//! [`YieldCurve`](crate::core::curves::YieldCurve)s directly.

pub mod black;
pub mod jamshidian;

pub use black::{implied_black_vol, implied_normal_vol, swaption_from_vol, RateVol};
pub use jamshidian::{caplet, coupon_bond_option, european_swaption, floorlet};
