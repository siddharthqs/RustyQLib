//! How it gets priced: the pricing methods for rate options.
//!
//! - [`jamshidian`] — the analytic engine for one-factor affine
//!   short-rate models: zero-bond options composed into coupon-bond
//!   options, European swaptions, caplets and floorlets.
//! - [`black`] — the market-quote formulas: Black-76 and Bachelier on
//!   the forward rate times the annuity, and the implied-vol
//!   inversions that read a premium back as a quote.
//! - [`hw_grid`] — backward induction on the Hull-White state grid for
//!   Bermudan exercise: Bermudan swaptions and callable bonds.
//!
//! Linear products (swaps, futures) need no engine: they discount off
//! [`YieldCurve`](crate::core::curves::YieldCurve)s directly.

pub mod black;
pub mod fd_g2pp;
pub mod fd_hull_white;
pub mod gaussian1d;
pub mod hw_grid;
pub mod jamshidian;
pub mod mc_hull_white;

pub use black::{implied_black_vol, implied_normal_vol, swaption_from_vol, RateVol, RateVolKind};
pub use fd_g2pp::FdG2Config;
pub use fd_hull_white::FdConfig;
pub use hw_grid::GridConfig;
pub use jamshidian::{caplet, coupon_bond_option, european_swaption, floorlet};
pub use mc_hull_white::{McConfig, McResult};
