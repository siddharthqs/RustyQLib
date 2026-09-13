//! What you can trade: the rate products and their product logic.
//!
//! Linear:
//! - [`VanillaSwap`] — fixed-for-floating interest rate swap
//! - [`OvernightIndexSwap`] — fixed versus daily-compounded overnight
//! - [`BasisSwap`] — floating-for-floating with a spread
//! - [`FedFundsFuture`], [`SofrFuture`] — money-market futures
//!
//! Optional (priced under a short-rate model from [`models`]):
//! - [`Swaption`] — European option to enter a [`VanillaSwap`]
//! - [`BermudanSwaption`] — the same right on any of several dates, on
//!   the Hull-White grid
//! - [`CapFloor`] — a strip of caplets or floorlets over a schedule
//!
//! Every product owns its dates and conventions; the optional products
//! map them to the year fractions the engines work in against an
//! anchor curve's reference date and day count.
//!
//! [`models`]: crate::rates::models

pub mod basis_swap;
pub mod bermudan_swaption;
pub mod cap_floor;
pub mod fed_funds_future;
pub mod ois;
pub mod sofr_future;
pub mod swaption;
pub mod vanilla_swap;

pub use basis_swap::{BasisSwap, BasisSwapLeg};
pub use bermudan_swaption::BermudanSwaption;
pub use cap_floor::{CapFloor, CapOrFloor, CapletValue};
pub use fed_funds_future::FedFundsFuture;
pub use ois::OvernightIndexSwap;
pub use sofr_future::{hull_convexity_adjustment, SofrContract, SofrFuture};
pub use swaption::Swaption;
pub use vanilla_swap::VanillaSwap;
