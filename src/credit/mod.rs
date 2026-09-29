//! Credit: the hazard-rate curve, credit default swaps and their
//! calibration.
//!
//! [`CreditCurve`] is the reduced-form term structure — piecewise-
//! constant default intensity — shared by the risky bond pricers in
//! [`bonds::credit`](crate::bonds::credit), the jump-to-default
//! convertible in [`hybrid`](crate::hybrid) and the swaps here.
//! [`CreditDefaultSwap`] prices the two legs on it with the ISDA
//! standard model's integrals, gives par spreads, points upfront and
//! the flat-hazard conversion between them; [`bootstrap_cds_curve`]
//! builds the curve from par-spread quotes. [`CdsOption`] is the
//! European option on that swap: forward legs read off the same
//! integrals, front-end protection for the no-knockout convention, and
//! Black or Bachelier on the loss-adjusted forward spread.
//!
//! This module depends on [`core`](crate::core), and on
//! [`rates::engines::black`](crate::rates::engines::black) for the
//! shared rate-option kernel and its implied-vol inversion; the bond
//! and hybrid modules depend on it.

pub mod bootstrap;
pub mod cds;
pub mod cds_option;
pub mod curve;

pub use bootstrap::{bootstrap_cds_curve, CdsQuote};
pub use cds::{CreditDefaultSwap, ProtectionSide};
pub use cds_option::CdsOption;
pub use curve::CreditCurve;
