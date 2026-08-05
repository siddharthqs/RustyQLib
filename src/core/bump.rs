//! Scalar market bumps: the **one** definition of "reprice under a shifted
//! market" that every engine consumes.
//!
//! A [`Bump`] says *what moved* — spot, parallel implied vol, parallel
//! zero rate, elapsed calendar time — without knowing who consumes it.
//! Consumers pair it with market data (the equity side wraps both in
//! [`BumpedMarket`](crate::equity::bump::BumpedMarket)) and engines read
//! the shifted values from that pairing; no engine applies a shift itself.
//!
//! Sign and unit conventions (fixed here, nowhere else):
//! - `d_spot`: absolute shift of the spot (or futures) level;
//! - `d_vol`: absolute parallel shift of implied volatility (0.01 = 1 pt);
//! - `d_rate`: absolute parallel shift of the zero curve (0.0001 = 1 bp);
//! - `d_time`: **elapsed calendar time in years** — positive means time
//!   passes and the remaining maturity shrinks.
//!
//! Relation to the object-level scenario store: [`Market::bumped`]
//! (crate::core::market::Market::bumped) applies typed [`Shock`]
//! (crate::core::market::Shock)s to real market *objects* (quotes, curves,
//! surfaces) for stress and scenario work. `Bump` is the scalar fast path
//! underneath Greeks stencils and PnL attribution: no allocation, engine
//! bit-identity, sub-day time shifts.

use serde::{Deserialize, Serialize};

/// An additive market scenario; see the module docs for conventions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bump {
    /// Absolute spot shift.
    pub d_spot: f64,
    /// Parallel implied-vol shift.
    pub d_vol: f64,
    /// Parallel zero-rate shift.
    pub d_rate: f64,
    /// Elapsed calendar time in years (positive shortens maturity).
    pub d_time: f64,
}

impl Bump {
    /// No shift: pricing under `NONE` is the base price, bit for bit.
    pub const NONE: Bump = Bump {
        d_spot: 0.0,
        d_vol: 0.0,
        d_rate: 0.0,
        d_time: 0.0,
    };

    pub fn spot(d_spot: f64) -> Bump {
        Bump {
            d_spot,
            ..Bump::NONE
        }
    }
    pub fn vol(d_vol: f64) -> Bump {
        Bump {
            d_vol,
            ..Bump::NONE
        }
    }
    pub fn rate(d_rate: f64) -> Bump {
        Bump {
            d_rate,
            ..Bump::NONE
        }
    }
    /// Elapsed calendar time (positive = time passes).
    pub fn elapsed(d_time: f64) -> Bump {
        Bump {
            d_time,
            ..Bump::NONE
        }
    }

    pub fn is_zero(&self) -> bool {
        *self == Bump::NONE
    }
}

impl Default for Bump {
    fn default() -> Self {
        Bump::NONE
    }
}
