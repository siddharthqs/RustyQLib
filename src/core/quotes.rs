//! Price observations: the scalar market observable pricing consumes.
//!
//! A [`Quote`] is what the pricing layer reads — a mark. It is
//! deliberately **not** an order book: trading systems keep the book
//! (ladders, order-by-order churn) in the execution layer and hand
//! pricing a derived observable. The book-shaped sibling is
//! [`MarketDepth`](crate::core::depth::MarketDepth), which lives under
//! its own market key and *produces* a `Quote` at the snapshot boundary.
//!
//! Derived values (`mid`, `spread`, `microprice`) are **functions, never
//! fields** — a stored mid can disagree with the bid/ask it came from;
//! a computed one cannot. Representation is private by construction
//! (enum variants are matched, not assigned), so richer quote shapes can
//! be added without touching pricing call sites: engines only call
//! [`mid`](Quote::mid).

use crate::core::errors::{Result, RustyQLibError};

/// One price observation for one instrument, in increasing order of
/// structure. Pricing reads [`mid`](Quote::mid) regardless of variant;
/// the richer variants exist so marking policies (bid/ask-side marks,
/// microprice) have honest inputs when a trading system supplies them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quote {
    /// A single price with no book behind it: a mark, settle, close or
    /// model price.
    Mid(f64),
    /// Best bid and offer. `mid` and `spread` are derived on demand.
    TopOfBook { bid: f64, ask: f64 },
    /// Best bid and offer with displayed sizes; enables the size-weighted
    /// [`microprice`](Quote::microprice).
    Sized { bid: f64, bid_size: f64, ask: f64, ask_size: f64 },
}

fn require_uncrossed(bid: f64, ask: f64) -> Result<()> {
    if !bid.is_finite() || !ask.is_finite() {
        return Err(RustyQLibError::invalid_input(
            "quote",
            format!("bid/ask must be finite, got bid={bid}, ask={ask}"),
        ));
    }
    if bid > ask {
        return Err(RustyQLibError::invalid_input(
            "quote",
            format!("crossed quote: bid {bid} > ask {ask}"),
        ));
    }
    Ok(())
}

impl Quote {
    /// A bare mid — the compatibility constructor (marks, settles, model
    /// prices, placeholders). Performs no validation; use
    /// [`from_bid_ask`](Self::from_bid_ask) for feed data.
    pub fn new(mid: f64) -> Self {
        Quote::Mid(mid)
    }

    /// Top of book. Rejects non-finite and crossed (`bid > ask`) input —
    /// crossed books are a feed problem to resolve upstream, not a state
    /// to price off.
    pub fn from_bid_ask(bid: f64, ask: f64) -> Result<Self> {
        require_uncrossed(bid, ask)?;
        Ok(Quote::TopOfBook { bid, ask })
    }

    /// Top of book with displayed sizes (both strictly positive and
    /// finite), enabling [`microprice`](Self::microprice).
    pub fn from_bid_ask_sized(bid: f64, bid_size: f64, ask: f64, ask_size: f64) -> Result<Self> {
        require_uncrossed(bid, ask)?;
        if !(bid_size.is_finite() && ask_size.is_finite() && bid_size > 0.0 && ask_size > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "quote",
                format!("sizes must be finite and positive, got bid_size={bid_size}, ask_size={ask_size}"),
            ));
        }
        Ok(Quote::Sized { bid, bid_size, ask, ask_size })
    }

    /// The mark pricing uses: the price itself for [`Mid`](Quote::Mid),
    /// the arithmetic bid/ask midpoint otherwise.
    pub fn mid(&self) -> f64 {
        match *self {
            Quote::Mid(value) => value,
            Quote::TopOfBook { bid, ask } | Quote::Sized { bid, ask, .. } => 0.5 * (bid + ask),
        }
    }

    /// Alias for [`mid`](Self::mid), kept for source compatibility with
    /// the field-based `Quote`.
    pub fn value(&self) -> f64 {
        self.mid()
    }

    /// Best bid, when a book side was observed.
    pub fn bid(&self) -> Option<f64> {
        match *self {
            Quote::Mid(_) => None,
            Quote::TopOfBook { bid, .. } | Quote::Sized { bid, .. } => Some(bid),
        }
    }

    /// Best ask, when a book side was observed.
    pub fn ask(&self) -> Option<f64> {
        match *self {
            Quote::Mid(_) => None,
            Quote::TopOfBook { ask, .. } | Quote::Sized { ask, .. } => Some(ask),
        }
    }

    /// `ask - bid`, when both sides were observed.
    pub fn spread(&self) -> Option<f64> {
        match *self {
            Quote::Mid(_) => None,
            Quote::TopOfBook { bid, ask } | Quote::Sized { bid, ask, .. } => Some(ask - bid),
        }
    }

    /// The size-weighted mid `(bid * ask_size + ask * bid_size) /
    /// (bid_size + ask_size)` — the standard short-horizon fair-value
    /// estimator (leans toward the side with less displayed size). Falls
    /// back to [`mid`](Self::mid) when sizes are not available.
    pub fn microprice(&self) -> f64 {
        match *self {
            Quote::Sized { bid, bid_size, ask, ask_size } => {
                (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
            }
            _ => self.mid(),
        }
    }

    /// Whether the mark is a usable positive price.
    pub fn valid_value(&self) -> bool {
        self.mid() > 0.0
    }

    /// Every price level scaled by `factor` (a relative bump: spread and
    /// levels scale together, sizes are untouched). The quote shape is
    /// preserved — bumping a scenario must not silently discard book
    /// information.
    pub fn scaled(&self, factor: f64) -> Quote {
        match *self {
            Quote::Mid(value) => Quote::Mid(value * factor),
            Quote::TopOfBook { bid, ask } => {
                Quote::TopOfBook { bid: bid * factor, ask: ask * factor }
            }
            Quote::Sized { bid, bid_size, ask, ask_size } => {
                Quote::Sized { bid: bid * factor, bid_size, ask: ask * factor, ask_size }
            }
        }
    }

    /// Every price level shifted by `delta` (an absolute bump: the spread
    /// is preserved exactly, sizes are untouched).
    pub fn shifted(&self, delta: f64) -> Quote {
        match *self {
            Quote::Mid(value) => Quote::Mid(value + delta),
            Quote::TopOfBook { bid, ask } => {
                Quote::TopOfBook { bid: bid + delta, ask: ask + delta }
            }
            Quote::Sized { bid, bid_size, ask, ask_size } => {
                Quote::Sized { bid: bid + delta, bid_size, ask: ask + delta, ask_size }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mid_is_derived_never_stored() {
        assert_eq!(Quote::new(100.0).mid(), 100.0);
        let top = Quote::from_bid_ask(99.0, 101.0).unwrap();
        assert_eq!(top.mid(), 100.0);
        assert_eq!(top.spread(), Some(2.0));
        assert_eq!(top.bid(), Some(99.0));
        assert_eq!(top.ask(), Some(101.0));
        // the compatibility alias agrees
        assert_eq!(top.value(), top.mid());
        // a bare mid has no book
        let mark = Quote::new(100.0);
        assert_eq!(mark.bid(), None);
        assert_eq!(mark.spread(), None);
    }

    #[test]
    fn constructors_reject_crossed_and_non_finite_input() {
        assert!(Quote::from_bid_ask(101.0, 99.0).is_err(), "crossed");
        assert!(Quote::from_bid_ask(f64::NAN, 100.0).is_err());
        assert!(Quote::from_bid_ask(99.0, f64::INFINITY).is_err());
        // touched (bid == ask) is legal: a locked book still has a mid
        assert!(Quote::from_bid_ask(100.0, 100.0).is_ok());
        assert!(Quote::from_bid_ask_sized(99.0, 0.0, 101.0, 5.0).is_err(), "zero size");
        assert!(Quote::from_bid_ask_sized(99.0, 5.0, 101.0, -1.0).is_err());
    }

    #[test]
    fn microprice_weights_toward_the_thin_side() {
        // 4x the size on the bid: fair value leans toward the ask
        let quote = Quote::from_bid_ask_sized(99.0, 400.0, 101.0, 100.0).unwrap();
        let micro = quote.microprice();
        assert!((micro - (99.0 * 100.0 + 101.0 * 400.0) / 500.0).abs() < 1e-12);
        assert!(micro > quote.mid(), "heavy bid pushes fair value up");
        // without sizes it degrades to the mid
        assert_eq!(Quote::from_bid_ask(99.0, 101.0).unwrap().microprice(), 100.0);
        assert_eq!(Quote::new(100.0).microprice(), 100.0);
    }

    #[test]
    fn bumps_preserve_shape_spread_and_sizes() {
        let quote = Quote::from_bid_ask_sized(99.0, 400.0, 101.0, 100.0).unwrap();
        let scaled = quote.scaled(0.8);
        assert!((scaled.mid() - 80.0).abs() < 1e-12);
        assert!((scaled.spread().unwrap() - 1.6).abs() < 1e-12, "relative bump scales the spread");
        let shifted = quote.shifted(-20.0);
        assert!((shifted.mid() - 80.0).abs() < 1e-12);
        assert!((shifted.spread().unwrap() - 2.0).abs() < 1e-12, "absolute bump preserves the spread");
        // sizes ride through both, and the variant is unchanged
        match (scaled, shifted) {
            (
                Quote::Sized { bid_size: s1, ask_size: a1, .. },
                Quote::Sized { bid_size: s2, ask_size: a2, .. },
            ) => {
                assert_eq!((s1, a1), (400.0, 100.0));
                assert_eq!((s2, a2), (400.0, 100.0));
            }
            other => panic!("bump must preserve the quote shape, got {other:?}"),
        }
        // a bare mid bumps as a scalar
        assert!((Quote::new(100.0).scaled(1.1).mid() - 110.0).abs() < 1e-12);
        assert!((Quote::new(100.0).shifted(-1.0).mid() - 99.0).abs() < 1e-12);
    }
}
