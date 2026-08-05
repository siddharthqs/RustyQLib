//! Limit-order-book depth: the **execution-layer** observable.
//!
//! Pricing never walks a ladder — it reads a [`Quote`] mark. Execution
//! analytics (slippage, liquidation value, sizing) do walk it. Keeping
//! [`MarketDepth`] as a sibling of [`Quote`] under its own market key
//! ([`Depth`](crate::core::market::Depth)) preserves that layering: a
//! trading system publishes the book here and the top-of-book collapses
//! into a [`Quote`] at the snapshot boundary ([`to_quote`]
//! (MarketDepth::to_quote)); the pricing layer never changes.
//!
//! Scenario shocks do not rewrite ladders: bump the pricing observables
//! ([`Spot`](crate::core::market::Spot)), let the adapter republish depth.

use crate::core::errors::{Result, RustyQLibError};
use crate::core::quotes::Quote;
use crate::core::trade::Transection;

/// One book level: a price and the size displayed at it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthLevel {
    pub price: f64,
    pub size: f64,
}

/// A price-aggregated (L2) snapshot of one instrument's book: bids best
/// (highest) first, asks best (lowest) first. Validated on construction —
/// sorted sides, positive finite sizes, uncrossed top — so consumers can
/// walk it without re-checking.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketDepth {
    bids: Vec<DepthLevel>,
    asks: Vec<DepthLevel>,
}

fn validate_side(levels: &[DepthLevel], side: &str, descending: bool) -> Result<()> {
    for level in levels {
        if !level.price.is_finite() || !level.size.is_finite() || level.size <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "market depth",
                format!("{side} level must have finite price and positive size, got {level:?}"),
            ));
        }
    }
    let ordered = levels.windows(2).all(|w| {
        if descending {
            w[1].price < w[0].price
        } else {
            w[1].price > w[0].price
        }
    });
    if !ordered {
        return Err(RustyQLibError::invalid_input(
            "market depth",
            format!("{side} levels must be strictly best-first, got {levels:?}"),
        ));
    }
    Ok(())
}

impl MarketDepth {
    /// Build a validated book. Either side may be empty (a one-sided
    /// market); a crossed top (`best_bid > best_ask`) is rejected.
    pub fn new(bids: Vec<DepthLevel>, asks: Vec<DepthLevel>) -> Result<Self> {
        validate_side(&bids, "bid", true)?;
        validate_side(&asks, "ask", false)?;
        if let (Some(bid), Some(ask)) = (bids.first(), asks.first()) {
            if bid.price > ask.price {
                return Err(RustyQLibError::invalid_input(
                    "market depth",
                    format!(
                        "crossed book: best bid {} > best ask {}",
                        bid.price, ask.price
                    ),
                ));
            }
        }
        Ok(MarketDepth { bids, asks })
    }

    pub fn bids(&self) -> &[DepthLevel] {
        &self.bids
    }

    pub fn asks(&self) -> &[DepthLevel] {
        &self.asks
    }

    pub fn best_bid(&self) -> Option<DepthLevel> {
        self.bids.first().copied()
    }

    pub fn best_ask(&self) -> Option<DepthLevel> {
        self.asks.first().copied()
    }

    /// Top-of-book midpoint, when both sides exist.
    pub fn mid(&self) -> Option<f64> {
        Some(0.5 * (self.best_bid()?.price + self.best_ask()?.price))
    }

    /// Volume-weighted average execution price for `quantity`, walking
    /// the book: a `Buy` lifts asks, a `Sell` hits bids. `None` when the
    /// displayed depth cannot fill the quantity (or it is not positive) —
    /// the honest answer, not an extrapolation.
    pub fn vwap_for_size(&self, side: Transection, quantity: f64) -> Option<f64> {
        if quantity <= 0.0 || !quantity.is_finite() {
            return None;
        }
        let levels = match side {
            Transection::Buy => &self.asks,
            Transection::Sell => &self.bids,
        };
        let mut remaining = quantity;
        let mut cost = 0.0;
        for level in levels {
            let fill = remaining.min(level.size);
            cost += fill * level.price;
            remaining -= fill;
            if remaining <= 0.0 {
                return Some(cost / quantity);
            }
        }
        None
    }

    /// Slippage of filling `quantity` versus marking at mid: `vwap - mid`
    /// for a buy, `mid - vwap` for a sell (so it is a cost when positive).
    pub fn slippage_for_size(&self, side: Transection, quantity: f64) -> Option<f64> {
        let mid = self.mid()?;
        let vwap = self.vwap_for_size(side.clone(), quantity)?;
        Some(match side {
            Transection::Buy => vwap - mid,
            Transection::Sell => mid - vwap,
        })
    }

    /// Collapse to the pricing observable: a sized top-of-book [`Quote`],
    /// when both sides exist. This is the snapshot-boundary conversion a
    /// trading adapter runs to feed the pricing layer.
    pub fn to_quote(&self) -> Option<Quote> {
        let bid = self.best_bid()?;
        let ask = self.best_ask()?;
        Quote::from_bid_ask_sized(bid.price, bid.size, ask.price, ask.size).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn level(price: f64, size: f64) -> DepthLevel {
        DepthLevel { price, size }
    }

    fn book() -> MarketDepth {
        MarketDepth::new(
            vec![level(99.0, 100.0), level(98.5, 200.0), level(98.0, 500.0)],
            vec![
                level(101.0, 150.0),
                level(101.5, 300.0),
                level(102.0, 400.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn construction_validates_ordering_sizes_and_crossing() {
        // unsorted bids (must be descending)
        assert!(MarketDepth::new(vec![level(98.0, 1.0), level(99.0, 1.0)], vec![]).is_err());
        // unsorted asks (must be ascending)
        assert!(MarketDepth::new(vec![], vec![level(102.0, 1.0), level(101.0, 1.0)]).is_err());
        // crossed top
        assert!(MarketDepth::new(vec![level(101.5, 1.0)], vec![level(101.0, 1.0)]).is_err());
        // non-positive size
        assert!(MarketDepth::new(vec![level(99.0, 0.0)], vec![]).is_err());
        // one-sided and empty books are legal
        assert!(MarketDepth::new(vec![level(99.0, 1.0)], vec![]).is_ok());
        assert!(MarketDepth::new(vec![], vec![]).is_ok());
        assert_eq!(book().mid(), Some(100.0));
    }

    #[test]
    fn vwap_walks_the_ladder_and_refuses_to_extrapolate() {
        let depth = book();
        // inside the top level: pay the touch
        assert_eq!(depth.vwap_for_size(Transection::Buy, 150.0), Some(101.0));
        // 300 lifts 150 @ 101 + 150 @ 101.5
        let vwap = depth.vwap_for_size(Transection::Buy, 300.0).unwrap();
        assert!((vwap - (150.0 * 101.0 + 150.0 * 101.5) / 300.0).abs() < 1e-12);
        // selling walks the bid side
        let sell = depth.vwap_for_size(Transection::Sell, 250.0).unwrap();
        assert!((sell - (100.0 * 99.0 + 150.0 * 98.5) / 250.0).abs() < 1e-12);
        // more than the displayed book: None, not a guess
        assert_eq!(depth.vwap_for_size(Transection::Buy, 1_000.0), None);
        assert_eq!(depth.vwap_for_size(Transection::Buy, 0.0), None);
        // slippage is a positive cost on both sides of this book
        assert!(depth.slippage_for_size(Transection::Buy, 300.0).unwrap() > 0.0);
        assert!(depth.slippage_for_size(Transection::Sell, 250.0).unwrap() > 0.0);
    }

    #[test]
    fn to_quote_collapses_the_top_of_book_for_pricing() {
        let quote = book().to_quote().unwrap();
        assert_eq!(quote.bid(), Some(99.0));
        assert_eq!(quote.ask(), Some(101.0));
        assert_eq!(quote.mid(), 100.0);
        match quote {
            Quote::Sized {
                bid_size, ask_size, ..
            } => {
                assert_eq!((bid_size, ask_size), (100.0, 150.0));
            }
            other => panic!("expected a sized quote, got {other:?}"),
        }
        // a one-sided book has no priceable top
        let one_sided = MarketDepth::new(vec![level(99.0, 1.0)], vec![]).unwrap();
        assert_eq!(one_sided.to_quote(), None);
    }
}
