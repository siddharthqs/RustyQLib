//! Shared market state, separated from contracts: the **pricing context**.
//!
//! Instruments constructed from JSON or the builder embed the market they
//! were built with — convenient for a stateless pricing service, but a
//! desk wants the other shape too: one market snapshot shared across a
//! book, bumped once, and the whole book repriced under it. That is this
//! module:
//!
//! ```text
//! let market = MarketData::capture(&book);          // snapshot the book's market
//! let crash  = market.bump(&scenario.shocks)?;      // -20% spot, +10 vol pts, ...
//! let pnl    = book.npv_under(&crash)? - book.npv_under(&market)?;
//! ```
//!
//! A [`MarketData`] holds scalar market *levels* — per-underlying spot and
//! implied vol, one rate, one valuation date. Repricing computes each
//! position's deltas against its embedded market and revalues through
//! [`EquityOption::price_with`] (full revaluation on the position's own
//! engine, parallel vol shift, common random numbers under Monte Carlo).
//! The TOML stress runner ([`risk::stress`](crate::risk::stress)) is a
//! consumer of these primitives.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use serde::Deserialize;

use crate::core::errors::RustyQLibError;
use crate::equity::portfolio::EquityPortfolio;
use crate::equity::vanilla_option::EquityOption;

// ── Shock vocabulary ────────────────────────────────────────────────────

/// How a shock size is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BumpMode {
    /// `size` is a fraction of the current market level.
    Relative,
    /// `size` is added to the current market level (for `time`: days).
    Absolute,
}

/// The risk factor a shock applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiskFactor {
    Spot,
    #[serde(alias = "volatility")]
    Vol,
    #[serde(alias = "rates")]
    Rate,
    /// Calendar decay, in days (absolute only).
    Time,
}

/// One shock on one factor.
#[derive(Debug, Clone, Deserialize)]
pub struct Shock {
    pub factor: RiskFactor,
    pub mode: BumpMode,
    pub size: f64,
    /// Restrict to one underlying symbol (`None` / `"*"` = every one).
    pub underlying: Option<String>,
}

impl Shock {
    /// Whether this shock applies to `symbol` (case-insensitive filter).
    pub fn applies_to(&self, symbol: &str) -> bool {
        match self.underlying.as_deref() {
            None | Some("*") => true,
            Some(name) => name.eq_ignore_ascii_case(symbol),
        }
    }
}

// ── The market snapshot ─────────────────────────────────────────────────

/// Scalar market levels for one underlying.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquityLevels {
    pub spot: f64,
    /// Implied volatility level; deltas against it are applied as a
    /// parallel shift of the position's surface.
    pub vol: f64,
}

/// A market snapshot shared across a book: per-underlying levels, one
/// rate, one valuation date. Build it by [`capture`](MarketData::capture)
/// from a book, or by hand for a hypothetical market.
#[derive(Debug, Clone, PartialEq)]
pub struct MarketData {
    pub valuation_date: NaiveDate,
    /// Flat risk-free rate level; deltas shift each position's curve in
    /// parallel.
    pub rate: f64,
    equities: BTreeMap<String, EquityLevels>,
}

impl MarketData {
    pub fn new(valuation_date: NaiveDate, rate: f64) -> Self {
        MarketData { valuation_date, rate, equities: BTreeMap::new() }
    }

    /// Add or replace one underlying's levels (chainable).
    pub fn with_equity(mut self, symbol: &str, spot: f64, vol: f64) -> Self {
        self.equities.insert(symbol.to_string(), EquityLevels { spot, vol });
        self
    }

    pub fn spot(&self, symbol: &str) -> Option<f64> {
        self.equities.get(symbol).map(|l| l.spot)
    }

    pub fn vol(&self, symbol: &str) -> Option<f64> {
        self.equities.get(symbol).map(|l| l.vol)
    }

    pub fn symbols(&self) -> impl Iterator<Item = &str> {
        self.equities.keys().map(String::as_str)
    }

    /// Snapshot the market embedded in a book: valuation date and rate
    /// from the first position, per-underlying levels from the first
    /// position on each symbol. Repricing the book under the captured
    /// (unbumped) snapshot reproduces its base value exactly.
    pub fn capture(book: &EquityPortfolio) -> MarketData {
        let (valuation_date, rate) = match book.positions.first() {
            Some(p) => (p.option.base.valuation_date, p.option.base.risk_free_rate()),
            None => (chrono::Local::now().date_naive(), 0.0),
        };
        let mut market = MarketData::new(valuation_date, rate);
        for position in &book.positions {
            let base = &position.option.base;
            market.equities.entry(base.symbol.clone()).or_insert(EquityLevels {
                spot: base.underlying_price.value(),
                vol: base.volatility(),
            });
        }
        market
    }

    /// Snapshot a single option's embedded market.
    pub fn capture_option(option: &EquityOption) -> MarketData {
        let base = &option.base;
        MarketData::new(base.valuation_date, base.risk_free_rate()).with_equity(
            &base.symbol,
            base.underlying_price.value(),
            base.volatility(),
        )
    }

    /// Apply a scenario, returning the bumped market. Relative shocks
    /// scale by this snapshot's levels, repeated factors compose
    /// additively, `underlying` filters restrict spot/vol shocks to one
    /// name. Time shocks advance the valuation date and must be absolute
    /// (in days).
    pub fn bump(&self, shocks: &[Shock]) -> Result<MarketData, RustyQLibError> {
        let mut bumped = self.clone();
        let mut elapsed_days = 0.0;
        for shock in shocks {
            match shock.factor {
                RiskFactor::Spot | RiskFactor::Vol => {
                    for (symbol, levels) in bumped.equities.iter_mut() {
                        if !shock.applies_to(symbol) {
                            continue;
                        }
                        let base = self.equities[symbol];
                        match shock.factor {
                            RiskFactor::Spot => {
                                levels.spot += match shock.mode {
                                    BumpMode::Relative => base.spot * shock.size,
                                    BumpMode::Absolute => shock.size,
                                };
                            }
                            RiskFactor::Vol => {
                                levels.vol += match shock.mode {
                                    BumpMode::Relative => base.vol * shock.size,
                                    BumpMode::Absolute => shock.size,
                                };
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                RiskFactor::Rate => {
                    bumped.rate += match shock.mode {
                        BumpMode::Relative => self.rate * shock.size,
                        BumpMode::Absolute => shock.size,
                    };
                }
                RiskFactor::Time => {
                    if shock.mode == BumpMode::Relative {
                        return Err(RustyQLibError::invalid_input(
                            "shock",
                            "time shocks are absolute horizons in days; relative makes no sense",
                        ));
                    }
                    elapsed_days += shock.size;
                }
            }
        }
        if elapsed_days != 0.0 {
            bumped.valuation_date += chrono::Duration::days(elapsed_days.round() as i64);
        }
        Ok(bumped)
    }
}

// ── Repricing under a context ───────────────────────────────────────────

impl EquityOption {
    /// Value under a market snapshot: this position's deltas against its
    /// embedded market, revalued on its own engine. Errors when the
    /// snapshot has no levels for the option's underlying.
    pub fn npv_under(&self, market: &MarketData) -> Result<f64, RustyQLibError> {
        let levels = market.equities.get(&self.base.symbol).ok_or_else(|| {
            RustyQLibError::invalid_input(
                "market",
                format!("no market data for underlying '{}'", self.base.symbol),
            )
        })?;
        let d_spot = levels.spot - self.base.underlying_price.value();
        let d_vol = levels.vol - self.base.volatility();
        let d_rate = market.rate - self.base.risk_free_rate();
        let d_time =
            (market.valuation_date - self.base.valuation_date).num_days() as f64 / 365.0;
        Ok(self.price_with(d_spot, d_vol, d_rate, d_time))
    }
}

impl EquityPortfolio {
    /// Book value under a market snapshot (quantity-weighted).
    pub fn npv_under(&self, market: &MarketData) -> Result<f64, RustyQLibError> {
        let mut total = 0.0;
        for position in &self.positions {
            total += position.quantity * position.option.npv_under(market)?;
        }
        Ok(total)
    }

    /// Per-position values under a market snapshot, in book order.
    pub fn position_values_under(
        &self,
        market: &MarketData,
    ) -> Result<Vec<f64>, RustyQLibError> {
        self.positions
            .iter()
            .map(|p| p.option.npv_under(market).map(|v| p.quantity * v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;

    fn option(symbol: &str, strike: f64, engine: Engine) -> EquityOption {
        EquityOptionBuilder::new()
            .symbol(symbol)
            .spot(100.0)
            .strike(strike)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(engine)
            .build()
            .expect("option must build")
    }

    fn book() -> EquityPortfolio {
        // EquityPortfolio books are single-underlying; multi-underlying
        // markets are exercised through hand-built MarketData below
        let mut book = EquityPortfolio::new();
        book.add(option("ACME", 95.0, Engine::BlackScholes), 10.0);
        book.add(option("ACME", 105.0, Engine::BlackScholes), -5.0);
        book.add(option("ACME", 100.0, Engine::FiniteDifference), 3.0);
        book
    }

    #[test]
    fn captured_market_reproduces_the_base_value_exactly() {
        let book = book();
        let market = MarketData::capture(&book);
        assert_eq!(market.spot("ACME"), Some(100.0));
        assert_eq!(market.vol("ACME"), Some(0.25));
        let direct: f64 =
            book.positions.iter().map(|p| p.quantity * p.option.npv()).sum();
        let under = book.npv_under(&market).unwrap();
        assert!((under - direct).abs() < 1e-10, "under {under} direct {direct}");
    }

    #[test]
    fn bumped_spot_matches_direct_repricing() {
        let book = book();
        let market = MarketData::capture(&book);
        let shocks = [Shock {
            factor: RiskFactor::Spot,
            mode: BumpMode::Relative,
            size: -0.20,
            underlying: Some("ACME".to_string()),
        }];
        let crash = market.bump(&shocks).unwrap();
        assert_eq!(crash.spot("ACME"), Some(80.0));
        // a shock filtered to another name leaves this book untouched
        let other = [Shock {
            factor: RiskFactor::Spot,
            mode: BumpMode::Relative,
            size: -0.20,
            underlying: Some("ZENO".to_string()),
        }];
        let unmoved = market.bump(&other).unwrap();
        assert_eq!(unmoved.spot("ACME"), Some(100.0), "filter must spare ACME");
        assert!(
            (book.npv_under(&unmoved).unwrap() - book.npv_under(&market).unwrap()).abs() < 1e-10
        );
        // position-level check against price_with
        let p = &book.positions[0];
        let expected = p.quantity * p.option.price_with(-20.0, 0.0, 0.0, 0.0);
        let got = p.quantity * p.option.npv_under(&crash).unwrap();
        assert!((got - expected).abs() < 1e-10);
    }

    #[test]
    fn shocks_compose_additively_and_relative_scales_by_base() {
        let market = MarketData::new(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(), 0.03)
            .with_equity("ACME", 100.0, 0.25);
        let shocks = [
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Relative, size: -0.10, underlying: None },
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Absolute, size: 2.0, underlying: None },
            Shock { factor: RiskFactor::Vol, mode: BumpMode::Absolute, size: 0.10, underlying: None },
            Shock { factor: RiskFactor::Rate, mode: BumpMode::Relative, size: 1.0, underlying: None },
        ];
        let bumped = market.bump(&shocks).unwrap();
        // both spot shocks scale off the ORIGINAL level: -10 + 2
        assert!((bumped.spot("ACME").unwrap() - 92.0).abs() < 1e-12);
        assert!((bumped.vol("ACME").unwrap() - 0.35).abs() < 1e-12);
        assert!((bumped.rate - 0.06).abs() < 1e-12);
    }

    #[test]
    fn time_bump_advances_the_valuation_date_and_decays_value() {
        let book = book();
        let market = MarketData::capture(&book);
        let month = [Shock {
            factor: RiskFactor::Time,
            mode: BumpMode::Absolute,
            size: 30.0,
            underlying: None,
        }];
        let later = market.bump(&month).unwrap();
        assert_eq!(later.valuation_date, NaiveDate::from_ymd_opt(2026, 2, 4).unwrap());
        // a long option book decays as time passes, everything else equal
        let base = book.npv_under(&market).unwrap();
        let aged = book.npv_under(&later).unwrap();
        assert!(aged < base, "aged {aged} must be below base {base}");

        // relative time shocks are rejected
        let bad = [Shock {
            factor: RiskFactor::Time,
            mode: BumpMode::Relative,
            size: 0.1,
            underlying: None,
        }];
        assert!(market.bump(&bad).is_err());
    }

    #[test]
    fn missing_underlying_is_a_typed_error() {
        let market = MarketData::new(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(), 0.03);
        let opt = option("ACME", 100.0, Engine::BlackScholes);
        match opt.npv_under(&market) {
            Err(RustyQLibError::InvalidInput { field, reason }) => {
                assert_eq!(field, "market");
                assert!(reason.contains("ACME"));
            }
            other => panic!("expected InvalidInput, got {other:?}"),
        }
    }

    #[test]
    fn one_market_serves_positions_on_different_underlyings() {
        let acme = option("ACME", 100.0, Engine::BlackScholes);
        let zeno = option("ZENO", 100.0, Engine::FiniteDifference);
        let market = MarketData::new(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(), 0.03)
            .with_equity("ACME", 100.0, 0.25)
            .with_equity("ZENO", 100.0, 0.25);
        // base: matches each option's own value
        assert!((acme.npv_under(&market).unwrap() - acme.npv()).abs() < 1e-10);
        assert!((zeno.npv_under(&market).unwrap() - zeno.npv()).abs() < 1e-10);
        // crash only ACME: ZENO is untouched under the same bumped market
        let crash = market
            .bump(&[Shock {
                factor: RiskFactor::Spot,
                mode: BumpMode::Relative,
                size: -0.20,
                underlying: Some("ACME".to_string()),
            }])
            .unwrap();
        assert!(acme.npv_under(&crash).unwrap() < acme.npv());
        assert!((zeno.npv_under(&crash).unwrap() - zeno.npv()).abs() < 1e-10);
    }

    #[test]
    fn hand_built_market_prices_a_hypothetical() {
        // mark the same option under a hand-made market 10% higher
        let opt = option("ACME", 100.0, Engine::BlackScholes);
        let market = MarketData::new(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(), 0.03)
            .with_equity("ACME", 110.0, 0.25);
        let up = opt.npv_under(&market).unwrap();
        assert!((up - opt.price_with(10.0, 0.0, 0.0, 0.0)).abs() < 1e-12);
        assert!(up > opt.npv());
    }
}
