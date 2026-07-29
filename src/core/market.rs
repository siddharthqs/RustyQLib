//! The typed, open-ended market data container: the **pricing context**.
//!
//! A [`Market`] is not a struct of fields — it is a type-safe dictionary.
//! Each kind of market datum is addressed by its own **key type** (a
//! [`MarketKey`] implementor), and the key pins the value type at compile
//! time: looking up [`Spot`] returns a [`Quote`], looking up [`Vol`]
//! returns a [`VolSurface`]. Storage is open-ended — a new instrument that
//! needs a datum nobody anticipated (a correlation term structure, an FX
//! vol, a forecast curve) defines a new key type and inserts it; no core
//! code changes.
//!
//! ```text
//! Spot("ACME")     -> Quote(100.0)
//! Vol("ACME")      -> VolSurface {...}
//! Discount("USD")  -> YieldCurve {...}
//! ```
//!
//! The only runtime failure mode is *missing data*
//! ([`RustyQLibError::MissingMarketData`]) — a wrong-type lookup cannot be
//! written, because the key's [`MarketKey::Value`] fixes the return type.
//!
//! Instruments consume a market through their own wiring (e.g.
//! [`EquityOption::npv_in`](crate::equity::vanilla_option::EquityOption)),
//! keeping contract terms (immutable) separate from market state (shared,
//! bumped, repriced).

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::hash::Hash;

use chrono::NaiveDate;
use serde::Deserialize;

use crate::core::curves::{RateShift, YieldCurve};
use crate::core::errors::{Result, RustyQLibError};
use crate::core::quotes::Quote;
use crate::core::vols::{VolShift, VolSurface};

// ── The key protocol ────────────────────────────────────────────────────

/// A typed address for one piece of market data.
///
/// Implementing this trait is the extension point of the whole design:
/// the key names *what* is identified (symbol, currency, pair, ...) and
/// its associated `Value` fixes *which type* a lookup returns. Anyone —
/// including downstream crates and tests — can add key types without
/// touching [`Market`].
///
/// ```ignore
/// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// struct Correlation(String, String);
/// impl MarketKey for Correlation { type Value = CorrelationTermStructure; }
/// ```
pub trait MarketKey: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    /// The type a lookup with this key returns. `Clone` so a whole market
    /// can be cloned (the basis of scenario bumping: copy, then perturb).
    type Value: Clone + Debug + Send + Sync + 'static;
}

// ── Built-in keys ───────────────────────────────────────────────────────

/// Spot quote of one underlying, by symbol.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Spot(pub String);
impl MarketKey for Spot {
    type Value = Quote;
}

/// Implied volatility surface of one underlying, by symbol.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Vol(pub String);
impl MarketKey for Vol {
    type Value = VolSurface;
}

/// Discount curve, by currency code (e.g. `"USD"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Discount(pub String);
impl MarketKey for Discount {
    type Value = YieldCurve;
}

/// Currency assumed when an instrument does not state one, so that such
/// instruments and hand-built markets agree on the same [`Discount`] key.
pub const DEFAULT_CURRENCY: &str = "USD";

// ── Type-erased storage ─────────────────────────────────────────────────
//
// One inner `HashMap<K, K::Value>` per key type, held behind a small
// object-safe trait so the outer map can store them uniformly and clone
// the whole market. The downcasts below are the only ones in the design,
// and they cannot fail: the outer map is keyed by `TypeId::of::<K>()`, so
// the entry for that id *is* a `HashMap<K, K::Value>` by construction.

trait AnyStore: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn AnyStore>;
    fn len(&self) -> usize;
}

impl<K: MarketKey> AnyStore for HashMap<K, K::Value> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn AnyStore> {
        Box::new(self.clone())
    }
    fn len(&self) -> usize {
        HashMap::len(self)
    }
}

// ── The container ───────────────────────────────────────────────────────

/// A market snapshot: valuation date plus a typed store of market data,
/// shared across a book. See the [module docs](self) for the design.
pub struct Market {
    valuation_date: NaiveDate,
    stores: HashMap<TypeId, Box<dyn AnyStore>>,
}

impl Clone for Market {
    fn clone(&self) -> Self {
        Market {
            valuation_date: self.valuation_date,
            stores: self.stores.iter().map(|(&id, s)| (id, s.clone_box())).collect(),
        }
    }
}

impl Debug for Market {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Market")
            .field("valuation_date", &self.valuation_date)
            .field("entries", &self.len())
            .finish()
    }
}

impl Market {
    pub fn new(valuation_date: NaiveDate) -> Self {
        Market { valuation_date, stores: HashMap::new() }
    }

    pub fn valuation_date(&self) -> NaiveDate {
        self.valuation_date
    }

    /// Add or replace the datum at `key`.
    pub fn insert<K: MarketKey>(&mut self, key: K, value: K::Value) {
        self.stores
            .entry(TypeId::of::<K>())
            .or_insert_with(|| Box::new(HashMap::<K, K::Value>::new()))
            .as_any_mut()
            .downcast_mut::<HashMap<K, K::Value>>()
            .expect("store type is pinned by the TypeId key")
            .insert(key, value);
    }

    /// Chainable [`insert`](Self::insert), for building markets by hand.
    pub fn with<K: MarketKey>(mut self, key: K, value: K::Value) -> Self {
        self.insert(key, value);
        self
    }

    /// The datum at `key`, or `None` when absent.
    pub fn try_get<K: MarketKey>(&self, key: &K) -> Option<&K::Value> {
        self.stores
            .get(&TypeId::of::<K>())?
            .as_any()
            .downcast_ref::<HashMap<K, K::Value>>()
            .expect("store type is pinned by the TypeId key")
            .get(key)
    }

    /// The datum at `key`, or a typed
    /// [`MissingMarketData`](RustyQLibError::MissingMarketData) error naming
    /// the key (e.g. `Spot("ACME")`).
    pub fn get<K: MarketKey>(&self, key: &K) -> Result<&K::Value> {
        self.try_get(key)
            .ok_or_else(|| RustyQLibError::MissingMarketData { key: format!("{key:?}") })
    }

    pub fn contains<K: MarketKey>(&self, key: &K) -> bool {
        self.try_get(key).is_some()
    }

    /// Every key of type `K` in the store (in no particular order).
    pub fn keys<K: MarketKey>(&self) -> impl Iterator<Item = &K> {
        self.stores
            .get(&TypeId::of::<K>())
            .map(|s| {
                s.as_any()
                    .downcast_ref::<HashMap<K, K::Value>>()
                    .expect("store type is pinned by the TypeId key")
                    .keys()
            })
            .into_iter()
            .flatten()
    }

    /// Total number of stored data across all key types.
    pub fn len(&self) -> usize {
        self.stores.values().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Scenarios: the shock vocabulary ─────────────────────────────────────

/// How a shock size is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BumpMode {
    /// `size` scales the current level (spot x `(1+size)`, vols/zero rates
    /// scaled by `(1+size)`).
    Relative,
    /// `size` is added to the current level (for `time`: days).
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

/// One shock on one factor. The *how* of each bump lives with the bumped
/// type ([`VolSurface::bumped`], [`YieldCurve::bumped`]); a shock only
/// names the factor, the sizing and an optional underlying filter.
#[derive(Debug, Clone, Deserialize)]
pub struct Shock {
    pub factor: RiskFactor,
    pub mode: BumpMode,
    pub size: f64,
    /// Restrict a spot/vol shock to one underlying symbol (`None` / `"*"`
    /// = every one). Rate and time shocks are market-wide; the filter is
    /// ignored for them.
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

impl Market {
    /// Apply a scenario, returning the bumped market. Shocks apply **in
    /// order**, each to the market produced by the previous one; each
    /// datum performs its own bump ([`VolSurface::bumped`],
    /// [`YieldCurve::bumped`]). Spot/vol shocks honour the `underlying`
    /// filter; rate shocks hit every discount curve; time shocks advance
    /// the valuation date and must be absolute (in days).
    pub fn bumped(&self, shocks: &[Shock]) -> Result<Market> {
        let mut bumped = self.clone();
        for shock in shocks {
            match shock.factor {
                RiskFactor::Spot => {
                    let keys: Vec<Spot> = bumped
                        .keys::<Spot>()
                        .filter(|k| shock.applies_to(&k.0))
                        .cloned()
                        .collect();
                    for key in keys {
                        let level = bumped.get(&key)?.value();
                        let shifted = match shock.mode {
                            BumpMode::Relative => level * (1.0 + shock.size),
                            BumpMode::Absolute => level + shock.size,
                        };
                        bumped.insert(key, Quote::new(shifted));
                    }
                }
                RiskFactor::Vol => {
                    let shift = match shock.mode {
                        BumpMode::Relative => VolShift::ParallelRelative(shock.size),
                        BumpMode::Absolute => VolShift::ParallelAbsolute(shock.size),
                    };
                    let keys: Vec<Vol> = bumped
                        .keys::<Vol>()
                        .filter(|k| shock.applies_to(&k.0))
                        .cloned()
                        .collect();
                    for key in keys {
                        let surface = bumped.get(&key)?.bumped(shift)?;
                        bumped.insert(key, surface);
                    }
                }
                RiskFactor::Rate => {
                    let shift = match shock.mode {
                        BumpMode::Relative => RateShift::ParallelRelative(shock.size),
                        BumpMode::Absolute => RateShift::ParallelAbsolute(shock.size),
                    };
                    let keys: Vec<Discount> = bumped.keys::<Discount>().cloned().collect();
                    for key in keys {
                        let curve = bumped.get(&key)?.bumped(shift);
                        bumped.insert(key, curve);
                    }
                }
                RiskFactor::Time => {
                    if shock.mode == BumpMode::Relative {
                        return Err(RustyQLibError::invalid_input(
                            "shock",
                            "time shocks are absolute horizons in days; relative makes no sense",
                        ));
                    }
                    bumped.valuation_date += chrono::Duration::days(shock.size.round() as i64);
                }
            }
        }
        Ok(bumped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use crate::core::curves::Compounding;

    fn date() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 5).unwrap()
    }

    fn sample_market() -> Market {
        let curve = YieldCurve::flat(0.03, date(), DayCountConvention::Act365, Compounding::Continuous)
            .expect("curve must build");
        let surf = VolSurface::flat(0.25, date(), DayCountConvention::Act365).expect("surface");
        Market::new(date())
            .with(Spot("ACME".into()), Quote::new(100.0))
            .with(Spot("ZENO".into()), Quote::new(50.0))
            .with(Vol("ACME".into()), surf)
            .with(Discount("USD".into()), curve)
    }

    #[test]
    fn typed_roundtrip_per_key() {
        let market = sample_market();
        assert_eq!(market.get(&Spot("ACME".into())).unwrap().value(), 100.0);
        assert_eq!(market.get(&Spot("ZENO".into())).unwrap().value(), 50.0);
        // the Vol lookup statically returns a VolSurface — vol() is callable
        let sigma = market.get(&Vol("ACME".into())).unwrap().vol(100.0, 100.0, 1.0);
        assert!((sigma - 0.25).abs() < 1e-12);
        assert_eq!(market.len(), 4);
    }

    #[test]
    fn same_name_under_different_key_types_does_not_collide() {
        let market = sample_market();
        // "ACME" exists as a Spot key and a Vol key; each resolves its own type
        assert!(market.contains(&Spot("ACME".into())));
        assert!(market.contains(&Vol("ACME".into())));
        assert!(!market.contains(&Vol("ZENO".into())), "no surface stored for ZENO");
    }

    #[test]
    fn missing_data_is_a_typed_error_naming_the_key() {
        let market = sample_market();
        match market.get(&Vol("ZENO".into())) {
            Err(RustyQLibError::MissingMarketData { key }) => {
                assert!(key.contains("Vol") && key.contains("ZENO"), "got key `{key}`");
            }
            other => panic!("expected MissingMarketData, got {other:?}"),
        }
    }

    #[test]
    fn insert_replaces_existing_entry() {
        let mut market = sample_market();
        market.insert(Spot("ACME".into()), Quote::new(120.0));
        assert_eq!(market.get(&Spot("ACME".into())).unwrap().value(), 120.0);
        assert_eq!(market.len(), 4, "replace must not grow the store");
    }

    #[test]
    fn user_defined_key_types_extend_the_market() {
        // the open-world guarantee: a key type defined OUTSIDE core (here,
        // in a test) stores and retrieves its own value type
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        struct Correlation(String, String);
        impl MarketKey for Correlation {
            type Value = f64;
        }

        let market =
            sample_market().with(Correlation("ACME".into(), "ZENO".into()), 0.65);
        let rho = market.get(&Correlation("ACME".into(), "ZENO".into())).unwrap();
        assert_eq!(*rho, 0.65);
        assert!(market.get(&Correlation("ACME".into(), "OTHER".into())).is_err());
    }

    #[test]
    fn bumped_market_delegates_to_each_factor_and_honours_filters() {
        let market = sample_market();
        let shocks = [
            Shock {
                factor: RiskFactor::Spot,
                mode: BumpMode::Relative,
                size: -0.20,
                underlying: Some("ACME".into()),
            },
            Shock { factor: RiskFactor::Vol, mode: BumpMode::Absolute, size: 0.05, underlying: None },
            Shock { factor: RiskFactor::Rate, mode: BumpMode::Absolute, size: 0.01, underlying: None },
        ];
        let bumped = market.bumped(&shocks).unwrap();
        assert!((bumped.get(&Spot("ACME".into())).unwrap().value() - 80.0).abs() < 1e-12);
        // the filter spares ZENO
        assert!((bumped.get(&Spot("ZENO".into())).unwrap().value() - 50.0).abs() < 1e-12);
        let vol = bumped.get(&Vol("ACME".into())).unwrap().vol(100.0, 100.0, 1.0);
        assert!((vol - 0.30).abs() < 1e-12);
        let zero = bumped
            .get(&Discount("USD".into()))
            .unwrap()
            .zero_rate_with(1.0, Compounding::Continuous);
        assert!((zero - 0.04).abs() < 1e-12);
        // the base market is untouched
        assert!((market.get(&Spot("ACME".into())).unwrap().value() - 100.0).abs() < 1e-12);
    }

    #[test]
    fn time_shocks_advance_the_date_and_must_be_absolute() {
        let market = sample_market();
        let week = [Shock {
            factor: RiskFactor::Time,
            mode: BumpMode::Absolute,
            size: 7.0,
            underlying: None,
        }];
        let later = market.bumped(&week).unwrap();
        assert_eq!(later.valuation_date(), NaiveDate::from_ymd_opt(2026, 1, 12).unwrap());
        let bad = [Shock {
            factor: RiskFactor::Time,
            mode: BumpMode::Relative,
            size: 0.1,
            underlying: None,
        }];
        assert!(market.bumped(&bad).is_err());
    }

    #[test]
    fn cloned_market_is_independent() {
        let market = sample_market();
        let mut bumped = market.clone();
        bumped.insert(Spot("ACME".into()), Quote::new(80.0));
        assert_eq!(bumped.get(&Spot("ACME".into())).unwrap().value(), 80.0);
        assert_eq!(
            market.get(&Spot("ACME".into())).unwrap().value(),
            100.0,
            "clone must not alias the original"
        );
    }
}
