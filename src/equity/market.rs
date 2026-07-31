//! Binding equity instruments to a shared [`Market`]: the **pricing
//! context**, separated from contracts.
//!
//! Instruments constructed from JSON or the builder embed the market they
//! were built with — convenient for a stateless pricing service, but a
//! desk wants the other shape too: one market snapshot shared across a
//! book, bumped once, and the whole book repriced under it:
//!
//! ```text
//! let market  = book.snapshot_market();              // typed store: Spot/Vol/Discount
//! let crash   = market.bumped(&scenario.shocks)?;    // -20% spot, +10 vol pts, ...
//! let pnl     = book.npv_in(&crash)? - book.npv_in(&market)?;
//! ```
//!
//! The store itself ([`core::market`](crate::core::market)) holds real
//! objects — [`Quote`](crate::core::quotes::Quote) spots,
//! [`VolSurface`](crate::core::vols::VolSurface)s,
//! [`YieldCurve`](crate::core::curves::YieldCurve)s — keyed by symbol and
//! currency, and each object owns its own bump semantics. This module is
//! the equity wiring between the store and the instrument's **bound**
//! market ([`EquityMarketData`](crate::equity::vanilla_option::EquityMarketData),
//! the `market` field engines read): snapshotting a bound market into a
//! store, and rebinding an instrument to a store for repricing on its own
//! engine (full revaluation; Monte Carlo keeps its seed, so bumped-minus-
//! base differences are free of sampling noise). The TOML stress runner
//! ([`risk::stress`](crate::risk::stress)) is a consumer of these
//! primitives.
//!
//! Dividend yield, borrow cost and discrete cash dividends live on the
//! bound [`EquityMarketData`](crate::equity::vanilla_option::EquityMarketData)
//! but are not yet keyed in the store; they gain keys when a consumer
//! needs to bump them.

use crate::core::errors::RustyQLibError;
use crate::core::market::{Discount, Market, Spot, Vol};
use crate::core::traits::Instrument;
use crate::equity::portfolio::EquityPortfolio;
use crate::equity::vanilla_option::EquityOption;

impl EquityOption {
    /// Snapshot this option's embedded market objects into a typed
    /// [`Market`] anchored at the option's valuation date. Repricing under
    /// the unmodified snapshot reproduces `npv()` exactly.
    pub fn snapshot_market(&self) -> Market {
        Market::new(self.market.valuation_date)
            .with(Spot(self.base.symbol.clone()), self.market.spot.clone())
            .with(Vol(self.base.symbol.clone()), self.market.vol_surface.clone())
            .with(
                Discount(self.base.currency_code().to_string()),
                self.market.discount_curve.clone(),
            )
    }

    /// This contract rebound to `market`: spot, vol surface and discount
    /// curve are taken from the store (by symbol / currency code) and the
    /// valuation date from the snapshot; contract terms and engine are
    /// unchanged. Errors name the missing key when the market lacks data
    /// for this option.
    ///
    /// The model moves with the market where it must: a Heston model's
    /// parameters follow the surface's parallel shift (measured at this
    /// contract's strike and maturity) via
    /// [`Model::with_vol_shift`](crate::equity::utils::Model::with_vol_shift),
    /// so vol scenarios reach Heston-priced positions without
    /// recalibration.
    ///
    /// The market's objects are expected to be anchored at its valuation
    /// date (as [`snapshot_market`](Self::snapshot_market) guarantees).
    pub fn with_market(&self, market: &Market) -> Result<EquityOption, RustyQLibError> {
        let spot = market.get(&Spot(self.base.symbol.clone()))?;
        let vol = market.get(&Vol(self.base.symbol.clone()))?;
        let curve = market.get(&Discount(self.base.currency_code().to_string()))?;
        let mut option = self.clone();
        option.market.spot = spot.clone();
        option.market.vol_surface = vol.clone();
        option.market.discount_curve = curve.clone();
        option.market.valuation_date = market.valuation_date();
        if option.model.is_heston() {
            let t = option.time_to_maturity();
            if t > 0.0 {
                // the surface's parallel shift at this contract's anchor
                // (strike, spot-as-forward-proxy, maturity)
                let k = option.base.strike_price;
                let f = option.market.spot.value();
                let shift = option.market.vol_surface.vol(k, f, t)
                    - self.market.vol_surface.vol(k, f, t);
                if shift != 0.0 {
                    option.model = option.model.with_vol_shift(shift);
                }
            }
        }
        Ok(option)
    }

    /// Value under a typed market snapshot: rebind, then price on the
    /// option's own engine through the ordinary `npv` path.
    pub fn npv_in(&self, market: &Market) -> Result<f64, RustyQLibError> {
        self.with_market(market)?.try_npv()
    }
}

impl EquityPortfolio {
    /// Snapshot the market embedded in a book into a typed [`Market`]:
    /// valuation date and discount curve from the first position, one
    /// spot/vol entry per underlying (first position on each symbol wins).
    pub fn snapshot_market(&self) -> Market {
        match self.positions.first() {
            Some(first) => {
                let mut market = first.option.snapshot_market();
                for position in &self.positions[1..] {
                    let option = &position.option;
                    if !market.contains(&Spot(option.base.symbol.clone())) {
                        market
                            .insert(Spot(option.base.symbol.clone()), option.market.spot.clone());
                        market.insert(
                            Vol(option.base.symbol.clone()),
                            option.market.vol_surface.clone(),
                        );
                    }
                }
                market
            }
            None => Market::new(chrono::Local::now().date_naive()),
        }
    }

    /// Book value under a typed market snapshot (quantity-weighted).
    pub fn npv_in(&self, market: &Market) -> Result<f64, RustyQLibError> {
        let mut total = 0.0;
        for position in &self.positions {
            total += position.quantity * position.option.npv_in(market)?;
        }
        Ok(total)
    }

    /// Per-position values under a typed market snapshot, in book order.
    pub fn position_values_in(
        &self,
        market: &Market,
    ) -> Result<Vec<f64>, RustyQLibError> {
        self.positions
            .iter()
            .map(|p| p.option.npv_in(market).map(|v| p.quantity * v))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::market::{BumpMode, RiskFactor, Shock};
    use crate::core::trade::PutOrCall;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::{Engine, Model};
    use chrono::NaiveDate;

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

    fn shock(factor: RiskFactor, mode: BumpMode, size: f64) -> Shock {
        Shock { factor, mode, size, underlying: None, tenors: None, shifts: None }
    }

    // ── snapshot / rebind parity ────────────────────────────────────

    #[test]
    fn snapshot_market_reproduces_npv_on_every_engine() {
        for engine in [
            Engine::BlackScholes,
            Engine::Binomial,
            Engine::FiniteDifference,
            Engine::MonteCarlo,
        ] {
            let label = format!("{engine:?}");
            let opt = option("ACME", 100.0, engine);
            let market = opt.snapshot_market();
            let rebound = opt.npv_in(&market).expect("snapshot must price");
            let direct = opt.npv();
            assert!(
                (rebound - direct).abs() < 1e-12,
                "{label}: rebound {rebound} direct {direct}"
            );
        }
    }

    #[test]
    fn rebinding_to_a_moved_market_prices_the_new_levels() {
        let opt = option("ACME", 100.0, Engine::BlackScholes);
        let mut market = opt.snapshot_market();
        market.insert(Spot("ACME".to_string()), crate::core::quotes::Quote::new(110.0));
        let moved = opt.npv_in(&market).unwrap();
        // reference: the same contract built directly at the new spot
        let rebuilt = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(110.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap();
        assert!((moved - rebuilt.npv()).abs() < 1e-12, "moved {moved} rebuilt {}", rebuilt.npv());
        // the original instrument is untouched
        assert_eq!(opt.market.spot.value(), 100.0);
    }

    #[test]
    fn npv_in_missing_symbol_names_the_key() {
        let opt = option("ACME", 100.0, Engine::BlackScholes);
        let empty = Market::new(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap());
        match opt.npv_in(&empty) {
            Err(RustyQLibError::MissingMarketData { key }) => {
                assert!(key.contains("Spot") && key.contains("ACME"), "got key `{key}`");
            }
            other => panic!("expected MissingMarketData, got {other:?}"),
        }
    }

    // ── bumped markets against the price_with reference ────────────
    //
    // While the per-engine `price_with` scalar path still exists, it is
    // the independent reference implementation for these parities: a
    // bumped market repriced through `npv_in` must agree with the same
    // shifts applied as scalar deltas.

    #[test]
    fn spot_vol_and_rate_bumps_match_price_with_on_every_engine() {
        for engine in [
            Engine::BlackScholes,
            Engine::Binomial,
            Engine::FiniteDifference,
            Engine::MonteCarlo,
        ] {
            let label = format!("{engine:?}");
            let opt = option("ACME", 100.0, engine);
            let market = opt.snapshot_market();
            let cases: [(&str, Shock, [f64; 4]); 4] = [
                (
                    "spot -20%",
                    shock(RiskFactor::Spot, BumpMode::Relative, -0.20),
                    [-20.0, 0.0, 0.0, 0.0],
                ),
                (
                    "vol +10pts",
                    shock(RiskFactor::Vol, BumpMode::Absolute, 0.10),
                    [0.0, 0.10, 0.0, 0.0],
                ),
                (
                    "rate +100bp",
                    shock(RiskFactor::Rate, BumpMode::Absolute, 0.01),
                    [0.0, 0.0, 0.01, 0.0],
                ),
                (
                    "vol +10% relative",
                    shock(RiskFactor::Vol, BumpMode::Relative, 0.10),
                    [0.0, 0.025, 0.0, 0.0], // 0.25 * 10%
                ),
            ];
            for (name, s, [ds, dv, dr, dt]) in cases {
                let bumped = market.bumped(std::slice::from_ref(&s)).unwrap();
                let via_market = opt.npv_in(&bumped).unwrap();
                let via_deltas = opt.price_with(ds, dv, dr, dt);
                assert!(
                    (via_market - via_deltas).abs() < 1e-10,
                    "{label} {name}: market {via_market} deltas {via_deltas}"
                );
            }
        }
    }

    #[test]
    fn time_bump_advances_the_valuation_date_and_decays_value() {
        let opt = option("ACME", 100.0, Engine::BlackScholes);
        let market = opt.snapshot_market();
        let month = shock(RiskFactor::Time, BumpMode::Absolute, 30.0);
        let later = market.bumped(std::slice::from_ref(&month)).unwrap();
        assert_eq!(later.valuation_date(), NaiveDate::from_ymd_opt(2026, 2, 4).unwrap());
        let aged = opt.npv_in(&later).unwrap();
        let expected = opt.price_with(0.0, 0.0, 0.0, 30.0 / 365.0);
        assert!((aged - expected).abs() < 1e-10, "aged {aged} expected {expected}");
        assert!(aged < opt.npv(), "a long option must decay");
        // relative time shocks are rejected
        let bad = shock(RiskFactor::Time, BumpMode::Relative, 0.1);
        assert!(market.bumped(std::slice::from_ref(&bad)).is_err());
    }

    #[test]
    fn shocks_apply_in_order_and_filters_spare_other_names() {
        let acme = option("ACME", 100.0, Engine::BlackScholes);
        let zeno = option("ZENO", 100.0, Engine::FiniteDifference);
        let market = acme
            .snapshot_market()
            .with(Spot("ZENO".to_string()), zeno.market.spot.clone())
            .with(Vol("ZENO".to_string()), zeno.market.vol_surface.clone());
        // -10% then +2 absolute, ACME only: 100 * 0.9 + 2 = 92
        let shocks = [
            Shock {
                factor: RiskFactor::Spot,
                mode: BumpMode::Relative,
                size: -0.10,
                underlying: Some("ACME".to_string()),
                tenors: None,
                shifts: None,
            },
            Shock {
                factor: RiskFactor::Spot,
                mode: BumpMode::Absolute,
                size: 2.0,
                underlying: Some("ACME".to_string()),
                tenors: None,
                shifts: None,
            },
        ];
        let bumped = market.bumped(&shocks).unwrap();
        assert!((bumped.get(&Spot("ACME".to_string())).unwrap().value() - 92.0).abs() < 1e-12);
        // ZENO untouched under the same bumped market
        assert!((zeno.npv_in(&bumped).unwrap() - zeno.npv()).abs() < 1e-10);
        assert!((acme.npv_in(&bumped).unwrap() - acme.price_with(-8.0, 0.0, 0.0, 0.0)).abs() < 1e-10);
    }

    #[test]
    fn heston_model_follows_the_surface_shift() {
        use crate::equity::heston::HestonParams;
        let mut opt = option("ACME", 100.0, Engine::BlackScholes);
        opt.model = Model::Heston(HestonParams {
            v0: 0.0625,
            kappa: 1.5,
            theta: 0.0625,
            vol_of_vol: 0.4,
            rho: -0.6,
        });
        let market = opt.snapshot_market();
        // base parity first
        assert!((opt.npv_in(&market).unwrap() - opt.npv()).abs() < 1e-12);
        // a +2pt vol scenario must reach the Heston params (the reference
        // scalar path shifts sqrt(v0)/sqrt(theta) — with_market must agree)
        let bumped = market
            .bumped(&[shock(RiskFactor::Vol, BumpMode::Absolute, 0.02)])
            .unwrap();
        let via_market = opt.npv_in(&bumped).unwrap();
        let via_deltas = opt.price_with(0.0, 0.02, 0.0, 0.0);
        assert!(
            (via_market - via_deltas).abs() < 1e-10,
            "market {via_market} deltas {via_deltas}"
        );
        assert!(via_market > opt.npv(), "long vega: higher vol must raise the value");
    }

    // ── portfolio-level ─────────────────────────────────────────────

    #[test]
    fn portfolio_snapshot_covers_every_underlying_and_reprices_exactly() {
        // EquityPortfolio books are single-underlying; multi-underlying
        // repricing is exercised option-by-option against one Market
        let mut book = EquityPortfolio::new();
        book.add(option("ACME", 95.0, Engine::BlackScholes), 10.0);
        book.add(option("ACME", 105.0, Engine::Binomial), -5.0);
        book.add(option("ACME", 100.0, Engine::FiniteDifference), 3.0);
        let market = book.snapshot_market();
        assert!(market.contains(&Spot("ACME".to_string())));
        assert!(market.contains(&Vol("ACME".to_string())));
        let direct: f64 = book.positions.iter().map(|p| p.quantity * p.option.npv()).sum();
        let under = book.npv_in(&market).unwrap();
        assert!((under - direct).abs() < 1e-10, "under {under} direct {direct}");
        // per-position values sum to the book value
        let values = book.position_values_in(&market).unwrap();
        let sum: f64 = values.iter().sum();
        assert!((sum - under).abs() < 1e-12);
    }
}
