//! Stress MtM: scenario revaluation of an options book driven by a
//! **TOML shock configuration**.
//!
//! A config declares named scenarios, each a list of shocks on risk
//! factors (`spot`, `vol`, `rate`, `time`) with `relative` or
//! `absolute` sizing and an optional `underlying` filter:
//!
//! ```toml
//! [[scenarios]]
//! name = "equity_crash"
//!
//! [[scenarios.shocks]]
//! factor = "spot"
//! mode = "relative"
//! size = -0.20            # spot down 20%
//!
//! [[scenarios.shocks]]
//! factor = "vol"
//! mode = "absolute"
//! size = 0.10             # implied vol up 10 points
//! underlying = "ACME"     # only this name (omit or "*" for all)
//!
//! [[scenarios.shocks]]
//! factor = "rate"
//! mode = "absolute"
//! size = 0.005
//! tenors = [1.0, 2.0]     # key-rate: bump only this part of the curve
//! # shifts = [0.005, 0.003]  # optional per-tenor sizes (default: size)
//!
//! [arbitrage]             # optional guard on bumped curves
//! policy = "warn"         # allow | warn (default) | reject
//! forward_floor = 0.0
//! ```
//!
//! The book's market is snapshotted once into a typed
//! [`Market`](crate::core::market::Market) store, each scenario bumps it
//! ([`Market::bumped`](crate::core::market::Market::bumped) — every risk
//! factor object performs its own bump, shocks apply in order, relative
//! shocks scale the current level), and every position reprices fully on
//! its own engine under the bumped snapshot
//! ([`EquityOption::npv_in`](crate::equity::vanilla_option::EquityOption)).
//! Results come back **per trade** and **aggregated per scenario**, with
//! the aggregation identity `portfolio = sum(trades)` exact by
//! construction.
//!
//! The `time` factor is an absolute horizon in days (theta-inclusive
//! stresses); dividend/carry shocks are not yet supported by the
//! repricer and are rejected at parse time by omission from the enum.

use serde::Deserialize;

use crate::core::market::{Discount, Market};
use crate::equity::portfolio::EquityPortfolio;
use crate::equity::utils::PayoffType;
use crate::equity::vanilla_option::EquityOption;
use crate::core::errors::RustyQLibError;

// the shock vocabulary is the market layer's; re-exported here so stress
// configs keep their import paths
pub use crate::core::market::{BumpMode, RiskFactor, Shock};

/// A named collection of shocks applied together.
#[derive(Debug, Clone, Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub shocks: Vec<Shock>,
}

/// What to do when a bumped scenario curve implies a forward rate below
/// the configured floor (see
/// [`min_forward`](crate::core::curves::YieldCurve::min_forward)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ArbitragePolicy {
    /// Accept the curve silently.
    Allow,
    /// Log a warning and revalue anyway — the default: stress scenarios
    /// are deliberately extreme, and the P&L number is usually still wanted.
    #[default]
    Warn,
    /// Fail the run, naming the scenario, curve and offending segment.
    Reject,
}

/// The no-arbitrage guard applied to every bumped curve of every scenario.
///
/// ```toml
/// [arbitrage]
/// policy = "reject"        # allow | warn (default) | reject
/// forward_floor = 0.0      # smallest admissible continuous forward
/// ```
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct ArbitrageCheck {
    pub policy: ArbitragePolicy,
    /// Smallest admissible continuously compounded forward. `0.0` is the
    /// classic no-arbitrage bound; set it negative to tolerate negative
    /// forwards (the library allows negative rates).
    pub forward_floor: f64,
}

impl Default for ArbitrageCheck {
    fn default() -> Self {
        ArbitrageCheck { policy: ArbitragePolicy::default(), forward_floor: 0.0 }
    }
}

/// The whole stress configuration (one or more scenarios).
#[derive(Debug, Clone, Deserialize)]
pub struct StressConfig {
    pub scenarios: Vec<StressScenario>,
    /// No-arbitrage guard on bumped curves; defaults to warn at floor 0.
    #[serde(default)]
    pub arbitrage: ArbitrageCheck,
}

impl StressConfig {
    /// Parse from TOML text. Requires the `stress-config` feature.
    #[cfg(feature = "stress-config")]
    pub fn from_toml_str(text: &str) -> Result<StressConfig, RustyQLibError> {
        let config: StressConfig =
            toml::from_str(text).map_err(|e| RustyQLibError::ParseError(format!("invalid stress config: {e}")))?;
        config.validate()?;
        Ok(config)
    }

    /// Load and parse a TOML file. Requires the `stress-config` feature.
    #[cfg(feature = "stress-config")]
    pub fn from_toml_file(path: &str) -> Result<StressConfig, RustyQLibError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RustyQLibError::ParseError(format!("cannot read stress config '{path}': {e}")))?;
        Self::from_toml_str(&text)
    }

    #[cfg(feature = "stress-config")]
    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.scenarios.is_empty() {
            return Err(RustyQLibError::ParseError("stress config has no scenarios".to_string()));
        }
        for scenario in &self.scenarios {
            if scenario.shocks.is_empty() {
                return Err(RustyQLibError::ParseError(format!("scenario '{}' has no shocks", scenario.name)));
            }
            for shock in &scenario.shocks {
                if shock.factor == RiskFactor::Time && shock.mode == BumpMode::Relative {
                    return Err(RustyQLibError::ParseError(format!(
                        "scenario '{}': time shocks must be absolute (days)",
                        scenario.name
                    )));
                }
                let scenario_error = |reason: &str| {
                    RustyQLibError::ParseError(format!("scenario '{}': {reason}", scenario.name))
                };
                if let Some(tenors) = &shock.tenors {
                    if shock.factor != RiskFactor::Rate {
                        return Err(scenario_error("tenors are only supported on rate shocks"));
                    }
                    if shock.mode == BumpMode::Relative {
                        return Err(scenario_error("key-rate rate shocks must be absolute"));
                    }
                    if tenors.is_empty() {
                        return Err(scenario_error("tenors must not be empty"));
                    }
                    if tenors.windows(2).any(|w| w[1] <= w[0]) || tenors.iter().any(|&t| t <= 0.0) {
                        return Err(scenario_error("tenors must be positive and strictly increasing"));
                    }
                    if let Some(shifts) = &shock.shifts {
                        if shifts.len() != tenors.len() {
                            return Err(scenario_error("shifts must match tenors in length"));
                        }
                    }
                } else if shock.shifts.is_some() {
                    return Err(scenario_error("shifts require tenors"));
                }
            }
        }
        Ok(())
    }
}

/// One trade's stress result.
#[derive(Debug, Clone)]
pub struct TradeStress {
    /// Human-readable trade tag: symbol, payoff kind, strike, quantity.
    pub label: String,
    pub quantity: f64,
    pub base_mtm: f64,
    pub stressed_mtm: f64,
    /// `stressed - base`.
    pub stress_pnl: f64,
}

/// One scenario's book-level result with the per-trade breakdown.
#[derive(Debug, Clone)]
pub struct ScenarioResult {
    pub scenario: String,
    pub trades: Vec<TradeStress>,
    pub base_mtm: f64,
    pub stressed_mtm: f64,
    pub stress_pnl: f64,
}

fn trade_label(option: &EquityOption, quantity: f64) -> String {
    format!(
        "{} {:?} K={} x {}",
        option.base.symbol,
        option.payoff.payoff_kind(),
        option.base.strike_price,
        quantity
    )
}

/// Enforce the config's [`ArbitrageCheck`] on every discount curve of a
/// bumped scenario market.
fn check_arbitrage(
    market: &Market,
    check: &ArbitrageCheck,
    scenario: &str,
) -> Result<(), RustyQLibError> {
    if check.policy == ArbitragePolicy::Allow {
        return Ok(());
    }
    let keys: Vec<Discount> = market.keys::<Discount>().cloned().collect();
    for key in keys {
        let worst = market.get(&key)?.min_forward();
        if worst.forward < check.forward_floor {
            let detail = format!(
                "scenario '{scenario}': curve {key:?} implies forward {:.6} on [{:.4}, {:.4}], below floor {}",
                worst.forward, worst.t1, worst.t2, check.forward_floor
            );
            match check.policy {
                ArbitragePolicy::Warn => log::warn!("{detail}"),
                ArbitragePolicy::Reject => {
                    return Err(RustyQLibError::invalid_input("stress scenario", detail));
                }
                ArbitragePolicy::Allow => unreachable!("handled above"),
            }
        }
    }
    Ok(())
}

/// Run every scenario in `config` over the book, through the pricing
/// context: the book's market is snapshotted once, each scenario bumps it
/// ([`Market::bumped`]), the bumped curves pass the config's no-arbitrage
/// guard ([`ArbitrageCheck`]), and every position revalues fully on its
/// own engine under the bumped snapshot. Reported per trade and
/// aggregated; `portfolio = sum(trades)` is exact by construction.
/// Errors on a malformed shock, a rejected arbitrage check, or a position
/// the captured market cannot price.
pub fn stress_mtm(
    book: &EquityPortfolio,
    config: &StressConfig,
) -> Result<Vec<ScenarioResult>, RustyQLibError> {
    let base_market = book.snapshot_market();
    // base MtM is scenario-independent: price the book once, reuse the
    // per-position values across every scenario
    let base_values = book.position_values_in(&base_market)?;
    let base_total: f64 = base_values.iter().sum();
    let mut results = Vec::with_capacity(config.scenarios.len());
    for scenario in &config.scenarios {
        let stressed_market = base_market.bumped(&scenario.shocks)?;
        check_arbitrage(&stressed_market, &config.arbitrage, &scenario.name)?;
        let mut trades = Vec::with_capacity(book.positions.len());
        let mut stressed_total = 0.0;
        for (position, &base) in book.positions.iter().zip(&base_values) {
            let stressed = position.quantity * position.option.npv_in(&stressed_market)?;
            stressed_total += stressed;
            trades.push(TradeStress {
                label: trade_label(&position.option, position.quantity),
                quantity: position.quantity,
                base_mtm: base,
                stressed_mtm: stressed,
                stress_pnl: stressed - base,
            });
        }
        results.push(ScenarioResult {
            scenario: scenario.name.clone(),
            trades,
            base_mtm: base_total,
            stressed_mtm: stressed_total,
            stress_pnl: stressed_total - base_total,
        });
    }
    Ok(results)
}

// silence the unused-import lint path for PayoffType (used in labels)
const _: fn(&EquityOption) -> PayoffType = |o| o.payoff.payoff_kind();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    #[cfg(feature = "stress-config")]
    const CONFIG: &str = r#"
        [[scenarios]]
        name = "equity_crash"
        [[scenarios.shocks]]
        factor = "spot"
        mode = "relative"
        size = -0.20
        [[scenarios.shocks]]
        factor = "vol"
        mode = "absolute"
        size = 0.10

        [[scenarios]]
        name = "rates_up_acme_only"
        [[scenarios.shocks]]
        factor = "rate"
        mode = "absolute"
        size = 0.01
        underlying = "ACME"

        [[scenarios]]
        name = "one_week_decay"
        [[scenarios.shocks]]
        factor = "time"
        mode = "absolute"
        size = 7.0
    "#;

    fn option(symbol: &str, pc: PutOrCall, strike: f64) -> EquityOption {
        EquityOptionBuilder::new()
            .symbol(symbol)
            .spot(100.0)
            .strike(strike)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(pc)
            .engine(Engine::BlackScholes)
            .build().expect("option must build")
    }

    #[cfg(feature = "stress-config")]
    fn book() -> EquityPortfolio {
        let mut b = EquityPortfolio::new();
        b.add(option("ACME", PutOrCall::Call, 100.0), 100.0);
        b.add(option("ACME", PutOrCall::Put, 90.0), 50.0);
        b
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn toml_config_parses_scenarios_shocks_and_filters() {
        let config = StressConfig::from_toml_str(CONFIG).unwrap();
        assert_eq!(config.scenarios.len(), 3);
        let crash = &config.scenarios[0];
        assert_eq!(crash.shocks.len(), 2);
        assert_eq!(crash.shocks[0].factor, RiskFactor::Spot);
        assert_eq!(crash.shocks[0].mode, BumpMode::Relative);
        assert_eq!(crash.shocks[1].factor, RiskFactor::Vol);
        assert_eq!(config.scenarios[1].shocks[0].underlying.as_deref(), Some("ACME"));
        // rejects an empty config and relative time shocks
        assert!(StressConfig::from_toml_str("scenarios = []").is_err());
        let bad = r#"
            [[scenarios]]
            name = "bad"
            [[scenarios.shocks]]
            factor = "time"
            mode = "relative"
            size = 0.1
        "#;
        assert!(StressConfig::from_toml_str(bad).is_err());
        // unknown factors fail loudly at parse time
        let unknown = r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "dividend"
            mode = "absolute"
            size = 0.01
        "#;
        assert!(StressConfig::from_toml_str(unknown).is_err());
    }

    #[test]
    fn shocks_bump_the_market_levels_in_order_and_honour_filters() {
        use crate::core::market::{Spot, Vol};
        let opt = option("ACME", PutOrCall::Call, 100.0);
        let market = opt.snapshot_market();
        let shocks = vec![
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Relative, size: -0.2, underlying: None, tenors: None, shifts: None },
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Absolute, size: -1.0, underlying: None, tenors: None, shifts: None },
            // filtered out: different underlying
            Shock { factor: RiskFactor::Vol, mode: BumpMode::Absolute, size: 0.1, underlying: Some("OTHER".into()), tenors: None, shifts: None },
        ];
        let bumped = market.bumped(&shocks).unwrap();
        // in order: 100 * 0.8 = 80, then - 1
        let spot = bumped.get(&Spot("ACME".to_string())).unwrap().value();
        assert!((spot - 79.0).abs() < 1e-12, "composed spot {spot}");
        let vol = bumped.get(&Vol("ACME".to_string())).unwrap().vol(100.0, 100.0, 1.0);
        assert!((vol - 0.25).abs() < 1e-12, "filtered vol shock must not apply, got {vol}");
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn stress_mtm_matches_direct_repricing_and_aggregates_exactly() {
        let b = book();
        let config = StressConfig::from_toml_str(CONFIG).unwrap();
        let results = stress_mtm(&b, &config).unwrap();
        assert_eq!(results.len(), 3);
        let crash = &results[0];
        assert_eq!(crash.trades.len(), 2);
        // trade-level equals a direct price_with reprice
        let call = option("ACME", PutOrCall::Call, 100.0);
        let expected_stressed = 100.0 * call.price_with(-20.0, 0.10, 0.0, 0.0);
        assert!(
            (crash.trades[0].stressed_mtm - expected_stressed).abs() < 1e-10,
            "{} vs {expected_stressed}",
            crash.trades[0].stressed_mtm
        );
        // aggregation identity: portfolio = sum of trades, exactly
        for result in &results {
            let sum_pnl: f64 = result.trades.iter().map(|t| t.stress_pnl).sum();
            assert!((result.stress_pnl - sum_pnl).abs() < 1e-10, "{}", result.scenario);
            let sum_base: f64 = result.trades.iter().map(|t| t.base_mtm).sum();
            assert!((result.base_mtm - sum_base).abs() < 1e-10);
        }
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn scenario_economics_move_the_right_trades() {
        let b = book();
        let config = StressConfig::from_toml_str(CONFIG).unwrap();
        let results = stress_mtm(&b, &config).unwrap();
        let crash = &results[0];
        // spot -20% + vol +10pts: the long call loses, the long put gains
        assert!(crash.trades[0].stress_pnl < 0.0, "call {:?}", crash.trades[0]);
        assert!(crash.trades[1].stress_pnl > 0.0, "put {:?}", crash.trades[1]);
        // a week of pure decay costs a long-options book money
        let decay = &results[2];
        assert!(decay.stress_pnl < 0.0, "theta scenario {:?}", decay.stress_pnl);
        // the ACME-only rate shock hits every trade in this single-name book
        assert!(results[1].trades.iter().all(|t| t.stress_pnl != 0.0));
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn key_rate_shocks_parse_and_reprice_between_parallel_and_base() {
        let config = StressConfig::from_toml_str(
            r#"
            [[scenarios]]
            name = "front_end_up"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "absolute"
            size = 0.01
            tenors = [2.0]
        "#,
        )
        .unwrap();
        let shock = &config.scenarios[0].shocks[0];
        assert_eq!(shock.tenors.as_deref(), Some(&[2.0][..]));
        assert_eq!(shock.shifts, None);
        assert_eq!(config.arbitrage.policy, ArbitragePolicy::Warn, "default policy");

        // a 1.5y option sits between the 1y (unbumped) and 2y (bumped)
        // pillars: the 2y key-rate bump must move it less than the
        // same-size parallel bump, and more than not bumping at all
        let mid_pillar_option = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 7, 1).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .expect("option must build");
        let mut b = EquityPortfolio::new();
        b.add(mid_pillar_option, 100.0);
        let key_rate = stress_mtm(&b, &config).unwrap();
        let parallel = StressConfig::from_toml_str(
            r#"
            [[scenarios]]
            name = "all_up"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "absolute"
            size = 0.01
        "#,
        )
        .unwrap();
        let parallel = stress_mtm(&b, &parallel).unwrap();
        assert!(key_rate[0].stress_pnl.abs() > 1e-8, "key-rate shock must move the book");
        assert!(
            key_rate[0].stress_pnl.abs() < parallel[0].stress_pnl.abs(),
            "key-rate {} vs parallel {}",
            key_rate[0].stress_pnl,
            parallel[0].stress_pnl
        );

        // malformed key-rate configs fail at parse time
        for bad in [
            // shifts without tenors
            r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "absolute"
            size = 0.01
            shifts = [0.01]
            "#,
            // tenors on a vol shock
            r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "vol"
            mode = "absolute"
            size = 0.01
            tenors = [1.0]
            "#,
            // relative key-rate
            r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "relative"
            size = 0.01
            tenors = [1.0]
            "#,
            // length mismatch
            r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "absolute"
            size = 0.01
            tenors = [1.0, 2.0]
            shifts = [0.01]
            "#,
            // non-increasing tenors
            r#"
            [[scenarios]]
            name = "x"
            [[scenarios.shocks]]
            factor = "rate"
            mode = "absolute"
            size = 0.01
            tenors = [2.0, 1.0]
            "#,
        ] {
            assert!(StressConfig::from_toml_str(bad).is_err(), "must reject: {bad}");
        }
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn arbitrage_policy_rejects_curves_with_forwards_below_the_floor() {
        // -200bp at the 10y pillar alone: the preceding forward goes to
        // 3% - 2% * 10/3 < 0 (zero bumps amplify into forwards by t/dt)
        let toml = |policy: &str| {
            format!(
                r#"
                [[scenarios]]
                name = "long_end_collapse"
                [[scenarios.shocks]]
                factor = "rate"
                mode = "absolute"
                size = -0.02
                tenors = [10.0]

                [arbitrage]
                policy = "{policy}"
            "#
            )
        };
        let b = book();
        let rejecting = StressConfig::from_toml_str(&toml("reject")).unwrap();
        assert_eq!(rejecting.arbitrage.policy, ArbitragePolicy::Reject);
        let err = stress_mtm(&b, &rejecting).unwrap_err();
        assert!(err.to_string().contains("long_end_collapse"), "{err}");
        // warn and allow both let the run complete
        for policy in ["warn", "allow"] {
            let config = StressConfig::from_toml_str(&toml(policy)).unwrap();
            assert!(stress_mtm(&b, &config).is_ok(), "policy {policy} must not fail");
        }
        // a floor can also be relaxed instead of the policy
        let relaxed = StressConfig::from_toml_str(
            &(toml("reject") + "forward_floor = -0.10\n"),
        )
        .unwrap();
        assert!((relaxed.arbitrage.forward_floor + 0.10).abs() < 1e-12);
        assert!(stress_mtm(&b, &relaxed).is_ok());
    }

    #[test]
    #[cfg(feature = "stress-config")]
    fn config_file_round_trip() {
        let path = std::env::temp_dir().join("rustyqlib_stress_test.toml");
        std::fs::write(&path, CONFIG).unwrap();
        let config = StressConfig::from_toml_file(path.to_str().unwrap()).unwrap();
        assert_eq!(config.scenarios.len(), 3);
        let _ = std::fs::remove_file(&path);
        assert!(StressConfig::from_toml_file("no_such_file.toml").is_err());
    }
}
