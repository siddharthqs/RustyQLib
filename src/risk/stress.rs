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
//! ```
//!
//! The bump layer converts each shock into the absolute market deltas
//! the pricing layer understands — relative shocks are scaled by the
//! position's own market level (its spot, its smile vol, its curve
//! rate), multiple shocks on one factor compose additively on the base
//! market — and every position reprices through
//! [`EquityOption::price_with`], i.e. full revaluation on its own
//! engine. Results come back **per trade** and **aggregated per
//! scenario**, with the aggregation identity `portfolio = sum(trades)`
//! exact by construction.
//!
//! The `time` factor is an absolute horizon in days (theta-inclusive
//! stresses); dividend/carry shocks are not yet supported by the
//! repricer and are rejected at parse time by omission from the enum.

use serde::Deserialize;

use crate::equity::portfolio::EquityPortfolio;
use crate::equity::utils::PayoffType;
use crate::equity::vanilla_option::EquityOption;

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
    fn applies_to(&self, symbol: &str) -> bool {
        match self.underlying.as_deref() {
            None | Some("*") => true,
            Some(name) => name.eq_ignore_ascii_case(symbol),
        }
    }
}

/// A named collection of shocks applied together.
#[derive(Debug, Clone, Deserialize)]
pub struct StressScenario {
    pub name: String,
    pub shocks: Vec<Shock>,
}

/// The whole stress configuration (one or more scenarios).
#[derive(Debug, Clone, Deserialize)]
pub struct StressConfig {
    pub scenarios: Vec<StressScenario>,
}

impl StressConfig {
    /// Parse from TOML text.
    pub fn from_toml_str(text: &str) -> Result<StressConfig, String> {
        let config: StressConfig =
            toml::from_str(text).map_err(|e| format!("invalid stress config: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Load and parse a TOML file.
    pub fn from_toml_file(path: &str) -> Result<StressConfig, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read stress config '{path}': {e}"))?;
        Self::from_toml_str(&text)
    }

    fn validate(&self) -> Result<(), String> {
        if self.scenarios.is_empty() {
            return Err("stress config has no scenarios".into());
        }
        for scenario in &self.scenarios {
            if scenario.shocks.is_empty() {
                return Err(format!("scenario '{}' has no shocks", scenario.name));
            }
            for shock in &scenario.shocks {
                if shock.factor == RiskFactor::Time && shock.mode == BumpMode::Relative {
                    return Err(format!(
                        "scenario '{}': time shocks must be absolute (days)",
                        scenario.name
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The absolute market deltas a scenario implies for one position — the
/// shape [`EquityOption::price_with`] consumes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MarketBump {
    pub d_spot: f64,
    pub d_vol: f64,
    pub d_rate: f64,
    /// Elapsed calendar time in years.
    pub d_time: f64,
}

/// Convert a scenario's shocks into the absolute bumps for `option`:
/// relative shocks scale by the position's own market levels; shocks
/// filtered to other underlyings are skipped; repeated factors add.
pub fn prepare_bump(option: &EquityOption, shocks: &[Shock]) -> MarketBump {
    let mut bump = MarketBump::default();
    for shock in shocks {
        if !shock.applies_to(&option.base.symbol) {
            continue;
        }
        match shock.factor {
            RiskFactor::Spot => {
                bump.d_spot += match shock.mode {
                    BumpMode::Relative => option.base.underlying_price.value() * shock.size,
                    BumpMode::Absolute => shock.size,
                };
            }
            RiskFactor::Vol => {
                bump.d_vol += match shock.mode {
                    BumpMode::Relative => option.base.volatility() * shock.size,
                    BumpMode::Absolute => shock.size,
                };
            }
            RiskFactor::Rate => {
                bump.d_rate += match shock.mode {
                    BumpMode::Relative => option.base.risk_free_rate() * shock.size,
                    BumpMode::Absolute => shock.size,
                };
            }
            RiskFactor::Time => {
                // absolute, in days (validated at parse time)
                bump.d_time += shock.size / 365.0;
            }
        }
    }
    bump
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

/// Run every scenario in `config` over the book: full revaluation of
/// each position on its own engine, reported per trade and aggregated.
pub fn stress_mtm(book: &EquityPortfolio, config: &StressConfig) -> Vec<ScenarioResult> {
    config
        .scenarios
        .iter()
        .map(|scenario| {
            let mut trades = Vec::with_capacity(book.positions.len());
            let mut base_total = 0.0;
            let mut stressed_total = 0.0;
            for position in &book.positions {
                let bump = prepare_bump(&position.option, &scenario.shocks);
                let base = position.quantity * position.option.price_with(0.0, 0.0, 0.0, 0.0);
                let stressed = position.quantity
                    * position.option.price_with(bump.d_spot, bump.d_vol, bump.d_rate, bump.d_time);
                base_total += base;
                stressed_total += stressed;
                trades.push(TradeStress {
                    label: trade_label(&position.option, position.quantity),
                    quantity: position.quantity,
                    base_mtm: base,
                    stressed_mtm: stressed,
                    stress_pnl: stressed - base,
                });
            }
            ScenarioResult {
                scenario: scenario.name.clone(),
                trades,
                base_mtm: base_total,
                stressed_mtm: stressed_total,
                stress_pnl: stressed_total - base_total,
            }
        })
        .collect()
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
            .build()
    }

    fn book() -> EquityPortfolio {
        let mut b = EquityPortfolio::new();
        b.add(option("ACME", PutOrCall::Call, 100.0), 100.0);
        b.add(option("ACME", PutOrCall::Put, 90.0), 50.0);
        b
    }

    #[test]
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
    fn bumps_scale_relative_shocks_by_the_position_market() {
        let opt = option("ACME", PutOrCall::Call, 100.0);
        let shocks = vec![
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Relative, size: -0.2, underlying: None },
            Shock { factor: RiskFactor::Vol, mode: BumpMode::Absolute, size: 0.1, underlying: None },
            Shock { factor: RiskFactor::Spot, mode: BumpMode::Absolute, size: -1.0, underlying: None },
            // filtered out: different underlying
            Shock { factor: RiskFactor::Rate, mode: BumpMode::Absolute, size: 0.05, underlying: Some("OTHER".into()) },
        ];
        let bump = prepare_bump(&opt, &shocks);
        assert!((bump.d_spot - (-21.0)).abs() < 1e-12, "composed spot {}", bump.d_spot);
        assert!((bump.d_vol - 0.1).abs() < 1e-12);
        assert_eq!(bump.d_rate, 0.0, "filtered shock must not apply");
    }

    #[test]
    fn stress_mtm_matches_direct_repricing_and_aggregates_exactly() {
        let b = book();
        let config = StressConfig::from_toml_str(CONFIG).unwrap();
        let results = stress_mtm(&b, &config);
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
    fn scenario_economics_move_the_right_trades() {
        let b = book();
        let config = StressConfig::from_toml_str(CONFIG).unwrap();
        let results = stress_mtm(&b, &config);
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
    fn config_file_round_trip() {
        let path = std::env::temp_dir().join("rustyqlib_stress_test.toml");
        std::fs::write(&path, CONFIG).unwrap();
        let config = StressConfig::from_toml_file(path.to_str().unwrap()).unwrap();
        assert_eq!(config.scenarios.len(), 3);
        let _ = std::fs::remove_file(&path);
        assert!(StressConfig::from_toml_file("no_such_file.toml").is_err());
    }
}
