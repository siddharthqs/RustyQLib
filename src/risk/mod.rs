//! Risk analytics: Value-at-Risk, Expected Shortfall and the
//! surrounding toolkit, one concern per file.
//!
//! - [`measures`]: VaR and ES — historical (empirical quantile / tail
//!   mean), parametric normal, Cornish-Fisher higher-moment corrected,
//!   and delta-normal multi-asset VaR with the exact Euler
//!   component/marginal decomposition;
//! - [`portfolio_risk`]: scenario VaR/ES for an options book
//!   ([`EquityPortfolio`](crate::equity::portfolio::EquityPortfolio)) —
//!   **delta-gamma-vega-theta** from the aggregated Greeks and **full
//!   revaluation** through the book's repricer, on shared spot/vol
//!   scenarios so their difference isolates the Taylor error;
//! - [`volatility`]: realized and EWMA (RiskMetrics) estimators;
//! - [`performance`]: max drawdown, Sharpe and Sortino ratios;
//! - [`backtest`]: the Kupiec proportion-of-failures VaR backtest;
//! - [`stress`]: TOML-configured **stress MtM** — named shock scenarios
//!   (relative/absolute, per-underlying) bumped into the market data and
//!   fully revalued, reported per trade and aggregated per scenario.
//!
//! Conventions: confidence levels are one-sided (0.99), VaR/ES are
//! positive loss amounts, and every simulation is deterministic per
//! seed.

pub mod backtest;
pub mod stress;
pub mod measures;
pub mod performance;
pub mod portfolio_risk;
pub mod volatility;

pub use backtest::{kupiec_pof, KupiecTest};
pub use measures::{
    cornish_fisher_var, delta_normal_var, historical_expected_shortfall, historical_var,
    parametric_expected_shortfall, parametric_var, DeltaNormalVar,
};
pub use performance::{max_drawdown, sharpe_ratio, sortino_ratio};
pub use portfolio_risk::{delta_gamma_var, full_revaluation_var, PortfolioRisk, RiskConfig};
pub use stress::{
    prepare_bump, stress_mtm, BumpMode, MarketBump, RiskFactor, ScenarioResult, Shock,
    StressConfig, StressScenario, TradeStress,
};
pub use volatility::{ewma_volatility, realized_volatility};
