//! VaR and Expected Shortfall for an options book
//! ([`EquityPortfolio`]), by scenario simulation over the underlying
//! and its implied volatility:
//!
//! - **Delta-gamma(-vega-theta)**: P&L approximated from the book's
//!   aggregated Greeks — fast, and exactly the Taylor expansion the
//!   portfolio's PnL attribution uses;
//! - **Full revaluation**: every scenario reprices every position
//!   through [`EquityPortfolio::price_with`] — exact payoff convexity,
//!   at pricing cost.
//!
//! Scenarios are joint lognormal-spot / normal-vol moves with a
//! spot-vol correlation (negative in equities), deterministic per seed.
//! Both estimators share scenarios, so their difference is purely the
//! Taylor truncation — a direct read on how non-linear the book is.

use crate::core::montecarlo::path_rng;
use crate::equity::portfolio::EquityPortfolio;
use rand::Rng;
use rand_distr::StandardNormal;

use super::measures::{historical_expected_shortfall, historical_var};

/// Scenario-generation settings for portfolio VaR.
#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    /// Horizon in years (1 trading day = 1/252).
    pub horizon: f64,
    /// Annualized volatility of the underlying's return.
    pub spot_vol: f64,
    /// Annualized volatility of the implied-vol move (absolute, e.g.
    /// 0.8 means a 1-day vol move of ~0.8/sqrt(252) ~ 5 vol points).
    pub vol_of_vol: f64,
    /// Spot-vol move correlation (negative in equity markets).
    pub spot_vol_corr: f64,
    pub scenarios: usize,
    pub confidence: f64,
    pub seed: u64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        RiskConfig {
            horizon: 1.0 / 252.0,
            spot_vol: 0.2,
            vol_of_vol: 0.0,
            spot_vol_corr: -0.5,
            scenarios: 20_000,
            confidence: 0.99,
            seed: 42,
        }
    }
}

/// VaR / ES output with the scenario P&L retained for inspection.
#[derive(Debug, Clone)]
pub struct PortfolioRisk {
    pub var: f64,
    pub expected_shortfall: f64,
    pub mean_pnl: f64,
    pub scenarios: usize,
}

fn scenario_moves(cfg: &RiskConfig, spot: f64, i: u64) -> (f64, f64) {
    let mut rng = path_rng(cfg.seed, i);
    let z1: f64 = rng.sample(StandardNormal);
    let z2: f64 = rng.sample(StandardNormal);
    let zv = cfg.spot_vol_corr * z1
        + (1.0 - cfg.spot_vol_corr * cfg.spot_vol_corr).sqrt() * z2;
    let sq = cfg.horizon.sqrt();
    // lognormal spot move, arithmetic vol move
    let d_spot = spot * ((-0.5 * cfg.spot_vol * cfg.spot_vol * cfg.horizon
        + cfg.spot_vol * sq * z1)
        .exp()
        - 1.0);
    let d_vol = cfg.vol_of_vol * sq * zv;
    (d_spot, d_vol)
}

/// Delta-gamma-vega-theta VaR: scenario P&L from the book's aggregated
/// Greeks (one Greeks computation, then arithmetic per scenario).
pub fn delta_gamma_var(book: &EquityPortfolio, spot: f64, cfg: &RiskConfig) -> PortfolioRisk {
    let g = book.greeks();
    let pnl: Vec<f64> = (0..cfg.scenarios as u64)
        .map(|i| {
            let (ds, dv) = scenario_moves(cfg, spot, i);
            g.delta * ds
                + 0.5 * g.gamma * ds * ds
                + g.vega * dv
                + 0.5 * g.volga * dv * dv
                + g.vanna * ds * dv
                + g.theta * cfg.horizon
        })
        .collect();
    summarize(&pnl, cfg)
}

/// Full-revaluation VaR: every scenario reprices the whole book via
/// [`EquityPortfolio::price_with`] (same scenarios as
/// [`delta_gamma_var`], so the difference isolates the Taylor error).
pub fn full_revaluation_var(
    book: &EquityPortfolio,
    spot: f64,
    cfg: &RiskConfig,
) -> PortfolioRisk {
    let base: f64 = book
        .positions
        .iter()
        .map(|p| p.quantity * p.option.price_with(0.0, 0.0, 0.0, 0.0))
        .sum();
    let pnl: Vec<f64> = (0..cfg.scenarios as u64)
        .map(|i| {
            let (ds, dv) = scenario_moves(cfg, spot, i);
            let revalued: f64 = book
                .positions
                .iter()
                .map(|p| p.quantity * p.option.price_with(ds, dv, 0.0, cfg.horizon))
                .sum();
            revalued - base
        })
        .collect();
    summarize(&pnl, cfg)
}

fn summarize(pnl: &[f64], cfg: &RiskConfig) -> PortfolioRisk {
    PortfolioRisk {
        var: historical_var(pnl, cfg.confidence),
        expected_shortfall: historical_expected_shortfall(pnl, cfg.confidence),
        mean_pnl: pnl.iter().sum::<f64>() / pnl.len() as f64,
        scenarios: pnl.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use crate::core::traits::Instrument;
    use crate::equity::vanilla_option::EquityOption;
    use chrono::NaiveDate;

    const SPOT: f64 = 100.0;

    fn option(pc: PutOrCall, strike: f64, qty_engine: Engine) -> EquityOption {
        EquityOptionBuilder::new()
            .symbol("RISK")
            .spot(SPOT)
            .strike(strike)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
            .vanilla(pc)
            .engine(qty_engine)
            .build()
    }

    fn book(positions: &[(PutOrCall, f64, f64)]) -> EquityPortfolio {
        let mut b = EquityPortfolio::new();
        for &(pc, k, qty) in positions {
            b.add(option(pc, k, Engine::BlackScholes), qty);
        }
        b
    }

    #[test]
    fn long_option_var_is_bounded_by_premium_and_es_dominates() {
        // a long call can lose at most its value
        let b = book(&[(PutOrCall::Call, 100.0, 100.0)]);
        let value: f64 = 100.0 * option(PutOrCall::Call, 100.0, Engine::BlackScholes).npv();
        let cfg = RiskConfig { scenarios: 10_000, ..RiskConfig::default() };
        let full = full_revaluation_var(&b, SPOT, &cfg);
        assert!(full.var > 0.0 && full.var < value, "var {} value {value}", full.var);
        assert!(full.expected_shortfall >= full.var);
        let dg = delta_gamma_var(&b, SPOT, &cfg);
        assert!(dg.expected_shortfall >= dg.var);
    }

    #[test]
    fn delta_gamma_tracks_full_revaluation_for_one_day() {
        let b = book(&[(PutOrCall::Call, 100.0, 100.0), (PutOrCall::Put, 95.0, 50.0)]);
        let cfg = RiskConfig { scenarios: 10_000, vol_of_vol: 0.5, ..RiskConfig::default() };
        let dg = delta_gamma_var(&b, SPOT, &cfg);
        let full = full_revaluation_var(&b, SPOT, &cfg);
        // one-day moves: the Taylor truncation is small
        assert!(
            (dg.var - full.var).abs() < 0.10 * full.var.max(1.0),
            "dg {} vs full {}",
            dg.var,
            full.var
        );
    }

    #[test]
    fn hedging_reduces_var_and_gamma_shows_in_the_comparison() {
        let cfg = RiskConfig { scenarios: 10_000, ..RiskConfig::default() };
        // naked short call vs the same with a long ATM call hedge
        let naked = book(&[(PutOrCall::Call, 100.0, -100.0)]);
        let hedged = book(&[
            (PutOrCall::Call, 100.0, -100.0),
            (PutOrCall::Call, 105.0, 100.0),
        ]);
        let naked_var = full_revaluation_var(&naked, SPOT, &cfg).var;
        let hedged_var = full_revaluation_var(&hedged, SPOT, &cfg).var;
        assert!(hedged_var < naked_var, "hedged {hedged_var} vs naked {naked_var}");
        // for the short book the delta-gamma estimate must not report a
        // negative-loss (profit) VaR
        assert!(delta_gamma_var(&naked, SPOT, &cfg).var > 0.0);
    }

    #[test]
    fn vol_scenarios_add_risk_to_a_vega_book() {
        // long straddle: pure spot scenarios miss the vega risk of a
        // vol crush; adding vol scenarios raises the VaR
        let b = book(&[(PutOrCall::Call, 100.0, 100.0), (PutOrCall::Put, 100.0, 100.0)]);
        let no_vol = RiskConfig { scenarios: 10_000, vol_of_vol: 0.0, ..RiskConfig::default() };
        let with_vol = RiskConfig { scenarios: 10_000, vol_of_vol: 0.8, ..RiskConfig::default() };
        let base = full_revaluation_var(&b, SPOT, &no_vol).var;
        let vol_aware = full_revaluation_var(&b, SPOT, &with_vol).var;
        assert!(vol_aware > base, "with vol {vol_aware} vs without {base}");
    }
}
