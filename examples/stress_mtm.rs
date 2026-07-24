//! Stress MtM from a TOML shock configuration: trade-level results and
//! per-scenario portfolio aggregation.
//!
//! Run with:  cargo run --release --example stress_mtm

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::portfolio::EquityPortfolio;
use rustyqlib::equity::utils::Engine;
use rustyqlib::risk::{stress_mtm, StressConfig};

fn option(pc: PutOrCall, strike: f64, months: u32) -> rustyqlib::equity::vanilla_option::EquityOption {
    EquityOptionBuilder::new()
        .symbol("ACME")
        .spot(100.0)
        .strike(strike)
        .flat_vol(0.25)
        .flat_rate(0.03)
        .dividend_yield(0.01)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2026, 1 + months, 1).unwrap())
        .vanilla(pc)
        .engine(Engine::BlackScholes)
        .build()
}

fn main() {
    common::title("STRESS MtM — TOML scenarios, trade-level and aggregated");

    let mut book = EquityPortfolio::new();
    book.add(option(PutOrCall::Call, 100.0, 11), 100.0);
    book.add(option(PutOrCall::Call, 110.0, 5), -150.0);
    book.add(option(PutOrCall::Put, 90.0, 3), 80.0);

    let config = StressConfig::from_toml_file("src/examples/stress_config.toml")
        .expect("loading stress config");
    common::note(&format!("loaded {} scenarios from src/examples/stress_config.toml",
        config.scenarios.len()));

    for result in stress_mtm(&book, &config) {
        common::section(&format!("Scenario: {}", result.scenario));
        println!("  {:<34} {:>12} {:>12} {:>12}", "trade", "base MtM", "stressed", "stress P&L");
        for trade in &result.trades {
            println!(
                "  {:<34} {:>12.2} {:>12.2} {:>+12.2}",
                trade.label, trade.base_mtm, trade.stressed_mtm, trade.stress_pnl
            );
        }
        println!("  {:-<74}", "");
        println!(
            "  {:<34} {:>12.2} {:>12.2} {:>+12.2}",
            "PORTFOLIO", result.base_mtm, result.stressed_mtm, result.stress_pnl
        );
    }
    println!();
}
