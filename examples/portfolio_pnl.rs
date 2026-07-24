//! Portfolio Greeks aggregation and risk-based PnL attribution for a book of
//! options on one underlying.
//!
//! Run with:  cargo run --release --example portfolio_pnl

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::portfolio::{EquityPortfolio, MarketMove, PnlAttribution};
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::EquityOption;

const SPOT: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

fn option(put_or_call: PutOrCall, strike: f64, months: u32) -> EquityOption {
    let maturity = if months >= 12 {
        NaiveDate::from_ymd_opt(2027, months - 11, 1).unwrap()
    } else {
        NaiveDate::from_ymd_opt(2026, months + 1, 1).unwrap()
    };
    EquityOptionBuilder::new()
        .symbol("ACME")
        .spot(SPOT)
        .strike(strike)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(asof())
        .maturity_date(maturity)
        .vanilla(put_or_call)
        .engine(Engine::BlackScholes)
        .build()
}

fn print_attribution(a: &PnlAttribution) {
    println!("  {:<14} {:>12.4}", "delta", a.delta_pnl);
    println!("  {:<14} {:>12.4}", "gamma", a.gamma_pnl);
    println!("  {:<14} {:>12.4}", "vega", a.vega_pnl);
    println!("  {:<14} {:>12.4}", "volga", a.volga_pnl);
    println!("  {:<14} {:>12.4}", "vanna", a.vanna_pnl);
    println!("  {:<14} {:>12.4}", "theta", a.theta_pnl);
    println!("  {:<14} {:>12.4}", "rho", a.rho_pnl);
    println!("  {:<14} {:>12.4}", "explained", a.explained);
    println!("  {:<14} {:>12.4}", "actual", a.actual);
    println!("  {:<14} {:>12.4}  ({:.2}% of actual)",
        "unexplained", a.unexplained,
        if a.actual.abs() > 1e-12 { 100.0 * a.unexplained / a.actual } else { 0.0 });
}

fn main() {
    common::title("PORTFOLIO — one underlying, aggregated Greeks and PnL attribution");

    // a realistic single-name book: long 1y ATM calls, short 6m upside
    // calls (call spread financing), long 3m downside puts (crash hedge)
    let mut book = EquityPortfolio::new();
    book.add(option(PutOrCall::Call, 100.0, 12), 100.0);
    book.add(option(PutOrCall::Call, 110.0, 6), -150.0);
    book.add(option(PutOrCall::Put, 90.0, 3), 80.0);

    common::section("Positions");
    println!("  {:>8}  {:<6} {:>8} {:>10}  {:>12} {:>10}",
        "qty", "type", "strike", "expiry", "npv", "delta");
    for p in &book.positions {
        println!("  {:>8.0}  {:<6} {:>8.2} {:>10}  {:>12.4} {:>10.4}",
            p.quantity,
            format!("{:?}", p.option.payoff.put_or_call()),
            p.option.base.strike_price,
            p.option.base.maturity_date,
            p.option.npv(),
            p.option.delta());
    }

    common::section("Aggregated book Greeks (quantity-weighted)");
    let g = book.greeks();
    println!("  {:<10} {:>14.4}", "npv", g.npv);
    println!("  {:<10} {:>14.4}", "delta", g.delta);
    println!("  {:<10} {:>14.4}", "gamma", g.gamma);
    println!("  {:<10} {:>14.4}", "vega", g.vega);
    println!("  {:<10} {:>14.4}", "theta", g.theta);
    println!("  {:<10} {:>14.4}", "rho", g.rho);
    println!("  {:<10} {:>14.4}", "vanna", g.vanna);
    println!("  {:<10} {:>14.4}", "charm", g.charm);
    println!("  {:<10} {:>14.4}", "zomma", g.zomma);
    println!("  {:<10} {:>14.4}", "volga", g.volga);

    common::section("Scenario 1: quiet day — spot +0.5, vol -25bp, one day");
    let quiet = MarketMove { d_spot: 0.5, d_vol: -0.0025, d_rate: 0.0, d_time: 1.0 / 365.0 };
    let a1 = book.pnl_attribution(&quiet);
    print_attribution(&a1);

    common::section("Scenario 2: risk-off — spot -5, vol +4pts, one day");
    let riskoff = MarketMove { d_spot: -5.0, d_vol: 0.04, d_rate: -0.001, d_time: 1.0 / 365.0 };
    let a2 = book.pnl_attribution(&riskoff);
    print_attribution(&a2);
    common::note("larger moves leave more in unexplained: the Taylor expansion");
    common::note("is second order, so the residual grows with the cube of the move");

    common::section("Checks");
    common::check(
        "explained + unexplained = actual (scenario 2)",
        a2.explained + a2.unexplained,
        a2.actual,
        1e-10,
    );
    // a delta hedge of -delta shares removes exactly the delta bucket: the
    // hedged book's quiet-day PnL is the attribution minus the delta term
    let hedge_pnl = -g.delta * quiet.d_spot;
    common::check(
        "delta-hedged quiet-day PnL = actual - delta bucket",
        a1.actual + hedge_pnl,
        a1.actual - a1.delta_pnl,
        1e-10,
    );
    common::note(&format!(
        "stock hedge: {} {:.2} shares; unexplained: quiet {:+.4} vs risk-off {:+.4}",
        if g.delta < 0.0 { "buy" } else { "sell" },
        g.delta.abs(),
        a1.unexplained,
        a2.unexplained
    ));
    println!();
}
