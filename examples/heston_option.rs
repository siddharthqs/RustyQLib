//! Heston stochastic volatility: semi-analytic characteristic-function
//! pricing vs Monte Carlo, and the smile the model produces.
//!
//! Run with:  cargo run --release --example heston_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::barrier::{BarrierDirection, KnockType};
use rustyqlib::equity::blackscholes::{bs_price, implied_vol_from_price};
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::heston::{heston_price, HestonParams};
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanila_option::BinaryType;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;

fn params() -> HestonParams {
    HestonParams { v0: 0.09, kappa: 2.0, theta: 0.09, vol_of_vol: 0.4, rho: -0.7 }
}

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("HESTON")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(0.30) // only used as the vega-bump reference
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
        .heston(params())
}

fn main() {
    let p = params();
    common::title(&format!(
        "HESTON — v0={} kappa={} theta={} vol-of-vol={} rho={}",
        p.v0, p.kappa, p.theta, p.vol_of_vol, p.rho
    ));
    common::note(&format!(
        "Feller condition 2*kappa*theta >= vol_of_vol^2: {}",
        if p.feller_condition_holds() { "holds" } else { "VIOLATED (variance can touch zero)" }
    ));

    common::section("Vanilla: semi-analytic vs Monte Carlo");
    common::table_header();
    for pc in [PutOrCall::Call, PutOrCall::Put] {
        common::row(
            &format!("Analytical (char. function), {pc:?}"),
            &base().vanilla(pc).engine(Engine::BlackScholes).build(),
        );
        common::row(
            &format!("Monte Carlo (full-trunc Euler), {pc:?}"),
            &base().vanilla(pc).engine(Engine::MonteCarlo).paths(100_000).build(),
        );
    }
    common::row(
        "Finite difference (unsupported)",
        &base().vanilla(PutOrCall::Call).engine(Engine::FiniteDifference).build(),
    );
    common::note("MC vega/theta bump sqrt(v0) and sqrt(theta) in parallel");

    common::section("Binaries under Heston");
    common::table_header();
    common::row(
        "Cash-or-nothing call (analytic)",
        &base()
            .binary(PutOrCall::Call, BinaryType::CashOrNothing, 1.0)
            .engine(Engine::BlackScholes)
            .build(),
    );
    common::row(
        "Cash-or-nothing call (MC)",
        &base()
            .binary(PutOrCall::Call, BinaryType::CashOrNothing, 1.0)
            .engine(Engine::MonteCarlo)
            .paths(100_000)
            .build(),
    );
    common::row(
        "Asset-or-nothing call (analytic)",
        &base()
            .binary(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0)
            .engine(Engine::BlackScholes)
            .build(),
    );

    common::section("Path-dependent payoffs (Monte Carlo only)");
    common::table_header();
    common::row(
        "Down-and-out call H=85",
        &base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 85.0)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );
    common::row(
        "Down-and-in put H=85",
        &base()
            .barrier(PutOrCall::Put, BarrierDirection::Down, KnockType::In, 85.0)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );

    common::section("Identities");
    let call = base().vanilla(PutOrCall::Call).engine(Engine::BlackScholes).build();
    let put = base().vanilla(PutOrCall::Put).engine(Engine::BlackScholes).build();
    let parity = SPOT * (-DIV * 1.0_f64).exp() - STRIKE * (-RATE * 1.0_f64).exp();
    common::check("put-call parity", call.npv() - put.npv(), parity, 1e-10);
    let asset = base()
        .binary(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0)
        .engine(Engine::BlackScholes)
        .build();
    let k_cash = base()
        .binary(PutOrCall::Call, BinaryType::CashOrNothing, STRIKE)
        .engine(Engine::BlackScholes)
        .build();
    common::check("vanilla = asset digital - K cash digitals", call.npv(), asset.npv() - k_cash.npv(), 1e-10);
    common::check(
        "vol-of-vol -> 0 degenerates to Black-Scholes",
        heston_price(
            SPOT,
            STRIKE,
            RATE,
            DIV,
            1.0,
            &HestonParams { vol_of_vol: 1e-4, ..params() },
            PutOrCall::Call,
        ),
        bs_price(SPOT, STRIKE, RATE, DIV, p.v0.sqrt(), 1.0, PutOrCall::Call),
        1e-4,
    );

    common::section("The Heston smile (implied vol backed out of Heston prices)");
    println!("  {:>8} {:>14} {:>14}", "strike", "heston price", "implied vol");
    for k in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0] {
        let price = heston_price(SPOT, k, RATE, DIV, 1.0, &p, PutOrCall::Call);
        let iv = implied_vol_from_price(SPOT, k, RATE, DIV, 1.0, price, PutOrCall::Call)
            .unwrap_or(f64::NAN);
        println!("  {k:>8.1} {price:>14.6} {:>13.4}%", iv * 100.0);
    }
    common::note("rho < 0 tilts the smile: low strikes carry higher implied vol");

    common::section("Correlation and vol-of-vol control the smile shape");
    println!("  {:>6} {:>8} {:>12} {:>12} {:>12}", "rho", "vol-of-vol", "iv(80)", "iv(100)", "iv(120)");
    for (rho, vov) in [(-0.7, 0.4), (0.0, 0.4), (0.7, 0.4), (-0.7, 0.1), (-0.7, 0.8)] {
        let hp = HestonParams { rho, vol_of_vol: vov, ..params() };
        let iv = |k: f64| {
            let price = heston_price(SPOT, k, RATE, DIV, 1.0, &hp, PutOrCall::Call);
            implied_vol_from_price(SPOT, k, RATE, DIV, 1.0, price, PutOrCall::Call)
                .unwrap_or(f64::NAN)
                * 100.0
        };
        println!("  {rho:>6.1} {vov:>10.1} {:>11.3}% {:>11.3}% {:>11.3}%", iv(80.0), iv(100.0), iv(120.0));
    }
    common::note("rho controls the skew (tilt); vol-of-vol controls the smile (curvature)");
    println!();
}
