//! Forward-start options: strike fixed at a future date as k * S(t_f).
//! The product that exposes the *forward smile*, so Heston and
//! Black-Scholes disagree by construction.
//!
//! Run with:  cargo run --release --example forward_start_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::forward_start_option::forward_start_price;
use rustyqlib::equity::heston::HestonParams;
use rustyqlib::equity::montecarlo::McModel;
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;
const START: f64 = 0.5; // fixing at half the option's life

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("FWDSTART")
        .spot(SPOT)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
}

fn heston_params(vol_of_vol: f64, rho: f64) -> HestonParams {
    HestonParams { v0: VOL * VOL, kappa: 2.0, theta: VOL * VOL, vol_of_vol, rho }
}

fn main() {
    common::title("FORWARD-START OPTION — S=100, strike = 1.0 x S(0.5y), T=1y, sigma=30%");

    common::section("Black-Scholes: analytic vs Monte Carlo");
    common::table_header();
    common::row(
        "Analytical (Rubinstein)",
        &base()
            .forward_start(PutOrCall::Call, 1.0, START)
            .engine(Engine::BlackScholes)
            .build(),
    );
    common::row(
        "Monte Carlo (GBM)",
        &base()
            .forward_start(PutOrCall::Call, 1.0, START)
            .engine(Engine::MonteCarlo)
            .paths(100_000)
            .build(),
    );
    common::row(
        "Finite difference (unsupported)",
        &base()
            .forward_start(PutOrCall::Call, 1.0, START)
            .engine(Engine::FiniteDifference)
            .build(),
    );

    common::section("Forward smile: Heston vs Black-Scholes");
    common::table_header();
    let bs = base()
        .forward_start(PutOrCall::Call, 1.0, START)
        .engine(Engine::BlackScholes)
        .build()
        .npv();
    common::row(
        "Heston vol-of-vol=0.001 (-> BS)",
        &base()
            .forward_start(PutOrCall::Call, 1.0, START)
            .engine(Engine::MonteCarlo)
            .heston(heston_params(1e-3, 0.0))
            .paths(50_000)
            .build(),
    );
    for (vov, rho) in [(0.2, -0.7), (0.4, -0.7), (0.6, -0.7), (0.4, 0.0)] {
        common::row(
            &format!("Heston vol-of-vol={vov}, rho={rho}"),
            &base()
                .forward_start(PutOrCall::Call, 1.0, START)
                .engine(Engine::MonteCarlo)
                .heston(heston_params(vov, rho))
                .paths(50_000)
                .build(),
        );
    }
    common::note(&format!("Black-Scholes reference: {bs:.6}"));
    common::note("the gap is the forward-smile effect — the reason to price these on a stoch-vol model");

    common::section("Strike fraction sweep (analytic)");
    common::table_header();
    for k in [0.9, 0.95, 1.0, 1.05, 1.1] {
        common::row(
            &format!("strike = {k} x S(t_f), call"),
            &base().forward_start(PutOrCall::Call, k, START).engine(Engine::BlackScholes).build(),
        );
    }

    common::section("Fixing date sweep (analytic, ATM)");
    common::table_header();
    for start in [0.1, 0.25, 0.5, 0.75, 0.9] {
        common::row(
            &format!("fixing at {:.0}% of life", start * 100.0),
            &base().forward_start(PutOrCall::Call, 1.0, start).engine(Engine::BlackScholes).build(),
        );
    }
    common::note("later fixing leaves less time to expiry, so the option is worth less");

    common::section("Identities");
    common::check(
        "immediate fixing -> vanilla struck at S0",
        forward_start_price(SPOT, 1.0, RATE, DIV, VOL, 1e-6, 1.0, PutOrCall::Call),
        base()
            .strike(SPOT)
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .npv(),
        1e-3,
    );
    let p100 = forward_start_price(100.0, 1.0, RATE, DIV, VOL, 0.5, 1.0, PutOrCall::Call);
    let p200 = forward_start_price(200.0, 1.0, RATE, DIV, VOL, 0.5, 1.0, PutOrCall::Call);
    common::check("homogeneity: price(2S) = 2 price(S)", p200, 2.0 * p100, 1e-12);
    let fs = base()
        .forward_start(PutOrCall::Call, 1.0, START)
        .engine(Engine::BlackScholes)
        .build();
    common::check("delta = price / spot (homogeneity)", fs.delta(), fs.npv() / SPOT, 1e-6);
    println!();
}
