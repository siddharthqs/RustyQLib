//! Options on futures priced with Black-76: standard (discounted, premium
//! paid up front) and futures-style (margined, undiscounted).
//!
//! Run with:  cargo run --release --example futures_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::black76::{price, FuturesSettlement};
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::EquityOption;

const F: f64 = 100.0; // futures price
const K: f64 = 100.0;
const VOL: f64 = 0.30;
const R: f64 = 0.05;
const T: f64 = 1.0;

fn futures_option(pc: PutOrCall, settlement: FuturesSettlement) -> EquityOption {
    EquityOptionBuilder::new()
        .symbol("FUT")
        .spot(F) // the futures price F
        .strike(K)
        .flat_vol(VOL)
        .flat_rate(R)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
        .vanilla(pc)
        .on_future(settlement)
        .engine(Engine::BlackScholes)
        .build()
}

fn main() {
    common::title("OPTIONS ON FUTURES (Black-76) — F=100 K=100 sigma=30% r=5% T=1y");

    for (name, settlement) in [
        ("Discounted (standard Black-76)", FuturesSettlement::Discounted),
        ("Margined (futures-style)", FuturesSettlement::Margined),
    ] {
        common::section(name);
        common::table_header();
        common::row("call", &futures_option(PutOrCall::Call, settlement));
        common::row("put", &futures_option(PutOrCall::Put, settlement));
    }
    common::note("margined has zero rho (no discounting) and a larger vega/theta");

    common::section("Settlement effect: margined = discounted / e^{-rT}");
    let disc = futures_option(PutOrCall::Call, FuturesSettlement::Discounted).npv();
    let marg = futures_option(PutOrCall::Call, FuturesSettlement::Margined).npv();
    println!("  discounted call {disc:.6}   margined call {marg:.6}   ratio {:.6} (= e^rT {:.6})",
        marg / disc, (R * T).exp());

    common::section("Identities");
    let dc = futures_option(PutOrCall::Call, FuturesSettlement::Discounted);
    let dp = futures_option(PutOrCall::Put, FuturesSettlement::Discounted);
    common::check(
        "discounted parity C - P = e^{-rT}(F - K)",
        dc.npv() - dp.npv(),
        (-R * T).exp() * (F - K),
        1e-10,
    );
    let mc = futures_option(PutOrCall::Call, FuturesSettlement::Margined);
    let mp = futures_option(PutOrCall::Put, FuturesSettlement::Margined);
    common::check("margined parity C - P = F - K", mc.npv() - mp.npv(), F - K, 1e-10);
    common::check(
        "margined rho is exactly zero",
        futures_option(PutOrCall::Call, FuturesSettlement::Margined).rho(),
        0.0,
        1e-15,
    );

    common::section("Black-76 on the forward reproduces spot Black-Scholes");
    // an option on F = S e^{(r-q)T} equals the equivalent spot option
    let (s, q) = (100.0, 0.02);
    let fwd = s * ((R - q) * T).exp();
    let on_forward = price(fwd, K, R, VOL, T, PutOrCall::Call, FuturesSettlement::Discounted);
    let spot_bsm = bs_price(s, K, R, q, VOL, T, PutOrCall::Call);
    common::check("black76(F = S e^{(r-q)T}) = BSM(S, q)", on_forward, spot_bsm, 1e-10);

    common::section("Skew across strikes (discounted put)");
    common::table_header();
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
        common::row(
            &format!("K = {k}"),
            &EquityOptionBuilder::new()
                .spot(F)
                .strike(k)
                .flat_vol(VOL)
                .flat_rate(R)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
                .vanilla(PutOrCall::Put)
                .on_future(FuturesSettlement::Discounted)
                .engine(Engine::BlackScholes)
                .build(),
        );
    }
    println!();
}
