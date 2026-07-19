//! Carry inputs: continuous dividend yield, discrete cash dividends and
//! stock borrow cost, and how each engine treats them.
//!
//! Run with:  cargo run --release --example dividends_and_borrow

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::barrier::{BarrierDirection, KnockType};
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}
fn expiry() -> NaiveDate {
    NaiveDate::from_ymd_opt(2027, 1, 1).unwrap()
}

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("CARRY")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .valuation_date(asof())
        .maturity_date(expiry())
}

fn main() {
    common::title("DIVIDENDS AND BORROW COST — S=100 K=100 sigma=30% r=5% T=1y");

    common::section("Continuous carry: dividend yield and borrow cost are interchangeable");
    common::table_header();
    common::row("no carry", &base().vanilla(PutOrCall::Call).build());
    common::row("q = 4%", &base().dividend_yield(0.04).vanilla(PutOrCall::Call).build());
    common::row("borrow = 4%", &base().borrow_cost(0.04).vanilla(PutOrCall::Call).build());
    common::row(
        "q = 1% + borrow = 3%",
        &base().dividend_yield(0.01).borrow_cost(0.03).vanilla(PutOrCall::Call).build(),
    );
    common::note("carry_yield() = dividend_yield + borrow_cost enters every formula as 'q'");

    let q_only = base().dividend_yield(0.04).vanilla(PutOrCall::Call).build();
    let split = base().dividend_yield(0.01).borrow_cost(0.03).vanilla(PutOrCall::Call).build();
    common::check("q=4% vs q=1%+b=3%", split.npv(), q_only.npv(), 1e-12);

    common::section("Hard-to-borrow names: high borrow cost lowers the forward");
    common::table_header();
    for b in [0.0, 0.02, 0.05, 0.15] {
        let option = base().borrow_cost(b).vanilla(PutOrCall::Call).build();
        common::row(&format!("borrow = {:.0}%", b * 100.0), &option);
    }
    let hard = base().borrow_cost(0.15).vanilla(PutOrCall::Call).build();
    println!(
        "  forward with 15% borrow: {:.4} (vs spot {SPOT})",
        hard.base.forward_price()
    );

    common::section("Discrete cash dividends: 2 x 1.50 over the year");
    let with_divs = |b: EquityOptionBuilder| {
        b.cash_dividend(NaiveDate::from_ymd_opt(2026, 4, 1).unwrap(), 1.5)
            .cash_dividend(NaiveDate::from_ymd_opt(2026, 10, 1).unwrap(), 1.5)
    };
    let analytic = with_divs(base()).vanilla(PutOrCall::Call).build();
    println!(
        "  spot {SPOT} - PV(dividends) {:.6} = escrowed spot {:.6}",
        analytic.base.pv_cash_dividends(),
        analytic.base.effective_spot()
    );
    common::table_header();
    common::row("Analytical (escrowed model)", &analytic);
    common::row(
        "Binomial (escrowed)",
        &with_divs(base()).vanilla(PutOrCall::Call).engine(Engine::Binomial).build(),
    );
    common::row(
        "Finite difference (jump model)",
        &with_divs(base()).vanilla(PutOrCall::Call).engine(Engine::FiniteDifference).build(),
    );
    common::row(
        "Monte Carlo terminal (escrowed)",
        &with_divs(base()).vanilla(PutOrCall::Call).engine(Engine::MonteCarlo).build(),
    );
    common::row(
        "Monte Carlo path-wise (jump model)",
        &with_divs(base())
            .vanilla(PutOrCall::Call)
            .engine(Engine::MonteCarlo)
            .mc_time_steps(200)
            .paths(50_000)
            .build(),
    );
    common::note("escrowed: lognormal on S - PV(divs); jump: dividends subtracted at each ex-date");
    common::note("the two models differ slightly by construction — that gap is expected, not a bug");

    common::check(
        "escrowed analytic == BS on the escrowed spot",
        analytic.npv(),
        bs_price(analytic.base.effective_spot(), STRIKE, RATE, 0.0, VOL, 1.0, PutOrCall::Call),
        1e-10,
    );

    common::section("Where the jump model matters: American exercise and barriers");
    common::table_header();
    common::row(
        "American put, FD (jumps)",
        &with_divs(base())
            .american()
            .vanilla(PutOrCall::Put)
            .engine(Engine::FiniteDifference)
            .build(),
    );
    common::row(
        "American put, no dividends",
        &base().american().vanilla(PutOrCall::Put).engine(Engine::FiniteDifference).build(),
    );
    common::row(
        "Down-and-out call H=85, MC (jumps)",
        &with_divs(base())
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 85.0)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );
    common::row(
        "Down-and-out call H=85, no dividends",
        &base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 85.0)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );
    common::note("dividend drops push the path toward a down barrier and change exercise timing");

    common::section("Put-call parity with full carry");
    let call = with_divs(base()).borrow_cost(0.02).vanilla(PutOrCall::Call).build();
    let put = with_divs(base()).borrow_cost(0.02).vanilla(PutOrCall::Put).build();
    let parity = call.base.effective_spot() * (-call.base.carry_yield() * 1.0_f64).exp()
        - STRIKE * (-RATE * 1.0_f64).exp();
    common::check("C - P = S_eff e^{-(q+b)T} - K e^{-rT}", call.npv() - put.npv(), parity, 1e-10);
    println!();
}
