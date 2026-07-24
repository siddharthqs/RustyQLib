//! American options via the Barone-Adesi-Whaley (BAW) quadratic approximation
//! — a fast analytic alternative to a tree or PDE solve.
//!
//! Run with:  cargo run --release --example american_baw

mod common;

use std::time::Instant;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::baw;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanila_option::EquityOption;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.25;
const RATE: f64 = 0.08;
const DIV: f64 = 0.04; // continuous carry (dividend + borrow); makes early exercise bite

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

/// Market/contract stub without the payoff — the caller adds the exercise
/// style and `.vanilla(pc)`. Order matters: `.american()` must precede
/// `.vanilla()`, which is when the payoff (and its exercise style) is built.
fn contract() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("ACME")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(asof())
        .maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap()) // ~0.5y
}

fn american(put_or_call: PutOrCall, engine: Engine) -> EquityOption {
    contract().american().vanilla(put_or_call).engine(engine).build()
}

fn european(put_or_call: PutOrCall) -> EquityOption {
    contract().vanilla(put_or_call).engine(Engine::BlackScholes).build()
}

fn main() {
    common::title("AMERICAN OPTIONS, ANALYTIC: BAW & BJERKSUND-STENSLAND 2002 — S=100 K=100 sigma=25% r=8% q=4% T=0.5y");

    for pc in [PutOrCall::Put, PutOrCall::Call] {
        common::section(&format!("American {pc:?}: analytic approximations vs the exact engines"));
        common::table_header();
        common::row("Barone-Adesi-Whaley (analytic)", &american(pc, Engine::BaroneAdesiWhaley));
        common::row("Bjerksund-Stensland 2002", &american(pc, Engine::BjerksundStensland));
        common::row("Binomial (CRR tree)", &american(pc, Engine::Binomial));
        common::row("Finite difference (Brennan-Schwartz)", &american(pc, Engine::FiniteDifference));
        common::row("Monte Carlo (Longstaff-Schwartz)",
            &contract().american().vanilla(pc).engine(Engine::MonteCarlo).paths(50_000).build());
        // the analytic Black-Scholes engine has no American method and says so
        common::row("Black-Scholes (rejects American)", &american(pc, Engine::BlackScholes));

        // early-exercise premium and the exercise boundary
        let european = european(pc).npv();
        let baw_opt = american(pc, Engine::BaroneAdesiWhaley);
        let premium = baw_opt.npv() - european;
        let boundary = baw::critical_spot(&baw_opt);
        common::note(&format!("European price {european:.6}"));
        common::note(&format!("early-exercise premium (BAW - European) = {premium:.6}"));
        common::note(&format!(
            "critical exercise spot S* = {boundary:.4}  (exercise once spot {} it)",
            if matches!(pc, PutOrCall::Put) { "falls below" } else { "rises above" }
        ));
    }

    common::section("Accuracy: both approximations within a few cents of a fine tree");
    common::table_header();
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
        let baw_opt = contract().strike(k).american().vanilla(PutOrCall::Put)
            .engine(Engine::BaroneAdesiWhaley).build();
        common::row(&format!("BAW    put K={k}"), &baw_opt);
    }
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
        let bs = contract().strike(k).american().vanilla(PutOrCall::Put)
            .engine(Engine::BjerksundStensland).build();
        common::row(&format!("BS2002 put K={k}"), &bs);
    }
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
        let tree = contract().strike(k).american().vanilla(PutOrCall::Put)
            .engine(Engine::Binomial).build();
        common::row(&format!("tree   put K={k}"), &tree);
    }
    common::note("BS2002 uses a feasible two-step exercise boundary, so it is a lower");
    common::note("bound on the true price; BAW is not a bound and can land either side.");

    common::section("Speed: why you would reach for BAW");
    let baw_opt = american(PutOrCall::Put, Engine::BaroneAdesiWhaley);
    let tree = american(PutOrCall::Put, Engine::Binomial);
    let reps = 20_000;
    let t0 = Instant::now();
    let mut acc = 0.0;
    for _ in 0..reps {
        acc += baw_opt.npv();
    }
    let baw_ns = t0.elapsed().as_nanos() as f64 / reps as f64;
    let t1 = Instant::now();
    for _ in 0..reps {
        acc += tree.npv();
    }
    let tree_ns = t1.elapsed().as_nanos() as f64 / reps as f64;
    std::hint::black_box(acc);
    println!("  BAW    : {baw_ns:>10.0} ns / price");
    println!("  tree   : {tree_ns:>10.0} ns / price");
    println!("  BAW is ~{:.0}x faster for ~{:.3} of price error",
        tree_ns / baw_ns, (baw_opt.npv() - tree.npv()).abs());

    common::section("Checks");
    let put = american(PutOrCall::Put, Engine::BaroneAdesiWhaley);
    let tree_put = american(PutOrCall::Put, Engine::Binomial);
    common::check("BAW American put ~= binomial", put.npv(), tree_put.npv(), 0.05);
    // a non-dividend American call is never exercised early -> equals European
    let call_no_div = EquityOptionBuilder::new()
        .symbol("ACME").spot(SPOT).strike(STRIKE).flat_vol(VOL).flat_rate(RATE)
        .dividend_yield(0.0)
        .valuation_date(asof()).maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
        .american().vanilla(PutOrCall::Call).engine(Engine::BaroneAdesiWhaley).build();
    let euro_call = EquityOptionBuilder::new()
        .symbol("ACME").spot(SPOT).strike(STRIKE).flat_vol(VOL).flat_rate(RATE)
        .dividend_yield(0.0)
        .valuation_date(asof()).maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
        .vanilla(PutOrCall::Call).engine(Engine::BlackScholes).build();
    common::check("non-dividend American call = European", call_no_div.npv(), euro_call.npv(), 1e-9);
    println!();
}
