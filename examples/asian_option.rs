//! Asian (average) options: arithmetic / geometric, fixed / floating strike.
//!
//! Run with:  cargo run --release --example asian_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::asian::{
    geometric_asian_price, turnbull_wakeman_price, AsianStrikeType, AveragingType,
};
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::montecarlo::DiscretizationScheme;
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("ASIAN")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
}

fn main() {
    common::title("ASIAN OPTIONS — S=100 K=100 sigma=30% r=5% q=2% T=1y");

    common::section("Fixed strike (average price) call");
    common::table_header();
    common::row(
        "Geometric, analytic (exact)",
        &base()
            .asian(PutOrCall::Call, AveragingType::Geometric, AsianStrikeType::FixedStrike)
            .engine(Engine::BlackScholes)
            .build(),
    );
    common::row(
        "Geometric, Monte Carlo",
        &base()
            .asian(PutOrCall::Call, AveragingType::Geometric, AsianStrikeType::FixedStrike)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );
    common::row(
        "Arithmetic, Turnbull-Wakeman",
        &base()
            .asian(PutOrCall::Call, AveragingType::Arithmetic, AsianStrikeType::FixedStrike)
            .engine(Engine::BlackScholes)
            .build(),
    );
    common::row(
        "Arithmetic, MC + geometric CV",
        &base()
            .asian(PutOrCall::Call, AveragingType::Arithmetic, AsianStrikeType::FixedStrike)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );

    common::section("Control variate effect (same path count)");
    common::table_header();
    let with_cv = base()
        .asian(PutOrCall::Call, AveragingType::Arithmetic, AsianStrikeType::FixedStrike)
        .engine(Engine::MonteCarlo)
        .paths(20_000)
        .build();
    let without_cv = base()
        .asian(PutOrCall::Call, AveragingType::Arithmetic, AsianStrikeType::FixedStrike)
        .engine(Engine::MonteCarlo)
        .paths(20_000)
        .mc_config({
            // Euler stepping disables the control variate precondition
            let mut c = rustyqlib::equity::montecarlo::MonteCarloConfig::default();
            c.paths = 20_000;
            c.scheme = DiscretizationScheme::Euler;
            c.time_steps = 100;
            c
        })
        .build();
    common::row("with geometric control variate", &with_cv);
    common::row("without (Euler path route)", &without_cv);
    common::note("compare the std err column: the CV collapses the variance");

    common::section("Floating strike (average strike)");
    common::table_header();
    for pc in [PutOrCall::Call, PutOrCall::Put] {
        common::row(
            &format!("Monte Carlo, {pc:?}"),
            &base()
                .asian(pc, AveragingType::Arithmetic, AsianStrikeType::FloatingStrike)
                .engine(Engine::MonteCarlo)
                .paths(50_000)
                .build(),
        );
        common::row(
            &format!("Analytic (unsupported), {pc:?}"),
            &base()
                .asian(pc, AveragingType::Arithmetic, AsianStrikeType::FloatingStrike)
                .engine(Engine::BlackScholes)
                .build(),
        );
    }

    common::section("Orderings and limits");
    let vanilla = base().vanilla(PutOrCall::Call).engine(Engine::BlackScholes).build().npv();
    let geo = geometric_asian_price(SPOT, STRIKE, RATE, DIV, VOL, 1.0, None, PutOrCall::Call);
    let arith = turnbull_wakeman_price(SPOT, STRIKE, RATE, DIV, VOL, 1.0, PutOrCall::Call);
    println!("  geometric {geo:.6} < arithmetic {arith:.6} < vanilla {vanilla:.6}");
    common::note("AM-GM: the arithmetic average dominates the geometric one");
    common::note("averaging reduces effective volatility (sigma^2 T / 3), so both sit below vanilla");
    common::check(
        "discrete geometric (n=1e5) -> continuous",
        geometric_asian_price(SPOT, STRIKE, RATE, DIV, VOL, 1.0, Some(100_000), PutOrCall::Call),
        geo,
        1e-3,
    );

    common::section("Averaging frequency (geometric, exact)");
    for n in [4usize, 12, 52, 252] {
        let price =
            geometric_asian_price(SPOT, STRIKE, RATE, DIV, VOL, 1.0, Some(n), PutOrCall::Call);
        println!("  {n:>4} fixings: {price:.6}");
    }
    println!();
}
