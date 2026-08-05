//! Stress / what-if benchmarks: the scenario revaluation path.
//!
//! Measures the market-context machinery around pricing — snapshotting a
//! book into a typed `Market`, bumping it per scenario, and rebinding
//! every position for full revaluation — on cheap analytic engines, so
//! the context overhead (clones, allocation, rebinds) is what dominates
//! and regressions in it are visible. Run with `cargo bench --bench stress`;
//! criterion compares against the previous run.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::portfolio::EquityPortfolio;
use rustyqlib::equity::utils::Engine;
use rustyqlib::risk::stress::{
    stress_mtm, ArbitrageCheck, BumpMode, RiskFactor, Shock, StressConfig, StressScenario,
};
use rustyqlib::Instrument;

fn shock(factor: RiskFactor, mode: BumpMode, size: f64, underlying: Option<&str>) -> Shock {
    Shock {
        factor,
        mode,
        size,
        underlying: underlying.map(str::to_string),
        tenors: None,
        shifts: None,
    }
}

/// 100 Black-Scholes vanillas on one name (EquityPortfolio books are
/// single-underlying): a strike ladder, mixed long/short.
fn book() -> EquityPortfolio {
    let mut book = EquityPortfolio::new();
    for j in 0..100 {
        let strike = 60.0 + 0.8 * j as f64;
        let option = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(strike)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(if j % 2 == 0 {
                PutOrCall::Call
            } else {
                PutOrCall::Put
            })
            .engine(Engine::BlackScholes)
            .build()
            .expect("bench option must build");
        book.add(option, if j % 3 == 0 { -50.0 } else { 100.0 });
    }
    book
}

/// Three scenarios of the shapes a desk actually runs: a broad crash, a
/// single-name vol move, and a parallel rate shift.
fn config() -> StressConfig {
    StressConfig {
        scenarios: vec![
            StressScenario {
                name: "equity_crash".into(),
                shocks: vec![
                    shock(RiskFactor::Spot, BumpMode::Relative, -0.20, None),
                    shock(RiskFactor::Vol, BumpMode::Absolute, 0.10, None),
                ],
            },
            StressScenario {
                name: "acme_vol_up".into(),
                shocks: vec![shock(
                    RiskFactor::Vol,
                    BumpMode::Absolute,
                    0.05,
                    Some("ACME"),
                )],
            },
            StressScenario {
                name: "rates_up_100bp".into(),
                shocks: vec![shock(RiskFactor::Rate, BumpMode::Absolute, 0.01, None)],
            },
        ],
        arbitrage: ArbitrageCheck::default(),
    }
}

fn stress_run(c: &mut Criterion) {
    let book = book();
    let cfg = config();
    // the full runner: snapshot, bump per scenario, revalue every position
    c.bench_function("stress_mtm_100pos_3scenarios", |b| {
        b.iter(|| stress_mtm(black_box(&book), black_box(&cfg)).unwrap())
    });
}

fn market_context(c: &mut Criterion) {
    let book = book();
    let first = &book.positions[0].option;
    let market = book.snapshot_market();

    // rebind + reprice one position vs its direct npv(): the difference is
    // pure context overhead (key lookups, clone of the bound market)
    c.bench_function("npv_in_single_rebind", |b| {
        b.iter(|| black_box(first).npv_in(black_box(&market)).unwrap())
    });
    c.bench_function("npv_direct_reference", |b| {
        b.iter(|| black_box(first).npv())
    });

    // snapshotting the whole book and bumping the whole market: the
    // per-scenario fixed costs of the stress runner
    c.bench_function("snapshot_market_100pos", |b| {
        b.iter(|| black_box(&book).snapshot_market())
    });
    let crash = vec![
        shock(RiskFactor::Spot, BumpMode::Relative, -0.20, None),
        shock(RiskFactor::Vol, BumpMode::Absolute, 0.10, None),
    ];
    c.bench_function("market_bumped_crash", |b| {
        b.iter(|| black_box(&market).bumped(black_box(&crash)).unwrap())
    });
}

criterion_group!(benches, stress_run, market_context);
criterion_main!(benches);
