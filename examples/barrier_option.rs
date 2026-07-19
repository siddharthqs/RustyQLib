//! Barrier options: all eight knock-in / knock-out types.
//!
//! Run with:  cargo run --release --example barrier_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::barrier::{barrier_price, BarrierDirection, KnockType};
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::montecarlo::McModel;
use rustyqlib::equity::utils::Engine;
use rustyqlib::core::vols::VolSurface;
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::curves::Tenor;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("BARRIER")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(asof())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
}

fn main() {
    common::title("BARRIER OPTIONS — S=100 K=100 sigma=30% r=5% q=2% T=1y");

    common::section("All eight types, analytic (Reiner-Rubinstein)");
    common::table_header();
    for (dir, knock, pc, level) in [
        (BarrierDirection::Down, KnockType::In, PutOrCall::Call, 90.0),
        (BarrierDirection::Down, KnockType::Out, PutOrCall::Call, 90.0),
        (BarrierDirection::Down, KnockType::In, PutOrCall::Put, 90.0),
        (BarrierDirection::Down, KnockType::Out, PutOrCall::Put, 90.0),
        (BarrierDirection::Up, KnockType::In, PutOrCall::Call, 120.0),
        (BarrierDirection::Up, KnockType::Out, PutOrCall::Call, 120.0),
        (BarrierDirection::Up, KnockType::In, PutOrCall::Put, 120.0),
        (BarrierDirection::Up, KnockType::Out, PutOrCall::Put, 120.0),
    ] {
        common::row(
            &format!("{dir:?}-and-{knock:?} {pc:?} H={level}"),
            &base().barrier(pc, dir, knock, level).engine(Engine::BlackScholes).build(),
        );
    }

    common::section("Engine comparison: down-and-out call, H=90");
    common::table_header();
    for (label, engine) in [
        ("Analytical (Reiner-Rubinstein)", Engine::BlackScholes),
        ("Finite difference (absorbing)", Engine::FiniteDifference),
        ("Monte Carlo (Brownian bridge)", Engine::MonteCarlo),
        ("Binomial (unsupported)", Engine::Binomial),
    ] {
        common::row(
            label,
            &base()
                .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 90.0)
                .engine(engine)
                .build(),
        );
    }
    common::note("MC applies a bridge crossing correction, so monitoring is effectively continuous");

    common::section("In-out parity: KI + KO = vanilla");
    let vanilla = base().vanilla(PutOrCall::Call).engine(Engine::BlackScholes).build();
    for level in [80.0, 90.0, 99.0] {
        let ki = base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::In, level)
            .engine(Engine::BlackScholes)
            .build();
        let ko = base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, level)
            .engine(Engine::BlackScholes)
            .build();
        common::check(
            &format!("H={level}: KI + KO"),
            ki.npv() + ko.npv(),
            vanilla.npv(),
            1e-10,
        );
    }

    common::section("Limits");
    common::check(
        "far barrier: KO call -> vanilla",
        barrier_price(SPOT, STRIKE, 1e-4, RATE, DIV, VOL, 1.0, BarrierDirection::Down, KnockType::Out, PutOrCall::Call),
        vanilla.npv(),
        1e-9,
    );
    common::check(
        "up-and-out call with K >= H is worthless",
        barrier_price(SPOT, 110.0, 105.0, RATE, DIV, VOL, 1.0, BarrierDirection::Up, KnockType::Out, PutOrCall::Call),
        0.0,
        1e-12,
    );
    common::check(
        "spot at barrier: KO = 0",
        base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, SPOT)
            .engine(Engine::BlackScholes)
            .build()
            .npv(),
        0.0,
        1e-12,
    );

    common::section("Barrier level sweep: down-and-out call");
    common::table_header();
    for level in [50.0, 70.0, 85.0, 95.0, 99.0] {
        common::row(
            &format!("H={level}"),
            &base()
                .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, level)
                .engine(Engine::BlackScholes)
                .build(),
        );
    }
    common::note("value decreases as the barrier approaches spot; delta can exceed 1 near it");

    common::section("Smile matters: down-and-out call under local vol");
    let skewed = VolSurface::from_strike_grid(
        &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
        &[70.0, 85.0, 100.0, 115.0, 130.0],
        &[
            vec![0.38, 0.34, 0.30, 0.28, 0.27],
            vec![0.37, 0.34, 0.30, 0.29, 0.28],
            vec![0.36, 0.33, 0.30, 0.29, 0.28],
        ],
        asof(),
        DayCountConvention::Act365,
    )
    .unwrap();
    common::table_header();
    common::row(
        "GBM (flat 30%)",
        &base()
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 90.0)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build(),
    );
    common::row(
        "Local vol (skewed surface)",
        &base()
            .vol_surface(skewed)
            .barrier(PutOrCall::Call, BarrierDirection::Down, KnockType::Out, 90.0)
            .engine(Engine::MonteCarlo)
            .model(McModel::LocalVol)
            .paths(50_000)
            .build(),
    );
    common::note("downside skew raises the knock-out probability, lowering the price");
    println!();
}
