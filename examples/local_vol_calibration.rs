//! The local volatility workflow end to end: quoted option prices ->
//! implied vols -> implied surface -> Dupire local vol -> reprice.
//!
//! Run with:  cargo run --release --example local_vol_calibration

mod common;

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, Tenor, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::quotes::Quote;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::core::vols::VolSurface;
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::local_vol::LocalVol;
use rustyqlib::equity::montecarlo::McModel;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vol_surface::build_implied_vol_surface;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

/// The "true" market smile we will generate quotes from and recover.
fn true_vol(strike: f64, base: f64) -> f64 {
    base - 0.001 * (strike - 100.0)
}

fn main() {
    common::title("LOCAL VOLATILITY — quotes -> implied surface -> Dupire -> reprice");

    let maturities = [
        (NaiveDate::from_ymd_opt(2026, 7, 2).unwrap(), 0.23),
        (NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(), 0.25),
    ];

    common::section("Step 1: generate market quotes from a known skew");
    println!("  sigma(K, T) = base(T) - 0.001 * (K - 100)");
    let mut quotes = Vec::new();
    for (maturity, base_vol) in maturities {
        let t = (maturity - asof()).num_days() as f64 / 365.0;
        for i in 0..13 {
            let strike = 70.0 + 5.0 * i as f64;
            let vol = true_vol(strike, base_vol);
            let price = bs_price(SPOT, strike, RATE, 0.0, vol, t, PutOrCall::Call);
            let mut option = EquityOptionBuilder::new()
                .spot(SPOT)
                .strike(strike)
                .flat_vol(0.2) // placeholder: the solve does not use it
                .flat_rate(RATE)
                .valuation_date(asof())
                .maturity_date(maturity)
                .vanilla(PutOrCall::Call)
                .build();
            option.base.current_price = Quote::new(price);
            quotes.push(Box::new(option));
        }
    }
    println!("  {} quotes across {} expiries", quotes.len(), maturities.len());

    common::section("Step 2: back out implied vols and build the surface");
    let surface = build_implied_vol_surface(&quotes).expect("calibration failed");
    println!("{surface}");

    common::section("Step 3: check the surface recovers the input smile");
    for (t, base_vol) in [(182.0 / 365.0, 0.23), (1.0, 0.25)] {
        for strike in [70.0, 85.0, 100.0, 115.0, 130.0] {
            let recovered = surface.vol(strike, SPOT, t);
            common::check(
                &format!("T={t:.3} K={strike}"),
                recovered,
                true_vol(strike, base_vol),
                1e-6,
            );
        }
    }

    common::section("Step 4: Dupire local volatility from that surface");
    let curve =
        YieldCurve::flat(RATE, asof(), DayCountConvention::Act365, Compounding::Continuous).unwrap();
    let lv = LocalVol::new(&surface, &curve, SPOT, 0.0, 0.0);
    println!("  {:>8} {:>12} {:>12} {:>12}", "level", "t=0.25", "t=0.50", "t=1.00");
    for level in [70.0, 85.0, 100.0, 115.0, 130.0] {
        println!(
            "  {level:>8.1} {:>12.4} {:>12.4} {:>12.4}",
            lv.vol(level, 0.25),
            lv.vol(level, 0.50),
            lv.vol(level, 1.00)
        );
    }
    common::note("local vol is steeper in strike than implied vol (the 'twice the slope' rule)");
    common::note("the far wings are noisy: Dupire takes numerical derivatives of a");
    common::note("piecewise-linear surface with flat extrapolation — trust the interior.");

    common::section("Step 5: reprice the calibrating vanillas through local vol MC");
    common::table_header();
    for strike in [90.0, 100.0, 110.0] {
        let expected = bs_price(SPOT, strike, RATE, 0.0, true_vol(strike, 0.25), 1.0, PutOrCall::Call);
        common::row(
            &format!("local vol MC, K={strike}"),
            &EquityOptionBuilder::new()
                .spot(SPOT)
                .strike(strike)
                .vol_surface(surface.clone())
                .flat_rate(RATE)
                .valuation_date(asof())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
                .vanilla(PutOrCall::Call)
                .engine(Engine::MonteCarlo)
                .model(McModel::LocalVol)
                .paths(50_000)
                .build(),
        );
        println!("{:<34} {expected:>12.6}  <- Black-Scholes target at the quoted smile vol", "");
    }

    common::section("Local vol on the finite difference engine (no sampling noise)");
    common::table_header();
    for strike in [90.0, 100.0, 110.0] {
        common::row(
            &format!("local vol FD, K={strike}"),
            &EquityOptionBuilder::new()
                .spot(SPOT)
                .strike(strike)
                .vol_surface(surface.clone())
                .flat_rate(RATE)
                .valuation_date(asof())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
                .vanilla(PutOrCall::Call)
                .engine(Engine::FiniteDifference)
                .model(McModel::LocalVol)
                .build(),
        );
    }

    common::section("Sanity: a flat surface must give flat local vol");
    let flat = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
    let flat_lv = LocalVol::new(&flat, &curve, SPOT, 0.0, 0.0);
    for (level, t) in [(70.0, 0.25), (100.0, 1.0), (130.0, 2.0)] {
        common::check(&format!("sigma_loc({level}, {t})"), flat_lv.vol(level, t), 0.25, 1e-6);
    }

    common::section("Term structure: local vol is the forward variance");
    let term = VolSurface::from_strike_smiles(
        &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
        &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
        asof(),
        DayCountConvention::Act365,
    )
    .unwrap();
    let term_lv = LocalVol::new(&term, &curve, SPOT, 0.0, 0.0);
    // (0.25^2 * 1 - 0.20^2 * 0.5) / 0.5 = 0.085
    common::check(
        "sigma_loc between pillars = sqrt(fwd variance)",
        term_lv.vol(100.0, 0.75),
        0.085_f64.sqrt(),
        1e-3,
    );
    println!();
}
