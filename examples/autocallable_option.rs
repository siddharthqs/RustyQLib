//! Autocallable note with coupon (rebate) and knock-in capital protection.
//! Priced under GBM, Dupire local volatility and Heston.
//!
//! Run with:  cargo run --release --example autocallable_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::curves::Tenor;
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::traits::Instrument;
use rustyqlib::core::vols::VolSurface;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::heston::HestonParams;
use rustyqlib::equity::montecarlo::McModel;
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;
const NOTIONAL: f64 = 100.0;
const AUTOCALL: f64 = 100.0;
const PROTECTION: f64 = 70.0;
const COUPON: f64 = 6.0;
const OBSERVATIONS: usize = 4;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("ATHENA")
        .spot(SPOT)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(asof())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
        .engine(Engine::MonteCarlo)
        .paths(50_000)
}

fn note(autocall: f64, protection: f64, coupon: f64) -> EquityOptionBuilder {
    base().autocallable(autocall, protection, coupon, OBSERVATIONS, NOTIONAL)
}

/// Downward-skewed surface: the shape that actually drives these notes.
fn skewed_surface() -> VolSurface {
    VolSurface::from_strike_grid(
        &[Tenor::YearFraction(0.25), Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
        &[60.0, 70.0, 85.0, 100.0, 115.0, 130.0],
        &[
            vec![0.42, 0.38, 0.33, 0.29, 0.27, 0.26],
            vec![0.41, 0.37, 0.33, 0.30, 0.28, 0.27],
            vec![0.40, 0.37, 0.33, 0.30, 0.29, 0.28],
        ],
        asof(),
        DayCountConvention::Act365,
    )
    .unwrap()
}

fn main() {
    common::title(&format!(
        "AUTOCALLABLE NOTE — N={NOTIONAL} autocall={AUTOCALL} protection={PROTECTION} coupon={COUPON}/period, {OBSERVATIONS} observations, T=1y"
    ));
    common::note("pays N + m*coupon if S >= autocall barrier at observation m;");
    common::note("otherwise N at maturity, or N*S_T/S_0 if the protection barrier was breached.");

    common::section("Model comparison");
    common::table_header();
    common::row("GBM (flat 30%)", &note(AUTOCALL, PROTECTION, COUPON).build());
    common::row(
        "Local vol (skewed surface)",
        &note(AUTOCALL, PROTECTION, COUPON)
            .vol_surface(skewed_surface())
            .model(McModel::LocalVol)
            .build(),
    );
    common::row(
        "Heston (vol-of-vol=0.4, rho=-0.7)",
        &note(AUTOCALL, PROTECTION, COUPON)
            .heston(HestonParams {
                v0: VOL * VOL,
                kappa: 2.0,
                theta: VOL * VOL,
                vol_of_vol: 0.4,
                rho: -0.7,
            })
            .build(),
    );
    common::row(
        "Analytical (unsupported)",
        &note(AUTOCALL, PROTECTION, COUPON).engine(Engine::BlackScholes).build(),
    );
    common::note("skew/stoch-vol raise the knock-in probability, lowering the note value");

    common::section("Structure sensitivity (GBM)");
    common::table_header();
    for coupon in [0.0, 3.0, 6.0, 9.0] {
        common::row(&format!("coupon = {coupon}/period"), &note(AUTOCALL, PROTECTION, coupon).build());
    }
    for protection in [50.0, 60.0, 70.0, 80.0] {
        common::row(
            &format!("protection barrier = {protection}"),
            &note(AUTOCALL, protection, COUPON).build(),
        );
    }
    for autocall in [95.0, 100.0, 105.0, 110.0] {
        common::row(
            &format!("autocall barrier = {autocall}"),
            &note(autocall, PROTECTION, COUPON).build(),
        );
    }

    common::section("Observation frequency (GBM)");
    common::table_header();
    for obs in [1usize, 2, 4, 12] {
        common::row(
            &format!("{obs} observations"),
            &base().autocallable(AUTOCALL, PROTECTION, COUPON, obs, NOTIONAL).build(),
        );
    }

    common::section("Degenerate cases (exact identities)");
    let always_calls = base()
        .autocallable(1e-9, 50.0, COUPON, OBSERVATIONS, NOTIONAL)
        .build()
        .npv();
    common::check(
        "barrier at 0 -> called at t1 with 1 coupon",
        always_calls,
        (NOTIONAL + COUPON) * (-RATE * 0.25_f64).exp(),
        1e-8,
    );
    let never_calls = base()
        .autocallable(1e12, 1e-9, COUPON, OBSERVATIONS, NOTIONAL)
        .build()
        .npv();
    common::check(
        "unreachable barriers -> zero-coupon bond",
        never_calls,
        NOTIONAL * (-RATE * 1.0_f64).exp(),
        1e-8,
    );
    let full_downside = base()
        .autocallable(1e12, 1e12, 0.0, OBSERVATIONS, NOTIONAL)
        .dividend_yield(0.0)
        .build()
        .npv();
    common::check(
        "always knocked in, no coupon -> discounted forward",
        full_downside,
        NOTIONAL,
        0.3,
    );
    println!();
}
