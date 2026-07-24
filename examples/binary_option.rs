//! Binary (digital) options: cash-or-nothing and asset-or-nothing.
//!
//! Run with:  cargo run --release --example binary_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::BinaryType;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.30;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;
const CASH: f64 = 1.0;

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("BINARY")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
}

fn main() {
    common::title("BINARY OPTION — S=100 K=100 sigma=30% r=5% q=2% T=1y");

    for (name, binary_type, cash) in [
        ("cash-or-nothing (1 unit)", BinaryType::CashOrNothing, CASH),
        ("asset-or-nothing", BinaryType::AssetOrNothing, 0.0),
    ] {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            common::section(&format!("{name} {pc:?}"));
            common::table_header();
            for (label, engine) in [
                ("Analytical (closed form)", Engine::BlackScholes),
                ("Binomial (1000 steps)", Engine::Binomial),
                ("Finite difference", Engine::FiniteDifference),
                ("Monte Carlo (Sobol, 100k)", Engine::MonteCarlo),
            ] {
                common::row(label, &base().binary(pc, binary_type, cash).engine(engine).build());
            }
        }
    }
    common::note("the tree oscillates on digitals: the strike falls between terminal nodes");

    common::section("Cash amount scales linearly");
    common::table_header();
    for cash in [1.0, 100.0, 1000.0] {
        common::row(
            &format!("cash-or-nothing call, cash={cash}"),
            &base()
                .binary(PutOrCall::Call, BinaryType::CashOrNothing, cash)
                .engine(Engine::BlackScholes)
                .build(),
        );
    }

    common::section("Identities");
    let cash_call = base()
        .binary(PutOrCall::Call, BinaryType::CashOrNothing, CASH)
        .engine(Engine::BlackScholes)
        .build();
    let cash_put = base()
        .binary(PutOrCall::Put, BinaryType::CashOrNothing, CASH)
        .engine(Engine::BlackScholes)
        .build();
    common::check(
        "cash call + cash put = e^{-rT}",
        cash_call.npv() + cash_put.npv(),
        (-RATE * 1.0_f64).exp(),
        1e-12,
    );

    let asset_call = base()
        .binary(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0)
        .engine(Engine::BlackScholes)
        .build();
    let asset_put = base()
        .binary(PutOrCall::Put, BinaryType::AssetOrNothing, 0.0)
        .engine(Engine::BlackScholes)
        .build();
    common::check(
        "asset call + asset put = S e^{-qT}",
        asset_call.npv() + asset_put.npv(),
        SPOT * (-DIV * 1.0_f64).exp(),
        1e-10,
    );

    common::section("Replication: asset digital = vanilla call + K cash digitals");
    let vanilla = base().vanilla(PutOrCall::Call).engine(Engine::BlackScholes).build();
    let k_cash = base()
        .binary(PutOrCall::Call, BinaryType::CashOrNothing, STRIKE)
        .engine(Engine::BlackScholes)
        .build();
    common::check("npv", asset_call.npv(), vanilla.npv() + k_cash.npv(), 1e-10);
    common::check("delta", asset_call.delta(), vanilla.delta() + k_cash.delta(), 1e-10);
    common::check("gamma", asset_call.gamma(), vanilla.gamma() + k_cash.gamma(), 1e-10);
    common::check("vega", asset_call.vega(), vanilla.vega() + k_cash.vega(), 1e-10);
    common::check("theta", asset_call.theta(), vanilla.theta() + k_cash.theta(), 1e-10);
    common::check("rho", asset_call.rho(), vanilla.rho() + k_cash.rho(), 1e-10);
    common::note("both sides are implemented independently, so this is a real cross-check");

    common::section("Digital risk: delta and gamma explode near the strike at expiry");
    common::table_header();
    for years in [1.0, 0.25, 0.05, 0.01] {
        common::row(
            &format!("cash-or-nothing call, T={years}y"),
            &base()
                .years_to_maturity(years)
                .binary(PutOrCall::Call, BinaryType::CashOrNothing, CASH)
                .engine(Engine::BlackScholes)
                .build(),
        );
    }

    digital_greek_surfaces();
    println!();
}

/// The digital's Greeks are the standout case for visualizing (lack of)
/// smoothness: as maturity shrinks the delta spikes into a tall bump at the
/// strike and gamma flips sign right across it. Saved as self-contained
/// interactive HTML to `runs/binary_option/`.
fn digital_greek_surfaces() {
    use common::plot3d::{greek_surface, linspace, save_surface_html, Labels};
    use rustyqlib::equity::vanilla_option::EquityOption;

    common::section("Digital Greek surfaces over (moneyness, maturity) -> runs/binary_option/*.html");

    // tighter moneyness band and shorter maturities: that is where the
    // digital's delta/gamma structure lives
    let moneyness = linspace(0.8, 1.2, 80);
    let mats = linspace(0.02, 1.0, 60);

    let greek = |select: fn(&EquityOption) -> f64| {
        move |m: f64, years: f64| -> f64 {
            let option = base()
                .spot(m * STRIKE)
                .years_to_maturity(years)
                .binary(PutOrCall::Call, BinaryType::CashOrNothing, CASH)
                .engine(Engine::BlackScholes)
                .build();
            select(&option)
        }
    };

    for (name, file, select) in [
        ("Delta", "delta", (|o: &EquityOption| o.delta()) as fn(&EquityOption) -> f64),
        ("Gamma", "gamma", |o: &EquityOption| o.gamma()),
    ] {
        let surface = greek_surface(&moneyness, &mats, greek(select));
        save_surface_html(
            &surface,
            &format!("runs/binary_option/{file}_surface.html"),
            &Labels {
                title: &format!("Cash digital call {name} (K=100) — note the near-expiry spike"),
                x: "moneyness (S/K)",
                y: "maturity (y)",
                z: name,
            },
        );
    }
    common::note("contrast with the vanilla surfaces: the digital is far from smooth near the strike");
}
