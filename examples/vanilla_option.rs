//! Vanilla European and American options across every pricing engine.
//!
//! Run with:  cargo run --release --example vanilla_option

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::montecarlo::{McModel, Sampler};
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanila_option::EquityOption;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const VOL: f64 = 0.80;
const RATE: f64 = 0.06;
const DIV: f64 = 0.00;

fn asof() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
}

fn base(put_or_call: PutOrCall) -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .symbol("VANILLA")
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(asof())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
        .vanilla(put_or_call)
}

fn priced(builder: EquityOptionBuilder, engine: Engine) -> EquityOption {
    builder.engine(engine).build()
}

fn main() {
    common::title("VANILLA OPTION — S=100 K=100 sigma=30% r=5% q=2% T=1y");

    for pc in [PutOrCall::Call, PutOrCall::Put] {
        common::section(&format!("European {pc:?}"));
        common::table_header();
        common::row("Analytical (Black-Scholes)", &priced(base(pc), Engine::BlackScholes));
        // common::row("Binomial (1000 steps)", &priced(base(pc), Engine::Binomial));
        // common::row("Finite difference (400x400)", &priced(base(pc), Engine::FiniteDifference));
        // common::row("Monte Carlo (Sobol, 100k)", &priced(base(pc), Engine::MonteCarlo));
        // common::row(
        //     "Monte Carlo (pseudo, 100k)",
        //     &base(pc)
        //         .engine(Engine::MonteCarlo)
        //         .mc_config({
        //             let mut c = rustyqlib::equity::montecarlo::MonteCarloConfig::default();
        //             c.sampler = Sampler::PseudoRandom;
        //             c
        //         })
        //         .build(),
        // );
    }

    // common::section("American put (early exercise premium)");
    // common::table_header();
    // let european_put = priced(base(PutOrCall::Put), Engine::BlackScholes).npv();
    // common::row(
    //     "Analytical (rejects American)",
    //     &base(PutOrCall::Put).american().vanilla(PutOrCall::Put).engine(Engine::BlackScholes).build(),
    // );
    // common::row(
    //     "Binomial",
    //     &base(PutOrCall::Put).american().vanilla(PutOrCall::Put).engine(Engine::Binomial).build(),
    // );
    // common::row(
    //     "Finite difference (Brennan-Schwartz)",
    //     &base(PutOrCall::Put)
    //         .american()
    //         .vanilla(PutOrCall::Put)
    //         .engine(Engine::FiniteDifference)
    //         .build(),
    // );
    // common::row(
    //     "Monte Carlo (Longstaff-Schwartz)",
    //     &base(PutOrCall::Put)
    //         .american()
    //         .vanilla(PutOrCall::Put)
    //         .engine(Engine::MonteCarlo)
    //         .paths(50_000)
    //         .build(),
    // );
    // common::note(&format!("European put for reference: {european_put:.6}"));
    //common::note("FD and MC report true American Greeks (grid / LSMC repricing);");
    //common::note("the tree falls back to analytic European Greeks — note the delta gap.");

    //common::section("Model comparison (same flat 30% vol)");
    //common::table_header();
    //common::row("GBM", &priced(base(PutOrCall::Call), Engine::MonteCarlo));
    //common::row(
    //    "Local vol (flat surface)",
    //    &base(PutOrCall::Call)
    //        .engine(Engine::MonteCarlo)
    //        .model(McModel::LocalVol)
    //        .paths(50_000)
    //        .build(),
    //);
    // common::row(
    //     "Heston (vol-of-vol -> 0)",
    //     &base(PutOrCall::Call)
    //         .engine(Engine::MonteCarlo)
    //         .heston(rustyqlib::equity::heston::HestonParams {
    //             v0: VOL * VOL,
    //             kappa: 1.0,
    //             theta: VOL * VOL,
    //             vol_of_vol: 1e-3,
    //             rho: 0.0,
    //         })
    //         .paths(50_000)
    //         .build(),
    // );
    //common::note("all three must agree: flat surface and zero vol-of-vol are Black-Scholes");

    // common::section("Identities");
    // let call = priced(base(PutOrCall::Call), Engine::BlackScholes);
    // let put = priced(base(PutOrCall::Put), Engine::BlackScholes);
    // let parity = SPOT * (-DIV * 1.0_f64).exp() - STRIKE * (-RATE * 1.0_f64).exp();
    // common::check("put-call parity: C - P", call.npv() - put.npv(), parity, 1e-10);
    // common::check(
    //     "closed form vs bs_price()",
    //     call.npv(),
    //     bs_price(SPOT, STRIKE, RATE, DIV, VOL, 1.0, PutOrCall::Call),
    //     1e-12,
    // );
    // common::check(
    //     "delta_call - delta_put = e^{-qT}",
    //     call.delta() - put.delta(),
    //     (-DIV * 1.0_f64).exp(),
    //     1e-10,
    // );

    // common::section("Implied volatility round trip");
    // let mut iv_option = priced(base(PutOrCall::Call), Engine::BlackScholes);
    // let market_price = iv_option.npv();
    // let recovered = iv_option.imp_vol(market_price);
    // common::check("implied vol recovers input", recovered, VOL, 1e-10);
    //
    // common::section("Greeks vs bump-and-reprice (finite difference of the closed form)");
    // let h = 0.01;
    // let up = base(PutOrCall::Call).spot(SPOT + h).engine(Engine::BlackScholes).build();
    // let dn = base(PutOrCall::Call).spot(SPOT - h).engine(Engine::BlackScholes).build();
    // common::check("delta", call.delta(), (up.npv() - dn.npv()) / (2.0 * h), 1e-6);
    // common::check(
    //     "gamma",
    //     call.gamma(),
    //     (up.npv() - 2.0 * call.npv() + dn.npv()) / (h * h),
    //     1e-4,
    // );

    greek_surfaces();
    println!();
}

/// Save interactive 3D surfaces of the Greeks over (moneyness, maturity) so
/// their shape and smoothness can be inspected. Written as self-contained
/// HTML to `runs/vanilla_option/`.
fn greek_surfaces() {
    use common::plot3d::{greek_surface, linspace, save_surface_html, Labels};

    common::section("Greek surfaces over (moneyness, maturity) -> runs/vanilla_option/*.html");

    // x = moneyness S/K (0.4 .. 1.6, i.e. spot 40..160 for K=100);
    // y = maturity 0.05..2.0y (short-dated ATM is where the structure lives)
    let moneyness = linspace(0.5, 1.5, 72);
    let mats = linspace(0.01, 1.0, 56);

    // a call priced analytically at (moneyness, maturity); spot = m * K
    let greek = |select: fn(&EquityOption) -> f64| {
        move |m: f64, years: f64| -> f64 {
            let option = EquityOptionBuilder::new()
                .spot(m * STRIKE)
                .strike(STRIKE)
                .flat_vol(VOL)
                .flat_rate(RATE)
                .dividend_yield(DIV)
                .valuation_date(asof())
                .years_to_maturity(years)
                .vanilla(PutOrCall::Call)
                .engine(Engine::BlackScholes)
                .build();
            select(&option)
        }
    };

    for (name, file, select) in [
        ("Delta", "delta", (|o: &EquityOption| o.delta()) as fn(&EquityOption) -> f64),
        ("Gamma", "gamma", |o: &EquityOption| o.gamma()),
        ("Vega", "vega", |o: &EquityOption| o.vega()),
        ("Theta", "theta", |o: &EquityOption| o.theta()),
        ("Vanna", "vanna", |o: &EquityOption| o.vanna()),
        ("Charm", "charm", |o: &EquityOption| o.charm()),
        ("Gamma P", "gamma_p", |o: &EquityOption| o.gamma_p()),
        ("Zomma", "zomma", |o: &EquityOption| o.zomma()),
    ] {
        let surface = greek_surface(&moneyness, &mats, greek(select));
        save_surface_html(
            &surface,
            &format!("runs/vanilla_option/{file}_surface.html"),
            &Labels {
                title: &format!("Vanilla call {name} (K=100, sigma=30%, r=5%, q=2%)"),
                x: "moneyness (S/K)",
                y: "maturity (y)",
                z: name,
            },
        );
    }
    common::note("open the HTML in a browser to rotate, zoom and hover the surfaces");
}
