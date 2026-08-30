//! SABR for equity derivatives: the Hagan smile, pluggable pricing on
//! the Analytical and Monte Carlo engines, calibration to a quoted
//! smile, and smoothing a noisy implied-vol surface through the
//! per-expiry SABR fit.
//!
//! Run with:  cargo run --release --example sabr_smile

mod common;

use chrono::NaiveDate;
use rustyqlib::core::curves::Tenor;
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::core::vols::VolSurface;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::sabr::{sabr_price, SabrParams, SabrSurfaceFit};
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;
const RATE: f64 = 0.03;
const DIV: f64 = 0.01;

fn params() -> SabrParams {
    SabrParams {
        alpha: 0.22, // ATM vol level (beta = 1: alpha IS the ATM lognormal vol)
        beta: 1.0,   // lognormal backbone, the equity convention
        rho: -0.55,  // equity skew
        nu: 0.7,     // vol-of-vol: smile curvature
    }
}

fn sabr_option(
    strike: f64,
    pc: PutOrCall,
    engine: Engine,
) -> rustyqlib::equity::vanilla_option::EquityOption {
    EquityOptionBuilder::new()
        .symbol("SABR")
        .spot(SPOT)
        .strike(strike)
        .flat_vol(params().alpha) // surface anchor for vega bumps
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .years_to_maturity(1.0)
        .vanilla(pc)
        .engine(engine)
        .sabr(params())
        .build()
        .expect("option must build")
}

fn main() {
    let sp = params();
    common::title(&format!(
        "SABR — alpha={} beta={} rho={} nu={}",
        sp.alpha, sp.beta, sp.rho, sp.nu
    ));
    common::note("pluggable dynamics: mc_model = \"sabr\", swaps in for Heston/rBergomi");

    // ── the smile the parameters generate ───────────────────────────────
    common::section("The one-year Hagan smile");
    let forward = SPOT * ((RATE - DIV) * 1.0f64).exp();
    println!("  {:>8} {:>14}", "strike", "implied vol");
    for k in [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0] {
        println!("  {k:>8.1} {:>13.2}%", sp.vol(forward, k, 1.0) * 100.0);
    }
    common::note("rho < 0 tilts the put wing up; nu bends both wings");

    // ── pricing: analytic (Hagan -> Black-Scholes) vs Monte Carlo ───────
    common::section("Analytic vs Monte Carlo (true two-factor dynamics, 100k paths)");
    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "strike", "analytic", "monte carlo", "diff"
    );
    for k in [85.0, 100.0, 115.0] {
        let analytic = sabr_option(k, PutOrCall::Call, Engine::BlackScholes).npv();
        let mut mc = sabr_option(k, PutOrCall::Call, Engine::MonteCarlo);
        mc.engine = rustyqlib::equity::utils::PricingEngine::MonteCarlo(
            rustyqlib::equity::montecarlo::MonteCarloConfig {
                paths: 100_000,
                ..Default::default()
            },
        );
        let simulated = mc.npv();
        println!(
            "  {k:>8.1} {analytic:>12.4} {simulated:>12.4} {:>12.4}",
            simulated - analytic
        );
    }
    common::note("MC simulates dF = alpha F^beta dW, dalpha = nu alpha dW_a on the forward");

    // greeks come off the same bump machinery as every other model
    let atm = sabr_option(100.0, PutOrCall::Call, Engine::BlackScholes);
    common::section("ATM call greeks (bump-and-reprice through the Hagan smile)");
    println!(
        "  delta {:>8.4}   gamma {:>8.5}   vega {:>8.4}   theta {:>8.4}",
        atm.delta(),
        atm.gamma(),
        atm.vega(),
        atm.theta()
    );

    // ── calibration: recover the smile from noisy quotes ────────────────
    common::section("Calibration to a noisy quoted smile (beta fixed at 1)");
    let t = 1.0;
    let quotes: Vec<(f64, f64)> = (0..13)
        .map(|i| {
            let k = forward * (-0.3 + i as f64 * 0.05f64).exp();
            let noise = if i % 2 == 0 { 0.002 } else { -0.002 }; // +-20bp
            (k, sp.vol(forward, k, t) + noise)
        })
        .collect();
    let fit = SabrParams::calibrate(&quotes, forward, t, 1.0).expect("SABR calibration failed");
    println!(
        "  truth:  alpha={:.4} rho={:.3} nu={:.3}",
        sp.alpha, sp.rho, sp.nu
    );
    println!(
        "  fitted: alpha={:.4} rho={:.3} nu={:.3}   (rmse {:.1} bps, {} iterations)",
        fit.params.alpha,
        fit.params.rho,
        fit.params.nu,
        fit.rmse * 1e4,
        fit.iterations
    );
    common::note("the fit lands between the +-20bp noise, not on it — that is the smoothing");

    // ── surface smoothing: per-expiry SABR fit of a noisy surface ───────
    common::section("Smoothing a noisy two-expiry surface (SabrSurfaceFit)");
    let reference = NaiveDate::from_ymd_opt(2026, 8, 13).unwrap();
    let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
    let smiles: Vec<Vec<(f64, f64)>> = [0.5, 1.0]
        .iter()
        .map(|&te| {
            (0..11)
                .map(|i| {
                    let k = forward * (-0.25 + i as f64 * 0.05f64).exp();
                    let noise = if i % 2 == 0 { 0.0025 } else { -0.0025 };
                    (k, sp.vol(forward, k, te) + noise)
                })
                .collect()
        })
        .collect();
    let noisy =
        VolSurface::from_strike_smiles(&expiries, &smiles, reference, DayCountConvention::Act365)
            .expect("surface must build");
    let surface_fit = SabrSurfaceFit::fit(&noisy, |_| forward, 1.0).expect("fit must succeed");
    println!(
        "  {:>6} {:>10} {:>8} {:>8} {:>12} {:>10}",
        "T", "alpha", "rho", "nu", "rmse (bps)", "min g"
    );
    for slice in &surface_fit.slices {
        println!(
            "  {:>6.2} {:>10.4} {:>8.3} {:>8.3} {:>12.1} {:>10.4}",
            slice.t,
            slice.params.alpha,
            slice.params.rho,
            slice.params.nu,
            slice.rmse * 1e4,
            slice.min_g
        );
    }
    println!(
        "  smoothed ATM vols: 6m {:.2}%  12m {:.2}%   (noise-free truth {:.2}% / {:.2}%)",
        surface_fit.vol(forward, 0.5) * 100.0,
        surface_fit.vol(forward, 1.0) * 100.0,
        sp.vol(forward, forward, 0.5) * 100.0,
        sp.vol(forward, forward, 1.0) * 100.0
    );
    let sampled = surface_fit
        .to_vol_surface(41)
        .expect("sampling must succeed");
    common::note(&format!(
        "sampled back into a pricing VolSurface with {} expiries (min g > 0: no butterfly arbitrage in the quoted range)",
        sampled.expiry_times().len()
    ));

    // smile-consistent pricing straight off the calibrated params
    common::section("Reprice a 90-strike put off the calibrated smile");
    let put = sabr_price(SPOT, 90.0, RATE, DIV, 1.0, &fit.params, PutOrCall::Put);
    println!(
        "  NPV {put:>10.4}   at Hagan vol {:.2}%",
        fit.params.vol(forward, 90.0, 1.0) * 100.0
    );
    println!();
}
