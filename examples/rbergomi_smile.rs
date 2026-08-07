//! Rough Bergomi: the smile and the skew term structure the model
//! generates, priced with the exact-scheme Monte Carlo engine and the
//! conditional (Romano-Touzi) estimator, then inverted to implied vol.
//!
//! The parameter set is the Bayer-Friz-Gatheral SPX-like fit; the
//! Heston column uses a comparable diffusive parameterization to show
//! the structural difference in the short-dated skew.
//!
//! Run with:  cargo run --release --example rbergomi_smile

mod common;

use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::blackscholes::implied_vol_from_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::heston::{heston_price, HestonParams};
use rustyqlib::equity::rbergomi::RBergomiParams;
use rustyqlib::equity::utils::Engine;

const SPOT: f64 = 100.0;

fn params() -> RBergomiParams {
    RBergomiParams {
        xi0: 0.235 * 0.235, // flat forward variance, sqrt = 23.5% vol
        eta: 1.9,
        hurst: 0.07,
        rho: -0.9,
    }
}

/// Comparable diffusive parameterization: same variance level, deeply
/// negative correlation, vol-of-vol chosen so the one-year smiles are
/// in the same ballpark.
fn heston_params() -> HestonParams {
    HestonParams {
        v0: 0.235 * 0.235,
        kappa: 2.0,
        theta: 0.235 * 0.235,
        vol_of_vol: 1.0,
        rho: -0.9,
    }
}

/// rBergomi implied vol at (strike, maturity) off the Monte Carlo
/// engine (mixed estimator for vanillas).
fn rbergomi_iv(k: f64, t: f64, paths: usize) -> f64 {
    let pc = if k < SPOT {
        PutOrCall::Put
    } else {
        PutOrCall::Call
    };
    let price = EquityOptionBuilder::new()
        .symbol("RBERGOMI")
        .spot(SPOT)
        .strike(k)
        .flat_vol(params().xi0.sqrt())
        .flat_rate(0.0)
        .years_to_maturity(t)
        .vanilla(pc)
        .engine(Engine::MonteCarlo)
        .rbergomi(params())
        .paths(paths)
        .build()
        .expect("option must build")
        .npv();
    implied_vol_from_price(SPOT, k, 0.0, 0.0, t, price, pc).unwrap_or(f64::NAN)
}

fn heston_iv(k: f64, t: f64) -> f64 {
    let pc = if k < SPOT {
        PutOrCall::Put
    } else {
        PutOrCall::Call
    };
    let price = heston_price(SPOT, k, 0.0, 0.0, t, &heston_params(), pc);
    implied_vol_from_price(SPOT, k, 0.0, 0.0, t, price, pc).unwrap_or(f64::NAN)
}

fn main() {
    let p = params();
    common::title(&format!(
        "ROUGH BERGOMI — xi0={:.6} (vol {:.1}%) eta={} H={} rho={}",
        p.xi0,
        p.xi0.sqrt() * 100.0,
        p.eta,
        p.hurst,
        p.rho
    ));
    common::note("Bayer-Friz-Gatheral SPX-like fit; r = q = 0 so F = S");

    common::section("The six-month smile (implied vol from MC prices, 100k paths)");
    println!("  {:>8} {:>14}", "strike", "implied vol");
    for k in [80.0, 90.0, 100.0, 110.0, 120.0] {
        let iv = rbergomi_iv(k, 0.5, 100_000);
        println!("  {k:>8.1} {:>13.2}%", iv * 100.0);
    }
    common::note("deep put skew from rho = -0.9; eta sets how deep");

    common::section("ATM skew term structure: rough vs diffusive");
    println!(
        "  {:>6} {:>16} {:>14} {:>16}",
        "T", "rBergomi skew", "Heston skew", "rB skew * T^0.43"
    );
    // skew measured as the implied-vol difference across +-2% log-strike
    let (k_lo, k_hi) = (SPOT * (-0.02f64).exp(), SPOT * (0.02f64).exp());
    for t in [1.0 / 52.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0] {
        let rb = (rbergomi_iv(k_hi, t, 100_000) - rbergomi_iv(k_lo, t, 100_000)) / 0.04;
        let he = (heston_iv(k_hi, t) - heston_iv(k_lo, t)) / 0.04;
        println!(
            "  {t:>6.3} {:>15.3} {:>14.3} {:>16.3}",
            rb,
            he,
            rb * t.powf(0.5 - p.hurst)
        );
    }
    common::note("the rescaled rBergomi column is ~flat: skew ~ T^(H - 1/2)");
    common::note("Heston's skew saturates as T -> 0; the market's does not");
    println!();
}
