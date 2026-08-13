//! Stochastic short-rate models: fit Hull-White to a market curve,
//! price zero-bond options, European swaptions and caps analytically,
//! and demonstrate the simulation contract a cross-asset hybrid uses.
//!
//! Run with:  cargo run --release --example short_rate_models

use chrono::NaiveDate;
use rand::{Rng, SeedableRng};
use rustyqlib::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::models::pricers::{caplet, european_swaption, floorlet};
use rustyqlib::{HullWhite, PayerReceiver, ShortRateModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let asof = NaiveDate::from_ymd_opt(2026, 8, 13).unwrap();
    let curve = YieldCurve::from_zero_rates(
        &[
            Tenor::YearFraction(0.5),
            Tenor::YearFraction(1.0),
            Tenor::YearFraction(2.0),
            Tenor::YearFraction(5.0),
            Tenor::YearFraction(10.0),
            Tenor::YearFraction(30.0),
        ],
        &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
        asof,
        DayCountConvention::Act365,
        Compounding::Continuous,
        InterpolationMethod::LogLinearDf,
    )?;

    // a = mean reversion, sigma = short-rate vol (normally calibrated
    // to swaptions; here representative values)
    let model = HullWhite::new(0.05, 0.011, curve.clone())?;
    println!("Hull-White fitted to the market curve (a=0.05, sigma=110bp):");
    println!(
        "  initial short rate  {:.4}%",
        model.initial_short_rate() * 100.0
    );
    for t in [1.0, 5.0, 10.0, 30.0] {
        println!(
            "  P(0,{t:>4}): model {:.8}  market {:.8}",
            model.zero_bond(0.0, t, model.initial_short_rate())?,
            curve.df(t)
        );
    }

    // European swaption: 1y into 5y annual fixed leg
    let fixed_leg: Vec<(f64, f64)> = (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect();
    let annuity: f64 = fixed_leg.iter().map(|&(t, tau)| tau * curve.df(t)).sum();
    let atm = (curve.df(1.0) - curve.df(6.0)) / annuity;
    println!(
        "\n1y5y European swaptions on 10mm (ATM forward {:.4}%):",
        atm * 100.0
    );
    for strike in [atm - 0.005, atm, atm + 0.005] {
        let payer = european_swaption(
            &model,
            1.0,
            &fixed_leg,
            strike,
            10_000_000.0,
            PayerReceiver::Payer,
        )?;
        let receiver = european_swaption(
            &model,
            1.0,
            &fixed_leg,
            strike,
            10_000_000.0,
            PayerReceiver::Receiver,
        )?;
        println!(
            "  K = {:.4}%  payer {:>12.2}  receiver {:>12.2}",
            strike * 100.0,
            payer,
            receiver
        );
    }

    // Cap/floor on a quarterly period 2y out
    let cap = caplet(&model, 2.0, 2.25, 0.25, 0.045, 10_000_000.0)?;
    let floor = floorlet(&model, 2.0, 2.25, 0.25, 0.045, 10_000_000.0)?;
    println!("\nquarterly caplet/floorlet 2y out, K = 4.5%, 10mm:");
    println!("  caplet {cap:.2}   floorlet {floor:.2}");

    // The cross-asset simulation contract: evolve the short rate with
    // the exact transition and discount along paths — a hybrid equity
    // model does exactly this for its stochastic funding leg
    let (horizon, steps, paths) = (10.0_f64, 200usize, 50_000usize);
    let dt = horizon / steps as f64;
    let mut rng = rand_pcg::Pcg64::seed_from_u64(2026);
    let mut sum = 0.0;
    for _ in 0..paths {
        let mut r = model.initial_short_rate();
        let mut integral = 0.0;
        for step in 0..steps {
            let z: f64 = rng.sample(rand_distr::StandardNormal);
            let next = model.evolve(step as f64 * dt, r, dt, z)?;
            integral += 0.5 * (r + next) * dt;
            r = next;
        }
        sum += (-integral).exp();
    }
    let mc_df = sum / paths as f64;
    println!("\nsimulated 10y discounting ({paths} paths, exact transitions):");
    println!(
        "  Monte Carlo df {mc_df:.6}  market df {:.6}",
        curve.df(10.0)
    );
    Ok(())
}
