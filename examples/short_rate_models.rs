//! Stochastic short-rate models: fit Hull-White to a market curve,
//! price zero-bond options, European swaptions and caps analytically,
//! and demonstrate the simulation contract a cross-asset hybrid uses.
//!
//! Run with:  cargo run --release --example short_rate_models

use chrono::NaiveDate;
use rand::{Rng, SeedableRng};
use rustyqlib::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::engines::jamshidian::european_swaption;
use rustyqlib::rates::models::calibration::calibrate_hull_white_piecewise;
use rustyqlib::{
    BermudanSwaption, CapFloor, CapOrFloor, GridConfig, HullWhite, PayerReceiver, RateVol,
    RateVolKind, ShortRateModel, Swaption, SwaptionVolSurface, VanillaSwap,
};

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

    // The dated products: schedules, day counts and calendars live on
    // the product; the model stays in year fractions. A 1y-into-5y USD
    // swaption exercised two business days before the swap starts...
    let d = |y, m, day| NaiveDate::from_ymd_opt(y, m, day).unwrap();
    let probe = VanillaSwap::usd_standard(
        10_000_000.0,
        0.04,
        PayerReceiver::Payer,
        d(2027, 8, 16),
        d(2032, 8, 16),
    )?;
    let atm_dated = probe.par_rate(&curve, &curve)?;
    let underlying = VanillaSwap::usd_standard(
        10_000_000.0,
        atm_dated,
        PayerReceiver::Payer,
        d(2027, 8, 16),
        d(2032, 8, 16),
    )?;
    let swaption = Swaption::new(underlying, d(2027, 8, 12))?;
    println!(
        "\n1y5y USD payer swaption on 10mm, expiry 12-Aug-2027, struck ATM ({:.4}%):",
        atm_dated * 100.0
    );
    let hw_price = swaption.npv_hull_white(&model)?;
    println!(
        "  Hull-White {:.2}  = {:.1} bp normal vol  ({:.2}% Black vol)",
        hw_price,
        swaption.implied_normal_vol_hull_white(&model)? * 10_000.0,
        swaption.implied_black_vol(&curve, &curve, hw_price, 0.0)? * 100.0
    );
    // ...and the other way: a screen quote of 85bp normal into a price
    println!(
        "  at 85bp normal vol: {:.2}",
        swaption.npv_black(&curve, &curve, RateVol::Normal(0.0085))?
    );

    // ...and a 3y quarterly cap and floor starting a year forward
    let cap = CapFloor::usd_standard(
        10_000_000.0,
        0.045,
        CapOrFloor::Cap,
        d(2027, 8, 16),
        d(2030, 8, 16),
    )?;
    let floor = CapFloor::usd_standard(
        10_000_000.0,
        0.045,
        CapOrFloor::Floor,
        d(2027, 8, 16),
        d(2030, 8, 16),
    )?;
    println!(
        "\n3y quarterly cap/floor from 16-Aug-2027, K = 4.5%, 10mm (ATM {:.4}%):",
        cap.atm_strike(&curve)? * 100.0
    );
    println!(
        "  cap {:.2}   floor {:.2}",
        cap.npv_hull_white(&model)?,
        floor.npv_hull_white(&model)?
    );
    for c in cap.caplet_values(&model, &curve, None)?.iter().take(3) {
        println!(
            "  caplet {} -> {}  forward {:.4}%  value {:.2}",
            c.start,
            c.end,
            c.forward_rate * 100.0,
            c.value
        );
    }

    // A screen of ATM normal vols -> a piecewise-sigma Hull-White fitted
    // expiry by expiry -> a Bermudan priced on the grid
    let surface = SwaptionVolSurface::new(
        vec![1.0, 2.0, 3.0, 5.0],
        vec![5.0],
        vec![vec![0.0095], vec![0.0092], vec![0.0090], vec![0.0086]],
        RateVolKind::Normal,
    )?;
    let quotes = surface.usd_standard_quotes(&curve)?;
    let fit = calibrate_hull_white_piecewise(&curve, &quotes, 0.05)?;
    println!("\npiecewise Hull-White bootstrapped to a 5y-tenor column (a = 5%):");
    for (i, sigma) in fit.model.sigmas().iter().enumerate() {
        let from = if i == 0 {
            0.0
        } else {
            fit.model.sigma_times()[i - 1]
        };
        let to = fit
            .model
            .sigma_times()
            .get(i)
            .map(|t| format!("{t:.2}"))
            .unwrap_or_else(|| "inf".into());
        println!("  sigma on [{from:.2}, {to}) = {:.1} bp", sigma * 10_000.0);
    }
    let bermudan = BermudanSwaption::on_fixed_period_starts(
        VanillaSwap::usd_standard(
            10_000_000.0,
            atm_dated,
            PayerReceiver::Payer,
            d(2026, 8, 17),
            d(2032, 8, 17),
        )?,
        d(2027, 8, 1),
    )?;
    let europeans = bermudan.european_values_hull_white(&fit.model)?;
    println!(
        "\n6nc1 Bermudan payer swaption on 10mm at {:.4}%: {:.2}  (best European {:.2}, {} exercise dates)",
        atm_dated * 100.0,
        bermudan.npv_hull_white(&fit.model, &GridConfig::default())?,
        europeans.iter().cloned().fold(f64::MIN, f64::max),
        bermudan.exercise_dates.len()
    );

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
