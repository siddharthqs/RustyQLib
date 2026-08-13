//! Corporate bond analytics: spreads over a Treasury curve, a call
//! schedule with yield-to-worst, hazard-rate pricing, and a floating
//! rate note quoted by discount margin.
//!
//! Run with:  cargo run --release --example corporate_bond

use chrono::NaiveDate;
use rustyqlib::bonds::g_spread;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::{CallOption, FixedRateBond, FloatingRateNote};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A 5.5% 2031 investment-grade corporate: 30/360 semiannual, T+2
    let bond = FixedRateBond::us_corporate(100.0, 0.055, date(2026, 5, 15), date(2031, 5, 15))?;
    let settlement = bond.settlement_date(date(2026, 8, 5)); // Fri Aug 7
    let treasury_curve = YieldCurve::flat(
        0.04,
        settlement,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;

    let clean = 98.75;
    let ytm = bond.yield_from_clean_price(clean, settlement)?;
    let z = bond.z_spread(clean, &treasury_curve, settlement)?;
    println!("5.5% May-31 corporate at {clean} (settles {settlement}):");
    println!("  yield to maturity  {:.4}%", ytm * 100.0);
    println!(
        "  accrued (30/360)   {:.6}",
        bond.accrued_interest(settlement)?
    );
    println!(
        "  z-spread           {:.1} bp over the Treasury curve",
        z * 10_000.0
    );
    println!(
        "  spread DV01        {:.4} per 100 face / bp",
        bond.spread_dv01(&treasury_curve, z, settlement)?
    );
    // G-spread against on-the-run Treasury yields
    let benchmarks = [(2.0, 0.0428), (5.0, 0.0405), (10.0, 0.0415)];
    println!(
        "  G-spread           {:.1} bp (vs interpolated on-the-runs)",
        g_spread(ytm, 4.77, &benchmarks)? * 10_000.0
    );

    // Callable at 102.75 in 2028 stepping down to 100 in 2030
    let calls = [
        CallOption {
            call_date: date(2028, 5, 15),
            call_price: 102.75,
        },
        CallOption {
            call_date: date(2029, 5, 15),
            call_price: 101.375,
        },
        CallOption {
            call_date: date(2030, 5, 15),
            call_price: 100.0,
        },
    ];
    println!("\ncall schedule (price steps 102.75 / 101.375 / 100.00):");
    for call in &calls {
        println!(
            "  yield to {}  {:.4}%",
            call.call_date,
            bond.yield_to_call(clean, settlement, call)? * 100.0
        );
    }
    println!(
        "  yield to worst   {:.4}%",
        bond.yield_to_worst(clean, settlement, &calls)? * 100.0
    );

    // Credit view: what default intensity does the price imply at 40%
    // recovery, and how does it tie back to the spread?
    let recovery = 0.40;
    let hazard = bond.implied_hazard_rate(clean, &treasury_curve, recovery, settlement)?;
    println!("\ncredit view at {:.0}% recovery:", recovery * 100.0);
    println!(
        "  implied hazard rate       {:.2}% per year",
        hazard * 100.0
    );
    println!(
        "  credit triangle check     lambda*(1-R) = {:.1} bp vs z-spread {:.1} bp",
        hazard * (1.0 - recovery) * 10_000.0,
        z * 10_000.0
    );
    println!(
        "  reprice at that hazard    {:.6}",
        bond.risky_clean_price(&treasury_curve, hazard, recovery, settlement)?
    );

    // A 2-year FRN paying index + 75bp, quoted at a 95bp discount margin
    let frn = FloatingRateNote::usd_standard(
        100.0,
        0.0075,
        date(2026, 8, 6),
        date(2028, 8, 6),
        Some(0.0505), // current coupon already fixed at 5.05%
    )?;
    let index_curve = YieldCurve::flat(
        0.043,
        settlement,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    let frn_clean = frn.clean_price_from_discount_margin(&index_curve, 0.0095, settlement)?;
    println!("\n2y FRN, index + 75bp, current coupon fixed at 5.05%:");
    println!("  clean price at a 95bp discount margin  {frn_clean:.6}");
    println!(
        "  margin recovered from that price       {:.1} bp",
        frn.discount_margin_from_price(frn_clean, &index_curve, settlement)? * 10_000.0
    );

    // ── Option-model view of the same callable ─────────────────────────
    // Calibrate Hull-White to ATM swaptions, then price the call right
    // properly instead of quoting yield-to-worst.
    use rustyqlib::rates::models::calibration::{
        atm_swap_rate, calibrate_hull_white_sigma, SwaptionQuote,
    };
    use rustyqlib::rates::models::pricers::european_swaption;
    use rustyqlib::{HullWhite, PayerReceiver};

    // synthetic ATM swaption market generated at 110bp normal vol
    let market_model = HullWhite::new(0.05, 0.011, treasury_curve.clone())?;
    let quotes: Vec<SwaptionQuote> = [(1.0, 4usize), (2.0, 3usize), (3.0, 2usize)]
        .iter()
        .map(|&(expiry, tenor)| {
            let fixed_leg: Vec<(f64, f64)> =
                (1..=tenor).map(|i| (expiry + i as f64, 1.0)).collect();
            let strike = atm_swap_rate(&treasury_curve, expiry, &fixed_leg).unwrap();
            let price = european_swaption(
                &market_model,
                expiry,
                &fixed_leg,
                strike,
                1.0,
                PayerReceiver::Payer,
            )
            .unwrap();
            SwaptionQuote {
                expiry,
                fixed_leg,
                strike_rate: strike,
                market_price: price,
                payer_receiver: PayerReceiver::Payer,
            }
        })
        .collect();
    let fit = calibrate_hull_white_sigma(&treasury_curve, &quotes, 0.05, 0.02)?;
    println!("\nHull-White calibrated to {} ATM swaptions:", quotes.len());
    println!(
        "  sigma {:.2} bp (a fixed at 5%), price RMSE {:.2e}",
        fit.model.sigma * 10_000.0,
        fit.price_rmse
    );

    use rustyqlib::BondOptionality;
    let optionality = BondOptionality::from_calls(&calls);
    let oas = bond.oas_hw(clean, &fit.model, &optionality, settlement)?;
    let option_value = bond.call_option_value_hw(&fit.model, &calls, oas, settlement)?;
    let (eff_dur, eff_cvx) =
        bond.effective_duration_convexity_hw(&fit.model, &optionality, oas, settlement, 0.0025)?;
    let (straight_dur, _) = bond.effective_duration_convexity_hw(
        &fit.model,
        &BondOptionality::none(),
        oas,
        settlement,
        0.0025,
    )?;
    println!("\noption-model view of the callable at {clean}:");
    println!(
        "  OAS                 {:.1} bp (z-spread was {:.1} bp)",
        oas * 10_000.0,
        z * 10_000.0
    );
    println!("  call option value   {option_value:.4} per 100");
    println!("  effective duration  {eff_dur:.4} years (straight {straight_dur:.4})");
    println!("  effective convexity {eff_cvx:.4}");
    Ok(())
}
