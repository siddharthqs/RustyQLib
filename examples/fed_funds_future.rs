//! Money-market futures: CME fed funds (ZQ) and SOFR (SR1, SR3).
//!
//! Fair pricing off a curve, partially realized contracts, the FOMC
//! hike probability implied by a quoted ZQ price, and the arithmetic
//! (SR1) versus compounded (SR3) SOFR averaging.
//!
//! Run with:  cargo run --release --example fed_funds_future

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::overnight_forward;
use rustyqlib::{FedFundsFuture, RateFixings, SofrFuture};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let asof = date(2026, 9, 11);
    let curve = YieldCurve::flat(
        0.0435,
        asof,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;

    // October 2026 contract: not yet started, priced purely off the curve
    let october = FedFundsFuture::for_month(2026, 10)?;
    println!("ZQ Oct-26 ({} calendar days):", october.days_in_month());
    println!(
        "  fair average rate  {:.4}%",
        october.fair_rate(&curve)? * 100.0
    );
    println!(
        "  fair price         {:.4}  (rounded {:.3})",
        october.fair_price(&curve)?,
        FedFundsFuture::round_price(october.fair_price(&curve)?)
    );
    println!("  DV01               ${:.2} per contract", october.dv01());

    // September 2026 contract: 10 days realized at 4.33%, rest projected
    let september = FedFundsFuture::for_month(2026, 9)?;
    let mut fixings = RateFixings::new();
    fixings.insert(date(2026, 8, 31), 0.0433);
    let blended = september.fair_rate_with_fixings(&curve, &fixings, asof)?;
    println!("\nZQ Sep-26, partially realized as of {asof}:");
    println!("  realized (1-10 Sep) 4.3300%  at 4.33% EFFR");
    println!(
        "  projected (11-30)   {:.4}%",
        overnight_forward(&curve, date(2026, 9, 20))? * 100.0
    );
    println!("  blended average    {:.4}%", blended * 100.0);
    println!(
        "  fair price         {:.4}",
        september.fair_price_with_fixings(&curve, &fixings, asof)?
    );

    // FOMC: a hike effective Oct 29 splits the November contract
    let november = FedFundsFuture::for_month(2026, 11)?;
    let effective = date(2026, 11, 5); // day after the meeting decision
    let (before, after) = november.split_at(effective)?;
    let pre_rate = 0.0433;
    println!("\nZQ Nov-26 with a rate change effective {effective}:");
    println!("  {before} days at the prevailing rate, {after} days at the new one");
    for quoted in [95.67, 95.60, 95.55] {
        let post = november.implied_rate_after_from_price(quoted, pre_rate, effective)?;
        let probability =
            november.implied_move_probability(quoted, pre_rate, 0.0025, effective)? * 100.0;
        println!(
            "  price {quoted:.2} -> implied post-meeting {:.4}%, P(25bp hike) {:.1}%",
            post * 100.0,
            probability
        );
    }

    // Position P&L: long 20 contracts, price rallies 2.5bp
    println!(
        "\nlong 20 ZQ from 95.6700 to 95.6950: P&L ${:.2}",
        november.pnl(95.67, 95.695, 20.0)
    );

    // SOFR futures: SR1 averages arithmetically over the calendar month,
    // SR3 compounds daily over an IMM quarter
    let sr1 = SofrFuture::one_month(2026, 10)?;
    let sr3 = SofrFuture::three_month(2026, 9)?;
    println!("\nSOFR futures on the same curve:");
    println!(
        "  SR1 Oct-26  {} days, arithmetic  rate {:.4}%  price {:.4}  DV01 ${:.2}",
        sr1.reference_days(),
        sr1.fair_rate(&curve)? * 100.0,
        sr1.fair_price(&curve)?,
        sr1.dv01()
    );
    println!(
        "  SR3 Sep-26  {} days, compounded  rate {:.4}%  price {:.4}  DV01 ${:.2}",
        sr3.reference_days(),
        sr3.fair_rate(&curve)? * 100.0,
        sr3.fair_price(&curve)?,
        sr3.dv01()
    );
    println!(
        "  SR3 reference period {} to {} (IMM Wednesdays)",
        sr3.start, sr3.end
    );
    println!(
        "  compounding pickup over the overnight equivalent: {:.2} bp",
        (sr3.fair_rate(&curve)? - 0.0435 * 360.0 / 365.0) * 10_000.0
    );

    // The futures/forward convexity bias on a deferred SR3
    let deferred = SofrFuture::three_month(2027, 9)?;
    let futures_rate = deferred.fair_rate(&curve)?;
    let forward = deferred.forward_rate_from_futures(
        futures_rate,
        0.011, // 110bp annualized rate vol
        asof,
        DayCountConvention::Act365,
    )?;
    println!(
        "\nSR3 Sep-27 futures {:.4}% -> forward {:.4}% (convexity {:.2} bp at 110bp vol)",
        futures_rate * 100.0,
        forward * 100.0,
        (futures_rate - forward) * 10_000.0
    );
    Ok(())
}
