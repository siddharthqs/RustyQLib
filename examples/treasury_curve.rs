//! Bootstrap a US Treasury discount curve from bill and note quotes,
//! verify it reprices the inputs, and use it to value an off-the-run
//! bond and its key-rate risk.
//!
//! Run with:  cargo run --release --example treasury_curve

use chrono::NaiveDate;
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::{
    bootstrap_curve, BillQuote, BondQuote, CurveInstrument, FixedRateBond, RateShift, TreasuryBill,
};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settlement = date(2026, 8, 6);

    // The quoted market: two bills (discount rates) and three notes
    // (clean prices) on the May 15 / Nov 15 coupon cycle.
    let bills = [
        (date(2026, 11, 5), 0.0480), // ~3M
        (date(2027, 2, 4), 0.0470),  // ~6M
    ];
    let notes = [
        (0.0450, date(2028, 5, 15), 99.50),  // 2Y 4.50%
        (0.0425, date(2031, 5, 15), 101.25), // 5Y 4.25%
        (0.0440, date(2036, 5, 15), 102.10), // 10Y 4.40%
    ];

    let mut instruments: Vec<Box<dyn CurveInstrument>> = Vec::new();
    for &(maturity, discount_rate) in &bills {
        instruments.push(Box::new(BillQuote::new(
            TreasuryBill::new(100.0, maturity)?,
            discount_rate,
            settlement,
        )?));
    }
    for &(coupon, maturity, clean) in &notes {
        let bond = FixedRateBond::us_treasury(100.0, coupon, date(2026, 5, 15), maturity)?;
        instruments.push(Box::new(BondQuote::new(bond, clean, settlement)?));
    }

    let curve = bootstrap_curve(&instruments, settlement, DayCountConvention::Act365)?;
    println!("bootstrapped Treasury curve (as of {settlement}):");
    println!(
        "{:>12} {:>10} {:>12} {:>10}",
        "pillar", "time", "df", "zero(cc)"
    );
    for p in curve.pillars() {
        let date = p.date.map_or_else(|| "-".into(), |d| d.to_string());
        println!(
            "{date:>12} {:>10.4} {:>12.8} {:>9.4}%",
            p.time,
            p.df,
            p.zero_rate * 100.0
        );
    }

    // The curve must reprice its inputs exactly
    println!("\nrepricing check:");
    for &(coupon, maturity, clean) in &notes {
        let bond = FixedRateBond::us_treasury(100.0, coupon, date(2026, 5, 15), maturity)?;
        let repriced = bond.clean_price_from_curve(&curve, settlement)?;
        println!(
            "  {:>5.2}% {maturity}: quoted {clean:>7.3}, curve {repriced:>10.6}",
            coupon * 100.0
        );
    }

    // Value an off-the-run 3.875% Feb 2033 bond on the fitted curve
    let off_the_run =
        FixedRateBond::us_treasury(100.0, 0.03875, date(2026, 2, 15), date(2033, 2, 15))?;
    let clean = off_the_run.clean_price_from_curve(&curve, settlement)?;
    let implied_yield = off_the_run.yield_from_clean_price(clean, settlement)?;
    println!("\noff-the-run 3.875% Feb 2033 on the fitted curve:");
    println!(
        "  clean price {clean:.6}, implied street yield {:.4}%",
        implied_yield * 100.0
    );

    // Key-rate DV01s: bump each pillar zero by 1bp and reprice
    println!("\nkey-rate DV01s (per 100 face / bp):");
    let base = off_the_run.dirty_price_from_curve(&curve, settlement)?;
    let mut total = 0.0;
    for p in curve.pillars() {
        let bumped = curve.bumped(&RateShift::KeyRateAbsolute {
            tenors: vec![p.time],
            shifts: vec![0.0001],
        })?;
        let moved = off_the_run.dirty_price_from_curve(&bumped, settlement)?;
        let dv01 = base - moved;
        total += dv01;
        println!("  {:>6.2}y {dv01:>9.6}", p.time);
    }
    let parallel = curve.bumped(&RateShift::ParallelAbsolute(0.0001))?;
    let parallel_dv01 = base - off_the_run.dirty_price_from_curve(&parallel, settlement)?;
    println!("  ladder sum {total:.6} vs parallel DV01 {parallel_dv01:.6}");
    Ok(())
}
