//! US Treasury note analytics: accrued interest, price/yield, duration,
//! convexity, DV01 and discount-curve pricing.
//!
//! Run with:  cargo run --release --example us_treasury_bond

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::{FixedRateBond, RateShift, TreasuryBill};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A 4.25% 10-year note on the May 15 / Nov 15 cycle, street
    // conventions applied automatically (semiannual Act/Act ICMA, SIFMA
    // calendar, T+1 settlement).
    let bond =
        FixedRateBond::us_treasury(1_000_000.0, 0.0425, date(2026, 5, 15), date(2036, 5, 15))?;

    let trade_date = date(2026, 8, 5);
    let settlement = bond.settlement_date(trade_date);
    println!("trade {trade_date}, settles {settlement} (T+1)\n");

    // Quote-side analytics at a 4.10% street yield
    let yield_rate = 0.0410;
    let clean = bond.clean_price_from_yield(yield_rate, settlement)?;
    let accrued = bond.accrued_interest(settlement)?;
    let dirty = bond.dirty_price_from_yield(yield_rate, settlement)?;
    println!("at {:.3}% yield:", yield_rate * 100.0);
    println!("  clean price   {clean:>10.6} per 100");
    println!("  accrued       {accrued:>10.6} per 100");
    println!("  dirty price   {dirty:>10.6} per 100");

    // and back: price -> yield
    let implied = bond.yield_from_clean_price(clean, settlement)?;
    println!(
        "  yield from clean price: {:.6}% (round trip)",
        implied * 100.0
    );

    // Risk measures
    println!("\nrisk at {:.3}% yield:", yield_rate * 100.0);
    println!(
        "  Macaulay duration {:>8.4} years",
        bond.macaulay_duration(yield_rate, settlement)?
    );
    println!(
        "  modified duration {:>8.4} years",
        bond.modified_duration(yield_rate, settlement)?
    );
    println!(
        "  convexity         {:>8.4} years^2",
        bond.convexity(yield_rate, settlement)?
    );
    println!(
        "  DV01              {:>8.4} per 100 face / bp",
        bond.dv01(yield_rate, settlement)?
    );

    // Curve-side pricing: a flat 4% discount curve, then a +1bp parallel
    // bump to read a curve DV01
    let curve = YieldCurve::flat(
        0.04,
        settlement,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    let clean_on_curve = bond.clean_price_from_curve(&curve, settlement)?;
    let bumped = curve.bumped(&RateShift::ParallelAbsolute(0.0001))?;
    let clean_bumped = bond.clean_price_from_curve(&bumped, settlement)?;
    println!("\non a flat 4% (continuous) curve:");
    println!("  clean price   {clean_on_curve:>10.6} per 100");
    println!(
        "  curve DV01    {:>10.6} per 100 face / bp",
        clean_on_curve - clean_bumped
    );
    println!("  PV of {:.0} face: {:.2}", 1_000_000.0, bond.pv(&curve));

    // A 13-week bill quoted at a 4.85% discount rate
    let bill = TreasuryBill::new(1_000_000.0, date(2026, 11, 5))?;
    let discount_rate = 0.0485;
    println!("\n13-week bill at {:.3}% discount:", discount_rate * 100.0);
    println!(
        "  price {:.6} per 100, BEY {:.4}%",
        bill.price_from_discount_rate(discount_rate, settlement)?,
        bill.bond_equivalent_yield(discount_rate, settlement)? * 100.0
    );
    Ok(())
}
