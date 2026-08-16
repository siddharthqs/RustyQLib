//! A 6-month monthly WTI-style commodity swap priced on a contango
//! forward strip, then revalued mid-life with published fixings.
//!
//! Run with:  cargo run --release --example commodity_swap

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::PayerReceiver;
use rustyqlib::{Calendar, CommodityForwardCurve, CommoditySwap, PriceFixings};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let effective = date(2026, 9, 1);
    let maturity = date(2027, 3, 1); // Sep-26 through Feb-27 strip
    let discount = YieldCurve::flat(
        0.04,
        effective,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    // contango: monthly forwards from $70 to $76 a barrel
    let forward = CommodityForwardCurve::from_prices(
        effective,
        (0..=6)
            .map(|i| {
                let m = 9 + i;
                let (y, m) = if m > 12 { (2027, m - 12) } else { (2026, m) };
                (date(y, m, 1), 70.0 + i as f64)
            })
            .collect(),
    )?;

    // consumer hedge: pay $71 fixed, receive the monthly average price
    // on 10,000 bbl a month, cash-settled 5 business days after month end
    let swap = CommoditySwap::monthly(
        10_000.0,
        71.0,
        PayerReceiver::Payer,
        effective,
        maturity,
        Calendar::WeekendsOnly,
    )?;

    println!("6m payer swap, 10,000 bbl/month at $71.00 fixed:");
    println!("  period      days  avg forward     df(pay)");
    for period in swap.periods()? {
        println!(
            "  {}    {:>2}    {:>8.4}    {:>8.6}",
            period.start.format("%b %Y"),
            swap.pricing_days(&period).len(),
            swap.average_price(&period, &forward)?,
            discount.df_date(period.payment),
        );
    }
    println!(
        "  float leg PV   {:>14.2}",
        swap.float_leg_pv(&discount, &forward)?
    );
    println!("  fixed leg PV   {:>14.2}", swap.fixed_leg_pv(&discount)?);
    println!("  swap PV        {:>14.2}", swap.pv(&discount, &forward)?);
    println!(
        "  par price      {:>14.4}",
        swap.par_price(&discount, &forward)?
    );
    println!(
        "  delta          {:>14.2} (per $1 of forward strip)",
        swap.delta(&discount)?
    );
    println!(
        "  fixed-price dv {:>14.2} (per $1 of fixed price)",
        swap.fixed_price_delta(&discount)?
    );
    println!(
        "  curve DV01     {:>14.2} (per bp of discounting)",
        swap.dv01(&discount, &forward)?
    );

    // mid-September revaluation: the first eleven pricing days settled
    // around $73.50, the forward strip has shifted up $2
    let asof = date(2026, 9, 16);
    let mut fixings = PriceFixings::new();
    let mut day = effective;
    while day < asof {
        fixings.insert(day, 73.50);
        day = day.succ_opt().unwrap();
    }
    let shifted = forward.bumped(2.0)?;
    println!("\nrevalued {asof}, fixings at $73.50, strip +$2.00:");
    println!(
        "  swap PV        {:>14.2}",
        swap.pv_with_fixings(&discount, &shifted, &fixings, asof)?
    );
    println!(
        "  par price      {:>14.4}",
        swap.par_price_with_fixings(&discount, &shifted, &fixings, asof)?
    );
    println!(
        "  delta          {:>14.2} (fixed days no longer float)",
        swap.delta_with_fixings(&discount, asof)?
    );
    Ok(())
}
