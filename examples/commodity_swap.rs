//! A 6-month monthly WTI-style commodity swap priced on a contango
//! forward strip, then revalued mid-life with published fixings.
//!
//! Run with:  cargo run --release --example commodity_swap

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::PayerReceiver;
use rustyqlib::{
    Calendar, ClewlowStrickland, CommodityBasisSwap, CommodityForwardCurve, CommoditySwap,
    CommoditySwaption, PriceFixings,
};

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

    // WTI-vs-Brent-style basis swap: receive the WTI strip + spread,
    // pay the Brent strip trading ~$4 over, each leg on its own calendar
    let brent = CommodityForwardCurve::from_prices(
        effective,
        forward
            .pillars()
            .iter()
            .map(|&(date, price)| (date, price + 4.10))
            .collect(),
    )?;
    let basis = CommodityBasisSwap::monthly(
        10_000.0,
        0.0,
        effective,
        maturity,
        Calendar::WeekendsOnly, // WTI-style pricing days
        Calendar::UkSettlement, // Brent-style pricing days
    )?;
    let fair = basis.fair_spread(&discount, &forward, &brent)?;
    println!("\n6m basis swap, receive WTI + spread vs pay Brent (+$4.10):");
    println!("  fair spread    {:>14.4}", fair);
    let mut at_fair = basis.clone();
    at_fair.spread = fair;
    println!(
        "  PV at fair     {:>14.6}",
        at_fair.pv(&discount, &forward, &brent)?
    );
    println!(
        "  delta A / B    {:>10.2} / {:.2} (outright-flat)",
        basis.delta_a(&discount)?,
        basis.delta_b(&discount)?
    );

    // swaption: the right (not obligation) to lock a $73 payer swap on
    // the Mar-Aug 27 strip, decided next February
    let deferred_swap = CommoditySwap::monthly(
        10_000.0,
        73.0,
        PayerReceiver::Payer,
        date(2027, 3, 1),
        date(2027, 9, 1),
        Calendar::WeekendsOnly,
    )?;
    let swaption = CommoditySwaption::new(deferred_swap, date(2027, 2, 22))?;
    let vol = 0.30;
    println!("\npayer swaption into a 6m 73.00 swap, expiring 2027-02-22:");
    println!(
        "  forward par    {:>14.4}",
        swaption.forward_par_price(&discount, &forward)?
    );
    println!("  annuity        {:>14.2}", swaption.annuity(&discount)?);
    println!(
        "  premium        {:>14.2}",
        swaption.price(&discount, &forward, vol)?
    );
    println!(
        "  delta          {:>14.2}",
        swaption.delta(&discount, &forward, vol)?
    );
    println!(
        "  vega           {:>14.2}",
        swaption.vega(&discount, &forward, vol)?
    );

    // Clewlow-Strickland term structure: calibrate (sigma, alpha) to a
    // Samuelson-shaped strip of monthly option vols, then reprice the
    // swaption with maturity-aware covariances
    let vol_quotes = [
        // (option expiry, futures maturity, quoted Black vol)
        (0.10, 0.15, 0.415),
        (0.35, 0.40, 0.345),
        (0.60, 0.65, 0.302),
        (0.85, 0.90, 0.272),
    ];
    let fit = ClewlowStrickland::calibrate(&vol_quotes)?;
    let cs = fit.model;
    println!("\nClewlow-Strickland fit to the monthly vol strip:");
    println!(
        "  sigma / alpha  {:>14.4} / {:.4}  (rmse {:.2e})",
        cs.sigma, cs.alpha, fit.rmse
    );
    println!(
        "  effective vols: 3m contract {:.4}, 12m contract {:.4}",
        cs.effective_vol(0.20, 0.25)?,
        cs.effective_vol(0.95, 1.0)?
    );
    println!(
        "  swaption flat at sigma {:>10.2}",
        swaption.price(&discount, &forward, cs.sigma)?
    );
    println!(
        "  swaption CS            {:>10.2} (Samuelson-damped)",
        swaption.price_cs(&discount, &forward, &cs)?
    );
    Ok(())
}
