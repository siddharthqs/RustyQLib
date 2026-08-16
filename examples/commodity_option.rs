//! Black-76 options on a WTI-style future: premium, Greeks and implied
//! vol read off a commodity forward curve and a discount curve.
//!
//! Run with:  cargo run --release --example commodity_option

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::{
    AveragePriceOption, Calendar, CommodityForwardCurve, CommodityOption, FuturesSettlement,
};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let valuation = date(2026, 9, 1);
    let discount = YieldCurve::flat(
        0.04,
        valuation,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    // contango strip: $70 spot rising to $76 a year out
    let forward = CommodityForwardCurve::from_prices(
        valuation,
        vec![(date(2026, 9, 1), 70.0), (date(2027, 9, 1), 76.0)],
    )?;

    // Jun-27 call: expires mid-May, the future delivers in June
    let option = CommodityOption::new(
        1_000.0, // 1,000 bbl per contract
        75.0,
        PutOrCall::Call,
        date(2027, 5, 17),
        date(2027, 6, 1),
        FuturesSettlement::Discounted,
    )?;

    let vol = 0.35;
    println!("Jun-27 75 call on 1,000 bbl, 35% vol:");
    println!("  futures price  {:>12.4}", option.forward_price(&forward)?);
    println!(
        "  premium        {:>12.2}",
        option.price(&discount, &forward, vol)?
    );
    let greeks = option.greeks(&discount, &forward, vol)?;
    println!(
        "  delta          {:>12.2} (per $1 of the future)",
        greeks.delta
    );
    println!("  gamma          {:>12.4}", greeks.gamma);
    println!(
        "  vega           {:>12.2} (per vol point x100)",
        greeks.vega
    );
    println!("  theta          {:>12.2} (per year)", greeks.theta);
    println!("  rho            {:>12.2}", greeks.rho);

    // round-trip a quoted premium back to its implied vol
    let quoted = 4_850.0;
    println!(
        "\n  implied vol of a {quoted:.0} premium: {:.4}",
        option.implied_vol(&discount, &forward, quoted)?
    );

    // the same contract margined (futures-style) carries no discounting
    let mut margined = option.clone();
    margined.settlement = FuturesSettlement::Margined;
    println!(
        "  margined premium {:>10.2} (undiscounted)",
        margined.price(&discount, &forward, vol)?
    );

    // the Asian-style APO on the same month: averages the daily index
    // over June's business days, so it samples less variance and prices
    // below the vanilla
    let apo = AveragePriceOption::for_month(
        1_000.0,
        75.0,
        PutOrCall::Call,
        2027,
        6,
        Calendar::WeekendsOnly,
    )?;
    println!("\nJun-27 75 average price call, same size and vol:");
    println!(
        "  averaging      {:>12} business days, settles {}",
        apo.pricing_days().len(),
        apo.settlement_date()
    );
    println!(
        "  premium        {:>12.2}",
        apo.price(&discount, &forward, vol)?
    );
    println!(
        "  delta          {:>12.2}",
        apo.delta(&discount, &forward, vol)?
    );
    println!(
        "  vega           {:>12.2}",
        apo.vega(&discount, &forward, vol)?
    );
    Ok(())
}
