//! Black-76 options on a WTI-style future: premium, Greeks and implied
//! vol read off a commodity forward curve and a discount curve.
//!
//! Run with:  cargo run --release --example commodity_option

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::{
    AveragePriceOption, Calendar, CommodityForwardCurve, CommodityOption, CommoditySpreadOption,
    CommodityVol, FuturesSettlement, ShiftedSabr,
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
    println!("  futures price  {:>12.4}", option.forward_price(&forward));
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

    // a Waha-style basis call: the strip is negative, so Black-76 cannot
    // price it — quote a Bachelier (normal) vol instead
    let waha = CommodityForwardCurve::from_prices(
        valuation,
        vec![(date(2026, 9, 1), -1.80), (date(2027, 9, 1), 0.40)],
    )?;
    let basis_call = CommodityOption::new(
        10_000.0, // MMBtu
        -0.50,
        PutOrCall::Call,
        date(2027, 5, 17),
        date(2027, 6, 1),
        FuturesSettlement::Discounted,
    )?;
    let normal_vol = CommodityVol::Normal(1.25); // $/MMBtu per sqrt(year)
    println!("\nJun-27 -0.50 basis call on a negative strip, Bachelier:");
    println!("  forward        {:>12.4}", basis_call.forward_price(&waha));
    println!(
        "  premium        {:>12.2}",
        basis_call.price(&discount, &waha, normal_vol)?
    );
    println!(
        "  delta          {:>12.2}",
        basis_call.greeks(&discount, &waha, normal_vol)?.delta
    );

    // Brent-WTI location spread call via Kirk: receive Brent, pay WTI,
    // struck at $3 on the differential
    let brent = CommodityForwardCurve::from_prices(
        valuation,
        vec![(date(2026, 9, 1), 74.1), (date(2027, 9, 1), 80.1)],
    )?;
    let spread = CommoditySpreadOption::new(
        1_000.0,
        3.0,
        PutOrCall::Call,
        date(2027, 5, 17),
        date(2027, 6, 1), // Jun-27 Brent
        date(2027, 6, 1), // Jun-27 WTI
        FuturesSettlement::Discounted,
    )?;
    let (rho, vol_brent, vol_wti) = (0.85, 0.32, 0.35);
    let (f_brent, f_wti) = spread.forward_prices(&brent, &forward);
    println!("\nJun-27 Brent-WTI spread call struck at 3.00 (Kirk):");
    println!("  forwards       {f_brent:>12.4} / {f_wti:.4}");
    println!(
        "  premium        {:>12.2}",
        spread.price(&discount, &brent, &forward, vol_brent, vol_wti, rho)?
    );
    println!(
        "  delta A / B    {:>12.2} / {:.2}",
        spread.delta_a(&discount, &brent, &forward, vol_brent, vol_wti, rho)?,
        spread.delta_b(&discount, &brent, &forward, vol_brent, vol_wti, rho)?
    );
    println!(
        "  cega           {:>12.2} (per unit of correlation)",
        spread.cega(&discount, &brent, &forward, vol_brent, vol_wti, rho)?
    );

    // shifted SABR on the negative basis strip: calibrate (alpha, rho,
    // nu) at beta 0.7 and shift $10 to quoted shifted-Black vols, then
    // price the basis call smile-consistently, strike by strike
    let basis_forward = basis_call.forward_price(&waha);
    let smile_quotes = [
        (-3.0, 0.52),
        (-1.5, 0.46),
        (-0.5, 0.43),
        (0.5, 0.42),
        (2.0, 0.44),
    ];
    let fit = ShiftedSabr::calibrate(&smile_quotes, basis_forward, 0.71, 0.7, 10.0)?;
    println!("\nshifted SABR fit to the basis smile (beta 0.7, shift 10.00):");
    println!(
        "  alpha/rho/nu   {:>12.4} / {:.4} / {:.4}  (rmse {:.2e})",
        fit.sabr.params.alpha, fit.sabr.params.rho, fit.sabr.params.nu, fit.rmse
    );
    for strike in [-2.0, -0.5, 1.0] {
        let mut k_call = basis_call.clone();
        k_call.strike = strike;
        let quote = fit.sabr.quote_for(&k_call, &discount, &waha)?;
        println!(
            "  K {strike:>5.2}  vol {:>7.4}  premium {:>10.2}",
            quote.vol(),
            k_call.price(&discount, &waha, quote)?
        );
    }
    Ok(())
}
