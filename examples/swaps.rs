//! Linear rates products: a vanilla IRS, a SOFR-style OIS and a basis
//! swap priced on flat curves.
//!
//! Run with:  cargo run --release --example swaps

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::rates::{BasisSwap, BasisSwapLeg, OvernightIndexSwap, PayerReceiver, VanillaSwap};
use rustyqlib::{BusinessDayConvention, Calendar, Frequency};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn flat(rate: f64, reference: NaiveDate) -> YieldCurve {
    YieldCurve::flat(
        rate,
        reference,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let effective = date(2026, 8, 6);
    let maturity = date(2031, 8, 6);
    let discount = flat(0.040, effective); // OIS discounting
    let forecast = flat(0.0425, effective); // projection curve

    // 5y payer IRS, USD conventions (semiannual 30/360 vs quarterly Act/360)
    let swap = VanillaSwap::usd_standard(
        10_000_000.0,
        0.042,
        PayerReceiver::Payer,
        effective,
        maturity,
    )?;
    println!("5y payer IRS, 10mm at 4.20% fixed (dual curve):");
    println!("  fixed leg PV   {:>14.2}", swap.fixed_leg_pv(&discount)?);
    println!(
        "  float leg PV   {:>14.2}",
        swap.float_leg_pv(&discount, &forecast)?
    );
    println!(
        "  swap PV        {:>14.2}",
        swap.pv_with(&discount, &forecast)?
    );
    println!(
        "  par rate       {:>13.4}%",
        swap.par_rate(&discount, &forecast)? * 100.0
    );
    println!(
        "  fixed-rate PV01 {:>13.2} (per bp of coupon)",
        swap.fixed_rate_pv01(&discount)?
    );
    println!(
        "  curve DV01     {:>14.2} (per bp parallel)",
        swap.dv01(&discount, &forecast)?
    );

    // 2y SOFR OIS: annual Act/360, T+2 payment lag, single curve
    let ois = OvernightIndexSwap::sofr_standard(
        10_000_000.0,
        0.0405,
        PayerReceiver::Receiver,
        effective,
        date(2028, 8, 6),
    )?;
    println!("\n2y receiver SOFR OIS, 10mm at 4.05% fixed:");
    println!("  PV             {:>14.2}", ois.pv(&discount)?);
    println!(
        "  par rate       {:>13.4}%",
        ois.par_rate(&discount, &discount)? * 100.0
    );

    // 3y basis swap: receive quarterly leg + spread, pay semiannual leg,
    // forecast curves 25bp apart
    let basis = BasisSwap::new(
        10_000_000.0,
        0.0,
        effective,
        date(2029, 8, 6),
        BasisSwapLeg {
            frequency: Frequency::Quarterly,
            day_count: DayCountConvention::Act360,
        },
        BasisSwapLeg {
            frequency: Frequency::Semiannual,
            day_count: DayCountConvention::Act360,
        },
        Calendar::UsGovernmentBond,
        BusinessDayConvention::ModifiedFollowing,
    )?;
    let fair = basis.fair_spread(&discount, &discount, &forecast)?;
    println!("\n3y basis swap, receive 3M leg + spread vs pay 6M leg (+25bp curve):");
    println!("  fair spread    {:>13.2} bp", fair * 10_000.0);
    let mut at_fair = basis.clone();
    at_fair.spread = fair;
    println!(
        "  PV at fair     {:>14.6}",
        at_fair.pv(&discount, &discount, &forecast)?
    );
    Ok(())
}
