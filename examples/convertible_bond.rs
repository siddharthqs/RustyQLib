//! Convertible bond under Tsiveriotis-Fernandes: price across the
//! moneyness spectrum, the bond floor and parity boundaries, soft-call
//! and put effects, and the implied credit spread.
//!
//! Run with:  cargo run --release --example convertible_bond

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::{
    CallOption, ConvertibleBond, ConvertibleMarket, ConvertiblePreferred, FixedRateBond, PutOption,
};

fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2% coupon 2031 convertible, 1000 face, 20 shares per bond
    // (conversion price 50), issuer soft-callable at 102 from 2028 when
    // the stock trades at 130% of conversion, holder put at par in 2029
    let chassis = FixedRateBond::us_corporate(1000.0, 0.02, date(2026, 5, 15), date(2031, 5, 15))?;
    let mut convertible = ConvertibleBond::new(chassis, 20.0)?;
    convertible.calls = vec![CallOption {
        call_date: date(2028, 5, 15),
        call_price: 102.0,
    }];
    convertible.soft_call_trigger = Some(65.0);
    convertible.puts = vec![PutOption {
        put_date: date(2029, 5, 15),
        put_price: 100.0,
    }];

    let curve = YieldCurve::flat(
        0.04,
        date(2026, 8, 13),
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    let settlement = date(2026, 8, 14);

    println!(
        "2% 2031 convertible, ratio 20 (conversion price {:.2}):",
        convertible.conversion_price()
    );
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "spot", "clean", "floor", "parity", "premium", "delta"
    );
    for spot in [30.0, 40.0, 48.0, 55.0, 65.0, 80.0] {
        let market = ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        };
        let clean = convertible.clean_price(&market, &curve, settlement)?;
        let floor = convertible.bond_floor(&market, &curve, settlement)?;
        let parity = convertible.parity(spot);
        let premium = convertible.conversion_premium(clean, spot)?;
        let delta = convertible.delta(&market, &curve, settlement)?;
        println!(
            "{spot:>8.2} {clean:>10.4} {floor:>10.4} {parity:>10.2} {:>9.1}% {delta:>9.4}",
            premium * 100.0
        );
    }

    // read the market's credit view out of a quoted price
    let market = ConvertibleMarket {
        spot: 48.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.02,
    };
    let quoted = convertible.clean_price(&market, &curve, settlement)? - 1.5; // trading cheap
    let implied = convertible.implied_credit_spread(quoted, &market, &curve, settlement)?;
    println!(
        "\nquoted {quoted:.4} vs model at 200bp: implied credit spread {:.0} bp",
        implied * 10_000.0
    );

    // --- convertible preferred stock ---------------------------------
    // 5.5% cumulative perpetual on a $100 preference, converting into
    // 1.6 common shares (conversion price 62.50), callable at 101 from
    // 2029 once the stock trades at 130% of the conversion price
    let mut preferred = ConvertiblePreferred::new(100.0, 0.055, date(2026, 5, 15), 1.6)?;
    preferred.calls = vec![CallOption {
        call_date: date(2029, 5, 15),
        call_price: 101.0,
    }];
    preferred.soft_call_trigger = Some(1.3 * preferred.conversion_price());

    println!(
        "\n5.5% perpetual convertible preferred, ratio 1.6 (conversion price {:.2}):",
        preferred.conversion_price()
    );
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "spot", "clean", "floor", "parity", "cur yld", "delta"
    );
    for spot in [30.0, 45.0, 62.5, 80.0, 100.0] {
        let market = ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            // preferreds trade wide of the same issuer's senior debt:
            // deferral risk and deep subordination live in the spread
            credit_spread: 0.035,
        };
        let clean = preferred.clean_price(&market, &curve, settlement)?;
        let floor = preferred.preferred_floor(&market, &curve, settlement)?;
        let parity = preferred.parity(spot);
        let current_yield = preferred.current_yield(clean)?;
        let delta = preferred.delta(&market, &curve, settlement)?;
        println!(
            "{spot:>8.2} {clean:>10.4} {floor:>10.4} {parity:>10.2} {:>9.2}% {delta:>9.4}",
            current_yield * 100.0
        );
    }
    Ok(())
}
