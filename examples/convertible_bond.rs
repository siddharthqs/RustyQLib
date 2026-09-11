//! Convertible bond under Tsiveriotis-Fernandes and jump to default:
//! price across the moneyness spectrum, the bond floor and parity
//! boundaries, soft-call and put effects, the implied credit spread,
//! and the hazard-rate view with its implied hazard.
//!
//! Run with:  cargo run --release --example convertible_bond

use chrono::NaiveDate;
use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::{
    dejump_implied_vol, CallOption, CashDividend, ContingentConversion, ConvertibleBond,
    ConvertibleFdGrid, ConvertibleMarket, ConvertiblePreferred, ConvertiblePricing,
    CouponMakeWhole, DividendProtection, EquityLinkedHazardMarket, FdVolModel, FixedRateBond,
    FundamentalChangeMakeWhole, HazardLevel, JumpToDefaultMarket, MandatoryConversion, PutOption,
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
    // or, pinning the spread at 200bp, read the same cheapness as vol
    let implied_vol = convertible.implied_volatility(quoted, &market, &curve, settlement)?;
    println!(
        "quoted {quoted:.4} vs model at 30% vol:  implied volatility {:.1}%",
        implied_vol * 100.0
    );

    // --- jump to default ---------------------------------------------
    // the same bond on the reduced-form model: the stock is absorbed at
    // zero on default (hazard 3%/yr, 40% recovery on face) and the
    // hedger pays 50bp to borrow the shares
    println!(
        "
jump to default, hazard 300bp, recovery 40%, borrow 50bp:"
    );
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "spot", "clean", "floor", "parity", "premium", "delta"
    );
    for spot in [30.0, 40.0, 48.0, 55.0, 65.0, 80.0] {
        let market = JumpToDefaultMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            borrow_cost: 0.005,
            hazard: HazardLevel::Flat(0.03),
            recovery_rate: 0.40,
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
    let market = JumpToDefaultMarket {
        spot: 48.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        borrow_cost: 0.005,
        hazard: HazardLevel::Flat(0.03),
        recovery_rate: 0.40,
    };
    // the put floors the cash leg, so the price is shallow in the hazard:
    // 0.75 points cheap already reads as roughly double the hazard
    let quoted = convertible.clean_price(&market, &curve, settlement)? - 0.75;
    let implied = convertible.implied_hazard_rate(quoted, &market, &curve, settlement)?;
    println!(
        "
quoted {quoted:.4} vs model at 300bp hazard: implied hazard rate {:.0} bp",
        implied * 10_000.0
    );
    // a listed 30% implied vol already carries the default jump: the
    // diffusion the model should be given is lower
    let survival_vol = dejump_implied_vol(0.30, 48.0, 48.0, 4.75, 0.04, 0.01, 0.005, 0.03)?;
    println!(
        "a listed 30% at-the-money vol de-jumps to {:.1}% at 300bp hazard over the bond's life",
        survival_vol * 100.0
    );

    // --- finite differences ------------------------------------------
    // the same two models on a Crank-Nicolson grid: one solve returns
    // the value with delta and gamma read off the spot axis
    println!(
        "
finite differences (400 x 400 grid) against the 800-step trees:"
    );
    println!(
        "{:>8} {:>10} {:>10} {:>9} {:>9} {:>10} {:>10} {:>9} {:>9}",
        "spot", "tf tree", "tf fd", "delta", "gamma", "jtd tree", "jtd fd", "delta", "gamma"
    );
    for spot in [30.0, 48.0, 65.0] {
        let tf = ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        };
        let jtd = JumpToDefaultMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            borrow_cost: 0.005,
            hazard: HazardLevel::Flat(0.03),
            recovery_rate: 0.40,
        };
        let grid = ConvertibleFdGrid::default();
        let tf_tree = convertible.clean_price(&tf, &curve, settlement)?;
        let tf_fd = convertible.fd_valuation(&tf, &curve, settlement, grid)?;
        let jtd_tree = convertible.clean_price(&jtd, &curve, settlement)?;
        let jtd_fd = convertible.fd_valuation(&jtd, &curve, settlement, grid)?;
        println!(
            "{spot:>8.2} {tf_tree:>10.4} {:>10.4} {:>9.4} {:>9.5} {jtd_tree:>10.4} {:>10.4} {:>9.4} {:>9.5}",
            tf_fd.clean_price, tf_fd.delta, tf_fd.gamma, jtd_fd.clean_price, jtd_fd.delta, jtd_fd.gamma
        );
    }

    // --- equity-linked hazard ----------------------------------------
    // the same 300bp at the reference price 48, with the hazard rising
    // as the stock falls: lambda = a (48 / S)^p; p = 0 is jump to default
    println!(
        "\
equity-linked hazard, 300bp at 48, 40% recovery (grid):"
    );
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}",
        "spot", "p = 0", "p = 1", "p = 2", "delta 0", "delta 1", "delta 2"
    );
    for spot in [30.0, 40.0, 48.0, 65.0] {
        let base = JumpToDefaultMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            borrow_cost: 0.005,
            hazard: HazardLevel::Flat(0.03),
            recovery_rate: 0.40,
        };
        let at = |p: f64| {
            let market = EquityLinkedHazardMarket {
                reference_spot: 48.0,
                ..EquityLinkedHazardMarket::new(base.clone(), p).unwrap()
            };
            convertible.fd_valuation(&market, &curve, settlement, ConvertibleFdGrid::default())
        };
        let (v0, v1, v2) = (at(0.0)?, at(1.0)?, at(2.0)?);
        println!(
            "{spot:>8.2} {:>10.4} {:>10.4} {:>10.4} {:>9.4} {:>9.4} {:>9.4}",
            v0.clean_price, v1.clean_price, v2.clean_price, v0.delta, v1.delta, v2.delta
        );
    }

    // --- pluggable volatility ----------------------------------------
    // the grid takes any volatility field: here a downside skew, 30% at
    // the conversion price rising below it, against the flat 30%
    let skew = |s: f64, _t: f64| (0.30 * (50.0 / s).powf(0.3)).clamp(0.05, 1.5);
    println!("\nflat 30% vs a downside skew (30% at the conversion price) on the grid:");
    println!(
        "{:>8} {:>10} {:>10} {:>9} {:>9}",
        "spot", "flat", "skewed", "delta", "delta"
    );
    for spot in [30.0, 48.0, 65.0] {
        let market = ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        };
        let grid = ConvertibleFdGrid::default();
        let flat = convertible.fd_valuation(&market, &curve, settlement, grid)?;
        let skewed = convertible.fd_valuation_with_vol(
            &market,
            &curve,
            settlement,
            grid,
            &FdVolModel::Custom(&skew),
        )?;
        println!(
            "{spot:>8.2} {:>10.4} {:>10.4} {:>9.4} {:>9.4}",
            flat.clean_price, skewed.clean_price, flat.delta, skewed.delta
        );
    }

    // --- discrete cash dividends -------------------------------------
    // the same bond with the 1% yield replaced by 0.48 a share paid each
    // May and November (the same 1% on a 48 stock): the ex-date jump
    // V(S) = V(S - D) on both engines
    let mut paying = convertible.clone();
    paying.cash_dividends = (0..10)
        .map(|k| CashDividend {
            ex_date: date(2026 + (k + 1) / 2, if k % 2 == 0 { 11 } else { 5 }, 1),
            amount: 0.48,
        })
        .collect();
    let with_yield = ConvertibleMarket {
        spot: 48.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.02,
    };
    let with_cash = ConvertibleMarket {
        dividend_yield: 0.0,
        ..with_yield
    };
    let mut protected = paying.clone();
    protected.dividend_protection = Some(DividendProtection { threshold: 0.25 });
    println!(
        "
dividends at spot 48: 1% yield {:.4}, ten cash dividends of 0.48 {:.4} (tree) / {:.4} (grid), protected above 0.25 {:.4}",
        convertible.clean_price(&with_yield, &curve, settlement)?,
        paying.clean_price(&with_cash, &curve, settlement)?,
        paying
            .fd_valuation(&with_cash, &curve, settlement, ConvertibleFdGrid::default())?
            .clean_price,
        protected.clean_price(&with_cash, &curve, settlement)?
    );

    // --- bump greeks on the grid -------------------------------------
    let tenors = [1.0, 2.0, 3.0, 5.0];
    let greeks = convertible.fd_greeks(
        &ConvertibleMarket {
            spot: 48.0,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        },
        &curve,
        settlement,
        ConvertibleFdGrid::default(),
        &tenors,
    )?;
    println!("\ngreeks at spot 48 (Tsiveriotis-Fernandes, per 100 face):");
    println!(
        "  vega {:.4} per vol point, theta {:+.4} per day, rate dv01 {:.4}, spread dv01 {:.4}",
        greeks.vega, greeks.theta, greeks.rate_dv01, greeks.credit_dv01
    );
    let key_rates: Vec<String> = tenors
        .iter()
        .zip(&greeks.key_rate_dv01)
        .map(|(t, dv01)| format!("{t:.0}y {dv01:.4}"))
        .collect();
    println!("  key-rate dv01: {}", key_rates.join(", "));

    // --- contractual features ----------------------------------------
    // the same bond at spot 48 with the usual indenture extras added one
    // at a time: contingent conversion at 130% until six months before
    // maturity, a coupon make-whole through the put date on the soft
    // call, and a fundamental-change make-whole table with a 5%/yr
    // takeover intensity
    let market = ConvertibleMarket {
        spot: 48.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.02,
    };
    let base = convertible.clean_price(&market, &curve, settlement)?;
    println!("\ncontractual features at spot 48 (Tsiveriotis-Fernandes):");
    println!("{:<44} {:>10} {:>8}", "feature", "clean", "change");
    println!(
        "{:<44} {base:>10.4} {:>8}",
        "soft call 102 from 2028, put 100 in 2029", ""
    );
    let mut coco = convertible.clone();
    coco.contingent_conversion = Some(ContingentConversion {
        trigger: 65.0,
        until: date(2030, 11, 15),
    });
    let coco_price = coco.clean_price(&market, &curve, settlement)?;
    // no bite here: with the coupon above the dividends the holder never
    // converts below the trigger anyway, and the soft call sits on it
    println!(
        "{:<44} {coco_price:>10.4} {:>+8.4}",
        "+ contingent conversion at 130% (no bite)",
        coco_price - base
    );
    let mut made_whole = coco.clone();
    made_whole.coupon_make_whole = Some(CouponMakeWhole {
        until: date(2029, 5, 15),
        spread: 0.005,
    });
    let mw_price = made_whole.clean_price(&market, &curve, settlement)?;
    println!(
        "{:<44} {mw_price:>10.4} {:>+8.4}",
        "+ coupon make-whole to the put date, T+50",
        mw_price - coco_price
    );
    let mut protected = made_whole.clone();
    protected.fundamental_change = Some(FundamentalChangeMakeWhole {
        event_intensity: 0.05,
        stock_prices: vec![40.0, 50.0, 60.0, 80.0, 100.0],
        effective_dates: vec![date(2026, 5, 15), date(2028, 5, 15), date(2031, 5, 15)],
        additional_shares: vec![
            vec![5.0, 3.0, 2.0, 1.0, 0.0],
            vec![3.0, 2.0, 1.0, 0.5, 0.0],
            vec![0.0; 5],
        ],
    });
    let fc_price = protected.clean_price(&market, &curve, settlement)?;
    println!(
        "{:<44} {fc_price:>10.4} {:>+8.4}",
        "+ fundamental-change make-whole, 5%/yr",
        fc_price - mw_price
    );

    // --- mandatory convertible ---------------------------------------
    // a 6% three-year mandatory on the same stock: 25 shares at or below
    // 40, 20 shares at or above 50 (a 25% conversion premium), shares
    // worth 1000 in between; no principal, so the floor is the coupons
    let chassis = FixedRateBond::us_corporate(1000.0, 0.06, date(2026, 5, 15), date(2029, 5, 15))?;
    let mut mandatory = ConvertibleBond::new(chassis, 20.0)?;
    mandatory.mandatory = Some(MandatoryConversion { max_ratio: 25.0 });
    println!("\n6% 2029 mandatory, 25 shares below 40, 20 shares above 50:");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "spot", "clean", "coupons", "min parity", "max parity", "delta"
    );
    for spot in [30.0, 40.0, 45.0, 50.0, 60.0] {
        let market = ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        };
        let valuation =
            mandatory.fd_valuation(&market, &curve, settlement, ConvertibleFdGrid::default())?;
        let floor = mandatory.bond_floor(&market, &curve, settlement)?;
        println!(
            "{spot:>8.2} {:>10.4} {floor:>10.4} {:>10.2} {:>10.2} {:>9.4}",
            valuation.clean_price,
            mandatory.parity(spot),
            25.0 * spot * 100.0 / 1000.0,
            valuation.delta
        );
    }

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
