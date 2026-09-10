//! Tests of the tree engines, the implied solves and the contractual
//! features (both credit models).

use chrono::NaiveDate;

use super::*;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::utils::norm_cdf;

fn d(y: i32, m: u32, day: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, day).unwrap()
}

fn flat(rate: f64) -> YieldCurve {
    YieldCurve::flat(
        rate,
        d(2026, 8, 13),
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .unwrap()
}

/// A 2% five-year convertible on a 1000 face, 20 shares per bond
/// (conversion price 50).
fn convertible() -> ConvertibleBond {
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    ConvertibleBond::new(bond, 20.0).unwrap()
}

fn market(spot: f64) -> ConvertibleMarket {
    ConvertibleMarket {
        spot,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.02,
    }
}

#[test]
fn deep_out_of_the_money_collapses_to_the_risky_straight_bond() {
    // with the shares nearly worthless the cash part evolves
    // deterministically, so the tree must reproduce the analytic
    // risky bond almost exactly
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let market = market(0.01);
    let tree = cv.clean_price(&market, &curve, settlement).unwrap();
    let straight = cv.bond_floor(&market, &curve, settlement).unwrap();
    assert!((tree - straight).abs() < 1e-6, "{tree} vs {straight}");
}

#[test]
fn deep_in_the_money_trades_at_parity() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let spot = 250.0; // parity 500 vs redemption 100
    let dirty = cv.dirty_price(&market(spot), &curve, settlement).unwrap();
    let parity = cv.parity(spot);
    assert!(
        (dirty - parity) / parity < 0.01,
        "dirty {dirty} vs parity {parity}"
    );
    assert!(dirty >= parity - 1e-9, "conversion floor violated");
    // delta approaches the full ratio per 100 face
    let delta = cv.delta(&market(spot), &curve, settlement).unwrap();
    let full = cv.conversion_ratio * 100.0 / cv.bond.face_value;
    assert!(
        (delta - full).abs() < 0.05 * full,
        "delta {delta} vs {full}"
    );
}

#[test]
fn maturity_only_conversion_is_bond_plus_european_call() {
    // restrict conversion to maturity with zero credit spread: the
    // convertible = risk-free straight bond + ratio European calls
    // struck at redemption/ratio (Black-Scholes anchor)
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.convert_from = Some(d(2031, 5, 14));
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = ConvertibleMarket {
        spot: 48.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.0,
    };
    let tree = cv
        .dirty_price_with_steps(&m, &curve, settlement, 1600)
        .unwrap();

    // Black-Scholes call on the terminal package: strike is the full
    // redemption (face + final coupon) per share
    let dc = curve.day_count();
    let last = cv.bond.cashflows().last().unwrap().clone();
    let t = dc.year_fraction(settlement, last.payment_date);
    let strike = last.amount / cv.conversion_ratio;
    let df = curve.df_date(last.payment_date) / curve.df_date(settlement);
    let forward = m.spot * (-m.dividend_yield * t).exp() / df;
    let sd = m.volatility * t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
    let call = df * (forward * norm_cdf(d1) - strike * norm_cdf(d1 - sd));
    let straight = cv.bond.dirty_price_from_curve(&curve, settlement).unwrap();
    let expected = straight + cv.conversion_ratio * call * 100.0 / cv.bond.face_value;
    assert!((tree - expected).abs() < 0.15, "{tree} vs {expected}");
}

#[test]
fn price_sits_above_both_floors_and_orders_in_vol_and_spread() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 45.0, 55.0, 70.0] {
        let m = market(spot);
        let clean = cv.clean_price(&m, &curve, settlement).unwrap();
        let floor = cv.bond_floor(&m, &curve, settlement).unwrap();
        let parity = cv.parity(spot);
        assert!(
            clean >= floor - 0.05,
            "spot {spot}: {clean} vs floor {floor}"
        );
        assert!(
            clean >= parity - cv.bond.accrued_interest(settlement).unwrap() - 0.05,
            "spot {spot}: {clean} vs parity {parity}"
        );
    }
    // vega and credit ordering at the money
    let base = cv.clean_price(&market(48.0), &curve, settlement).unwrap();
    let hot = ConvertibleMarket {
        volatility: 0.45,
        ..market(48.0)
    };
    assert!(cv.clean_price(&hot, &curve, settlement).unwrap() > base);
    let wide = ConvertibleMarket {
        credit_spread: 0.05,
        ..market(48.0)
    };
    assert!(cv.clean_price(&wide, &curve, settlement).unwrap() < base);
}

#[test]
fn calls_cap_puts_floor_and_the_soft_trigger_softens() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let base = convertible();
    let free = base.clean_price(&m, &curve, settlement).unwrap();

    let mut hard_called = base.clone();
    hard_called.calls = vec![CallOption {
        call_date: d(2028, 5, 15),
        call_price: 102.0,
    }];
    let called = hard_called.clean_price(&m, &curve, settlement).unwrap();
    assert!(called < free, "{called} vs {free}");

    // a high soft-call trigger makes the call harder to exercise:
    // price between hard-called and call-free
    let mut soft_called = hard_called.clone();
    soft_called.soft_call_trigger = Some(65.0); // 130% of conversion price
    let softened = soft_called.clean_price(&m, &curve, settlement).unwrap();
    assert!(
        called < softened && softened <= free + 1e-9,
        "{called} < {softened} <= {free}"
    );

    let mut puttable = base.clone();
    puttable.puts = vec![PutOption {
        put_date: d(2029, 5, 15),
        put_price: 100.0,
    }];
    let put_price = puttable.clean_price(&m, &curve, settlement).unwrap();
    assert!(put_price > free, "{put_price} vs {free}");
}

#[test]
fn implied_credit_spread_round_trips() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let clean = cv.clean_price(&m, &curve, settlement).unwrap();
    let implied = cv
        .implied_credit_spread(clean, &m, &curve, settlement)
        .unwrap();
    assert!(
        (implied - m.credit_spread).abs() < 1e-5,
        "implied {implied}"
    );
}

#[test]
fn parity_premium_and_validation() {
    let cv = convertible();
    assert!((cv.conversion_price() - 50.0).abs() < 1e-12);
    assert!((cv.parity(48.0) - 96.0).abs() < 1e-12);
    let premium = cv.conversion_premium(105.6, 48.0).unwrap();
    assert!((premium - 0.10).abs() < 1e-12);
    // validation
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    assert!(ConvertibleBond::new(bond.clone(), 0.0).is_err());
    let cv = ConvertibleBond::new(bond, 20.0).unwrap();
    let curve = flat(0.04);
    let bad = ConvertibleMarket {
        spot: -1.0,
        volatility: 0.3,
        dividend_yield: 0.0,
        credit_spread: 0.0,
    };
    assert!(cv.dirty_price(&bad, &curve, d(2026, 8, 14)).is_err());
    let m = market(48.0);
    assert!(cv
        .dirty_price_with_steps(&m, &curve, d(2026, 8, 14), 5)
        .is_err());
    assert!(cv
        .implied_credit_spread(-10.0, &m, &curve, d(2026, 8, 14))
        .is_err());
}
#[test]
fn implied_volatility_round_trips_under_both_models() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let clean = cv.clean_price(&m, &curve, settlement).unwrap();
    // the solve starts from a wrong vol and must find the right one
    let guess = ConvertibleMarket {
        volatility: 0.5,
        ..m
    };
    let implied = cv
        .implied_volatility(clean, &guess, &curve, settlement)
        .unwrap();
    assert!((implied - m.volatility).abs() < 1e-5, "implied {implied}");

    let jm = jtd_market(48.0);
    let clean = cv.clean_price(&jm, &curve, settlement).unwrap();
    let guess = JumpToDefaultMarket {
        volatility: 0.15,
        ..jm
    };
    let implied = cv
        .implied_volatility(clean, &guess, &curve, settlement)
        .unwrap();
    assert!((implied - jm.volatility).abs() < 1e-5, "implied {implied}");

    // a quote no volatility can reach is an error, not a number
    assert!(cv
        .implied_volatility(clean + 100.0, &m, &curve, settlement)
        .is_err());
    assert!(cv.implied_volatility(-1.0, &m, &curve, settlement).is_err());
}

/// Cross-check against QuantLib 1.43's `BinomialConvertibleEngine`
/// (CRR, 3200 steps) on this bond: 2% semiannual 30/360, 2026-05-15 to
/// 2031-05-15, two shares per 100 face, flat 4% continuous curve, 30%
/// vol, 1% dividend yield, settlement 2026-08-14, clean call and put
/// prices. At zero credit spread the two engines share the model and
/// agree to a thousandth per 100 (QuantLib discounts each step at
/// `1 / (1 + r dt)`, we at `exp(-r dt)`). With a spread they differ by
/// design: QuantLib uses Hull's blended-discount variant of
/// Tsiveriotis-Fernandes (the node discounts at a conversion-
/// probability-weighted mix of the risk-free and risky rates), this
/// library the paper's equity/cash split, so those numbers are not
/// asserted. QuantLib's soft-call trigger is a multiple of the
/// conversion price (1.3 here), ours a share price (65).
#[test]
fn zero_spread_prices_match_quantlib() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let market = |spot: f64| ConvertibleMarket {
        spot,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.0,
    };
    let price = |cv: &ConvertibleBond, spot: f64| {
        cv.dirty_price_with_steps(&market(spot), &curve, settlement, 3200)
            .unwrap()
    };
    let bullet = convertible();
    for (spot, quantlib) in [(30.0, 99.4954), (48.0, 118.6952), (70.0, 151.7988)] {
        let ours = price(&bullet, spot);
        assert!(
            (ours - quantlib).abs() < 0.002,
            "spot {spot}: {ours} vs {quantlib}"
        );
    }
    let mut european = convertible();
    european.convert_from = Some(d(2031, 5, 14));
    assert!((price(&european, 48.0) - 118.4142).abs() < 0.002);

    let mut hard = convertible();
    hard.calls = vec![CallOption {
        call_date: d(2028, 5, 15),
        call_price: 102.0,
    }];
    hard.puts = vec![PutOption {
        put_date: d(2029, 5, 15),
        put_price: 100.0,
    }];
    assert!(
        (price(&hard, 70.0) - 145.6464).abs() < 0.002,
        "{}",
        price(&hard, 70.0)
    );
    let mut put_only = convertible();
    put_only.puts = hard.puts.clone();
    assert!((price(&put_only, 70.0) - 151.9594).abs() < 0.002);
    let mut soft = hard.clone();
    soft.soft_call_trigger = Some(65.0); // 130% of the conversion price
    assert!(
        (price(&soft, 70.0) - 149.8032).abs() < 0.002,
        "{}",
        price(&soft, 70.0)
    );

    // the busted limit at 200bp: both engines reprice the risky
    // straight bond, QuantLib to within its step-compounding error
    let busted = ConvertibleMarket {
        credit_spread: 0.02,
        ..market(0.01)
    };
    let ours = bullet
        .dirty_price_with_steps(&busted, &curve, settlement, 3200)
        .unwrap();
    assert!((ours - 83.8236).abs() < 0.005, "{ours}");
}

// --- cash dividends ------------------------------------------------

#[test]
fn cash_dividends_lower_the_price_and_only_before_maturity() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let price = |cv: &ConvertibleBond| cv.dirty_price(&m, &curve, settlement).unwrap();
    let free = price(&convertible());
    let mut paying = convertible();
    paying.cash_dividends = vec![
        CashDividend {
            ex_date: d(2027, 3, 1),
            amount: 0.5,
        },
        CashDividend {
            ex_date: d(2029, 3, 1),
            amount: 0.5,
        },
    ];
    let with = price(&paying);
    assert!(with < free, "{with} vs {free}");
    let mut bigger = paying.clone();
    bigger.cash_dividends[0].amount = 2.0;
    assert!(price(&bigger) < with);
    // after maturity, or before settlement: no effect
    let mut late = convertible();
    late.cash_dividends = vec![CashDividend {
        ex_date: d(2031, 6, 1),
        amount: 5.0,
    }];
    assert!((price(&late) - free).abs() < 1e-12);
    let mut early = convertible();
    early.cash_dividends = vec![CashDividend {
        ex_date: d(2026, 8, 1),
        amount: 5.0,
    }];
    assert!((price(&early) - free).abs() < 1e-12);
    // jump to default and the grid see the dividend too
    let jm = jtd_market(48.0);
    assert!(
        paying.dirty_price(&jm, &curve, settlement).unwrap()
            < convertible().dirty_price(&jm, &curve, settlement).unwrap()
    );
    let mut bad = convertible();
    bad.cash_dividends = vec![CashDividend {
        ex_date: d(2027, 3, 1),
        amount: -1.0,
    }];
    assert!(bad.dirty_price(&m, &curve, settlement).is_err());
}

#[test]
fn deep_in_the_money_conversion_forgoes_the_discounted_dividends() {
    // with conversion at maturity only, no yield and no credit, a bond
    // far in the money is the coupons (bar the final one, forfeited on
    // conversion) plus ratio times the share price less the present
    // value of the dividends the holder never sees; the spot is far
    // enough out that the redemption floor is worthless
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.convert_from = Some(d(2031, 5, 14));
    cv.cash_dividends = vec![
        CashDividend {
            ex_date: d(2027, 3, 1),
            amount: 5.0,
        },
        CashDividend {
            ex_date: d(2029, 3, 1),
            amount: 5.0,
        },
    ];
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = ConvertibleMarket {
        spot: 1000.0,
        volatility: 0.30,
        dividend_yield: 0.0,
        credit_spread: 0.0,
    };
    let tree = cv
        .dirty_price_with_steps(&m, &curve, settlement, 3200)
        .unwrap();
    let df = |date: NaiveDate| curve.df_date(date) / curve.df_date(settlement);
    let last = cv.bond.cashflows().last().unwrap().clone();
    let coupons = cv.bond.dirty_price_from_curve(&curve, settlement).unwrap()
        - last.amount / 10.0 * df(last.payment_date);
    let dividends: f64 = cv
        .cash_dividends
        .iter()
        .map(|d| d.amount * df(d.ex_date))
        .sum();
    let expected = coupons + 2.0 * (m.spot - dividends);
    assert!((tree - expected).abs() < 0.05, "{tree} vs {expected}");
}

// --- jump to default -----------------------------------------------

fn jtd_market(spot: f64) -> JumpToDefaultMarket {
    JumpToDefaultMarket {
        spot,
        volatility: 0.30,
        dividend_yield: 0.01,
        borrow_cost: 0.005,
        hazard_rate: 0.03,
        recovery_rate: 0.40,
    }
}

#[test]
fn jtd_busted_convertible_reprices_the_hazard_rate_bond() {
    // with the shares nearly worthless only the survival-weighted
    // coupons and the recovery leg remain, which is exactly the
    // credit module's risky bond (recovery paid at the end of the
    // default period); the residual is the step-boundary
    // misalignment of the default mass against the coupon periods
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = jtd_market(0.01);
    let tree = cv.clean_price(&m, &curve, settlement).unwrap();
    let floor = cv.bond_floor(&m, &curve, settlement).unwrap();
    assert!((tree - floor).abs() < 0.01, "{tree} vs {floor}");
    // and well below the risk-free straight bond
    let riskfree = cv.bond.clean_price_from_curve(&curve, settlement).unwrap();
    assert!(floor < riskfree - 5.0, "{floor} vs {riskfree}");
}

#[test]
fn jtd_without_hazard_is_tsiveriotis_fernandes_without_spread() {
    // no default and no borrow: both engines run the same tree
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 48.0, 70.0] {
        let tf = ConvertibleMarket {
            credit_spread: 0.0,
            ..market(spot)
        };
        let jtd = JumpToDefaultMarket {
            hazard_rate: 0.0,
            borrow_cost: 0.0,
            ..jtd_market(spot)
        };
        let a = cv.dirty_price(&tf, &curve, settlement).unwrap();
        let b = cv.dirty_price(&jtd, &curve, settlement).unwrap();
        assert!((a - b).abs() < 1e-9, "spot {spot}: TF {a} vs JTD {b}");
    }
}

#[test]
fn jtd_deep_in_the_money_trades_at_parity() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let spot = 250.0; // parity 500 vs redemption 100
    let m = jtd_market(spot);
    let dirty = cv.dirty_price(&m, &curve, settlement).unwrap();
    let parity = cv.parity(spot);
    assert!(
        (dirty - parity) / parity < 0.01,
        "dirty {dirty} vs parity {parity}"
    );
    assert!(dirty >= parity - 1e-9, "conversion floor violated");
    let delta = cv.delta(&m, &curve, settlement).unwrap();
    let full = cv.conversion_ratio * 100.0 / cv.bond.face_value;
    assert!(
        (delta - full).abs() < 0.05 * full,
        "delta {delta} vs {full}"
    );
}

#[test]
fn jtd_maturity_only_conversion_is_risky_bond_plus_jump_to_default_call() {
    // restrict conversion to maturity: the convertible = hazard-rate
    // risky bond + ratio European calls struck at redemption/ratio
    // that pay only on survival. On the survival branch the stock is
    // lognormal with drift r + lambda - q - b, so the call is
    // Black-Scholes on the survival-conditional forward, discounted
    // at r + lambda (the model's signature)
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.convert_from = Some(d(2031, 5, 14));
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = jtd_market(48.0);
    let tree = cv
        .dirty_price_with_steps(&m, &curve, settlement, 1600)
        .unwrap();

    let dc = curve.day_count();
    let last = cv.bond.cashflows().last().unwrap().clone();
    let t = dc.year_fraction(settlement, last.payment_date);
    let strike = last.amount / cv.conversion_ratio;
    let df = curve.df_date(last.payment_date) / curve.df_date(settlement);
    let survival = (-m.hazard_rate * t).exp();
    let forward = m.spot * (-(m.dividend_yield + m.borrow_cost) * t).exp() / (df * survival);
    let sd = m.volatility * t.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
    let call = df * survival * (forward * norm_cdf(d1) - strike * norm_cdf(d1 - sd));
    let straight = cv
        .bond
        .risky_dirty_price(&curve, m.hazard_rate, m.recovery_rate, settlement)
        .unwrap();
    let expected = straight + cv.conversion_ratio * call * 100.0 / cv.bond.face_value;
    assert!((tree - expected).abs() < 0.15, "{tree} vs {expected}");
}

#[test]
fn jtd_price_sits_above_both_floors_and_orders_in_the_inputs() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let accrued = cv.bond.accrued_interest(settlement).unwrap();
    for spot in [30.0, 45.0, 55.0, 70.0] {
        let m = jtd_market(spot);
        let clean = cv.clean_price(&m, &curve, settlement).unwrap();
        let floor = cv.bond_floor(&m, &curve, settlement).unwrap();
        let parity = cv.parity(spot);
        assert!(
            clean >= floor - 0.05,
            "spot {spot}: {clean} vs floor {floor}"
        );
        assert!(
            clean >= parity - accrued - 0.05,
            "spot {spot}: {clean} vs parity {parity}"
        );
    }
    let base_market = jtd_market(48.0);
    let base = cv.clean_price(&base_market, &curve, settlement).unwrap();
    let price = |m: JumpToDefaultMarket| cv.clean_price(&m, &curve, settlement).unwrap();
    // more default risk cheapens every claim
    assert!(
        price(JumpToDefaultMarket {
            hazard_rate: 0.08,
            ..base_market
        }) < base
    );
    // a better recovery is worth more
    assert!(
        price(JumpToDefaultMarket {
            recovery_rate: 0.7,
            ..base_market
        }) > base
    );
    // vega
    assert!(
        price(JumpToDefaultMarket {
            volatility: 0.45,
            ..base_market
        }) > base
    );
    // borrow lowers the forward, hence the conversion value
    assert!(
        price(JumpToDefaultMarket {
            borrow_cost: 0.03,
            ..base_market
        }) < base
    );
    // calls cap, puts floor
    let mut called = cv.clone();
    called.calls = vec![CallOption {
        call_date: d(2028, 5, 15),
        call_price: 102.0,
    }];
    let called_price = called
        .clean_price(&base_market, &curve, settlement)
        .unwrap();
    assert!(called_price < base, "{called_price} vs {base}");
    let mut soft = called.clone();
    soft.soft_call_trigger = Some(65.0);
    let softened = soft.clean_price(&base_market, &curve, settlement).unwrap();
    assert!(
        called_price < softened && softened <= base + 1e-9,
        "{called_price} < {softened} <= {base}"
    );
    let mut puttable = cv.clone();
    puttable.puts = vec![PutOption {
        put_date: d(2029, 5, 15),
        put_price: 100.0,
    }];
    let put_price = puttable
        .clean_price(&base_market, &curve, settlement)
        .unwrap();
    assert!(put_price > base, "{put_price} vs {base}");
}

#[test]
fn jtd_implied_hazard_rate_round_trips() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = jtd_market(48.0);
    let clean = cv.clean_price(&m, &curve, settlement).unwrap();
    let implied = cv
        .implied_hazard_rate(clean, &m, &curve, settlement)
        .unwrap();
    assert!((implied - m.hazard_rate).abs() < 1e-5, "implied {implied}");
    // a quote at the zero-hazard value reads as no credit risk
    let riskless = JumpToDefaultMarket {
        hazard_rate: 0.0,
        ..m
    };
    let top = cv.clean_price(&riskless, &curve, settlement).unwrap();
    let implied = cv.implied_hazard_rate(top, &m, &curve, settlement).unwrap();
    assert!(implied.abs() < 1e-6, "implied {implied}");
    // above the zero-hazard value no hazard reproduces the quote;
    // below the model's minimum (the recovery leg takes over
    // beyond a 15% or so hazard here) neither does
    assert!(matches!(
        cv.implied_hazard_rate(top + 1.0, &m, &curve, settlement),
        Err(RustyQLibError::CalibrationFailed { .. })
    ));
    assert!(matches!(
        cv.implied_hazard_rate(100.0, &m, &curve, settlement),
        Err(RustyQLibError::CalibrationFailed { .. })
    ));
}

#[test]
fn jtd_validation() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let good = jtd_market(48.0);
    assert!(cv.dirty_price(&good, &curve, settlement).is_ok());
    let bad_hazard = JumpToDefaultMarket {
        hazard_rate: -0.01,
        ..good
    };
    assert!(cv.dirty_price(&bad_hazard, &curve, settlement).is_err());
    let bad_recovery = JumpToDefaultMarket {
        recovery_rate: 1.5,
        ..good
    };
    assert!(cv.dirty_price(&bad_recovery, &curve, settlement).is_err());
    let bad_borrow = JumpToDefaultMarket {
        borrow_cost: f64::NAN,
        ..good
    };
    assert!(cv.dirty_price(&bad_borrow, &curve, settlement).is_err());
    let bad_spot = JumpToDefaultMarket { spot: 0.0, ..good };
    assert!(cv.dirty_price(&bad_spot, &curve, settlement).is_err());
    assert!(cv
        .dirty_price_with_steps(&good, &curve, settlement, 5)
        .is_err());
    assert!(cv
        .implied_hazard_rate(-10.0, &good, &curve, settlement)
        .is_err());
}

// --- contractual features ------------------------------------------

fn make_whole_table() -> FundamentalChangeMakeWhole {
    FundamentalChangeMakeWhole {
        event_intensity: 0.05,
        stock_prices: vec![40.0, 50.0, 60.0, 80.0, 100.0],
        effective_dates: vec![d(2026, 5, 15), d(2028, 5, 15), d(2031, 5, 15)],
        additional_shares: vec![
            vec![5.0, 3.0, 2.0, 1.0, 0.0],
            vec![3.0, 2.0, 1.0, 0.5, 0.0],
            vec![0.0; 5],
        ],
    }
}

#[test]
fn contingent_conversion_gates_the_holder() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let price = |cv: &ConvertibleBond| cv.clean_price(&m, &curve, settlement).unwrap();
    let free = price(&convertible());
    // a trigger below any reachable price changes nothing
    let mut low = convertible();
    low.contingent_conversion = Some(ContingentConversion {
        trigger: 1e-6,
        until: d(2031, 5, 15),
    });
    assert!((price(&low) - free).abs() < 1e-9);
    // 130% of the conversion price: less than unconditional, more
    // than conversion at maturity only
    let mut coco = convertible();
    coco.contingent_conversion = Some(ContingentConversion {
        trigger: 65.0,
        until: d(2031, 5, 15),
    });
    let gated = price(&coco);
    let mut at_maturity = convertible();
    at_maturity.convert_from = Some(d(2031, 5, 14));
    let maturity_only = price(&at_maturity);
    assert!(
        maturity_only < gated && gated < free,
        "{maturity_only} < {gated} < {free}"
    );
    // the same ordering under jump to default
    let jm = jtd_market(48.0);
    let gated_jtd = coco.clean_price(&jm, &curve, settlement).unwrap();
    let free_jtd = convertible().clean_price(&jm, &curve, settlement).unwrap();
    let maturity_jtd = at_maturity.clean_price(&jm, &curve, settlement).unwrap();
    assert!(maturity_jtd < gated_jtd && gated_jtd < free_jtd);
    // an unreachable trigger lifting six months before maturity is a
    // conversion window opening on that date
    let mut late = convertible();
    late.contingent_conversion = Some(ContingentConversion {
        trigger: 1e9,
        until: d(2030, 11, 15),
    });
    let mut window = convertible();
    window.convert_from = Some(d(2030, 11, 15));
    assert!((price(&late) - price(&window)).abs() < 1e-9);
}

#[test]
fn coupon_make_whole_compensates_a_provisional_call() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let price = |cv: &ConvertibleBond, m: &ConvertibleMarket| {
        cv.clean_price(m, &curve, settlement).unwrap()
    };
    let mut called = convertible();
    called.calls = vec![CallOption {
        call_date: d(2028, 5, 15),
        call_price: 100.0,
    }];
    called.soft_call_trigger = Some(65.0);
    let free = price(&convertible(), &m);
    let plain_call = price(&called, &m);
    // two more coupons made whole at T+50
    let mut made_whole = called.clone();
    made_whole.coupon_make_whole = Some(CouponMakeWhole {
        until: d(2029, 5, 15),
        spread: 0.005,
    });
    let mw = price(&made_whole, &m);
    assert!(
        plain_call < mw && mw <= free + 1e-9,
        "{plain_call} < {mw} <= {free}"
    );
    // a make-whole ending on the call date is empty
    let mut empty = called.clone();
    empty.coupon_make_whole = Some(CouponMakeWhole {
        until: d(2028, 5, 15),
        spread: 0.0,
    });
    assert!((price(&empty, &m) - plain_call).abs() < 1e-9);
    // deep in the money with no dividends the holder never converts
    // early, so the call is certain and answered by conversion, and
    // the make-whole (two 1-point coupons, discounted) is paid on
    // top of the shares
    let deep = ConvertibleMarket {
        dividend_yield: 0.0,
        ..market(250.0)
    };
    let gain = price(&made_whole, &deep) - price(&called, &deep);
    assert!(gain > 1.6 && gain < 2.0, "{gain}");
    // jump to default sees the same gain
    // (and no borrow either: a borrow cost, like a dividend, is a
    // carry the holder forgoes by staying in the bond)
    let jm = JumpToDefaultMarket {
        dividend_yield: 0.0,
        borrow_cost: 0.0,
        ..jtd_market(250.0)
    };
    let gain_jtd = made_whole.clean_price(&jm, &curve, settlement).unwrap()
        - called.clean_price(&jm, &curve, settlement).unwrap();
    assert!(gain_jtd > 1.6 && gain_jtd < 2.0, "{gain_jtd}");
}

#[test]
fn fundamental_change_make_whole_adds_value() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let price = |cv: &ConvertibleBond, spot: f64| {
        cv.clean_price(&market(spot), &curve, settlement).unwrap()
    };
    let free = price(&convertible(), 48.0);
    let mut with = convertible();
    with.fundamental_change = Some(make_whole_table());
    let protected = price(&with, 48.0);
    assert!(protected > free, "{protected} vs {free}");
    // no events, no effect
    let mut quiet = with.clone();
    quiet.fundamental_change.as_mut().unwrap().event_intensity = 0.0;
    assert!((price(&quiet, 48.0) - free).abs() < 1e-9);
    // more events, more value
    let mut busy = with.clone();
    busy.fundamental_change.as_mut().unwrap().event_intensity = 0.10;
    assert!(price(&busy, 48.0) > protected);
    // an empty table on a deep-in-the-money bond changes nothing:
    // the holder already sits above par and parity
    let mut empty = with.clone();
    let fc = empty.fundamental_change.as_mut().unwrap();
    fc.additional_shares = vec![vec![0.0; 5]; 3];
    assert!((price(&empty, 250.0) - price(&convertible(), 250.0)).abs() < 1e-6);
    // but a busted bond gains the par put
    assert!(price(&empty, 5.0) > price(&convertible(), 5.0) + 0.5);
    // jump to default agrees on the ordering
    let jm = jtd_market(48.0);
    assert!(
        with.clean_price(&jm, &curve, settlement).unwrap()
            > convertible().clean_price(&jm, &curve, settlement).unwrap()
    );
}

#[test]
fn feature_validation() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(48.0);
    let mut bad = convertible();
    bad.contingent_conversion = Some(ContingentConversion {
        trigger: -1.0,
        until: d(2031, 5, 15),
    });
    assert!(bad.dirty_price(&m, &curve, settlement).is_err());
    let mut bad = convertible();
    bad.coupon_make_whole = Some(CouponMakeWhole {
        until: d(2029, 5, 15),
        spread: f64::NAN,
    });
    assert!(bad.dirty_price(&m, &curve, settlement).is_err());
    let table = make_whole_table();
    let mut ragged = table.clone();
    ragged.additional_shares[1].pop();
    let mut unsorted = table.clone();
    unsorted.stock_prices.swap(0, 1);
    let mut negative = table.clone();
    negative.additional_shares[0][0] = -1.0;
    let mut bad_intensity = table.clone();
    bad_intensity.event_intensity = -0.1;
    for fc in [ragged, unsorted, negative, bad_intensity] {
        let mut bad = convertible();
        bad.fundamental_change = Some(fc);
        assert!(bad.dirty_price(&m, &curve, settlement).is_err());
    }
    let mut good = convertible();
    good.fundamental_change = Some(table);
    assert!(good.dirty_price(&m, &curve, settlement).is_ok());
}

/// A 6% three-year mandatory on a 1000 face: 25 shares at or below
/// 40, 20 shares at or above 50, 1000 worth of shares between.
fn mandatory() -> ConvertibleBond {
    let bond = FixedRateBond::us_corporate(1000.0, 0.06, d(2026, 5, 15), d(2029, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.mandatory = Some(MandatoryConversion { max_ratio: 25.0 });
    cv
}

#[test]
fn mandatory_terminal_schedule_and_floor() {
    let cv = mandatory();
    assert!((cv.maturity_shares(30.0) - 25.0).abs() < 1e-12);
    assert!((cv.maturity_shares(40.0) - 25.0).abs() < 1e-12);
    assert!((cv.maturity_shares(45.0) * 45.0 - 1000.0).abs() < 1e-9);
    assert!((cv.maturity_shares(50.0) - 20.0).abs() < 1e-12);
    assert!((cv.maturity_shares(80.0) - 20.0).abs() < 1e-12);
    // the floor is the coupons alone: chassis less the principal
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = market(45.0);
    let floor = cv.bond_floor(&m, &curve, settlement).unwrap();
    let chassis = cv
        .bond
        .clean_price_from_curve_with_spread(&curve, m.credit_spread, settlement)
        .unwrap();
    let t = curve
        .day_count()
        .year_fraction(settlement, cv.bond.cashflows().last().unwrap().payment_date);
    let principal = 100.0 * (-(0.04 + m.credit_spread) * t).exp();
    assert!((floor - (chassis - principal)).abs() < 1e-6, "{floor}");
    // and under jump to default the survival-weighted principal
    let jm = jtd_market(45.0);
    let floor_jtd = cv.bond_floor(&jm, &curve, settlement).unwrap();
    let risky = cv
        .bond
        .risky_clean_price(&curve, jm.hazard_rate, jm.recovery_rate, settlement)
        .unwrap();
    let principal = 100.0 * (-(0.04 + jm.hazard_rate) * t).exp();
    assert!(
        (floor_jtd - (risky - principal)).abs() < 1e-6,
        "{floor_jtd}"
    );
}

#[test]
fn mandatory_is_the_stock_less_a_call_spread_plus_coupons() {
    // conversion at maturity only, no credit: max_ratio shares less
    // max_ratio calls at the lower threshold plus min_ratio calls at
    // the upper one, plus the risk-free coupons
    let mut cv = mandatory();
    cv.convert_from = Some(d(2029, 5, 14));
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = ConvertibleMarket {
        spot: 45.0,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.0,
    };
    let tree = cv
        .dirty_price_with_steps(&m, &curve, settlement, 1600)
        .unwrap();

    let last = cv.bond.cashflows().last().unwrap().clone();
    let t = curve
        .day_count()
        .year_fraction(settlement, last.payment_date);
    let df = curve.df_date(last.payment_date) / curve.df_date(settlement);
    let forward = m.spot * (-m.dividend_yield * t).exp() / df;
    let sd = m.volatility * t.sqrt();
    let call = |strike: f64| {
        let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
        df * (forward * norm_cdf(d1) - strike * norm_cdf(d1 - sd))
    };
    let (max_ratio, min_ratio) = (25.0, 20.0);
    let equity = max_ratio * (df * forward - call(40.0)) + min_ratio * call(50.0);
    let coupons = cv.bond.dirty_price_from_curve(&curve, settlement).unwrap() - 100.0 * df;
    let expected = coupons + equity * 100.0 / cv.bond.face_value;
    assert!((tree - expected).abs() < 0.15, "{tree} vs {expected}");
}

#[test]
fn mandatory_limits_and_ordering_under_both_models() {
    let cv = mandatory();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let tf = |spot: f64| cv.clean_price(&market(spot), &curve, settlement).unwrap();
    let jtd = |spot: f64| {
        cv.clean_price(&jtd_market(spot), &curve, settlement)
            .unwrap()
    };
    // increasing in the share price
    for price in [&tf as &dyn Fn(f64) -> f64, &jtd] {
        let (low, mid_low, mid_high, high) = (price(30.0), price(40.0), price(50.0), price(60.0));
        assert!(low < mid_low && mid_low < mid_high && mid_high < high);
        // far above the upper threshold: the minimum ratio's parity
        // plus the coupons, never below parity at that ratio (the
        // clean price sits below the dirty by the accrued)
        let deep = price(500.0);
        let accrued = cv.bond.accrued_interest(settlement).unwrap();
        assert!(deep > cv.parity(500.0) - accrued - 0.05, "{deep}");
        // far below the lower threshold: the maximum ratio's shares
        // (forward) plus the coupons, well below par
        let busted = price(4.0);
        assert!(
            busted < 50.0 && busted > 25.0 * 4.0 * 0.9 * 100.0 / 1000.0,
            "{busted}"
        );
    }
    // the coupons are credit-risky: wider credit, lower price
    let wide = ConvertibleMarket {
        credit_spread: 0.05,
        ..market(45.0)
    };
    assert!(cv.clean_price(&wide, &curve, settlement).unwrap() < tf(45.0));
    // the dead zone shows only near maturity, when the diffusion no
    // longer smooths it away: a month out the slope between the
    // thresholds is well below the slopes outside them
    let late = d(2029, 4, 14);
    let near = |spot: f64| cv.clean_price(&market(spot), &curve, late).unwrap();
    let (low, mid_low, mid_high, high) = (near(30.0), near(40.0), near(50.0), near(60.0));
    let outer = ((mid_low - low) + (high - mid_high)) / 2.0;
    assert!(
        mid_high - mid_low < 0.6 * outer,
        "{low} {mid_low} {mid_high} {high}"
    );
    // validation: the maximum ratio must exceed the minimum
    let mut bad = mandatory();
    bad.mandatory = Some(MandatoryConversion { max_ratio: 20.0 });
    assert!(bad.dirty_price(&market(45.0), &curve, settlement).is_err());
}
