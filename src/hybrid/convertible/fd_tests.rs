//! Tests of the finite-difference engine: agreement with the tree,
//! the grid greeks, the bump greeks and the pluggable volatility.

use chrono::NaiveDate;

use super::*;
use crate::core::curves::{Compounding, RateShift, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::utils::norm_cdf;
use crate::core::vols::VolSurface;

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

/// The same bond with the example's soft call and put.
fn structured() -> ConvertibleBond {
    let mut cv = convertible();
    cv.calls = vec![CallOption {
        call_date: d(2028, 5, 15),
        call_price: 102.0,
    }];
    cv.soft_call_trigger = Some(65.0);
    cv.puts = vec![PutOption {
        put_date: d(2029, 5, 15),
        put_price: 100.0,
    }];
    cv
}

fn tf_market(spot: f64) -> ConvertibleMarket {
    ConvertibleMarket {
        spot,
        volatility: 0.30,
        dividend_yield: 0.01,
        credit_spread: 0.02,
    }
}

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

const GRID: ConvertibleFdGrid = ConvertibleFdGrid {
    time_steps: 400,
    space_steps: 400,
    grid_stdevs: 5.0,
};

/// The refined grid used against a 1600-step tree for the
/// structured bond, whose soft-call trigger is a discontinuity in
/// the share price that both engines resolve only slowly.
const FINE: ConvertibleFdGrid = ConvertibleFdGrid {
    time_steps: 800,
    space_steps: 800,
    grid_stdevs: 5.0,
};

#[test]
fn tsiveriotis_fernandes_agrees_with_the_tree() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for (cv, grid, steps) in [(convertible(), GRID, 800), (structured(), FINE, 1600)] {
        for spot in [30.0, 48.0, 70.0] {
            let m = tf_market(spot);
            let fd = cv.fd_valuation(&m, &curve, settlement, grid).unwrap();
            let tree = cv
                .dirty_price_with_steps(&m, &curve, settlement, steps)
                .unwrap();
            assert!(
                (fd.dirty_price - tree).abs() < 0.15,
                "spot {spot}: fd {} vs tree {tree}",
                fd.dirty_price
            );
            // the tree's bump delta carries lattice noise of a few
            // hundredths, and near the soft-call trigger the bumps
            // straddle a discontinuity, so only the plain bond's
            // delta is a usable reference
            if cv.calls.is_empty() {
                let delta = cv.delta(&m, &curve, settlement).unwrap();
                assert!(
                    (fd.delta - delta).abs() < 0.15,
                    "spot {spot}: fd delta {} vs tree {delta}",
                    fd.delta
                );
            }
        }
    }
}

#[test]
fn jump_to_default_agrees_with_the_tree() {
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for (cv, grid, steps) in [(convertible(), GRID, 800), (structured(), FINE, 1600)] {
        for spot in [30.0, 48.0, 70.0] {
            let m = jtd_market(spot);
            let fd = cv.fd_valuation(&m, &curve, settlement, grid).unwrap();
            let tree = cv
                .dirty_price_with_steps(&m, &curve, settlement, steps)
                .unwrap();
            assert!(
                (fd.dirty_price - tree).abs() < 0.15,
                "spot {spot}: fd {} vs tree {tree}",
                fd.dirty_price
            );
            if cv.calls.is_empty() {
                let delta = cv.delta(&m, &curve, settlement).unwrap();
                assert!(
                    (fd.delta - delta).abs() < 0.15,
                    "spot {spot}: fd delta {} vs tree {delta}",
                    fd.delta
                );
            }
        }
    }
}

#[test]
fn grid_delta_matches_a_bumped_solve() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 48.0, 70.0] {
        let m = jtd_market(spot);
        let fd = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
        let bump = 0.01 * spot;
        let price = |s: f64| {
            cv.fd_valuation(
                &JumpToDefaultMarket { spot: s, ..m },
                &curve,
                settlement,
                GRID,
            )
            .unwrap()
            .dirty_price
        };
        let bumped = (price(spot + bump) - price(spot - bump)) / (2.0 * bump);
        assert!(
            (fd.delta - bumped).abs() < 0.01 * bumped,
            "spot {spot}: grid delta {} vs bumped {bumped}",
            fd.delta
        );
    }
}

/// The structured bond with every contractual extra switched on.
fn fully_featured() -> ConvertibleBond {
    let mut cv = structured();
    cv.contingent_conversion = Some(ContingentConversion {
        trigger: 60.0,
        until: d(2030, 11, 15),
    });
    cv.coupon_make_whole = Some(CouponMakeWhole {
        until: d(2029, 5, 15),
        spread: 0.005,
    });
    cv.fundamental_change = Some(FundamentalChangeMakeWhole {
        event_intensity: 0.05,
        stock_prices: vec![40.0, 50.0, 60.0, 80.0, 100.0],
        effective_dates: vec![d(2026, 5, 15), d(2028, 5, 15), d(2031, 5, 15)],
        additional_shares: vec![
            vec![5.0, 3.0, 2.0, 1.0, 0.0],
            vec![3.0, 2.0, 1.0, 0.5, 0.0],
            vec![0.0; 5],
        ],
    });
    cv
}

#[test]
fn fully_featured_bond_agrees_with_the_tree() {
    let cv = fully_featured();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 48.0, 70.0] {
        let tf = tf_market(spot);
        let fd = cv.fd_valuation(&tf, &curve, settlement, FINE).unwrap();
        let tree = cv
            .dirty_price_with_steps(&tf, &curve, settlement, 1600)
            .unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.2,
            "TF spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
        let jtd = jtd_market(spot);
        let fd = cv.fd_valuation(&jtd, &curve, settlement, FINE).unwrap();
        let tree = cv
            .dirty_price_with_steps(&jtd, &curve, settlement, 1600)
            .unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.2,
            "JTD spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
    }
}

#[test]
fn mandatory_agrees_with_the_tree() {
    let bond = FixedRateBond::us_corporate(1000.0, 0.06, d(2026, 5, 15), d(2029, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.mandatory = Some(MandatoryConversion { max_ratio: 25.0 });
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 45.0, 60.0] {
        let tf = tf_market(spot);
        let fd = cv.fd_valuation(&tf, &curve, settlement, GRID).unwrap();
        let tree = cv.dirty_price(&tf, &curve, settlement).unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.15,
            "TF spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
        let jtd = jtd_market(spot);
        let fd = cv.fd_valuation(&jtd, &curve, settlement, GRID).unwrap();
        let tree = cv.dirty_price(&jtd, &curve, settlement).unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.15,
            "JTD spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
    }
}

#[test]
fn cash_dividends_agree_between_the_engines() {
    let mut cv = structured();
    cv.cash_dividends = vec![
        CashDividend {
            ex_date: d(2027, 3, 1),
            amount: 0.75,
        },
        CashDividend {
            ex_date: d(2028, 3, 1),
            amount: 0.75,
        },
        CashDividend {
            ex_date: d(2029, 3, 1),
            amount: 0.75,
        },
    ];
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    for spot in [30.0, 48.0, 70.0] {
        let tf = tf_market(spot);
        let fd = cv.fd_valuation(&tf, &curve, settlement, FINE).unwrap();
        let tree = cv
            .dirty_price_with_steps(&tf, &curve, settlement, 1600)
            .unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.2,
            "TF spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
        let jtd = jtd_market(spot);
        let fd = cv.fd_valuation(&jtd, &curve, settlement, FINE).unwrap();
        let tree = cv
            .dirty_price_with_steps(&jtd, &curve, settlement, 1600)
            .unwrap();
        assert!(
            (fd.dirty_price - tree).abs() < 0.2,
            "JTD spot {spot}: fd {} vs tree {tree}",
            fd.dirty_price
        );
    }
}

#[test]
fn busted_limits_reprice_the_straight_bonds() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let tf = tf_market(0.01);
    let fd = cv.fd_valuation(&tf, &curve, settlement, GRID).unwrap();
    let floor = cv.bond_floor(&tf, &curve, settlement).unwrap();
    assert!(
        (fd.clean_price - floor).abs() < 0.01,
        "{} vs {floor}",
        fd.clean_price
    );
    assert!(fd.delta.abs() < 1e-3 && fd.gamma.abs() < 1e-3);

    let jtd = jtd_market(0.01);
    let fd = cv.fd_valuation(&jtd, &curve, settlement, GRID).unwrap();
    let floor = cv.bond_floor(&jtd, &curve, settlement).unwrap();
    assert!(
        (fd.clean_price - floor).abs() < 0.01,
        "{} vs {floor}",
        fd.clean_price
    );
}

#[test]
fn greeks_read_off_the_grid_are_consistent() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    // at the money: positive gamma, delta between the busted and the
    // full-conversion limits
    let m = jtd_market(48.0);
    let fd = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
    let full = cv.conversion_ratio * 100.0 / cv.bond.face_value;
    assert!(fd.gamma > 0.0, "gamma {}", fd.gamma);
    assert!(fd.delta > 0.0 && fd.delta < full, "delta {}", fd.delta);
    // the grid gamma matches a central difference of the grid delta
    let bump = 0.5;
    let up = cv
        .fd_valuation(
            &JumpToDefaultMarket {
                spot: 48.0 + bump,
                ..m
            },
            &curve,
            settlement,
            GRID,
        )
        .unwrap();
    let down = cv
        .fd_valuation(
            &JumpToDefaultMarket {
                spot: 48.0 - bump,
                ..m
            },
            &curve,
            settlement,
            GRID,
        )
        .unwrap();
    let bumped_gamma = (up.delta - down.delta) / (2.0 * bump);
    assert!(
        (fd.gamma - bumped_gamma).abs() < 0.1 * fd.gamma,
        "grid gamma {} vs bumped {bumped_gamma}",
        fd.gamma
    );
    // deep in the money: delta is the full ratio
    let deep = cv
        .fd_valuation(&jtd_market(250.0), &curve, settlement, GRID)
        .unwrap();
    assert!(
        (deep.delta - full).abs() < 0.02 * full,
        "delta {} vs {full}",
        deep.delta
    );
}

#[test]
fn maturity_only_conversion_matches_the_closed_form() {
    // same anchor as the tree test: risky bond + ratio European calls
    // on the survival branch, Black-Scholes at rate r + lambda
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.convert_from = Some(d(2031, 5, 14));
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = jtd_market(48.0);
    let fd = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();

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
    assert!(
        (fd.dirty_price - expected).abs() < 0.05,
        "{} vs {expected}",
        fd.dirty_price
    );
    // and the analytic delta of the package
    let expected_delta = cv.conversion_ratio
        * (-(m.dividend_yield + m.borrow_cost) * t).exp()
        * norm_cdf(d1)
        * 100.0
        / cv.bond.face_value;
    assert!(
        (fd.delta - expected_delta).abs() < 0.01,
        "delta {} vs {expected_delta}",
        fd.delta
    );
}

#[test]
fn bump_greeks_have_the_right_signs_and_limits() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let tenors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0];

    // busted: the rate and spread DV01s are the straight bond's, and
    // there is no vega
    let busted = tf_market(0.01);
    let g = cv
        .fd_greeks(&busted, &curve, settlement, GRID, &tenors)
        .unwrap();
    let straight_spread_dv01 = cv
        .bond
        .spread_dv01(&curve, busted.credit_spread, settlement)
        .unwrap();
    assert!(
        (g.credit_dv01 - straight_spread_dv01).abs() < 0.002,
        "{} vs {straight_spread_dv01}",
        g.credit_dv01
    );
    let parallel = curve.bumped(&RateShift::ParallelAbsolute(1e-4)).unwrap();
    let straight_rate_dv01 = cv
        .bond
        .clean_price_from_curve_with_spread(&curve, busted.credit_spread, settlement)
        .unwrap()
        - cv.bond
            .clean_price_from_curve_with_spread(&parallel, busted.credit_spread, settlement)
            .unwrap();
    assert!(
        (g.rate_dv01 - straight_rate_dv01).abs() < 0.002,
        "{} vs {straight_rate_dv01}",
        g.rate_dv01
    );
    assert!(g.vega.abs() < 0.01, "vega {}", g.vega);
    // the key-rate DV01s partition the parallel one
    let sum: f64 = g.key_rate_dv01.iter().sum();
    assert!(
        (sum - g.rate_dv01).abs() < 0.05 * g.rate_dv01,
        "key-rate sum {sum} vs parallel {}",
        g.rate_dv01
    );
    assert!(g.key_rate_dv01.iter().all(|x| *x >= -1e-9));

    // at the money: long vol, less spread-sensitive than the straight
    // bond (the equity part is spread-free), rates partition again
    let atm = tf_market(48.0);
    let g = cv
        .fd_greeks(&atm, &curve, settlement, GRID, &tenors)
        .unwrap();
    assert!(g.vega > 0.1, "vega {}", g.vega);
    assert!(
        g.credit_dv01 > 0.0 && g.credit_dv01 < straight_spread_dv01,
        "spread dv01 {} vs straight {straight_spread_dv01}",
        g.credit_dv01
    );
    assert!(g.theta.is_finite());
    let sum: f64 = g.key_rate_dv01.iter().sum();
    assert!(
        (sum - g.rate_dv01).abs() < 0.05 * g.rate_dv01.abs().max(1e-3),
        "key-rate sum {sum} vs parallel {}",
        g.rate_dv01
    );
    assert!(
        (g.valuation.delta
            - cv.fd_valuation(&atm, &curve, settlement, GRID)
                .unwrap()
                .delta)
            .abs()
            < 1e-12
    );

    // jump to default: a hazard DV01 at the money, and the key rates
    // still partition
    let g = cv
        .fd_greeks(&jtd_market(48.0), &curve, settlement, GRID, &tenors)
        .unwrap();
    assert!(g.vega > 0.1 && g.credit_dv01 > 0.0, "{g:?}");
    let sum: f64 = g.key_rate_dv01.iter().sum();
    assert!((sum - g.rate_dv01).abs() < 0.05 * g.rate_dv01.abs().max(1e-3));
    // no tenors, no key rates
    let g = cv
        .fd_greeks(&jtd_market(48.0), &curve, settlement, GRID, &[])
        .unwrap();
    assert!(g.key_rate_dv01.is_empty());
}

#[test]
fn pluggable_volatility_reduces_to_flat() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = tf_market(48.0);
    let flat_value = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
    // a constant custom field is the flat engine to the bit
    let constant = |_s: f64, _t: f64| 0.30;
    let custom = cv
        .fd_valuation_with_vol(&m, &curve, settlement, GRID, &FdVolModel::Custom(&constant))
        .unwrap();
    assert!((custom.dirty_price - flat_value.dirty_price).abs() < 1e-12);
    assert!((custom.delta - flat_value.delta).abs() < 1e-12);
    // Dupire local vol of a flat implied surface is that flat vol
    let surface = VolSurface::flat(0.30, d(2026, 8, 13), DayCountConvention::Act365).unwrap();
    let local = cv
        .local_vol_grid(&surface, &curve, settlement, m.spot, m.dividend_yield)
        .unwrap();
    let with_local = cv
        .fd_valuation_with_vol(&m, &curve, settlement, GRID, &FdVolModel::Local(&local))
        .unwrap();
    assert!(
        (with_local.dirty_price - flat_value.dirty_price).abs() < 0.02,
        "{} vs {}",
        with_local.dirty_price,
        flat_value.dirty_price
    );
    // and under jump to default
    let jm = jtd_market(48.0);
    let flat_jtd = cv.fd_valuation(&jm, &curve, settlement, GRID).unwrap();
    let local_jtd = cv
        .fd_valuation_with_vol(&jm, &curve, settlement, GRID, &FdVolModel::Local(&local))
        .unwrap();
    assert!((local_jtd.dirty_price - flat_jtd.dirty_price).abs() < 0.02);
}

#[test]
fn time_dependent_volatility_prices_by_total_variance() {
    // a European conversion right only sees the integrated variance:
    // a two-regime term structure with the flat 30% total variance
    // reprices the flat engine
    let bond = FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
    let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
    cv.convert_from = Some(d(2031, 5, 14));
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = ConvertibleMarket {
        credit_spread: 0.0,
        ..tf_market(48.0)
    };
    let t0 = curve
        .day_count()
        .year_fraction(curve.reference_date(), settlement);
    let horizon = curve.day_count().year_fraction(
        curve.reference_date(),
        cv.bond.cashflows().last().unwrap().payment_date,
    );
    let mid = 0.5 * (t0 + horizon);
    // 0.2 then sigma_2 with 0.2^2/2 + sigma_2^2/2 = 0.3^2
    let late_vol = (2.0 * 0.09 - 0.04_f64).sqrt();
    let regimes = move |_s: f64, t: f64| if t < mid { 0.20 } else { late_vol };
    let flat_value = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
    let term = cv
        .fd_valuation_with_vol(&m, &curve, settlement, GRID, &FdVolModel::Custom(&regimes))
        .unwrap();
    assert!(
        (term.dirty_price - flat_value.dirty_price).abs() < 0.05,
        "{} vs {}",
        term.dirty_price,
        flat_value.dirty_price
    );
}

#[test]
fn skewed_local_volatility_moves_the_price_and_the_greeks_run() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = tf_market(48.0);
    // a downside skew: more vol below the conversion price
    let skew = |s: f64, _t: f64| (0.30 * (50.0 / s).powf(0.3)).clamp(0.05, 1.5);
    let model = FdVolModel::Custom(&skew);
    let skewed = cv
        .fd_valuation_with_vol(&m, &curve, settlement, GRID, &model)
        .unwrap();
    let flat_value = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
    assert!((skewed.dirty_price - flat_value.dirty_price).abs() > 0.05);
    let greeks = cv
        .fd_greeks_with_vol(&m, &curve, settlement, GRID, &[1.0, 3.0, 5.0], &model)
        .unwrap();
    assert!(greeks.vega > 0.1 && greeks.credit_dv01 > 0.0, "{greeks:?}");
    assert!((greeks.valuation.dirty_price - skewed.dirty_price).abs() < 1e-12);
    let greeks = cv
        .fd_greeks_with_vol(&jtd_market(48.0), &curve, settlement, GRID, &[], &model)
        .unwrap();
    assert!(greeks.vega > 0.1, "{greeks:?}");
}

#[test]
fn dejumping_removes_the_default_from_an_implied_vol() {
    // the survival vol reproduces the listed call under jump to default
    let (spot, r, q, b) = (48.0, 0.04, 0.01, 0.0);
    // no hazard, no change
    assert!(
        (dejump_implied_vol(0.30, spot, spot, 5.0, r, q, b, 0.0).unwrap() - 0.30).abs() < 1e-12
    );
    // 30% at the money, five years: 3% hazard leaves about 20.6% of
    // diffusion, and the skew a flat implied vol hides
    let atm = dejump_implied_vol(0.30, spot, spot, 5.0, r, q, b, 0.03).unwrap();
    assert!((atm - 0.206).abs() < 0.005, "{atm}");
    let low = dejump_implied_vol(0.30, spot, 40.0, 5.0, r, q, b, 0.03).unwrap();
    let high = dejump_implied_vol(0.30, spot, 80.0, 5.0, r, q, b, 0.03).unwrap();
    assert!(low < atm && atm < high, "{low} {atm} {high}");
    // shorter and safer, less to remove
    let short = dejump_implied_vol(0.30, spot, spot, 1.0, r, q, b, 0.03).unwrap();
    let safer = dejump_implied_vol(0.30, spot, spot, 5.0, r, q, b, 0.01).unwrap();
    assert!(short > atm && safer > atm);
    // the jump alone exceeds the option: no diffusion fits
    assert!(matches!(
        dejump_implied_vol(0.30, spot, spot, 5.0, r, q, b, 0.10),
        Err(RustyQLibError::CalibrationFailed { .. })
    ));
    assert!(dejump_implied_vol(-0.1, spot, spot, 5.0, r, q, b, 0.03).is_err());
}

#[test]
fn dejumped_surface_feeds_a_consistent_local_vol() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = jtd_market(48.0);
    let listed = VolSurface::flat(0.30, d(2026, 8, 13), DayCountConvention::Act365).unwrap();
    // a flat surface prices deep out-of-the-money puts near zero, below
    // the default leg a 3% hazard implies, so the grid stays where a
    // flat quote and the hazard are consistent (a listed surface carries
    // that value in its put skew)
    let strikes = [40.0, 48.0, 60.0, 80.0, 120.0];
    let expiries = [1.0, 2.0, 3.0, 5.0];
    let dejumped = m
        .dejump_surface(&listed, &curve, &strikes, &expiries)
        .unwrap();
    // the surface holds the de-jumped vols: lower, and rising in strike
    let at = |k: f64, t: f64| dejumped.vol(k, 48.0, t);
    assert!(
        at(48.0, 5.0) < 0.25 && at(48.0, 5.0) > 0.15,
        "{}",
        at(48.0, 5.0)
    );
    assert!(at(40.0, 5.0) < at(48.0, 5.0) && at(48.0, 5.0) < at(80.0, 5.0));
    // its local vol prices the convertible below the flat 30% under jump
    // to default, and the engine runs end to end
    let local = cv
        .local_vol_grid(&dejumped, &curve, settlement, m.spot, m.dividend_yield)
        .unwrap();
    let with_local = cv
        .fd_valuation_with_vol(&m, &curve, settlement, GRID, &FdVolModel::Local(&local))
        .unwrap();
    let flat_value = cv.fd_valuation(&m, &curve, settlement, GRID).unwrap();
    assert!(
        with_local.dirty_price < flat_value.dirty_price - 1.0,
        "{} vs {}",
        with_local.dirty_price,
        flat_value.dirty_price
    );
    // a hazard the quotes cannot carry fails, naming the point; so does
    // a strike whose flat-vol put is worth less than the default leg
    let hot = JumpToDefaultMarket {
        hazard_rate: 0.10,
        ..m
    };
    assert!(hot
        .dejump_surface(&listed, &curve, &strikes, &expiries)
        .is_err());
    assert!(m
        .dejump_surface(&listed, &curve, &[20.0, 48.0], &[0.5])
        .is_err());
}

#[test]
fn grid_validation() {
    let cv = convertible();
    let curve = flat(0.04);
    let settlement = d(2026, 8, 14);
    let m = tf_market(48.0);
    let coarse_time = ConvertibleFdGrid {
        time_steps: 5,
        ..GRID
    };
    assert!(cv
        .fd_valuation(&m, &curve, settlement, coarse_time)
        .is_err());
    let coarse_space = ConvertibleFdGrid {
        space_steps: 10,
        ..GRID
    };
    assert!(cv
        .fd_valuation(&m, &curve, settlement, coarse_space)
        .is_err());
    let narrow = ConvertibleFdGrid {
        grid_stdevs: 0.0,
        ..GRID
    };
    assert!(cv.fd_valuation(&m, &curve, settlement, narrow).is_err());
    // an odd space count is rounded up, not rejected
    let odd = ConvertibleFdGrid {
        space_steps: 201,
        ..GRID
    };
    assert!(cv.fd_valuation(&m, &curve, settlement, odd).is_ok());
    let bad = ConvertibleMarket {
        volatility: -0.1,
        ..m
    };
    assert!(cv.fd_valuation(&bad, &curve, settlement, GRID).is_err());
    let bad = JumpToDefaultMarket {
        hazard_rate: -0.1,
        ..jtd_market(48.0)
    };
    assert!(cv.fd_valuation(&bad, &curve, settlement, GRID).is_err());
}
