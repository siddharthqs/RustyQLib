//! Credit-spread analytics for bonds priced against a benchmark curve.
//!
//! - **Z-spread**: the constant spread added to every continuously
//!   compounded zero rate of the benchmark curve that reprices the bond
//!   exactly. Implemented by shifting the curve with
//!   [`RateShift::ParallelAbsolute`], which is exact everywhere
//!   (including between pillars and in extrapolation), so pricing with
//!   a z-spread and recovering it are perfect inverses.
//! - **Spread DV01**: price change for a one-basis-point widening of
//!   the z-spread — the spread-duration risk number.
//! - **G-spread**: bond yield minus the benchmark yield linearly
//!   interpolated at the bond's maturity.

use chrono::NaiveDate;

use crate::bonds::FixedRateBond;
use crate::core::calendar::{BusinessDayConvention, Frequency};
use crate::core::curves::{RateShift, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::rates::leg::accrual_periods;

/// Search bracket for the z-spread solve, in absolute rate terms.
const SPREAD_BRACKET: (f64, f64) = (-0.5, 3.0);

impl FixedRateBond {
    /// Dirty price per 100 face off `curve` shifted by a constant
    /// continuous `spread` (the z-spread pricing direction).
    pub fn dirty_price_from_curve_with_spread(
        &self,
        curve: &YieldCurve,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "spread",
                format!("spread must be finite, got {spread}"),
            ));
        }
        let shifted = curve.bumped(&RateShift::ParallelAbsolute(spread))?;
        self.dirty_price_from_curve(&shifted, settlement)
    }

    /// Clean price per 100 face off `curve` plus a constant `spread`.
    pub fn clean_price_from_curve_with_spread(
        &self,
        curve: &YieldCurve,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.dirty_price_from_curve_with_spread(curve, spread, settlement)?
                - self.accrued_interest(settlement)?,
        )
    }

    /// The z-spread: the constant continuous spread over `curve` that
    /// reprices the bond at `clean_price`.
    pub fn z_spread(
        &self,
        clean_price: f64,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "z_spread",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // price is decreasing in the spread, so target - price(s) is
        // increasing: a clean sign-change bracket for bisection
        let objective = |s: f64| {
            clean_price
                - self
                    .clean_price_from_curve_with_spread(curve, s, settlement)
                    .expect("spread bracket keeps the shifted curve valid")
        };
        // surface real pricing errors (bad settlement, degenerate curve)
        // once before entering the solver
        self.clean_price_from_curve_with_spread(curve, SPREAD_BRACKET.0, settlement)?;
        let root =
            Solver1d::new(1e-12, 200).bisection(objective, SPREAD_BRACKET.0, SPREAD_BRACKET.1)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "z-spread solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }

    /// Spread DV01 per 100 face: the price drop for a one-basis-point
    /// widening of the z-spread.
    pub fn spread_dv01(
        &self,
        curve: &YieldCurve,
        spread: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let base = self.clean_price_from_curve_with_spread(curve, spread, settlement)?;
        let wider = self.clean_price_from_curve_with_spread(curve, spread + 1e-4, settlement)?;
        Ok(base - wider)
    }
}

impl FixedRateBond {
    /// Par-par asset swap spread: the spread over the floating leg that
    /// compensates a par buyer of the bond package,
    /// `(PV of the bond's flows on the swap curve - market dirty) /
    /// float annuity`, per 100 of outstanding face. The floating leg is
    /// quarterly Act/360 (the USD convention), modified following on the
    /// bond's calendar, from settlement to maturity.
    pub fn asset_swap_spread(
        &self,
        clean_price: f64,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "asset_swap_spread",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        let dirty_market = clean_price + self.accrued_interest(settlement)?;
        let dirty_on_curve = self.dirty_price_from_curve(curve, settlement)?;

        // float-leg annuity per unit notional, discounted to settlement
        let periods = accrual_periods(
            settlement,
            self.maturity_date,
            Frequency::Quarterly,
            &self.calendar,
            BusinessDayConvention::ModifiedFollowing,
            0,
        )?;
        let df_settlement = curve.df_date(settlement);
        let annuity: f64 = periods
            .iter()
            .map(|p| {
                DayCountConvention::Act360.year_fraction(p.start, p.end) * curve.df_date(p.payment)
                    / df_settlement
            })
            .sum();
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive float annuity {annuity}"
            )));
        }
        Ok((dirty_on_curve - dirty_market) / (100.0 * annuity))
    }
}

/// Linearly interpolated benchmark yield at time `t` (years) from
/// `(tenor, yield)` points sorted by tenor; flat beyond the ends.
pub fn interpolated_benchmark_yield(
    t: f64,
    benchmarks: &[(f64, f64)],
) -> Result<f64, RustyQLibError> {
    if benchmarks.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "g_spread",
            "no benchmark points",
        ));
    }
    if benchmarks.windows(2).any(|w| w[1].0 <= w[0].0) {
        return Err(RustyQLibError::invalid_input(
            "g_spread",
            "benchmark tenors must be strictly increasing",
        ));
    }
    let first = benchmarks[0];
    let last = benchmarks[benchmarks.len() - 1];
    if t <= first.0 {
        return Ok(first.1);
    }
    if t >= last.0 {
        return Ok(last.1);
    }
    let i = benchmarks.partition_point(|&(tenor, _)| tenor < t);
    let (t0, y0) = benchmarks[i - 1];
    let (t1, y1) = benchmarks[i];
    Ok(y0 + (y1 - y0) * (t - t0) / (t1 - t0))
}

/// G-spread: the bond's street yield minus the benchmark (government)
/// yield interpolated at the bond's remaining maturity in years.
pub fn g_spread(
    bond_yield: f64,
    maturity_years: f64,
    benchmarks: &[(f64, f64)],
) -> Result<f64, RustyQLibError> {
    Ok(bond_yield - interpolated_benchmark_yield(maturity_years, benchmarks)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
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

    fn corporate() -> FixedRateBond {
        FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap()
    }

    #[test]
    fn z_spread_round_trips_through_the_price() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        for spread in [-0.005, 0.0, 0.0125, 0.045] {
            let clean = bond
                .clean_price_from_curve_with_spread(&curve, spread, settlement)
                .unwrap();
            let recovered = bond.z_spread(clean, &curve, settlement).unwrap();
            assert!(
                (recovered - spread).abs() < 1e-10,
                "spread {spread}: {recovered}"
            );
        }
    }

    #[test]
    fn zero_spread_reduces_to_plain_curve_pricing() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let plain = bond.clean_price_from_curve(&curve, settlement).unwrap();
        let spread_zero = bond
            .clean_price_from_curve_with_spread(&curve, 0.0, settlement)
            .unwrap();
        assert!((plain - spread_zero).abs() < 1e-12);
        // a Treasury priced on its own curve carries no z-spread
        let z = bond.z_spread(plain, &curve, settlement).unwrap();
        assert!(z.abs() < 1e-10, "z {z}");
    }

    #[test]
    fn wider_spread_means_cheaper_bond_and_positive_spread_dv01() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let tight = bond
            .clean_price_from_curve_with_spread(&curve, 0.005, settlement)
            .unwrap();
        let wide = bond
            .clean_price_from_curve_with_spread(&curve, 0.02, settlement)
            .unwrap();
        assert!(wide < tight);
        let sdv01 = bond.spread_dv01(&curve, 0.0125, settlement).unwrap();
        assert!(sdv01 > 0.0);
        // spread duration of a ~4.75y bond: DV01 around 4 cents per 100
        assert!(sdv01 > 0.02 && sdv01 < 0.06, "spread dv01 {sdv01}");
    }

    #[test]
    fn g_spread_interpolates_the_benchmark_linearly() {
        let benchmarks = [(2.0, 0.041), (5.0, 0.043), (10.0, 0.045)];
        // midway between the 2y and 5y points
        let mid = interpolated_benchmark_yield(3.5, &benchmarks).unwrap();
        assert!((mid - 0.042).abs() < 1e-15);
        // flat beyond the ends
        assert_eq!(
            interpolated_benchmark_yield(1.0, &benchmarks).unwrap(),
            0.041
        );
        assert_eq!(
            interpolated_benchmark_yield(30.0, &benchmarks).unwrap(),
            0.045
        );
        // a 5.5% corporate yield over the interpolated 4.2% = 130bp
        let g = g_spread(0.055, 3.5, &benchmarks).unwrap();
        assert!((g - 0.013).abs() < 1e-15);
        // unsorted benchmarks are rejected
        assert!(interpolated_benchmark_yield(3.0, &[(5.0, 0.04), (2.0, 0.04)]).is_err());
        assert!(interpolated_benchmark_yield(3.0, &[]).is_err());
    }

    #[test]
    fn asset_swap_spread_is_zero_on_the_curve_and_tracks_cheapness() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        // a bond priced exactly on the curve swaps flat
        let fair = bond.clean_price_from_curve(&curve, settlement).unwrap();
        let asw = bond.asset_swap_spread(fair, &curve, settlement).unwrap();
        assert!(asw.abs() < 1e-12, "asw {asw}");
        // one point cheap: ASW = 1 / (100 * annuity) exactly (linearity)
        let one_cheap = bond
            .asset_swap_spread(fair - 1.0, &curve, settlement)
            .unwrap();
        let two_cheap = bond
            .asset_swap_spread(fair - 2.0, &curve, settlement)
            .unwrap();
        assert!(one_cheap > 0.0);
        assert!(
            (two_cheap - 2.0 * one_cheap).abs() < 1e-14,
            "linear in price"
        );
        // for a moderately cheap bond ASW sits near the z-spread
        let clean = 98.75;
        let asw = bond.asset_swap_spread(clean, &curve, settlement).unwrap();
        let z = bond.z_spread(clean, &curve, settlement).unwrap();
        assert!((asw - z).abs() < 0.2 * z, "asw {asw} vs z {z}");
        assert!(bond.asset_swap_spread(-1.0, &curve, settlement).is_err());
    }
}
