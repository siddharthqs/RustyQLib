//! Quoted Treasury instruments as curve pillars.
//!
//! A [`BillQuote`] (discount rate) or [`BondQuote`] (clean price) pairs an
//! instrument with its market quote and settlement date, implementing
//! [`CurveInstrument`] so [`bootstrap_curve`](crate::bonds::bootstrap_curve)
//! can build a Treasury discount curve from them.
//!
//! Unlike deposits and FRAs, a coupon bond does not pin its pillar in
//! closed form: coupons paid before maturity discount off earlier parts
//! of the curve, and coupons falling between the last known pillar and
//! the bond's maturity depend on the new pillar through interpolation.
//! Each quote therefore solves a 1-D root problem — find the maturity
//! discount factor such that the candidate curve (known pillars plus the
//! trial pillar) reprices the quote exactly.

use chrono::NaiveDate;

use crate::bonds::{CurveInstrument, FixedRateBond, TreasuryBill};
use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// A Treasury bill with its quoted Act/360 discount rate.
#[derive(Debug, Clone)]
pub struct BillQuote {
    pub bill: TreasuryBill,
    pub discount_rate: f64,
    pub settlement: NaiveDate,
}

impl BillQuote {
    pub fn new(
        bill: TreasuryBill,
        discount_rate: f64,
        settlement: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        // validates the rate, the settlement/maturity order and the
        // implied price in one go
        bill.price_from_discount_rate(discount_rate, settlement)?;
        Ok(BillQuote {
            bill,
            discount_rate,
            settlement,
        })
    }
}

impl CurveInstrument for BillQuote {
    fn maturity_date(&self) -> NaiveDate {
        self.bill.maturity_date
    }

    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError> {
        let price = self
            .bill
            .price_from_discount_rate(self.discount_rate, self.settlement)?;
        let settlement = self.settlement;
        solve_pillar_df(
            self.bill.maturity_date,
            reference_date,
            curve_so_far,
            price,
            move |candidate| {
                Ok(100.0 * candidate.df_date(self.bill.maturity_date)
                    / candidate.df_date(settlement))
            },
        )
    }
}

/// A fixed-rate bond with its quoted clean price per 100 face.
#[derive(Debug, Clone)]
pub struct BondQuote {
    pub bond: FixedRateBond,
    pub clean_price: f64,
    pub settlement: NaiveDate,
}

impl BondQuote {
    pub fn new(
        bond: FixedRateBond,
        clean_price: f64,
        settlement: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond quote",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // surfaces settlement-after-maturity errors at construction
        bond.accrued_interest(settlement)?;
        Ok(BondQuote {
            bond,
            clean_price,
            settlement,
        })
    }
}

impl CurveInstrument for BondQuote {
    fn maturity_date(&self) -> NaiveDate {
        self.bond.maturity_date
    }

    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError> {
        let target_dirty = self.clean_price + self.bond.accrued_interest(self.settlement)?;
        let settlement = self.settlement;
        solve_pillar_df(
            self.bond.maturity_date,
            reference_date,
            curve_so_far,
            target_dirty,
            move |candidate| self.bond.dirty_price_from_curve(candidate, settlement),
        )
    }
}

/// Bracket for the pillar discount-factor search: wide enough for deeply
/// negative rates while keeping the candidate curve valid (df > 0).
const DF_BRACKET: (f64, f64) = (1e-8, 4.0);

/// Find the discount factor at `maturity` such that the candidate curve
/// — `curve_so_far`'s pillars plus the trial pillar — makes
/// `model_price(candidate)` equal `target_price`. `model_price` must be
/// increasing in the trial df (true for any instrument whose remaining
/// cash flows are positive).
fn solve_pillar_df(
    maturity: NaiveDate,
    reference_date: NaiveDate,
    curve_so_far: Option<&YieldCurve>,
    target_price: f64,
    model_price: impl Fn(&YieldCurve) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    // surface real construction/pricing errors once, at a bracket end;
    // inside the objective the same operations cannot fail for any df in
    // the bracket (times are fixed and every df stays positive)
    let probe = candidate_curve(curve_so_far, reference_date, maturity, DF_BRACKET.0)?;
    model_price(&probe)?;

    let objective = |df: f64| {
        let candidate = candidate_curve(curve_so_far, reference_date, maturity, df)
            .expect("candidate construction cannot fail inside the bracket");
        model_price(&candidate).expect("model price cannot fail inside the bracket") - target_price
    };
    let root = Solver1d::new(1e-10, 200).bisection(objective, DF_BRACKET.0, DF_BRACKET.1)?;
    if !root.converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations: root.iterations,
            residual: objective(root.x).abs(),
            reason: format!("pillar solve for maturity {maturity} did not converge"),
        });
    }
    Ok(root.x)
}

/// The pillars of `curve_so_far` (kept at their exact times) plus a trial
/// pillar at `maturity` with discount factor `df`.
fn candidate_curve(
    curve_so_far: Option<&YieldCurve>,
    reference_date: NaiveDate,
    maturity: NaiveDate,
    df: f64,
) -> Result<YieldCurve, RustyQLibError> {
    let (mut tenors, mut dfs, day_count) = match curve_so_far {
        Some(curve) => {
            let pillars = curve.pillars();
            let tenors: Vec<Tenor> = pillars
                .iter()
                .map(|p| Tenor::YearFraction(p.time))
                .collect();
            let dfs: Vec<f64> = pillars.iter().map(|p| p.df).collect();
            (tenors, dfs, curve.day_count())
        }
        None => (Vec::new(), Vec::new(), DayCountConvention::Act365),
    };
    tenors.push(Tenor::Date(maturity));
    dfs.push(df);
    Ok(YieldCurve::from_discount_factors(
        &tenors,
        &dfs,
        reference_date,
        day_count,
        Compounding::Continuous,
        InterpolationMethod::LogLinearDf,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::bootstrap_curve;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn bill_pillar_is_price_over_face_when_settling_on_reference() {
        let reference = d(2026, 8, 6);
        let bill = TreasuryBill::new(100.0, d(2026, 11, 5)).unwrap();
        let price = bill.price_from_discount_rate(0.05, reference).unwrap();
        let quote = BillQuote::new(bill, 0.05, reference).unwrap();
        let curve = bootstrap_curve(
            &[Box::new(quote) as Box<dyn CurveInstrument>],
            reference,
            DayCountConvention::Act365,
        )
        .unwrap();
        assert!(
            (curve.df_date(d(2026, 11, 5)) - price / 100.0).abs() < 1e-9,
            "df {} vs price/100 {}",
            curve.df_date(d(2026, 11, 5)),
            price / 100.0
        );
    }

    #[test]
    fn bill_with_forward_settlement_is_self_consistent() {
        // curve as of trade date, bill settling T+1: the solved curve
        // must satisfy df(maturity)/df(settlement) = price/100
        let reference = d(2026, 8, 5);
        let settlement = d(2026, 8, 6);
        let bill = TreasuryBill::new(100.0, d(2026, 11, 5)).unwrap();
        let price = bill.price_from_discount_rate(0.05, settlement).unwrap();
        let quote = BillQuote::new(bill, 0.05, settlement).unwrap();
        let curve = bootstrap_curve(
            &[Box::new(quote) as Box<dyn CurveInstrument>],
            reference,
            DayCountConvention::Act365,
        )
        .unwrap();
        let ratio = curve.df_date(d(2026, 11, 5)) / curve.df_date(settlement);
        assert!((ratio - price / 100.0).abs() < 1e-9, "ratio {ratio}");
    }

    /// Bills and notes on the May 15 / Nov 15 cycle quoted for 2026-08-06.
    fn treasury_market(
        settlement: NaiveDate,
    ) -> (
        Vec<Box<dyn CurveInstrument>>,
        Vec<BondQuote>,
        Vec<BillQuote>,
    ) {
        let bill_3m = BillQuote::new(
            TreasuryBill::new(100.0, d(2026, 11, 5)).unwrap(),
            0.048,
            settlement,
        )
        .unwrap();
        let bill_6m = BillQuote::new(
            TreasuryBill::new(100.0, d(2027, 2, 4)).unwrap(),
            0.047,
            settlement,
        )
        .unwrap();
        let note_2y = BondQuote::new(
            FixedRateBond::us_treasury(100.0, 0.045, d(2026, 5, 15), d(2028, 5, 15)).unwrap(),
            99.50,
            settlement,
        )
        .unwrap();
        let note_5y = BondQuote::new(
            FixedRateBond::us_treasury(100.0, 0.0425, d(2026, 5, 15), d(2031, 5, 15)).unwrap(),
            101.25,
            settlement,
        )
        .unwrap();
        let bonds = vec![note_2y.clone(), note_5y.clone()];
        let bills = vec![bill_3m.clone(), bill_6m.clone()];
        let instruments: Vec<Box<dyn CurveInstrument>> = vec![
            Box::new(bill_3m),
            Box::new(bill_6m),
            Box::new(note_2y),
            Box::new(note_5y),
        ];
        (instruments, bonds, bills)
    }

    #[test]
    fn treasury_curve_reprices_every_input_quote() {
        let settlement = d(2026, 8, 6);
        let (instruments, bonds, bills) = treasury_market(settlement);
        let curve = bootstrap_curve(&instruments, settlement, DayCountConvention::Act365).unwrap();
        assert_eq!(curve.pillars().len(), 4);

        for quote in &bills {
            let quoted = quote
                .bill
                .price_from_discount_rate(quote.discount_rate, settlement)
                .unwrap();
            let repriced = 100.0 * curve.df_date(quote.bill.maturity_date);
            assert!(
                (repriced - quoted).abs() < 1e-7,
                "bill {}: {repriced} vs {quoted}",
                quote.bill.maturity_date
            );
        }
        for quote in &bonds {
            let repriced = quote
                .bond
                .clean_price_from_curve(&curve, settlement)
                .unwrap();
            assert!(
                (repriced - quote.clean_price).abs() < 1e-7,
                "bond {}: {repriced} vs {}",
                quote.bond.maturity_date,
                quote.clean_price
            );
        }
        // sanity: discount factors decrease along the curve
        let dfs: Vec<f64> = curve.pillars().iter().map(|p| p.df).collect();
        assert!(dfs.windows(2).all(|w| w[1] < w[0]), "{dfs:?}");
    }

    #[test]
    fn curve_yields_are_near_the_quoted_coupons() {
        // coarse sanity on levels: zeros should sit in the vicinity of
        // the quoted 4.2-4.8% Treasury market
        let settlement = d(2026, 8, 6);
        let (instruments, _, _) = treasury_market(settlement);
        let curve = bootstrap_curve(&instruments, settlement, DayCountConvention::Act365).unwrap();
        for pillar in curve.pillars() {
            assert!(
                pillar.zero_rate > 0.03 && pillar.zero_rate < 0.06,
                "zero {} at t={}",
                pillar.zero_rate,
                pillar.time
            );
        }
    }

    #[test]
    fn absurd_bond_quote_fails_the_solve() {
        let settlement = d(2026, 8, 6);
        let bond =
            FixedRateBond::us_treasury(100.0, 0.045, d(2026, 5, 15), d(2028, 5, 15)).unwrap();
        // a price no positive curve can reach
        let quote = BondQuote::new(bond, 1e6, settlement).unwrap();
        let result = bootstrap_curve(
            &[Box::new(quote) as Box<dyn CurveInstrument>],
            settlement,
            DayCountConvention::Act365,
        );
        assert!(result.is_err());
    }

    #[test]
    fn quote_construction_rejects_bad_inputs() {
        let bill = TreasuryBill::new(100.0, d(2026, 11, 5)).unwrap();
        // settlement after maturity
        assert!(BillQuote::new(bill.clone(), 0.05, d(2027, 1, 1)).is_err());
        // discount rate implying a negative price
        assert!(BillQuote::new(bill, 5.0, d(2026, 8, 6)).is_err());
        let bond =
            FixedRateBond::us_treasury(100.0, 0.045, d(2026, 5, 15), d(2028, 5, 15)).unwrap();
        assert!(BondQuote::new(bond.clone(), -10.0, d(2026, 8, 6)).is_err());
        assert!(BondQuote::new(bond, 99.5, d(2029, 1, 1)).is_err());
    }
}
