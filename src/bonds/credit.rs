//! Credit-risky bond pricing on a flat hazard rate.
//!
//! The reduced-form model: default arrives at a constant intensity
//! `lambda`, so survival to time `t` (measured from settlement on the
//! curve's day count) is `exp(-lambda * t)`. A risky bond is then
//!
//! - the promised cash flows, each weighted by survival to its payment,
//!   plus
//! - `recovery * face` paid at the end of the period in which default
//!   happens, weighted by the probability of defaulting in that period.
//!
//! Everything discounts on the risk-free `curve`. For small rates the
//! classic credit-triangle identity holds: the z-spread of the risky
//! price is approximately `lambda * (1 - recovery)` — a good
//! cross-check between this module and
//! [`spreads`](crate::bonds::spreads).

use chrono::NaiveDate;

pub use crate::credit::CreditCurve;

use crate::bonds::FixedRateBond;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

fn validate_credit(hazard_rate: f64, recovery_rate: f64) -> Result<(), RustyQLibError> {
    if !hazard_rate.is_finite() || hazard_rate < 0.0 {
        return Err(RustyQLibError::invalid_input(
            "hazard_rate",
            format!("must be non-negative, got {hazard_rate}"),
        ));
    }
    if !recovery_rate.is_finite() || !(0.0..=1.0).contains(&recovery_rate) {
        return Err(RustyQLibError::invalid_input(
            "recovery_rate",
            format!("must be in [0, 1], got {recovery_rate}"),
        ));
    }
    Ok(())
}

impl FixedRateBond {
    /// Dirty price per 100 face at `settlement` under a survival
    /// function measured from settlement: promised flows weighted by
    /// survival plus the recovery leg, discounted on `curve`.
    fn risky_dirty_price_with(
        &self,
        curve: &YieldCurve,
        survival: &dyn Fn(f64) -> f64,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let df_settlement = curve.df_date(settlement);
        if !df_settlement.is_finite() || df_settlement <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive discount factor at settlement {settlement}"
            )));
        }
        // ensures settlement < maturity like the risk-free pricers
        self.accrued_interest(settlement)?;

        let day_count = curve.day_count();
        let mut pv = 0.0;
        let mut survival_start = 1.0; // survival to the period start
        for cashflow in self
            .cashflows()
            .iter()
            .filter(|cf| cf.accrual_end > settlement)
        {
            let t_end = day_count.year_fraction(settlement, cashflow.accrual_end.max(settlement));
            let survival_end = survival(t_end);
            let df = curve.df_date(cashflow.payment_date) / df_settlement;
            // promised flow if the issuer survives the period
            pv += cashflow.amount * survival_end * df;
            // recovery on the face outstanding during the period if
            // default arrives inside it (paid at the period end)
            pv += recovery_rate
                * self.outstanding_face(cashflow.accrual_start)
                * (survival_start - survival_end)
                * df;
            survival_start = survival_end;
        }
        Ok(pv * 100.0 / self.outstanding_face(settlement))
    }

    /// Dirty price per 100 face at `settlement` under a flat hazard
    /// rate: promised flows weighted by survival plus the recovery leg,
    /// discounted on `curve`.
    pub fn risky_dirty_price(
        &self,
        curve: &YieldCurve,
        hazard_rate: f64,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        validate_credit(hazard_rate, recovery_rate)?;
        let survival = |t: f64| (-hazard_rate * t).exp();
        self.risky_dirty_price_with(curve, &survival, recovery_rate, settlement)
    }

    /// Clean price per 100 face under a flat hazard rate.
    pub fn risky_clean_price(
        &self,
        curve: &YieldCurve,
        hazard_rate: f64,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.risky_dirty_price(curve, hazard_rate, recovery_rate, settlement)?
                - self.accrued_interest(settlement)?,
        )
    }

    /// Dirty price per 100 face on a hazard **term structure** (times
    /// measured from `settlement` on the discount curve's day count).
    pub fn risky_dirty_price_on_curve(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        validate_credit(0.0, recovery_rate)?;
        let survival = |t: f64| credit.survival(t);
        self.risky_dirty_price_with(curve, &survival, recovery_rate, settlement)
    }

    /// Clean price per 100 face on a hazard term structure.
    pub fn risky_clean_price_on_curve(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.risky_dirty_price_on_curve(curve, credit, recovery_rate, settlement)?
                - self.accrued_interest(settlement)?,
        )
    }

    /// The flat hazard rate implied by a clean price, given the
    /// recovery assumption.
    pub fn implied_hazard_rate(
        &self,
        clean_price: f64,
        curve: &YieldCurve,
        recovery_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        validate_credit(0.0, recovery_rate)?;
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "implied_hazard_rate",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // price is decreasing in the hazard (recovery < 1 destroys value,
        // and with recovery = 1 coupons are still lost), so
        // target - price(lambda) is increasing: bisection on [0, 10]
        let objective = |lambda: f64| {
            clean_price
                - self
                    .risky_clean_price(curve, lambda, recovery_rate, settlement)
                    .expect("hazard bracket is valid")
        };
        // surface real pricing errors before entering the solver
        self.risky_clean_price(curve, 0.0, recovery_rate, settlement)?;
        let root = Solver1d::new(1e-12, 200).bisection(objective, 0.0, 10.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "hazard-rate solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }
}

/// Bootstrap a piecewise-constant hazard curve from a strip of bonds of
/// one issuer, quoted as `(bond, clean_price)` — the credit analogue of
/// the discount-curve bootstrap. Bonds are processed in maturity order;
/// each pins the hazard on the segment from the previous bond's
/// maturity to its own, chosen so the bond reprices exactly given the
/// segments already solved. All prices share one `settlement` and one
/// `recovery_rate`; times run from settlement on the discount curve's
/// day count.
pub fn bootstrap_credit_curve(
    quotes: &[(FixedRateBond, f64)],
    curve: &YieldCurve,
    recovery_rate: f64,
    settlement: NaiveDate,
) -> Result<CreditCurve, RustyQLibError> {
    validate_credit(0.0, recovery_rate)?;
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "credit bootstrap",
            "no bond quotes",
        ));
    }
    let mut order: Vec<usize> = (0..quotes.len()).collect();
    order.sort_by_key(|&i| quotes[i].0.maturity_date);
    for pair in order.windows(2) {
        if quotes[pair[0]].0.maturity_date == quotes[pair[1]].0.maturity_date {
            return Err(RustyQLibError::invalid_input(
                "credit bootstrap",
                format!(
                    "two bonds share the maturity {}",
                    quotes[pair[0]].0.maturity_date
                ),
            ));
        }
    }

    let day_count = curve.day_count();
    let mut pillars: Vec<(f64, f64)> = Vec::with_capacity(quotes.len());
    for &index in &order {
        let (bond, clean_price) = &quotes[index];
        if *clean_price <= 0.0 || !clean_price.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "credit bootstrap",
                format!("clean prices must be positive, got {clean_price}"),
            ));
        }
        let pillar_time = day_count.year_fraction(settlement, bond.maturity_date);
        if pillar_time <= pillars.last().map_or(0.0, |&(t, _)| t) {
            return Err(RustyQLibError::invalid_input(
                "credit bootstrap",
                format!(
                    "bond maturing {} does not extend the curve",
                    bond.maturity_date
                ),
            ));
        }
        // solve the new segment's hazard so this bond reprices exactly;
        // price is decreasing in the hazard, so bisection is safe
        let objective = |hazard: f64| -> f64 {
            let mut candidate = pillars.clone();
            candidate.push((pillar_time, hazard));
            let credit = CreditCurve::new(&candidate)
                .expect("candidate pillars are increasing and the hazard is in [0, 10]");
            clean_price
                - bond
                    .risky_clean_price_on_curve(curve, &credit, recovery_rate, settlement)
                    .expect("pricing succeeded at the bracket ends")
        };
        // surface real pricing errors once before the solver
        {
            let mut candidate = pillars.clone();
            candidate.push((pillar_time, 0.0));
            let credit = CreditCurve::new(&candidate)?;
            bond.risky_clean_price_on_curve(curve, &credit, recovery_rate, settlement)?;
        }
        let root = Solver1d::new(1e-9, 200).bisection(objective, 0.0, 10.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: format!(
                    "credit bootstrap failed at the {} pillar",
                    bond.maturity_date
                ),
            });
        }
        pillars.push((pillar_time, root.x));
    }
    CreditCurve::new(&pillars)
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
    fn zero_hazard_reduces_to_risk_free_pricing() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let risky = bond
            .risky_dirty_price(&curve, 0.0, 0.4, settlement)
            .unwrap();
        let risk_free = bond.dirty_price_from_curve(&curve, settlement).unwrap();
        assert!((risky - risk_free).abs() < 1e-12, "{risky} vs {risk_free}");
    }

    #[test]
    fn hazard_cheapens_and_recovery_cushions() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let risk_free = bond
            .risky_clean_price(&curve, 0.0, 0.4, settlement)
            .unwrap();
        let risky = bond
            .risky_clean_price(&curve, 0.02, 0.4, settlement)
            .unwrap();
        assert!(risky < risk_free, "{risky} vs {risk_free}");
        // higher recovery is worth more at the same hazard
        let high_recovery = bond
            .risky_clean_price(&curve, 0.02, 0.8, settlement)
            .unwrap();
        assert!(high_recovery > risky);
        // even full recovery of face loses the coupons, so still below
        // the risk-free price
        let full_recovery = bond
            .risky_clean_price(&curve, 0.02, 1.0, settlement)
            .unwrap();
        assert!(full_recovery < risk_free);
    }

    #[test]
    fn implied_hazard_round_trips() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        for hazard in [0.0, 0.008, 0.02, 0.15] {
            let clean = bond
                .risky_clean_price(&curve, hazard, 0.4, settlement)
                .unwrap();
            let implied = bond
                .implied_hazard_rate(clean, &curve, 0.4, settlement)
                .unwrap();
            assert!(
                (implied - hazard).abs() < 1e-9,
                "hazard {hazard}: {implied}"
            );
        }
    }

    #[test]
    fn credit_triangle_ties_hazard_to_the_z_spread() {
        // z-spread of the risky price ~ lambda * (1 - recovery)
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let (hazard, recovery) = (0.02, 0.4);
        let clean = bond
            .risky_clean_price(&curve, hazard, recovery, settlement)
            .unwrap();
        let z = bond.z_spread(clean, &curve, settlement).unwrap();
        let triangle = hazard * (1.0 - recovery);
        assert!(
            (z - triangle).abs() < 0.15 * triangle,
            "z {z} vs credit triangle {triangle}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        assert!(bond
            .risky_dirty_price(&curve, -0.01, 0.4, settlement)
            .is_err());
        assert!(bond
            .risky_dirty_price(&curve, 0.02, 1.5, settlement)
            .is_err());
        assert!(bond
            .implied_hazard_rate(-5.0, &curve, 0.4, settlement)
            .is_err());
        // a price no non-negative hazard can reach (above risk-free)
        let risk_free = bond
            .risky_clean_price(&curve, 0.0, 0.4, settlement)
            .unwrap();
        assert!(bond
            .implied_hazard_rate(risk_free + 5.0, &curve, 0.4, settlement)
            .is_err());
    }

    #[test]
    fn survival_integrates_the_piecewise_hazard_by_hand() {
        let credit = CreditCurve::new(&[(1.0, 0.01), (3.0, 0.03)]).unwrap();
        assert_eq!(credit.survival(0.0), 1.0);
        // inside the first segment
        assert!((credit.survival(0.5) - (-0.005_f64).exp()).abs() < 1e-15);
        // spanning both segments: 0.01*1 + 0.03*1
        assert!((credit.survival(2.0) - (-0.04_f64).exp()).abs() < 1e-15);
        // beyond the last pillar the 3% hazard extends flat
        assert!((credit.survival(5.0) - (-(0.01 + 0.06 + 0.06_f64)).exp()).abs() < 1e-15);
        // default probabilities partition survival
        let p1 = credit.default_probability(0.0, 2.0);
        let p2 = credit.default_probability(2.0, 5.0);
        assert!((p1 + p2 - (1.0 - credit.survival(5.0))).abs() < 1e-15);
        // validation
        assert!(CreditCurve::new(&[]).is_err());
        assert!(CreditCurve::new(&[(1.0, -0.01)]).is_err());
        assert!(CreditCurve::new(&[(2.0, 0.01), (1.0, 0.01)]).is_err());
    }

    #[test]
    fn flat_curve_pricing_matches_the_flat_hazard_method() {
        let bond = corporate();
        let curve = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let credit = CreditCurve::flat(0.02).unwrap();
        let via_curve = bond
            .risky_dirty_price_on_curve(&curve, &credit, 0.4, settlement)
            .unwrap();
        let via_flat = bond
            .risky_dirty_price(&curve, 0.02, 0.4, settlement)
            .unwrap();
        assert!(
            (via_curve - via_flat).abs() < 1e-12,
            "{via_curve} vs {via_flat}"
        );
    }

    #[test]
    fn bootstrap_recovers_a_generating_hazard_curve() {
        // three bonds of one issuer priced from a known upward-sloping
        // hazard curve; the bootstrap must recover it and reprice all
        let discount = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        let recovery = 0.4;
        let dc = discount.day_count();
        let bonds: Vec<FixedRateBond> = [d(2028, 5, 15), d(2031, 5, 15), d(2034, 5, 15)]
            .iter()
            .map(|&maturity| {
                FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), maturity).unwrap()
            })
            .collect();
        let truth = CreditCurve::new(&[
            (dc.year_fraction(settlement, bonds[0].maturity_date), 0.008),
            (dc.year_fraction(settlement, bonds[1].maturity_date), 0.015),
            (dc.year_fraction(settlement, bonds[2].maturity_date), 0.024),
        ])
        .unwrap();
        let quotes: Vec<(FixedRateBond, f64)> = bonds
            .iter()
            .map(|bond| {
                let clean = bond
                    .risky_clean_price_on_curve(&discount, &truth, recovery, settlement)
                    .unwrap();
                (bond.clone(), clean)
            })
            .collect();

        let bootstrapped =
            bootstrap_credit_curve(&quotes, &discount, recovery, settlement).unwrap();
        for ((_, h_true), (_, h_fit)) in truth.pillars().iter().zip(bootstrapped.pillars()) {
            assert!((h_true - h_fit).abs() < 1e-7, "{h_true} vs {h_fit}");
        }
        // and every quote reprices exactly on the bootstrapped curve
        for (bond, clean) in &quotes {
            let repriced = bond
                .risky_clean_price_on_curve(&discount, &bootstrapped, recovery, settlement)
                .unwrap();
            assert!((repriced - clean).abs() < 1e-7, "{repriced} vs {clean}");
        }
        // wider spreads at the long end imply rising hazards
        let hazards: Vec<f64> = bootstrapped.pillars().iter().map(|&(_, h)| h).collect();
        assert!(hazards.windows(2).all(|w| w[1] > w[0]), "{hazards:?}");
    }

    #[test]
    fn bootstrap_rejects_bad_inputs() {
        let discount = flat(0.04, d(2026, 8, 7));
        let settlement = d(2026, 8, 7);
        assert!(bootstrap_credit_curve(&[], &discount, 0.4, settlement).is_err());
        let bond = corporate();
        // duplicate maturities
        let duplicated = vec![(bond.clone(), 99.0), (bond.clone(), 98.0)];
        assert!(bootstrap_credit_curve(&duplicated, &discount, 0.4, settlement).is_err());
        // a price above the risk-free value cannot be bootstrapped
        let risk_free = bond
            .risky_clean_price(&discount, 0.0, 0.4, settlement)
            .unwrap();
        let rich = vec![(bond.clone(), risk_free + 5.0)];
        assert!(bootstrap_credit_curve(&rich, &discount, 0.4, settlement).is_err());
        // negative price
        let negative = vec![(bond, -1.0)];
        assert!(bootstrap_credit_curve(&negative, &discount, 0.4, settlement).is_err());
    }
}
