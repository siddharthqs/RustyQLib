//! Single-name credit default swaps.
//!
//! A CDS exchanges a running premium for protection against the
//! reference entity's default. On the [`CreditCurve`] and a risk-free
//! discount curve, with default at intensity `lambda(t)` and recovery
//! `R`:
//!
//! - the **premium leg** pays the coupon on each accrual period's end
//!   if the name survives, plus the accrued premium at default when the
//!   contract settles accrual (the standard);
//! - the **protection leg** pays `1 - R` at default.
//!
//! Both are integrated exactly on segments where the hazard and the
//! curve's forward rate are constant: with `mu = r + lambda` over a
//! segment of length `d` starting at time `a`,
//!
//! ```text
//! protection = (1 - R) df(a) S(a) lambda / mu (1 - exp(-mu d))
//! accrual    = coupon df(a) S(a) lambda [ (a - s)(1 - exp(-mu d)) / mu
//!                                       + (1 - exp(-mu d)(1 + mu d)) / mu^2 ]
//! ```
//!
//! where `s` is the period start — the ISDA standard model's integrals.
//! The segments are the accrual periods cut at the hazard pillars and
//! subdivided to at most a month, so a curve whose discount factors are
//! not log-linear between pillars is still integrated to well under a
//! basis point.
//!
//! Conventions: quarterly periods rolled back from the maturity (the
//! IMM dates when the maturity is one), a front stub from the effective
//! date, Act/360 accrual, coupons paid on the unadjusted period end.
//! The valuation date is the protection start; the ISDA one-day
//! extension of the final period and the business-day adjustment of
//! payment dates are not applied. The par spread is the coupon that
//! zeroes the value; the points upfront of a contract with a fixed
//! coupon are its value to the buyer per unit notional, and the
//! market's flat-hazard conversion between the two is
//! [`upfront_for_par_spread`](CreditDefaultSwap::upfront_for_par_spread)
//! and its inverse.

use chrono::{Months, NaiveDate};

use super::curve::CreditCurve;
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// Which side of the protection the holder is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtectionSide {
    /// Pays the premium, receives the loss on default.
    Buyer,
    /// Receives the premium, pays the loss on default.
    Seller,
}

/// The longest integration segment, in years, when cutting accrual
/// periods for the default integrals.
const MAX_SEGMENT: f64 = 1.0 / 12.0;

/// A single-name credit default swap.
#[derive(Debug, Clone)]
pub struct CreditDefaultSwap {
    pub notional: f64,
    /// Running premium as a decimal (`0.01` for the standard 100bp).
    pub coupon: f64,
    /// Recovery on default, in `[0, 1)`.
    pub recovery_rate: f64,
    /// Protection start; the valuation date of every method.
    pub effective_date: NaiveDate,
    pub maturity: NaiveDate,
    /// Months per accrual period (3 for the standard quarterly coupon).
    pub period_months: u32,
    /// Day count of the premium accrual (Act/360 standard).
    pub day_count: DayCountConvention,
    /// Whether the accrued premium is paid at default (standard).
    pub pays_accrued_on_default: bool,
    pub side: ProtectionSide,
}

impl CreditDefaultSwap {
    /// A standard contract: quarterly Act/360 premium with accrual paid
    /// on default, seen from the protection buyer.
    pub fn new(
        notional: f64,
        coupon: f64,
        recovery_rate: f64,
        effective_date: NaiveDate,
        maturity: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        let cds = CreditDefaultSwap {
            notional,
            coupon,
            recovery_rate,
            effective_date,
            maturity,
            period_months: 3,
            day_count: DayCountConvention::Act360,
            pays_accrued_on_default: true,
            side: ProtectionSide::Buyer,
        };
        cds.validate()?;
        Ok(cds)
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        let invalid = |what: String| Err(RustyQLibError::invalid_input("cds", what));
        if !(self.notional > 0.0 && self.notional.is_finite()) {
            return invalid(format!("notional must be positive, got {}", self.notional));
        }
        if !(self.coupon.is_finite() && self.coupon >= 0.0) {
            return invalid(format!("coupon must be non-negative, got {}", self.coupon));
        }
        if !(self.recovery_rate.is_finite() && (0.0..1.0).contains(&self.recovery_rate)) {
            return invalid(format!(
                "recovery must be in [0, 1), got {}",
                self.recovery_rate
            ));
        }
        if self.maturity <= self.effective_date {
            return invalid(format!(
                "maturity {} must follow the effective date {}",
                self.maturity, self.effective_date
            ));
        }
        if self.period_months == 0 {
            return invalid("the accrual period must be at least one month".to_string());
        }
        Ok(())
    }

    /// The signed multiplier for this side: the buyer's value is
    /// protection less premium.
    fn sign(&self) -> f64 {
        match self.side {
            ProtectionSide::Buyer => 1.0,
            ProtectionSide::Seller => -1.0,
        }
    }

    /// Accrual periods `(start, end)` rolled back from the maturity,
    /// with a front stub from the effective date.
    pub fn accrual_periods(&self) -> Vec<(NaiveDate, NaiveDate)> {
        let mut ends = Vec::new();
        let mut k = 0;
        loop {
            let date = self.maturity - Months::new(k * self.period_months);
            if date <= self.effective_date {
                break;
            }
            ends.push(date);
            k += 1;
        }
        ends.reverse();
        let mut periods = Vec::with_capacity(ends.len());
        let mut start = self.effective_date;
        for end in ends {
            periods.push((start, end));
            start = end;
        }
        periods
    }

    /// The two legs per unit notional on the curves, as
    /// `(risky annuity per unit coupon, protection)`.
    fn legs(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<(f64, f64), RustyQLibError> {
        if valuation < self.effective_date || valuation >= self.maturity {
            return Err(RustyQLibError::invalid_input(
                "cds",
                format!(
                    "valuation {valuation} must lie in the protection period {} to {}",
                    self.effective_date, self.maturity
                ),
            ));
        }
        let curve_dc = curve.day_count();
        let time = |date: NaiveDate| curve_dc.year_fraction(valuation, date);
        let df0 = curve.df_date(valuation);
        let df =
            |t: f64| curve.df(curve_dc.year_fraction(curve.reference_date(), valuation) + t) / df0;
        let pillars: Vec<f64> = credit.pillars().iter().map(|&(t, _)| t).collect();

        let mut annuity = 0.0;
        let mut protection = 0.0;
        for (start_date, end_date) in self.accrual_periods() {
            if end_date <= valuation {
                continue;
            }
            // the period from the valuation date on
            let accrual_start = start_date.max(valuation);
            let (s, e) = (time(accrual_start), time(end_date));
            let period_fraction = self.day_count.year_fraction(accrual_start, end_date);
            // coupon on survival to the period end
            annuity += period_fraction * df(e) * credit.survival(e);

            // the default integrals on segments of constant hazard and rate
            let mut cuts: Vec<f64> = vec![s];
            for &p in &pillars {
                if p > s + 1e-12 && p < e - 1e-12 {
                    cuts.push(p);
                }
            }
            cuts.push(e);
            cuts.sort_by(|a, b| a.total_cmp(b));
            let mut grid: Vec<f64> = Vec::new();
            for w in cuts.windows(2) {
                let n = ((w[1] - w[0]) / MAX_SEGMENT).ceil().max(1.0) as usize;
                for i in 0..n {
                    grid.push(w[0] + (w[1] - w[0]) * i as f64 / n as f64);
                }
            }
            grid.push(e);
            for w in grid.windows(2) {
                let (a, b) = (w[0], w[1]);
                let d = b - a;
                if d <= 0.0 {
                    continue;
                }
                let (dfa, dfb) = (df(a), df(b));
                let (sa, sb) = (credit.survival(a), credit.survival(b));
                if sa <= 0.0 || dfa <= 0.0 {
                    continue;
                }
                let r = -(dfb / dfa).ln() / d;
                let lambda = -(sb / sa).ln() / d;
                if lambda <= 0.0 {
                    continue;
                }
                let mu = r + lambda;
                let (decay, second) = if mu.abs() * d < 1e-8 {
                    (d, 0.5 * d * d)
                } else {
                    let x = (-mu * d).exp();
                    ((1.0 - x) / mu, (1.0 - x * (1.0 + mu * d)) / (mu * mu))
                };
                let weight = dfa * sa * lambda;
                protection += weight * decay;
                if self.pays_accrued_on_default {
                    // accrued from the period start to the default time,
                    // scaled to the period's accrual fraction
                    let accrued = period_fraction / (e - s) * ((a - s) * decay + second);
                    annuity += weight * accrued;
                }
            }
        }
        Ok((annuity, (1.0 - self.recovery_rate) * protection))
    }

    /// The risky annuity per unit notional: the value of one unit of
    /// running coupon (the risky PV01 in price terms).
    pub fn risky_annuity(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.legs(curve, credit, valuation)?.0)
    }

    /// Present value of the premium leg.
    pub fn premium_leg_pv(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.notional * self.coupon * self.risky_annuity(curve, credit, valuation)?)
    }

    /// Present value of the protection leg.
    pub fn protection_leg_pv(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.notional * self.legs(curve, credit, valuation)?.1)
    }

    /// The coupon at which the contract is worth zero.
    pub fn par_spread(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, protection) = self.legs(curve, credit, valuation)?;
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(
                "the risky annuity is not positive".to_string(),
            ));
        }
        Ok(protection / annuity)
    }

    /// Value to this side: protection less premium for the buyer.
    pub fn npv(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, protection) = self.legs(curve, credit, valuation)?;
        Ok(self.sign() * self.notional * (protection - self.coupon * annuity))
    }

    /// Points upfront per unit notional: the buyer's value of the
    /// contract at its fixed coupon on these curves, positive when the
    /// buyer pays to enter.
    pub fn points_upfront(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, protection) = self.legs(curve, credit, valuation)?;
        Ok(protection - self.coupon * annuity)
    }

    /// The first-order value change for a one-basis-point rise in the
    /// par spread, signed for this side: the risky annuity per basis
    /// point on the notional (the buyer gains as spreads widen).
    pub fn risky_pv01(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.sign() * self.notional * self.risky_annuity(curve, credit, valuation)? * 1e-4)
    }

    /// The flat hazard rate at which this contract's par spread equals
    /// `par_spread` — the market's single-curve conversion.
    pub fn flat_hazard_for_par_spread(
        &self,
        curve: &YieldCurve,
        valuation: NaiveDate,
        par_spread: f64,
    ) -> Result<f64, RustyQLibError> {
        if !(par_spread > 0.0 && par_spread.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cds",
                format!("the par spread must be positive, got {par_spread}"),
            ));
        }
        let spread_at = |hazard: f64| -> Result<f64, RustyQLibError> {
            self.par_spread(curve, &CreditCurve::flat(hazard)?, valuation)
        };
        // the par spread rises with the hazard; check the bracket first
        spread_at(1e-9)?;
        spread_at(10.0)?;
        let objective =
            |hazard: f64| spread_at(hazard).expect("the hazard bracket was priced") - par_spread;
        let root = Solver1d::new(1e-12, 200).bisection(objective, 1e-9, 10.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "flat hazard for the par spread did not converge".to_string(),
            });
        }
        Ok(root.x)
    }

    /// Points upfront the buyer pays for this contract, at its fixed
    /// coupon, when the market quotes the maturity at `par_spread`:
    /// the flat-hazard conversion.
    pub fn upfront_for_par_spread(
        &self,
        curve: &YieldCurve,
        valuation: NaiveDate,
        par_spread: f64,
    ) -> Result<f64, RustyQLibError> {
        let hazard = self.flat_hazard_for_par_spread(curve, valuation, par_spread)?;
        self.points_upfront(curve, &CreditCurve::flat(hazard)?, valuation)
    }

    /// The par spread the market implies when this contract, at its
    /// fixed coupon, trades at `upfront` points: the inverse of
    /// [`upfront_for_par_spread`](Self::upfront_for_par_spread).
    pub fn par_spread_for_upfront(
        &self,
        curve: &YieldCurve,
        valuation: NaiveDate,
        upfront: f64,
    ) -> Result<f64, RustyQLibError> {
        if !upfront.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "cds",
                "the upfront must be finite",
            ));
        }
        let upfront_at = |hazard: f64| -> Result<f64, RustyQLibError> {
            self.points_upfront(curve, &CreditCurve::flat(hazard)?, valuation)
        };
        upfront_at(1e-9)?;
        upfront_at(10.0)?;
        // the upfront rises with the hazard (more protection value)
        let objective =
            |hazard: f64| upfront_at(hazard).expect("the hazard bracket was priced") - upfront;
        let root = Solver1d::new(1e-12, 200).bisection(objective, 1e-9, 10.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "par spread for the upfront did not converge".to_string(),
            });
        }
        self.par_spread(curve, &CreditCurve::flat(root.x)?, valuation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat_curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(2026, 8, 14),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// 10mm, 100bp, 40% recovery, protection from the valuation date to
    /// the June 2031 IMM date.
    fn standard() -> CreditDefaultSwap {
        CreditDefaultSwap::new(10_000_000.0, 0.01, 0.40, d(2026, 8, 14), d(2031, 6, 20)).unwrap()
    }

    #[test]
    fn periods_roll_back_from_maturity_with_a_front_stub() {
        let periods = standard().accrual_periods();
        assert_eq!(periods[0], (d(2026, 8, 14), d(2026, 9, 20)));
        assert_eq!(periods[1], (d(2026, 9, 20), d(2026, 12, 20)));
        assert_eq!(periods.last().unwrap().1, d(2031, 6, 20));
        assert_eq!(periods.len(), 20);
        for w in periods.windows(2) {
            assert_eq!(w[0].1, w[1].0);
        }
    }

    /// QuantLib 1.43 `IsdaCdsEngine` (no accrual bias, piecewise
    /// forwards) on the same contract, flat 4% continuous Act/365
    /// discount and flat 2% hazard: NPV 80420.88, fair spread
    /// 118.9345bp, coupon leg 424732.07, default leg 505152.95.
    #[test]
    fn matches_quantlib_isda_engine() {
        let cds = standard();
        let curve = flat_curve(0.04);
        let credit = CreditCurve::flat(0.02).unwrap();
        let valuation = d(2026, 8, 14);
        let par = cds.par_spread(&curve, &credit, valuation).unwrap();
        assert!((par * 1e4 - 118.9345).abs() < 0.05, "{}", par * 1e4);
        let npv = cds.npv(&curve, &credit, valuation).unwrap();
        assert!((npv - 80420.88).abs() < 60.0, "{npv}");
        let protection = cds.protection_leg_pv(&curve, &credit, valuation).unwrap();
        assert!((protection - 505152.95).abs() < 60.0, "{protection}");
        let premium = cds.premium_leg_pv(&curve, &credit, valuation).unwrap();
        assert!((premium - 424732.07).abs() < 60.0, "{premium}");
    }

    #[test]
    fn protection_leg_matches_the_closed_form_on_flat_curves() {
        let cds = standard();
        let curve = flat_curve(0.04);
        let credit = CreditCurve::flat(0.02).unwrap();
        let valuation = d(2026, 8, 14);
        let t = DayCountConvention::Act365.year_fraction(valuation, d(2031, 6, 20));
        let mu: f64 = 0.06;
        let expected = 10_000_000.0 * 0.6 * 0.02 / mu * (1.0 - (-mu * t).exp());
        let protection = cds.protection_leg_pv(&curve, &credit, valuation).unwrap();
        assert!(
            (protection - expected).abs() < 1e-3,
            "{protection} vs {expected}"
        );
    }

    #[test]
    fn credit_triangle_limits_and_sides() {
        let cds = standard();
        let curve = flat_curve(0.04);
        let valuation = d(2026, 8, 14);
        // par spread ~ hazard x loss, a shade below with quarterly premium
        let par = cds
            .par_spread(&curve, &CreditCurve::flat(0.02).unwrap(), valuation)
            .unwrap();
        assert!((par - 0.012).abs() < 0.0002 && par < 0.012, "{par}");
        // no default risk: no protection value, a risk-free annuity
        let riskless = CreditCurve::flat(0.0).unwrap();
        assert_eq!(
            cds.protection_leg_pv(&curve, &riskless, valuation).unwrap(),
            0.0
        );
        let annuity = cds.risky_annuity(&curve, &riskless, valuation).unwrap();
        let df0 = curve.df_date(valuation);
        let riskfree: f64 = cds
            .accrual_periods()
            .iter()
            .map(|&(s, e)| DayCountConvention::Act360.year_fraction(s, e) * curve.df_date(e) / df0)
            .sum();
        assert!((annuity - riskfree).abs() < 1e-12);
        // more hazard, wider; more recovery, tighter
        let wider = cds
            .par_spread(&curve, &CreditCurve::flat(0.04).unwrap(), valuation)
            .unwrap();
        assert!(wider > par);
        let mut generous = standard();
        generous.recovery_rate = 0.6;
        assert!(
            generous
                .par_spread(&curve, &CreditCurve::flat(0.02).unwrap(), valuation)
                .unwrap()
                < par
        );
        // the seller holds the mirror image
        let credit = CreditCurve::flat(0.02).unwrap();
        let mut seller = standard();
        seller.side = ProtectionSide::Seller;
        assert!(
            (seller.npv(&curve, &credit, valuation).unwrap()
                + cds.npv(&curve, &credit, valuation).unwrap())
            .abs()
                < 1e-9
        );
        // a contract struck at par is worth nothing to either side
        let mut at_par = standard();
        at_par.coupon = par;
        assert!(at_par.npv(&curve, &credit, valuation).unwrap().abs() < 1e-6);
    }

    #[test]
    fn upfront_and_par_spread_conversions_round_trip() {
        let cds = standard();
        let curve = flat_curve(0.04);
        let valuation = d(2026, 8, 14);
        let hazard = cds
            .flat_hazard_for_par_spread(&curve, valuation, 0.015)
            .unwrap();
        let par = cds
            .par_spread(&curve, &CreditCurve::flat(hazard).unwrap(), valuation)
            .unwrap();
        assert!((par - 0.015).abs() < 1e-10, "{par}");
        // 150bp quoted against a 100bp coupon: the buyer pays upfront
        let upfront = cds
            .upfront_for_par_spread(&curve, valuation, 0.015)
            .unwrap();
        assert!(upfront > 0.0 && upfront < 0.05, "{upfront}");
        let back = cds
            .par_spread_for_upfront(&curve, valuation, upfront)
            .unwrap();
        assert!((back - 0.015).abs() < 1e-8, "{back}");
        // a quote at the coupon costs nothing upfront
        assert!(
            cds.upfront_for_par_spread(&curve, valuation, 0.01)
                .unwrap()
                .abs()
                < 1e-9
        );
        // and the buyer receives upfront when the quote is inside the coupon
        assert!(
            cds.upfront_for_par_spread(&curve, valuation, 0.005)
                .unwrap()
                < 0.0
        );
    }

    #[test]
    fn risky_pv01_and_validation() {
        let cds = standard();
        let curve = flat_curve(0.04);
        let credit = CreditCurve::flat(0.02).unwrap();
        let valuation = d(2026, 8, 14);
        let pv01 = cds.risky_pv01(&curve, &credit, valuation).unwrap();
        let annuity = cds.risky_annuity(&curve, &credit, valuation).unwrap();
        assert!((pv01 - 10_000_000.0 * annuity * 1e-4).abs() < 1e-9 && pv01 > 0.0);
        let mut seller = standard();
        seller.side = ProtectionSide::Seller;
        assert!(seller.risky_pv01(&curve, &credit, valuation).unwrap() < 0.0);
        // validation
        assert!(CreditDefaultSwap::new(-1.0, 0.01, 0.4, d(2026, 8, 14), d(2031, 6, 20)).is_err());
        assert!(CreditDefaultSwap::new(1.0, -0.01, 0.4, d(2026, 8, 14), d(2031, 6, 20)).is_err());
        assert!(CreditDefaultSwap::new(1.0, 0.01, 1.0, d(2026, 8, 14), d(2031, 6, 20)).is_err());
        assert!(CreditDefaultSwap::new(1.0, 0.01, 0.4, d(2031, 6, 20), d(2026, 8, 14)).is_err());
        assert!(cds.npv(&curve, &credit, d(2032, 1, 1)).is_err());
        assert!(cds
            .flat_hazard_for_par_spread(&curve, valuation, -0.01)
            .is_err());
    }
}
