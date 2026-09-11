//! Calibrating the equity-linked hazard level `a(t)` to a CDS curve
//! through the model.
//!
//! Under `lambda(S, t) = a(t) (S0 / S)^p` the unconditional survival to
//! `T` is an expectation over share-price paths,
//! `E[exp(-integral of lambda)]`, which the grid computes as the value
//! of a **survival claim**: a zero-coupon, zero-recovery claim that pays
//! one at `T` if the name survives, priced on the risk-free curve and
//! divided by the discount factor (rates are deterministic, so
//! discounting factors out). For `p > 0` that survival sits below
//! `exp(-integral of a)` because the hazard is convex in the share
//! price, so the level cannot be read off the CDS curve — it comes out
//! below the curve's hazards — and is solved for instead.
//! [`EquityLinkedHazardMarket::calibrated_to`] takes the CDS
//! curve's pillars in order and, on each segment, bisects the level
//! until the model's survival to the pillar matches the curve's — the
//! credit analogue of the discount-curve bootstrap, with a grid solve
//! per trial. At `p = 0` it returns the CDS hazards themselves (to the
//! grid's time-stepping accuracy).

use chrono::{Days, NaiveDate};

use super::fd::{valuation, ConvertibleFdGrid, FdVolModel};
use super::models::{EquityLinkedHazardMarket, HazardLevel};
use super::ConvertibleBond;
use crate::bonds::FixedRateBond;
use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::credit::CreditCurve;

/// The widest hazard level the calibration searches, per year.
const MAX_LEVEL: f64 = 5.0;

impl EquityLinkedHazardMarket {
    /// The model's unconditional survival probability from `settlement`
    /// to `horizon` (years on the curve's day count): a survival claim
    /// priced on `grid`.
    pub fn survival_probability(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        horizon: f64,
        grid: ConvertibleFdGrid,
    ) -> Result<f64, RustyQLibError> {
        if !(horizon > 0.0 && horizon.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "calibration",
                format!("the horizon must be positive, got {horizon}"),
            ));
        }
        // the claim: a zero-coupon bond over the horizon whose
        // conversion right never opens, with no recovery, paying on the
        // horizon date itself (no business-day adjustment, so the
        // survival is to the horizon exactly)
        let days = (horizon * 365.0).round().max(1.0) as u64;
        let maturity = settlement
            .checked_add_days(Days::new(days))
            .ok_or_else(|| RustyQLibError::invalid_input("calibration", "date overflow"))?;
        let bond = FixedRateBond::new(
            100.0,
            0.0,
            Frequency::Annual,
            settlement,
            maturity,
            DayCountConvention::Act365,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
            0,
            false,
        )?;
        let mut claim = ConvertibleBond::new(bond, 1.0)?;
        claim.convert_from = Some(maturity);
        claim.convert_until = Some(settlement);
        let mut market = self.clone();
        market.recovery_rate = 0.0;
        let value = valuation(&claim, &market, curve, settlement, grid, &FdVolModel::Flat)?;
        let payment = claim
            .bond
            .cashflows()
            .last()
            .map(|cf| cf.payment_date)
            .unwrap_or(maturity);
        let df = curve.df_date(payment) / curve.df_date(settlement);
        Ok(value.dirty_price / 100.0 / df)
    }

    /// The model with its level `a(t)` re-solved, pillar by pillar, so
    /// that the survival probabilities to the CDS curve's pillars match
    /// the curve's. The elasticity, reference price and the other
    /// inputs are kept; the level becomes a term structure on the
    /// curve's pillars (a flat curve gives one pillar).
    pub fn calibrated_to(
        &self,
        credit: &CreditCurve,
        curve: &YieldCurve,
        settlement: NaiveDate,
        grid: ConvertibleFdGrid,
    ) -> Result<Self, RustyQLibError> {
        self.validate()?;
        let mut pillars: Vec<(f64, f64)> = Vec::new();
        for (time, _) in credit.pillars() {
            // the claim's maturity is a whole number of days, so the
            // target is the curve's survival to that exact horizon
            let horizon = (time * 365.0).round().max(1.0) / 365.0;
            let target = credit.survival(horizon);
            let survival_at = |level: f64| -> Result<f64, RustyQLibError> {
                let mut trial = pillars.clone();
                trial.push((time, level));
                let market = EquityLinkedHazardMarket {
                    hazard: HazardLevel::Term(CreditCurve::new(&trial)?),
                    ..self.clone()
                };
                market.survival_probability(curve, settlement, horizon, grid)
            };
            survival_at(0.0)?;
            survival_at(MAX_LEVEL)?;
            // survival falls as the level rises
            let objective =
                |level: f64| target - survival_at(level).expect("the level bracket was priced");
            let root = Solver1d::new(1e-12, 200).bisection(objective, 0.0, MAX_LEVEL)?;
            if !root.converged {
                return Err(RustyQLibError::CalibrationFailed {
                    iterations: root.iterations,
                    residual: objective(root.x).abs(),
                    reason: format!("the hazard level at {time:.3}y did not converge"),
                });
            }
            pillars.push((time, root.x));
        }
        Ok(EquityLinkedHazardMarket {
            hazard: HazardLevel::Term(CreditCurve::new(&pillars)?),
            ..self.clone()
        })
    }
}
