//! Floating-for-floating (basis) swap.
//!
//! Receive leg A's floating rate plus `spread`, pay leg B's floating
//! rate. Each leg has its own frequency, day count and forecast curve —
//! the classic tenor-basis (3M vs 6M) or index-basis (IBOR vs OIS)
//! trade. Under a single forecast curve with matching legs both sides
//! telescope to the same value, so the fair spread is zero.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::{accrual_periods, annuity, float_leg_pv, AccrualPeriod};

/// One floating leg's conventions.
#[derive(Debug, Clone, Copy)]
pub struct BasisSwapLeg {
    pub frequency: Frequency,
    pub day_count: DayCountConvention,
}

/// A basis swap from the point of view of the party **receiving leg A
/// plus the spread** and paying leg B.
#[derive(Debug, Clone)]
pub struct BasisSwap {
    pub notional: f64,
    /// Spread added to leg A's floating rate (e.g. `0.0025` = 25bp).
    pub spread: f64,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub leg_a: BasisSwapLeg,
    pub leg_b: BasisSwapLeg,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
}

impl BasisSwap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        spread: f64,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        leg_a: BasisSwapLeg,
        leg_b: BasisSwapLeg,
        calendar: Calendar,
        convention: BusinessDayConvention,
    ) -> Result<Self, RustyQLibError> {
        if !notional.is_finite() || notional <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "basis swap",
                format!("notional must be positive, got {notional}"),
            ));
        }
        if !spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "basis swap",
                format!("spread must be finite, got {spread}"),
            ));
        }
        if maturity_date <= effective_date {
            return Err(RustyQLibError::invalid_input(
                "basis swap",
                format!("maturity {maturity_date} must be after effective {effective_date}"),
            ));
        }
        Ok(BasisSwap {
            notional,
            spread,
            effective_date,
            maturity_date,
            leg_a,
            leg_b,
            calendar,
            convention,
        })
    }

    fn periods(&self, leg: &BasisSwapLeg) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        accrual_periods(
            self.effective_date,
            self.maturity_date,
            leg.frequency,
            &self.calendar,
            self.convention,
            0,
        )
    }

    /// PV of one leg on unit notional with the given spread.
    fn leg_pv(
        &self,
        leg: &BasisSwapLeg,
        spread: f64,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        float_leg_pv(
            &self.periods(leg)?,
            spread,
            leg.day_count,
            discount,
            forecast,
        )
    }

    /// Swap PV: receive leg A (+ spread) forecast on `forecast_a`, pay
    /// leg B forecast on `forecast_b`, both discounted on `discount`.
    pub fn pv(
        &self,
        discount: &YieldCurve,
        forecast_a: &YieldCurve,
        forecast_b: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let receive = self.leg_pv(&self.leg_a, self.spread, discount, forecast_a)?;
        let pay = self.leg_pv(&self.leg_b, 0.0, discount, forecast_b)?;
        Ok(self.notional * (receive - pay))
    }

    /// The fair spread on leg A: the spread that makes the PV zero.
    pub fn fair_spread(
        &self,
        discount: &YieldCurve,
        forecast_a: &YieldCurve,
        forecast_b: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let a_flat = self.leg_pv(&self.leg_a, 0.0, discount, forecast_a)?;
        let b = self.leg_pv(&self.leg_b, 0.0, discount, forecast_b)?;
        let annuity = annuity(&self.periods(&self.leg_a)?, self.leg_a.day_count, discount);
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive leg A annuity {annuity}"
            )));
        }
        Ok((b - a_flat) / annuity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

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

    fn quarterly_leg() -> BasisSwapLeg {
        BasisSwapLeg {
            frequency: Frequency::Quarterly,
            day_count: DayCountConvention::Act360,
        }
    }

    fn two_year_basis(spread: f64, leg_b: BasisSwapLeg) -> BasisSwap {
        BasisSwap::new(
            1_000_000.0,
            spread,
            d(2026, 8, 6),
            d(2028, 8, 6),
            quarterly_leg(),
            leg_b,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
        .unwrap()
    }

    #[test]
    fn identical_legs_on_one_curve_have_zero_fair_spread() {
        let curve = flat(0.04, d(2026, 8, 6));
        let swap = two_year_basis(0.0, quarterly_leg());
        assert!(swap.pv(&curve, &curve, &curve).unwrap().abs() < 1e-10);
        let fair = swap.fair_spread(&curve, &curve, &curve).unwrap();
        assert!(fair.abs() < 1e-14, "fair spread {fair}");
    }

    #[test]
    fn fair_spread_recovers_the_forecast_curve_gap() {
        // leg B forecasts 25bp above leg A: receiving A needs ~25bp
        let reference = d(2026, 8, 6);
        let discount = flat(0.04, reference);
        let forecast_a = flat(0.04, reference);
        let forecast_b = flat(0.0425, reference);
        let swap = two_year_basis(0.0, quarterly_leg());
        let fair = swap
            .fair_spread(&discount, &forecast_a, &forecast_b)
            .unwrap();
        assert!((fair - 0.0025).abs() < 2e-4, "fair spread {fair}");
        // priced at the fair spread the swap is worth zero
        let mut at_fair = swap.clone();
        at_fair.spread = fair;
        let pv = at_fair.pv(&discount, &forecast_a, &forecast_b).unwrap();
        assert!(pv.abs() < 1e-8, "pv {pv}");
        // and with no spread, receiving the lower-forecast leg loses money
        assert!(swap.pv(&discount, &forecast_a, &forecast_b).unwrap() < 0.0);
    }

    #[test]
    fn tenor_basis_legs_can_differ_in_frequency() {
        // quarterly vs semiannual legs on one curve: both telescope to
        // the same 1 - df(T) (unadjusted, no lag), so the fair spread of
        // the mixed-frequency swap stays tiny
        let reference = d(2026, 8, 6);
        let curve = flat(0.04, reference);
        let semi = BasisSwapLeg {
            frequency: Frequency::Semiannual,
            day_count: DayCountConvention::Act360,
        };
        let swap = BasisSwap::new(
            1_000_000.0,
            0.0,
            reference,
            d(2028, 8, 6),
            quarterly_leg(),
            semi,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
        )
        .unwrap();
        let fair = swap.fair_spread(&curve, &curve, &curve).unwrap();
        assert!(fair.abs() < 1e-12, "fair spread {fair}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let e = d(2026, 8, 6);
        let leg = quarterly_leg();
        let cal = Calendar::WeekendsOnly;
        let conv = BusinessDayConvention::Following;
        assert!(BasisSwap::new(0.0, 0.0, e, d(2028, 8, 6), leg, leg, cal.clone(), conv).is_err());
        assert!(
            BasisSwap::new(1e6, f64::NAN, e, d(2028, 8, 6), leg, leg, cal.clone(), conv).is_err()
        );
        assert!(BasisSwap::new(1e6, 0.0, e, e, leg, leg, cal, conv).is_err());
    }
}
