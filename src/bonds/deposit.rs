//! Money-market deposit.
//!
//! An agreement to place `notional` at `start_date` and receive principal
//! plus simple interest at `fix_rate` (under `day_count`) back at
//! `maturity_date`. Deposits quote the very short end of a discount curve.

use chrono::NaiveDate;

use crate::bonds::{df_to_start, CurveInstrument};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

#[derive(Debug, Clone)]
pub struct Deposit {
    pub start_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub notional: f64,
    pub fix_rate: f64,
    pub day_count: DayCountConvention,
}

impl Deposit {
    pub fn new(
        start_date: NaiveDate,
        maturity_date: NaiveDate,
        notional: f64,
        fix_rate: f64,
        day_count: DayCountConvention,
    ) -> Result<Self, RustyQLibError> {
        if maturity_date <= start_date {
            return Err(RustyQLibError::invalid_input(
                "deposit",
                format!("maturity {maturity_date} must be after start {start_date}"),
            ));
        }
        if !notional.is_finite() || notional <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "deposit",
                format!("notional must be positive, got {notional}"),
            ));
        }
        if !fix_rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "deposit",
                format!("fix_rate must be finite, got {fix_rate}"),
            ));
        }
        Ok(Deposit {
            start_date,
            maturity_date,
            notional,
            fix_rate,
            day_count,
        })
    }

    /// Accrual year fraction from start to maturity under the deposit's
    /// own day count.
    pub fn accrual(&self) -> f64 {
        self.day_count
            .year_fraction(self.start_date, self.maturity_date)
    }

    /// Principal plus interest repaid at maturity.
    pub fn maturity_value(&self) -> f64 {
        self.notional * (1.0 + self.fix_rate * self.accrual())
    }

    /// Discount factor over the deposit period implied by the fixed rate:
    /// `1 / (1 + r * tau)`.
    pub fn period_df(&self) -> f64 {
        1.0 / (1.0 + self.fix_rate * self.accrual())
    }

    /// Present value of the maturity repayment discounted on `curve`.
    pub fn pv(&self, curve: &YieldCurve) -> f64 {
        self.maturity_value() * curve.df_date(self.maturity_date)
    }
}

impl CurveInstrument for Deposit {
    fn maturity_date(&self) -> NaiveDate {
        self.maturity_date
    }

    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError> {
        let df_start = df_to_start("deposit", self.start_date, reference_date, curve_so_far)?;
        Ok(df_start * self.period_df())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn spot_deposit() -> Deposit {
        // 90-day deposit at 5% Act/360: tau = 0.25
        Deposit::new(
            d(2026, 1, 15),
            d(2026, 4, 15),
            1_000_000.0,
            0.05,
            DayCountConvention::Act360,
        )
        .unwrap()
    }

    #[test]
    fn accrual_and_period_df() {
        let dep = spot_deposit();
        assert!((dep.accrual() - 0.25).abs() < 1e-12);
        assert!((dep.period_df() - 1.0 / 1.0125).abs() < 1e-12);
        assert!((dep.maturity_value() - 1_012_500.0).abs() < 1e-6);
    }

    #[test]
    fn spot_start_needs_no_curve() {
        let dep = spot_deposit();
        let df = dep.implied_df(d(2026, 1, 15), None).unwrap();
        assert!((df - dep.period_df()).abs() < 1e-15);
    }

    #[test]
    fn forward_start_without_curve_is_an_error() {
        let dep = spot_deposit();
        assert!(dep.implied_df(d(2026, 1, 1), None).is_err());
    }

    #[test]
    fn start_before_reference_is_an_error() {
        let dep = spot_deposit();
        assert!(dep.implied_df(d(2026, 2, 1), None).is_err());
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let dc = DayCountConvention::Act360;
        assert!(Deposit::new(d(2026, 4, 15), d(2026, 1, 15), 1e6, 0.05, dc).is_err());
        assert!(Deposit::new(d(2026, 1, 15), d(2026, 4, 15), 0.0, 0.05, dc).is_err());
        assert!(Deposit::new(d(2026, 1, 15), d(2026, 4, 15), 1e6, f64::NAN, dc).is_err());
    }
}
