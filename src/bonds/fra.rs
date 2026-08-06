//! Forward rate agreement.
//!
//! An agreement to exchange a pre-agreed simple `fix_rate` for the
//! floating rate that sets on `start_date`, over the period from
//! `start_date` to `maturity_date`. In curve building an FRA quote pins
//! the forward discount factor over its period, extending the curve one
//! pillar beyond the deposits.

use chrono::NaiveDate;

use crate::bonds::{df_to_start, CurveInstrument};
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

#[derive(Debug, Clone)]
pub struct Fra {
    pub start_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub notional: f64,
    pub fix_rate: f64,
    pub day_count: DayCountConvention,
}

impl Fra {
    pub fn new(
        start_date: NaiveDate,
        maturity_date: NaiveDate,
        notional: f64,
        fix_rate: f64,
        day_count: DayCountConvention,
    ) -> Result<Self, RustyQLibError> {
        if maturity_date <= start_date {
            return Err(RustyQLibError::invalid_input(
                "fra",
                format!("maturity {maturity_date} must be after start {start_date}"),
            ));
        }
        if !notional.is_finite() || notional <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "fra",
                format!("notional must be positive, got {notional}"),
            ));
        }
        if !fix_rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "fra",
                format!("fix_rate must be finite, got {fix_rate}"),
            ));
        }
        Ok(Fra {
            start_date,
            maturity_date,
            notional,
            fix_rate,
            day_count,
        })
    }

    /// Accrual year fraction of the FRA period under its own day count.
    pub fn accrual(&self) -> f64 {
        self.day_count
            .year_fraction(self.start_date, self.maturity_date)
    }

    /// Discount factor over the FRA period implied by the fixed rate:
    /// `1 / (1 + r * tau)`.
    pub fn period_df(&self) -> f64 {
        1.0 / (1.0 + self.fix_rate * self.accrual())
    }

    /// The simple forward rate over the FRA period read off `curve` —
    /// the fair fixed rate for this FRA on that curve.
    pub fn forward_rate(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        let dc = curve.day_count();
        let t1 = dc.year_fraction(curve.reference_date(), self.start_date);
        let t2 = dc.year_fraction(curve.reference_date(), self.maturity_date);
        Ok(curve.forward_rate_with(t1, t2, Compounding::Simple)?)
    }
}

impl CurveInstrument for Fra {
    fn maturity_date(&self) -> NaiveDate {
        self.maturity_date
    }

    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError> {
        let df_start = df_to_start("fra", self.start_date, reference_date, curve_so_far)?;
        Ok(df_start * self.period_df())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{InterpolationMethod, Tenor};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn implied_df_chains_off_the_partial_curve() {
        let reference = d(2026, 1, 15);
        let start = d(2026, 4, 15);
        // short curve: one pillar at the FRA start with df = 0.99
        let curve = YieldCurve::from_discount_factors(
            &[Tenor::Date(start)],
            &[0.99],
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        // 3M FRA at 6% Act/360 over 91 days
        let fra = Fra::new(
            start,
            d(2026, 7, 15),
            1_000_000.0,
            0.06,
            DayCountConvention::Act360,
        )
        .unwrap();
        let tau = 91.0 / 360.0;
        let expected = 0.99 / (1.0 + 0.06 * tau);
        let df = fra.implied_df(reference, Some(&curve)).unwrap();
        assert!(
            (df - expected).abs() < 1e-12,
            "df={df}, expected={expected}"
        );
    }

    #[test]
    fn fair_forward_rate_recovers_the_quote() {
        let reference = d(2026, 1, 15);
        let start = d(2026, 4, 15);
        let end = d(2026, 7, 15);
        let fra = Fra::new(start, end, 1e6, 0.06, DayCountConvention::Act365).unwrap();
        // build a two-pillar curve consistent with df(start)=0.99 and the
        // FRA's own implied maturity df, then read the forward back
        let curve_short = YieldCurve::from_discount_factors(
            &[Tenor::Date(start)],
            &[0.99],
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let df_end = fra.implied_df(reference, Some(&curve_short)).unwrap();
        let curve = YieldCurve::from_discount_factors(
            &[Tenor::Date(start), Tenor::Date(end)],
            &[0.99, df_end],
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let fwd = fra.forward_rate(&curve).unwrap();
        assert!((fwd - 0.06).abs() < 1e-10, "fwd={fwd}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let dc = DayCountConvention::Act360;
        assert!(Fra::new(d(2026, 7, 15), d(2026, 4, 15), 1e6, 0.06, dc).is_err());
        assert!(Fra::new(d(2026, 4, 15), d(2026, 7, 15), -1.0, 0.06, dc).is_err());
    }
}
