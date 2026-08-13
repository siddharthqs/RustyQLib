//! Floating rate note (FRN) with discount-margin analytics.
//!
//! An FRN pays `index + quoted_margin` each period. The market's price
//! measure is the **discount margin**: the constant spread over the
//! index that, used for discounting period by period, reproduces the
//! traded price. Future index fixings are projected as simple forwards
//! from the index curve; the current period's rate is already fixed and
//! can be supplied via `current_coupon`.
//!
//! Pricing is the standard street recursion, backward from redemption:
//!
//! ```text
//! V_n = 100
//! V_{i-1} = (V_i + 100*(f_i + qm)*tau_i) / (1 + (f_i + dm)*tau_i)
//! dirty   = (V_k + coupon_k) / (1 + (r_k + dm)*tau_remaining)
//! ```
//!
//! The defining property (tested): on a reset date with `dm = qm` the
//! note prices at exactly par, whatever the level or shape of rates —
//! that is what "floating" means.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::rates::leg::{accrual_periods, AccrualPeriod};

/// Search bracket for the discount-margin solve.
const MARGIN_BRACKET: (f64, f64) = (-0.5, 3.0);

#[derive(Debug, Clone)]
pub struct FloatingRateNote {
    pub face_value: f64,
    /// Contractual spread over the index paid in each coupon.
    pub quoted_margin: f64,
    /// Interest accrual start.
    pub dated_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub frequency: Frequency,
    pub day_count: DayCountConvention,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
    /// The full coupon rate (index fixing + quoted margin) already set
    /// for the current interest period; `None` projects it from the
    /// curve like the later periods.
    pub current_coupon: Option<f64>,
}

impl FloatingRateNote {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        face_value: f64,
        quoted_margin: f64,
        dated_date: NaiveDate,
        maturity_date: NaiveDate,
        frequency: Frequency,
        day_count: DayCountConvention,
        calendar: Calendar,
        convention: BusinessDayConvention,
        current_coupon: Option<f64>,
    ) -> Result<Self, RustyQLibError> {
        if !face_value.is_finite() || face_value <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "frn",
                format!("face value must be positive, got {face_value}"),
            ));
        }
        if !quoted_margin.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "frn",
                format!("quoted margin must be finite, got {quoted_margin}"),
            ));
        }
        if maturity_date <= dated_date {
            return Err(RustyQLibError::invalid_input(
                "frn",
                format!("maturity {maturity_date} must be after the dated date {dated_date}"),
            ));
        }
        Ok(FloatingRateNote {
            face_value,
            quoted_margin,
            dated_date,
            maturity_date,
            frequency,
            day_count,
            calendar,
            convention,
            current_coupon,
        })
    }

    /// A USD-style FRN: quarterly Act/360 coupons, modified following on
    /// the US bond-market calendar.
    pub fn usd_standard(
        face_value: f64,
        quoted_margin: f64,
        dated_date: NaiveDate,
        maturity_date: NaiveDate,
        current_coupon: Option<f64>,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            face_value,
            quoted_margin,
            dated_date,
            maturity_date,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            current_coupon,
        )
    }

    /// The note's accrual periods (business-day adjusted, like swaps).
    pub fn periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        accrual_periods(
            self.dated_date,
            self.maturity_date,
            self.frequency,
            &self.calendar,
            self.convention,
            0,
        )
    }

    /// Simple forward of the index over one accrual period, on the
    /// note's day count.
    fn forward(&self, curve: &YieldCurve, period: &AccrualPeriod) -> Result<f64, RustyQLibError> {
        let tau = self.day_count.year_fraction(period.start, period.end);
        if tau <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "degenerate accrual period ending {}",
                period.end
            )));
        }
        let df_start = curve.df_date(period.start);
        let df_end = curve.df_date(period.end);
        if !df_start.is_finite() || df_start <= 0.0 || !df_end.is_finite() || df_end <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive discount factor in the period ending {}",
                period.end
            )));
        }
        Ok((df_start / df_end - 1.0) / tau)
    }

    /// The current-period index+margin rate: the supplied fixing, or
    /// the period's projected forward plus the quoted margin.
    fn current_rate(
        &self,
        curve: &YieldCurve,
        period: &AccrualPeriod,
    ) -> Result<f64, RustyQLibError> {
        match self.current_coupon {
            Some(rate) => Ok(rate),
            None => Ok(self.forward(curve, period)? + self.quoted_margin),
        }
    }

    /// Accrued interest per 100 face at `settlement`.
    pub fn accrued_interest(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (periods, current) = self.locate(settlement)?;
        let rate = self.current_rate(curve, &periods[current])?;
        Ok(100.0
            * rate
            * self
                .day_count
                .year_fraction(periods[current].start, settlement))
    }

    /// Dirty price per 100 face for a given discount margin `dm`.
    pub fn dirty_price_from_discount_margin(
        &self,
        curve: &YieldCurve,
        dm: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !dm.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "discount_margin",
                format!("must be finite, got {dm}"),
            ));
        }
        let (periods, current) = self.locate(settlement)?;

        // backward recursion over the future periods
        let mut value = 100.0;
        for period in periods[current + 1..].iter().rev() {
            let tau = self.day_count.year_fraction(period.start, period.end);
            let forward = self.forward(curve, period)?;
            value = (value + 100.0 * (forward + self.quoted_margin) * tau)
                / (1.0 + (forward + dm) * tau);
        }

        // current period: the coupon is (partly) fixed, discount over the
        // remaining fraction
        let period = &periods[current];
        let rate = self.current_rate(curve, period)?;
        let tau_full = self.day_count.year_fraction(period.start, period.end);
        let tau_remaining = self.day_count.year_fraction(settlement, period.end);
        let coupon = 100.0 * rate * tau_full;
        let index_rate = rate - self.quoted_margin;
        Ok((value + coupon) / (1.0 + (index_rate + dm) * tau_remaining))
    }

    /// Clean price per 100 face for a given discount margin.
    pub fn clean_price_from_discount_margin(
        &self,
        curve: &YieldCurve,
        dm: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.dirty_price_from_discount_margin(curve, dm, settlement)?
                - self.accrued_interest(curve, settlement)?,
        )
    }

    /// The discount margin implied by a clean price.
    pub fn discount_margin_from_price(
        &self,
        clean_price: f64,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !clean_price.is_finite() || clean_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "discount_margin",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // price is decreasing in the margin: bisect target - price(dm)
        let objective = |dm: f64| {
            clean_price
                - self
                    .clean_price_from_discount_margin(curve, dm, settlement)
                    .expect("margin bracket keeps discounting valid")
        };
        self.clean_price_from_discount_margin(curve, MARGIN_BRACKET.0, settlement)?;
        let root =
            Solver1d::new(1e-12, 200).bisection(objective, MARGIN_BRACKET.0, MARGIN_BRACKET.1)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "discount-margin solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }

    /// The periods and the index of the one containing `settlement`.
    fn locate(&self, settlement: NaiveDate) -> Result<(Vec<AccrualPeriod>, usize), RustyQLibError> {
        let periods = self.periods()?;
        if settlement < periods[0].start || settlement >= periods[periods.len() - 1].end {
            return Err(RustyQLibError::invalid_input(
                "frn",
                format!(
                    "settlement {settlement} must lie inside the note's life {} to {}",
                    periods[0].start,
                    periods[periods.len() - 1].end
                ),
            ));
        }
        let current = periods
            .iter()
            .position(|p| settlement < p.end)
            .expect("settlement is inside the life");
        Ok((periods, current))
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

    fn two_year_frn(quoted_margin: f64) -> FloatingRateNote {
        FloatingRateNote::usd_standard(100.0, quoted_margin, d(2026, 8, 6), d(2028, 8, 6), None)
            .unwrap()
    }

    #[test]
    fn prices_par_on_a_reset_date_when_margins_match() {
        // the defining FRN property, at wildly different rate levels
        let frn = two_year_frn(0.0075);
        for rate in [0.01, 0.04, 0.09] {
            let curve = flat(rate, d(2026, 8, 6));
            let dirty = frn
                .dirty_price_from_discount_margin(&curve, 0.0075, d(2026, 8, 6))
                .unwrap();
            assert!((dirty - 100.0).abs() < 1e-10, "rate {rate}: {dirty}");
            // no accrued on the reset date
            assert!(frn.accrued_interest(&curve, d(2026, 8, 6)).unwrap().abs() < 1e-12);
        }
    }

    #[test]
    fn discount_margin_round_trips_and_orders_prices() {
        let frn = two_year_frn(0.0075);
        let curve = flat(0.04, d(2026, 8, 6));
        let settlement = d(2026, 9, 10); // mid-period
        for dm in [0.0, 0.0075, 0.02] {
            let clean = frn
                .clean_price_from_discount_margin(&curve, dm, settlement)
                .unwrap();
            let recovered = frn
                .discount_margin_from_price(clean, &curve, settlement)
                .unwrap();
            assert!((recovered - dm).abs() < 1e-10, "dm {dm}: {recovered}");
        }
        // wider margin, cheaper note; trading above the quoted margin
        // means below par
        let tight = frn
            .clean_price_from_discount_margin(&curve, 0.005, settlement)
            .unwrap();
        let wide = frn
            .clean_price_from_discount_margin(&curve, 0.02, settlement)
            .unwrap();
        assert!(wide < tight);
        assert!(wide < 100.0);
    }

    #[test]
    fn current_coupon_fixing_overrides_the_projection() {
        let curve = flat(0.04, d(2026, 8, 6));
        let settlement = d(2026, 9, 10);
        let projected = two_year_frn(0.0075);
        let mut fixed = projected.clone();
        // period fixed well above where the curve projects it
        fixed.current_coupon = Some(0.07);
        let p_dirty = projected
            .dirty_price_from_discount_margin(&curve, 0.0075, settlement)
            .unwrap();
        let f_dirty = fixed
            .dirty_price_from_discount_margin(&curve, 0.0075, settlement)
            .unwrap();
        assert!(f_dirty > p_dirty, "{f_dirty} vs {p_dirty}");
        // and the accrued reflects the fixing
        let accrued = fixed.accrued_interest(&curve, settlement).unwrap();
        let expected = 100.0 * 0.07 * (35.0 / 360.0); // Aug 6 -> Sep 10
        assert!((accrued - expected).abs() < 1e-12, "accrued {accrued}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let curve = flat(0.04, d(2026, 8, 6));
        assert!(
            FloatingRateNote::usd_standard(0.0, 0.0075, d(2026, 8, 6), d(2028, 8, 6), None)
                .is_err()
        );
        assert!(
            FloatingRateNote::usd_standard(100.0, 0.0075, d(2028, 8, 6), d(2026, 8, 6), None)
                .is_err()
        );
        let frn = two_year_frn(0.0075);
        // settlement outside the life (Aug 6 2028 is a Sunday, so the
        // adjusted final accrual runs to Monday Aug 7)
        assert!(frn
            .dirty_price_from_discount_margin(&curve, 0.0075, d(2026, 8, 5))
            .is_err());
        assert!(frn
            .dirty_price_from_discount_margin(&curve, 0.0075, d(2028, 8, 7))
            .is_err());
        assert!(frn
            .discount_margin_from_price(-10.0, &curve, d(2026, 9, 10))
            .is_err());
    }
}
