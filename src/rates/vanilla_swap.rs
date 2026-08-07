//! Fixed-for-floating interest rate swap.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::{accrual_periods, annuity, fixed_leg_pv, float_leg_pv, AccrualPeriod};
use crate::rates::PayerReceiver;

/// A vanilla interest rate swap: a periodic fixed leg against a
/// periodic floating leg, both from `effective_date` to
/// `maturity_date`. PV is quoted from the position's point of view
/// (`Payer` pays fixed).
#[derive(Debug, Clone)]
pub struct VanillaSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub payer_receiver: PayerReceiver,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub fixed_frequency: Frequency,
    pub fixed_day_count: DayCountConvention,
    pub float_frequency: Frequency,
    pub float_day_count: DayCountConvention,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
}

impl VanillaSwap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        payer_receiver: PayerReceiver,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        fixed_frequency: Frequency,
        fixed_day_count: DayCountConvention,
        float_frequency: Frequency,
        float_day_count: DayCountConvention,
        calendar: Calendar,
        convention: BusinessDayConvention,
    ) -> Result<Self, RustyQLibError> {
        if !notional.is_finite() || notional <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "swap",
                format!("notional must be positive, got {notional}"),
            ));
        }
        if !fixed_rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "swap",
                format!("fixed rate must be finite, got {fixed_rate}"),
            ));
        }
        if maturity_date <= effective_date {
            return Err(RustyQLibError::invalid_input(
                "swap",
                format!("maturity {maturity_date} must be after effective {effective_date}"),
            ));
        }
        Ok(VanillaSwap {
            notional,
            fixed_rate,
            payer_receiver,
            effective_date,
            maturity_date,
            fixed_frequency,
            fixed_day_count,
            float_frequency,
            float_day_count,
            calendar,
            convention,
        })
    }

    /// A USD-style swap: semiannual 30/360 fixed versus quarterly
    /// Act/360 floating, modified following on the US bond-market
    /// calendar.
    pub fn usd_standard(
        notional: f64,
        fixed_rate: f64,
        payer_receiver: PayerReceiver,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            notional,
            fixed_rate,
            payer_receiver,
            effective_date,
            maturity_date,
            Frequency::Semiannual,
            DayCountConvention::Thirty360,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
    }

    /// The fixed leg's accrual periods.
    pub fn fixed_periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        accrual_periods(
            self.effective_date,
            self.maturity_date,
            self.fixed_frequency,
            &self.calendar,
            self.convention,
            0,
        )
    }

    /// The floating leg's accrual periods.
    pub fn float_periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        accrual_periods(
            self.effective_date,
            self.maturity_date,
            self.float_frequency,
            &self.calendar,
            self.convention,
            0,
        )
    }

    /// PV of the fixed leg (positive, before the payer/receiver sign).
    pub fn fixed_leg_pv(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(self.notional
            * fixed_leg_pv(
                &self.fixed_periods()?,
                self.fixed_rate,
                self.fixed_day_count,
                discount,
            ))
    }

    /// PV of the floating leg forecast on `forecast` and discounted on
    /// `discount` (positive, before the payer/receiver sign).
    pub fn float_leg_pv(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.notional
            * float_leg_pv(
                &self.float_periods()?,
                0.0,
                self.float_day_count,
                discount,
                forecast,
            )?)
    }

    /// Swap PV under dual curves: `sign * (float - fixed)`.
    pub fn pv_with(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.payer_receiver.sign()
            * (self.float_leg_pv(discount, forecast)? - self.fixed_leg_pv(discount)?))
    }

    /// Single-curve PV: forecast and discount on the same curve.
    pub fn pv(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        self.pv_with(curve, curve)
    }

    /// The fair (par) fixed rate: the rate that makes the PV zero.
    pub fn par_rate(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let annuity = annuity(&self.fixed_periods()?, self.fixed_day_count, discount);
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive fixed-leg annuity {annuity}"
            )));
        }
        let float_pv = float_leg_pv(
            &self.float_periods()?,
            0.0,
            self.float_day_count,
            discount,
            forecast,
        )?;
        Ok(float_pv / annuity)
    }

    /// PV change for a one-basis-point increase of the fixed rate,
    /// signed from the position's point of view (a payer gains when its
    /// contractual rate would have been 1bp lower — this is the
    /// fixed-leg annuity in disguise).
    pub fn fixed_rate_pv01(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        let annuity = annuity(&self.fixed_periods()?, self.fixed_day_count, discount);
        Ok(-self.payer_receiver.sign() * self.notional * annuity / 10_000.0)
    }

    /// Curve DV01: PV change for a one-basis-point parallel increase of
    /// both curves' zero rates.
    pub fn dv01(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let shift = crate::core::curves::RateShift::ParallelAbsolute(0.0001);
        let base = self.pv_with(discount, forecast)?;
        let bumped = self.pv_with(&discount.bumped(&shift)?, &forecast.bumped(&shift)?)?;
        Ok(bumped - base)
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

    fn two_year_payer(fixed_rate: f64) -> VanillaSwap {
        VanillaSwap::usd_standard(
            1_000_000.0,
            fixed_rate,
            PayerReceiver::Payer,
            d(2026, 8, 6),
            d(2028, 8, 6),
        )
        .unwrap()
    }

    #[test]
    fn par_swap_has_zero_pv_and_par_matches_the_flat_curve() {
        let curve = flat(0.04, d(2026, 8, 6));
        let swap = two_year_payer(0.04);
        let par = swap.par_rate(&curve, &curve).unwrap();
        // semiannual-equivalent of 4% continuous is 2*(e^0.02 - 1) = 4.0402%;
        // schedule adjustments move it only slightly
        assert!((par - 0.0404).abs() < 5e-4, "par {par}");
        let at_par = two_year_payer(par);
        assert!(at_par.pv(&curve).unwrap().abs() < 1e-8);
    }

    #[test]
    fn payer_and_receiver_are_antisymmetric() {
        let curve = flat(0.045, d(2026, 8, 6));
        let payer = two_year_payer(0.04);
        let mut receiver = payer.clone();
        receiver.payer_receiver = PayerReceiver::Receiver;
        let p = payer.pv(&curve).unwrap();
        let r = receiver.pv(&curve).unwrap();
        assert!((p + r).abs() < 1e-10, "{p} vs {r}");
        // paying 4% fixed when rates are ~4.5%: the payer is in the money
        assert!(p > 0.0);
    }

    #[test]
    fn fixed_rate_pv01_is_exactly_linear() {
        let curve = flat(0.04, d(2026, 8, 6));
        let swap = two_year_payer(0.04);
        let bumped = two_year_payer(0.04 + 1e-4);
        let actual = bumped.pv(&curve).unwrap() - swap.pv(&curve).unwrap();
        let pv01 = swap.fixed_rate_pv01(&curve).unwrap();
        assert!((actual - pv01).abs() < 1e-9, "{actual} vs {pv01}");
        // a payer loses when the contractual fixed rate rises
        assert!(pv01 < 0.0);
    }

    #[test]
    fn dual_curve_forecasting_moves_the_float_leg() {
        let reference = d(2026, 8, 6);
        let discount = flat(0.04, reference);
        let higher_forecast = flat(0.0425, reference);
        let swap = two_year_payer(0.04);
        let single = swap.pv(&discount).unwrap();
        let dual = swap.pv_with(&discount, &higher_forecast).unwrap();
        // a payer receives the (higher-forecast) float leg
        assert!(dual > single, "{dual} vs {single}");
        // and the par rate rises roughly with the forecast curve
        let par = swap.par_rate(&discount, &higher_forecast).unwrap();
        assert!(par > 0.0425 && par < 0.0445, "par {par}");
    }

    #[test]
    fn dv01_sign_and_magnitude_are_sensible() {
        let curve = flat(0.04, d(2026, 8, 6));
        let swap = two_year_payer(0.04);
        let dv01 = swap.dv01(&curve, &curve).unwrap();
        // payer gains when rates rise; ~2y annuity on 1mm is ~190 per bp
        assert!(dv01 > 100.0 && dv01 < 300.0, "dv01 {dv01}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let e = d(2026, 8, 6);
        assert!(
            VanillaSwap::usd_standard(0.0, 0.04, PayerReceiver::Payer, e, d(2028, 8, 6)).is_err()
        );
        assert!(
            VanillaSwap::usd_standard(1e6, f64::NAN, PayerReceiver::Payer, e, d(2028, 8, 6))
                .is_err()
        );
        assert!(VanillaSwap::usd_standard(1e6, 0.04, PayerReceiver::Payer, e, e).is_err());
    }
}
