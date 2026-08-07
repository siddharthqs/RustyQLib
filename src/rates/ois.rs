//! Overnight indexed swap (OIS): fixed versus daily-compounded
//! overnight, SOFR-style.
//!
//! Both legs share one payment schedule (annual for USD SOFR OIS), and
//! payments may lag the accrual end by a couple of business days. With
//! no fixings modelled, the compounded overnight accrual over `[s, e]`
//! is forecast from the OIS curve as `df(s)/df(e) - 1` — exactly the
//! telescoped product of the daily forward factors.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::{accrual_periods, annuity, float_leg_pv, AccrualPeriod};
use crate::rates::PayerReceiver;

#[derive(Debug, Clone)]
pub struct OvernightIndexSwap {
    pub notional: f64,
    pub fixed_rate: f64,
    pub payer_receiver: PayerReceiver,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    /// One schedule for both legs (annual is the USD SOFR standard).
    pub frequency: Frequency,
    /// One day count for both legs (Act/360 for SOFR).
    pub day_count: DayCountConvention,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
    /// Business days between accrual end and payment (2 for SOFR OIS).
    pub payment_lag: i64,
}

impl OvernightIndexSwap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        fixed_rate: f64,
        payer_receiver: PayerReceiver,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        frequency: Frequency,
        day_count: DayCountConvention,
        calendar: Calendar,
        convention: BusinessDayConvention,
        payment_lag: i64,
    ) -> Result<Self, RustyQLibError> {
        if !notional.is_finite() || notional <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "ois",
                format!("notional must be positive, got {notional}"),
            ));
        }
        if !fixed_rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "ois",
                format!("fixed rate must be finite, got {fixed_rate}"),
            ));
        }
        if maturity_date <= effective_date {
            return Err(RustyQLibError::invalid_input(
                "ois",
                format!("maturity {maturity_date} must be after effective {effective_date}"),
            ));
        }
        Ok(OvernightIndexSwap {
            notional,
            fixed_rate,
            payer_receiver,
            effective_date,
            maturity_date,
            frequency,
            day_count,
            calendar,
            convention,
            payment_lag,
        })
    }

    /// A USD SOFR OIS: annual Act/360 legs, modified following on the US
    /// bond-market calendar, T+2 payment lag.
    pub fn sofr_standard(
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
            Frequency::Annual,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            2,
        )
    }

    /// The shared accrual periods of both legs.
    pub fn periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        accrual_periods(
            self.effective_date,
            self.maturity_date,
            self.frequency,
            &self.calendar,
            self.convention,
            self.payment_lag,
        )
    }

    /// Swap PV: `sign * (float - fixed)`, forecast on `forecast`,
    /// discounted on `discount`.
    pub fn pv_with(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let periods = self.periods()?;
        let float = float_leg_pv(&periods, 0.0, self.day_count, discount, forecast)?;
        let fixed = self.fixed_rate * annuity(&periods, self.day_count, discount);
        Ok(self.payer_receiver.sign() * self.notional * (float - fixed))
    }

    /// Single-curve PV (the usual OIS setup: discount and forecast are
    /// the same overnight curve).
    pub fn pv(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        self.pv_with(curve, curve)
    }

    /// The fair fixed rate.
    pub fn par_rate(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let periods = self.periods()?;
        let annuity = annuity(&periods, self.day_count, discount);
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive annuity {annuity}"
            )));
        }
        Ok(float_leg_pv(&periods, 0.0, self.day_count, discount, forecast)? / annuity)
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

    fn two_year_sofr(fixed_rate: f64) -> OvernightIndexSwap {
        OvernightIndexSwap::sofr_standard(
            1_000_000.0,
            fixed_rate,
            PayerReceiver::Payer,
            d(2026, 8, 6),
            d(2028, 8, 6),
        )
        .unwrap()
    }

    #[test]
    fn par_ois_has_zero_pv_and_a_sensible_level() {
        let curve = flat(0.04, d(2026, 8, 6));
        let ois = two_year_sofr(0.04);
        let par = ois.par_rate(&curve, &curve).unwrap();
        // annual Act/360 quote of a 4% Act/365-continuous curve: the
        // accrual e^0.04 - 1 spread over tau = 365/360 gives roughly
        // (360/365) * (e^0.04 - 1) = 4.025%
        assert!((par - 0.04025).abs() < 8e-4, "par {par}");
        let at_par = two_year_sofr(par);
        assert!(at_par.pv(&curve).unwrap().abs() < 1e-8);
    }

    #[test]
    fn payment_lag_is_applied_and_costs_a_little_pv() {
        let curve = flat(0.04, d(2026, 8, 6));
        let lagged = two_year_sofr(0.04);
        let mut spot_paid = lagged.clone();
        spot_paid.payment_lag = 0;
        for p in lagged.periods().unwrap() {
            assert!(p.payment > p.end);
        }
        // paying two days later discounts every net cash flow a touch
        // more; with a positive float-fixed gap the payer PV shrinks
        let pv_lagged = lagged.pv(&curve).unwrap();
        let pv_spot = spot_paid.pv(&curve).unwrap();
        assert!(pv_lagged < pv_spot, "{pv_lagged} vs {pv_spot}");
        // but only slightly
        assert!((pv_lagged - pv_spot).abs() / pv_spot.abs() < 1e-3);
    }

    #[test]
    fn receiver_mirrors_payer() {
        let curve = flat(0.045, d(2026, 8, 6));
        let payer = two_year_sofr(0.04);
        let mut receiver = payer.clone();
        receiver.payer_receiver = PayerReceiver::Receiver;
        assert!((payer.pv(&curve).unwrap() + receiver.pv(&curve).unwrap()).abs() < 1e-10);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let e = d(2026, 8, 6);
        assert!(OvernightIndexSwap::sofr_standard(
            -1.0,
            0.04,
            PayerReceiver::Payer,
            e,
            d(2028, 8, 6)
        )
        .is_err());
        assert!(OvernightIndexSwap::sofr_standard(1e6, 0.04, PayerReceiver::Payer, e, e).is_err());
    }
}
