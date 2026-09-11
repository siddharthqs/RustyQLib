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
use crate::rates::leg::{annuity, float_leg_pv, float_leg_pv_with_fixing, AccrualPeriod};
use crate::rates::overnight::{compounded_overnight_accrual, OvernightConvention, RateFixings};
use crate::rates::schedule::{LegSchedule, RollConvention, StubConvention};
use crate::rates::{validate_swap_terms, PayerReceiver};

/// With the default [`OvernightConvention`] a fully forecast period is
/// the curve ratio `df(s)/df(e) - 1`; with a lookback, lockout or
/// observation shift — or once fixings enter through
/// [`pv_with_fixings`](Self::pv_with_fixings) — each period is
/// compounded day by day.
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
    /// Stub placement.
    pub stub: StubConvention,
    /// Anchor-day roll.
    pub roll: RollConvention,
    /// Lookback, lockout and observation shift of the compounding.
    pub overnight: OvernightConvention,
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
        validate_swap_terms(
            "ois",
            notional,
            fixed_rate,
            "fixed rate",
            effective_date,
            maturity_date,
        )?;
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
            stub: StubConvention::default(),
            roll: RollConvention::default(),
            overnight: OvernightConvention::default(),
        })
    }

    /// Place the stub.
    pub fn with_stub(mut self, stub: StubConvention) -> Self {
        self.stub = stub;
        self
    }

    /// Pin the anchor day.
    pub fn with_roll(mut self, roll: RollConvention) -> Self {
        self.roll = roll;
        self
    }

    /// Set the compounding conventions (lookback, lockout, shift).
    pub fn with_overnight(mut self, overnight: OvernightConvention) -> Self {
        self.overnight = overnight;
        self
    }

    /// The shared schedule terms of both legs.
    pub fn schedule(&self) -> LegSchedule {
        LegSchedule::new(
            self.effective_date,
            self.maturity_date,
            self.frequency,
            self.calendar.clone(),
            self.convention,
        )
        .with_stub(self.stub)
        .with_roll(self.roll)
        .with_payment_lag(self.payment_lag)
    }

    /// PV of the floating leg on unit notional: the curve ratio per
    /// period under plain conventions with no fixings, otherwise the
    /// daily compounding under `self.overnight`, realized days from
    /// `fixings`.
    fn float_leg(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        fixings: Option<&RateFixings>,
    ) -> Result<f64, RustyQLibError> {
        let periods = self.periods()?;
        if self.overnight.is_plain() && fixings.is_none() {
            return float_leg_pv(&periods, 0.0, self.day_count, discount, forecast);
        }
        let valuation = discount.reference_date();
        let mut pv = 0.0;
        for p in &periods {
            if p.payment <= valuation {
                continue;
            }
            let accrual = compounded_overnight_accrual(
                forecast,
                fixings,
                p.start,
                p.end,
                &self.calendar,
                self.overnight,
            )?;
            pv += accrual * discount.df_date(p.payment);
        }
        Ok(pv)
    }

    /// [`pv_with`](Self::pv_with) for a seasoned swap with the
    /// published overnight fixings: the period in progress compounds
    /// its realized days from `fixings` and its remaining days off the
    /// forecast curve, under the swap's [`OvernightConvention`].
    pub fn pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        fixings: &RateFixings,
    ) -> Result<f64, RustyQLibError> {
        let periods = self.periods()?;
        let float = self.float_leg(discount, forecast, Some(fixings))?;
        let fixed = self.fixed_rate * annuity(&periods, self.day_count, discount);
        Ok(self.payer_receiver.sign() * self.notional * (float - fixed))
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
        self.schedule().accrual_periods()
    }

    /// Swap PV: `sign * (float - fixed)`, forecast on `forecast`,
    /// discounted on `discount`.
    pub fn pv_with(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let periods = self.periods()?;
        let float = self.float_leg(discount, forecast, None)?;
        let fixed = self.fixed_rate * annuity(&periods, self.day_count, discount);
        Ok(self.payer_receiver.sign() * self.notional * (float - fixed))
    }

    /// [`pv_with`](Self::pv_with) for a seasoned swap: `realized_rate`
    /// is the annualized simple-rate equivalent of the compounding
    /// realized from the current period's start through the forecast
    /// curve's reference date. Settled periods contribute nothing.
    pub fn pv_with_fixing(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        realized_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        if !self.overnight.is_plain() {
            return Err(RustyQLibError::invalid_input(
                "ois",
                "a swap with lookback, lockout or observation shift compounds day by day: \
                 supply the fixings through pv_with_fixings",
            ));
        }
        let periods = self.periods()?;
        let float = float_leg_pv_with_fixing(
            &periods,
            0.0,
            self.day_count,
            discount,
            forecast,
            Some(realized_rate),
        )?;
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
        Ok(self.float_leg(discount, forecast, None)? / annuity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

    #[test]
    fn daily_compounding_matches_the_curve_ratio_and_conventions_perturb_it() {
        let reference = d(2026, 8, 6);
        let curve = YieldCurve::from_zero_rates(
            &[
                crate::core::curves::Tenor::YearFraction(0.5),
                crate::core::curves::Tenor::YearFraction(3.0),
            ],
            &[0.04, 0.05],
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
            crate::core::curves::InterpolationMethod::LinearZero,
        )
        .unwrap();
        let swap = OvernightIndexSwap::sofr_standard(
            1_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 9, 1),
            d(2028, 9, 1),
        )
        .unwrap();
        let plain = swap.pv(&curve).unwrap();
        // the daily loop with plain conventions and an empty history
        // reproduces the telescoped ratio
        let daily = swap
            .pv_with_fixings(&curve, &curve, &RateFixings::new())
            .unwrap();
        assert!((plain - daily).abs() < 1e-6, "{plain} vs {daily}");
        // lookback and lockout are small corrections, not zero
        let lookback = swap
            .clone()
            .with_overnight(OvernightConvention {
                lookback_days: 5,
                ..Default::default()
            })
            .pv(&curve)
            .unwrap();
        assert!(
            lookback != plain && (lookback - plain).abs() < 500.0,
            "{lookback} vs {plain}"
        );
        let lockout = swap
            .clone()
            .with_overnight(OvernightConvention {
                lockout_days: 2,
                ..Default::default()
            })
            .pv(&curve)
            .unwrap();
        assert!(
            lockout != plain && (lockout - plain).abs() < 100.0,
            "{lockout} vs {plain}"
        );
        // the single-rate path is refused once conventions are set
        assert!(swap
            .clone()
            .with_overnight(OvernightConvention {
                lookback_days: 5,
                ..Default::default()
            })
            .pv_with_fixing(&curve, &curve, 0.045)
            .is_err());
        // stub and roll flow through to the schedule
        let back = swap.clone().with_stub(StubConvention::ShortBack);
        assert_eq!(back.periods().unwrap().len(), swap.periods().unwrap().len());
        // an IMM roll on a non-IMM effective date: a 15-day front stub
        // to the September 2026 IMM date, then annual IMM anchors
        let imm = swap
            .clone()
            .with_roll(RollConvention::Imm)
            .periods()
            .unwrap();
        assert_eq!(imm[0].end, d(2026, 9, 16));
        assert_eq!(imm[1].end, d(2027, 9, 15));
    }

    #[test]
    fn seasoned_ois_compounds_realized_fixings_then_forecasts() {
        // effective Sep 1, valued Oct 1 with September fixed at 5%
        let effective = d(2026, 9, 1);
        let valuation = d(2026, 10, 1);
        let curve = YieldCurve::flat(
            0.045,
            valuation,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let swap = OvernightIndexSwap::sofr_standard(
            1_000_000.0,
            0.045,
            PayerReceiver::Payer,
            effective,
            d(2027, 9, 1),
        )
        .unwrap();
        assert!(swap
            .pv_with_fixings(&curve, &curve, &RateFixings::new())
            .is_err());
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 8, 31), 0.05);
        let high = swap.pv_with_fixings(&curve, &curve, &fixings).unwrap();
        fixings.insert(d(2026, 8, 31), 0.045);
        let flat = swap.pv_with_fixings(&curve, &curve, &fixings).unwrap();
        // 50bp over a month on 1mm, roughly 400 to the payer
        assert!(
            high - flat > 350.0 && high - flat < 450.0,
            "{high} vs {flat}"
        );
        // the one-period swap assembles from the compounded accrual, the
        // fixed coupon and the lagged payment discount
        let p = swap.periods().unwrap()[0];
        let accrual = compounded_overnight_accrual(
            &curve,
            Some(&fixings),
            p.start,
            p.end,
            &swap.calendar,
            OvernightConvention::default(),
        )
        .unwrap();
        let tau = DayCountConvention::Act360.year_fraction(p.start, p.end);
        let expected = 1_000_000.0 * (accrual - 0.045 * tau) * curve.df_date(p.payment);
        assert!((flat - expected).abs() < 1e-6, "{flat} vs {expected}");
    }

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
    fn seasoned_ois_prices_with_the_realized_rate() {
        // priced mid first annual period: the realized compounding since
        // Aug 2026 cannot be read off a Jan 2027 curve
        let ois = two_year_sofr(0.04);
        let curve = flat(0.05, d(2027, 1, 15));
        assert!(ois.pv(&curve).is_err());
        // with it supplied, paying 4% fixed against ~5% is in the money
        let pv = ois.pv_with_fixing(&curve, &curve, 0.05).unwrap();
        assert!(pv > 0.0, "{pv}");
        assert!(pv < 50_000.0, "{pv}");
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
