//! Fixed-for-floating interest rate swap.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::{
    annuity, fixed_leg_pv, float_leg_pv_compounded, float_leg_pv_with_fixing, float_periods,
    AccrualPeriod, CompoundingMethod, FloatPeriod,
};
use crate::rates::overnight::RateFixings;
use crate::rates::schedule::{LegSchedule, RollConvention, StubConvention};
use crate::rates::{validate_swap_terms, PayerReceiver};

/// A vanilla interest rate swap: a periodic fixed leg against a
/// periodic floating leg, both from `effective_date` to
/// `maturity_date`. PV is quoted from the position's point of view
/// (`Payer` pays fixed).
///
/// The constructors give the market defaults — short front stub, the
/// maturity's roll day, one reset per floating payment; [`with_stub`],
/// [`with_roll`] and [`with_float_reset`] change them.
///
/// [`with_stub`]: Self::with_stub
/// [`with_roll`]: Self::with_roll
/// [`with_float_reset`]: Self::with_float_reset
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
    /// Stub placement, shared by both legs.
    pub stub: StubConvention,
    /// Anchor-day roll, shared by both legs.
    pub roll: RollConvention,
    /// The floating index's reset frequency when it resets more often
    /// than the leg pays (`None`: one reset per payment).
    pub float_reset: Option<Frequency>,
    /// How reset sub-periods compound into a payment.
    pub float_compounding: CompoundingMethod,
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
        validate_swap_terms(
            "swap",
            notional,
            fixed_rate,
            "fixed rate",
            effective_date,
            maturity_date,
        )?;
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
            stub: StubConvention::default(),
            roll: RollConvention::default(),
            float_reset: None,
            float_compounding: CompoundingMethod::default(),
        })
    }

    /// Place the stub (both legs).
    pub fn with_stub(mut self, stub: StubConvention) -> Self {
        self.stub = stub;
        self
    }

    /// Pin the anchor day (both legs).
    pub fn with_roll(mut self, roll: RollConvention) -> Self {
        self.roll = roll;
        self
    }

    /// Reset the floating index every `reset` inside each payment
    /// period, compounding the sub-periods by `compounding` — a 1M
    /// index on a quarterly leg, say. The reset frequency must divide
    /// the floating payment frequency.
    pub fn with_float_reset(mut self, reset: Frequency, compounding: CompoundingMethod) -> Self {
        self.float_reset = Some(reset);
        self.float_compounding = compounding;
        self
    }

    /// The fixed leg's schedule terms.
    pub fn fixed_schedule(&self) -> LegSchedule {
        LegSchedule::new(
            self.effective_date,
            self.maturity_date,
            self.fixed_frequency,
            self.calendar.clone(),
            self.convention,
        )
        .with_stub(self.stub)
        .with_roll(self.roll)
    }

    /// The floating leg's payment schedule terms.
    pub fn float_schedule(&self) -> LegSchedule {
        LegSchedule::new(
            self.effective_date,
            self.maturity_date,
            self.float_frequency,
            self.calendar.clone(),
            self.convention,
        )
        .with_stub(self.stub)
        .with_roll(self.roll)
    }

    /// The floating leg's payment periods with their reset sub-periods.
    pub fn float_reset_periods(&self) -> Result<Vec<FloatPeriod>, RustyQLibError> {
        float_periods(&self.float_schedule(), self.float_reset)
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
        self.fixed_schedule().accrual_periods()
    }

    /// The floating leg's payment periods.
    pub fn float_periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        self.float_schedule().accrual_periods()
    }

    /// The single-rate fixing path applies only to a leg that resets
    /// once per payment; a compounding leg takes a fixing history.
    fn require_single_reset(&self, method: &str) -> Result<(), RustyQLibError> {
        if self.float_reset.is_some() {
            return Err(RustyQLibError::invalid_input(
                "swap",
                format!(
                    "{method} takes one realized rate per period; a leg with sub-period \
                     resets needs pv_with_fixings and a fixing history"
                ),
            ));
        }
        Ok(())
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
            * float_leg_pv_compounded(
                &self.float_reset_periods()?,
                0.0,
                self.float_day_count,
                self.float_compounding,
                discount,
                forecast,
                None,
            )?)
    }

    /// PV of a seasoned floating leg: `realized_rate` is the annualized
    /// simple rate realized from the current period's start through the
    /// forecast curve's reference date (see
    /// [`float_leg_pv_with_fixing`]).
    pub fn float_leg_pv_with_fixing(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        realized_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        self.require_single_reset("float_leg_pv_with_fixing")?;
        Ok(self.notional
            * float_leg_pv_with_fixing(
                &self.float_periods()?,
                0.0,
                self.float_day_count,
                discount,
                forecast,
                Some(realized_rate),
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

    /// [`pv_with`](Self::pv_with) for a seasoned swap, with the current
    /// float period's realized rate supplied. Settled periods on both
    /// legs contribute nothing.
    pub fn pv_with_fixing(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        realized_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        self.require_single_reset("pv_with_fixing")?;
        Ok(self.payer_receiver.sign()
            * (self.float_leg_pv_with_fixing(discount, forecast, realized_rate)?
                - self.fixed_leg_pv(discount)?))
    }

    /// PV of the floating leg for a seasoned swap with a fixing
    /// history: every reset that started before the forecast curve's
    /// reference date takes its published rate from `fixings`, keyed by
    /// the reset's accrual start. Works for single- and multi-reset legs.
    pub fn float_leg_pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        fixings: &RateFixings,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.notional
            * float_leg_pv_compounded(
                &self.float_reset_periods()?,
                0.0,
                self.float_day_count,
                self.float_compounding,
                discount,
                forecast,
                Some(fixings),
            )?)
    }

    /// [`pv_with`](Self::pv_with) for a seasoned swap with a fixing
    /// history (see [`float_leg_pv_with_fixings`](Self::float_leg_pv_with_fixings)).
    pub fn pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        fixings: &RateFixings,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.payer_receiver.sign()
            * (self.float_leg_pv_with_fixings(discount, forecast, fixings)?
                - self.fixed_leg_pv(discount)?))
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
        let float_pv = self.float_leg_pv(discount, forecast)? / self.notional;
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

    #[test]
    fn conventions_change_the_schedule_and_compounding_the_leg() {
        let reference = d(2026, 8, 6);
        let curve = flat(0.045, reference);
        // 14 months: a stub somewhere
        let base = VanillaSwap::usd_standard(
            1_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 8, 20),
            d(2027, 10, 20),
        )
        .unwrap();
        let short_front = base.fixed_periods().unwrap();
        let long_front = base
            .clone()
            .with_stub(StubConvention::LongFront)
            .fixed_periods()
            .unwrap();
        let short_back = base
            .clone()
            .with_stub(StubConvention::ShortBack)
            .fixed_periods()
            .unwrap();
        assert_eq!(short_front.len(), 3);
        assert_eq!(long_front.len(), 2);
        assert_eq!(short_back.len(), 3);
        assert_eq!(short_front[0].end, d(2026, 10, 20));
        assert_eq!(short_back[0].end, d(2027, 2, 22)); // Feb 20 2027 is a Saturday
                                                       // IMM roll on an IMM-dated swap
        let imm = VanillaSwap::usd_standard(
            1_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 9, 16),
            d(2027, 9, 15),
        )
        .unwrap()
        .with_roll(RollConvention::Imm);
        let periods = imm.float_periods().unwrap();
        assert_eq!(periods[0].end, d(2026, 12, 16));
        assert_eq!(periods[1].end, d(2027, 3, 17));
        // monthly resets, straight compounding, zero spread: same value
        // as the plain quarterly leg (the df ratios telescope)
        let plain = base.float_leg_pv(&curve, &curve).unwrap();
        let compounded = base
            .clone()
            .with_float_reset(Frequency::Monthly, CompoundingMethod::Straight)
            .float_leg_pv(&curve, &curve)
            .unwrap();
        assert!((plain - compounded).abs() < 1e-6, "{plain} vs {compounded}");
        // no compounding on monthly resets undervalues it by the lost
        // intra-quarter compounding, well under a percent of the leg
        let simple = base
            .clone()
            .with_float_reset(Frequency::Monthly, CompoundingMethod::None)
            .float_leg_pv(&curve, &curve)
            .unwrap();
        assert!(
            simple < plain && plain - simple < 1e-2 * plain,
            "{simple} vs {plain}"
        );
        // the par rate follows the leg it prices
        let par = base
            .clone()
            .with_float_reset(Frequency::Monthly, CompoundingMethod::None)
            .par_rate(&curve, &curve)
            .unwrap();
        assert!(par < base.par_rate(&curve, &curve).unwrap());
        // the single-rate fixing path refuses a compounding leg
        assert!(base
            .clone()
            .with_float_reset(Frequency::Monthly, CompoundingMethod::Straight)
            .pv_with_fixing(&curve, &curve, 0.04)
            .is_err());
    }

    #[test]
    fn seasoned_swap_prices_off_its_fixing_history() {
        // a 2y quarterly swap valued five weeks in: the first quarter has
        // fixed; with a fixing equal to the curve forward the history
        // path matches the single-rate path
        let effective = d(2026, 8, 6);
        let valuation = d(2026, 9, 10);
        let curve = flat(0.045, valuation);
        let swap = VanillaSwap::new(
            1_000_000.0,
            0.045,
            PayerReceiver::Payer,
            effective,
            d(2028, 8, 6),
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
        )
        .unwrap();
        assert!(swap
            .pv_with_fixings(&curve, &curve, &RateFixings::new())
            .is_err());
        let mut fixings = RateFixings::new();
        fixings.insert(effective, 0.05);
        let history = swap.pv_with_fixings(&curve, &curve, &fixings).unwrap();
        let single = swap.pv_with_fixing(&curve, &curve, 0.05).unwrap();
        // the two seasoned conventions differ in how the fixed rate
        // carries to the period end: an IBOR fixing applies for the
        // whole period, while the single-rate path grows the realized
        // stub at the curve from the valuation date on
        let p = swap.float_periods().unwrap()[0];
        let dc = DayCountConvention::Act360;
        let ibor = 0.05 * dc.year_fraction(p.start, p.end);
        let stub = (1.0 + 0.05 * dc.year_fraction(p.start, valuation)) / curve.df_date(p.end) - 1.0;
        let expected_gap = 1_000_000.0 * (ibor - stub) * curve.df_date(p.payment);
        assert!(
            (history - single - expected_gap).abs() < 1e-6,
            "{history} vs {single}: gap {expected_gap}"
        );
        // a higher fixing is worth more to the payer (receives floating)
        fixings.insert(effective, 0.06);
        assert!(swap.pv_with_fixings(&curve, &curve, &fixings).unwrap() > history);
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
    fn seasoned_swap_needs_and_uses_the_current_fixing() {
        // swap effective Aug 2026, priced mid-life in Jan 2027: the old
        // code summed past fixed coupons at face and dropped past float
        // periods, silently
        let swap = two_year_payer(0.04);
        let curve = flat(0.05, d(2027, 1, 15));
        // projecting the live float period now errors instead
        assert!(swap.pv(&curve).is_err());
        // with the current fixing supplied, paying 4% fixed against ~5%
        // rates leaves the payer in the money, at remaining-leg scale
        let pv = swap.pv_with_fixing(&curve, &curve, 0.05).unwrap();
        assert!(pv > 0.0, "{pv}");
        assert!(pv < 50_000.0, "{pv}");
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
