//! Fixed-for-floating commodity swap.
//!
//! The canonical commodity hedge (a WTI calendar swap, a natural gas
//! swap, a gold swap): over each calculation period the floating leg
//! pays the **arithmetic average of the commodity index price over the
//! period's pricing days** (every business day of the period), the
//! fixed leg pays an agreed fixed price, and the difference — times the
//! period's notional quantity — is cash-settled shortly after the
//! period ends. No commodity changes hands.
//!
//! Because the average is arithmetic, the product is linear in the
//! daily prices, so (as with the money-market futures in [`crate::rates`])
//! partially realized periods blend published fixings with curve
//! forwards exactly, and the forward-price delta is analytic.
//!
//! Pricing days run from a period's start (inclusive) to its end
//! (exclusive), so with month-start boundaries — the market standard,
//! see [`CommoditySwap::monthly`] — each period prices over exactly its
//! own month's business days and every business day belongs to one
//! period.

use chrono::NaiveDate;
use std::collections::BTreeMap;

use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::{accrual_periods, AccrualPeriod};
use crate::rates::overnight::fixing_on_or_before;
use crate::rates::PayerReceiver;

/// Published daily index prices (settlement prices), keyed by pricing
/// date. Missing days — an index holiday the calendar does not model —
/// carry the most recent earlier fixing forward.
pub type PriceFixings = BTreeMap<NaiveDate, f64>;

/// A fixed-for-floating commodity swap. PV is quoted from the
/// position's point of view: a [`PayerReceiver::Payer`] pays the fixed
/// price and receives the floating average (the consumer's hedge), a
/// `Receiver` the reverse (the producer's hedge).
#[derive(Debug, Clone)]
pub struct CommoditySwap {
    /// Notional quantity per calculation period, in the index's units
    /// (barrels, MMBtu, troy ounces, ...).
    pub quantity: f64,
    /// The fixed price per unit, in the index's currency.
    pub fixed_price: f64,
    pub payer_receiver: PayerReceiver,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub frequency: Frequency,
    /// The pricing calendar of the index: its business days are the
    /// averaging observations.
    pub calendar: Calendar,
    /// Adjustment for the period boundaries. Commodity calculation
    /// periods are usually calendar months, so `Unadjusted` is standard.
    pub convention: BusinessDayConvention,
    /// Business days between a period's end date (the exclusive
    /// boundary, e.g. the next month's first) and its cash settlement
    /// (5 is the oil-market standard).
    pub payment_lag: i64,
}

impl CommoditySwap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        quantity: f64,
        fixed_price: f64,
        payer_receiver: PayerReceiver,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        frequency: Frequency,
        calendar: Calendar,
        convention: BusinessDayConvention,
        payment_lag: i64,
    ) -> Result<Self, RustyQLibError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity swap",
                format!("quantity must be positive, got {quantity}"),
            ));
        }
        if !fixed_price.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "commodity swap",
                format!("fixed price must be finite, got {fixed_price}"),
            ));
        }
        if maturity_date <= effective_date {
            return Err(RustyQLibError::invalid_input(
                "commodity swap",
                format!("maturity {maturity_date} must be after effective {effective_date}"),
            ));
        }
        Ok(CommoditySwap {
            quantity,
            fixed_price,
            payer_receiver,
            effective_date,
            maturity_date,
            frequency,
            calendar,
            convention,
            payment_lag,
        })
    }

    /// A standard monthly swap: unadjusted calendar-month periods
    /// settling 5 business days after each month ends. Quote
    /// `effective_date` and `maturity_date` as month starts (e.g.
    /// 2026-09-01 to 2027-09-01 for the Sep-26 through Aug-27 strip) so
    /// each period prices over exactly its month.
    pub fn monthly(
        quantity: f64,
        fixed_price: f64,
        payer_receiver: PayerReceiver,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        calendar: Calendar,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            quantity,
            fixed_price,
            payer_receiver,
            effective_date,
            maturity_date,
            Frequency::Monthly,
            calendar,
            BusinessDayConvention::Unadjusted,
            5,
        )
    }

    /// The calculation periods, each with its lagged settlement date.
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

    /// The averaging observations of one period: every business day `d`
    /// on the pricing calendar with `start <= d < end`.
    pub fn pricing_days(&self, period: &AccrualPeriod) -> Vec<NaiveDate> {
        business_days_in(&self.calendar, period.start, period.end)
    }

    /// The floating average of one period, fully projected on the
    /// forward curve.
    pub fn average_price(
        &self,
        period: &AccrualPeriod,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.average_price_with_fixings(period, forward, &PriceFixings::new(), period.start)
    }

    /// The floating average of a partially realized period: pricing
    /// days strictly before `asof` come from `fixings` (carried forward
    /// over gaps), days from `asof` onward from the forward curve.
    pub fn average_price_with_fixings(
        &self,
        period: &AccrualPeriod,
        forward: &CommodityForwardCurve,
        fixings: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        period_average(
            &self.calendar,
            period.start,
            period.end,
            forward,
            fixings,
            asof,
        )
    }

    /// PV of the floating leg (positive, before the payer/receiver
    /// sign): `quantity * sum avg_i * df(pay_i)`.
    pub fn float_leg_pv(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.float_leg_pv_with_fixings(discount, forward, &PriceFixings::new(), self.effective_date)
    }

    /// PV of the floating leg with realized fixings up to `asof`.
    /// Periods already settled (payment before `asof`) contribute
    /// nothing.
    pub fn float_leg_pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        fixings: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let mut pv = 0.0;
        for period in self.unsettled_periods(asof)? {
            let avg = self.average_price_with_fixings(&period, forward, fixings, asof)?;
            pv += avg * discount.df_date(period.payment);
        }
        Ok(self.quantity * pv)
    }

    /// PV of the fixed leg (positive, before the payer/receiver sign):
    /// `quantity * fixed_price * sum df(pay_i)`.
    pub fn fixed_leg_pv(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(self.fixed_price * self.settlement_annuity(discount, self.effective_date)?)
    }

    /// Swap PV, everything projected on the forward curve:
    /// `sign * (float - fixed)`.
    pub fn pv(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.pv_with_fixings(discount, forward, &PriceFixings::new(), self.effective_date)
    }

    /// Swap PV mid-life: realized pricing days from `fixings`, the rest
    /// from the forward curve, periods already settled dropped.
    pub fn pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        fixings: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let float = self.float_leg_pv_with_fixings(discount, forward, fixings, asof)?;
        let fixed = self.fixed_price * self.settlement_annuity(discount, asof)?;
        Ok(self.payer_receiver.sign() * (float - fixed))
    }

    /// The fair fixed price — the discount-weighted average of the
    /// period averages, which makes the PV zero.
    pub fn par_price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.par_price_with_fixings(discount, forward, &PriceFixings::new(), self.effective_date)
    }

    /// The fair fixed price mid-life, given realized fixings.
    pub fn par_price_with_fixings(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        fixings: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let annuity = self.settlement_annuity(discount, asof)?;
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive settlement annuity {annuity}"
            )));
        }
        let float = self.float_leg_pv_with_fixings(discount, forward, fixings, asof)?;
        Ok(float / annuity)
    }

    /// PV change per one-unit increase of the contractual fixed price,
    /// signed from the position's point of view (a payer loses; this is
    /// minus the settlement annuity, and the relation is exactly linear).
    pub fn fixed_price_delta(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(-self.payer_receiver.sign() * self.settlement_annuity(discount, self.effective_date)?)
    }

    /// Forward-price delta: PV change per one-unit parallel increase of
    /// the forward curve, with every pricing day still floating. Exact,
    /// because the averages are linear in the daily prices.
    pub fn delta(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        self.delta_with_fixings(discount, self.effective_date)
    }

    /// Forward-price delta mid-life: only pricing days on or after
    /// `asof` still float, so each period contributes its unfixed
    /// fraction.
    pub fn delta_with_fixings(
        &self,
        discount: &YieldCurve,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let mut delta = 0.0;
        for period in self.unsettled_periods(asof)? {
            let days = self.pricing_days(&period);
            if days.is_empty() {
                continue;
            }
            let unfixed = days.iter().filter(|&&day| day >= asof).count();
            delta += unfixed as f64 / days.len() as f64 * discount.df_date(period.payment);
        }
        Ok(self.payer_receiver.sign() * self.quantity * delta)
    }

    /// Discounting DV01: PV change for a one-basis-point parallel
    /// increase of the discount curve's zero rates (the forward curve is
    /// prices, not rates, so it does not move).
    pub fn dv01(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        let shift = crate::core::curves::RateShift::ParallelAbsolute(0.0001);
        let base = self.pv(discount, forward)?;
        let bumped = self.pv(&discount.bumped(&shift)?, forward)?;
        Ok(bumped - base)
    }

    /// Periods whose settlement has not happened yet as of `asof`.
    fn unsettled_periods(&self, asof: NaiveDate) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        Ok(self
            .periods()?
            .into_iter()
            .filter(|p| p.payment >= asof)
            .collect())
    }

    /// `quantity * sum df(pay_i)` over unsettled periods: the PV of one
    /// unit of fixed price (shared with the swaption, whose value is
    /// quoted per unit of this annuity).
    pub(crate) fn settlement_annuity(
        &self,
        discount: &YieldCurve,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.quantity
            * self
                .unsettled_periods(asof)?
                .iter()
                .map(|p| discount.df_date(p.payment))
                .sum::<f64>())
    }
}

// ── Averaging machinery shared with the basis swap ──────────────────────

/// Every business day of `calendar` in `[start, end)` — the averaging
/// observations of one calculation period.
pub(crate) fn business_days_in(
    calendar: &Calendar,
    start: NaiveDate,
    end: NaiveDate,
) -> Vec<NaiveDate> {
    let mut days = Vec::new();
    let mut day = start;
    while day < end {
        if calendar.is_business_day(day) {
            days.push(day);
        }
        day = day.succ_opt().expect("date in range");
    }
    days
}

/// Arithmetic average of the index over one period's business days:
/// days strictly before `asof` from `fixings` (carried forward over
/// gaps), days from `asof` onward from the forward curve.
pub(crate) fn period_average(
    calendar: &Calendar,
    start: NaiveDate,
    end: NaiveDate,
    forward: &CommodityForwardCurve,
    fixings: &PriceFixings,
    asof: NaiveDate,
) -> Result<f64, RustyQLibError> {
    let days = business_days_in(calendar, start, end);
    if days.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "commodity swap",
            format!("no pricing days between {start} and {end}"),
        ));
    }
    let mut total = 0.0;
    for &day in &days {
        total += if day < asof {
            fixing_on_or_before(fixings, day)?
        } else {
            forward.price(day)
        };
    }
    Ok(total / days.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat_discount(rate: f64, reference: NaiveDate) -> YieldCurve {
        YieldCurve::flat(
            rate,
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// Six-month monthly WTI-style payer swap, Sep-26 through Feb-27.
    fn six_month_payer(fixed_price: f64) -> CommoditySwap {
        CommoditySwap::monthly(
            10_000.0,
            fixed_price,
            PayerReceiver::Payer,
            d(2026, 9, 1),
            d(2027, 3, 1),
            Calendar::WeekendsOnly,
        )
        .unwrap()
    }

    #[test]
    fn monthly_periods_price_over_their_own_month() {
        let swap = six_month_payer(70.0);
        let periods = swap.periods().unwrap();
        assert_eq!(periods.len(), 6);
        // unadjusted month boundaries, contiguous
        assert_eq!(periods[0].start, d(2026, 9, 1));
        assert_eq!(periods[0].end, d(2026, 10, 1));
        assert_eq!(periods[5].end, d(2027, 3, 1));
        for pair in periods.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        // settlement lags 5 business days off the Oct 1 boundary -> Oct 8
        assert_eq!(periods[0].payment, d(2026, 10, 8));
        // September 2026 has 22 weekdays, all inside [Sep 1, Oct 1)
        let days = swap.pricing_days(&periods[0]);
        assert_eq!(days.len(), 22);
        assert_eq!(days[0], d(2026, 9, 1));
        assert_eq!(*days.last().unwrap(), d(2026, 9, 30));
    }

    #[test]
    fn flat_forward_prices_at_par_and_pv_is_the_discounted_spread() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward = CommodityForwardCurve::flat(72.0, d(2026, 9, 1)).unwrap();
        let swap = six_month_payer(70.0);
        // on a flat curve every period average is the curve level
        assert!((swap.par_price(&discount, &forward).unwrap() - 72.0).abs() < 1e-12);
        // and the PV is (F - K) * quantity * sum df
        let annuity: f64 = swap
            .periods()
            .unwrap()
            .iter()
            .map(|p| discount.df_date(p.payment))
            .sum();
        let expected = 2.0 * 10_000.0 * annuity;
        let pv = swap.pv(&discount, &forward).unwrap();
        assert!((pv - expected).abs() < 1e-8, "{pv} vs {expected}");
        // paying 70 when forwards are 72: the payer is in the money
        assert!(pv > 0.0);
    }

    #[test]
    fn at_par_the_pv_is_zero_and_sides_are_antisymmetric() {
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        // contango: 70 rising to 76 over the strip
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), 70.0), (d(2027, 3, 1), 76.0)],
        )
        .unwrap();
        let payer = six_month_payer(70.0);
        let par = payer.par_price(&discount, &forward).unwrap();
        assert!(par > 70.0 && par < 76.0, "par {par}");
        let mut at_par = payer.clone();
        at_par.fixed_price = par;
        assert!(at_par.pv(&discount, &forward).unwrap().abs() < 1e-8);
        let mut receiver = payer.clone();
        receiver.payer_receiver = PayerReceiver::Receiver;
        let p = payer.pv(&discount, &forward).unwrap();
        let r = receiver.pv(&discount, &forward).unwrap();
        assert!((p + r).abs() < 1e-10, "{p} vs {r}");
    }

    #[test]
    fn fixings_blend_into_a_partially_realized_period() {
        let asof = d(2026, 9, 16);
        let forward = CommodityForwardCurve::flat(75.0, asof).unwrap();
        // every realized day fixed at 71
        let mut fixings = PriceFixings::new();
        let mut day = d(2026, 9, 1);
        while day < asof {
            fixings.insert(day, 71.0);
            day = day.succ_opt().unwrap();
        }
        let swap = six_month_payer(70.0);
        let first = swap.periods().unwrap()[0].clone();
        // September 2026: 22 weekdays, 11 before the 16th
        let avg = swap
            .average_price_with_fixings(&first, &forward, &fixings, asof)
            .unwrap();
        let expected = (11.0 * 71.0 + 11.0 * 75.0) / 22.0;
        assert!((avg - expected).abs() < 1e-12, "{avg} vs {expected}");
        // later periods are untouched by fixings
        let second = swap.periods().unwrap()[1].clone();
        let avg2 = swap
            .average_price_with_fixings(&second, &forward, &fixings, asof)
            .unwrap();
        assert!((avg2 - 75.0).abs() < 1e-12);
    }

    #[test]
    fn settled_periods_drop_out_of_the_pv() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward = CommodityForwardCurve::flat(72.0, d(2026, 9, 1)).unwrap();
        let swap = six_month_payer(70.0);
        // fix all of September at 80 and early October on the curve at
        // 72; value the day after September's Oct 8 settlement
        let mut fixings = PriceFixings::new();
        let mut day = d(2026, 9, 1);
        while day < d(2026, 10, 1) {
            fixings.insert(day, 80.0);
            day = day.succ_opt().unwrap();
        }
        while day < d(2026, 10, 9) {
            fixings.insert(day, 72.0);
            day = day.succ_opt().unwrap();
        }
        let asof = d(2026, 10, 9);
        let pv = swap
            .pv_with_fixings(&discount, &forward, &fixings, asof)
            .unwrap();
        // identical to a five-month swap starting in October
        let stub = CommoditySwap::monthly(
            10_000.0,
            70.0,
            PayerReceiver::Payer,
            d(2026, 10, 1),
            d(2027, 3, 1),
            Calendar::WeekendsOnly,
        )
        .unwrap();
        let expected = stub.pv(&discount, &forward).unwrap();
        assert!((pv - expected).abs() < 1e-8, "{pv} vs {expected}");
    }

    #[test]
    fn delta_matches_a_parallel_forward_bump_exactly() {
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), 70.0), (d(2027, 3, 1), 76.0)],
        )
        .unwrap();
        let swap = six_month_payer(70.0);
        let delta = swap.delta(&discount).unwrap();
        let bumped = swap.pv(&discount, &forward.bumped(1.0).unwrap()).unwrap();
        let base = swap.pv(&discount, &forward).unwrap();
        assert!((bumped - base - delta).abs() < 1e-8);
        // a payer of fixed is long the commodity
        assert!(delta > 0.0);
        // and short the fixed price by the same discounted quantity
        let k_delta = swap.fixed_price_delta(&discount).unwrap();
        assert!((delta + k_delta).abs() < 1e-10, "{delta} vs {k_delta}");
    }

    #[test]
    fn mid_life_delta_scales_with_the_unfixed_fraction() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let swap = six_month_payer(70.0);
        let full = swap.delta(&discount).unwrap();
        // half of September fixed: delta must sit between 5.5 and 6 months' worth
        let mid = swap.delta_with_fixings(&discount, d(2026, 9, 16)).unwrap();
        assert!(mid < full && mid > full * 5.0 / 6.0, "{mid} vs {full}");
    }

    #[test]
    fn dv01_discounts_the_in_the_money_payer_less() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward = CommodityForwardCurve::flat(75.0, d(2026, 9, 1)).unwrap();
        let swap = six_month_payer(70.0);
        // positive future cashflows discounted harder: PV falls as rates rise
        assert!(swap.dv01(&discount, &forward).unwrap() < 0.0);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let e = d(2026, 9, 1);
        let m = d(2027, 3, 1);
        let cal = Calendar::WeekendsOnly;
        assert!(
            CommoditySwap::monthly(0.0, 70.0, PayerReceiver::Payer, e, m, cal.clone()).is_err()
        );
        assert!(CommoditySwap::monthly(
            10_000.0,
            f64::NAN,
            PayerReceiver::Payer,
            e,
            m,
            cal.clone()
        )
        .is_err());
        assert!(CommoditySwap::monthly(10_000.0, 70.0, PayerReceiver::Payer, e, e, cal).is_err());
    }
}
