//! Floating-for-floating commodity basis swap.
//!
//! The differential trade between two commodity indexes — WTI versus
//! Brent, a delivery point versus the hub benchmark (location basis), or
//! one grade versus another (quality basis). Over each calculation
//! period the position **receives the arithmetic average of index A's
//! daily price plus a fixed `spread`, and pays the average of index
//! B's**, each averaged over its own pricing calendar's business days,
//! times the period's notional quantity, cash-settled after the period
//! ends.
//!
//! Averaging conventions (pricing days, fixings carry-forward, the
//! settlement lag) are exactly those of
//! [`CommoditySwap`](crate::cmdty::CommoditySwap); the two legs share
//! one calculation-period schedule but may observe different holiday
//! calendars, as cross-market indexes do.

use chrono::NaiveDate;

use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::cmdty::swap::{
    business_days_in, period_average, positive_annuity, PeriodSchedule, PriceFixings,
};
use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::leg::AccrualPeriod;

const FIELD: &str = "commodity basis swap";

/// A commodity basis swap from the point of view of the party
/// **receiving index A plus the spread** and paying index B.
#[derive(Debug, Clone)]
pub struct CommodityBasisSwap {
    /// Notional quantity per calculation period, in the indexes' units.
    pub quantity: f64,
    /// Fixed differential added to leg A's average, in price units per
    /// unit (e.g. `-4.25` $/bbl for a WTI-minus-Brent leg).
    pub spread: f64,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub frequency: Frequency,
    /// Pricing calendar of index A (the received leg).
    pub calendar_a: Calendar,
    /// Pricing calendar of index B (the paid leg).
    pub calendar_b: Calendar,
    /// Adjustment for the shared period boundaries (`Unadjusted` is the
    /// calendar-month standard). The schedule and the settlement lag run
    /// on `calendar_a`.
    pub convention: BusinessDayConvention,
    /// Business days between a period's end date and its cash
    /// settlement, counted on `calendar_a`.
    pub payment_lag: i64,
}

impl CommodityBasisSwap {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        quantity: f64,
        spread: f64,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        frequency: Frequency,
        calendar_a: Calendar,
        calendar_b: Calendar,
        convention: BusinessDayConvention,
        payment_lag: i64,
    ) -> Result<Self, RustyQLibError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity basis swap",
                format!("quantity must be positive, got {quantity}"),
            ));
        }
        if !spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "commodity basis swap",
                format!("spread must be finite, got {spread}"),
            ));
        }
        if maturity_date <= effective_date {
            return Err(RustyQLibError::invalid_input(
                "commodity basis swap",
                format!("maturity {maturity_date} must be after effective {effective_date}"),
            ));
        }
        if payment_lag < 0 {
            return Err(RustyQLibError::invalid_input(
                "commodity basis swap",
                format!("payment lag must be non-negative, got {payment_lag}"),
            ));
        }
        Ok(CommodityBasisSwap {
            quantity,
            spread,
            effective_date,
            maturity_date,
            frequency,
            calendar_a,
            calendar_b,
            convention,
            payment_lag,
        })
    }

    /// A standard monthly basis swap: unadjusted calendar-month periods
    /// settling 5 business days after each month ends (the
    /// [`CommoditySwap::monthly`](crate::cmdty::CommoditySwap::monthly)
    /// conventions).
    pub fn monthly(
        quantity: f64,
        spread: f64,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        calendar_a: Calendar,
        calendar_b: Calendar,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            quantity,
            spread,
            effective_date,
            maturity_date,
            Frequency::Monthly,
            calendar_a,
            calendar_b,
            BusinessDayConvention::Unadjusted,
            5,
        )
    }

    /// The quantity-free calculation-period schedule (on `calendar_a`).
    fn schedule(&self) -> PeriodSchedule<'_> {
        PeriodSchedule {
            field: FIELD,
            effective: self.effective_date,
            maturity: self.maturity_date,
            frequency: self.frequency,
            calendar: &self.calendar_a,
            convention: self.convention,
            payment_lag: self.payment_lag,
        }
    }

    /// The shared calculation periods, each with its lagged settlement
    /// date (schedule on `calendar_a`).
    pub fn periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        self.schedule().periods()
    }

    /// One leg's averaging observations in a period: the business days
    /// of that leg's calendar in `[start, end)`.
    pub fn pricing_days(&self, period: &AccrualPeriod, calendar: &Calendar) -> Vec<NaiveDate> {
        business_days_in(calendar, period.start, period.end)
    }

    /// Swap PV as of the discount curve's reference date, both legs
    /// fully projected: per period,
    /// `quantity * (avg_a + spread - avg_b) * df(payment)`. A seasoned
    /// swap needs [`pv_with_fixings`](Self::pv_with_fixings).
    pub fn pv(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.pv_with_fixings(
            discount,
            forward_a,
            forward_b,
            &PriceFixings::new(),
            &PriceFixings::new(),
            discount.reference_date(),
        )
    }

    /// Swap PV mid-life: each leg blends its own realized fixings with
    /// its forward curve, and periods already settled (payment on or
    /// before `asof`) drop out. `asof` may not precede the discount
    /// curve's reference date.
    #[allow(clippy::too_many_arguments)]
    pub fn pv_with_fixings(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        fixings_a: &PriceFixings,
        fixings_b: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let mut pv = 0.0;
        for period in self.unsettled_periods(discount, asof)? {
            let avg_a = period_average(
                FIELD,
                &self.calendar_a,
                period.start,
                period.end,
                forward_a,
                fixings_a,
                asof,
            )?;
            let avg_b = period_average(
                FIELD,
                &self.calendar_b,
                period.start,
                period.end,
                forward_b,
                fixings_b,
                asof,
            )?;
            pv += (avg_a + self.spread - avg_b) * discount.df_date(period.payment);
        }
        Ok(self.quantity * pv)
    }

    /// The fair spread on leg A: the differential that makes the PV
    /// zero — the discount-weighted average of `avg_b - avg_a` — as of
    /// the discount curve's reference date.
    pub fn fair_spread(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.fair_spread_with_fixings(
            discount,
            forward_a,
            forward_b,
            &PriceFixings::new(),
            &PriceFixings::new(),
            discount.reference_date(),
        )
    }

    /// The fair spread mid-life, given each leg's realized fixings.
    #[allow(clippy::too_many_arguments)]
    pub fn fair_spread_with_fixings(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        fixings_a: &PriceFixings,
        fixings_b: &PriceFixings,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let annuity = positive_annuity(FIELD, self.discount_annuity(discount, asof)?)?;
        // PV is linear in the spread with slope quantity * annuity
        let mut zero_spread = self.clone();
        zero_spread.spread = 0.0;
        let pv = zero_spread
            .pv_with_fixings(discount, forward_a, forward_b, fixings_a, fixings_b, asof)?;
        Ok(-pv / (self.quantity * annuity))
    }

    /// PV change per one-unit parallel increase of index A's forward
    /// curve, everything still floating — `quantity * sum df` (a parallel
    /// bump moves every period average one-for-one, whatever the day
    /// count). Exact, by linearity.
    pub fn delta_a(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(self.quantity * self.discount_annuity(discount, discount.reference_date())?)
    }

    /// PV change per one-unit parallel increase of index B's forward
    /// curve: minus [`delta_a`](Self::delta_a) — the basis position is
    /// flat the outright price and long only the differential.
    pub fn delta_b(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(-self.delta_a(discount)?)
    }

    fn unsettled_periods(
        &self,
        discount: &YieldCurve,
        asof: NaiveDate,
    ) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        self.schedule().unsettled_periods(discount, asof)
    }

    /// `sum df(pay_i)` over unsettled periods — per unit of quantity,
    /// unlike [`CommoditySwap::settlement_annuity`], which folds the
    /// notional in.
    fn discount_annuity(
        &self,
        discount: &YieldCurve,
        asof: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.schedule().discount_annuity(discount, asof)
    }
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

    /// Six-month monthly WTI-vs-Brent-style basis swap, Sep-26 through
    /// Feb-27, on 10,000 bbl a month.
    fn six_month_basis(spread: f64, calendar_b: Calendar) -> CommodityBasisSwap {
        CommodityBasisSwap::monthly(
            10_000.0,
            spread,
            d(2026, 9, 1),
            d(2027, 3, 1),
            Calendar::WeekendsOnly,
            calendar_b,
        )
        .unwrap()
    }

    #[test]
    fn identical_legs_price_to_the_spread_annuity() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward = CommodityForwardCurve::flat(72.0, d(2026, 9, 1)).unwrap();
        // same index both sides: PV is exactly the spread annuity
        let swap = six_month_basis(0.0, Calendar::WeekendsOnly);
        assert_eq!(swap.pv(&discount, &forward, &forward).unwrap(), 0.0);
        let fair = swap.fair_spread(&discount, &forward, &forward).unwrap();
        assert!(fair.abs() < 1e-14, "fair {fair}");
        let with_spread = six_month_basis(1.5, Calendar::WeekendsOnly);
        let annuity: f64 = with_spread
            .periods()
            .unwrap()
            .iter()
            .map(|p| discount.df_date(p.payment))
            .sum();
        let pv = with_spread.pv(&discount, &forward, &forward).unwrap();
        let expected = 10_000.0 * 1.5 * annuity;
        assert!((pv - expected).abs() < 1e-8, "{pv} vs {expected}");
    }

    #[test]
    fn fair_spread_recovers_the_curve_differential() {
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        // Brent-style leg B trades $4.25 over WTI-style leg A
        let wti = CommodityForwardCurve::flat(72.0, reference).unwrap();
        let brent = CommodityForwardCurve::flat(76.25, reference).unwrap();
        let swap = six_month_basis(0.0, Calendar::WeekendsOnly);
        let fair = swap.fair_spread(&discount, &wti, &brent).unwrap();
        assert!((fair - 4.25).abs() < 1e-12, "fair {fair}");
        // priced at the fair spread the swap is worth zero
        let mut at_fair = swap.clone();
        at_fair.spread = fair;
        assert!(at_fair.pv(&discount, &wti, &brent).unwrap().abs() < 1e-8);
        // receiving the cheaper index with no spread loses money
        assert!(swap.pv(&discount, &wti, &brent).unwrap() < 0.0);
    }

    #[test]
    fn legs_average_over_their_own_calendars() {
        // September 2026: Labor Day (Sep 7) is a US holiday, so the NYSE
        // leg observes one day fewer than the weekends-only leg
        let swap = six_month_basis(0.0, Calendar::UsNyse);
        let first = swap.periods().unwrap()[0];
        let days_a = swap.pricing_days(&first, &swap.calendar_a);
        let days_b = swap.pricing_days(&first, &swap.calendar_b);
        assert_eq!(days_a.len(), 22);
        assert_eq!(days_b.len(), 21);
        assert!(days_a.contains(&d(2026, 9, 7)));
        assert!(!days_b.contains(&d(2026, 9, 7)));
        // flat curves: averages coincide regardless of the day sets, so
        // the calendar mismatch alone contributes no fair spread
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward = CommodityForwardCurve::flat(72.0, d(2026, 9, 1)).unwrap();
        let fair = swap.fair_spread(&discount, &forward, &forward).unwrap();
        assert!(fair.abs() < 1e-12, "fair {fair}");
    }

    #[test]
    fn fixings_blend_per_leg_and_settled_periods_drop() {
        // one-month swap valued mid-month with different realized levels
        let swap = CommodityBasisSwap::monthly(
            10_000.0,
            0.0,
            d(2026, 9, 1),
            d(2026, 10, 1),
            Calendar::WeekendsOnly,
            Calendar::WeekendsOnly,
        )
        .unwrap();
        let asof = d(2026, 9, 16);
        let discount = flat_discount(0.04, asof);
        let forward_a = CommodityForwardCurve::flat(76.0, asof).unwrap();
        let forward_b = CommodityForwardCurve::flat(72.0, asof).unwrap();
        let (mut fixings_a, mut fixings_b) = (PriceFixings::new(), PriceFixings::new());
        let mut day = d(2026, 9, 1);
        while day < asof {
            fixings_a.insert(day, 80.0);
            fixings_b.insert(day, 71.0);
            day = day.succ_opt().unwrap();
        }
        let pv = swap
            .pv_with_fixings(
                &discount, &forward_a, &forward_b, &fixings_a, &fixings_b, asof,
            )
            .unwrap();
        // 11 of September's 22 weekdays realized on each side
        let avg_a = (11.0 * 80.0 + 11.0 * 76.0) / 22.0;
        let avg_b = (11.0 * 71.0 + 11.0 * 72.0) / 22.0;
        let period = swap.periods().unwrap()[0];
        let expected = 10_000.0 * (avg_a - avg_b) * discount.df_date(period.payment);
        assert!((pv - expected).abs() < 1e-8, "{pv} vs {expected}");
        // valued after settlement the swap has nothing left
        let after = d(2026, 10, 9);
        let late = swap
            .pv_with_fixings(
                &flat_discount(0.04, after),
                &forward_a,
                &forward_b,
                &fixings_a,
                &fixings_b,
                after,
            )
            .unwrap();
        assert_eq!(late, 0.0);
    }

    #[test]
    fn deltas_are_equal_opposite_and_match_curve_bumps() {
        let discount = flat_discount(0.04, d(2026, 9, 1));
        let forward_a = CommodityForwardCurve::flat(72.0, d(2026, 9, 1)).unwrap();
        let forward_b = CommodityForwardCurve::flat(76.0, d(2026, 9, 1)).unwrap();
        let swap = six_month_basis(4.0, Calendar::UsNyse);
        let delta_a = swap.delta_a(&discount).unwrap();
        let delta_b = swap.delta_b(&discount).unwrap();
        assert!((delta_a + delta_b).abs() < 1e-10);
        let base = swap.pv(&discount, &forward_a, &forward_b).unwrap();
        let bumped_a = swap
            .pv(&discount, &forward_a.bumped(1.0).unwrap(), &forward_b)
            .unwrap();
        let bumped_b = swap
            .pv(&discount, &forward_a, &forward_b.bumped(1.0).unwrap())
            .unwrap();
        assert!((bumped_a - base - delta_a).abs() < 1e-8);
        assert!((bumped_b - base - delta_b).abs() < 1e-8);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let (e, m) = (d(2026, 9, 1), d(2027, 3, 1));
        let cal = Calendar::WeekendsOnly;
        assert!(CommodityBasisSwap::monthly(0.0, 0.0, e, m, cal.clone(), cal.clone()).is_err());
        assert!(
            CommodityBasisSwap::monthly(1e4, f64::NAN, e, m, cal.clone(), cal.clone()).is_err()
        );
        assert!(CommodityBasisSwap::monthly(1e4, 0.0, e, e, cal.clone(), cal.clone()).is_err());
        // a negative payment lag is rejected at construction
        assert!(CommodityBasisSwap::new(
            1e4,
            0.0,
            e,
            m,
            Frequency::Monthly,
            cal.clone(),
            cal,
            BusinessDayConvention::Unadjusted,
            -1,
        )
        .is_err());
    }

    #[test]
    fn seasoned_valuation_needs_fixings_and_respects_the_reference_date() {
        let asof = d(2026, 9, 16);
        let discount = flat_discount(0.04, asof);
        let forward = CommodityForwardCurve::flat(72.0, asof).unwrap();
        let swap = six_month_basis(0.0, Calendar::WeekendsOnly);
        // September's realized days cannot be read off the curve
        assert!(swap.pv(&discount, &forward, &forward).is_err());
        assert!(swap.fair_spread(&discount, &forward, &forward).is_err());
        // an as-of before the curve's reference is rejected outright
        let (fa, fb) = (PriceFixings::new(), PriceFixings::new());
        assert!(swap
            .pv_with_fixings(&discount, &forward, &forward, &fa, &fb, d(2026, 9, 1))
            .is_err());
        // a period paying exactly on the valuation date has settled
        let settled = d(2026, 10, 8);
        let late = flat_discount(0.04, settled);
        let periods = swap.periods().unwrap();
        assert_eq!(periods[0].payment, settled);
        let annuity: f64 = periods
            .iter()
            .filter(|p| p.payment > settled)
            .map(|p| late.df_date(p.payment))
            .sum();
        assert!((swap.delta_a(&late).unwrap() - 10_000.0 * annuity).abs() < 1e-10);
    }
}
