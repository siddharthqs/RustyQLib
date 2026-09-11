//! Shared leg machinery for swaps: schedule construction and the fixed
//! and floating leg present values.
//!
//! A floating leg may **reset more often than it pays** — a 1M index on
//! a quarterly leg — in which case the sub-period accruals are combined
//! by a [`CompoundingMethod`] (ISDA 2006 straight, flat and
//! spread-exclusive compounding, or none). [`float_leg_pv_compounded`]
//! prices such a leg from [`FloatPeriod`]s, taking past resets from a
//! fixing history keyed by the reset period's accrual start.

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::checked_df;
use crate::rates::overnight::RateFixings;
use crate::rates::schedule::LegSchedule;

/// One swap accrual period. Unlike bonds, swap accrual runs between
/// business-day **adjusted** dates; payment can lag the accrual end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AccrualPeriod {
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub payment: NaiveDate,
}

/// Build the accrual periods of one leg under the default conventions:
/// unadjusted anchors rolled backward from `maturity` (short stub at
/// the front, the maturity's day of month), each anchor adjusted on
/// `calendar` under `convention`, payments lagged `payment_lag` business
/// days after the accrual end. Anchors that collapse after adjustment
/// are dropped. For other stub and roll conventions build a
/// [`LegSchedule`].
pub fn accrual_periods(
    effective: NaiveDate,
    maturity: NaiveDate,
    frequency: Frequency,
    calendar: &Calendar,
    convention: BusinessDayConvention,
    payment_lag: i64,
) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
    LegSchedule::new(effective, maturity, frequency, calendar.clone(), convention)
        .with_payment_lag(payment_lag)
        .accrual_periods()
}

/// How the accruals of a floating leg's reset sub-periods combine into
/// one payment (ISDA 2006 Definitions). Irrelevant when the leg resets
/// once per payment period.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompoundingMethod {
    /// No compounding: the sub-period accruals `(r_i + s) tau_i` are
    /// summed.
    #[default]
    None,
    /// Straight: each sub-period's amount accrues on the notional plus
    /// every earlier amount, at rate plus spread —
    /// `prod(1 + (r_i + s) tau_i) - 1`.
    Straight,
    /// Flat: the spread accrues on the notional only; earlier amounts
    /// (spread included) compound at the rate alone —
    /// `A_i = (r_i + s) tau_i + (sum_{j<i} A_j) r_i tau_i`.
    Flat,
    /// Spread exclusive: the rate compounds, the spread never does —
    /// `prod(1 + r_i tau_i) - 1 + s sum tau_i`.
    SpreadExclusive,
}

impl CompoundingMethod {
    /// Combine `(rate, tau)` sub-period pairs and the spread into the
    /// payment period's accrual on unit notional.
    pub fn accrual(self, resets: &[(f64, f64)], spread: f64) -> f64 {
        match self {
            CompoundingMethod::None => resets.iter().map(|&(r, tau)| (r + spread) * tau).sum(),
            CompoundingMethod::Straight => {
                resets
                    .iter()
                    .fold(1.0, |acc, &(r, tau)| acc * (1.0 + (r + spread) * tau))
                    - 1.0
            }
            CompoundingMethod::Flat => {
                let mut accrued = 0.0;
                for &(r, tau) in resets {
                    accrued += (r + spread) * tau + accrued * r * tau;
                }
                accrued
            }
            CompoundingMethod::SpreadExclusive => {
                let compounded = resets
                    .iter()
                    .fold(1.0, |acc, &(r, tau)| acc * (1.0 + r * tau))
                    - 1.0;
                compounded + spread * resets.iter().map(|&(_, tau)| tau).sum::<f64>()
            }
        }
    }
}

/// One payment period of a floating leg with its reset sub-periods
/// (accrual start and end of each reset; a single reset spanning the
/// period when the leg resets as often as it pays).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatPeriod {
    pub period: AccrualPeriod,
    pub resets: Vec<(NaiveDate, NaiveDate)>,
}

/// The floating periods of `schedule`, each split into resets every
/// `reset` (`None`: one reset per payment period). The reset schedule
/// shares the payment schedule's stub, roll and adjustment terms, so
/// the two nest when the payment frequency is a multiple of the reset
/// frequency; anything else is an error.
pub fn float_periods(
    schedule: &LegSchedule,
    reset: Option<Frequency>,
) -> Result<Vec<FloatPeriod>, RustyQLibError> {
    let payments = schedule.accrual_periods()?;
    let Some(reset) = reset else {
        return Ok(payments
            .into_iter()
            .map(|p| FloatPeriod {
                resets: vec![(p.start, p.end)],
                period: p,
            })
            .collect());
    };
    if reset.months() > schedule.frequency.months()
        || !schedule.frequency.months().is_multiple_of(reset.months())
    {
        return Err(RustyQLibError::invalid_input(
            "float leg",
            format!(
                "reset frequency {reset:?} must divide the payment frequency {:?}",
                schedule.frequency
            ),
        ));
    }
    let fine = schedule.at_frequency(reset).accrual_periods()?;
    let mut out = Vec::with_capacity(payments.len());
    for p in payments {
        let resets: Vec<(NaiveDate, NaiveDate)> = fine
            .iter()
            .filter(|f| f.start >= p.start && f.end <= p.end)
            .map(|f| (f.start, f.end))
            .collect();
        let covers = resets.first().map(|r| r.0) == Some(p.start)
            && resets.last().map(|r| r.1) == Some(p.end)
            && resets.windows(2).all(|w| w[0].1 == w[1].0);
        if !covers {
            return Err(RustyQLibError::invalid_input(
                "float leg",
                format!(
                    "the reset schedule does not tile the payment period {}..{}",
                    p.start, p.end
                ),
            ));
        }
        out.push(FloatPeriod { period: p, resets });
    }
    Ok(out)
}

/// PV on unit notional of a floating leg with reset sub-periods. Each
/// reset's simple rate is the forward off `forecast` — or, for a reset
/// that started before the forecast curve's reference date, the
/// published rate in `fixings` keyed by that reset's accrual start
/// (an error if missing). Sub-periods combine by `compounding`, and
/// each payment discounts on `discount`; settled periods contribute
/// nothing.
pub fn float_leg_pv_compounded(
    periods: &[FloatPeriod],
    spread: f64,
    day_count: DayCountConvention,
    compounding: CompoundingMethod,
    discount: &YieldCurve,
    forecast: &YieldCurve,
    fixings: Option<&RateFixings>,
) -> Result<f64, RustyQLibError> {
    let valuation = discount.reference_date();
    let fixing_horizon = forecast.reference_date();
    let mut pv = 0.0;
    for fp in periods {
        if fp.period.payment <= valuation {
            continue;
        }
        let mut resets = Vec::with_capacity(fp.resets.len());
        for &(start, end) in &fp.resets {
            let tau = day_count.year_fraction(start, end);
            let rate = if start >= fixing_horizon {
                (checked_df(forecast, start)? / checked_df(forecast, end)? - 1.0) / tau
            } else {
                fixings
                    .and_then(|f| f.get(&start))
                    .copied()
                    .ok_or_else(|| {
                        RustyQLibError::invalid_input(
                            "swap leg",
                            format!(
                                "the reset starting {start} fixed before the forecast curve \
                                 reference {fixing_horizon}: supply its fixing"
                            ),
                        )
                    })?
            };
            resets.push((rate, tau));
        }
        pv += compounding.accrual(&resets, spread) * discount.df_date(fp.period.payment);
    }
    Ok(pv)
}

/// PV of a fixed leg: `sum rate * tau_i * df(pay_i)` on unit notional
/// over the unsettled periods (see [`annuity`]).
pub fn fixed_leg_pv(
    periods: &[AccrualPeriod],
    rate: f64,
    day_count: DayCountConvention,
    discount: &YieldCurve,
) -> f64 {
    rate * annuity(periods, day_count, discount)
}

/// The fixed-leg annuity `sum tau_i * df(pay_i)` on unit notional — the
/// PV of 1 unit of rate, and the sensitivity of the swap to its fixed
/// rate. Periods already settled — payment on or before the discount
/// curve's reference date — contribute nothing, so a seasoned swap
/// prices only its remaining cashflows (the curve would otherwise clamp
/// their discount factors to 1 and sum past coupons at face).
pub fn annuity(
    periods: &[AccrualPeriod],
    day_count: DayCountConvention,
    discount: &YieldCurve,
) -> f64 {
    let valuation = discount.reference_date();
    periods
        .iter()
        .filter(|p| p.payment > valuation)
        .map(|p| day_count.year_fraction(p.start, p.end) * discount.df_date(p.payment))
        .sum()
}

/// PV of a floating leg plus `spread` on unit notional. Each period's
/// floating accrual is forecast from the curve ratio
/// `df(start)/df(end) - 1` (the simple forward, and equally the
/// compounded overnight accrual); the spread accrues on `day_count`.
/// Periods already settled (payment on or before the discount curve's
/// reference date) contribute nothing; a period that began before the
/// forecast curve's reference date fixed in the past and is an error
/// here — supply its realized rate through
/// [`float_leg_pv_with_fixing`].
pub fn float_leg_pv(
    periods: &[AccrualPeriod],
    spread: f64,
    day_count: DayCountConvention,
    discount: &YieldCurve,
    forecast: &YieldCurve,
) -> Result<f64, RustyQLibError> {
    float_leg_pv_with_fixing(periods, spread, day_count, discount, forecast, None)
}

/// [`float_leg_pv`] for a seasoned leg. `realized_rate` is the
/// annualized simple rate (on `day_count`) realized from the current
/// period's start through the forecast curve's reference date — for a
/// compounded overnight leg, the simple-rate equivalent of the
/// compounding realized so far. The period in progress then accrues
/// `(1 + realized_rate * tau_elapsed) / df(end) - 1` — the realized
/// stub grown at the curve — and every other period prices as usual.
///
/// A period that is fully elapsed on the forecast curve but pays after
/// the valuation date (a payment lag straddling the reference) cannot
/// be represented by the single stub rate and is rejected.
pub fn float_leg_pv_with_fixing(
    periods: &[AccrualPeriod],
    spread: f64,
    day_count: DayCountConvention,
    discount: &YieldCurve,
    forecast: &YieldCurve,
    realized_rate: Option<f64>,
) -> Result<f64, RustyQLibError> {
    if let Some(rate) = realized_rate {
        if !rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "realized_rate",
                format!("must be finite, got {rate}"),
            ));
        }
    }
    let valuation = discount.reference_date();
    let fixing_horizon = forecast.reference_date();
    let mut pv = 0.0;
    for p in periods {
        if p.payment <= valuation {
            continue; // settled
        }
        let tau = day_count.year_fraction(p.start, p.end);
        let accrual = if p.start >= fixing_horizon {
            checked_df(forecast, p.start)? / checked_df(forecast, p.end)? - 1.0
        } else if p.end > fixing_horizon {
            // the period in progress: realized stub, grown at the curve
            let Some(rate) = realized_rate else {
                return Err(RustyQLibError::invalid_input(
                    "swap leg",
                    format!(
                        "the period starting {} began before the forecast curve reference \
                         {}: supply the realized rate — a historical fixing cannot be \
                         read off the curve",
                        p.start, fixing_horizon
                    ),
                ));
            };
            let realized = 1.0 + rate * day_count.year_fraction(p.start, fixing_horizon);
            realized / checked_df(forecast, p.end)? - 1.0
        } else {
            // fully elapsed, but paying after the valuation date
            return Err(RustyQLibError::invalid_input(
                "swap leg",
                format!(
                    "the period ending {} is fully elapsed on the forecast curve reference \
                     {} but pays {} — its realized accrual cannot be read off the curve",
                    p.end, fixing_horizon, p.payment
                ),
            ));
        };
        pv += (accrual + spread * tau) * discount.df_date(p.payment);
    }
    Ok(pv)
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

    #[test]
    fn compounding_methods_match_the_isda_formulas_and_order() {
        // two monthly resets at 4% and 5%, 25bp spread, tau = 1/12
        let resets = [(0.04, 1.0 / 12.0), (0.05, 1.0 / 12.0)];
        let s = 0.0025;
        let none = CompoundingMethod::None.accrual(&resets, s);
        let straight = CompoundingMethod::Straight.accrual(&resets, s);
        let flat = CompoundingMethod::Flat.accrual(&resets, s);
        let exclusive = CompoundingMethod::SpreadExclusive.accrual(&resets, s);
        let tau = 1.0 / 12.0;
        assert!((none - (0.0425 + 0.0525) * tau).abs() < 1e-15);
        assert!((straight - ((1.0 + 0.0425 * tau) * (1.0 + 0.0525 * tau) - 1.0)).abs() < 1e-15);
        let a1 = 0.0425 * tau;
        assert!((flat - (a1 + 0.0525 * tau + a1 * 0.05 * tau)).abs() < 1e-15);
        assert!(
            (exclusive - ((1.0 + 0.04 * tau) * (1.0 + 0.05 * tau) - 1.0 + s * 2.0 * tau)).abs()
                < 1e-15
        );
        // straight compounds the spread, flat compounds earlier spread
        // amounts at the rate, exclusive never compounds the spread
        assert!(straight > flat && flat > exclusive && exclusive > none);
        // one reset: every method is the simple accrual
        let one = [(0.04, 0.25)];
        for m in [
            CompoundingMethod::None,
            CompoundingMethod::Straight,
            CompoundingMethod::Flat,
            CompoundingMethod::SpreadExclusive,
        ] {
            assert!((m.accrual(&one, s) - 0.0425 * 0.25).abs() < 1e-15, "{m:?}");
        }
    }

    #[test]
    fn monthly_resets_on_a_quarterly_leg_telescope_under_straight_compounding() {
        // zero spread, straight compounding: the product of the monthly
        // df ratios is the quarterly df ratio, so the leg equals the
        // single-reset leg exactly
        let reference = d(2026, 8, 6);
        let curve = flat(0.045, reference);
        let schedule = LegSchedule::new(
            reference,
            d(2028, 8, 6),
            Frequency::Quarterly,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
        );
        let monthly = float_periods(&schedule, Some(Frequency::Monthly)).unwrap();
        assert_eq!(monthly.len(), 8);
        assert!(monthly.iter().all(|p| p.resets.len() == 3));
        let single = float_periods(&schedule, None).unwrap();
        let dc = DayCountConvention::Act360;
        let compounded = float_leg_pv_compounded(
            &monthly,
            0.0,
            dc,
            CompoundingMethod::Straight,
            &curve,
            &curve,
            None,
        )
        .unwrap();
        let plain = float_leg_pv_compounded(
            &single,
            0.0,
            dc,
            CompoundingMethod::None,
            &curve,
            &curve,
            None,
        )
        .unwrap();
        let legacy = float_leg_pv(
            &schedule.accrual_periods().unwrap(),
            0.0,
            dc,
            &curve,
            &curve,
        )
        .unwrap();
        assert!(
            (compounded - plain).abs() < 1e-12,
            "{compounded} vs {plain}"
        );
        assert!((plain - legacy).abs() < 1e-15);
        // a reset frequency that does not divide the payment frequency is refused
        assert!(float_periods(&schedule, Some(Frequency::Semiannual)).is_err());
    }

    #[test]
    fn past_resets_come_from_the_fixing_history() {
        // valued four weeks into the first quarter: the first monthly
        // reset fixed in the past, the second has not started
        let effective = d(2026, 8, 6);
        let valuation = d(2026, 9, 3);
        let curve = flat(0.045, valuation);
        let schedule = LegSchedule::new(
            effective,
            d(2027, 8, 6),
            Frequency::Quarterly,
            Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
        );
        let periods = float_periods(&schedule, Some(Frequency::Monthly)).unwrap();
        let dc = DayCountConvention::Act360;
        // without the fixing: refused
        assert!(float_leg_pv_compounded(
            &periods,
            0.0,
            dc,
            CompoundingMethod::None,
            &curve,
            &curve,
            None
        )
        .is_err());
        let mut fixings = RateFixings::new();
        fixings.insert(effective, 0.05);
        let with = float_leg_pv_compounded(
            &periods,
            0.0,
            dc,
            CompoundingMethod::None,
            &curve,
            &curve,
            Some(&fixings),
        )
        .unwrap();
        // the fixed month at 5% versus the curve's 4.5%: the leg is worth
        // more than the same leg with a 4.5% fixing
        fixings.insert(effective, 0.045);
        let lower = float_leg_pv_compounded(
            &periods,
            0.0,
            dc,
            CompoundingMethod::None,
            &curve,
            &curve,
            Some(&fixings),
        )
        .unwrap();
        let tau = dc.year_fraction(effective, d(2026, 9, 6));
        let expected_gap = 0.005 * tau * curve.df_date(d(2026, 11, 6));
        assert!(
            (with - lower - expected_gap).abs() < 1e-12,
            "{with} vs {lower}"
        );
    }

    #[test]
    fn quarterly_schedule_rolls_backward_and_adjusts() {
        // effective Thu 2026-08-06, 1y quarterly, modified following
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            0,
        )
        .unwrap();
        assert_eq!(periods.len(), 4);
        // contiguous: each period starts where the previous ended
        for pair in periods.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        // Nov 6 2026 is a Friday, stays; Feb 6 2027 is a Saturday -> Mon Feb 8
        assert_eq!(periods[0].end, d(2026, 11, 6));
        assert_eq!(periods[1].end, d(2027, 2, 8));
        // Aug 6 2027 is a Friday
        assert_eq!(periods[3].end, d(2027, 8, 6));
        // no lag: payment = accrual end
        assert!(periods.iter().all(|p| p.payment == p.end));
    }

    #[test]
    fn payment_lag_shifts_payments_by_business_days() {
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            2,
        )
        .unwrap();
        // first period ends Fri Nov 6 2026 -> T+2 is Tue Nov 10
        assert_eq!(periods[0].payment, d(2026, 11, 10));
        for p in &periods {
            assert!(p.payment > p.end);
        }
    }

    #[test]
    fn float_leg_telescopes_on_a_single_curve() {
        // unadjusted periods and zero lag: the df ratios telescope so the
        // float leg is exactly df(effective) - df(maturity) = 1 - df(T)
        let reference = d(2026, 8, 6);
        let curve = flat(0.045, reference);
        let periods = accrual_periods(
            reference,
            d(2028, 8, 6),
            Frequency::Quarterly,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
            0,
        )
        .unwrap();
        let pv = float_leg_pv(&periods, 0.0, DayCountConvention::Act360, &curve, &curve).unwrap();
        let expected = 1.0 - curve.df_date(d(2028, 8, 6));
        assert!((pv - expected).abs() < 1e-14, "{pv} vs {expected}");
    }

    #[test]
    fn annuity_is_the_fixed_rate_sensitivity() {
        let reference = d(2026, 8, 6);
        let curve = flat(0.04, reference);
        let periods = accrual_periods(
            reference,
            d(2028, 8, 6),
            Frequency::Semiannual,
            &Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
            0,
        )
        .unwrap();
        let dc = DayCountConvention::Thirty360;
        let a = annuity(&periods, dc, &curve);
        assert!(a > 0.0);
        // fixed_leg_pv is linear in the rate with slope = annuity
        let pv1 = fixed_leg_pv(&periods, 0.04, dc, &curve);
        let pv2 = fixed_leg_pv(&periods, 0.05, dc, &curve);
        assert!(((pv2 - pv1) - 0.01 * a).abs() < 1e-15);
    }

    #[test]
    fn seasoned_legs_drop_settled_periods_and_need_the_current_fixing() {
        // 2y quarterly leg effective 2026-08-06, priced mid-life on
        // 2027-01-15: two periods settled, one in progress, the rest ahead
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2028, 8, 6),
            Frequency::Quarterly,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Unadjusted,
            0,
        )
        .unwrap();
        let reference = d(2027, 1, 15);
        let curve = flat(0.04, reference);
        let dc = DayCountConvention::Act360;
        assert!(
            periods.iter().any(|p| p.payment <= reference),
            "the test needs settled periods"
        );

        // annuity: only periods paying after the reference remain (the
        // old behavior summed past coupons at df = 1)
        let a = annuity(&periods, dc, &curve);
        let manual: f64 = periods
            .iter()
            .filter(|p| p.payment > reference)
            .map(|p| dc.year_fraction(p.start, p.end) * curve.df_date(p.payment))
            .sum();
        assert!((a - manual).abs() < 1e-15, "{a} vs {manual}");

        // the float leg refuses to project the period fixed in the past...
        let err = float_leg_pv(&periods, 0.0, dc, &curve, &curve);
        assert!(format!("{}", err.unwrap_err()).contains("realized rate"));

        // ...and with the realized rate supplied, prices settled-free with
        // the realized stub grown at the curve
        let realized = 0.05;
        let pv =
            float_leg_pv_with_fixing(&periods, 0.0, dc, &curve, &curve, Some(realized)).unwrap();
        let mut manual_pv = 0.0;
        for p in &periods {
            if p.payment <= reference {
                continue;
            }
            let accrual = if p.start >= reference {
                curve.df_date(p.start) / curve.df_date(p.end) - 1.0
            } else {
                (1.0 + realized * dc.year_fraction(p.start, reference)) / curve.df_date(p.end) - 1.0
            };
            manual_pv += accrual * curve.df_date(p.payment);
        }
        assert!((pv - manual_pv).abs() < 1e-15, "{pv} vs {manual_pv}");
    }

    #[test]
    fn elapsed_but_unpaid_periods_are_rejected() {
        // T+5 payment lag and a reference between a period's end and its
        // payment: the single stub rate cannot represent that period's
        // fully historical accrual
        let periods = accrual_periods(
            d(2026, 8, 6),
            d(2027, 8, 6),
            Frequency::Quarterly,
            &Calendar::WeekendsOnly,
            BusinessDayConvention::Following,
            5,
        )
        .unwrap();
        // first period ends Fri 2026-11-06 and pays five business days on
        let reference = d(2026, 11, 9);
        assert!(periods[0].end < reference && reference < periods[0].payment);
        let curve = flat(0.04, reference);
        let err = float_leg_pv_with_fixing(
            &periods,
            0.0,
            DayCountConvention::Act360,
            &curve,
            &curve,
            Some(0.04),
        );
        assert!(format!("{}", err.unwrap_err()).contains("fully elapsed"));
    }

    #[test]
    fn rejects_bad_inputs() {
        let e = d(2026, 8, 6);
        let cal = Calendar::WeekendsOnly;
        let conv = BusinessDayConvention::Following;
        assert!(accrual_periods(e, e, Frequency::Quarterly, &cal, conv, 0).is_err());
        assert!(accrual_periods(e, d(2027, 8, 6), Frequency::Quarterly, &cal, conv, -1).is_err());
    }
}
