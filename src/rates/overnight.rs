//! Shared overnight-rate machinery for OIS legs and money-market futures.
//!
//! Both fed funds and SOFR futures settle on an average of a daily
//! overnight rate, and an OIS leg on its daily compounding: published
//! fixings for days already realized, curve forwards for the rest. The
//! pieces they share live here, including the compounded-in-arrears
//! accrual with the ISDA/ARRC conventions — [`OvernightConvention`]'s
//! **lookback** (each day's rate is the fixing `k` business days
//! earlier), **lockout** (the last `k` business days of a period reuse
//! the rate of the day before the lockout) and **observation shift**
//! (with a lookback, each rate is weighted by its own observation
//! day's calendar days rather than the accrual day's).

use std::collections::BTreeMap;

use chrono::{Duration, NaiveDate};

use crate::core::calendar::Calendar;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// Conventions of a compounded-in-arrears overnight leg. The default —
/// no lookback, no lockout, no shift — is plain compounding, where a
/// fully forecast period telescopes to the curve's `df(s)/df(e) - 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OvernightConvention {
    /// Business days between an accrual day and the day whose fixing
    /// it uses (SOFR loans commonly use 5; swaps 0).
    pub lookback_days: i64,
    /// Business days at the end of each period during which the rate
    /// is frozen at the last observed one.
    pub lockout_days: i64,
    /// With a lookback, weight each rate by the observation day's
    /// calendar days instead of the accrual day's.
    pub observation_shift: bool,
}

impl OvernightConvention {
    pub fn validate(self) -> Result<(), RustyQLibError> {
        if self.lookback_days < 0 || self.lockout_days < 0 {
            return Err(RustyQLibError::invalid_input(
                "overnight convention",
                format!(
                    "lookback and lockout must be non-negative, got {} and {}",
                    self.lookback_days, self.lockout_days
                ),
            ));
        }
        if self.observation_shift && self.lookback_days == 0 {
            return Err(RustyQLibError::invalid_input(
                "overnight convention",
                "an observation shift needs a lookback",
            ));
        }
        Ok(())
    }

    /// Whether every day's rate is its own day's, weighted by its own
    /// days — the case that telescopes to the discount-factor ratio.
    pub fn is_plain(self) -> bool {
        self == Self::default()
    }
}

/// The compounded overnight accrual `prod(1 + r_d n_d / 360) - 1` over
/// `[start, end)` on `calendar` business days, Act/360. Day `d`'s rate
/// is the fixing (or, from the curve's reference date on, the curve's
/// overnight forward) of its observation day under `convention`; days
/// before the reference need `fixings` and error without them.
pub fn compounded_overnight_accrual(
    curve: &YieldCurve,
    fixings: Option<&RateFixings>,
    start: NaiveDate,
    end: NaiveDate,
    calendar: &Calendar,
    convention: OvernightConvention,
) -> Result<f64, RustyQLibError> {
    convention.validate()?;
    if end <= start {
        return Err(RustyQLibError::invalid_input(
            "overnight accrual",
            format!("end {end} must be after start {start}"),
        ));
    }
    // accrual segments: each business day (the start counts as one)
    // runs to the next business day, the last truncated at the end
    let mut segments: Vec<(NaiveDate, i64)> = Vec::new();
    let mut day = start;
    while day < end {
        let next = calendar.add_business_days(day, 1).min(end);
        segments.push((day, (next - day).num_days()));
        day = next;
    }
    let n = segments.len();
    let lockout = (convention.lockout_days as usize).min(n.saturating_sub(1));
    let reference = curve.reference_date();
    let mut factor = 1.0;
    for (i, &(accrual_day, accrual_days)) in segments.iter().enumerate() {
        // lockout: the last `lockout` days reuse the last free day
        let source_day = if i >= n - lockout {
            segments[n - lockout - 1].0
        } else {
            accrual_day
        };
        let observation = calendar.add_business_days(source_day, -convention.lookback_days);
        let observation_days =
            (calendar.add_business_days(observation, 1) - observation).num_days();
        let rate = if observation < reference {
            let Some(fixings) = fixings else {
                return Err(RustyQLibError::invalid_input(
                    "overnight accrual",
                    format!(
                        "the rate for {accrual_day} observes {observation}, before the curve \
                         reference {reference}: supply the fixings"
                    ),
                ));
            };
            fixing_on_or_before(fixings, observation)?
        } else {
            simple_forward(curve, observation, observation_days)?
        };
        let days = if convention.observation_shift {
            observation_days
        } else {
            accrual_days
        };
        factor *= 1.0 + rate * days as f64 / 360.0;
    }
    Ok(factor - 1.0)
}

/// Published daily fixings, keyed by the date the rate applies to.
/// Gaps (weekends, holidays) are filled by carrying the previous
/// business day's rate forward.
pub type RateFixings = BTreeMap<NaiveDate, f64>;

/// The fixing applying to `day`: its own, or the most recent earlier one
/// (weekend and holiday carry-forward).
pub fn fixing_on_or_before(fixings: &RateFixings, day: NaiveDate) -> Result<f64, RustyQLibError> {
    fixings
        .range(..=day)
        .next_back()
        .map(|(_, &rate)| rate)
        .ok_or_else(|| {
            RustyQLibError::invalid_input(
                "fixings",
                format!("no published fixing on or before {day}"),
            )
        })
}

/// Simple overnight forward rate from `day` to the next calendar day,
/// quoted Act/360 off `curve`.
pub fn overnight_forward(curve: &YieldCurve, day: NaiveDate) -> Result<f64, RustyQLibError> {
    simple_forward(curve, day, 1)
}

/// Simple forward rate over `days` calendar days from `day`, quoted
/// Act/360 — the rate `r` with `1 + r * days/360 = df(day)/df(day+days)`.
pub fn simple_forward(
    curve: &YieldCurve,
    day: NaiveDate,
    days: i64,
) -> Result<f64, RustyQLibError> {
    if days <= 0 {
        return Err(RustyQLibError::invalid_input(
            "simple_forward",
            format!("the period must be at least one day, got {days}"),
        ));
    }
    // the curve clamps df to 1 before its reference, which would return
    // exactly 0% for elapsed days instead of their realized fixings
    if day < curve.reference_date() {
        return Err(RustyQLibError::invalid_input(
            "simple_forward",
            format!(
                "{day} is before the curve reference {}: past overnight rates are \
                 fixings, not forwards",
                curve.reference_date()
            ),
        ));
    }
    let df0 = curve.df_date(day);
    let df1 = curve.df_date(day + Duration::days(days));
    if !df0.is_finite() || df0 <= 0.0 || !df1.is_finite() || df1 <= 0.0 {
        return Err(RustyQLibError::NumericalError(format!(
            "non-positive discount factor around {day}"
        )));
    }
    Ok((df0 / df1 - 1.0) * 360.0 / days as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat_curve(rate: f64, reference: NaiveDate) -> YieldCurve {
        YieldCurve::flat(
            rate,
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn plain_compounding_telescopes_to_the_discount_ratio() {
        let reference = d(2026, 8, 6);
        let curve = flat_curve(0.045, reference);
        let (start, end) = (d(2026, 9, 1), d(2027, 3, 1));
        let accrual = compounded_overnight_accrual(
            &curve,
            None,
            start,
            end,
            &Calendar::UsGovernmentBond,
            OvernightConvention::default(),
        )
        .unwrap();
        let ratio = curve.df_date(start) / curve.df_date(end) - 1.0;
        assert!((accrual - ratio).abs() < 1e-13, "{accrual} vs {ratio}");
    }

    #[test]
    fn lookback_lockout_and_shift_perturb_the_plain_accrual_slightly() {
        // an upward-sloping curve so shifting observation days matters
        let reference = d(2026, 8, 6);
        let curve = YieldCurve::from_zero_rates(
            &[
                crate::core::curves::Tenor::YearFraction(0.5),
                crate::core::curves::Tenor::YearFraction(2.0),
            ],
            &[0.04, 0.05],
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
            crate::core::curves::InterpolationMethod::LinearZero,
        )
        .unwrap();
        let (start, end) = (d(2026, 9, 1), d(2027, 9, 1));
        let cal = Calendar::UsGovernmentBond;
        let plain = compounded_overnight_accrual(
            &curve,
            None,
            start,
            end,
            &cal,
            OvernightConvention::default(),
        )
        .unwrap();
        let lookback = OvernightConvention {
            lookback_days: 5,
            ..Default::default()
        };
        let lb = compounded_overnight_accrual(&curve, None, start, end, &cal, lookback).unwrap();
        // observing 5 days earlier on a rising curve lowers the accrual, a little
        assert!(
            lb < plain && (plain - lb) < 1e-4 * plain.max(1e-9) * 100.0,
            "{lb} vs {plain}"
        );
        let shifted = OvernightConvention {
            lookback_days: 5,
            observation_shift: true,
            ..Default::default()
        };
        let sh = compounded_overnight_accrual(&curve, None, start, end, &cal, shifted).unwrap();
        assert!(sh != lb && (sh - lb).abs() < 1e-3 * plain, "{sh} vs {lb}");
        let lockout = OvernightConvention {
            lockout_days: 2,
            ..Default::default()
        };
        let lo = compounded_overnight_accrual(&curve, None, start, end, &cal, lockout).unwrap();
        assert!(
            lo != plain && (lo - plain).abs() < 1e-4 * plain,
            "{lo} vs {plain}"
        );
        // a shift without a lookback is meaningless
        assert!(OvernightConvention {
            observation_shift: true,
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn realized_days_come_from_fixings_and_are_required() {
        // a period one month in progress: September fixed at 5%, the
        // rest forecast at the curve's 4.5%
        let start = d(2026, 9, 1);
        let reference = d(2026, 10, 1);
        let end = d(2026, 12, 1);
        let curve = flat_curve(0.045, reference);
        let cal = Calendar::UsGovernmentBond;
        assert!(compounded_overnight_accrual(
            &curve,
            None,
            start,
            end,
            &cal,
            OvernightConvention::default()
        )
        .is_err());
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 8, 31), 0.05);
        let accrual = compounded_overnight_accrual(
            &curve,
            Some(&fixings),
            start,
            end,
            &cal,
            OvernightConvention::default(),
        )
        .unwrap();
        // realized part: 30 days of 5% compounded on business days;
        // forecast part: the df ratio from the reference to the end
        let mut realized = 1.0;
        let mut day = start;
        while day < reference {
            let next = cal.add_business_days(day, 1).min(reference);
            realized *= 1.0 + 0.05 * (next - day).num_days() as f64 / 360.0;
            day = next;
        }
        let forecast = curve.df_date(reference) / curve.df_date(end);
        assert!(
            (accrual - (realized * forecast - 1.0)).abs() < 1e-13,
            "{accrual}"
        );
    }

    #[test]
    fn carry_forward_picks_the_latest_earlier_fixing() {
        let mut fixings = RateFixings::new();
        fixings.insert(d(2026, 9, 4), 0.05);
        fixings.insert(d(2026, 9, 8), 0.04);
        // exact hit, and the weekend days after the Friday fixing
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 4)).unwrap(), 0.05);
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 6)).unwrap(), 0.05);
        assert_eq!(fixing_on_or_before(&fixings, d(2026, 9, 8)).unwrap(), 0.04);
        // before the first fixing there is nothing to carry
        assert!(fixing_on_or_before(&fixings, d(2026, 9, 1)).is_err());
    }

    #[test]
    fn simple_forward_inverts_the_discount_ratio() {
        let curve = YieldCurve::flat(
            0.04,
            d(2026, 8, 6),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        for days in [1, 3, 7, 91] {
            let day = d(2026, 9, 16);
            let r = simple_forward(&curve, day, days).unwrap();
            let ratio = curve.df_date(day) / curve.df_date(day + Duration::days(days));
            assert!(
                (1.0 + r * days as f64 / 360.0 - ratio).abs() < 1e-14,
                "{days}d"
            );
        }
        // the one-day case is the overnight forward
        let day = d(2026, 9, 16);
        assert_eq!(
            overnight_forward(&curve, day).unwrap(),
            simple_forward(&curve, day, 1).unwrap()
        );
        assert!(simple_forward(&curve, day, 0).is_err());
    }

    #[test]
    fn past_days_are_fixings_not_forwards() {
        let curve = YieldCurve::flat(
            0.04,
            d(2026, 8, 6),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        // the old df clamp returned exactly 0% here
        assert!(simple_forward(&curve, d(2026, 8, 5), 1).is_err());
        assert!(simple_forward(&curve, d(2026, 8, 6), 1).is_ok());
    }
}
