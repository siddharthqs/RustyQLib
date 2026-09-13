//! Bermudan swaption: the right to enter a [`VanillaSwap`] on any one
//! of several exercise dates, priced by backward induction on the
//! Hull-White grid ([`hw_grid`]).
//!
//! On exercise at date `E` the holder enters the swap's **remaining**
//! periods — those whose accrual starts on or after `E` (the usual
//! "exercise into the tail" of a `10nc1`-style trade). Under the model
//! the exercise value at a grid node is closed form: the floating leg
//! is worth `N (P(E, S) - P(E, T_n))` with `S` the first remaining
//! start and `T_n` the last payment, the fixed leg is the remaining
//! coupons plus nothing, each discounted with the node's own zero-bond
//! prices, and a payer receives `float - fixed`. The induction rolls
//! `max(continuation, exercise)` back between exercise dates.
//!
//! [`european_values_hull_white`](BermudanSwaption::european_values_hull_white)
//! prices each exercise date as a European swaption on the same tail,
//! by Jamshidian — the Bermudan is bounded below by the largest of them
//! and, with a single exercise date, equals it, which is the engine's
//! accuracy check.
//!
//! [`hw_grid`]: crate::rates::engines::hw_grid

use chrono::NaiveDate;

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::contracts::swaption::{year_fraction_from, Swaption};
use crate::rates::contracts::vanilla_swap::VanillaSwap;
use crate::rates::engines::hw_grid::{self, GridConfig};
use crate::rates::models::{HullWhite, ShortRateModel};
use crate::rates::PayerReceiver;

const FIELD: &str = "bermudan swaption";

/// A Bermudan option to enter `swap` (its side and fixed rate) on any
/// one of `exercise_dates`.
#[derive(Debug, Clone)]
pub struct BermudanSwaption {
    pub swap: VanillaSwap,
    /// Ascending exercise dates, each before the swap's maturity and on
    /// or before some fixed period's start.
    pub exercise_dates: Vec<NaiveDate>,
}

/// One exercise opportunity, in year fractions from the anchor.
struct Exercise {
    time: f64,
    /// First remaining fixed period's start.
    start: f64,
    /// Remaining fixed coupons `(payment time, amount)`.
    coupons: Vec<(f64, f64)>,
    /// The last payment time (notional exchange).
    last: f64,
}

impl BermudanSwaption {
    pub fn new(swap: VanillaSwap, exercise_dates: Vec<NaiveDate>) -> Result<Self, RustyQLibError> {
        if exercise_dates.is_empty() {
            return Err(RustyQLibError::invalid_input(FIELD, "no exercise dates"));
        }
        for pair in exercise_dates.windows(2) {
            if pair[1] <= pair[0] {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!(
                        "exercise dates must be ascending, got {} after {}",
                        pair[1], pair[0]
                    ),
                ));
            }
        }
        let last_start = swap
            .fixed_periods()?
            .last()
            .map(|p| p.start)
            .expect("a swap has at least one period");
        if let Some(&late) = exercise_dates.iter().find(|&&d| d > last_start) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "exercise {late} is after the last fixed period start {last_start}: \
                     nothing would remain to enter"
                ),
            ));
        }
        Ok(BermudanSwaption {
            swap,
            exercise_dates,
        })
    }

    /// Exercise on every fixed period start from `first_exercise` on
    /// (excluding the final period's start only if it is the maturity)
    /// — the standard Bermudan schedule.
    pub fn on_fixed_period_starts(
        swap: VanillaSwap,
        first_exercise: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        let dates: Vec<NaiveDate> = swap
            .fixed_periods()?
            .iter()
            .map(|p| p.start)
            .filter(|&start| start >= first_exercise)
            .collect();
        Self::new(swap, dates)
    }

    /// The exercise opportunities against `anchor`.
    fn schedule(&self, anchor: &YieldCurve) -> Result<Vec<Exercise>, RustyQLibError> {
        let periods = self.swap.fixed_periods()?;
        let n = self.swap.notional;
        let mut out = Vec::with_capacity(self.exercise_dates.len());
        for &date in &self.exercise_dates {
            let time = year_fraction_from(anchor, date);
            if time <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!(
                        "exercise {date} is not after the anchor date {}",
                        anchor.reference_date()
                    ),
                ));
            }
            let remaining: Vec<_> = periods.iter().filter(|p| p.start >= date).collect();
            let first = remaining.first().ok_or_else(|| {
                RustyQLibError::invalid_input(
                    FIELD,
                    format!("no fixed period starts on or after the exercise {date}"),
                )
            })?;
            let coupons: Vec<(f64, f64)> = remaining
                .iter()
                .map(|p| {
                    (
                        year_fraction_from(anchor, p.payment),
                        n * self.swap.fixed_rate
                            * self.swap.fixed_day_count.year_fraction(p.start, p.end),
                    )
                })
                .collect();
            out.push(Exercise {
                time,
                start: year_fraction_from(anchor, first.start),
                last: coupons.last().map(|c| c.0).expect("remaining is non-empty"),
                coupons,
            });
        }
        Ok(out)
    }

    /// Value under Hull-White by backward induction on the grid,
    /// anchored on the model's fitted curve.
    pub fn npv_hull_white(
        &self,
        model: &HullWhite,
        config: &GridConfig,
    ) -> Result<f64, RustyQLibError> {
        let schedule = self.schedule(model.curve())?;
        let times: Vec<f64> = schedule.iter().map(|e| e.time).collect();
        let n = self.swap.notional;
        let sign = match self.swap.payer_receiver {
            PayerReceiver::Payer => 1.0,
            PayerReceiver::Receiver => -1.0,
        };
        hw_grid::backward_induction(
            model,
            &times,
            0.0,
            |k, rate, continuation| {
                let e = &schedule[k];
                let float = n
                    * (model.zero_bond(e.time, e.start, rate)?
                        - model.zero_bond(e.time, e.last, rate)?);
                let mut fixed = 0.0;
                for &(pay, amount) in &e.coupons {
                    fixed += amount * model.zero_bond(e.time, pay, rate)?;
                }
                Ok(continuation.max(sign * (float - fixed)))
            },
            config,
        )
    }

    /// Value under any [`Gaussian1dModel`] by deflated backward induction
    /// on its state grid, dates mapped against `anchor`.
    ///
    /// [`Gaussian1dModel`]: crate::rates::models::gaussian1d::Gaussian1dModel
    pub fn npv_gaussian1d(
        &self,
        model: &impl crate::rates::models::gaussian1d::Gaussian1dModel,
        anchor: &YieldCurve,
        config: &GridConfig,
    ) -> Result<f64, RustyQLibError> {
        use crate::rates::models::black_karasinski::TailSwap;
        let tails: Vec<TailSwap> = self
            .schedule(anchor)?
            .into_iter()
            .map(|e| TailSwap {
                expiry: e.time,
                start: e.start,
                coupons: e.coupons,
                last: e.last,
            })
            .collect();
        crate::rates::engines::gaussian1d::bermudan_swaption(
            model,
            &tails,
            self.swap.notional,
            self.swap.payer_receiver,
            config,
        )
    }

    /// Value under Black-Karasinski by backward induction on its fitted
    /// tree, anchored on the model's curve.
    pub fn npv_black_karasinski(
        &self,
        model: &crate::rates::models::black_karasinski::BlackKarasinski,
    ) -> Result<f64, RustyQLibError> {
        use crate::rates::models::black_karasinski::TailSwap;
        let tails: Vec<TailSwap> = self
            .schedule(model.curve())?
            .into_iter()
            .map(|e| TailSwap {
                expiry: e.time,
                start: e.start,
                coupons: e.coupons,
                last: e.last,
            })
            .collect();
        model.bermudan_swaption(&tails, self.swap.notional, self.swap.payer_receiver)
    }

    /// The tails as the engines see them, against `anchor`.
    fn tails(
        &self,
        anchor: &YieldCurve,
    ) -> Result<Vec<crate::rates::models::black_karasinski::TailSwap>, RustyQLibError> {
        Ok(self
            .schedule(anchor)?
            .into_iter()
            .map(|e| crate::rates::models::black_karasinski::TailSwap {
                expiry: e.time,
                start: e.start,
                coupons: e.coupons,
                last: e.last,
            })
            .collect())
    }

    /// Value under Hull-White by finite differences on the state PDE,
    /// anchored on the model's curve.
    pub fn npv_fd_hull_white(
        &self,
        model: &HullWhite,
        config: &crate::rates::engines::fd_hull_white::FdConfig,
    ) -> Result<f64, RustyQLibError> {
        crate::rates::engines::fd_hull_white::bermudan_swaption(
            model,
            &self.tails(model.curve())?,
            self.swap.notional,
            self.swap.payer_receiver,
            config,
        )
    }

    /// Value under G2++ by ADI finite differences on the two-factor
    /// PDE, anchored on the model's curve.
    pub fn npv_fd_g2pp(
        &self,
        model: &crate::rates::models::g2pp::G2pp,
        config: &crate::rates::engines::fd_g2pp::FdG2Config,
    ) -> Result<f64, RustyQLibError> {
        crate::rates::engines::fd_g2pp::bermudan_swaption(
            model,
            &self.tails(model.curve())?,
            self.swap.notional,
            self.swap.payer_receiver,
            config,
        )
    }

    /// Each exercise date as a European swaption on the same remaining
    /// swap, priced analytically (Jamshidian) under the model. The
    /// Bermudan is worth at least the largest of these.
    pub fn european_values_hull_white(
        &self,
        model: &HullWhite,
    ) -> Result<Vec<f64>, RustyQLibError> {
        let periods = self.swap.fixed_periods()?;
        self.exercise_dates
            .iter()
            .map(|&date| {
                let start = periods
                    .iter()
                    .find(|p| p.start >= date)
                    .map(|p| p.start)
                    .expect("validated at construction");
                let tail = VanillaSwap {
                    effective_date: start,
                    ..self.swap.clone()
                };
                Swaption::new(tail, date)?.npv_hull_white(model)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn model(sigma: f64) -> HullWhite {
        HullWhite::new(0.05, sigma, market_curve()).unwrap()
    }

    /// A 6y swap on 10mm, callable from year one: "6nc1".
    fn swap(rate: f64, side: PayerReceiver) -> VanillaSwap {
        VanillaSwap::usd_standard(10_000_000.0, rate, side, d(2026, 8, 17), d(2032, 8, 17)).unwrap()
    }

    #[test]
    fn a_single_exercise_date_is_the_european_swaption() {
        let m = model(0.011);
        let s = swap(0.045, PayerReceiver::Payer);
        let start = s.fixed_periods().unwrap()[2].start; // one year in
        let b = BermudanSwaption::new(s, vec![start]).unwrap();
        let on_grid = b.npv_hull_white(&m, &GridConfig::default()).unwrap();
        let european = b.european_values_hull_white(&m).unwrap()[0];
        assert!(
            (on_grid - european).abs() < 2e-3 * european,
            "grid {on_grid} vs Jamshidian {european}"
        );
    }

    #[test]
    fn bermudan_dominates_every_european_and_grows_with_volatility() {
        let m = model(0.011);
        for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
            let b =
                BermudanSwaption::on_fixed_period_starts(swap(0.045, side), d(2027, 8, 1)).unwrap();
            assert_eq!(b.exercise_dates.len(), 10);
            let bermudan = b.npv_hull_white(&m, &GridConfig::default()).unwrap();
            let europeans = b.european_values_hull_white(&m).unwrap();
            let best = europeans.iter().cloned().fold(f64::MIN, f64::max);
            assert!(
                bermudan > best,
                "{side:?}: bermudan {bermudan} vs best european {best}"
            );
            // and by a margin, but well under the sum of the Europeans
            assert!(bermudan < europeans.iter().sum::<f64>());
            assert!(
                bermudan - best > 0.02 * best,
                "{side:?}: switch value too small"
            );
            let wild = b
                .npv_hull_white(&model(0.014), &GridConfig::default())
                .unwrap();
            assert!(wild > bermudan);
        }
    }

    #[test]
    fn zero_volatility_is_the_best_intrinsic_exercise() {
        // no uncertainty: exercise on the date whose forward swap is
        // worth most today, computed off the curve alone
        let m = model(1e-6);
        let s = swap(0.040, PayerReceiver::Payer);
        let b = BermudanSwaption::on_fixed_period_starts(s.clone(), d(2027, 8, 1)).unwrap();
        let curve = m.curve();
        let best_intrinsic = b
            .exercise_dates
            .iter()
            .map(|&date| {
                let start = s
                    .fixed_periods()
                    .unwrap()
                    .iter()
                    .find(|p| p.start >= date)
                    .unwrap()
                    .start;
                let tail = VanillaSwap {
                    effective_date: start,
                    ..s.clone()
                };
                tail.pv(curve).unwrap().max(0.0)
            })
            .fold(0.0_f64, f64::max);
        let bermudan = b.npv_hull_white(&m, &GridConfig::default()).unwrap();
        assert!(
            (bermudan - best_intrinsic).abs() < 1e-3 * best_intrinsic.max(1.0),
            "{bermudan} vs {best_intrinsic}"
        );
        assert!(best_intrinsic > 0.0);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let s = swap(0.045, PayerReceiver::Payer);
        assert!(BermudanSwaption::new(s.clone(), vec![]).is_err());
        assert!(BermudanSwaption::new(s.clone(), vec![d(2028, 8, 17), d(2027, 8, 17)]).is_err());
        assert!(BermudanSwaption::new(s.clone(), vec![d(2032, 8, 10)]).is_err());
        // an exercise date at or before the anchor
        let b = BermudanSwaption::new(s, vec![d(2026, 8, 13)]).unwrap();
        assert!(b
            .npv_hull_white(&model(0.01), &GridConfig::default())
            .is_err());
    }
}
