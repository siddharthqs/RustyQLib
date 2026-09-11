//! Interest rate cap and floor: a strip of caplets (floorlets) on the
//! simple rate over each accrual period of a schedule, priced under a
//! one-factor affine short-rate model.
//!
//! The product owns the schedule — effective and maturity dates, the
//! payment frequency, day count, calendar and roll convention, exactly
//! as a swap's floating leg — and values each period's caplet through
//! the [`jamshidian`] engine (a caplet is `1 + K tau` zero-bond puts
//! struck at `1/(1 + K tau)`), summing over the periods. Dates map to
//! year fractions against an **anchor** curve's reference date and day
//! count; for Hull-White that is the model's own fitted curve, see
//! [`npv_hull_white`](CapFloor::npv_hull_white).
//!
//! Each period's rate fixes at its start and pays at its end, with no
//! payment lag. A period whose start is on or before the anchor date
//! has already fixed: it contributes nothing to [`npv`](CapFloor::npv)
//! — the market convention that a spot-starting cap's first caplet is
//! excluded — unless the realized rate is supplied through
//! [`npv_with_fixing`](CapFloor::npv_with_fixing), in which case it
//! contributes its discounted intrinsic. Periods already paid
//! contribute nothing either way.
//!
//! The cap-floor parity `cap - floor = payer swap` on the same
//! schedule holds exactly, and the ATM strike — the forward swap rate
//! of the schedule — is where cap and floor are equal.
//!
//! Caps are quoted in a **flat** vol — one normal or Black vol for every
//! caplet of the strip: [`npv_black`](CapFloor::npv_black) prices from
//! one, [`implied_flat_normal_vol`](CapFloor::implied_flat_normal_vol)
//! and [`implied_flat_black_vol`](CapFloor::implied_flat_black_vol)
//! invert a premium to one, and
//! [`implied_normal_vol_hull_white`](CapFloor::implied_normal_vol_hull_white)
//! reads the model price as the vol it implies.
//!
//! [`jamshidian`]: crate::rates::engines::jamshidian

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Frequency};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::rates::contracts::swaption::year_fraction_from;
use crate::rates::engines::black::{invert_premium, rate_option_kernel, RateVol};
use crate::rates::engines::jamshidian::{caplet, floorlet};
use crate::rates::leg::AccrualPeriod;
use crate::rates::models::{HullWhite, OneFactorAffine};
use crate::rates::schedule::{LegSchedule, RollConvention, StubConvention};
use crate::rates::{checked_df, validate_swap_terms};

const FIELD: &str = "cap/floor";

/// Cap (a strip of caplets: pays `(L - K)^+`) or floor (floorlets:
/// `(K - L)^+`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapOrFloor {
    Cap,
    Floor,
}

impl CapOrFloor {
    /// The period payoff per unit notional and accrual, given the
    /// period's rate `rate` and the strike.
    pub fn intrinsic(self, rate: f64, strike: f64) -> f64 {
        match self {
            CapOrFloor::Cap => (rate - strike).max(0.0),
            CapOrFloor::Floor => (strike - rate).max(0.0),
        }
    }
}

/// An interest rate cap or floor on `notional`, struck at `strike`,
/// over the accrual periods from `effective_date` to `maturity_date`.
#[derive(Debug, Clone)]
pub struct CapFloor {
    pub notional: f64,
    pub strike: f64,
    pub cap_or_floor: CapOrFloor,
    pub effective_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub frequency: Frequency,
    pub day_count: DayCountConvention,
    pub calendar: Calendar,
    pub convention: BusinessDayConvention,
    /// Stub placement.
    pub stub: StubConvention,
    /// Anchor-day roll.
    pub roll: RollConvention,
}

/// One caplet (floorlet) of the strip, as valued.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapletValue {
    pub start: NaiveDate,
    pub end: NaiveDate,
    /// Accrual on the product's day count.
    pub tau: f64,
    /// The period's simple forward rate off the anchor curve (or the
    /// realized rate, for a period that has fixed).
    pub forward_rate: f64,
    /// PV on the product's notional.
    pub value: f64,
}

impl CapFloor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        notional: f64,
        strike: f64,
        cap_or_floor: CapOrFloor,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
        frequency: Frequency,
        day_count: DayCountConvention,
        calendar: Calendar,
        convention: BusinessDayConvention,
    ) -> Result<Self, RustyQLibError> {
        validate_swap_terms(
            FIELD,
            notional,
            strike,
            "strike",
            effective_date,
            maturity_date,
        )?;
        if strike <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("the zero-bond-option equivalence needs a positive strike, got {strike}"),
            ));
        }
        Ok(CapFloor {
            notional,
            strike,
            cap_or_floor,
            effective_date,
            maturity_date,
            frequency,
            day_count,
            calendar,
            convention,
            stub: StubConvention::default(),
            roll: RollConvention::default(),
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

    /// The strip's schedule terms.
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
    }

    /// A USD-style cap/floor: quarterly Act/360, modified following on
    /// the US bond-market calendar — the conventions of the floating
    /// leg it hedges.
    pub fn usd_standard(
        notional: f64,
        strike: f64,
        cap_or_floor: CapOrFloor,
        effective_date: NaiveDate,
        maturity_date: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            notional,
            strike,
            cap_or_floor,
            effective_date,
            maturity_date,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
    }

    /// The accrual periods of the strip (payment at each period's end).
    pub fn periods(&self) -> Result<Vec<AccrualPeriod>, RustyQLibError> {
        self.schedule().accrual_periods()
    }

    /// The ATM strike: the forward swap rate of the schedule over its
    /// unfixed periods, `(df(first start) - df(last end)) / annuity` —
    /// where cap and floor are worth the same.
    pub fn atm_strike(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        let live: Vec<AccrualPeriod> = self
            .periods()?
            .into_iter()
            .filter(|p| p.start > curve.reference_date())
            .collect();
        let (first, last) = match (live.first(), live.last()) {
            (Some(first), Some(last)) => (*first, *last),
            _ => {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    "no unfixed periods left on the schedule",
                ))
            }
        };
        let annuity: f64 = live
            .iter()
            .map(|p| self.day_count.year_fraction(p.start, p.end) * curve.df_date(p.payment))
            .sum();
        if annuity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive annuity {annuity}"
            )));
        }
        Ok((curve.df_date(first.start) - curve.df_date(last.end)) / annuity)
    }

    /// Value each caplet (floorlet) of the strip under `model`, dates
    /// mapped against `anchor`. Periods that have already fixed are
    /// left out (see the module docs); pass the realized rate of the
    /// period in progress through `realized_rate` to include it at its
    /// discounted intrinsic.
    pub fn caplet_values(
        &self,
        model: &impl OneFactorAffine,
        anchor: &YieldCurve,
        realized_rate: Option<f64>,
    ) -> Result<Vec<CapletValue>, RustyQLibError> {
        let valuation = anchor.reference_date();
        let mut values = Vec::new();
        for p in self.periods()? {
            if p.payment <= valuation {
                continue; // settled
            }
            let tau = self.day_count.year_fraction(p.start, p.end);
            if p.start <= valuation {
                // fixed: intrinsic on the realized rate, if we have it
                if let Some(rate) = realized_rate {
                    let value = self.notional
                        * tau
                        * self.cap_or_floor.intrinsic(rate, self.strike)
                        * anchor.df_date(p.payment);
                    values.push(CapletValue {
                        start: p.start,
                        end: p.end,
                        tau,
                        forward_rate: rate,
                        value,
                    });
                }
                continue;
            }
            let start = year_fraction_from(anchor, p.start);
            let end = year_fraction_from(anchor, p.end);
            let forward_rate =
                (checked_df(anchor, p.start)? / checked_df(anchor, p.end)? - 1.0) / tau;
            let value = match self.cap_or_floor {
                CapOrFloor::Cap => caplet(model, start, end, tau, self.strike, self.notional)?,
                CapOrFloor::Floor => floorlet(model, start, end, tau, self.strike, self.notional)?,
            };
            values.push(CapletValue {
                start: p.start,
                end: p.end,
                tau,
                forward_rate,
                value,
            });
        }
        Ok(values)
    }

    /// Value under `model`: the sum of the strip's caplets (floorlets),
    /// with dates mapped to year fractions against `anchor`. For a
    /// curve-fitted model pass the curve it was fitted to (or use
    /// [`npv_hull_white`](Self::npv_hull_white)).
    pub fn npv(
        &self,
        model: &impl OneFactorAffine,
        anchor: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        Ok(self
            .caplet_values(model, anchor, None)?
            .iter()
            .map(|c| c.value)
            .sum())
    }

    /// [`npv`](Self::npv) for a seasoned cap/floor, with the current
    /// period's realized rate (annualized simple, on the product's day
    /// count) included at its discounted intrinsic.
    pub fn npv_with_fixing(
        &self,
        model: &impl OneFactorAffine,
        anchor: &YieldCurve,
        realized_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        Ok(self
            .caplet_values(model, anchor, Some(realized_rate))?
            .iter()
            .map(|c| c.value)
            .sum())
    }

    /// [`npv`](Self::npv) under Hull-White, anchored on the model's own
    /// fitted curve.
    pub fn npv_hull_white(&self, model: &HullWhite) -> Result<f64, RustyQLibError> {
        self.npv(model, model.curve())
    }

    // ── market vol: Black / Bachelier caplets at one flat vol ──────────

    /// The market formula: each unfixed caplet (floorlet) is
    /// `notional * tau * df(pay) * kernel(forward, strike, vol, T_fix)`
    /// with the period forward off `forecast` and discounting off
    /// `discount`, all at the one **flat** `vol`. Periods that have
    /// already fixed are left out, as in [`npv`](Self::npv).
    pub fn npv_black(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        vol: RateVol,
    ) -> Result<f64, RustyQLibError> {
        let valuation = discount.reference_date();
        let side = match self.cap_or_floor {
            CapOrFloor::Cap => PutOrCall::Call,
            CapOrFloor::Floor => PutOrCall::Put,
        };
        let mut value = 0.0;
        for p in self.periods()? {
            if p.start <= valuation {
                continue;
            }
            let tau = self.day_count.year_fraction(p.start, p.end);
            let forward =
                (checked_df(forecast, p.start)? / checked_df(forecast, p.end)? - 1.0) / tau;
            let fixing = year_fraction_from(discount, p.start);
            value += self.notional
                * tau
                * discount.df_date(p.payment)
                * rate_option_kernel(forward, self.strike, fixing, vol, side)?;
        }
        Ok(value)
    }

    /// The flat Bachelier (normal) vol, in absolute rate units per
    /// √year, that reproduces `premium` for the whole strip.
    pub fn implied_flat_normal_vol(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        premium: f64,
    ) -> Result<f64, RustyQLibError> {
        invert_premium(premium, 10.0 * (self.strike + 0.05), |v| {
            self.npv_black(discount, forecast, RateVol::Normal(v))
        })
    }

    /// The flat (shifted) Black-76 vol that reproduces `premium`;
    /// `shift` zero for the plain lognormal quote.
    pub fn implied_flat_black_vol(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        premium: f64,
        shift: f64,
    ) -> Result<f64, RustyQLibError> {
        invert_premium(premium, 10.0, |vol| {
            self.npv_black(discount, forecast, RateVol::ShiftedLognormal { vol, shift })
        })
    }

    /// The flat normal vol the Hull-White model implies for this strip:
    /// its model price read through the market formula on the model's
    /// own curve.
    pub fn implied_normal_vol_hull_white(&self, model: &HullWhite) -> Result<f64, RustyQLibError> {
        let curve = model.curve();
        self.implied_flat_normal_vol(curve, curve, self.npv_hull_white(model)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::rates::contracts::vanilla_swap::VanillaSwap;
    use crate::rates::PayerReceiver;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn asof() -> NaiveDate {
        d(2026, 8, 13)
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
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn model(sigma: f64) -> HullWhite {
        HullWhite::new(0.05, sigma, market_curve()).unwrap()
    }

    /// A 3-year quarterly cap/floor starting 1y forward, on 10mm.
    fn strip(strike: f64, side: CapOrFloor) -> CapFloor {
        CapFloor::usd_standard(10_000_000.0, strike, side, d(2027, 8, 16), d(2030, 8, 16)).unwrap()
    }

    #[test]
    fn cap_minus_floor_is_the_payer_swap_on_the_same_schedule() {
        let m = model(0.011);
        let curve = m.curve();
        let strike = 0.045;
        let cap = strip(strike, CapOrFloor::Cap).npv_hull_white(&m).unwrap();
        let floor = strip(strike, CapOrFloor::Floor).npv_hull_white(&m).unwrap();
        // the same schedule as a swap: quarterly Act/360 on both legs
        let swap = VanillaSwap::new(
            10_000_000.0,
            strike,
            PayerReceiver::Payer,
            d(2027, 8, 16),
            d(2030, 8, 16),
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
        .unwrap();
        let payer_swap = swap.pv(curve).unwrap();
        assert!(
            (cap - floor - payer_swap).abs() < 1e-6 * 10_000_000.0,
            "{cap} - {floor} vs {payer_swap}"
        );
        assert!(cap > 0.0 && floor > 0.0);
    }

    #[test]
    fn strip_is_the_sum_of_its_caplets_and_grows_with_volatility() {
        let calm = model(0.005);
        let wild = model(0.012);
        let cap = strip(0.045, CapOrFloor::Cap);
        let caplets = cap.caplet_values(&calm, calm.curve(), None).unwrap();
        // 12 quarterly periods over 3 years, all unfixed
        assert_eq!(caplets.len(), 12);
        let total: f64 = caplets.iter().map(|c| c.value).sum();
        assert_eq!(total, cap.npv_hull_white(&calm).unwrap());
        assert!(caplets
            .iter()
            .all(|c| c.value > 0.0 && c.tau > 0.2 && c.tau < 0.3));
        assert!(cap.npv_hull_white(&wild).unwrap() > total);
        // a deep out-of-the-money cap is nearly worthless, a deep
        // in-the-money one is nearly the swap
        let deep_otm = strip(0.15, CapOrFloor::Cap).npv_hull_white(&calm).unwrap();
        assert!(deep_otm < 1e-3 * total, "{deep_otm}");
    }

    #[test]
    fn at_the_atm_strike_cap_and_floor_are_equal() {
        let m = model(0.011);
        let curve = m.curve();
        let atm = strip(0.04, CapOrFloor::Cap).atm_strike(curve).unwrap();
        assert!(atm > 0.03 && atm < 0.06, "atm {atm}");
        let cap = strip(atm, CapOrFloor::Cap).npv_hull_white(&m).unwrap();
        let floor = strip(atm, CapOrFloor::Floor).npv_hull_white(&m).unwrap();
        assert!(
            (cap - floor).abs() < 1e-6 * 10_000_000.0,
            "{cap} vs {floor}"
        );
    }

    #[test]
    fn fixed_period_is_excluded_unless_its_rate_is_supplied() {
        // a spot-starting cap: the first quarter fixed today
        let m = model(0.011);
        let curve = m.curve();
        let cap = CapFloor::usd_standard(
            10_000_000.0,
            0.03,
            CapOrFloor::Cap,
            d(2026, 8, 13),
            d(2028, 8, 13),
        )
        .unwrap();
        let periods = cap.periods().unwrap();
        assert_eq!(periods.len(), 8);
        assert!(periods[0].start <= curve.reference_date());
        // 7 live caplets without a fixing
        let live = cap.caplet_values(&m, curve, None).unwrap();
        assert_eq!(live.len(), 7);
        let without = cap.npv_hull_white(&m).unwrap();
        // in the money at 5%: the fixed period adds its intrinsic
        let with = cap.npv_with_fixing(&m, curve, 0.05).unwrap();
        let first = periods[0];
        let tau = DayCountConvention::Act360.year_fraction(first.start, first.end);
        let intrinsic = 10_000_000.0 * tau * (0.05 - 0.03) * curve.df_date(first.payment);
        assert!(
            (with - without - intrinsic).abs() < 1e-8,
            "{with} vs {without}"
        );
        // out of the money: nothing added
        assert_eq!(cap.npv_with_fixing(&m, curve, 0.02).unwrap(), without);
    }

    #[test]
    fn flat_vol_prices_keep_parity_and_implied_vols_round_trip() {
        let m = model(0.011);
        let curve = m.curve();
        let strike = 0.045;
        let cap = strip(strike, CapOrFloor::Cap);
        let floor = strip(strike, CapOrFloor::Floor);
        // cap - floor under a flat vol is the same payer swap
        let vol = RateVol::Normal(0.0080);
        let c = cap.npv_black(curve, curve, vol).unwrap();
        let f = floor.npv_black(curve, curve, vol).unwrap();
        let swap = VanillaSwap::new(
            10_000_000.0,
            strike,
            PayerReceiver::Payer,
            d(2027, 8, 16),
            d(2030, 8, 16),
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Frequency::Quarterly,
            DayCountConvention::Act360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::ModifiedFollowing,
        )
        .unwrap();
        assert!((c - f - swap.pv(curve).unwrap()).abs() < 1e-6, "{c} - {f}");
        // round trips
        let v = cap.implied_flat_normal_vol(curve, curve, c).unwrap();
        assert!((v - 0.0080).abs() < 1e-10, "normal {v}");
        let black = cap
            .npv_black(curve, curve, RateVol::Lognormal(0.25))
            .unwrap();
        let v = cap
            .implied_flat_black_vol(curve, curve, black, 0.0)
            .unwrap();
        assert!((v - 0.25).abs() < 1e-10, "black {v}");
        // the Hull-White strip reads as a flat normal vol near sigma
        let hw_vol = cap.implied_normal_vol_hull_white(&m).unwrap();
        assert!(
            hw_vol > 0.009 && hw_vol < 0.0115,
            "HW-implied flat vol {hw_vol}"
        );
        assert!(cap.implied_normal_vol_hull_white(&model(0.013)).unwrap() > hw_vol);
        // a premium below intrinsic has no vol
        let floor_intrinsic = floor.npv_black(curve, curve, RateVol::Normal(0.0)).unwrap();
        assert!(floor
            .implied_flat_normal_vol(curve, curve, 0.5 * floor_intrinsic)
            .is_err());
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(CapFloor::usd_standard(
            10_000_000.0,
            0.0,
            CapOrFloor::Cap,
            d(2027, 8, 16),
            d(2030, 8, 16)
        )
        .is_err());
        assert!(CapFloor::usd_standard(
            -1.0,
            0.04,
            CapOrFloor::Cap,
            d(2027, 8, 16),
            d(2030, 8, 16)
        )
        .is_err());
        assert!(CapFloor::usd_standard(
            1.0,
            0.04,
            CapOrFloor::Floor,
            d(2030, 8, 16),
            d(2027, 8, 16)
        )
        .is_err());
        // a strip entirely in the past has no ATM strike
        let expired =
            CapFloor::usd_standard(1.0, 0.04, CapOrFloor::Cap, d(2020, 8, 16), d(2022, 8, 16))
                .unwrap();
        assert!(expired.atm_strike(&market_curve()).is_err());
        assert_eq!(expired.npv_hull_white(&model(0.01)).unwrap(), 0.0);
    }
}
