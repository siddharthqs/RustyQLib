//! Average price option (APO) — the Asian-style commodity option.
//!
//! The workhorse OTC commodity option (and CME's settled-price options,
//! e.g. AO on WTI): at settlement it pays the difference, when
//! favourable, between the **arithmetic average of the daily index
//! price over the averaging period's business days** and the strike,
//! times the notional quantity. The averaging is exactly the floating
//! leg of one [`CommoditySwap`](crate::cmdty::CommoditySwap) period, so
//! pricing days, fixings carry-forward and the settlement lag all
//! follow the swap's conventions.
//!
//! # Model
//!
//! Priced by discrete moment matching (Levy 1992): under one-factor
//! Black-76 dynamics with flat volatility `sigma`, each unrealized
//! observation `F_i` is lognormal with variance `sigma^2 t_i` at its
//! observation date, and observations are perfectly correlated through
//! the common driving factor, so
//!
//! - `E[A] = (1/n) sum F_i`
//! - `E[A^2] = (1/n^2) sum_ij F_i F_j exp(sigma^2 min(t_i, t_j))`
//!
//! and the average is approximated as lognormal with those two moments,
//! priced in a Black formula of total variance `ln E[A^2] - 2 ln E[A]`.
//! Realized fixings shift into an adjusted strike on the remaining
//! average, so a partially fixed (or fully fixed) option prices
//! consistently down to its deterministic settlement value.

use chrono::NaiveDate;

use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::cmdty::swap::PriceFixings;
use crate::core::calendar::Calendar;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::core::utils::norm_cdf;
use crate::rates::overnight::fixing_on_or_before;

/// An average price option on one averaging period. Premium is quoted
/// for the whole contract (`quantity` units).
#[derive(Debug, Clone)]
pub struct AveragePriceOption {
    /// Contract size in the index's units.
    pub quantity: f64,
    /// Strike price per unit.
    pub strike: f64,
    pub put_or_call: PutOrCall,
    /// First day of the averaging period (inclusive).
    pub averaging_start: NaiveDate,
    /// End of the averaging period (exclusive, like a swap period): the
    /// observations are the calendar's business days in
    /// `[averaging_start, averaging_end)`.
    pub averaging_end: NaiveDate,
    /// The pricing calendar of the index.
    pub calendar: Calendar,
    /// Business days between `averaging_end` and cash settlement.
    pub payment_lag: i64,
}

impl AveragePriceOption {
    pub fn new(
        quantity: f64,
        strike: f64,
        put_or_call: PutOrCall,
        averaging_start: NaiveDate,
        averaging_end: NaiveDate,
        calendar: Calendar,
        payment_lag: i64,
    ) -> Result<Self, RustyQLibError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("quantity must be positive, got {quantity}"),
            ));
        }
        if !strike.is_finite() || strike <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("strike must be positive, got {strike}"),
            ));
        }
        if averaging_end <= averaging_start {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("averaging end {averaging_end} must be after start {averaging_start}"),
            ));
        }
        if payment_lag < 0 {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("payment lag must be non-negative, got {payment_lag}"),
            ));
        }
        Ok(AveragePriceOption {
            quantity,
            strike,
            put_or_call,
            averaging_start,
            averaging_end,
            calendar,
            payment_lag,
        })
    }

    /// The standard monthly contract: averaging over `year`/`month`'s
    /// business days, settling 5 business days after the month ends.
    pub fn for_month(
        quantity: f64,
        strike: f64,
        put_or_call: PutOrCall,
        year: i32,
        month: u32,
        calendar: Calendar,
    ) -> Result<Self, RustyQLibError> {
        let start = NaiveDate::from_ymd_opt(year, month, 1).ok_or_else(|| {
            RustyQLibError::invalid_input(
                "average price option",
                format!("invalid contract month {year}-{month}"),
            )
        })?;
        let end = if month == 12 {
            NaiveDate::from_ymd_opt(year + 1, 1, 1)
        } else {
            NaiveDate::from_ymd_opt(year, month + 1, 1)
        }
        .expect("valid month start");
        Self::new(quantity, strike, put_or_call, start, end, calendar, 5)
    }

    /// The averaging observations: every business day in
    /// `[averaging_start, averaging_end)`.
    pub fn pricing_days(&self) -> Vec<NaiveDate> {
        let mut days = Vec::new();
        let mut day = self.averaging_start;
        while day < self.averaging_end {
            if self.calendar.is_business_day(day) {
                days.push(day);
            }
            day = day.succ_opt().expect("date in range");
        }
        days
    }

    /// Cash settlement date: `payment_lag` business days after the
    /// averaging period ends.
    pub fn settlement_date(&self) -> NaiveDate {
        self.calendar
            .add_business_days(self.averaging_end, self.payment_lag)
    }

    /// Premium for an option whose averaging has not started (every
    /// observation still floats). Once the discount curve's reference
    /// date is inside the averaging period, use
    /// [`price_with_fixings`](Self::price_with_fixings).
    pub fn price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<f64, RustyQLibError> {
        self.price_with_fixings(discount, forward, vol, &PriceFixings::new())
    }

    /// Premium with realized fixings: pricing days strictly before the
    /// discount curve's reference date (the valuation date) come from
    /// `fixings` (carried forward over gaps), the rest float on the
    /// forward curve.
    pub fn price_with_fixings(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
        fixings: &PriceFixings,
    ) -> Result<f64, RustyQLibError> {
        if !vol.is_finite() || vol < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("volatility must be non-negative, got {vol}"),
            ));
        }
        let valuation = discount.reference_date();
        let settlement = self.settlement_date();
        if settlement < valuation {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!("settled {settlement} (valuing {valuation})"),
            ));
        }
        let days = self.pricing_days();
        if days.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "average price option",
                format!(
                    "no pricing days between {} and {}",
                    self.averaging_start, self.averaging_end
                ),
            ));
        }
        let n = days.len() as f64;
        let df = discount.df_date(settlement);

        // realized part, and the still-floating observations
        let mut fixed_sum = 0.0;
        let mut unfixed: Vec<(f64, f64)> = Vec::with_capacity(days.len()); // (t_i, F_i)
        for &day in &days {
            if day < valuation {
                fixed_sum += fixing_on_or_before(fixings, day)?;
            } else {
                let f = forward.price(day);
                if f <= 0.0 {
                    return Err(RustyQLibError::invalid_input(
                        "average price option",
                        format!("moment matching needs a positive forward, got {f} at {day}"),
                    ));
                }
                let t = discount.day_count().year_fraction(valuation, day);
                unfixed.push((t.max(0.0), f));
            }
        }

        let intrinsic = |avg_minus_k: f64| match self.put_or_call {
            PutOrCall::Call => avg_minus_k.max(0.0),
            PutOrCall::Put => (-avg_minus_k).max(0.0),
        };

        // fully realized: the settlement amount is deterministic
        if unfixed.is_empty() {
            return Ok(self.quantity * df * intrinsic(fixed_sum / n - self.strike));
        }

        let nu = unfixed.len() as f64;
        let m1 = unfixed.iter().map(|&(_, f)| f).sum::<f64>() / nu;
        // remaining strike on the floating average; the realized part is a
        // known shift
        let k_eff = (n * self.strike - fixed_sum) / nu;
        let weight = self.quantity * (nu / n) * df;

        // the whole distribution sits above the adjusted strike: the call
        // is a forward purchase, the put is worthless
        if k_eff <= 0.0 {
            return Ok(match self.put_or_call {
                PutOrCall::Call => weight * (m1 - k_eff),
                PutOrCall::Put => 0.0,
            });
        }

        // E[A^2]: with observations sorted by date, min(t_i, t_j) = t_i for
        // j > i, so sum_ij F_i F_j e^{s^2 min} folds into one pass over
        // suffix sums
        let sig2 = vol * vol;
        let mut suffix = m1 * nu; // sum of F_j for j >= i, walked down
        let mut sum2 = 0.0;
        for &(t, f) in &unfixed {
            suffix -= f;
            sum2 += (sig2 * t).exp() * f * (f + 2.0 * suffix);
        }
        let m2 = sum2 / (nu * nu);
        // total variance of the lognormal proxy; clamp numerical noise
        let v = (m2.ln() - 2.0 * m1.ln()).max(0.0);

        if v <= 1e-300 {
            return Ok(weight * intrinsic(m1 - k_eff));
        }
        let sq = v.sqrt();
        let d1 = (m1 / k_eff).ln() / sq + 0.5 * sq;
        let d2 = d1 - sq;
        Ok(match self.put_or_call {
            PutOrCall::Call => weight * (m1 * norm_cdf(d1) - k_eff * norm_cdf(d2)),
            PutOrCall::Put => weight * (k_eff * norm_cdf(-d2) - m1 * norm_cdf(-d1)),
        })
    }

    /// Delta against a parallel move of the forward strip, by central
    /// bump (averaging has not started; mid-life, central-bump
    /// [`price_with_fixings`](Self::price_with_fixings) instead).
    pub fn delta(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<f64, RustyQLibError> {
        let h = 1e-4;
        let up = self.price(discount, &forward.bumped(h)?, vol)?;
        let down = self.price(discount, &forward.bumped(-h)?, vol)?;
        Ok((up - down) / (2.0 * h))
    }

    /// Vega per unit of volatility, by central bump.
    pub fn vega(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<f64, RustyQLibError> {
        let h = 1e-4;
        let up = self.price(discount, forward, vol + h)?;
        let down = self.price(discount, forward, (vol - h).max(0.0))?;
        Ok((up - down) / (2.0 * h))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;
    use crate::equity::black76::{self, FuturesSettlement};

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

    /// Jun-27 monthly APO valued from Sep 1 2026.
    fn jun27(strike: f64, put_or_call: PutOrCall) -> AveragePriceOption {
        AveragePriceOption::for_month(
            1_000.0,
            strike,
            put_or_call,
            2027,
            6,
            Calendar::WeekendsOnly,
        )
        .unwrap()
    }

    #[test]
    fn monthly_contract_covers_its_month() {
        let apo = jun27(75.0, PutOrCall::Call);
        let days = apo.pricing_days();
        // June 2027 has 22 weekdays
        assert_eq!(days.len(), 22);
        assert_eq!(days[0], d(2027, 6, 1));
        assert_eq!(*days.last().unwrap(), d(2027, 6, 30));
        // settlement 5 business days after the Jul 1 boundary (Thursday)
        assert_eq!(apo.settlement_date(), d(2027, 7, 8));
    }

    #[test]
    fn single_observation_reduces_to_black76() {
        // averaging window with exactly one business day: Wed Jun 16 2027
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::flat(72.0, valuation).unwrap();
        let apo = AveragePriceOption::new(
            1_000.0,
            70.0,
            PutOrCall::Call,
            d(2027, 6, 16),
            d(2027, 6, 17),
            Calendar::WeekendsOnly,
            0,
        )
        .unwrap();
        assert_eq!(apo.pricing_days(), vec![d(2027, 6, 16)]);
        // one observation: the average IS the price, variance sigma^2 t_obs;
        // our df runs to settlement, so compare against the margined
        // (undiscounted) Black-76 kernel discounted by hand
        let t_obs = DayCountConvention::Act365.year_fraction(valuation, d(2027, 6, 16));
        let kernel = black76::price(
            72.0,
            70.0,
            0.0,
            0.35,
            t_obs,
            PutOrCall::Call,
            FuturesSettlement::Margined,
        );
        let expected = 1_000.0 * discount.df_date(d(2027, 6, 17)) * kernel;
        let price = apo.price(&discount, &forward, 0.35).unwrap();
        assert!((price - expected).abs() < 1e-8, "{price} vs {expected}");
    }

    #[test]
    fn put_call_parity_holds() {
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::from_prices(
            valuation,
            vec![(d(2026, 9, 1), 70.0), (d(2027, 9, 1), 78.0)],
        )
        .unwrap();
        let call = jun27(75.0, PutOrCall::Call);
        let put = jun27(75.0, PutOrCall::Put);
        let c = call.price(&discount, &forward, 0.35).unwrap();
        let p = put.price(&discount, &forward, 0.35).unwrap();
        // c - p = df * (E[A] - K) * quantity; E[A] over June's 22 weekdays
        let days = call.pricing_days();
        let avg = days.iter().map(|&day| forward.price(day)).sum::<f64>() / days.len() as f64;
        let parity = 1_000.0 * discount.df_date(call.settlement_date()) * (avg - 75.0);
        assert!((c - p - parity).abs() < 1e-8, "{} vs {parity}", c - p);
    }

    #[test]
    fn moment_matching_agrees_with_one_factor_monte_carlo() {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};

        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::from_prices(
            valuation,
            vec![(d(2026, 9, 1), 70.0), (d(2027, 9, 1), 78.0)],
        )
        .unwrap();
        let apo = jun27(75.0, PutOrCall::Call);
        let vol = 0.35;
        let analytic = apo.price(&discount, &forward, vol).unwrap();

        // simulate the single driving factor at each observation date
        let days = apo.pricing_days();
        let obs: Vec<(f64, f64)> = days
            .iter()
            .map(|&day| {
                (
                    DayCountConvention::Act365.year_fraction(valuation, day),
                    forward.price(day),
                )
            })
            .collect();
        let df = discount.df_date(apo.settlement_date());
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        let paths = 100_000;
        let mut sum_payoff = 0.0;
        for _ in 0..paths {
            let z: Vec<f64> = (0..obs.len())
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            // antithetic pair
            for sign in [1.0, -1.0] {
                let (mut w, mut t_prev, mut avg) = (0.0, 0.0f64, 0.0);
                for (i, &(t, f)) in obs.iter().enumerate() {
                    w += (t - t_prev).sqrt() * sign * z[i];
                    t_prev = t;
                    avg += f * (vol * w - 0.5 * vol * vol * t).exp();
                }
                avg /= obs.len() as f64;
                sum_payoff += (avg - 75.0).max(0.0);
            }
        }
        let mc = 1_000.0 * df * sum_payoff / (2.0 * paths as f64);
        // ~8.5k premium; Levy error and MC noise are both well inside 1%
        let tolerance = 0.01 * analytic;
        assert!(
            (analytic - mc).abs() < tolerance,
            "analytic {analytic} vs MC {mc}"
        );
    }

    #[test]
    fn zero_vol_prices_the_discounted_intrinsic() {
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::flat(78.0, valuation).unwrap();
        let apo = jun27(75.0, PutOrCall::Call);
        let price = apo.price(&discount, &forward, 0.0).unwrap();
        let expected = 1_000.0 * discount.df_date(apo.settlement_date()) * 3.0;
        assert!((price - expected).abs() < 1e-8, "{price} vs {expected}");
        // and an OTM put at zero vol is worthless
        assert_eq!(
            jun27(75.0, PutOrCall::Put)
                .price(&discount, &forward, 0.0)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn fixings_shift_the_strike_and_fully_fixed_is_deterministic() {
        // value mid-June: half the month realized at 80
        let valuation = d(2027, 6, 16);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::flat(76.0, valuation).unwrap();
        let mut fixings = PriceFixings::new();
        let mut day = d(2027, 6, 1);
        while day < d(2027, 7, 1) {
            fixings.insert(day, 80.0);
            day = day.succ_opt().unwrap();
        }
        let call = jun27(75.0, PutOrCall::Call);
        let mid = call
            .price_with_fixings(&discount, &forward, 0.35, &fixings)
            .unwrap();
        // 11 of 22 days fixed at 80: adjusted strike (22*75 - 11*80)/11 = 70,
        // on a floating average worth 76 — comfortably in the money
        let df = discount.df_date(call.settlement_date());
        let deep_floor = 1_000.0 * df * ((11.0 * 80.0 + 11.0 * 76.0) / 22.0 - 75.0);
        assert!(mid > deep_floor, "{mid} vs floor {deep_floor}");
        // fully realized (valuing after the month): deterministic payout
        let after = flat_discount(0.04, d(2027, 7, 2));
        let settled = call
            .price_with_fixings(&after, &forward, 0.35, &fixings)
            .unwrap();
        let expected = 1_000.0 * after.df_date(call.settlement_date()) * 5.0;
        assert!((settled - expected).abs() < 1e-8, "{settled} vs {expected}");
        // the put against those fixings is worthless once k_eff < 0
        let put = jun27(35.0, PutOrCall::Put);
        assert_eq!(
            put.price_with_fixings(&discount, &forward, 0.35, &fixings)
                .unwrap(),
            0.0
        );
    }

    #[test]
    fn averaging_discount_makes_the_apo_cheaper_than_the_vanilla() {
        // same strike and terminal date: averaging samples earlier (less
        // variance), so the APO must cost less than the vanilla on the
        // last observation
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::flat(75.0, valuation).unwrap();
        let apo = jun27(75.0, PutOrCall::Call)
            .price(&discount, &forward, 0.35)
            .unwrap();
        let t_last = DayCountConvention::Act365.year_fraction(valuation, d(2027, 6, 30));
        let vanilla = 1_000.0
            * discount.df_date(d(2027, 7, 8))
            * black76::price(
                75.0,
                75.0,
                0.0,
                0.35,
                t_last,
                PutOrCall::Call,
                FuturesSettlement::Margined,
            );
        assert!(apo < vanilla, "{apo} vs {vanilla}");
    }

    #[test]
    fn delta_and_vega_have_option_signs() {
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let forward = CommodityForwardCurve::flat(75.0, valuation).unwrap();
        let call = jun27(75.0, PutOrCall::Call);
        let delta = call.delta(&discount, &forward, 0.35).unwrap();
        // ATM call: delta near half the discounted quantity
        assert!(delta > 300.0 && delta < 700.0, "{delta}");
        assert!(call.vega(&discount, &forward, 0.35).unwrap() > 0.0);
        let put = jun27(75.0, PutOrCall::Put);
        assert!(put.delta(&discount, &forward, 0.35).unwrap() < 0.0);
    }

    #[test]
    fn validation_and_degenerate_inputs_error() {
        let cal = Calendar::WeekendsOnly;
        let (s, e) = (d(2027, 6, 1), d(2027, 7, 1));
        let pc = PutOrCall::Call;
        assert!(AveragePriceOption::new(0.0, 75.0, pc, s, e, cal.clone(), 5).is_err());
        assert!(AveragePriceOption::new(1e3, -75.0, pc, s, e, cal.clone(), 5).is_err());
        assert!(AveragePriceOption::new(1e3, 75.0, pc, e, s, cal.clone(), 5).is_err());
        assert!(AveragePriceOption::new(1e3, 75.0, pc, s, e, cal.clone(), -1).is_err());
        let valuation = d(2026, 9, 1);
        let discount = flat_discount(0.04, valuation);
        let apo = jun27(75.0, pc);
        // negative vol, negative forwards, and valuing after settlement
        let forward = CommodityForwardCurve::flat(75.0, valuation).unwrap();
        assert!(apo.price(&discount, &forward, -0.1).is_err());
        let negative = CommodityForwardCurve::flat(-37.63, valuation).unwrap();
        assert!(apo.price(&discount, &negative, 0.35).is_err());
        let late = flat_discount(0.04, d(2027, 7, 9));
        assert!(apo.price(&late, &forward, 0.35).is_err());
        // averaging already started but no fixings supplied
        let mid = flat_discount(0.04, d(2027, 6, 16));
        assert!(apo.price(&mid, &forward, 0.35).is_err());
    }
}
