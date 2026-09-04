//! European commodity swaption: the option to enter a
//! [`CommoditySwap`] at expiry.
//!
//! A payer swaption (the underlying swap's side is
//! [`PayerReceiver::Payer`]) is the right to start paying the fixed
//! price `swap.fixed_price` and receiving the floating average — a
//! producer-consumer hedge on future hedging levels. Exercise is
//! rational when the swap's par price at expiry exceeds the strike, so
//! the payoff is
//!
//! `annuity(T_ex) * max(P(T_ex) - K, 0)`
//!
//! (reversed for a receiver), where `P` is the par price and the
//! annuity is `quantity * sum df(pay_i)` over the swap's settlement
//! dates. Cash and physical settlement are equivalent here.
//!
//! # Model
//!
//! Black-on-par times the annuity — and under this library's one-factor
//! flat-vol dynamics that formula is **exact**, not an approximation:
//! at expiry every forward in the strip carries the same accumulated
//! shock, so the par price — a weighted average of forwards — is itself
//! exactly lognormal with total variance `sigma^2 T_ex` (exactly
//! normal under a [`CommodityVol::Normal`] quote, exactly displaced
//! lognormal under a shifted quote). Rates are deterministic (the
//! annuity does not diffuse), the standard assumption for commodity
//! swaptions, where the commodity price dwarfs rate risk.

use chrono::NaiveDate;

use crate::cmdty::apo::black_on_lognormal_moments;
use crate::cmdty::bachelier;
use crate::cmdty::clewlow_strickland::ClewlowStrickland;
use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::cmdty::swap::{positive_annuity, CommoditySwap};
use crate::cmdty::vol::CommodityVol;
use crate::cmdty::{expiry_inputs, vol_time};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::equity::black76::{self, FuturesSettlement};
use crate::rates::PayerReceiver;

const FIELD: &str = "commodity swaption";

/// A European option to enter `swap` at `expiry_date`. The strike is
/// the swap's `fixed_price`; the payer/receiver side is the swap's.
#[derive(Debug, Clone)]
pub struct CommoditySwaption {
    pub swap: CommoditySwap,
    /// Exercise date; on exercise the swap runs as written. Must not be
    /// after the swap's effective date.
    pub expiry_date: NaiveDate,
}

impl CommoditySwaption {
    pub fn new(swap: CommoditySwap, expiry_date: NaiveDate) -> Result<Self, RustyQLibError> {
        if expiry_date > swap.effective_date {
            return Err(RustyQLibError::invalid_input(
                "commodity swaption",
                format!(
                    "expiry {expiry_date} must not be after the swap's effective date {} \
                     (exercising into a running swap is not modelled)",
                    swap.effective_date
                ),
            ));
        }
        Ok(CommoditySwaption { swap, expiry_date })
    }

    /// Today's forward par price of the underlying swap — the
    /// swaption's underlying.
    pub fn forward_par_price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<f64, RustyQLibError> {
        self.swap.par_price(discount, forward)
    }

    /// The settlement annuity `quantity * sum df(pay_i)`: the swaption
    /// value is the annuity times an option on the par price.
    pub fn annuity(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        self.swap
            .settlement_annuity(discount, self.swap.effective_date)
    }

    /// The option side on the par price: a payer exercises when par
    /// exceeds the strike (a call), a receiver the other way.
    fn option_side(&self) -> PutOrCall {
        match self.swap.payer_receiver {
            PayerReceiver::Payer => PutOrCall::Call,
            PayerReceiver::Receiver => PutOrCall::Put,
        }
    }

    /// Premium. A bare `f64` vol is a Black (lognormal) vol on the par
    /// price; pass a [`CommodityVol`] to select the shifted or normal
    /// model. Within the one-factor flat-vol dynamics each formula is
    /// exact (see the module docs).
    pub fn price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: impl Into<CommodityVol>,
    ) -> Result<f64, RustyQLibError> {
        let quote = vol.into().validated("commodity swaption")?;
        let t = expiry_inputs(FIELD, self.expiry_date, discount)?.t;
        let p0 = self.forward_par_price(discount, forward)?;
        let k = self.swap.fixed_price;
        let pc = self.option_side();
        // the annuity carries all the discounting, so the kernel runs
        // undiscounted (margined, r = 0)
        let undiscounted = match quote {
            CommodityVol::Lognormal(v) => {
                displaced_par(p0, k, 0.0)?;
                black76::price(p0, k, 0.0, v, t, pc, FuturesSettlement::Margined)
            }
            CommodityVol::ShiftedLognormal { vol, shift } => {
                displaced_par(p0, k, shift)?;
                black76::price(
                    p0 + shift,
                    k + shift,
                    0.0,
                    vol,
                    t,
                    pc,
                    FuturesSettlement::Margined,
                )
            }
            CommodityVol::Normal(v) => {
                bachelier::price(p0, k, 0.0, v, t, pc, FuturesSettlement::Margined)
            }
        };
        Ok(self.annuity(discount)? * undiscounted)
    }

    /// Premium under Clewlow–Strickland forward dynamics (lognormal
    /// only). The par price is a weighted basket of the strip's daily
    /// forwards, and at expiry each carries the variance of **its own
    /// maturity** — a prompt month has moved more than a distant one —
    /// so the basket is no longer exactly lognormal and is
    /// moment-matched (Levy) with the CS covariances over `[0, T_ex]`.
    /// `alpha = 0` reproduces [`price`](Self::price) with the flat vol
    /// `sigma` exactly.
    pub fn price_cs(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        model: &ClewlowStrickland,
    ) -> Result<f64, RustyQLibError> {
        let valuation = discount.reference_date();
        let t_ex = expiry_inputs(FIELD, self.expiry_date, discount)?.t;
        let periods = self.swap.periods()?;
        let dfs: Vec<f64> = periods
            .iter()
            .map(|p| discount.df_date(p.payment))
            .collect();
        let a_df = positive_annuity(FIELD, dfs.iter().sum())?;
        // the par price as a weighted basket of the daily forwards
        let mut obs: Vec<(f64, f64, f64)> = Vec::new(); // (T_d, F_d, w_d)
        for (period, &df_p) in periods.iter().zip(&dfs) {
            let days = self.swap.pricing_days(period);
            if days.is_empty() {
                return Err(RustyQLibError::invalid_input(
                    "commodity swaption",
                    format!(
                        "no pricing days between {} and {}",
                        period.start, period.end
                    ),
                ));
            }
            let w = df_p / (a_df * days.len() as f64);
            for day in days {
                let f = forward.price(day);
                if f <= 0.0 {
                    return Err(RustyQLibError::invalid_input(
                        "commodity swaption",
                        format!("lognormal moment matching needs positive forwards, got {f}"),
                    ));
                }
                // the observation's own maturity, on the same vol clock
                // as the exercise date
                obs.push((vol_time(valuation, day), f, w));
            }
        }
        let m1: f64 = obs.iter().map(|&(_, f, w)| w * f).sum();
        let k = self.swap.fixed_price;
        let pc = self.option_side();
        let annuity = self.annuity(discount)?;
        // a non-positive strike on a positive basket: the payer always
        // exercises, the receiver never does
        if k <= 0.0 {
            return Ok(match pc {
                PutOrCall::Call => annuity * (m1 - k),
                PutOrCall::Put => 0.0,
            });
        }
        // E[P^2]: every forward observed over [0, T_ex] with the loading
        // of its own maturity (the weights already sum to one)
        let mut m2 = 0.0;
        for &(cap_ti, f_i, w_i) in &obs {
            for &(cap_tj, f_j, w_j) in &obs {
                m2 += w_i * w_j * f_i * f_j * model.covariance(t_ex, cap_ti, t_ex, cap_tj).exp();
            }
        }
        let v = (m2.ln() - 2.0 * m1.ln()).max(0.0);
        Ok(annuity * black_on_lognormal_moments(m1, k, v, pc))
    }

    /// Delta against a parallel move of the forward strip, by central
    /// bump (a $1 strip move lifts the par price by exactly $1).
    pub fn delta(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: impl Into<CommodityVol>,
    ) -> Result<f64, RustyQLibError> {
        let quote = vol.into();
        let h = 1e-4;
        let up = self.price(discount, &forward.bumped(h)?, quote)?;
        let down = self.price(discount, &forward.bumped(-h)?, quote)?;
        Ok((up - down) / (2.0 * h))
    }

    /// Vega per unit of the quote's vol, by central bump.
    ///
    /// The down bump is floored at zero vol, so at (or just above) a
    /// zero quote the pair is one-sided; dividing by the **realized**
    /// bump width rather than `2h` keeps it a true difference quotient
    /// instead of halving the forward difference.
    pub fn vega(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: impl Into<CommodityVol>,
    ) -> Result<f64, RustyQLibError> {
        let quote = vol.into();
        let h = 1e-4;
        let (qu, qd) = (quote.bumped_vol(h), quote.bumped_vol(-h));
        let width = qu.vol() - qd.vol();
        if width <= 0.0 {
            return Ok(0.0);
        }
        let up = self.price(discount, forward, qu)?;
        let down = self.price(discount, forward, qd)?;
        Ok((up - down) / width)
    }
}

/// The lognormal kernels need the (displaced) par price and strike
/// positive.
fn displaced_par(p0: f64, k: f64, shift: f64) -> Result<(), RustyQLibError> {
    if p0 + shift <= 0.0 || k + shift <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "commodity swaption",
            format!(
                "par price {p0} / strike {k} not priceable lognormally with shift {shift}; \
                 use a larger shift or a normal vol"
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::calendar::Calendar;
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

    /// Swaption expiring Aug 24 2027 into a 6m monthly swap running
    /// Sep-27 through Feb-28 on 10,000 bbl a month.
    fn swaption(fixed_price: f64, side: PayerReceiver) -> CommoditySwaption {
        let swap = CommoditySwap::monthly(
            10_000.0,
            fixed_price,
            side,
            d(2027, 9, 1),
            d(2028, 3, 1),
            Calendar::WeekendsOnly,
        )
        .unwrap();
        CommoditySwaption::new(swap, d(2027, 8, 24)).unwrap()
    }

    fn market() -> (YieldCurve, CommodityForwardCurve) {
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), 70.0), (d(2028, 3, 1), 79.0)],
        )
        .unwrap();
        (discount, forward)
    }

    #[test]
    fn matches_black_on_par_times_annuity_by_hand() {
        let (discount, forward) = market();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let p0 = payer.forward_par_price(&discount, &forward).unwrap();
        let a = payer.annuity(&discount).unwrap();
        let t = DayCountConvention::Act365.year_fraction(d(2026, 9, 1), d(2027, 8, 24));
        let expected = a * black76::price(
            p0,
            74.0,
            0.0,
            0.30,
            t,
            PutOrCall::Call,
            FuturesSettlement::Margined,
        );
        let price = payer.price(&discount, &forward, 0.30).unwrap();
        assert!((price - expected).abs() < 1e-8, "{price} vs {expected}");
        // the par sits inside the strip and the annuity is ~6 months of
        // discounted barrels
        assert!(p0 > 70.0 && p0 < 79.0, "par {p0}");
        assert!(a > 0.0 && a < 60_000.0, "annuity {a}");
    }

    #[test]
    fn payer_minus_receiver_is_the_forward_swap_pv() {
        // swaption parity: exercising one side or the other always nets
        // to the swap itself
        let (discount, forward) = market();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let receiver = swaption(74.0, PayerReceiver::Receiver);
        let diff = payer.price(&discount, &forward, 0.30).unwrap()
            - receiver.price(&discount, &forward, 0.30).unwrap();
        let swap_pv = payer.swap.pv(&discount, &forward).unwrap();
        assert!((diff - swap_pv).abs() < 1e-8, "{diff} vs {swap_pv}");
        // and at the forward par strike the two sides are worth the same
        let p0 = payer.forward_par_price(&discount, &forward).unwrap();
        let atm_payer = swaption(p0, PayerReceiver::Payer);
        let atm_receiver = swaption(p0, PayerReceiver::Receiver);
        let cp = atm_payer.price(&discount, &forward, 0.30).unwrap();
        let rp = atm_receiver.price(&discount, &forward, 0.30).unwrap();
        assert!((cp - rp).abs() < 1e-8, "{cp} vs {rp}");
        assert!(cp > 0.0);
    }

    #[test]
    fn zero_vol_collapses_to_the_positive_part_of_the_swap_pv() {
        let (discount, forward) = market();
        // strike below par: the payer swaption at zero vol IS the swap
        let payer = swaption(72.0, PayerReceiver::Payer);
        let price = payer.price(&discount, &forward, 0.0).unwrap();
        let swap_pv = payer.swap.pv(&discount, &forward).unwrap();
        assert!(swap_pv > 0.0);
        assert!((price - swap_pv).abs() < 1e-8, "{price} vs {swap_pv}");
        // strike above par: worthless at zero vol
        let otm = swaption(85.0, PayerReceiver::Payer);
        assert_eq!(otm.price(&discount, &forward, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn shifted_model_is_the_lognormal_on_displaced_market() {
        let (discount, forward) = market();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let shift = 20.0;
        let via_shift = payer
            .price(
                &discount,
                &forward,
                CommodityVol::ShiftedLognormal { vol: 0.30, shift },
            )
            .unwrap();
        // same swaption on the displaced market: curve + 20, strike + 20
        let displaced_curve = CommodityForwardCurve::from_prices(
            d(2026, 9, 1),
            vec![(d(2026, 9, 1), 90.0), (d(2028, 3, 1), 99.0)],
        )
        .unwrap();
        let displaced = swaption(94.0, PayerReceiver::Payer);
        let via_displacement = displaced.price(&discount, &displaced_curve, 0.30).unwrap();
        assert!(
            (via_shift - via_displacement).abs() < 1e-8,
            "{via_shift} vs {via_displacement}"
        );
    }

    #[test]
    fn normal_model_handles_a_negative_par_price() {
        // a Waha-style negative strip: lognormal refuses, Bachelier prices
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), -2.0), (d(2028, 3, 1), -0.5)],
        )
        .unwrap();
        let swap = CommoditySwap::monthly(
            10_000.0,
            -1.0,
            PayerReceiver::Payer,
            d(2027, 9, 1),
            d(2028, 3, 1),
            Calendar::WeekendsOnly,
        )
        .unwrap();
        let swaption = CommoditySwaption::new(swap, d(2027, 8, 24)).unwrap();
        assert!(swaption.price(&discount, &forward, 0.30).is_err());
        let price = swaption
            .price(&discount, &forward, CommodityVol::Normal(1.5))
            .unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn cs_with_zero_alpha_reduces_to_black_on_par() {
        let (discount, forward) = market();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let flat = payer.price(&discount, &forward, 0.30).unwrap();
        let cs = ClewlowStrickland::new(0.30, 0.0).unwrap();
        let via_cs = payer.price_cs(&discount, &forward, &cs).unwrap();
        assert!((flat - via_cs).abs() < 1e-8 * flat, "{flat} vs {via_cs}");
    }

    #[test]
    fn samuelson_decay_cheapens_the_swaption_and_parity_survives() {
        let (discount, forward) = market();
        let cs = ClewlowStrickland::new(0.30, 1.4).unwrap();
        let payer = swaption(74.0, PayerReceiver::Payer);
        // every covariance entry is damped versus flat sigma^2 t, so the
        // near-the-money option loses time value
        let flat = payer.price(&discount, &forward, 0.30).unwrap();
        let damped = payer.price_cs(&discount, &forward, &cs).unwrap();
        assert!(damped < flat, "{damped} vs {flat}");
        // moment matching preserves parity: payer - receiver = swap PV
        let receiver = swaption(74.0, PayerReceiver::Receiver);
        let diff = damped - receiver.price_cs(&discount, &forward, &cs).unwrap();
        let swap_pv = payer.swap.pv(&discount, &forward).unwrap();
        assert!((diff - swap_pv).abs() < 1e-8, "{diff} vs {swap_pv}");
    }

    #[test]
    fn cs_rejects_a_negative_strip() {
        let reference = d(2026, 9, 1);
        let discount = flat_discount(0.04, reference);
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), -2.0), (d(2028, 3, 1), -0.5)],
        )
        .unwrap();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let cs = ClewlowStrickland::new(0.30, 1.4).unwrap();
        assert!(payer.price_cs(&discount, &forward, &cs).is_err());
    }

    #[test]
    fn delta_and_vega_have_option_signs() {
        let (discount, forward) = market();
        let payer = swaption(74.0, PayerReceiver::Payer);
        let annuity = payer.annuity(&discount).unwrap();
        let delta = payer.delta(&discount, &forward, 0.30).unwrap();
        // a payer swaption is long the strip, capped by the annuity
        assert!(delta > 0.0 && delta < annuity, "{delta} vs {annuity}");
        assert!(payer.vega(&discount, &forward, 0.30).unwrap() > 0.0);
        let receiver = swaption(74.0, PayerReceiver::Receiver);
        assert!(receiver.delta(&discount, &forward, 0.30).unwrap() < 0.0);
    }

    #[test]
    fn vega_at_a_zero_vol_quote_uses_the_realized_bump_width() {
        // bumping down from zero is clamped at zero, so the pair spans
        // h, not 2h: dividing by 2h would halve the sensitivity
        let (discount, forward) = market();
        let h = 1e-4;
        // struck at the forward par, so a vol bump is pure time value
        let p0 = swaption(74.0, PayerReceiver::Payer)
            .forward_par_price(&discount, &forward)
            .unwrap();
        let atm = swaption(p0, PayerReceiver::Payer);
        let vega = atm.vega(&discount, &forward, 0.0).unwrap();
        let up = atm.price(&discount, &forward, h).unwrap();
        let down = atm.price(&discount, &forward, 0.0).unwrap();
        assert_eq!(down, 0.0);
        assert!(vega > 0.0, "{vega}");
        assert!((vega - (up - down) / h).abs() < 1e-10, "{vega}");
        // the nominal 2h would have reported half of it
        assert!(((up - down) / (2.0 * h) - 0.5 * vega).abs() < 1e-10);
        // and the same holds for a normal quote
        let vega_n = atm
            .vega(&discount, &forward, CommodityVol::Normal(0.0))
            .unwrap();
        let up_n = atm
            .price(&discount, &forward, CommodityVol::Normal(h))
            .unwrap();
        let down_n = atm
            .price(&discount, &forward, CommodityVol::Normal(0.0))
            .unwrap();
        assert!(vega_n > 0.0, "{vega_n}");
        assert!((vega_n - (up_n - down_n) / h).abs() < 1e-10, "{vega_n}");
        // well away from zero the bump is two-sided again, so nothing
        // about the ordinary case changed
        let live = atm.vega(&discount, &forward, 0.30).unwrap();
        let u = atm.price(&discount, &forward, 0.30 + h).unwrap();
        let dn = atm.price(&discount, &forward, 0.30 - h).unwrap();
        let width = (0.30 + h) - (0.30 - h);
        assert!((live - (u - dn) / width).abs() < 1e-10, "{live}");
    }

    #[test]
    fn validation_and_expiry_errors() {
        let swap = CommoditySwap::monthly(
            10_000.0,
            74.0,
            PayerReceiver::Payer,
            d(2027, 9, 1),
            d(2028, 3, 1),
            Calendar::WeekendsOnly,
        )
        .unwrap();
        // expiry after the swap starts is rejected
        assert!(CommoditySwaption::new(swap.clone(), d(2027, 9, 2)).is_err());
        let swaption = CommoditySwaption::new(swap, d(2027, 8, 24)).unwrap();
        let (_, forward) = market();
        // valuing after expiry is rejected
        let late = flat_discount(0.04, d(2027, 8, 25));
        assert!(swaption.price(&late, &forward, 0.30).is_err());
        // bad vols are rejected
        let (discount, _) = market();
        assert!(swaption.price(&discount, &forward, -0.1).is_err());
    }
}
