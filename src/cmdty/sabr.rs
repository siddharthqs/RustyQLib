//! Shifted SABR: the SABR smile on a displaced underlying.
//!
//! The market-standard smile model when the underlying can print
//! negative or sit near zero — adopted by rates desks for negative
//! rates and the natural fit for steep NG and basis smiles here. The
//! SABR dynamics ([`crate::equity::sabr`], Hagan et al. 2002) are
//! written on the **displaced** forward `F + shift`:
//!
//! ```text
//! d(F + shift) = alpha (F + shift)^beta dW_f
//! dalpha       = nu alpha dW_a,    d<W_f, W_a> = rho dt
//! ```
//!
//! so Hagan's expansion evaluated at `(F + shift, K + shift)` yields a
//! **shifted-Black implied vol** — precisely the vol a
//! [`CommodityVol::ShiftedLognormal`] quote carries. That makes the
//! integration seamless: [`ShiftedSabr::vol_quote`] turns the smile
//! into the quote and every commodity option prices through its
//! existing displaced Black-76 dispatch, strike by strike,
//! smile-consistently.
//!
//! Two conventions to keep straight (both from the rates market):
//! the `shift` is **declared, not calibrated** — chosen per hub and
//! published alongside the quotes — and quoted vols are only meaningful
//! **relative to their shift** (the same smile quoted at a different
//! shift has different vol numbers). Calibration therefore takes the
//! shift as an input and fits `(alpha, rho, nu)` at fixed `beta` in
//! displaced space, reusing the equity module's Levenberg-Marquardt
//! machinery.

use crate::cmdty::vol::CommodityVol;
use crate::core::errors::RustyQLibError;
use crate::equity::sabr::SabrParams;

/// A SABR smile on the displaced forward `F + shift`. `shift = 0` is
/// plain SABR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShiftedSabr {
    pub params: SabrParams,
    /// Displacement, in price units: supports prices down to `-shift`.
    pub shift: f64,
}

/// Result of a shifted SABR smile calibration.
#[derive(Debug, Clone)]
pub struct ShiftedSabrFit {
    pub sabr: ShiftedSabr,
    /// Root-mean-square error in (shifted-Black) implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl ShiftedSabr {
    pub fn new(params: SabrParams, shift: f64) -> Result<Self, RustyQLibError> {
        params.validate()?;
        if !shift.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("shift must be finite, got {shift}"),
            ));
        }
        Ok(ShiftedSabr { params, shift })
    }

    /// The shifted-Black implied vol at strike `k` for forward `f` and
    /// expiry `t`: Hagan's expansion on the displaced pair
    /// `(f + shift, k + shift)`.
    pub fn vol(&self, f: f64, k: f64, t: f64) -> Result<f64, RustyQLibError> {
        let (fs, ks) = (f + self.shift, k + self.shift);
        if fs <= 0.0 || ks <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!(
                    "displaced forward {fs} / strike {ks} must be positive \
                     (forward {f}, strike {k}, shift {})",
                    self.shift
                ),
            ));
        }
        if !t.is_finite() || t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("expiry must be positive, got {t}"),
            ));
        }
        Ok(self.params.vol(fs, ks, t))
    }

    /// The pricing quote for one strike: the smile vol wrapped as a
    /// [`CommodityVol::ShiftedLognormal`] carrying this smile's shift —
    /// feed it straight into any commodity option's `price`/`greeks`.
    pub fn vol_quote(&self, f: f64, k: f64, t: f64) -> Result<CommodityVol, RustyQLibError> {
        Ok(CommodityVol::ShiftedLognormal {
            vol: self.vol(f, k, t)?,
            shift: self.shift,
        })
    }

    /// The quote for a [`CommodityOption`](crate::cmdty::CommodityOption),
    /// resolving the forward and time to expiry exactly as its pricing
    /// does (the forward curve at the underlying date, the Act/365 vol
    /// time to expiry — see the [`crate::cmdty`] conventions).
    pub fn quote_for(
        &self,
        option: &crate::cmdty::CommodityOption,
        discount: &crate::core::curves::YieldCurve,
        forward: &crate::cmdty::CommodityForwardCurve,
    ) -> Result<CommodityVol, RustyQLibError> {
        let f = option.forward_price(forward);
        let t = crate::cmdty::vol_time(discount.reference_date(), option.expiry_date);
        self.vol_quote(f, option.strike, t)
    }

    /// Calibrate `(alpha, rho, nu)` at fixed `beta` and **given
    /// `shift`** to one expiry's `(strike, shifted-Black vol)` quotes —
    /// the vols must be quoted against the same shift. Runs the equity
    /// module's Levenberg-Marquardt fit on the displaced market.
    pub fn calibrate(
        quotes: &[(f64, f64)],
        forward: f64,
        t: f64,
        beta: f64,
        shift: f64,
    ) -> Result<ShiftedSabrFit, RustyQLibError> {
        if quotes.len() < 3 {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!(
                    "three parameters need at least three quotes, got {}",
                    quotes.len()
                ),
            ));
        }
        if !shift.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("shift must be finite, got {shift}"),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("beta must lie in [0, 1], got {beta}"),
            ));
        }
        if !t.is_finite() || t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("expiry must be positive, got {t}"),
            ));
        }
        if forward + shift <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "shifted sabr",
                format!("displaced forward {} must be positive", forward + shift),
            ));
        }
        let mut displaced = Vec::with_capacity(quotes.len());
        for &(k, v) in quotes {
            if k + shift <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "shifted sabr",
                    format!("displaced strike {} must be positive", k + shift),
                ));
            }
            if !v.is_finite() || v <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "shifted sabr",
                    format!("quoted vol must be positive, got {v} at strike {k}"),
                ));
            }
            displaced.push((k + shift, v));
        }
        let fit = SabrParams::calibrate(&displaced, forward + shift, t, beta)?;
        Ok(ShiftedSabrFit {
            sabr: ShiftedSabr {
                params: fit.params,
                shift,
            },
            rmse: fit.rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmdty::{CommodityForwardCurve, CommodityOption, FuturesSettlement};
    use crate::core::curves::{Compounding, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::core::trade::PutOrCall;
    use crate::equity::black76;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn params(alpha: f64, beta: f64, rho: f64, nu: f64) -> SabrParams {
        SabrParams {
            alpha,
            beta,
            rho,
            nu,
        }
    }

    #[test]
    fn zero_shift_is_plain_hagan() {
        let p = params(0.35, 1.0, -0.3, 0.6);
        let sabr = ShiftedSabr::new(p, 0.0).unwrap();
        for k in [60.0, 72.0, 85.0] {
            assert_eq!(sabr.vol(72.0, k, 0.75).unwrap(), p.vol(72.0, k, 0.75));
        }
    }

    #[test]
    fn flat_limit_and_skew_signs_on_a_negative_market() {
        // Waha-style: forward -1.5, shift 10 -> displaced forward 8.5
        let (f, t, shift) = (-1.5, 0.75, 10.0);
        // beta = 1, nu = 0: the displaced smile is exactly flat at alpha
        let flat = ShiftedSabr::new(params(0.40, 1.0, 0.0, 0.0), shift).unwrap();
        for k in [-4.0, -1.5, 2.0] {
            assert!((flat.vol(f, k, t).unwrap() - 0.40).abs() < 1e-12);
        }
        // negative rho: downside strikes carry higher vol
        let skewed = ShiftedSabr::new(params(0.40, 1.0, -0.5, 0.8), shift).unwrap();
        let down = skewed.vol(f, -4.0, t).unwrap();
        let up = skewed.vol(f, 2.0, t).unwrap();
        let atm = skewed.vol(f, f, t).unwrap();
        assert!(down > atm, "{down} vs atm {atm}");
        assert!(down > up, "{down} vs {up}");
    }

    #[test]
    fn calibration_round_trips_a_known_smile() {
        let truth = ShiftedSabr::new(params(0.38, 0.7, -0.35, 0.65), 10.0).unwrap();
        let (f, t) = (-1.5, 0.75);
        let strikes = [-5.0, -3.0, -1.5, 0.5, 3.0, 6.0];
        let quotes: Vec<(f64, f64)> = strikes
            .iter()
            .map(|&k| (k, truth.vol(f, k, t).unwrap()))
            .collect();
        let fit = ShiftedSabr::calibrate(&quotes, f, t, 0.7, 10.0).unwrap();
        assert!(fit.converged);
        assert!(fit.rmse < 1e-8, "rmse {}", fit.rmse);
        // the recovered smile reprices every quote
        for &(k, v) in &quotes {
            let vol = fit.sabr.vol(f, k, t).unwrap();
            assert!((vol - v).abs() < 1e-6, "strike {k}: {vol} vs {v}");
        }
    }

    #[test]
    fn vol_quote_prices_smile_consistently_through_the_option() {
        // an OTM call on a negative forward priced off the smile equals
        // displaced Black-76 at the Hagan vol, assembled by hand
        let valuation = d(2026, 9, 1);
        let discount = YieldCurve::flat(
            0.04,
            valuation,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let curve = CommodityForwardCurve::flat(-1.5, valuation).unwrap();
        let option = CommodityOption::new(
            10_000.0,
            0.5,
            PutOrCall::Call,
            d(2027, 9, 1), // t = 1.0 exactly on Act/365
            d(2027, 9, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        let sabr = ShiftedSabr::new(params(0.38, 1.0, -0.35, 0.65), 10.0).unwrap();
        let quote = sabr.quote_for(&option, &discount, &curve).unwrap();
        let price = option.price(&discount, &curve, quote).unwrap();
        let vol = sabr.vol(-1.5, 0.5, 1.0).unwrap();
        let r = -discount.df_date(d(2027, 9, 1)).ln();
        let expected = 10_000.0
            * black76::price(
                -1.5 + 10.0,
                0.5 + 10.0,
                r,
                vol,
                1.0,
                PutOrCall::Call,
                FuturesSettlement::Discounted,
            );
        assert!((price - expected).abs() < 1e-6, "{price} vs {expected}");
        // and the smile makes the downside strike genuinely more expensive
        // in vol terms than the flat-quote alternative
        let atm_vol = sabr.vol(-1.5, -1.5, 1.0).unwrap();
        let mut put = option.clone();
        put.put_or_call = PutOrCall::Put;
        put.strike = -4.0;
        let smile_price = put
            .price(
                &discount,
                &curve,
                sabr.quote_for(&put, &discount, &curve).unwrap(),
            )
            .unwrap();
        let flat_price = put
            .price(
                &discount,
                &curve,
                CommodityVol::ShiftedLognormal {
                    vol: atm_vol,
                    shift: 10.0,
                },
            )
            .unwrap();
        assert!(smile_price > flat_price, "{smile_price} vs {flat_price}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let good = params(0.38, 0.7, -0.35, 0.65);
        // parameter validation flows through from SabrParams
        assert!(ShiftedSabr::new(params(-0.1, 0.7, 0.0, 0.5), 10.0).is_err());
        assert!(ShiftedSabr::new(good, f64::NAN).is_err());
        let sabr = ShiftedSabr::new(good, 10.0).unwrap();
        // displaced strike or forward non-positive, bad expiry
        assert!(sabr.vol(-1.5, -11.0, 0.75).is_err());
        assert!(sabr.vol(-15.0, 0.5, 0.75).is_err());
        assert!(sabr.vol(-1.5, 0.5, 0.0).is_err());
        // calibration guards: too few quotes, bad vols, bad displacement
        let quotes = [(-3.0, 0.4), (-1.5, 0.38)];
        assert!(ShiftedSabr::calibrate(&quotes, -1.5, 0.75, 0.7, 10.0).is_err());
        let bad_vol = [(-3.0, 0.4), (-1.5, -0.1), (0.5, 0.39)];
        assert!(ShiftedSabr::calibrate(&bad_vol, -1.5, 0.75, 0.7, 10.0).is_err());
        let bad_strike = [(-12.0, 0.4), (-1.5, 0.38), (0.5, 0.39)];
        assert!(ShiftedSabr::calibrate(&bad_strike, -1.5, 0.75, 0.7, 10.0).is_err());
        // beta outside [0, 1] on an otherwise valid quote set
        let ok = [(-3.0, 0.4), (-1.5, 0.38), (0.5, 0.39)];
        assert!(ShiftedSabr::calibrate(&ok, -1.5, 0.75, 1.5, 10.0).is_err());
    }
}
