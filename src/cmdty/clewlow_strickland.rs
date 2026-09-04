//! Clewlow–Strickland (1999) one-factor forward-curve dynamics.
//!
//! Every futures price is a driftless lognormal martingale whose
//! instantaneous vol decays exponentially in time-to-maturity:
//!
//! ```text
//! dF(t,T) / F(t,T) = sigma * e^{-alpha (T - t)} dW(t)
//! ```
//!
//! — the Samuelson effect as dynamics: prompt contracts rattle, distant
//! contracts barely move, with one overall level `sigma` and one decay
//! rate `alpha` (`alpha = 0` recovers the flat-vol model used
//! elsewhere in this module). It is the Hull-White of commodities:
//! today's forward curve is repriced by construction, and everything a
//! European pricer needs is the closed-form **integrated covariance**
//!
//! ```text
//! Cov[ln F(.,T_i), ln F(.,T_j)] over [0, m] =
//!     sigma^2 (e^{-alpha(T_i+T_j-2m)} - e^{-alpha(T_i+T_j)}) / (2 alpha)
//! ```
//!
//! Three uses here:
//!
//! - **Vanilla term structure** — [`effective_vol`](ClewlowStrickland::effective_vol)
//!   turns the integrated variance into the Black vol for an option
//!   expiring at `t` on the `T`-maturity future, emitted as a
//!   [`CommodityVol`] quote ([`vol_quote`](ClewlowStrickland::vol_quote) /
//!   [`quote_for`](ClewlowStrickland::quote_for)) — the model generates
//!   quotes across maturities the way [`ShiftedSabr`](crate::cmdty::ShiftedSabr)
//!   does across strikes.
//! - **Multi-date products** — the APO
//!   ([`price_cs`](crate::cmdty::AveragePriceOption::price_cs)) and the
//!   swaption ([`price_cs`](crate::cmdty::CommoditySwaption::price_cs))
//!   replace the flat `sigma^2 min(t_i, t_j)` covariance in their
//!   moment matching with the CS one, so strips stop treating a prompt
//!   observation and a year-out observation as equally volatile.
//! - **Calibration** — [`calibrate`](ClewlowStrickland::calibrate) fits
//!   `(sigma, alpha)` to a term structure of quoted Black vols by
//!   Levenberg-Marquardt in log space, the caplet-style workflow.

use crate::cmdty::vol::CommodityVol;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};

/// One-factor Clewlow–Strickland forward vol: level `sigma`, Samuelson
/// decay `alpha`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClewlowStrickland {
    /// Instantaneous vol of the maturing contract (lognormal, per √year).
    pub sigma: f64,
    /// Exponential decay of vol in time-to-maturity; `0` is flat vol.
    pub alpha: f64,
}

/// Result of a Clewlow–Strickland term-structure calibration.
#[derive(Debug, Clone)]
pub struct ClewlowStricklandFit {
    pub model: ClewlowStrickland,
    /// Root-mean-square error in Black vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl ClewlowStrickland {
    pub fn new(sigma: f64, alpha: f64) -> Result<Self, RustyQLibError> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "clewlow-strickland",
                format!("sigma must be positive, got {sigma}"),
            ));
        }
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "clewlow-strickland",
                format!("alpha must be non-negative, got {alpha}"),
            ));
        }
        Ok(ClewlowStrickland { sigma, alpha })
    }

    /// Integrated covariance of `ln F(., T_i)` and `ln F(., T_j)` over
    /// the common observation window `[0, min(t_i, t_j)]` — the entry
    /// every moment-matching pricer needs. At `alpha = 0` this is the
    /// flat-vol `sigma^2 min(t_i, t_j)`.
    ///
    /// # Preconditions
    ///
    /// Each contract must still be alive over the whole window: both
    /// maturities at or after `min(t_i, t_j)`. A maturity inside the
    /// window integrates the vol of a contract that has already
    /// expired and inflates the covariance without bound as `alpha`
    /// grows. Checked by `debug_assert` only — this is the inner term
    /// of an O(n^2) loop inside a calibration closure, and every caller
    /// in this module builds the arguments from a schedule that
    /// satisfies it by construction.
    pub fn covariance(&self, t_i: f64, cap_ti: f64, t_j: f64, cap_tj: f64) -> f64 {
        let m = t_i.min(t_j);
        debug_assert!(
            cap_ti >= m && cap_tj >= m,
            "covariance window [0, {m}] runs past a maturity (T_i = {cap_ti}, T_j = {cap_tj})"
        );
        if m <= 0.0 {
            return 0.0;
        }
        let s2 = self.sigma * self.sigma;
        let a = self.alpha;
        if a < 1e-10 {
            return s2 * m;
        }
        let sum = cap_ti + cap_tj;
        s2 * ((-a * (sum - 2.0 * m)).exp() - (-a * sum).exp()) / (2.0 * a)
    }

    /// Total Black variance to expiry `t` of the `T`-maturity future.
    pub fn total_variance(&self, t: f64, cap_t: f64) -> f64 {
        self.covariance(t, cap_t, t, cap_t)
    }

    /// The effective Black vol for an option expiring at `t` on the
    /// `T`-maturity future: `sqrt(total_variance / t)`.
    pub fn effective_vol(&self, t: f64, cap_t: f64) -> Result<f64, RustyQLibError> {
        if !t.is_finite() || t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "clewlow-strickland",
                format!("expiry must be positive, got {t}"),
            ));
        }
        if !cap_t.is_finite() || cap_t < t {
            return Err(RustyQLibError::invalid_input(
                "clewlow-strickland",
                format!("underlying maturity {cap_t} must not precede expiry {t}"),
            ));
        }
        Ok((self.total_variance(t, cap_t) / t).sqrt())
    }

    /// The effective vol wrapped as a [`CommodityVol::Lognormal`]
    /// pricing quote.
    pub fn vol_quote(&self, t: f64, cap_t: f64) -> Result<CommodityVol, RustyQLibError> {
        Ok(CommodityVol::Lognormal(self.effective_vol(t, cap_t)?))
    }

    /// The quote for a [`CommodityOption`](crate::cmdty::CommodityOption),
    /// resolving expiry and underlying maturity exactly as its pricing
    /// does (the Act/365 vol time from the discount curve's reference
    /// date — see the [`crate::cmdty`] conventions).
    pub fn quote_for(
        &self,
        option: &crate::cmdty::CommodityOption,
        discount: &crate::core::curves::YieldCurve,
    ) -> Result<CommodityVol, RustyQLibError> {
        let valuation = discount.reference_date();
        let t = crate::cmdty::vol_time(valuation, option.expiry_date);
        let cap_t = crate::cmdty::vol_time(valuation, option.underlying_date);
        self.vol_quote(t, cap_t)
    }

    /// Calibrate `(sigma, alpha)` to a term structure of quoted Black
    /// vols, each quote `(t, T, vol)`: option expiry, underlying
    /// maturity, implied vol. Levenberg-Marquardt in log space, so
    /// every trial parameter set is admissible.
    pub fn calibrate(quotes: &[(f64, f64, f64)]) -> Result<ClewlowStricklandFit, RustyQLibError> {
        if quotes.len() < 2 {
            return Err(RustyQLibError::invalid_input(
                "clewlow-strickland",
                format!(
                    "two parameters need at least two quotes, got {}",
                    quotes.len()
                ),
            ));
        }
        let mut max_vol: f64 = 0.0;
        for &(t, cap_t, v) in quotes {
            if !t.is_finite() || t <= 0.0 || !cap_t.is_finite() || cap_t < t {
                return Err(RustyQLibError::invalid_input(
                    "clewlow-strickland",
                    format!("quote needs 0 < expiry <= maturity, got ({t}, {cap_t})"),
                ));
            }
            if !v.is_finite() || v <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "clewlow-strickland",
                    format!("quoted vol must be positive, got {v} at ({t}, {cap_t})"),
                ));
            }
            max_vol = max_vol.max(v);
        }
        let unpack = |u: &[f64]| ClewlowStrickland {
            sigma: u[0].exp(),
            alpha: u[1].exp(),
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            let m = unpack(u);
            quotes
                .iter()
                .map(|&(t, cap_t, v)| (m.total_variance(t, cap_t) / t).sqrt() - v)
                .collect()
        };
        // effective vols sit below sigma, so start the level above the
        // largest quote; moderate decay
        let x0 = [(1.25 * max_vol).ln(), 1.0_f64.ln()];
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let model = unpack(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(t, cap_t, v)| ((model.total_variance(t, cap_t) / t).sqrt() - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        Ok(ClewlowStricklandFit {
            model,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_alpha_is_the_flat_vol_model() {
        let cs = ClewlowStrickland::new(0.40, 0.0).unwrap();
        for (t, cap_t) in [(0.25, 0.3), (0.5, 1.5), (1.0, 3.0)] {
            assert!((cs.effective_vol(t, cap_t).unwrap() - 0.40).abs() < 1e-12);
        }
        assert!((cs.covariance(0.5, 0.6, 0.75, 2.0) - 0.16 * 0.5).abs() < 1e-12);
    }

    #[test]
    fn samuelson_shapes_the_term_structure() {
        let cs = ClewlowStrickland::new(0.45, 1.4).unwrap();
        // same expiry, farther underlying: quieter
        let near = cs.effective_vol(0.25, 0.3).unwrap();
        let far = cs.effective_vol(0.25, 2.0).unwrap();
        assert!(near > far, "{near} vs {far}");
        // the maturing contract's vol approaches sigma as t -> T -> 0
        let prompt = cs.effective_vol(0.01, 0.01).unwrap();
        assert!((prompt - 0.45).abs() < 0.01, "{prompt}");
        // effective vol of a fixed contract rises as expiry nears delivery
        let early = cs.effective_vol(0.25, 1.0).unwrap();
        let late = cs.effective_vol(0.99, 1.0).unwrap();
        assert!(late > early, "{late} vs {early}");
    }

    #[test]
    fn covariance_matches_numerical_quadrature() {
        let cs = ClewlowStrickland::new(0.45, 1.4).unwrap();
        let (t_i, cap_ti, t_j, cap_tj): (f64, f64, f64, f64) = (0.6, 0.9, 0.8, 1.7);
        // Simpson on sigma^2 e^{-a(Ti-u)} e^{-a(Tj-u)} over [0, min]
        let m = t_i.min(t_j);
        let n = 2_000;
        let h = m / n as f64;
        let f = |u: f64| {
            cs.sigma
                * cs.sigma
                * (-cs.alpha * (cap_ti - u)).exp()
                * (-cs.alpha * (cap_tj - u)).exp()
        };
        let mut simpson = f(0.0) + f(m);
        for k in 1..n {
            simpson += f(k as f64 * h) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        simpson *= h / 3.0;
        let closed = cs.covariance(t_i, cap_ti, t_j, cap_tj);
        assert!((closed - simpson).abs() < 1e-10, "{closed} vs {simpson}");
    }

    #[test]
    fn calibration_round_trips_a_known_term_structure() {
        let truth = ClewlowStrickland::new(0.48, 1.6).unwrap();
        let points = [
            (0.1, 0.15),
            (0.25, 0.35),
            (0.5, 0.6),
            (0.75, 1.3),
            (1.0, 2.0),
        ];
        let quotes: Vec<(f64, f64, f64)> = points
            .iter()
            .map(|&(t, cap_t)| (t, cap_t, truth.effective_vol(t, cap_t).unwrap()))
            .collect();
        let fit = ClewlowStrickland::calibrate(&quotes).unwrap();
        assert!(fit.converged);
        assert!(fit.rmse < 1e-10, "rmse {}", fit.rmse);
        assert!((fit.model.sigma - 0.48).abs() < 1e-6, "{:?}", fit.model);
        assert!((fit.model.alpha - 1.6).abs() < 1e-5, "{:?}", fit.model);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(ClewlowStrickland::new(0.0, 1.0).is_err());
        assert!(ClewlowStrickland::new(0.4, -0.1).is_err());
        let cs = ClewlowStrickland::new(0.4, 1.0).unwrap();
        // maturity before expiry, and non-positive expiry
        assert!(cs.effective_vol(1.0, 0.5).is_err());
        assert!(cs.effective_vol(0.0, 0.5).is_err());
        // calibration guards
        assert!(ClewlowStrickland::calibrate(&[(0.5, 1.0, 0.4)]).is_err());
        assert!(ClewlowStrickland::calibrate(&[(0.5, 0.4, 0.4), (0.6, 0.8, 0.4)]).is_err());
        assert!(ClewlowStrickland::calibrate(&[(0.5, 1.0, -0.4), (0.6, 0.8, 0.4)]).is_err());
    }
}
