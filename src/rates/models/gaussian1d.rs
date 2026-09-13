//! The Gaussian1d framework (QuantLib's `Gaussian1dModel`): any
//! one-factor model driven by a Gaussian Markov state `x` under the
//! **numeraire measure** of a terminal bond `P(., T*)`, described by
//! two functions of the state — the numeraire `N(t, x) = P(t, T*, x)`
//! and the zero bonds `P(t, T, x)` — plus the state's transition law.
//!
//! Every deflated price `V(t, x) / N(t, x)` is a martingale under that
//! measure, so one set of engines ([`gaussian1d`]) prices European and
//! Bermudan swaptions on any model that implements the trait: Gaussian
//! quadrature at expiry for a European, backward induction of deflated
//! values on an `x`-grid for a Bermudan. Two models implement it here:
//!
//! - Hull-White / GSR, through [`HullWhite::gaussian1d`], whose
//!   numeraire and bonds are closed form — the state is `x = r -
//!   alpha(t)` and the measure change adds a deterministic drift;
//! - the [`MarkovFunctional`] model, whose numeraire is a **calibrated
//!   functional** of the state read off the market's swaption digitals
//!   rather than a formula.
//!
//! [`gaussian1d`]: crate::rates::engines::gaussian1d
//! [`MarkovFunctional`]: crate::rates::models::markov_functional::MarkovFunctional

use crate::core::errors::RustyQLibError;
use crate::rates::models::{HullWhite, ShortRateModel};

/// A one-factor model in Gaussian1d form.
pub trait Gaussian1dModel {
    /// The numeraire bond's maturity `T*`.
    fn numeraire_time(&self) -> f64;

    /// The state at the anchor.
    fn initial_state(&self) -> f64 {
        0.0
    }

    /// The law of `x(t1)` given `x(t0)` under the numeraire measure:
    /// `x(t1) = decay * x(t0) + shift + std * Z`, returned as
    /// `(decay, shift, std)`.
    fn transition(&self, t0: f64, t1: f64) -> (f64, f64, f64);

    /// `P(t, maturity | x)`.
    fn zerobond(&self, t: f64, maturity: f64, x: f64) -> Result<f64, RustyQLibError>;

    /// `N(t, x) = P(t, T* | x)`.
    fn numeraire(&self, t: f64, x: f64) -> Result<f64, RustyQLibError> {
        self.zerobond(t, self.numeraire_time(), x)
    }
}

/// Hull-White (any of its term-structure forms) as a Gaussian1d model
/// under the `T*`-forward measure.
#[derive(Debug, Clone)]
pub struct HullWhite1d<'a> {
    pub model: &'a HullWhite,
    pub numeraire_time: f64,
}

impl HullWhite {
    /// The model in Gaussian1d form with `P(., numeraire_time)` as the
    /// numeraire — usually the last payment date of the product.
    pub fn gaussian1d(&self, numeraire_time: f64) -> HullWhite1d<'_> {
        HullWhite1d {
            model: self,
            numeraire_time,
        }
    }
}

impl Gaussian1dModel for HullWhite1d<'_> {
    fn numeraire_time(&self) -> f64 {
        self.numeraire_time
    }

    fn transition(&self, t0: f64, t1: f64) -> (f64, f64, f64) {
        let m = self.model;
        // under the T*-forward measure the OU state x = r - alpha drifts
        // by -int sigma^2 e^{-K(u,t1)} B(u,T*) du
        //   = -[M(t0,t1) + B(t1,T*) V(t0,t1)]
        let variance = m.short_rate_variance(t0, t1);
        let shift =
            -(m.forward_measure_shift(t0, t1) + m.b_factor(t1, self.numeraire_time) * variance);
        (m.decay(t0, t1), shift, variance.sqrt())
    }

    fn zerobond(&self, t: f64, maturity: f64, x: f64) -> Result<f64, RustyQLibError> {
        self.model.zero_bond(t, maturity, x + self.model.alpha(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::core::utils::norm_pdf;
    use chrono::NaiveDate;

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
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    #[test]
    fn deflated_bonds_are_martingales_under_the_numeraire_measure() {
        // P(0,S)/P(0,T*) = E^{T*}[P(T,S,x)/P(T,T*,x)], the measure
        // change's drift being exactly what makes it so — for constant
        // and piecewise coefficients alike
        let curve = market_curve();
        let models = [
            HullWhite::new(0.1, 0.01, curve.clone()).unwrap(),
            HullWhite::generalized(
                &[2.0],
                &[0.03, 0.12],
                &[1.0],
                &[0.012, 0.008],
                curve.clone(),
            )
            .unwrap(),
        ];
        for m in &models {
            let g = m.gaussian1d(10.0);
            for (t, s) in [(1.0, 3.0), (4.0, 7.0), (2.0, 10.0)] {
                let (decay, shift, std) = g.transition(0.0, t);
                let mean = decay * g.initial_state() + shift;
                let n = 801;
                let dz = 16.0 / (n - 1) as f64;
                let mut expectation = 0.0;
                for k in 0..n {
                    let z = -8.0 + k as f64 * dz;
                    let simpson = if k == 0 || k == n - 1 {
                        1.0
                    } else if k % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    };
                    let x = mean + std * z;
                    let deflated = g.zerobond(t, s, x).unwrap() / g.numeraire(t, x).unwrap();
                    expectation += simpson * dz / 3.0 * norm_pdf(z) * deflated;
                }
                let forward = curve.df(s) / curve.df(10.0);
                assert!(
                    (expectation / forward - 1.0).abs() < 1e-6,
                    "{t}->{s}: {expectation} vs {forward}"
                );
            }
        }
    }
}
