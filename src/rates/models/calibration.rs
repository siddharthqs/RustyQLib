//! Hull-White calibration to European swaption quotes.
//!
//! The market-standard workflow (QuantLib's swaption-helper calibration):
//! given a set of swaption prices — typically ATM across a grid of
//! expiries and tenors — find the `(a, sigma)` that minimizes the sum of
//! squared **relative** price errors, pricing each quote with the
//! model's own analytic Jamshidian swaption formula. Optimization runs
//! over `(ln a, ln sigma)` with Nelder-Mead, so positivity is built in
//! and no gradients are needed.
//!
//! Desks often fix the mean reversion (it is weakly identified by
//! coterminal swaptions) and calibrate `sigma` alone —
//! [`calibrate_hull_white_sigma`] does exactly that.

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::{minimize, Method, OptimConfig, Problem};
use crate::rates::models::pricers::european_swaption;
use crate::rates::models::HullWhite;
use crate::rates::PayerReceiver;

/// One European swaption quote, on unit notional.
#[derive(Debug, Clone)]
pub struct SwaptionQuote {
    /// Option expiry (= swap start), in years from the curve anchor.
    pub expiry: f64,
    /// Fixed leg `(payment_time, accrual)` pairs.
    pub fixed_leg: Vec<(f64, f64)>,
    /// Fixed rate of the underlying swap.
    pub strike_rate: f64,
    /// Market price per unit notional.
    pub market_price: f64,
    pub payer_receiver: PayerReceiver,
}

/// The forward par rate of the swap underlying a quote — the natural
/// ATM strike.
pub fn atm_swap_rate(
    curve: &YieldCurve,
    expiry: f64,
    fixed_leg: &[(f64, f64)],
) -> Result<f64, RustyQLibError> {
    let last = fixed_leg.last().ok_or_else(|| {
        RustyQLibError::invalid_input("atm_swap_rate", "the fixed leg has no payments")
    })?;
    let annuity: f64 = fixed_leg
        .iter()
        .map(|&(time, tau)| tau * curve.df(time))
        .sum();
    if annuity <= 0.0 {
        return Err(RustyQLibError::NumericalError(format!(
            "non-positive annuity {annuity}"
        )));
    }
    Ok((curve.df(expiry) - curve.df(last.0)) / annuity)
}

/// The result of a calibration.
#[derive(Debug, Clone)]
pub struct HullWhiteFit {
    pub model: HullWhite,
    /// Root-mean-square relative price error over the quotes.
    pub price_rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

fn validate_quotes(quotes: &[SwaptionQuote]) -> Result<(), RustyQLibError> {
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "calibration",
            "no swaption quotes",
        ));
    }
    for quote in quotes {
        if !(quote.market_price > 0.0 && quote.market_price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "calibration",
                format!(
                    "market prices must be positive, got {} at expiry {}",
                    quote.market_price, quote.expiry
                ),
            ));
        }
    }
    Ok(())
}

/// Sum of squared relative price errors for `(a, sigma)`; a large
/// penalty when a quote cannot be priced (keeps the simplex away from
/// pathological corners).
fn objective(curve: &YieldCurve, quotes: &[SwaptionQuote], a: f64, sigma: f64) -> f64 {
    let model = match HullWhite::new(a, sigma, curve.clone()) {
        Ok(model) => model,
        Err(_) => return 1e10,
    };
    let mut sum = 0.0;
    for quote in quotes {
        let price = european_swaption(
            &model,
            quote.expiry,
            &quote.fixed_leg,
            quote.strike_rate,
            1.0,
            quote.payer_receiver,
        );
        match price {
            Ok(price) => {
                let error = (price - quote.market_price) / quote.market_price;
                sum += error * error;
            }
            Err(_) => return 1e10,
        }
    }
    sum
}

fn finish_fit(
    curve: &YieldCurve,
    quotes: &[SwaptionQuote],
    a: f64,
    sigma: f64,
    value: f64,
    iterations: usize,
    converged: bool,
) -> Result<HullWhiteFit, RustyQLibError> {
    if !converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations,
            residual: value,
            reason: "Hull-White swaption calibration did not converge".to_string(),
        });
    }
    Ok(HullWhiteFit {
        model: HullWhite::new(a, sigma, curve.clone())?,
        price_rmse: (value / quotes.len() as f64).sqrt(),
        iterations,
        converged,
    })
}

/// Calibrate both `a` and `sigma` to the quotes, starting from
/// `(a0, sigma0)`.
pub fn calibrate_hull_white(
    curve: &YieldCurve,
    quotes: &[SwaptionQuote],
    a0: f64,
    sigma0: f64,
) -> Result<HullWhiteFit, RustyQLibError> {
    validate_quotes(quotes)?;
    if a0 <= 0.0 || sigma0 <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "calibration",
            format!("starting values must be positive, got a0={a0}, sigma0={sigma0}"),
        ));
    }
    let f = |x: &[f64]| objective(curve, quotes, x[0].exp(), x[1].exp());
    let problem = Problem::scalar(&f, vec![a0.ln(), sigma0.ln()]);
    let result = minimize(&OptimConfig::new(1e-14, 600), Method::NelderMead, &problem)?;
    finish_fit(
        curve,
        quotes,
        result.x[0].exp(),
        result.x[1].exp(),
        result.value,
        result.iterations,
        result.converged,
    )
}

/// Calibrate `sigma` alone with the mean reversion fixed at `a` — the
/// common desk setup.
pub fn calibrate_hull_white_sigma(
    curve: &YieldCurve,
    quotes: &[SwaptionQuote],
    a: f64,
    sigma0: f64,
) -> Result<HullWhiteFit, RustyQLibError> {
    validate_quotes(quotes)?;
    if a <= 0.0 || sigma0 <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "calibration",
            format!("parameters must be positive, got a={a}, sigma0={sigma0}"),
        ));
    }
    let f = |x: &[f64]| objective(curve, quotes, a, x[0].exp());
    let problem = Problem::scalar(&f, vec![sigma0.ln()]);
    let result = minimize(&OptimConfig::new(1e-14, 400), Method::NelderMead, &problem)?;
    finish_fit(
        curve,
        quotes,
        a,
        result.x[0].exp(),
        result.value,
        result.iterations,
        result.converged,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.041, 0.042, 0.044, 0.045, 0.046],
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    /// Annual fixed leg starting at `expiry` for `tenor` years.
    fn leg(expiry: f64, tenor: usize) -> Vec<(f64, f64)> {
        (1..=tenor).map(|i| (expiry + i as f64, 1.0)).collect()
    }

    /// ATM quotes generated from a known model — the calibration target.
    fn quotes_from(model: &HullWhite) -> Vec<SwaptionQuote> {
        let curve = model.curve();
        [(1.0, 2), (1.0, 5), (2.0, 5), (3.0, 7), (5.0, 5)]
            .iter()
            .map(|&(expiry, tenor)| {
                let fixed_leg = leg(expiry, tenor);
                let strike = atm_swap_rate(curve, expiry, &fixed_leg).unwrap();
                let price =
                    european_swaption(model, expiry, &fixed_leg, strike, 1.0, PayerReceiver::Payer)
                        .unwrap();
                SwaptionQuote {
                    expiry,
                    fixed_leg,
                    strike_rate: strike,
                    market_price: price,
                    payer_receiver: PayerReceiver::Payer,
                }
            })
            .collect()
    }

    #[test]
    fn full_calibration_recovers_the_generating_model() {
        let curve = market_curve();
        let truth = HullWhite::new(0.08, 0.012, curve.clone()).unwrap();
        let quotes = quotes_from(&truth);
        // start well away from the truth
        let fit = calibrate_hull_white(&curve, &quotes, 0.03, 0.025).unwrap();
        assert!(fit.price_rmse < 1e-6, "rmse {}", fit.price_rmse);
        assert!(
            (fit.model.sigma - 0.012).abs() / 0.012 < 0.01,
            "sigma {}",
            fit.model.sigma
        );
        assert!(
            (fit.model.a - 0.08).abs() / 0.08 < 0.05,
            "a {}",
            fit.model.a
        );
    }

    #[test]
    fn sigma_only_calibration_is_sharp() {
        let curve = market_curve();
        let truth = HullWhite::new(0.08, 0.012, curve.clone()).unwrap();
        let quotes = quotes_from(&truth);
        let fit = calibrate_hull_white_sigma(&curve, &quotes, 0.08, 0.03).unwrap();
        assert!(fit.price_rmse < 1e-6, "rmse {}", fit.price_rmse);
        assert!(
            (fit.model.sigma - 0.012).abs() < 2e-5,
            "sigma {}",
            fit.model.sigma
        );
        // wrong fixed mean reversion still fits, but not perfectly
        let biased = calibrate_hull_white_sigma(&curve, &quotes, 0.30, 0.03).unwrap();
        assert!(biased.price_rmse > fit.price_rmse);
    }

    #[test]
    fn atm_rate_matches_the_curve_forward_swap() {
        let curve = market_curve();
        let fixed_leg = leg(1.0, 5);
        let atm = atm_swap_rate(&curve, 1.0, &fixed_leg).unwrap();
        // reconstruct: fixed leg at the ATM rate must equal the float leg
        let fixed_pv: f64 = fixed_leg
            .iter()
            .map(|&(t, tau)| atm * tau * curve.df(t))
            .sum();
        let float_pv = curve.df(1.0) - curve.df(6.0);
        assert!((fixed_pv - float_pv).abs() < 1e-15);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let curve = market_curve();
        assert!(calibrate_hull_white(&curve, &[], 0.05, 0.01).is_err());
        let bad = SwaptionQuote {
            expiry: 1.0,
            fixed_leg: leg(1.0, 5),
            strike_rate: 0.04,
            market_price: -1.0,
            payer_receiver: PayerReceiver::Payer,
        };
        assert!(calibrate_hull_white(&curve, std::slice::from_ref(&bad), 0.05, 0.01).is_err());
        let ok = SwaptionQuote {
            market_price: 0.01,
            ..bad
        };
        assert!(calibrate_hull_white(&curve, std::slice::from_ref(&ok), -0.05, 0.01).is_err());
        assert!(calibrate_hull_white_sigma(&curve, &[ok], 0.05, -0.01).is_err());
    }
}
