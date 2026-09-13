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
use crate::rates::engines::jamshidian::european_swaption_settled;
use crate::rates::models::HullWhite;
use crate::rates::PayerReceiver;

/// One European swaption quote, on unit notional.
#[derive(Debug, Clone)]
pub struct SwaptionQuote {
    /// Option expiry, in years from the curve anchor.
    pub expiry: f64,
    /// Start of the underlying swap: the expiry itself, or a settlement
    /// lag after it.
    pub swap_start: f64,
    /// Fixed leg `(payment_time, accrual)` pairs.
    pub fixed_leg: Vec<(f64, f64)>,
    /// Fixed rate of the underlying swap.
    pub strike_rate: f64,
    /// Market price per unit notional.
    pub market_price: f64,
    pub payer_receiver: PayerReceiver,
}

/// The forward par rate of the swap starting at `swap_start` with this
/// fixed leg — the natural ATM strike of a quote.
pub fn atm_swap_rate(
    curve: &YieldCurve,
    swap_start: f64,
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
    Ok((curve.df(swap_start) - curve.df(last.0)) / annuity)
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
        let price = european_swaption_settled(
            &model,
            quote.expiry,
            quote.swap_start,
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

/// Bootstrap a **piecewise-constant** `sigma` to a column of swaption
/// quotes with the mean reversion fixed at `a`: one interval per
/// distinct expiry, `sigma_i` on `[expiry_{i-1}, expiry_i)` solved so
/// the quotes expiring at `expiry_i` reprice given the earlier
/// intervals (a swaption sees only the variance up to its expiry, so
/// the solve is exactly sequential). One quote per expiry is hit to
/// tolerance by bisection; several quotes at one expiry (a smile or
/// several tenors) are fitted in the least-squares sense. The last
/// interval's sigma extends beyond the last expiry.
///
/// This is how a Bermudan or callable desk fits Hull-White: the
/// coterminal (or a column of) swaptions, each expiry pinning its own
/// sigma, with `a` chosen separately.
pub fn calibrate_hull_white_piecewise(
    curve: &YieldCurve,
    quotes: &[SwaptionQuote],
    a: f64,
) -> Result<HullWhiteFit, RustyQLibError> {
    validate_quotes(quotes)?;
    if a <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "calibration",
            format!("mean reversion must be positive, got {a}"),
        ));
    }
    let mut order: Vec<usize> = (0..quotes.len()).collect();
    order.sort_by(|&i, &j| quotes[i].expiry.total_cmp(&quotes[j].expiry));
    // group by expiry (within a day)
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for &i in &order {
        match groups.last_mut() {
            Some((expiry, members)) if (quotes[i].expiry - *expiry).abs() < 1.0 / 365.0 => {
                members.push(i)
            }
            _ => groups.push((quotes[i].expiry, vec![i])),
        }
    }
    let mut times: Vec<f64> = Vec::new();
    let mut sigmas: Vec<f64> = Vec::new();
    let mut total_iterations = 0;
    for (index, (expiry, members)) in groups.iter().enumerate() {
        // the sigma on the interval ending at this expiry: extend the
        // fitted structure with a trial value and price this expiry's
        // quotes; earlier intervals are already fixed
        let priced_error = |sigma: f64| -> Result<f64, RustyQLibError> {
            let mut trial = sigmas.clone();
            trial.push(sigma);
            let model = HullWhite::with_piecewise_sigma(a, &times, &trial, curve.clone())?;
            let mut sum = 0.0;
            for &i in members {
                let quote = &quotes[i];
                let price = european_swaption_settled(
                    &model,
                    quote.expiry,
                    quote.swap_start,
                    &quote.fixed_leg,
                    quote.strike_rate,
                    1.0,
                    quote.payer_receiver,
                )?;
                let error = (price - quote.market_price) / quote.market_price;
                sum += if members.len() == 1 {
                    error
                } else {
                    error * error
                };
            }
            Ok(sum)
        };
        // a 10% short-rate vol is far beyond any market; wider brackets
        // degenerate the bond prices inside the Jamshidian solve
        let (lo, hi) = (1e-8, 0.1);
        let sigma = if members.len() == 1 {
            // price is increasing in this interval's sigma: bisect
            let f_lo = priced_error(lo)?;
            let f_hi = priced_error(hi)?;
            if f_lo > 0.0 || f_hi < 0.0 {
                return Err(RustyQLibError::CalibrationFailed {
                    iterations: total_iterations,
                    residual: f_lo.abs().min(f_hi.abs()),
                    reason: format!(
                        "the quote expiring at {expiry} cannot be reached with a sigma in \
                         [{lo}, {hi}] given the earlier intervals"
                    ),
                });
            }
            let root = crate::core::solvers::Solver1d::new(1e-12, 200).bisection(
                |s| priced_error(s).unwrap_or(f64::NAN),
                lo,
                hi,
            )?;
            total_iterations += root.iterations;
            root.x
        } else {
            let f = |x: &[f64]| priced_error(x[0].exp()).unwrap_or(1e10);
            let start = sigmas.last().copied().unwrap_or(0.01).max(1e-4);
            let problem = Problem::scalar(&f, vec![start.ln()]);
            let result = minimize(&OptimConfig::new(1e-14, 400), Method::NelderMead, &problem)?;
            if !result.converged {
                return Err(RustyQLibError::CalibrationFailed {
                    iterations: result.iterations,
                    residual: result.value,
                    reason: format!("the fit at expiry {expiry} did not converge"),
                });
            }
            total_iterations += result.iterations;
            result.x[0].exp()
        };
        sigmas.push(sigma);
        if index + 1 < groups.len() {
            times.push(*expiry);
        }
    }
    let model = HullWhite::with_piecewise_sigma(a, &times, &sigmas, curve.clone())?;
    // report the fit over all quotes
    let mut sum = 0.0;
    for quote in quotes {
        let price = european_swaption_settled(
            &model,
            quote.expiry,
            quote.swap_start,
            &quote.fixed_leg,
            quote.strike_rate,
            1.0,
            quote.payer_receiver,
        )?;
        let error = (price - quote.market_price) / quote.market_price;
        sum += error * error;
    }
    Ok(HullWhiteFit {
        model,
        price_rmse: (sum / quotes.len() as f64).sqrt(),
        iterations: total_iterations,
        converged: true,
    })
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
    use crate::rates::engines::jamshidian::european_swaption;
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
                    swap_start: expiry,
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
            (fit.model.sigma() - 0.012).abs() / 0.012 < 0.01,
            "sigma {}",
            fit.model.sigma()
        );
        assert!(
            (fit.model.a() - 0.08).abs() / 0.08 < 0.05,
            "a {}",
            fit.model.a()
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
            (fit.model.sigma() - 0.012).abs() < 2e-5,
            "sigma {}",
            fit.model.sigma()
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
    fn piecewise_bootstrap_recovers_a_generating_term_structure() {
        // a column of coterminal swaptions (1y, 2y, 3y into 5y) priced by
        // a piecewise model, then bootstrapped back: every expiry
        // reprices to tolerance and every interval's sigma is recovered
        let curve = market_curve();
        let truth = HullWhite::with_piecewise_sigma(
            0.05,
            &[1.0, 2.0],
            &[0.012, 0.009, 0.011],
            curve.clone(),
        )
        .unwrap();
        let quotes: Vec<SwaptionQuote> = [1.0, 2.0, 3.0]
            .iter()
            .map(|&expiry| {
                let fixed_leg: Vec<(f64, f64)> =
                    (1..=5).map(|i| (expiry + i as f64, 1.0)).collect();
                let strike = atm_swap_rate(&curve, expiry, &fixed_leg).unwrap();
                let price = european_swaption(
                    &truth,
                    expiry,
                    &fixed_leg,
                    strike,
                    1.0,
                    PayerReceiver::Payer,
                )
                .unwrap();
                SwaptionQuote {
                    expiry,
                    swap_start: expiry,
                    fixed_leg,
                    strike_rate: strike,
                    market_price: price,
                    payer_receiver: PayerReceiver::Payer,
                }
            })
            .collect();
        let fit = calibrate_hull_white_piecewise(&curve, &quotes, 0.05).unwrap();
        assert!(fit.price_rmse < 1e-9, "rmse {}", fit.price_rmse);
        assert_eq!(fit.model.sigma_times(), &[1.0, 2.0]);
        for (got, want) in fit.model.sigmas().iter().zip([0.012, 0.009, 0.011]) {
            assert!((got - want).abs() < 1e-7, "{got} vs {want}");
        }
        // a constant-sigma fit to the same column cannot hit all three
        let flat = calibrate_hull_white_sigma(&curve, &quotes, 0.05, 0.01).unwrap();
        assert!(flat.price_rmse > 1e-3, "flat rmse {}", flat.price_rmse);
        // quote order does not matter
        let mut shuffled = quotes.clone();
        shuffled.reverse();
        let again = calibrate_hull_white_piecewise(&curve, &shuffled, 0.05).unwrap();
        assert_eq!(again.model.sigmas(), fit.model.sigmas());
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let curve = market_curve();
        assert!(calibrate_hull_white(&curve, &[], 0.05, 0.01).is_err());
        let bad = SwaptionQuote {
            expiry: 1.0,
            swap_start: 1.0,
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
