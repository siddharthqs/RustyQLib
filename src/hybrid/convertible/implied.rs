//! The implied-parameter solves: the credit spread, the hazard rate,
//! and the volatility that reproduce a quoted clean price, plus the
//! central-difference spot delta.

use super::models::ConvertibleMarket;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// Central difference of `price` in the spot from a symmetric 1% bump.
pub(crate) fn central_spot_delta(
    spot: f64,
    price: impl Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    let bump = 0.01 * spot;
    Ok((price(spot + bump)? - price(spot - bump)?) / (2.0 * bump))
}

/// The credit spread at which `price` reproduces `target`, holding the
/// other market inputs fixed. Shared by the convertible bond and the
/// convertible preferred.
pub(crate) fn solve_implied_credit_spread(
    target: f64,
    market: &ConvertibleMarket,
    price: impl Fn(&ConvertibleMarket) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    // price is decreasing in the spread (only the cash part reacts)
    solve_implied_parameter(target, -0.2, 3.0, "credit spread", |credit_spread| {
        price(&ConvertibleMarket {
            credit_spread,
            ..*market
        })
    })
}

/// The value in `[lo, hi]` of a parameter in which `price` is
/// decreasing at which it reproduces `target`, by bisection. Both
/// bracket ends are priced first so an invalid tree surfaces as an
/// error rather than inside the solve.
fn solve_implied_parameter(
    target: f64,
    lo: f64,
    hi: f64,
    what: &str,
    price: impl Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    price(lo)?;
    price(hi)?;
    // target - price(x) is increasing; the tree stays valid inside a
    // bracket whose ends are valid (the risk-neutral probability is
    // monotone in the parameter)
    let objective = |x: f64| target - price(x).expect("the bracket ends were priced successfully");
    let root = Solver1d::new(1e-8, 100).bisection(objective, lo, hi)?;
    if !root.converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations: root.iterations,
            residual: objective(root.x).abs(),
            reason: format!("implied {what} solve did not converge"),
        });
    }
    Ok(root.x)
}

/// The hazard rate on the decreasing branch of `price` at which it
/// reproduces `target`: the bracket is grown geometrically from zero
/// until the price crosses the target, and the search stops (with an
/// error) as soon as the price turns upward.
pub(crate) fn solve_implied_hazard_rate(
    target: f64,
    price: impl Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    const MAX_HAZARD: f64 = 3.0;
    let mut lo = 0.0;
    let mut price_lo = price(lo)?;
    if target >= price_lo {
        if target - price_lo < 1e-8 {
            return Ok(0.0);
        }
        return Err(RustyQLibError::CalibrationFailed {
            iterations: 1,
            residual: target - price_lo,
            reason: format!(
                "clean price {target} is above the zero-hazard value {price_lo}; \
                 no non-negative hazard rate reproduces it"
            ),
        });
    }
    let mut hi = 0.01;
    let mut evaluations = 1;
    loop {
        let price_hi = price(hi)?;
        evaluations += 1;
        if price_hi <= target {
            break;
        }
        if price_hi >= price_lo || hi >= MAX_HAZARD {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: evaluations,
                residual: price_hi - target,
                reason: format!(
                    "clean price {target} is below the model's reach: the value bottoms \
                     out near {price_lo} at a hazard rate of {lo}, beyond which the \
                     recovery leg outweighs the survival claims"
                ),
            });
        }
        lo = hi;
        price_lo = price_hi;
        hi = (hi * 2.0).min(MAX_HAZARD);
    }
    solve_implied_parameter(target, lo, hi, "hazard rate", price)
}

/// The volatility in `[0.02, 3.0]` at which `price` reproduces
/// `target`. The price rises with the volatility, so the shared
/// decreasing-parameter solve is run on the negated price.
pub(crate) fn solve_implied_volatility(
    target: f64,
    price: impl Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    solve_implied_parameter(-target, 0.02, 3.0, "volatility", |volatility| {
        price(volatility).map(|p| -p)
    })
}
