//! De-jumping listed implied volatilities for the jump-to-default
//! model.
//!
//! An option-implied volatility is a *risky* vol: the option price it
//! quotes already carries the stock's jump to zero on default. The
//! jump-to-default model's `sigma` is the diffusion *conditional on
//! survival*, so feeding an implied vol in double counts the default.
//! Under the model a European call is Black-Scholes on the
//! survival-conditional forward with the risky discount:
//!
//! ```text
//! C = S exp(-(q + b) T) N(d1) - K exp(-(r + lambda) T) N(d2)
//! d1 = [ln(S / K) + (r + lambda - q - b + sigma^2 / 2) T] / (sigma sqrt T)
//! ```
//!
//! so the de-jumped vol is the `sigma` at which this reproduces the
//! market call priced at the implied vol. It is always lower, by an
//! amount that grows with the hazard and the maturity, and a flat
//! implied vol de-jumps to an upward-sloping one in strike: the model
//! generates the downside skew on its own. When the jump alone is
//! worth more than the option, no diffusion is consistent with the
//! quote and the solve fails — the hazard and that vol disagree. That
//! happens first for deep out-of-the-money puts (equivalently deep
//! in-the-money calls, by put-call parity, which the model preserves):
//! the default leg alone is worth `K exp(-rT) (1 - exp(-lambda T))`,
//! and a quote below that cannot carry the hazard. A listed surface on
//! a credit-risky name holds that value in its put skew; a flat test
//! surface does not, so sample it where the two are consistent.
//!
//! [`dejump_surface`] samples a whole surface on a strike x expiry grid,
//! for the Dupire local vol
//! ([`local_vol_grid`](super::ConvertiblePricing::local_vol_grid)) under
//! jump to default.

use crate::core::curves::{Compounding, Tenor, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::utils::norm_cdf;
use crate::core::vols::VolSurface;

const MIN_VOL: f64 = 1e-4;

/// Undiscounted-forward Black call `df * (F N(d1) - K N(d2))`.
fn black_call(forward: f64, strike: f64, t: f64, sigma: f64, df: f64) -> f64 {
    let sd = sigma * t.sqrt();
    if sd <= 0.0 {
        return df * (forward - strike).max(0.0);
    }
    let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
    df * (forward * norm_cdf(d1) - strike * norm_cdf(d1 - sd))
}

/// A European call under jump to default with total loss: Black on the
/// survival-conditional forward, discounted at `r + lambda`.
fn jump_to_default_call(
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    carry: f64,
    hazard: f64,
    sigma: f64,
) -> f64 {
    let df = (-(rate + hazard) * t).exp();
    let forward = spot * ((rate + hazard - carry) * t).exp();
    black_call(forward, strike, t, sigma, df)
}

/// The diffusive volatility conditional on survival that, under jump
/// to default with `hazard`, reproduces a call quoted at `implied_vol`
/// on the ordinary forward `S exp((r - q - b) t)`. `rate` is the
/// continuous zero rate to `t`; `dividend_yield` and `borrow_cost` are
/// the carry on both sides, so only the hazard separates them.
#[allow(clippy::too_many_arguments)]
pub fn dejump_implied_vol(
    implied_vol: f64,
    spot: f64,
    strike: f64,
    t: f64,
    rate: f64,
    dividend_yield: f64,
    borrow_cost: f64,
    hazard: f64,
) -> Result<f64, RustyQLibError> {
    for (name, value) in [
        ("spot", spot),
        ("strike", strike),
        ("t", t),
        ("implied vol", implied_vol),
    ] {
        if !(value > 0.0 && value.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "dejump",
                format!("{name} must be positive, got {value}"),
            ));
        }
    }
    if !(hazard >= 0.0 && hazard.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "dejump",
            format!("hazard must be non-negative, got {hazard}"),
        ));
    }
    if hazard == 0.0 {
        return Ok(implied_vol);
    }
    let carry = dividend_yield + borrow_cost;
    let target = black_call(
        spot * ((rate - carry) * t).exp(),
        strike,
        t,
        implied_vol,
        (-rate * t).exp(),
    );
    let model = |sigma: f64| jump_to_default_call(spot, strike, t, rate, carry, hazard, sigma);
    if model(MIN_VOL) > target {
        return Err(RustyQLibError::CalibrationFailed {
            iterations: 0,
            residual: model(MIN_VOL) - target,
            reason: format!(
                "the jump alone is worth more than the option: no diffusion reproduces an \
                 implied vol of {implied_vol} at strike {strike}, {t:.3}y with a hazard of {hazard}"
            ),
        });
    }
    // the call rises with sigma; the de-jumped vol sits in (0, implied]
    let (mut lo, mut hi) = (MIN_VOL, implied_vol);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if model(mid) > target {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo < 1e-12 {
            break;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// `surface` de-jumped for a flat hazard, sampled on an absolute strike
/// x expiry grid (expiries as year fractions on the surface's axis; the
/// curve is read at the same times, so it should share the surface's
/// reference date). Every point must de-jump; a point where the jump
/// exceeds the option value fails the whole surface, naming it.
#[allow(clippy::too_many_arguments)]
pub fn dejump_surface(
    surface: &VolSurface,
    curve: &YieldCurve,
    spot: f64,
    dividend_yield: f64,
    borrow_cost: f64,
    hazard: f64,
    strikes: &[f64],
    expiries: &[f64],
) -> Result<VolSurface, RustyQLibError> {
    dejump_surface_with(
        surface,
        curve,
        spot,
        dividend_yield,
        borrow_cost,
        |_| hazard,
        strikes,
        expiries,
    )
}

/// [`dejump_surface`] with the hazard given per expiry: for a term
/// structure, the average hazard to each expiry, which has the same
/// survival as the flat rate the call formula assumes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dejump_surface_with(
    surface: &VolSurface,
    curve: &YieldCurve,
    spot: f64,
    dividend_yield: f64,
    borrow_cost: f64,
    hazard_to: impl Fn(f64) -> f64,
    strikes: &[f64],
    expiries: &[f64],
) -> Result<VolSurface, RustyQLibError> {
    let mut rows = Vec::with_capacity(expiries.len());
    for &t in expiries {
        let rate = curve.zero_rate_with(t, Compounding::Continuous);
        let forward = spot * ((rate - dividend_yield - borrow_cost) * t).exp();
        let hazard = hazard_to(t);
        let row = strikes
            .iter()
            .map(|&strike| {
                let implied = surface.vol(strike, forward, t);
                dejump_implied_vol(
                    implied,
                    spot,
                    strike,
                    t,
                    rate,
                    dividend_yield,
                    borrow_cost,
                    hazard,
                )
            })
            .collect::<Result<Vec<f64>, _>>()?;
        rows.push(row);
    }
    let tenors: Vec<Tenor> = expiries.iter().map(|&t| Tenor::YearFraction(t)).collect();
    VolSurface::from_strike_grid(
        &tenors,
        strikes,
        &rows,
        surface.reference_date(),
        surface.day_count(),
    )
    .map_err(|e| RustyQLibError::invalid_input("dejump", format!("{e:?}")))
}
