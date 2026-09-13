//! Monte Carlo under Hull-White with **exact** steps (QuantLib's
//! `MCHullWhiteCapFloorEngine`, without its discretization bias).
//!
//! The state `x = r - alpha(t)` and its time integral are jointly
//! Gaussian over any step: given `x(t)`,
//!
//! ```text
//! x(T)             ~ N( x(t) e^{-K(t,T)},  V(t,T) )
//! int_t^T x(u) du  ~ N( x(t) B(t,T),      W(t,T) ),   Cov = M(t,T)
//! ```
//!
//! with `V`, `W`, `M` the model's closed-form integrals, and
//! `int alpha du` deterministic. So a path steps straight from one
//! event date to the next — one bivariate draw per step — and the bank
//! account `exp(-int r)` is exact along it: no time grid between the
//! fixings, no bias, and the discount factor reproduces the curve to
//! Monte Carlo error alone. Any payoff written on the short rates at
//! the event dates and the discount factors to them prices through
//! [`simulate`]; [`cap_floor`] is the caplet strip, each caplet's
//! payment discounted with the conditional expectation
//! `D(t_i) P(t_i, t_pay | r_i)` rather than a further draw, which
//! removes the payment-date noise.

use rand::{Rng, SeedableRng};

use crate::core::errors::RustyQLibError;
use crate::rates::contracts::cap_floor::CapFloor;
use crate::rates::contracts::swaption::year_fraction_from;
use crate::rates::models::{HullWhite, ShortRateModel};

const FIELD: &str = "mc hull-white";

/// Path count and randomness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct McConfig {
    /// Paths (antithetic pairs count as two).
    pub paths: usize,
    pub seed: u64,
    /// Pair every path with its mirror image in the normal draws.
    pub antithetic: bool,
}

impl Default for McConfig {
    fn default() -> Self {
        McConfig {
            paths: 20_000,
            seed: 42,
            antithetic: true,
        }
    }
}

/// A Monte Carlo estimate and its standard error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct McResult {
    pub value: f64,
    pub std_error: f64,
}

/// One exact step's moments.
struct Step {
    decay: f64,
    b: f64,
    std_x: f64,
    /// `Cov / std_x`, the loading of the integral on the state draw.
    cov_load: f64,
    /// The integral's residual std after that loading.
    std_i: f64,
    alpha_int: f64,
    alpha_end: f64,
}

/// Simulate `payoff(rates, discount_factors)` — the short rate at each
/// of `times` and the exact discount factor `exp(-int_0^t r)` to each —
/// under `model`, and return the discounted mean with its standard
/// error. `times` ascending and positive.
pub fn simulate(
    model: &HullWhite,
    times: &[f64],
    config: &McConfig,
    payoff: impl Fn(&[f64], &[f64]) -> f64,
) -> Result<McResult, RustyQLibError> {
    if times.is_empty() || times[0] <= 0.0 || times.windows(2).any(|w| w[1] <= w[0]) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "event times must be positive and strictly ascending",
        ));
    }
    if config.paths < 2 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "need at least two paths",
        ));
    }
    let steps: Vec<Step> = times
        .iter()
        .enumerate()
        .map(|(k, &t)| {
            let t0 = if k == 0 { 0.0 } else { times[k - 1] };
            let var_x = model.short_rate_variance(t0, t);
            let var_i = model.integrated_variance(t0, t);
            let cov = model.forward_measure_shift(t0, t);
            let std_x = var_x.sqrt();
            let cov_load = if std_x > 0.0 { cov / std_x } else { 0.0 };
            Step {
                decay: model.decay(t0, t),
                b: model.b_factor(t0, t),
                std_x,
                cov_load,
                std_i: (var_i - cov_load * cov_load).max(0.0).sqrt(),
                alpha_int: model.integrated_alpha(t0, t),
                alpha_end: model.alpha(t),
            }
        })
        .collect();

    let mut rng = rand_pcg::Pcg64::seed_from_u64(config.seed);
    let n = times.len();
    let mut rates = vec![0.0; n];
    let mut dfs = vec![0.0; n];
    let mut run = |draws: &[(f64, f64)]| -> f64 {
        let (mut x, mut integral) = (0.0_f64, 0.0_f64);
        for (k, step) in steps.iter().enumerate() {
            let (z1, z2) = draws[k];
            let x_next = x * step.decay + step.std_x * z1;
            integral += x * step.b + step.cov_load * z1 + step.std_i * z2 + step.alpha_int;
            x = x_next;
            rates[k] = x + step.alpha_end;
            dfs[k] = (-integral).exp();
        }
        payoff(&rates, &dfs)
    };
    let samples = if config.antithetic {
        config.paths.div_ceil(2)
    } else {
        config.paths
    };
    let mut draws = vec![(0.0, 0.0); n];
    let (mut sum, mut sum_sq) = (0.0, 0.0);
    for _ in 0..samples {
        for d in draws.iter_mut() {
            *d = (
                rng.sample(rand_distr::StandardNormal),
                rng.sample(rand_distr::StandardNormal),
            );
        }
        let value = if config.antithetic {
            let up = run(&draws);
            let mirrored: Vec<(f64, f64)> = draws.iter().map(|&(a, b)| (-a, -b)).collect();
            0.5 * (up + run(&mirrored))
        } else {
            run(&draws)
        };
        sum += value;
        sum_sq += value * value;
    }
    let mean = sum / samples as f64;
    let variance = (sum_sq / samples as f64 - mean * mean).max(0.0);
    Ok(McResult {
        value: mean,
        std_error: (variance / samples as f64).sqrt(),
    })
}

/// A cap or floor by simulation: the strip's unfixed caplets, each
/// paying `notional * tau * (L - K)^+` (or the floorlet) on the rate
/// `L` read from the model's own zero bond at the fixing, discounted
/// by the exact bank account to the fixing and the conditional bond
/// price to the payment.
pub fn cap_floor(
    model: &HullWhite,
    product: &CapFloor,
    config: &McConfig,
) -> Result<McResult, RustyQLibError> {
    let anchor = model.curve();
    let valuation = anchor.reference_date();
    let mut fixings = Vec::new();
    let mut legs = Vec::new(); // (fixing, end, payment, tau)
    for p in product.periods()? {
        if p.start <= valuation {
            continue;
        }
        let fixing = year_fraction_from(anchor, p.start);
        fixings.push(fixing);
        legs.push((
            fixing,
            year_fraction_from(anchor, p.end),
            year_fraction_from(anchor, p.payment),
            product.day_count.year_fraction(p.start, p.end),
        ));
    }
    if fixings.is_empty() {
        return Ok(McResult {
            value: 0.0,
            std_error: 0.0,
        });
    }
    let (notional, strike, side) = (product.notional, product.strike, product.cap_or_floor);
    simulate(model, &fixings, config, |rates, dfs| {
        let mut total = 0.0;
        for (i, &(fixing, end, payment, tau)) in legs.iter().enumerate() {
            let r = rates[i];
            let p_end = model.zero_bond(fixing, end, r).unwrap_or(f64::NAN);
            let p_pay = model.zero_bond(fixing, payment, r).unwrap_or(f64::NAN);
            let libor = (1.0 / p_end - 1.0) / tau;
            total += notional * tau * side.intrinsic(libor, strike) * p_pay * dfs[i];
        }
        total
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::contracts::cap_floor::CapOrFloor;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
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
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    #[test]
    fn exact_steps_reprice_the_curve_without_bias() {
        // one step to 5y and four coarse steps: both discount to the
        // market df within Monte Carlo error, because every step is exact
        for m in [
            HullWhite::new(0.05, 0.011, market_curve()).unwrap(),
            HullWhite::generalized(
                &[2.0],
                &[0.03, 0.12],
                &[1.0],
                &[0.012, 0.008],
                market_curve(),
            )
            .unwrap(),
        ] {
            for times in [vec![5.0], vec![0.5, 2.0, 3.5, 5.0]] {
                let config = McConfig {
                    paths: 40_000,
                    ..McConfig::default()
                };
                let r = simulate(&m, &times, &config, |_, dfs| *dfs.last().unwrap()).unwrap();
                let df = m.curve().df(5.0);
                assert!(
                    (r.value - df).abs() < 3.0 * r.std_error.max(1e-6)
                        && (r.value / df - 1.0).abs() < 2e-3,
                    "{times:?}: MC {} +- {} vs {df}",
                    r.value,
                    r.std_error
                );
            }
        }
    }

    #[test]
    fn caps_and_floors_match_the_analytic_strip() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let config = McConfig {
            paths: 60_000,
            ..McConfig::default()
        };
        for (strike, side) in [
            (0.045, CapOrFloor::Cap),
            (0.045, CapOrFloor::Floor),
            (0.06, CapOrFloor::Cap),
        ] {
            let product =
                CapFloor::usd_standard(10_000_000.0, strike, side, d(2027, 8, 16), d(2030, 8, 16))
                    .unwrap();
            let analytic = product.npv_hull_white(&m).unwrap();
            let mc = product.npv_mc_hull_white(&m, &config).unwrap();
            assert!(
                (mc.value - analytic).abs() < 3.0 * mc.std_error
                    && (mc.value / analytic - 1.0).abs() < 0.01,
                "{side:?} K={strike}: MC {} +- {} vs {analytic}",
                mc.value,
                mc.std_error
            );
        }
        // an already-expired strip is worth nothing
        let expired =
            CapFloor::usd_standard(1.0, 0.04, CapOrFloor::Cap, d(2020, 8, 16), d(2022, 8, 16))
                .unwrap();
        assert_eq!(expired.npv_mc_hull_white(&m, &config).unwrap().value, 0.0);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let config = McConfig::default();
        assert!(simulate(&m, &[], &config, |_, _| 0.0).is_err());
        assert!(simulate(&m, &[2.0, 1.0], &config, |_, _| 0.0).is_err());
        assert!(simulate(&m, &[0.0], &config, |_, _| 0.0).is_err());
        let one = McConfig {
            paths: 1,
            ..McConfig::default()
        };
        assert!(simulate(&m, &[1.0], &one, |_, _| 0.0).is_err());
    }
}
