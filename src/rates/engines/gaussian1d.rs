//! Engines on the [`Gaussian1dModel`] trait: European swaptions by
//! quadrature at expiry, Bermudan swaptions by backward induction of
//! **deflated** values on a state grid.
//!
//! With a terminal-bond numeraire `N`, every price satisfies
//! `V(t,x)/N(t,x) = E[V(T,x')/N(T,x') | x]`, so no per-step discounting
//! appears: the induction carries `V/N` between event dates with the
//! model's own transition law and multiplies by `N(0, x0)` at the end.
//! The engines never look inside the model — the same code prices on
//! Hull-White, whose numeraire is a formula, and on the Markov
//! functional model, whose numeraire is a calibrated table.

use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::utils::norm_pdf;
use crate::rates::engines::hw_grid::GridConfig;
use crate::rates::models::black_karasinski::TailSwap;
use crate::rates::models::gaussian1d::Gaussian1dModel;
use crate::rates::PayerReceiver;

const FIELD: &str = "gaussian1d";
/// Nodes and half-width (in stds) of the expiry quadrature.
const X_NODES: usize = 801;
const X_SPAN: f64 = 8.0;

/// The signed swap value at `(t, x)` of a tail (payer sign).
fn tail_value(
    model: &impl Gaussian1dModel,
    tail: &TailSwap,
    notional: f64,
    x: f64,
) -> Result<f64, RustyQLibError> {
    let t = tail.expiry;
    let mut fixed = 0.0;
    for &(pay, amount) in &tail.coupons {
        fixed += amount * model.zerobond(t, pay, x)?;
    }
    Ok(notional * (model.zerobond(t, tail.start, x)? - model.zerobond(t, tail.last, x)?) - fixed)
}

fn side_sign(payer_receiver: PayerReceiver) -> f64 {
    match payer_receiver {
        PayerReceiver::Payer => 1.0,
        PayerReceiver::Receiver => -1.0,
    }
}

/// Simpson over `[lo, hi]` of `f(x) phi((x - mean)/std)/std`.
fn integrate(
    lo: f64,
    hi: f64,
    mean: f64,
    std: f64,
    f: &dyn Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    if hi <= lo {
        return Ok(0.0);
    }
    let dx = (hi - lo) / (X_NODES - 1) as f64;
    let mut total = 0.0;
    for k in 0..X_NODES {
        let x = lo + k as f64 * dx;
        let simpson = if k == 0 || k == X_NODES - 1 {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += simpson * dx / 3.0 * norm_pdf((x - mean) / std) / std * f(x)?;
    }
    Ok(total)
}

/// European swaption on a swap starting at `swap_start >= expiry`: the
/// fixed leg pays `notional * strike_rate * tau` at each
/// `(payment_time, tau)` and the notional at the last payment. The
/// exercise boundary in `x` is located by bisection (the swap value is
/// monotone in the state) and the exercise side integrated.
pub fn european_swaption(
    model: &impl Gaussian1dModel,
    expiry: f64,
    swap_start: f64,
    fixed_leg: &[(f64, f64)],
    strike_rate: f64,
    notional: f64,
    payer_receiver: PayerReceiver,
) -> Result<f64, RustyQLibError> {
    if !(expiry > 0.0 && swap_start >= expiry && notional > 0.0 && strike_rate > 0.0) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "need expiry > 0, swap start >= expiry, positive notional and strike",
        ));
    }
    let Some(&(last, _)) = fixed_leg.last() else {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "the fixed leg has no payments",
        ));
    };
    let tail = TailSwap {
        expiry,
        start: swap_start,
        coupons: fixed_leg
            .iter()
            .map(|&(t, tau)| (t, notional * strike_rate * tau))
            .collect(),
        last,
    };
    let sign = side_sign(payer_receiver);
    let x0 = model.initial_state();
    let (decay, shift, std) = model.transition(0.0, expiry);
    let mean = decay * x0 + shift;
    let signed = |x: f64| -> Result<f64, RustyQLibError> {
        Ok(sign * tail_value(model, &tail, notional, x)?)
    };
    let (lo, hi) = (mean - X_SPAN * std, mean + X_SPAN * std);
    if std <= 0.0 {
        return Ok(
            model.numeraire(0.0, x0)? * signed(mean)?.max(0.0) / model.numeraire(expiry, mean)?
        );
    }
    // the payer value rises with x (bonds fall), the receiver falls
    let (v_lo, v_hi) = (signed(lo)?, signed(hi)?);
    let root = if v_lo.signum() == v_hi.signum() {
        None
    } else {
        Some(
            Solver1d::new(1e-14, 200)
                .bisection(|x| signed(x).unwrap_or(f64::NAN), lo, hi)?
                .x,
        )
    };
    let deflated = |x: f64| -> Result<f64, RustyQLibError> {
        Ok(signed(x)?.max(0.0) / model.numeraire(expiry, x)?)
    };
    let integral = match root {
        None => {
            if v_lo > 0.0 {
                integrate(lo, hi, mean, std, &deflated)?
            } else {
                0.0
            }
        }
        Some(x_star) => {
            if v_hi > 0.0 {
                integrate(x_star, hi, mean, std, &deflated)?
            } else {
                integrate(lo, x_star, mean, std, &deflated)?
            }
        }
    };
    Ok(model.numeraire(0.0, x0)? * integral)
}

/// Backward induction of deflated values on a state grid between
/// `times`. `at_event(k, x, continuation)` returns the deflated value
/// at event `k` given the deflated continuation (zero after the last).
pub fn backward_induction_deflated(
    model: &impl Gaussian1dModel,
    times: &[f64],
    mut at_event: impl FnMut(usize, f64, f64) -> Result<f64, RustyQLibError>,
    config: &GridConfig,
) -> Result<f64, RustyQLibError> {
    if times.is_empty() {
        return Ok(0.0);
    }
    for pair in times.windows(2) {
        if pair[1] <= pair[0] {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "event times must be strictly ascending",
            ));
        }
    }
    if config.nodes < 3
        || config.nodes % 2 == 0
        || config.quad_nodes < 3
        || config.quad_nodes % 2 == 0
    {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            "grid and quadrature nodes must be odd and at least 3",
        ));
    }
    let x0 = model.initial_state();
    let horizon = times
        .last()
        .copied()
        .unwrap_or(0.0)
        .max(config.min_horizon)
        .max(1e-8);
    let (_, shift_h, std_h) = model.transition(0.0, horizon);
    let nodes = config.nodes;
    let half_width = (config.stds * std_h + shift_h.abs()).max(1e-4);
    let dx = 2.0 * half_width / (nodes - 1) as f64;
    let grid: Vec<f64> = (0..nodes)
        .map(|j| x0 - half_width + j as f64 * dx)
        .collect();

    let quad_nodes = config.quad_nodes;
    let dz = 2.0 * config.quad_span / (quad_nodes - 1) as f64;
    let mut quad: Vec<(f64, f64)> = (0..quad_nodes)
        .map(|k| {
            let z = -config.quad_span + k as f64 * dz;
            let simpson = if k == 0 || k == quad_nodes - 1 {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            (z, simpson * norm_pdf(z) * dz / 3.0)
        })
        .collect();
    let total: f64 = quad.iter().map(|&(_, w)| w).sum();
    for (_, w) in quad.iter_mut() {
        *w /= total;
    }
    let interpolate = |values: &[f64], x: f64| -> f64 {
        if x <= grid[0] {
            return values[0];
        }
        if x >= grid[nodes - 1] {
            return values[nodes - 1];
        }
        let position = (x - grid[0]) / dx;
        let j = (position.floor() as usize).min(nodes - 2);
        let w = position - j as f64;
        values[j] * (1.0 - w) + values[j + 1] * w
    };

    let mut values = vec![0.0_f64; nodes];
    for (index, &time) in times.iter().enumerate().rev() {
        for (j, value) in values.iter_mut().enumerate() {
            *value = at_event(index, grid[j], *value)?;
        }
        let t_previous = if index == 0 { 0.0 } else { times[index - 1] };
        let (decay, shift, std) = model.transition(t_previous, time);
        let mut next = vec![0.0_f64; nodes];
        for (j, &x) in grid.iter().enumerate() {
            let mean = decay * x + shift;
            next[j] = if std > 0.0 {
                quad.iter()
                    .map(|&(z, w)| w * interpolate(&values, mean + std * z))
                    .sum()
            } else {
                interpolate(&values, mean)
            };
        }
        values = next;
    }
    Ok(model.numeraire(0.0, x0)? * interpolate(&values, x0))
}

/// Bermudan swaption: on each `tails[k].expiry` (ascending) the holder
/// may enter that tail.
pub fn bermudan_swaption(
    model: &impl Gaussian1dModel,
    tails: &[TailSwap],
    notional: f64,
    payer_receiver: PayerReceiver,
    config: &GridConfig,
) -> Result<f64, RustyQLibError> {
    if tails.is_empty() {
        return Err(RustyQLibError::invalid_input(FIELD, "no exercise dates"));
    }
    let times: Vec<f64> = tails.iter().map(|t| t.expiry).collect();
    let sign = side_sign(payer_receiver);
    backward_induction_deflated(
        model,
        &times,
        |k, x, continuation| {
            let exercise = sign * tail_value(model, &tails[k], notional, x)?
                / model.numeraire(tails[k].expiry, x)?;
            Ok(continuation.max(exercise))
        },
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::contracts::{BermudanSwaption, VanillaSwap};
    use crate::rates::engines::jamshidian::european_swaption_settled;
    use crate::rates::models::HullWhite;
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
    fn european_swaptions_on_hull_white_match_jamshidian() {
        let curve = market_curve();
        let models = [
            HullWhite::new(0.05, 0.011, curve.clone()).unwrap(),
            HullWhite::generalized(
                &[2.0],
                &[0.03, 0.12],
                &[1.0],
                &[0.012, 0.008],
                curve.clone(),
            )
            .unwrap(),
        ];
        let leg: Vec<(f64, f64)> = (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect();
        for m in &models {
            let g = m.gaussian1d(6.0);
            for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
                for (expiry, start) in [(1.0, 1.0), (1.0, 1.01)] {
                    let engine =
                        european_swaption(&g, expiry, start, &leg, 0.045, 1_000_000.0, side)
                            .unwrap();
                    let analytic =
                        european_swaption_settled(m, expiry, start, &leg, 0.045, 1_000_000.0, side)
                            .unwrap();
                    assert!(
                        (engine - analytic).abs() < 1e-6 * analytic,
                        "{side:?} {expiry}/{start}: {engine} vs {analytic}"
                    );
                }
            }
        }
    }

    #[test]
    fn bermudans_on_hull_white_match_the_short_rate_grid() {
        let m = HullWhite::new(0.05, 0.011, market_curve()).unwrap();
        let swap = VanillaSwap::usd_standard(
            10_000_000.0,
            0.045,
            PayerReceiver::Payer,
            d(2026, 8, 17),
            d(2032, 8, 17),
        )
        .unwrap();
        let b = BermudanSwaption::on_fixed_period_starts(swap, d(2027, 8, 1)).unwrap();
        let on_rate_grid = b.npv_hull_white(&m, &GridConfig::default()).unwrap();
        let on_deflated_grid = b
            .npv_gaussian1d(&m.gaussian1d(6.05), m.curve(), &GridConfig::default())
            .unwrap();
        assert!(
            (on_rate_grid - on_deflated_grid).abs() < 2e-3 * on_rate_grid,
            "{on_rate_grid} vs {on_deflated_grid}"
        );
    }
}
