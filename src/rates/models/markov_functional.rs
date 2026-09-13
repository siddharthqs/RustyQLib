//! The Markov functional model (Hunt, Kennedy, Pelsser 2000) in its
//! one-factor, terminal-bond-numeraire form.
//!
//! Instead of postulating short-rate dynamics and hoping the swaption
//! market falls out, the model takes the market's **marginal
//! distributions** as given and builds the numeraire around them. A
//! Gaussian Markov driver `x` (an Ornstein-Uhlenbeck process under the
//! `T*`-forward measure) carries the randomness; at each expiry `T_i`
//! of a coterminal swaption column the swap rate `S_i` is a monotone
//! **functional** `S_i(x)`, fixed so that the model's digital swaption
//! prices match the market's at every strike:
//!
//! ```text
//! P(0,T*) E[ A_i(x)/N(T_i,x) 1{x > x_b} ]  =  A_i(0) Phi((F_i - K)/(sigma_i sqrt T_i))
//! ```
//!
//! read as an equation for `K = S_i(x_b)`. The numeraire follows from
//! the swap-rate identity `1/N(T_i,x) = 1 + S_i(x) A_i(x)/N(T_i,x)`, and
//! the annuity ratio `A_i/N` at `T_i` is known from the later expiries
//! by Gaussian integration — so the construction runs backward from
//! `T*`, one expiry at a time, with no optimization anywhere. The
//! result reproduces the calibrating swaptions at **all strikes**, not
//! just at the money, which is what makes the model the reference for
//! Bermudan swaptions against a smile. The market marginals are either
//! flat Bachelier (one normal vol per expiry, [`calibrate`]) or a SABR
//! smile per expiry ([`calibrate_with_smiles`]), whose digitals enter
//! the same equation as minus the strike derivative of the smile's
//! Bachelier price.
//!
//! [`calibrate`]: MarkovFunctional::calibrate
//! [`calibrate_with_smiles`]: MarkovFunctional::calibrate_with_smiles
//!
//! The model exposes itself through the [`Gaussian1dModel`] trait, so
//! the [`gaussian1d`] engines price European and Bermudan swaptions on
//! it unchanged. Bonds are defined on the expiry grid (and `T*`): the
//! model knows `P(T_i, T_j, x)` and `P(T_i, T*, x)`, which is what a
//! coterminal Bermudan needs.
//!
//! [`gaussian1d`]: crate::rates::engines::gaussian1d

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::utils::{inv_norm_cdf, norm_pdf};
use crate::rates::models::gaussian1d::Gaussian1dModel;

const FIELD: &str = "markov functional";
const GRID_NODES: usize = 241;
const GRID_STDS: f64 = 6.0;
const QUAD_NODES: usize = 121;
const QUAD_SPAN: f64 = 6.0;
const TIME_TOLERANCE: f64 = 1e-9;

#[derive(Debug, Clone)]
pub struct MarkovFunctional {
    /// Mean reversion of the driver.
    pub a: f64,
    /// Volatility of the driver (its scale is absorbed by the
    /// functionals; it sets the driver's autocorrelation together
    /// with `a`, which is what the model's Bermudan prices depend on).
    pub sigma: f64,
    curve: YieldCurve,
    numeraire_time: f64,
    expiries: Vec<f64>,
    /// The driver grid at each expiry.
    grids: Vec<Vec<f64>>,
    /// `N(T_i, x) = P(T_i, T*, x)` on the grid.
    numeraires: Vec<Vec<f64>>,
    /// `S_i(x)`: the coterminal swap rate functional on the grid.
    swap_rates: Vec<Vec<f64>>,
}

impl MarkovFunctional {
    /// Calibrate to a coterminal column: swaptions expiring at
    /// `expiries` (ascending, positive) into swaps running to
    /// `numeraire_time`, with fixed payments on the later expiries and
    /// `numeraire_time` (accrual = the gaps), each quoted at its
    /// Bachelier `normal_vols[i]`.
    pub fn calibrate(
        a: f64,
        sigma: f64,
        curve: YieldCurve,
        expiries: &[f64],
        numeraire_time: f64,
        normal_vols: &[f64],
    ) -> Result<Self, RustyQLibError> {
        if expiries.len() != normal_vols.len() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need one normal vol per expiry",
            ));
        }
        if normal_vols.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "normal vols must be positive",
            ));
        }
        // flat Bachelier marginal: the digital inverts in closed form
        Self::build(
            a,
            sigma,
            curve,
            expiries,
            numeraire_time,
            &|i, forward, annuity_0, t, digital| {
                let p = (digital / annuity_0).clamp(1e-7, 1.0 - 1e-7);
                forward - normal_vols[i] * t.sqrt() * inv_norm_cdf(p)
            },
        )
    }

    /// Calibrate to a coterminal column quoted with a **smile**: at each
    /// expiry a [`RateSabr`] gives the normal vol at every strike, and
    /// the market digital — minus the strike derivative of the smile's
    /// Bachelier price — replaces the flat Bachelier digital. The model
    /// then reproduces the smile's swaption prices at every strike —
    /// to the extent the smile is arbitrage-free: the construction
    /// matches the digitals, so a smile whose implied density does not
    /// integrate to the forward (Hagan's expansion in the wings at
    /// large `nu sqrt(T)`) shows up as a parity gap between the
    /// model's payer and receiver, not as a calibration failure.
    ///
    /// [`RateSabr`]: crate::rates::models::sabr::RateSabr
    pub fn calibrate_with_smiles(
        a: f64,
        sigma: f64,
        curve: YieldCurve,
        expiries: &[f64],
        numeraire_time: f64,
        smiles: &[crate::rates::models::sabr::RateSabr],
    ) -> Result<Self, RustyQLibError> {
        use crate::rates::engines::black::{swaption_from_vol, RateVol};
        use crate::rates::PayerReceiver;
        if expiries.len() != smiles.len() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need one smile per expiry",
            ));
        }
        Self::build(
            a,
            sigma,
            curve,
            expiries,
            numeraire_time,
            &|i, forward, annuity_0, t, digital| {
                let smile = &smiles[i];
                // the smile's payer price and its digital, -dC/dK
                let price = |k: f64| -> f64 {
                    smile
                        .normal_vol(forward, k, t)
                        .and_then(|v| {
                            swaption_from_vol(
                                annuity_0,
                                forward,
                                k,
                                t,
                                RateVol::Normal(v),
                                PayerReceiver::Payer,
                            )
                        })
                        .unwrap_or(f64::NAN)
                };
                let atm = smile
                    .normal_vol(forward, forward, t)
                    .unwrap_or(0.01)
                    .max(1e-5);
                let h = 1e-4 * atm * t.sqrt();
                let market_digital = |k: f64| -> f64 { -(price(k + h) - price(k - h)) / (2.0 * h) };
                // the digital falls with the strike: bracket and bisect
                let (mut lo, mut hi) = (
                    forward - 8.0 * atm * t.sqrt(),
                    forward + 8.0 * atm * t.sqrt(),
                );
                if market_digital(lo) <= digital {
                    return lo;
                }
                if market_digital(hi) >= digital {
                    return hi;
                }
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    if market_digital(mid) > digital {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                0.5 * (lo + hi)
            },
        )
    }

    /// The backward construction shared by the flat and smile markets:
    /// `strike_for(i, forward, annuity_0, t_i, digital)` inverts the
    /// market digital (in currency, on unit notional) to the strike.
    fn build(
        a: f64,
        sigma: f64,
        curve: YieldCurve,
        expiries: &[f64],
        numeraire_time: f64,
        strike_for: &dyn Fn(usize, f64, f64, f64, f64) -> f64,
    ) -> Result<Self, RustyQLibError> {
        if !(a > 0.0 && a.is_finite() && sigma > 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("need positive a and sigma, got a={a}, sigma={sigma}"),
            ));
        }
        if expiries.is_empty() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need at least one expiry",
            ));
        }
        if expiries[0] <= 0.0 || expiries.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("expiries must be positive and increasing, got {expiries:?}"),
            ));
        }
        if numeraire_time <= expiries[expiries.len() - 1] {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "the numeraire time must lie after the last expiry",
            ));
        }
        let n = expiries.len();
        let mut model = MarkovFunctional {
            a,
            sigma,
            curve,
            numeraire_time,
            expiries: expiries.to_vec(),
            grids: Vec::with_capacity(n),
            numeraires: vec![Vec::new(); n],
            swap_rates: vec![Vec::new(); n],
        };
        for &t in expiries {
            let s = model.driver_std(t);
            model.grids.push(
                (0..GRID_NODES)
                    .map(|k| {
                        -GRID_STDS * s + k as f64 * 2.0 * GRID_STDS * s / (GRID_NODES - 1) as f64
                    })
                    .collect(),
            );
        }
        // payment times of tail i: expiries[i+1..], then T*
        let pay_time = |k: usize| -> f64 {
            if k < n {
                expiries[k]
            } else {
                numeraire_time
            }
        };
        for i in (0..n).rev() {
            let t_i = expiries[i];
            let grid = model.grids[i].clone();
            // annuity ratio A_i(x)/N(T_i,x) = sum tau_k E[1/N(T_{k+1})|x]
            let annuity_ratio: Vec<f64> = grid
                .iter()
                .map(|&x| {
                    let mut total = 0.0;
                    for k in i..n {
                        let tau = pay_time(k + 1) - pay_time(k);
                        total += tau * model.inverse_numeraire_expectation(t_i, x, k + 1);
                    }
                    total
                })
                .collect();
            // the model digital from each grid node upward, with the
            // unconditional N(0, std^2) law of the driver at T_i
            let std = model.driver_std(t_i);
            let density: Vec<f64> = grid.iter().map(|&x| norm_pdf(x / std) / std).collect();
            let mut digital = vec![0.0; GRID_NODES];
            for k in (0..GRID_NODES - 1).rev() {
                let dx = grid[k + 1] - grid[k];
                digital[k] = digital[k + 1]
                    + 0.5
                        * dx
                        * (annuity_ratio[k] * density[k] + annuity_ratio[k + 1] * density[k + 1]);
            }
            let p_star = model.curve.df(numeraire_time);
            // the market: annuity, forward swap rate, Bachelier digital
            let annuity_0: f64 = (i..n)
                .map(|k| (pay_time(k + 1) - pay_time(k)) * model.curve.df(pay_time(k + 1)))
                .sum();
            let forward = (model.curve.df(t_i) - p_star) / annuity_0;
            // each grid node's model digital, in currency, inverts to the
            // strike the market assigns that probability
            let swap_rate: Vec<f64> = digital
                .iter()
                .map(|&m| strike_for(i, forward, annuity_0, t_i, p_star * m))
                .collect();
            let numeraire: Vec<f64> = swap_rate
                .iter()
                .zip(&annuity_ratio)
                .map(|(s, ar)| 1.0 / (1.0 + (s * ar).max(-0.95)))
                .collect();
            model.swap_rates[i] = swap_rate;
            model.numeraires[i] = numeraire;
        }
        Ok(model)
    }

    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    /// The calibrated swap-rate functional at expiry `i` on its grid,
    /// as `(x, S_i(x))` pairs.
    pub fn swap_rate_functional(&self, i: usize) -> Vec<(f64, f64)> {
        self.grids[i]
            .iter()
            .copied()
            .zip(self.swap_rates[i].iter().copied())
            .collect()
    }

    fn driver_std(&self, t: f64) -> f64 {
        (self.sigma * self.sigma * (1.0 - (-2.0 * self.a * t).exp()) / (2.0 * self.a)).sqrt()
    }

    fn expiry_index(&self, t: f64) -> Option<usize> {
        self.expiries
            .iter()
            .position(|&e| (e - t).abs() < TIME_TOLERANCE)
    }

    /// Linear interpolation of a grid table at `x`, flat outside.
    fn interpolate(grid: &[f64], values: &[f64], x: f64) -> f64 {
        let n = grid.len();
        if x <= grid[0] {
            return values[0];
        }
        if x >= grid[n - 1] {
            return values[n - 1];
        }
        let dx = grid[1] - grid[0];
        let position = (x - grid[0]) / dx;
        let k = (position.floor() as usize).min(n - 2);
        let w = position - k as f64;
        values[k] * (1.0 - w) + values[k + 1] * w
    }

    /// `E[1/N(T_j, x') | x at t]` for payment index `j` (`j == n`
    /// means `T*`, where the numeraire is one).
    fn inverse_numeraire_expectation(&self, t: f64, x: f64, j: usize) -> f64 {
        if j >= self.expiries.len() {
            return 1.0;
        }
        let t_j = self.expiries[j];
        let decay = (-self.a * (t_j - t)).exp();
        let std = self.driver_std(t_j - t);
        let mean = decay * x;
        let dz = 2.0 * QUAD_SPAN / (QUAD_NODES - 1) as f64;
        let mut total = 0.0;
        let mut weight_sum = 0.0;
        for k in 0..QUAD_NODES {
            let z = -QUAD_SPAN + k as f64 * dz;
            let simpson = if k == 0 || k == QUAD_NODES - 1 {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            let w = simpson * dz / 3.0 * norm_pdf(z);
            let n = Self::interpolate(&self.grids[j], &self.numeraires[j], mean + std * z);
            total += w / n;
            weight_sum += w;
        }
        total / weight_sum
    }
}

impl Gaussian1dModel for MarkovFunctional {
    fn numeraire_time(&self) -> f64 {
        self.numeraire_time
    }

    /// The driver is an Ornstein-Uhlenbeck process under the `T*`
    /// measure by construction: no drift correction.
    fn transition(&self, t0: f64, t1: f64) -> (f64, f64, f64) {
        ((-self.a * (t1 - t0)).exp(), 0.0, self.driver_std(t1 - t0))
    }

    fn zerobond(&self, t: f64, maturity: f64, x: f64) -> Result<f64, RustyQLibError> {
        if (maturity - t).abs() < TIME_TOLERANCE {
            return Ok(1.0);
        }
        if maturity < t {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("maturity {maturity} precedes t {t}"),
            ));
        }
        if t.abs() < TIME_TOLERANCE {
            // the anchor: the curve itself
            return Ok(self.curve.df(maturity));
        }
        let Some(i) = self.expiry_index(t) else {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("bonds are defined on the expiry grid; {t} is not an expiry"),
            ));
        };
        let numeraire = Self::interpolate(&self.grids[i], &self.numeraires[i], x);
        if (maturity - self.numeraire_time).abs() < TIME_TOLERANCE {
            return Ok(numeraire);
        }
        let Some(j) = self.expiry_index(maturity) else {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("bonds are defined on the expiry grid and T*; {maturity} is neither"),
            ));
        };
        Ok(numeraire * self.inverse_numeraire_expectation(t, x, j))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::engines::black::{swaption_from_vol, RateVol};
    use crate::rates::engines::gaussian1d::{bermudan_swaption, european_swaption};
    use crate::rates::engines::hw_grid::GridConfig;
    use crate::rates::models::black_karasinski::TailSwap;
    use crate::rates::PayerReceiver;
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

    /// Annual expiries 1..5 into a 6y terminal date, normal vols
    /// falling with expiry.
    fn column() -> (Vec<f64>, f64, Vec<f64>) {
        (
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            6.0,
            vec![0.0095, 0.0092, 0.0090, 0.0088, 0.0086],
        )
    }

    fn model() -> MarkovFunctional {
        let (expiries, t_star, vols) = column();
        MarkovFunctional::calibrate(0.05, 0.01, market_curve(), &expiries, t_star, &vols).unwrap()
    }

    /// The tail swap of expiry `i` on unit notional at `strike`.
    fn tail(expiries: &[f64], t_star: f64, i: usize, strike: f64) -> (Vec<(f64, f64)>, f64) {
        let mut pays: Vec<f64> = expiries[i + 1..].to_vec();
        pays.push(t_star);
        let mut prev = expiries[i];
        let leg: Vec<(f64, f64)> = pays
            .iter()
            .map(|&p| {
                let tau = p - prev;
                prev = p;
                (p, tau)
            })
            .collect();
        (leg, strike)
    }

    #[test]
    fn the_functionals_are_monotone_and_the_curve_is_reproduced() {
        let m = model();
        for i in 0..5 {
            let f = m.swap_rate_functional(i);
            assert!(f.windows(2).all(|w| w[1].1 >= w[0].1), "S_{i} not monotone");
            // rates near the money at the center of the grid
            let center = f[f.len() / 2].1;
            assert!(center > 0.03 && center < 0.06, "S_{i}(0) = {center}");
            // P(0, T_i) = P(0,T*) E[1/N(T_i)]
            let implied = m.curve().df(6.0) * m.inverse_numeraire_expectation(0.0, 0.0, i);
            let market = m.curve().df(m.expiries()[i]);
            assert!(
                (implied / market - 1.0).abs() < 2e-4,
                "T_{i}: {implied} vs {market}"
            );
        }
    }

    #[test]
    fn calibrating_swaptions_reprice_at_every_strike() {
        // the whole point: the model matches the market marginal, so the
        // Bachelier prices come back off the money as well as at it
        let m = model();
        let (expiries, t_star, vols) = column();
        let curve = m.curve();
        for i in [0, 2, 4] {
            let (leg, _) = tail(&expiries, t_star, i, 0.0);
            let annuity: f64 = leg.iter().map(|&(t, tau)| tau * curve.df(t)).sum();
            let forward = (curve.df(expiries[i]) - curve.df(t_star)) / annuity;
            for offset in [-0.005, 0.0, 0.005] {
                let strike = forward + offset;
                for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
                    let market = swaption_from_vol(
                        annuity,
                        forward,
                        strike,
                        expiries[i],
                        RateVol::Normal(vols[i]),
                        side,
                    )
                    .unwrap();
                    let model_price =
                        european_swaption(&m, expiries[i], expiries[i], &leg, strike, 1.0, side)
                            .unwrap();
                    assert!(
                        (model_price - market).abs() < 3e-3 * market.max(1e-4),
                        "expiry {} strike {strike} {side:?}: model {model_price} vs market {market}",
                        expiries[i]
                    );
                }
            }
        }
    }

    #[test]
    fn bermudan_dominates_the_europeans() {
        let m = model();
        let (expiries, t_star, _) = column();
        let strike = 0.045;
        let tails: Vec<TailSwap> = (0..5)
            .map(|i| {
                let (leg, _) = tail(&expiries, t_star, i, strike);
                TailSwap {
                    expiry: expiries[i],
                    start: expiries[i],
                    coupons: leg
                        .iter()
                        .map(|&(t, tau)| (t, 1_000_000.0 * strike * tau))
                        .collect(),
                    last: t_star,
                }
            })
            .collect();
        let bermudan = bermudan_swaption(
            &m,
            &tails,
            1_000_000.0,
            PayerReceiver::Payer,
            &GridConfig::default(),
        )
        .unwrap();
        let best = (0..5)
            .map(|i| {
                let (leg, _) = tail(&expiries, t_star, i, strike);
                european_swaption(
                    &m,
                    expiries[i],
                    expiries[i],
                    &leg,
                    strike,
                    1_000_000.0,
                    PayerReceiver::Payer,
                )
                .unwrap()
            })
            .fold(f64::MIN, f64::max);
        assert!(bermudan > best, "{bermudan} vs {best}");
        assert!(bermudan < 1.5 * best.max(1.0) + 100_000.0);
        // a single-date Bermudan is the European
        let single = bermudan_swaption(
            &m,
            &tails[..1],
            1_000_000.0,
            PayerReceiver::Payer,
            &GridConfig::default(),
        )
        .unwrap();
        let (leg, _) = tail(&expiries, t_star, 0, strike);
        let european = european_swaption(
            &m,
            1.0,
            1.0,
            &leg,
            strike,
            1_000_000.0,
            PayerReceiver::Payer,
        )
        .unwrap();
        assert!(
            (single - european).abs() < 2e-3 * european,
            "{single} vs {european}"
        );
    }

    #[test]
    fn a_smile_calibration_reprices_the_smile_at_every_strike() {
        use crate::rates::models::sabr::RateSabr;
        let (expiries, t_star, vols) = column();
        let smiles: Vec<RateSabr> = vols
            .iter()
            .map(|&v| RateSabr::new(v, 0.0, -0.3, 0.2, 0.0).unwrap())
            .collect();
        let m = MarkovFunctional::calibrate_with_smiles(
            0.05,
            0.01,
            market_curve(),
            &expiries,
            t_star,
            &smiles,
        )
        .unwrap();
        let curve = m.curve();
        for i in [0, 2, 4] {
            let (leg, _) = tail(&expiries, t_star, i, 0.0);
            let annuity: f64 = leg.iter().map(|&(t, tau)| tau * curve.df(t)).sum();
            let forward = (curve.df(expiries[i]) - curve.df(t_star)) / annuity;
            for offset in [-0.01, -0.005, 0.0, 0.005, 0.01] {
                let strike = forward + offset;
                let vol = smiles[i].normal_vol(forward, strike, expiries[i]).unwrap();
                for side in [PayerReceiver::Payer, PayerReceiver::Receiver] {
                    let market = swaption_from_vol(
                        annuity,
                        forward,
                        strike,
                        expiries[i],
                        RateVol::Normal(vol),
                        side,
                    )
                    .unwrap();
                    let model_price =
                        european_swaption(&m, expiries[i], expiries[i], &leg, strike, 1.0, side)
                            .unwrap();
                    assert!(
                        (model_price - market).abs() < 5e-3 * market.max(1e-4),
                        "expiry {} strike {strike} {side:?}: model {model_price} vs smile {market}",
                        expiries[i]
                    );
                }
            }
        }
        // the flat-vol model is the smile model with nu = 0
        let flat_smiles: Vec<RateSabr> = vols
            .iter()
            .map(|&v| RateSabr::new(v, 0.0, 0.0, 0.0, 0.0).unwrap())
            .collect();
        let via_smile = MarkovFunctional::calibrate_with_smiles(
            0.05,
            0.01,
            market_curve(),
            &expiries,
            t_star,
            &flat_smiles,
        )
        .unwrap();
        let flat = model();
        let (leg, _) = tail(&expiries, t_star, 1, 0.0);
        let a = european_swaption(&via_smile, 2.0, 2.0, &leg, 0.045, 1.0, PayerReceiver::Payer)
            .unwrap();
        let b = european_swaption(&flat, 2.0, 2.0, &leg, 0.045, 1.0, PayerReceiver::Payer).unwrap();
        assert!((a - b).abs() < 1e-3 * b, "{a} vs {b}");
        assert!(MarkovFunctional::calibrate_with_smiles(
            0.05,
            0.01,
            market_curve(),
            &expiries,
            t_star,
            &smiles[..2]
        )
        .is_err());
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let c = market_curve();
        assert!(MarkovFunctional::calibrate(0.0, 0.01, c.clone(), &[1.0], 2.0, &[0.01]).is_err());
        assert!(
            MarkovFunctional::calibrate(0.05, 0.01, c.clone(), &[1.0, 2.0], 3.0, &[0.01]).is_err()
        );
        assert!(MarkovFunctional::calibrate(
            0.05,
            0.01,
            c.clone(),
            &[2.0, 1.0],
            3.0,
            &[0.01, 0.01]
        )
        .is_err());
        assert!(MarkovFunctional::calibrate(0.05, 0.01, c.clone(), &[1.0], 1.0, &[0.01]).is_err());
        let m = model();
        // bonds only on the grid
        assert!(m.zerobond(1.5, 3.0, 0.0).is_err());
        assert!(m.zerobond(1.0, 2.5, 0.0).is_err());
        assert!((m.zerobond(1.0, 1.0, 0.3).unwrap() - 1.0).abs() < 1e-15);
    }
}
