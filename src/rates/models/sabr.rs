//! SABR for rates: the smile across strikes at one swaption or caplet
//! expiry, and the cube of smiles over the swaption grid.
//!
//! Hagan, Kumar, Lesniewski and Woodward's dynamics on the forward
//! rate `F` (shifted by `s` so the backbone stays positive when rates
//! do not),
//!
//! ```text
//! d(F + s) = alpha (F + s)^beta dW,   dalpha = nu alpha dZ,   d<W,Z> = rho dt
//! ```
//!
//! with the two implied-vol expansions the market quotes in: the
//! **normal** (Bachelier) vol of eq. A.69/A.70 — the swaption and cap
//! standard, valid for any rate sign at `beta = 0` — and the
//! **lognormal** (Black) vol of eq. A.69 on the shifted rate. `beta` is
//! chosen, not fitted (it is collinear with `rho`), and `(alpha, rho,
//! nu)` are calibrated per node to the strike quotes by
//! Levenberg-Marquardt in a transform space.
//!
//! [`SabrSwaptionCube`] holds one smile per expiry-tenor node and
//! interpolates the parameters between nodes, so a swaption at any
//! strike prices off it ([`Swaption::npv_sabr`]), a cap prices caplet
//! by caplet against its own smile
//! ([`CapFloor::npv_with_smile`]), and the Markov functional model
//! takes the smile's digitals as its market
//! ([`MarkovFunctional::calibrate_with_smiles`]).
//!
//! [`Swaption::npv_sabr`]: crate::rates::contracts::swaption::Swaption::npv_sabr
//! [`CapFloor::npv_with_smile`]: crate::rates::contracts::cap_floor::CapFloor::npv_with_smile
//! [`MarkovFunctional::calibrate_with_smiles`]: crate::rates::models::markov_functional::MarkovFunctional::calibrate_with_smiles

use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::equity::models::sabr::SabrParams;
use crate::rates::engines::black::{RateVol, RateVolKind};
use crate::rates::models::vol_surface::bracket;

const FIELD: &str = "rate sabr";

/// One SABR smile: `alpha`, `beta`, `rho`, `nu` and the displacement
/// `shift` (zero for plain SABR).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateSabr {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub nu: f64,
    pub shift: f64,
}

/// The result of a smile calibration.
#[derive(Debug, Clone)]
pub struct RateSabrFit {
    pub smile: RateSabr,
    /// Root-mean-square vol error, in the quoted units.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// `z / x(z)` of the Hagan expansion, `x(z) = ln[(sqrt(1 - 2 rho z +
/// z^2) + z - rho) / (1 - rho)]`, one at the money.
fn z_over_x(z: f64, rho: f64) -> f64 {
    if z.abs() < 1e-7 {
        return 1.0;
    }
    let x = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho);
    z / x.ln()
}

impl RateSabr {
    pub fn new(
        alpha: f64,
        beta: f64,
        rho: f64,
        nu: f64,
        shift: f64,
    ) -> Result<Self, RustyQLibError> {
        if !(alpha > 0.0 && alpha.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("alpha must be positive, got {alpha}"),
            ));
        }
        if !(0.0..=1.0).contains(&beta) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("beta must lie in [0, 1], got {beta}"),
            ));
        }
        if !(rho > -1.0 && rho < 1.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("rho must lie in (-1, 1), got {rho}"),
            ));
        }
        if !(nu >= 0.0 && nu.is_finite()) || !shift.is_finite() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("nu must be non-negative and shift finite, got nu={nu}, shift={shift}"),
            ));
        }
        Ok(RateSabr {
            alpha,
            beta,
            rho,
            nu,
            shift,
        })
    }

    /// The displaced forward and strike, which the backbone needs
    /// positive unless `beta == 0`.
    fn displaced(&self, forward: f64, strike: f64) -> Result<(f64, f64), RustyQLibError> {
        let (f, k) = (forward + self.shift, strike + self.shift);
        if self.beta > 0.0 && (f <= 0.0 || k <= 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the shifted forward {f} and strike {k} must be positive for beta = {}; \
                     raise the shift or use beta = 0",
                    self.beta
                ),
            ));
        }
        Ok((f, k))
    }

    /// Hagan's normal (Bachelier) implied vol at `strike` (eq. A.69a
    /// for `beta > 0`, the exact-in-`F - K` form A.70 for `beta = 0`).
    pub fn normal_vol(&self, forward: f64, strike: f64, t: f64) -> Result<f64, RustyQLibError> {
        let (f, k) = self.displaced(forward, strike)?;
        let (alpha, beta, rho, nu) = (self.alpha, self.beta, self.rho, self.nu);
        if beta == 0.0 {
            let z = nu / alpha * (f - k);
            let correction = 1.0 + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0 * t;
            return Ok(alpha * z_over_x(z, rho) * correction);
        }
        let omb = 1.0 - beta;
        let fk = f * k;
        let fk_mid = fk.powf(0.5 * omb); // (FK)^{(1-beta)/2}
        let correction = 1.0
            + t * (-beta * (2.0 - beta) * alpha * alpha / (24.0 * fk.powf(omb))
                + rho * beta * nu * alpha / (4.0 * fk_mid)
                + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0);
        if (f - k).abs() < 1e-10 * f.abs().max(1e-4) {
            return Ok(alpha * f.powf(beta) * correction);
        }
        let l = (f / k).ln();
        let l2 = l * l;
        let numerator = 1.0 + l2 / 24.0 + l2 * l2 / 1920.0;
        let denominator = 1.0 + omb * omb * l2 / 24.0 + omb.powi(4) * l2 * l2 / 1920.0;
        let z = nu / alpha * fk_mid * l;
        Ok(alpha * fk.powf(0.5 * beta) * numerator / denominator * z_over_x(z, rho) * correction)
    }

    /// Hagan's lognormal (Black) implied vol at `strike` on the shifted
    /// rate; needs a positive shifted forward and strike.
    pub fn lognormal_vol(&self, forward: f64, strike: f64, t: f64) -> Result<f64, RustyQLibError> {
        let (f, k) = (forward + self.shift, strike + self.shift);
        if f <= 0.0 || k <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "a lognormal vol needs positive shifted forward and strike, got {f} and {k}"
                ),
            ));
        }
        Ok(SabrParams {
            alpha: self.alpha,
            beta: self.beta,
            rho: self.rho,
            nu: self.nu,
        }
        .vol(f, k, t))
    }

    /// The vol at `strike` as a quote of `kind`: normal, lognormal (with
    /// `shift = 0`) or shifted lognormal (with the smile's shift).
    pub fn quote(
        &self,
        kind: RateVolKind,
        forward: f64,
        strike: f64,
        t: f64,
    ) -> Result<RateVol, RustyQLibError> {
        Ok(match kind {
            RateVolKind::Normal => RateVol::Normal(self.normal_vol(forward, strike, t)?),
            RateVolKind::Lognormal => {
                if self.shift != 0.0 {
                    return Err(RustyQLibError::invalid_input(
                        FIELD,
                        "a shifted smile quotes shifted-lognormal vols, not plain lognormal",
                    ));
                }
                RateVol::Lognormal(self.lognormal_vol(forward, strike, t)?)
            }
            RateVolKind::ShiftedLognormal { shift } => {
                if (shift - self.shift).abs() > 1e-12 {
                    return Err(RustyQLibError::invalid_input(
                        FIELD,
                        format!(
                            "quote shift {shift} differs from the smile's {}",
                            self.shift
                        ),
                    ));
                }
                RateVol::ShiftedLognormal {
                    vol: self.lognormal_vol(forward, strike, t)?,
                    shift,
                }
            }
        })
    }

    /// The vol in the quoted units, as a bare number (for fitting).
    fn vol_number(&self, kind: RateVolKind, forward: f64, strike: f64, t: f64) -> f64 {
        match kind {
            RateVolKind::Normal => self.normal_vol(forward, strike, t).unwrap_or(f64::NAN),
            _ => self.lognormal_vol(forward, strike, t).unwrap_or(f64::NAN),
        }
    }

    /// Calibrate `(alpha, rho, nu)` at fixed `beta` and `shift` to one
    /// expiry's `(strike, vol)` quotes in `kind` — at least three.
    pub fn calibrate(
        quotes: &[(f64, f64)],
        forward: f64,
        t: f64,
        beta: f64,
        shift: f64,
        kind: RateVolKind,
    ) -> Result<RateSabrFit, RustyQLibError> {
        if quotes.len() < 3 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("need at least three strike quotes, got {}", quotes.len()),
            ));
        }
        if quotes
            .iter()
            .any(|&(k, v)| !k.is_finite() || !(v.is_finite() && v > 0.0))
            || !t.is_finite()
            || t <= 0.0
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "quotes need finite strikes and positive vols, and a positive expiry",
            ));
        }
        if !(0.0..=1.0).contains(&beta) || !shift.is_finite() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("beta must lie in [0, 1] and the shift be finite, got {beta}, {shift}"),
            ));
        }
        // starting alpha from the quote nearest the money
        let atm_vol = quotes
            .iter()
            .min_by(|a, b| (a.0 - forward).abs().total_cmp(&(b.0 - forward).abs()))
            .map(|&(_, v)| v)
            .expect("non-empty");
        let f = forward + shift;
        let alpha0 = match kind {
            RateVolKind::Normal => atm_vol / f.abs().max(1e-6).powf(beta),
            _ => atm_vol * f.abs().max(1e-6).powf(1.0 - beta),
        }
        .max(1e-6);
        let scale = match kind {
            RateVolKind::Normal => 1e4,
            _ => 1e2,
        };
        let unpack = |u: &[f64]| RateSabr {
            alpha: u[0].exp(),
            beta,
            rho: u[1].tanh(),
            nu: u[2].exp(),
            shift,
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            let smile = unpack(u);
            quotes
                .iter()
                .map(|&(k, v)| scale * (smile.vol_number(kind, forward, k, t) - v))
                .collect()
        };
        let x0 = [alpha0.ln(), (-0.3_f64).atanh(), 0.4_f64.ln()];
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 300), &residuals, None, &x0);
        let smile = unpack(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(k, v)| (smile.vol_number(kind, forward, k, t) - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        Ok(RateSabrFit {
            smile,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged && rmse.is_finite(),
        })
    }
}

/// The swaption smile cube: one [`RateSabr`] per expiry-tenor node,
/// with the node's forward swap rate, parameters interpolated
/// bilinearly between nodes (`beta` and `shift` common to the cube).
#[derive(Debug, Clone)]
pub struct SabrSwaptionCube {
    expiries: Vec<f64>,
    tenors: Vec<f64>,
    /// `smiles[i][j]` at `expiries[i]`, `tenors[j]`.
    smiles: Vec<Vec<RateSabr>>,
    kind: RateVolKind,
}

impl SabrSwaptionCube {
    /// A cube from calibrated smiles; every smile must share `beta`
    /// and `shift`.
    pub fn new(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        smiles: Vec<Vec<RateSabr>>,
        kind: RateVolKind,
    ) -> Result<Self, RustyQLibError> {
        for (name, axis) in [("expiries", &expiries), ("tenors", &tenors)] {
            if axis.is_empty() || axis[0] <= 0.0 || axis.windows(2).any(|w| w[1] <= w[0]) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("{name} must be positive and strictly increasing, got {axis:?}"),
                ));
            }
        }
        if smiles.len() != expiries.len() || smiles.iter().any(|row| row.len() != tenors.len()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need one smile per expiry-tenor node",
            ));
        }
        let first = smiles[0][0];
        if smiles
            .iter()
            .flatten()
            .any(|s| s.beta != first.beta || s.shift != first.shift)
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "every smile of a cube must share beta and shift",
            ));
        }
        Ok(SabrSwaptionCube {
            expiries,
            tenors,
            smiles,
            kind,
        })
    }

    /// Calibrate every node: `quotes[i][j]` is `(forward, strike quotes)`
    /// at `expiries[i]`, `tenors[j]`, the quotes `(strike, vol)` in
    /// `kind`. A node whose fit does not converge is an error.
    pub fn calibrate(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        quotes: &[Vec<(f64, Vec<(f64, f64)>)>],
        beta: f64,
        shift: f64,
        kind: RateVolKind,
    ) -> Result<Self, RustyQLibError> {
        if quotes.len() != expiries.len() || quotes.iter().any(|row| row.len() != tenors.len()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need one quote set per expiry-tenor node",
            ));
        }
        let mut smiles = Vec::with_capacity(expiries.len());
        for (i, row) in quotes.iter().enumerate() {
            let mut fitted = Vec::with_capacity(tenors.len());
            for (j, (forward, strikes)) in row.iter().enumerate() {
                let fit = RateSabr::calibrate(strikes, *forward, expiries[i], beta, shift, kind)?;
                if !fit.converged {
                    return Err(RustyQLibError::CalibrationFailed {
                        iterations: fit.iterations,
                        residual: fit.rmse,
                        reason: format!(
                            "the smile at expiry {} tenor {} did not converge",
                            expiries[i], tenors[j]
                        ),
                    });
                }
                fitted.push(fit.smile);
            }
            smiles.push(fitted);
        }
        Self::new(expiries, tenors, smiles, kind)
    }

    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    pub fn kind(&self) -> RateVolKind {
        self.kind
    }

    /// The calibrated smile at a node.
    pub fn node(&self, i: usize, j: usize) -> RateSabr {
        self.smiles[i][j]
    }

    /// The smile at `(expiry, tenor)`: `alpha`, `rho`, `nu` bilinear
    /// between the nodes, flat outside the grid.
    pub fn smile(&self, expiry: f64, tenor: f64) -> RateSabr {
        let (i, wi) = bracket(&self.expiries, expiry);
        let (j, wj) = bracket(&self.tenors, tenor);
        let blend = |pick: &dyn Fn(&RateSabr) -> f64| {
            let (i1, j1) = (
                (i + 1).min(self.expiries.len() - 1),
                (j + 1).min(self.tenors.len() - 1),
            );
            let row0 = pick(&self.smiles[i][j]) * (1.0 - wj) + pick(&self.smiles[i][j1]) * wj;
            let row1 = pick(&self.smiles[i1][j]) * (1.0 - wj) + pick(&self.smiles[i1][j1]) * wj;
            row0 * (1.0 - wi) + row1 * wi
        };
        RateSabr {
            alpha: blend(&|s| s.alpha),
            beta: self.smiles[0][0].beta,
            rho: blend(&|s| s.rho),
            nu: blend(&|s| s.nu),
            shift: self.smiles[0][0].shift,
        }
    }

    /// The quote for a swaption of `(expiry, tenor)` at `strike`, given
    /// its forward swap rate.
    pub fn vol(
        &self,
        expiry: f64,
        tenor: f64,
        forward: f64,
        strike: f64,
    ) -> Result<RateVol, RustyQLibError> {
        self.smile(expiry, tenor)
            .quote(self.kind, forward, strike, expiry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_sabr_limits_and_shape() {
        // beta = 0, nu = 0: the normal vol is alpha at every strike
        let flat = RateSabr::new(0.008, 0.0, 0.0, 0.0, 0.0).unwrap();
        for k in [0.02, 0.045, 0.07] {
            assert!((flat.normal_vol(0.045, k, 2.0).unwrap() - 0.008).abs() < 1e-15);
        }
        // negative rates are fine at beta = 0 and with a shift
        let smile = RateSabr::new(0.006, 0.0, -0.3, 0.4, 0.0).unwrap();
        assert!(smile.normal_vol(-0.002, -0.005, 1.0).unwrap() > 0.0);
        let shifted = RateSabr::new(0.25, 0.5, -0.3, 0.4, 0.03).unwrap();
        assert!(shifted.normal_vol(-0.002, 0.0, 1.0).unwrap() > 0.0);
        assert!(RateSabr::new(0.25, 0.5, -0.3, 0.4, 0.0)
            .unwrap()
            .normal_vol(-0.002, 0.0, 1.0)
            .is_err());
        // negative rho: a downward-sloping skew; rho = 0: a symmetric smile
        let skewed = RateSabr::new(0.008, 0.0, -0.5, 0.5, 0.0).unwrap();
        let (lo, atm, hi) = (
            skewed.normal_vol(0.045, 0.025, 2.0).unwrap(),
            skewed.normal_vol(0.045, 0.045, 2.0).unwrap(),
            skewed.normal_vol(0.045, 0.065, 2.0).unwrap(),
        );
        assert!(lo > atm && lo > hi, "{lo} {atm} {hi}");
        let symmetric = RateSabr::new(0.008, 0.0, 0.0, 0.5, 0.0).unwrap();
        let (lo, atm, hi) = (
            symmetric.normal_vol(0.045, 0.025, 2.0).unwrap(),
            symmetric.normal_vol(0.045, 0.045, 2.0).unwrap(),
            symmetric.normal_vol(0.045, 0.065, 2.0).unwrap(),
        );
        assert!(
            lo > atm && hi > atm && (lo - hi).abs() < 1e-12,
            "{lo} {atm} {hi}"
        );
        // beta > 0 general formula is continuous at the money
        let cev = RateSabr::new(0.05, 0.5, -0.2, 0.3, 0.0).unwrap();
        let at = cev.normal_vol(0.045, 0.045, 2.0).unwrap();
        let near = cev.normal_vol(0.045, 0.045 + 1e-7, 2.0).unwrap();
        assert!((at - near).abs() < 1e-7, "{at} vs {near}");
        // lognormal at beta = 1 is the equity Hagan formula
        let ln = RateSabr::new(0.2, 1.0, -0.3, 0.4, 0.0).unwrap();
        let equity = SabrParams {
            alpha: 0.2,
            beta: 1.0,
            rho: -0.3,
            nu: 0.4,
        };
        assert_eq!(
            ln.lognormal_vol(0.045, 0.05, 2.0).unwrap(),
            equity.vol(0.045, 0.05, 2.0)
        );
    }

    #[test]
    fn calibration_recovers_a_generating_smile_in_both_quotes() {
        let (forward, t) = (0.045, 2.0);
        let strikes: Vec<f64> = (-4..=4).map(|i| forward + 0.005 * i as f64).collect();
        let truth = RateSabr::new(0.0085, 0.0, -0.35, 0.45, 0.0).unwrap();
        let quotes: Vec<(f64, f64)> = strikes
            .iter()
            .map(|&k| (k, truth.normal_vol(forward, k, t).unwrap()))
            .collect();
        let fit = RateSabr::calibrate(&quotes, forward, t, 0.0, 0.0, RateVolKind::Normal).unwrap();
        assert!(fit.converged && fit.rmse < 1e-8, "rmse {}", fit.rmse);
        assert!((fit.smile.alpha - 0.0085).abs() < 1e-5);
        assert!((fit.smile.rho + 0.35).abs() < 1e-3);
        assert!((fit.smile.nu - 0.45).abs() < 1e-3);
        let truth_ln = RateSabr::new(0.05, 0.5, -0.2, 0.3, 0.0).unwrap();
        let quotes_ln: Vec<(f64, f64)> = strikes
            .iter()
            .map(|&k| (k, truth_ln.lognormal_vol(forward, k, t).unwrap()))
            .collect();
        let fit =
            RateSabr::calibrate(&quotes_ln, forward, t, 0.5, 0.0, RateVolKind::Lognormal).unwrap();
        assert!(
            fit.converged && fit.rmse < 1e-8,
            "lognormal rmse {}",
            fit.rmse
        );
        assert!((fit.smile.alpha - 0.05).abs() < 1e-4);
        assert!(
            RateSabr::calibrate(&quotes[..2], forward, t, 0.0, 0.0, RateVolKind::Normal).is_err()
        );
    }

    #[test]
    fn the_cube_interpolates_parameters_and_quotes_by_strike() {
        let mk = |alpha: f64, nu: f64| RateSabr::new(alpha, 0.0, -0.3, nu, 0.0).unwrap();
        let cube = SabrSwaptionCube::new(
            vec![1.0, 5.0],
            vec![2.0, 10.0],
            vec![
                vec![mk(0.009, 0.5), mk(0.008, 0.4)],
                vec![mk(0.007, 0.3), mk(0.006, 0.2)],
            ],
            RateVolKind::Normal,
        )
        .unwrap();
        let mid = cube.smile(3.0, 6.0);
        assert!((mid.alpha - 0.0075).abs() < 1e-15);
        assert!((mid.nu - 0.35).abs() < 1e-15);
        assert_eq!(cube.smile(0.5, 1.0), mk(0.009, 0.5));
        assert_eq!(cube.smile(9.0, 30.0), mk(0.006, 0.2));
        let atm = cube.vol(1.0, 2.0, 0.04, 0.04).unwrap();
        let low = cube.vol(1.0, 2.0, 0.04, 0.02).unwrap();
        assert!(low.vol() > atm.vol());
        assert!(SabrSwaptionCube::new(
            vec![1.0],
            vec![2.0, 10.0],
            vec![vec![
                mk(0.009, 0.5),
                RateSabr::new(0.2, 1.0, -0.3, 0.4, 0.0).unwrap()
            ]],
            RateVolKind::Normal
        )
        .is_err());
    }
}
