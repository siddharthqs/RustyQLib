//! ZABR (Andreasen and Huge, "ZABR — expansions for the masses", 2011):
//! SABR with a free vol-of-vol exponent, priced arbitrage-free.
//!
//! ```text
//! dF = z sigma(F) dW,   dz = eps z^gamma dZ,   d<W,Z> = rho dt,   z(0) = 1
//! sigma(F) = alpha (F + shift)^beta
//! ```
//!
//! `gamma = 1` is SABR; `gamma < 1` damps the vol of vol at high vol
//! levels and flattens the wings, `gamma > 1` steepens them — the one
//! extra parameter that lets a smile fit its wings without giving up
//! its body. The model has no expansion of Hagan's kind; instead:
//!
//! 1. **The short-maturity smile** is the geodesic distance `d(K)` from
//!    the start `(F, z = 1)` to the strike line `{K} x z` in the metric
//!    of the two-factor diffusion — the Berestycki-Busca-Florent
//!    result `sigma_N(K) -> (K - F) / d(K)`. In the variable `y =
//!    int dF / sigma(F)` the metric is invariant under `y`-shifts, so
//!    the `y`-momentum `p` is conserved along a geodesic and the
//!    shortest path meets the strike line orthogonally, at the level
//!    `z* = 1 / p`. The distance is found by shooting: the Hamiltonian
//!    flow is integrated (RK4 in arc length) from the start in every
//!    departure direction, and the direction whose arrival momentum
//!    `p_z` vanishes is the geodesic — a scalar root per strike, warm
//!    started from the neighbouring strike. The departure may point
//!    anywhere on the circle: for a far strike under a negative
//!    correlation the cheapest path first climbs in vol, even slightly
//!    away from the strike, before curving toward it. For `gamma = 1`
//!    the result reproduces Hagan's `z / x(z)` to 1e-6, the test that
//!    pins the machinery.
//! 2. **The local volatility** consistent with that smile is
//!    `sigma_loc(K) = sigma(K) / p(K) = sigma(K) z*(K)`: `p = dd/dy` by
//!    Hamilton-Jacobi, so `1 / d'(K)` — the local vol of any diffusion
//!    reproducing the distance — is `sigma(K) / p`.
//! 3. **Prices** come from the Dupire forward equation `C_T = 1/2
//!    sigma_loc(K)^2 C_KK` solved implicitly in strike from the
//!    intrinsic payoff, which keeps them decreasing and convex in the
//!    strike — no negative densities in the wings, unlike the SABR
//!    expansion at large `nu sqrt(T)` — and the implied normal vol is
//!    read back from them.
//!
//! A [`ZabrSmile`] is built for one expiry and forward and then quotes
//! any strike; [`ZabrSmile::calibrate`] fits `(alpha, rho, eps)` at
//! chosen `beta`, `gamma` and shift to normal-vol quotes.

use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::rates::engines::black::{implied_normal_vol, RateVol};

const FIELD: &str = "zabr";

/// The dynamics parameters of a ZABR smile.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    /// Vol of vol.
    pub eps: f64,
    /// Vol-of-vol exponent; `1` is SABR.
    pub gamma: f64,
    pub shift: f64,
}

impl ZabrParams {
    pub fn new(
        alpha: f64,
        beta: f64,
        rho: f64,
        eps: f64,
        gamma: f64,
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
        if !(eps >= 0.0 && eps.is_finite())
            || !(gamma >= 0.0 && gamma.is_finite())
            || !shift.is_finite()
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "need eps >= 0, gamma >= 0 and a finite shift, got {eps}, {gamma}, {shift}"
                ),
            ));
        }
        Ok(ZabrParams {
            alpha,
            beta,
            rho,
            eps,
            gamma,
            shift,
        })
    }

    /// `sigma(F) = alpha (F + shift)^beta`.
    fn backbone(&self, rate: f64) -> f64 {
        self.alpha * (rate + self.shift).abs().powf(self.beta)
    }

    /// `y(K) = int_F^K du / sigma(u)`.
    fn y_of(&self, forward: f64, strike: f64) -> f64 {
        let (f, k) = (forward + self.shift, strike + self.shift);
        if self.beta == 0.0 {
            return (k - f) / self.alpha;
        }
        if (self.beta - 1.0).abs() < 1e-12 {
            return (k / f).ln() / self.alpha;
        }
        let omb = 1.0 - self.beta;
        (k.powf(omb) - f.powf(omb)) / (self.alpha * omb)
    }

    /// One geodesic shot forward from the start `(0, 1)` in the metric
    /// direction `phi` (angle in the `(y, z)` plane), following the
    /// Hamiltonian flow in arc length until the path reaches the
    /// strike line `y = target`: returns the arrival momentum `p_z`
    /// (zero for the geodesic that meets the line orthogonally — the
    /// shortest one), the conserved `y`-momentum `p` and the length,
    /// as `(p_z, p, length)`; `None` if the path never gets there.
    fn shoot(&self, target: f64, phi: f64, rho: f64, steps: usize) -> Option<(f64, f64, f64)> {
        let (eps, gamma) = (self.eps, self.gamma);
        // covariance at the start (z = 1) and its inverse: the unit
        // tangent v in direction phi has momentum p = Sigma^{-1} v
        let det = eps * eps * (1.0 - rho * rho);
        let (vy, vz) = (phi.cos(), phi.sin());
        let py = (eps * eps * vy - rho * eps * vz) / det;
        let pz = (-rho * eps * vy + vz) / det;
        let norm = (py * vy + pz * vz).sqrt(); // g(v, v) = p . v
        let p = py / norm;
        let mut q = pz / norm;
        let flow = |z: f64, q: f64| -> (f64, f64, f64) {
            let zg = z.powf(gamma);
            let fy = z * z * p + rho * eps * zg * z * q;
            let fz = rho * eps * zg * z * p + eps * eps * zg * zg * q;
            let fq = -(z * p * p
                + rho * eps * (gamma + 1.0) * zg * p * q
                + gamma * eps * eps * zg * zg / z * q * q);
            (fy, fz, fq)
        };
        let rk4 = |yy: f64, z: f64, q: f64, dh: f64| -> (f64, f64, f64) {
            let (k1y, k1z, k1q) = flow(z, q);
            let (k2y, k2z, k2q) = flow(z + 0.5 * dh * k1z, q + 0.5 * dh * k1q);
            let (k3y, k3z, k3q) = flow(z + 0.5 * dh * k2z, q + 0.5 * dh * k2q);
            let (k4y, k4z, k4q) = flow(z + dh * k3z, q + dh * k3q);
            (
                yy + dh / 6.0 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y),
                z + dh / 6.0 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z),
                q + dh / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q),
            )
        };
        // the path along z = 1 has length `target`; the geodesic is shorter
        let h = 1.5 * target / steps as f64;
        let (mut yy, mut z, mut s) = (0.0_f64, 1.0_f64, 0.0_f64);
        for _ in 0..steps {
            let (y0, z0, q0) = (yy, z, q);
            (yy, z, q) = rk4(yy, z, q, h);
            s += h;
            if !(z > 0.0 && z.is_finite() && yy.is_finite() && q.is_finite()) {
                return None;
            }
            if yy >= target {
                // locate the arrival inside this step with fine substeps
                let substeps = 32usize;
                let dh = h / substeps as f64;
                let (mut fy, mut fz, mut fq, mut fs) = (y0, z0, q0, s - h);
                for _ in 0..substeps {
                    let (py, pq) = (fy, fq);
                    (fy, fz, fq) = rk4(fy, fz, fq, dh);
                    fs += dh;
                    if fy >= target {
                        let fraction = (target - py) / (fy - py);
                        return Some((pq + fraction * (fq - pq), p, fs - dh + fraction * dh));
                    }
                }
                return Some((fq, p, fs));
            }
        }
        None
    }

    /// The geodesic distance `d(y)` from `(0, 1)` to the strike line at
    /// signed `y`, and the momentum `p = dd/dy = 1 / z*` there: the
    /// departure angle is the root of "arrives orthogonally", found by
    /// scanning the half-plane of directions for sign changes of the
    /// arrival `p_z` and bisecting, the shortest candidate taken.
    fn distance(
        &self,
        y: f64,
        steps: usize,
        hint: Option<f64>,
    ) -> Result<(f64, f64, f64), RustyQLibError> {
        if y.abs() < 1e-14 || self.eps == 0.0 {
            return Ok((y.abs(), 1.0, 0.0));
        }
        let (target, rho) = if y > 0.0 {
            (y, self.rho)
        } else {
            (-y, -self.rho)
        };
        let pi = std::f64::consts::PI;
        // the departure may point anywhere: for a far strike under a
        // negative correlation the shortest path first climbs in z, even
        // slightly away from the strike, before curving toward it
        let search =
            |lo: f64, hi: f64, scan: usize| -> Result<Option<(f64, f64, f64)>, RustyQLibError> {
                let mut best: Option<(f64, f64, f64)> = None;
                let mut previous: Option<(f64, f64)> = None;
                for k in 0..=scan {
                    let phi = lo + (hi - lo) * k as f64 / scan as f64;
                    let Some((miss, _, _)) = self.shoot(target, phi, rho, steps) else {
                        previous = None;
                        continue;
                    };
                    if let Some((prev_phi, prev_miss)) = previous {
                        if prev_miss.signum() != miss.signum() {
                            let root = Solver1d::new(1e-12, 100)
                                .bisection(
                                    |x| {
                                        self.shoot(target, x, rho, steps)
                                            .map(|(m, _, _)| m)
                                            .unwrap_or(f64::NAN)
                                    },
                                    prev_phi,
                                    phi,
                                )?
                                .x;
                            if let Some((m, p, length)) = self.shoot(target, root, rho, steps) {
                                if m.abs() < 1e-5 && best.is_none_or(|(d, _, _)| length < d) {
                                    best = Some((length, p, root));
                                }
                            }
                        }
                    }
                    previous = Some((phi, miss));
                }
                Ok(best)
            };
        // a hint from a neighbouring strike brackets the root cheaply;
        // otherwise, or if that misses, scan the whole circle
        if let Some(phi0) = hint {
            if let Some(found) = search(phi0 - 0.2, phi0 + 0.2, 8)? {
                return Ok(found);
            }
        }
        match search(-pi, pi, 90)? {
            Some(found) => Ok(found),
            None => Err(RustyQLibError::NumericalError(format!(
                "no geodesic reaches the strike line at y = {y} (parameters {self:?})"
            ))),
        }
    }

    /// The short-maturity implied normal vol `(K - F) / d(K)` — the
    /// zero-order expansion, exact as `T -> 0`.
    pub fn short_maturity_normal_vol(
        &self,
        forward: f64,
        strike: f64,
    ) -> Result<f64, RustyQLibError> {
        if (strike - forward).abs() < 1e-12 {
            return Ok(self.backbone(forward));
        }
        let (d, _, _) = self.distance(self.y_of(forward, strike), 400, None)?;
        Ok((strike - forward).abs() / d)
    }

    /// The local volatility `sigma(K) z*(K)` consistent with the
    /// short-maturity smile.
    pub fn local_vol(&self, forward: f64, strike: f64) -> Result<f64, RustyQLibError> {
        let (_, p, _) = self.distance(self.y_of(forward, strike), 400, None)?;
        Ok(self.backbone(strike) / p)
    }
}

/// Resolution of the forward-PDE solve behind a [`ZabrSmile`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZabrConfig {
    /// Strike nodes (odd, so the forward is a node).
    pub strike_nodes: usize,
    /// Half-width of the strike grid in short-maturity ATM stds.
    pub stds: f64,
    /// Implicit time steps to expiry.
    pub time_steps: usize,
    /// Strikes at which the geodesic is solved (odd); the local vol is
    /// interpolated between them onto the strike grid.
    pub geodesic_nodes: usize,
    /// RK4 steps per geodesic shot.
    pub ode_steps: usize,
}

impl Default for ZabrConfig {
    fn default() -> Self {
        ZabrConfig {
            strike_nodes: 401,
            stds: 8.0,
            time_steps: 25,
            geodesic_nodes: 41,
            ode_steps: 300,
        }
    }
}

/// One ZABR smile at a forward and expiry: the arbitrage-free call
/// prices on a strike grid and the normal vols they imply.
#[derive(Debug, Clone)]
pub struct ZabrSmile {
    pub params: ZabrParams,
    pub forward: f64,
    pub expiry: f64,
    strikes: Vec<f64>,
    calls: Vec<f64>,
}

/// The result of a smile calibration.
#[derive(Debug, Clone)]
pub struct ZabrFit {
    pub smile: ZabrSmile,
    /// Root-mean-square normal-vol error.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl ZabrSmile {
    /// Build the smile: local vol on the strike grid from the geodesic
    /// distance, then the implicit Dupire solve to `expiry`.
    pub fn new(
        params: ZabrParams,
        forward: f64,
        expiry: f64,
        config: &ZabrConfig,
    ) -> Result<Self, RustyQLibError> {
        if !(expiry > 0.0 && expiry.is_finite()) || !forward.is_finite() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need a positive expiry and a finite forward",
            ));
        }
        if config.strike_nodes < 5
            || config.strike_nodes % 2 == 0
            || config.time_steps == 0
            || !(config.stds > 0.0)
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "strike_nodes must be odd and at least 5, time_steps at least 1, stds positive",
            ));
        }
        let n = config.strike_nodes;
        let atm = params.backbone(forward);
        // widen for the vol of vol over the horizon
        let half = config.stds * atm * expiry.sqrt() * (1.0 + params.eps * expiry.sqrt());
        let dk = 2.0 * half / (n - 1) as f64;
        let strikes: Vec<f64> = (0..n).map(|i| forward - half + i as f64 * dk).collect();
        if config.geodesic_nodes < 3 || config.geodesic_nodes % 2 == 0 || config.ode_steps < 10 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "geodesic_nodes must be odd and at least 3, ode_steps at least 10",
            ));
        }
        // the local vol on a coarse strike grid, solved outward from the
        // forward on each side so every shot starts near its neighbour
        let m = config.geodesic_nodes;
        let coarse: Vec<f64> = (0..m)
            .map(|i| forward - half + i as f64 * 2.0 * half / (m - 1) as f64)
            .collect();
        let mut coarse_local = vec![0.0; m];
        let center = m / 2;
        for side in [1i64, -1i64] {
            let mut hint = None;
            let mut i = center as i64;
            while i >= 0 && (i as usize) < m {
                let k = coarse[i as usize];
                let (_, p, phi) =
                    params.distance(params.y_of(forward, k), config.ode_steps, hint)?;
                coarse_local[i as usize] = params.backbone(k) / p;
                hint = if (k - forward).abs() > 1e-12 {
                    Some(phi)
                } else {
                    None
                };
                i += side;
            }
        }
        let local: Vec<f64> = strikes
            .iter()
            .map(|&k| {
                let position =
                    ((k - coarse[0]) / (coarse[1] - coarse[0])).clamp(0.0, (m - 1) as f64);
                let i = (position.floor() as usize).min(m - 2);
                let w = position - i as f64;
                coarse_local[i] * (1.0 - w) + coarse_local[i + 1] * w
            })
            .collect();
        // Dupire forward equation, implicit Euler in T: C_T = 1/2 s^2 C_KK
        let mut calls: Vec<f64> = strikes.iter().map(|&k| (forward - k).max(0.0)).collect();
        let dt = expiry / config.time_steps as f64;
        let mut sub = vec![0.0; n - 1];
        let mut diag = vec![1.0; n];
        let mut sup = vec![0.0; n - 1];
        for i in 1..n - 1 {
            let c = 0.5 * local[i] * local[i] * dt / (dk * dk);
            sub[i - 1] = -c;
            diag[i] = 1.0 + 2.0 * c;
            sup[i] = -c;
        }
        for _ in 0..config.time_steps {
            // boundaries: intrinsic (linear) far in and zero far out
            calls[0] = forward - strikes[0];
            calls[n - 1] = 0.0;
            calls =
                crate::core::fd_solvers::tridiagonal::thomas_algorithm(&sub, &diag, &sup, &calls);
        }
        Ok(ZabrSmile {
            params,
            forward,
            expiry,
            strikes,
            calls,
        })
    }

    /// The arbitrage-free undiscounted call price at `strike` (linear
    /// between grid nodes, intrinsic outside).
    pub fn call_price(&self, strike: f64) -> f64 {
        let n = self.strikes.len();
        if strike <= self.strikes[0] {
            return self.forward - strike;
        }
        if strike >= self.strikes[n - 1] {
            return 0.0;
        }
        let dk = self.strikes[1] - self.strikes[0];
        let position = (strike - self.strikes[0]) / dk;
        let i = (position.floor() as usize).min(n - 2);
        let w = position - i as f64;
        self.calls[i] * (1.0 - w) + self.calls[i + 1] * w
    }

    /// The implied normal vol at `strike`, read back from the price.
    pub fn normal_vol(&self, strike: f64) -> Result<f64, RustyQLibError> {
        let price = self.call_price(strike).max(0.0);
        implied_normal_vol(
            1.0,
            self.forward,
            strike,
            self.expiry,
            PutOrCall::Call,
            price,
        )
    }

    /// The quote at `strike`.
    pub fn quote(&self, strike: f64) -> Result<RateVol, RustyQLibError> {
        Ok(RateVol::Normal(self.normal_vol(strike)?))
    }

    /// Calibrate `(alpha, rho, eps)` at fixed `beta`, `gamma` and
    /// `shift` to `(strike, normal vol)` quotes — at least three.
    pub fn calibrate(
        quotes: &[(f64, f64)],
        forward: f64,
        expiry: f64,
        beta: f64,
        gamma: f64,
        shift: f64,
        config: &ZabrConfig,
    ) -> Result<ZabrFit, RustyQLibError> {
        if quotes.len() < 3 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("need at least three strike quotes, got {}", quotes.len()),
            ));
        }
        if quotes
            .iter()
            .any(|&(k, v)| !k.is_finite() || !(v.is_finite() && v > 0.0))
        {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "quotes need finite strikes and positive vols",
            ));
        }
        let atm_vol = quotes
            .iter()
            .min_by(|a, b| (a.0 - forward).abs().total_cmp(&(b.0 - forward).abs()))
            .map(|&(_, v)| v)
            .expect("non-empty");
        let alpha0 = (atm_vol / (forward + shift).abs().max(1e-6).powf(beta)).max(1e-6);
        let unpack = |u: &[f64]| ZabrParams {
            alpha: u[0].exp(),
            beta,
            rho: u[1].tanh(),
            eps: u[2].exp(),
            gamma,
            shift,
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            match ZabrSmile::new(unpack(u), forward, expiry, config) {
                Ok(smile) => quotes
                    .iter()
                    .map(|&(k, v)| 1e4 * (smile.normal_vol(k).unwrap_or(f64::NAN) - v))
                    .collect(),
                Err(_) => vec![f64::NAN; quotes.len()],
            }
        };
        let x0 = [alpha0.ln(), (-0.3_f64).atanh(), 0.4_f64.ln()];
        let fit = levenberg_marquardt(&OptimConfig::new(1e-12, 200), &residuals, None, &x0);
        let smile = ZabrSmile::new(unpack(&fit.x), forward, expiry, config)?;
        let rmse = (quotes
            .iter()
            .map(|&(k, v)| (smile.normal_vol(k).unwrap_or(f64::NAN) - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        Ok(ZabrFit {
            smile,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged && rmse.is_finite(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rates::models::sabr::RateSabr;

    fn params(gamma: f64) -> ZabrParams {
        ZabrParams::new(0.0085, 0.0, -0.3, 0.45, gamma, 0.0).unwrap()
    }

    #[test]
    fn at_gamma_one_the_geodesic_reproduces_hagan_leading_order() {
        // normal SABR: sigma_N -> alpha z / x(z), z = (eps/alpha)(K - F)
        let p = params(1.0);
        let (alpha, rho, eps, f) = (p.alpha, p.rho, p.eps, 0.045);
        for k in [0.015, 0.03, 0.04, 0.05, 0.06, 0.08] {
            let z = eps / alpha * (f - k);
            let x = ((1.0 - 2.0 * rho * z + z * z).sqrt() + z - rho) / (1.0 - rho);
            let hagan = alpha * z / x.ln();
            let geodesic = p.short_maturity_normal_vol(f, k).unwrap();
            assert!(
                (geodesic - hagan).abs() < 1e-6 * hagan,
                "K={k}: {geodesic} vs {hagan}"
            );
        }
        assert!((p.short_maturity_normal_vol(f, f).unwrap() - alpha).abs() < 1e-15);
        // the local vol at the money is the backbone (z* = 1)
        assert!((p.local_vol(f, f).unwrap() - alpha).abs() < 1e-9);
    }

    #[test]
    fn the_pde_smile_is_arbitrage_free_and_near_sabr_at_gamma_one() {
        let f = 0.045;
        let smile = ZabrSmile::new(params(1.0), f, 1.0, &ZabrConfig::default()).unwrap();
        // decreasing and convex call prices
        let ks: Vec<f64> = (0..40).map(|i| 0.01 + 0.002 * i as f64).collect();
        let cs: Vec<f64> = ks.iter().map(|&k| smile.call_price(k)).collect();
        for w in cs.windows(3) {
            assert!(w[1] <= w[0] + 1e-12 && w[2] - 2.0 * w[1] + w[0] >= -1e-9);
        }
        // near Hagan's normal SABR at one year (the expansion is fine there)
        let sabr = RateSabr::new(0.0085, 0.0, -0.3, 0.45, 0.0).unwrap();
        for k in [0.03, 0.04, 0.045, 0.05, 0.06] {
            let z = smile.normal_vol(k).unwrap();
            let h = sabr.normal_vol(f, k, 1.0).unwrap();
            assert!((z - h).abs() < 0.03 * h, "K={k}: zabr {z} vs sabr {h}");
        }
    }

    #[test]
    fn gamma_shapes_the_wings() {
        // lower gamma: the vol of vol fades as vol rises, so the wings
        // flatten relative to SABR; higher gamma steepens them
        let f = 0.045;
        let wing = |gamma: f64| {
            ZabrSmile::new(params(gamma), f, 1.0, &ZabrConfig::default())
                .unwrap()
                .normal_vol(0.075)
                .unwrap()
        };
        let (low, sabr, high) = (wing(0.5), wing(1.0), wing(1.5));
        assert!(low < sabr && sabr < high, "{low} {sabr} {high}");
        // at the money all three agree closely
        let atm = |gamma: f64| {
            ZabrSmile::new(params(gamma), f, 1.0, &ZabrConfig::default())
                .unwrap()
                .normal_vol(f)
                .unwrap()
        };
        assert!((atm(0.5) - atm(1.5)).abs() < 0.03 * atm(1.0));
    }

    #[test]
    fn calibration_recovers_a_generating_smile() {
        let f = 0.045;
        let truth = ZabrSmile::new(params(0.7), f, 2.0, &ZabrConfig::default()).unwrap();
        let quotes: Vec<(f64, f64)> = (-4..=4)
            .map(|i| {
                let k = f + 0.006 * i as f64;
                (k, truth.normal_vol(k).unwrap())
            })
            .collect();
        let fit =
            ZabrSmile::calibrate(&quotes, f, 2.0, 0.0, 0.7, 0.0, &ZabrConfig::default()).unwrap();
        assert!(fit.converged && fit.rmse < 1e-6, "rmse {}", fit.rmse);
        assert!(
            (fit.smile.params.alpha - 0.0085).abs() < 1e-4,
            "alpha {}",
            fit.smile.params.alpha
        );
        assert!(
            (fit.smile.params.rho + 0.3).abs() < 2e-2,
            "rho {}",
            fit.smile.params.rho
        );
        assert!(
            (fit.smile.params.eps - 0.45).abs() < 2e-2,
            "eps {}",
            fit.smile.params.eps
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(ZabrParams::new(0.0, 0.0, -0.3, 0.4, 1.0, 0.0).is_err());
        assert!(ZabrParams::new(0.01, 0.0, 1.0, 0.4, 1.0, 0.0).is_err());
        assert!(ZabrParams::new(0.01, 0.0, -0.3, -0.4, 1.0, 0.0).is_err());
        assert!(ZabrSmile::new(params(1.0), 0.045, 0.0, &ZabrConfig::default()).is_err());
        let even = ZabrConfig {
            strike_nodes: 400,
            ..ZabrConfig::default()
        };
        assert!(ZabrSmile::new(params(1.0), 0.045, 1.0, &even).is_err());
        assert!(ZabrSmile::calibrate(
            &[(0.04, 0.01)],
            0.045,
            1.0,
            0.0,
            1.0,
            0.0,
            &ZabrConfig::default()
        )
        .is_err());
    }
}
