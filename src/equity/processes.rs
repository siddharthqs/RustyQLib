//! Equity model dynamics as [`StochasticProcess1D`] / [`StochasticProcess`]
//! implementations — the bridge between the model layer (GBM, Dupire
//! local vol, Heston) and the generic stepping in
//! [`core::montecarlo::process`](crate::core::montecarlo::process).
//!
//! The Monte Carlo engine consumes these; the same objects can feed any
//! future coefficient-driven method (finite difference via the
//! Feynman-Kac link, other samplers) without touching the models.

use libm::exp;

use crate::core::montecarlo::process::{
    numeric_diffusion_dx, DiscretizationScheme, StochasticProcess, StochasticProcess1D,
};
use super::heston::HestonParams;
use super::local_vol::LocalVol;

// ── Black-Scholes / local volatility ────────────────────────────────────

/// Volatility dynamics along a path: constant (GBM) or Dupire local vol.
pub enum VolDynamics<'a> {
    Const(f64),
    Local(LocalVol<'a>),
}

/// Risk-neutral lognormal dynamics `dS = (r - q) S dt + sigma(S, t) S dW`
/// with constant or local volatility.
///
/// `evolve` is overridden so each step costs one volatility lookup shared
/// by drift and diffusion (a local-vol lookup is several surface
/// interpolations); the trait's coefficient methods expose the same
/// dynamics to generic consumers.
pub struct BlackScholesProcess<'a> {
    /// Risk-neutral drift rate `r - q` (of the *rate*, not `a(t,x)`).
    drift_rate: f64,
    vol: VolDynamics<'a>,
}

impl<'a> BlackScholesProcess<'a> {
    pub fn new(drift_rate: f64, vol: VolDynamics<'a>) -> Self {
        BlackScholesProcess { drift_rate, vol }
    }

    /// The volatility used to diffuse at level `s`, time `t`.
    pub fn vol(&self, s: f64, t: f64) -> f64 {
        match &self.vol {
            VolDynamics::Const(v) => *v,
            VolDynamics::Local(lv) => lv.vol(s, t),
        }
    }

    /// One step with an externally supplied volatility — for callers that
    /// already looked it up for their own purposes (the barrier engine
    /// needs `sigma` for the Brownian-bridge crossing probability).
    ///
    /// `Exact` is the closed-form lognormal transition under constant
    /// vol; under local vol it is the standard frozen-coefficient
    /// log-Euler step (not exact, but positivity-preserving and the
    /// usual choice). Milstein's `∂b/∂x` keeps the `S ∂sigma/∂S` term
    /// under local vol via the numeric derivative.
    pub(crate) fn step_with_vol(
        &self,
        scheme: DiscretizationScheme,
        t: f64,
        s: f64,
        dt: f64,
        dw: f64,
        sigma: f64,
    ) -> f64 {
        let mu = self.drift_rate;
        let next = match scheme {
            DiscretizationScheme::Exact => {
                s * exp((mu - 0.5 * sigma * sigma) * dt + sigma * dw)
            }
            DiscretizationScheme::Euler => s * (1.0 + mu * dt + sigma * dw),
            DiscretizationScheme::Milstein => {
                // b = sigma(S,t) S, so b' = sigma under constant vol and
                // sigma + S dsigma/dS under local vol
                let b_dx = self.diffusion_dx(t, s);
                s * (1.0 + mu * dt + sigma * dw + 0.5 * sigma * b_dx * (dw * dw - dt))
            }
        };
        next.max(0.0)
    }
}

impl StochasticProcess1D for BlackScholesProcess<'_> {
    fn drift(&self, _t: f64, x: f64) -> f64 {
        self.drift_rate * x
    }

    fn diffusion(&self, t: f64, x: f64) -> f64 {
        self.vol(x, t) * x
    }

    fn diffusion_dx(&self, t: f64, x: f64) -> f64 {
        match &self.vol {
            VolDynamics::Const(sigma) => *sigma,
            VolDynamics::Local(_) => numeric_diffusion_dx(self, t, x),
        }
    }

    fn exact_step(&self, t: f64, x: f64, dt: f64, dw: f64) -> Option<f64> {
        let sigma = self.vol(x, t);
        Some(x * exp((self.drift_rate - 0.5 * sigma * sigma) * dt + sigma * dw))
    }

    /// A lognormal spot cannot go negative; approximate schemes can.
    fn constrain(&self, x: f64) -> f64 {
        x.max(0.0)
    }

    fn evolve(&self, scheme: DiscretizationScheme, t: f64, x: f64, dt: f64, dw: f64) -> f64 {
        let sigma = self.vol(x, t);
        self.step_with_vol(scheme, t, x, dt, dw, sigma)
    }
}

// ── Heston stochastic volatility ────────────────────────────────────────

/// Heston-specific stepping schemes — model-owned, because good variance
/// stepping is genuinely model-specific and no generic Euler/Milstein
/// switch covers it (the QuantLib / TF Quant Finance pattern).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HestonScheme {
    /// Euler with the variance floored at zero inside the coefficients
    /// while the state keeps its (possibly negative) excursion; log-Euler
    /// spot stepping. Simple, O(dt) biased.
    FullTruncation,
    /// Andersen (2008) Quadratic-Exponential with martingale correction
    /// (QE-M): the variance transition is moment-matched to the exact
    /// non-central chi-squared law — a squared Gaussian where the
    /// distribution is peaked (`psi <= 1.5`), a mass-at-zero /
    /// exponential-tail mixture where it is not — and the spot step's
    /// constant is chosen so `E[S_{t+dt} | S_t, v_t]` is exact. Near
    /// bias-free even with coarse time grids.
    QuadraticExponential,
}

/// Andersen's switching threshold between the quadratic and exponential
/// variance samplers (any value in [1, 2] is valid; 1.5 is his choice).
const QE_PSI_SWITCH: f64 = 1.5;

/// Which QE sampler fired, with the parameters the martingale correction
/// needs.
enum QeBranch {
    /// `v_next = a (b + Z)^2`.
    Quadratic { a: f64, b2: f64 },
    /// Mass at zero with probability `p`, exponential tail of rate `beta`.
    Exponential { p: f64, beta: f64 },
}

/// One Andersen QE draw of the CIR variance transition, plus the branch
/// bookkeeping.
fn qe_variance_draw(hp: &HestonParams, v: f64, dt: f64, z_v: f64) -> (f64, QeBranch) {
    let (kappa, theta, xi) = (hp.kappa, hp.theta, hp.vol_of_vol);
    // conditional mean and variance of v_{t+dt} | v_t (exact CIR moments)
    let e = (-kappa * dt).exp();
    let m = theta + (v - theta) * e;
    let s2 = v * xi * xi * e * (1.0 - e) / kappa
        + theta * xi * xi * (1.0 - e) * (1.0 - e) / (2.0 * kappa);
    let psi = s2 / (m * m);

    if psi <= QE_PSI_SWITCH {
        // squared-Gaussian branch: v_next = a (b + Z)^2 matching (m, s2)
        let inv = 2.0 / psi;
        let b2 = inv - 1.0 + inv.sqrt() * (inv - 1.0).sqrt();
        let a = m / (1.0 + b2);
        let bz = b2.sqrt() + z_v;
        (a * bz * bz, QeBranch::Quadratic { a, b2 })
    } else {
        // mass-at-zero / exponential-tail branch, driven through the
        // normal's uniform so the caller's draw pipeline is unchanged
        let p = (psi - 1.0) / (psi + 1.0);
        let beta = (1.0 - p) / m;
        let u = crate::core::utils::norm_cdf(z_v);
        let v_next = if u <= p { 0.0 } else { ((1.0 - p) / (1.0 - u)).ln() / beta };
        (v_next, QeBranch::Exponential { p, beta })
    }
}

/// One Andersen QE draw of the CIR variance transition
/// `v_{t+dt} | v_t = v` from a standard normal `z_v` — the sampler
/// matches the exact conditional mean and variance of the square-root
/// process, switching between a squared-Gaussian and a
/// mass-at-zero/exponential form. Public for consumers that step the
/// variance leg on its own (the SLV engine pairs it with a
/// leverage-adjusted spot step).
pub fn qe_variance_step(hp: &HestonParams, v: f64, dt: f64, z_v: f64) -> f64 {
    qe_variance_draw(hp, v.max(0.0), dt, z_v).0
}

/// Heston dynamics as a 2-state, 2-factor process, state `[S, v]`:
///
/// ```text
/// dS = (r - q) S dt + sqrt(v) S dW_s
/// dv = kappa (theta - v) dt + vol_of_vol sqrt(v) dW_v
/// ```
///
/// `dw` carries **independent** increments; the spot/variance correlation
/// is applied inside — through the Cholesky rows of the diffusion matrix
/// for full truncation, analytically through the `K1`/`K2` coefficients
/// for QE. `evolve` overrides the generic Euler with the selected
/// [`HestonScheme`].
pub struct HestonProcess {
    /// Risk-neutral drift rate `r - q`.
    pub drift_rate: f64,
    pub params: HestonParams,
    pub scheme: HestonScheme,
}

impl StochasticProcess for HestonProcess {
    fn dim(&self) -> usize {
        2
    }

    fn factors(&self) -> usize {
        2
    }

    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let v_pos = x[1].max(0.0);
        out[0] = self.drift_rate * x[0];
        out[1] = self.params.kappa * (self.params.theta - v_pos);
    }

    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let sqrt_v = x[1].max(0.0).sqrt();
        let rho = self.params.rho;
        // row-major 2x2: spot row, then variance row (correlation folded
        // into the Cholesky structure)
        out[0] = sqrt_v * x[0];
        out[1] = 0.0;
        out[2] = self.params.vol_of_vol * sqrt_v * rho;
        out[3] = self.params.vol_of_vol * sqrt_v * (1.0 - rho * rho).sqrt();
    }

    fn evolve(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) {
        match self.scheme {
            HestonScheme::FullTruncation => self.step_full_truncation(x, dt, dw, out),
            HestonScheme::QuadraticExponential => self.step_qe(x, dt, dw, out),
        }
    }
}

impl HestonProcess {
    fn step_full_truncation(&self, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) {
        let hp = &self.params;
        let rho_perp = (1.0 - hp.rho * hp.rho).sqrt();
        let v_pos = x[1].max(0.0);
        let sqrt_v = v_pos.sqrt();
        let dw_v = hp.rho * dw[0] + rho_perp * dw[1];
        out[0] = x[0] * exp((self.drift_rate - 0.5 * v_pos) * dt + sqrt_v * dw[0]);
        out[1] = x[1] + hp.kappa * (hp.theta - v_pos) * dt + hp.vol_of_vol * sqrt_v * dw_v;
    }

    /// One Andersen QE-M step. `dw[0]` drives the spot, `dw[1]` the
    /// variance; both are plain Brownian increments, converted back to
    /// standard normals internally (the exponential branch further maps
    /// its normal to a uniform through the Gaussian CDF, so the caller's
    /// draw pipeline — antithetics included — is unchanged).
    fn step_qe(&self, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) {
        let hp = &self.params;
        let (kappa, theta, xi, rho) = (hp.kappa, hp.theta, hp.vol_of_vol, hp.rho);
        let v = x[1].max(0.0);
        let sqrt_dt = dt.sqrt();
        let z_s = dw[0] / sqrt_dt;
        let z_v = dw[1] / sqrt_dt;

        let (v_next, branch) = qe_variance_draw(hp, v, dt, z_v);

        // spot-step coefficients, central discretization (gamma1 = gamma2 = 1/2)
        let k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi;
        let k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi;
        let k3 = 0.5 * dt * (1.0 - rho * rho);
        let k4 = k3;
        // exponent in the martingale correction E[exp(a_mc * v_next)]
        let a_mc = k2 + 0.5 * k4;
        // plain-QE constant, the fallback where the correction's
        // moment-generating function does not exist
        let k0_plain = -rho * kappa * theta * dt / xi;

        let k0 = match branch {
            QeBranch::Quadratic { a, b2 } if 2.0 * a_mc * a < 1.0 => {
                -a_mc * b2 * a / (1.0 - 2.0 * a_mc * a)
                    + 0.5 * (1.0 - 2.0 * a_mc * a).ln()
                    - (k1 + 0.5 * k3) * v
            }
            QeBranch::Exponential { p, beta } if a_mc < beta => {
                -(p + beta * (1.0 - p) / (beta - a_mc)).ln() - (k1 + 0.5 * k3) * v
            }
            _ => k0_plain,
        };

        out[0] = x[0]
            * exp(self.drift_rate * dt
                + k0
                + k1 * v
                + k2 * v_next
                + (k3 * v + k4 * v_next).sqrt() * z_s);
        out[1] = v_next;
    }
}

// ── Correlated multi-asset lognormal dynamics ───────────────────────────

/// N correlated lognormal assets as one N-state, N-factor process:
///
/// ```text
/// dS_i = (r - q_i) S_i dt + sigma_i S_i dW_i,   d<W_i, W_j> = rho_ij dt
/// ```
///
/// The correlation enters through the rows of the lower-triangular
/// Cholesky factor, so `dw` carries **independent** increments (the
/// [`StochasticProcess`] contract). `evolve` overrides the generic Euler
/// with the exact per-asset lognormal transition — under constant
/// coefficients the joint law is exact at any step size, so coarse
/// grids only cost monitoring resolution, never bias.
pub struct MultiAssetGbmProcess {
    /// Per-asset risk-neutral drift rates `r - q_i`.
    pub drift_rates: Vec<f64>,
    pub vols: Vec<f64>,
    /// Lower-triangular Cholesky factor of the asset correlation matrix.
    pub chol: Vec<Vec<f64>>,
}

impl StochasticProcess for MultiAssetGbmProcess {
    fn dim(&self) -> usize {
        self.vols.len()
    }

    fn factors(&self) -> usize {
        self.vols.len()
    }

    fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        for i in 0..self.dim() {
            out[i] = self.drift_rates[i] * x[i];
        }
    }

    fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
        let n = self.dim();
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] =
                    if j <= i { self.vols[i] * x[i] * self.chol[i][j] } else { 0.0 };
            }
        }
    }

    fn evolve(&self, _t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) {
        for i in 0..self.dim() {
            // correlated Brownian increment of asset i
            let dwi: f64 = (0..=i).map(|j| self.chol[i][j] * dw[j]).sum();
            let sigma = self.vols[i];
            out[i] =
                x[i] * exp((self.drift_rates[i] - 0.5 * sigma * sigma) * dt + sigma * dwi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gbm() -> BlackScholesProcess<'static> {
        BlackScholesProcess::new(0.03, VolDynamics::Const(0.2))
    }

    #[test]
    fn coefficients_are_multiplicative() {
        let p = gbm();
        assert!((StochasticProcess1D::drift(&p, 0.0, 100.0) - 3.0).abs() < 1e-14);
        assert!((StochasticProcess1D::diffusion(&p, 0.0, 100.0) - 20.0).abs() < 1e-14);
        assert!((p.diffusion_dx(0.0, 100.0) - 0.2).abs() < 1e-14);
    }

    #[test]
    fn schemes_match_the_classical_gbm_formulas() {
        let p = gbm();
        let (s, dt, dw) = (100.0, 0.01, 0.05);
        let exact = p.evolve(DiscretizationScheme::Exact, 0.0, s, dt, dw);
        assert!((exact - s * ((0.03 - 0.5 * 0.04) * dt + 0.2 * dw).exp()).abs() < 1e-10);
        let euler = p.evolve(DiscretizationScheme::Euler, 0.0, s, dt, dw);
        assert!((euler - s * (1.0 + 0.03 * dt + 0.2 * dw)).abs() < 1e-12);
        let milstein = p.evolve(DiscretizationScheme::Milstein, 0.0, s, dt, dw);
        let expect =
            s * (1.0 + 0.03 * dt + 0.2 * dw + 0.5 * 0.04 * (dw * dw - dt));
        assert!((milstein - expect).abs() < 1e-12);
    }

    #[test]
    fn spot_is_floored_at_zero() {
        // a catastrophic Euler shock cannot take a lognormal spot negative
        let p = gbm();
        let next = p.evolve(DiscretizationScheme::Euler, 0.0, 100.0, 0.01, -10.0);
        assert_eq!(next, 0.0);
    }

    #[test]
    fn heston_evolve_matches_full_truncation_euler() {
        let hp = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.05, vol_of_vol: 0.5, rho: -0.7 };
        let p = HestonProcess {
            drift_rate: 0.02,
            params: hp,
            scheme: HestonScheme::FullTruncation,
        };
        let x = [100.0, -0.01]; // negative variance excursion
        let (dt, dw) = (0.004, [0.03, -0.02]);
        let mut out = [0.0; 2];
        p.evolve(0.0, &x, dt, &dw, &mut out);
        // v_pos = 0: the spot diffuses at zero vol, the variance keeps
        // its negative state and mean-reverts from the floored value
        assert!((out[0] - 100.0 * (0.02_f64 * dt).exp()).abs() < 1e-10);
        assert!((out[1] - (-0.01 + 1.5 * 0.05 * dt)).abs() < 1e-12);
    }

    #[test]
    fn heston_diffusion_matrix_encodes_the_correlation() {
        let hp = HestonParams { v0: 0.04, kappa: 1.5, theta: 0.05, vol_of_vol: 0.5, rho: -0.7 };
        let p = HestonProcess {
            drift_rate: 0.0,
            params: hp,
            scheme: HestonScheme::FullTruncation,
        };
        let mut b = [0.0; 4];
        StochasticProcess::diffusion(&p, 0.0, &[100.0, 0.04], &mut b);
        // spot row: [sqrt(v) S, 0]; variance row: xi sqrt(v) [rho, sqrt(1-rho^2)]
        assert!((b[0] - 20.0).abs() < 1e-12 && b[1] == 0.0);
        let xi_sv = 0.5 * 0.2;
        assert!((b[2] - xi_sv * -0.7).abs() < 1e-12);
        assert!((b[3] - xi_sv * (1.0_f64 - 0.49).sqrt()).abs() < 1e-12);
        // row 2 has squared norm (xi sqrt(v))^2 — the Cholesky property
        assert!((b[2] * b[2] + b[3] * b[3] - xi_sv * xi_sv).abs() < 1e-12);
    }

    // ── Andersen QE ─────────────────────────────────────────────────────

    /// Sample one QE step's variance from `v` many times and return the
    /// sample (mean, variance) of `v_next` plus `E[S_next] / S0`.
    fn qe_sample(hp: HestonParams, drift: f64, v: f64, dt: f64, n: usize) -> (f64, f64, f64) {
        use crate::core::montecarlo::path_normals;
        let p = HestonProcess {
            drift_rate: drift,
            params: hp,
            scheme: HestonScheme::QuadraticExponential,
        };
        let sqrt_dt = dt.sqrt();
        let (mut sum_v, mut sum_v2, mut sum_s) = (0.0, 0.0, 0.0);
        let mut out = [0.0; 2];
        let mut z = [0.0; 2];
        for i in 0..n {
            // independent per-sample streams: the spot and variance
            // normals must not be antithetic partners of each other
            path_normals(7, i as u64, &mut z);
            let dw = [sqrt_dt * z[0], sqrt_dt * z[1]];
            p.evolve(0.0, &[100.0, v], dt, &dw, &mut out);
            sum_v += out[1];
            sum_v2 += out[1] * out[1];
            sum_s += out[0];
        }
        let mean = sum_v / n as f64;
        (mean, sum_v2 / n as f64 - mean * mean, sum_s / n as f64 / 100.0)
    }

    /// Exact CIR conditional moments of `v_{t+dt} | v_t`.
    fn cir_moments(hp: &HestonParams, v: f64, dt: f64) -> (f64, f64) {
        let e = (-hp.kappa * dt).exp();
        let xi2 = hp.vol_of_vol * hp.vol_of_vol;
        let m = hp.theta + (v - hp.theta) * e;
        let s2 = v * xi2 * e * (1.0 - e) / hp.kappa
            + hp.theta * xi2 * (1.0 - e) * (1.0 - e) / (2.0 * hp.kappa);
        (m, s2)
    }

    #[test]
    fn qe_quadratic_branch_matches_the_cir_moments() {
        // small psi => squared-Gaussian sampler
        let hp = HestonParams { v0: 0.09, kappa: 2.0, theta: 0.09, vol_of_vol: 0.4, rho: -0.7 };
        let (v, dt) = (0.09, 0.01);
        let (m, s2) = cir_moments(&hp, v, dt);
        assert!(s2 / (m * m) <= QE_PSI_SWITCH, "test must hit the quadratic branch");
        let (mean, var, _) = qe_sample(hp, 0.0, v, dt, 200_000);
        assert!((mean - m).abs() / m < 0.01, "mean {mean} vs {m}");
        assert!((var - s2).abs() / s2 < 0.02, "var {var} vs {s2}");
    }

    #[test]
    fn qe_exponential_branch_matches_the_cir_moments() {
        // near-zero variance, high vol-of-vol, coarse step => large psi
        let hp = HestonParams { v0: 0.001, kappa: 0.5, theta: 0.04, vol_of_vol: 1.0, rho: -0.7 };
        let (v, dt) = (0.001, 1.0);
        let (m, s2) = cir_moments(&hp, v, dt);
        assert!(s2 / (m * m) > QE_PSI_SWITCH, "test must hit the exponential branch");
        let (mean, var, _) = qe_sample(hp, 0.0, v, dt, 200_000);
        assert!((mean - m).abs() / m < 0.02, "mean {mean} vs {m}");
        assert!((var - s2).abs() / s2 < 0.03, "var {var} vs {s2}");
    }

    #[test]
    fn qe_spot_step_is_a_martingale_in_both_branches() {
        // the martingale correction makes E[S_next | S, v] = S e^{mu dt}
        // exact; with mu = 0 the discounted spot ratio must be 1 up to
        // sampling noise
        let quad = HestonParams { v0: 0.09, kappa: 2.0, theta: 0.09, vol_of_vol: 0.4, rho: -0.7 };
        let (_, _, ratio) = qe_sample(quad, 0.0, 0.09, 0.05, 400_000);
        assert!((ratio - 1.0).abs() < 2e-3, "quadratic branch ratio {ratio}");

        let expo = HestonParams { v0: 0.001, kappa: 0.5, theta: 0.04, vol_of_vol: 1.0, rho: -0.7 };
        let (_, _, ratio) = qe_sample(expo, 0.0, 0.001, 1.0, 400_000);
        assert!((ratio - 1.0).abs() < 5e-3, "exponential branch ratio {ratio}");
    }

    #[test]
    fn qe_variance_is_never_negative() {
        let hp = HestonParams { v0: 0.02, kappa: 1.0, theta: 0.03, vol_of_vol: 0.9, rho: -0.5 };
        let p = HestonProcess {
            drift_rate: 0.0,
            params: hp,
            scheme: HestonScheme::QuadraticExponential,
        };
        let mut out = [0.0; 2];
        for z in [-4.0, -1.0, 0.0, 1.0, 4.0] {
            let dt: f64 = 0.02;
            let dw = [0.0, dt.sqrt() * z];
            p.evolve(0.0, &[100.0, 0.0001], dt, &dw, &mut out);
            assert!(out[1] >= 0.0, "z={z}: v_next={}", out[1]);
            assert!(out[0] > 0.0);
        }
    }

    // ── Multi-asset GBM ─────────────────────────────────────────────────

    fn two_asset_gbm(rho: f64) -> MultiAssetGbmProcess {
        MultiAssetGbmProcess {
            drift_rates: vec![0.03, 0.01],
            vols: vec![0.2, 0.3],
            chol: vec![vec![1.0, 0.0], vec![rho, (1.0 - rho * rho).sqrt()]],
        }
    }

    #[test]
    fn multi_gbm_recovers_forwards_and_correlation() {
        use crate::core::montecarlo::path_normals;
        let rho = -0.6;
        let p = two_asset_gbm(rho);
        let (dt, n) = (0.02_f64, 200_000);
        let mut z = [0.0; 2];
        let mut out = [0.0; 2];
        let (mut m0, mut m1) = (0.0, 0.0);
        let (mut c00, mut c11, mut c01) = (0.0, 0.0, 0.0);
        let sqrt_dt = dt.sqrt();
        for i in 0..n {
            path_normals(11, i as u64, &mut z);
            let dw = [sqrt_dt * z[0], sqrt_dt * z[1]];
            p.evolve(0.0, &[100.0, 50.0], dt, &dw, &mut out);
            m0 += out[0];
            m1 += out[1];
            let (l0, l1) = ((out[0] / 100.0).ln(), (out[1] / 50.0).ln());
            c00 += l0 * l0;
            c11 += l1 * l1;
            c01 += l0 * l1;
        }
        let nf = n as f64;
        let (t0, t1) = (100.0 * (0.03_f64 * dt).exp(), 50.0 * (0.01_f64 * dt).exp());
        assert!((m0 / nf - t0).abs() / t0 < 1e-3, "asset 0 forward {}", m0 / nf);
        assert!((m1 / nf - t1).abs() / t1 < 1e-3, "asset 1 forward {}", m1 / nf);
        // sample correlation of the log-returns (means are O(dt), ignorable)
        let corr = c01 / (c00 * c11).sqrt();
        assert!((corr - rho).abs() < 0.01, "log-return correlation {corr} vs {rho}");
    }

    #[test]
    fn multi_gbm_diffusion_matrix_is_the_scaled_cholesky() {
        let p = two_asset_gbm(0.5);
        let mut b = [0.0; 4];
        StochasticProcess::diffusion(&p, 0.0, &[100.0, 50.0], &mut b);
        assert!((b[0] - 0.2 * 100.0).abs() < 1e-12 && b[1] == 0.0);
        assert!((b[2] - 0.3 * 50.0 * 0.5).abs() < 1e-12);
        assert!((b[3] - 0.3 * 50.0 * 0.75_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn perfectly_correlated_identical_assets_move_in_lockstep() {
        let p = MultiAssetGbmProcess {
            drift_rates: vec![0.02, 0.02],
            vols: vec![0.25, 0.25],
            chol: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        let mut out = [0.0; 2];
        p.evolve(0.0, &[80.0, 80.0], 0.01, &[0.03, -0.4], &mut out);
        assert!((out[0] - out[1]).abs() < 1e-12);
    }
}
