//! Rough Bergomi (rBergomi) stochastic volatility
//! (Bayer-Friz-Gatheral 2016).
//!
//! Risk-neutral dynamics, `X = ln S`:
//!
//! ```text
//! dX   = (r - q - V/2) dt + sqrt(V) dW,   W = rho W^v + sqrt(1-rho^2) W_perp
//! V_t  = xi0 * exp( eta * VH(t) - eta^2/2 * t^{2H} )
//! VH(t) = sqrt(2H) int_0^t (t-s)^{H-1/2} dW^v_s
//! ```
//!
//! `VH` is the Riemann-Liouville Volterra process: Gaussian, centered,
//! with `Var[VH(t)] = t^{2H}`. For Hurst exponent `H < 1/2` its paths are
//! rougher than Brownian motion, which is what makes the model reproduce
//! the `T^{H-1/2}` explosion of the short-dated ATM implied-vol skew that
//! diffusive models (Heston, SABR) cannot. The `exp(...)`-compensated
//! form keeps `E[V_t] = xi0` exactly, so `xi0` **is** the flat forward
//! variance curve and the model's fair variance-swap strike.
//!
//! `VH` is **not** Markovian — no per-step [`StochasticProcess`]
//! (crate::core::montecarlo::process::StochasticProcess) recursion can
//! generate it — so simulation samples the whole vol leg jointly, by one
//! of two schemes behind [`RBergomiGenerator`]:
//!
//! - **Exact scheme** ([`RBergomiPaths`]): on the pricing grid,
//!   `(dW^v_1..n, VH(t_1)..VH(t_n))` is one jointly Gaussian vector
//!   whose covariance is known in closed/quadrature form; a Cholesky
//!   factor turns i.i.d. normals into exact joint draws — no
//!   discretization bias in the variance path at all. The factorization
//!   costs `O(steps^3)` once per pricing and each path costs
//!   `O(steps^2)`.
//! - **Hybrid scheme** ([`RBergomiHybrid`], Bennedsen-Lunde-Pakkanen
//!   2017): the kernel integral over the most recent step — where the
//!   `(t-s)^{H-1/2}` singularity lives — is drawn **exactly** from its
//!   known 2x2 joint law with the Brownian increment, and every older
//!   step enters through a first-moment-matched kernel weight, so the
//!   history term is one linear convolution of the increments,
//!   FFT-accelerated through a precomputed
//!   [`RealConvolver`](crate::core::fft::RealConvolver) plan: `O(steps)`
//!   setup and `O(steps log steps)` per path. The approximation error is
//!   tiny (the marginal variance is matched to a fraction of a percent)
//!   and vanishes as the grid refines.
//!
//! The Monte Carlo engine selects the exact scheme up to
//! [`RBergomiGenerator::EXACT_MAX_STEPS`] and the hybrid scheme on finer
//! grids, where the exact scheme's quadratic path cost would dominate.
//!
//! Pricing lives in the Monte Carlo engine
//! ([`montecarlo`](crate::equity::montecarlo)): European vanillas use the
//! conditional (Romano-Touzi / "mixed") estimator — conditional on the
//! vol path the spot is lognormal, so each path contributes a Black
//! price, integrating out the orthogonal spot noise analytically — and
//! every other payoff simulates full spot paths. There is no
//! characteristic function, so the COS/analytic routes do not apply.

use libm::exp;
use serde::{Deserialize, Serialize};

use crate::core::errors::RustyQLibError;
use crate::core::fft::{Complex, RealConvolver};
use crate::core::linalg::cholesky_factor;
use crate::core::trade::PutOrCall;
use crate::core::utils::norm_cdf;

// ── Model parameters ────────────────────────────────────────────────────

/// Rough Bergomi parameters. `xi0` is the (flat) forward variance —
/// `E[V_t] = xi0` for all `t` — `eta` the vol-of-vol, `hurst` the
/// roughness exponent `H` in `(0, 1/2]`, `rho` the spot-vol correlation.
///
/// `H = 1/2` degenerates to a driftless lognormal-vol model (the Volterra
/// process becomes a Brownian motion); the rough regime — and the model's
/// point — is `H` well below `1/2` (equity fits sit around 0.05–0.15).
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct RBergomiParams {
    /// Flat forward variance (initial instantaneous variance).
    #[serde(alias = "xi")]
    pub xi0: f64,
    /// Vol-of-vol (often written nu).
    #[serde(alias = "nu")]
    pub eta: f64,
    /// Hurst exponent `H` in `(0, 1/2]`.
    #[serde(alias = "h")]
    pub hurst: f64,
    /// Spot-vol correlation in `[-1, 1]` (equity fits are deeply negative).
    pub rho: f64,
}

impl RBergomiParams {
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.xi0 <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "rbergomi params",
                "rBergomi xi0 (forward variance) must be positive".to_string(),
            ));
        }
        if self.eta < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "rbergomi params",
                "rBergomi eta (vol-of-vol) must be non-negative".to_string(),
            ));
        }
        if !(self.hurst > 0.0 && self.hurst <= 0.5) {
            return Err(RustyQLibError::invalid_input(
                "rbergomi params",
                "rBergomi Hurst exponent must lie in (0, 1/2] — the rough regime \
                 (H = 1/2 is the lognormal-vol limit)"
                    .to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(RustyQLibError::invalid_input(
                "rbergomi params",
                "rBergomi rho must be in [-1, 1]".to_string(),
            ));
        }
        Ok(())
    }

    /// Parameters with a parallel shift applied to the forward vol
    /// `sqrt(xi0)` — the library's vega bump convention, mirroring
    /// [`HestonParams::with_vol_shift`](crate::equity::heston::HestonParams::with_vol_shift).
    pub fn with_vol_shift(&self, shift: f64) -> RBergomiParams {
        let vol = (self.xi0.sqrt() + shift).max(1e-6);
        RBergomiParams {
            xi0: vol * vol,
            ..*self
        }
    }

    /// Instantaneous variance at time `t` given the Volterra value
    /// `VH(t)`: `xi0 * exp(eta VH - eta^2/2 t^{2H})`. The compensator
    /// makes `E[V_t] = xi0` exact (`VH(t)` is centered Gaussian with
    /// variance `t^{2H}`), so `xi0` is also the model's fair
    /// variance-swap strike for every maturity.
    pub fn variance(&self, t: f64, volterra: f64) -> f64 {
        self.xi0 * exp(self.eta * volterra - 0.5 * self.eta * self.eta * t.powf(2.0 * self.hurst))
    }
}

// ── Volterra covariance structure ───────────────────────────────────────
//
// All second moments of (W^v, VH) have closed or one-dimensional
// quadrature form; with alpha = H + 1/2 and gamma = 1/2 - H:
//
//   Var[VH(t)]        = t^{2H}
//   Cov[VH(s), VH(t)] = 2H int_0^s (t-u)^{-gamma} (s-u)^{-gamma} du   (s <= t)
//   Cov[W^v_s, VH(t)] = sqrt(2H)/alpha * (t^alpha - (t - min(s,t))^alpha)

/// Composite Simpson on `[a, b]` with `n` (even) intervals.
fn simpson(a: f64, b: f64, n: usize, f: impl Fn(f64) -> f64) -> f64 {
    debug_assert!(n >= 2 && n.is_multiple_of(2));
    let h = (b - a) / n as f64;
    let mut s = f(a) + f(b);
    for i in 1..n {
        let w = if i.is_multiple_of(2) { 2.0 } else { 4.0 };
        s += w * f(a + i as f64 * h);
    }
    s * h / 3.0
}

/// `E[VH(s) VH(t)]` of the `sqrt(2H)`-normalized Riemann-Liouville
/// Volterra process.
///
/// After `u -> s - u`, the integral is
/// `int_0^s (delta + u)^{-gamma} u^{-gamma} du` with `delta = t - s`. The
/// `u^{-gamma}` endpoint singularity is removed exactly by `y = u^alpha`
/// (then `u^{-gamma} du = dy / alpha`); the integral is split at
/// `u = delta` so each piece is smooth on its own scale — the near piece
/// under the power substitution, the far piece (integrand `~ u^{-2gamma}`
/// over possibly many decades) on a log grid.
pub(crate) fn volterra_cov(hurst: f64, s: f64, t: f64) -> f64 {
    let (s, t) = if s <= t { (s, t) } else { (t, s) };
    if s <= 0.0 {
        return 0.0;
    }
    let two_h = 2.0 * hurst;
    let delta = t - s;
    if delta == 0.0 {
        return s.powf(two_h);
    }
    let gamma = 0.5 - hurst;
    if gamma == 0.0 {
        // H = 1/2: VH is a standard Brownian motion
        return s;
    }
    let alpha = hurst + 0.5;
    let m = delta.min(s);
    // near piece u in [0, m]: y = u^alpha
    let i1 = simpson(0.0, m.powf(alpha), 64, |y| {
        (delta + y.powf(1.0 / alpha)).powf(-gamma)
    }) / alpha;
    // far piece u in [m, s] on a log grid: u = e^y, u^{-gamma} du = e^{(1-gamma) y} dy
    let i2 = if s > m {
        let (a, b) = (m.ln(), s.ln());
        // resolution scales with the number of decades spanned
        let n = (((b - a) * 48.0).ceil() as usize).clamp(64, 512).div_ceil(2) * 2;
        simpson(a, b, n, |y| {
            let u = y.exp();
            (delta + u).powf(-gamma) * u.powf(1.0 - gamma)
        })
    } else {
        0.0
    };
    two_h * (i1 + i2)
}

/// `E[W^v_s VH(t)]` — the driving Brownian motion at `s` against the
/// Volterra value at `t` (closed form).
pub(crate) fn wv_volterra_cov(hurst: f64, s: f64, t: f64) -> f64 {
    let m = s.min(t);
    if m <= 0.0 || t <= 0.0 {
        return 0.0;
    }
    let alpha = hurst + 0.5;
    (2.0 * hurst).sqrt() / alpha * (t.powf(alpha) - (t - m).powf(alpha))
}

// ── Exact joint path generation ─────────────────────────────────────────

/// Exact-scheme generator for rough Bergomi paths on a uniform grid
/// `t_i = i * dt`, `i = 1..=steps`.
///
/// Holds the Cholesky factor of the `2n x 2n` covariance of the jointly
/// Gaussian vector `(dW^v_1, .., dW^v_n, VH(t_1), .., VH(t_n))` — the
/// driving Brownian **increments** first (their block is `dt * I`, which
/// keeps the factorization well-conditioned), the Volterra values second.
/// [`correlate`](Self::correlate) maps `2n` i.i.d. standard normals to
/// one exact joint draw; the marginal law of the variance path has no
/// time-discretization error.
pub struct RBergomiPaths {
    n: usize,
    /// Grid times `t_1..t_n`.
    times: Vec<f64>,
    /// Row-major dense lower-triangular Cholesky factor, `(2n)^2` long.
    chol: Vec<f64>,
}

impl RBergomiPaths {
    /// Build the joint covariance for `steps` uniform steps to `t` and
    /// factor it. Cost is `O(steps^2)` quadratures + one `O(steps^3)`
    /// Cholesky, paid once per pricing and shared by every path.
    // index loops mirror the block-matrix subscripts (`a[n+j][i]`);
    // iterator forms would obscure the layout
    #[allow(clippy::needless_range_loop)]
    pub fn new(hurst: f64, steps: usize, t: f64) -> Result<Self, RustyQLibError> {
        if steps == 0 || t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "rbergomi grid",
                "rBergomi path generation needs steps >= 1 and t > 0".to_string(),
            ));
        }
        let n = steps;
        let dt = t / n as f64;
        let times: Vec<f64> = (1..=n).map(|i| i as f64 * dt).collect();
        let dim = 2 * n;
        let mut a = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            a[i][i] = dt;
        }
        for j in 0..n {
            let tj = times[j];
            // increment i spans (i*dt, (i+1)*dt]; increments after t_j are
            // independent of VH(t_j)
            for i in 0..=j {
                let c = wv_volterra_cov(hurst, times[i], tj)
                    - if i == 0 {
                        0.0
                    } else {
                        wv_volterra_cov(hurst, times[i - 1], tj)
                    };
                a[n + j][i] = c;
                a[i][n + j] = c;
            }
            for k in 0..=j {
                let c = volterra_cov(hurst, times[k], tj);
                a[n + j][n + k] = c;
                a[n + k][n + j] = c;
            }
        }
        let l = match cholesky_factor(&a) {
            Ok(l) => l,
            Err(_) => {
                // quadrature rounding can push a tiny Schur pivot just
                // past the factorizer's tolerance; a ridge of 1e-10 on
                // the Volterra diagonal is far below the covariance
                // scale and restores positive semi-definiteness
                for j in 0..n {
                    a[n + j][n + j] += 1e-10 * (1.0 + a[n + j][n + j]);
                }
                cholesky_factor(&a)?
            }
        };
        let mut chol = vec![0.0; dim * dim];
        for (i, row) in l.iter().enumerate() {
            chol[i * dim..i * dim + i + 1].copy_from_slice(&row[..i + 1]);
        }
        Ok(RBergomiPaths { n, times, chol })
    }

    pub fn steps(&self) -> usize {
        self.n
    }

    /// Grid times `t_1..t_n`.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// One exact joint draw from `2n` i.i.d. standard normals
    /// (`z[..2n]`): `dwv` receives the `W^v` increments over each step,
    /// `volterra` the Volterra values at `t_1..t_n`. Negating `z`
    /// negates the whole draw (the Gaussian vector is centered), so
    /// antithetic pairing works exactly as for Brownian increments.
    pub fn correlate(&self, z: &[f64], dwv: &mut [f64], volterra: &mut [f64]) {
        let (n, dim) = (self.n, 2 * self.n);
        debug_assert!(z.len() >= dim && dwv.len() == n && volterra.len() == n);
        for (i, d) in dwv.iter_mut().enumerate() {
            let row = &self.chol[i * dim..i * dim + i + 1];
            *d = row.iter().zip(z).map(|(l, zk)| l * zk).sum();
        }
        for (j, v) in volterra.iter_mut().enumerate() {
            let row = &self.chol[(n + j) * dim..(n + j) * dim + n + j + 1];
            *v = row.iter().zip(z).map(|(l, zk)| l * zk).sum();
        }
    }
}

// ── Hybrid-scheme path generation (Bennedsen-Lunde-Pakkanen 2017) ──────

/// Hybrid-scheme generator for the rough Bergomi vol leg on a uniform
/// grid, the fine-grid alternative to the exact scheme.
///
/// Decompose `VH(t_i) = sqrt(2H) sum_k int_{t_{k-1}}^{t_k}
/// (t_i - s)^{H-1/2} dW^v_s`. The most recent interval (`k = i`), which
/// carries the kernel singularity, is drawn **exactly**: the pair
/// `(dW^v_i, W2_i)` of the Brownian increment and the kernel integral
/// over that interval is jointly Gaussian with closed-form 2x2
/// covariance. Every older interval is approximated by
/// `g_l * dW^v_{i-l}` with the first-moment-matched weight
///
/// ```text
/// g_l = (b_{l+1} dt)^{H-1/2},
/// b_k = ((k^{a+1} - (k-1)^{a+1}) / (a+1))^{1/a},   a = H - 1/2
/// ```
///
/// (`g_l dt` equals the kernel's average over its interval exactly), so
/// the history term is one linear convolution of the increments with a
/// fixed kernel — evaluated through a precomputed FFT plan
/// ([`RealConvolver`]) at `O(steps log steps)` per path. Setup is
/// `O(steps)` plus one kernel FFT, against the exact scheme's
/// `O(steps^3)` factorization.
pub struct RBergomiHybrid {
    n: usize,
    /// Read by the test-only marginal-variance diagnostic.
    #[cfg_attr(not(test), allow(dead_code))]
    hurst: f64,
    times: Vec<f64>,
    sqrt_2h: f64,
    /// Cholesky of the per-interval `(dW, W2)` law: `dW = l11 z_a`,
    /// `W2 = l21 z_a + l22 z_b`.
    l11: f64,
    l21: f64,
    l22: f64,
    conv: RealConvolver,
}

/// First-moment-matched hybrid lag weight `g_l` (zero at lag zero, where
/// the exact per-interval pair takes over): `g_l * dt` equals the kernel
/// `(t - s)^{H-1/2}` averaged over the interval at distance
/// `[l dt, (l+1) dt]`.
fn hybrid_lag_weight(hurst: f64, dt: f64, l: usize) -> f64 {
    if l == 0 {
        return 0.0;
    }
    let a = hurst - 0.5;
    if a == 0.0 {
        // H = 1/2: the kernel is identically one
        return 1.0;
    }
    let alpha = hurst + 0.5;
    let k = (l + 1) as f64;
    let b = ((k.powf(alpha) - (k - 1.0).powf(alpha)) / alpha).powf(1.0 / a);
    (b * dt).powf(a)
}

impl RBergomiHybrid {
    pub fn new(hurst: f64, steps: usize, t: f64) -> Result<Self, RustyQLibError> {
        if steps == 0 || t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "rbergomi grid",
                "rBergomi path generation needs steps >= 1 and t > 0".to_string(),
            ));
        }
        let n = steps;
        let dt = t / n as f64;
        let alpha = hurst + 0.5;
        // exact last-interval pair: Var[dW] = dt,
        // Cov[dW, W2] = dt^{a+1}/(a+1), Var[W2] = dt^{2a+1}/(2a+1)
        let l11 = dt.sqrt();
        let l21 = dt.powf(hurst) / alpha;
        let l22 = dt.powf(hurst) * (1.0 / (2.0 * hurst) - 1.0 / (alpha * alpha)).max(0.0).sqrt();
        let g: Vec<f64> = (0..n).map(|l| hybrid_lag_weight(hurst, dt, l)).collect();
        Ok(RBergomiHybrid {
            n,
            hurst,
            times: (1..=n).map(|i| i as f64 * dt).collect(),
            sqrt_2h: (2.0 * hurst).sqrt(),
            l11,
            l21,
            l22,
            conv: RealConvolver::new(&g, n),
        })
    }

    /// Grid times `t_1..t_n`.
    pub fn times(&self) -> &[f64] {
        &self.times
    }

    /// One hybrid draw from `2n` i.i.d. standard normals: `z[..n]`
    /// drives the Brownian increments, `z[n..2n]` the orthogonal part of
    /// the exact last-interval kernel integrals. Outputs match
    /// [`RBergomiPaths::correlate`]; `scratch` is the FFT work buffer,
    /// reusable across paths.
    pub fn correlate(
        &self,
        z: &[f64],
        dwv: &mut [f64],
        volterra: &mut [f64],
        scratch: &mut Vec<Complex>,
    ) {
        let n = self.n;
        debug_assert!(z.len() >= 2 * n && dwv.len() == n && volterra.len() == n);
        for (j, d) in dwv.iter_mut().enumerate() {
            *d = self.l11 * z[j];
        }
        // history term: volterra[i] first receives sum_l g[l] dW_{i-l}
        self.conv.convolve_into(dwv, volterra, scratch);
        for (i, v) in volterra.iter_mut().enumerate() {
            let w2 = self.l21 * z[i] + self.l22 * z[n + i];
            *v = self.sqrt_2h * (w2 + *v);
        }
    }

    /// Model variance of `VH(t_i)` (`i` 1-based) under the hybrid
    /// approximation — the exact last-interval variance plus the squared
    /// moment-matched weights (used to quantify the scheme's bias in
    /// tests).
    #[cfg(test)]
    fn marginal_variance(&self, i: usize) -> f64 {
        let dt = self.times[0];
        let c22 = self.l21 * self.l21 + self.l22 * self.l22;
        let hist: f64 = (1..i)
            .map(|l| {
                let g = hybrid_lag_weight(self.hurst, dt, l);
                g * g * dt
            })
            .sum();
        self.sqrt_2h * self.sqrt_2h * (c22 + hist)
    }
}

/// The vol-leg sampler behind the Monte Carlo engine: exact scheme up to
/// [`EXACT_MAX_STEPS`](Self::EXACT_MAX_STEPS) steps, hybrid scheme on
/// finer grids where the exact scheme's `O(steps^2)` per-path cost and
/// `O(steps^3)` setup would dominate. Both variants share the draw
/// layout (`z[..2n]` for the vol leg) and output contract, so pricing
/// code is scheme-agnostic.
pub enum RBergomiGenerator {
    Exact(RBergomiPaths),
    Hybrid(RBergomiHybrid),
}

impl RBergomiGenerator {
    /// Largest grid the exact scheme is selected for. At this size the
    /// per-path cost of the two schemes crosses over; beyond it the
    /// hybrid's `O(steps log steps)` wins outright and its bias is
    /// already far below Monte Carlo noise.
    pub const EXACT_MAX_STEPS: usize = 256;

    /// Scheme-selecting constructor for a `steps`-step grid to `t`.
    pub fn for_grid(hurst: f64, steps: usize, t: f64) -> Result<Self, RustyQLibError> {
        if steps <= Self::EXACT_MAX_STEPS {
            Ok(RBergomiGenerator::Exact(RBergomiPaths::new(
                hurst, steps, t,
            )?))
        } else {
            Ok(RBergomiGenerator::Hybrid(RBergomiHybrid::new(
                hurst, steps, t,
            )?))
        }
    }

    /// Grid times `t_1..t_n`.
    pub fn times(&self) -> &[f64] {
        match self {
            RBergomiGenerator::Exact(g) => g.times(),
            RBergomiGenerator::Hybrid(g) => g.times(),
        }
    }

    /// One joint vol-leg draw from `z[..2n]` i.i.d. standard normals;
    /// `scratch` is used by the hybrid variant only.
    pub fn correlate(
        &self,
        z: &[f64],
        dwv: &mut [f64],
        volterra: &mut [f64],
        scratch: &mut Vec<Complex>,
    ) {
        match self {
            RBergomiGenerator::Exact(g) => g.correlate(z, dwv, volterra),
            RBergomiGenerator::Hybrid(g) => g.correlate(z, dwv, volterra, scratch),
        }
    }
}

// ── Conditional Black price (the mixed-estimator kernel) ────────────────

/// Undiscounted Black price on a forward: the per-path kernel of the
/// conditional (Romano-Touzi) estimator — conditional on the vol path,
/// the terminal spot is lognormal around the leverage-adjusted forward
/// with the orthogonal share of the integrated variance.
pub(crate) fn black_on_forward(f: f64, k: f64, stdev: f64, put_or_call: PutOrCall) -> f64 {
    if stdev <= 1e-12 {
        return match put_or_call {
            PutOrCall::Call => (f - k).max(0.0),
            PutOrCall::Put => (k - f).max(0.0),
        };
    }
    let d1 = ((f / k).ln() + 0.5 * stdev * stdev) / stdev;
    let d2 = d1 - stdev;
    match put_or_call {
        PutOrCall::Call => f * norm_cdf(d1) - k * norm_cdf(d2),
        PutOrCall::Put => k * norm_cdf(-d2) - f * norm_cdf(-d1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::montecarlo::path_normals;

    // ── covariance quadrature ───────────────────────────────────────────

    /// Brute-force reference: midpoint rule on the globally
    /// singularity-removed form `(1/alpha) int_0^{s^alpha}
    /// (delta + y^{1/alpha})^{-gamma} dy` with a million points — a
    /// different rule and a different domain split than the production
    /// quadrature.
    fn cov_reference(hurst: f64, s: f64, t: f64) -> f64 {
        let (gamma, alpha) = (0.5 - hurst, hurst + 0.5);
        let delta = t - s;
        let b = s.powf(alpha);
        let n = 1_000_000usize;
        let h = b / n as f64;
        let sum: f64 = (0..n)
            .map(|i| {
                let y = (i as f64 + 0.5) * h;
                (delta + y.powf(1.0 / alpha)).powf(-gamma)
            })
            .sum();
        2.0 * hurst * sum * h / alpha
    }

    #[test]
    fn volterra_cov_matches_a_brute_force_reference() {
        for (h, s, t) in [
            (0.07, 0.5, 0.504), // adjacent grid times, very rough
            (0.07, 0.3, 1.2),   // wide separation, very rough
            (0.10, 0.004, 1.0), // first grid point vs maturity
            (0.30, 1.0, 1.1),
            (0.45, 0.2, 2.0), // near the lognormal limit
        ] {
            let got = volterra_cov(h, s, t);
            let want = cov_reference(h, s, t);
            assert!(
                (got - want).abs() < 1e-5 * want.max(1e-3),
                "H={h} s={s} t={t}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn volterra_cov_is_brownian_at_h_half() {
        for (s, t) in [(0.3, 0.7), (1.0, 1.0), (2.0, 0.5)] {
            assert!((volterra_cov(0.5, s, t) - s.min(t)).abs() < 1e-14);
        }
    }

    #[test]
    fn volterra_variance_is_t_to_the_2h() {
        for h in [0.07, 0.25, 0.5] {
            for t in [0.1, 1.0, 3.0] {
                assert!((volterra_cov(h, t, t) - t.powf(2.0 * h)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ordering_of_arguments_does_not_matter() {
        let a = volterra_cov(0.12, 0.4, 1.3);
        let b = volterra_cov(0.12, 1.3, 0.4);
        assert!((a - b).abs() < 1e-15);
    }

    // ── joint factorization ─────────────────────────────────────────────

    #[test]
    // index loops mirror the block-matrix subscripts, as in `new`
    #[allow(clippy::needless_range_loop)]
    fn factor_reconstructs_the_joint_covariance() {
        let (h, n, t) = (0.1, 16usize, 1.0);
        let gen = RBergomiPaths::new(h, n, t).unwrap();
        let dt = t / n as f64;
        let dim = 2 * n;
        let times = gen.times();
        // expected covariance, rebuilt from the analytic entries
        let mut a = vec![vec![0.0; dim]; dim];
        for i in 0..n {
            a[i][i] = dt;
        }
        for j in 0..n {
            for i in 0..=j {
                let c = wv_volterra_cov(h, times[i], times[j])
                    - if i == 0 {
                        0.0
                    } else {
                        wv_volterra_cov(h, times[i - 1], times[j])
                    };
                a[n + j][i] = c;
                a[i][n + j] = c;
            }
            for k in 0..=j {
                let c = volterra_cov(h, times[k], times[j]);
                a[n + j][n + k] = c;
                a[n + k][n + j] = c;
            }
        }
        for i in 0..dim {
            for j in 0..dim {
                let recon: f64 = (0..=i.min(j))
                    .map(|k| gen.chol[i * dim + k] * gen.chol[j * dim + k])
                    .sum();
                assert!(
                    (recon - a[i][j]).abs() < 1e-8,
                    "[{i}][{j}]: {recon} vs {}",
                    a[i][j]
                );
            }
        }
    }

    #[test]
    fn sampled_moments_match_the_analytic_covariance() {
        let (h, n, t) = (0.25, 8usize, 0.5);
        let gen = RBergomiPaths::new(h, n, t).unwrap();
        let dt = t / n as f64;
        let paths = 100_000usize;
        let mut z = vec![0.0; 2 * n];
        let mut dwv = vec![0.0; n];
        let mut vh = vec![0.0; n];
        let (mut var_vh, mut var_dw0, mut cross) = (0.0, 0.0, 0.0);
        for i in 0..paths {
            path_normals(17, i as u64, &mut z);
            gen.correlate(&z, &mut dwv, &mut vh);
            let wv_t: f64 = dwv.iter().sum();
            var_vh += vh[n - 1] * vh[n - 1];
            var_dw0 += dwv[0] * dwv[0];
            cross += wv_t * vh[n - 1];
        }
        let nf = paths as f64;
        let want_var = t.powf(2.0 * h);
        assert!(
            (var_vh / nf - want_var).abs() / want_var < 0.02,
            "Var[VH(T)] {} vs {want_var}",
            var_vh / nf
        );
        assert!(
            (var_dw0 / nf - dt).abs() / dt < 0.02,
            "Var[dW^v_1] {} vs {dt}",
            var_dw0 / nf
        );
        let want_cross = wv_volterra_cov(h, t, t);
        assert!(
            (cross / nf - want_cross).abs() / want_cross < 0.05,
            "Cov[W^v_T, VH(T)] {} vs {want_cross}",
            cross / nf
        );
    }

    #[test]
    fn compensated_variance_has_mean_xi0() {
        // E[V_t] = xi0 exactly — the exp compensator against the exact
        // Volterra marginal; checked by simulation at full vol-of-vol
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.0,
            hurst: 0.3,
            rho: -0.7,
        };
        let (n, t) = (16usize, 1.0);
        let gen = RBergomiPaths::new(params.hurst, n, t).unwrap();
        let paths = 200_000usize;
        let mut z = vec![0.0; 2 * n];
        let mut dwv = vec![0.0; n];
        let mut vh = vec![0.0; n];
        let mut sum_v = 0.0;
        for i in 0..paths {
            // antithetic pairs halve the lognormal skew's sampling noise
            path_normals(23, (i / 2) as u64, &mut z);
            if i % 2 == 1 {
                for zi in z.iter_mut() {
                    *zi = -*zi;
                }
            }
            gen.correlate(&z, &mut dwv, &mut vh);
            sum_v += params.variance(t, vh[n - 1]);
        }
        let mean = sum_v / paths as f64;
        assert!(
            (mean - params.xi0).abs() / params.xi0 < 0.01,
            "E[V_T] {mean} vs {}",
            params.xi0
        );
    }

    // ── parameters ──────────────────────────────────────────────────────

    #[test]
    fn validate_rejects_out_of_domain_parameters() {
        let good = RBergomiParams {
            xi0: 0.04,
            eta: 1.9,
            hurst: 0.07,
            rho: -0.9,
        };
        assert!(good.validate().is_ok());
        assert!(RBergomiParams { xi0: 0.0, ..good }.validate().is_err());
        assert!(RBergomiParams { eta: -0.1, ..good }.validate().is_err());
        assert!(RBergomiParams { hurst: 0.0, ..good }.validate().is_err());
        assert!(RBergomiParams { hurst: 0.6, ..good }.validate().is_err());
        assert!(RBergomiParams { rho: -1.5, ..good }.validate().is_err());
    }

    #[test]
    fn vol_shift_bumps_the_forward_vol() {
        let p = RBergomiParams {
            xi0: 0.04,
            eta: 1.9,
            hurst: 0.07,
            rho: -0.9,
        };
        let bumped = p.with_vol_shift(0.01);
        assert!((bumped.xi0 - 0.21f64 * 0.21).abs() < 1e-15);
        assert_eq!(bumped.eta, p.eta);
        assert_eq!(bumped.hurst, p.hurst);
        assert_eq!(bumped.rho, p.rho);
    }

    #[test]
    fn variance_at_time_zero_is_xi0() {
        let p = RBergomiParams {
            xi0: 0.09,
            eta: 2.0,
            hurst: 0.1,
            rho: 0.0,
        };
        assert!((p.variance(0.0, 0.0) - 0.09).abs() < 1e-15);
    }

    #[test]
    fn params_deserialize_with_paper_aliases() {
        let p: RBergomiParams =
            serde_json::from_str(r#"{"xi": 0.055225, "nu": 1.9, "h": 0.07, "rho": -0.9}"#).unwrap();
        assert!((p.xi0 - 0.055225).abs() < 1e-15);
        assert!((p.eta - 1.9).abs() < 1e-15);
        assert!((p.hurst - 0.07).abs() < 1e-15);
    }

    // ── hybrid scheme ───────────────────────────────────────────────────

    #[test]
    fn hybrid_is_brownian_at_h_half() {
        // H = 1/2 collapses every weight to one and the exact pair to
        // the plain increment, so VH must be the running sum of dW —
        // an algebraic identity that pins down the convolution indexing
        let n = 32;
        let gen = RBergomiHybrid::new(0.5, n, 1.0).unwrap();
        let mut z = vec![0.0; 2 * n];
        path_normals(31, 0, &mut z);
        let (mut dwv, mut vh) = (vec![0.0; n], vec![0.0; n]);
        let mut scratch = Vec::new();
        gen.correlate(&z, &mut dwv, &mut vh, &mut scratch);
        let mut cum = 0.0;
        for i in 0..n {
            cum += dwv[i];
            assert!((vh[i] - cum).abs() < 1e-10, "step {i}: {} vs {cum}", vh[i]);
        }
    }

    #[test]
    fn hybrid_marginal_variance_tracks_the_power_law() {
        // the scheme's only bias: Jensen on the moment-matched weights
        // pulls Var[VH(t_i)] slightly below t_i^{2H}; it must stay
        // within a fraction of a percent on a practical grid
        for h in [0.07, 0.1, 0.3] {
            let n = 64;
            let gen = RBergomiHybrid::new(h, n, 1.0).unwrap();
            for i in [n / 2, n] {
                let t_i = gen.times()[i - 1];
                let want = t_i.powf(2.0 * h);
                let got = gen.marginal_variance(i);
                assert!(
                    got <= want && (want - got) / want < 0.005,
                    "H={h} t={t_i}: hybrid {got} vs exact {want}"
                );
            }
        }
    }

    #[test]
    fn generator_switches_schemes_at_the_threshold() {
        let at = RBergomiGenerator::for_grid(0.1, RBergomiGenerator::EXACT_MAX_STEPS, 1.0).unwrap();
        assert!(matches!(at, RBergomiGenerator::Exact(_)));
        let above =
            RBergomiGenerator::for_grid(0.1, RBergomiGenerator::EXACT_MAX_STEPS + 1, 1.0).unwrap();
        assert!(matches!(above, RBergomiGenerator::Hybrid(_)));
        assert_eq!(above.times().len(), RBergomiGenerator::EXACT_MAX_STEPS + 1);
    }

    /// Conditional (mixed) European call price off a given vol-leg
    /// generator — the scheme-comparison harness (spot 100, rate 2%).
    fn mixed_call_price(
        gen: &RBergomiGenerator,
        params: &RBergomiParams,
        k: f64,
        t: f64,
        paths: usize,
        seed: u64,
    ) -> f64 {
        let (s0, r) = (100.0, 0.02);
        let n = gen.times().len();
        let dt = t / n as f64;
        let df = exp(-r * t);
        let forward = s0 * exp(r * t);
        let (mut z, mut dwv, mut vh) = (vec![0.0; 2 * n], vec![0.0; n], vec![0.0; n]);
        let mut scratch = Vec::new();
        let mut sum = 0.0;
        for i in 0..paths {
            path_normals(seed, (i / 2) as u64, &mut z);
            if i % 2 == 1 {
                for zi in z.iter_mut() {
                    *zi = -*zi;
                }
            }
            gen.correlate(&z, &mut dwv, &mut vh, &mut scratch);
            let mut v_left = params.variance(0.0, 0.0);
            let (mut stoch_int, mut int_var) = (0.0, 0.0);
            for j in 0..n {
                stoch_int += v_left.sqrt() * dwv[j];
                int_var += v_left * dt;
                v_left = params.variance(gen.times()[j], vh[j]);
            }
            let f_cond =
                forward * exp(params.rho * stoch_int - 0.5 * params.rho * params.rho * int_var);
            let stdev = ((1.0 - params.rho * params.rho) * int_var).sqrt();
            sum += df * black_on_forward(f_cond, k, stdev, PutOrCall::Call);
        }
        sum / paths as f64
    }

    #[test]
    fn hybrid_and_exact_schemes_price_alike() {
        // same model, same grid, both generators: prices must agree to
        // Monte Carlo noise (the hybrid's variance bias is far smaller)
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.5,
            hurst: 0.1,
            rho: -0.7,
        };
        let (n, t, paths) = (128usize, 1.0, 10_000usize);
        let exact = RBergomiGenerator::Exact(RBergomiPaths::new(params.hurst, n, t).unwrap());
        let hybrid = RBergomiGenerator::Hybrid(RBergomiHybrid::new(params.hurst, n, t).unwrap());
        let pe = mixed_call_price(&exact, &params, 105.0, t, paths, 7);
        let ph = mixed_call_price(&hybrid, &params, 105.0, t, paths, 11);
        assert!((pe - ph).abs() < 0.45, "exact={pe} hybrid={ph}");
    }

    // ── Monte Carlo engine integration ──────────────────────────────────

    use crate::core::traits::Instrument;
    use crate::core::utils::ContractStyle;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use crate::equity::vanilla_option::EquityOption;

    fn engine_option(
        params: RBergomiParams,
        put_or_call: PutOrCall,
        strike: f64,
        t: f64,
        rate: f64,
        paths: usize,
    ) -> EquityOption {
        EquityOptionBuilder::new()
            .spot(100.0)
            .strike(strike)
            .flat_vol(params.xi0.sqrt())
            .flat_rate(rate)
            .years_to_maturity(t)
            .vanilla(put_or_call)
            .engine(Engine::MonteCarlo)
            .rbergomi(params)
            .paths(paths)
            .build()
            .expect("rbergomi option must build")
    }

    #[test]
    fn degenerate_dynamics_reduce_to_black_scholes_exactly() {
        // eta = 0 makes the variance deterministic (V = xi0) and rho = 0
        // removes the leverage adjustment, so every path of the mixed
        // estimator contributes the identical Black price: the Monte
        // Carlo value must equal the analytic engine to float precision
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 0.0,
            hurst: 0.3,
            rho: 0.0,
        };
        let mc = engine_option(params, PutOrCall::Call, 105.0, 1.0, 0.03, 64).npv();
        let analytic = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(105.0)
            .flat_vol(0.2)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        assert!((mc - analytic).abs() < 1e-9, "mc={mc} analytic={analytic}");
    }

    #[test]
    fn mixed_estimator_agrees_with_pathwise_simulation() {
        // the engine's conditional (Romano-Touzi) estimator against an
        // independent full-path simulation on the same exact scheme and
        // grid — same model, two estimators
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.5,
            hurst: 0.1,
            rho: -0.7,
        };
        let (t, k, r, steps) = (1.0, 105.0, 0.02, 100usize);
        let paths = 20_000usize;
        let option = engine_option(params, PutOrCall::Call, k, t, r, paths);
        let mixed = option.npv();

        let gen = RBergomiPaths::new(params.hurst, steps, t).unwrap();
        let dt = t / steps as f64;
        let sqrt_dt = dt.sqrt();
        let rho_perp = (1.0f64 - params.rho * params.rho).sqrt();
        let df = exp(-r * t);
        let mut z = vec![0.0; 3 * steps];
        let mut dwv = vec![0.0; steps];
        let mut vh = vec![0.0; steps];
        let mut sum = 0.0;
        for i in 0..paths {
            path_normals(99, (i / 2) as u64, &mut z);
            if i % 2 == 1 {
                for zi in z.iter_mut() {
                    *zi = -*zi;
                }
            }
            gen.correlate(&z, &mut dwv, &mut vh);
            let mut s = 100.0f64;
            let mut v_left = params.variance(0.0, 0.0);
            for j in 0..steps {
                let dw = params.rho * dwv[j] + rho_perp * sqrt_dt * z[2 * steps + j];
                s *= exp((r - 0.5 * v_left) * dt + v_left.sqrt() * dw);
                v_left = params.variance(gen.times()[j], vh[j]);
            }
            sum += df * (s - k).max(0.0);
        }
        let plain = sum / paths as f64;
        assert!(
            (mixed - plain).abs() < 0.35,
            "mixed={mixed} pathwise={plain}"
        );
    }

    #[test]
    fn rough_smile_shows_the_leverage_skew() {
        // the Bayer-Friz-Gatheral SPX-like fit: deeply negative rho and
        // a small Hurst exponent must produce a steep negative smile —
        // the model's reason to exist
        let params = RBergomiParams {
            xi0: 0.235 * 0.235,
            eta: 1.9,
            hurst: 0.07,
            rho: -0.9,
        };
        let t = 0.5;
        let put = engine_option(params, PutOrCall::Put, 90.0, t, 0.0, 20_000).npv();
        let call = engine_option(params, PutOrCall::Call, 110.0, t, 0.0, 20_000).npv();
        let iv_put = crate::equity::blackscholes::implied_vol_from_price(
            100.0,
            90.0,
            0.0,
            0.0,
            t,
            put,
            PutOrCall::Put,
        )
        .unwrap();
        let iv_call = crate::equity::blackscholes::implied_vol_from_price(
            100.0,
            110.0,
            0.0,
            0.0,
            t,
            call,
            PutOrCall::Call,
        )
        .unwrap();
        assert!(
            iv_put - iv_call > 0.05,
            "90% put vol {iv_put} vs 110% call vol {iv_call}: skew too flat"
        );
    }

    #[test]
    fn fine_grids_price_through_the_hybrid_scheme() {
        // above EXACT_MAX_STEPS the engine switches generators; the
        // price must stay consistent with a coarse exact-scheme grid
        // (differences: spot discretization + Monte Carlo noise)
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.5,
            hurst: 0.1,
            rho: -0.7,
        };
        let build = |steps: usize| {
            EquityOptionBuilder::new()
                .spot(100.0)
                .strike(105.0)
                .flat_vol(0.2)
                .flat_rate(0.02)
                .years_to_maturity(1.0)
                .vanilla(PutOrCall::Call)
                .engine(Engine::MonteCarlo)
                .rbergomi(params)
                .paths(10_000)
                .mc_time_steps(steps)
                .build()
                .unwrap()
                .npv()
        };
        let coarse_exact = build(100);
        let fine_hybrid = build(300);
        assert!(
            (coarse_exact - fine_hybrid).abs() < 0.45,
            "exact(100)={coarse_exact} hybrid(300)={fine_hybrid}"
        );
    }

    #[test]
    fn knocked_out_barrier_is_bounded_by_the_vanilla() {
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.5,
            hurst: 0.1,
            rho: -0.7,
        };
        let vanilla = engine_option(params, PutOrCall::Call, 100.0, 0.5, 0.02, 10_000).npv();
        let barrier = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.2)
            .flat_rate(0.02)
            .years_to_maturity(0.5)
            .barrier(
                PutOrCall::Call,
                crate::equity::barrier::BarrierDirection::Up,
                crate::equity::barrier::KnockType::Out,
                130.0,
            )
            .engine(Engine::MonteCarlo)
            .rbergomi(params)
            .paths(10_000)
            .build()
            .unwrap()
            .npv();
        assert!(barrier > 0.0, "up-and-out call {barrier} must retain value");
        assert!(
            barrier < vanilla,
            "up-and-out {barrier} must be worth less than the vanilla {vanilla}"
        );
    }

    #[test]
    fn unsupported_combinations_are_rejected_at_build() {
        let params = RBergomiParams {
            xi0: 0.04,
            eta: 1.9,
            hurst: 0.07,
            rho: -0.9,
        };
        // American exercise: the exercise rule would need the whole
        // variance history
        let american = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.2)
            .flat_rate(0.02)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Put)
            .exercise_style(ContractStyle::American)
            .engine(Engine::MonteCarlo)
            .rbergomi(params)
            .build();
        assert!(matches!(
            american,
            Err(RustyQLibError::UnsupportedEngine(_))
        ));
        // non-Monte-Carlo engines have no non-Markovian route
        for engine in [Engine::BlackScholes, Engine::Binomial, Engine::FiniteDifference] {
            let built = EquityOptionBuilder::new()
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.2)
                .flat_rate(0.02)
                .years_to_maturity(1.0)
                .vanilla(PutOrCall::Call)
                .engine(engine)
                .rbergomi(params)
                .build();
            assert!(
                matches!(built, Err(RustyQLibError::UnsupportedEngine(_))),
                "{:?} must be rejected",
                built.map(|o| o.npv())
            );
        }
    }
}
