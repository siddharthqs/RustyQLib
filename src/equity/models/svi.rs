//! SVI and SSVI implied-volatility parameterizations (Gatheral 2004;
//! Gatheral & Jacquier 2014).
//!
//! **SVI** (raw form) parameterizes one expiry's total variance in
//! log-moneyness `k = ln(K/F)`:
//!
//! ```text
//! w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]
//! ```
//!
//! five parameters per smile: level `a`, wing slope `b`, skew `rho`,
//! shift `m`, ATM curvature `sigma`. Wings are asymptotically linear
//! with slopes `b(1 - rho)` (put side) and `b(1 + rho)` (call side).
//!
//! **SSVI** parameterizes the whole surface from the ATM total-variance
//! term structure `theta_t` and three global parameters `(rho, eta,
//! gamma)` through the power-law curvature
//! `phi(theta) = eta / (theta^gamma (1 + theta)^(1-gamma))`:
//!
//! ```text
//! w(k, t) = theta_t/2 [ 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ]
//! ```
//!
//! SVI smiles calibrate through the pluggable [`SviCalibration`]
//! methods — Zeliade's quasi-explicit reduction by default, plain or
//! polishing Levenberg-Marquardt on request — while SSVI calibrates by
//! Levenberg-Marquardt
//! ([`core::optimization`](crate::core::optimization)) in a transformed
//! parameter space, the same pattern as
//! [`heston::calibrate`](crate::equity::heston::calibrate). Butterfly
//! arbitrage is checked through the Gatheral-Jacquier `g(k)` density
//! condition (SVI) and the power-law sufficient conditions (SSVI), and
//! fitted smiles sample into the pricing
//! [`VolSurface`](crate::core::vols::VolSurface) via
//! [`Ssvi::to_vol_surface`].

use chrono::NaiveDate;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::numerics::solve_dense;
use crate::core::optimization::{levenberg_marquardt, nelder_mead, OptimConfig};
use crate::core::vols::{VolError, VolSurface};
use crate::equity::smoothed_surface::{
    interpolate_slices, SmoothedSurface, VarianceDerivatives, MIN_TIME,
};

// ── SVI: one expiry ─────────────────────────────────────────────────────

/// Raw SVI parameters for a single expiry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

/// Result of an SVI smile calibration (`rmse` in implied vol).
pub type SviFit = crate::equity::models::calibration::Fit<SviParams>;

/// The calibration algorithm [`SviParams::calibrate_with`] runs.
///
/// - [`QuasiExplicit`](Self::QuasiExplicit) (the default): Zeliade's
///   quasi-explicit calibration (De Marco & Martini 2009). For fixed
///   `(m, sigma)` the smile is *linear* in `(a, d, c) = (a, rho b
///   sigma, b sigma)`, so those three solve exactly as a tiny
///   constrained least squares and only `(m, sigma)` need a numerical
///   search — a 2-D landscape mild enough for a grid seed plus
///   Nelder-Mead. Deterministic and start-point free, and the
///   constraint box (`a >= 0`, `|d| <= c`, `c + |d| <= 4 sigma / t`)
///   enforces non-negative variance and Lee's wing bound by
///   construction — which matters when fits are repeated under bumped
///   surfaces for Greeks, where basin-hopping between fits shows up as
///   noise.
/// - [`QuasiExplicitThenLm`](Self::QuasiExplicitThenLm): the
///   quasi-explicit stage finds the basin, then Levenberg-Marquardt
///   polishes all five parameters for the last fraction of residual.
///   The polish is unconstrained (transformed parameters stay
///   admissible, but the fit may leave the arbitrage box above).
/// - [`LevenbergMarquardt`](Self::LevenbergMarquardt): single-start
///   Levenberg-Marquardt from a heuristic guess — the fastest, but
///   exposed to local minima on strongly skewed or sparse smiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SviCalibration {
    #[default]
    QuasiExplicit,
    QuasiExplicitThenLm,
    LevenbergMarquardt,
}

impl SviParams {
    /// Total variance `w(k)` at log-moneyness `k = ln(K/F)`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let d = k - self.m;
        self.a + self.b * (self.rho * d + (d * d + self.sigma * self.sigma).sqrt())
    }

    /// Implied vol at log-moneyness `k` for expiry `t`.
    pub fn vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k).max(0.0) / t).sqrt()
    }

    /// Static parameter constraints: `b >= 0`, `|rho| < 1`, `sigma > 0`
    /// and non-negative minimum variance `a + b sigma sqrt(1 - rho^2)`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.b < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "b must be non-negative",
            ));
        }
        if !(-1.0..1.0).contains(&self.rho) || self.rho == -1.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "rho must be in (-1, 1)",
            ));
        }
        if self.sigma <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "sigma must be positive",
            ));
        }
        if self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "minimum total variance is negative",
            ));
        }
        Ok(())
    }

    /// Total variance and its first two log-moneyness derivatives at
    /// `k` — all closed-form, which is what makes SVI-based Dupire
    /// local vol smooth by construction.
    pub fn variance_derivatives(&self, k: f64) -> (f64, f64, f64) {
        let d = k - self.m;
        let root = (d * d + self.sigma * self.sigma).sqrt();
        let w = self.a + self.b * (self.rho * d + root);
        let w1 = self.b * (self.rho + d / root);
        let w2 = self.b * self.sigma * self.sigma / (root * root * root);
        (w, w1, w2)
    }

    /// The Gatheral-Jacquier butterfly function
    /// `g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2`,
    /// which must stay non-negative for an arbitrage-free density.
    pub fn butterfly_g(&self, k: f64) -> f64 {
        let (w, w1, w2) = self.variance_derivatives(k);
        (1.0 - k * w1 / (2.0 * w)).powi(2) - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    }

    /// Minimum of `g(k)` over a wide log-moneyness scan; negative means
    /// the smile carries butterfly arbitrage.
    pub fn min_butterfly_g(&self) -> f64 {
        (0..=800)
            .map(|i| self.butterfly_g(-2.0 + i as f64 * 0.005))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn has_butterfly_arbitrage(&self) -> bool {
        self.min_butterfly_g() < 0.0
    }

    /// Calibrate to one expiry's quotes `(k, implied vol)` with the
    /// default method ([`SviCalibration::QuasiExplicit`]).
    pub fn calibrate(quotes: &[(f64, f64)], t: f64) -> SviFit {
        Self::calibrate_with(quotes, t, SviCalibration::default())
    }

    /// Calibrate to one expiry's quotes `(k, implied vol)` on
    /// total-variance residuals with the chosen [`SviCalibration`]
    /// (`rmse` reported in implied vol).
    pub fn calibrate_with(quotes: &[(f64, f64)], t: f64, method: SviCalibration) -> SviFit {
        Self::calibrate_weighted(quotes, t, method, &vec![1.0; quotes.len()])
    }

    /// [`calibrate_with`](Self::calibrate_with) with per-quote weights
    /// on the total-variance residuals: the objective becomes
    /// `sum_i weights[i] (w_fit(k_i) - w_i)^2` and `rmse` the
    /// weight-averaged implied-vol error, so uniform weights reproduce
    /// `calibrate_with` exactly and a zero weight excludes its quote.
    /// To weight *implied-vol* errors by `u_i` instead (vega or
    /// spread weighting), pass `weights[i] = u_i / (2 v_i t)^2` — the
    /// delta-method conversion between the two residual spaces.
    /// Weights must be finite and non-negative with a positive sum.
    pub fn calibrate_weighted(
        quotes: &[(f64, f64)],
        t: f64,
        method: SviCalibration,
        weights: &[f64],
    ) -> SviFit {
        assert!(
            quotes.len() >= 5,
            "SVI has five parameters; need at least five quotes"
        );
        assert!(t > 0.0);
        assert_eq!(
            weights.len(),
            quotes.len(),
            "one weight per quote (got {} weights for {} quotes)",
            weights.len(),
            quotes.len()
        );
        assert!(
            weights.iter().all(|w| w.is_finite() && *w >= 0.0),
            "weights must be finite and non-negative"
        );
        assert!(
            weights.iter().filter(|w| **w > 0.0).count() >= 5,
            "a zero weight excludes its quote: need at least five with positive weight"
        );
        let weight_sum: f64 = weights.iter().sum();
        let w_target: Vec<(f64, f64)> = quotes.iter().map(|&(k, v)| (k, v * v * t)).collect();
        let (params, iterations, converged) = match method {
            SviCalibration::QuasiExplicit => Self::fit_qe(&w_target, weights, t),
            SviCalibration::QuasiExplicitThenLm => {
                let (start, _, _) = Self::fit_qe(&w_target, weights, t);
                Self::fit_lm(&w_target, weights, Some(start))
            }
            SviCalibration::LevenbergMarquardt => Self::fit_lm(&w_target, weights, None),
        };
        let rmse = (quotes
            .iter()
            .zip(weights)
            .map(|(&(k, v), &wt)| wt * (params.vol(k, t) - v).powi(2))
            .sum::<f64>()
            / weight_sum)
            .sqrt();
        SviFit {
            params,
            rmse,
            iterations,
            converged,
        }
    }

    /// Levenberg-Marquardt on the total-variance targets (residuals
    /// scaled by the root of each quote's weight), with `b` and `sigma`
    /// in log space and `rho` through `tanh` so every trial is
    /// admissible; `start` seeds the search (heuristic guess when
    /// absent).
    fn fit_lm(
        w_target: &[(f64, f64)],
        weights: &[f64],
        start: Option<SviParams>,
    ) -> (SviParams, usize, bool) {
        let scale: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
        let x0 = match start {
            Some(p) => vec![
                p.a,
                p.b.max(1e-8).ln(),
                p.rho.clamp(-1.0 + 1e-9, 1.0 - 1e-9).atanh(),
                p.m,
                p.sigma.ln(),
            ],
            None => {
                let (w_min, w_max) = w_target
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, w)| {
                        (lo.min(w), hi.max(w))
                    });
                let k_at_min = w_target
                    .iter()
                    .fold(
                        (0.0, f64::INFINITY),
                        |acc, &(k, w)| if w < acc.1 { (k, w) } else { acc },
                    )
                    .0;
                let k_span = w_target.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max)
                    - w_target.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
                // start: level at the observed floor, gentle wings, no skew
                vec![
                    0.5 * w_min,                                          // a
                    (((w_max - w_min) / k_span.max(0.1)).max(1e-3)).ln(), // ln b
                    0.0,                                                  // atanh rho
                    k_at_min,                                             // m
                    0.2_f64.ln(),                                         // ln sigma
                ]
            }
        };
        let unpack = |u: &[f64]| SviParams {
            a: u[0],
            b: u[1].exp(),
            rho: u[2].tanh(),
            m: u[3],
            sigma: u[4].exp(),
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            let p = unpack(u);
            w_target
                .iter()
                .zip(&scale)
                .map(|(&(k, w), &s)| s * (p.total_variance(k) - w))
                .collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        (unpack(&fit.x), fit.iterations, fit.converged)
    }

    /// Zeliade's quasi-explicit fit on the total-variance targets: a
    /// deterministic grid seed then Nelder-Mead over `(m, ln sigma)`,
    /// with `(a, d, c)` solved exactly per trial by [`qe_inner`].
    fn fit_qe(w_target: &[(f64, f64)], weights: &[f64], t: f64) -> (SviParams, usize, bool) {
        let (k_min, k_max) = w_target
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(k, _)| {
                (lo.min(k), hi.max(k))
            });
        let span = (k_max - k_min).max(0.1);
        let objective = |m: f64, sigma: f64| {
            qe_inner(w_target, weights, m, sigma, 4.0 * sigma / t)
                .map(|(_, sse)| sse)
                .unwrap_or(f64::INFINITY)
        };
        // seed: m across the quoted span (widened by half a span each
        // side), sigma log-spaced over the plausible curvature range
        let mut cells: Vec<(f64, f64, f64)> = Vec::new(); // (sse, m, sigma)
        for i in 0..=14 {
            let m = k_min - 0.5 * span + 2.0 * span * i as f64 / 14.0;
            for j in 0..=9 {
                let sigma = 0.01 * 200f64.powf(j as f64 / 9.0); // 0.01 .. 2
                cells.push((objective(m, sigma), m, sigma));
            }
        }
        cells.sort_by(|x, y| x.0.total_cmp(&y.0));
        // refine from the best grid cells in normalized coordinates:
        // u = (1, 1) at the seed keeps Nelder-Mead's proportional first
        // step spanning a real fraction of the cell whatever the seed's
        // magnitude (an m of 0.0 would otherwise get a ~1e-4 simplex
        // that cannot travel), and the restart re-inflates a collapsed
        // simplex
        let cfg = OptimConfig::new(1e-13, 400);
        let mut iterations = 0;
        let mut best: Option<(f64, f64, f64, bool)> = None; // (sse, m, sigma, converged)
        for &(seed_sse, m0, s0) in cells.iter().take(3) {
            if !seed_sse.is_finite() {
                continue;
            }
            let obj = |u: &[f64]| {
                let m = m0 + span * (u[0] - 1.0);
                let sigma = (s0 * (u[1] - 1.0).exp()).clamp(1e-4, 10.0);
                objective(m, sigma)
            };
            let first = nelder_mead(&cfg, &obj, &[1.0, 1.0]);
            let nm = nelder_mead(&cfg, &obj, &first.x);
            iterations += first.iterations + nm.iterations;
            let m = m0 + span * (nm.x[0] - 1.0);
            let sigma = (s0 * (nm.x[1] - 1.0).exp()).clamp(1e-4, 10.0);
            if best.is_none_or(|(sse, ..)| nm.value < sse) {
                best = Some((nm.value, m, sigma, nm.converged));
            }
        }
        let (_, m, sigma, converged) = best.unwrap_or((cells[0].0, cells[0].1, cells[0].2, false));
        let ([a, d, c], m, sigma) = qe_inner(w_target, weights, m, sigma, 4.0 * sigma / t)
            .map(|(x, _)| (x, m, sigma))
            .or_else(|| {
                let (_, m0, s0) = cells[0];
                qe_inner(w_target, weights, m0, s0, 4.0 * s0 / t).map(|(x, _)| (x, m0, s0))
            })
            .expect("quasi-explicit inner solve failed on the seed grid");
        let (b, rho) = if c <= 1e-12 {
            (0.0, 0.0) // flat slice: the skew direction is undetermined
        } else {
            (c / sigma, (d / c).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        };
        (
            SviParams {
                a: a.max(0.0),
                b,
                rho,
                m,
                sigma,
            },
            iterations,
            converged,
        )
    }
}

/// The quasi-explicit inner solve (De Marco & Martini 2009): for fixed
/// `(m, sigma)` total variance is linear in `x = (a, d, c)` with
/// `d = rho b sigma`, `c = b sigma`, so the best fit is a tiny convex
/// least squares (per-quote weights on the squared residuals) over the
/// no-arbitrage box `a in [0, max w]`, `c >= 0`, `|d| <= c` and
/// `c + |d| <= slope_cap` (Lee's wing bound, `slope_cap = 4 sigma / t`
/// on total variance). Solved exactly by active-set enumeration: the
/// unconstrained normal equations first, otherwise every KKT system
/// over the seven constraint faces, keeping the feasible candidate with
/// the smallest sum of squares. Returns `(x, sse)` against the
/// total-variance targets.
fn qe_inner(
    w_target: &[(f64, f64)],
    weights: &[f64],
    m: f64,
    sigma: f64,
    slope_cap: f64,
) -> Option<([f64; 3], f64)> {
    let rows: Vec<[f64; 3]> = w_target
        .iter()
        .map(|&(k, _)| {
            let y = (k - m) / sigma;
            [1.0, y, (y * y + 1.0).sqrt()]
        })
        .collect();
    // the level box only spans quotes the fit actually sees
    let w_max = w_target
        .iter()
        .zip(weights)
        .filter(|(_, &wt)| wt > 0.0)
        .fold(0.0f64, |acc, (&(_, w), _)| acc.max(w));
    // weighted normal matrix q = G^T W G and right-hand side g = G^T W w
    let mut q = [[0.0; 3]; 3];
    let mut g = [0.0; 3];
    for ((row, &(_, w)), &wt) in rows.iter().zip(w_target).zip(weights) {
        for i in 0..3 {
            g[i] += wt * row[i] * w;
            for j in 0..3 {
                q[i][j] += wt * row[i] * row[j];
            }
        }
    }
    // faces n.x <= h of the constraint box on x = (a, d, c)
    let faces: [([f64; 3], f64); 7] = [
        ([-1.0, 0.0, 0.0], 0.0),       // a >= 0
        ([1.0, 0.0, 0.0], w_max),      // a <= max w
        ([0.0, 0.0, -1.0], 0.0),       // c >= 0   (b >= 0)
        ([0.0, 1.0, -1.0], 0.0),       // d <= c   (rho <= 1)
        ([0.0, -1.0, -1.0], 0.0),      // -d <= c  (rho >= -1)
        ([0.0, 1.0, 1.0], slope_cap),  // c + d <= cap (Lee, call wing)
        ([0.0, -1.0, 1.0], slope_cap), // c - d <= cap (Lee, put wing)
    ];
    let feasible = |x: &[f64; 3]| {
        faces
            .iter()
            .all(|&(n, h)| n[0] * x[0] + n[1] * x[1] + n[2] * x[2] <= h + 1e-9 * (1.0 + h.abs()))
    };
    let sse = |x: &[f64; 3]| -> f64 {
        rows.iter()
            .zip(w_target)
            .zip(weights)
            .map(|((row, &(_, w)), &wt)| {
                let e = row[0] * x[0] + row[1] * x[1] + row[2] * x[2] - w;
                wt * e * e
            })
            .sum()
    };
    let mut best: Option<([f64; 3], f64)> = None;
    for mask in 0u32..(1 << faces.len()) {
        if mask.count_ones() > 3 {
            continue; // three unknowns: more active faces is redundant
        }
        let active: Vec<usize> = (0..faces.len()).filter(|i| mask >> i & 1 == 1).collect();
        // stationarity on the active faces: [2q N^T; N 0](x, lambda) = (2g, h)
        let n = 3 + active.len();
        let mut lhs = vec![vec![0.0; n]; n];
        let mut rhs = vec![0.0; n];
        for i in 0..3 {
            rhs[i] = 2.0 * g[i];
            for j in 0..3 {
                lhs[i][j] = 2.0 * q[i][j];
            }
        }
        for (r, &face) in active.iter().enumerate() {
            let (normal, h) = faces[face];
            for j in 0..3 {
                lhs[3 + r][j] = normal[j];
                lhs[j][3 + r] = normal[j];
            }
            rhs[3 + r] = h;
        }
        let Some(sol) = solve_dense(&mut lhs, &mut rhs) else {
            continue;
        };
        let x = [sol[0], sol[1], sol[2]];
        if !(x.iter().all(|v| v.is_finite()) && feasible(&x)) {
            continue;
        }
        let value = sse(&x);
        if mask == 0 {
            return Some((x, value)); // interior optimum: global by convexity
        }
        if best.is_none_or(|(_, b)| value < b) {
            best = Some((x, value));
        }
    }
    best
}

// ── Per-expiry SVI surface fit ──────────────────────────────────────────

/// One fitted expiry slice of a [`SviSurfaceFit`].
#[derive(Debug, Clone)]
pub struct SviSlice {
    /// Expiry time (year fraction).
    pub t: f64,
    /// Forward the slice's log-moneyness is measured against.
    pub forward: f64,
    pub params: SviParams,
    /// Fit error in implied vol against the input pillars.
    pub rmse: f64,
    pub converged: bool,
    /// The quoted log-moneyness span the fit is anchored on.
    pub k_range: (f64, f64),
    /// Minimum of Gatheral's `g(k)` over the quoted span; negative
    /// means the *fit itself* carries butterfly arbitrage there.
    pub min_g: f64,
}

/// A per-expiry SVI fit of an implied surface: one [`SviParams`] smile
/// per pillar expiry, linear total variance in time between slices at
/// fixed log-moneyness (with a forward-variance floor for calendar
/// safety), and **analytic** Dupire local vol from SVI's closed-form
/// derivatives.
///
/// This is the smoother, where
/// [`repair_arbitrage`](crate::equity::surface_repair::repair_arbitrage)
/// is the repair: every point moves a little (by the fit RMSE), in
/// exchange for a C^2 smile that Dupire can differentiate without the
/// spikes piecewise-linear interpolation produces. Fit it to the
/// *cleaned* surface so outright arbitrage is gone before smoothing.
#[derive(Debug, Clone)]
pub struct SviSurfaceFit {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    /// Slices in increasing expiry order.
    pub slices: Vec<SviSlice>,
    /// Input expiries skipped for having fewer than five pillar quotes.
    pub skipped_slices: usize,
    /// Largest total-variance decrease between adjacent fitted slices
    /// over the quoted span (0 = calendar-clean fit); evaluation floors
    /// forward variance, so this measures fit tension, not arbitrage in
    /// the output.
    pub max_calendar_crossing: f64,
}

impl SviSurfaceFit {
    /// Fit one SVI smile per pillar expiry of `surface` (its per-expiry
    /// point smiles, on any coordinate). `forward` maps expiry time to
    /// the underlying's forward, exactly as for
    /// [`VolSurface::diagnostics`](crate::core::vols::VolSurface::diagnostics).
    /// Expiries with fewer than five pillars (SVI has five parameters)
    /// are skipped and counted. Slices calibrate with the default
    /// [`SviCalibration`]; [`fit_with`](Self::fit_with) picks another.
    pub fn fit(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
    ) -> Result<SviSurfaceFit, RustyQLibError> {
        Self::fit_with(surface, forward, SviCalibration::default())
    }

    /// [`fit`](Self::fit) with an explicit per-slice [`SviCalibration`].
    pub fn fit_with(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
        method: SviCalibration,
    ) -> Result<SviSurfaceFit, RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "svi fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };
        let mut slices = Vec::new();
        let mut skipped = 0usize;
        for (tenor, smile) in expiries.iter().zip(&smiles) {
            let t = match tenor {
                Tenor::YearFraction(t) => *t,
                Tenor::Date(_) => continue, // to_input never emits dates
            };
            if smile.len() < 5 {
                skipped += 1;
                continue;
            }
            let f = forward(t);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let k = match coordinate {
                        SmileCoordinate::Strike => (x / f).ln(),
                        SmileCoordinate::Moneyness => x.ln(),
                        SmileCoordinate::LogMoneyness => x,
                    };
                    (k, vol)
                })
                .collect();
            let fit = SviParams::calibrate_with(&quotes, t, method);
            let (k_lo, k_hi) = quotes
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &(k, _)| {
                    (lo.min(k), hi.max(k))
                });
            let min_g = (0..=200)
                .map(|i| {
                    fit.params
                        .butterfly_g(k_lo + (k_hi - k_lo) * i as f64 / 200.0)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(SviSlice {
                t,
                forward: f,
                params: fit.params,
                rmse: fit.rmse,
                converged: fit.converged,
                k_range: (k_lo, k_hi),
                min_g,
            });
        }
        if slices.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "svi fit",
                format!("no expiry has the five quotes an SVI fit needs ({skipped} skipped)"),
            ));
        }
        slices.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        // fit tension: does total variance ever fall between slices?
        let mut max_crossing: f64 = 0.0;
        for pair in slices.windows(2) {
            let (lo, hi) = (
                pair[0].k_range.0.min(pair[1].k_range.0),
                pair[0].k_range.1.max(pair[1].k_range.1),
            );
            for i in 0..=100 {
                let k = lo + (hi - lo) * i as f64 / 100.0;
                let crossing = pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
                max_crossing = max_crossing.max(crossing);
            }
        }
        Ok(SviSurfaceFit {
            reference_date: surface.reference_date(),
            day_count: surface.day_count(),
            slices,
            skipped_slices: skipped,
            max_calendar_crossing: max_crossing,
        })
    }

    /// The slice pair bracketing `t`, with the interpolation weight on
    /// the later slice (0 at or below the earlier, 1 at or beyond the
    /// later; a single-slice surface brackets with itself).
    fn bracket(&self, t: f64) -> (&SviSlice, &SviSlice, f64) {
        let n = self.slices.len();
        if n == 1 || t <= self.slices[0].t {
            return (&self.slices[0], &self.slices[0], 0.0);
        }
        if t >= self.slices[n - 1].t {
            return (&self.slices[n - 1], &self.slices[n - 1], 0.0);
        }
        let idx = self.slices.partition_point(|s| s.t < t);
        let (a, b) = (&self.slices[idx - 1], &self.slices[idx]);
        (a, b, (t - a.t) / (b.t - a.t))
    }

    /// Forward at `t`: linear between the slice forwards, flat outside.
    pub fn forward(&self, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        a.forward + (b.forward - a.forward) * weight
    }

    /// Total variance at log-moneyness `k`: linear in time between
    /// slices at fixed `k`, floored to be non-decreasing; proportional
    /// to `t` below the first slice (variance accrues from zero); beyond
    /// the last slice the smile shape is held and variance keeps
    /// accruing at the last segment's forward rate, so implied vol tends
    /// to a level instead of decaying like `1/sqrt(t)`.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        let wa = a.params.total_variance(k);
        let wb = b.params.total_variance(k).max(wa);
        let w = if t <= a.t {
            wa * (t / a.t).min(1.0)
        } else if a.t == b.t {
            // beyond the last pillar (the bracket clamps to it)
            wb + self.last_segment_dwdt(k) * (t - b.t)
        } else {
            wa + (wb - wa) * weight
        };
        w.max(0.0)
    }

    /// Implied vol for an absolute `strike` at `t`.
    pub fn vol(&self, strike: f64, t: f64) -> f64 {
        let k = (strike / self.forward(t)).ln();
        (self.total_variance(k, t).max(1e-12) / t.max(1e-8)).sqrt()
    }

    /// Analytic Dupire local vol at underlying `level` and time `t`:
    /// Gatheral's formula with `w`, `w_k`, `w_kk` in closed form from
    /// the bracketing SVI slices (interpolated linearly in time) and
    /// `dw/dt` as the floored forward variance between them. Clamped to
    /// the same `[1%, 300%]` band as the numerical
    /// [`LocalVol`](crate::equity::local_vol::LocalVol).
    pub fn local_vol(&self, level: f64, t: f64) -> f64 {
        self.local_vol_checked(level, t).0
    }

    /// [`local_vol`](Self::local_vol) plus whether a guard fired
    /// (`true` = implied vol was returned instead of the Dupire value:
    /// vanishing variance or a non-positive density denominator).
    ///
    /// Delegates to the shared
    /// [`SmoothedSurface`](crate::equity::smoothed_surface::SmoothedSurface)
    /// path, so SVI, SABR and SSVI reach Dupire's formula, the guards and
    /// the clamps through identical code.
    pub fn local_vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        <Self as SmoothedSurface>::local_vol_checked(self, level, t)
    }

    /// Forward variance of the last inter-slice segment at `k` (floored
    /// at zero for calendar safety), used to extrapolate beyond the
    /// final pillar (a single-slice fit accrues its variance from zero
    /// instead, extending the slice at flat implied vol).
    fn last_segment_dwdt(&self, k: f64) -> f64 {
        let n = self.slices.len();
        if n == 1 {
            let s = &self.slices[0];
            return s.params.total_variance(k) / s.t;
        }
        let (prev, last) = (&self.slices[n - 2], &self.slices[n - 1]);
        ((last.params.total_variance(k) - prev.params.total_variance(k)) / (last.t - prev.t))
            .max(0.0)
    }

    /// Sample the fit into the canonical pricing [`VolSurface`]: per
    /// slice, `samples` strikes across its own quoted log-moneyness
    /// span, through the floored [`Self::total_variance`] so the
    /// calendar floor is baked into the artifact. The sampled surface
    /// serializes, plots and prices like any other; Dupire should use
    /// [`Self::local_vol`] directly, which stays analytic. (Sub-basis-
    /// point calendar crossings can survive in the sampled wings where
    /// grazing weekly fits overlap — the diagnostics in the build
    /// metadata report them; the analytic path floors them.)
    pub fn to_vol_surface(&self, samples: usize) -> Result<VolSurface, VolError> {
        let expiries: Vec<Tenor> = self
            .slices
            .iter()
            .map(|s| Tenor::YearFraction(s.t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = self
            .slices
            .iter()
            .map(|slice| {
                let (lo, hi) = slice.k_range;
                let n = samples.max(5);
                (0..n)
                    .map(|i| {
                        let k = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                        // sample through the floored accessor, so the
                        // calendar floor between grazing fitted slices
                        // is baked into the sampled artifact too
                        let vol = (self.total_variance(k, slice.t).max(1e-12) / slice.t).sqrt();
                        (slice.forward * k.exp(), vol)
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, self.reference_date, self.day_count)
    }

    /// Sample [`Self::local_vol`] on a `levels` x `times` grid
    /// (`grid[i][j]` = level i, time j — the plotting layout).
    pub fn local_vol_grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        <Self as SmoothedSurface>::local_vol_grid(self, levels, times)
    }

    /// Fit-quality metadata for the surface document: per-slice params,
    /// RMSE in vol basis points, convergence, `min g`, and the global
    /// calendar-tension figure.
    pub fn metadata(&self) -> serde_json::Value {
        let slices: Vec<serde_json::Value> = self
            .slices
            .iter()
            .map(|s| {
                serde_json::json!({
                    "t": s.t,
                    "forward": s.forward,
                    "params": {
                        "a": s.params.a, "b": s.params.b, "rho": s.params.rho,
                        "m": s.params.m, "sigma": s.params.sigma,
                    },
                    "rmse_vol_bps": s.rmse * 1e4,
                    "converged": s.converged,
                    "min_butterfly_g": s.min_g,
                    "k_range": [s.k_range.0, s.k_range.1],
                })
            })
            .collect();
        serde_json::json!({
            "model": "per-expiry raw SVI (Gatheral), linear total variance in time",
            "slices": slices,
            "skipped_slices": self.skipped_slices,
            "max_calendar_crossing": self.max_calendar_crossing,
        })
    }
}

/// SVI reaches local volatility with **closed-form** strike derivatives
/// (from [`SviParams::variance_derivatives`]) and a **per-expiry** time
/// structure: independent slices interpolated linearly in total variance,
/// so `dw/dt` is piecewise constant and the local volatility steps in
/// time at each pillar. Both properties are the model's, not the
/// implementation's — everything downstream is shared.
impl SmoothedSurface for SviSurfaceFit {
    fn forward(&self, t: f64) -> f64 {
        SviSurfaceFit::forward(self, t)
    }

    fn total_variance(&self, k: f64, t: f64) -> f64 {
        SviSurfaceFit::total_variance(self, k, t)
    }

    fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives {
        let t = t.max(MIN_TIME);
        let (a, b, weight) = self.bracket(t);
        let (wa, wa1, wa2) = a.params.variance_derivatives(k);
        let (wb, wb1, wb2) = b.params.variance_derivatives(k);
        interpolate_slices(
            (a.t, [wa, wa1, wa2]),
            (b.t, [wb, wb1, wb2]),
            t,
            weight,
            || self.last_segment_dwdt(k),
        )
    }
}

// ── SSVI: the whole surface ─────────────────────────────────────────────

/// SSVI surface: ATM total-variance pillars plus global `(rho, eta,
/// gamma)` with the power-law curvature.
#[derive(Debug, Clone)]
pub struct Ssvi {
    pub rho: f64,
    pub eta: f64,
    /// Power-law exponent in `(0, 1]`.
    pub gamma: f64,
    /// `(t, theta_t)` pillars, `t` and `theta` strictly increasing.
    pub theta_pillars: Vec<(f64, f64)>,
}

/// Result of an SSVI calibration.
#[derive(Debug, Clone)]
pub struct SsviFit {
    pub surface: Ssvi,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl Ssvi {
    /// ATM total variance at `t`: proportional below the first pillar
    /// (variance accrues from zero), linear between pillars, and
    /// continued with the last segment's slope beyond.
    pub fn theta(&self, t: f64) -> f64 {
        let p = &self.theta_pillars;
        let n = p.len();
        if t <= 0.0 {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1 * t / p[0].0;
        }
        if t >= p[n - 1].0 {
            if n == 1 {
                return p[0].1 * t / p[0].0;
            }
            let slope = (p[n - 1].1 - p[n - 2].1) / (p[n - 1].0 - p[n - 2].0);
            return p[n - 1].1 + slope * (t - p[n - 1].0);
        }
        let idx = p.partition_point(|&(ti, _)| ti < t);
        let (t0, w0) = p[idx - 1];
        let (t1, w1) = p[idx];
        w0 + (w1 - w0) * (t - t0) / (t1 - t0)
    }

    /// `d theta / dt`, the slope of the ATM total-variance term
    /// structure — the time derivative [`SmoothedSurface`] needs.
    /// Mirrors [`Self::theta`] branch for branch, so it is **piecewise
    /// constant**: linear interpolation between pillars means the slope
    /// steps at each one. SSVI's `dw/dt` therefore also steps at pillars,
    /// but only by a factor uniform in `k` (the shape factor
    /// `dw/dtheta` stays continuous), where a per-expiry smoother's
    /// forward variance changes shape across the seam.
    pub fn theta_slope(&self, t: f64) -> f64 {
        let p = &self.theta_pillars;
        let n = p.len();
        if t <= 0.0 {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1 / p[0].0;
        }
        if t >= p[n - 1].0 {
            if n == 1 {
                return p[0].1 / p[0].0;
            }
            return (p[n - 1].1 - p[n - 2].1) / (p[n - 1].0 - p[n - 2].0);
        }
        let idx = p.partition_point(|&(ti, _)| ti < t);
        let (t0, w0) = p[idx - 1];
        let (t1, w1) = p[idx];
        (w1 - w0) / (t1 - t0)
    }

    /// Total variance and its derivatives at `(k, t)`, **all in closed
    /// form**. Writing `u = phi k + rho` and `R = sqrt(u^2 + 1 - rho^2)`:
    ///
    /// ```text
    /// w    = theta/2 (1 + rho phi k + R)
    /// w_k  = theta phi / 2 (rho + u/R)
    /// w_kk = theta phi^2 (1 - rho^2) / (2 R^3)
    /// w_t  = (dtheta/dt) [ w/theta + k (phi'/phi) w_k ]
    /// ```
    ///
    /// the last line following from the chain rule through `phi(theta)`,
    /// whose logarithmic derivative for the power law
    /// \eqref-free form `phi = eta / (theta^gamma (1+theta)^(1-gamma))`
    /// is `phi'/phi = -[gamma/theta + (1-gamma)/(1+theta)]`.
    pub fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives {
        let theta = self.theta(t);
        if theta <= 0.0 {
            return VarianceDerivatives {
                w: 0.0,
                dk: 0.0,
                dkk: 0.0,
                dt: 0.0,
            };
        }
        let phi = self.phi(theta);
        let rho = self.rho;
        let u = phi * k + rho;
        let r = (u * u + 1.0 - rho * rho).sqrt();
        let w = 0.5 * theta * (1.0 + rho * phi * k + r);
        let dk = 0.5 * theta * phi * (rho + u / r);
        let dkk = 0.5 * theta * phi * phi * (1.0 - rho * rho) / (r * r * r);
        // phi'/phi for the power-law curvature
        let dlog_phi = -(self.gamma / theta + (1.0 - self.gamma) / (1.0 + theta));
        let dw_dtheta = w / theta + k * dlog_phi * dk;
        VarianceDerivatives {
            w,
            dk,
            dkk,
            dt: self.theta_slope(t) * dw_dtheta,
        }
    }

    /// Power-law curvature `phi(theta)`.
    pub fn phi(&self, theta: f64) -> f64 {
        self.eta / (theta.powf(self.gamma) * (1.0 + theta).powf(1.0 - self.gamma))
    }

    /// Total variance `w(k, t)`.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let theta = self.theta(t);
        if theta <= 0.0 {
            return 0.0;
        }
        let phi = self.phi(theta);
        let pk = phi * k;
        0.5 * theta
            * (1.0 + self.rho * pk + ((pk + self.rho).powi(2) + 1.0 - self.rho * self.rho).sqrt())
    }

    /// Implied vol for `strike` given the `forward` at expiry `t`.
    pub fn vol(&self, strike: f64, forward: f64, t: f64) -> f64 {
        (self.total_variance((strike / forward).ln(), t) / t).sqrt()
    }

    /// Static no-arbitrage checks (Gatheral-Jacquier): admissible
    /// parameters, nondecreasing `theta` (calendar), the power-law
    /// sufficient condition `eta (1 + |rho|) <= 2`, and the per-pillar
    /// butterfly bounds `theta phi (1 + |rho|) <= 4` and
    /// `theta phi^2 (1 + |rho|) <= 4`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if !(-1.0..1.0).contains(&self.rho) || self.rho == -1.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "rho must be in (-1, 1)",
            ));
        }
        if self.eta <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "eta must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&self.gamma) || self.gamma == 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "gamma must be in (0, 1]",
            ));
        }
        if self.theta_pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "need at least one theta pillar",
            ));
        }
        if self
            .theta_pillars
            .iter()
            .any(|&(t, w)| t <= 0.0 || w <= 0.0)
        {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "theta pillars must have positive times and variances",
            ));
        }
        if self
            .theta_pillars
            .windows(2)
            .any(|p| p[1].0 <= p[0].0 || p[1].1 < p[0].1)
        {
            return Err(RustyQLibError::invalid_input("svi params", "theta pillars must be increasing in time and nondecreasing in variance (calendar arbitrage)"));
        }
        if self.eta * (1.0 + self.rho.abs()) > 2.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "eta (1 + |rho|) must not exceed 2 (static arbitrage)",
            ));
        }
        for &(_, theta) in &self.theta_pillars {
            let phi = self.phi(theta);
            if theta * phi * (1.0 + self.rho.abs()) > 4.0
                || theta * phi * phi * (1.0 + self.rho.abs()) > 4.0
            {
                return Err(RustyQLibError::invalid_input(
                    "svi params",
                    "butterfly bound violated at a theta pillar",
                ));
            }
        }
        Ok(())
    }

    /// Calibrate `(rho, eta, gamma)` to surface quotes `(t, k, vol)`
    /// given the ATM total-variance pillars, by Levenberg-Marquardt on
    /// total-variance residuals (`tanh` / `exp` / logistic transforms
    /// keep every trial admissible).
    pub fn calibrate(
        quotes: &[(f64, f64, f64)],
        theta_pillars: &[(f64, f64)],
        start: (f64, f64, f64),
    ) -> SsviFit {
        assert!(
            quotes.len() >= 3,
            "need at least three quotes for three parameters"
        );
        let make = |u: &[f64]| Ssvi {
            rho: u[0].tanh(),
            eta: u[1].exp(),
            gamma: 1.0 / (1.0 + (-u[2]).exp()),
            theta_pillars: theta_pillars.to_vec(),
        };
        let (rho0, eta0, gamma0) = start;
        let x0 = vec![
            rho0.clamp(-0.999, 0.999).atanh(),
            eta0.ln(),
            (gamma0.clamp(1e-3, 1.0 - 1e-9) / (1.0 - gamma0.clamp(1e-3, 1.0 - 1e-9))).ln(),
        ];
        let residuals = |u: &[f64]| -> Vec<f64> {
            let s = make(u);
            quotes
                .iter()
                .map(|&(t, k, v)| s.total_variance(k, t) - v * v * t)
                .collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let surface = make(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(t, k, v)| ((surface.total_variance(k, t) / t).sqrt() - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SsviFit {
            surface,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        }
    }

    /// Sample the SSVI surface into the canonical pricing
    /// [`VolSurface`]: per expiry `(t, forward)`, strikes are placed at
    /// `forward * exp(k)` over the log-moneyness grid.
    pub fn to_vol_surface(
        &self,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        expiry_forwards: &[(f64, f64)],
        log_moneyness_grid: &[f64],
    ) -> Result<VolSurface, VolError> {
        let expiries: Vec<Tenor> = expiry_forwards
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = expiry_forwards
            .iter()
            .map(|&(t, forward)| {
                log_moneyness_grid
                    .iter()
                    .map(|&k| {
                        let strike = forward * k.exp();
                        (strike, self.vol(strike, forward, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, reference_date, day_count)
    }
}

// ── SSVI as a fitted surface ────────────────────────────────────────────

/// A calibrated SSVI surface bound to a forward curve — the SSVI sibling
/// of [`SviSurfaceFit`] and
/// [`SabrSurfaceFit`](crate::equity::sabr::SabrSurfaceFit), and the form
/// in which SSVI enters a like-for-like comparison.
///
/// [`Ssvi`] alone cannot be a [`SmoothedSurface`]: it parameterizes total
/// variance in log-moneyness but carries no forwards, so it cannot map a
/// strike to a `k`. This wrapper adds the forward term structure, the
/// quoted spans, and the fit diagnostics.
///
/// Structurally it differs from the per-expiry fits in exactly one way
/// that matters downstream: its term structure is a single continuous
/// `theta_t` threaded through every expiry, so the smile *shape* factor
/// of `dw/dt` never breaks across a pillar. The per-expiry smoothers
/// re-derive that shape on each segment.
#[derive(Debug, Clone)]
pub struct SsviSurfaceFit {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    /// The calibrated global surface.
    pub ssvi: Ssvi,
    /// `(t, forward)` pillars in increasing time order.
    pub forwards: Vec<(f64, f64)>,
    /// Quoted log-moneyness span per pillar, in the same order.
    pub k_ranges: Vec<(f64, f64)>,
    /// Root-mean-square fit error in implied vol over all quotes.
    pub rmse: f64,
    pub converged: bool,
    /// Input expiries dropped for having no usable quotes.
    pub skipped_slices: usize,
}

impl SsviSurfaceFit {
    /// Calibrate one global SSVI surface to every quote on `surface`.
    ///
    /// ATM total variance pillars `theta_t` are read off the surface at
    /// each expiry's forward (not fitted), leaving the three shape
    /// parameters `(rho, eta, gamma)` to the optimizer — the standard
    /// split, and the one that keeps the ATM term structure exact by
    /// construction. Pillars are floored to be non-decreasing so the
    /// calendar condition holds even if the input surface grazes.
    pub fn fit(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
    ) -> Result<SsviSurfaceFit, RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "ssvi fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };
        let mut pillars: Vec<(f64, f64)> = Vec::new();
        let mut forwards: Vec<(f64, f64)> = Vec::new();
        let mut k_ranges: Vec<(f64, f64)> = Vec::new();
        let mut quotes: Vec<(f64, f64, f64)> = Vec::new();
        let mut skipped = 0usize;
        for (tenor, smile) in expiries.iter().zip(&smiles) {
            let t = match tenor {
                Tenor::YearFraction(t) => *t,
                Tenor::Date(_) => continue, // to_input never emits dates
            };
            if smile.is_empty() || t <= 0.0 {
                skipped += 1;
                continue;
            }
            let f = forward(t);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            for &(x, vol) in smile {
                let k = match coordinate {
                    SmileCoordinate::Strike => (x / f).ln(),
                    SmileCoordinate::Moneyness => x.ln(),
                    SmileCoordinate::LogMoneyness => x,
                };
                lo = lo.min(k);
                hi = hi.max(k);
                quotes.push((t, k, vol));
            }
            // ATM total variance read at the forward
            let atm = surface.vol(f, f, t);
            pillars.push((t, atm * atm * t));
            forwards.push((t, f));
            k_ranges.push((lo, hi));
        }
        if quotes.len() < 3 {
            return Err(RustyQLibError::invalid_input(
                "ssvi fit",
                format!(
                    "SSVI needs at least three quotes for three parameters \
                     ({} found, {skipped} expiries skipped)",
                    quotes.len()
                ),
            ));
        }
        // calendar floor on the ATM pillars: theta must not decrease
        for i in 1..pillars.len() {
            if pillars[i].1 < pillars[i - 1].1 {
                pillars[i].1 = pillars[i - 1].1;
            }
        }
        let fit = Ssvi::calibrate(&quotes, &pillars, (-0.5, 0.5, 0.5));
        Ok(SsviSurfaceFit {
            reference_date: surface.reference_date(),
            day_count: surface.day_count(),
            ssvi: fit.surface,
            forwards,
            k_ranges,
            rmse: fit.rmse,
            converged: fit.converged,
            skipped_slices: skipped,
        })
    }

    /// Dupire local vol at underlying `level` and time `t`, from the
    /// closed-form SSVI derivatives.
    pub fn local_vol(&self, level: f64, t: f64) -> f64 {
        <Self as SmoothedSurface>::local_vol(self, level, t)
    }

    /// [`local_vol`](Self::local_vol) plus whether a guard fired.
    pub fn local_vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        <Self as SmoothedSurface>::local_vol_checked(self, level, t)
    }

    /// Sample [`Self::local_vol`] on a `levels` x `times` grid
    /// (`grid[i][j]` = level i, time j — the plotting layout).
    pub fn local_vol_grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        <Self as SmoothedSurface>::local_vol_grid(self, levels, times)
    }

    /// Sample the fit into the canonical pricing [`VolSurface`]: per
    /// pillar, `samples` strikes across that pillar's quoted span.
    pub fn to_vol_surface(&self, samples: usize) -> Result<VolSurface, VolError> {
        let n = samples.max(5);
        let tenors: Vec<Tenor> = self
            .forwards
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = self
            .forwards
            .iter()
            .zip(&self.k_ranges)
            .map(|(&(t, f), &(lo, hi))| {
                (0..n)
                    .map(|i| {
                        let k = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                        let strike = f * k.exp();
                        (strike, self.ssvi.vol(strike, f, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&tenors, &smiles, self.reference_date, self.day_count)
    }

    /// Fit-quality metadata for the surface document.
    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "model": "global SSVI (Gatheral-Jacquier) with power-law curvature",
            "params": {
                "rho": self.ssvi.rho,
                "eta": self.ssvi.eta,
                "gamma": self.ssvi.gamma,
            },
            "theta_pillars": self.ssvi.theta_pillars,
            "rmse_vol_bps": self.rmse * 1e4,
            "converged": self.converged,
            "skipped_slices": self.skipped_slices,
            "static_arbitrage_free": self.ssvi.validate().is_ok(),
        })
    }
}

/// SSVI reaches local volatility with **closed-form** derivatives in both
/// strike and time — the only contender that does. Everything downstream
/// is the shared path.
impl SmoothedSurface for SsviSurfaceFit {
    fn forward(&self, t: f64) -> f64 {
        let p = &self.forwards;
        let n = p.len();
        if n == 1 || t <= p[0].0 {
            return p[0].1;
        }
        if t >= p[n - 1].0 {
            return p[n - 1].1;
        }
        let idx = p.partition_point(|&(ti, _)| ti < t);
        let (t0, f0) = p[idx - 1];
        let (t1, f1) = p[idx];
        f0 + (f1 - f0) * (t - t0) / (t1 - t0)
    }

    fn total_variance(&self, k: f64, t: f64) -> f64 {
        self.ssvi.total_variance(k, t)
    }

    fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives {
        self.ssvi.variance_derivatives(k, t.max(MIN_TIME))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sane() -> SviParams {
        SviParams {
            a: 0.03,
            b: 0.12,
            rho: -0.4,
            m: -0.02,
            sigma: 0.3,
        }
    }

    #[test]
    fn svi_shape_matches_the_closed_form_structure() {
        let p = sane();
        p.validate().unwrap();
        // total variance at k = m is a + b sigma
        assert!((p.total_variance(p.m) - (p.a + p.b * p.sigma)).abs() < 1e-14);
        // asymptotic wing slopes b (1 +- rho), measured per unit of |k|
        let far = 60.0;
        let call_slope = p.total_variance(far + 1.0) - p.total_variance(far);
        let put_slope = p.total_variance(-far - 1.0) - p.total_variance(-far);
        assert!(
            (call_slope - p.b * (1.0 + p.rho)).abs() < 1e-3,
            "{call_slope}"
        );
        assert!(
            (put_slope - p.b * (1.0 - p.rho)).abs() < 1e-3,
            "{put_slope}"
        );
    }

    #[test]
    fn vogt_example_carries_butterfly_arbitrage_and_sane_params_do_not() {
        // the classic arbitrageable SVI smile (Gatheral-Jacquier 2014 §3)
        let vogt = SviParams {
            a: -0.0410,
            b: 0.1331,
            rho: 0.3060,
            m: 0.3586,
            sigma: 0.4153,
        };
        assert!(
            vogt.has_butterfly_arbitrage(),
            "min g = {}",
            vogt.min_butterfly_g()
        );
        assert!((vogt.min_butterfly_g() - -0.0329).abs() < 2e-3);
        assert!(
            !sane().has_butterfly_arbitrage(),
            "min g = {}",
            sane().min_butterfly_g()
        );
    }

    #[test]
    fn svi_calibration_round_trips_with_every_method() {
        let truth = sane();
        let t = 0.75;
        let quotes: Vec<(f64, f64)> = (0..15)
            .map(|i| -0.42 + i as f64 * 0.06)
            .map(|k| (k, truth.vol(k, t)))
            .collect();
        for method in [
            SviCalibration::QuasiExplicit,
            SviCalibration::QuasiExplicitThenLm,
            SviCalibration::LevenbergMarquardt,
        ] {
            let fit = SviParams::calibrate_with(&quotes, t, method);
            assert!(
                fit.rmse < 1e-6,
                "{method:?}: vol rmse {} params {:?}",
                fit.rmse,
                fit.params
            );
            assert!(fit.params.validate().is_ok(), "{method:?}");
            // the fitted smile matches off the quote grid too
            for i in 0..=20 {
                let k = -0.5 + i as f64 * 0.05;
                assert!(
                    (fit.params.vol(k, t) - truth.vol(k, t)).abs() < 1e-4,
                    "{method:?}: k = {k}"
                );
            }
        }
        // the default is the quasi-explicit method
        let default = SviParams::calibrate(&quotes, t);
        let qe = SviParams::calibrate_with(&quotes, t, SviCalibration::QuasiExplicit);
        assert_eq!(default.params, qe.params);
    }

    #[test]
    fn quasi_explicit_fit_stays_in_the_arbitrage_box() {
        // noisy skewed quotes the smile cannot match exactly: the
        // quasi-explicit constraints must still hold on the fit
        let truth = SviParams {
            a: 0.02,
            b: 0.4,
            rho: -0.7,
            m: 0.05,
            sigma: 0.15,
        };
        let t = 2.0;
        let quotes: Vec<(f64, f64)> = (0..13)
            .map(|i| -0.6 + i as f64 * 0.1)
            .map(|k| (k, truth.vol(k, t) + 0.004 * (17.0 * k).sin()))
            .collect();
        let fit = SviParams::calibrate_with(&quotes, t, SviCalibration::QuasiExplicit);
        let p = fit.params;
        assert!(p.validate().is_ok(), "{p:?}");
        assert!(p.a >= 0.0, "a {}", p.a);
        // Lee's wing bound b (1 + |rho|) <= 4/t, up to mapping tolerance
        assert!(
            p.b * (1.0 + p.rho.abs()) <= 4.0 / t + 1e-6,
            "b(1+|rho|) = {}",
            p.b * (1.0 + p.rho.abs())
        );
        assert!(fit.rmse < 0.01, "vol rmse {}", fit.rmse);
    }

    #[test]
    fn uniform_weights_reproduce_the_unweighted_fit() {
        let truth = sane();
        let t = 0.75;
        let quotes: Vec<(f64, f64)> = (0..15)
            .map(|i| -0.42 + i as f64 * 0.06)
            .map(|k| (k, truth.vol(k, t) + 0.003 * (11.0 * k).sin()))
            .collect();
        for method in [
            SviCalibration::QuasiExplicit,
            SviCalibration::LevenbergMarquardt,
        ] {
            let plain = SviParams::calibrate_with(&quotes, t, method);
            let ones = SviParams::calibrate_weighted(&quotes, t, method, &vec![1.0; quotes.len()]);
            assert_eq!(plain.params, ones.params, "{method:?}");
            assert_eq!(plain.rmse, ones.rmse, "{method:?}");
        }
    }

    #[test]
    fn zero_weight_excludes_an_outlier_quote() {
        let truth = sane();
        let t = 0.75;
        let mut quotes: Vec<(f64, f64)> = (0..15)
            .map(|i| -0.42 + i as f64 * 0.06)
            .map(|k| (k, truth.vol(k, t)))
            .collect();
        quotes[7].1 += 0.05; // a bad print near the money
        let max_clean_err = |p: &SviParams| {
            quotes
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != 7)
                .map(|(_, &(k, _))| (p.vol(k, t) - truth.vol(k, t)).abs())
                .fold(0.0, f64::max)
        };
        for method in [
            SviCalibration::QuasiExplicit,
            SviCalibration::QuasiExplicitThenLm,
            SviCalibration::LevenbergMarquardt,
        ] {
            let polluted = SviParams::calibrate_with(&quotes, t, method);
            let mut weights = vec![1.0; quotes.len()];
            weights[7] = 0.0;
            let cleaned = SviParams::calibrate_weighted(&quotes, t, method, &weights);
            // the outlier drags the unweighted fit; zero-weighting it
            // recovers the generating smile on the clean quotes
            assert!(
                max_clean_err(&cleaned.params) < 1e-5,
                "{method:?}: cleaned err {}",
                max_clean_err(&cleaned.params)
            );
            assert!(
                max_clean_err(&polluted.params) > 1e-3,
                "{method:?}: polluted err {}",
                max_clean_err(&polluted.params)
            );
        }
    }

    #[test]
    fn lm_polish_never_worsens_the_quasi_explicit_fit() {
        let truth = sane();
        let t = 1.5;
        let quotes: Vec<(f64, f64)> = (0..11)
            .map(|i| -0.35 + i as f64 * 0.07)
            .map(|k| (k, truth.vol(k, t) + 0.002 * (23.0 * k).cos()))
            .collect();
        let sse_w = |p: &SviParams| -> f64 {
            quotes
                .iter()
                .map(|&(k, v)| (p.total_variance(k) - v * v * t).powi(2))
                .sum()
        };
        let qe = SviParams::calibrate_with(&quotes, t, SviCalibration::QuasiExplicit);
        let polished = SviParams::calibrate_with(&quotes, t, SviCalibration::QuasiExplicitThenLm);
        // LM only accepts cost-decreasing steps from the QE start
        assert!(
            sse_w(&polished.params) <= sse_w(&qe.params) * (1.0 + 1e-12),
            "qe {} polished {}",
            sse_w(&qe.params),
            sse_w(&polished.params)
        );
    }

    fn surface_from(slices: &[(f64, SviParams, f64)]) -> VolSurface {
        // sample each known smile onto pillar strikes, as a chain would
        let expiries: Vec<Tenor> = slices
            .iter()
            .map(|&(t, _, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = slices
            .iter()
            .map(|&(t, p, f)| {
                (0..11)
                    .map(|i| {
                        let k = -0.3 + i as f64 * 0.06;
                        (f * k.exp(), p.vol(k, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap()
    }

    #[test]
    fn per_expiry_fit_recovers_generating_smiles() {
        let front = sane();
        let back = SviParams { a: 0.055, ..sane() };
        let surface = surface_from(&[(0.5, front, 101.0), (1.0, back, 102.0)]);
        let forward = |t: f64| if t < 0.75 { 101.0 } else { 102.0 };
        let fit = SviSurfaceFit::fit(&surface, forward).unwrap();
        assert_eq!(fit.slices.len(), 2);
        assert_eq!(fit.skipped_slices, 0);
        for slice in &fit.slices {
            assert!(slice.rmse < 1e-5, "rmse {}", slice.rmse);
            assert!(slice.min_g > 0.0, "min g {}", slice.min_g);
        }
        assert!(fit.max_calendar_crossing <= 1e-10);
        // fitted vols agree with the generators off the pillar grid too
        for i in 0..=12 {
            let k = -0.28 + i as f64 * 0.05;
            let strike = 101.0 * k.exp();
            assert!(
                (fit.vol(strike, 0.5) - front.vol((strike / 101.0_f64).ln(), 0.5)).abs() < 5e-4,
                "k = {k}"
            );
        }
        // sampled surface matches the fit at its own nodes
        let sampled = fit.to_vol_surface(41).unwrap();
        assert_eq!(sampled.expiry_times().len(), 2);
        let probe = 101.0;
        assert!((sampled.vol(probe, probe, 0.5) - fit.vol(probe, 0.5)).abs() < 1e-3);
        // metadata carries per-slice fit quality
        let meta = fit.metadata();
        assert_eq!(meta["slices"].as_array().unwrap().len(), 2);
        assert!(meta["slices"][0]["rmse_vol_bps"].as_f64().unwrap() < 0.5);
    }

    #[test]
    fn surface_fit_calibration_method_is_pluggable() {
        let surface = surface_from(&[(0.5, sane(), 101.0)]);
        for method in [
            SviCalibration::QuasiExplicitThenLm,
            SviCalibration::LevenbergMarquardt,
        ] {
            let fit = SviSurfaceFit::fit_with(&surface, |_| 101.0, method).unwrap();
            assert!(fit.slices[0].rmse < 1e-5, "{method:?}: {}", fit.slices[0].rmse);
        }
    }

    #[test]
    fn flat_svi_term_structure_gives_flat_local_vol() {
        // b = 0 collapses SVI to w(k) = a: constant vol per slice
        let vol = 0.3_f64;
        let slice = |t: f64| SviParams {
            a: vol * vol * t,
            b: 0.0,
            rho: 0.0,
            m: 0.0,
            sigma: 0.3,
        };
        let surface = surface_from(&[(0.5, slice(0.5), 100.0), (1.0, slice(1.0), 100.0)]);
        let fit = SviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        // sigma_loc = sigma_imp everywhere: interior, between slices,
        // below the first pillar and beyond the last
        for level in [80.0, 100.0, 120.0] {
            for t in [0.1, 0.5, 0.75, 1.0, 1.4] {
                let lv = fit.local_vol(level, t);
                assert!((lv - vol).abs() < 5e-3, "level {level} t {t}: {lv}");
            }
        }
    }

    #[test]
    fn sparse_slices_are_skipped_not_fatal() {
        let p = sane();
        let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
        let smiles = vec![
            // three quotes: below SVI's five-parameter minimum
            vec![(90.0, 0.25), (100.0, 0.24), (110.0, 0.23)],
            (0..9)
                .map(|i| {
                    let k = -0.2 + i as f64 * 0.05;
                    (100.0 * k.exp(), p.vol(k, 1.0))
                })
                .collect(),
        ];
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let fit = SviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        assert_eq!(fit.slices.len(), 1);
        assert_eq!(fit.skipped_slices, 1);
        // a surface with no fittable slice errors instead
        let tiny = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5)],
            &[vec![(100.0, 0.2), (105.0, 0.19)]],
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        assert!(SviSurfaceFit::fit(&tiny, |_| 100.0).is_err());
    }

    fn ssvi() -> Ssvi {
        Ssvi {
            rho: -0.55,
            eta: 0.9,
            gamma: 0.45,
            theta_pillars: vec![(0.25, 0.012), (0.5, 0.023), (1.0, 0.045), (2.0, 0.09)],
        }
    }

    #[test]
    fn ssvi_reproduces_the_atm_term_structure_and_skew_sign() {
        let s = ssvi();
        s.validate().unwrap();
        for &(t, theta) in &s.theta_pillars {
            assert!(
                (s.total_variance(0.0, t) - theta).abs() < 1e-14,
                "w(0, {t})"
            );
        }
        // negative rho: puts richer than calls
        assert!(s.total_variance(-0.2, 1.0) > s.total_variance(0.2, 1.0));
        // calendar: total variance nondecreasing in t at fixed k
        for i in 1..40 {
            let (t0, t1) = (i as f64 * 0.05, (i + 1) as f64 * 0.05);
            assert!(
                s.total_variance(0.15, t1) >= s.total_variance(0.15, t0),
                "t = {t0}"
            );
        }
    }

    #[test]
    fn ssvi_no_arbitrage_bounds_are_enforced() {
        let mut bad = ssvi();
        bad.eta = 1.5; // eta (1 + |rho|) = 2.325 > 2
        assert!(bad.validate().is_err());
        let mut decreasing = ssvi();
        decreasing.theta_pillars[2].1 = 0.01; // calendar violation
        assert!(decreasing.validate().is_err());
    }

    #[test]
    fn ssvi_calibration_round_trips() {
        let truth = ssvi();
        let mut quotes = Vec::new();
        for &(t, _) in &truth.theta_pillars {
            for i in 0..7 {
                let k = -0.3 + i as f64 * 0.1;
                quotes.push((t, k, (truth.total_variance(k, t) / t).sqrt()));
            }
        }
        let fit = Ssvi::calibrate(&quotes, &truth.theta_pillars, (-0.2, 0.5, 0.5));
        assert!(fit.rmse < 1e-8, "vol rmse {}", fit.rmse);
        assert!(
            (fit.surface.rho - truth.rho).abs() < 1e-4,
            "rho {}",
            fit.surface.rho
        );
        assert!(
            (fit.surface.eta - truth.eta).abs() < 1e-3,
            "eta {}",
            fit.surface.eta
        );
        assert!(fit.surface.validate().is_ok());
    }

    // ── SSVI derivatives and local volatility ───────────────────────────

    #[test]
    fn ssvi_analytic_derivatives_match_finite_differences() {
        // the closed forms in `Ssvi::variance_derivatives` are the whole
        // reason SSVI can be differentiated exactly; check every one of
        // them against a difference quotient
        use crate::equity::smoothed_surface::numeric_k_derivatives;
        let s = ssvi();
        // strictly between pillars, so d theta / dt is unambiguous
        for t in [0.35, 0.7, 1.4] {
            for k in [-0.4, -0.1, 0.0, 0.2, 0.5] {
                let d = s.variance_derivatives(k, t);
                let [w_num, dk_num, dkk_num] =
                    numeric_k_derivatives(|kk| s.total_variance(kk, t), k);
                assert!((d.w - w_num).abs() < 1e-14, "w at k={k} t={t}");
                assert!(
                    (d.dk - dk_num).abs() < 1e-7,
                    "w_k at k={k} t={t}: {} vs {dk_num}",
                    d.dk
                );
                assert!(
                    (d.dkk - dkk_num).abs() < 1e-4,
                    "w_kk at k={k} t={t}: {} vs {dkk_num}",
                    d.dkk
                );
                // time derivative by central difference inside the segment
                let h = 1e-6;
                let dt_num = (s.total_variance(k, t + h) - s.total_variance(k, t - h)) / (2.0 * h);
                assert!(
                    (d.dt - dt_num).abs() < 1e-5,
                    "w_t at k={k} t={t}: {} vs {dt_num}",
                    d.dt
                );
            }
        }
    }

    #[test]
    fn variance_keeps_accruing_beyond_the_last_pillar() {
        // two flat smiles (b = 0) at the same implied vol: w = vol^2 t at
        // both pillars, so the extrapolated term structure must stay flat
        let vol = 0.2_f64;
        let flat = |t: f64| SviSlice {
            t,
            forward: 100.0,
            params: SviParams {
                a: vol * vol * t,
                b: 0.0,
                rho: 0.0,
                m: 0.0,
                sigma: 0.1,
            },
            rmse: 0.0,
            converged: true,
            k_range: (-0.5, 0.5),
            min_g: 0.0,
        };
        let reference = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let fit = SviSurfaceFit {
            reference_date: reference,
            day_count: DayCountConvention::Act365,
            slices: vec![flat(0.5), flat(1.0)],
            skipped_slices: 0,
            max_calendar_crossing: 0.0,
        };
        // w keeps accruing at the last segment's forward variance...
        assert!((fit.total_variance(0.0, 2.0) - vol * vol * 2.0).abs() < 1e-14);
        // ...so implied vol holds its level instead of decaying like 1/sqrt(t)
        for t in [1.0, 2.0, 5.0, 30.0] {
            assert!((fit.vol(100.0, t) - vol).abs() < 1e-9, "t = {t}");
        }
        // the Dupire derivatives path sees the same extension
        let d = fit.variance_derivatives(0.0, 4.0);
        assert!((d.w - fit.total_variance(0.0, 4.0)).abs() < 1e-14);
        assert!((d.dt - vol * vol).abs() < 1e-14);

        // a single-slice fit extends at flat implied vol
        let single = SviSurfaceFit {
            reference_date: reference,
            day_count: DayCountConvention::Act365,
            slices: vec![flat(0.5)],
            skipped_slices: 0,
            max_calendar_crossing: 0.0,
        };
        assert!((single.vol(100.0, 3.0) - vol).abs() < 1e-9);

        // a decreasing last segment (fit tension) is floored: variance
        // is held beyond the pillar rather than bled away
        let tense = SviSurfaceFit {
            reference_date: reference,
            day_count: DayCountConvention::Act365,
            slices: vec![flat(0.5), {
                let mut s = flat(1.0);
                s.params.a = 0.9 * vol * vol * 0.5; // below the first pillar's w
                s
            }],
            skipped_slices: 0,
            max_calendar_crossing: 0.0,
        };
        let held = tense.total_variance(0.0, 1.0);
        assert!((tense.total_variance(0.0, 5.0) - held).abs() < 1e-14);
    }

    #[test]
    fn ssvi_second_derivative_is_positive_so_the_density_is_admissible() {
        // w_kk = theta phi^2 (1 - rho^2) / (2 R^3) > 0 identically
        let s = ssvi();
        for t in [0.25, 1.0, 2.0] {
            for i in 0..=20 {
                let k = -1.0 + i as f64 * 0.1;
                assert!(
                    s.variance_derivatives(k, t).dkk > 0.0,
                    "w_kk at k={k} t={t}"
                );
            }
        }
    }

    fn ssvi_surface_from(s: &Ssvi, forward: f64) -> VolSurface {
        let expiries: Vec<Tenor> = s
            .theta_pillars
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = s
            .theta_pillars
            .iter()
            .map(|&(t, _)| {
                (0..11)
                    .map(|i| {
                        let k = -0.3 + i as f64 * 0.06;
                        let strike = forward * k.exp();
                        (strike, s.vol(strike, forward, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 19).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap()
    }

    #[test]
    fn flat_ssvi_gives_flat_local_vol() {
        // eta -> 0 collapses the smile: w(k, t) = theta_t for every k, so
        // with theta_t = sigma^2 t the local vol must be sigma everywhere
        let vol = 0.28_f64;
        let flat = Ssvi {
            rho: -0.3,
            eta: 1e-10,
            gamma: 0.5,
            theta_pillars: vec![
                (0.5, vol * vol * 0.5),
                (1.0, vol * vol),
                (2.0, vol * vol * 2.0),
            ],
        };
        let surface = ssvi_surface_from(&flat, 100.0);
        let fit = SsviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        for level in [80.0, 100.0, 125.0] {
            for t in [0.1, 0.5, 0.75, 1.0, 2.0, 2.5] {
                let (lv, guarded) = fit.local_vol_checked(level, t);
                assert!(!guarded, "level {level} t {t} should not need a guard");
                assert!((lv - vol).abs() < 5e-3, "level {level} t {t}: {lv}");
            }
        }
    }

    #[test]
    fn ssvi_surface_fit_recovers_the_generating_surface() {
        let truth = ssvi();
        let surface = ssvi_surface_from(&truth, 100.0);
        let fit = SsviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        assert_eq!(fit.skipped_slices, 0);
        assert!(fit.rmse < 1e-4, "vol rmse {}", fit.rmse);
        assert!(
            (fit.ssvi.rho - truth.rho).abs() < 1e-2,
            "rho {}",
            fit.ssvi.rho
        );
        // the ATM pillars are read off the surface, not fitted, so they
        // must reproduce the generator's ATM variance
        for (&(t, theta), &(_, want)) in fit.ssvi.theta_pillars.iter().zip(&truth.theta_pillars) {
            assert!(
                (theta - want).abs() < 1e-6,
                "theta at t={t}: {theta} vs {want}"
            );
        }
        // local vol is finite, positive and guard-free across the quoted box
        let mut guarded = 0;
        for i in 0..=10 {
            let k: f64 = -0.28 + i as f64 * 0.056;
            for t in [0.3, 0.5, 1.0, 2.0] {
                let (lv, g) = fit.local_vol_checked(100.0 * k.exp(), t);
                assert!(lv.is_finite() && lv > 0.0, "k={k} t={t}");
                guarded += g as usize;
            }
        }
        assert_eq!(guarded, 0, "a clean SSVI fit needs no guards");
        // and it samples back into a pricing surface
        let sampled = fit.to_vol_surface(31).unwrap();
        assert_eq!(sampled.expiry_times().len(), truth.theta_pillars.len());
        let meta = fit.metadata();
        assert!(meta["static_arbitrage_free"].as_bool().unwrap());
    }

    #[test]
    fn sampled_vol_surface_agrees_with_the_parametric_form() {
        use chrono::NaiveDate;
        let s = ssvi();
        let reference = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let forwards = [(0.25, 101.0), (1.0, 104.0), (2.0, 108.0)];
        let grid: Vec<f64> = (0..13).map(|i| -0.3 + i as f64 * 0.05).collect();
        let surface = s
            .to_vol_surface(reference, DayCountConvention::Act365, &forwards, &grid)
            .unwrap();
        // exact at the sampled nodes
        for &(t, f) in &forwards {
            for &k in &grid {
                let strike = f * k.exp();
                let sampled = surface.vol(strike, f, t);
                let parametric = s.vol(strike, f, t);
                assert!(
                    (sampled - parametric).abs() < 1e-10,
                    "t = {t}, k = {k}: {sampled} vs {parametric}"
                );
            }
        }
    }
}
