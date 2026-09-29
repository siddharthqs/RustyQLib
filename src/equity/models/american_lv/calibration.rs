//! Tikhonov calibration of the B-spline local volatility (and the
//! per-expiry carry) to a chain of American quotes: the least-squares
//! problem with two-sided and one-sided (at-intrinsic) quotes, the
//! projected Levenberg–Marquardt driver with alpha continuation and the
//! discrepancy principle, the linearized first Gauss–Newton step (M1), and
//! the European variant used by the pointwise-identification benchmark
//! (M0).
//!
//! ```text
//! Unknowns:   theta (M B-spline coefficients, index j_y * n_t_basis + j_t), q_k (carry per
//!             expiry interval I_k = [T_{k-1}, T_k), E of them; none when carry is frozen)
//! Residuals:  two-sided in-sample   r_i = (F_i - m_i) / s_i
//!             one-sided (intrinsic) r_i = max(F_i - target_i, 0) / s_i
//!             held-out              evaluated, excluded from the fit
//! Objective:  J = sum_fit r_i^2 + alpha theta^T Q theta + beta sum_k ((q_k - q_prior,k)/q_scale)^2
//! Normal eq.: (J^T J + alpha Q + beta/q_scale^2 I_q + lambda_LM diag) delta = -(J^T r + alpha Q theta + ...)
//!             one triangle assembled and mirrored; Cholesky; delta_theta clamped to the bounds
//! Jacobian:   per quote one retained forward solve + one adjoint march, streamed:
//!             dF/dtheta_m = sum_{j,n} g_{j,n} B_m(y_{j,n}, t_n)   (separable basis cache)
//!             dF/dq_k     = - sum_{n : t_mid[n] in I_k} h_n
//! Continuation: alpha_0 -> alpha_0 / ratio -> ... ; stop at the first level with
//!             sum_{two-sided in-sample} r_i^2 <= target, then one bisection in log alpha
//!             between the last two levels; target = tau N_eff, or with a noise floor
//!             delta: sum_i (delta / s_i)^2 (plateau stop when the level-to-level
//!             decrease is below plateau_fraction).
//! ```
//!
//! Cost model (the study runs ~1000 captures x ~100 Jacobians): every
//! Jacobian pass reuses thread-local workspaces (one per rayon worker),
//! never allocates grids per quote, evaluates the surface once per expiry
//! per theta through the [`BasisCache`], and processes the longest meshes
//! first for load balance. A rejected LM trial costs one pass (the trial
//! evaluation *is* the Jacobian pass, so an accepted step needs no second
//! forward sweep); the Jacobian at a level's converged theta is reused as
//! the first Jacobian of the next level.
//!
//! The coordinate of the surface is `y = x - ln F_ref(t)` with `F_ref` the
//! PRIOR forward fixed for the whole calibration ([`CaptureMeshes::f_ref_table`]):
//! the carry enters only through the drift of the pricing PDE, so a change
//! of `q` never rebuilds the basis caches.

// Stencil-style index loops over quotes x columns and over triangles are
// clearer than zipped iterators here.
#![allow(clippy::needless_range_loop)]

use super::adjoint::{adjoint_stream, AdjointWorkspace};
use super::bspline::{project_onto_bounds, BSplineLocalVol, BasisCache};
use super::european_surface::{european_prices_backward, otm_implied_vol, IvError};
use super::grid::{
    effective_rates, forward_at_levels, n_t_default, n_t_fine, MarketSlice, Mesh, N_X_DEFAULT,
    N_X_FINE,
};
use super::solver::{price_only, solve_into, Mode, QuoteSpec, Solution, Workspace, RHO_DEFAULT};
use super::vol_field::{NodeField, VolField};
use crate::core::errors::RustyQLibError;
use crate::core::linalg::decomp::cholesky::{cholesky_factor, cholesky_solve};
use crate::core::trade::PutOrCall;
use crate::equity::blackscholes::bs_price;
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

/// Smallest Marquardt parameter after successive accepted steps.
pub const LM_LAMBDA_MIN: f64 = 1e-12;
/// Marquardt parameter beyond which a level is declared stagnated.
pub const LM_LAMBDA_MAX: f64 = 1e8;
/// Smallest Marquardt parameter after a rejected trial.
pub const LM_LAMBDA_REJECT_FLOOR: f64 = 1e-4;
/// Tolerance on `|quote.t - expiry|` when matching a quote to its mesh.
const EXPIRY_TOL: f64 = 1e-9;
/// Sentinel of the support-map bins for nodes outside the display range.
const NO_BIN: u16 = u16::MAX;

// ── Quotes ───────────────────────────────────────────────────────────────

/// One listed quote as the calibration sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Quote {
    /// Strike (spot currency).
    pub strike: f64,
    /// Expiry in years (must equal the capture's expiry `expiry_index`).
    pub t: f64,
    /// Index of the expiry (and mesh) in the [`CaptureMeshes`].
    pub expiry_index: usize,
    /// Put or call.
    pub right: PutOrCall,
    /// Mid price `m_i` (the fitted target of a two-sided quote).
    pub mid: f64,
    /// Half-spread `s_i` (the residual scale unless `weight_override`).
    pub half_spread: f64,
    /// Bid.
    pub bid: f64,
    /// Ask.
    pub ask: f64,
    /// The bid sits at intrinsic: the quote is fitted one-sidedly
    /// (`F_i <= one_sided_target`).
    pub at_intrinsic: bool,
    /// Upper bound of a one-sided quote, `max(mid + s, intrinsic + tick)`.
    pub one_sided_target: f64,
    /// Excluded from the fit, evaluated by every method.
    pub held_out: bool,
    /// Replaces `half_spread` as the residual scale when set (the
    /// propagated `s_i nu^E/nu^A` of the identification benchmark).
    pub weight_override: Option<f64>,
}

impl Quote {
    /// A two-sided quote from bid and ask.
    pub fn two_sided(
        strike: f64,
        t: f64,
        expiry_index: usize,
        right: PutOrCall,
        bid: f64,
        ask: f64,
    ) -> Quote {
        let mid = 0.5 * (bid + ask);
        let half_spread = 0.5 * (ask - bid);
        Quote {
            strike,
            t,
            expiry_index,
            right,
            mid,
            half_spread,
            bid,
            ask,
            at_intrinsic: false,
            one_sided_target: mid + half_spread,
            held_out: false,
            weight_override: None,
        }
    }

    /// A two-sided quote from its mid and half-spread (`bid = mid - s`,
    /// `ask = mid + s`).
    pub fn from_mid(
        strike: f64,
        t: f64,
        expiry_index: usize,
        right: PutOrCall,
        mid: f64,
        half_spread: f64,
    ) -> Quote {
        Quote::two_sided(
            strike,
            t,
            expiry_index,
            right,
            mid - half_spread,
            mid + half_spread,
        )
    }

    /// Classify against the intrinsic value: `at_intrinsic = bid <=
    /// intrinsic + tick`, `one_sided_target = max(mid + s, intrinsic +
    /// tick)`.
    pub fn classify_intrinsic(mut self, intrinsic: f64, tick: f64) -> Quote {
        self.at_intrinsic = self.bid <= intrinsic + tick;
        self.one_sided_target = (self.mid + self.half_spread).max(intrinsic + tick);
        self
    }

    /// Mark the quote as held out.
    pub fn held_out(mut self) -> Quote {
        self.held_out = true;
        self
    }

    /// Override the residual scale.
    pub fn with_weight(mut self, scale: f64) -> Quote {
        self.weight_override = Some(scale);
        self
    }

    /// Plain intrinsic value `psi(S0)`.
    pub fn intrinsic(&self, s0: f64) -> f64 {
        match self.right {
            PutOrCall::Call => (s0 - self.strike).max(0.0),
            PutOrCall::Put => (self.strike - s0).max(0.0),
        }
    }

    /// The solver spec at the mesh's exact expiry.
    pub fn spec(&self, t_expiry: f64) -> QuoteSpec {
        QuoteSpec::new(self.strike, t_expiry, self.right)
    }

    /// Residual scale `s_i` (the override when set).
    #[inline]
    pub fn scale(&self) -> f64 {
        self.weight_override.unwrap_or(self.half_spread)
    }

    /// In the fit (not held out).
    #[inline]
    pub fn in_fit(&self) -> bool {
        !self.held_out
    }

    /// Two-sided and in the fit: counts toward the discrepancy.
    #[inline]
    pub fn two_sided_in_fit(&self) -> bool {
        !self.held_out && !self.at_intrinsic
    }

    /// Whether a one-sided quote is active at the model price `F`
    /// (always `true` for two-sided quotes).
    #[inline]
    pub fn active_at(&self, price: f64) -> bool {
        !self.at_intrinsic || price > self.one_sided_target
    }

    /// Unscaled misfit: `F - m` (two-sided) or `max(F - target, 0)`
    /// (one-sided).
    #[inline]
    pub fn misfit(&self, price: f64) -> f64 {
        if self.at_intrinsic {
            (price - self.one_sided_target).max(0.0)
        } else {
            price - self.mid
        }
    }

    /// Scaled residual `misfit / s_i`.
    #[inline]
    pub fn residual(&self, price: f64) -> f64 {
        self.misfit(price) / self.scale()
    }

    fn validate(&self, capture: &CaptureMeshes) -> Result<(), RustyQLibError> {
        if !(self.strike.is_finite() && self.strike > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "quote.strike",
                format!("strike must be positive and finite, got {}", self.strike),
            ));
        }
        if self.expiry_index >= capture.n_expiries() {
            return Err(RustyQLibError::invalid_input(
                "quote.expiry_index",
                format!(
                    "expiry index {} but the capture has {} expiries",
                    self.expiry_index,
                    capture.n_expiries()
                ),
            ));
        }
        let t_mesh = capture.expiries[self.expiry_index];
        if !self.t.is_finite() || (self.t - t_mesh).abs() > EXPIRY_TOL * t_mesh.max(1.0) {
            return Err(RustyQLibError::invalid_input(
                "quote.t",
                format!(
                    "quote expiry {} does not match expiry {} = {}",
                    self.t, self.expiry_index, t_mesh
                ),
            ));
        }
        let s = self.scale();
        if !(s.is_finite() && s > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "quote.half_spread",
                format!("residual scale must be positive and finite, got {s}"),
            ));
        }
        if !(self.mid.is_finite() && self.one_sided_target.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "quote.mid",
                "mid and one-sided target must be finite",
            ));
        }
        Ok(())
    }
}

// ── Configuration ────────────────────────────────────────────────────────

/// Knobs of the calibration. Construct with [`CalibrationConfig::american`]
/// (M2), [`CalibrationConfig::european`] (M0) or
/// [`CalibrationConfig::synthetic`] and adjust fields.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationConfig {
    /// Spatial intervals of the working mesh (recorded; the meshes are
    /// built by [`CaptureMeshes`]).
    pub n_x: usize,
    /// Penalty rate of the American solver (1/year).
    pub rho: f64,
    /// Ratio between consecutive alpha levels.
    pub alpha_ratio: f64,
    /// Maximum number of alpha levels (the bisection level not counted).
    pub max_alpha_levels: usize,
    /// LM iteration cap at intermediate levels.
    pub lm_iters_intermediate: usize,
    /// LM iteration cap at the level that satisfies the discrepancy (and
    /// at the bisection level).
    pub lm_iters_final: usize,
    /// Cap on Jacobian passes (accepted and rejected trials, plus the
    /// initial pass) per calibration.
    pub jacobian_budget: usize,
    /// Initial Marquardt parameter (`0` gives a pure Gauss–Newton first
    /// trial).
    pub lm_lambda0: f64,
    /// Multiplicative Marquardt update (`x` on rejection, `/` on
    /// acceptance).
    pub lm_lambda_factor: f64,
    /// Stop a level when the relative objective decrease of an accepted
    /// step falls below this (intermediate levels).
    pub rel_decrease_stop_intermediate: f64,
    /// Same at the final level.
    pub rel_decrease_stop_final: f64,
    /// Discrepancy factor: stop at the first level with `sum r_i^2 <= tau
    /// N_eff` (1.0 empirical, 1.1 synthetic Gaussian noise).
    pub tau: f64,
    /// Weight `beta` of the carry prior.
    pub beta_carry: f64,
    /// Scale of the carry prior (1% by default).
    pub q_scale: f64,
    /// Hold the carry at its prior (no `q` columns).
    pub carry_frozen: bool,
    /// Coefficient bounds `(sigma_min, sigma_max)` enforced by projection.
    pub bounds: (f64, f64),
    /// Carry bounds enforced by projection.
    pub q_bounds: (f64, f64),
    /// Noise-floor variant for noiseless data: the discrepancy target
    /// becomes `sum_i (noise_floor / s_i)^2` (an absolute price, e.g.
    /// `1e-4 S0`), and the continuation also stops when the discrepancy
    /// decreases by less than `plateau_fraction` between levels.
    pub noise_floor: Option<f64>,
    /// Relative level-to-level decrease below which the noise-floor
    /// continuation stops.
    pub plateau_fraction: f64,
    /// Regularizer weights `(lambda_y, lambda_t)`; `None` uses the
    /// surface's [`BSplineLocalVol::default_lambdas`].
    pub lambdas: Option<(f64, f64)>,
    /// Maximum alpha levels of the linear path in [`linearized_step`].
    pub max_linear_levels: usize,
}

impl CalibrationConfig {
    /// The M2 (full nonlinear, American) defaults: Jacobian budget 48,
    /// `tau = 1`, carry free with the 1% prior.
    pub fn american() -> Self {
        CalibrationConfig {
            n_x: N_X_DEFAULT,
            rho: RHO_DEFAULT,
            alpha_ratio: 4.0,
            max_alpha_levels: 8,
            lm_iters_intermediate: 8,
            lm_iters_final: 20,
            jacobian_budget: 48,
            lm_lambda0: 1e-3,
            lm_lambda_factor: 3.0,
            rel_decrease_stop_intermediate: 1e-4,
            rel_decrease_stop_final: 1e-6,
            tau: 1.0,
            beta_carry: 1.0,
            q_scale: 0.01,
            carry_frozen: false,
            bounds: (super::bspline::SIGMA_MIN, super::bspline::SIGMA_MAX),
            q_bounds: (-0.5, 0.5),
            noise_floor: None,
            plateau_fraction: 0.01,
            lambdas: None,
            max_linear_levels: 24,
        }
    }

    /// The M0 (European identification benchmark) defaults: Jacobian
    /// budget 32, carry frozen at the prior.
    pub fn european() -> Self {
        CalibrationConfig {
            jacobian_budget: 32,
            carry_frozen: true,
            ..Self::american()
        }
    }

    /// Synthetic-study defaults: Jacobian budget 96, `tau = 1.1`.
    pub fn synthetic() -> Self {
        CalibrationConfig {
            jacobian_budget: 96,
            tau: 1.1,
            ..Self::american()
        }
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        let ok = self.alpha_ratio > 1.0
            && self.max_alpha_levels >= 1
            && self.jacobian_budget >= 1
            && self.lm_lambda0 >= 0.0
            && self.lm_lambda_factor > 1.0
            && self.tau > 0.0
            && self.beta_carry >= 0.0
            && self.q_scale > 0.0
            && self.bounds.0 > 0.0
            && self.bounds.0 < self.bounds.1
            && self.q_bounds.0 < self.q_bounds.1
            && self.noise_floor.is_none_or(|d| d > 0.0)
            && self.plateau_fraction >= 0.0
            && self.rho > 0.0;
        if ok {
            Ok(())
        } else {
            Err(RustyQLibError::invalid_input(
                "calibration_config",
                format!("{self:?} is not a valid configuration"),
            ))
        }
    }
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self::american()
    }
}

// ── Capture meshes ───────────────────────────────────────────────────────

/// The rate input of a capture.
#[derive(Clone, Copy)]
pub enum RateInput<'a> {
    /// A flat continuous rate.
    Flat(f64),
    /// An instantaneous forward-rate function `r(t)`.
    Instantaneous(&'a (dyn Fn(f64) -> f64 + Sync)),
    /// A discount-factor function `D(t)` (per-step forwards are exact).
    DiscountFactors(&'a (dyn Fn(f64) -> f64 + Sync)),
}

/// The meshes of one capture: a shared log-spot grid, one time grid per
/// expiry (with the cash dividends), the per-step market slices at the
/// current carry vector, and the carry intervals `[T_{k-1}, T_k)`.
#[derive(Debug, Clone)]
pub struct CaptureMeshes {
    /// Spot.
    pub s0: f64,
    /// Half-width of the log-spot grid.
    pub half_width: f64,
    /// Spatial intervals.
    pub n_x: usize,
    /// Expiries in years, strictly increasing.
    pub expiries: Vec<f64>,
    /// Cash dividends `(t, amount)` applied to every mesh that spans them.
    pub dividends: Vec<(f64, f64)>,
    /// One mesh per expiry.
    pub meshes: Vec<Mesh>,
    /// One market slice per expiry at the current `q`.
    pub markets: Vec<MarketSlice>,
    /// Carry per expiry interval.
    pub q: Vec<f64>,
    /// Half-open carry intervals `[a_k, b_k)`, one per expiry.
    pub intervals: Vec<(f64, f64)>,
}

impl CaptureMeshes {
    /// Build the meshes: `n_x` intervals on `[ln s0 - half_width, ln s0 +
    /// half_width]`, `n_t_of(T)` uniform steps per expiry, the dividends
    /// inserted, per-step rates from `rates`, carry `q[k]` on interval
    /// `k`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        s0: f64,
        half_width: f64,
        n_x: usize,
        expiries: &[f64],
        dividends: &[(f64, f64)],
        rates: RateInput<'_>,
        q: &[f64],
        n_t_of: &dyn Fn(f64) -> usize,
    ) -> Result<Self, RustyQLibError> {
        if expiries.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "expiries",
                "at least one expiry is required",
            ));
        }
        for (i, &t) in expiries.iter().enumerate() {
            let ok = t.is_finite() && t > 0.0 && (i == 0 || t > expiries[i - 1]);
            if !ok {
                return Err(RustyQLibError::invalid_input(
                    "expiries",
                    "expiries must be finite, positive and strictly increasing",
                ));
            }
        }
        if q.len() != expiries.len() {
            return Err(RustyQLibError::invalid_input(
                "q",
                format!("{} carries for {} expiries", q.len(), expiries.len()),
            ));
        }
        let meshes = expiries
            .iter()
            .map(|&t| Mesh::new(s0, half_width, n_x, t, n_t_of(t), dividends))
            .collect::<Result<Vec<_>, _>>()?;
        let mut intervals = Vec::with_capacity(expiries.len());
        let mut lo = 0.0;
        for &t in expiries {
            intervals.push((lo, t));
            lo = t;
        }
        let zero = |_: f64| 0.0;
        let markets = meshes
            .iter()
            .map(|mesh| match rates {
                RateInput::Flat(r) => MarketSlice::flat(mesh, s0, r, 0.0, dividends),
                RateInput::Instantaneous(f) => {
                    MarketSlice::from_curves(mesh, s0, f, &zero, dividends)
                }
                RateInput::DiscountFactors(df) => {
                    MarketSlice::from_discount_factors(mesh, s0, df, &zero, dividends)
                }
            })
            .collect();
        let mut out = CaptureMeshes {
            s0,
            half_width,
            n_x,
            expiries: expiries.to_vec(),
            dividends: dividends.to_vec(),
            meshes,
            markets,
            q: vec![0.0; expiries.len()],
            intervals,
        };
        out.set_q(q)?;
        Ok(out)
    }

    /// The working mesh: [`N_X_DEFAULT`] intervals and [`n_t_default`]
    /// steps per expiry.
    pub fn standard(
        s0: f64,
        half_width: f64,
        expiries: &[f64],
        dividends: &[(f64, f64)],
        rates: RateInput<'_>,
        q: &[f64],
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            s0,
            half_width,
            N_X_DEFAULT,
            expiries,
            dividends,
            rates,
            q,
            &n_t_default,
        )
    }

    /// The fine reference mesh for synthetic truth: [`N_X_FINE`] intervals
    /// and [`n_t_fine`] steps per expiry.
    pub fn fine_reference(
        s0: f64,
        half_width: f64,
        expiries: &[f64],
        dividends: &[(f64, f64)],
        rates: RateInput<'_>,
        q: &[f64],
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            s0, half_width, N_X_FINE, expiries, dividends, rates, q, &n_t_fine,
        )
    }

    /// Number of expiries.
    #[inline]
    pub fn n_expiries(&self) -> usize {
        self.expiries.len()
    }

    /// The carry interval containing `t` (`[a_k, b_k)`; the last interval
    /// for `t` beyond the last expiry).
    pub fn interval_of(&self, t: f64) -> usize {
        let k = self.expiries.partition_point(|&e| e <= t);
        k.min(self.expiries.len() - 1)
    }

    /// Interval index of every step of expiry `e` (by step mid-time).
    pub fn step_intervals(&self, e: usize) -> Vec<usize> {
        self.meshes[e]
            .t_mid
            .iter()
            .map(|&tm| self.interval_of(tm))
            .collect()
    }

    /// Market slices at the carry vector `q` (rates and dividends
    /// unchanged; `carry[n] = q[interval_of(t_mid[n])]`).
    pub fn markets_for(&self, q: &[f64]) -> Result<Vec<MarketSlice>, RustyQLibError> {
        if q.len() != self.n_expiries() || q.iter().any(|v| !v.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "q",
                format!("expected {} finite carries, got {q:?}", self.n_expiries()),
            ));
        }
        Ok(self
            .meshes
            .iter()
            .zip(&self.markets)
            .map(|(mesh, mk)| {
                let carry = mesh
                    .t_mid
                    .iter()
                    .map(|&tm| q[self.interval_of(tm)])
                    .collect();
                MarketSlice {
                    s0: mk.s0,
                    rate: mk.rate.clone(),
                    carry,
                    dividends: mk.dividends.clone(),
                }
            })
            .collect())
    }

    /// Replace the carry vector and rebuild the market slices.
    pub fn set_q(&mut self, q: &[f64]) -> Result<(), RustyQLibError> {
        self.markets = self.markets_for(q)?;
        self.q = q.to_vec();
        Ok(())
    }

    /// `(r_eff, q_eff, forward, df)` of expiry `e` at the current carry.
    pub fn effective_rates(&self, e: usize) -> (f64, f64, f64, f64) {
        effective_rates(&self.meshes[e], &self.markets[e])
    }

    /// The `(t, ln F(t))` table of the current carry and dividends at
    /// `t = 0` and every expiry: the prior forward of the B-spline
    /// coordinate when built at the prior `q`.
    pub fn f_ref_table(&self) -> Vec<(f64, f64)> {
        let mut table = vec![(0.0, self.s0.ln())];
        for (mesh, mk) in self.meshes.iter().zip(&self.markets) {
            let f = *forward_at_levels(mesh, mk)
                .last()
                .expect("mesh has a terminal level");
            table.push((mesh.t_expiry, f.max(1e-300).ln()));
        }
        table
    }
}

// ── Path, flags, outputs ─────────────────────────────────────────────────

/// How a path point was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathKind {
    /// A regular continuation level.
    Level,
    /// The bisection point in `log alpha` between the last two levels.
    Bisection,
}

/// One point of the alpha continuation path.
#[derive(Debug, Clone, PartialEq)]
pub struct PathPoint {
    /// Regularization weight.
    pub alpha: f64,
    /// `sum_{two-sided in-sample} r_i^2` at the level's converged point.
    pub discrepancy: f64,
    /// `theta^T Q theta`.
    pub regularizer: f64,
    /// Full objective `J`.
    pub objective: f64,
    /// Accepted LM iterations at the level.
    pub iterations: usize,
    /// Jacobian passes spent at the level (accepted and rejected trials).
    pub jacobians: usize,
    /// One-sided quotes whose active status changed over the level.
    pub active_set_switches: usize,
    /// Level or bisection.
    pub kind: PathKind,
}

/// Diagnostic flags of a calibration.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CalibrationFlags {
    /// No level satisfied the discrepancy principle.
    pub discrepancy_not_reached: bool,
    /// The Jacobian budget stopped the calibration.
    pub jacobian_budget_exhausted: bool,
    /// The noise-floor plateau rule stopped the continuation.
    pub plateau_stop: bool,
    /// Some level ended because no trial step decreased the objective.
    pub lm_stagnated: bool,
    /// The bisection point replaced the last level as `alpha*`.
    pub bisection_accepted: bool,
    /// `alpha_0` came from the cold rule (no warm value supplied).
    pub cold_alpha0: bool,
    /// The first level already satisfied the target and the continuation
    /// searched upward in `alpha` to bracket `alpha*`.
    pub alpha_searched_upward: bool,
    /// The upward search ended (level cap, budget or a discrepancy that
    /// does not grow with `alpha`) without a level above the target:
    /// `alpha*` is the largest satisfying `alpha` seen, not bracketed.
    pub alpha_unbracketed_above: bool,
    /// The level cap was hit before the discrepancy.
    pub max_levels_reached: bool,
    /// Cholesky failures of the damped normal matrix (each increased
    /// `lambda_LM`).
    pub cholesky_failures: usize,
    /// Penalty-inconsistent steps summed over the quotes of the final
    /// Jacobian pass.
    pub penalty_inconsistent_steps: usize,
}

/// A display grid `(y, t)` on which the data-support map is accumulated.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportGrid {
    /// Forward log-moneyness nodes, increasing.
    pub y: Vec<f64>,
    /// Time nodes, increasing.
    pub t: Vec<f64>,
}

/// The data-support map `S(y, t) = sum_i sum_{(j,n) -> (y,t)} |g_i(j,n)| /
/// s_i` accumulated over every quote in a Jacobian pass by nearest-node
/// binning of the mesh points; `values[iy * t.len() + it]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SupportMap {
    /// The grid's `y` nodes.
    pub y: Vec<f64>,
    /// The grid's `t` nodes.
    pub t: Vec<f64>,
    /// Row-major values, `y` outer.
    pub values: Vec<f64>,
}

impl SupportMap {
    /// Value at `(iy, it)`.
    #[inline]
    pub fn at(&self, iy: usize, it: usize) -> f64 {
        self.values[iy * self.t.len() + it]
    }

    /// Mask of the cells with support at least `fraction` of the maximum.
    pub fn mask(&self, fraction: f64) -> Vec<bool> {
        let max = self.values.iter().cloned().fold(0.0, f64::max);
        self.values.iter().map(|&v| v >= fraction * max).collect()
    }
}

/// Outcome of [`calibrate_american`] / [`calibrate_european`].
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Calibrated coefficients (index `j_y * n_t_basis + j_t`).
    pub theta: Vec<f64>,
    /// Calibrated carry per expiry interval (the prior when frozen).
    pub q: Vec<f64>,
    /// Selected `alpha*`.
    pub alpha: f64,
    /// Continuation path, sorted by decreasing `alpha`.
    pub path: Vec<PathPoint>,
    /// Model price of every quote at the solution (input order).
    pub prices: Vec<f64>,
    /// Scaled residual of every quote (input order; held-out and one-sided
    /// quotes with their own formula).
    pub residuals: Vec<f64>,
    /// `sum g` (the parallel-shift vega) of every quote at the solution.
    pub vegas: Vec<f64>,
    /// Active status of every quote at the solution (`false` only for
    /// inactive one-sided quotes).
    pub active: Vec<bool>,
    /// The last Jacobian rows at the solution, `N x n_cols` (input order):
    /// `dF_i/dtheta` then `dF_i/dq_k`, unscaled.
    pub rows: Vec<f64>,
    /// Columns per row (`M + E`).
    pub n_cols: usize,
    /// Number of coefficients `M`.
    pub n_theta: usize,
    /// Discrepancy `sum_{two-sided in-sample} r_i^2` at the solution.
    pub discrepancy: f64,
    /// Discrepancy target the continuation aimed at.
    pub discrepancy_target: f64,
    /// `theta^T Q theta` at the solution.
    pub regularizer: f64,
    /// Objective at the solution.
    pub objective: f64,
    /// Jacobian passes performed.
    pub jacobians: usize,
    /// Accepted LM iterations performed.
    pub iterations: usize,
    /// Wall-clock of the calibration in milliseconds.
    pub wall_ms: f64,
    /// Standard error of each carry from the inverse normal matrix
    /// (residual units: half-spreads); empty when frozen.
    pub q_se: Vec<f64>,
    /// Multiple correlation of each carry with the coefficients,
    /// `sqrt(1 - 1 / (H_kk (H^-1)_kk))`; empty when frozen.
    pub q_theta_correlation: Vec<f64>,
    /// Data-support map accumulated in the final Jacobian pass.
    pub support_map: Option<SupportMap>,
    /// Diagnostic flags.
    pub flags: CalibrationFlags,
}

impl CalibrationResult {
    /// Jacobian row of quote `i` (unscaled).
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        &self.rows[i * self.n_cols..(i + 1) * self.n_cols]
    }

    /// `dF_i/dtheta` block of quote `i`.
    #[inline]
    pub fn row_theta(&self, i: usize) -> &[f64] {
        &self.rows[i * self.n_cols..i * self.n_cols + self.n_theta]
    }

    /// First-order prediction `J^A_i . (theta - theta_ref) / nu^A_i` of
    /// the American implied-vol change from `theta_ref` to the solution,
    /// from the final rows (the `pred_at_star` column of the study).
    pub fn prediction_from(&self, theta_ref: &[f64]) -> Vec<f64> {
        (0..self.prices.len())
            .map(|i| dot(self.row_theta(i), &self.theta) - dot(self.row_theta(i), theta_ref))
            .zip(&self.vegas)
            .map(|(d, &v)| if v > 0.0 { d / v } else { f64::NAN })
            .collect()
    }
}

/// One Jacobian pass at a fixed point, for callers that need the rows
/// (first-order predictions) or want to time a pass.
#[derive(Debug, Clone)]
pub struct JacobianSnapshot {
    /// `N x n_cols` rows in input order (`dF/dtheta`, then `dF/dq`).
    pub rows: Vec<f64>,
    /// Model prices.
    pub prices: Vec<f64>,
    /// `sum g` per quote.
    pub vegas: Vec<f64>,
    /// Active status per quote.
    pub active: Vec<bool>,
    /// Columns per row.
    pub n_cols: usize,
    /// Coefficients `M`.
    pub n_theta: usize,
    /// Data-support map when a grid was supplied.
    pub support_map: Option<SupportMap>,
    /// Wall-clock of the pass in milliseconds (excluding cache
    /// construction).
    pub wall_ms: f64,
    /// Wall-clock of building the basis caches, in milliseconds.
    pub setup_ms: f64,
}

/// Outcome of the linearized step [`linearized_step`] (M1).
#[derive(Debug, Clone)]
pub struct Linearized {
    /// The start `theta_0`.
    pub theta0: Vec<f64>,
    /// The start (prior) carry.
    pub q0: Vec<f64>,
    /// `clamp(theta_0 + delta_theta(alpha_lin))`.
    pub theta_lin: Vec<f64>,
    /// `q_0 + delta_q(alpha_lin)` (equal to `q0` when frozen).
    pub q_lin: Vec<f64>,
    /// The alpha selected on the linear discrepancy path.
    pub alpha_lin: f64,
    /// The linear path (discrepancy of the linear residual `r_0 + J
    /// delta`), sorted by decreasing alpha.
    pub path: Vec<PathPoint>,
    /// Jacobian rows of the primary mode at `theta_0`, `N x n_cols`.
    pub rows_a: Vec<f64>,
    /// European Jacobian rows at `theta_0`, `N x n_cols` (the European
    /// kernels; identical to `rows_a` when the primary mode is European).
    pub rows_e: Vec<f64>,
    /// `nu^A_i = sum g^A_i`.
    pub vega_a: Vec<f64>,
    /// `nu^E_i = sum g^E_i`.
    pub vega_e: Vec<f64>,
    /// Primary-mode prices at `theta_0`.
    pub prices_a: Vec<f64>,
    /// European prices at `theta_0`.
    pub prices_e: Vec<f64>,
    /// Active status at `theta_0`.
    pub active: Vec<bool>,
    /// Columns per row.
    pub n_cols: usize,
    /// Coefficients `M`.
    pub n_theta: usize,
    /// Jacobian passes (2 for American, 1 for European).
    pub jacobians: usize,
    /// Discrepancy target used.
    pub discrepancy_target: f64,
    /// Flags (`discrepancy_not_reached`, `plateau_stop`, ...).
    pub flags: CalibrationFlags,
    /// Wall-clock in milliseconds.
    pub wall_ms: f64,
}

impl Linearized {
    /// `dF^A_i/dtheta` block.
    #[inline]
    pub fn row_a(&self, i: usize) -> &[f64] {
        &self.rows_a[i * self.n_cols..i * self.n_cols + self.n_theta]
    }

    /// `dF^E_i/dtheta` block.
    #[inline]
    pub fn row_e(&self, i: usize) -> &[f64] {
        &self.rows_e[i * self.n_cols..i * self.n_cols + self.n_theta]
    }

    /// The first-order prediction of the identification error at every
    /// quote for the coefficients `theta_star`:
    /// `pred_i = J^A_i . d / nu^A_i - J^E_i . d / nu^E_i`, `d = theta_star -
    /// theta_0` (`NaN` where a vega is not positive).
    pub fn prediction(&self, theta_star: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = theta_star
            .iter()
            .zip(&self.theta0)
            .map(|(a, b)| a - b)
            .collect();
        (0..self.prices_a.len())
            .map(|i| {
                let (va, ve) = (self.vega_a[i], self.vega_e[i]);
                if va > 0.0 && ve > 0.0 {
                    dot(self.row_a(i), &d) / va - dot(self.row_e(i), &d) / ve
                } else {
                    f64::NAN
                }
            })
            .collect()
    }

    /// The paper's first-order M1 surface at the quotes, `sigma_0 + J^E_i
    /// . (theta - theta_0) / nu^E_i` (`NaN` where `nu^E` is not positive).
    pub fn first_order_sigma_e(&self, sigma0: f64, theta: &[f64]) -> Vec<f64> {
        let d: Vec<f64> = theta.iter().zip(&self.theta0).map(|(a, b)| a - b).collect();
        (0..self.prices_e.len())
            .map(|i| {
                let ve = self.vega_e[i];
                if ve > 0.0 {
                    sigma0 + dot(self.row_e(i), &d) / ve
                } else {
                    f64::NAN
                }
            })
            .collect()
    }
}

// ── Small linear algebra ─────────────────────────────────────────────────

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn quad_form(a: &[Vec<f64>], x: &[f64]) -> f64 {
    dot(&mat_vec(a, x), x)
}

// ── The Jacobian engine ──────────────────────────────────────────────────

/// Per-worker scratch of the Jacobian pass.
struct ThreadState {
    ws: Workspace,
    aws: AdjointWorkspace,
    sol: Solution,
    partial: Vec<f64>,
    support: Vec<f64>,
}

/// Nearest-node bins of every mesh point of every expiry on the display
/// grid (`NO_BIN` outside the `y` range).
struct SupportBins {
    n_t: usize,
    n_cells: usize,
    y_bin: Vec<Vec<u16>>,
    t_bin: Vec<Vec<u16>>,
}

/// Nearest index of `v` in the increasing `grid`, `None` when `v` lies
/// more than half a local spacing outside the grid.
fn nearest_bin(grid: &[f64], v: f64) -> Option<usize> {
    let n = grid.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(0);
    }
    let pos = grid.partition_point(|&g| g < v);
    let i = if pos == 0 {
        0
    } else if pos >= n {
        n - 1
    } else if (grid[pos] - v).abs() < (v - grid[pos - 1]).abs() {
        pos
    } else {
        pos - 1
    };
    let spacing = if i + 1 < n {
        grid[i + 1] - grid[i]
    } else {
        grid[i] - grid[i - 1]
    };
    if (v - grid[i]).abs() <= 0.5 * spacing + 1e-12 {
        Some(i)
    } else {
        None
    }
}

impl SupportBins {
    fn build(grid: &SupportGrid, capture: &CaptureMeshes, surface: &BSplineLocalVol) -> Self {
        let n_t = grid.t.len();
        let mut y_bin = Vec::with_capacity(capture.n_expiries());
        let mut t_bin = Vec::with_capacity(capture.n_expiries());
        for mesh in &capture.meshes {
            let mut yb = Vec::with_capacity(mesh.n_steps() * mesh.n_nodes());
            let mut tb = Vec::with_capacity(mesh.n_steps());
            for &tm in &mesh.t_mid {
                let lf = surface.ln_f_ref(tm);
                tb.push(nearest_bin(&grid.t, tm).map_or(NO_BIN, |i| i as u16));
                for &x in &mesh.x {
                    yb.push(nearest_bin(&grid.y, x - lf).map_or(NO_BIN, |i| i as u16));
                }
            }
            y_bin.push(yb);
            t_bin.push(tb);
        }
        SupportBins {
            n_t,
            n_cells: grid.y.len() * n_t,
            y_bin,
            t_bin,
        }
    }
}

/// Outputs of one Jacobian pass over the (sorted) quotes.
#[derive(Clone)]
struct PassData {
    rows: Vec<f64>,
    prices: Vec<f64>,
    vegas: Vec<f64>,
    active: Vec<bool>,
    support: Vec<f64>,
    penalty_inconsistent: usize,
}

impl PassData {
    fn new(n: usize, w: usize, n_cells: usize) -> Self {
        PassData {
            rows: vec![0.0; n * w],
            prices: vec![f64::NAN; n],
            vegas: vec![f64::NAN; n],
            active: vec![true; n],
            support: vec![0.0; n_cells],
            penalty_inconsistent: 0,
        }
    }
}

/// The streaming Jacobian: thread-local workspaces, per-expiry basis
/// caches and node fields, quotes sorted by descending mesh length.
struct Engine<'a> {
    quotes: Vec<Quote>,
    orig: Vec<usize>,
    specs: Vec<QuoteSpec>,
    capture: &'a CaptureMeshes,
    caches: Vec<BasisCache>,
    step_interval: Vec<Vec<usize>>,
    bins: Option<SupportBins>,
    fields: Vec<NodeField>,
    markets: Vec<MarketSlice>,
    m: usize,
    e: usize,
    w: usize,
    mode: Mode,
    states: Vec<Mutex<ThreadState>>,
    data: PassData,
    passes: usize,
    setup_ms: f64,
}

impl<'a> Engine<'a> {
    fn new(
        quotes: &[Quote],
        capture: &'a CaptureMeshes,
        surface: &BSplineLocalVol,
        mode: Mode,
        carry_free: bool,
        support_grid: Option<&SupportGrid>,
    ) -> Result<Self, RustyQLibError> {
        let setup = Instant::now();
        if quotes.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "quotes",
                "at least one quote is required",
            ));
        }
        for q in quotes {
            q.validate(capture)?;
        }
        for (mesh, mk) in capture.meshes.iter().zip(&capture.markets) {
            mk.validate(mesh)?;
        }
        // longest meshes first: the work per quote is proportional to the
        // step count and varies ~6x across the tenor mix
        let mut orig: Vec<usize> = (0..quotes.len()).collect();
        orig.sort_by_key(|&i| std::cmp::Reverse(capture.meshes[quotes[i].expiry_index].n_steps()));
        let sorted: Vec<Quote> = orig.iter().map(|&i| quotes[i].clone()).collect();
        let specs: Vec<QuoteSpec> = sorted
            .iter()
            .map(|q| q.spec(capture.meshes[q.expiry_index].t_expiry))
            .collect();
        let caches: Vec<BasisCache> = capture
            .meshes
            .par_iter()
            .map(|mesh| surface.basis_cache(&mesh.x, &mesh.t_mid))
            .collect();
        let step_interval = (0..capture.n_expiries())
            .map(|e| capture.step_intervals(e))
            .collect();
        let bins = support_grid.map(|g| SupportBins::build(g, capture, surface));
        let n_cells = bins.as_ref().map_or(0, |b| b.n_cells);
        let fields = caches
            .iter()
            .map(|c| c.node_field(&surface.theta))
            .collect();
        let m = surface.len();
        let e = if carry_free { capture.n_expiries() } else { 0 };
        let w = m + e;
        let n_threads = rayon::current_num_threads();
        let states = (0..=n_threads)
            .map(|_| {
                Mutex::new(ThreadState {
                    ws: Workspace::new(),
                    aws: AdjointWorkspace::new(),
                    sol: Solution::new(),
                    partial: vec![0.0; surface.n_y()],
                    support: vec![0.0; n_cells],
                })
            })
            .collect();
        let data = PassData::new(sorted.len(), w, n_cells);
        Ok(Engine {
            quotes: sorted,
            orig,
            specs,
            capture,
            caches,
            step_interval,
            bins,
            fields,
            markets: capture.markets.clone(),
            m,
            e,
            w,
            mode,
            states,
            data,
            passes: 0,
            setup_ms: setup.elapsed().as_secs_f64() * 1e3,
        })
    }

    fn n(&self) -> usize {
        self.quotes.len()
    }

    /// One Jacobian pass at `(theta, q)`: fills `self.data`.
    fn run(&mut self, theta: &[f64], q: &[f64]) -> Result<(), RustyQLibError> {
        if theta.len() != self.m {
            return Err(RustyQLibError::invalid_input(
                "theta",
                format!("expected {} coefficients, got {}", self.m, theta.len()),
            ));
        }
        if theta.iter().any(|v| !(v.is_finite() && *v > 0.0)) {
            return Err(RustyQLibError::NumericalError(
                "non-positive or non-finite coefficient in theta".to_string(),
            ));
        }
        // per-expiry node fields of this theta (shared by the expiry's quotes)
        let caches = &self.caches;
        self.fields
            .par_iter_mut()
            .zip(caches.par_iter())
            .for_each(|(f, c)| c.fill_node_values(theta, &mut f.values));
        if self.e > 0 {
            self.markets = self.capture.markets_for(q)?;
        }
        for st in &self.states {
            let mut st = st.lock().unwrap_or_else(|p| p.into_inner());
            st.support.iter_mut().for_each(|v| *v = 0.0);
        }
        let n = self.n();
        let (m, e, w) = (self.m, self.e, self.w);
        let quotes = &self.quotes;
        let specs = &self.specs;
        let capture = self.capture;
        let step_interval = &self.step_interval;
        let bins = self.bins.as_ref();
        let fields = &self.fields;
        let markets = &self.markets;
        let mode = self.mode;
        let states = &self.states;
        let fallback = states.len() - 1;
        let mut errors: Vec<Option<RustyQLibError>> = (0..n).map(|_| None).collect();
        let mut inconsistent: Vec<usize> = vec![0; n];
        let data = &mut self.data;
        data.rows
            .par_chunks_mut(w)
            .zip(data.prices.par_iter_mut())
            .zip(data.vegas.par_iter_mut())
            .zip(data.active.par_iter_mut())
            .zip(errors.par_iter_mut())
            .zip(inconsistent.par_iter_mut())
            .enumerate()
            .for_each(|(p, (((((row, price), vega), active), err), inc))| {
                let idx = rayon::current_thread_index().unwrap_or(fallback);
                let mut st = states[idx].lock().unwrap_or_else(|e| e.into_inner());
                let ThreadState {
                    ws,
                    aws,
                    sol,
                    partial,
                    support,
                } = &mut *st;
                let quote = &quotes[p];
                let ex = quote.expiry_index;
                let mesh = &capture.meshes[ex];
                let market = &markets[ex];
                let field = &fields[ex];
                let cache = &caches[ex];
                let spec = &specs[p];
                row.iter_mut().for_each(|v| *v = 0.0);
                let res: Result<(f64, f64), RustyQLibError> = (|| {
                    solve_into(mesh, spec, field, market, mode, true, ws, sol)?;
                    let pr = sol.price;
                    *inc = sol.penalty_inconsistent_steps;
                    let stride = mesh.n_nodes();
                    let nt = cache.n_t_basis;
                    let partial = &mut partial[..cache.n_y];
                    let inv_s = 1.0 / quote.scale();
                    let (row_theta, row_q) = row.split_at_mut(m);
                    let si = &step_interval[ex][..];
                    let sup = bins.map(|b| (&b.y_bin[ex][..], &b.t_bin[ex][..], b.n_t));
                    let nu =
                        adjoint_stream(mesh, spec, sol, market, field, aws, |step, g_row, h| {
                            partial.iter_mut().for_each(|v| *v = 0.0);
                            let base = step * stride;
                            match sup {
                                Some((yb, tb, n_tg)) if tb[step] != NO_BIN => {
                                    let tbin = tb[step] as usize;
                                    for j in 1..stride - 1 {
                                        let g = g_row[j];
                                        if g == 0.0 {
                                            continue;
                                        }
                                        let k = base + j;
                                        let by = &cache.y_vals[k];
                                        let y0 = cache.y_first[k];
                                        partial[y0] += g * by[0];
                                        partial[y0 + 1] += g * by[1];
                                        partial[y0 + 2] += g * by[2];
                                        partial[y0 + 3] += g * by[3];
                                        let yb_k = yb[k];
                                        if yb_k != NO_BIN {
                                            support[yb_k as usize * n_tg + tbin] += g.abs() * inv_s;
                                        }
                                    }
                                }
                                _ => {
                                    for j in 1..stride - 1 {
                                        let g = g_row[j];
                                        if g == 0.0 {
                                            continue;
                                        }
                                        let k = base + j;
                                        let by = &cache.y_vals[k];
                                        let y0 = cache.y_first[k];
                                        partial[y0] += g * by[0];
                                        partial[y0 + 1] += g * by[1];
                                        partial[y0 + 2] += g * by[2];
                                        partial[y0 + 3] += g * by[3];
                                    }
                                }
                            }
                            let bt = cache.t_vals[step];
                            let t0 = cache.t_first[step];
                            for (a, &pv) in partial.iter().enumerate() {
                                if pv == 0.0 {
                                    continue;
                                }
                                let r = a * nt + t0;
                                row_theta[r] += pv * bt[0];
                                row_theta[r + 1] += pv * bt[1];
                                row_theta[r + 2] += pv * bt[2];
                                row_theta[r + 3] += pv * bt[3];
                            }
                            if e > 0 {
                                row_q[si[step]] -= h;
                            }
                        })?;
                    Ok((pr, nu))
                })();
                match res {
                    Ok((pr, nu)) => {
                        *price = pr;
                        *vega = nu;
                        *active = quote.active_at(pr);
                    }
                    Err(er) => *err = Some(er),
                }
            });
        if let Some(er) = errors.into_iter().flatten().next() {
            return Err(er);
        }
        data.penalty_inconsistent = inconsistent.iter().sum();
        data.support.iter_mut().for_each(|v| *v = 0.0);
        for st in &self.states {
            let st = st.lock().unwrap_or_else(|p| p.into_inner());
            for (o, &v) in data.support.iter_mut().zip(&st.support) {
                *o += v;
            }
        }
        self.passes += 1;
        Ok(())
    }

    /// A vector in sorted order mapped back to input order.
    fn unsort<T: Clone>(&self, v: &[T]) -> Vec<T> {
        let mut out = vec![v[0].clone(); v.len()];
        for (p, &i) in self.orig.iter().enumerate() {
            out[i] = v[p].clone();
        }
        out
    }

    /// Rows in sorted order mapped back to input order.
    fn unsort_rows(&self, rows: &[f64]) -> Vec<f64> {
        let w = self.w;
        let mut out = vec![0.0; rows.len()];
        for (p, &i) in self.orig.iter().enumerate() {
            out[i * w..(i + 1) * w].copy_from_slice(&rows[p * w..(p + 1) * w]);
        }
        out
    }

    fn support_map(&self, grid: Option<&SupportGrid>, data: &PassData) -> Option<SupportMap> {
        grid.map(|g| SupportMap {
            y: g.y.clone(),
            t: g.t.clone(),
            values: data.support.clone(),
        })
    }
}

// ── Normal equations and objective ───────────────────────────────────────

/// Shared read-only context of one calibration.
struct Ctx<'a> {
    cfg: &'a CalibrationConfig,
    q_mat: Vec<Vec<f64>>,
    q_prior: Vec<f64>,
    m: usize,
    e: usize,
    w: usize,
}

/// Objective pieces at a pass.
#[derive(Debug, Clone, Copy)]
struct Eval {
    objective: f64,
    discrepancy: f64,
    regularizer: f64,
}

fn evaluate(
    ctx: &Ctx<'_>,
    quotes: &[Quote],
    prices: &[f64],
    alpha: f64,
    theta: &[f64],
    q: &[f64],
) -> Eval {
    let mut fit = 0.0;
    let mut disc = 0.0;
    for (quote, &price) in quotes.iter().zip(prices) {
        if !quote.in_fit() {
            continue;
        }
        let r = quote.residual(price);
        fit += r * r;
        if !quote.at_intrinsic {
            disc += r * r;
        }
    }
    let reg = quad_form(&ctx.q_mat, theta);
    let mut carry = 0.0;
    if ctx.e > 0 {
        let inv = 1.0 / ctx.cfg.q_scale;
        for (qk, qp) in q.iter().zip(&ctx.q_prior) {
            let d = (qk - qp) * inv;
            carry += d * d;
        }
    }
    Eval {
        objective: fit + alpha * reg + ctx.cfg.beta_carry * carry,
        discrepancy: disc,
        regularizer: reg,
    }
}

/// `J^T J` (lower triangle mirrored) and `J^T r` over the fit rows, with
/// the weights `1/s_i^2` and the one-sided activity of `data`.
fn assemble_jtj(quotes: &[Quote], data: &PassData, w: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = quotes.len();
    let use_row: Vec<Option<(f64, f64)>> = (0..n)
        .map(|p| {
            let q = &quotes[p];
            if q.in_fit() && data.active[p] {
                let s = q.scale();
                Some((1.0 / (s * s), q.misfit(data.prices[p])))
            } else {
                None
            }
        })
        .collect();
    let rows = &data.rows;
    let mut jtj: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|a| {
            let mut row = vec![0.0; w];
            for p in 0..n {
                if let Some((wt, _)) = use_row[p] {
                    let rp = &rows[p * w..(p + 1) * w];
                    let wa = wt * rp[a];
                    if wa == 0.0 {
                        continue;
                    }
                    for b in 0..=a {
                        row[b] += wa * rp[b];
                    }
                }
            }
            row
        })
        .collect();
    for a in 0..w {
        for b in 0..a {
            let v = jtj[a][b];
            jtj[b][a] = v;
        }
    }
    let mut jtr = vec![0.0; w];
    for p in 0..n {
        if let Some((wt, misfit)) = use_row[p] {
            let rp = &rows[p * w..(p + 1) * w];
            let c = wt * misfit;
            for a in 0..w {
                jtr[a] += c * rp[a];
            }
        }
    }
    (jtj, jtr)
}

/// `H0 = J^T J + alpha Q + beta/q_scale^2 I_q` and the gradient
/// `g = J^T r + alpha Q theta + beta/q_scale^2 (q - q_prior)`.
fn normal_with_alpha(
    ctx: &Ctx<'_>,
    jtj: &[Vec<f64>],
    jtr: &[f64],
    alpha: f64,
    theta: &[f64],
    q: &[f64],
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let (m, w) = (ctx.m, ctx.w);
    let mut h0 = jtj.to_vec();
    for a in 0..m {
        let qa = &ctx.q_mat[a];
        let ha = &mut h0[a];
        for b in 0..m {
            ha[b] += alpha * qa[b];
        }
    }
    let bq = ctx.cfg.beta_carry / (ctx.cfg.q_scale * ctx.cfg.q_scale);
    for k in m..w {
        h0[k][k] += bq;
    }
    let mut g = jtr.to_vec();
    let qtheta = mat_vec(&ctx.q_mat, theta);
    for a in 0..m {
        g[a] += alpha * qtheta[a];
    }
    for k in 0..ctx.e {
        g[m + k] += bq * (q[k] - ctx.q_prior[k]);
    }
    (h0, g)
}

/// Solve `(H0 + lambda diag H0) delta = -g` by Cholesky.
fn solve_step(h0: &[Vec<f64>], g: &[f64], lambda: f64) -> Result<Vec<f64>, RustyQLibError> {
    let w = g.len();
    let mut h = h0.to_vec();
    for a in 0..w {
        h[a][a] += lambda * h0[a][a].abs();
    }
    let l = cholesky_factor(&h)?;
    let rhs: Vec<f64> = g.iter().map(|v| -v).collect();
    cholesky_solve(&l, &rhs)
}

/// Apply a step and project onto the bounds.
fn apply_step(ctx: &Ctx<'_>, theta: &[f64], q: &[f64], delta: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut theta_new: Vec<f64> = theta.iter().zip(delta).map(|(t, d)| t + d).collect();
    project_onto_bounds(&mut theta_new, ctx.cfg.bounds.0, ctx.cfg.bounds.1);
    // frozen carry: no q columns, the prior is carried through unchanged
    let mut q_new: Vec<f64> = if ctx.e == 0 {
        q.to_vec()
    } else {
        q.iter()
            .zip(&delta[ctx.m..])
            .map(|(qk, d)| qk + d)
            .collect()
    };
    project_onto_bounds(&mut q_new, ctx.cfg.q_bounds.0, ctx.cfg.q_bounds.1);
    (theta_new, q_new)
}

/// Discrepancy target and `N_eff`.
fn discrepancy_target(quotes: &[Quote], cfg: &CalibrationConfig) -> (f64, usize) {
    let mut n_eff = 0usize;
    let mut floor_sum = 0.0;
    for q in quotes {
        if q.two_sided_in_fit() {
            n_eff += 1;
            if let Some(d) = cfg.noise_floor {
                let s = q.scale();
                floor_sum += (d / s) * (d / s);
            }
        }
    }
    let target = match cfg.noise_floor {
        Some(_) => floor_sum,
        None => cfg.tau * n_eff as f64,
    };
    (target, n_eff)
}

/// Cold `alpha_0 = 10 max diag(J^T J)_theta / max diag(Q)`.
fn cold_alpha0(jtj: &[Vec<f64>], q_mat: &[Vec<f64>], m: usize) -> f64 {
    let max_j = (0..m).map(|a| jtj[a][a]).fold(0.0, f64::max);
    let max_q = (0..m).map(|a| q_mat[a][a]).fold(0.0, f64::max);
    if max_j > 0.0 && max_q > 0.0 {
        10.0 * max_j / max_q
    } else {
        1.0
    }
}

/// Carry standard errors and multiple correlations from `H0^{-1}`.
fn carry_uncertainty(h0: &[Vec<f64>], m: usize, e: usize) -> (Vec<f64>, Vec<f64>) {
    let mut se = vec![f64::NAN; e];
    let mut corr = vec![f64::NAN; e];
    if e == 0 {
        return (se, corr);
    }
    let l = match cholesky_factor(h0) {
        Ok(l) => l,
        Err(err) => {
            log::warn!("carry uncertainty: normal matrix not factorizable ({err})");
            return (se, corr);
        }
    };
    let w = h0.len();
    for k in 0..e {
        let mut unit = vec![0.0; w];
        unit[m + k] = 1.0;
        if let Ok(col) = cholesky_solve(&l, &unit) {
            let var = col[m + k];
            if var > 0.0 {
                se[k] = var.sqrt();
                let hkk = h0[m + k][m + k];
                corr[k] = (1.0 - 1.0 / (hkk * var)).max(0.0).sqrt();
            }
        }
    }
    (se, corr)
}

// ── Levenberg–Marquardt ──────────────────────────────────────────────────

/// The accepted point of the iteration.
#[derive(Clone)]
struct Accepted {
    theta: Vec<f64>,
    q: Vec<f64>,
    data: PassData,
    eval: Eval,
}

#[derive(Debug, Clone, Copy, Default)]
struct LevelStats {
    iterations: usize,
    jacobians: usize,
    switches: usize,
    stagnated: bool,
    budget_hit: bool,
    cholesky_failures: usize,
}

/// Projected LM at fixed `alpha` from `acc`, updating it in place.
#[allow(clippy::too_many_arguments)]
fn lm_level(
    ctx: &Ctx<'_>,
    engine: &mut Engine<'_>,
    acc: &mut Accepted,
    alpha: f64,
    max_iters: usize,
    rel_stop: f64,
    jac_count: &mut usize,
) -> Result<LevelStats, RustyQLibError> {
    let cfg = ctx.cfg;
    let mut stats = LevelStats::default();
    let mut lambda = cfg.lm_lambda0;
    // the objective is re-evaluated at this level's alpha
    acc.eval = evaluate(
        ctx,
        &engine.quotes,
        &acc.data.prices,
        alpha,
        &acc.theta,
        &acc.q,
    );
    while stats.iterations < max_iters {
        if *jac_count >= cfg.jacobian_budget {
            stats.budget_hit = true;
            break;
        }
        let (jtj, jtr) = assemble_jtj(&engine.quotes, &acc.data, ctx.w);
        let (h0, g) = normal_with_alpha(ctx, &jtj, &jtr, alpha, &acc.theta, &acc.q);
        let mut accepted = false;
        let mut rel = 0.0;
        loop {
            if *jac_count >= cfg.jacobian_budget {
                stats.budget_hit = true;
                break;
            }
            let delta = match solve_step(&h0, &g, lambda) {
                Ok(d) => d,
                Err(err) => {
                    stats.cholesky_failures += 1;
                    log::debug!("lm: Cholesky failed at lambda = {lambda} ({err}); damping more");
                    lambda = (lambda.max(1e-8)) * cfg.lm_lambda_factor;
                    if lambda > LM_LAMBDA_MAX {
                        break;
                    }
                    continue;
                }
            };
            let (theta_t, q_t) = apply_step(ctx, &acc.theta, &acc.q, &delta);
            engine.run(&theta_t, &q_t)?;
            *jac_count += 1;
            stats.jacobians += 1;
            let ev = evaluate(
                ctx,
                &engine.quotes,
                &engine.data.prices,
                alpha,
                &theta_t,
                &q_t,
            );
            if ev.objective.is_finite() && ev.objective < acc.eval.objective {
                rel = (acc.eval.objective - ev.objective) / acc.eval.objective.max(1e-300);
                stats.switches += engine
                    .data
                    .active
                    .iter()
                    .zip(&acc.data.active)
                    .filter(|(a, b)| a != b)
                    .count();
                std::mem::swap(&mut engine.data, &mut acc.data);
                acc.theta = theta_t;
                acc.q = q_t;
                acc.eval = ev;
                lambda = (lambda / cfg.lm_lambda_factor).max(LM_LAMBDA_MIN);
                accepted = true;
                break;
            }
            // a rejection from a vanishing lambda jumps to LM_LAMBDA_REJECT_FLOOR
            // rather than climbing from 1e-12 one factor per wasted Jacobian
            lambda = (lambda * cfg.lm_lambda_factor).max(LM_LAMBDA_REJECT_FLOOR);
            if lambda > LM_LAMBDA_MAX {
                break;
            }
        }
        if !accepted {
            if !stats.budget_hit {
                stats.stagnated = true;
            }
            break;
        }
        stats.iterations += 1;
        if rel < rel_stop {
            break;
        }
    }
    Ok(stats)
}

fn path_point(acc: &Accepted, alpha: f64, stats: LevelStats, kind: PathKind) -> PathPoint {
    PathPoint {
        alpha,
        discrepancy: acc.eval.discrepancy,
        regularizer: acc.eval.regularizer,
        objective: acc.eval.objective,
        iterations: stats.iterations,
        jacobians: stats.jacobians,
        active_set_switches: stats.switches,
        kind,
    }
}

/// The nonlinear calibration shared by [`calibrate_american`] and
/// [`calibrate_european`].
fn calibrate_core(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    mode: Mode,
    alpha0: Option<f64>,
    support_grid: Option<&SupportGrid>,
) -> Result<CalibrationResult, RustyQLibError> {
    let start = Instant::now();
    cfg.validate()?;
    if let Some(a) = alpha0 {
        if !(a.is_finite() && a > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "alpha0",
                format!("alpha_0 must be positive and finite, got {a}"),
            ));
        }
    }
    let carry_free = !cfg.carry_frozen;
    let mut engine = Engine::new(quotes, capture, surface, mode, carry_free, support_grid)?;
    let (lambda_y, lambda_t) = cfg.lambdas.unwrap_or_else(|| surface.default_lambdas());
    let (target, _) = discrepancy_target(quotes, cfg);
    let ctx = Ctx {
        cfg,
        q_mat: surface.regularizer_matrix(lambda_y, lambda_t),
        q_prior: capture.q.clone(),
        m: engine.m,
        e: engine.e,
        w: engine.w,
    };
    let mut theta = surface.theta.clone();
    project_onto_bounds(&mut theta, cfg.bounds.0, cfg.bounds.1);
    let q = capture.q.clone();
    let mut flags = CalibrationFlags::default();

    engine.run(&theta, &q)?;
    let mut jac_count = 1usize;
    let mut alpha = match alpha0 {
        Some(a) => a,
        None => {
            flags.cold_alpha0 = true;
            let (jtj, _) = assemble_jtj(&engine.quotes, &engine.data, ctx.w);
            cold_alpha0(&jtj, &ctx.q_mat, ctx.m)
        }
    };
    let mut acc = Accepted {
        theta: theta.clone(),
        q: q.clone(),
        data: engine.data.clone(),
        eval: evaluate(&ctx, &engine.quotes, &engine.data.prices, alpha, &theta, &q),
    };

    let mut path: Vec<PathPoint> = Vec::new();
    let mut prev: Option<(f64, Accepted)> = None;
    let mut reached = false;
    let mut total_iterations = 0usize;
    for level in 0..cfg.max_alpha_levels {
        let stats = lm_level(
            &ctx,
            &mut engine,
            &mut acc,
            alpha,
            cfg.lm_iters_intermediate,
            cfg.rel_decrease_stop_intermediate,
            &mut jac_count,
        )?;
        total_iterations += stats.iterations;
        flags.cholesky_failures += stats.cholesky_failures;
        flags.lm_stagnated |= stats.stagnated;
        path.push(path_point(&acc, alpha, stats, PathKind::Level));
        log::debug!(
            "calibration level {level}: alpha = {alpha:.4e}, discrepancy = {:.4} (target {:.4}), \
             R = {:.4e}, {} iterations, {} Jacobians",
            acc.eval.discrepancy,
            target,
            acc.eval.regularizer,
            stats.iterations,
            stats.jacobians
        );
        if acc.eval.discrepancy <= target {
            reached = true;
            break;
        }
        if cfg.noise_floor.is_some() {
            if let Some((_, p)) = &prev {
                let before = p.eval.discrepancy;
                if before - acc.eval.discrepancy < cfg.plateau_fraction * before {
                    flags.plateau_stop = true;
                    break;
                }
            }
        }
        if stats.budget_hit || jac_count >= cfg.jacobian_budget {
            flags.jacobian_budget_exhausted = true;
            break;
        }
        if level + 1 == cfg.max_alpha_levels {
            flags.max_levels_reached = true;
            break;
        }
        prev = Some((alpha, acc.clone()));
        alpha /= cfg.alpha_ratio;
    }

    if reached && prev.is_none() {
        // the start already satisfies the target (a warm alpha_0 below
        // alpha*): search UPWARD by the same ratio until a level fails, so
        // that alpha* is bracketed from above and the bisection applies
        flags.alpha_searched_upward = true;
        for _ in 1..cfg.max_alpha_levels {
            if jac_count >= cfg.jacobian_budget {
                flags.jacobian_budget_exhausted = true;
                break;
            }
            let alpha_up = alpha * cfg.alpha_ratio;
            let mut acc_up = acc.clone();
            let stats = lm_level(
                &ctx,
                &mut engine,
                &mut acc_up,
                alpha_up,
                cfg.lm_iters_intermediate,
                cfg.rel_decrease_stop_intermediate,
                &mut jac_count,
            )?;
            total_iterations += stats.iterations;
            flags.cholesky_failures += stats.cholesky_failures;
            flags.lm_stagnated |= stats.stagnated;
            path.push(path_point(&acc_up, alpha_up, stats, PathKind::Level));
            if acc_up.eval.discrepancy > target {
                prev = Some((alpha_up, acc_up));
                break;
            }
            // a discrepancy that does not grow with alpha (data fitted by
            // the regularizer's kernel, e.g. a flat truth) never brackets:
            // keep the largest alpha that satisfies the target and stop
            let grew = acc_up.eval.discrepancy - acc.eval.discrepancy
                > cfg.plateau_fraction * acc.eval.discrepancy.max(1e-12 * target.max(1e-300));
            acc = acc_up;
            alpha = alpha_up;
            if !grew {
                break;
            }
        }
        if prev.is_none() {
            flags.alpha_unbracketed_above = true;
        }
    }
    // index of alpha*'s path point (the upward search may have pushed
    // levels after it)
    let star_idx = path
        .iter()
        .position(|p| p.alpha == alpha)
        .unwrap_or(path.len().saturating_sub(1));

    if reached {
        // polish at alpha* with the final settings
        if jac_count < cfg.jacobian_budget {
            let stats = lm_level(
                &ctx,
                &mut engine,
                &mut acc,
                alpha,
                cfg.lm_iters_final,
                cfg.rel_decrease_stop_final,
                &mut jac_count,
            )?;
            total_iterations += stats.iterations;
            flags.cholesky_failures += stats.cholesky_failures;
            flags.lm_stagnated |= stats.stagnated;
            if let Some(star) = path.get_mut(star_idx) {
                star.iterations += stats.iterations;
                star.jacobians += stats.jacobians;
                star.active_set_switches += stats.switches;
                star.discrepancy = acc.eval.discrepancy;
                star.regularizer = acc.eval.regularizer;
                star.objective = acc.eval.objective;
            }
            if stats.budget_hit {
                flags.jacobian_budget_exhausted = true;
            }
        } else {
            flags.jacobian_budget_exhausted = true;
        }
        // one bisection in log alpha between the last two levels
        if let Some((alpha_prev, _)) = &prev {
            if jac_count < cfg.jacobian_budget {
                let alpha_mid = (alpha_prev * alpha).sqrt();
                let mut acc_mid = acc.clone();
                let stats = lm_level(
                    &ctx,
                    &mut engine,
                    &mut acc_mid,
                    alpha_mid,
                    cfg.lm_iters_final,
                    cfg.rel_decrease_stop_final,
                    &mut jac_count,
                )?;
                total_iterations += stats.iterations;
                flags.cholesky_failures += stats.cholesky_failures;
                flags.lm_stagnated |= stats.stagnated;
                if stats.budget_hit {
                    flags.jacobian_budget_exhausted = true;
                }
                path.push(path_point(&acc_mid, alpha_mid, stats, PathKind::Bisection));
                if acc_mid.eval.discrepancy <= target {
                    acc = acc_mid;
                    alpha = alpha_mid;
                    flags.bisection_accepted = true;
                }
            }
        }
    } else {
        flags.discrepancy_not_reached = true;
    }
    path.sort_by(|a, b| {
        b.alpha
            .partial_cmp(&a.alpha)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // uncertainty of the carry from the final normal matrix
    let (jtj, jtr) = assemble_jtj(&engine.quotes, &acc.data, ctx.w);
    let (h0, _) = normal_with_alpha(&ctx, &jtj, &jtr, alpha, &acc.theta, &acc.q);
    let (q_se, q_theta_correlation) = carry_uncertainty(&h0, ctx.m, ctx.e);
    flags.penalty_inconsistent_steps = acc.data.penalty_inconsistent;

    let prices = engine.unsort(&acc.data.prices);
    let residuals: Vec<f64> = quotes
        .iter()
        .zip(&prices)
        .map(|(q, &p)| q.residual(p))
        .collect();
    let support_map = engine.support_map(support_grid, &acc.data);
    Ok(CalibrationResult {
        theta: acc.theta.clone(),
        q: acc.q.clone(),
        alpha,
        path,
        prices,
        residuals,
        vegas: engine.unsort(&acc.data.vegas),
        active: engine.unsort(&acc.data.active),
        rows: engine.unsort_rows(&acc.data.rows),
        n_cols: ctx.w,
        n_theta: ctx.m,
        discrepancy: acc.eval.discrepancy,
        discrepancy_target: target,
        regularizer: acc.eval.regularizer,
        objective: acc.eval.objective,
        jacobians: jac_count,
        iterations: total_iterations,
        wall_ms: start.elapsed().as_secs_f64() * 1e3,
        q_se,
        q_theta_correlation,
        support_map,
        flags,
    })
}

// ── Public entry points ──────────────────────────────────────────────────

/// M2: the full nonlinear Tikhonov calibration of `surface` (its `theta`
/// is the start, e.g. `theta_lin`) and, unless `cfg.carry_frozen`, the
/// carry per expiry interval (prior `capture.q`) to the American quotes,
/// with the penalty forward map at `cfg.rho`. `alpha0` is the first level
/// of the continuation (`16 alpha_lin` from [`linearized_step`] for the
/// panel); `None` uses the cold rule `10 max diag(J^T J)/max diag(Q)`.
/// The continuation descends by `cfg.alpha_ratio` until the discrepancy
/// principle holds (or ascends when `alpha0` already satisfies it), then
/// bisects once in `log alpha`. The data-support map is accumulated on
/// `support_grid` when given.
pub fn calibrate_american(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    alpha0: Option<f64>,
    support_grid: Option<&SupportGrid>,
) -> Result<CalibrationResult, RustyQLibError> {
    calibrate_core(
        quotes,
        capture,
        surface,
        cfg,
        Mode::American { rho: cfg.rho },
        alpha0,
        support_grid,
    )
}

/// M0: the same Tikhonov calibration with the EUROPEAN forward map, for
/// price-converted identification quotes (see [`identification_quotes`]).
pub fn calibrate_european(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    alpha0: Option<f64>,
    support_grid: Option<&SupportGrid>,
) -> Result<CalibrationResult, RustyQLibError> {
    calibrate_core(
        quotes,
        capture,
        surface,
        cfg,
        Mode::European,
        alpha0,
        support_grid,
    )
}

/// One Jacobian pass at the surface's `theta` and the capture's `q`.
pub fn jacobian_rows(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    mode: Mode,
    support_grid: Option<&SupportGrid>,
) -> Result<JacobianSnapshot, RustyQLibError> {
    cfg.validate()?;
    let mut engine = Engine::new(
        quotes,
        capture,
        surface,
        mode,
        !cfg.carry_frozen,
        support_grid,
    )?;
    let start = Instant::now();
    engine.run(&surface.theta, &capture.q)?;
    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
    Ok(JacobianSnapshot {
        rows: engine.unsort_rows(&engine.data.rows),
        prices: engine.unsort(&engine.data.prices),
        vegas: engine.unsort(&engine.data.vegas),
        active: engine.unsort(&engine.data.active),
        n_cols: engine.w,
        n_theta: engine.m,
        support_map: engine.support_map(support_grid, &engine.data),
        wall_ms,
        setup_ms: engine.setup_ms,
    })
}

/// One projected Gauss–Newton step (`lambda_LM = 0`) at `alpha` from the
/// surface's `theta` and the capture's `q`: the step the LM driver takes
/// first when started there with `lm_lambda0 = 0`. Equals
/// [`linearized_step`]'s solution at the same `alpha` (unit-tested).
pub fn gauss_newton_step(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    mode: Mode,
    alpha: f64,
) -> Result<(Vec<f64>, Vec<f64>), RustyQLibError> {
    cfg.validate()?;
    let mut engine = Engine::new(quotes, capture, surface, mode, !cfg.carry_frozen, None)?;
    let (lambda_y, lambda_t) = cfg.lambdas.unwrap_or_else(|| surface.default_lambdas());
    let ctx = Ctx {
        cfg,
        q_mat: surface.regularizer_matrix(lambda_y, lambda_t),
        q_prior: capture.q.clone(),
        m: engine.m,
        e: engine.e,
        w: engine.w,
    };
    let mut theta = surface.theta.clone();
    project_onto_bounds(&mut theta, cfg.bounds.0, cfg.bounds.1);
    engine.run(&theta, &capture.q)?;
    let (jtj, jtr) = assemble_jtj(&engine.quotes, &engine.data, ctx.w);
    let (h0, g) = normal_with_alpha(&ctx, &jtj, &jtr, alpha, &theta, &capture.q);
    let delta = solve_step(&h0, &g, 0.0)?;
    Ok(apply_step(&ctx, &theta, &capture.q, &delta))
}

/// M1: the linearized problem at the surface's (flat) `theta_0`. One
/// Jacobian pass of `mode` (and one European pass when `mode` is
/// American, for the European kernels), then the linear Tikhonov path
/// `(J^T J + alpha Q + beta/q_scale^2 I) delta = -(J^T r_0 + alpha Q
/// theta_0)` from the cold `alpha_0` down by `cfg.alpha_ratio`, stopping
/// at the first alpha whose linear discrepancy `sum (r_0 + J delta)_i^2`
/// (two-sided in-sample, step clamped to the bounds) meets the target,
/// then one bisection in `log alpha`. Returns `theta_lin`, `alpha_lin`, the
/// path and the rows of both Jacobians.
pub fn linearized_step(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    surface: &BSplineLocalVol,
    cfg: &CalibrationConfig,
    mode: Mode,
) -> Result<Linearized, RustyQLibError> {
    let start = Instant::now();
    cfg.validate()?;
    let carry_free = !cfg.carry_frozen;
    let mut engine = Engine::new(quotes, capture, surface, mode, carry_free, None)?;
    let (lambda_y, lambda_t) = cfg.lambdas.unwrap_or_else(|| surface.default_lambdas());
    let (target, _) = discrepancy_target(quotes, cfg);
    let ctx = Ctx {
        cfg,
        q_mat: surface.regularizer_matrix(lambda_y, lambda_t),
        q_prior: capture.q.clone(),
        m: engine.m,
        e: engine.e,
        w: engine.w,
    };
    let mut theta0 = surface.theta.clone();
    project_onto_bounds(&mut theta0, cfg.bounds.0, cfg.bounds.1);
    let q0 = capture.q.clone();
    engine.run(&theta0, &q0)?;
    let mut jacobians = 1usize;
    let (jtj, jtr) = assemble_jtj(&engine.quotes, &engine.data, ctx.w);
    let mut flags = CalibrationFlags {
        cold_alpha0: true,
        ..Default::default()
    };

    // the linear discrepancy of a (clamped) step
    let linear_eval = |alpha: f64| -> Result<(Vec<f64>, Vec<f64>, f64, f64), RustyQLibError> {
        let (h0, g) = normal_with_alpha(&ctx, &jtj, &jtr, alpha, &theta0, &q0);
        let delta = solve_step(&h0, &g, 0.0)?;
        let (theta_t, q_t) = apply_step(&ctx, &theta0, &q0, &delta);
        let mut d = vec![0.0; ctx.w];
        for a in 0..ctx.m {
            d[a] = theta_t[a] - theta0[a];
        }
        for k in 0..ctx.e {
            d[ctx.m + k] = q_t[k] - q0[k];
        }
        let mut disc = 0.0;
        let w = ctx.w;
        for (p, quote) in engine.quotes.iter().enumerate() {
            if !quote.two_sided_in_fit() {
                continue;
            }
            let row = &engine.data.rows[p * w..(p + 1) * w];
            let r = (quote.misfit(engine.data.prices[p]) + dot(row, &d)) / quote.scale();
            disc += r * r;
        }
        let reg = quad_form(&ctx.q_mat, &theta_t);
        Ok((theta_t, q_t, disc, reg))
    };

    let mut alpha = cold_alpha0(&jtj, &ctx.q_mat, ctx.m);
    let mut path = Vec::new();
    let mut best: Option<(f64, Vec<f64>, Vec<f64>)> = None;
    let mut prev_alpha: Option<f64> = None;
    let mut prev_disc = f64::INFINITY;
    let mut reached = false;
    for level in 0..cfg.max_linear_levels {
        let (theta_t, q_t, disc, reg) = linear_eval(alpha)?;
        path.push(PathPoint {
            alpha,
            discrepancy: disc,
            regularizer: reg,
            objective: disc + alpha * reg,
            iterations: 1,
            jacobians: 0,
            active_set_switches: 0,
            kind: PathKind::Level,
        });
        best = Some((alpha, theta_t, q_t));
        if disc <= target {
            reached = true;
            break;
        }
        if cfg.noise_floor.is_some()
            && level > 0
            && prev_disc - disc < cfg.plateau_fraction * prev_disc
        {
            flags.plateau_stop = true;
            break;
        }
        if level + 1 == cfg.max_linear_levels {
            flags.max_levels_reached = true;
            break;
        }
        prev_alpha = Some(alpha);
        prev_disc = disc;
        alpha /= cfg.alpha_ratio;
    }
    if reached {
        if let Some(ap) = prev_alpha {
            let alpha_mid = (ap * alpha).sqrt();
            let (theta_t, q_t, disc, reg) = linear_eval(alpha_mid)?;
            path.push(PathPoint {
                alpha: alpha_mid,
                discrepancy: disc,
                regularizer: reg,
                objective: disc + alpha_mid * reg,
                iterations: 1,
                jacobians: 0,
                active_set_switches: 0,
                kind: PathKind::Bisection,
            });
            if disc <= target {
                alpha = alpha_mid;
                best = Some((alpha_mid, theta_t, q_t));
                flags.bisection_accepted = true;
            }
        }
    } else {
        flags.discrepancy_not_reached = true;
    }
    path.sort_by(|a, b| {
        b.alpha
            .partial_cmp(&a.alpha)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (alpha_lin, theta_lin, q_lin) = best.expect("at least one linear level was evaluated");
    debug_assert!((alpha_lin - alpha).abs() <= 1e-12 * alpha);

    let rows_a = engine.unsort_rows(&engine.data.rows);
    let prices_a = engine.unsort(&engine.data.prices);
    let vega_a = engine.unsort(&engine.data.vegas);
    let active = engine.unsort(&engine.data.active);
    let (rows_e, prices_e, vega_e) = if mode.is_american() {
        let mut eng_e = Engine::new(quotes, capture, surface, Mode::European, carry_free, None)?;
        eng_e.run(&theta0, &q0)?;
        jacobians += 1;
        (
            eng_e.unsort_rows(&eng_e.data.rows),
            eng_e.unsort(&eng_e.data.prices),
            eng_e.unsort(&eng_e.data.vegas),
        )
    } else {
        (rows_a.clone(), prices_a.clone(), vega_a.clone())
    };
    Ok(Linearized {
        theta0,
        q0,
        theta_lin,
        q_lin,
        alpha_lin,
        path,
        rows_a,
        rows_e,
        vega_a,
        vega_e,
        prices_a,
        prices_e,
        active,
        n_cols: ctx.w,
        n_theta: ctx.m,
        jacobians,
        discrepancy_target: target,
        flags,
        wall_ms: start.elapsed().as_secs_f64() * 1e3,
    })
}

// ── Evaluation helpers ───────────────────────────────────────────────────

/// Model prices of every quote under `vol` on the capture's meshes
/// (`mode` American or European), in parallel with thread-local
/// workspaces; the field is sampled once per expiry.
pub fn model_prices(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    vol: &dyn VolField,
    mode: Mode,
) -> Result<Vec<f64>, RustyQLibError> {
    for q in quotes {
        q.validate(capture)?;
    }
    let fields: Vec<NodeField> = capture
        .meshes
        .par_iter()
        .map(|mesh| mesh.node_field(vol))
        .collect();
    quotes
        .par_iter()
        .map_init(Workspace::new, |ws, q| {
            let e = q.expiry_index;
            let mesh = &capture.meshes[e];
            price_only(
                mesh,
                &q.spec(mesh.t_expiry),
                &fields[e],
                &capture.markets[e],
                mode,
                ws,
            )
        })
        .collect()
}

/// American prices of every quote under `vol` (penalty `rho`).
pub fn american_prices(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    vol: &dyn VolField,
    rho: f64,
) -> Result<Vec<f64>, RustyQLibError> {
    model_prices(quotes, capture, vol, Mode::American { rho })
}

/// European prices of every quote under `vol` by the backward European
/// solver on each quote's own mesh ([`european_prices_backward`] per
/// expiry): THE source of `sigma^E` at the quotes.
pub fn european_prices(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    vol: &dyn VolField,
) -> Result<Vec<f64>, RustyQLibError> {
    for q in quotes {
        q.validate(capture)?;
    }
    let mut out = vec![f64::NAN; quotes.len()];
    for e in 0..capture.n_expiries() {
        let mesh = &capture.meshes[e];
        let idx: Vec<usize> = (0..quotes.len())
            .filter(|&i| quotes[i].expiry_index == e)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let specs: Vec<QuoteSpec> = idx.iter().map(|&i| quotes[i].spec(mesh.t_expiry)).collect();
        let prices = european_prices_backward(mesh, &specs, vol, &capture.markets[e])?;
        for (&i, p) in idx.iter().zip(prices) {
            out[i] = p;
        }
    }
    Ok(out)
}

/// `sigma^E` of every quote from its European price: Black-76 on the
/// expiry's model forward (identical to a Black-Scholes inversion at the
/// effective flat rates), inverted on the out-of-the-money right.
pub fn european_implied_vols(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    prices: &[f64],
) -> Vec<Result<f64, IvError>> {
    quotes
        .iter()
        .zip(prices)
        .map(|(q, &p)| {
            let (_, _, forward, df) = capture.effective_rates(q.expiry_index);
            let t = capture.expiries[q.expiry_index];
            otm_implied_vol(forward, q.strike, t, p / df, q.right)
        })
        .collect()
}

/// Scaled residuals `quote.residual(price)` of every quote.
pub fn scaled_residuals(quotes: &[Quote], prices: &[f64]) -> Vec<f64> {
    quotes
        .iter()
        .zip(prices)
        .map(|(q, &p)| q.residual(p))
        .collect()
}

/// The identification (M0) quotes: every quote with a finite `sigma_a`
/// becomes a two-sided EUROPEAN quote at `P_hat^E = BS^E(sigma_a)` with
/// the expiry's effective flat rates, scale `s_i nu^E/nu^A = s_i /
/// vega_ratio_i` when `propagated_weights` (the most favourable treatment
/// of the industry method) or the plain `s_i` otherwise; quotes without a
/// `sigma_a` (at intrinsic, non-finite) keep their American mid and are
/// marked `held_out`. Held-out flags of the input are preserved.
pub fn identification_quotes(
    quotes: &[Quote],
    capture: &CaptureMeshes,
    sigma_a: &[f64],
    vega_ratio: &[f64],
    propagated_weights: bool,
) -> Vec<Quote> {
    quotes
        .iter()
        .enumerate()
        .map(|(i, q)| {
            let sa = sigma_a.get(i).copied().unwrap_or(f64::NAN);
            if !(sa.is_finite() && sa > 0.0) {
                return q.clone().held_out();
            }
            let (r_eff, q_eff, _, _) = capture.effective_rates(q.expiry_index);
            let t = capture.expiries[q.expiry_index];
            let p_hat = bs_price(capture.s0, q.strike, r_eff, q_eff, sa, t, q.right);
            let ratio = vega_ratio.get(i).copied().unwrap_or(f64::NAN);
            let s = if propagated_weights && ratio.is_finite() && ratio > 0.0 {
                q.half_spread / ratio
            } else {
                q.half_spread
            };
            let mut out = Quote::from_mid(q.strike, q.t, q.expiry_index, q.right, p_hat, s);
            out.held_out = q.held_out;
            out
        })
        .collect()
}

#[cfg(test)]
#[path = "calibration_tests.rs"]
mod calibration_tests;
