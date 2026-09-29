//! The dense European surface: Dupire's forward equation, Black-76
//! implied-volatility inversion with effective flat rates, and
//! static-arbitrage scans.
//!
//! The backward European solver (`solver.rs`) is THE source of `sigma^E`
//! at the listed quotes, because it shares the mesh of `sigma^A` and the
//! discretization error cancels in `E = sigma^A - sigma^E`. Everything
//! here serves the *display* surface and the arbitrage tables: one forward
//! march gives the call price at every strike and every maturity at once.
//!
//! ```text
//! Forward equation in k = ln K (Dupire 1994), between ex-dividend dates:
//!   d/dT C = 1/2 Sigma(K,T)^2 (C_kk - C_k) - (r - q) C_k - q C,   C(k, 0) = (S0 - e^k)^+
//! Cash dividend delta at t_j (S -> S - delta):
//!   C(K, t_j^+) = C(K + delta, t_j^-)          (affine strike shift, interpolated in k)
//! Edge rows (linear in K, C_KK = 0  <=>  C_kk = C_k; Windcliff-Forsyth-Vetzal 2004):
//!   (L C)_0 = -(r - q)(C_1 - C_0)/dk - q C_0,   (L C)_n = -(r - q)(C_n - C_{n-1})/dk - q C_n
//! ```
//!
//! Discretization conventions (stated because the same choices are made
//! by the backward solver and must not drift apart):
//!
//! - `k` is a uniform grid; `Sigma` is read at `vol(k_j, t_{n+1/2})`, the
//!   node's log-strike and the mid-time of the step, and ONE operator per
//!   step serves both the explicit and the implicit half.
//! - Rates and carry enter as the step averages of piecewise-constant
//!   term structures, so the discrete discount factor product equals the
//!   exact `exp(-int r)` at every maturity node.
//! - Time grid: geometric from [`DupireConfig::first_step`] with at least
//!   [`DupireConfig::min_steps_first`] steps before the first anchor
//!   (maturity or ex-date), uniform at most [`DupireConfig::max_dt`]
//!   afterwards; every anchor is a node. Crank-Nicolson after
//!   [`DupireConfig::rannacher_steps`] fully implicit steps at `t = 0` and
//!   after every dividend jump.
//! - Interior rows use central differences whenever the off-diagonals
//!   stay nonnegative (`|b| dk / 2 <= Sigma^2 / 2`), first-order upwinding
//!   for the drift otherwise (counted in [`DenseSurface::upwinded_rows`]).
//! - The initial layer is the exact cell average of the payoff over
//!   `[k - dk/2, k + dk/2]`; centre the grid on `ln S0` with
//!   [`log_strike_grid`] so the kink sits on a node.
//! - A maturity that coincides with an ex-date reads the PRE-jump layer
//!   (the option settles on the cum-dividend price); the forward used for
//!   its implied vols excludes that dividend consistently.
//!
//! Implied vols are Black-76 on the model forward, `price / df` against
//! `F(T)`, which is identical to a Black-Scholes inversion with the
//! effective flat rates `r_eff = -ln D(T)/T`, `q_eff = r_eff - ln(F/S0)/T`
//! ([`effective_flat_rates`]); the inversion is always done on the
//! out-of-the-money right via parity, where the price-to-vol map is best
//! conditioned.

// Stencil loops index three neighbours per row; iterator chains would
// hide the recurrence, so the range-loop lint is silenced for this file.
#![allow(clippy::needless_range_loop)]

use crate::core::errors::{Result, RustyQLibError};
use crate::core::trade::PutOrCall;
use crate::equity::blackscholes::implied_vol_from_price;

use super::vol_field::{FlatVol, VolField};

/// The crate inverter's volatility floor: a vol at or below it means the
/// price sat within round-off of the discounted intrinsic.
const IV_FLOOR: f64 = 1e-4;

/// Two time nodes closer than this are the same node.
const NODE_TOL: f64 = 1e-12;

/// `true` unless `w[1] > w[0]` holds as an ordered comparison (so a NaN
/// counts as "not increasing", which every validation here wants).
#[inline]
fn not_increasing(w: &[f64]) -> bool {
    w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater)
}

/// Nodes whose out-of-the-money undiscounted price is below this fraction
/// of `s0` are skipped by the calendar check of [`arbitrage_scan`]: the
/// vol there is dominated by round-off of a price that is numerically
/// zero. Recorded in [`ArbReport::n_skipped`].
pub const CALENDAR_PRICE_FLOOR: f64 = 1e-6;

// ── Market description ───────────────────────────────────────────────────

/// A piecewise-constant instantaneous rate: `values[0]` on
/// `[0, breaks[0])`, `values[i]` on `[breaks[i-1], breaks[i])`, the last
/// value beyond the last break (and before `t = 0`).
///
/// This is exactly what a log-linear discount-factor curve implies, so a
/// bootstrapped `YieldCurve` maps onto it losslessly (continuous forward
/// rates between pillars).
#[derive(Debug, Clone, PartialEq)]
pub struct TermStructure {
    breaks: Vec<f64>,
    values: Vec<f64>,
}

impl TermStructure {
    /// A constant rate.
    pub fn flat(value: f64) -> Self {
        TermStructure {
            breaks: Vec::new(),
            values: vec![value],
        }
    }

    /// Piecewise-constant with `values.len() == breaks.len() + 1`;
    /// breaks must be finite, positive and strictly increasing.
    pub fn piecewise(breaks: Vec<f64>, values: Vec<f64>) -> Result<Self> {
        if values.len() != breaks.len() + 1 {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                format!(
                    "expected {} values for {} breaks, got {}",
                    breaks.len() + 1,
                    breaks.len(),
                    values.len()
                ),
            ));
        }
        if breaks.iter().any(|b| !(b.is_finite() && *b > 0.0))
            || breaks.windows(2).any(not_increasing)
        {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                "breaks must be finite, positive and strictly increasing",
            ));
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                "values must be finite",
            ));
        }
        Ok(TermStructure { breaks, values })
    }

    /// The piecewise-constant forward rates implied by discount factors
    /// `dfs[i]` at `times[i]` (positive, strictly increasing; `df(0) = 1`
    /// implied), extrapolated flat beyond the last pillar. This is the
    /// lossless image of a log-linear discount curve.
    pub fn from_discount_factors(times: &[f64], dfs: &[f64]) -> Result<Self> {
        if times.is_empty() || times.len() != dfs.len() {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                "times and discount factors must be nonempty and of equal length",
            ));
        }
        if dfs.iter().any(|d| !(d.is_finite() && *d > 0.0)) {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                "discount factors must be positive and finite",
            ));
        }
        if times.iter().any(|t| !(t.is_finite() && *t > 0.0))
            || times.windows(2).any(not_increasing)
        {
            return Err(RustyQLibError::invalid_input(
                "term_structure",
                "times must be finite, positive and strictly increasing",
            ));
        }
        let mut values = Vec::with_capacity(times.len());
        let (mut t_prev, mut df_prev) = (0.0, 1.0);
        for (&t, &d) in times.iter().zip(dfs) {
            values.push(-(d / df_prev).ln() / (t - t_prev));
            t_prev = t;
            df_prev = d;
        }
        let breaks = times[..times.len() - 1].to_vec();
        TermStructure::piecewise(breaks, values)
    }

    /// The breakpoints.
    pub fn breaks(&self) -> &[f64] {
        &self.breaks
    }

    /// The values, one more than the breakpoints.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The value in force at `t` (right-continuous).
    pub fn at(&self, t: f64) -> f64 {
        let i = self.breaks.partition_point(|&b| b <= t);
        self.values[i]
    }

    /// `int_{t1}^{t2} value(s) ds`, exact; antisymmetric when `t2 < t1`.
    pub fn integral(&self, t1: f64, t2: f64) -> f64 {
        if t2 < t1 {
            return -self.integral(t2, t1);
        }
        let mut total = 0.0;
        let mut lo = t1;
        for (i, &b) in self.breaks.iter().enumerate() {
            if b <= lo {
                continue;
            }
            if b >= t2 {
                break;
            }
            total += self.values[i] * (b - lo);
            lo = b;
        }
        total + self.at(lo) * (t2 - lo)
    }

    /// The average value over `[t1, t2]` (the value at `t1` when the
    /// interval is empty).
    pub fn average(&self, t1: f64, t2: f64) -> f64 {
        if t2 > t1 {
            self.integral(t1, t2) / (t2 - t1)
        } else {
            self.at(t1)
        }
    }
}

/// The deterministic part of the model: spot, rate and carry term
/// structures, and cash dividends `(t_j, delta_j)` (sorted by time).
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardMarket {
    /// Spot at the valuation date.
    pub s0: f64,
    /// Continuous risk-free rate `r(t)`.
    pub rate: TermStructure,
    /// Continuous carry `q(t)` (dividend yield plus borrow).
    pub carry: TermStructure,
    /// Cash dividends `(ex-time in years, amount)`, sorted by time.
    pub dividends: Vec<(f64, f64)>,
}

impl ForwardMarket {
    /// Validate and sort the dividends (times positive and finite, amounts
    /// nonnegative and finite).
    pub fn new(
        s0: f64,
        rate: TermStructure,
        carry: TermStructure,
        mut dividends: Vec<(f64, f64)>,
    ) -> Result<Self> {
        if !(s0.is_finite() && s0 > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "s0",
                format!("spot must be positive and finite, got {s0}"),
            ));
        }
        for &(t, d) in &dividends {
            if !(t.is_finite() && t > 0.0 && d.is_finite() && d >= 0.0) {
                return Err(RustyQLibError::invalid_input(
                    "dividends",
                    format!("ex-time must be positive and amount nonnegative, got ({t}, {d})"),
                ));
            }
        }
        dividends.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Ok(ForwardMarket {
            s0,
            rate,
            carry,
            dividends,
        })
    }

    /// Flat rate and carry, no cash dividends.
    pub fn flat(s0: f64, r: f64, q: f64) -> Self {
        ForwardMarket {
            s0,
            rate: TermStructure::flat(r),
            carry: TermStructure::flat(q),
            dividends: Vec::new(),
        }
    }

    /// Discount factor `exp(-int_0^t r)`.
    pub fn df(&self, t: f64) -> f64 {
        (-self.rate.integral(0.0, t)).exp()
    }

    /// Growth factor `exp(int_{t1}^{t2} (r - q))`.
    pub fn growth(&self, t1: f64, t2: f64) -> f64 {
        (self.rate.integral(t1, t2) - self.carry.integral(t1, t2)).exp()
    }

    /// The model forward including cash dividends strictly before `t`:
    /// `F(t) = S0 e^{int (r - q)} - sum_{t_j < t} delta_j e^{int_{t_j}^t (r - q)}`.
    pub fn forward(&self, t: f64) -> f64 {
        let mut f = self.s0 * self.growth(0.0, t);
        for &(tj, dj) in &self.dividends {
            if tj < t - NODE_TOL {
                f -= dj * self.growth(tj, t);
            }
        }
        f
    }

    /// `(r_eff, q_eff)` at `t`: see [`effective_flat_rates`].
    pub fn effective_rates(&self, t: f64) -> (f64, f64) {
        effective_flat_rates(self.s0, t, self.df(t), self.forward(t))
    }
}

/// Effective flat rates for a Black-Scholes inversion at maturity `t`:
/// `r_eff = -ln(df)/t`, `q_eff = r_eff - ln(forward / s0)/t`, so that
/// `exp(-r_eff t) = df` and `s0 exp((r_eff - q_eff) t) = forward`.
pub fn effective_flat_rates(s0: f64, t: f64, df: f64, forward: f64) -> (f64, f64) {
    let r = -df.ln() / t;
    let q = r - (forward / s0).ln() / t;
    (r, q)
}

// ── Time grid and configuration ──────────────────────────────────────────

/// Knobs of the forward march.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DupireConfig {
    /// First time step in years (default `1e-4`).
    pub first_step: f64,
    /// Minimum number of steps before the first anchor (default 40).
    pub min_steps_first: usize,
    /// Geometric growth of consecutive steps until `max_dt` (default 1.1).
    pub growth: f64,
    /// Largest step in years (default `1/365`).
    pub max_dt: f64,
    /// Fully implicit steps at `t = 0` and after each dividend jump
    /// (default 4).
    pub rannacher_steps: usize,
}

impl Default for DupireConfig {
    fn default() -> Self {
        DupireConfig {
            first_step: 1e-4,
            min_steps_first: 40,
            growth: 1.1,
            max_dt: 1.0 / 365.0,
            rannacher_steps: 4,
        }
    }
}

impl DupireConfig {
    fn validate(&self) -> Result<()> {
        let ok = self.first_step.is_finite()
            && self.first_step > 0.0
            && self.max_dt.is_finite()
            && self.max_dt > 0.0
            && self.growth.is_finite()
            && self.growth >= 1.0
            && self.min_steps_first >= 1;
        if ok {
            Ok(())
        } else {
            Err(RustyQLibError::invalid_input(
                "dupire_config",
                format!("{self:?} is not a valid configuration"),
            ))
        }
    }
}

/// Geometric steps `first * rho^j`, `j = 0..n`, summing to `total`
/// (uniform when `first * n >= total`).
fn geometric_steps(first: f64, total: f64, n: usize) -> Vec<f64> {
    let nf = n as f64;
    if first * nf >= total {
        return vec![total / nf; n];
    }
    let sum = |rho: f64| first * (rho.powi(n as i32) - 1.0) / (rho - 1.0);
    let (mut lo, mut hi) = (1.0 + 1e-12, 2.0);
    while sum(hi) < total {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if sum(mid) < total {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-15 * hi {
            break;
        }
    }
    let rho = 0.5 * (lo + hi);
    let mut steps: Vec<f64> = (0..n).map(|j| first * rho.powi(j as i32)).collect();
    let s: f64 = steps.iter().sum();
    for d in &mut steps {
        *d *= total / s;
    }
    steps
}

/// The graded time grid through the sorted, distinct, positive `anchors`
/// (maturities and ex-dates): `0 = t_0 < ... `, geometric from
/// `cfg.first_step` with at least `cfg.min_steps_first` steps up to the
/// first anchor, then uniform steps of at most `cfg.max_dt` between
/// consecutive anchors. Every anchor is a node (bit-exact).
pub fn graded_time_grid(anchors: &[f64], cfg: &DupireConfig) -> Result<Vec<f64>> {
    cfg.validate()?;
    if anchors.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "anchors",
            "at least one maturity is required",
        ));
    }
    if anchors.iter().any(|a| !(a.is_finite() && *a > 0.0))
        || anchors.windows(2).any(not_increasing)
    {
        return Err(RustyQLibError::invalid_input(
            "anchors",
            "anchors must be finite, positive and strictly increasing",
        ));
    }
    let a1 = anchors[0];
    let mut steps: Vec<f64> = Vec::new();
    let mut dt = cfg.first_step;
    let mut cum = 0.0;
    while cum < a1 {
        let d = dt.min(cfg.max_dt);
        steps.push(d);
        cum += d;
        dt *= cfg.growth;
    }
    if steps.len() < cfg.min_steps_first {
        steps = geometric_steps(cfg.first_step, a1, cfg.min_steps_first);
    } else {
        let scale = a1 / cum;
        for d in &mut steps {
            *d *= scale;
        }
    }
    let mut nodes = Vec::with_capacity(steps.len() + 1);
    nodes.push(0.0);
    let mut t = 0.0;
    for d in &steps {
        t += d;
        nodes.push(t);
    }
    *nodes.last_mut().unwrap() = a1;
    for w in anchors.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let len = hi - lo;
        let n = ((len / cfg.max_dt) - 1e-9).ceil().max(1.0) as usize;
        for i in 1..=n {
            nodes.push(lo + len * i as f64 / n as f64);
        }
        *nodes.last_mut().unwrap() = hi;
    }
    Ok(nodes)
}

/// Uniform log-strike nodes `ln s0 + i * (half_width / n_half)` for
/// `i = -n_half..=n_half` (so `ln s0` is a node and the payoff kink sits
/// on it).
pub fn log_strike_grid(s0: f64, half_width: f64, n_half: usize) -> Vec<f64> {
    let dk = half_width / n_half as f64;
    let k0 = s0.ln();
    (0..=2 * n_half)
        .map(|i| k0 + (i as f64 - n_half as f64) * dk)
        .collect()
}

/// Uniform spacing of `k` (error unless `k.len() >= 5`, increasing and
/// uniform to `1e-9`).
fn uniform_spacing(k: &[f64]) -> Result<f64> {
    if k.len() < 5 {
        return Err(RustyQLibError::invalid_input(
            "k_grid",
            "at least five log-strike nodes are required",
        ));
    }
    let n = k.len();
    let dk = (k[n - 1] - k[0]) / (n - 1) as f64;
    if !(dk.is_finite() && dk > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "k_grid",
            "log-strike nodes must be finite and increasing",
        ));
    }
    for (j, &kj) in k.iter().enumerate() {
        if (kj - (k[0] + j as f64 * dk)).abs() > 1e-9 {
            return Err(RustyQLibError::invalid_input(
                "k_grid",
                format!("log-strike nodes must be uniform (node {j} is off by more than 1e-9)"),
            ));
        }
    }
    Ok(dk)
}

// ── Numerical kernels ────────────────────────────────────────────────────

/// Allocation-free Thomas solve of the tridiagonal system with
/// sub-diagonal `a` (`n-1`), diagonal `b` (`n`), super-diagonal `c`
/// (`n-1`) and right-hand side `d`; `cw`, `dw` are scratch of length `n`,
/// `x` receives the solution. Returns `false` on a near-zero pivot
/// instead of panicking.
fn thomas_inplace(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    cw: &mut [f64],
    dw: &mut [f64],
    x: &mut [f64],
) -> bool {
    let n = d.len();
    debug_assert!(b.len() == n && a.len() + 1 == n && c.len() + 1 == n);
    debug_assert!(cw.len() == n && dw.len() == n && x.len() == n);
    if n == 0 {
        return true;
    }
    let piv_ok = |p: f64, i: usize| p.abs() > 1e-14 * (1.0 + b[i].abs());
    if !piv_ok(b[0], 0) {
        return false;
    }
    cw[0] = if n > 1 { c[0] / b[0] } else { 0.0 };
    dw[0] = d[0] / b[0];
    for i in 1..n {
        let piv = b[i] - a[i - 1] * cw[i - 1];
        if !piv_ok(piv, i) {
            return false;
        }
        let id = 1.0 / piv;
        if i + 1 < n {
            cw[i] = c[i] * id;
        }
        dw[i] = (d[i] - a[i - 1] * dw[i - 1]) * id;
    }
    x[n - 1] = dw[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dw[i] - cw[i] * x[i + 1];
    }
    true
}

/// Four-point Lagrange (cubic) interpolation of `f` sampled on the uniform
/// grid `k0 + j dk` at `x`; the stencil is clamped to the grid, so values
/// outside it are cubic extrapolations of the nearest four nodes.
fn lagrange4(k0: f64, dk: f64, f: &[f64], x: f64) -> f64 {
    let n = f.len();
    debug_assert!(n >= 4);
    let pos = (x - k0) / dk;
    let j = (pos.floor() as isize - 1).clamp(0, n as isize - 4) as usize;
    let s = pos - j as f64;
    let w0 = -(s - 1.0) * (s - 2.0) * (s - 3.0) / 6.0;
    let w1 = s * (s - 2.0) * (s - 3.0) / 2.0;
    let w2 = -s * (s - 1.0) * (s - 3.0) / 2.0;
    let w3 = s * (s - 1.0) * (s - 2.0) / 6.0;
    w0 * f[j] + w1 * f[j + 1] + w2 * f[j + 2] + w3 * f[j + 3]
}

/// Exact average of the payoff over the cell `[k - dk/2, k + dk/2]`.
fn cell_average_payoff(k: f64, dk: f64, s0: f64, right: PutOrCall) -> f64 {
    let a = k - 0.5 * dk;
    let b = k + 0.5 * dk;
    let c = s0.ln().clamp(a, b);
    match right {
        PutOrCall::Call => (s0 * (c - a) - (c.exp() - a.exp())) / dk,
        PutOrCall::Put => ((b.exp() - c.exp()) - s0 * (b - c)) / dk,
    }
}

/// Scratch storage of the forward march, reusable across calls with the
/// same (or a smaller) number of log-strike nodes.
#[derive(Debug, Clone, Default)]
pub struct DupireWorkspace {
    lower: Vec<f64>,
    diag: Vec<f64>,
    upper: Vec<f64>,
    sub: Vec<f64>,
    dia: Vec<f64>,
    sup: Vec<f64>,
    rhs: Vec<f64>,
    cw: Vec<f64>,
    dw: Vec<f64>,
    cur: Vec<f64>,
    next: Vec<f64>,
}

impl DupireWorkspace {
    /// Allocate for `n_k` log-strike nodes.
    pub fn new(n_k: usize) -> Self {
        let mut ws = DupireWorkspace::default();
        ws.resize(n_k);
        ws
    }

    fn resize(&mut self, n_k: usize) {
        let m = n_k.saturating_sub(1);
        for v in [
            &mut self.lower,
            &mut self.diag,
            &mut self.upper,
            &mut self.dia,
            &mut self.rhs,
            &mut self.cw,
            &mut self.dw,
            &mut self.cur,
            &mut self.next,
        ] {
            v.clear();
            v.resize(n_k, 0.0);
        }
        for v in [&mut self.sub, &mut self.sup] {
            v.clear();
            v.resize(m, 0.0);
        }
    }
}

/// Assemble the operator bands `(lower, diag, upper)` of `L` at the step
/// mid-time; returns the number of upwinded interior rows.
///
/// The three band slices are passed separately (rather than the whole
/// workspace) so the caller keeps disjoint borrows of its other buffers.
#[allow(clippy::too_many_arguments)]
fn assemble_operator(
    vol: &dyn VolField,
    k: &[f64],
    dk: f64,
    t_mid: f64,
    r: f64,
    q: f64,
    lower: &mut [f64],
    diag: &mut [f64],
    upper: &mut [f64],
) -> Result<usize> {
    let n = k.len();
    let mut upwinded = 0;
    let inv_dk2 = 1.0 / (dk * dk);
    for j in 1..n - 1 {
        let sigma = vol.vol(k[j], t_mid);
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(RustyQLibError::NumericalError(format!(
                "local volatility {sigma} at k = {} (K = {}), t = {t_mid}",
                k[j],
                k[j].exp()
            )));
        }
        let s2 = 0.5 * sigma * sigma;
        let b = -(s2 + r - q);
        let diff = s2 * inv_dk2;
        if b.abs() * dk * 0.5 <= s2 {
            lower[j] = diff - b / (2.0 * dk);
            diag[j] = -2.0 * diff - q;
            upper[j] = diff + b / (2.0 * dk);
        } else {
            upwinded += 1;
            if b > 0.0 {
                lower[j] = diff;
                diag[j] = -2.0 * diff - b / dk - q;
                upper[j] = diff + b / dk;
            } else {
                lower[j] = diff - b / dk;
                diag[j] = -2.0 * diff + b / dk - q;
                upper[j] = diff;
            }
        }
    }
    let m = r - q;
    lower[0] = 0.0;
    diag[0] = m / dk - q;
    upper[0] = -m / dk;
    lower[n - 1] = m / dk;
    diag[n - 1] = -m / dk - q;
    upper[n - 1] = 0.0;
    Ok(upwinded)
}

// ── The dense surface ────────────────────────────────────────────────────

/// Prices of one right on a dense `(k, T)` grid, row-major
/// `prices[i * n_k + j]` for maturity `i` and log-strike node `j`,
/// discounted to the valuation date, with the discount factor and the
/// (dividend-adjusted) forward of every maturity.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseSurface {
    /// The right the prices refer to.
    pub right: PutOrCall,
    /// Spot at the valuation date.
    pub s0: f64,
    /// Uniform log-strike nodes.
    pub k: Vec<f64>,
    /// Node spacing.
    pub dk: f64,
    /// Maturities in years, increasing.
    pub maturities: Vec<f64>,
    /// Discount factor per maturity.
    pub df: Vec<f64>,
    /// Model forward per maturity.
    pub forward: Vec<f64>,
    /// Discounted prices, `maturities.len() * k.len()` entries.
    pub prices: Vec<f64>,
    /// The time nodes the march used (empty for a hand-built surface).
    pub t_nodes: Vec<f64>,
    /// Interior rows that fell back to upwinding, summed over steps.
    pub upwinded_rows: usize,
}

impl DenseSurface {
    /// A surface from given prices (for hand-built or externally computed
    /// surfaces); validates the grid and the lengths.
    pub fn new(
        right: PutOrCall,
        s0: f64,
        k: Vec<f64>,
        maturities: Vec<f64>,
        df: Vec<f64>,
        forward: Vec<f64>,
        prices: Vec<f64>,
    ) -> Result<Self> {
        let dk = uniform_spacing(&k)?;
        let n_t = maturities.len();
        if n_t == 0 || maturities.windows(2).any(not_increasing) {
            return Err(RustyQLibError::invalid_input(
                "maturities",
                "must be nonempty and strictly increasing",
            ));
        }
        if df.len() != n_t || forward.len() != n_t || prices.len() != n_t * k.len() {
            return Err(RustyQLibError::invalid_input(
                "dense_surface",
                "df, forward and prices must match the maturity and strike counts",
            ));
        }
        Ok(DenseSurface {
            right,
            s0,
            k,
            dk,
            maturities,
            df,
            forward,
            prices,
            t_nodes: Vec::new(),
            upwinded_rows: 0,
        })
    }

    /// Number of log-strike nodes.
    #[inline]
    pub fn n_k(&self) -> usize {
        self.k.len()
    }

    /// Number of maturities.
    #[inline]
    pub fn n_t(&self) -> usize {
        self.maturities.len()
    }

    /// Discounted price at maturity `i`, node `j`.
    #[inline]
    pub fn price(&self, i: usize, j: usize) -> f64 {
        self.prices[i * self.k.len() + j]
    }

    /// One maturity's row of discounted prices.
    #[inline]
    pub fn row(&self, i: usize) -> &[f64] {
        let n = self.k.len();
        &self.prices[i * n..(i + 1) * n]
    }

    /// Undiscounted CALL price at maturity `i`, node `j` (puts converted by
    /// parity `c = p + F - K`).
    pub fn undiscounted_call(&self, i: usize, j: usize) -> f64 {
        let u = self.price(i, j) / self.df[i];
        match self.right {
            PutOrCall::Call => u,
            PutOrCall::Put => u + self.forward[i] - self.k[j].exp(),
        }
    }

    /// Discounted price at maturity `i` and an arbitrary `strike`, by
    /// four-point Lagrange interpolation in log-strike (cubic
    /// extrapolation outside the grid — keep listed strikes inside).
    pub fn price_at(&self, i: usize, strike: f64) -> f64 {
        lagrange4(self.k[0], self.dk, self.row(i), strike.ln())
    }

    /// Black-76 implied vol at maturity `i`, node `j` (out-of-the-money
    /// right via parity).
    pub fn implied_vol(&self, i: usize, j: usize) -> std::result::Result<f64, IvError> {
        otm_implied_vol(
            self.forward[i],
            self.k[j].exp(),
            self.maturities[i],
            self.price(i, j) / self.df[i],
            self.right,
        )
    }

    /// Black-76 implied vol at maturity `i` and a listed `strike`
    /// (interpolated price, out-of-the-money right via parity).
    pub fn implied_vol_at(&self, i: usize, strike: f64) -> std::result::Result<f64, IvError> {
        otm_implied_vol(
            self.forward[i],
            strike,
            self.maturities[i],
            self.price_at(i, strike) / self.df[i],
            self.right,
        )
    }

    /// Implied vol at every node, row-major like [`DenseSurface::prices`].
    pub fn implied_vol_grid(&self) -> Vec<std::result::Result<f64, IvError>> {
        (0..self.n_t())
            .flat_map(|i| (0..self.n_k()).map(move |j| (i, j)))
            .map(|(i, j)| self.implied_vol(i, j))
            .collect()
    }

    /// Total implied variance `sigma^2 T` at every node (`None` where the
    /// inversion failed), row-major.
    pub fn total_variance_grid(&self) -> Vec<Option<f64>> {
        self.implied_vol_grid()
            .into_iter()
            .enumerate()
            .map(|(idx, v)| {
                let t = self.maturities[idx / self.n_k()];
                v.ok().map(|s| s * s * t)
            })
            .collect()
    }
}

/// Dupire forward march for CALLS: the discounted call price on
/// `k_grid x maturities` under the local volatility `vol`.
pub fn dupire_forward(
    vol: &dyn VolField,
    market: &ForwardMarket,
    k_grid: &[f64],
    maturities: &[f64],
    cfg: &DupireConfig,
) -> Result<DenseSurface> {
    let mut ws = DupireWorkspace::new(k_grid.len());
    dupire_forward_with(
        vol,
        market,
        k_grid,
        maturities,
        cfg,
        PutOrCall::Call,
        &mut ws,
    )
}

/// Dupire forward march for either right with a caller-owned workspace.
/// Puts solve the same equation from `(e^k - S0)^+` with the same
/// (linear-in-K) edge rows, so `C - P = D(T)(F(T) - K)` holds up to the
/// scheme's `O(dk^2)` non-preservation of exponentials.
#[allow(clippy::too_many_arguments)]
pub fn dupire_forward_with(
    vol: &dyn VolField,
    market: &ForwardMarket,
    k_grid: &[f64],
    maturities: &[f64],
    cfg: &DupireConfig,
    right: PutOrCall,
    ws: &mut DupireWorkspace,
) -> Result<DenseSurface> {
    cfg.validate()?;
    let dk = uniform_spacing(k_grid)?;
    if maturities.is_empty()
        || maturities.iter().any(|t| !(t.is_finite() && *t > 0.0))
        || maturities.windows(2).any(not_increasing)
    {
        return Err(RustyQLibError::invalid_input(
            "maturities",
            "must be nonempty, finite, positive and strictly increasing",
        ));
    }
    if !(market.s0.is_finite() && market.s0 > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "s0",
            format!("spot must be positive and finite, got {}", market.s0),
        ));
    }
    let n_k = k_grid.len();
    let n_mat = maturities.len();
    let s0 = market.s0;
    let t_max = maturities[n_mat - 1];

    // anchors: maturities and the ex-dates before the last maturity
    let mut anchors: Vec<f64> = maturities.to_vec();
    for &(td, _) in &market.dividends {
        if td < t_max - NODE_TOL {
            anchors.push(td);
        }
    }
    anchors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    anchors.dedup_by(|a, b| (*a - *b).abs() <= NODE_TOL);
    let t = graded_time_grid(&anchors, cfg)?;
    let locate = |x: f64| -> Result<usize> {
        let idx = t.partition_point(|&v| v < x - NODE_TOL);
        if idx < t.len() && (t[idx] - x).abs() <= 1e-9 {
            Ok(idx)
        } else {
            Err(RustyQLibError::NumericalError(format!(
                "time {x} is not a node of the graded grid"
            )))
        }
    };
    let mat_node: Vec<usize> = maturities
        .iter()
        .map(|&m| locate(m))
        .collect::<Result<Vec<_>>>()?;
    let mut div_events: Vec<(usize, f64)> = Vec::new();
    for &(td, amount) in &market.dividends {
        if td < t_max - NODE_TOL {
            div_events.push((locate(td)?, amount));
        }
    }

    ws.resize(n_k);
    for j in 0..n_k {
        ws.cur[j] = cell_average_payoff(k_grid[j], dk, s0, right);
    }

    let mut prices = vec![0.0; n_mat * n_k];
    let mut df = vec![0.0; n_mat];
    let mut forward = vec![0.0; n_mat];
    let mut implicit_left = cfg.rannacher_steps;
    let mut upwinded = 0usize;
    let mut next_mat = 0usize;
    let mut next_div = 0usize;

    for n in 0..t.len() - 1 {
        let (t1, t2) = (t[n], t[n + 1]);
        let dt = t2 - t1;
        let t_mid = 0.5 * (t1 + t2);
        let r = market.rate.average(t1, t2);
        let q = market.carry.average(t1, t2);
        let theta = if implicit_left > 0 {
            implicit_left -= 1;
            1.0
        } else {
            0.5
        };
        upwinded += assemble_operator(
            vol,
            k_grid,
            dk,
            t_mid,
            r,
            q,
            &mut ws.lower,
            &mut ws.diag,
            &mut ws.upper,
        )?;

        // explicit half: rhs = u + (1 - theta) dt L u
        let e = (1.0 - theta) * dt;
        let cur = &ws.cur;
        ws.rhs[0] = cur[0] + e * (ws.diag[0] * cur[0] + ws.upper[0] * cur[1]);
        for j in 1..n_k - 1 {
            ws.rhs[j] = cur[j]
                + e * (ws.lower[j] * cur[j - 1] + ws.diag[j] * cur[j] + ws.upper[j] * cur[j + 1]);
        }
        ws.rhs[n_k - 1] =
            cur[n_k - 1] + e * (ws.lower[n_k - 1] * cur[n_k - 2] + ws.diag[n_k - 1] * cur[n_k - 1]);

        // implicit half: (I - theta dt L) u_new = rhs
        let th = theta * dt;
        for j in 0..n_k {
            ws.dia[j] = 1.0 - th * ws.diag[j];
        }
        for j in 0..n_k - 1 {
            ws.sub[j] = -th * ws.lower[j + 1];
            ws.sup[j] = -th * ws.upper[j];
        }
        if !thomas_inplace(
            &ws.sub,
            &ws.dia,
            &ws.sup,
            &ws.rhs,
            &mut ws.cw,
            &mut ws.dw,
            &mut ws.next,
        ) {
            return Err(RustyQLibError::NumericalError(format!(
                "Dupire forward march: tridiagonal solve broke down at step {n} (t = {t2})"
            )));
        }
        std::mem::swap(&mut ws.cur, &mut ws.next);

        // events at the node just reached: maturities read the pre-jump layer
        while next_mat < n_mat && mat_node[next_mat] == n + 1 {
            let i = next_mat;
            prices[i * n_k..(i + 1) * n_k].copy_from_slice(&ws.cur);
            df[i] = market.df(maturities[i]);
            forward[i] = market.forward(maturities[i]);
            next_mat += 1;
        }
        while next_div < div_events.len() && div_events[next_div].0 == n + 1 {
            let delta = div_events[next_div].1;
            for j in 0..n_k {
                let x = (k_grid[j].exp() + delta).ln();
                ws.next[j] = lagrange4(k_grid[0], dk, &ws.cur, x);
            }
            std::mem::swap(&mut ws.cur, &mut ws.next);
            implicit_left = cfg.rannacher_steps;
            next_div += 1;
        }
    }
    if upwinded > 0 {
        log::debug!(
            "dupire_forward: {upwinded} interior rows upwinded over {} steps",
            t.len() - 1
        );
    }
    Ok(DenseSurface {
        right,
        s0,
        k: k_grid.to_vec(),
        dk,
        maturities: maturities.to_vec(),
        df,
        forward,
        prices,
        t_nodes: t,
        upwinded_rows: upwinded,
    })
}

// ── Implied volatility ───────────────────────────────────────────────────

/// Why a price could not be inverted to a Black-76 volatility.
#[derive(Debug, Clone, PartialEq)]
pub enum IvError {
    /// The (undiscounted, parity-converted) price is not strictly positive.
    NonPositivePrice,
    /// The inverter returned its `1e-4` floor: the price sits within
    /// round-off of the intrinsic bound and carries no vol information.
    AtFloor,
    /// The crate inverter rejected the inputs (price outside the arbitrage
    /// bounds, vol above 5.0, expired, non-finite); the message is kept.
    Rejected(String),
}

impl std::fmt::Display for IvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IvError::NonPositivePrice => f.write_str("price is not positive"),
            IvError::AtFloor => f.write_str("implied vol at the 1e-4 floor"),
            IvError::Rejected(m) => write!(f, "implied vol rejected: {m}"),
        }
    }
}

/// Black-76 implied vol of an UNDISCOUNTED price against the `forward`:
/// `implied_vol_from_price(forward, strike, 0, 0, t, price, right)` with
/// the crate's silent floor and its errors mapped to [`IvError`].
pub fn black76_implied_vol(
    forward: f64,
    strike: f64,
    t: f64,
    undiscounted_price: f64,
    right: PutOrCall,
) -> std::result::Result<f64, IvError> {
    // NaN is "not positive" too; +inf falls through to the crate's bounds
    if undiscounted_price.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(IvError::NonPositivePrice);
    }
    match implied_vol_from_price(forward, strike, 0.0, 0.0, t, undiscounted_price, right) {
        Ok(v) if v <= IV_FLOOR * (1.0 + 1e-9) => Err(IvError::AtFloor),
        Ok(v) => Ok(v),
        Err(e) => Err(IvError::Rejected(e.to_string())),
    }
}

/// [`black76_implied_vol`] after converting the price to the
/// out-of-the-money right by parity (`c - p = forward - strike`), where
/// the inversion is best conditioned.
pub fn otm_implied_vol(
    forward: f64,
    strike: f64,
    t: f64,
    undiscounted_price: f64,
    right: PutOrCall,
) -> std::result::Result<f64, IvError> {
    let (target, otm_right) = match right {
        PutOrCall::Call if strike < forward => {
            (undiscounted_price - (forward - strike), PutOrCall::Put)
        }
        PutOrCall::Put if strike >= forward => {
            (undiscounted_price + (forward - strike), PutOrCall::Call)
        }
        _ => (undiscounted_price, right),
    };
    black76_implied_vol(forward, strike, t, target, otm_right)
}

/// Implied vols of a dense surface at listed `strikes`, one inner vector
/// per maturity (interpolated price, Black-76 on the forward, i.e. the
/// effective flat rates of every expiry).
pub fn implied_vols(
    surface: &DenseSurface,
    strikes: &[f64],
) -> Vec<Vec<std::result::Result<f64, IvError>>> {
    (0..surface.n_t())
        .map(|i| {
            strikes
                .iter()
                .map(|&strike| surface.implied_vol_at(i, strike))
                .collect()
        })
        .collect()
}

// ── Static-arbitrage scans ───────────────────────────────────────────────

/// Outcome of [`arbitrage_scan`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ArbReport {
    /// Adjacent-strike butterflies below `-tol_butterfly` (undiscounted
    /// price units).
    pub butterfly_count: usize,
    /// Calendar pairs whose total variance decreased by more than
    /// `tol_calendar` at matched forward log-moneyness.
    pub calendar_count: usize,
    /// The most negative butterfly value seen (`0` when nothing was
    /// checked; positive when every triple was strictly convex).
    pub worst_butterfly: f64,
    /// The largest decrease `w(T_{i-1}) - w(T_i)` seen (`0` when nothing
    /// was checked; negative when total variance always increased).
    pub worst_calendar: f64,
    /// Butterfly triples plus calendar pairs evaluated.
    pub n_checked: usize,
    /// Butterfly triples evaluated.
    pub n_checked_butterfly: usize,
    /// Calendar pairs evaluated.
    pub n_checked_calendar: usize,
    /// Calendar pairs skipped because a vol was unavailable or the node
    /// was below [`CALENDAR_PRICE_FLOOR`].
    pub n_skipped: usize,
}

/// Scan a dense surface for static arbitrage.
///
/// Butterfly: for every maturity and every adjacent strike triple
/// `K1 < K2 < K3`, `w c1 + (1 - w) c3 - c2` with `w = (K3 - K2)/(K3 - K1)`
/// in UNDISCOUNTED call price units (the formula of the library's
/// `VolSurface::diagnostics`); a violation is a value below
/// `-tol_butterfly`. Calendar: for every pair of adjacent maturities and
/// every node `j` of the later one, the total implied variance at forward
/// log-moneyness `y = k_j - ln F_i` must not fall below the earlier
/// maturity's total variance at the same `y` (cubic interpolation in `k`)
/// by more than `tol_calendar`. Tolerances are meant to be calibrated on
/// a flat-vol control ([`flat_vol_control`]).
pub fn arbitrage_scan(surface: &DenseSurface, tol_butterfly: f64, tol_calendar: f64) -> ArbReport {
    let n_k = surface.n_k();
    let n_t = surface.n_t();
    let mut report = ArbReport::default();
    let mut worst_b = f64::INFINITY;
    let mut worst_c = f64::NEG_INFINITY;

    // butterfly in strike units
    let strikes: Vec<f64> = surface.k.iter().map(|k| k.exp()).collect();
    for i in 0..n_t {
        for j in 1..n_k - 1 {
            let (k1, k2, k3) = (strikes[j - 1], strikes[j], strikes[j + 1]);
            let w = (k3 - k2) / (k3 - k1);
            let b = w * surface.undiscounted_call(i, j - 1)
                + (1.0 - w) * surface.undiscounted_call(i, j + 1)
                - surface.undiscounted_call(i, j);
            report.n_checked_butterfly += 1;
            worst_b = worst_b.min(b);
            if b < -tol_butterfly {
                report.butterfly_count += 1;
            }
        }
    }

    // calendar in total variance at matched forward log-moneyness; NaN
    // marks a node without a vol and propagates through the interpolation
    let floor = CALENDAR_PRICE_FLOOR * surface.s0;
    let tv: Vec<f64> = surface
        .total_variance_grid()
        .into_iter()
        .enumerate()
        .map(|(idx, v)| {
            let (i, j) = (idx / n_k, idx % n_k);
            let c = surface.undiscounted_call(i, j);
            let otm = c.min(c - (surface.forward[i] - strikes[j]));
            match v {
                Some(w) if otm >= floor => w,
                _ => f64::NAN,
            }
        })
        .collect();
    for i in 1..n_t {
        let prev = &tv[(i - 1) * n_k..i * n_k];
        let ln_f_prev = surface.forward[i - 1].ln();
        let ln_f = surface.forward[i].ln();
        for j in 0..n_k {
            let w_cur = tv[i * n_k + j];
            let k_prev = surface.k[j] - ln_f + ln_f_prev;
            if w_cur.is_nan() || k_prev < surface.k[0] || k_prev > surface.k[n_k - 1] {
                report.n_skipped += 1;
                continue;
            }
            let w_prev = lagrange4(surface.k[0], surface.dk, prev, k_prev);
            if w_prev.is_nan() {
                report.n_skipped += 1;
                continue;
            }
            let d = w_prev - w_cur;
            report.n_checked_calendar += 1;
            worst_c = worst_c.max(d);
            if d > tol_calendar {
                report.calendar_count += 1;
            }
        }
    }
    report.worst_butterfly = if worst_b.is_finite() { worst_b } else { 0.0 };
    report.worst_calendar = if worst_c.is_finite() { worst_c } else { 0.0 };
    report.n_checked = report.n_checked_butterfly + report.n_checked_calendar;
    report
}

/// The flat-vol control: the identical dense pipeline under a constant
/// volatility, whose worst violations are pure discretization noise and
/// set the tolerances of the real scans.
#[derive(Debug, Clone, PartialEq)]
pub struct FlatControl {
    /// Most negative butterfly of the flat surface (undiscounted price).
    pub worst_butterfly: f64,
    /// Largest calendar decrease of the flat surface (total variance).
    pub worst_calendar: f64,
    /// `safety * max(0, -worst_butterfly)`, floored at `1e-12 s0`.
    pub tol_butterfly: f64,
    /// `safety * max(0, worst_calendar)`, floored at `1e-12`.
    pub tol_calendar: f64,
    /// The zero-tolerance report of the flat surface.
    pub report: ArbReport,
}

/// Run [`dupire_forward`] with [`FlatVol`]`(sigma)` on the given grid and
/// scan it at zero tolerance; the suggested tolerances are `safety` times
/// the worst violations (2 is the spec's choice).
pub fn flat_vol_control(
    sigma: f64,
    market: &ForwardMarket,
    k_grid: &[f64],
    maturities: &[f64],
    cfg: &DupireConfig,
    safety: f64,
) -> Result<FlatControl> {
    let surface = dupire_forward(&FlatVol(sigma), market, k_grid, maturities, cfg)?;
    let report = arbitrage_scan(&surface, 0.0, 0.0);
    let vb = (-report.worst_butterfly).max(0.0);
    let vc = report.worst_calendar.max(0.0);
    Ok(FlatControl {
        worst_butterfly: report.worst_butterfly,
        worst_calendar: report.worst_calendar,
        tol_butterfly: (safety * vb).max(1e-12 * market.s0),
        tol_calendar: (safety * vc).max(1e-12),
        report,
    })
}

/// A listed quote as a discounted CALL price band `[bid, ask]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuoteBand {
    /// Strike.
    pub strike: f64,
    /// Discounted call bid.
    pub bid: f64,
    /// Discounted call ask.
    pub ask: f64,
}

impl QuoteBand {
    /// A call quote.
    pub fn call(strike: f64, bid: f64, ask: f64) -> Self {
        QuoteBand { strike, bid, ask }
    }

    /// A put quote converted to a call band by parity,
    /// `c = p + df (forward - strike)` applied to both edges.
    pub fn from_put(strike: f64, bid: f64, ask: f64, df: f64, forward: f64) -> Self {
        let shift = df * (forward - strike);
        QuoteBand {
            strike,
            bid: bid + shift,
            ask: ask + shift,
        }
    }
}

/// The listed call bands of one expiry.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpiryQuotes {
    /// Time to expiry in years.
    pub t: f64,
    /// Discount factor to expiry.
    pub df: f64,
    /// Forward at expiry.
    pub forward: f64,
    /// Call bands (any order; sorted by strike internally).
    pub calls: Vec<QuoteBand>,
}

/// Outcome of [`arbitrage_scan_quotes`]: raw counts use the mids;
/// beyond-spread counts keep only the violations that no choice of
/// prices inside the bands removes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct QuoteArbReport {
    /// Butterfly violations of the mids.
    pub butterfly_raw: usize,
    /// Butterflies still negative with `ask1, ask3` and `bid2`.
    pub butterfly_beyond_spread: usize,
    /// Calendar violations of the mids.
    pub calendar_raw: usize,
    /// Calendar violations that persist with the later expiry at its ask
    /// vol and the earlier expiry at its bid vol.
    pub calendar_beyond_spread: usize,
    /// Most negative mid butterfly (undiscounted).
    pub worst_butterfly: f64,
    /// Largest mid total-variance decrease.
    pub worst_calendar: f64,
    /// Butterfly triples evaluated.
    pub n_checked_butterfly: usize,
    /// Calendar comparisons evaluated (mid vols available on both sides).
    pub n_checked_calendar: usize,
}

/// One listed strike with undiscounted prices and total variances at the
/// three band points (`w_bid = 0` when the bid is below the vol floor,
/// `w_ask = inf` when the ask is above the upper bound).
struct BandRow {
    strike: f64,
    y: f64,
    bid: f64,
    mid: f64,
    ask: f64,
    w_bid: f64,
    w_mid: Option<f64>,
    w_ask: f64,
}

/// Scan listed quotes with bid/ask bands.
///
/// Butterfly per expiry over adjacent strike triples, undiscounted, with
/// the dense weight `w = (K3 - K2)/(K3 - K1)`: raw uses the mids; a
/// violation is beyond-spread when `w ask1 + (1 - w) ask3 - bid2` (the
/// least negative value the bands allow) is still below
/// `-tol_butterfly`. Calendar between adjacent expiries at every strike
/// of the later expiry whose forward log-moneyness lies inside the
/// earlier expiry's range: the earlier total variance is linearly
/// interpolated in log-moneyness; raw compares mids, beyond-spread
/// compares the later ask vol against the earlier interpolated bid vol.
pub fn arbitrage_scan_quotes(
    expiries: &[ExpiryQuotes],
    tol_butterfly: f64,
    tol_calendar: f64,
) -> QuoteArbReport {
    let mut report = QuoteArbReport::default();
    let mut worst_b = f64::INFINITY;
    let mut worst_c = f64::NEG_INFINITY;

    let mut order: Vec<usize> = (0..expiries.len()).collect();
    order.sort_by(|&a, &b| expiries[a].t.partial_cmp(&expiries[b].t).unwrap());
    let rows: Vec<Vec<BandRow>> = order
        .iter()
        .map(|&e| {
            let ex = &expiries[e];
            let mut calls = ex.calls.clone();
            calls.sort_by(|a, b| a.strike.partial_cmp(&b.strike).unwrap());
            // (undiscounted price, strike) -> total implied variance
            let tv = |c: (f64, f64)| {
                otm_implied_vol(ex.forward, c.1, ex.t, c.0, PutOrCall::Call)
                    .ok()
                    .map(|v| v * v * ex.t)
            };
            calls
                .iter()
                .map(|qb| {
                    let bid = qb.bid / ex.df;
                    let ask = qb.ask / ex.df;
                    let mid = 0.5 * (bid + ask);
                    BandRow {
                        strike: qb.strike,
                        y: (qb.strike / ex.forward).ln(),
                        bid,
                        mid,
                        ask,
                        w_bid: tv((bid, qb.strike)).unwrap_or(0.0),
                        w_mid: tv((mid, qb.strike)),
                        w_ask: tv((ask, qb.strike)).unwrap_or(f64::INFINITY),
                    }
                })
                .collect()
        })
        .collect();

    for ex in &rows {
        for w3 in ex.windows(3) {
            let (r1, r2, r3) = (&w3[0], &w3[1], &w3[2]);
            let w = (r3.strike - r2.strike) / (r3.strike - r1.strike);
            let raw = w * r1.mid + (1.0 - w) * r3.mid - r2.mid;
            let best = w * r1.ask + (1.0 - w) * r3.ask - r2.bid;
            report.n_checked_butterfly += 1;
            worst_b = worst_b.min(raw);
            if raw < -tol_butterfly {
                report.butterfly_raw += 1;
                if best < -tol_butterfly {
                    report.butterfly_beyond_spread += 1;
                }
            }
        }
    }

    for pair in rows.windows(2) {
        let (earlier, later) = (&pair[0], &pair[1]);
        if earlier.len() < 2 {
            continue;
        }
        for row in later {
            let y = row.y;
            let Some(w_cur) = row.w_mid else { continue };
            // bracketing strikes of the earlier expiry in log-moneyness
            let pos = earlier.partition_point(|r| r.y <= y);
            if pos == 0 || pos >= earlier.len() {
                continue;
            }
            let (a, b) = (&earlier[pos - 1], &earlier[pos]);
            let lam = (y - a.y) / (b.y - a.y);
            let (Some(wa), Some(wb)) = (a.w_mid, b.w_mid) else {
                continue;
            };
            let w_prev = (1.0 - lam) * wa + lam * wb;
            let w_prev_min = (1.0 - lam) * a.w_bid + lam * b.w_bid;
            report.n_checked_calendar += 1;
            let d = w_prev - w_cur;
            worst_c = worst_c.max(d);
            if d > tol_calendar {
                report.calendar_raw += 1;
                if w_prev_min - row.w_ask > tol_calendar {
                    report.calendar_beyond_spread += 1;
                }
            }
        }
    }
    report.worst_butterfly = if worst_b.is_finite() { worst_b } else { 0.0 };
    report.worst_calendar = if worst_c.is_finite() { worst_c } else { 0.0 };
    report
}

// ── Backward European prices at the quotes ───────────────────────────────

/// European prices of `quotes` (all at the mesh's expiry) under `vol` by
/// the BACKWARD European solver on `mesh` with `market`: the same grid,
/// time steps and evaluation point as the American solves of `sigma^A`,
/// so the discretization error cancels in `E = sigma^A - sigma^E`. The
/// field is sampled once at the mesh's nodes and step mid-times; the
/// quotes are priced in parallel with thread-local solver workspaces.
pub fn european_prices_backward(
    mesh: &super::grid::Mesh,
    quotes: &[super::solver::QuoteSpec],
    vol: &dyn VolField,
    market: &super::grid::MarketSlice,
) -> Result<Vec<f64>> {
    use super::solver::{price_only, Mode, Workspace};
    use rayon::prelude::*;
    market.validate(mesh)?;
    let field = mesh.node_field(vol);
    quotes
        .par_iter()
        .map_init(Workspace::new, |ws, spec| {
            price_only(mesh, spec, &field, market, Mode::European, ws)
        })
        .collect()
}

#[cfg(test)]
#[path = "european_surface_tests.rs"]
mod tests;
