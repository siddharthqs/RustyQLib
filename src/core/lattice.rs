//! Recombining binomial lattices, asset-class agnostic.
//!
//! Every classic parameterization reduces to one recombining structure:
//! a node after `j` up-moves out of `i` steps sits at
//! `S(i, j) = S0 * exp(j*log_up + (i-j)*log_down)`, with up-probability
//! `p_up` — [`LatticeParams`]. The [`BinomialTreeType`] enum supplies the
//! `(log_up, log_down, p_up)` triple for Cox-Ross-Rubinstein, Jarrow-Rudd,
//! Tian, Trigeorgis, Leisen-Reimer (Peizer-Pratt method 2, strike-aware,
//! second-order smooth convergence) and the equal-probability additive
//! tree (Clewlow-Strickland / QuantLib's `AdditiveEQPBinomialTree`).
//!
//! Three engines share the parameterization:
//! - [`price_backward`] — the production engine: a rolling one-dimensional
//!   value array (O(n) memory, no tree materialized) with the layer spot
//!   levels rebuilt from two precomputed power tables.
//! - [`price_backward_with_greeks`] — the same rolling pass, additionally
//!   keeping the first two layers so the value and the tree
//!   delta/gamma/theta come from a single induction.
//! - [`price_with_diagnostics`] — the debug engine: keeps the full spot
//!   and value trees, records the early-exercise boundary per layer, tree
//!   Greeks read off the first layers, and wall-clock time.
//!
//! [`convergence_study`] prices across a ladder of step counts (with
//! per-point timing) to expose each scheme's convergence behavior — CRR
//! oscillates at first order, Leisen-Reimer converges smoothly at second.
//!
//! Payoffs and early exercise enter as closures, so the same lattice
//! prices equity payoffs today and other asset classes later.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::core::errors::RustyQLibError;

// ── Tree types ──────────────────────────────────────────────────────────

/// The lattice parameterization: how `(u, d, p)` are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinomialTreeType {
    /// `u = e^{sigma sqrt(dt)}`, `d = 1/u`; the classic. First-order,
    /// oscillating convergence.
    #[serde(alias = "CRR", alias = "crr")]
    CoxRossRubinstein,
    /// Equal probabilities with the drift in the node spacing.
    #[serde(alias = "JR", alias = "jr")]
    JarrowRudd,
    /// Matches the first three moments of the lognormal step.
    Tian,
    /// Additive in log-space with drift-adjusted spacing.
    Trigeorgis,
    /// Peizer-Pratt inversion centered on the strike; needs an odd step
    /// count (even counts are bumped up by one). Second-order, smooth —
    /// the default: at ~100 steps it matches CRR at 1000.
    #[default]
    #[serde(alias = "LR", alias = "lr")]
    LeisenReimer,
    /// Equal-probability additive tree (Clewlow-Strickland).
    #[serde(alias = "EQP", alias = "eqp")]
    AdditiveEqp,
}

impl std::str::FromStr for BinomialTreeType {
    type Err = RustyQLibError;
    fn from_str(s: &str) -> Result<Self, RustyQLibError> {
        use BinomialTreeType::*;
        Ok(match s.trim().to_lowercase().as_str() {
            "crr" | "coxrossrubinstein" | "cox_ross_rubinstein" => CoxRossRubinstein,
            "jr" | "jarrowrudd" | "jarrow_rudd" => JarrowRudd,
            "tian" => Tian,
            "trigeorgis" => Trigeorgis,
            "lr" | "leisenreimer" | "leisen_reimer" => LeisenReimer,
            "eqp" | "additiveeqp" | "additive_eqp" => AdditiveEqp,
            other => {
                return Err(RustyQLibError::invalid_input(
                    "tree_type",
                    format!(
                        "unknown tree type '{other}' (use CRR, JarrowRudd, Tian, \
                         Trigeorgis, LeisenReimer or EQP)"
                    ),
                ))
            }
        })
    }
}

/// Lattice engine configuration carried by an instrument.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatticeConfig {
    pub tree_type: BinomialTreeType,
    pub steps: usize,
    /// Price on the [`TermLattice`]: term structures of rates, carry and
    /// volatility applied per step (variance-equal time grid). When set,
    /// `tree_type` is ignored — the term lattice has its own
    /// CRR-in-variance spacing.
    #[serde(default)]
    pub term_structure: bool,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        LatticeConfig {
            tree_type: BinomialTreeType::LeisenReimer,
            steps: 1000,
            term_structure: false,
        }
    }
}

/// The general recombining step: `S(i, j) = S0 e^{j lu + (i-j) ld}`,
/// up with probability `p_up`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeParams {
    pub log_up: f64,
    pub log_down: f64,
    pub p_up: f64,
}

impl BinomialTreeType {
    /// The step count this scheme actually uses (Leisen-Reimer needs an
    /// odd number of steps for the Peizer-Pratt inversion).
    pub fn effective_steps(&self, steps: usize) -> usize {
        let steps = steps.max(2);
        match self {
            BinomialTreeType::LeisenReimer if steps % 2 == 0 => steps + 1,
            _ => steps,
        }
    }

    /// The `(log_up, log_down, p_up)` triple for `n` steps over life `t`
    /// with carry drift `b` (`r - q`) and volatility `sigma`. `s0` and
    /// `strike` are used by the strike-aware Leisen-Reimer scheme only.
    pub fn params(
        &self,
        s0: f64,
        strike: f64,
        b: f64,
        sigma: f64,
        t: f64,
        n: usize,
    ) -> Result<LatticeParams, RustyQLibError> {
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "sigma",
                format!("lattice volatility must be positive and finite, got {sigma}"),
            ));
        }
        if !(t > 0.0) || n < 2 {
            return Err(RustyQLibError::invalid_input(
                "lattice",
                "the lattice needs positive maturity and at least two steps",
            ));
        }
        let dt = t / n as f64;
        let nu = b - 0.5 * sigma * sigma;
        let params = match self {
            BinomialTreeType::CoxRossRubinstein => {
                let dx = sigma * dt.sqrt();
                let (u, d) = (dx.exp(), (-dx).exp());
                LatticeParams {
                    log_up: dx,
                    log_down: -dx,
                    p_up: ((b * dt).exp() - d) / (u - d),
                }
            }
            BinomialTreeType::JarrowRudd => LatticeParams {
                log_up: nu * dt + sigma * dt.sqrt(),
                log_down: nu * dt - sigma * dt.sqrt(),
                p_up: 0.5,
            },
            BinomialTreeType::Tian => {
                let m = (b * dt).exp();
                let v = (sigma * sigma * dt).exp();
                let root = (v * v + 2.0 * v - 3.0).sqrt();
                let u = 0.5 * m * v * (v + 1.0 + root);
                let d = 0.5 * m * v * (v + 1.0 - root);
                LatticeParams { log_up: u.ln(), log_down: d.ln(), p_up: (m - d) / (u - d) }
            }
            BinomialTreeType::Trigeorgis => {
                let dx = (sigma * sigma * dt + nu * nu * dt * dt).sqrt();
                LatticeParams {
                    log_up: dx,
                    log_down: -dx,
                    p_up: 0.5 + 0.5 * nu * dt / dx,
                }
            }
            BinomialTreeType::LeisenReimer => {
                if !(s0 > 0.0 && strike > 0.0) {
                    return Err(RustyQLibError::invalid_input(
                        "lattice",
                        "Leisen-Reimer needs positive spot and strike",
                    ));
                }
                let n_odd = self.effective_steps(n);
                if n_odd != n {
                    return Err(RustyQLibError::invalid_input(
                        "lattice",
                        "Leisen-Reimer needs an odd step count (use effective_steps)",
                    ));
                }
                let sq_t = sigma * t.sqrt();
                let d1 = ((s0 / strike).ln() + (b + 0.5 * sigma * sigma) * t) / sq_t;
                let d2 = d1 - sq_t;
                let p = peizer_pratt(d2, n);
                let p_star = peizer_pratt(d1, n);
                let m = (b * dt).exp();
                let u = m * p_star / p;
                let d = (m - p * u) / (1.0 - p);
                if !(d > 0.0 && u > d) {
                    return Err(RustyQLibError::NumericalError(format!(
                        "Leisen-Reimer step degenerated (u {u}, d {d}); increase the step count"
                    )));
                }
                LatticeParams { log_up: u.ln(), log_down: d.ln(), p_up: p }
            }
            BinomialTreeType::AdditiveEqp => {
                let disc = 4.0 * sigma * sigma * dt - 3.0 * nu * nu * dt * dt;
                if disc <= 0.0 {
                    return Err(RustyQLibError::NumericalError(
                        "EQP tree needs 4 sigma^2 dt > 3 nu^2 dt^2; increase the step count"
                            .to_string(),
                    ));
                }
                LatticeParams {
                    log_up: 0.5 * nu * dt + 0.5 * disc.sqrt(),
                    log_down: 1.5 * nu * dt - 0.5 * disc.sqrt(),
                    p_up: 0.5,
                }
            }
        };
        if !(params.p_up > 0.0 && params.p_up < 1.0) {
            return Err(RustyQLibError::NumericalError(format!(
                "lattice probability {:.6} outside (0, 1): the time step is too \
                 large for this drift/volatility (increase the step count)",
                params.p_up
            )));
        }
        Ok(params)
    }
}

/// Peizer-Pratt method-2 inversion of the binomial CDF.
fn peizer_pratt(z: f64, n: usize) -> f64 {
    let nf = n as f64;
    let scaled = z / (nf + 1.0 / 3.0 + 0.1 / (nf + 1.0));
    let inner = (1.0 - (-scaled * scaled * (nf + 1.0 / 6.0)).exp()).sqrt();
    0.5 + 0.5 * inner.copysign(z)
}

// ── Optimized engine ────────────────────────────────────────────────────

/// Backward induction on a rolling one-dimensional array: O(n) memory,
/// O(n^2) work, spot levels rebuilt from two precomputed power tables.
///
/// `terminal(spot)` is the payoff at expiry; `exercise` (when given)
/// maps `(step, spot, continuation)` to the node value, which expresses
/// American (`intrinsic.max(cont)` at every step), Bermudan (only on
/// listed steps) or any custom early-exercise rule.
pub fn price_backward(
    s0: f64,
    params: &LatticeParams,
    n: usize,
    df_step: f64,
    terminal: &dyn Fn(f64) -> f64,
    exercise: Option<&dyn Fn(usize, f64, f64) -> f64>,
) -> f64 {
    // spot(i, j) = s0 * up_pow[j] * down_pow[i - j]
    let up_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_up).exp()).collect();
    let down_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_down).exp()).collect();
    let spot = |i: usize, j: usize| s0 * up_pow[j] * down_pow[i - j];

    let mut v: Vec<f64> = (0..=n).map(|j| terminal(spot(n, j))).collect();
    let (p, q) = (params.p_up, 1.0 - params.p_up);
    for i in (0..n).rev() {
        for j in 0..=i {
            // from (i, j): up -> (i+1, j+1), down -> (i+1, j)
            let cont = df_step * (p * v[j + 1] + q * v[j]);
            v[j] = match exercise {
                Some(ex) => ex(i, spot(i, j), cont),
                None => cont,
            };
        }
    }
    v[0]
}

/// Value and the tree Greeks read off a single backward pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeSolution {
    pub price: f64,
    /// Tree delta from the first layer.
    pub delta: f64,
    /// Tree gamma from the second layer.
    pub gamma: f64,
    /// Calendar theta (per year) from the second-layer center vs the root,
    /// drift-corrected with the tree's own delta and gamma so asymmetric
    /// trees (Leisen-Reimer, Jarrow-Rudd, Tian) are handled too.
    pub theta: f64,
}

/// The same rolling-array induction as [`price_backward`] (identical price,
/// bit for bit) that additionally keeps the first two layers, so the value
/// and the tree delta/gamma/theta come out of **one** pass — no re-pricing
/// per Greek.
pub fn price_backward_with_greeks(
    s0: f64,
    params: &LatticeParams,
    n: usize,
    dt: f64,
    df_step: f64,
    terminal: &dyn Fn(f64) -> f64,
    exercise: Option<&dyn Fn(usize, f64, f64) -> f64>,
) -> LatticeSolution {
    let up_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_up).exp()).collect();
    let down_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_down).exp()).collect();
    let spot = |i: usize, j: usize| s0 * up_pow[j] * down_pow[i - j];

    let mut v: Vec<f64> = (0..=n).map(|j| terminal(spot(n, j))).collect();
    let mut layer2 = [0.0; 3];
    let mut layer1 = [0.0; 2];
    if n == 2 {
        layer2.copy_from_slice(&v[0..3]);
    }
    let (p, q) = (params.p_up, 1.0 - params.p_up);
    for i in (0..n).rev() {
        for j in 0..=i {
            let cont = df_step * (p * v[j + 1] + q * v[j]);
            v[j] = match exercise {
                Some(ex) => ex(i, spot(i, j), cont),
                None => cont,
            };
        }
        match i {
            2 => layer2.copy_from_slice(&v[0..3]),
            1 => layer1.copy_from_slice(&v[0..2]),
            _ => {}
        }
    }

    let price = v[0];
    let delta = (layer1[1] - layer1[0]) / (spot(1, 1) - spot(1, 0));
    let (s_uu, s_ud, s_dd) = (spot(2, 2), spot(2, 1), spot(2, 0));
    let d_up = (layer2[2] - layer2[1]) / (s_uu - s_ud);
    let d_down = (layer2[1] - layer2[0]) / (s_ud - s_dd);
    let gamma = (d_up - d_down) / (0.5 * (s_uu - s_dd));
    // the second-layer center sits at s0 only on symmetric trees; remove
    // the spot displacement with the tree's delta and gamma before reading
    // the calendar decay over the 2*dt elapsed
    let ds = s_ud - s0;
    let theta = (layer2[1] - price - delta * ds - 0.5 * gamma * ds * ds) / (2.0 * dt);

    LatticeSolution { price, delta, gamma, theta }
}

// ── Diagnostic engine ───────────────────────────────────────────────────

/// Everything the debug lattice records beyond the price.
#[derive(Debug, Clone)]
pub struct LatticeDiagnostics {
    pub price: f64,
    pub tree_type: BinomialTreeType,
    pub steps: usize,
    pub params: LatticeParams,
    /// Wall-clock time of the build + induction.
    pub elapsed: Duration,
    /// `spot_tree[i][j]`: layer `i` has `i + 1` nodes, `j` up-moves.
    pub spot_tree: Vec<Vec<f64>>,
    pub value_tree: Vec<Vec<f64>>,
    /// Per layer, the `(min, max)` spot at which early exercise was
    /// optimal; `None` where the option was never exercised.
    pub exercise_boundary: Vec<Option<(f64, f64)>>,
    /// Tree delta from the first layer.
    pub delta: f64,
    /// Tree gamma from the second layer.
    pub gamma: f64,
    /// Tree theta from the second-layer center vs the root, per year
    /// (exact for symmetric trees where that node returns to `s0`,
    /// approximate otherwise).
    pub theta: f64,
}

/// The debug engine: same induction as [`price_backward`] but keeping
/// every layer, the exercise boundary, tree Greeks and timing.
pub fn price_with_diagnostics(
    tree_type: BinomialTreeType,
    s0: f64,
    params: &LatticeParams,
    n: usize,
    dt: f64,
    df_step: f64,
    terminal: &dyn Fn(f64) -> f64,
    exercise: Option<&dyn Fn(usize, f64, f64) -> f64>,
) -> LatticeDiagnostics {
    let start = Instant::now();
    // identical spot computation to `price_backward`, so the two engines
    // agree bit-for-bit
    let up_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_up).exp()).collect();
    let down_pow: Vec<f64> = (0..=n).map(|j| (j as f64 * params.log_down).exp()).collect();
    let spot_tree: Vec<Vec<f64>> = (0..=n)
        .map(|i| (0..=i).map(|j| s0 * up_pow[j] * down_pow[i - j]).collect())
        .collect();

    let mut value_tree: Vec<Vec<f64>> = spot_tree
        .iter()
        .map(|layer| layer.iter().map(|_| 0.0).collect())
        .collect();
    value_tree[n] = spot_tree[n].iter().map(|&s| terminal(s)).collect();

    let mut exercise_boundary: Vec<Option<(f64, f64)>> = vec![None; n + 1];
    let (p, q) = (params.p_up, 1.0 - params.p_up);
    for i in (0..n).rev() {
        for j in 0..=i {
            let cont = df_step * (p * value_tree[i + 1][j + 1] + q * value_tree[i + 1][j]);
            let spot = spot_tree[i][j];
            let value = match exercise {
                Some(ex) => ex(i, spot, cont),
                None => cont,
            };
            if value > cont {
                let entry = exercise_boundary[i].get_or_insert((spot, spot));
                entry.0 = entry.0.min(spot);
                entry.1 = entry.1.max(spot);
            }
            value_tree[i][j] = value;
        }
    }

    let price = value_tree[0][0];
    let delta = (value_tree[1][1] - value_tree[1][0]) / (spot_tree[1][1] - spot_tree[1][0]);
    let (s_uu, s_ud, s_dd) = (spot_tree[2][2], spot_tree[2][1], spot_tree[2][0]);
    let (v_uu, v_ud, v_dd) = (value_tree[2][2], value_tree[2][1], value_tree[2][0]);
    let d_up = (v_uu - v_ud) / (s_uu - s_ud);
    let d_down = (v_ud - v_dd) / (s_ud - s_dd);
    let gamma = (d_up - d_down) / (0.5 * (s_uu - s_dd));
    let theta = (v_ud - price) / (2.0 * dt);

    LatticeDiagnostics {
        price,
        tree_type,
        steps: n,
        params: *params,
        elapsed: start.elapsed(),
        spot_tree,
        value_tree,
        exercise_boundary,
        delta,
        gamma,
        theta,
    }
}

// ── Convergence study ───────────────────────────────────────────────────

/// One rung of a convergence ladder.
#[derive(Debug, Clone, Copy)]
pub struct ConvergencePoint {
    pub steps: usize,
    pub price: f64,
    pub elapsed: Duration,
}

/// Price the same contract across a ladder of step counts on the
/// optimized engine, timing each rung — the raw material for studying a
/// scheme's convergence order and oscillation.
#[allow(clippy::too_many_arguments)]
pub fn convergence_study(
    tree_type: BinomialTreeType,
    s0: f64,
    strike: f64,
    b: f64,
    sigma: f64,
    r: f64,
    t: f64,
    steps_ladder: &[usize],
    terminal: &dyn Fn(f64) -> f64,
    exercise: Option<&dyn Fn(usize, f64, f64) -> f64>,
) -> Result<Vec<ConvergencePoint>, RustyQLibError> {
    steps_ladder
        .iter()
        .map(|&steps| {
            let n = tree_type.effective_steps(steps);
            let params = tree_type.params(s0, strike, b, sigma, t, n)?;
            let df_step = (-r * t / n as f64).exp();
            let start = Instant::now();
            let price = price_backward(s0, &params, n, df_step, terminal, exercise);
            Ok(ConvergencePoint { steps: n, price, elapsed: start.elapsed() })
        })
        .collect()
}


// ── Time-dependent (term-structure) lattice ─────────────────────────────

/// A recombining binomial lattice under **time-dependent parameters**:
/// term structures of rates, carry and volatility applied directly on
/// the tree.
///
/// Construction (the standard variance-grid method):
/// 1. The time grid is warped so every step accrues equal variance
///    `w = V(T)/n`, where `V(t)` is the cumulative variance supplied by
///    the caller. Fixed log-spacing `dx = sqrt(w)` then keeps the tree
///    recombining even though volatility varies with time.
/// 2. Each step's drift is matched exactly by a **per-step probability**
///    from the forward rate and carry over that step, and each step
///    discounts with its own forward discount factor.
///
/// So flat inputs reduce to the classic CRR tree, while curved inputs
/// reprice the exact term structure: for a European payoff the tree
/// converges to Black-Scholes with the equivalent average variance and
/// the curve's exact discount factor.
#[derive(Debug, Clone)]
pub struct TermLattice {
    /// Layer times `t_0 = 0 .. t_n = T` (unequal spacing in general).
    pub times: Vec<f64>,
    /// Fixed log-spacing between adjacent nodes.
    pub dx: f64,
    /// Per-step up-probability (drift-matched from the forward rates).
    pub p_up: Vec<f64>,
    /// Per-step discount factor (forward rate over the step).
    pub df: Vec<f64>,
}

impl TermLattice {
    /// Build the grid for `n` steps over `[0, t]`.
    ///
    /// - `forward_rate(t1, t2)`: continuously compounded forward rate
    ///   over the step (from a discount curve:
    ///   `ln(df(t1)/df(t2)) / (t2 - t1)`).
    /// - `forward_carry(t1, t2)`: forward dividend + borrow yield over
    ///   the step; the drift per step is `rate - carry`.
    /// - `total_variance(t)`: cumulative variance `sigma(t)^2 * t` (or
    ///   an integral of instantaneous variance); must be strictly
    ///   increasing — a calendar-arbitrage-free vol term structure.
    pub fn build(
        n: usize,
        t: f64,
        forward_rate: &dyn Fn(f64, f64) -> f64,
        forward_carry: &dyn Fn(f64, f64) -> f64,
        total_variance: &dyn Fn(f64) -> f64,
    ) -> Result<TermLattice, RustyQLibError> {
        if !(t > 0.0) || n < 2 {
            return Err(RustyQLibError::invalid_input(
                "lattice",
                "the lattice needs positive maturity and at least two steps",
            ));
        }
        let w_total = total_variance(t);
        if !(w_total > 0.0 && w_total.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "total_variance",
                format!("total variance to maturity must be positive and finite, got {w_total}"),
            ));
        }

        // variance-equal time grid: invert V(t) = i * w by bisection
        let mut times = Vec::with_capacity(n + 1);
        times.push(0.0);
        for i in 1..n {
            let target = w_total * i as f64 / n as f64;
            let (mut lo, mut hi) = (*times.last().expect("nonempty"), t);
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                if total_variance(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let ti = 0.5 * (lo + hi);
            if ti <= *times.last().expect("nonempty") {
                return Err(RustyQLibError::NumericalError(
                    "cumulative variance is not strictly increasing (calendar \
                     arbitrage in the vol term structure)"
                        .to_string(),
                ));
            }
            times.push(ti);
        }
        times.push(t);

        let dx = (w_total / n as f64).sqrt();
        let (u, d) = (dx.exp(), (-dx).exp());
        let mut p_up = Vec::with_capacity(n);
        let mut df = Vec::with_capacity(n);
        for i in 0..n {
            let (t1, t2) = (times[i], times[i + 1]);
            let dt = t2 - t1;
            let r = forward_rate(t1, t2);
            let b = r - forward_carry(t1, t2);
            let growth = (b * dt).exp();
            let p = (growth - d) / (u - d);
            if !(p > 0.0 && p < 1.0) {
                return Err(RustyQLibError::NumericalError(format!(
                    "step {i} probability {p:.6} outside (0, 1): the local drift \
                     is too large for the variance spacing (increase the step count)"
                )));
            }
            p_up.push(p);
            df.push((-r * dt).exp());
        }

        Ok(TermLattice { times, dx, p_up, df })
    }

    pub fn steps(&self) -> usize {
        self.p_up.len()
    }

    /// Backward induction on the rolling array with per-step
    /// probabilities and discounting. The early-exercise closure receives
    /// `(step, time, spot, continuation)` — time-aware because the layer
    /// times are unequal.
    pub fn price(
        &self,
        s0: f64,
        terminal: &dyn Fn(f64) -> f64,
        exercise: Option<&dyn Fn(usize, f64, f64, f64) -> f64>,
    ) -> f64 {
        let n = self.steps();
        // spot(i, j) = s0 * e^{(2j - i) dx}: one power table over [-n, n]
        let pow: Vec<f64> = (0..=2 * n).map(|k| ((k as f64 - n as f64) * self.dx).exp()).collect();
        let spot = |i: usize, j: usize| s0 * pow[2 * j + n - i];

        let mut v: Vec<f64> = (0..=n).map(|j| terminal(spot(n, j))).collect();
        for i in (0..n).rev() {
            let (p, q, df) = (self.p_up[i], 1.0 - self.p_up[i], self.df[i]);
            for j in 0..=i {
                let cont = df * (p * v[j + 1] + q * v[j]);
                v[j] = match exercise {
                    Some(ex) => ex(i, self.times[i], spot(i, j), cont),
                    None => cont,
                };
            }
        }
        v[0]
    }

    /// The same induction as [`price`](Self::price) (identical price, bit
    /// for bit) keeping the first two layers, so the value and the tree
    /// delta/gamma/theta come out of one pass. The log-grid is symmetric,
    /// so the second-layer center returns exactly to `s0` and theta needs
    /// no drift correction; the elapsed time is the grid's own `times[2]`
    /// (the layer times are unequal).
    pub fn price_with_greeks(
        &self,
        s0: f64,
        terminal: &dyn Fn(f64) -> f64,
        exercise: Option<&dyn Fn(usize, f64, f64, f64) -> f64>,
    ) -> LatticeSolution {
        let n = self.steps();
        let pow: Vec<f64> = (0..=2 * n).map(|k| ((k as f64 - n as f64) * self.dx).exp()).collect();
        let spot = |i: usize, j: usize| s0 * pow[2 * j + n - i];

        let mut v: Vec<f64> = (0..=n).map(|j| terminal(spot(n, j))).collect();
        let mut layer2 = [0.0; 3];
        let mut layer1 = [0.0; 2];
        if n == 2 {
            layer2.copy_from_slice(&v[0..3]);
        }
        for i in (0..n).rev() {
            let (p, q, df) = (self.p_up[i], 1.0 - self.p_up[i], self.df[i]);
            for j in 0..=i {
                let cont = df * (p * v[j + 1] + q * v[j]);
                v[j] = match exercise {
                    Some(ex) => ex(i, self.times[i], spot(i, j), cont),
                    None => cont,
                };
            }
            match i {
                2 => layer2.copy_from_slice(&v[0..3]),
                1 => layer1.copy_from_slice(&v[0..2]),
                _ => {}
            }
        }

        let price = v[0];
        let delta = (layer1[1] - layer1[0]) / (spot(1, 1) - spot(1, 0));
        let (s_uu, s_ud, s_dd) = (spot(2, 2), spot(2, 1), spot(2, 0));
        let d_up = (layer2[2] - layer2[1]) / (s_uu - s_ud);
        let d_down = (layer2[1] - layer2[0]) / (s_ud - s_dd);
        let gamma = (d_up - d_down) / (0.5 * (s_uu - s_dd));
        let theta = (layer2[1] - price) / self.times[2];
        LatticeSolution { price, delta, gamma, theta }
    }
}


// ── Trinomial lattice ───────────────────────────────────────────────────

/// One node's branching: the **middle child's absolute index** on the
/// next layer and the probabilities onto `(target+1, target, target-1)`.
///
/// A plain diffusion always targets its own index; a mean-reverting
/// short-rate tree (Hull-White) shifts the target at the edge nodes so
/// probabilities stay positive — the reason rates trees are trinomial.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrinomialBranch {
    pub target: i32,
    pub p_up: f64,
    pub p_mid: f64,
    pub p_down: f64,
}

/// A recombining trinomial lattice over the integer state grid
/// `x_j = j * dx` (the caller maps `j` to its own state, e.g.
/// `r(i, j) = alpha_i + j * dx` for a fitted short-rate tree).
///
/// Built from a per-node branching closure, so edge-switching trees
/// (Hull-White clamping) and plain diffusions use the same engine. The
/// per-layer index ranges follow from reachability. Discounting is
/// **per node** — `df(layer, j)` — because for fixed income the short
/// rate lives on the node; equity-style trees pass a constant.
#[derive(Debug, Clone)]
pub struct TrinomialLattice {
    pub dt: f64,
    pub dx: f64,
    j_min: Vec<i32>,
    j_max: Vec<i32>,
    /// `branches[i][j - j_min[i]]` for layers `0..n`.
    branches: Vec<Vec<TrinomialBranch>>,
}

impl TrinomialLattice {
    /// Build `n` steps of the lattice from the branching rule.
    /// Probabilities are validated per node; targets may shift by at most
    /// one index per step (`|target - j| <= 1`), which keeps the tree
    /// recombining.
    pub fn build(
        n: usize,
        dt: f64,
        dx: f64,
        branching: &dyn Fn(usize, i32) -> TrinomialBranch,
    ) -> Result<TrinomialLattice, RustyQLibError> {
        if n < 1 || !(dt > 0.0) || !(dx > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "trinomial",
                "the lattice needs at least one step and positive dt / dx",
            ));
        }
        let mut j_min = vec![0i32];
        let mut j_max = vec![0i32];
        let mut branches: Vec<Vec<TrinomialBranch>> = Vec::with_capacity(n);
        for i in 0..n {
            let (lo, hi) = (j_min[i], j_max[i]);
            let mut layer = Vec::with_capacity((hi - lo + 1) as usize);
            let (mut next_lo, mut next_hi) = (i32::MAX, i32::MIN);
            for j in lo..=hi {
                let b = branching(i, j);
                if (b.target - j).abs() > 1 {
                    return Err(RustyQLibError::NumericalError(format!(
                        "node ({i}, {j}) branches to target {} — more than one \
                         index away, which breaks recombination",
                        b.target
                    )));
                }
                for (name, prob) in
                    [("p_up", b.p_up), ("p_mid", b.p_mid), ("p_down", b.p_down)]
                {
                    if !(prob >= 0.0 && prob <= 1.0) {
                        return Err(RustyQLibError::NumericalError(format!(
                            "node ({i}, {j}): {name} = {prob:.6} outside [0, 1] \
                             (adjust the spacing or the branching rule)"
                        )));
                    }
                }
                if (b.p_up + b.p_mid + b.p_down - 1.0).abs() > 1e-9 {
                    return Err(RustyQLibError::NumericalError(format!(
                        "node ({i}, {j}): probabilities sum to {:.9}, not 1",
                        b.p_up + b.p_mid + b.p_down
                    )));
                }
                next_lo = next_lo.min(b.target - 1);
                next_hi = next_hi.max(b.target + 1);
                layer.push(b);
            }
            branches.push(layer);
            j_min.push(next_lo);
            j_max.push(next_hi);
        }
        Ok(TrinomialLattice { dt, dx, j_min, j_max, branches })
    }

    pub fn steps(&self) -> usize {
        self.branches.len()
    }

    /// Node index range `(j_min, j_max)` of a layer.
    pub fn layer_range(&self, i: usize) -> (i32, i32) {
        (self.j_min[i], self.j_max[i])
    }

    /// Backward induction. `node_df(i, j)` is the one-step discount at
    /// the node (state-dependent: `e^{-r(i,j) dt}` on a short-rate
    /// tree); `terminal(j)` values the final layer; `exercise` (when
    /// given) maps `(layer, j, continuation)` to the node value.
    pub fn price(
        &self,
        node_df: &dyn Fn(usize, i32) -> f64,
        terminal: &dyn Fn(i32) -> f64,
        exercise: Option<&dyn Fn(usize, i32, f64) -> f64>,
    ) -> f64 {
        let n = self.steps();
        let (lo_n, hi_n) = (self.j_min[n], self.j_max[n]);
        let mut values: Vec<f64> = (lo_n..=hi_n).map(terminal).collect();
        for i in (0..n).rev() {
            let (lo, hi) = (self.j_min[i], self.j_max[i]);
            let next_lo = self.j_min[i + 1];
            let mut layer = Vec::with_capacity((hi - lo + 1) as usize);
            for j in lo..=hi {
                let b = self.branches[i][(j - lo) as usize];
                let k = (b.target - next_lo) as usize;
                let expected = b.p_up * values[k + 1] + b.p_mid * values[k]
                    + b.p_down * values[k - 1];
                let cont = node_df(i, j) * expected;
                layer.push(match exercise {
                    Some(ex) => ex(i, j, cont),
                    None => cont,
                });
            }
            values = layer;
        }
        values[0]
    }

    /// Arrow-Debreu state prices by forward induction: `Q[i][j - j_min[i]]`
    /// is the value today of receiving 1 at node `(i, j)`. The workhorse of
    /// short-rate curve fitting — Hull-White's `alpha_i` shifts solve
    /// `sum_j Q[i][j] e^{-(alpha_i + j dx) dt} = P(0, t_{i+1})` layer by
    /// layer. `sum_j Q[n][j]` is the tree's discount factor to `t_n`.
    pub fn arrow_debreu(&self, node_df: &dyn Fn(usize, i32) -> f64) -> Vec<Vec<f64>> {
        let n = self.steps();
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
        q.push(vec![1.0]);
        for i in 0..n {
            let (lo, hi) = (self.j_min[i], self.j_max[i]);
            let (next_lo, next_hi) = (self.j_min[i + 1], self.j_max[i + 1]);
            let mut next = vec![0.0; (next_hi - next_lo + 1) as usize];
            for j in lo..=hi {
                let b = self.branches[i][(j - lo) as usize];
                let flow = q[i][(j - lo) as usize] * node_df(i, j);
                let k = (b.target - next_lo) as usize;
                next[k + 1] += b.p_up * flow;
                next[k] += b.p_mid * flow;
                next[k - 1] += b.p_down * flow;
            }
            q.push(next);
        }
        q
    }
}

/// Moment-matched branching for a constant-coefficient diffusion
/// `dx_t = nu dt + sigma dW` on spacing `dx` (Boyle / Kamrad-Ritchken:
/// `dx = sigma sqrt(3 dt)` gives the classic 1/6, 2/3, 1/6 weights at
/// zero drift). Same rule at every node — the equity-style tree.
pub fn diffusion_branching(nu: f64, sigma: f64, dt: f64, dx: f64) -> TrinomialBranch {
    let v = (sigma * sigma * dt + nu * nu * dt * dt) / (dx * dx);
    let m = nu * dt / dx;
    TrinomialBranch {
        target: 0, // filled per node by the caller closure (target = j)
        p_up: 0.5 * (v + m),
        p_mid: 1.0 - v,
        p_down: 0.5 * (v - m),
    }
}

/// Hull-White branching for the mean-reverting state
/// `dx_t = -a x_t dt + sigma dW` with `dx = sigma sqrt(3 dt)`:
/// standard branching in the interior, switching to downward branching at
/// `+j_cap` and upward at `-j_cap` so probabilities stay positive
/// (Hull's `j_max = ceil(0.184 / (a dt))` is the usual cap).
pub fn hull_white_branching(a: f64, dt: f64, j: i32, j_cap: i32) -> TrinomialBranch {
    let ajdt = a * j as f64 * dt;
    let ajdt2 = ajdt * ajdt;
    if j >= j_cap {
        // downward branching: children (j, j-1, j-2)
        TrinomialBranch {
            target: j - 1,
            p_up: 7.0 / 6.0 + 0.5 * (ajdt2 - 3.0 * ajdt),
            p_mid: -1.0 / 3.0 - ajdt2 + 2.0 * ajdt,
            p_down: 1.0 / 6.0 + 0.5 * (ajdt2 - ajdt),
        }
    } else if j <= -j_cap {
        // upward branching: children (j+2, j+1, j)
        TrinomialBranch {
            target: j + 1,
            p_up: 1.0 / 6.0 + 0.5 * (ajdt2 + ajdt),
            p_mid: -1.0 / 3.0 - ajdt2 - 2.0 * ajdt,
            p_down: 7.0 / 6.0 + 0.5 * (ajdt2 + 3.0 * ajdt),
        }
    } else {
        TrinomialBranch {
            target: j,
            p_up: 1.0 / 6.0 + 0.5 * (ajdt2 - ajdt),
            p_mid: 2.0 / 3.0 - ajdt2,
            p_down: 1.0 / 6.0 + 0.5 * (ajdt2 + ajdt),
        }
    }
}

/// Hull's recommended clamp for [`hull_white_branching`].
pub fn hull_white_j_cap(a: f64, dt: f64) -> i32 {
    (0.184 / (a * dt)).ceil() as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::norm_cdf;

    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIGMA: f64 = 0.3;
    const T: f64 = 1.0;

    fn bs_call() -> f64 {
        let b = R - Q;
        let d1 = ((S / K).ln() + (b + 0.5 * SIGMA * SIGMA) * T) / (SIGMA * T.sqrt());
        let d2 = d1 - SIGMA * T.sqrt();
        S * ((b - R) * T).exp() * norm_cdf(d1) - K * (-R * T).exp() * norm_cdf(d2)
    }

    fn tree_call(tree_type: BinomialTreeType, steps: usize) -> f64 {
        let n = tree_type.effective_steps(steps);
        let params = tree_type.params(S, K, R - Q, SIGMA, T, n).unwrap();
        let df = (-R * T / n as f64).exp();
        price_backward(S, &params, n, df, &|s| (s - K).max(0.0), None)
    }

    fn all_types() -> [BinomialTreeType; 6] {
        use BinomialTreeType::*;
        [CoxRossRubinstein, JarrowRudd, Tian, Trigeorgis, LeisenReimer, AdditiveEqp]
    }

    #[test]
    fn every_scheme_converges_to_black_scholes() {
        let reference = bs_call();
        for tree_type in all_types() {
            let price = tree_call(tree_type, 1000);
            // EQP converges at O(sqrt(dt)) — the known laggard, kept for
            // completeness (same formulation as QuantLib's AdditiveEQP)
            let tol = if tree_type == BinomialTreeType::AdditiveEqp { 2e-2 } else { 5e-3 };
            assert!(
                (price - reference).abs() < tol,
                "{tree_type:?}: {price} vs BS {reference}"
            );
        }
    }

    #[test]
    fn leisen_reimer_beats_crr_by_orders_of_magnitude() {
        let reference = bs_call();
        let lr_err = (tree_call(BinomialTreeType::LeisenReimer, 101) - reference).abs();
        let crr_err = (tree_call(BinomialTreeType::CoxRossRubinstein, 101) - reference).abs();
        assert!(
            lr_err < 1e-4,
            "LR at 101 steps is second order, err {lr_err}"
        );
        assert!(
            lr_err * 10.0 < crr_err,
            "LR(101) err {lr_err} must be >10x tighter than CRR(101) err {crr_err}"
        );
    }

    #[test]
    fn leisen_reimer_bumps_even_step_counts_to_odd() {
        assert_eq!(BinomialTreeType::LeisenReimer.effective_steps(100), 101);
        assert_eq!(BinomialTreeType::LeisenReimer.effective_steps(101), 101);
        assert_eq!(BinomialTreeType::CoxRossRubinstein.effective_steps(100), 100);
    }

    #[test]
    fn optimized_and_diagnostic_engines_agree_exactly() {
        for tree_type in all_types() {
            let n = tree_type.effective_steps(200);
            let params = tree_type.params(S, K, R - Q, SIGMA, T, n).unwrap();
            let dt = T / n as f64;
            let df = (-R * dt).exp();
            let terminal = |s: f64| (K - s).max(0.0);
            let exercise = |_: usize, s: f64, cont: f64| (K - s).max(0.0).max(cont);
            let fast = price_backward(S, &params, n, df, &terminal, Some(&exercise));
            let diag = price_with_diagnostics(
                tree_type, S, &params, n, dt, df, &terminal, Some(&exercise),
            );
            assert_eq!(fast, diag.price, "{tree_type:?} engines disagree");
            assert_eq!(diag.spot_tree.len(), n + 1);
            assert_eq!(diag.spot_tree[n].len(), n + 1);
        }
    }

    #[test]
    fn one_pass_greeks_match_black_scholes() {
        let norm_pdf = |x: f64| (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let b = R - Q;
        let sq_t = SIGMA * T.sqrt();
        let d1 = ((S / K).ln() + (b + 0.5 * SIGMA * SIGMA) * T) / sq_t;
        let d2 = d1 - sq_t;
        let carry_df = ((b - R) * T).exp(); // e^{-qT}
        let bs_delta = carry_df * norm_cdf(d1);
        let bs_gamma = carry_df * norm_pdf(d1) / (S * sq_t);
        let bs_theta = -S * carry_df * norm_pdf(d1) * SIGMA / (2.0 * T.sqrt())
            - (b - R) * S * carry_df * norm_cdf(d1)
            - R * K * (-R * T).exp() * norm_cdf(d2);

        // the asymmetric schemes exercise the drift-corrected theta read
        for tree_type in [BinomialTreeType::LeisenReimer, BinomialTreeType::JarrowRudd] {
            let n = tree_type.effective_steps(1001);
            let params = tree_type.params(S, K, b, SIGMA, T, n).unwrap();
            let dt = T / n as f64;
            let df = (-R * dt).exp();
            let terminal = |s: f64| (s - K).max(0.0);
            let sol = price_backward_with_greeks(S, &params, n, dt, df, &terminal, None);
            let plain = price_backward(S, &params, n, df, &terminal, None);
            assert_eq!(sol.price, plain, "{tree_type:?}: one-pass price must match exactly");
            assert!(
                (sol.delta - bs_delta).abs() < 2e-3,
                "{tree_type:?} delta {} vs BS {bs_delta}",
                sol.delta
            );
            assert!(
                (sol.gamma - bs_gamma).abs() < 2e-4,
                "{tree_type:?} gamma {} vs BS {bs_gamma}",
                sol.gamma
            );
            assert!(
                (sol.theta - bs_theta).abs() < 2e-2,
                "{tree_type:?} theta {} vs BS {bs_theta}",
                sol.theta
            );
        }
    }

    #[test]
    fn one_pass_greeks_agree_with_the_diagnostic_engine() {
        // American put: same induction, so delta and gamma must be identical
        let tree_type = BinomialTreeType::CoxRossRubinstein;
        let n = 200;
        let params = tree_type.params(S, K, R - Q, SIGMA, T, n).unwrap();
        let dt = T / n as f64;
        let df = (-R * dt).exp();
        let terminal = |s: f64| (K - s).max(0.0);
        let exercise = |_: usize, s: f64, cont: f64| (K - s).max(0.0).max(cont);
        let sol =
            price_backward_with_greeks(S, &params, n, dt, df, &terminal, Some(&exercise));
        let diag = price_with_diagnostics(
            tree_type, S, &params, n, dt, df, &terminal, Some(&exercise),
        );
        assert_eq!(sol.price, diag.price);
        assert_eq!(sol.delta, diag.delta);
        assert_eq!(sol.gamma, diag.gamma);
        // CRR is symmetric (the layer-2 center returns to s0), so the
        // drift correction vanishes and the thetas coincide
        assert!((sol.theta - diag.theta).abs() < 1e-10, "{} vs {}", sol.theta, diag.theta);
    }

    #[test]
    fn american_put_diagnostics_show_the_exercise_region() {
        let tree_type = BinomialTreeType::LeisenReimer;
        let n = tree_type.effective_steps(201);
        let params = tree_type.params(S, K, R - Q, SIGMA, T, n).unwrap();
        let dt = T / n as f64;
        let df = (-R * dt).exp();
        let terminal = |s: f64| (K - s).max(0.0);
        let exercise = |_: usize, s: f64, cont: f64| (K - s).max(0.0).max(cont);
        let diag = price_with_diagnostics(
            tree_type, S, &params, n, dt, df, &terminal, Some(&exercise),
        );
        // an American put on a dividend payer exercises early somewhere
        let exercised_layers = diag.exercise_boundary.iter().flatten().count();
        assert!(exercised_layers > 0, "the put must have an exercise region");
        // the boundary lies below the strike and its max is below spot levels
        for (lo, hi) in diag.exercise_boundary.iter().flatten() {
            assert!(*lo <= *hi && *hi < K);
        }
        // early exercise premium over the European put
        let euro = price_backward(S, &params, n, df, &terminal, None);
        assert!(diag.price > euro + 1e-4, "american {} european {euro}", diag.price);
        // tree Greeks are sane for an ATM put
        assert!(diag.delta > -1.0 && diag.delta < 0.0);
        assert!(diag.gamma > 0.0);
        assert!(diag.theta < 0.0);
        assert!(diag.elapsed > Duration::ZERO);
    }

    #[test]
    fn convergence_ladder_reports_prices_and_timing() {
        let terminal = |s: f64| (s - K).max(0.0);
        let ladder =
            convergence_study(
                BinomialTreeType::LeisenReimer, S, K, R - Q, SIGMA, R, T,
                &[25, 51, 101, 201], &terminal, None,
            )
            .unwrap();
        assert_eq!(ladder.len(), 4);
        let reference = bs_call();
        let errors: Vec<f64> = ladder.iter().map(|p| (p.price - reference).abs()).collect();
        // LR converges monotonically (smoothly) for the vanilla
        assert!(
            errors.windows(2).all(|w| w[1] <= w[0] * 1.5),
            "LR errors should shrink along the ladder: {errors:?}"
        );
        assert!(errors[3] < 2e-5, "LR(201) err {}", errors[3]);
    }

    #[test]
    fn degenerate_steps_are_rejected_with_typed_errors() {
        // enormous drift with one coarse step: probability leaves (0, 1)
        let r = BinomialTreeType::CoxRossRubinstein.params(100.0, 100.0, 2.0, 0.05, 1.0, 2);
        assert!(matches!(r, Err(RustyQLibError::NumericalError(_))), "{r:?}");
        // EQP discriminant failure on the same setup
        let r = BinomialTreeType::AdditiveEqp.params(100.0, 100.0, 2.0, 0.05, 1.0, 2);
        assert!(matches!(r, Err(RustyQLibError::NumericalError(_))), "{r:?}");
        // invalid vol
        let r = BinomialTreeType::Tian.params(100.0, 100.0, 0.03, -0.1, 1.0, 100);
        assert!(matches!(r, Err(RustyQLibError::InvalidInput { .. })));
    }

    #[test]
    fn trinomial_diffusion_converges_to_black_scholes() {
        // GBM in log space: nu = b - sigma^2/2, terminal S = S0 e^{j dx}
        let nu = (R - Q) - 0.5 * SIGMA * SIGMA;
        let price_at = |n: usize| {
            let dt = T / n as f64;
            let dx = SIGMA * (3.0 * dt).sqrt();
            let proto = diffusion_branching(nu, SIGMA, dt, dx);
            let branching = |_: usize, j: i32| TrinomialBranch { target: j, ..proto };
            let lattice = TrinomialLattice::build(n, dt, dx, &branching).unwrap();
            let df = (-R * dt).exp();
            (
                lattice.price(&|_, _| df, &|j| (S * (j as f64 * dx).exp() - K).max(0.0), None),
                dx,
                lattice,
                df,
            )
        };
        let reference = bs_call();
        let coarse_err = (price_at(250).0 - reference).abs();
        let (call, dx, lattice, df) = price_at(1000);
        let fine_err = (call - reference).abs();
        assert!(fine_err < 5e-3, "trinomial {call} vs BS {reference}");
        // first-order convergence: quadrupling the steps shrinks the error
        assert!(
            fine_err < coarse_err,
            "error must shrink with steps: {fine_err} vs {coarse_err}"
        );
        let n = 1000usize;
        let _ = n;

        // American put dominates European on the same tree
        let terminal_put = |j: i32| (K - S * (j as f64 * dx).exp()).max(0.0);
        let euro = lattice.price(&|_, _| df, &terminal_put, None);
        let ex = |_: usize, j: i32, cont: f64| {
            (K - S * (j as f64 * dx).exp()).max(0.0).max(cont)
        };
        let amer = lattice.price(&|_, _| df, &terminal_put, Some(&ex));
        assert!(amer >= euro - 1e-12, "american {amer} vs european {euro}");
    }

    #[test]
    fn arrow_debreu_prices_sum_to_the_discount_factor() {
        let n = 100;
        let dt = 0.01;
        let dx = 0.2 * (3.0_f64 * dt).sqrt(); // Kamrad-Ritchken spacing
        let proto = diffusion_branching(0.0, 0.2, dt, dx);
        let branching = |_: usize, j: i32| TrinomialBranch { target: j, ..proto };
        let lattice = TrinomialLattice::build(n, dt, dx, &branching).unwrap();
        let df = (-0.05_f64 * dt).exp();
        let q = lattice.arrow_debreu(&|_, _| df);
        // sum of state prices at layer i = P(0, t_i) under a flat rate
        for i in [1usize, 50, 100] {
            let total: f64 = q[i].iter().sum();
            let expected = (-0.05 * i as f64 * dt).exp();
            assert!(
                (total - expected).abs() < 1e-12,
                "layer {i}: sum {total} vs df {expected}"
            );
        }
    }

    #[test]
    fn hull_white_tree_reprices_the_vasicek_bond() {
        // Vasicek with b = 0, r0 = 0: dr = -a r dt + sigma dW. The tree
        // state IS the short rate; per-node discounting e^{-r(i,j) dt}.
        // Closed form: P(0,T) = exp(sigma^2 (T - B)/(2 a^2) - sigma^2 B^2/(4a)),
        // B = (1 - e^{-aT})/a.
        let (a, sigma, t_mat) = (0.10, 0.015, 5.0);
        let n = 500;
        let dt = t_mat / n as f64;
        let dx = sigma * (3.0 * dt).sqrt();
        let cap = hull_white_j_cap(a, dt);
        let branching = |_: usize, j: i32| hull_white_branching(a, dt, j, cap);
        let lattice = TrinomialLattice::build(n, dt, dx, &branching).unwrap();
        // the clamp must actually bite for the edge formulas to be used
        assert!(lattice.layer_range(n).1 == cap, "edge branching untested");
        let node_df = |_: usize, j: i32| (-(j as f64 * dx) * dt).exp();
        let tree_price = lattice.price(&node_df, &|_| 1.0, None);

        let b_t = (1.0 - (-a * t_mat).exp()) / a;
        let closed_form = (sigma * sigma * (t_mat - b_t) / (2.0 * a * a)
            - sigma * sigma * b_t * b_t / (4.0 * a))
            .exp();
        let rel_err = (tree_price - closed_form).abs() / closed_form;
        assert!(
            rel_err < 1e-3,
            "tree {tree_price} vs Vasicek {closed_form} (rel err {rel_err:.2e})"
        );

        // Arrow-Debreu consistency: state prices reprice the same bond
        let q = lattice.arrow_debreu(&node_df);
        let via_q: f64 = q[n].iter().sum();
        assert!((via_q - tree_price).abs() < 1e-12);
    }

    #[test]
    fn trinomial_rejects_invalid_branching() {
        // fat drift on a coarse grid: probabilities leave [0, 1]
        let proto = diffusion_branching(5.0, 0.05, 0.5, 0.02);
        let branching = |_: usize, j: i32| TrinomialBranch { target: j, ..proto };
        let r = TrinomialLattice::build(4, 0.5, 0.02, &branching);
        assert!(matches!(r, Err(RustyQLibError::NumericalError(_))), "{r:?}");
        // non-recombining target shift
        let jumpy = |_: usize, j: i32| TrinomialBranch {
            target: j + 2,
            p_up: 1.0 / 6.0,
            p_mid: 2.0 / 3.0,
            p_down: 1.0 / 6.0,
        };
        let r = TrinomialLattice::build(4, 0.01, 0.02, &jumpy);
        assert!(matches!(r, Err(RustyQLibError::NumericalError(_))), "{r:?}");
    }

    #[test]
    fn term_lattice_with_flat_inputs_matches_the_uniform_tree() {
        let n = 500;
        let flat_rate = |_: f64, _: f64| R;
        let flat_carry = |_: f64, _: f64| Q;
        let flat_var = |t: f64| SIGMA * SIGMA * t;
        let lattice = TermLattice::build(n, T, &flat_rate, &flat_carry, &flat_var).unwrap();
        let term = lattice.price(S, &|s| (s - K).max(0.0), None);
        // flat inputs reduce to the CRR tree
        let params = BinomialTreeType::CoxRossRubinstein.params(S, K, R - Q, SIGMA, T, n).unwrap();
        let uniform =
            price_backward(S, &params, n, (-R * T / n as f64).exp(), &|s| (s - K).max(0.0), None);
        assert!((term - uniform).abs() < 1e-9, "term {term} vs uniform {uniform}");
    }

    #[test]
    fn time_dependent_vol_prices_to_the_equivalent_total_variance() {
        // two vol regimes: 20% for the first half-year, 40% after; a
        // European option only sees the total variance, so the tree must
        // converge to BS at the equivalent vol sqrt((0.2^2 + 0.4^2)/2)
        let (s1, s2) = (0.2, 0.4);
        let var = move |t: f64| {
            if t <= 0.5 { s1 * s1 * t } else { s1 * s1 * 0.5 + s2 * s2 * (t - 0.5) }
        };
        let sigma_eq = (0.5f64 * (s1 * s1 + s2 * s2)).sqrt();
        let flat_rate = |_: f64, _: f64| R;
        let flat_carry = |_: f64, _: f64| Q;
        let lattice = TermLattice::build(2000, T, &flat_rate, &flat_carry, &var).unwrap();
        let term = lattice.price(S, &|s| (s - K).max(0.0), None);
        let b = R - Q;
        let d1 = ((S / K).ln() + (b + 0.5 * sigma_eq * sigma_eq) * T) / (sigma_eq * T.sqrt());
        let d2 = d1 - sigma_eq * T.sqrt();
        let bs = S * ((b - R) * T).exp() * norm_cdf(d1) - K * (-R * T).exp() * norm_cdf(d2);
        assert!((term - bs).abs() < 5e-3, "term {term} vs BS(sigma_eq) {bs}");
    }

    #[test]
    fn time_dependent_rates_discount_and_drift_exactly() {
        // stepwise forward rates: 2% then 8%; the European price must
        // match BS with the average rate 5% (exact discounting + drift)
        let fwd = |t1: f64, t2: f64| {
            let integral = |t: f64| {
                if t <= 0.5 { 0.02 * t } else { 0.02 * 0.5 + 0.08 * (t - 0.5) }
            };
            (integral(t2) - integral(t1)) / (t2 - t1)
        };
        let flat_carry = |_: f64, _: f64| Q;
        let var = |t: f64| SIGMA * SIGMA * t;
        let lattice = TermLattice::build(2000, T, &fwd, &flat_carry, &var).unwrap();
        let term = lattice.price(S, &|s| (s - K).max(0.0), None);
        let r_eq = 0.05;
        let b = r_eq - Q;
        let d1 = ((S / K).ln() + (b + 0.5 * SIGMA * SIGMA) * T) / (SIGMA * T.sqrt());
        let d2 = d1 - SIGMA * T.sqrt();
        let bs = S * ((b - r_eq) * T).exp() * norm_cdf(d1) - K * (-r_eq * T).exp() * norm_cdf(d2);
        assert!((term - bs).abs() < 5e-3, "term {term} vs BS(r_eq) {bs}");

        // American exercise still works and dominates European
        let exercise = |_: usize, _: f64, s: f64, cont: f64| (K - s).max(0.0).max(cont);
        let amer = lattice.price(S, &|s| (K - s).max(0.0), Some(&exercise));
        let euro = lattice.price(S, &|s| (K - s).max(0.0), None);
        assert!(amer >= euro - 1e-12, "american {amer} vs european {euro}");
    }

    #[test]
    fn term_lattice_rejects_bad_term_structures() {
        let flat_rate = |_: f64, _: f64| R;
        let flat_carry = |_: f64, _: f64| Q;
        // decreasing total variance = calendar arbitrage
        let bad_var = |t: f64| 0.09 * (1.0 - t).max(0.01);
        let r = TermLattice::build(100, T, &flat_rate, &flat_carry, &bad_var);
        assert!(r.is_err(), "{r:?}");
        // giant drift on a coarse grid: probability leaves (0, 1)
        let big_rate = |_: f64, _: f64| 5.0;
        let var = |t: f64| 0.01 * t;
        let r = TermLattice::build(4, T, &big_rate, &flat_carry, &var);
        assert!(matches!(r, Err(RustyQLibError::NumericalError(_))), "{r:?}");
    }

    #[test]
    fn tree_type_parses_from_contract_strings() {
        use std::str::FromStr;
        for (s, expected) in [
            ("CRR", BinomialTreeType::CoxRossRubinstein),
            ("LeisenReimer", BinomialTreeType::LeisenReimer),
            ("lr", BinomialTreeType::LeisenReimer),
            ("jarrow_rudd", BinomialTreeType::JarrowRudd),
            ("Tian", BinomialTreeType::Tian),
            ("trigeorgis", BinomialTreeType::Trigeorgis),
            ("EQP", BinomialTreeType::AdditiveEqp),
        ] {
            assert_eq!(BinomialTreeType::from_str(s).unwrap(), expected);
        }
        assert!(BinomialTreeType::from_str("no_such_tree").is_err());
    }
}
