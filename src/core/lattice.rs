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
//! Two engines share the parameterization:
//! - [`price_backward`] — the production engine: a rolling one-dimensional
//!   value array (O(n) memory, no tree materialized) with the layer spot
//!   levels rebuilt from two precomputed power tables.
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
