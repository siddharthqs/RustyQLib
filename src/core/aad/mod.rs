//! Adjoint Algorithmic Differentiation (AAD): tape-based reverse-mode
//! differentiation for pricing code.
//!
//! Write a pricer over [`Var`] instead of `f64` — the operator
//! overloading records every operation on a [`Tape`] — and one backward
//! sweep ([`Var::grad`]) returns the sensitivity of the output to
//! **every** input at once, at a fixed small multiple of the pricing
//! cost. That is the AAD trade against bump-and-reprice: bumping costs
//! one full reprice *per input*, the adjoint sweep costs ~one reprice
//! *total*, however many inputs there are — the difference between
//! seconds and hours on a book with per-pillar curve and surface
//! sensitivities.
//!
//! Two worked and tested quant applications live in this module's tests:
//!
//! - [`black_scholes`]: the closed form written over `Var`; a single
//!   sweep produces delta, dual delta, rho, carry rho, vega and the
//!   maturity sensitivity simultaneously, matching the library's
//!   closed-form Greeks to near machine precision;
//! - **pathwise Monte Carlo Greeks**: differentiate through the
//!   simulation itself (one small tape per path), giving delta, vega
//!   and rho from the same paths that price — the standard pathwise
//!   estimator, validated against the closed forms.
//!
//! The `max(x, 0)` payoff kink is handled by the almost-everywhere
//! derivative ([`Var::maxf`]), which is exactly the classical pathwise
//! estimator's requirement (fine for vanillas and smooth-density
//! payoffs; digitals need smoothing or likelihood-ratio methods).

pub mod tape;
pub mod var;

pub use tape::{Gradients, Tape};
pub use var::Var;

use crate::core::trade::PutOrCall;

/// Black-Scholes price recorded on the tape: differentiate to get every
/// first-order Greek from one backward sweep.
pub fn black_scholes<'a>(
    s: Var<'a>,
    k: Var<'a>,
    r: Var<'a>,
    q: Var<'a>,
    sigma: Var<'a>,
    t: Var<'a>,
    put_or_call: PutOrCall,
) -> Var<'a> {
    let sqrt_t = t.sqrt();
    let st = sigma * sqrt_t;
    let d1 = ((s / k).ln() + (r - q + sigma * sigma * 0.5) * t) / st;
    let d2 = d1 - st;
    let df_q = (-(q * t)).exp();
    let df_r = (-(r * t)).exp();
    match put_or_call {
        PutOrCall::Call => s * df_q * d1.norm_cdf() - k * df_r * d2.norm_cdf(),
        PutOrCall::Put => k * df_r * (-d2).norm_cdf() - s * df_q * (-d1).norm_cdf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::{bs_price, bs_vega};

    const S: f64 = 100.0;
    const K: f64 = 105.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIG: f64 = 0.3;
    const T: f64 = 1.0;

    #[test]
    fn one_sweep_reproduces_every_black_scholes_greek() {
        let tape = Tape::new();
        let (s, k, r, q, sigma, t) =
            (tape.var(S), tape.var(K), tape.var(R), tape.var(Q), tape.var(SIG), tape.var(T));
        let price = black_scholes(s, k, r, q, sigma, t, PutOrCall::Call);
        assert!((price.value() - bs_price(S, K, R, Q, SIG, T, PutOrCall::Call)).abs() < 1e-12);

        // ONE backward pass: six sensitivities
        let g = price.grad();

        // vega against the library closed form
        assert!((g.wrt(sigma) - bs_vega(S, K, R, Q, SIG, T)).abs() < 1e-10, "vega");
        // the rest against tight central differences of bs_price
        let h = 1e-6;
        let fd = |f: &dyn Fn(f64) -> f64| (f(h) - f(-h)) / (2.0 * h);
        let cases: [(f64, Box<dyn Fn(f64) -> f64>); 5] = [
            (g.wrt(s), Box::new(|e| bs_price(S + e, K, R, Q, SIG, T, PutOrCall::Call))),
            (g.wrt(k), Box::new(|e| bs_price(S, K + e, R, Q, SIG, T, PutOrCall::Call))),
            (g.wrt(r), Box::new(|e| bs_price(S, K, R + e, Q, SIG, T, PutOrCall::Call))),
            (g.wrt(q), Box::new(|e| bs_price(S, K, R, Q + e, SIG, T, PutOrCall::Call))),
            (g.wrt(t), Box::new(|e| bs_price(S, K, R, Q, SIG, T + e, PutOrCall::Call))),
        ];
        for (i, (aad, f)) in cases.iter().enumerate() {
            let numeric = fd(f);
            assert!((aad - numeric).abs() < 1e-7, "greek {i}: aad {aad} vs fd {numeric}");
        }
        // put side too
        let put = black_scholes(s, k, r, q, sigma, t, PutOrCall::Put);
        let gp = put.grad();
        let put_delta_fd =
            (bs_price(S + h, K, R, Q, SIG, T, PutOrCall::Put)
                - bs_price(S - h, K, R, Q, SIG, T, PutOrCall::Put))
                / (2.0 * h);
        assert!((gp.wrt(s) - put_delta_fd).abs() < 1e-7);
    }

    #[test]
    fn pathwise_monte_carlo_greeks_from_one_sweep_per_path() {
        // differentiate straight through a GBM simulation: delta, vega
        // and rho of a European call from the same paths that price it
        use crate::core::montecarlo::path_rng;
        use rand::Rng;

        let n_paths = 40_000;
        // accumulate mean and variance of each estimator so the
        // assertions are proper statistical bands, not magic numbers
        let mut acc = [[0.0f64; 2]; 4]; // [sum, sum_sq] x {price, delta, vega, rho}
        for i in 0..n_paths {
            let mut rng = path_rng(2026, i);
            let z: f64 = rng.sample(rand_distr::StandardNormal);
            let tape = Tape::new();
            let s0 = tape.var(S);
            let sigma = tape.var(SIG);
            let r = tape.var(R);
            let drift = (r - Q - sigma * sigma * 0.5) * T;
            let s_t = s0 * (drift + sigma * (T.sqrt() * z)).exp();
            let payoff = (s_t - K).maxf(0.0) * (-(r * T)).exp();
            let g = payoff.grad();
            for (slot, x) in
                [payoff.value(), g.wrt(s0), g.wrt(sigma), g.wrt(r)].into_iter().enumerate()
            {
                acc[slot][0] += x;
                acc[slot][1] += x * x;
            }
        }
        let n = n_paths as f64;
        let stats = |slot: usize| -> (f64, f64) {
            let mean = acc[slot][0] / n;
            let var = (acc[slot][1] / n - mean * mean).max(0.0);
            (mean, (var / n).sqrt())
        };
        let h = 1e-5;
        let bs = |s: f64, sig: f64, r: f64| bs_price(s, K, r, Q, sig, T, PutOrCall::Call);
        let truths = [
            bs(S, SIG, R),
            (bs(S + h, SIG, R) - bs(S - h, SIG, R)) / (2.0 * h),
            bs_vega(S, K, R, Q, SIG, T),
            (bs(S, SIG, R + h) - bs(S, SIG, R - h)) / (2.0 * h),
        ];
        for (slot, name) in ["price", "delta", "vega", "rho"].iter().enumerate() {
            let (mean, se) = stats(slot);
            assert!(
                (mean - truths[slot]).abs() < 4.0 * se + 1e-10,
                "{name}: {mean} vs {} (se {se})",
                truths[slot]
            );
        }
    }

    #[test]
    fn tape_cost_is_a_small_constant_multiple_of_pricing() {
        // the adjoint promise: node count (a proxy for work) does not
        // grow with the number of sensitivities requested
        let tape = Tape::new();
        let (s, k, r, q, sigma, t) =
            (tape.var(S), tape.var(K), tape.var(R), tape.var(Q), tape.var(SIG), tape.var(T));
        let price = black_scholes(s, k, r, q, sigma, t, PutOrCall::Call);
        let nodes = tape.len();
        assert!(nodes < 60, "tape has {nodes} nodes");
        // one sweep serves all six inputs
        let g = price.grad();
        let six = [g.wrt(s), g.wrt(k), g.wrt(r), g.wrt(q), g.wrt(sigma), g.wrt(t)];
        assert!(six.iter().all(|x| x.is_finite() && *x != 0.0));
    }
}
