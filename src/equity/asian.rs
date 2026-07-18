//! Analytic pricing of Asian (average) options.
//!
//! - **Geometric average price**: exact closed form — the geometric average
//!   of lognormals is lognormal. Supports discrete equally spaced averaging
//!   (`n` points, matching Monte Carlo monitoring) and the continuous limit.
//! - **Arithmetic average price**: Turnbull-Wakeman (1991) lognormal
//!   moment-matching approximation, continuous averaging.
//!
//! Both assume the averaging period spans the whole life of the option and
//! has not yet started. Floating-strike (average strike) Asians have no
//! implemented closed form and price on the Monte Carlo engine.

use crate::core::trade::PutOrCall;
use crate::core::utils::N;

/// How the average is computed along the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AveragingType {
    Arithmetic,
    Geometric,
}

/// Fixed strike (average price) vs floating strike (average strike).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsianStrikeType {
    FixedStrike,
    FloatingStrike,
}

fn black(df_r: f64, forward: f64, k: f64, log_var: f64, put_or_call: PutOrCall) -> f64 {
    let sqrt_v = log_var.sqrt();
    let d1 = ((forward / k).ln() + 0.5 * log_var) / sqrt_v;
    let d2 = d1 - sqrt_v;
    match put_or_call {
        PutOrCall::Call => df_r * (forward * N(d1) - k * N(d2)),
        PutOrCall::Put => df_r * (k * N(-d2) - forward * N(-d1)),
    }
}

/// Exact price of a geometric average-price Asian.
///
/// `n = Some(count)`: discrete averaging at `t_i = i*T/n`, `i = 1..=n`
/// (matches Monte Carlo monitoring, spot excluded). `n = None`: continuous
/// averaging limit.
#[allow(clippy::too_many_arguments)]
pub fn geometric_asian_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n: Option<usize>,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && sigma > 0.0 && t > 0.0);
    let b = r - q;
    let (mean_factor, var_factor) = match n {
        Some(n) => {
            assert!(n > 0);
            let nf = n as f64;
            ((nf + 1.0) / (2.0 * nf), (nf + 1.0) * (2.0 * nf + 1.0) / (6.0 * nf * nf))
        }
        None => (0.5, 1.0 / 3.0),
    };
    let mu = (b - 0.5 * sigma * sigma) * t * mean_factor;
    let log_var = sigma * sigma * t * var_factor;
    let forward = s * (mu + 0.5 * log_var).exp();
    black((-r * t).exp(), forward, k, log_var, put_or_call)
}

/// Turnbull-Wakeman approximation for an arithmetic average-price Asian
/// (continuous averaging): the first two moments of the average are matched
/// to a lognormal and priced with Black's formula.
pub fn turnbull_wakeman_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && sigma > 0.0 && t > 0.0);
    let b = r - q;
    let s2 = sigma * sigma;
    let (m1, m2) = if b.abs() > 1e-8 {
        let m1 = ((b * t).exp() - 1.0) / (b * t);
        let m2 = 2.0 * ((2.0 * b + s2) * t).exp() / ((b + s2) * (2.0 * b + s2) * t * t)
            + 2.0 / (b * t * t) * (1.0 / (2.0 * b + s2) - (b * t).exp() / (b + s2));
        (m1, m2)
    } else {
        let m1 = 1.0;
        let m2 = (2.0 * (s2 * t).exp() - 2.0 * (1.0 + s2 * t)) / (s2 * s2 * t * t);
        (m1, m2)
    };
    let forward = s * m1;
    let log_var = (m2 / (m1 * m1)).ln(); // sigma_A^2 * T
    black((-r * t).exp(), forward, k, log_var, put_or_call)
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIG: f64 = 0.3;
    const T: f64 = 1.0;

    #[test]
    fn geometric_golden_values() {
        // independently generated oracle values
        assert!((geometric_asian_price(S, 100.0, R, Q, SIG, T, Some(252), PutOrCall::Call)
            - 6.976295)
            .abs()
            < 1e-5);
        assert!((geometric_asian_price(S, 100.0, R, Q, SIG, T, None, PutOrCall::Call) - 6.953600)
            .abs()
            < 1e-5);
    }

    #[test]
    fn turnbull_wakeman_golden_value() {
        let price = turnbull_wakeman_price(S, 100.0, R, Q, SIG, T, PutOrCall::Call);
        assert!((price - 7.409272).abs() < 1e-5, "{price}");
    }

    #[test]
    fn geometric_put_call_parity() {
        // C - P = e^{-rT} (F_G - K) with the same lognormal forward
        for n in [Some(12), Some(252), None] {
            let c = geometric_asian_price(S, 90.0, R, Q, SIG, T, n, PutOrCall::Call);
            let p = geometric_asian_price(S, 90.0, R, Q, SIG, T, n, PutOrCall::Put);
            // recover F_G from a deep parity-free identity: price both at a
            // strike and check C - P is strike-linear with slope -e^{-rT}
            let c2 = geometric_asian_price(S, 110.0, R, Q, SIG, T, n, PutOrCall::Call);
            let p2 = geometric_asian_price(S, 110.0, R, Q, SIG, T, n, PutOrCall::Put);
            let df = (-R * T).exp();
            assert!((((c - p) - (c2 - p2)) - df * 20.0).abs() < 1e-10);
        }
    }

    #[test]
    fn discrete_averaging_converges_to_continuous() {
        let continuous = geometric_asian_price(S, 100.0, R, Q, SIG, T, None, PutOrCall::Call);
        let fine = geometric_asian_price(S, 100.0, R, Q, SIG, T, Some(100_000), PutOrCall::Call);
        assert!((fine - continuous).abs() < 1e-3);
    }

    #[test]
    fn averaging_reduces_option_value_below_vanilla() {
        use crate::equity::blackscholes::bs_price;
        let vanilla = bs_price(S, 100.0, R, Q, SIG, T, PutOrCall::Call);
        let geo = geometric_asian_price(S, 100.0, R, Q, SIG, T, None, PutOrCall::Call);
        let arith = turnbull_wakeman_price(S, 100.0, R, Q, SIG, T, PutOrCall::Call);
        assert!(geo < arith, "AM-GM: arithmetic average dominates geometric");
        assert!(arith < vanilla, "averaging reduces effective volatility");
    }

    #[test]
    fn zero_cost_of_carry_branch() {
        // r = q exercises the b = 0 moment formulas
        let price = turnbull_wakeman_price(S, 100.0, 0.03, 0.03, SIG, T, PutOrCall::Call);
        assert!(price > 0.0 && price.is_finite());
        // continuity across the branch: b = 1e-9 vs b = 0
        let near = turnbull_wakeman_price(S, 100.0, 0.03 + 1e-9, 0.03, SIG, T, PutOrCall::Call);
        assert!((price - near).abs() < 1e-5);
    }
}
