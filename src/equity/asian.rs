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
use crate::core::utils::norm_cdf;

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
        PutOrCall::Call => df_r * (forward * norm_cdf(d1) - k * norm_cdf(d2)),
        PutOrCall::Put => df_r * (k * norm_cdf(-d2) - forward * norm_cdf(-d1)),
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

/// Exact price of a geometric average-**strike** (floating-strike) Asian:
/// the call pays `(S_T - G)+` and the put `(G - S_T)+`, with `G` the
/// geometric average of `n` equally spaced fixings (continuous averaging
/// when `None`).
///
/// `ln S_T` and `ln G` are jointly normal, so the payoff is an exchange
/// option between two lognormals and prices exactly by
/// `E[(A - B)+] = E[A] N(d1) - E[B] N(d2)` with
/// `s^2 = Var(ln S_T - ln G)`. With a single fixing `G = S_T` and the
/// option is worthless; more fixings decouple the average from the
/// terminal spot and raise the value toward the continuous limit.
pub fn geometric_average_strike_price(
    s: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    n: Option<usize>,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && sigma > 0.0 && t > 0.0);
    let b = r - q;
    // E[ln G] factor, Var(ln G) factor, Cov(ln S_T, ln G) factor (in
    // units of (b - sigma^2/2) t / sigma^2 t)
    let (mean_factor, var_factor, cov_factor) = match n {
        Some(n) => {
            assert!(n > 0);
            let nf = n as f64;
            (
                (nf + 1.0) / (2.0 * nf),
                (nf + 1.0) * (2.0 * nf + 1.0) / (6.0 * nf * nf),
                (nf + 1.0) / (2.0 * nf),
            )
        }
        None => (0.5, 1.0 / 3.0, 0.5),
    };
    let e_terminal = s * (b * t).exp();
    let e_average = s
        * ((b - 0.5 * sigma * sigma) * t * mean_factor
            + 0.5 * sigma * sigma * t * var_factor)
            .exp();
    let spread_var = sigma * sigma * t * (1.0 + var_factor - 2.0 * cov_factor);
    let df = (-r * t).exp();
    if spread_var <= 1e-16 {
        // single fixing: G = S_T, zero intrinsic either way
        return match put_or_call {
            PutOrCall::Call => df * (e_terminal - e_average).max(0.0),
            PutOrCall::Put => df * (e_average - e_terminal).max(0.0),
        };
    }
    let sv = spread_var.sqrt();
    let d1 = ((e_terminal / e_average).ln() + 0.5 * spread_var) / sv;
    let d2 = d1 - sv;
    match put_or_call {
        PutOrCall::Call => df * (e_terminal * norm_cdf(d1) - e_average * norm_cdf(d2)),
        PutOrCall::Put => df * (e_average * norm_cdf(-d2) - e_terminal * norm_cdf(-d1)),
    }
}

/// Arithmetic average-**strike** (floating-strike) Asian via the
/// Henderson-Wojakowski (2002) symmetry: at inception under GBM, a
/// floating-strike call paying `(S_T - A)+` is **exactly** a
/// fixed-strike put struck at spot with the roles of `r` and `q`
/// interchanged (and vice versa for the put). The fixed-strike side is
/// then priced with the Turnbull-Wakeman moment match, so the symmetry
/// step is exact and TW is the only approximation layer. Continuous
/// averaging; exact put-call parity
/// `C - P = e^{-rT} (E[S_T] - E[A])` is preserved.
pub fn turnbull_wakeman_average_strike_price(
    s: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    match put_or_call {
        PutOrCall::Call => turnbull_wakeman_price(s, s, q, r, sigma, t, PutOrCall::Put),
        PutOrCall::Put => turnbull_wakeman_price(s, s, q, r, sigma, t, PutOrCall::Call),
    }
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

    #[test]
    fn average_strike_single_fixing_is_worthless() {
        // n = 1: the average IS the terminal spot
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let p = geometric_average_strike_price(S, R, Q, SIG, 1.0, Some(1), pc);
            assert!(p.abs() < 1e-12, "{pc:?}: {p}");
        }
    }

    #[test]
    fn average_strike_exchange_parity_is_exact() {
        // C - P = e^{-rT} (E[S_T] - E[G]) for jointly lognormal legs
        for n in [Some(4), Some(12), None] {
            let c = geometric_average_strike_price(S, R, Q, SIG, 1.0, n, PutOrCall::Call);
            let p = geometric_average_strike_price(S, R, Q, SIG, 1.0, n, PutOrCall::Put);
            let b = R - Q;
            let (mf, vf) = match n {
                Some(n) => {
                    let nf = n as f64;
                    ((nf + 1.0) / (2.0 * nf), (nf + 1.0) * (2.0 * nf + 1.0) / (6.0 * nf * nf))
                }
                None => (0.5, 1.0 / 3.0),
            };
            let e_st = S * b.exp();
            let e_g = S * ((b - 0.5 * SIG * SIG) * mf + 0.5 * SIG * SIG * vf).exp();
            let parity = (-R * 1.0f64).exp() * (e_st - e_g);
            assert!((c - p - parity).abs() < 1e-12, "n = {n:?}");
        }
    }

    #[test]
    fn arithmetic_average_strike_preserves_exact_parity() {
        // the Henderson-Wojakowski swap preserves put-call parity
        // C - P = e^{-rT} (E[S_T] - E[A]) exactly, with
        // E[A] = S (e^{bT} - 1)/(bT)
        let t = 1.0;
        let b = R - Q;
        let c = turnbull_wakeman_average_strike_price(S, R, Q, SIG, t, PutOrCall::Call);
        let p = turnbull_wakeman_average_strike_price(S, R, Q, SIG, t, PutOrCall::Put);
        let parity = (-R * t).exp() * (S * (b * t).exp() - S * ((b * t).exp() - 1.0) / (b * t));
        assert!((c - p - parity).abs() < 1e-12, "{}", c - p - parity);
        // and the zero-carry branch works through the swap (b' = -b = 0)
        let c0 = turnbull_wakeman_average_strike_price(S, 0.03, 0.03, SIG, t, PutOrCall::Call);
        let p0 = turnbull_wakeman_average_strike_price(S, 0.03, 0.03, SIG, t, PutOrCall::Put);
        assert!(c0 > 0.0 && p0 > 0.0);
        assert!((c0 - p0).abs() < 1e-12, "b = 0 parity: E[S_T] = E[A]");
    }

    #[test]
    fn arithmetic_average_strike_orders_against_geometric_by_am_gm() {
        // the arithmetic average dominates the geometric, so the
        // average-strike CALL is cheaper and the PUT richer than the
        // geometric-average versions
        let t = 1.0;
        let arith_call = turnbull_wakeman_average_strike_price(S, R, Q, SIG, t, PutOrCall::Call);
        let geo_call = geometric_average_strike_price(S, R, Q, SIG, t, None, PutOrCall::Call);
        assert!(arith_call < geo_call, "call: arith {arith_call} vs geo {geo_call}");
        let arith_put = turnbull_wakeman_average_strike_price(S, R, Q, SIG, t, PutOrCall::Put);
        let geo_put = geometric_average_strike_price(S, R, Q, SIG, t, None, PutOrCall::Put);
        assert!(arith_put > geo_put, "put: arith {arith_put} vs geo {geo_put}");
    }

    #[test]
    fn average_strike_value_grows_with_fixings_toward_the_continuous_limit() {
        let price = |n| geometric_average_strike_price(S, R, Q, SIG, 1.0, n, PutOrCall::Call);
        assert!(price(Some(2)) > 0.0);
        assert!(price(Some(4)) > price(Some(2)));
        assert!(price(Some(12)) > price(Some(4)));
        let continuous = price(None);
        assert!(price(Some(12)) < continuous);
        assert!((price(Some(5000)) - continuous).abs() < 2e-3, "limit");
    }
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
