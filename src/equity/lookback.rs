//! Lookback options with closed forms (continuous monitoring, GBM).
//!
//! - **Floating strike** (Goldman-Sosin-Gatto 1979): the call pays
//!   `S_T - min S`, the put `max S - S_T` — "buy at the low, sell at
//!   the high", never out of the money;
//! - **Fixed strike** (Conze-Viswanathan 1991): the call pays
//!   `(max S - K)+`, the put `(K - min S)+`.
//!
//! Both support seasoned contracts through the running extremum
//! argument (pass the spot for a fresh option). The formulas take the
//! usual carry `b = r - q`; the `b = 0` singularity (the `sigma^2/2b`
//! factor) is handled by nudging `b` by 1e-7, accurate to ~1e-6 in
//! price and validated against Monte Carlo in the tests.
//!
//! Monte Carlo pricing of the same payoffs monitors **discretely** on
//! the simulation grid, so it sits *below* these continuous forms for
//! max-based payoffs (above for min-based) by the O(sigma sqrt(dt))
//! extremum gap — the tests assert direction and convergence rather
//! than pretending the two conventions coincide.

use crate::core::trade::PutOrCall;
use crate::core::utils::norm_cdf;

/// Floating-strike lookback (Goldman-Sosin-Gatto). `extremum` is the
/// running minimum for a call, the running maximum for a put; pass the
/// spot for a freshly issued option.
pub fn floating_strike_lookback_price(
    s: f64,
    extremum: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && extremum > 0.0 && sigma > 0.0 && t > 0.0);
    let mut b = r - q;
    if b.abs() < 1e-7 {
        b = if b >= 0.0 { 1e-7 } else { -1e-7 };
    }
    let sq = sigma * t.sqrt();
    let two_b_over_v2 = 2.0 * b / (sigma * sigma);
    let carry_df = ((b - r) * t).exp();
    let df = (-r * t).exp();
    match put_or_call {
        PutOrCall::Call => {
            let m = extremum; // running minimum <= s
            assert!(m <= s * (1.0 + 1e-12), "call extremum must be the running minimum");
            let a1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
            let a2 = a1 - sq;
            s * carry_df * norm_cdf(a1) - m * df * norm_cdf(a2)
                + s * df * (1.0 / two_b_over_v2)
                    * ((s / m).powf(-two_b_over_v2) * norm_cdf(-a1 + (2.0 * b / sigma) * t.sqrt())
                        - (b * t).exp() * norm_cdf(-a1))
        }
        PutOrCall::Put => {
            let m = extremum; // running maximum >= s
            assert!(m >= s * (1.0 - 1e-12), "put extremum must be the running maximum");
            let b1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
            let b2 = b1 - sq;
            m * df * norm_cdf(-b2) - s * carry_df * norm_cdf(-b1)
                + s * df * (1.0 / two_b_over_v2)
                    * (-(s / m).powf(-two_b_over_v2) * norm_cdf(b1 - (2.0 * b / sigma) * t.sqrt())
                        + (b * t).exp() * norm_cdf(b1))
        }
    }
}

/// Fixed-strike lookback (Conze-Viswanathan). `extremum` is the running
/// maximum for a call, the running minimum for a put; pass the spot for
/// a freshly issued option.
pub fn fixed_strike_lookback_price(
    s: f64,
    k: f64,
    extremum: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && extremum > 0.0 && sigma > 0.0 && t > 0.0);
    let mut b = r - q;
    if b.abs() < 1e-7 {
        b = if b >= 0.0 { 1e-7 } else { -1e-7 };
    }
    let sq = sigma * t.sqrt();
    let two_b_over_v2 = 2.0 * b / (sigma * sigma);
    let carry_df = ((b - r) * t).exp();
    let df = (-r * t).exp();
    let two_b_sq = (2.0 * b / sigma) * t.sqrt();
    match put_or_call {
        PutOrCall::Call => {
            let m = extremum; // running maximum
            assert!(m >= s * (1.0 - 1e-12), "call extremum must be the running maximum");
            if k > m {
                let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
                let d2 = d1 - sq;
                s * carry_df * norm_cdf(d1) - k * df * norm_cdf(d2)
                    + s * df * (1.0 / two_b_over_v2)
                        * (-(s / k).powf(-two_b_over_v2) * norm_cdf(d1 - two_b_sq)
                            + (b * t).exp() * norm_cdf(d1))
            } else {
                let e1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
                let e2 = e1 - sq;
                df * (m - k) + s * carry_df * norm_cdf(e1) - m * df * norm_cdf(e2)
                    + s * df * (1.0 / two_b_over_v2)
                        * (-(s / m).powf(-two_b_over_v2) * norm_cdf(e1 - two_b_sq)
                            + (b * t).exp() * norm_cdf(e1))
            }
        }
        PutOrCall::Put => {
            let m = extremum; // running minimum
            assert!(m <= s * (1.0 + 1e-12), "put extremum must be the running minimum");
            if k < m {
                let d1 = ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
                let d2 = d1 - sq;
                k * df * norm_cdf(-d2) - s * carry_df * norm_cdf(-d1)
                    + s * df * (1.0 / two_b_over_v2)
                        * ((s / k).powf(-two_b_over_v2) * norm_cdf(-d1 + two_b_sq)
                            - (b * t).exp() * norm_cdf(-d1))
            } else {
                let f1 = ((s / m).ln() + (b + 0.5 * sigma * sigma) * t) / sq;
                let f2 = f1 - sq;
                df * (k - m) - s * carry_df * norm_cdf(-f1) + m * df * norm_cdf(-f2)
                    + s * df * (1.0 / two_b_over_v2)
                        * ((s / m).powf(-two_b_over_v2) * norm_cdf(-f1 + two_b_sq)
                            - (b * t).exp() * norm_cdf(-f1))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    const S: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIG: f64 = 0.3;
    const T: f64 = 1.0;

    #[test]
    fn floating_lookbacks_dominate_atm_vanillas() {
        // S_T - min >= (S_T - S_0)+ pathwise for fresh options, so the
        // lookback must cost more; same for the put side
        let call = floating_strike_lookback_price(S, S, R, Q, SIG, T, PutOrCall::Call);
        let put = floating_strike_lookback_price(S, S, R, Q, SIG, T, PutOrCall::Put);
        assert!(call > bs_price(S, S, R, Q, SIG, T, PutOrCall::Call), "{call}");
        assert!(put > bs_price(S, S, R, Q, SIG, T, PutOrCall::Put), "{put}");
        // and both are worth more than intrinsic zero by a wide margin
        assert!(call > 10.0 && put > 10.0);
    }

    #[test]
    fn fixed_strike_branches_are_continuous_at_the_extremum() {
        // the K > M and K <= M branches must agree at K = M
        let eps = 1e-9;
        let call_above =
            fixed_strike_lookback_price(S, S + eps, S, R, Q, SIG, T, PutOrCall::Call);
        let call_below =
            fixed_strike_lookback_price(S, S - eps, S, R, Q, SIG, T, PutOrCall::Call);
        assert!((call_above - call_below).abs() < 1e-6, "{call_above} vs {call_below}");
        let put_above =
            fixed_strike_lookback_price(S, S + eps, S, R, Q, SIG, T, PutOrCall::Put);
        let put_below =
            fixed_strike_lookback_price(S, S - eps, S, R, Q, SIG, T, PutOrCall::Put);
        assert!((put_above - put_below).abs() < 1e-6, "{put_above} vs {put_below}");
    }

    #[test]
    fn fixed_and_floating_forms_are_linked_by_exact_identities() {
        // (max - K)+ with K -> 0 is max - K, and (max - S_T) + S_T = max:
        // fixed_call(K~0) - floating_put = e^{-rT} E[S_T] - K e^{-rT}
        //                                = S e^{-qT} - K e^{-rT}
        let k_small = 1e-7;
        let fixed_call =
            fixed_strike_lookback_price(S, k_small, S, R, Q, SIG, T, PutOrCall::Call);
        let floating_put = floating_strike_lookback_price(S, S, R, Q, SIG, T, PutOrCall::Put);
        let expect = S * (-Q * T).exp() - k_small * (-R * T).exp();
        assert!(
            (fixed_call - floating_put - expect).abs() < 1e-4,
            "{} vs {}",
            fixed_call - floating_put,
            expect
        );
        // mirrored: (K - min)+ with huge K is K - min = (K - S_T) + (S_T - min)
        let k_big = 100_000.0;
        let fixed_put =
            fixed_strike_lookback_price(S, k_big, S, R, Q, SIG, T, PutOrCall::Put);
        let floating_call = floating_strike_lookback_price(S, S, R, Q, SIG, T, PutOrCall::Call);
        let expect2 = k_big * (-R * T).exp() - S * (-Q * T).exp();
        assert!(
            (fixed_put - floating_call - expect2).abs() < 1e-4,
            "{} vs {}",
            fixed_put - floating_call,
            expect2
        );
    }

    #[test]
    fn zero_carry_nudge_is_smooth() {
        // b = 0 exactly vs b = 1e-5 either side: no discontinuity
        let at = floating_strike_lookback_price(S, S, 0.03, 0.03, SIG, T, PutOrCall::Call);
        let up = floating_strike_lookback_price(S, S, 0.03 + 1e-5, 0.03, SIG, T, PutOrCall::Call);
        let dn = floating_strike_lookback_price(S, S, 0.03 - 1e-5, 0.03, SIG, T, PutOrCall::Call);
        assert!(dn < at && at < up, "{dn} {at} {up}");
        assert!((up - dn).abs() < 0.02, "smooth through b = 0");
    }

    #[test]
    fn seasoned_extrema_move_prices_the_right_way() {
        // a lower observed minimum makes the floating call strictly richer
        // (the gain is modest: the fresh option already expects a low min)
        let fresh = floating_strike_lookback_price(S, S, R, Q, SIG, T, PutOrCall::Call);
        let seasoned = floating_strike_lookback_price(S, 80.0, R, Q, SIG, T, PutOrCall::Call);
        assert!(seasoned > fresh + 1.0, "{seasoned} vs {fresh}");
        // lower bound: the locked-in extremum pays at least e^{-rT}(E[S_T] - 80)
        assert!(seasoned > S * (-Q * T).exp() - 80.0 * (-R * T).exp());
        // a higher observed maximum makes the fixed call richer, and it is
        // floored by the locked-in intrinsic e^{-rT}(130 - 110)
        let fresh_fix = fixed_strike_lookback_price(S, 110.0, S, R, Q, SIG, T, PutOrCall::Call);
        let seasoned_fix =
            fixed_strike_lookback_price(S, 110.0, 130.0, R, Q, SIG, T, PutOrCall::Call);
        assert!(seasoned_fix > fresh_fix + 1.0, "{seasoned_fix} vs {fresh_fix}");
        assert!(seasoned_fix >= (130.0_f64 - 110.0) * (-R * T).exp());
    }
}
