//! Perpetual (infinite-maturity) American options — Merton (1973)
//! closed forms. Unlike the finite-maturity approximations
//! ([`baw`](crate::equity::baw),
//! [`bjerksund_stensland`](crate::equity::bjerksund_stensland)) these are
//! **exact**: with no expiry the American value solves the stationary ODE
//! `1/2 sigma^2 S^2 V'' + b S V' - r V = 0` with value matching and
//! smooth pasting at a constant exercise boundary.
//!
//! The exponents `y1 > 1 > 0 > y2` are the roots of the quadratic
//! `1/2 sigma^2 y (y - 1) + b y - r = 0` — the same `beta` that drives
//! the Barone-Adesi-Whaley and Bjerksund-Stensland boundaries, whose
//! infinite-maturity limit these formulas are. Finite-maturity American
//! prices increase in maturity toward the perpetual value (tested).
//!
//! Conventions match the rest of the library: `q` is the total carry
//! (dividend yield + borrow), `b = r - q`.

/// The positive (`y1`) and negative (`y2`) roots of the fundamental
/// quadratic.
fn roots(r: f64, b: f64, sigma: f64) -> (f64, f64) {
    let v2 = sigma * sigma;
    let half_shift = 0.5 - b / v2;
    let disc = ((b / v2 - 0.5).powi(2) + 2.0 * r / v2).sqrt();
    (half_shift + disc, half_shift - disc)
}

/// Perpetual American call.
///
/// Requires `q > 0` (i.e. `b < r`) for a finite value: without a carry
/// cost early exercise is never optimal and the value equals the spot
/// (`b = r`); with `b > r` the value is unbounded and infinity is
/// returned.
pub fn perpetual_call(s: f64, k: f64, r: f64, q: f64, sigma: f64) -> f64 {
    assert!(s > 0.0 && k > 0.0 && sigma > 0.0, "need positive spot, strike and vol");
    let b = r - q;
    if b > r {
        return f64::INFINITY; // undiscounted growth beats financing
    }
    if b == r {
        return s; // never exercised; the option is worth the stock
    }
    let (y1, _) = roots(r, b, sigma);
    let boundary = y1 / (y1 - 1.0) * k;
    if s >= boundary {
        return s - k;
    }
    k / (y1 - 1.0) * (((y1 - 1.0) / y1) * (s / k)).powf(y1)
}

/// Perpetual American put. Requires `r > 0` (with no discounting the
/// optimal-stopping problem degenerates).
pub fn perpetual_put(s: f64, k: f64, r: f64, q: f64, sigma: f64) -> f64 {
    assert!(s > 0.0 && k > 0.0 && sigma > 0.0, "need positive spot, strike and vol");
    assert!(r > 0.0, "the perpetual put needs a positive risk-free rate");
    let b = r - q;
    let (_, y2) = roots(r, b, sigma);
    let boundary = y2 / (y2 - 1.0) * k;
    if s <= boundary {
        return k - s;
    }
    k / (1.0 - y2) * (((y2 - 1.0) / y2) * (s / k)).powf(y2)
}

/// The constant early-exercise boundary: exercise the call once the spot
/// rises to `y1/(y1-1) K`, the put once it falls to `y2/(y2-1) K`.
pub fn exercise_boundary(
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    put_or_call: crate::core::trade::PutOrCall,
) -> f64 {
    let b = r - q;
    let (y1, y2) = roots(r, b, sigma);
    match put_or_call {
        crate::core::trade::PutOrCall::Call => {
            assert!(b < r, "the perpetual call is never exercised when b >= r");
            y1 / (y1 - 1.0) * k
        }
        crate::core::trade::PutOrCall::Put => y2 / (y2 - 1.0) * k,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;

    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.03;
    const V: f64 = 0.25;

    #[test]
    fn golden_values_match_the_validated_reference() {
        assert!((perpetual_call(S, K, R, Q, V) - 40.3730823948).abs() < 1e-9);
        assert!((perpetual_put(S, K, R, Q, V) - 23.4169723789).abs() < 1e-9);
        assert!((exercise_boundary(K, R, Q, V, PutOrCall::Call) - 318.505635).abs() < 1e-5);
        assert!((exercise_boundary(K, R, Q, V, PutOrCall::Put) - 52.327698).abs() < 1e-5);
    }

    #[test]
    fn solves_the_stationary_ode_exactly() {
        // 1/2 v^2 S^2 V'' + b S V' - r V = 0 in the continuation region
        let b = R - Q;
        for f in [
            (|s: f64| perpetual_call(s, K, R, Q, V)) as fn(f64) -> f64,
            |s: f64| perpetual_put(s, K, R, Q, V),
        ] {
            for s in [60.0, 80.0, 100.0, 150.0] {
                let h = s * 1e-4;
                let (v0, vp, vm) = (f(s), f(s + h), f(s - h));
                let d1 = (vp - vm) / (2.0 * h);
                let d2 = (vp - 2.0 * v0 + vm) / (h * h);
                let residual = 0.5 * V * V * s * s * d2 + b * s * d1 - R * v0;
                assert!(residual.abs() < 1e-5 * (1.0 + v0), "S = {s}: residual {residual}");
            }
        }
    }

    #[test]
    fn value_matching_and_smooth_pasting_at_the_boundary() {
        let call_boundary = exercise_boundary(K, R, Q, V, PutOrCall::Call);
        let put_boundary = exercise_boundary(K, R, Q, V, PutOrCall::Put);
        // value matching: the formula meets intrinsic at the boundary
        assert!((perpetual_call(call_boundary, K, R, Q, V) - (call_boundary - K)).abs() < 1e-9);
        assert!((perpetual_put(put_boundary, K, R, Q, V) - (K - put_boundary)).abs() < 1e-9);
        // smooth pasting: the derivative meets +-1 there
        let h = 1e-5;
        let call_slope =
            (perpetual_call(call_boundary - h, K, R, Q, V)
                - perpetual_call(call_boundary - 2.0 * h, K, R, Q, V))
                / h;
        assert!((call_slope - 1.0).abs() < 1e-4, "call slope {call_slope}");
        let put_slope = (perpetual_put(put_boundary + 2.0 * h, K, R, Q, V)
            - perpetual_put(put_boundary + h, K, R, Q, V))
            / h;
        assert!((put_slope + 1.0).abs() < 1e-4, "put slope {put_slope}");
    }

    #[test]
    fn put_call_duality_holds() {
        // McDonald-Schroder: P(S, K, r, q) = C(K, S, r' = q, q' = r)
        let p = perpetual_put(S, K, R, Q, V);
        let c = perpetual_call(K, S, Q, R, V);
        assert!((p - c).abs() < 1e-12, "{p} vs {c}");
    }

    #[test]
    fn degenerate_carry_cases() {
        // no carry cost: the perpetual call is worth the stock
        assert_eq!(perpetual_call(100.0, 80.0, 0.05, 0.0, 0.3), 100.0);
        // negative q (carry above r): unbounded
        assert!(perpetual_call(100.0, 80.0, 0.05, -0.01, 0.3).is_infinite());
        // deep in the exercise regions: intrinsic
        assert_eq!(perpetual_call(500.0, 100.0, R, Q, V), 400.0);
        assert_eq!(perpetual_put(30.0, 100.0, R, Q, V), 70.0);
    }

    #[test]
    fn finite_maturity_american_prices_increase_toward_the_perpetual() {
        // Bjerksund-Stensland is a lower bound on the true American price,
        // and the true price is bounded by the perpetual — so the BS2002
        // ladder must increase in maturity and stay below the perpetual.
        // (BAW is NOT a bound: at T = 40 it overshoots this perpetual by
        // ~0.6, which is exactly why it is not used for this test.)
        use crate::equity::bjerksund_stensland;
        let perpetual = perpetual_put(S, K, R, Q, V);
        let mut last = 0.0;
        for t in [1.0, 5.0, 15.0, 40.0] {
            let finite = bjerksund_stensland::price(S, K, R, Q, V, t, PutOrCall::Put);
            assert!(finite > last, "not increasing at T = {t}");
            assert!(finite <= perpetual + 1e-9, "T = {t}: {finite} above perpetual {perpetual}");
            last = finite;
        }
        // by T = 40 the finite price is close to the perpetual limit
        assert!(perpetual - last < 0.10 * perpetual, "T=40 {last} vs perpetual {perpetual}");
    }
}
