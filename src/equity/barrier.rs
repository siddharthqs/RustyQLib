//! Analytic pricing of continuously monitored barrier options
//! (Reiner-Rubinstein 1991), all eight types: up/down x in/out x call/put,
//! without rebate.
//!
//! Implemented as a pure function of the market inputs so Greeks can be
//! taken by bumping arguments, and so the formulas can be validated
//! independently of the option object (in-out parity, vanilla limits,
//! Monte Carlo agreement).

use crate::core::trade::PutOrCall;
use crate::core::utils::norm_cdf;
use super::blackscholes::bs_price;

/// Which side the barrier sits on relative to the spot at inception.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierDirection {
    Up,
    Down,
}

/// Knock-in options come alive when the barrier is touched; knock-out
/// options die.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnockType {
    In,
    Out,
}

/// Reiner-Rubinstein price of a European barrier option (no rebate).
///
/// If the spot is already at or beyond the barrier the option is treated as
/// knocked: an `Out` option is worthless, an `In` option is the vanilla.
#[allow(clippy::too_many_arguments)]
pub fn barrier_price(
    s: f64,
    k: f64,
    h: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    direction: BarrierDirection,
    knock: KnockType,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && h > 0.0 && sigma > 0.0 && t > 0.0);
    let down = direction == BarrierDirection::Down;
    let knocked_now = if down { s <= h } else { s >= h };
    if knocked_now {
        return match knock {
            KnockType::Out => 0.0,
            KnockType::In => bs_price(s, k, r, q, sigma, t, put_or_call),
        };
    }

    let call = put_or_call == PutOrCall::Call;
    let phi: f64 = if call { 1.0 } else { -1.0 };
    let eta: f64 = if down { 1.0 } else { -1.0 };
    let st = sigma * t.sqrt();
    let mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma);
    let df_q = (-q * t).exp();
    let df_r = (-r * t).exp();
    let hs = h / s;

    let x1 = (s / k).ln() / st + (1.0 + mu) * st;
    let x2 = (s / h).ln() / st + (1.0 + mu) * st;
    let y1 = (h * h / (s * k)).ln() / st + (1.0 + mu) * st;
    let y2 = (h / s).ln() / st + (1.0 + mu) * st;

    let a = phi * s * df_q * norm_cdf(phi * x1) - phi * k * df_r * norm_cdf(phi * x1 - phi * st);
    let b = phi * s * df_q * norm_cdf(phi * x2) - phi * k * df_r * norm_cdf(phi * x2 - phi * st);
    let c = phi * s * df_q * hs.powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y1)
        - phi * k * df_r * hs.powf(2.0 * mu) * norm_cdf(eta * y1 - eta * st);
    let d = phi * s * df_q * hs.powf(2.0 * (mu + 1.0)) * norm_cdf(eta * y2)
        - phi * k * df_r * hs.powf(2.0 * mu) * norm_cdf(eta * y2 - eta * st);

    let k_above_barrier = k >= h;
    let knock_in = match (call, down) {
        (true, true) => if k_above_barrier { c } else { a - b + d },
        (true, false) => if k_above_barrier { a } else { b - c + d },
        (false, true) => if k_above_barrier { b - c + d } else { a },
        (false, false) => if k_above_barrier { a - b + d } else { c },
    };
    match knock {
        KnockType::In => knock_in,
        // in-out parity (no rebate): out = vanilla - in
        KnockType::Out => bs_price(s, k, r, q, sigma, t, put_or_call) - knock_in,
    }
}

/// When a knock-out rebate is paid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebateTiming {
    /// Paid the moment the barrier is touched (Reiner-Rubinstein `F`).
    AtHit,
    /// Paid at expiry if the barrier was touched.
    AtExpiry,
}

/// Present value of the rebate leg of a single-barrier option.
///
/// - **Knock-out**: `rebate` is paid when the barrier is hit — either at
///   the touch time (`AtHit`, the Reiner-Rubinstein `F` term) or at
///   expiry (`AtExpiry`, the complement of the survival term);
/// - **Knock-in**: `rebate` is paid **at expiry** when the option never
///   knocked in (the `E` term); `timing` is ignored.
#[allow(clippy::too_many_arguments)]
pub fn barrier_rebate_value(
    s: f64,
    h: f64,
    rebate: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    direction: BarrierDirection,
    knock: KnockType,
    timing: RebateTiming,
) -> f64 {
    assert!(s > 0.0 && h > 0.0 && sigma > 0.0 && t > 0.0);
    if rebate == 0.0 {
        return 0.0;
    }
    let down = direction == BarrierDirection::Down;
    let eta: f64 = if down { 1.0 } else { -1.0 };
    let knocked_now = if down { s <= h } else { s >= h };
    let df_r = (-r * t).exp();
    if knocked_now {
        return match knock {
            KnockType::In => 0.0, // knocked in: no rebate
            KnockType::Out => match timing {
                RebateTiming::AtHit => rebate,
                RebateTiming::AtExpiry => rebate * df_r,
            },
        };
    }
    let st = sigma * t.sqrt();
    let mu = (r - q - 0.5 * sigma * sigma) / (sigma * sigma);
    let hs = h / s;
    // discounted probability of never touching the barrier (the E term)
    let x2 = (s / h).ln() / st + (1.0 + mu) * st;
    let y2 = (h / s).ln() / st + (1.0 + mu) * st;
    let survival_pv =
        rebate * df_r * (norm_cdf(eta * (x2 - st)) - hs.powf(2.0 * mu) * norm_cdf(eta * (y2 - st)));
    match knock {
        KnockType::In => survival_pv,
        KnockType::Out => match timing {
            RebateTiming::AtExpiry => rebate * df_r - survival_pv,
            RebateTiming::AtHit => {
                // first-touch value (the F term)
                let lambda = (mu * mu + 2.0 * r / (sigma * sigma)).sqrt();
                let z = (h / s).ln() / st + lambda * st;
                rebate
                    * (hs.powf(mu + lambda) * norm_cdf(eta * z)
                        + hs.powf(mu - lambda) * norm_cdf(eta * (z - 2.0 * lambda * st)))
            }
        },
    }
}

/// Single-barrier option with a rebate leg: the Reiner-Rubinstein price
/// plus [`barrier_rebate_value`].
#[allow(clippy::too_many_arguments)]
pub fn barrier_price_with_rebate(
    s: f64,
    k: f64,
    h: f64,
    rebate: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    direction: BarrierDirection,
    knock: KnockType,
    timing: RebateTiming,
    put_or_call: PutOrCall,
) -> f64 {
    barrier_price(s, k, h, r, q, sigma, t, direction, knock, put_or_call)
        + barrier_rebate_value(s, h, rebate, r, q, sigma, t, direction, knock, timing)
}

/// Double-barrier option (flat lower `l` and upper `u` barriers), by the
/// Ikeda-Kunitomo image-series expansion (continuous monitoring, no
/// rebate). Knock-in prices through in-out parity against the vanilla.
///
/// The series converges extremely fast; five image pairs are far below
/// f64 precision for practical inputs. The single-barrier limits
/// (`l -> 0`, `u -> infinity`) reproduce the Reiner-Rubinstein prices
/// (tested).
#[allow(clippy::too_many_arguments)]
pub fn double_barrier_price(
    s: f64,
    k: f64,
    l: f64,
    u: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    knock: KnockType,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && sigma > 0.0 && t > 0.0);
    assert!(l < u, "lower barrier must be below the upper barrier");
    let knocked_now = s <= l || s >= u;
    let vanilla = bs_price(s, k, r, q, sigma, t, put_or_call);
    if knocked_now {
        return match knock {
            KnockType::Out => 0.0,
            KnockType::In => vanilla,
        };
    }
    let b = r - q;
    let st = sigma * t.sqrt();
    let mu = 2.0 * b / (sigma * sigma) + 1.0;
    let df_q = ((b - r) * t).exp();
    let df_r = (-r * t).exp();
    let call = put_or_call == PutOrCall::Call;
    // the effective cap/floor of the payoff region inside the corridor
    let f = if call { u } else { l };
    let mut spot_sum = 0.0;
    let mut strike_sum = 0.0;
    for n in -5i32..=5 {
        let un = u.powi(n);
        let ln = l.powi(n);
        let ratio1 = (un / ln).powf(mu);
        let ratio2 = (l.powi(n + 1) / (un * s)).powf(mu);
        let d1 = ((s * un * un / (k * ln * ln)).ln() + (b + 0.5 * sigma * sigma) * t) / st;
        let d2 = ((s * un * un / (f * ln * ln)).ln() + (b + 0.5 * sigma * sigma) * t) / st;
        let d3 = ((l.powi(2 * n + 2) / (k * s * un * un)).ln()
            + (b + 0.5 * sigma * sigma) * t)
            / st;
        let d4 = ((l.powi(2 * n + 2) / (f * s * un * un)).ln()
            + (b + 0.5 * sigma * sigma) * t)
            / st;
        if call {
            spot_sum += ratio1 * (norm_cdf(d1) - norm_cdf(d2)) - ratio2 * (norm_cdf(d3) - norm_cdf(d4));
            strike_sum += (un / ln).powf(mu - 2.0) * (norm_cdf(d1 - st) - norm_cdf(d2 - st))
                - (l.powi(n + 1) / (un * s)).powf(mu - 2.0) * (norm_cdf(d3 - st) - norm_cdf(d4 - st));
        } else {
            // put: the payoff region is [l, k], integrated with positive
            // normal arguments (d2 anchors the floor f = l, d1 the strike)
            spot_sum += ratio1 * (norm_cdf(d2) - norm_cdf(d1)) - ratio2 * (norm_cdf(d4) - norm_cdf(d3));
            strike_sum += (un / ln).powf(mu - 2.0) * (norm_cdf(d2 - st) - norm_cdf(d1 - st))
                - (l.powi(n + 1) / (un * s)).powf(mu - 2.0) * (norm_cdf(d4 - st) - norm_cdf(d3 - st));
        }
    }
    let phi = if call { 1.0 } else { -1.0 };
    let out = phi * (s * df_q * spot_sum - k * df_r * strike_sum);
    match knock {
        KnockType::Out => out.max(0.0),
        KnockType::In => (vanilla - out.max(0.0)).max(0.0),
    }
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
    fn rebate_identities_hold_exactly() {
        use BarrierDirection::*;
        let rebate = 5.0;
        for (dir, h) in [(Down, 85.0), (Up, 115.0)] {
            // hit and no-hit rebates at expiry are complementary events
            let ki = barrier_rebate_value(S, h, rebate, R, Q, SIG, T, dir, KnockType::In,
                RebateTiming::AtExpiry);
            let ko_exp = barrier_rebate_value(S, h, rebate, R, Q, SIG, T, dir, KnockType::Out,
                RebateTiming::AtExpiry);
            assert!((ki + ko_exp - rebate * (-R * T).exp()).abs() < 1e-12, "{dir:?}");
            // paying at the touch is worth more than waiting until expiry
            let ko_hit = barrier_rebate_value(S, h, rebate, R, Q, SIG, T, dir, KnockType::Out,
                RebateTiming::AtHit);
            assert!(ko_hit > ko_exp, "{dir:?}: {ko_hit} vs {ko_exp}");
            // with r = 0 timing is irrelevant
            let hit0 = barrier_rebate_value(S, h, rebate, 0.0, Q, SIG, T, dir, KnockType::Out,
                RebateTiming::AtHit);
            let exp0 = barrier_rebate_value(S, h, rebate, 0.0, Q, SIG, T, dir, KnockType::Out,
                RebateTiming::AtExpiry);
            assert!((hit0 - exp0).abs() < 1e-10, "{dir:?}: {hit0} vs {exp0}");
        }
        // barrier far away: the knock-out never pays, the knock-in always
        let far = barrier_rebate_value(S, 1e4, rebate, R, Q, SIG, T, Up, KnockType::Out,
            RebateTiming::AtHit);
        assert!(far < 1e-8, "{far}");
        let sure = barrier_rebate_value(S, 1e4, rebate, R, Q, SIG, T, Up, KnockType::In,
            RebateTiming::AtExpiry);
        assert!((sure - rebate * (-R * T).exp()).abs() < 1e-8);
        // already-touched knock-out pays the rebate immediately
        let now = barrier_rebate_value(100.0, 100.0, rebate, R, Q, SIG, T, Up, KnockType::Out,
            RebateTiming::AtHit);
        assert!((now - rebate).abs() < 1e-12);
    }

    #[test]
    fn rebate_at_hit_matches_a_first_touch_monte_carlo() {
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let (h, rebate) = (115.0, 10.0);
        let analytic = barrier_rebate_value(S, h, rebate, R, Q, SIG, T,
            BarrierDirection::Up, KnockType::Out, RebateTiming::AtHit);
        // dense-grid first-touch simulation (discrete monitoring misses
        // some touches, so the MC sits slightly below)
        let (paths, steps) = (60_000, 2000);
        let dt = T / steps as f64;
        let drift = (R - Q - 0.5 * SIG * SIG) * dt;
        let vol = SIG * dt.sqrt();
        let mut sum = 0.0;
        for i in 0..paths {
            let mut rng = path_rng(97, i as u64);
            let mut spot = S;
            for step in 1..=steps {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                spot *= (drift + vol * z).exp();
                if spot >= h {
                    sum += rebate * (-R * step as f64 * dt).exp();
                    break;
                }
            }
        }
        let mc = sum / paths as f64;
        assert!(mc < analytic + 0.02, "discrete monitoring should undercount");
        assert!((mc - analytic).abs() < 0.20, "mc {mc} vs analytic {analytic}");
    }

    #[test]
    fn double_barrier_limits_reproduce_single_barriers() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            for k in [90.0, 100.0, 110.0] {
                // lower barrier -> 0: pure up-and-out
                let dko = double_barrier_price(S, k, 10.0, 120.0, R, Q, SIG, T,
                    KnockType::Out, pc);
                let uo = barrier_price(S, k, 120.0, R, Q, SIG, T,
                    BarrierDirection::Up, KnockType::Out, pc);
                assert!((dko - uo).abs() < 1e-6, "{pc:?} K={k}: {dko} vs UO {uo}");
                // upper barrier -> infinity: pure down-and-out
                let dko2 = double_barrier_price(S, k, 80.0, 1000.0, R, Q, SIG, T,
                    KnockType::Out, pc);
                let down_out = barrier_price(S, k, 80.0, R, Q, SIG, T,
                    BarrierDirection::Down, KnockType::Out, pc);
                assert!((dko2 - down_out).abs() < 1e-6, "{pc:?} K={k}: {dko2} vs DO {down_out}");
                // both far: the vanilla
                let wide = double_barrier_price(S, k, 10.0, 1000.0, R, Q, SIG, T,
                    KnockType::Out, pc);
                let vanilla = crate::equity::blackscholes::bs_price(S, k, R, Q, SIG, T, pc);
                assert!((wide - vanilla).abs() < 1e-6, "{pc:?} K={k}");
            }
        }
    }

    #[test]
    fn double_barrier_parity_and_bounds() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let (l, u, k) = (85.0, 120.0, 100.0);
            let out = double_barrier_price(S, k, l, u, R, Q, SIG, T, KnockType::Out, pc);
            let inn = double_barrier_price(S, k, l, u, R, Q, SIG, T, KnockType::In, pc);
            let vanilla = crate::equity::blackscholes::bs_price(S, k, R, Q, SIG, T, pc);
            assert!((out + inn - vanilla).abs() < 1e-10, "{pc:?} parity");
            // the corridor is more restrictive than either single barrier
            let uo = barrier_price(S, k, u, R, Q, SIG, T,
                BarrierDirection::Up, KnockType::Out, pc);
            let down_out = barrier_price(S, k, l, R, Q, SIG, T,
                BarrierDirection::Down, KnockType::Out, pc);
            assert!(out <= uo + 1e-10 && out <= down_out + 1e-10, "{pc:?} bounds");
            assert!(out > 0.0);
        }
    }

    #[test]
    fn double_barrier_matches_a_dense_monte_carlo() {
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let (l, u, k) = (85.0, 120.0, 100.0);
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let analytic = double_barrier_price(S, k, l, u, R, Q, SIG, T, KnockType::Out, pc);
            let (paths, steps) = (60_000, 2000);
            let dt = T / steps as f64;
            let drift = (R - Q - 0.5 * SIG * SIG) * dt;
            let vol = SIG * dt.sqrt();
            let mut sum = 0.0;
            for i in 0..paths {
                let mut rng = path_rng(13, i as u64);
                let mut spot = S;
                let mut alive = true;
                for _ in 0..steps {
                    let z: f64 = rng.sample(rand_distr::StandardNormal);
                    spot *= (drift + vol * z).exp();
                    if spot <= l || spot >= u {
                        alive = false;
                        break;
                    }
                }
                if alive {
                    sum += match pc {
                        PutOrCall::Call => (spot - k).max(0.0),
                        PutOrCall::Put => (k - spot).max(0.0),
                    };
                }
            }
            let mc = (-R * T).exp() * sum / paths as f64;
            // discrete monitoring survives more often -> MC above analytic
            assert!(mc > analytic - 0.02, "{pc:?}: {mc} vs {analytic}");
            assert!((mc - analytic).abs() < 0.25, "{pc:?}: mc {mc} vs analytic {analytic}");
        }
    }

    #[test]
    fn matches_independent_oracle_goldens() {
        use BarrierDirection::*;
        use KnockType::*;
        use PutOrCall::*;
        // values from an independently coded Reiner-Rubinstein implementation
        let cases = [
            (Down, In, Call, 90.0, 4.5095197744),
            (Down, Out, Call, 90.0, 8.5107614943),
            (Down, In, Put, 90.0, 10.0710164338),
            (Down, Out, Put, 90.0, 0.0523399543),
            (Up, In, Call, 120.0, 12.5974705742),
            (Up, Out, Call, 120.0, 0.4228106946),
            (Up, In, Put, 120.0, 1.4297711810),
            (Up, Out, Put, 120.0, 8.6935852071),
        ];
        for (direction, knock, pc, h, expected) in cases {
            let price = barrier_price(S, 100.0, h, R, Q, SIG, T, direction, knock, pc);
            assert!(
                (price - expected).abs() < 1e-8,
                "{direction:?} {knock:?} {pc:?} H={h}: {price} vs {expected}"
            );
        }
    }

    #[test]
    fn in_plus_out_equals_vanilla() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            for k in [90.0, 100.0, 110.0] {
                for (direction, h) in [
                    (BarrierDirection::Down, 80.0),
                    (BarrierDirection::Down, 99.0),
                    (BarrierDirection::Up, 101.0),
                    (BarrierDirection::Up, 130.0),
                ] {
                    let vanilla = bs_price(S, k, R, Q, SIG, T, pc);
                    let ki = barrier_price(S, k, h, R, Q, SIG, T, direction, KnockType::In, pc);
                    let ko = barrier_price(S, k, h, R, Q, SIG, T, direction, KnockType::Out, pc);
                    assert!(
                        (ki + ko - vanilla).abs() < 1e-10,
                        "{pc:?} K={k} {direction:?} H={h}: {ki} + {ko} != {vanilla}"
                    );
                }
            }
        }
    }

    #[test]
    fn far_barriers_reduce_to_vanilla_or_zero() {
        let vanilla_call = bs_price(S, 100.0, R, Q, SIG, T, PutOrCall::Call);
        // barrier so far away it is never hit: out = vanilla, in = 0
        let ko = barrier_price(S, 100.0, 1e-4, R, Q, SIG, T, BarrierDirection::Down, KnockType::Out, PutOrCall::Call);
        let ki = barrier_price(S, 100.0, 1e-4, R, Q, SIG, T, BarrierDirection::Down, KnockType::In, PutOrCall::Call);
        assert!((ko - vanilla_call).abs() < 1e-9);
        assert!(ki.abs() < 1e-9);
        let ko_up = barrier_price(S, 100.0, 1e6, R, Q, SIG, T, BarrierDirection::Up, KnockType::Out, PutOrCall::Call);
        assert!((ko_up - vanilla_call).abs() < 1e-9);
    }

    #[test]
    fn already_knocked_positions() {
        let vanilla = bs_price(S, 100.0, R, Q, SIG, T, PutOrCall::Call);
        // spot at the barrier counts as touched
        let ko = barrier_price(S, 100.0, 100.0, R, Q, SIG, T, BarrierDirection::Down, KnockType::Out, PutOrCall::Call);
        let ki = barrier_price(S, 100.0, 100.0, R, Q, SIG, T, BarrierDirection::Down, KnockType::In, PutOrCall::Call);
        assert_eq!(ko, 0.0);
        assert!((ki - vanilla).abs() < 1e-12);
    }

    #[test]
    fn up_out_call_with_strike_above_barrier_is_worthless() {
        // any payoff requires S_T > K >= H, which forces a knock
        let price = barrier_price(
            S, 110.0, 105.0, R, Q, SIG, T,
            BarrierDirection::Up, KnockType::Out, PutOrCall::Call,
        );
        assert!(price.abs() < 1e-12, "{price}");
    }
}
