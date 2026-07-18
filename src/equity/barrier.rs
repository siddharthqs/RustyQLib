//! Analytic pricing of continuously monitored barrier options
//! (Reiner-Rubinstein 1991), all eight types: up/down x in/out x call/put,
//! without rebate.
//!
//! Implemented as a pure function of the market inputs so Greeks can be
//! taken by bumping arguments, and so the formulas can be validated
//! independently of the option object (in-out parity, vanilla limits,
//! Monte Carlo agreement).

use crate::core::trade::PutOrCall;
use crate::core::utils::N;
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

    let a = phi * s * df_q * N(phi * x1) - phi * k * df_r * N(phi * x1 - phi * st);
    let b = phi * s * df_q * N(phi * x2) - phi * k * df_r * N(phi * x2 - phi * st);
    let c = phi * s * df_q * hs.powf(2.0 * (mu + 1.0)) * N(eta * y1)
        - phi * k * df_r * hs.powf(2.0 * mu) * N(eta * y1 - eta * st);
    let d = phi * s * df_q * hs.powf(2.0 * (mu + 1.0)) * N(eta * y2)
        - phi * k * df_r * hs.powf(2.0 * mu) * N(eta * y2 - eta * st);

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

#[cfg(test)]
mod tests {
    use super::*;

    const S: f64 = 100.0;
    const R: f64 = 0.05;
    const Q: f64 = 0.02;
    const SIG: f64 = 0.3;
    const T: f64 = 1.0;

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
