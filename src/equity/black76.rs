//! Black-76 (1976): European options on a future/forward price `F`.
//!
//! Two settlement styles:
//! - **Discounted** (standard Black-76): the premium is paid up front and
//!   the payoff is discounted, `call = e^{-rT}[F N(d1) - K N(d2)]`.
//! - **Margined** (futures-style / "future-style"): the option premium is
//!   itself margined daily like the future, so there is no discounting,
//!   `call = F N(d1) - K N(d2)`. Common for options on futures on many
//!   non-US derivatives exchanges (e.g. Eurex, ICE, ASX).
//!
//! `F` is the futures price directly — Black-76 has no spot, dividend or
//! carry, since a future already embeds the cost of carry. All Greeks are
//! sensitivities with respect to `F` (delta/gamma), `sigma`, `r` and time.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::core::trade::PutOrCall;
use crate::core::utils::{dN, N};

/// How an option on a future is settled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuturesSettlement {
    /// Premium paid up front; the payoff is discounted at the risk-free rate.
    Discounted,
    /// Futures-style: the premium is margined, so the payoff is undiscounted.
    Margined,
}

impl FromStr for FuturesSettlement {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "discounted" | "black76" | "premium" | "premium_settled" => {
                Ok(FuturesSettlement::Discounted)
            }
            "margined" | "futures_style" | "future_style" | "futures-style" => {
                Ok(FuturesSettlement::Margined)
            }
            other => Err(format!(
                "Invalid futures settlement '{other}' (use 'discounted' or 'margined')"
            )),
        }
    }
}

impl FuturesSettlement {
    /// Discount factor applied to the payoff: `e^{-rT}` when the premium is
    /// paid up front, `1` when it is margined.
    pub fn discount_factor(&self, r: f64, t: f64) -> f64 {
        match self {
            FuturesSettlement::Discounted => (-r * t).exp(),
            FuturesSettlement::Margined => 1.0,
        }
    }
}

fn d1_d2(f: f64, k: f64, sigma: f64, t: f64) -> (f64, f64) {
    let st = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sigma * sigma * t) / st;
    (d1, d1 - st)
}

/// Black-76 price of a European option on a future.
pub fn price(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    assert!(f > 0.0 && k > 0.0, "futures price and strike must be positive");
    let df = settlement.discount_factor(r, t);
    if t <= 0.0 || sigma <= 0.0 {
        let intrinsic = match put_or_call {
            PutOrCall::Call => (f - k).max(0.0),
            PutOrCall::Put => (k - f).max(0.0),
        };
        return df * intrinsic;
    }
    let (d1, d2) = d1_d2(f, k, sigma, t);
    match put_or_call {
        PutOrCall::Call => df * (f * N(d1) - k * N(d2)),
        PutOrCall::Put => df * (k * N(-d2) - f * N(-d1)),
    }
}

/// Delta with respect to the futures price `F`.
pub fn delta(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let (d1, _) = d1_d2(f, k, sigma, t);
    match put_or_call {
        PutOrCall::Call => df * N(d1),
        PutOrCall::Put => -df * N(-d1),
    }
}

/// Gamma with respect to the futures price `F` (same for calls and puts).
pub fn gamma(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let (d1, _) = d1_d2(f, k, sigma, t);
    df * dN(d1) / (f * sigma * t.sqrt())
}

/// Vega (per unit of vol; same for calls and puts).
pub fn vega(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let (d1, _) = d1_d2(f, k, sigma, t);
    df * f * dN(d1) * t.sqrt()
}

/// Rho (sensitivity to the risk-free rate). Zero for margined options,
/// which have no discounting; `-T * price` for discounted options (the
/// futures price is exogenous, so `r` enters only through the discount).
pub fn rho(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    match settlement {
        FuturesSettlement::Margined => 0.0,
        FuturesSettlement::Discounted => -t * price(f, k, r, sigma, t, put_or_call, settlement),
    }
}

/// Theta (calendar time decay, `dV/dt = -dV/dT`).
pub fn theta(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let (d1, _) = d1_d2(f, k, sigma, t);
    // volatility bleed term F df dN(d1) sigma / (2 sqrt(T)), common to both
    // settlement styles and to calls and puts
    let bleed = df * f * dN(d1) * sigma / (2.0 * t.sqrt());
    match settlement {
        FuturesSettlement::Margined => -bleed,
        FuturesSettlement::Discounted => {
            r * price(f, k, r, sigma, t, put_or_call, settlement) - bleed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    const F: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.05;
    const SIG: f64 = 0.30;
    const T: f64 = 1.0;

    // Golden values, bump-verified against an independent reference
    #[test]
    fn discounted_golden_values() {
        use FuturesSettlement::Discounted as D;
        assert!((price(F, K, R, SIG, T, PutOrCall::Call, D) - 11.34202064).abs() < 1e-7);
        assert!((delta(F, K, R, SIG, T, PutOrCall::Call, D) - 0.53232482).abs() < 1e-7);
        assert!((delta(F, K, R, SIG, T, PutOrCall::Put, D) + 0.41890461).abs() < 1e-7);
        assert!((gamma(F, K, R, SIG, T, D) - 0.01250801).abs() < 1e-7);
        assert!((vega(F, K, R, SIG, T, D) - 37.52403469).abs() < 1e-6);
        assert!((rho(F, K, R, SIG, T, PutOrCall::Call, D) + 11.34202064).abs() < 1e-6);
        assert!((theta(F, K, R, SIG, T, PutOrCall::Call, D) + 5.06150417).abs() < 1e-6);
    }

    #[test]
    fn margined_golden_values() {
        use FuturesSettlement::Margined as M;
        assert!((price(F, K, R, SIG, T, PutOrCall::Call, M) - 11.92353847).abs() < 1e-7);
        assert!((delta(F, K, R, SIG, T, PutOrCall::Call, M) - 0.55961769).abs() < 1e-7);
        assert!((gamma(F, K, R, SIG, T, M) - 0.01314931).abs() < 1e-7);
        assert!((vega(F, K, R, SIG, T, M) - 39.44793309).abs() < 1e-6);
        assert!((theta(F, K, R, SIG, T, PutOrCall::Call, M) + 5.91718996).abs() < 1e-6);
    }

    #[test]
    fn margined_rho_is_zero() {
        // no discounting -> no rate sensitivity at all
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            assert_eq!(rho(F, K, R, SIG, T, pc, FuturesSettlement::Margined), 0.0);
        }
    }

    #[test]
    fn margined_exceeds_discounted() {
        // the undiscounted premium is worth more than the discounted one
        let disc = price(F, K, R, SIG, T, PutOrCall::Call, FuturesSettlement::Discounted);
        let marg = price(F, K, R, SIG, T, PutOrCall::Call, FuturesSettlement::Margined);
        assert!(marg > disc);
        // margined = discounted / e^{-rT}
        assert!((marg - disc * (R * T).exp()).abs() < 1e-10);
    }

    #[test]
    fn put_call_parity_both_styles() {
        for (s, factor) in [
            (FuturesSettlement::Discounted, (-R * T).exp()),
            (FuturesSettlement::Margined, 1.0),
        ] {
            let c = price(F, 95.0, R, SIG, T, PutOrCall::Call, s);
            let p = price(F, 95.0, R, SIG, T, PutOrCall::Put, s);
            assert!((c - p - factor * (F - 95.0)).abs() < 1e-10, "{s:?}");
        }
    }

    #[test]
    fn discounted_black76_equals_black_scholes_at_the_forward() {
        // Black-76 on F = S e^{(r-q)T} must reproduce Black-Scholes-Merton
        let (s, q) = (100.0, 0.02);
        let fwd = s * ((R - q) * T).exp();
        let b76 = price(fwd, K, R, SIG, T, PutOrCall::Call, FuturesSettlement::Discounted);
        let bsm = bs_price(s, K, R, q, SIG, T, PutOrCall::Call);
        assert!((b76 - bsm).abs() < 1e-10, "b76 {b76} vs bsm {bsm}");
    }

    #[test]
    fn settlement_parses_from_strings() {
        assert_eq!("discounted".parse::<FuturesSettlement>().unwrap(), FuturesSettlement::Discounted);
        assert_eq!("black76".parse::<FuturesSettlement>().unwrap(), FuturesSettlement::Discounted);
        assert_eq!("margined".parse::<FuturesSettlement>().unwrap(), FuturesSettlement::Margined);
        assert_eq!("futures_style".parse::<FuturesSettlement>().unwrap(), FuturesSettlement::Margined);
        assert!("bad".parse::<FuturesSettlement>().is_err());
    }
}
