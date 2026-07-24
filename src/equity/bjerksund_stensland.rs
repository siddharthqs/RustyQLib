//! Bjerksund & Stensland (2002) two-boundary approximation for American
//! options — the tighter sibling of Barone-Adesi-Whaley.
//!
//! The exercise boundary is approximated by a two-step flat barrier: one
//! level over the first `t1 = (sqrt(5) - 1)/2 * T` of the option's life
//! and a lower one after, which turns the American value into a
//! closed-form sum of single (`phi`) and bivariate (`psi`) normal
//! probabilities. Because the strategy is feasible but suboptimal the
//! price is a **lower bound** on the true American value — typically
//! within a few cents. Like [`baw`](crate::equity::baw), everything is
//! expressed with the cost of carry `b = r - q`; puts price through the
//! Bjerksund-Stensland transformation
//! `P(S, K, r, b) = C(K, S, r - b, -b)`.
//!
//! Use it (or BAW) when speed matters more than the last cent; a tree or
//! PDE solve when precision is paramount. Between the two analytic
//! engines, BS2002's two-step boundary is generally the tighter fit,
//! especially for longer maturities.

use crate::core::trade::PutOrCall;
use crate::core::utils::{bivariate_norm_cdf, ContractStyle, norm_cdf};
use crate::equity::blackscholes::bs_price;
use crate::equity::vanilla_option::EquityOption;

/// Bjerksund-Stensland (2002) American vanilla price. `q` is the total
/// continuous carry (`b = r - q`). Degenerate inputs return intrinsic.
pub fn price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, put_or_call: PutOrCall) -> f64 {
    let intrinsic = match put_or_call {
        PutOrCall::Call => (s - k).max(0.0),
        PutOrCall::Put => (k - s).max(0.0),
    };
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic;
    }
    let b = r - q;
    let value = match put_or_call {
        PutOrCall::Call => call_2002(s, k, t, r, b, sigma),
        // put transformation: swap spot/strike, r -> r - b, b -> -b
        PutOrCall::Put => call_2002(k, s, t, r - b, -b, sigma),
    };
    // the approximation is a lower bound; intrinsic and European floors
    // cost nothing and guard the deep tails
    value.max(intrinsic).max(bs_price(s, k, r, q, sigma, t, put_or_call))
}

/// Early-exercise premium over the European price (non-negative).
pub fn early_exercise_premium(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    price(s, k, r, q, sigma, t, put_or_call) - bs_price(s, k, r, q, sigma, t, put_or_call)
}

/// The flat-boundary American call (Bjerksund-Stensland 2002, eq. 4-6).
fn call_2002(s: f64, k: f64, t: f64, r: f64, b: f64, sigma: f64) -> f64 {
    if b >= r {
        // never exercised early: European
        return bs_price(s, k, r, r - b, sigma, t, PutOrCall::Call);
    }
    let v2 = sigma * sigma;
    let t1 = 0.5 * (5.0_f64.sqrt() - 1.0) * t;
    let beta = (0.5 - b / v2) + ((b / v2 - 0.5).powi(2) + 2.0 * r / v2).sqrt();
    let b_inf = beta / (beta - 1.0) * k;
    let b0 = if r - b > 0.0 { k.max(r / (r - b) * k) } else { k };
    let h_t = -(b * t + 2.0 * sigma * t.sqrt()) * k * k / ((b_inf - b0) * b0);
    let h_t1 = -(b * t1 + 2.0 * sigma * t1.sqrt()) * k * k / ((b_inf - b0) * b0);
    // near boundary (after t1) and far boundary (before t1)
    let i1 = b0 + (b_inf - b0) * (1.0 - h_t1.exp());
    let i2 = b0 + (b_inf - b0) * (1.0 - h_t.exp());
    if s >= i2 {
        return s - k;
    }
    let alpha1 = (i1 - k) * i1.powf(-beta);
    let alpha2 = (i2 - k) * i2.powf(-beta);

    alpha2 * s.powf(beta) - alpha2 * phi(s, t1, beta, i2, i2, r, b, sigma)
        + phi(s, t1, 1.0, i2, i2, r, b, sigma)
        - phi(s, t1, 1.0, i1, i2, r, b, sigma)
        - k * phi(s, t1, 0.0, i2, i2, r, b, sigma)
        + k * phi(s, t1, 0.0, i1, i2, r, b, sigma)
        + alpha1 * phi(s, t1, beta, i1, i2, r, b, sigma)
        - alpha1 * psi(s, t, beta, i1, i2, i1, t1, r, b, sigma)
        + psi(s, t, 1.0, i1, i2, i1, t1, r, b, sigma)
        - psi(s, t, 1.0, k, i2, i1, t1, r, b, sigma)
        - k * psi(s, t, 0.0, i1, i2, i1, t1, r, b, sigma)
        + k * psi(s, t, 0.0, k, i2, i1, t1, r, b, sigma)
}

/// Single-boundary expectation `phi`: the value of `S_T^gamma` collected
/// on `{S_T beyond H}` under a flat barrier `I`, by the reflection
/// principle.
fn phi(s: f64, t: f64, gamma: f64, h: f64, i: f64, r: f64, b: f64, sigma: f64) -> f64 {
    let sqt = sigma * t.sqrt();
    let lambda = (-r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma * sigma) * t;
    let d = -((s / h).ln() + (b + (gamma - 0.5) * sigma * sigma) * t) / sqt;
    let kappa = 2.0 * b / (sigma * sigma) + 2.0 * gamma - 1.0;
    lambda.exp()
        * s.powf(gamma)
        * (norm_cdf(d) - (i / s).powf(kappa) * norm_cdf(d - 2.0 * (i / s).ln() / sqt))
}

/// Two-period expectation `psi`: the `S_T^gamma` payoff surviving the
/// far boundary `i2` until `t1` and the near boundary `i1` after, as
/// four bivariate-normal terms (direct, two single images, one double
/// image).
#[allow(clippy::too_many_arguments)]
fn psi(
    s: f64,
    t2: f64,
    gamma: f64,
    h: f64,
    i2: f64,
    i1: f64,
    t1: f64,
    r: f64,
    b: f64,
    sigma: f64,
) -> f64 {
    let mu = b + (gamma - 0.5) * sigma * sigma;
    let (st1, st2) = (sigma * t1.sqrt(), sigma * t2.sqrt());
    let d1 = ((s / i1).ln() + mu * t1) / st1;
    let d2 = ((i2 * i2 / (s * i1)).ln() + mu * t1) / st1;
    let d3 = ((s / i1).ln() - mu * t1) / st1;
    let d4 = ((i2 * i2 / (s * i1)).ln() - mu * t1) / st1;
    let f1 = ((s / h).ln() + mu * t2) / st2;
    let f2 = ((i2 * i2 / (s * h)).ln() + mu * t2) / st2;
    let f3 = ((i1 * i1 / (s * h)).ln() + mu * t2) / st2;
    let f4 = ((s * i1 * i1 / (h * i2 * i2)).ln() + mu * t2) / st2;
    let rho = (t1 / t2).sqrt();
    let lambda = -r + gamma * b + 0.5 * gamma * (gamma - 1.0) * sigma * sigma;
    let kappa = 2.0 * b / (sigma * sigma) + 2.0 * gamma - 1.0;
    (lambda * t2).exp()
        * s.powf(gamma)
        * (bivariate_norm_cdf(-d1, -f1, rho) - (i2 / s).powf(kappa) * bivariate_norm_cdf(-d2, -f2, rho)
            - (i1 / s).powf(kappa) * bivariate_norm_cdf(-d3, -f3, -rho)
            + (i1 / i2).powf(kappa) * bivariate_norm_cdf(-d4, -f4, -rho))
}

// ── EquityOption integration (mirrors the BAW engine) ───────────────────

fn reprice(option: &EquityOption, d_spot: f64, d_vol: f64, d_rate: f64, d_maturity: f64) -> f64 {
    let s = option.base.effective_spot() + d_spot;
    let k = option.base.strike_price;
    let r = option.base.risk_free_rate() + d_rate;
    let q = option.base.carry_yield();
    let sigma = option.base.volatility() + d_vol;
    let t = (option.time_to_maturity() + d_maturity).max(1e-8);
    let pc = *option.payoff.put_or_call();
    match option.payoff.exercise_style() {
        ContractStyle::American => price(s, k, r, q, sigma, t, pc),
        ContractStyle::European => bs_price(s, k, r, q, sigma, t, pc),
    }
}

pub fn npv(option: &EquityOption) -> f64 {
    reprice(option, 0.0, 0.0, 0.0, 0.0)
}

/// Reprice under a market move for portfolio PnL attribution.
pub fn price_with(option: &EquityOption, d_spot: f64, d_vol: f64, d_rate: f64, d_time: f64) -> f64 {
    reprice(option, d_spot, d_vol, d_rate, -d_time)
}

// Greeks by bump-and-reprice on the fast closed form.

fn spot_bump(option: &EquityOption) -> f64 {
    option.base.effective_spot() * 1e-4
}

pub fn delta(option: &EquityOption) -> f64 {
    let h = spot_bump(option);
    (reprice(option, h, 0.0, 0.0, 0.0) - reprice(option, -h, 0.0, 0.0, 0.0)) / (2.0 * h)
}

pub fn gamma(option: &EquityOption) -> f64 {
    let h = spot_bump(option);
    (reprice(option, h, 0.0, 0.0, 0.0) - 2.0 * reprice(option, 0.0, 0.0, 0.0, 0.0)
        + reprice(option, -h, 0.0, 0.0, 0.0))
        / (h * h)
}

pub fn vega(option: &EquityOption) -> f64 {
    let h = 1e-4;
    (reprice(option, 0.0, h, 0.0, 0.0) - reprice(option, 0.0, -h, 0.0, 0.0)) / (2.0 * h)
}

pub fn rho(option: &EquityOption) -> f64 {
    let h = 1e-4;
    (reprice(option, 0.0, 0.0, h, 0.0) - reprice(option, 0.0, 0.0, -h, 0.0)) / (2.0 * h)
}

pub fn theta(option: &EquityOption) -> f64 {
    let h = (1.0 / 365.0_f64).min(0.5 * option.time_to_maturity());
    -(reprice(option, 0.0, 0.0, 0.0, h) - reprice(option, 0.0, 0.0, 0.0, -h)) / (2.0 * h)
}

pub fn vanna(option: &EquityOption) -> f64 {
    let hs = spot_bump(option);
    let hv = 1e-4;
    (reprice(option, hs, hv, 0.0, 0.0) - reprice(option, -hs, hv, 0.0, 0.0)
        - reprice(option, hs, -hv, 0.0, 0.0)
        + reprice(option, -hs, -hv, 0.0, 0.0))
        / (4.0 * hs * hv)
}

pub fn charm(option: &EquityOption) -> f64 {
    let hs = spot_bump(option);
    let ht = (1.0 / 365.0_f64).min(0.5 * option.time_to_maturity());
    -(reprice(option, hs, 0.0, 0.0, ht) - reprice(option, -hs, 0.0, 0.0, ht)
        - reprice(option, hs, 0.0, 0.0, -ht)
        + reprice(option, -hs, 0.0, 0.0, -ht))
        / (4.0 * hs * ht)
}

pub fn zomma(option: &EquityOption) -> f64 {
    let hs = spot_bump(option);
    let hv = 1e-4;
    let gamma_at = |dv: f64| {
        (reprice(option, hs, dv, 0.0, 0.0) - 2.0 * reprice(option, 0.0, dv, 0.0, 0.0)
            + reprice(option, -hs, dv, 0.0, 0.0))
            / (hs * hs)
    };
    (gamma_at(hv) - gamma_at(-hv)) / (2.0 * hv)
}

pub fn volga(option: &EquityOption) -> f64 {
    let hv = 1e-3;
    (reprice(option, 0.0, hv, 0.0, 0.0) - 2.0 * reprice(option, 0.0, 0.0, 0.0, 0.0)
        + reprice(option, 0.0, -hv, 0.0, 0.0))
        / (hv * hv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_values_match_the_validated_reference() {
        // pinned against the cross-checked Python implementation of the
        // published 2002 formulas
        let c = price(100.0, 100.0, 0.10, 0.10, 0.25, 0.5, PutOrCall::Call);
        assert!((c - 6.7661200846).abs() < 1e-6, "call {c}");
        let p = price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, PutOrCall::Put);
        assert!((p - 6.0159338278).abs() < 1e-6, "put {p}");
        let d = price(100.0, 100.0, 0.08, 0.12, 0.30, 3.0, PutOrCall::Call);
        assert!((d - 13.8483486158).abs() < 1e-6, "dividend call {d}");
    }

    #[test]
    fn non_dividend_call_equals_european() {
        for s in [80.0, 100.0, 120.0] {
            let a = price(s, 100.0, 0.05, 0.0, 0.30, 1.0, PutOrCall::Call);
            let e = bs_price(s, 100.0, 0.05, 0.0, 0.30, 1.0, PutOrCall::Call);
            assert!((a - e).abs() < 1e-10, "s={s}");
        }
    }

    // reference binomial tree for accuracy and lower-bound checks
    fn crr(pc: PutOrCall, s: f64, k: f64, r: f64, b: f64, v: f64, t: f64, steps: usize) -> f64 {
        let dt = t / steps as f64;
        let u = (v * dt.sqrt()).exp();
        let d = 1.0 / u;
        let p = ((b * dt).exp() - d) / (u - d);
        let disc = (-r * dt).exp();
        let intrinsic = |sp: f64| match pc {
            PutOrCall::Call => (sp - k).max(0.0),
            PutOrCall::Put => (k - sp).max(0.0),
        };
        let mut v_nodes: Vec<f64> = (0..=steps)
            .map(|j| intrinsic(s * u.powi(j as i32) * d.powi((steps - j) as i32)))
            .collect();
        for i in (0..steps).rev() {
            for j in 0..=i {
                let cont = disc * (p * v_nodes[j + 1] + (1.0 - p) * v_nodes[j]);
                let sp = s * u.powi(j as i32) * d.powi((i - j) as i32);
                v_nodes[j] = cont.max(intrinsic(sp));
            }
        }
        v_nodes[0]
    }

    #[test]
    fn is_a_lower_bound_that_tracks_the_tree() {
        let cases = [
            (PutOrCall::Call, 100.0, 100.0, 0.10, 0.10, 0.25, 0.5),
            (PutOrCall::Call, 110.0, 100.0, 0.10, 0.10, 0.25, 0.5),
            (PutOrCall::Put, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0),
            (PutOrCall::Put, 90.0, 100.0, 0.10, 0.0, 0.25, 0.5),
            (PutOrCall::Call, 100.0, 100.0, 0.08, 0.12, 0.30, 3.0),
            (PutOrCall::Put, 100.0, 100.0, 0.06, 0.02, 0.40, 0.25),
        ];
        for (pc, s, k, r, q, v, t) in cases {
            let approx = price(s, k, r, q, v, t, pc);
            let tree = crr(pc, s, k, r, r - q, v, t, 2000);
            // feasible-strategy value: below the true price, within cents
            assert!(approx <= tree + 5e-3, "{pc:?} s={s}: approx {approx} above tree {tree}");
            assert!(tree - approx < 0.10, "{pc:?} s={s}: approx {approx} vs tree {tree}");
        }
    }

    #[test]
    fn deep_in_the_money_is_intrinsic_or_better() {
        let p = price(55.0, 100.0, 0.10, 0.0, 0.20, 0.5, PutOrCall::Put);
        assert!(p >= 45.0 - 1e-12, "{p}");
        let c = price(250.0, 100.0, 0.10, 0.04, 0.20, 0.5, PutOrCall::Call);
        assert!(c >= 150.0 - 1e-12, "{c}");
    }

    #[test]
    fn engine_dispatch_prices_and_greeks() {
        use crate::core::traits::Instrument;
        use crate::equity::builder::EquityOptionBuilder;
        use crate::equity::utils::Engine;
        use chrono::NaiveDate;

        let build = |engine: Engine| {
            EquityOptionBuilder::new()
                .symbol("ACME")
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.25)
                .flat_rate(0.08)
                .dividend_yield(0.04)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
                .american()
                .vanilla(PutOrCall::Put)
                .engine(engine)
                .build()
        };
        let bs2002 = build(Engine::BjerksundStensland);
        let tree = build(Engine::Binomial);
        assert!((bs2002.npv() - tree.npv()).abs() < 0.05,
            "bs2002 {} vs tree {}", bs2002.npv(), tree.npv());
        // true American Greeks: delta/gamma near the FD engine's values
        let fd = build(Engine::FiniteDifference);
        assert!((bs2002.delta() - fd.delta()).abs() < 0.01,
            "delta {} vs fd {}", bs2002.delta(), fd.delta());
        assert!(bs2002.gamma() > 0.0 && bs2002.vega() > 0.0);
    }
}
