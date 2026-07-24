//! Barone-Adesi & Whaley (1987) quadratic approximation for American options.
//!
//! A fast, closed-form-ish alternative to a tree or PDE for American vanillas:
//! the early-exercise premium is approximated by the dominant term of the
//! quadratic (MacMillan) approximation to the American PDE, so pricing is a
//! European Black-Scholes evaluation plus a short Newton solve for the
//! critical exercise price. Accuracy is a few cents versus a fine binomial
//! tree for typical parameters — use it when speed matters more than the last
//! basis point (e.g. a large book revalued many times), and a tree/FD solve
//! when precision is paramount.
//!
//! Everything is expressed with a continuous cost of carry `b = r - q`, where
//! `q` is the total carry (dividend yield plus borrow). With `b = r - q` the
//! generalized Black-Scholes formula is exactly
//! [`bs_price`](crate::equity::blackscholes::bs_price), so the European leg is
//! shared with the rest of the library.
//!
//! Key properties the implementation preserves:
//! - an American call on a non-dividend payer (`b >= r`, i.e. `q <= 0`) is
//!   never exercised early, so it equals the European call;
//! - the price is bounded below by intrinsic value and by the European price
//!   (the early-exercise premium is non-negative).

use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::core::utils::{norm_pdf, ContractStyle, norm_cdf};
use crate::equity::blackscholes::bs_price;
use crate::equity::vanilla_option::EquityOption;

/// Relative convergence tolerance (in units of strike) for the critical-price
/// Newton iteration, and its iteration cap.
const CRIT_TOL: f64 = 1e-6;
const CRIT_MAX_ITER: usize = 100;

/// American vanilla price via the Barone-Adesi-Whaley approximation.
///
/// `q` is the total continuous carry, so the cost of carry is `b = r - q`.
/// Degenerate inputs (`t <= 0` or `sigma <= 0`) return intrinsic value.
pub fn price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, put_or_call: PutOrCall) -> f64 {
    let intrinsic = match put_or_call {
        PutOrCall::Call => (s - k).max(0.0),
        PutOrCall::Put => (k - s).max(0.0),
    };
    if t <= 0.0 || sigma <= 0.0 {
        return intrinsic;
    }
    let b = r - q;
    let euro = bs_price(s, k, r, q, sigma, t, put_or_call);

    match put_or_call {
        PutOrCall::Call => {
            // never optimal to exercise a call early when b >= r
            if b >= r {
                return euro;
            }
            let s_star = critical_call(k, r, b, sigma, t);
            if s >= s_star {
                return intrinsic;
            }
            let q2 = quadratic_root(r, b, sigma, t, true);
            let d1 = d1_of(s_star, k, b, sigma, t);
            let a2 = (s_star / q2) * (1.0 - ((b - r) * t).exp() * norm_cdf(d1));
            euro + a2 * (s / s_star).powf(q2)
        }
        PutOrCall::Put => {
            let s_star = critical_put(k, r, b, sigma, t);
            if s <= s_star {
                return intrinsic;
            }
            let q1 = quadratic_root(r, b, sigma, t, false);
            let d1 = d1_of(s_star, k, b, sigma, t);
            let a1 = -(s_star / q1) * (1.0 - ((b - r) * t).exp() * norm_cdf(-d1));
            euro + a1 * (s / s_star).powf(q1)
        }
    }
}

/// Early-exercise premium: the BAW American price minus the European price.
/// Non-negative by construction.
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

fn d1_of(s: f64, k: f64, b: f64, sigma: f64, t: f64) -> f64 {
    ((s / k).ln() + (b + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
}

/// The `q1` (put, `+` root false) or `q2` (call, `+` root true) exponent of
/// the quadratic approximation, using the finite-maturity `K` factor.
fn quadratic_root(r: f64, b: f64, sigma: f64, t: f64, call: bool) -> f64 {
    let n = 2.0 * b / (sigma * sigma);
    let kf = 2.0 * r / (sigma * sigma * (1.0 - (-r * t).exp()));
    let disc = ((n - 1.0).powi(2) + 4.0 * kf).sqrt();
    if call {
        (-(n - 1.0) + disc) / 2.0
    } else {
        (-(n - 1.0) - disc) / 2.0
    }
}

/// Critical (early-exercise boundary) spot for an American call, solved by
/// Newton-Raphson on the value-matching residual `(S* - K) - RHS(S*)`; the
/// Barone-Adesi-Whaley update `(K + RHS - b_i S) / (1 - b_i)` is exactly the
/// Newton step for this residual with slope `1 - b_i`.
fn critical_call(k: f64, r: f64, b: f64, sigma: f64, t: f64) -> f64 {
    let n = 2.0 * b / (sigma * sigma);
    let m = 2.0 * r / (sigma * sigma);
    let q2u = (-(n - 1.0) + ((n - 1.0).powi(2) + 4.0 * m).sqrt()) / 2.0;
    let su = k / (1.0 - 1.0 / q2u); // perpetual (infinite-maturity) boundary
    let h2 = -(b * t + 2.0 * sigma * t.sqrt()) * k / (su - k);
    let seed = k + (su - k) * (1.0 - h2.exp());

    let q2 = quadratic_root(r, b, sigma, t, true);
    let sqt = sigma * t.sqrt();
    let rhs = |si: f64| {
        let d1 = d1_of(si, k, b, sigma, t);
        bs_price(si, k, r, r - b, sigma, t, PutOrCall::Call)
            + (1.0 - ((b - r) * t).exp() * norm_cdf(d1)) * si / q2
    };
    // slope b_i of RHS from the Barone-Adesi-Whaley paper
    let bi = |si: f64| {
        let d1 = d1_of(si, k, b, sigma, t);
        ((b - r) * t).exp() * norm_cdf(d1) * (1.0 - 1.0 / q2)
            + (1.0 - ((b - r) * t).exp() * norm_pdf(d1) / sqt) / q2
    };
    Solver1d::new(CRIT_TOL * k, CRIT_MAX_ITER)
        .newton_raphson(|si| (si - k) - rhs(si), |si| 1.0 - bi(si), seed)
        .x
}

/// Critical (early-exercise boundary) spot for an American put; Newton on
/// the residual `(K - S*) - RHS(S*)` with slope `-(1 + b_i)`.
fn critical_put(k: f64, r: f64, b: f64, sigma: f64, t: f64) -> f64 {
    let n = 2.0 * b / (sigma * sigma);
    let m = 2.0 * r / (sigma * sigma);
    let q1u = (-(n - 1.0) - ((n - 1.0).powi(2) + 4.0 * m).sqrt()) / 2.0;
    let su = k / (1.0 - 1.0 / q1u);
    let h1 = (b * t - 2.0 * sigma * t.sqrt()) * k / (k - su);
    let seed = su + (k - su) * h1.exp();

    let q1 = quadratic_root(r, b, sigma, t, false);
    let sqt = sigma * t.sqrt();
    let rhs = |si: f64| {
        let d1 = d1_of(si, k, b, sigma, t);
        bs_price(si, k, r, r - b, sigma, t, PutOrCall::Put)
            - (1.0 - ((b - r) * t).exp() * norm_cdf(-d1)) * si / q1
    };
    let bi = |si: f64| {
        let d1 = d1_of(si, k, b, sigma, t);
        -((b - r) * t).exp() * norm_cdf(-d1) * (1.0 - 1.0 / q1)
            - (1.0 + ((b - r) * t).exp() * norm_pdf(-d1) / sqt) / q1
    };
    Solver1d::new(CRIT_TOL * k, CRIT_MAX_ITER)
        .newton_raphson(|si| (k - si) - rhs(si), |si| -1.0 - bi(si), seed)
        .x
}

// ── EquityOption integration ────────────────────────────────────────────
// Flat Black-Scholes inputs are read the same way the analytic vanilla
// pricer reads them: escrowed spot (cash dividends carved out), the curve's
// continuous zero rate, the total carry, and the surface vol at this strike.

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
        // BAW on a European contract is just the European price
        ContractStyle::European => bs_price(s, k, r, q, sigma, t, pc),
    }
}

pub fn npv(option: &EquityOption) -> f64 {
    reprice(option, 0.0, 0.0, 0.0, 0.0)
}

/// Critical early-exercise spot for this option (the BAW boundary `S*`).
pub fn critical_spot(option: &EquityOption) -> f64 {
    let (k, r, b, sigma, t) = (
        option.base.strike_price,
        option.base.risk_free_rate(),
        option.base.risk_free_rate() - option.base.carry_yield(),
        option.base.volatility(),
        option.time_to_maturity(),
    );
    match option.payoff.put_or_call() {
        PutOrCall::Call => critical_call(k, r, b, sigma, t),
        PutOrCall::Put => critical_put(k, r, b, sigma, t),
    }
}

/// Reprice under a market move for portfolio PnL attribution: spot `+ d_spot`,
/// a parallel vol shift `+ d_vol`, rate `+ d_rate`, and `d_time` years of
/// elapsed calendar time (which shortens maturity).
pub fn price_with(option: &EquityOption, d_spot: f64, d_vol: f64, d_rate: f64, d_time: f64) -> f64 {
    reprice(option, d_spot, d_vol, d_rate, -d_time)
}

// Greeks by bump-and-reprice on the fast closed form. The American price is
// smooth below the exercise boundary, so central differences are well-behaved.

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
    // calendar theta = dV/dt = -dV/dT
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
    // charm = -d(delta)/dT; the maturity bump is +ht
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
    fn golden_values_match_reference() {
        // American put S=100 K=100 T=1 r=5% q=0 (b=5%) sigma=20%
        let p = price(100.0, 100.0, 0.05, 0.0, 0.20, 1.0, PutOrCall::Put);
        assert!((p - 6.09762).abs() < 1e-4, "put {p}");
        // American call S=100 K=100 T=0.5 r=10% q=10% (b=0) sigma=25%
        let c = price(100.0, 100.0, 0.10, 0.10, 0.25, 0.5, PutOrCall::Call);
        assert!((c - 6.80134).abs() < 1e-4, "call {c}");
    }

    #[test]
    fn non_dividend_call_equals_european() {
        // q = 0 => b = r, an American call is never exercised early
        for s in [80.0, 100.0, 120.0] {
            let a = price(s, 100.0, 0.05, 0.0, 0.30, 1.0, PutOrCall::Call);
            let e = bs_price(s, 100.0, 0.05, 0.0, 0.30, 1.0, PutOrCall::Call);
            assert!((a - e).abs() < 1e-10, "s={s} amer {a} euro {e}");
        }
    }

    #[test]
    fn premium_is_non_negative_and_bounded_by_intrinsic() {
        for &pc in &[PutOrCall::Call, PutOrCall::Put] {
            for s in [70.0, 85.0, 100.0, 115.0, 130.0] {
                let a = price(s, 100.0, 0.08, 0.04, 0.25, 0.75, pc);
                let e = bs_price(s, 100.0, 0.08, 0.04, 0.25, 0.75, pc);
                let intrinsic = match pc {
                    PutOrCall::Call => (s - 100.0).max(0.0),
                    PutOrCall::Put => (100.0 - s).max(0.0),
                };
                assert!(a >= e - 1e-9, "american {a} below european {e}");
                assert!(a >= intrinsic - 1e-9, "american {a} below intrinsic {intrinsic}");
            }
        }
    }

    #[test]
    fn deep_in_the_money_put_is_intrinsic() {
        // far below the exercise boundary: exercise now, value = K - S
        let p = price(60.0, 100.0, 0.10, 0.0, 0.20, 0.5, PutOrCall::Put);
        assert!((p - 40.0).abs() < 1e-6, "{p}");
    }

    // simple reference binomial tree for cross-checking the approximation
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
    fn engine_prices_and_greeks_match_binomial() {
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
        let euro = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.08)
            .dividend_yield(0.04)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2026, 7, 2).unwrap())
            .vanilla(PutOrCall::Put)
            .engine(Engine::BlackScholes)
            .build();

        let baw_opt = build(Engine::BaroneAdesiWhaley);
        let tree = build(Engine::Binomial);
        let fd = build(Engine::FiniteDifference);

        // the approximation lands within a couple of cents of the tree
        assert!((baw_opt.npv() - tree.npv()).abs() < 0.05,
            "baw {} vs tree {}", baw_opt.npv(), tree.npv());
        // and it carries a genuine early-exercise premium over European
        assert!(baw_opt.npv() > euro.npv(), "baw {} not above euro {}", baw_opt.npv(), euro.npv());
        // BAW reports true American Greeks; check delta/gamma against the FD
        // grid (which also solves the American problem), not the tree (whose
        // Greeks fall back to the European closed form)
        assert!(baw_opt.delta() < 0.0 && baw_opt.delta() > -1.0);
        assert!(baw_opt.gamma() > 0.0);
        assert!(baw_opt.vega() > 0.0);
        assert!((baw_opt.delta() - fd.delta()).abs() < 0.01,
            "baw delta {} vs fd {}", baw_opt.delta(), fd.delta());
        assert!((baw_opt.gamma() - fd.gamma()).abs() < 0.01,
            "baw gamma {} vs fd {}", baw_opt.gamma(), fd.gamma());
    }

    #[test]
    fn tracks_binomial_within_a_few_cents() {
        // (pc, s, k, r, q, v, t): b = r - q
        let cases = [
            (PutOrCall::Put, 100.0, 100.0, 0.05, 0.0, 0.20, 1.0),
            (PutOrCall::Put, 100.0, 100.0, 0.10, 0.0, 0.25, 0.5),
            (PutOrCall::Call, 110.0, 100.0, 0.10, 0.10, 0.25, 0.5),
            (PutOrCall::Put, 95.0, 100.0, 0.08, 0.03, 0.30, 0.25),
            (PutOrCall::Call, 100.0, 100.0, 0.06, 0.09, 0.20, 1.0),
        ];
        for (pc, s, k, r, q, v, t) in cases {
            let baw = price(s, k, r, q, v, t, pc);
            let tree = crr(pc, s, k, r, r - q, v, t, 3000);
            assert!(
                (baw - tree).abs() < 0.05,
                "{pc:?} s={s}: baw {baw} vs tree {tree}"
            );
        }
    }
}
