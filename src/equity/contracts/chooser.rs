//! Chooser ("as-you-like-it") options: at the choice date `t1` the
//! holder decides whether the contract becomes a European **call or put**
//! (Rubinstein 1991).
//!
//! **Simple chooser** — both legs share the strike and expiry. By
//! put-call parity the choice value `max(C, P)` at `t1` decomposes into
//! the call plus a parity put, so today's price is two vanilla
//! evaluations:
//!
//! ```text
//! w = C(S, K, T)  +  e^{-q (T - t1)} * P(S, K e^{-(r - q)(T - t1)}, t1)
//! ```
//!
//! Limits recover the familiar contracts: `t1 -> 0` gives `max(C, P)`
//! (the choice is made now), `t1 -> T` gives the straddle `C + P`
//! (choose at expiry). Value increases in `t1` — a later choice is worth
//! more.
//!
//! **Complex chooser** — each leg has its own strike and expiry, so
//! parity no longer collapses the choice. The decision boundary is the
//! critical spot `I` at which the two leg values cross (a short Newton
//! solve, as in the Barone-Adesi-Whaley boundary), and the price is a
//! Geske-style compound assembly of **bivariate normal** terms
//! correlating the spot at the choice date with the spot at each leg's
//! expiry (`rho = sqrt(t1 / T_leg)`). The simple chooser is the
//! degenerate case of equal legs.
//!
//! Prices on the **Analytical engine only** (`check_engine_support`
//! rejects the rest); Greeks come from the central sensitivity engine
//! through the analytic pricer's bump arms.

use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::core::utils::{bivariate_norm_cdf, norm_cdf, ContractStyle};
use crate::equity::blackscholes::bs_price;
use crate::equity::utils::{Payoff, PayoffType};

/// Simple chooser price (Rubinstein 1991): choice at `t1`, both legs
/// struck at `k` expiring at `t`, carry `q` (dividend yield + borrow).
pub fn simple_chooser_price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t1: f64, t: f64) -> f64 {
    let tau = t - t1;
    let call = bs_price(s, k, r, q, sigma, t, PutOrCall::Call);
    // parity put: strike discounted to the choice date at the net carry
    let parity_strike = k * (-(r - q) * tau).exp();
    let put = bs_price(s, parity_strike, r, q, sigma, t1, PutOrCall::Put);
    call + (-q * tau).exp() * put
}

/// Complex chooser price (Rubinstein 1991): at `t1` the holder picks
/// between a call `(k_call, t_call)` and a put `(k_put, t_put)`, all
/// times in years from today with `t1 <= min(t_call, t_put)`.
///
/// The critical spot `I` where the leg values cross at the choice date is
/// solved by Newton (the difference is strictly increasing in spot), then
/// each leg is a "conditional vanilla" received only on its side of `I` —
/// bivariate normal terms over the choice date and the leg expiry.
// the literature's flat signature: spot, two strikes, carry, vol and
// three dates read better positionally than through a struct
#[allow(clippy::too_many_arguments)]
pub fn complex_chooser_price(
    s: f64,
    k_call: f64,
    k_put: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t1: f64,
    t_call: f64,
    t_put: f64,
) -> f64 {
    let b = r - q;
    let half_var = b + 0.5 * sigma * sigma;
    // ── critical spot at the choice date ────────────────────────────────
    let tau_call = (t_call - t1).max(1e-8);
    let tau_put = (t_put - t1).max(1e-8);
    let d1_at = |x: f64, k: f64, tau: f64| ((x / k).ln() + half_var * tau) / (sigma * tau.sqrt());
    let value_gap = |x: f64| {
        bs_price(x, k_call, r, q, sigma, tau_call, PutOrCall::Call)
            - bs_price(x, k_put, r, q, sigma, tau_put, PutOrCall::Put)
    };
    // gap slope = call delta - put delta > 0: Newton is well-behaved
    let gap_slope = |x: f64| {
        (-q * tau_call).exp() * norm_cdf(d1_at(x, k_call, tau_call))
            + (-q * tau_put).exp() * norm_cdf(-d1_at(x, k_put, tau_put))
    };
    let seed = 0.5 * (k_call + k_put);
    let critical = Solver1d::new(1e-10 * k_call.max(k_put), 100)
        .newton_raphson(value_gap, gap_slope, seed)
        .x;

    // ── Geske-style assembly over (choice date, leg expiry) ─────────────
    let st1 = sigma * t1.sqrt();
    let d1 = ((s / critical).ln() + half_var * t1) / st1;
    let d2 = d1 - st1;
    let y_call = ((s / k_call).ln() + half_var * t_call) / (sigma * t_call.sqrt());
    let y_put = ((s / k_put).ln() + half_var * t_put) / (sigma * t_put.sqrt());
    let rho_call = (t1 / t_call).sqrt();
    let rho_put = (t1 / t_put).sqrt();
    s * (-q * t_call).exp() * bivariate_norm_cdf(d1, y_call, rho_call)
        - k_call
            * (-r * t_call).exp()
            * bivariate_norm_cdf(d2, y_call - sigma * t_call.sqrt(), rho_call)
        - s * (-q * t_put).exp() * bivariate_norm_cdf(-d1, -y_put, rho_put)
        + k_put
            * (-r * t_put).exp()
            * bivariate_norm_cdf(-d2, -y_put + sigma * t_put.sqrt(), rho_put)
}

/// Per-leg terms of a **complex** chooser. Expiries are fractions of the
/// option's time to maturity (1.0 = the maturity date), strictly after
/// the choice fraction.
#[derive(Debug, Clone, Copy)]
pub struct ChooserLegs {
    pub call_strike: f64,
    pub put_strike: f64,
    /// Call-leg expiry as a fraction of the maturity, in (choice, 1].
    pub call_expiry_fraction: f64,
    /// Put-leg expiry as a fraction of the maturity, in (choice, 1].
    pub put_expiry_fraction: f64,
}

/// The chooser contract: the choice date is stored as a fraction of the
/// (build-time) life, like the forward-start fixing. `legs: None` is the
/// simple chooser (both legs share the option's strike and expiry);
/// `Some` gives each leg its own strike and expiry.
#[derive(Debug, Clone)]
pub struct ChooserPayoff {
    pub exercise_style: ContractStyle,
    /// Choice time as a fraction of the time to maturity, in (0, 1).
    pub choice_fraction: f64,
    /// Per-leg strikes and expiries of a complex chooser.
    pub legs: Option<ChooserLegs>,
}

impl Payoff for ChooserPayoff {
    /// Intrinsic were the choice made at expiry (display/premium-at-risk
    /// only; pricing is closed-form): the straddle `|S - K|` for the
    /// simple chooser, the better leg intrinsic for the complex one.
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match self.legs {
            None => (spot - strike).abs(),
            Some(legs) => (spot - legs.call_strike)
                .max(0.0)
                .max((legs.put_strike - spot).max(0.0)),
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Chooser
    }
    /// A chooser has no side until the choice date; reported as a call by
    /// convention.
    fn put_or_call(&self) -> &PutOrCall {
        &PutOrCall::Call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: f64 = 50.0;
    const K: f64 = 50.0;
    const R: f64 = 0.08;
    const SIGMA: f64 = 0.25;

    #[test]
    fn golden_value_matches_the_published_reference() {
        // Haug, "The Complete Guide to Option Pricing Formulas": simple
        // chooser with S=50, K=50, t1=0.25, T=0.5, r=b=8%, sigma=25%
        let w = simple_chooser_price(S, K, R, 0.0, SIGMA, 0.25, 0.5);
        assert!((w - 6.1071).abs() < 1e-3, "chooser {w} vs Haug 6.1071");
    }

    #[test]
    fn late_choice_is_a_straddle_and_early_choice_is_the_better_vanilla() {
        let t = 0.5;
        let call = bs_price(S, K, R, 0.0, SIGMA, t, PutOrCall::Call);
        let put = bs_price(S, K, R, 0.0, SIGMA, t, PutOrCall::Put);
        // t1 -> T: the holder chooses at expiry, i.e. holds a straddle
        let late = simple_chooser_price(S, K, R, 0.0, SIGMA, t, t);
        assert!(
            (late - (call + put)).abs() < 1e-12,
            "late {late} vs straddle {}",
            call + put
        );
        // t1 -> 0: the choice is made now — the better of call and put
        let early = simple_chooser_price(S, K, R, 0.0, SIGMA, 1e-9, t);
        assert!(
            (early - call.max(put)).abs() < 1e-6,
            "early {early} vs max {}",
            call.max(put)
        );
    }

    #[test]
    fn value_increases_with_the_choice_date_and_is_bounded_by_the_straddle() {
        let t = 1.0;
        let call = bs_price(S, K, R, 0.02, SIGMA, t, PutOrCall::Call);
        let put = bs_price(S, K, R, 0.02, SIGMA, t, PutOrCall::Put);
        let mut previous = call.max(put);
        for t1 in [0.1, 0.25, 0.5, 0.75, 1.0] {
            let w = simple_chooser_price(S, K, R, 0.02, SIGMA, t1, t);
            assert!(
                w >= previous - 1e-12,
                "must be monotone in t1: {w} < {previous}"
            );
            assert!(w <= call + put + 1e-12, "straddle bounds the chooser");
            previous = w;
        }
    }

    #[test]
    fn complex_golden_value_matches_the_published_reference() {
        // Haug: complex chooser with S=50, Kc=55, Kp=48, t1=0.25,
        // Tc=0.5, Tp=0.5833, r=10%, b=5% (q=5%), sigma=35%
        let w = complex_chooser_price(50.0, 55.0, 48.0, 0.10, 0.05, 0.35, 0.25, 0.5, 0.5833);
        assert!(
            (w - 6.0508).abs() < 1e-3,
            "complex chooser {w} vs Haug 6.0508"
        );
    }

    #[test]
    fn complex_with_equal_legs_degenerates_to_the_simple_chooser() {
        let (t1, t) = (0.25, 0.75);
        let simple = simple_chooser_price(S, K, R, 0.02, SIGMA, t1, t);
        let complex = complex_chooser_price(S, K, K, R, 0.02, SIGMA, t1, t, t);
        assert!(
            (simple - complex).abs() < 1e-6,
            "simple {simple} vs degenerate complex {complex}"
        );
    }

    #[test]
    fn complex_value_sits_between_the_better_leg_and_both_legs() {
        let (k_call, k_put) = (55.0, 48.0);
        let (t1, t_call, t_put) = (0.25, 0.5, 0.75);
        let call = bs_price(S, k_call, R, 0.02, SIGMA, t_call, PutOrCall::Call);
        let put = bs_price(S, k_put, R, 0.02, SIGMA, t_put, PutOrCall::Put);
        let w = complex_chooser_price(S, k_call, k_put, R, 0.02, SIGMA, t1, t_call, t_put);
        assert!(
            w > call.max(put),
            "{w} must beat the better leg {}",
            call.max(put)
        );
        assert!(
            w < call + put,
            "{w} must be cheaper than owning both legs {}",
            call + put
        );
    }

    fn chooser_builder() -> crate::equity::builder::EquityOptionBuilder {
        use chrono::NaiveDate;
        crate::equity::builder::EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .chooser(0.5)
    }

    #[test]
    fn option_level_pricing_greeks_and_engine_guard() {
        use crate::core::traits::Instrument;
        use crate::equity::utils::Engine;
        let option = chooser_builder()
            .engine(Engine::BlackScholes)
            .build()
            .unwrap();
        // ATM chooser: worth more than either vanilla, less than the straddle
        let call = chooser_builder().vanilla(PutOrCall::Call).build().unwrap();
        let put = chooser_builder().vanilla(PutOrCall::Put).build().unwrap();
        let w = option.npv();
        assert!(w > call.npv().max(put.npv()), "{w} vs vanillas");
        assert!(w < call.npv() + put.npv(), "{w} vs straddle");
        // greeks: near-symmetric ATM chooser is long gamma and vega, and
        // the batch path equals the accessors
        assert!(option.gamma() > 0.0);
        assert!(option.vega() > 0.0);
        assert!(option.delta().abs() < 1.0);
        let result = option.price().unwrap();
        assert_eq!(result.pv, w);
        assert_eq!(result.greeks.delta, option.delta());
        assert_eq!(result.greeks.vega, option.vega());
        // other engines are rejected (the builder validates engine support)
        assert!(
            chooser_builder()
                .engine(Engine::MonteCarlo)
                .build()
                .is_err(),
            "chooser must be Analytical-only"
        );
    }

    #[test]
    fn complex_option_level_pricing_and_batch_greeks() {
        use crate::core::traits::Instrument;
        use crate::equity::utils::Engine;
        // strangle-like chooser: 105 call to maturity, 95 put to mid-life
        let option = crate::equity::builder::EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(chrono::NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(chrono::NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .complex_chooser(0.25, 105.0, 1.0, 95.0, 0.5)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap();
        let w = option.npv();
        assert!(w > 0.0);
        assert!(option.gamma() > 0.0);
        assert!(option.vega() > 0.0);
        let result = option.price().unwrap();
        assert_eq!(result.pv, w);
        assert_eq!(result.greeks.delta, option.delta());
        // leg expiry at or before the choice date must fail validation
        let bad = chooser_builder();
        assert!(bad
            .complex_chooser(0.5, 105.0, 0.4, 95.0, 1.0)
            .build()
            .is_err());
    }

    #[test]
    fn builds_from_json_contract_data() {
        use crate::core::data_models::EquityOptionData;
        use crate::core::traits::Instrument;
        let json = r#"{
            "symbol": "ACME",
            "underlying_price": 100.0,
            "valuation_date": "2026-01-05",
            "put_or_call": "C",
            "payoff_type": "chooser",
            "strike_price": 100.0,
            "maturity": "2027-01-04",
            "choice_date": "2026-07-05",
            "volatility": 0.25,
            "risk_free_rate": 0.03,
            "pricer": "Analytical"
        }"#;
        let data: EquityOptionData = serde_json::from_str(json).expect("contract must parse");
        let option = crate::equity::vanilla_option::EquityOption::try_from_json(&data)
            .expect("chooser must build from JSON");
        let w = option.npv();
        assert!(w > 0.0);
        // the fraction is the calendar ratio of (choice - valuation) to life
        let payoff = option
            .payoff
            .as_any()
            .downcast_ref::<ChooserPayoff>()
            .unwrap();
        assert!(
            (payoff.choice_fraction - 0.5).abs() < 5e-3,
            "{}",
            payoff.choice_fraction
        );
        // a choice date outside the option's life is rejected
        let mut bad = data.clone();
        bad.choice_date = Some("2027-06-01".to_string());
        assert!(crate::equity::vanilla_option::EquityOption::try_from_json(&bad).is_err());
        // per-leg fields upgrade the contract to a complex chooser, with
        // absent legs defaulting to the option's strike and maturity
        let mut cx = data.clone();
        cx.chooser_call_strike = Some(105.0);
        cx.chooser_put_strike = Some(95.0);
        cx.chooser_put_expiry = Some("2026-10-05".to_string());
        let complex = crate::equity::vanilla_option::EquityOption::try_from_json(&cx).unwrap();
        let legs = complex
            .payoff
            .as_any()
            .downcast_ref::<ChooserPayoff>()
            .unwrap()
            .legs
            .expect("per-leg fields must build a complex chooser");
        assert_eq!(legs.call_strike, 105.0);
        assert_eq!(legs.put_strike, 95.0);
        assert!(
            (legs.call_expiry_fraction - 1.0).abs() < 1e-12,
            "call leg defaults to maturity"
        );
        assert!(legs.put_expiry_fraction > 0.5 && legs.put_expiry_fraction < 1.0);
        assert!(complex.npv() > 0.0);
    }
}
