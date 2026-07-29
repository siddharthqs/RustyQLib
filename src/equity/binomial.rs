//! Binomial lattice engine for equity options: a thin adapter over the
//! asset-class-agnostic [`core::lattice`](crate::core::lattice) framework.
//!
//! The tree parameterization and step count come from the option's
//! [`LatticeConfig`](crate::core::lattice::LatticeConfig) (`tree_type` /
//! `tree_steps` in JSON, `.tree_type()` / `.tree_steps()` on the builder).
//! The default is **Leisen-Reimer** at 1001 steps — strictly more accurate
//! than the historical CRR-1000 default at the same cost; select `CRR`
//! explicitly to reproduce the classic tree. European, American and Bermudan exercise
//! all price on the optimized rolling-array engine;
//! [`npv_with_diagnostics`] runs the debug engine instead, keeping the
//! full trees, the exercise boundary, tree Greeks and timing.
//!
//! Greeks are engine-consistent (so American and Bermudan sensitivities
//! reflect the exercise boundary, not a European closed form):
//! [`solution`] reads value, delta, gamma and theta off **one** backward
//! pass, and [`pricing_result`] adds the bump-based Greeks (vega, rho,
//! vanna, charm, zomma) from a shared set of shifted trees — seven passes
//! for the value plus all nine Greeks, instead of a rebuild per Greek.

use crate::core::lattice::{
    price_backward, price_backward_with_greeks, price_with_diagnostics, LatticeDiagnostics,
    LatticeParams, LatticeSolution, TermLattice,
};
use crate::core::results::{Greeks, PricingResult};
use crate::core::utils::{times_to_grid_steps, ContractStyle};
use super::vanilla_option::EquityOption;

struct TreeSetup {
    params: LatticeParams,
    n: usize,
    df_step: f64,
    dt: f64,
    s0: f64,
    t: f64,
}

/// Tree build under a shifted market — spot `+ d_spot`, a parallel vol
/// shift `+ d_vol`, rate `+ d_rate`, `d_time` years of elapsed calendar
/// time. All-zero shifts reproduce the base tree bit for bit.
fn setup_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> TreeSetup {
    assert!(option.market.spot.value >= 0.0);
    let t = (option.time_to_maturity() - d_time).max(1e-6);
    let r = option.risk_free_rate() + d_rate;
    let b = r - option.carry_yield();
    let s0 = option.effective_spot() + d_spot;
    let cfg = option.lattice_cfg();
    let n = cfg.tree_type.effective_steps(cfg.steps);
    let params = cfg
        .tree_type
        .params(
            s0,
            option.base.strike_price,
            b,
            option.volatility() + d_vol,
            t,
            n,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let dt = t / n as f64;
    TreeSetup { params, n, df_step: (-r * dt).exp(), dt, s0, t }
}

/// Early-exercise rule for the option's style: intrinsic-vs-continuation
/// at every layer (American), only at mapped layers (Bermudan), or none.
fn exercise_rule<'a>(
    option: &'a EquityOption,
    t: f64,
    n: usize,
) -> Option<Box<dyn Fn(usize, f64, f64) -> f64 + 'a>> {
    let strike = option.base.strike_price;
    match option.payoff.exercise_style() {
        ContractStyle::European => None,
        ContractStyle::American => Some(Box::new(move |_i, spot, cont| {
            option.payoff.payoff(spot, strike).max(cont)
        })),
        ContractStyle::Bermudan(times) => {
            let mut exercisable = vec![false; n + 1];
            for idx in times_to_grid_steps(times, t, n) {
                exercisable[idx] = true;
            }
            Some(Box::new(move |i, spot, cont| {
                if exercisable[i] {
                    option.payoff.payoff(spot, strike).max(cont)
                } else {
                    cont
                }
            }))
        }
    }
}

/// Term-structure lattice: forward rates from the option's discount
/// curve, its carry, and the vol surface's term structure at the strike,
/// applied per step on a variance-equal time grid. The market shifts
/// enter the same way as in [`setup_with`]: parallel on the forward
/// rates and the implied surface, additive on spot, elapsed on time.
/// One pass yields the value and the tree delta/gamma/theta together.
fn term_solution_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> LatticeSolution {
    let t = (option.time_to_maturity() - d_time).max(1e-6);
    let s0 = option.effective_spot() + d_spot;
    let strike = option.base.strike_price;
    let carry = option.carry_yield();
    let curve = &option.market.discount_curve;
    let forward_rate =
        |t1: f64, t2: f64| (curve.df(t1) / curve.df(t2)).ln() / (t2 - t1) + d_rate;
    let forward_carry = |_: f64, _: f64| carry;
    let total_variance = |tt: f64| {
        if tt <= 1e-12 {
            return 0.0;
        }
        // strike-frozen implied term structure: sigma(K, t)^2 * t
        let fwd = s0 / curve.df(tt) * (-carry * tt).exp();
        let sigma = option.market.vol_surface.vol(strike, fwd, tt) + d_vol;
        sigma * sigma * tt
    };
    let lattice =
        TermLattice::build(option.lattice_cfg().steps, t, &forward_rate, &forward_carry, &total_variance)
            .unwrap_or_else(|e| panic!("{e}"));
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    match option.payoff.exercise_style() {
        ContractStyle::European => lattice.price_with_greeks(s0, &terminal, None),
        ContractStyle::American => {
            let ex = |_: usize, _: f64, spot: f64, cont: f64| {
                option.payoff.payoff(spot, strike).max(cont)
            };
            lattice.price_with_greeks(s0, &terminal, Some(&ex))
        }
        ContractStyle::Bermudan(times) => {
            // unequal layer times: map each exercise date to the nearest
            // interior layer by calendar time
            let n = lattice.steps();
            let mut exercisable = vec![false; n];
            for tm in times {
                let mut best = 1usize;
                for i in 1..n {
                    if (lattice.times[i] - tm).abs() < (lattice.times[best] - tm).abs() {
                        best = i;
                    }
                }
                exercisable[best] = true;
            }
            let ex = move |i: usize, _: f64, spot: f64, cont: f64| {
                if exercisable[i] {
                    option.payoff.payoff(spot, strike).max(cont)
                } else {
                    cont
                }
            };
            lattice.price_with_greeks(s0, &terminal, Some(&ex))
        }
    }
}

/// Lattice price on the optimized rolling-array engine; routes to the
/// term-structure lattice when `lattice.term_structure` is set.
pub fn npv(option: &EquityOption) -> f64 {
    npv_with(option, 0.0, 0.0, 0.0, 0.0)
}

/// Lattice price under a shifted market (spot / parallel vol / rate /
/// elapsed time) — the bump machinery behind the higher-order Greeks and
/// the PnL-attribution reprice. Zero shifts equal [`npv`] bit for bit.
pub(crate) fn npv_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> f64 {
    if option.lattice_cfg().term_structure {
        return term_solution_with(option, d_spot, d_vol, d_rate, d_time).price;
    }
    let s = setup_with(option, d_spot, d_vol, d_rate, d_time);
    let strike = option.base.strike_price;
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    let exercise = exercise_rule(option, s.t, s.n);
    price_backward(s.s0, &s.params, s.n, s.df_step, &terminal, exercise.as_deref())
}

/// Value and the tree delta/gamma/theta from **one** backward pass, on
/// both the uniform and the term-structure lattice. Same price as
/// [`npv`], bit for bit.
pub fn solution(option: &EquityOption) -> LatticeSolution {
    solution_with(option, 0.0, 0.0, 0.0, 0.0)
}

fn solution_with(
    option: &EquityOption,
    d_spot: f64,
    d_vol: f64,
    d_rate: f64,
    d_time: f64,
) -> LatticeSolution {
    if option.lattice_cfg().term_structure {
        return term_solution_with(option, d_spot, d_vol, d_rate, d_time);
    }
    let s = setup_with(option, d_spot, d_vol, d_rate, d_time);
    let strike = option.base.strike_price;
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    let exercise = exercise_rule(option, s.t, s.n);
    price_backward_with_greeks(
        s.s0,
        &s.params,
        s.n,
        s.dt,
        s.df_step,
        &terminal,
        exercise.as_deref(),
    )
}

// Bump sizes shared with the finite-difference engine's Greeks.
const VOL_BUMP: f64 = 1e-3;
const RATE_BUMP: f64 = 1e-4;
const VOLGA_BUMP: f64 = 1e-2;

pub fn delta(option: &EquityOption) -> f64 {
    solution(option).delta
}
pub fn gamma(option: &EquityOption) -> f64 {
    solution(option).gamma
}
pub fn theta(option: &EquityOption) -> f64 {
    solution(option).theta
}
pub fn vega(option: &EquityOption) -> f64 {
    let h = VOL_BUMP;
    (npv_with(option, 0.0, h, 0.0, 0.0) - npv_with(option, 0.0, -h, 0.0, 0.0)) / (2.0 * h)
}
pub fn rho(option: &EquityOption) -> f64 {
    let h = RATE_BUMP;
    (npv_with(option, 0.0, 0.0, h, 0.0) - npv_with(option, 0.0, 0.0, -h, 0.0)) / (2.0 * h)
}

/// Vanna from the change in the tree delta under a parallel vol bump.
pub fn vanna(option: &EquityOption) -> f64 {
    let h = VOL_BUMP;
    (solution_with(option, 0.0, h, 0.0, 0.0).delta
        - solution_with(option, 0.0, -h, 0.0, 0.0).delta)
        / (2.0 * h)
}

/// Charm from the spot derivative of the tree's calendar theta.
pub fn charm(option: &EquityOption) -> f64 {
    let h = option.market.spot.value() * 1e-3;
    (solution_with(option, h, 0.0, 0.0, 0.0).theta
        - solution_with(option, -h, 0.0, 0.0, 0.0).theta)
        / (2.0 * h)
}

/// Zomma from the change in the tree gamma under a parallel vol bump.
pub fn zomma(option: &EquityOption) -> f64 {
    let h = VOL_BUMP;
    (solution_with(option, 0.0, h, 0.0, 0.0).gamma
        - solution_with(option, 0.0, -h, 0.0, 0.0).gamma)
        / (2.0 * h)
}

/// Volga as the second price derivative under a parallel vol bump (the
/// larger step tempers roundoff in the second difference).
pub fn volga(option: &EquityOption) -> f64 {
    let h = VOLGA_BUMP;
    (npv_with(option, 0.0, h, 0.0, 0.0) - 2.0 * npv(option)
        + npv_with(option, 0.0, -h, 0.0, 0.0))
        / (h * h)
}

/// Value and all nine Greeks from a **shared** set of tree passes instead
/// of a rebuild per Greek: the base pass yields the price plus
/// delta/gamma/theta for free, the two vol-bumped passes yield vega, vanna
/// and zomma together, two rate bumps yield rho, and two spot-bumped
/// passes yield charm — seven passes in total.
pub fn pricing_result(option: &EquityOption) -> PricingResult {
    let base = solution(option);
    let hv = VOL_BUMP;
    let vol_up = solution_with(option, 0.0, hv, 0.0, 0.0);
    let vol_down = solution_with(option, 0.0, -hv, 0.0, 0.0);
    let hr = RATE_BUMP;
    let rho = (npv_with(option, 0.0, 0.0, hr, 0.0) - npv_with(option, 0.0, 0.0, -hr, 0.0))
        / (2.0 * hr);
    let hs = option.market.spot.value() * 1e-3;
    let charm = (solution_with(option, hs, 0.0, 0.0, 0.0).theta
        - solution_with(option, -hs, 0.0, 0.0, 0.0).theta)
        / (2.0 * hs);
    let gamma_p = if base.delta == 0.0 {
        f64::NAN
    } else {
        option.market.spot.value() * base.gamma / base.delta
    };
    PricingResult {
        pv: base.price,
        greeks: Greeks {
            delta: base.delta,
            gamma: base.gamma,
            vega: (vol_up.price - vol_down.price) / (2.0 * hv),
            theta: base.theta,
            rho,
            vanna: (vol_up.delta - vol_down.delta) / (2.0 * hv),
            charm,
            gamma_p,
            zomma: (vol_up.gamma - vol_down.gamma) / (2.0 * hv),
        },
        std_err: None,
    }
}

/// Lattice price on the debug engine: the full spot/value trees, the
/// early-exercise boundary per layer, tree Greeks and wall-clock time.
/// Same price as [`npv`], bit for bit — greeks come from
/// [`solution`] / [`pricing_result`] on the production engine.
pub fn npv_with_diagnostics(option: &EquityOption) -> LatticeDiagnostics {
    let s = setup_with(option, 0.0, 0.0, 0.0, 0.0);
    let strike = option.base.strike_price;
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    let exercise = exercise_rule(option, s.t, s.n);
    price_with_diagnostics(
        option.lattice_cfg().tree_type,
        s.s0,
        &s.params,
        s.n,
        s.dt,
        s.df_step,
        &terminal,
        exercise.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    fn builder(put_or_call: PutOrCall, engine: Engine) -> EquityOptionBuilder {
        EquityOptionBuilder::new()
            .symbol("TEST")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .dividend_yield(0.02)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(put_or_call)
            .engine(engine)
    }

    #[test]
    fn european_tree_greeks_match_the_analytic_engine() {
        let tree = builder(PutOrCall::Call, Engine::Binomial).build().unwrap();
        let analytic = builder(PutOrCall::Call, Engine::BlackScholes).build().unwrap();
        let result = tree.price().unwrap();
        assert_eq!(result.pv, tree.npv(), "pricing_result must reuse the npv price");
        assert!((result.greeks.delta - analytic.delta()).abs() < 2e-3);
        assert!((result.greeks.gamma - analytic.gamma()).abs() < 2e-4);
        assert!((result.greeks.theta - analytic.theta()).abs() < 2e-2);
        assert!((result.greeks.vega - analytic.vega()).abs() < 5e-2);
        assert!((result.greeks.rho - analytic.rho()).abs() < 5e-2);
        assert!((result.greeks.vanna - analytic.vanna()).abs() < 5e-3);
        assert!((result.greeks.charm - analytic.charm()).abs() < 5e-3);
        assert!((result.greeks.zomma - analytic.zomma()).abs() < 5e-3);
        assert!((tree.volga() - analytic.volga()).abs() < 5e-1);
    }

    #[test]
    fn pricing_result_matches_the_per_greek_dispatch() {
        // one-go greeks and the individual accessors share the same passes
        let option = builder(PutOrCall::Put, Engine::Binomial)
            .american()
            .build()
            .unwrap();
        let result = option.price().unwrap();
        assert_eq!(result.greeks.delta, option.delta());
        assert_eq!(result.greeks.gamma, option.gamma());
        assert_eq!(result.greeks.theta, option.theta());
        assert_eq!(result.greeks.vega, option.vega());
        assert_eq!(result.greeks.rho, option.rho());
        assert_eq!(result.greeks.vanna, option.vanna());
        assert_eq!(result.greeks.charm, option.charm());
        assert_eq!(result.greeks.zomma, option.zomma());
    }

    #[test]
    fn american_put_greeks_reflect_the_exercise_boundary() {
        let american = builder(PutOrCall::Put, Engine::Binomial).american().build().unwrap();
        let european = builder(PutOrCall::Put, Engine::Binomial).build().unwrap();
        // deeper (more negative) delta and faster decay than the European:
        // the tree greeks see the exercise boundary, the old analytic
        // fallback could not
        assert!(american.delta() < european.delta() - 1e-3);
        assert!(american.npv() > european.npv() + 1e-3);
        // sanity: an ATM American put is short delta, long gamma
        assert!(american.delta() > -1.0 && american.delta() < 0.0);
        assert!(american.gamma() > 0.0);
        assert!(american.theta() < 0.0);
    }

    #[test]
    fn price_with_zero_shifts_reproduces_npv() {
        let option = builder(PutOrCall::Put, Engine::Binomial).american().build().unwrap();
        assert_eq!(option.price_with(0.0, 0.0, 0.0, 0.0), option.npv());
        // a spot shift moves the reprice in the direction of delta
        let bumped = option.price_with(1.0, 0.0, 0.0, 0.0);
        assert!(bumped < option.npv(), "put value must fall as spot rises");
    }

    #[test]
    fn term_structure_lattice_reports_greeks_too() {
        let option = builder(PutOrCall::Call, Engine::Binomial)
            .tree_term_structure()
            .build()
            .unwrap();
        let analytic = builder(PutOrCall::Call, Engine::BlackScholes).build().unwrap();
        // flat inputs: the term lattice's bump greeks sit near the closed form
        assert!((option.delta() - analytic.delta()).abs() < 5e-3);
        assert!(
            (option.gamma() - analytic.gamma()).abs() < 2e-3,
            "term gamma {} vs analytic {}",
            option.gamma(),
            analytic.gamma()
        );
        assert!((option.theta() - analytic.theta()).abs() < 5e-2);
        assert!((option.vega() - analytic.vega()).abs() < 2e-1);
    }
}
