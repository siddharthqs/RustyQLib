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

use crate::core::lattice::{
    price_backward, price_with_diagnostics, LatticeDiagnostics, LatticeParams, TermLattice,
};
use crate::core::utils::{times_to_grid_steps, ContractStyle};
use super::vanilla_option::EquityOption;

struct TreeSetup {
    params: LatticeParams,
    n: usize,
    df_step: f64,
    dt: f64,
}

fn setup(option: &EquityOption) -> TreeSetup {
    assert!(option.base.underlying_price.value >= 0.0);
    let t = option.time_to_maturity();
    let r = option.base.risk_free_rate();
    let b = r - option.base.carry_yield();
    let cfg = option.lattice_cfg();
    let n = cfg.tree_type.effective_steps(cfg.steps);
    let params = cfg
        .tree_type
        .params(
            option.base.effective_spot(),
            option.base.strike_price,
            b,
            option.base.volatility(),
            t,
            n,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let dt = t / n as f64;
    TreeSetup { params, n, df_step: (-r * dt).exp(), dt }
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
/// applied per step on a variance-equal time grid.
fn term_npv(option: &EquityOption) -> f64 {
    let t = option.time_to_maturity();
    let s0 = option.base.effective_spot();
    let strike = option.base.strike_price;
    let carry = option.base.carry_yield();
    let curve = &option.base.discount_curve;
    let forward_rate =
        |t1: f64, t2: f64| (curve.df(t1) / curve.df(t2)).ln() / (t2 - t1);
    let forward_carry = |_: f64, _: f64| carry;
    let total_variance = |tt: f64| {
        if tt <= 1e-12 {
            return 0.0;
        }
        // strike-frozen implied term structure: sigma(K, t)^2 * t
        let fwd = s0 / curve.df(tt) * (-carry * tt).exp();
        let sigma = option.base.vol_surface.vol(strike, fwd, tt);
        sigma * sigma * tt
    };
    let lattice =
        TermLattice::build(option.lattice_cfg().steps, t, &forward_rate, &forward_carry, &total_variance)
            .unwrap_or_else(|e| panic!("{e}"));
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    match option.payoff.exercise_style() {
        ContractStyle::European => lattice.price(s0, &terminal, None),
        ContractStyle::American => {
            let ex = |_: usize, _: f64, spot: f64, cont: f64| {
                option.payoff.payoff(spot, strike).max(cont)
            };
            lattice.price(s0, &terminal, Some(&ex))
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
            lattice.price(s0, &terminal, Some(&ex))
        }
    }
}

/// Lattice price on the optimized rolling-array engine; routes to the
/// term-structure lattice when `lattice.term_structure` is set.
pub fn npv(option: &EquityOption) -> f64 {
    if option.lattice_cfg().term_structure {
        return term_npv(option);
    }
    let s = setup(option);
    let t = option.time_to_maturity();
    let strike = option.base.strike_price;
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    let exercise = exercise_rule(option, t, s.n);
    price_backward(
        option.base.effective_spot(),
        &s.params,
        s.n,
        s.df_step,
        &terminal,
        exercise.as_deref(),
    )
}

/// Lattice price on the debug engine: the full spot/value trees, the
/// early-exercise boundary per layer, tree Greeks and wall-clock time.
/// Same price as [`npv`], bit for bit.
pub fn npv_with_diagnostics(option: &EquityOption) -> LatticeDiagnostics {
    let s = setup(option);
    let t = option.time_to_maturity();
    let strike = option.base.strike_price;
    let terminal = |spot: f64| option.payoff.payoff(spot, strike);
    let exercise = exercise_rule(option, t, s.n);
    price_with_diagnostics(
        option.lattice_cfg().tree_type,
        option.base.effective_spot(),
        &s.params,
        s.n,
        s.dt,
        s.df_step,
        &terminal,
        exercise.as_deref(),
    )
}
