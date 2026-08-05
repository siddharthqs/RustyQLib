//! The central sensitivity engine: one implementation of bump-and-reprice
//! Greeks, one batch entry point, engine-native fast paths.
//!
//! Every Greek request routes through here. Engines fall into four routes:
//!
//! - **Grid** (finite difference): delta/gamma/theta are read off the
//!   solved grid; the higher orders difference whole grid *solutions*
//!   (vanna = d(grid delta)/dσ), which is smoother than price stencils.
//! - **Tree** (binomial): same idea on the lattice — value and
//!   delta/gamma/theta from one backward pass, higher orders from bumped
//!   tree solutions ([`binomial::pricing_result`] shares seven passes).
//! - **Analytic** (Black-Scholes engine): the payoff-aware
//!   [`BlackScholesPricer`] with closed forms where they exist (vanilla
//!   vanna/charm/zomma/volga, the Black-76 futures family).
//! - **Bump**: everything else (Monte Carlo, Barone-Adesi-Whaley,
//!   Bjerksund-Stensland, analytic Heston) shares the *one* set of
//!   central-difference stencils below over bumped-market reprices
//!   ([`EquityOption::price_bumped`]), which guarantee common random
//!   numbers under Monte Carlo. What used to be
//!   four hand-written copies of every stencil now differs only in a
//!   [`BumpPolicy`]: a per-engine table of bump sizes (preserved exactly,
//!   so values are bit-identical to the former per-engine code).
//!
//! The batch entry point [`pricing_result`] shares evaluations across
//! Greeks through a reprice cache: one Monte Carlo `price()` costs 17
//! simulations instead of 28, one finite-difference `price()` costs 9
//! solves instead of ~16.
//!
//! Within the bump route, engines expose **native fast paths** where
//! their structure allows a better estimator than a stencil:
//! Monte Carlo pathwise delta/vega on the terminal-GBM route
//! ([`montecarlo::pathwise_delta_vega`] — one simulation, no
//! finite-difference bias), the **adjoint (AAD) sweep** on the
//! path-simulation routes with continuous payoffs
//! ([`montecarlo::aad_greeks`] — delta, vega and rho from one backward
//! pass per path over the [`core::aad`](crate::core::aad) tape), the
//! Heston vanilla delta read off the price integration
//! ([`heston::native_vanilla_delta`]), and the Barone-Adesi-Whaley
//! boundary solve shared across the spot ladder ([`baw::SpotKernel`],
//! bit-identical values).

use std::collections::HashMap;

use crate::core::results::{Greeks, PricingResult};
use crate::equity::blackscholes::BlackScholesPricer;
use crate::equity::bump::{Bump, BumpedMarket};
use crate::equity::utils::PricingEngine;
use crate::equity::vanilla_option::EquityOption;
use crate::equity::{baw, binomial, finite_difference, heston, montecarlo};

// ── Bump policies ───────────────────────────────────────────────────────

/// Central-difference bump sizes for one engine. Sizes are inherited from
/// the engines' historical per-Greek choices (larger steps where the
/// kernel is noisier), so consolidating did not move any number.
#[derive(Debug, Clone, Copy)]
struct BumpPolicy {
    /// Spot bump for delta, vanna and charm.
    spot_bump: f64,
    /// Spot bump for gamma and the zomma inner stencil (second
    /// differences want a larger step).
    spot_bump_gamma: f64,
    /// Vol bump for vega, vanna and the zomma outer stencil.
    vol_bump: f64,
    /// Vol bump for volga.
    vol_bump_volga: f64,
    /// Rate bump for rho.
    rate_bump: f64,
    /// Maturity bump for theta and charm.
    time_bump: f64,
}

fn maturity_bump(option: &EquityOption) -> f64 {
    (1.0 / 365.0_f64).min(0.5 * option.time_to_maturity())
}

/// How Greeks are produced for this option's engine.
enum Route {
    Grid,
    Tree,
    Analytic,
    Bump(BumpPolicy),
}

fn route(option: &EquityOption) -> Route {
    match option.engine {
        PricingEngine::MonteCarlo(_) => {
            let s = option.market.spot.value();
            Route::Bump(BumpPolicy {
                spot_bump: s * 0.01,
                spot_bump_gamma: s * 0.01,
                vol_bump: 0.01,
                vol_bump_volga: 0.01,
                rate_bump: 1e-4,
                time_bump: maturity_bump(option),
            })
        }
        PricingEngine::FiniteDifference(_) => Route::Grid,
        PricingEngine::BaroneAdesiWhaley | PricingEngine::BjerksundStensland => {
            // the American approximations are smooth in the escrowed spot
            let s = option.effective_spot();
            Route::Bump(BumpPolicy {
                spot_bump: s * 1e-4,
                spot_bump_gamma: s * 1e-4,
                vol_bump: 1e-4,
                vol_bump_volga: 1e-3,
                rate_bump: 1e-4,
                time_bump: maturity_bump(option),
            })
        }
        _ if option.analytic_heston() => {
            let s = option.market.spot.value();
            Route::Bump(BumpPolicy {
                spot_bump: s * 1e-4,
                spot_bump_gamma: s * 1e-3,
                vol_bump: 1e-4,
                vol_bump_volga: 1e-2,
                rate_bump: 1e-5,
                time_bump: maturity_bump(option),
            })
        }
        PricingEngine::Binomial(_) => Route::Tree,
        _ => Route::Analytic,
    }
}

// ── The cached repricer ─────────────────────────────────────────────────

/// Memoized shifted reprices of one option, in the **maturity-shift**
/// convention: `reprice(ds, dv, dr, dt)` values the option with maturity
/// extended by `dt` (the stencils below read like the textbook formulas).
/// The [`Bump`] convention is elapsed calendar time, hence the sign flip
/// when the view is built.
///
/// On the Barone-Adesi-Whaley engine the repricer additionally caches the
/// spot-independent boundary work per `(dv, dr, dt)` shift
/// ([`baw::SpotKernel`]): the delta/gamma spot ladder solves the critical
/// price once instead of once per evaluation, with bit-identical values.
struct Repricer<'a> {
    option: &'a EquityOption,
    cache: HashMap<[u64; 4], f64>,
    /// `Some` on the BAW engine: boundary kernels keyed by (dv, dr, dt).
    baw_kernels: Option<HashMap<[u64; 3], baw::SpotKernel>>,
}

impl<'a> Repricer<'a> {
    fn new(option: &'a EquityOption) -> Self {
        let baw_kernels =
            matches!(option.engine, PricingEngine::BaroneAdesiWhaley).then(HashMap::new);
        Repricer {
            option,
            cache: HashMap::new(),
            baw_kernels,
        }
    }

    fn reprice(&mut self, ds: f64, dv: f64, dr: f64, dt: f64) -> f64 {
        let key = [ds.to_bits(), dv.to_bits(), dr.to_bits(), dt.to_bits()];
        if let Some(&cached) = self.cache.get(&key) {
            return cached;
        }
        let value = match &mut self.baw_kernels {
            Some(kernels) => {
                let kernel = kernels
                    .entry([dv.to_bits(), dr.to_bits(), dt.to_bits()])
                    .or_insert_with(|| baw::SpotKernel::new(self.option, dv, dr, dt));
                kernel.value(self.option.effective_spot() + ds)
            }
            None => {
                // the stencils shift the maturity; the bump convention is
                // elapsed calendar time — the one place the sign flips
                let bump = Bump {
                    d_spot: ds,
                    d_vol: dv,
                    d_rate: dr,
                    d_time: -dt,
                };
                self.option
                    .price_bumped(&BumpedMarket::new(&self.option.market, bump))
            }
        };
        self.cache.insert(key, value);
        value
    }
}

// ── The stencils (written once) ─────────────────────────────────────────

fn bump_delta(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let h = bumps.spot_bump;
    (repricer.reprice(h, 0.0, 0.0, 0.0) - repricer.reprice(-h, 0.0, 0.0, 0.0)) / (2.0 * h)
}

fn bump_gamma(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let h = bumps.spot_bump_gamma;
    (repricer.reprice(h, 0.0, 0.0, 0.0) - 2.0 * repricer.reprice(0.0, 0.0, 0.0, 0.0)
        + repricer.reprice(-h, 0.0, 0.0, 0.0))
        / (h * h)
}

fn bump_vega(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let h = bumps.vol_bump;
    (repricer.reprice(0.0, h, 0.0, 0.0) - repricer.reprice(0.0, -h, 0.0, 0.0)) / (2.0 * h)
}

fn bump_theta(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    // calendar theta = dV/dt = -dV/dT
    let h = bumps.time_bump;
    -(repricer.reprice(0.0, 0.0, 0.0, h) - repricer.reprice(0.0, 0.0, 0.0, -h)) / (2.0 * h)
}

fn bump_rho(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let h = bumps.rate_bump;
    (repricer.reprice(0.0, 0.0, h, 0.0) - repricer.reprice(0.0, 0.0, -h, 0.0)) / (2.0 * h)
}

fn bump_vanna(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let (hs, hv) = (bumps.spot_bump, bumps.vol_bump);
    (repricer.reprice(hs, hv, 0.0, 0.0)
        - repricer.reprice(-hs, hv, 0.0, 0.0)
        - repricer.reprice(hs, -hv, 0.0, 0.0)
        + repricer.reprice(-hs, -hv, 0.0, 0.0))
        / (4.0 * hs * hv)
}

fn bump_charm(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    // charm = d(delta)/dt = -d(delta)/dT
    let (hs, ht) = (bumps.spot_bump, bumps.time_bump);
    -(repricer.reprice(hs, 0.0, 0.0, ht)
        - repricer.reprice(-hs, 0.0, 0.0, ht)
        - repricer.reprice(hs, 0.0, 0.0, -ht)
        + repricer.reprice(-hs, 0.0, 0.0, -ht))
        / (4.0 * hs * ht)
}

fn bump_zomma(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let (hs, hv) = (bumps.spot_bump_gamma, bumps.vol_bump);
    let gamma_at = |repricer: &mut Repricer, dv: f64| {
        (repricer.reprice(hs, dv, 0.0, 0.0) - 2.0 * repricer.reprice(0.0, dv, 0.0, 0.0)
            + repricer.reprice(-hs, dv, 0.0, 0.0))
            / (hs * hs)
    };
    (gamma_at(repricer, hv) - gamma_at(repricer, -hv)) / (2.0 * hv)
}

fn bump_volga(repricer: &mut Repricer, bumps: &BumpPolicy) -> f64 {
    let h = bumps.vol_bump_volga;
    (repricer.reprice(0.0, h, 0.0, 0.0) - 2.0 * repricer.reprice(0.0, 0.0, 0.0, 0.0)
        + repricer.reprice(0.0, -h, 0.0, 0.0))
        / (h * h)
}

// ── Single-Greek entry points (the accessors' backend) ──────────────────

macro_rules! greek {
    ($name:ident, $stencil:ident, $grid:path, $tree:path, $analytic:ident) => {
        pub fn $name(option: &EquityOption) -> f64 {
            match route(option) {
                Route::Grid => $grid(option),
                Route::Tree => $tree(option),
                Route::Analytic => BlackScholesPricer::new().$analytic(option),
                Route::Bump(bumps) => $stencil(&mut Repricer::new(option), &bumps),
            }
        }
    };
}

greek!(
    gamma,
    bump_gamma,
    finite_difference::gamma,
    binomial::gamma,
    gamma
);
greek!(
    theta,
    bump_theta,
    finite_difference::theta,
    binomial::theta,
    theta
);
greek!(
    vanna,
    bump_vanna,
    finite_difference::vanna,
    binomial::vanna,
    vanna
);

// ── Engine-native fast paths inside the bump route ──────────────────────
//
// Better estimators than the stencils where the engine's structure allows:
// Monte Carlo pathwise delta/vega on the one-step terminal route, the
// adjoint (AAD) sweep for delta/vega/rho on the path-simulation routes
// with continuous payoffs, and the Heston vanilla delta that falls out of
// the price integration. Fall through to the stencils everywhere else.

fn native_delta(option: &EquityOption) -> Option<f64> {
    match option.engine {
        PricingEngine::MonteCarlo(_) => montecarlo::pathwise_delta_vega(option)
            .map(|(delta, _)| delta)
            .or_else(|| montecarlo::aad_greeks(option).map(|g| g.delta)),
        _ if option.analytic_heston() => heston::native_vanilla_delta(option),
        _ => None,
    }
}

fn native_vega(option: &EquityOption) -> Option<f64> {
    match option.engine {
        PricingEngine::MonteCarlo(_) => montecarlo::pathwise_delta_vega(option)
            .map(|(_, vega)| vega)
            .or_else(|| montecarlo::aad_greeks(option).map(|g| g.vega)),
        _ => None,
    }
}

fn native_rho(option: &EquityOption) -> Option<f64> {
    match option.engine {
        // the one-step terminal route keeps the (cheaper) bump stencil;
        // the adjoint sweep covers the path routes
        PricingEngine::MonteCarlo(_) if montecarlo::pathwise_delta_vega(option).is_none() => {
            montecarlo::aad_greeks(option).map(|g| g.rho)
        }
        _ => None,
    }
}

pub fn delta(option: &EquityOption) -> f64 {
    match route(option) {
        Route::Grid => finite_difference::delta(option),
        Route::Tree => binomial::delta(option),
        Route::Analytic => BlackScholesPricer::new().delta(option),
        Route::Bump(bumps) => {
            native_delta(option).unwrap_or_else(|| bump_delta(&mut Repricer::new(option), &bumps))
        }
    }
}

pub fn vega(option: &EquityOption) -> f64 {
    match route(option) {
        Route::Grid => finite_difference::vega(option),
        Route::Tree => binomial::vega(option),
        Route::Analytic => BlackScholesPricer::new().vega(option),
        Route::Bump(bumps) => {
            native_vega(option).unwrap_or_else(|| bump_vega(&mut Repricer::new(option), &bumps))
        }
    }
}

pub fn rho(option: &EquityOption) -> f64 {
    match route(option) {
        Route::Grid => finite_difference::rho(option),
        Route::Tree => binomial::rho(option),
        Route::Analytic => BlackScholesPricer::new().rho(option),
        Route::Bump(bumps) => {
            native_rho(option).unwrap_or_else(|| bump_rho(&mut Repricer::new(option), &bumps))
        }
    }
}
greek!(
    charm,
    bump_charm,
    finite_difference::charm,
    binomial::charm,
    charm
);
greek!(
    zomma,
    bump_zomma,
    finite_difference::zomma,
    binomial::zomma,
    zomma
);
greek!(
    volga,
    bump_volga,
    finite_difference::volga,
    binomial::volga,
    volga
);

/// Delta elasticity `S * gamma / delta`; `NaN` when delta is zero.
pub fn gamma_p(option: &EquityOption) -> f64 {
    let delta = delta(option);
    if delta == 0.0 {
        f64::NAN
    } else {
        option.market.spot.value() * gamma(option) / delta
    }
}

// ── The batch entry point ───────────────────────────────────────────────

/// Value plus all reported Greeks, sharing work across them:
/// grid/tree engines harvest delta/gamma/theta from the base solve, and
/// bump engines reuse every shifted reprice that appears in more than one
/// stencil through the [`Repricer`] cache.
pub fn pricing_result(option: &EquityOption) -> PricingResult {
    match route(option) {
        Route::Tree => binomial::pricing_result(option),
        Route::Grid => finite_difference::pricing_result(option),
        Route::Analytic => {
            let pricer = BlackScholesPricer::new();
            PricingResult {
                pv: pricer.npv(option),
                greeks: Greeks {
                    delta: pricer.delta(option),
                    gamma: pricer.gamma(option),
                    vega: pricer.vega(option),
                    theta: pricer.theta(option),
                    rho: pricer.rho(option),
                    vanna: pricer.vanna(option),
                    charm: pricer.charm(option),
                    gamma_p: pricer.gamma_p(option),
                    zomma: pricer.zomma(option),
                },
                std_err: None,
            }
        }
        Route::Bump(bumps) => {
            let (pv, std_err) = match option.engine {
                PricingEngine::MonteCarlo(_) => {
                    let stats = montecarlo::stats(option, None);
                    (stats.pv, Some(stats.std_err))
                }
                _ => (
                    option.price_bumped(&BumpedMarket::base(&option.market)),
                    None,
                ),
            };
            // one pathwise pass serves delta and vega under MC; one
            // adjoint sweep serves delta, vega AND rho on the path routes
            let pathwise = match option.engine {
                PricingEngine::MonteCarlo(_) => montecarlo::pathwise_delta_vega(option),
                _ => None,
            };
            let adjoint = match option.engine {
                PricingEngine::MonteCarlo(_) if pathwise.is_none() => {
                    montecarlo::aad_greeks(option)
                }
                _ => None,
            };
            let repricer = &mut Repricer::new(option);
            let delta = pathwise
                .map(|(delta, _)| delta)
                .or(adjoint.map(|g| g.delta))
                .or_else(|| {
                    option
                        .analytic_heston()
                        .then(|| heston::native_vanilla_delta(option))
                        .flatten()
                })
                .unwrap_or_else(|| bump_delta(repricer, &bumps));
            let vega = pathwise
                .map(|(_, vega)| vega)
                .or(adjoint.map(|g| g.vega))
                .unwrap_or_else(|| bump_vega(repricer, &bumps));
            let rho = adjoint
                .map(|g| g.rho)
                .unwrap_or_else(|| bump_rho(repricer, &bumps));
            let gamma = bump_gamma(repricer, &bumps);
            let gamma_p = if delta == 0.0 {
                f64::NAN
            } else {
                option.market.spot.value() * gamma / delta
            };
            PricingResult {
                pv,
                greeks: Greeks {
                    delta,
                    gamma,
                    vega,
                    theta: bump_theta(repricer, &bumps),
                    rho,
                    vanna: bump_vanna(repricer, &bumps),
                    charm: bump_charm(repricer, &bumps),
                    gamma_p,
                    zomma: bump_zomma(repricer, &bumps),
                },
                std_err,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::{Engine, Model};
    use chrono::NaiveDate;

    /// The stencil legs below bump the market themselves and reprice.
    fn spot_bumped(option: &EquityOption, h: f64) -> f64 {
        option.price_bumped(&BumpedMarket::new(&option.market, Bump::spot(h)))
    }

    fn option(engine: Engine, put_or_call: PutOrCall) -> EquityOption {
        EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(put_or_call)
            .engine(engine)
            .build()
            .expect("option must build")
    }

    #[test]
    fn mc_pathwise_delta_and_vega_match_the_analytic_values() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mc = option(Engine::MonteCarlo, pc);
            let bs = option(Engine::BlackScholes, pc);
            // the pathwise estimator is live for this option
            let (d, v) = montecarlo::pathwise_delta_vega(&mc).expect("pathwise must apply");
            assert_eq!(d, mc.delta(), "accessor must use the pathwise estimator");
            assert_eq!(v, mc.vega(), "accessor must use the pathwise estimator");
            // and agrees with the closed form (QMC terminal simulation)
            assert!(
                (d - bs.delta()).abs() < 5e-3,
                "{pc:?} delta {d} vs {}",
                bs.delta()
            );
            assert!(
                (v - bs.vega()).abs() < 0.2,
                "{pc:?} vega {v} vs {}",
                bs.vega()
            );
            // batch equals accessors exactly
            let result = mc.price().unwrap();
            assert_eq!(result.greeks.delta, d);
            assert_eq!(result.greeks.vega, v);
        }
    }

    #[test]
    fn mc_pathwise_declines_out_of_scope_and_the_adjoint_takes_over() {
        // multi-step path simulation: the terminal pathwise estimator
        // does not apply; the AAD sweep covers the route instead
        let mut mc = option(Engine::MonteCarlo, PutOrCall::Call);
        if let PricingEngine::MonteCarlo(cfg) = &mut mc.engine {
            cfg.time_steps = 12;
        }
        assert!(montecarlo::pathwise_delta_vega(&mc).is_none());
        let adjoint = montecarlo::aad_greeks(&mc).expect("AAD must cover multi-step vanilla");
        assert_eq!(
            mc.delta(),
            adjoint.delta,
            "accessor must use the adjoint estimator"
        );
        assert_eq!(mc.vega(), adjoint.vega);
        assert_eq!(mc.rho(), adjoint.rho);
        // one sweep agrees with the closed forms
        let bs = option(Engine::BlackScholes, PutOrCall::Call);
        assert!(
            (adjoint.delta - bs.delta()).abs() < 1e-2,
            "{} vs {}",
            adjoint.delta,
            bs.delta()
        );
        assert!(
            (adjoint.vega - bs.vega()).abs() < 0.5,
            "{} vs {}",
            adjoint.vega,
            bs.vega()
        );
        assert!(
            (adjoint.rho - bs.rho()).abs() < 0.5,
            "{} vs {}",
            adjoint.rho,
            bs.rho()
        );
        // batch equals accessors exactly
        let result = mc.price().unwrap();
        assert_eq!(result.greeks.delta, adjoint.delta);
        assert_eq!(result.greeks.vega, adjoint.vega);
        assert_eq!(result.greeks.rho, adjoint.rho);
    }

    #[test]
    fn aad_covers_continuous_path_dependents_and_declines_discontinuous() {
        use crate::core::utils::ContractStyle;
        use crate::equity::asian::{AsianStrikeType, AveragingType};
        use crate::equity::vanilla_option::AsianPayoff;
        // arithmetic fixed-strike Asian: continuous payoff, AAD applies
        let mut asian = option(Engine::MonteCarlo, PutOrCall::Call);
        asian.payoff = Box::new(AsianPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: ContractStyle::European,
            averaging: AveragingType::Arithmetic,
            strike_type: AsianStrikeType::FixedStrike,
        });
        let adjoint = montecarlo::aad_greeks(&asian).expect("AAD must cover Asians");
        assert_eq!(asian.delta(), adjoint.delta);
        // the adjoint agrees with the CRN bump stencil on the same option
        let h = asian.market.spot.value() * 0.01;
        let bump = (spot_bumped(&asian, h) - spot_bumped(&asian, -h)) / (2.0 * h);
        assert!(
            (adjoint.delta - bump).abs() < 0.03,
            "adjoint {} vs bump {bump}",
            adjoint.delta
        );
        assert!(adjoint.vega > 0.0 && adjoint.rho > 0.0);

        // a barrier's indicator has zero almost-everywhere derivative:
        // it must never opt into AAD, the bump stencils keep it
        use crate::equity::barrier::{BarrierDirection, KnockType};
        use crate::equity::vanilla_option::BarrierPayoff;
        let mut barrier = option(Engine::MonteCarlo, PutOrCall::Call);
        barrier.payoff = Box::new(BarrierPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: ContractStyle::European,
            direction: BarrierDirection::Up,
            knock: KnockType::Out,
            barrier: 130.0,
            barrier2: None,
            rebate: 0.0,
            rebate_at_hit: false,
        });
        assert!(montecarlo::aad_greeks(&barrier).is_none());
    }

    #[test]
    fn heston_native_delta_matches_the_bump_stencil() {
        use crate::equity::heston::HestonParams;
        let params = HestonParams {
            v0: 0.0625,
            kappa: 1.5,
            theta: 0.0625,
            vol_of_vol: 0.4,
            rho: -0.6,
        };
        let mut call = option(Engine::BlackScholes, PutOrCall::Call);
        call.model = Model::Heston(params);
        let mut put = option(Engine::BlackScholes, PutOrCall::Put);
        put.model = Model::Heston(params);
        // native delta agrees with the old central-difference stencil
        let h = call.market.spot.value() * 1e-4;
        let stencil = (spot_bumped(&call, h) - spot_bumped(&call, -h)) / (2.0 * h);
        assert!(
            (call.delta() - stencil).abs() < 1e-6,
            "native {} vs stencil {stencil}",
            call.delta()
        );
        // put-call delta parity: delta_C - delta_P = e^{-qT} (here q = 0)
        assert!((call.delta() - put.delta() - 1.0).abs() < 1e-9);
        // batch equals accessor exactly
        assert_eq!(call.price().unwrap().greeks.delta, call.delta());
    }

    #[test]
    fn baw_boundary_kernel_is_bit_identical_to_the_direct_reprice() {
        // American put: the early-exercise boundary is live
        let baw_put = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(PutOrCall::Put)
            .american()
            .engine(Engine::BaroneAdesiWhaley)
            .build()
            .expect("option must build");
        // the kernel path (used by delta/gamma) must reproduce the direct
        // bumped-market stencil bit for bit — sharing the boundary solve
        // is a pure speed optimization
        let h = baw_put.effective_spot() * 1e-4;
        let direct_delta = (spot_bumped(&baw_put, h) - spot_bumped(&baw_put, -h)) / (2.0 * h);
        assert_eq!(baw_put.delta(), direct_delta);
        let direct_gamma = (spot_bumped(&baw_put, h) - 2.0 * spot_bumped(&baw_put, 0.0)
            + spot_bumped(&baw_put, -h))
            / (h * h);
        assert_eq!(baw_put.gamma(), direct_gamma);
    }
}
