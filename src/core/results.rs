//! Structured pricing results: everything one pricing call produces.

use serde::{Deserialize, Serialize};

/// First- and second-order sensitivities of a priced instrument.
///
/// Instruments that do not report a sensitivity leave it at `0.0`
/// (e.g. spot Greeks of return-based payoffs such as cliquets, which are
/// spot-homogeneous by construction).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Greeks {
    /// Change in value per unit change in the underlying, `dV/dS`.
    pub delta: f64,
    /// Change in delta per unit change in the underlying, `d²V/dS²`.
    pub gamma: f64,
    /// Change in value per unit change in implied volatility, `dV/dσ`.
    pub vega: f64,
    /// Change in value per year of calendar time, `dV/dt`.
    pub theta: f64,
    /// Change in value per unit change in the risk-free rate, `dV/dr`.
    pub rho: f64,
    /// Change in delta per unit change in implied volatility, `d²V/(dS dσ)`.
    pub vanna: f64,
    /// Change in delta per year of calendar time, `d²V/(dS dt)`.
    pub charm: f64,
    /// Percentage gamma (Haug's GammaP), `S * gamma / 100`: the change
    /// in delta per 1% move in the underlying.
    pub gamma_p: f64,
    /// Change in gamma per unit change in implied volatility, `d³V/(dS² dσ)`.
    pub zomma: f64,
}

/// Per-underlying first-order sensitivities of a multi-asset
/// instrument, in leg order (`symbols[i]` names slot `i`).
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct PerAssetGreeks {
    pub symbols: Vec<String>,
    /// `dV/dS_i` per leg.
    pub deltas: Vec<f64>,
    /// `d²V/dS_i²` per leg (same-leg second differences; cross-gammas
    /// are not reported).
    pub gammas: Vec<f64>,
    /// `dV/dσ_i` per leg.
    pub vegas: Vec<f64>,
}

/// The result of a single [`price()`](crate::core::traits::Instrument::price)
/// call: present value, sensitivities, and the Monte Carlo standard error
/// when a simulation engine produced the value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PricingResult {
    /// Present value.
    pub pv: f64,
    /// Scalar sensitivities of `pv`. For multi-asset instruments the
    /// spot/vol slots stay zero — the per-leg values live in
    /// [`asset_greeks`](Self::asset_greeks).
    pub greeks: Greeks,
    /// Monte Carlo standard error of `pv`; `None` for deterministic engines.
    pub std_err: Option<f64>,
    /// Per-underlying sensitivities of multi-asset instruments; `None`
    /// for single-asset ones (and absent from serialized output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub asset_greeks: Option<PerAssetGreeks>,
}

impl PricingResult {
    /// A result with the given present value, zero Greeks and no standard
    /// error — the shape produced by deterministic pricers of instruments
    /// that do not report sensitivities.
    pub fn from_pv(pv: f64) -> Self {
        PricingResult {
            pv,
            greeks: Greeks::default(),
            std_err: None,
            asset_greeks: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;

    fn vanilla(engine: Engine) -> crate::equity::vanilla_option::EquityOption {
        EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(0.05)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Call)
            .engine(engine)
            .build()
            .expect("option must build")
    }

    #[test]
    fn price_matches_individual_accessors() {
        let option = vanilla(Engine::BlackScholes);
        let result = option.price().unwrap();
        assert_eq!(result.pv, option.npv());
        assert_eq!(result.greeks.delta, option.delta());
        assert_eq!(result.greeks.gamma, option.gamma());
        assert_eq!(result.greeks.vega, option.vega());
        assert_eq!(result.greeks.theta, option.theta());
        assert_eq!(result.greeks.rho, option.rho());
        assert_eq!(result.greeks.vanna, option.vanna());
        assert_eq!(result.greeks.charm, option.charm());
        assert_eq!(result.greeks.zomma, option.zomma());
        assert_eq!(result.std_err, None, "deterministic engine has no std_err");
    }

    #[test]
    fn monte_carlo_price_reports_std_err_and_reproducible_pv() {
        // the default low-discrepancy sampler reports no standard error
        // (deterministic points have no sample variance to report)
        let sobol = vanilla(Engine::MonteCarlo);
        assert_eq!(sobol.price().unwrap().std_err, None);

        let mut option = vanilla(Engine::MonteCarlo);
        option.mc_cfg_mut().sampler = crate::equity::montecarlo::Sampler::PseudoRandom;
        let result = option.price().unwrap();
        let se = result
            .std_err
            .expect("pseudo-random MC must report a standard error");
        assert!(se > 0.0 && se.is_finite());
        // bit-reproducible MC: price() sees the same paths as npv()
        assert_eq!(result.pv, option.npv());
    }

    #[test]
    fn unsupported_combination_errors_through_price() {
        use crate::core::errors::RustyQLibError;
        // build() enforces engine support, so force the bad combination
        // onto an already-built option to exercise the price()-time check
        let mut option = vanilla(Engine::MonteCarlo);
        option.payoff = Box::new(crate::equity::vanilla_option::LookbackPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: crate::core::utils::ContractStyle::European,
            lookback_type: crate::equity::vanilla_option::LookbackType::FloatingStrike,
        });
        option.engine = crate::equity::utils::PricingEngine::Binomial(Default::default());
        match option.price() {
            Err(RustyQLibError::UnsupportedEngine(msg)) => {
                assert!(
                    msg.contains("Binomial"),
                    "should explain the refusal: {msg}"
                )
            }
            other => panic!("expected UnsupportedEngine, got {other:?}"),
        }
    }
}
