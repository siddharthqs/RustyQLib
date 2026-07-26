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
    /// Delta elasticity, `S * gamma / delta`.
    pub gamma_p: f64,
    /// Change in gamma per unit change in implied volatility, `d³V/(dS² dσ)`.
    pub zomma: f64,
}

/// The result of a single [`price()`](crate::core::traits::Instrument::price)
/// call: present value, sensitivities, and the Monte Carlo standard error
/// when a simulation engine produced the value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PricingResult {
    /// Present value.
    pub pv: f64,
    /// Sensitivities of `pv`.
    pub greeks: Greeks,
    /// Monte Carlo standard error of `pv`; `None` for deterministic engines.
    pub std_err: Option<f64>,
}

impl PricingResult {
    /// A result with the given present value, zero Greeks and no standard
    /// error — the shape produced by deterministic pricers of instruments
    /// that do not report sensitivities.
    pub fn from_pv(pv: f64) -> Self {
        PricingResult { pv, greeks: Greeks::default(), std_err: None }
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
            .build().expect("option must build")
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
        let option = vanilla(Engine::MonteCarlo);
        let result = option.price().unwrap();
        let se = result.std_err.expect("MC engine must report a standard error");
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
                assert!(msg.contains("Binomial"), "should explain the refusal: {msg}")
            }
            other => panic!("expected UnsupportedEngine, got {other:?}"),
        }
    }
}
