use crate::core::errors::RustyQLibError;
use crate::core::traits::Instrument;
use crate::core::utils::{Contract,CombinedContract, ContractOutput};
use crate::core::data_models::ProductData;
use crate::equity::equity_forward::EquityForward;
use crate::equity::vanilla_option::EquityOption;
use crate::equity::equity_future::EquityFuture;

/// Price one contract, reporting any failure in the output's `error` field
/// so a batch of contracts always produces one result per contract. Typed
/// errors come from validation and pricing; a panic escaping a numerical
/// kernel is caught as a last resort and reported the same way.
pub fn handle_equity_contract(data: &Contract) -> serde_json::Value {
    let priced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        price_equity_contract(data)
    }));
    let output = match priced {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => ContractOutput::from_error(e.to_string()),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "pricing panicked".to_string());
            ContractOutput::from_error(msg)
        }
    };
    if let Some(err) = &output.error {
        eprintln!("contract error: {err}");
    }
    let combined_ = CombinedContract { contract: data.clone(), output };
    serde_json::to_value(&combined_).expect("Failed to generate output")
}

fn price_equity_contract(data: &Contract) -> Result<ContractOutput, RustyQLibError> {
    match &data.product_type {
        ProductData::Option(opt) => {
            let option = EquityOption::try_from_json(opt)?;
            let (pv, std_err) = match option.engine {
                crate::equity::utils::Engine::MonteCarlo => {
                    let stats = crate::equity::montecarlo::npv_with_stats(&option);
                    (stats.pv, Some(stats.std_err))
                }
                _ => (option.try_npv()?, None),
            };
            let contract_output = ContractOutput {
                pv,
                delta: option.delta(),
                gamma: option.gamma(),
                vega: option.vega(),
                theta: option.theta(),
                rho: option.rho(),
                vanna: option.vanna(),
                charm: option.charm(),
                gamma_p: option.gamma_p(),
                zomma: option.zomma(),
                std_err,
                deltas: None,
                vegas: None,
                error: None
            };
            println!("Theoretical Price ${}", contract_output.pv);
            println!("Delta ${}", contract_output.delta);
            Ok(contract_output)
        }
        ProductData::Future(fut) => {
            let future = EquityFuture::try_from_json(fut)?;
            let contract_output = ContractOutput {
                pv: future.try_npv()?,
                delta: future.delta(),
                gamma: future.gamma(),
                vega: future.vega(),
                theta: future.theta(),
                rho: future.rho(),
                vanna: future.vanna(),
                charm: future.charm(),
                gamma_p: future.gamma_p(),
                zomma: future.zomma(),
                std_err: None,
                deltas: None,
                vegas: None,
                error: None
            };
            println!("Equity Future Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::Forward(forward) => {
            let future = EquityForward::try_from_json(forward)?;
            let contract_output = ContractOutput {
                pv: future.try_npv()?,
                delta: future.delta(),
                gamma: future.gamma(),
                vega: future.vega(),
                theta: future.theta(),
                rho: future.rho(),
                vanna: future.vanna(),
                charm: future.charm(),
                gamma_p: future.gamma_p(),
                zomma: future.zomma(),
                std_err: None,
                deltas: None,
                vegas: None,
                error: None
            };
            println!("Equity Forward Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::RainbowOption(rb) => {
            let option = crate::equity::rainbow::RainbowOption::try_from_json(rb)?;
            let stats = option.npv_with_stats();
            let pv = match stats {
                Some(s) => s.pv,
                None => option.npv(),
            };
            let contract_output = ContractOutput {
                pv,
                // scalar Greeks are per-asset for rainbows: see deltas/vegas
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: option.theta(),
                rho: option.rho(),
                // Per-asset vanna/charm are not yet defined for rainbow payoffs.
                vanna: 0.0,
                charm: 0.0,
                gamma_p: 0.0,
                zomma: 0.0,
                std_err: stats.map(|s| s.std_err),
                deltas: Some(option.deltas()),
                vegas: Some(option.vegas()),
                error: None,
            };
            println!("Rainbow Option Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::CliquetOption(cq) => {
            let cliquet = crate::equity::cliquet::Cliquet::try_from_json(cq)?;
            let (pv, std_err) = match cliquet.pricer {
                crate::equity::cliquet::CliquetPricer::MonteCarlo => {
                    let (pv, se) = cliquet.mc_npv();
                    (pv, Some(se))
                }
                crate::equity::cliquet::CliquetPricer::Analytical => match cliquet.analytic_npv()
                {
                    Ok(pv) => (pv, None),
                    Err(_) => {
                        let (pv, se) = cliquet.mc_npv();
                        (pv, Some(se))
                    }
                },
            };
            let contract_output = ContractOutput {
                pv,
                // the return-based payoff is spot-homogeneous: no spot Greeks
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
                vanna: 0.0,
                charm: 0.0,
                gamma_p: 0.0,
                zomma: 0.0,
                std_err,
                deltas: None,
                vegas: None,
                error: None,
            };
            println!("Cliquet Option Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::Accumulator(acc) => {
            let accumulator = crate::equity::accumulator::Accumulator::try_from_json(acc)?;
            let (pv, std_err) = match accumulator.pricer {
                crate::equity::accumulator::AccumulatorPricer::MonteCarlo => {
                    let (pv, se) = accumulator.mc_npv();
                    (pv, Some(se))
                }
                crate::equity::accumulator::AccumulatorPricer::Analytical => {
                    (accumulator.analytic_npv(), None)
                }
            };
            let contract_output = ContractOutput {
                pv,
                delta: 0.0,
                gamma: 0.0,
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
                vanna: 0.0,
                charm: 0.0,
                gamma_p: 0.0,
                zomma: 0.0,
                std_err,
                deltas: None,
                vegas: None,
                error: None,
            };
            println!("Accumulator Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::VarianceSwap(vs) => {
            let swap = crate::equity::variance_swap::VarianceSwap::try_from_json(vs)?;
            let contract_output = ContractOutput {
                pv: swap.try_npv()?,
                delta: 0.0,
                gamma: 0.0,
                // a variance swap is pure vega-family exposure: report the
                // fair strike diagnostics through the print instead
                vega: 0.0,
                theta: 0.0,
                rho: 0.0,
                vanna: 0.0,
                charm: 0.0,
                gamma_p: 0.0,
                zomma: 0.0,
                std_err: None,
                deltas: None,
                vegas: None,
                error: None,
            };
            println!(
                "Variance Swap MtM: {} (fair strike {:.4} vol)",
                contract_output.pv,
                swap.fair_remaining_variance.sqrt()
            );
            Ok(contract_output)
        }
        #[allow(unreachable_patterns)]
        _ => Err(RustyQLibError::ParseError(
            "unsupported or missing product_type for asset EQ".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(product: serde_json::Value) -> Contract {
        serde_json::from_value(serde_json::json!({
            "action": "PV",
            "asset": "EQ",
            "product_type": product,
        }))
        .expect("test contract must deserialize")
    }

    #[test]
    fn invalid_contract_reports_error_instead_of_panicking() {
        let bad = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "maturity": "2030-01-01",
            "risk_free_rate": 0.05,
            "pricer": "NoSuchEngine",
        }));
        let out = handle_equity_contract(&bad);
        let err = out["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("pricer"), "error should name the field: {err}");
        assert_eq!(out["output"]["pv"], 0.0);
    }

    #[test]
    fn unsupported_engine_combination_reports_error() {
        // autocallable on the analytical engine is refused, not panicked
        let bad = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "autocallable",
            "autocall_barrier": 1.0,
            "protection_barrier": 0.7,
            "volatility": 0.3,
            "maturity": "2030-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }));
        let out = handle_equity_contract(&bad);
        let err = out["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("MonteCarlo"), "should point at the right engine: {err}");
    }

    #[test]
    fn valid_contract_still_prices_with_no_error() {
        let good = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "maturity": "2030-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }));
        let out = handle_equity_contract(&good);
        assert!(out["output"]["error"].is_null());
        assert!(out["output"]["pv"].as_f64().unwrap() > 0.0);
    }
}
