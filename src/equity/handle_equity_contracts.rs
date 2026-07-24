use crate::core::traits::Instrument;
use crate::core::utils::{Contract,CombinedContract, ContractOutput};
use crate::core::data_models::ProductData;
use crate::equity::equity_forward::EquityForward;
use crate::equity::vanilla_option::EquityOption;
use crate::equity::equity_future::EquityFuture;
pub fn handle_equity_contract(data: &Contract) -> serde_json::Value {
    match &data.product_type {
        ProductData::Option(opt) => {
            let option = EquityOption::from_json(opt);
            let (pv, std_err) = match option.engine {
                crate::equity::utils::Engine::MonteCarlo => {
                    let stats = crate::equity::montecarlo::npv_with_stats(&option);
                    (stats.pv, Some(stats.std_err))
                }
                _ => (option.npv(), None),
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
            let combined_ = CombinedContract{
                contract: data.clone(),
                output:contract_output
            };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::Future(fut) => {
            let future = EquityFuture::from_json(fut);
            let contract_output = ContractOutput {
                pv: future.npv(),
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
            let combined_ = CombinedContract {
                contract: data.clone(),
                output: contract_output
            };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::Forward(forward) => {
            let future = EquityForward::from_json(forward);
            let contract_output = ContractOutput {
                pv: future.npv(),
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
            let combined_ = CombinedContract {
                contract: data.clone(),
                output: contract_output
            };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::RainbowOption(rb) => {
            let option = crate::equity::rainbow::RainbowOption::from_json(rb);
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
            let combined_ = CombinedContract { contract: data.clone(), output: contract_output };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::CliquetOption(cq) => {
            let cliquet = crate::equity::cliquet::Cliquet::from_json(cq);
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
            let combined_ = CombinedContract { contract: data.clone(), output: contract_output };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::Accumulator(acc) => {
            let accumulator = crate::equity::accumulator::Accumulator::from_json(acc);
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
            let combined_ = CombinedContract { contract: data.clone(), output: contract_output };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        ProductData::VarianceSwap(vs) => {
            let swap = crate::equity::variance_swap::VarianceSwap::from_json(vs);
            let contract_output = ContractOutput {
                pv: swap.npv(),
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
            let combined_ = CombinedContract { contract: data.clone(), output: contract_output };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        #[allow(unreachable_patterns)]
        _ => {
            panic!("Unsupported or missing product_type for asset EQ");
        }
    }
}
