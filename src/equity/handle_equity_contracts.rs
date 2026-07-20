use crate::core::traits::Instrument;
use crate::core::utils::{Contract,CombinedContract, ContractOutput};
use crate::core::data_models::ProductData;
use crate::equity::equity_forward::EquityForward;
use crate::equity::vanila_option::EquityOption;
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
                std_err: stats.map(|s| s.std_err),
                deltas: Some(option.deltas()),
                vegas: Some(option.vegas()),
                error: None,
            };
            println!("Rainbow Option Price: {}", contract_output.pv);
            let combined_ = CombinedContract { contract: data.clone(), output: contract_output };
            serde_json::to_value(&combined_).expect("Failed to generate output")
        }
        #[allow(unreachable_patterns)]
        _ => {
            panic!("Unsupported or missing product_type for asset EQ");
        }
    }
}