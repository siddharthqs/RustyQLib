use crate::core::traits::Instrument;
use crate::core::utils::{Contract,CombinedContract, ContractOutput};
use crate::core::data_models::ProductData;
use crate::equity::equity_forward::EquityForward;
use crate::equity::vanila_option::EquityOption;
use crate::equity::equity_future::EquityFuture;
pub fn handle_equity_contract(data: &Contract) -> String {
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
                error: None
            };
            println!("Theoretical Price ${}", contract_output.pv);
            println!("Delta ${}", contract_output.delta);
            let combined_ = CombinedContract{
                contract: data.clone(),
                output:contract_output
            };
            serde_json::to_string(&combined_).expect("Failed to generate output")
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
                error: None
            };
            println!("Equity Future Price: {}", contract_output.pv);
            let combined_ = CombinedContract {
                contract: data.clone(),
                output: contract_output
            };
            serde_json::to_string(&combined_).expect("Failed to generate output")
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
                error: None
            };
            println!("Equity Forward Price: {}", contract_output.pv);
            let combined_ = CombinedContract {
                contract: data.clone(),
                output: contract_output
            };
            serde_json::to_string(&combined_).expect("Failed to generate output")
        }
        _ => {
            panic!("Unsupported or missing product_type for asset EQ");
        }
    }
}