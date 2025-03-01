use crate::core::traits::Instrument;
use crate::core::utils::{Contract,CombinedContract, ContractOutput};
use crate::equity::equity_forward::EquityForward;
use crate::equity::vanila_option::EquityOption;
use crate::equity::equity_future::EquityFuture;
pub fn handle_equity_contract(data: &Contract) -> String {
    match data.product_type.as_deref() {
        Some("Option") => {
            let option = EquityOption::from_json(&data);
            let contract_output = ContractOutput {
                pv: option.npv(),
                delta: option.delta(),
                gamma: option.gamma(),
                vega: option.vega(),
                theta: option.theta(),
                rho: option.rho(),
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
        Some("Future") => {
            let future = EquityFuture::from_json(data);
            let contract_output = ContractOutput {
                pv: future.npv(),
                delta: future.delta(),
                gamma: future.gamma(),
                vega: future.vega(),
                theta: future.theta(),
                rho: future.rho(),
                error: None
            };
            println!("Equity Future Price: {}", contract_output.pv);
            let combined_ = CombinedContract {
                contract: data.clone(),
                output: contract_output
            };
            serde_json::to_string(&combined_).expect("Failed to generate output")
        }
        Some("Forward") => {
            let future = EquityForward::from_json(data);
            let contract_output = ContractOutput {
                pv: future.npv(),
                delta: future.delta(),
                gamma: future.gamma(),
                vega: future.vega(),
                theta: future.theta(),
                rho: future.rho(),
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