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
            let contract_output = ContractOutput::from(option.price()?);
            println!("Theoretical Price ${}", contract_output.pv);
            println!("Delta ${}", contract_output.delta);
            Ok(contract_output)
        }
        ProductData::Future(fut) => {
            let future = EquityFuture::try_from_json(fut)?;
            let contract_output = ContractOutput::from(future.price()?);
            println!("Equity Future Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::Forward(forward) => {
            let future = EquityForward::try_from_json(forward)?;
            let contract_output = ContractOutput::from(future.price()?);
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
            let contract_output = ContractOutput::from(cliquet.price()?);
            println!("Cliquet Option Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::Accumulator(acc) => {
            let accumulator = crate::equity::accumulator::Accumulator::try_from_json(acc)?;
            let contract_output = ContractOutput::from(accumulator.price()?);
            println!("Accumulator Price: {}", contract_output.pv);
            Ok(contract_output)
        }
        ProductData::VarianceSwap(vs) => {
            let swap = crate::equity::variance_swap::VarianceSwap::try_from_json(vs)?;
            // a variance swap is pure vega-family exposure: the fair strike
            // diagnostics are reported through the print below
            let contract_output = ContractOutput::from(swap.price()?);
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
    fn explicit_valuation_date_prices_reproducibly() {
        // fixed valuation and maturity: exactly one year, so the pv must
        // hit the Black-Scholes golden value on any run date
        let contract_json = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "2026-01-01",
            "maturity": "2027-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }));
        let out = handle_equity_contract(&contract_json);
        assert!(out["output"]["error"].is_null());
        let pv = out["output"]["pv"].as_f64().unwrap();
        assert!((pv - 14.2312547860).abs() < 1e-8, "pv {pv} must be date-independent");
    }

    #[test]
    fn bad_or_expired_valuation_dates_are_rejected() {
        let bad_date = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "01/01/2026",
            "maturity": "2027-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }));
        let out = handle_equity_contract(&bad_date);
        let err = out["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("valuation_date"), "error should name the field: {err}");

        // valuation after maturity: expired, refused
        let expired = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "2028-01-01",
            "maturity": "2027-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }));
        let out = handle_equity_contract(&expired);
        let err = out["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("maturity"), "error should name the field: {err}");
    }

    #[test]
    fn bermudan_contract_prices_and_requires_dates() {
        let berm = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "P",
            "payoff_type": "vanilla",
            "exercise_style": "Bermudan",
            "exercise_dates": ["2026-04-06", "2026-07-06", "2026-10-05"],
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "2026-01-05",
            "maturity": "2027-01-04",
            "risk_free_rate": 0.05,
            "pricer": "Binomial",
        }));
        let out = handle_equity_contract(&berm);
        assert!(out["output"]["error"].is_null(), "error: {:?}", out["output"]["error"]);
        let pv = out["output"]["pv"].as_f64().unwrap();
        // between the European and American puts for these parameters
        assert!(pv > 9.0 && pv < 11.5, "Bermudan put pv {pv} out of range");

        // Bermudan without dates is rejected naming the field
        let missing = contract(serde_json::json!({
            "product_type": "option",
            "symbol": "ABC",
            "underlying_price": 100.0,
            "put_or_call": "P",
            "payoff_type": "vanilla",
            "exercise_style": "Bermudan",
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "2026-01-05",
            "maturity": "2027-01-04",
            "risk_free_rate": 0.05,
            "pricer": "Binomial",
        }));
        let out = handle_equity_contract(&missing);
        let err = out["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("exercise_dates"), "error should name the field: {err}");
    }

    #[test]
    fn tree_type_flows_through_the_contract() {
        let priced = |tree: &str| {
            let c = contract(serde_json::json!({
                "product_type": "option",
                "symbol": "ABC",
                "underlying_price": 100.0,
                "put_or_call": "P",
                "payoff_type": "vanilla",
                "exercise_style": "American",
                "strike_price": 100.0,
                "volatility": 0.3,
                "valuation_date": "2026-01-05",
                "maturity": "2027-01-05",
                "risk_free_rate": 0.05,
                "pricer": "Binomial",
                "tree_type": tree,
                "tree_steps": 501,
            }));
            handle_equity_contract(&c)
        };
        let lr = priced("LeisenReimer");
        assert!(lr["output"]["error"].is_null());
        let crr = priced("CRR");
        let (lr_pv, crr_pv) =
            (lr["output"]["pv"].as_f64().unwrap(), crr["output"]["pv"].as_f64().unwrap());
        assert!((lr_pv - crr_pv).abs() < 0.05, "schemes agree loosely: {lr_pv} vs {crr_pv}");
        // unknown scheme is rejected naming the field
        let bad = priced("no_such_tree");
        let err = bad["output"]["error"].as_str().expect("error must be set");
        assert!(err.contains("tree_type"), "{err}");
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
