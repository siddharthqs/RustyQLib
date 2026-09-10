use crate::core::data_models::ProductData;
use crate::core::errors::RustyQLibError;
use crate::core::utils::Contract;
use crate::equity::vanilla_option::EquityOption;

/// Build every contract in the batch as an [`EquityOption`], reporting the
/// offending contract (by position, and symbol when it has one) instead of
/// panicking when the batch holds a non-option or an invalid contract.
pub fn build_eq_contracts_from_json(
    data: Vec<Contract>,
) -> Result<Vec<Box<EquityOption>>, RustyQLibError> {
    data.iter()
        .enumerate()
        .map(|(index, x)| {
            let Some(ProductData::Option(opt_data)) = &x.product_type else {
                return Err(RustyQLibError::invalid_input(
                    format!("contracts[{index}]"),
                    "not an option contract",
                ));
            };
            // quotes used for implied vol calibration carry a market price but
            // no input vol; seed a placeholder flat vol (the implied solve does
            // not depend on it)
            let mut opt_data = opt_data.clone();
            if opt_data.volatility.is_none() && opt_data.vol_surface.is_none() {
                opt_data.volatility = Some(0.2);
            }
            EquityOption::try_from_json(&opt_data).map_err(|e| {
                RustyQLibError::invalid_input(
                    format!("contracts[{index}] ('{}')", opt_data.base.symbol),
                    e.to_string(),
                )
            })
        })
        .collect()
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

    fn option_contract(symbol: &str) -> Contract {
        contract(serde_json::json!({
            "product_type": "option",
            "symbol": symbol,
            "underlying_price": 100.0,
            "put_or_call": "C",
            "payoff_type": "vanilla",
            "strike_price": 100.0,
            "volatility": 0.3,
            "valuation_date": "2026-01-01",
            "maturity": "2027-01-01",
            "risk_free_rate": 0.05,
            "pricer": "Analytical",
        }))
    }

    #[test]
    fn valid_option_batch_builds() {
        let built =
            build_eq_contracts_from_json(vec![option_contract("ABC"), option_contract("ABC")])
                .expect("valid batch must build");
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].base.symbol, "ABC");
    }

    #[test]
    fn non_option_contract_is_an_error_naming_its_position() {
        let non_option: Contract = serde_json::from_value(serde_json::json!({
            "action": "PV",
            "asset": "EQ",
        }))
        .expect("test contract must deserialize");
        let err = build_eq_contracts_from_json(vec![option_contract("ABC"), non_option])
            .expect_err("a non-option in the batch must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("contracts[1]"),
            "error should name the offending contract: {msg}"
        );
        assert!(msg.contains("not an option"), "{msg}");
    }

    #[test]
    fn invalid_option_contract_is_an_error_not_a_panic() {
        let mut bad = option_contract("ABC");
        // maturity in an unparseable format: try_from_json refuses it
        if let Some(ProductData::Option(opt)) = &mut bad.product_type {
            opt.maturity = "01/01/2027".to_string();
        }
        let err = build_eq_contracts_from_json(vec![bad])
            .expect_err("an invalid contract in the batch must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("contracts[0]") && msg.contains("'ABC'"),
            "error should name the offending contract: {msg}"
        );
        assert!(
            msg.contains("maturity"),
            "error should name the field: {msg}"
        );
    }
}
