//extern crate ndarray;
use super::vanila_option::{EquityOption};
use super::utils::{Engine, Payoff};
use crate::core::trade::{ PutOrCall, Transection};
use crate::core::utils::{ContractStyle};
use ndarray::Array2;

/// Binomial tree model for European and American options
pub fn npv(option: &EquityOption) -> f64 {
    assert!(option.base.volatility >= 0.0);
    assert!(option.time_to_maturity() >= 0.0);
    assert!(option.base.underlying_price.value >= 0.0);
    let num_steps = 1000;

    let dt = option.time_to_maturity() / num_steps as f64;
    let discount_factor = (-option.base.risk_free_rate * dt).exp();
    // Calculate parameters for the binomial tree
    let u = (option.base.volatility*dt.sqrt()).exp(); //up movement
    let d = 1.0 / u; //down movement
    let a_factor = ((option.base.risk_free_rate-option.base.dividend_yield) * dt).exp();
    let p = (a_factor - d) / (u - d); //martingale probability
    // Create a 2D array to represent the binomial tree
    let mut tree = Array2::from_elem((num_steps + 1, num_steps + 1), 0.0);
    //println!("{:?}",tree);
    // Calculate option prices at the final time step (backward induction)
    let multiplier = if option.payoff.put_or_call() == &PutOrCall::Call { 1.0 } else { -1.0 };

    for j in 0..=num_steps {
        let spot_price_j = option.base.underlying_price.value * u.powi(num_steps as i32 - j as i32) * d.powi(j as i32);
        tree[[j,num_steps]] = (multiplier*(spot_price_j - option.base.strike_price)).max(0.0);
    }

    match option.payoff.exercise_style() {
        ContractStyle::European => {
            for i in (0..num_steps).rev() {
                for j in 0..=i {
                    //let spot_price_i =  option.underlying_price.value * u.powi(i as i32 - j as i32) * d.powi(j as i32);
                    let discounted_option_price = discount_factor * (p * tree[[ j,i+1]] + (1.0 - p) * tree[[ j + 1,i+1]]);
                    //tree[[j,i]] = (multiplier*(spot_price_i - option.strike_price)).max(discounted_option_price);
                    tree[[j,i]] = discounted_option_price;
                }
            }

        }
        ContractStyle::American => {

            for i in (0..num_steps).rev() {
                for j in 0..=i {
                    let spot_price_i =  option.base.underlying_price.value * u.powi(i as i32 - j as i32) * d.powi(j as i32);
                    //let intrinsic_value = (multiplier*(spot_price_i - option.strike_price)).max(0.0);
                    let discounted_option_price = discount_factor * (p * tree[[ j,i+1]] + (1.0 - p) * tree[[ j + 1,i+1]]);
                    tree[[j,i]] = (multiplier*(spot_price_i - option.base.strike_price)).max(discounted_option_price);
                }
            }

        }
        _ => {
            panic!("Invalid option style");
        }
    }


    return tree[[0,0]];
}

// Write a unit test for the binomial tree model

// #[cfg(test)]
// mod tests {
//     use assert_approx_eq::assert_approx_eq;
//     use super::*;
//     use crate::core::utils::{Contract,MarketData};
//     use crate::core::trade::{OptionType,Transection};
//     use crate::core::utils::{ContractStyle};
//     use crate::equity::vanila_option::{EquityOption};
//
//     use chrono::{NaiveDate};
//     use crate::core::traits::Instrument;
//
//
//     #[test]
//     fn test_binomial_tree() {
//         let mut data = Contract {
//             action: "PV".to_string(),
//             market_data: Some(MarketData {
//                 underlying_price: 100.0,
//                 strike_price: 100.0,
//                 volatility: Some(0.3),
//                 option_price: Some(10.0),
//                 risk_free_rate: Some(0.05),
//                 dividend: Some(0.0),
//                 maturity: "2024-01-01".to_string(),
//                 option_type: "C".to_string(),
//                 simulation: None
//             }),
//             pricer: "Binomial".to_string(),
//             asset: "".to_string(),
//             style: Some("European".to_string()),
//             rate_data: None
//         };
//         let mut option = EquityOption::from_json(&data);
//         option.valuation_date = NaiveDate::from_ymd(2023, 11, 06);
//         //Call European test
//         let npv = option.npv();
//         assert_approx_eq!(npv, 5.058163, 1e-6);
//         //Call American test
//         option.option_type = OptionType::Call;
//         option.style = ContractStyle::American;
//         let npv = option.npv();
//         assert_approx_eq!(npv, 5.058163, 1e-6);
//
//         //Put European test
//         option.option_type = OptionType::Put;
//         option.style = ContractStyle::European;
//         option.valuation_date = NaiveDate::from_ymd(2023, 11, 07);
//         let npv = option.npv();
//         assert_approx_eq!(npv, 4.259022688, 1e-6);
//
//         //Put American test
//         option.option_type = OptionType::Put;
//         option.style = ContractStyle::American;
//         let npv = option.npv();
//         assert_approx_eq!(npv, 4.315832381, 1e-6);
//     }
// }
