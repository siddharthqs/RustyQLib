//extern crate ndarray;
use super::vanila_option::{EquityOption};
use super::utils::{Engine};
use crate::core::trade::{OptionType,Transection};
use crate::core::utils::{ContractStyle};
use ndarray::Array2;
pub fn npv(option: &&EquityOption) -> f64 {
    assert!(option.volatility >= 0.0);
    assert!(option.time_to_maturity >= 0.0);
    assert!(option.current_price.value >= 0.0);
    let num_steps = 1000;

    let dt = option.time_to_maturity / num_steps as f64;
    let discount_factor = (-option.risk_free_rate * dt).exp();
    // Calculate parameters for the binomial tree
    let u = (option.volatility*dt.sqrt()).exp(); //up movement
    let d = 1.0 / u; //down movement
    let a_factor = ((option.risk_free_rate-option.dividend_yield) * dt).exp();
    let p = (a_factor - d) / (u - d); //martingale probability
    // Create a 2D array to represent the binomial tree
    let mut tree = Array2::from_elem((num_steps + 1, num_steps + 1), 0.0);
    //println!("{:?}",tree);
    // Calculate option prices at the final time step (backward induction)
    let multiplier = if option.option_type == OptionType::Call { 1.0 } else { -1.0 };
    for j in 0..=num_steps {
        let spot_price_j = option.current_price.value * u.powi(num_steps as i32 - j as i32) * d.powi(j as i32);
        tree[[j,num_steps]] = (multiplier*(spot_price_j - option.strike_price)).max(0.0);
    }
    //println!("{:?}",tree);
    match option.style {
        ContractStyle::European => {
            for i in (0..num_steps).rev() {
                for j in 0..=i {
                    let spot_price_i =  option.current_price.value * u.powi(i as i32 - j as i32) * d.powi(j as i32);
                    let discounted_option_price = discount_factor * (p * tree[[ j,i+1]] + (1.0 - p) * tree[[ j + 1,i+1]]);
                    //tree[[j,i]] = (multiplier*(spot_price_i - option.strike_price)).max(discounted_option_price);
                    tree[[j,i]] = discounted_option_price;
                }
            }
        }
        ContractStyle::American => {
            println!("American");
            for i in (0..num_steps).rev() {
                for j in 0..=i {
                    let spot_price_i =  option.current_price.value * u.powi(i as i32 - j as i32) * d.powi(j as i32);
                    //let intrinsic_value = (multiplier*(spot_price_i - option.strike_price)).max(0.0);
                    let discounted_option_price = discount_factor * (p * tree[[ j,i+1]] + (1.0 - p) * tree[[ j + 1,i+1]]);
                    tree[[j,i]] = (multiplier*(spot_price_i - option.strike_price)).max(discounted_option_price);
                }
            }
        }
        _ => {
            panic!("Invalid option style");
        }
    }


    return tree[[0,0]];
}