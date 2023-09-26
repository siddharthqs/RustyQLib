//extern crate ndarray;
use super::vanila_option::{EquityOption};
use super::utils::{Engine};
use crate::core::trade::{OptionType,Transection};
use ndarray::Array2;
pub fn npv(option: &&EquityOption,path_size: bool) -> f64 {
    assert!(option.volatility >= 0.0);
    assert!(option.time_to_maturity >= 0.0);
    assert!(option.current_price.value >= 0.0);
    let num_steps = 1000;

    let dt = option.time_to_maturity / num_steps as f64;
    let discount_factor = (-option.risk_free_rate * dt).exp();
    // Calculate parameters for the binomial tree
    let u = (option.volatility*dt.sqrt()).exp(); //up movement
    let d = 1.0 / u; //down movement
    let factor = (option.risk_free_rate * dt).exp();
    let p = (factor - d) / (u - d); //martingale probability
    // Create a 2D array to represent the binomial tree
    let mut tree = Array2::from_elem((num_steps + 1, num_steps + 1), 0.0);
    //println!("{:?}",tree);
    // Calculate option prices at the final time step (backward induction)
    for j in 0..=num_steps {
        let spot_price_j = option.current_price.value * u.powi(num_steps as i32 - j as i32) * d.powi(j as i32);
        tree[[j,num_steps]] = (spot_price_j - option.strike_price).max(0.0);
    }
    //println!("{:?}",tree);
    for i in (0..num_steps).rev() {
        for j in 0..=i {
            let spot_price_i =  option.current_price.value * u.powi(i as i32 - j as i32) * d.powi(j as i32);
            let discounted_option_price = discount_factor * (p * tree[[ j,i+1]] + (1.0 - p) * tree[[ j + 1,i+1]]);
            tree[[j,i]] = (spot_price_i - option.strike_price).max(discounted_option_price);
        }
    }
    return tree[[0,0]];
}