//use crate::rates;
//use crate::rates::deposits::Deposit;

use chrono::{NaiveDate,Local,Weekday};
use chrono::Datelike;
use crate::core::trade;
use super::vanilla_option::{EquityOption};
use super::super::core::termstructure::YieldTermStructure;
use crate::rates::utils::TermStructure;
use crate::core::vols::VolSurface;
use crate::rates::utils::{DayCountConvention};
use crate::core::quotes::Quote;
use crate::core::utils::{Contract,ContractStyle};
use crate::equity::utils::{Engine, Payoff};
use std::collections::BTreeMap;
use crate::core::data_models::ProductData;

pub fn build_eq_contracts_from_json(data: Vec<Contract>) -> Vec<Box<EquityOption>> {
    let derivatives:Vec<Box<EquityOption>> = data.iter().map(|x| {
        let ProductData::Option(opt_data) = &x.product_type else {
            panic!("Not an option!");
        };
        // quotes used for implied vol calibration carry a market price but
        // no input vol; seed a placeholder flat vol (the implied solve does
        // not depend on it)
        let mut opt_data = opt_data.clone();
        if opt_data.volatility.is_none() && opt_data.vol_surface.is_none() {
            opt_data.volatility = Some(0.2);
        }
        EquityOption::from_json(&opt_data)
    }).collect();
    return derivatives;
}
