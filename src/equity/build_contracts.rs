//use crate::rates;
//use crate::rates::deposits::Deposit;

use chrono::{NaiveDate,Local,Weekday};
use chrono::Datelike;
use crate::core::trade;
use super::vanila_option::{EquityOption};
use super::super::core::termstructure::YieldTermStructure;
use crate::rates::utils::TermStructure;
use crate::equity::vol_surface::VolSurface;
use crate::rates::utils::{DayCountConvention};
use crate::core::quotes::Quote;
use crate::core::utils::{Contract,ContractStyle};
use crate::equity::utils::{Engine, Payoff};
use std::collections::BTreeMap;
use crate::core::data_models::ProductData;

pub fn build_eq_contracts_from_json(mut data: Vec<Contract>) -> Vec<Box<EquityOption>> {
    let derivatives:Vec<Box<EquityOption>> = data.iter().map(|x| {
        let ProductData::Option(opt_data) = &x.product_type else {
            panic!("Not an option!");
        };
        EquityOption::from_json(opt_data)
    }).collect();
    return derivatives;
}
