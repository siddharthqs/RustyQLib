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
use crate::equity::utils::{Engine};
use std::collections::BTreeMap;

pub fn build_eq_contracts_from_json(data: Vec<Contract>) -> Vec<Box<EquityOption>> {
    let derivatives:Vec<Box<EquityOption>> = data.iter().map(|x| EquityOption::from_json(x.clone())).collect();
    return derivatives;
}
pub fn build_volatility_surface(mut contracts:Vec<Box<EquityOption>>) -> VolSurface {

    let mut vol_tree:BTreeMap<NaiveDate,Vec<(f64,f64)>> = BTreeMap::new();
    let spot_date = contracts[0].valuation_date;
    let spot_price = contracts[0].underlying_price.value;
    for i in 0..contracts.len(){
        let mut contract = contracts[i].as_mut();
        let moneyness = contract.underlying_price.value / contract.strike_price as f64;
        let volatility = contract.get_imp_vol();
        let maturity = contract.maturity_date;
        vol_tree.entry(maturity).or_insert(Vec::new()).push((moneyness,volatility));
    }
    let vol_surface:VolSurface = VolSurface::new(vol_tree, spot_price, spot_date,
                                                     DayCountConvention::Act365);
    return vol_surface;
}