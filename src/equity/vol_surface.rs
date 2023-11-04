use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use crate::rates::utils::{DayCountConvention};
use chrono::{NaiveDate};
use crate::core::trade::OptionType;
use super::vanila_option::{EquityOption};

/// Vol Surface is a collection of volatilities for different maturities and strikes
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct VolSurface{
    pub term_structure: BTreeMap<NaiveDate, Vec<(f64,f64)>>,
    pub spot: f64,
    pub spot_date: NaiveDate,
    pub day_count: DayCountConvention,
}

impl VolSurface {
    pub fn new(term_structure: BTreeMap<NaiveDate, Vec<(f64,f64)>>,spot:f64,spot_date:NaiveDate,day_count:DayCountConvention) -> VolSurface {
        VolSurface {
            term_structure,
            spot,
            spot_date,
            day_count
        }
    }
    pub fn get_vol(&self,val_date:NaiveDate,maturity_date:NaiveDate,strike:f64)-> f64{
        //TODO: Interpolate Vol Surface
        0.0
    }
    pub fn get_year_fraction(&self,val_date:NaiveDate,maturity_date:NaiveDate) -> f64 {
        self.day_count.get_year_fraction(val_date,maturity_date)
    }
    pub fn build_eq_vol(mut contracts:Vec<Box<EquityOption>>) -> VolSurface {
        let mut vol_tree:BTreeMap<NaiveDate,Vec<(f64,f64)>> = BTreeMap::new();
        let spot_date = contracts[0].valuation_date;
        let spot_price = contracts[0].underlying_price.value;
        for i in 0..contracts.len(){
            let mut moneyness=1.0;
            let mut contract = contracts[i].as_mut();
            match contract.option_type{
                OptionType::Call => {
                    moneyness = contract.underlying_price.value / contract.strike_price as f64;

                },
                OptionType::Put => {
                    moneyness = contract.strike_price / contract.underlying_price.value  as f64;
                }
                _ => {
                    panic!("Option type not supported");
                }
            }
            let volatility = contract.get_imp_vol();
            let maturity = contract.maturity_date;
            vol_tree.entry(maturity).or_insert(Vec::new()).push((moneyness,volatility));
        }
        let vol_surface:VolSurface = VolSurface::new(vol_tree, spot_price, spot_date,
                                                     DayCountConvention::Act365);
        return vol_surface;
    }

}

// #[cfg(test)]
// mod tests{
//     use super::*;
//     use crate::core::quotes::Quote;
//     use crate::core::utils::{Contract,ContractStyle};
//     use crate::equity::utils::{Engine};
//     use crate::core::trade::OptionType;
//     use crate::core::trade::OptionStyle;
//     use crate::core::trade::OptionStyle::European;
//
//     #[test]
//     fn test_build_eq_vol(){
//         // write a unit test for this function
//         let mut contract = C
//         let mut contracts:Vec<Box<EquityOption>> = Vec::new();
//
//
// }