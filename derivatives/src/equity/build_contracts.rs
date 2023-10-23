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
use std::collections::HashMap;
pub fn build_eq_contracts(data: Contract)-> Box<EquityOption>{
    let market_data = data.market_data.clone().unwrap();
    let underlying_quote = Quote::new( market_data.underlying_price);
    let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
    let rates = vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12];
    let ts = YieldTermStructure::new(date,rates);
    let option_type = &market_data.option_type;
    let side: trade::OptionType;
    match option_type.trim() {
        "C" | "c" | "Call" | "call" => side = trade::OptionType::Call,
        "P" | "p" | "Put" | "put" => side = trade::OptionType::Put,
        _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
    }
    let maturity_date = &market_data.maturity;
    let today = Local::today();
    let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
    let duration = future_date.signed_duration_since(today.naive_utc());
    let year_fraction = duration.num_days() as f64 / 365.0;
    let rf = Some(market_data.risk_free_rate).unwrap();
    let div = Some(market_data.dividend).unwrap();
    let price = Some(market_data.option_price).unwrap();
    let option_price = Quote::new(price.unwrap());
    let mut option = EquityOption {
        option_type: side,
        transection: trade::Transection::Buy,
        underlying_price: underlying_quote,
        current_price: option_price,
        strike_price: market_data.strike_price,
        volatility: 0.2,
        time_to_maturity: year_fraction,
        risk_free_rate: rf.unwrap_or(0.0),
        dividend_yield: div.unwrap_or(0.0),
        transection_price: 0.0,
        term_structure: ts,
        engine: Engine::BlackScholes,
        simulation: None,
        style: ContractStyle::European,
        //style: Option::from(data.style.as_ref().unwrap_or(&default_style)).map(|x| &**x),
    };
    match data.pricer.trim() {
        "Analytical" |"analytical" => {
            option.engine = Engine::BlackScholes;
        }
        "MonteCarlo" |"montecarlo"|"MC" => {
            option.engine = Engine::MonteCarlo;
        }
        "Binomial"|"binomial" => {
            option.engine = Engine::Binomial;
        }
        _ => {
            panic!("Invalid pricer");}
    }
    match data.style.as_ref().unwrap_or(&"European".to_string()).trim() {
        "European" |"european" => {
            option.style = ContractStyle::European;
        }
        "American" |"american" => {
            option.style = ContractStyle::American;
        }
        _ => {
            option.style = ContractStyle::European;}
    }
    option.set_risk_free_rate();
    option.volatility = option.imp_vol(option.current_price.value);
    return Box::new(option);
    }

pub fn build_eq_contracts_from_json(data: Vec<Contract>) -> Vec<Box<EquityOption>> {
    let mut derivatives:Vec<Box<EquityOption>> = Vec::new();
    for contract in data {
        let eq = build_eq_contracts(contract);
         derivatives.push(eq);
     }
    return derivatives;
}
pub fn build_vol_surface(mut contracts:Vec<Box<EquityOption>>) -> VolSurface {
    //let mut ts:rates::utils::TermStructure = rates::utils::TermStructure::new(vec![],vec![],vec![],
    //                                                                          rates::utils::DayCountConvention::Act360);
    // let mut vol_surface:VolSurface = VolSurface::new(Default::default(), 0.0, spot_date: NaiveDate::from_ymd(, 2020),
    //                                                  DayCountConvention::Act365);
    let mut hash_map:HashMap<NaiveDate,Vec<(f64,f64)>> = HashMap::new();
    let spot_date = Local::today();
    let spot_price = contracts[0].underlying_price.value;
    for i in 0..contracts.len(){
        let mut contract = contracts[i].as_mut();
        let stike = contract.underlying_price.value / contract.strike_price as f64;
        let vol = contract.volatility;
        let maturity = contract.time_to_maturity;
        hash_map.entry(maturity).or_insert(Vec::new()).push((stike,vol));
    }
    let vol_surface:VolSurface = VolSurface::new(hash_map, spot_price, spot_date,
                                                     DayCountConvention::Act365);
    return vol_surface;
}