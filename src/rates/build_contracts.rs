use crate::rates;
use crate::rates::deposits::Deposit;
use chrono::{NaiveDate,Local,Weekday};
use chrono::Datelike;
use crate::rates::fra::FRA;
use crate::core::traits::{Instrument,Rates};
use crate::core::utils::{Contract, Contracts};
use crate::rates::utils::TermStructure;

pub fn build_ir_contracts(data: Contract) -> Box<dyn Rates> {
    let rate_data = data.rate_data.clone().unwrap();
    let mut start_date_str = rate_data.start_date; // Only for 0M case
    let mut maturity_date_str = rate_data.maturity_date;
    let current_date = Local::today();
    let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
    let start_date = rates::utils::convert_mm_to_date(start_date_str);
    if rate_data.instrument.as_str() == "Deposit" {
        let mut deposit = Deposit {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: current_date.naive_utc(),
            notional: rate_data.notional,
            fix_rate: rate_data.fix_rate,
            day_count: rates::utils::DayCountConvention::Act360,
            business_day_adjustment: 0,
            term_structure: None
        };
        match rate_data.day_count.as_str() {
            "Act360" |"A360" => {
                deposit.day_count = rates::utils::DayCountConvention::Act360;
            }
            "Act365" |"A365" => {
                deposit.day_count = rates::utils::DayCountConvention::Act365;
            }
            "Thirty360" |"30/360" => {
                deposit.day_count = rates::utils::DayCountConvention::Thirty360;
            }
            _ => {}
        }
        let mut ird:Box<dyn Rates> = Box::new(deposit);
        return ird;
    }
    else if rate_data.instrument.as_str()=="FRA" {
        //let mut start_date_str = rate_data.start_date;
        //let mut maturity_date_str = rate_data.maturity_date;
        //let current_date = Local::today();
        //let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
        //let start_date = rates::utils::convert_mm_to_date(start_date_str);
        let mut fra = FRA {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: current_date.naive_utc(),
            notional: rate_data.notional,
            currency: rate_data.currency,
            fix_rate: rate_data.fix_rate,
            day_count: rates::utils::DayCountConvention::Act360,
            business_day_adjustment: 0,
            term_structure: None
        };
        match rate_data.day_count.as_str() {
            "Act360" |"A360" => {
                fra.day_count = rates::utils::DayCountConvention::Act360;
            }
            "Act365" |"A365" => {
                fra.day_count = rates::utils::DayCountConvention::Act365;
            }
            "Thirty360" |"30/360" => {
                fra.day_count = rates::utils::DayCountConvention::Thirty360;
            }
            _ => {}
        }
        let ird:Box<dyn Rates> = Box::new(fra);
        return ird;
    }
    else {
        panic!("Invalid asset");
    }
}

pub fn build_ir_contracts_from_json(data: Vec<Contract>) -> Vec<Box<dyn Rates>> {
    let mut irds:Vec<Box<dyn Rates>> = Vec::new();
    for contract in data {
        let ird = build_ir_contracts(contract);
        irds.push(ird);
    }
    return irds;
}
pub fn build_term_structure(mut contracts:Vec<Box<dyn Rates>>) -> TermStructure {
    let mut ts:rates::utils::TermStructure = rates::utils::TermStructure::new(vec![],vec![],vec![],
                                                                              rates::utils::DayCountConvention::Act360);
    let mut contract = contracts[0].as_mut();
    ts.discount_factor.push(contract.get_maturity_discount_factor());
    ts.date.push(contract.get_maturity_date());
    ts.rate.push(contract.get_rate());
    for i in 1..contracts.len(){
        let mut contract = contracts[i].as_mut();
        contract.set_term_structure(ts.clone());
        ts.discount_factor.push(contract.get_maturity_discount_factor());
        ts.date.push(contract.get_maturity_date());
        ts.rate.push(contract.get_rate());
    }
    return ts
}