use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::Read;
use chrono::{Datelike, Local, NaiveDate};
use crate::core::quotes::Quote;
use crate::core::termstructure::YieldTermStructure;
use crate::equity::vanila_option::{EquityOption};
use crate::core::utils::{dN, N};
//use super::vanila_option::{EquityOption};
use crate::equity::utils::{Engine};
use crate::cmdty::cmdty_option::{CmdtyOption};
use crate::core::trade;
use crate::cmdty::cmdty_option;
use crate::core::traits::Instrument;
use crate::core::utils;
use crate::core::utils::{CombinedContract, ContractOutput, Contracts, OutputJson,EngineType};
use crate::core::utils::ContractStyle;
use crate::core::traits::Greeks;
use std::io::Write;
use crate::read_csv::read_ts;
use crate::rates;
use crate::rates::deposits::Deposit;

pub fn build_curve(mut file: &mut File,output_filename: &str)->() {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read JSON file");

    let list_contracts: utils::Contracts = serde_json::from_str(&contents).expect("Failed to deserialize JSON");

    let mut contracts:Vec<Deposit> = Vec::new();
    contracts = list_contracts.contracts.into_iter()
        .map(|x| build_contracts(x)).collect();
    //let mut ts :rates::utils::TermStructure = rates::utils::TermStructure::new(vec![],vec![],vec![]);
    let dates = contracts.iter().map(|x| x.get_maturity_date()).collect();
    let rates = contracts.iter().map(|x| x.get_rate()).collect();
    let df = contracts.iter().map(|x| x.get_discount_factor()).collect();
    let day_count = contracts[0].day_count.clone();
    let ts = rates::utils::TermStructure::new(dates,df,rates,day_count);
    println!("Term Structure {:?}",ts);



}
pub fn build_contracts(data: utils::Contract) -> Deposit {
    if data.asset=="IR"{
        let rate_data = data.rate_data.clone().unwrap();
        let mut start_date_str = rate_data.start_date; // Only for 0M case
        let mut maturity_date_str = rate_data.maturity_date;
        let current_date = Local::today();
        let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
        let start_date = rates::utils::convert_mm_to_date(start_date_str);
        println!("Maturity Date {:?}",maturity_date);
        let mut deposit = rates::deposits::Deposit {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: current_date.naive_utc(),
            notional: rate_data.notional,
            fix_rate: rate_data.fix_rate,
            day_count: rates::utils::DayCountConvention::Act360,
            business_day_adjustment: 0
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
        let df = deposit.get_discount_factor();
        println!("Discount Factor {:?}",df);
        return deposit;
    }
    else {
        panic!("Invalid asset");
    }
}

pub fn parse_contract(mut file: &mut File,output_filename: &str) {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read JSON file");

    let list_contracts: utils::Contracts = serde_json::from_str(&contents).expect("Failed to deserialize JSON");

    //let data: utils::Contract = serde_json::from_str(&contents).expect("Failed to deserialize JSON");
    //let mut output: String = String::new();
    let mut output_vec:Vec<String> = Vec::new();
    for data in list_contracts.contracts.into_iter() {
        output_vec.push(process_contract(data));
    }

    let mut file = File::create(output_filename).expect("Failed to create file");
    //let mut output:OutputJson = OutputJson{contracts:output_vec};
    let output_str = output_vec.join(",");
    //let output_json = serde_json::to_string(&output_vec).expect("Failed to generate output");
    file.write_all(output_str.as_bytes()).expect("Failed to write to file");
}
pub fn process_contract(data: utils::Contract) -> String {
    //println!("Processing {:?}",data);
    let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
    let rates = vec![0.05,0.05,0.06,0.07,0.08,0.9,0.9,0.10];
    let ts = YieldTermStructure::new(date,rates);


    if data.action=="PV" && data.asset=="EQ"{
        let market_data = data.market_data.clone().unwrap();
        let sim = market_data.simulation;
        let curr_quote = Quote{value: market_data.underlying_price};
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
        let mut option = EquityOption {
            option_type: side,
            transection: trade::Transection::Buy,
            current_price: curr_quote,
            strike_price: market_data.strike_price,
            volatility: market_data.volatility,
            time_to_maturity: year_fraction,
            risk_free_rate: rf.unwrap_or(0.0),
            dividend_yield: div.unwrap_or(0.0),
            transection_price: 0.0,
            term_structure: ts,
            engine: Engine::BlackScholes,
            simulation: Option::from(sim.unwrap_or(10000)),
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
        let contract_output = utils::ContractOutput{pv:option.npv(),delta:option.delta(),gamma:option.gamma(),vega:option.vega(),theta:option.theta(),rho:option.rho(), error: None };
        println!("Theoretical Price ${}", contract_output.pv);
        println!("Delta ${}", contract_output.delta);
        let combined_ = utils::CombinedContract{
            contract: data,
            output:contract_output
        };
        let output_json = serde_json::to_string(&combined_).expect("Failed to generate output");
        return output_json;
    }
    else if data.action=="PV" && data.asset=="CO"{
        let market_data = data.market_data.clone().unwrap();
        let curr_quote = Quote{value: market_data.underlying_price};
        let option_type = &market_data.option_type;
        let side: trade::OptionType;
        match option_type.trim() {
            "C" | "c" | "Call" | "call" => side = trade::OptionType::Call,
            "P" | "p" | "Put" | "put" => side = trade:: OptionType::Put,
            _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
        }
        let maturity_date = &market_data.maturity;
        let today = Local::today();
        let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let duration = future_date.signed_duration_since(today.naive_utc());
        let year_fraction = duration.num_days() as f64 / 365.0;

        let sim = market_data.simulation;
        if data.pricer=="Analytical"{
            let mut option: CmdtyOption = CmdtyOption {
                option_type: side,
                transection: trade::Transection::Buy,
                current_price: curr_quote,
                strike_price: market_data.strike_price,
                volatility: market_data.volatility,
                time_to_maturity: year_fraction,
                transection_price: 0.0,
                term_structure: ts,
                engine: cmdty_option::Engine::Black76,
                simulation: Option::from(sim.unwrap_or(10000)),
                time_to_future_maturity: None,
                risk_free_rate: None
            };
            let contract_output = utils::ContractOutput{pv:option.npv(),delta:option.delta(),gamma:option.gamma(),vega:option.vega(),theta:option.theta(),rho:option.rho(), error: None };
            println!("Theoretical Price ${}", contract_output.pv);
            println!("Delta ${}", contract_output.delta);
            let combined_ = utils::CombinedContract{
                contract: data,
                output:contract_output
            };
            let output_json = serde_json::to_string(&combined_).expect("Failed to generate output");
            return output_json;


        }

    }
    else if data.action=="PV" && data.asset=="IR"{
        //println!("Processing {:?}",data);
        let rate_data = data.rate_data.clone().unwrap();
        let mut start_date_str = rate_data.start_date; // Only for 0M case
        let mut maturity_date_str = rate_data.maturity_date;
        let current_date = Local::today();
        let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
        let start_date = rates::utils::convert_mm_to_date(start_date_str);
        println!("Maturity Date {:?}",maturity_date);
        let mut deposit = rates::deposits::Deposit {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: current_date.naive_utc(),
            notional: rate_data.notional,
            fix_rate: rate_data.fix_rate,
            day_count: rates::utils::DayCountConvention::Act360,
            business_day_adjustment: 0
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
        let df = deposit.get_discount_factor();
        println!("Discount Factor {:?}",df);
        return "Work in progress".to_string();
    }
    else{
        panic!("Invalid action");
    }
    return "Invalid Action".to_string();

}