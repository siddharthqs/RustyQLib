use serde::{Deserialize, Serialize};
use std::fs::File;
use std::fs;
use byteorder::{ByteOrder, LittleEndian,BigEndian};
use std::io::Read;
use chrono::{Datelike, Local, NaiveDate};
use crate::core::quotes::Quote;
use crate::core::termstructure::YieldTermStructure;
use crate::equity::vanila_option::{EquityOption};
//use crate::core::utils::{dN, N};
//use super::vanila_option::{EquityOption};
use crate::equity::utils::{Engine};
use crate::cmdty::cmdty_option::{CmdtyOption};
use crate::core::trade;
use crate::cmdty::cmdty_option;
use crate::core::traits::{Instrument, Rates};
use crate::core::utils::{Contract,CombinedContract, ContractOutput, Contracts, OutputJson,EngineType};
use crate::core::utils::ContractStyle;
use crate::core::traits::Greeks;
use std::io::Write;
use std::env::temp_dir;
//use crate::read_csv::read_ts;
use crate::rates;
use crate::rates::deposits::Deposit;
use crate::rates::build_contracts::{build_ir_contracts, build_ir_contracts_from_json, build_term_structure};
use crate::equity::build_contracts::{build_eq_contracts_from_json};
use crate::equity::vol_surface::VolSurface;
use rayon::prelude::*;
/// This function saves the output to a file and returns the path to the file.
pub fn save_to_file<'a>(output_folder: &'a str, subfolder: &'a str, filename: &'a str, output: &'a str) -> String {
    let mut dir = std::path::PathBuf::from(output_folder);
    if subfolder.len() > 0 {
        dir.push(subfolder);
    }
    let _dir = dir.as_path();
    if !_dir.exists() {
        let _ = fs::create_dir(_dir);
    }
    dir.push(filename);
    let mut file = File::create(&dir).expect("Failed to create file");
    file.write_all(output.as_bytes()).expect("Failed to write to file");
    return dir.as_path().to_str().unwrap().to_string();
}

/// This function different types of curves such as term structure, volatility surface, etc.
pub fn build_curve(mut file: &mut File,output_filename: &str)->() {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read JSON file");
    let list_contracts: Contracts = serde_json::from_str(&contents).expect("Failed to deserialize JSON");
    if list_contracts.contracts.len() == 0 {
        panic!("No contracts found in JSON file");
    }
    else if list_contracts.asset=="EQ"{
        println!("Building volatility surface");
        let mut contracts:Vec<Box<EquityOption>> = build_eq_contracts_from_json(list_contracts.contracts);
        let vol_surface = VolSurface::build_eq_vol(contracts);
        let serialized_vol_surface = serde_json::to_string(&vol_surface).unwrap();
        let out_dir = save_to_file(output_filename, "vol_surface", "vol_surface.json", &serialized_vol_surface);
        println!("Volatility surface saved to {}", out_dir);
    }
    else if list_contracts.asset=="CO"{
        //Todo -build commodity vol surface
        panic!("Commodity contracts not supported");
    }
    else if list_contracts.asset=="IR"{
        let mut contracts:Vec<Box<dyn Rates>> = build_ir_contracts_from_json(list_contracts.contracts);
        let ts = build_term_structure(contracts);
        let mut output: String = String::new();
        for i in 0..ts.date.len(){
            output.push_str(&format!("{},{},{}\n",ts.date[i],ts.discount_factor[i],ts.rate[i]));
        }

        let out_dir = save_to_file(output_filename, "term_structure", "term_structure.csv", &output);
        println!("Term structure saved to {}", out_dir);

    }
    else{
        panic!("Asset class not supported");
    }
}

pub fn parse_contract(mut file: &mut File,output_filename: &str) {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read JSON file");

    let list_contracts: Contracts = serde_json::from_str(&contents).expect("Failed to deserialize JSON");

    if list_contracts.contracts.len() == 0 {
        println!("No contracts found in JSON file");
        return;
    }
    // parallel processing of each contract
    let mut output_vec: Vec<_> = list_contracts.contracts.par_iter().enumerate()
        .map(|(index,data)| (index,process_contract(data)))
        .collect();
    output_vec.sort_by_key(|k| k.0);

    let output_vec: Vec<String> = output_vec.into_iter().map(|(_,v)| v).collect();
    let output_str = output_vec.join(",");
    //Write to file
    let mut file = File::create(output_filename).expect("Failed to create file");
    file.write_all(output_str.as_bytes()).expect("Failed to write to file");
}
pub fn process_contract(data: &Contract) -> String {
    //println!("Processing {:?}",data);
    let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
    let rates = vec![0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05];
    let ts = YieldTermStructure::new(date,rates);

    if data.action=="PV" && data.asset=="EQ"{
        //let market_data = data.market_data.clone().unwrap();
        let option = EquityOption::from_json(&data);

        let contract_output = ContractOutput{pv:option.npv(),delta:option.delta(),gamma:option.gamma(),
            vega:option.vega(),theta:option.theta(),rho:option.rho(), error: None };
        println!("Theoretical Price ${}", contract_output.pv);
        println!("Delta ${}", contract_output.delta);
        let combined_ = CombinedContract{
            contract: data.clone(),
            output:contract_output
        };
        let output_json = serde_json::to_string(&combined_).expect("Failed to generate output");
        return output_json;
    }
    else if data.action=="PV" && data.asset=="CO"{
        let market_data = data.market_data.clone().unwrap();
        let curr_quote = Quote::new( market_data.underlying_price);
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
        let vol = Some(market_data.volatility).unwrap();

        let sim = market_data.simulation;
        if data.pricer=="Analytical"{
            let mut option: CmdtyOption = CmdtyOption {
                option_type: side,
                transection: trade::Transection::Buy,
                current_price: curr_quote,
                strike_price: market_data.strike_price,
                volatility: vol.unwrap(),
                time_to_maturity: year_fraction,
                transection_price: 0.0,
                term_structure: ts,
                engine: cmdty_option::Engine::Black76,
                simulation: Option::from(sim.unwrap_or(10000)),
                time_to_future_maturity: None,
                risk_free_rate: None
            };
            let contract_output = ContractOutput{pv:option.npv(),delta:option.delta(),gamma:option.gamma(),vega:option.vega(),theta:option.theta(),rho:option.rho(), error: None };
            println!("Theoretical Price ${}", contract_output.pv);
            println!("Delta ${}", contract_output.delta);
            let combined_ = CombinedContract{
                contract: data.clone(),
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
        let df = deposit.get_discount_factor();
        println!("Discount Factor {:?}",df);
        return "Work in progress".to_string();
    }
    else{
        panic!("Invalid action");
    }
    return "Invalid Action".to_string();

}