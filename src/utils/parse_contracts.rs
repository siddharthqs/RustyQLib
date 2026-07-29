use std::fs::File;
use std::fs;
use std::io::Read;
use chrono::Local;
use crate::equity::vanilla_option::{EquityOption};
use crate::core::traits::Rates;
use crate::core::utils::{Contract, Contracts};
use std::io::Write;
use crate::rates;
use crate::rates::deposits::Deposit;
use crate::rates::build_contracts::{build_ir_contracts_from_json, build_term_structure};
use crate::equity::build_contracts::{build_eq_contracts_from_json};
use crate::equity::handle_equity_contracts::handle_equity_contract;

use rayon::prelude::*;
use serde_json::Value;
use crate::core::serialization::{self, Format};
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
pub fn build_curve(file: &mut File, output_filename: &str) -> () {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read curve definition file");
    let format = Format::detect(&contents);
    let list_contracts: Contracts = serialization::parse(&contents, format)
        .unwrap_or_else(|e| panic!("Failed to read {format:?} curve definition: {e}"));
    if list_contracts.contracts.len() == 0 {
        panic!("No contracts found in JSON file");
    }
    else if list_contracts.asset=="EQ"{
        log::info!("building implied volatility surface");
        let contracts:Vec<Box<EquityOption>> = build_eq_contracts_from_json(list_contracts.contracts);
        let vol_surface = crate::equity::vol_surface::build_implied_vol_surface(&contracts)
            .expect("Failed to build implied vol surface");
        log::debug!("implied vol surface:\n{}", vol_surface);
        let vol_value = serde_json::to_value(&vol_surface).unwrap();
        let serialized_vol_surface =
            serialization::render_value(&vol_value, format, "vol_surface");
        let filename = format!("vol_surface.{}", format.extension());
        let out_dir = save_to_file(output_filename, "vol_surface", &filename, &serialized_vol_surface);
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

/// Price every contract in a document. The input format is detected from
/// the content (JSON or XML) and the output format from the output file
/// extension, defaulting to the input format.
pub fn parse_contract(file: &mut File, output_filename: &str) {
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read contract file");

    let in_format = Format::detect(&contents);
    let out_format = Format::from_path(output_filename).unwrap_or(in_format);

    let list_contracts: Contracts = serialization::parse(&contents, in_format)
        .unwrap_or_else(|e| panic!("Failed to read {in_format:?} contracts: {e}"));

    if list_contracts.contracts.is_empty() {
        log::warn!("no contracts found in the input document; nothing written");
        return;
    }
    // parallel processing of each contract using rayon
    let mut output_vec: Vec<_> = list_contracts.contracts.par_iter().enumerate()
        .map(|(index,data)| (index,process_contract(data)))
        .collect();
    output_vec.sort_by_key(|k| k.0);

    let results: Vec<Value> = output_vec.into_iter().map(|(_,v)| v).collect();
    let output_str = serialization::render_results(&results, out_format);
    //Write to file
    let mut file = File::create(output_filename).expect("Failed to create file");
    file.write_all(output_str.as_bytes()).expect("Failed to write to file");
}
pub fn process_contract(data: &Contract) -> serde_json::Value {

    if data.action=="PV" && data.asset=="EQ"{
        return handle_equity_contract(data);

    }
    else if data.action=="PV" && data.asset=="IR"{
        let rate_data = data.rate_data.clone().unwrap();
        let mut start_date_str = rate_data.start_date; // Only for 0M case
        let mut maturity_date_str = rate_data.maturity_date;
        let current_date = Local::now().date_naive();
        let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
        let start_date = rates::utils::convert_mm_to_date(start_date_str);
        log::debug!("deposit maturity date {:?}", maturity_date);
        let mut deposit = Deposit {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: current_date,
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
        log::debug!("deposit discount factor {:?}", df);
        return serde_json::Value::String("Work in progress".to_string());
    }
    else{
        panic!("Invalid action");
    }
    return serde_json::Value::String("Invalid Action".to_string());

}