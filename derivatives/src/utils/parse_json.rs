use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use chrono::{Local, NaiveDate};
use crate::core::quotes::Quote;
use crate::core::termstructure::YieldTermStructure;
use crate::equity::vanila_option::{Engine, EquityOption, OptionType, Transection};
use crate::core::traits::Instrument;
#[derive(Deserialize)]
struct MarketData {
    underlying_price:f64,
    option_type:String,
    strike_price:f64,
    volatility:f64,
    risk_free_rate:Option<f64>,
    maturity:String,
    dividend: Option<f64>,
    simulation:Option<u64>
}

#[derive(Deserialize)]
struct contract {
    action: String,
    pricer: String,
    asset: String,
    market_data: MarketData,
}
pub fn parse_contract(mut file: &mut File){
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read JSON file");
    let data: contract = serde_json::from_str(&contents).expect("Failed to deserialize JSON");
    if data.action=="PV" && data.asset=="EQ"{
        let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
        let rates = vec![0.05,0.05,0.06,0.07,0.08,0.9,0.9,0.10];
        let ts = YieldTermStructure::new(date,rates);
        let curr_quote = Quote{value: data.market_data.underlying_price};
        let option_type = data.market_data.option_type;
        let side: OptionType;
        match option_type.trim() {
            "C" | "c" | "Call" | "call" => side = OptionType::Call,
            "P" | "p" | "Put" | "put" => side = OptionType::Put,
            _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
        }
        let maturity_date = data.market_data.maturity;
        let today = Local::today();
        let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let duration = future_date.signed_duration_since(today.naive_utc());
        let year_fraction = duration.num_days() as f64 / 365.0;
        let rf = Some(data.market_data.risk_free_rate).unwrap();
        let div = Some(data.market_data.dividend).unwrap();
        let sim = data.market_data.simulation;
        if data.pricer=="BS"{
            let mut option = EquityOption {
                option_type: side,
                transection: Transection::Buy,
                current_price: curr_quote,
                strike_price: data.market_data.strike_price,
                volatility: data.market_data.volatility,
                time_to_maturity: year_fraction,
                risk_free_rate: rf.unwrap_or(0.0),
                dividend_yield: div.unwrap_or(0.0),
                transection_price: 0.0,
                term_structure: ts,
                engine: Engine::BlackScholes,
                simulation: Option::from(sim.unwrap_or(10000))
            };
            option.set_risk_free_rate();
            println!("Theoretical Price ${}", option.npv());
        }
        else if data.pricer=="MC" {
            let mut option = EquityOption {
                option_type: side,
                transection: Transection::Buy,
                current_price: curr_quote,
                strike_price: data.market_data.strike_price,
                volatility: data.market_data.volatility,
                time_to_maturity: year_fraction,
                risk_free_rate: rf.unwrap_or(0.0),
                dividend_yield: div.unwrap_or(0.0),
                transection_price: 0.0,
                term_structure: ts,
                engine: Engine::MonteCarlo,
                simulation:Option::from(sim.unwrap_or(10000))
            };
            option.set_risk_free_rate();
            println!("Theoretical Price ${}", option.npv());
        }

    }



}