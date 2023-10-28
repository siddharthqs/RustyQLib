use chrono::{Datelike, Local, NaiveDate};
use crate::equity::montecarlo;
use crate::equity::binomial;
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::quotes::Quote;
use super::super::core::traits::{Instrument,Greeks};
use super::blackscholes;
use crate::equity::utils::{Engine};
use crate::core::trade::{OptionType,Transection};
use crate::core::utils::{Contract,ContractStyle};
use crate::core::trade;

impl Instrument for EquityOption  {
    fn npv(&self) -> f64 {
        match self.engine{
            Engine::BlackScholes => {
                 let value = blackscholes::npv(&self);
                value
            }
            Engine::MonteCarlo => {
                println!("Using MonteCarlo ");
                let value = montecarlo::npv(&self,false);
                value
            }
            Engine::Binomial => {
                println!("Using Binomial ");
                let value = binomial::npv(&self);
                value
            }
            _ => {
                0.0
            }
        }
    }
}

#[derive(Debug)]
pub struct EquityOption {
    pub option_type: OptionType,
    pub transection: Transection,
    pub underlying_price: Quote,
    pub current_price: Quote,
    pub strike_price: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub term_structure: YieldTermStructure<f64>,
    pub risk_free_rate: f64,
    pub transection_price: f64,
    pub engine: Engine,
    pub simulation:Option<u64>,
    pub style: ContractStyle,
}
impl EquityOption{
    pub fn time_to_maturity(&self) -> f64{
        let time_to_maturity = (self.maturity_date - self.valuation_date).num_days() as f64/365.0;
        time_to_maturity
    }
}
impl EquityOption {
    pub fn equityoption_from_json(data: Contract) -> Box<EquityOption> {
        let market_data = data.market_data.unwrap();
        let underlying_quote = Quote::new(market_data.underlying_price);
        //TODO: Add term structure
        let date = vec![0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0];
        let rates = vec![0.05, 0.055, 0.06, 0.07, 0.07, 0.08, 0.1, 0.1];
        let ts = YieldTermStructure::new(date, rates);
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

        let risk_free_rate = Some(market_data.risk_free_rate).unwrap();
        let dividend = Some(market_data.dividend).unwrap();
        let option_price = Quote::new(match Some(market_data.option_price) {
            Some(x) => x.unwrap(),
            None => 0.0,
        });
        let mut option = EquityOption {
            option_type: side,
            transection: trade::Transection::Buy,
            underlying_price: underlying_quote,
            current_price: option_price,
            strike_price: market_data.strike_price,
            volatility: 0.2,
            maturity_date: future_date,
            risk_free_rate: risk_free_rate.unwrap_or(0.0),
            dividend_yield: dividend.unwrap_or(0.0),
            transection_price: 0.0,
            term_structure: ts,
            engine: Engine::BlackScholes,
            simulation: None,
            style: ContractStyle::European,
            valuation_date: today.naive_utc(),
        };
        match data.pricer.trim() {
            "Analytical" | "analytical" => {
                option.engine = Engine::BlackScholes;
            }
            "MonteCarlo" | "montecarlo" | "MC" => {
                option.engine = Engine::MonteCarlo;
            }
            "Binomial" | "binomial" => {
                option.engine = Engine::Binomial;
            }
            _ => {
                panic!("Invalid pricer");
            }
        }
        match data.style.as_ref().unwrap_or(&"European".to_string()).trim() {
            "European" | "european" => {
                option.style = ContractStyle::European;
            }
            "American" | "american" => {
                option.style = ContractStyle::American;
            }
            _ => {
                option.style = ContractStyle::European;
            }
        }
        option.set_risk_free_rate();
        return Box::new(option);
    }
}
