use crate::equity::montecarlo;
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::quotes::Quote;
use super::super::core::traits::{Instrument,Greeks};
use super::blackscholes;

#[derive(PartialEq, Debug)]
pub enum OptionType {
    Call,
    Put,
    Straddle,
}
pub enum Transection {
    Buy,
    Sell,
}

pub enum Engine{
    BlackScholes,
    MonteCarlo
}
impl Instrument for EquityOption  {
    fn npv(&self) -> f64 {
        match self.engine{
            Engine::BlackScholes => {
                 let value = blackscholes::npv(&self);
                value
            }
            Engine::MonteCarlo => {
                println!("BlackScholes");
                let value = blackscholes::npv(&self);
                println!("{}", value);

                println!("MonteCarlo Path Wise");
                let value = montecarlo::npv(&self,true);
                println!("{}", value);

                println!("MonteCarlo ");
                let value = montecarlo::npv(&self,false);
                println!("{}", value);
                value
            }
            _ => {
                0.0
            }
        }
    }
}
pub struct EquityOption {
    pub option_type: OptionType,
    pub transection: Transection,
    pub current_price: Quote,
    pub strike_price: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub time_to_maturity: f64,
    pub term_structure: YieldTermStructure<f64>,
    pub risk_free_rate: f64,
    pub transection_price: f64,
    pub engine: Engine
}
