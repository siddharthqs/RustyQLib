use std::error::Error;
use chrono::{Datelike, Local, NaiveDate};
use crate::equity::{binomial,finite_difference,montecarlo};
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::quotes::Quote;
use super::super::core::traits::{Instrument,Greeks};
use super::blackscholes;
use crate::equity::utils::{Engine,PayoffType,Payoff};
use crate::core::trade::{OptionType,Transection};
use crate::core::utils::{Contract,ContractStyle};
use crate::core::{interpolation, trade};
use serde::Deserialize;
use blackscholes::BlackScholesPricer;

#[derive(Debug)]
pub struct VanillaPayoff {
    pub option_type: OptionType,
}
#[derive(Debug)]
pub struct BinaryPayoff {
    pub option_type: OptionType,
}
#[derive(Debug)]
pub struct BarrierPayoff {
    pub option_type: OptionType,
}
#[derive(Debug)]
pub struct AsianPayoff {
    pub option_type: OptionType,
}
impl Payoff for VanillaPayoff {
    fn payoff_amount(&self, base: &EquityOptionBase) -> f64 {
        // implement vanilla payoff
        let intrinsic_value = base.underlying_price.value() - base.strike_price;
        match &self.option_type {
            OptionType::Call=> intrinsic_value.max(0.0),
            OptionType::Put=> (-intrinsic_value).max(0.0),
            _=>0.0
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Vanilla
    }
    fn option_type(&self)->OptionType{
        self.option_type.clone()
    }
}

#[derive(Debug)]
pub struct EquityOptionBase {
    pub underlying_price: Quote,
    pub current_price: Quote,
    pub strike_price: f64,
    pub transection: Transection,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub term_structure: YieldTermStructure<f64>,
    pub risk_free_rate: f64,
    pub transection_price: f64,
    pub style: ContractStyle,
}
#[derive(Debug)]
pub struct EquityOption {
    pub base: EquityOptionBase,
    pub payoff: Box<dyn Payoff>,
    pub engine: Engine,
    pub simulation: Option<u64>,

}
impl EquityOption{
    pub fn time_to_maturity(&self) -> f64{
        let time_to_maturity = (self.base.maturity_date - self.base.valuation_date).num_days() as f64/365.0;
        time_to_maturity
    }
    pub fn option_type(&self)->OptionType{
        self.payoff.option_type().clone()
    }
}
impl EquityOption {
    //pub fn new(x:T)->Self{
    //
    //}
    pub fn from_json(data: &Contract) -> Box<EquityOption> {
        let payoff_type = data.payoff_type.as_ref().unwrap().parse::<PayoffType>();
        let market_data = data.market_data.as_ref().unwrap();
        let option_type = &market_data.option_type;
        let side: OptionType;
        match option_type.trim() {
            "C" | "c" | "Call" | "call" => side = OptionType::Call,
            "P" | "p" | "Put" | "put" => side = OptionType::Put,
            _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
        }
        let payoff:Box<dyn Payoff> = match payoff_type {
            Ok(PayoffType::Vanilla) => Box::new(VanillaPayoff{option_type:side}),
            //Ok(PayoffType::Binary) => Box::new(BinaryPayoff{option_type:side}),
            //Ok(PayoffType::Barrier) => Box::new(BarrierPayoff{option_type:side}),
            //Ok(PayoffType::Asian) => Box::new(AsianPayoff{option_type:side}),
            _ => {Box::new(VanillaPayoff{option_type:side})}
        };

        let underlying_quote = Quote::new(market_data.underlying_price);
        //TODO: Add term structure
        let date = vec![0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0];
        let rates = vec![0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05];
        let ts = YieldTermStructure::new(date, rates);


        let maturity_date = &market_data.maturity;
        let today = Local::today();
        let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");

        let risk_free_rate = Some(market_data.risk_free_rate).unwrap();
        let dividend = Some(market_data.dividend).unwrap();
        //let mut op = 0.0;

        let option_price = Quote::new(match market_data.option_price {
            Some(x) => x,
            None => 0.0,
        });
        //let volatility = Some(market_data.volatility);
        let volatility = match market_data.volatility {
            Some(x) => {
                x
            }
            None => 0.2
        };
        let mut option = EquityOptionBase {
            transection: Transection::Buy,
            underlying_price: underlying_quote,
            current_price: option_price,
            strike_price: market_data.strike_price,
            volatility: volatility,
            maturity_date: future_date,
            risk_free_rate: risk_free_rate.unwrap_or(0.0),
            dividend_yield: dividend.unwrap_or(0.0),
            transection_price: 0.0,
            term_structure: ts,
            style: ContractStyle::European,
            valuation_date: today.naive_utc(),
        };
        option.set_risk_free_rate();
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
        let equityoption = EquityOption {
            base: option,
            payoff,
            engine: match data.pricer.trim() {
                "Analytical" | "analytical" | "bs" => Engine::BlackScholes,
                "MonteCarlo" | "montecarlo" | "MC" | "mc" => Engine::MonteCarlo,
                "Binomial" | "binomial" | "bino" => Engine::Binomial,
                "FiniteDifference" | "finitdifference" | "FD" | "fd" => Engine::FiniteDifference,
                _ => {
                    panic!("Invalid pricer");
                }
            },
            simulation: None
        };

        return Box::new(equityoption);
    }
}

impl EquityOptionBase {
    pub fn time_to_maturity(&self) -> f64{
        let time_to_maturity = (self.maturity_date - self.valuation_date).num_days() as f64/365.0;
        time_to_maturity
    }
    pub fn set_risk_free_rate(&mut self) {
        let model = interpolation::CubicSpline::new(&self.term_structure.date, &self.term_structure.rates);
        let r = model.interpolation(self.time_to_maturity());
        self.risk_free_rate = r;
    }
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let d1_numerator = (self.underlying_price.value() / self.strike_price).ln()
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility.powi(2))
            * self.time_to_maturity();

        let d1_denominator = self.volatility * (self.time_to_maturity().sqrt());
        return d1_numerator / d1_denominator;
    }
    pub fn d2(&self) -> f64 {
        let d2 = self.d1() - self.volatility * self.time_to_maturity().powf(0.5);
        return d2;
    }
}
impl EquityOption {
    pub fn get_premium_at_risk(&self) -> f64 {
        let value = self.npv();
        let mut pay_off = self.payoff.payoff_amount(&self.base);
        if pay_off > 0.0 {
            return value - pay_off;
        } else {
            return value;
        }
    }
    
    pub fn imp_vol(&mut self,option_price:f64) -> f64 {
        for i in 0..100{
            let d_sigma = (self.npv()-option_price)/self.vega();
            self.base.volatility -= d_sigma
        }
        self.base.volatility
    }
    pub fn get_imp_vol(&mut self) -> f64 {
        for i in 0..100{
            let d_sigma = (self.npv()-self.base.current_price.value)/self.vega();
            self.base.volatility -= d_sigma
        }
        self.base.volatility
    }
}


impl Instrument for EquityOption  {
    fn npv(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.npv(&self);
                value
            }
            _ => { 0.0 }
        }
        //     Engine::MonteCarlo => {
        //
        //         let value = montecarlo::npv(&self,false);
        //         value
        //     }
        //     Engine::Binomial => {
        //
        //         let value = binomial::npv(&self);
        //         value
        //     }
        //     Engine::FiniteDifference => {
        //         let value = finite_difference::npv(&self);
        //         value
        //     }

    }
}

impl EquityOption {
    pub fn delta(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.delta(&self);
                value
            }
            _ => { 0.0 }
        }
    }
    pub fn gamma(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.gamma(&self);
                value
            }
            _ => { 0.0 }
        }
    }
    pub fn vega(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.vega(&self);
                value
            }
            _ => { 0.0 }
        }
    }
    pub fn theta(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.theta(&self);
                value
            }
            _ => { 0.0 }
        }
    }
    pub fn rho(&self) -> f64 {
        match self.engine {
            Engine::BlackScholes => {
                let pricer = BlackScholesPricer::new();
                let value = pricer.rho(&self);
                value
            }
            _ => { 0.0 }
        }
    }
}
// #[cfg(test)]
// mod tests {
//     //write a unit test for from_json
//     use super::*;
//     use crate::core::utils::{Contract,MarketData};
//     use crate::core::trade::OptionType;
//     use crate::core::trade::Transection;
//     use crate::core::utils::ContractStyle;
//     use crate::core::termstructure::YieldTermStructure;
//     use crate::core::quotes::Quote;
//     use chrono::{Datelike, Local, NaiveDate};
//     #[test]
//     fn test_from_json() {
//         let data = Contract {
//             action: "PV".to_string(),
//             market_data: Some(MarketData {
//                 underlying_price: 100.0,
//                 strike_price: 100.0,
//                 volatility: None,
//                 option_price: Some(10.0),
//                 risk_free_rate: Some(0.05),
//                 dividend: Some(0.0),
//                 maturity: "2024-01-01".to_string(),
//                 option_type: "C".to_string(),
//                 simulation: None
//             }),
//             pricer: "Analytical".to_string(),
//             asset: "".to_string(),
//             style: Some("European".to_string()),
//             rate_data: None
//         };
//         let option = EquityOption::from_json(&data);
//         assert_eq!(option.option_type, OptionType::Call);
//         assert_eq!(option.transection, Transection::Buy);
//         assert_eq!(option.underlying_price.value, 100.0);
//         assert_eq!(option.strike_price, 100.0);
//         assert_eq!(option.current_price.value, 10.0);
//         assert_eq!(option.dividend_yield, 0.0);
//         assert_eq!(option.volatility, 0.2);
//         assert_eq!(option.maturity_date, NaiveDate::from_ymd(2024, 1, 1));
//         assert_eq!(option.valuation_date, Local::today().naive_utc());
//         assert_eq!(option.engine, Engine::BlackScholes);
//         assert_eq!(option.style, ContractStyle::European);
//     }
// }

