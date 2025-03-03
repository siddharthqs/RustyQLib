use std::error::Error;
use chrono::{Datelike, Local, NaiveDate};
use crate::equity::{binomial,finite_difference,montecarlo};
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::quotes::Quote;
use super::super::core::traits::{Instrument,Greeks};
use super::blackscholes;
use crate::equity::utils::{Engine, PayoffType, Payoff, LongShort};
use crate::core::trade::{PutOrCall,Transection};
use crate::core::utils::{Contract,ContractStyle};
use crate::core::{interpolation, trade};
use serde::Deserialize;
use blackscholes::BlackScholesPricer;
use crate::core::data_models::EquityOptionData;

#[derive(Debug)]
pub struct VanillaPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct BinaryPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct BarrierPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
#[derive(Debug)]
pub struct AsianPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}
impl Payoff for VanillaPayoff {
    fn payoff_amount(&self, base: &EquityOptionBase) -> f64 {
        // implement vanilla payoff
        let intrinsic_value = base.underlying_price.value() - base.strike_price;
        match &self.put_or_call {
            PutOrCall::Call=> intrinsic_value.max(0.0),
            PutOrCall::Put=> (-intrinsic_value).max(0.0),
            _=>0.0
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Vanilla
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }

}

#[derive(Debug)]
pub struct EquityOptionBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    //pub payoff_type: String, // Vanilla/Barrier/Binary
    pub underlying_price: Quote,
    pub current_price: Quote,
    pub strike_price: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub term_structure: YieldTermStructure<f64>,
    pub risk_free_rate: f64,
    pub entry_price: f64,
    pub long_short: LongShort,
    pub multiplier: f64,

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


}
impl EquityOption {

    pub fn from_json(data: &EquityOptionData) -> Box<EquityOption> {
        let date = vec![0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0];
        let rates = vec![0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05];
        let ts = YieldTermStructure::new(date, rates);
        let maturity_date = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").expect("Invalid date format");
        let mut base_option = EquityOptionBase {
            symbol:data.base.symbol.clone(),
            currency: data.base.currency.clone(),
            exchange:data.base.exchange.clone(),
            name: data.base.name.clone(),
            cusip: data.base.cusip.clone(),
            isin: data.base.isin.clone(),
            settlement_type: data.base.settlement_type.clone(),

            underlying_price: Quote::new(data.base.underlying_price),
            current_price: Quote::new(data.current_price.unwrap_or(0.0)),
            strike_price: data.strike_price,
            volatility: data.volatility,
            maturity_date,
            risk_free_rate: data.base.risk_free_rate.unwrap_or(0.0),
            entry_price: data.entry_price.unwrap_or(0.0),
            long_short: LongShort::LONG,
            dividend_yield: data.dividend.unwrap_or(0.0),
            term_structure: ts,
            valuation_date: Local::today().naive_utc(),
            multiplier: data.multiplier.unwrap_or(1.0),
        };
        let payoff_type = &data.payoff_type.parse::<PayoffType>().unwrap();
        let side: PutOrCall;
        let put_or_call = data.put_or_call.clone();
        match put_or_call.trim() {
            "C" | "c" | "Call" | "call" => side = PutOrCall::Call,
            "P" | "p" | "Put" | "put" => side = PutOrCall::Put,
            _ => panic!("Invalid side argument! Side has to be either 'C' or 'P'."),
        }

        let style = match data.exercise_style.as_ref().unwrap_or(&"European".to_string()).trim() {
            "European" | "european" => {
                ContractStyle::European
            }
            "American" | "american" => {
                ContractStyle::American
            }
            _ => {
                ContractStyle::European
            }
        };

        let payoff:Box<dyn Payoff> = match &payoff_type {
            PayoffType::Vanilla => Box::new(VanillaPayoff{
                put_or_call:side,
                exercise_style:style}),
            //Ok(PayoffType::Binary) => Box::new(BinaryPayoff{option_type:side}),
            //Ok(PayoffType::Barrier) => Box::new(BarrierPayoff{option_type:side}),
            //Ok(PayoffType::Asian) => Box::new(AsianPayoff{option_type:side}),
            _ => {Box::new(VanillaPayoff{
                put_or_call:side,
                exercise_style:style})}
        };

        base_option.set_risk_free_rate();
        let equityoption = EquityOption {
            base: base_option,
            payoff,
            engine: match data.pricer.as_ref().map_or("Analytical",|v| v).trim() {
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
        Box::new(equityoption)
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

