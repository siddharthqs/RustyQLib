use libm::{exp, log};
use std::f64::consts::{PI, SQRT_2};
use std::{io, thread};
use crate::core::quotes::Quote;
use chrono::{Datelike, Local, NaiveDate};
//use utils::{N,dN};
//use vanila_option::{EquityOption,OptionType};
use crate::core::utils::{ContractStyle, dN, N};
use crate::core::trade::{PutOrCall, Transection};
use super::vanila_option::{EquityOption, EquityOptionBase, VanillaPayoff};
use super::utils::{Engine, PayoffType, Payoff, LongShort};
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::traits::{Instrument,Greeks};
use super::super::core::interpolation;

pub struct BlackScholesPricer;
impl BlackScholesPricer {
    pub fn new() -> Self {
        BlackScholesPricer
    }
    pub fn npv(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(bsd_option.time_to_maturity() >= 0.0, "Option is expired or negative time");
        assert!(bsd_option.base.underlying_price.value >= 0.0, "Negative underlying price not allowed");
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.npv_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    pub fn delta(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(bsd_option.time_to_maturity() >= 0.0, "Option is expired or negative time");
        assert!(bsd_option.base.underlying_price.value >= 0.0, "Negative underlying price not allowed");
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.delta_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    pub fn gamma(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.gamma_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    pub fn vega(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.vega_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    pub fn theta(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.theta_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    pub fn rho(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.theta_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    fn npv_vanilla(&self, bsd_option: &EquityOption) -> f64 {

        let n_d1 = N(bsd_option.base.d1());
        let n_d2 = N(bsd_option.base.d2());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {bsd_option.base.underlying_price.value()*n_d1 *df_d
                -bsd_option.base.strike_price*n_d2*df_r
            }
            PutOrCall::Put => {bsd_option.base.strike_price*N(-bsd_option.base.d2())*df_r-
                bsd_option.base.underlying_price.value()*N(-bsd_option.base.d1()) *df_d
                }

        }
    }
    fn delta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        let n_d1 = N(bsd_option.base.d1());

        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {n_d1 *(df_r/df_d) }
            PutOrCall::Put => {(n_d1-1.0) *(df_r/df_d) }
        }
    }
    fn gamma_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        let dn_d1 = dN(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());
        let num = dn_d1*(df_r/df_d);
        let var_sqrt = bsd_option.base.volatility * (bsd_option.time_to_maturity().sqrt());
        num/ (bsd_option.base.underlying_price.value() * var_sqrt)
    }
    fn vega_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        let dn_d1 = dN(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());
        let df_S = bsd_option.base.underlying_price.value()*df_r/df_d;
        let vega = df_S * dn_d1 * bsd_option.time_to_maturity().sqrt();
        vega
    }
    fn theta_vanilla(&self, bsd_option: &EquityOption) -> f64 {

        //let vol_time_sqrt = bsd_option.base.volatility * (bsd_option.time_to_maturity().sqrt());
        let dn_d1 = dN(bsd_option.base.d1());
        let n_d1 = N(bsd_option.base.d1());
        let n_d2 = N(bsd_option.base.d2());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());
        let df_S = bsd_option.base.underlying_price.value()*df_r/df_d;
        let t1 = -df_S*dn_d1  * bsd_option.base.volatility
            / (2.0 * bsd_option.time_to_maturity().sqrt());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                let t2 = -df_S*n_d1*(bsd_option.base.dividend_yield-bsd_option.base.risk_free_rate);
                let t3 = -bsd_option.base.risk_free_rate*bsd_option.base.strike_price*df_r*n_d2;
                t1+t2+t3
            }
            PutOrCall::Put => {
                let t2 = df_S*N(-bsd_option.base.d1())*(bsd_option.base.dividend_yield-bsd_option.base.risk_free_rate);
                let t3 = bsd_option.base.risk_free_rate*bsd_option.base.strike_price*df_r*N(-bsd_option.base.d2());
                t1+t2+t3
            }

        }
    }
    fn rho_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        //let n_d1 = N(bsd_option.base.d1());
        let n_d2 = N(bsd_option.base.d2());
        //let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = exp(-bsd_option.base.risk_free_rate * bsd_option.time_to_maturity());
        let r1 = bsd_option.time_to_maturity()*bsd_option.base.strike_price;
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                r1*n_d2*df_r
            }
            PutOrCall::Put => {r1*N(-bsd_option.base.d2())*df_r
            }

        }
    }

}


pub fn option_pricing() {
    println!("Welcome to the Black-Scholes Option pricer.");
    print!(">>");
    println!(" What is the current price of the underlying asset?");
    print!(">>");
    let mut curr_price = String::new();
    io::stdin()
        .read_line(&mut curr_price)
        .expect("Failed to read line");
    println!(" Do you want a call option ('C') or a put option ('P') ?");
    print!(">>");
    let mut side_input = String::new();
    io::stdin()
        .read_line(&mut side_input)
        .expect("Failed to read line");
    let side: PutOrCall;
    match side_input.trim() {
        "C" | "c" | "Call" | "call" => side = PutOrCall::Call,
        "P" | "p" | "Put" | "put" => side = PutOrCall::Put,
        _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
    }
    println!("Stike price:");
    print!(">>");
    let mut strike = String::new();
    io::stdin()
        .read_line(&mut strike)
        .expect("Failed to read line");
    println!("Expected annualized volatility in %:");
    println!("E.g.: Enter 50% chance as 0.50 ");
    print!(">>");
    let mut vol = String::new();
    io::stdin()
        .read_line(&mut vol)
        .expect("Failed to read line");

    println!("Risk-free rate in %:");
    print!(">>");
    let mut rf = String::new();
    io::stdin().read_line(&mut rf).expect("Failed to read line");
    println!(" Maturity date in YYYY-MM-DD format:");

    let mut expiry = String::new();
    println!("E.g.: Enter 2020-12-31 for 31st December 2020");
    print!(">>");
    io::stdin()
        .read_line(&mut expiry)
        .expect("Failed to read line");
    println!("{:?}", expiry.trim());
    let _d = expiry.trim();
    let future_date = NaiveDate::parse_from_str(&_d, "%Y-%m-%d").expect("Invalid date format");
    //println!("{:?}", future_date);
    println!("Dividend yield on this stock:");
    print!(">>");
    let mut div = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");

    //let ts = YieldTermStructure{
    //    date: vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0],
    //    rates: vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12]
    //};
    let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
    let rates = vec![0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05];
    let ts = YieldTermStructure::new(date,rates);
    let curr_quote = Quote::new( curr_price.trim().parse::<f64>().unwrap());
    let mut option = EquityOptionBase {

        symbol:"ABC".to_string(),
        currency: None,
        exchange:None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: Some("ABC".to_string()),
        entry_price: 0.0,
        long_short: LongShort::LONG,
        underlying_price: curr_quote,
        current_price: Quote::new(0.0),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        volatility: vol.trim().parse::<f64>().unwrap(),
        maturity_date: future_date,
        risk_free_rate: rf.trim().parse::<f64>().unwrap(),
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        term_structure: ts,
        valuation_date: Local::today().naive_local(),
        multiplier: 1.0,
    };
    option.set_risk_free_rate();
    //println!("{:?}", option.time_to_maturity());
    let payoff = Box::new(VanillaPayoff{put_or_call:side,
                                    exercise_style:ContractStyle::European});
    let option = EquityOption {
        base: option,
        payoff:payoff,
        engine:Engine::BlackScholes,
        simulation: None
    };
    println!("Theoretical Price ${}", option.npv());
    println!("Premium at risk ${}", option.get_premium_at_risk());
    println!("Delta {}", option.delta());
    println!("Gamma {}", option.gamma());
    println!("Vega {}", option.vega() * 0.01);
    println!("Theta {}", option.theta() * (1.0 / 365.0));
    println!("Rho {}", option.rho() * 0.01);
    let mut wait = String::new();
    io::stdin()
        .read_line(&mut wait)
        .expect("Failed to read line");
}
pub fn implied_volatility(){}
// pub fn implied_volatility() {
//     println!("Welcome to the Black-Scholes Option pricer.");
//     println!("(Step 1/7) What is the current price of the underlying asset?");
//     let mut curr_price = String::new();
//     io::stdin()
//         .read_line(&mut curr_price)
//         .expect("Failed to read line");
//
//     println!("(Step 2/7) Do you want a call option ('C') or a put option ('P') ?");
//     let mut side_input = String::new();
//     io::stdin()
//         .read_line(&mut side_input)
//         .expect("Failed to read line");
//
//     let side: OptionType;
//     match side_input.trim() {
//         "C" | "c" | "Call" | "call" => side = OptionType::Call,
//         "P" | "p" | "Put" | "put" => side = OptionType::Put,
//         _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
//     }
//
//     println!("Stike price:");
//     let mut strike = String::new();
//     io::stdin()
//         .read_line(&mut strike)
//         .expect("Failed to read line");
//
//     println!("What is option price:");
//     let mut option_price = String::new();
//     io::stdin()
//         .read_line(&mut option_price)
//         .expect("Failed to read line");
//
//     println!("Risk-free rate in %:");
//     let mut rf = String::new();
//     io::stdin().read_line(&mut rf).expect("Failed to read line");
//
//     println!(" Maturity date in YYYY-MM-DD format:");
//     let mut expiry = String::new();
//     io::stdin()
//         .read_line(&mut expiry)
//         .expect("Failed to read line");
//     let future_date = NaiveDate::parse_from_str(&expiry.trim(), "%Y-%m-%d").expect("Invalid date format");
//     println!("Dividend yield on this stock:");
//     let mut div = String::new();
//     io::stdin()
//         .read_line(&mut div)
//         .expect("Failed to read line");
//
//     //let ts = YieldTermStructure{
//     //    date: vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0],
//     //    rates: vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12]
//     //};
//     let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
//     let rates = vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12];
//     let ts = YieldTermStructure::new(date,rates);
//     let curr_quote = Quote::new( curr_price.trim().parse::<f64>().unwrap());
//     let sim = Some(10000);
//     let mut option = EquityOption {
//         option_type: side,
//         transection: Transection::Buy,
//         underlying_price: curr_quote,
//         current_price: Quote::new(0.0),
//         strike_price: strike.trim().parse::<f64>().unwrap(),
//         volatility: 0.20,
//         maturity_date: future_date,
//         risk_free_rate: rf.trim().parse::<f64>().unwrap(),
//         dividend_yield: div.trim().parse::<f64>().unwrap(),
//         transection_price: 0.0,
//         term_structure: ts,
//         engine: Engine::BlackScholes,
//         simulation:sim,
//         //style:Option::from("European".to_string()),
//         style: ContractStyle::European,
//         valuation_date: Local::today().naive_utc(),
//     };
//     option.set_risk_free_rate();
//     println!("Implied Volatility  {}%", 100.0*option.imp_vol(option_price.trim().parse::<f64>().unwrap()));
//
//     let mut div1 = String::new();
//     io::stdin()
//         .read_line(&mut div)
//         .expect("Failed to read line");
// }

// #[cfg(test)]
// mod tests {
//
//     use assert_approx_eq::assert_approx_eq;
//     use super::*;
//     use crate::core::utils::{Contract,MarketData};
//     use crate::core::trade::{OptionType,Transection};
//     use crate::core::utils::{ContractStyle};
//     use crate::equity::vanila_option::{EquityOption};
//
//     #[test]
//     fn test_black_scholes() {
//         let mut data = Contract {
//             action: "PV".to_string(),
//             market_data: Some(MarketData {
//                 underlying_price: 100.0,
//                 strike_price: 100.0,
//                 volatility: Some(0.3),
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
//
//         let mut option = EquityOption::from_json(&data);
//         option.valuation_date = NaiveDate::from_ymd(2023, 11, 06);
//         //Call European test
//         let npv = option.npv();
//         assert_approx_eq!(npv, 5.05933313, 1e-6);
//
//         //Put European test
//         option.option_type = OptionType::Put;
//         option.style = ContractStyle::European;
//         option.valuation_date = NaiveDate::from_ymd(2023, 11, 07);
//         let npv = option.npv();
//         assert_approx_eq!(npv,4.2601813, 1e-6);
//
//     }
// }

