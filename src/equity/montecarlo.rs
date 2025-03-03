
use std::io;
use chrono::{Datelike, Local, NaiveDate};
use libm::exp;

//use crate::equity::vanila_option::{Engine, EquityOption, OptionType, Transection};
use crate::core::utils::{ContractStyle,dN, N};

use super::vanila_option::{EquityOption, EquityOptionBase, VanillaPayoff};
use super::utils::{Engine, LongShort, Payoff};
use crate::core::trade::{PutOrCall, Transection};
use super::super::utils::RNG;
use crate::core::quotes::Quote;
use crate::core::termstructure::YieldTermStructure;
use crate::core::traits::Instrument;
use serde::de::Unexpected::Option;
use crate::core::trade::PutOrCall::Put;

pub fn simulate_market(option: &EquityOption) -> Vec<f64>{
    let mut monte_carlo = RNG::MonteCarloSimulation{
        antithetic: true,
        moment_matching: true,
        dimentation: 1,
        size: option.simulation.unwrap(),
        standard_normal_vector: vec![] as Vec<f64>,
        standard_normal_matrix: vec![] as Vec<Vec<f64>>
    };
    monte_carlo.set_standard_normal_vector();
    let path = monte_carlo.get_standard_normal_vector();

    let mut market_at_maturity:Vec<f64> = Vec::new();
    for z in path{
        let sim_value = option.base.underlying_price.value()
            *exp(((option.base.risk_free_rate - option.base.dividend_yield - 0.5 * option.base.volatility.powi(2))
            * option.time_to_maturity())+option.base.volatility * option.time_to_maturity().sqrt()*z);
        market_at_maturity.push(sim_value);
    }
    market_at_maturity
}

pub fn simulate_market_path_wise(option: &EquityOption) -> Vec<f64>{
    let M = 1000;
    let N = 10000;
    let dt = option.time_to_maturity()/1000.0;
    let path = RNG::get_matrix_standard_normal(N,M);
    let mut market_at_maturity:Vec<f64> = Vec::new();
    for ipath in &path{
        let mut st = option.base.current_price.value();
        for z in ipath{
            st = st
                *exp(((option.base.risk_free_rate - option.base.dividend_yield - 0.5 * option.base.volatility.powi(2))
                * dt)+option.base.volatility * dt.sqrt()*z);
        }
        market_at_maturity.push(st);
    }
    market_at_maturity
}

pub fn payoff(market: &Vec<f64>,
              strike: &f64,
              option_type: &PutOrCall) -> Vec<f64>{
    let mut payoff_vec = Vec::new();
    match option_type{
        PutOrCall::Call=>{
            for st in market{
                let pay = (st - strike).max(0.0);
                payoff_vec.push(pay);
            }
        }
        PutOrCall::Put=>{
            for st in market{
                let pay = (strike-st).max(0.0);
                payoff_vec.push(pay);
            }
        }
        _ => {}
    }
    payoff_vec
}


pub fn npv(option: &EquityOption,path_size: bool) -> f64 {
    assert!(option.base.volatility >= 0.0);
    assert!(option.base.time_to_maturity() >= 0.0);
    assert!(option.base.underlying_price.value >= 0.0);
    let mut st = vec![];
    if path_size {
        st  = simulate_market_path_wise(&option);
    }
    else {
        //let sim_size = 10000;
        //println!("simulating{}",option.simulation.unwrap());
        st  = simulate_market(&option);
    }

    let payoff = payoff(&st,&option.base.strike_price,&option.payoff.put_or_call());
    let sum_pay:f64 = payoff.iter().sum();
    let num_of_simulations = st.len() as f64;
    let c0:f64 = (sum_pay / num_of_simulations)*exp(-(option.base.risk_free_rate)*option.time_to_maturity());
    c0
    }


pub fn option_pricing() {
    println!("Welcome to the Black-Scholes Option pricer.");
    println!("(Step 1/7) What is the current price of the underlying asset?");
    let mut curr_price = String::new();
    io::stdin()
        .read_line(&mut curr_price)
        .expect("Failed to read line");

    println!("(Step 2/7) Do you want a call option ('C') or a put option ('P') ?");
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
    let mut strike = String::new();
    io::stdin()
        .read_line(&mut strike)
        .expect("Failed to read line");

    println!("Expected annualized volatility in %:");
    println!("E.g.: Enter 50% chance as 0.50 ");
    let mut vol = String::new();
    io::stdin()
        .read_line(&mut vol)
        .expect("Failed to read line");

    println!("Risk-free rate in %:");
    let mut rf = String::new();
    io::stdin().read_line(&mut rf).expect("Failed to read line");

    println!("Maturity date in YYYY-MM-DD format:");
    let mut expiry = String::new();
    io::stdin()
        .read_line(&mut expiry)
        .expect("Failed to read line");
    let future_date = NaiveDate::parse_from_str(&expiry.trim(), "%Y-%m-%d").expect("Invalid date format");
    println!("Dividend yield on this stock:");
    let mut div = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");

    //let ts = YieldTermStructure{
    //    date: vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0],
    //    rates: vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12]
    //};
    let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
    let rates = vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12];
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
    println!("{:?}", option.time_to_maturity());
    let payoff = Box::new(VanillaPayoff{put_or_call:side,exercise_style:ContractStyle::European});
    let equityoption = EquityOption {
        base: option,
        payoff:payoff,
        engine:Engine::BlackScholes,
        simulation: None
    };

    println!("Theoretical Price ${}", equityoption.npv());
    // println!("Premium at risk ${}", option.get_premium_at_risk());
    // println!("Delata {}", option.delta());
    // println!("Gamma {}", option.gamma());
    // println!("Vega {}", option.vega() * 0.01);
    // println!("Theta {}", option.theta() * (1.0 / 365.0));
    // println!("Rho {}", option.rho() * 0.01);
    let mut div1 = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");
}