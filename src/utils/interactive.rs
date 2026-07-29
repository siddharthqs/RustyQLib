//! Interactive terminal wizards for the `interactive` CLI subcommand.
//!
//! These read from stdin and print to stdout, so they live behind the
//! `cli` feature rather than in the pricing modules.

use std::io;

use chrono::{Local, NaiveDate};

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::montecarlo::{npv_with_stats, MonteCarloConfig};
use crate::equity::utils::{LongShort, Model, PricingEngine};
use crate::equity::vanilla_option::{EquityMarketData, EquityOption, EquityOptionBase, VanillaPayoff};

/// Prompt for the terms of a European vanilla option and price it with
/// the Black-Scholes engine, printing the price and Greeks.
pub fn black_scholes_pricing() {
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
    let _d = expiry.trim();
    let future_date = NaiveDate::parse_from_str(&_d, "%Y-%m-%d").expect("Invalid date format");
    println!("Dividend yield on this stock:");
    print!(">>");
    let mut div = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");

    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        rf.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .expect("Invalid risk free rate");
    let vol_surface = VolSurface::flat(
        vol.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
    )
    .expect("Invalid volatility");
    let curr_quote = Quote::new(curr_price.trim().parse::<f64>().unwrap());
    let base = EquityOptionBase {
        symbol: "ABC".to_string(),
        currency: None,
        exchange: None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: Some("ABC".to_string()),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        maturity_date: future_date,
        futures_settlement: None,
        multiplier: 1.0,
        current_price: Quote::new(0.0),
        entry_price: 0.0,
        long_short: LongShort::LONG,
    };
    let market = EquityMarketData {
        valuation_date,
        spot: curr_quote,
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        borrow_cost: 0.0,
        cash_dividends: vec![],
        vol_surface,
        discount_curve,
    };
    let payoff = Box::new(VanillaPayoff {
        put_or_call: side,
        exercise_style: ContractStyle::European,
    });
    let option = EquityOption {
        base,
        market,
        payoff,
        engine: PricingEngine::BlackScholes,
        model: Model::Gbm,
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

/// Prompt for the terms of a European vanilla option and price it with
/// the Monte Carlo engine, printing the price and its standard error.
pub fn monte_carlo_pricing() {
    println!("Welcome to the Monte Carlo Option pricer.");
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

    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        rf.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .expect("Invalid risk free rate");
    let vol_surface = VolSurface::flat(
        vol.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
    )
    .expect("Invalid volatility");
    let curr_quote = Quote::new(curr_price.trim().parse::<f64>().unwrap());
    let base = EquityOptionBase {
        symbol: "ABC".to_string(),
        currency: None,
        exchange: None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: Some("ABC".to_string()),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        maturity_date: future_date,
        futures_settlement: None,
        multiplier: 1.0,
        current_price: Quote::new(0.0),
        entry_price: 0.0,
        long_short: LongShort::LONG,
    };
    let market = EquityMarketData {
        valuation_date,
        spot: curr_quote,
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        borrow_cost: 0.0,
        cash_dividends: vec![],
        vol_surface,
        discount_curve,
    };
    let payoff = Box::new(VanillaPayoff {
        put_or_call: side,
        exercise_style: ContractStyle::European,
    });
    let equityoption = EquityOption {
        base,
        market,
        payoff,
        engine: PricingEngine::MonteCarlo(MonteCarloConfig::default()),
        model: Model::Gbm,
    };

    let result = npv_with_stats(&equityoption);
    println!("Theoretical Price ${} (std err {})", result.pv, result.std_err);
    let mut wait = String::new();
    io::stdin()
        .read_line(&mut wait)
        .expect("Failed to read line");
}
