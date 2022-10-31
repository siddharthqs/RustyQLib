use libm::{exp, log};
use std::f64::consts::{PI, SQRT_2};
use std::{io, thread};
use crate::core::quotes::Quote;
//use utils::{N,dN};
//use vanila_option::{EquityOption,OptionType};
use super::utils::{dN, N};
use super::vanila_option::{EquityOption, OptionType, Transection,Engine};
use super::super::core::termstructure::YieldTermStructure;
use super::super::core::traits::{Instrument,Greeks};
use super::super::core::interpolation;


pub fn npv(bsd_option: &&EquityOption) -> f64 {
    assert!(bsd_option.volatility >= 0.0);
    assert!(bsd_option.time_to_maturity >= 0.0);
    assert!(bsd_option.current_price.value >= 0.0);
    if bsd_option.option_type == OptionType::Call {
        let option_price = bsd_option.current_price.value()
            * N(bsd_option.d1())
            * exp(-bsd_option.dividend_yield * bsd_option.time_to_maturity)
            - bsd_option.strike_price
            * exp(-bsd_option.risk_free_rate * bsd_option.time_to_maturity)
            * N(bsd_option.d2());
        return option_price;
    } else {
        let option_price = -bsd_option.current_price.value()
            * N(-bsd_option.d1())
            * exp(-bsd_option.dividend_yield * bsd_option.time_to_maturity)
            + bsd_option.strike_price
            * exp(-bsd_option.risk_free_rate * bsd_option.time_to_maturity)
            * N(-bsd_option.d2());
        return option_price;
    }
}

impl Greeks for EquityOption{
    fn delta(&self) -> f64 {
        let mut delta = N(self.d1());
        if self.option_type == OptionType::Call {
            delta = delta * exp(-self.dividend_yield * self.time_to_maturity);
        } else if self.option_type == OptionType::Put {
            delta = delta - 1.0;
        }
        return delta;
    }
    fn gamma(&self) -> f64 {
        let gamma = dN(self.d1());
        //(St * sigma * math.sqrt(T - t))
        let var_sqrt = self.volatility * (self.time_to_maturity.sqrt());
        return gamma / (self.current_price.value() * var_sqrt);
    }
    fn vega(&self) -> f64 {
        //St * dN(d1) * math.sqrt(T - t)
        let vega = self.current_price.value() * dN(self.d1()) * self.time_to_maturity.sqrt();
        return vega;
    }
    fn theta(&self) -> f64 {
        let mut theta = 0.0;
        if self.option_type == OptionType::Call {
            //-(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) + r * K * math.exp(-r * (T - t)) * N(d2))
            let t1 = -self.current_price.value() * dN(self.d1()) * self.volatility
                / (2.0 * self.time_to_maturity.sqrt());
            let t2 = (self.risk_free_rate - self.dividend_yield)
                * self.strike_price
                * exp(-(self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
                * N(self.d2());
            theta = t1 + t2;
        } else if self.option_type == OptionType::Put {
            //-(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) - r * K * math.exp(-r * (T - t)) * N(d2))
            let t1 = -self.current_price.value() * dN(self.d1()) * self.volatility
                / (2.0 * self.time_to_maturity.sqrt());
            let t2 = (self.risk_free_rate - self.dividend_yield)
                * self.strike_price
                * exp(-(self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
                * N(self.d2());
            theta = t1 - t2;
        }

        return theta;
    }
    fn rho(&self) -> f64 {
        //rho K * (T - t) * math.exp(-r * (T - t)) * N(d2)
        let mut rho = 0.0;
        if self.option_type == OptionType::Call {
            rho = self.strike_price
                * self.time_to_maturity
                * exp(-(self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
                * N(self.d2());
        } else if self.option_type == OptionType::Put {
            //put_rho = -K * (T - t) * math.exp(-r * (T - t)) * N(-d2)
            rho = -self.strike_price
                * self.time_to_maturity
                * exp(-(self.risk_free_rate - self.dividend_yield) * self.time_to_maturity)
                * N(-self.d2());
        }

        return rho;
    }
}

impl EquityOption {
    pub fn set_risk_free_rate(&mut self){
        let model = interpolation::CubicSpline::new(&self.term_structure.date, &self.term_structure.rates);
        let r = model.interpolation(self.time_to_maturity);
        self.risk_free_rate = r;
    }
    pub fn get_premium_at_risk(&self) -> f64 {
        let value = self.npv();
        let mut pay_off = 0.0;
        if self.option_type == OptionType::Call {
            pay_off = self.current_price.value() - self.strike_price;
        } else if self.option_type == OptionType::Put {
            pay_off = self.strike_price - self.current_price.value();
        }
        if pay_off > 0.0 {
            return value - pay_off;
        } else {
            return value;
        }
    }
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let tmp1 = (self.current_price.value() / self.strike_price).ln()
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility.powi(2))
                * self.time_to_maturity;

        let tmp2 = self.volatility * (self.time_to_maturity.sqrt());
        return tmp1 / tmp2;
    }
    pub fn d2(&self) -> f64 {
        let d2 = self.d1() - self.volatility * self.time_to_maturity.powf(0.5);
        return d2;
    }
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

    let side: OptionType;
    match side_input.trim() {
        "C" | "c" | "Call" | "call" => side = OptionType::Call,
        "P" | "p" | "Put" | "put" => side = OptionType::Put,
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

    println!("Time to maturity in years");
    let mut expiry = String::new();
    io::stdin()
        .read_line(&mut expiry)
        .expect("Failed to read line");

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
    let curr_quote = Quote{value: curr_price.trim().parse::<f64>().unwrap()};
    let mut option = EquityOption {
        option_type: side,
        transection: Transection::Buy,
        current_price: curr_quote,
        strike_price: strike.trim().parse::<f64>().unwrap(),
        volatility: vol.trim().parse::<f64>().unwrap(),
        time_to_maturity: expiry.trim().parse::<f64>().unwrap(),
        risk_free_rate: rf.trim().parse::<f64>().unwrap(),
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        transection_price: 0.0,
        term_structure: ts,
        engine: Engine::BlackScholes,
    };
    option.set_risk_free_rate();
    println!("Theoretical Price ${}", option.npv());
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
