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
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use super::super::core::traits::{Instrument,Greeks};

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
            PayoffType::Vanilla => self.rho_vanilla(bsd_option),
            _ => {0.0}
        }
    }
    fn npv_vanilla(&self, bsd_option: &EquityOption) -> f64 {

        let n_d1 = N(bsd_option.base.d1());
        let n_d2 = N(bsd_option.base.d2());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_r = bsd_option.base.maturity_discount_factor();
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
        // spot delta: e^{-qT} N(d1) for a call, e^{-qT}(N(d1)-1) for a put
        let n_d1 = N(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {n_d1 * df_d }
            PutOrCall::Put => {(n_d1-1.0) * df_d }
        }
    }
    fn gamma_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // e^{-qT} dN(d1) / (S sigma sqrt(T))
        let dn_d1 = dN(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let var_sqrt = bsd_option.base.volatility * (bsd_option.time_to_maturity().sqrt());
        dn_d1 * df_d / (bsd_option.base.underlying_price.value() * var_sqrt)
    }
    fn vega_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // S e^{-qT} dN(d1) sqrt(T)
        let dn_d1 = dN(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.dividend_yield * bsd_option.time_to_maturity());
        let df_S = bsd_option.base.underlying_price.value() * df_d;
        let vega = df_S * dn_d1 * bsd_option.time_to_maturity().sqrt();
        vega
    }
    fn theta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
        // put:  -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
        let q = bsd_option.base.dividend_yield;
        let r = bsd_option.base.risk_free_rate();
        let k = bsd_option.base.strike_price;
        let dn_d1 = dN(bsd_option.base.d1());
        let n_d1 = N(bsd_option.base.d1());
        let n_d2 = N(bsd_option.base.d2());
        let df_d = exp(-q * bsd_option.time_to_maturity());
        let df_r = bsd_option.base.maturity_discount_factor();
        let df_S = bsd_option.base.underlying_price.value() * df_d;
        let t1 = -df_S * dn_d1 * bsd_option.base.volatility
            / (2.0 * bsd_option.time_to_maturity().sqrt());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                t1 + q * df_S * n_d1 - r * k * df_r * n_d2
            }
            PutOrCall::Put => {
                t1 - q * df_S * N(-bsd_option.base.d1()) + r * k * df_r * N(-bsd_option.base.d2())
            }
        }
    }
    fn rho_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: K T e^{-rT} N(d2); put: -K T e^{-rT} N(-d2)
        let n_d2 = N(bsd_option.base.d2());
        let df_r = bsd_option.base.maturity_discount_factor();
        let r1 = bsd_option.time_to_maturity()*bsd_option.base.strike_price;
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                r1*n_d2*df_r
            }
            PutOrCall::Put => {-r1*N(-bsd_option.base.d2())*df_r
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

    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        rf.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .expect("Invalid risk free rate");
    let curr_quote = Quote::new( curr_price.trim().parse::<f64>().unwrap());
    let option = EquityOptionBase {

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
        discount_curve,
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        valuation_date,
        multiplier: 1.0,
    };
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


#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::core::utils::ContractStyle;

    /// S=100, K=100, sigma=30%, q=0, T=1y (2026-01-01 -> 2027-01-01, Act/365).
    fn test_option(put_or_call: PutOrCall, curve: YieldCurve) -> EquityOption {
        let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let base = EquityOptionBase {
            symbol: "TEST".to_string(),
            currency: None,
            exchange: None,
            name: None,
            cusip: None,
            isin: None,
            settlement_type: None,
            underlying_price: Quote::new(100.0),
            current_price: Quote::new(0.0),
            strike_price: 100.0,
            dividend_yield: 0.0,
            volatility: 0.3,
            maturity_date: NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(),
            valuation_date,
            discount_curve: curve,
            entry_price: 0.0,
            long_short: LongShort::LONG,
            multiplier: 1.0,
        };
        EquityOption {
            base,
            payoff: Box::new(VanillaPayoff {
                put_or_call,
                exercise_style: ContractStyle::European,
            }),
            engine: Engine::BlackScholes,
            simulation: None,
        }
    }

    fn flat_5pct() -> YieldCurve {
        YieldCurve::flat(
            0.05,
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    // Golden values computed independently (erf-based reference implementation)
    #[test]
    fn golden_call_npv_and_greeks() {
        let option = test_option(PutOrCall::Call, flat_5pct());
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.delta(), 0.6242517279, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        assert_approx_eq!(option.theta(), -8.1011898970, 1e-8);
        assert_approx_eq!(option.rho(), 48.1939180046, 1e-8);
    }

    #[test]
    fn golden_put_npv_and_greeks() {
        let option = test_option(PutOrCall::Put, flat_5pct());
        assert_approx_eq!(option.npv(), 9.3541972361, 1e-8);
        assert_approx_eq!(option.delta(), -0.3757482721, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        assert_approx_eq!(option.theta(), -3.3450427745, 1e-8);
        assert_approx_eq!(option.rho(), -46.9290244455, 1e-8);
    }

    #[test]
    fn put_call_parity() {
        let call = test_option(PutOrCall::Call, flat_5pct());
        let put = test_option(PutOrCall::Put, flat_5pct());
        let s = call.base.underlying_price.value();
        let k_df = call.base.strike_price * call.base.maturity_discount_factor();
        assert_approx_eq!(call.npv() - put.npv(), s - k_df, 1e-10);
    }

    #[test]
    fn zero_curve_prices_off_maturity_pillar() {
        // A non-flat zero curve whose 1y pillar is 5% must reproduce the
        // flat-5% price: discounting reads df at maturity, not any other node.
        let curve = YieldCurve::from_zero_rates(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
            &[0.02, 0.05, 0.07],
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let option = test_option(PutOrCall::Call, curve);
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.base.risk_free_rate(), 0.05, 1e-12);
    }
}
