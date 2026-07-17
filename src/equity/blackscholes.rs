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
use crate::core::vols::VolSurface;
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
            PayoffType::Binary => self.npv_binary(bsd_option),
            _ => {0.0}
        }
    }
    pub fn delta(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(bsd_option.time_to_maturity() >= 0.0, "Option is expired or negative time");
        assert!(bsd_option.base.underlying_price.value >= 0.0, "Negative underlying price not allowed");
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.delta_vanilla(bsd_option),
            PayoffType::Binary => self.delta_binary(bsd_option),
            _ => {0.0}
        }
    }
    pub fn gamma(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.gamma_vanilla(bsd_option),
            PayoffType::Binary => self.gamma_binary(bsd_option),
            _ => {0.0}
        }
    }
    pub fn vega(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.vega_vanilla(bsd_option),
            PayoffType::Binary => self.vega_binary(bsd_option),
            _ => {0.0}
        }
    }
    pub fn theta(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.theta_vanilla(bsd_option),
            PayoffType::Binary => self.theta_binary(bsd_option),
            _ => {0.0}
        }
    }
    pub fn rho(&self, bsd_option: &EquityOption) -> f64 {
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.rho_vanilla(bsd_option),
            PayoffType::Binary => self.rho_binary(bsd_option),
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
        let var_sqrt = bsd_option.base.volatility() * (bsd_option.time_to_maturity().sqrt());
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
        let t1 = -df_S * dn_d1 * bsd_option.base.volatility()
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

    // ── Cash-or-nothing binary (unit cash) ─────────────────────────────
    // call: e^{-rT} N(d2); put: e^{-rT} N(-d2)

    fn npv_binary(&self, bsd_option: &EquityOption) -> f64 {
        let df_r = bsd_option.base.maturity_discount_factor();
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => df_r * N(bsd_option.base.d2()),
            PutOrCall::Put => df_r * N(-bsd_option.base.d2()),
        }
    }
    fn delta_binary(&self, bsd_option: &EquityOption) -> f64 {
        // +- e^{-rT} dN(d2) / (S sigma sqrt(T))
        let df_r = bsd_option.base.maturity_discount_factor();
        let s = bsd_option.base.underlying_price.value();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let delta_call = df_r * dN(bsd_option.base.d2()) / (s * sigma * t.sqrt());
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => delta_call,
            PutOrCall::Put => -delta_call,
        }
    }
    fn gamma_binary(&self, bsd_option: &EquityOption) -> f64 {
        // -+ e^{-rT} dN(d2) d1 / (S^2 sigma^2 T)
        let df_r = bsd_option.base.maturity_discount_factor();
        let s = bsd_option.base.underlying_price.value();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let gamma_call =
            -df_r * dN(bsd_option.base.d2()) * bsd_option.base.d1() / (s * s * sigma * sigma * t);
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => gamma_call,
            PutOrCall::Put => -gamma_call,
        }
    }
    fn vega_binary(&self, bsd_option: &EquityOption) -> f64 {
        // -+ e^{-rT} dN(d2) d1 / sigma
        let df_r = bsd_option.base.maturity_discount_factor();
        let sigma = bsd_option.base.volatility();
        let vega_call = -df_r * dN(bsd_option.base.d2()) * bsd_option.base.d1() / sigma;
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => vega_call,
            PutOrCall::Put => -vega_call,
        }
    }
    fn theta_binary(&self, bsd_option: &EquityOption) -> f64 {
        // theta = -dV/dT with dd2/dT = (r - q - sigma^2/2)/(sigma sqrt(T)) - d2/(2T)
        let df_r = bsd_option.base.maturity_discount_factor();
        let r = bsd_option.base.risk_free_rate();
        let q = bsd_option.base.dividend_yield;
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let d2 = bsd_option.base.d2();
        let dd2_dt = (r - q - 0.5 * sigma * sigma) / (sigma * t.sqrt()) - d2 / (2.0 * t);
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => r * df_r * N(d2) - df_r * dN(d2) * dd2_dt,
            PutOrCall::Put => r * df_r * N(-d2) + df_r * dN(d2) * dd2_dt,
        }
    }
    fn rho_binary(&self, bsd_option: &EquityOption) -> f64 {
        // call: -T e^{-rT} N(d2) + e^{-rT} dN(d2) sqrt(T)/sigma
        let df_r = bsd_option.base.maturity_discount_factor();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let d2 = bsd_option.base.d2();
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => -t * df_r * N(d2) + df_r * dN(d2) * t.sqrt() / sigma,
            PutOrCall::Put => -t * df_r * N(-d2) - df_r * dN(d2) * t.sqrt() / sigma,
        }
    }

}


/// Black-Scholes price of a European vanilla as a pure function of its
/// inputs (no option object needed).
pub fn bs_price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, put_or_call: PutOrCall) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return match put_or_call {
            PutOrCall::Call => (s * exp(-q * t) - k * exp(-r * t)).max(0.0),
            PutOrCall::Put => (k * exp(-r * t) - s * exp(-q * t)).max(0.0),
        };
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    match put_or_call {
        PutOrCall::Call => s * exp(-q * t) * N(d1) - k * exp(-r * t) * N(d2),
        PutOrCall::Put => k * exp(-r * t) * N(-d2) - s * exp(-q * t) * N(-d1),
    }
}

/// Black-Scholes vega as a pure function (per unit of vol).
pub fn bs_vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    s * exp(-q * t) * dN(d1) * sqrt_t
}

const IMPLIED_VOL_MIN: f64 = 1e-4;
const IMPLIED_VOL_MAX: f64 = 5.0;

/// Implied Black-Scholes volatility for a European vanilla price.
///
/// Safeguarded Newton: full Newton steps while they stay inside the current
/// bisection bracket `[1e-4, 5.0]`, bisection otherwise, so it converges for
/// deep in/out-of-the-money quotes where raw Newton diverges. Prices outside
/// the arbitrage bounds return an error.
pub fn implied_vol_from_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    target: f64,
    put_or_call: PutOrCall,
) -> Result<f64, String> {
    if t <= 0.0 {
        return Err("option is expired".to_string());
    }
    let lower_bound = bs_price(s, k, r, q, 0.0, t, put_or_call);
    let upper_bound = match put_or_call {
        PutOrCall::Call => s * exp(-q * t),
        PutOrCall::Put => k * exp(-r * t),
    };
    if target < lower_bound - 1e-12 || target > upper_bound + 1e-12 {
        return Err(format!(
            "price {target} violates arbitrage bounds [{lower_bound}, {upper_bound}]"
        ));
    }

    let (mut lo, mut hi) = (IMPLIED_VOL_MIN, IMPLIED_VOL_MAX);
    if bs_price(s, k, r, q, lo, t, put_or_call) > target {
        return Ok(lo); // at or below the vol floor
    }
    if bs_price(s, k, r, q, hi, t, put_or_call) < target {
        return Err(format!("implied vol above {IMPLIED_VOL_MAX}"));
    }

    let mut sigma = 0.5_f64.min(hi).max(lo);
    let tol = 1e-12 * target.max(1.0);
    for _ in 0..100 {
        let diff = bs_price(s, k, r, q, sigma, t, put_or_call) - target;
        if diff.abs() < tol {
            return Ok(sigma);
        }
        if diff > 0.0 {
            hi = sigma;
        } else {
            lo = sigma;
        }
        let vega = bs_vega(s, k, r, q, sigma, t);
        let newton = sigma - diff / vega;
        sigma = if vega > 1e-12 && newton > lo && newton < hi {
            newton
        } else {
            0.5 * (lo + hi)
        };
        if hi - lo < 1e-14 {
            return Ok(sigma);
        }
    }
    Ok(sigma)
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
    let vol_surface = VolSurface::flat(
        vol.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
    )
    .expect("Invalid volatility");
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
        vol_surface,
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
        mc: crate::equity::montecarlo::MonteCarloConfig::default()
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
    fn test_option_with(payoff: Box<dyn Payoff>, curve: YieldCurve) -> EquityOption {
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
            vol_surface: VolSurface::flat(0.3, valuation_date, DayCountConvention::Act365)
                .unwrap(),
            maturity_date: NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(),
            valuation_date,
            discount_curve: curve,
            entry_price: 0.0,
            long_short: LongShort::LONG,
            multiplier: 1.0,
        };
        EquityOption {
            base,
            payoff,
            engine: Engine::BlackScholes,
            mc: crate::equity::montecarlo::MonteCarloConfig::default(),
        }
    }

    fn test_option(put_or_call: PutOrCall, curve: YieldCurve) -> EquityOption {
        test_option_with(
            Box::new(VanillaPayoff { put_or_call, exercise_style: ContractStyle::European }),
            curve,
        )
    }

    fn binary_option(put_or_call: PutOrCall) -> EquityOption {
        test_option_with(
            Box::new(super::super::vanila_option::BinaryPayoff {
                put_or_call,
                exercise_style: ContractStyle::European,
            }),
            flat_5pct(),
        )
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

    // Binary golden values computed independently and bump-verified
    #[test]
    fn golden_binary_call_npv_and_greeks() {
        let option = binary_option(PutOrCall::Call);
        assert_approx_eq!(option.npv(), 0.4819391800, 1e-8);
        assert_approx_eq!(option.delta(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.gamma(), -0.0001335042, 1e-8);
        assert_approx_eq!(option.vega(), -0.4005125405, 1e-8);
        assert_approx_eq!(option.theta(), 0.0209350179, 1e-8);
        assert_approx_eq!(option.rho(), 0.7828372637, 1e-8);
    }

    #[test]
    fn golden_binary_put_npv_and_greeks() {
        let option = binary_option(PutOrCall::Put);
        assert_approx_eq!(option.npv(), 0.4692902445, 1e-8);
        assert_approx_eq!(option.delta(), -0.0126477644, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0001335042, 1e-8);
        assert_approx_eq!(option.vega(), 0.4005125405, 1e-8);
        assert_approx_eq!(option.theta(), 0.0266264533, 1e-8);
        assert_approx_eq!(option.rho(), -1.7340666882, 1e-8);
    }

    #[test]
    fn binary_call_plus_put_equals_discount_factor() {
        let call = binary_option(PutOrCall::Call);
        let put = binary_option(PutOrCall::Put);
        assert_approx_eq!(call.npv() + put.npv(), call.base.maturity_discount_factor(), 1e-12);
    }

    // ── Cross-engine agreement ──────────────────────────────────────────

    #[test]
    fn finite_difference_matches_analytic_vanilla() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut option = test_option(pc, flat_5pct());
            let analytic = option.npv();
            option.engine = Engine::FiniteDifference;
            let fd = option.npv();
            assert!(
                (fd - analytic).abs() < 0.01,
                "{pc:?}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn finite_difference_matches_analytic_binary() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut option = binary_option(pc);
            let analytic = option.npv();
            option.engine = Engine::FiniteDifference;
            let fd = option.npv();
            assert!(
                (fd - analytic).abs() < 0.002,
                "{pc:?}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn binomial_matches_analytic_vanilla_and_binary() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut vanilla = test_option(pc, flat_5pct());
            let analytic = vanilla.npv();
            vanilla.engine = Engine::Binomial;
            let tree = vanilla.npv();
            assert!((tree - analytic).abs() < 0.02, "vanilla {pc:?}: tree={tree} bs={analytic}");

            let mut binary = binary_option(pc);
            let analytic = binary.npv();
            binary.engine = Engine::Binomial;
            let tree = binary.npv();
            assert!((tree - analytic).abs() < 0.02, "binary {pc:?}: tree={tree} bs={analytic}");
        }
    }

    #[test]
    fn monte_carlo_matches_analytic_vanilla_and_binary() {
        // default config: Sobol low-discrepancy terminal simulation
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut vanilla = test_option(pc, flat_5pct());
            let analytic = vanilla.npv();
            vanilla.engine = Engine::MonteCarlo;
            let mc = vanilla.npv();
            assert!((mc - analytic).abs() < 0.02, "vanilla {pc:?}: mc={mc} bs={analytic}");

            let mut binary = binary_option(pc);
            let analytic = binary.npv();
            binary.engine = Engine::MonteCarlo;
            let mc = binary.npv();
            assert!((mc - analytic).abs() < 0.005, "binary {pc:?}: mc={mc} bs={analytic}");
        }
    }

    #[test]
    fn monte_carlo_sobol_beats_default_tolerance_and_is_reproducible() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        let first = option.npv();
        let second = option.npv();
        assert_eq!(first, second, "deterministic sampler must reproduce exactly");
        assert!((first - 14.2312547860).abs() < 0.02, "sobol mc = {first}");
    }

    #[test]
    fn monte_carlo_path_wise_starts_at_spot_and_schemes_converge() {
        // regression for the path-wise bug (paths used to start at the
        // option premium instead of the underlying spot)
        let analytic = test_option(PutOrCall::Call, flat_5pct()).npv();
        for scheme in ["exact", "euler", "milstein"] {
            let mut option = test_option(PutOrCall::Call, flat_5pct());
            option.engine = Engine::MonteCarlo;
            option.mc.scheme = scheme.parse().unwrap();
            option.mc.time_steps = 252;
            option.mc.paths = 50_000;
            let mc = option.npv();
            assert!(
                (mc - analytic).abs() < 0.35,
                "{scheme}: mc={mc} analytic={analytic}"
            );
        }
    }

    #[test]
    fn monte_carlo_greeks_match_analytic() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        // common-random-number bumps against the analytic golden values
        assert!((option.delta() - 0.6242517279).abs() < 0.01, "delta {}", option.delta());
        assert!((option.gamma() - 0.0126477644).abs() < 0.003, "gamma {}", option.gamma());
        assert!((option.vega() - 37.9432933117).abs() < 1.0, "vega {}", option.vega());
        assert!((option.theta() - -8.1011898970).abs() < 0.5, "theta {}", option.theta());
        assert!((option.rho() - 48.1939180046).abs() < 0.5, "rho {}", option.rho());
    }

    #[test]
    fn lsmc_american_put_close_to_tree_and_dominates_european() {
        let european = test_option(PutOrCall::Put, flat_5pct()).npv();
        let mut tree_option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        tree_option.engine = Engine::Binomial;
        let tree = tree_option.npv();

        let mut lsmc_option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        lsmc_option.engine = Engine::MonteCarlo;
        lsmc_option.mc.paths = 20_000;
        let lsmc = lsmc_option.npv();

        // LSMC is biased slightly low (suboptimal exercise policy) but must
        // sit between the European price and just above the tree price
        assert!(lsmc > european, "lsmc {lsmc} must exceed european {european}");
        assert!((lsmc - tree).abs() < 0.25, "lsmc={lsmc} tree={tree}");
    }

    // ── Implied vol solver ──────────────────────────────────────────────

    #[test]
    fn implied_vol_round_trips_across_strikes_and_vols() {
        let (s, r, q) = (100.0, 0.05, 0.02);
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            for k in [50.0, 80.0, 100.0, 120.0, 200.0] {
                for vol in [0.05, 0.2, 0.6, 1.5] {
                    for t in [0.05, 0.5, 2.0] {
                        let price = bs_price(s, k, r, q, vol, t, pc);
                        // skip quotes indistinguishable from intrinsic
                        if price - bs_price(s, k, r, q, 0.0, t, pc) < 1e-10 {
                            continue;
                        }
                        let iv = implied_vol_from_price(s, k, r, q, t, price, pc).unwrap();
                        // deep in-the-money short-dated quotes have vega ~1e-7,
                        // so a double-precision price only pins the vol to
                        // ~1e-6 — 1e-5 is the attainable accuracy everywhere
                        assert!(
                            (iv - vol).abs() < 1e-5,
                            "{pc:?} K={k} vol={vol} t={t}: recovered {iv}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn implied_vol_rejects_arbitrage_violating_prices() {
        // below intrinsic
        assert!(implied_vol_from_price(100.0, 80.0, 0.05, 0.0, 1.0, 10.0, PutOrCall::Call)
            .is_err());
        // above the underlying
        assert!(implied_vol_from_price(100.0, 100.0, 0.05, 0.0, 1.0, 101.0, PutOrCall::Call)
            .is_err());
    }

    // ── Implied surface construction + Dupire local vol round trip ──────

    /// Quotes generated from a known smile: sigma(K, T) = base(T) - 0.001*(K-100)
    fn smile_vol(k: f64, base: f64) -> f64 {
        base - 0.001 * (k - 100.0)
    }

    fn quoted_option(
        k: f64,
        maturity: NaiveDate,
        market_price: f64,
    ) -> Box<EquityOption> {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.strike_price = k;
        option.base.maturity_date = maturity;
        option.base.current_price = Quote::new(market_price);
        Box::new(option)
    }

    fn build_surface_from_quotes() -> crate::core::vols::VolSurface {
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let maturities = [
            (NaiveDate::from_ymd_opt(2026, 7, 2).unwrap(), 0.23),
            (NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(), 0.25),
        ];
        let mut quotes = Vec::new();
        for (maturity, base) in maturities {
            let t = (maturity - valuation).num_days() as f64 / 365.0;
            for i in 0..13 {
                let k = 70.0 + 5.0 * i as f64;
                let vol = smile_vol(k, base);
                let price = bs_price(100.0, k, 0.05, 0.0, vol, t, PutOrCall::Call);
                quotes.push(quoted_option(k, maturity, price));
            }
        }
        crate::equity::vol_surface::build_implied_vol_surface(&quotes).unwrap()
    }

    #[test]
    fn implied_surface_recovers_input_vols() {
        let surface = build_surface_from_quotes();
        // exact at the quoted pillars (forward is irrelevant on a strike axis)
        for (t, base) in [(182.0 / 365.0, 0.23), (1.0, 0.25)] {
            for k in [70.0, 85.0, 100.0, 115.0, 130.0] {
                let vol = surface.vol(k, 100.0, t);
                assert!(
                    (vol - smile_vol(k, base)).abs() < 1e-7,
                    "K={k} t={t}: {vol} vs {}",
                    smile_vol(k, base)
                );
            }
        }
    }

    fn local_vol_option(surface: crate::core::vols::VolSurface, k: f64) -> EquityOption {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.strike_price = k;
        option.base.vol_surface = surface;
        option.engine = Engine::MonteCarlo;
        option.mc.model = crate::equity::montecarlo::McModel::LocalVol;
        option.mc.paths = 20_000;
        option
    }

    #[test]
    fn local_vol_prices_back_vanilla_from_calibrated_surface() {
        // implied quotes -> implied surface -> Dupire local vol -> MC price
        // must reproduce the original Black-Scholes prices
        let surface = build_surface_from_quotes();
        for k in [90.0, 100.0, 110.0] {
            let expected = bs_price(100.0, k, 0.05, 0.0, smile_vol(k, 0.25), 1.0, PutOrCall::Call);
            let lv_price = local_vol_option(surface.clone(), k).npv();
            assert!(
                (lv_price - expected).abs() < 0.3,
                "K={k}: local vol {lv_price} vs BS {expected}"
            );
        }
    }

    #[test]
    fn local_vol_flat_surface_reproduces_black_scholes() {
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface =
            crate::core::vols::VolSurface::flat(0.3, valuation, DayCountConvention::Act365)
                .unwrap();
        let expected = 14.2312547860; // flat-30% golden
        let lv_price = local_vol_option(surface, 100.0).npv();
        assert!((lv_price - expected).abs() < 0.3, "{lv_price} vs {expected}");
    }

    #[test]
    fn local_vol_term_structure_reproduces_terminal_implied() {
        // 20% to 6M, 25% to 1Y: pricing a 1Y option through the local vol
        // (which steps at ~20% then at the ~29.2% forward vol) must recover
        // the 25% terminal implied price
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface = crate::core::vols::VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
            valuation,
            DayCountConvention::Act365,
        )
        .unwrap();
        let expected = bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 1.0, PutOrCall::Call);
        let lv_price = local_vol_option(surface, 100.0).npv();
        assert!((lv_price - expected).abs() < 0.3, "{lv_price} vs {expected}");
    }

    #[test]
    #[should_panic(expected = "Analytical engine cannot price American")]
    fn analytic_engine_rejects_american_exercise() {
        let option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        option.npv();
    }

    #[test]
    fn american_put_fd_and_tree_agree_and_dominate_european() {
        let european_put = test_option(PutOrCall::Put, flat_5pct()).npv();
        let american = |engine: Engine| {
            let mut option = test_option_with(
                Box::new(VanillaPayoff {
                    put_or_call: PutOrCall::Put,
                    exercise_style: ContractStyle::American,
                }),
                flat_5pct(),
            );
            option.engine = engine;
            option.npv()
        };
        let fd = american(Engine::FiniteDifference);
        let tree = american(Engine::Binomial);
        assert!(fd > european_put, "american {fd} must exceed european {european_put}");
        assert!(tree > european_put);
        assert!((fd - tree).abs() < 0.02, "fd={fd} tree={tree}");
    }

    #[test]
    fn smile_surface_prices_with_interpolated_vol() {
        // K=100 sits midway between the 90 and 110 pillars at the 1y expiry,
        // so the option must price at the interpolated 30% vol — i.e. match
        // the flat-30% golden values exactly.
        let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface = crate::core::vols::VolSurface::from_strike_grid(
            &[Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
            &[90.0, 100.0, 110.0],
            &[vec![0.32, 0.30, 0.28], vec![0.36, 0.34, 0.32]],
            valuation_date,
            DayCountConvention::Act365,
        )
        .unwrap();
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.vol_surface = surface;
        assert_approx_eq!(option.base.volatility(), 0.30, 1e-14);
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        // a lower strike picks up the skew: vol(95) = 0.31
        option.base.strike_price = 95.0;
        assert_approx_eq!(option.base.volatility(), 0.31, 1e-14);
    }

    #[test]
    fn implied_vol_recovers_input_vol() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        let target_price = option.npv(); // priced at 30% flat
        // start the solve from a different vol level
        option.base.vol_surface = crate::core::vols::VolSurface::flat(
            0.6,
            option.base.valuation_date,
            DayCountConvention::Act365,
        )
        .unwrap();
        let iv = option.imp_vol(target_price);
        assert_approx_eq!(iv, 0.30, 1e-10);
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
