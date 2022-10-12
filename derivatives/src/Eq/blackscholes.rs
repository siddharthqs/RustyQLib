use std::f64::consts::{PI, SQRT_2};
use libm::{log, exp};
use std::{thread,io};
//use utils::{N,dN};
//use vanila_option::{EquityOption,OptionType};
use super::utils::{N, dN};
use super::vanila_option::{OptionType, EquityOption,Transection};


//mod dis;
impl EquityOption{
    pub fn pv(&self)->f64 {
        assert!(self.volatility>=0.0);
        assert!(self.time_to_maturity>=0.0);
        assert!(self.current_price>=0.0);
        if self.option_type == OptionType::Call {
            let option_price = self.current_price * N(self.d1()) * exp(-self.dividend_yield * self.time_to_maturity)
                - self.strike_price * exp(-self.risk_free_rate * self.time_to_maturity) * N(self.d2());
            return option_price
        }
        else {
            let option_price = -self.current_price * N(-self.d1()) * exp(-self.dividend_yield * self.time_to_maturity)
                + self.strike_price * exp(-self.risk_free_rate * self.time_to_maturity) * N(-self.d2());
            return option_price
        }

    }
    pub fn get_premium_at_risk(&self)->f64{
        let value = self.pv();
        let mut pay_off = 0.0;
        if self.option_type==OptionType::Call {
            pay_off = self.current_price - self.strike_price;
        }
        else if self.option_type==OptionType::Put {
             pay_off = self.strike_price -self.current_price;
        }
        if pay_off>0.0{
            return value -pay_off
        }
        else{
            return value
        }
    }
    pub fn delta(&self) -> f64 {
        let mut delta = N(self.d1());
        if self.option_type==OptionType::Call {
            delta = delta*exp(-self.dividend_yield * self.time_to_maturity);
        }
        else if self.option_type==OptionType::Put {
             delta = delta -1.0;
        }
        return delta;
    }
    pub fn gamma(&self) -> f64 {
        let mut gamma = dN(self.d1());
        //(St * sigma * math.sqrt(T - t))
        let var_sqrt = self.volatility*(self.time_to_maturity.sqrt());
        gamma = gamma/(self.current_price*var_sqrt);
        return gamma;
    }
    //ToDO
    //theta
    //rho
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let tmp1 = (self.current_price/self.strike_price).ln()+
            (self.risk_free_rate - self.dividend_yield+0.5*self.volatility.powi(2))*
                self.time_to_maturity;

        let tmp2 = self.volatility*(self.time_to_maturity.sqrt());
        return tmp1/tmp2;
    }
    pub fn d2(&self)->f64{
        let d2 = self.d1() - self.volatility* self.time_to_maturity.powf(0.5);
        return d2;
    }

}
pub fn option_pricing() {

    println!("Welcome to the Black-Scholes Option pricer.");
    println!("(Step 1/7) What is the current price of the underlying asset?");
    let mut curr_price = String::new();
    io::stdin().read_line(&mut curr_price).expect("Failed to read line");

    println!("(Step 2/7) Do you want a call option ('C') or a put option ('P') ?");
    let mut side_input = String::new();
    io::stdin().read_line(&mut side_input).expect("Failed to read line");

    let side: OptionType;
    match side_input.trim() {
        "C" | "c" | "Call" | "call" => side = OptionType::Call,
        "P" | "p" | "Put" | "put" => side = OptionType::Put,
        _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
    }

    println!("(Step 3/7) What's the stike price you want?");
    let mut strike = String::new();
    io::stdin().read_line(&mut strike).expect("Failed to read line");

    println!("(Step 4/7) What's the expected annualized volatility in %?");
    println!("E.g.: Enter 50% chance as 0.50 ");
    let mut vol = String::new();
    io::stdin().read_line(&mut vol).expect("Failed to read line");

    println!("(Step 5/7) What's the risk-free rate in %?");
    let mut rf = String::new();
    io::stdin().read_line(&mut rf).expect("Failed to read line");

    println!("(Step 6/7) In how many years do you want the expiry to be?");
    let mut expiry = String::new();
    io::stdin().read_line(&mut expiry).expect("Failed to read line");

    println!("(Step 7/7) What is the dividend yield on this stock?");
    let mut div = String::new();
    io::stdin().read_line(&mut div).expect("Failed to read line");

    let option = EquityOption {
        option_type: side,
        transection: Transection::Buy,
        current_price: curr_price.trim().parse::<f64>().unwrap(),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        volatility: vol.trim().parse::<f64>().unwrap(),
        time_to_maturity: expiry.trim().parse::<f64>().unwrap(),
        risk_free_rate: rf.trim().parse::<f64>().unwrap(),
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        transection_price: 0.0
    };
    println!("This option is currently worth ${}!", option.pv());

    let mut div1 = String::new();
    io::stdin().read_line(&mut div).expect("Failed to read line");
}
