use std::f64::consts::{PI, SQRT_2};
use probability;
use probability::distribution::Distribution;

fn dN(x:f64)->f64 {
    // Probability density function of standard normal random variable x.
    let t = (-0.5*x).powi(2);
    return t.exp() / (SQRT_2 * PI.sqrt());
}

fn N(x: f64) -> f64 {
    //umulative density function of standard normal random variable x.
    let m = probability::distribution::Gaussian::new(0.0,1.0);
    let cdf = m.distribution(x);
    return cdf
 }


pub enum OptionType{
    Call,
    Put
}
pub enum Transection{
    Buy,
    Sell
}

pub struct EquityOption {
    pub option_type: OptionType,
    pub transection : Transection,
    pub current_price: f64,
    pub strike_price: f64,
    pub dividend_yield: f64,
    pub volatility:f64,
    pub time_to_maturity:f64,
    pub risk_free_rate: f64,
    pub transection_price:f32,
}
impl EquityOption{
    pub fn calulate_price(&self){
        println!("This is price function");
    }
    pub fn d1(&self) -> f64 {
        //Black-Scholes-Merton d1 function Parameters
        let f1 = (self.current_price/self.strike_price).ln();
        let f2 = (self.risk_free_rate - self.dividend_yield+0.5*self.volatility.powi(2))*self.time_to_maturity;
        let f3 = self.volatility*(self.time_to_maturity.sqrt());
        return (f1+f2)/f3;

    }
}