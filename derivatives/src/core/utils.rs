//mod dis{
use libm::{exp, log};
use probability;
use probability::distribution::Distribution;
use std::f64::consts::{PI, SQRT_2};
use serde::Serialize;
use crate::Deserialize;

#[derive(Deserialize,Serialize)]
pub struct MarketData {
    pub underlying_price:f64,
    pub option_type:String,
    pub strike_price:f64,
    pub volatility:f64,
    pub risk_free_rate:Option<f64>,
    pub maturity:String,
    pub dividend: Option<f64>,
    pub simulation:Option<u64>
}

#[derive(Deserialize,Serialize)]
pub struct Contract {
    pub action: String,
    pub pricer: String,
    pub asset: String,
    pub market_data: MarketData,
}
#[derive(Deserialize,Serialize)]
pub struct CombinedContract{
    pub contract: Contract,
    pub output: ContractOutput
}

#[derive(Deserialize,Serialize)]
pub struct ContractOutput {
    pub pv: f64,
}

pub fn dN(x: f64) -> f64 {
    // Probability density function of standard normal random variable x.
    let t = -0.5 * x * x;
    return t.exp() / (SQRT_2 * PI.sqrt());
}

pub fn N(x: f64) -> f64 {
    //umulative density function of standard normal random variable x.
    let m = probability::distribution::Gaussian::new(0.0, 1.0);
    let cdf = m.distribution(x);
    return cdf;
}
//}
