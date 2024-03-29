//mod dis{
use libm::{exp, log};
use probability;
use probability::distribution::Distribution;
use std::f64::consts::{PI, SQRT_2};
use serde::Serialize;
use crate::Deserialize;

#[derive(PartialEq,Clone,Debug)]
pub enum ContractStyle {
    European,
    American,
}

#[derive(strum_macros::Display)]
pub enum EngineType {
    Analytical,
    MonteCarlo,
    Binomial,
    FiniteDifference,
    FFT,
}
impl EngineType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineType::Analytical => "Analytical",
            EngineType::MonteCarlo => "MonteCarlo",
            EngineType::Binomial => "Binomial",
            EngineType::FiniteDifference => "FiniteDifference",
            EngineType::FFT => "FFT",
        }
    }
}

#[derive(Clone,Debug,Deserialize,Serialize)]
pub struct MarketData {
    pub underlying_price:f64,
    pub option_type:String,
    pub strike_price:f64,
    pub volatility:Option<f64>,
    pub option_price:Option<f64>,
    pub risk_free_rate:Option<f64>,
    pub maturity:String,
    pub dividend: Option<f64>,
    pub simulation:Option<u64>,
}
#[derive(Clone,Debug,Deserialize,Serialize)]
pub struct RateData {
    pub instrument: String,
    pub currency: String,
    pub start_date: String,
    pub maturity_date: String,
    pub valuation_date: String,
    pub notional: f64,
    pub fix_rate: f64,
    pub day_count: String,
    pub business_day_adjustment: i8,
}

#[derive(Clone,Debug,Deserialize,Serialize)]
pub struct Contract {
    pub action: String,
    pub pricer: String,
    pub asset: String,
    pub style: Option<String>,
    pub market_data: Option<MarketData>,
    pub rate_data: Option<RateData>,
}
#[derive(Deserialize,Serialize)]
pub struct CombinedContract{
    pub contract: Contract,
    pub output: ContractOutput
}

#[derive(Debug, Deserialize,Serialize)]
pub struct Contracts {
    pub asset: String,
    pub contracts: Vec<Contract>,
}
#[derive(Debug, Deserialize,Serialize)]
pub struct OutputJson {
    pub contracts: Vec<String>,
}
#[derive(Deserialize,Serialize)]
pub struct ContractOutput {
    pub pv: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub error: Option<String>
}

pub fn dN(x: f64) -> f64 {
    /// Probability density function of standard normal random variable x.
    let t = -0.5 * x * x;
    return t.exp() / (SQRT_2 * PI.sqrt());
}

pub fn N(x: f64) -> f64 {
    ///Cumulative density function of standard normal random variable x.
    let m = probability::distribution::Gaussian::new(0.0, 1.0);
    let cdf = m.distribution(x);
    return cdf;
}

