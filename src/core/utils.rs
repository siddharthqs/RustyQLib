//mod dis{
use libm::erf;
use std::f64::consts::{PI, SQRT_2};
use serde::{Deserialize, Serialize};
use crate::core::data_models::ProductData;

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
pub trait Engine<I> {
    fn npv(&self, instrument: &I) -> f64;
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



// #[derive(Clone,Debug,Deserialize,Serialize)]
// pub struct MarketData {
//     pub underlying_price:f64,
//     pub option_type:Option<String>,
//     pub strike_price:Option<f64>,
//     pub volatility:Option<f64>,
//     pub option_price:Option<f64>,
//     pub risk_free_rate:Option<f64>,
//     pub maturity:String,
//     pub dividend: Option<f64>,
//     pub simulation:Option<u64>,
//     pub current_price:Option<f64>,
//     pub notional: Option<f64>,
//     pub long_short:Option<i32>,
//     pub multiplier:Option<f64>,
//     pub entry_price:Option<f64>,
// }


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
    pub asset: String,
    pub product_type: ProductData,
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
    /// Change in delta per unit change in implied volatility.
    pub vanna: f64,
    /// Change in delta per year of calendar time.
    pub charm: f64,
    /// Delta elasticity, `S * gamma / delta`.
    pub gamma_p: f64,
    /// Change in gamma per unit change in implied volatility.
    pub zomma: f64,
    /// Monte Carlo standard error of `pv` (None for deterministic engines).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_err: Option<f64>,
    /// Per-asset deltas for multi-asset (rainbow) products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deltas: Option<Vec<f64>>,
    /// Per-asset vegas for multi-asset (rainbow) products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vegas: Option<Vec<f64>>,
    pub error: Option<String>
}

/// Probability density function of a standard normal random variable x.
pub fn dN(x: f64) -> f64 {
    let t = -0.5 * x * x;
    t.exp() / (SQRT_2 * PI.sqrt())
}

/// Cumulative distribution function of a standard normal random variable x.
pub fn N(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Inverse of the standard normal CDF (quantile function).
///
/// Acklam's rational approximation refined with one Halley step against the
/// erf-based [`N`], giving close to machine precision. `p` must be in (0, 1);
/// values outside return NaN.
pub fn inv_N(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    let tail = |q: f64| -> f64 {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let mut x = if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    };
    // one Halley refinement step
    let e = N(x) - p;
    let u = e * (2.0 * PI).sqrt() * (x * x / 2.0).exp();
    x -= u / (1.0 + x * u / 2.0);
    x
}

