use serde::{Deserialize, Serialize};
use crate::core::curves::CurveInput;
use crate::core::vols::VolInput;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "product_type", rename_all = "snake_case")]
pub enum ProductData {
    Option(EquityOptionData),
    Future(EquityFutureData),
    Forward(EquityForwardData),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityInstrumentBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub underlying_price: f64,
    pub long_short: Option<i32>,
    pub risk_free_rate: Option<f64>,
    pub settlement_type: Option<String>,
}


#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityFutureData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub current_price: Option<f64>,
    pub multiplier:Option<f64>,
    pub entry_price:Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,

}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityForwardData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub current_price: Option<f64>,
    pub notional: Option<f64>,
    pub entry_price:Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,

}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityOptionData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub put_or_call: String, // "Call"/"Put"
    pub payoff_type: String, // Vanilla/Barrier/Binary
    pub strike_price: f64,
    /// Constant volatility; the simple alternative to `vol_surface`.
    pub volatility: Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,
    pub current_price: Option<f64>,
    pub multiplier:Option<f64>,
    pub entry_price:Option<f64>,
    /// Monte Carlo path count (engine "MC" only).
    pub simulation: Option<u64>,
    /// MC time steps: 1 = terminal simulation; > 1 = path-wise stepping.
    pub mc_time_steps: Option<usize>,
    /// "exact" (default) | "euler" | "milstein"
    pub mc_scheme: Option<String>,
    /// "sobol" (default, low-discrepancy) | "pseudo" (seeded PCG64)
    pub mc_sampler: Option<String>,
    pub mc_seed: Option<u64>,
    /// "gbm" (default, constant vol) | "local_vol" (Dupire from the
    /// option's vol surface)
    pub mc_model: Option<String>,
    pub exercise_style: Option<String>, //European, American,
    pub pricer:Option<String>,
    /// Optional discount curve; when absent a flat curve is built from
    /// `risk_free_rate` (which stays the simple way to specify a rate).
    pub discount_curve: Option<CurveInput>,
    /// Optional volatility surface; when absent a flat surface is built
    /// from `volatility`. One of the two must be provided.
    pub vol_surface: Option<VolInput>,
}

