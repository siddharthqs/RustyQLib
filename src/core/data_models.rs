use serde::{Deserialize, Serialize};

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
    pub volatility: f64,
    pub maturity: String,
    pub dividend: Option<f64>,
    pub current_price: Option<f64>,
    pub multiplier:Option<f64>,
    pub entry_price:Option<f64>,
    pub simulation: Option<u64>,
    pub exercise_style: Option<String>, //European, American,
    pub pricer:Option<String>
}

