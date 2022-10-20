use super::super::core::yc_term_structure::YieldTermStructure;
use super::super::core::quotes::Quote;

#[derive(PartialEq, Debug)]
pub enum OptionType {
    Call,
    Put,
    Straddle,
}
pub enum Transection {
    Buy,
    Sell,
}

pub struct EquityOption {
    pub option_type: OptionType,
    pub transection: Transection,
    pub current_price: Quote,
    pub strike_price: f64,
    pub dividend_yield: f64,
    pub volatility: f64,
    pub time_to_maturity: f64,
    pub term_structure: YieldTermStructure<f64>,
    pub risk_free_rate: f64,
    pub transection_price: f64,
}
