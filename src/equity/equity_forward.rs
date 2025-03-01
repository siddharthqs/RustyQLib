use chrono::NaiveDate;
use crate::core::quotes::Quote;
use crate::equity::utils::LongShort;
pub struct EquityForward {
    pub underlying_price: Quote,
    pub forward_price: Quote,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub long_short:LongShort,
    pub notional:f64
}

