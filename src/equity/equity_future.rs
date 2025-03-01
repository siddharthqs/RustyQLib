// An equity
use chrono::{Datelike, Local, NaiveDate};
use crate::core::quotes::Quote;
use crate::core::traits::Instrument;
use crate::core::utils::{Contract,ContractStyle};
//use crate::equity::vanila_option::EquityOption;

pub struct EquityFuture {
    pub underlying_price: Quote,
    pub current_price: Quote,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
}

impl EquityFuture {
    pub fn from_json(data: &Contract) -> Box<Self> {
        let market_data = data.market_data.as_ref().unwrap();
        //let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let today = Local::today();
        let maturity_date = NaiveDate::parse_from_str(&market_data.maturity, "%Y-%m-%d")
            .expect("Invalid maturity date");

        let underlying_quote = Quote::new(market_data.underlying_price);
        let quote = Some(market_data.current_price).unwrap();
        let current_quote = Quote::new(quote.unwrap_or(0.0));
        let risk_free_rate = Some(market_data.risk_free_rate).unwrap();
        let dividend = Some(market_data.dividend).unwrap();
        Box::new(Self {
            underlying_price: underlying_quote,
            current_price:current_quote,
            risk_free_rate: risk_free_rate.unwrap_or(0.0),
            dividend_yield: dividend.unwrap_or(0.0),
            maturity_date: maturity_date,
            valuation_date: today.naive_utc(),
        })
    }

    fn time_to_maturity(&self) -> f64 {
        let days = (self.maturity_date - self.valuation_date).num_days();
        (days as f64) / 365.0
    }
    fn premiun(&self)->f64{
        self.current_price.value()-self.underlying_price.value()
    }
}
impl Instrument for EquityFuture {
    fn npv(&self) -> f64 {
        // F_0 = S_0 * e^{(r - q)*t}
        let t = self.time_to_maturity();
        self.underlying_price.value() * ((self.risk_free_rate - self.dividend_yield) * t).exp()
    }
}
impl EquityFuture{
    pub fn delta(&self) -> f64 { 1.0 }
    pub fn gamma(&self) -> f64 { 0.0 }
    pub fn vega(&self) -> f64  { 0.0 }
    pub fn theta(&self) -> f64 { 0.0 }
    pub fn rho(&self) -> f64   { 0.0 }
}
// }