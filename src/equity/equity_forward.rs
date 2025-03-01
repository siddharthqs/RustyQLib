use chrono::{Local, NaiveDate};
use crate::core::quotes::Quote;
use crate::core::traits::Instrument;
use crate::core::utils::Contract;
use crate::equity::equity_future::EquityFuture;
use crate::equity::utils::LongShort;
///A forward contract is an agreement between two parties to buy or sell, as the case may be,
/// a commodity (or financial instrument or currency or any other underlying)
/// on a pre-determined future date at a price agreed when the contract is entered into.
pub struct EquityForward {
    pub underlying_price: Quote,
    pub forward_price: Quote, //Forward price you actually locked in.
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub long_short:LongShort,
    pub notional:f64
}
impl EquityForward  {
    pub fn from_json(data: &Contract) -> Box<Self> {
        let market_data = data.market_data.as_ref().unwrap();
        //let future_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let today = Local::today();
        let maturity_date = NaiveDate::parse_from_str(&market_data.maturity, "%Y-%m-%d")
            .expect("Invalid maturity date");

        let underlying_price = Quote::new(market_data.underlying_price);
        let quote = Some(market_data.current_price).unwrap();
        let current_quote = Quote::new(quote.unwrap_or(0.0));
        let risk_free_rate = Some(market_data.risk_free_rate).unwrap();
        let dividend = Some(market_data.dividend).unwrap();
        let long_short = market_data.long_short.unwrap();
        let position = match long_short{
            1=>LongShort::LONG,
            -1=>LongShort::SHORT,
            _=>LongShort::LONG,
        };
        Box::new(Self {
            underlying_price: underlying_price,
            forward_price:current_quote,
            risk_free_rate: risk_free_rate.unwrap_or(0.0),
            dividend_yield: dividend.unwrap_or(0.0),
            maturity_date: maturity_date,
            valuation_date: today.naive_utc(),
            notional:market_data.notional.unwrap_or(1.0),
            long_short:position
        })
    }

    fn time_to_maturity(&self) -> f64 {
        let days = (self.maturity_date - self.valuation_date).num_days();
        (days as f64) / 365.0
    }
    fn premiun(&self)->f64{
        self.forward()-self.underlying_price.value()
    }
    fn forward(&self)->f64{
        let discount_df = 1.0/(self.risk_free_rate*self.time_to_maturity()).exp();
        let dividend_df = 1.0/(self.dividend_yield*self.time_to_maturity()).exp();
        let forward = self.underlying_price.value()*dividend_df/discount_df;
        forward
    }
}
impl Instrument for EquityForward {
    fn npv(&self) -> f64 {
        // e −r(T−t) (Ft −K),
        let df_r = 1.0/(self.risk_free_rate*self.time_to_maturity()).exp();
        match self.long_short{
            LongShort::LONG => (self.forward()-self.forward_price.value()) * self.notional*df_r,
            LongShort::SHORT => -(self.forward()-self.forward_price.value()) * self.notional*df_r,
        }

    }
}

