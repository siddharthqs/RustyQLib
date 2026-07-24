use crate::core::errors::RustyQLibError;
use chrono::{Local, NaiveDate};
use crate::core::data_models::EquityForwardData;
use crate::core::quotes::Quote;
use crate::core::traits::Instrument;
use crate::core::utils::Contract;
use crate::equity::equity_future::EquityFuture;
use crate::equity::utils::LongShort;
///A forward contract is an agreement between two parties to buy or sell, as the case may be,
/// a commodity (or financial instrument or currency or any other underlying)
/// on a pre-determined future date at a price agreed when the contract is entered into.
pub struct EquityForward {

    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    pub underlying_price: Quote,
    pub forward_price: Quote, //Forward price you actually locked in.
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    /// Continuous stock borrow (repo) cost; part of the carry.
    pub borrow_cost: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub long_short:LongShort,
    pub notional:f64
}
impl EquityForward  {
    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`EquityForward::try_from_json`].
    pub fn from_json(data: &EquityForwardData) -> Box<Self> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &EquityForwardData) -> Result<Box<Self>, RustyQLibError> {
        let today = Local::now().date_naive();
        let maturity_date = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d")
            .map_err(|_| RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            ))?;

        let underlying_price = Quote::new(data.base.underlying_price);
        let entry_quote = Quote::new(data.entry_price.unwrap_or(0.0));
        let risk_free_rate = data.base.risk_free_rate.unwrap_or(0.0);
        let dividend = data.dividend.unwrap_or(0.0);
        let long_short = data.base.long_short.unwrap_or(1);
        let position = match long_short{
            1=>LongShort::LONG,
            -1=>LongShort::SHORT,
            _=>LongShort::LONG,
        };
        Ok(Box::new(Self {
            symbol:data.base.symbol.clone(),
            currency: data.base.currency.clone(),
            exchange:data.base.exchange.clone(),
            name: data.base.name.clone(),
            cusip: data.base.cusip.clone(),
            isin: data.base.isin.clone(),
            settlement_type: data.base.settlement_type.clone(),


            underlying_price: underlying_price,
            forward_price:entry_quote,
            risk_free_rate: risk_free_rate,
            dividend_yield: dividend,
            borrow_cost: data.base.borrow_cost.unwrap_or(0.0),
            maturity_date: maturity_date,
            valuation_date: today,
            notional:data.notional.unwrap_or(1.0),
            long_short:position
        }))
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
        let dividend_df = 1.0/((self.dividend_yield + self.borrow_cost)*self.time_to_maturity()).exp();
        let forward = self.underlying_price.value()*dividend_df/discount_df;
        forward
    }
}
impl Instrument for EquityForward {
    fn try_npv(&self) -> Result<f64, crate::core::errors::RustyQLibError> {
        // e −r(T−t) (Ft −K),
        let df_r = 1.0/(self.risk_free_rate*self.time_to_maturity()).exp();
        let share = self.notional/self.forward_price.value();
        Ok(match self.long_short{
            LongShort::LONG => (self.forward()-self.forward_price.value()) * share *df_r,
            LongShort::SHORT => -(self.forward()-self.forward_price.value()) * share *df_r,
        })
    }
}

impl EquityForward{
    pub fn delta(&self) -> f64 { 1.0 }
    pub fn gamma(&self) -> f64 { 0.0 }
    pub fn vega(&self) -> f64  { 0.0 }
    pub fn theta(&self) -> f64 { 0.0 }
    pub fn rho(&self) -> f64   { 0.0 }
    pub fn vanna(&self) -> f64 { 0.0 }
    pub fn charm(&self) -> f64 { 0.0 }
    pub fn gamma_p(&self) -> f64 { 0.0 }
    pub fn zomma(&self) -> f64 { 0.0 }
}
