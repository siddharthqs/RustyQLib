use chrono::NaiveDate;
use crate::core::errors::RustyQLibError;
use crate::core::results::PricingResult;
use crate::rates::utils::DayCountConvention;
use crate::rates::utils::TermStructure;

pub trait Instrument {
    /// Present value, or a typed error when the instrument cannot be priced
    /// (invalid inputs, or an engine/product combination the library
    /// refuses to price).
    fn try_npv(&self) -> Result<f64, RustyQLibError>;

    /// Present value, panicking on any pricing error. Convenience for
    /// instruments already known to be valid; fallible callers (batch
    /// pricing, services) should use [`Instrument::try_npv`].
    fn npv(&self) -> f64 {
        match self.try_npv() {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        }
    }

    /// Price the instrument once, returning value, Greeks and (for Monte
    /// Carlo engines) the standard error together in a [`PricingResult`].
    ///
    /// The default implementation reports the present value with zero
    /// Greeks; instruments that compute sensitivities override it.
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        Ok(PricingResult::from_pv(self.try_npv()?))
    }
}


pub trait Rates {
    fn get_implied_rates(&self) -> f64;
    fn get_maturity_date(&self) -> NaiveDate;
    fn get_rate(&self) -> f64;
    fn get_maturity_discount_factor(&self) -> f64;
    fn get_day_count(&self) -> &DayCountConvention;
    fn set_term_structure(&mut self,term_structure:TermStructure)->();
}

pub trait Observer{
    fn update(&mut self);
    fn reset(&mut self);
}
pub trait Observable{
    fn update(&mut self);
    fn reset(&mut self);
}
