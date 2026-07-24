use chrono::NaiveDate;
use crate::core::errors::RustyQLibError;
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
}

pub trait Greeks{
    fn delta(&self) -> f64;
    fn gamma(&self) -> f64;
    fn vega(&self) -> f64;
    fn theta(&self) -> f64;
    fn rho(&self) -> f64;
    /// Change in delta for a one-unit change in implied volatility,
    /// `d²V / (dS dσ)`.
    fn vanna(&self) -> f64;
    /// Change in delta as calendar time passes, `d²V / (dS dt)`.
    fn charm(&self) -> f64;
    /// Delta elasticity: the percentage change in delta for a percentage
    /// change in the underlying, `S * gamma / delta`.
    fn gamma_p(&self) -> f64;
    /// Zomma, the change in gamma per unit change in implied volatility.
    fn zomma(&self) -> f64;
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
