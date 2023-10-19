use chrono::NaiveDate;
use crate::rates::utils::DayCountConvention;
use crate::rates::utils::TermStructure;
pub trait Instrument{
    fn npv(&self)-> f64;
}

pub trait Greeks{
    fn delta(&self) -> f64;
    fn gamma(&self) -> f64;
    fn vega(&self) -> f64;
    fn theta(&self) -> f64;
    fn rho(&self) -> f64;
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