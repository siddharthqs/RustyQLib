use std::collections::HashMap;
use crate::rates::utils::{DayCountConvention};
use chrono::{NaiveDate};
#[derive(Clone,Debug)]
pub struct VolSurface{
    pub term_structure: HashMap<NaiveDate, Vec<(f64,f64)>>,
    pub spot: f64,
    pub spot_date: NaiveDate,
    pub day_count: DayCountConvention,
}

impl VolSurface {
    pub fn new(term_structure: HashMap<NaiveDate, Vec<(f64,f64)>>,spot:f64,spot_date:NaiveDate,day_count:DayCountConvention) -> VolSurface {
        VolSurface {
            term_structure,
            spot,
            spot_date,
            day_count
        }
    }
    pub fn get_vol(&self,val_date:NaiveDate,maturity_date:NaiveDate,strike:f64)-> f64{
        0.0
    }
    pub fn get_year_fraction(&self,val_date:NaiveDate,maturity_date:NaiveDate) -> f64 {
        self.day_count.year_fraction(val_date,maturity_date)
    }

}