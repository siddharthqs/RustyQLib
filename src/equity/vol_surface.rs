use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use crate::rates::utils::{DayCountConvention};
use chrono::{NaiveDate};

/// Vol Surface is a collection of volatilities for different maturities and strikes
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct VolSurface{
    pub term_structure: BTreeMap<NaiveDate, Vec<(f64,f64)>>,
    pub spot: f64,
    pub spot_date: NaiveDate,
    pub day_count: DayCountConvention,
}

impl VolSurface {
    pub fn new(term_structure: BTreeMap<NaiveDate, Vec<(f64,f64)>>,spot:f64,spot_date:NaiveDate,day_count:DayCountConvention) -> VolSurface {
        VolSurface {
            term_structure,
            spot,
            spot_date,
            day_count
        }
    }
    pub fn get_vol(&self,val_date:NaiveDate,maturity_date:NaiveDate,strike:f64)-> f64{
        //TODO: Interpolate Vol Surface
        0.0
    }
    pub fn get_year_fraction(&self,val_date:NaiveDate,maturity_date:NaiveDate) -> f64 {
        self.day_count.get_year_fraction(val_date,maturity_date)
    }

}