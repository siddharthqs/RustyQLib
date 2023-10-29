use chrono::{Local, NaiveDate};
use crate::core::traits::{Instrument,Rates};
use crate::rates::utils::{DayCountConvention,TermStructure};

/*
    A forward rate agreement or simply "forward contract" is an agreement to exchange a fixed pre-agreed rate for a
    floating rate is not known until some specified
    future fixing date. The FRA payment occurs on or soon after this date
    on the FRA settlement date. Typically the timing gap is two days.

 */

pub struct FRA {
    pub start_date: NaiveDate, //The date the FRA starts to accrue interest
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub notional: f64,
    pub currency: String,
    pub fix_rate: f64,
    pub day_count: DayCountConvention,
    pub business_day_adjustment: i8,
    pub term_structure: Option<TermStructure>,
}
impl FRA {
    pub fn new(start_date: NaiveDate, maturity_date: NaiveDate, valuation_date: NaiveDate,
               notional: f64, fix_rate: f64, day_count: DayCountConvention,
               business_day_adjustment: i8) -> FRA {
        FRA {
            start_date,
            maturity_date,
            valuation_date,
            notional,
            currency: String::from("USD"),
            fix_rate,
            day_count,
            business_day_adjustment,
            term_structure: None,
        }
    }
    pub fn builder(start_date: String,maturity_date:String,notional: f64, fix_rate: f64,day_count: String) ->FRA{

        let today = Local::today();
        let start_date = NaiveDate::parse_from_str(&start_date, "%Y-%m-%d").expect("Invalid date format");
        let maturity_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let mut fra = FRA {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: today.naive_utc(),
            notional: notional,
            currency: String::from("USD"),
            fix_rate: fix_rate,
            day_count: DayCountConvention::Act360,
            business_day_adjustment: 0,
            term_structure: None,
        };
        match day_count.as_str() {
            "Act360" |"A360" => {
                fra.day_count = DayCountConvention::Act360;
            }
            "Act365" |"A365" => {
                fra.day_count = DayCountConvention::Act365;
            }
            "Thirty360" |"30/360" => {
                fra.day_count = DayCountConvention::Thirty360;
            }
            _ => {}
        }
        fra
    }
    pub fn get_year_fraction(&self,date:NaiveDate) -> f64 {
        let duration = self.maturity_date.signed_duration_since(date);
        let year_fraction = duration.num_days() as f64 / self.day_count.num_of_days() as f64;
        year_fraction
    }
    pub fn get_discount_factor(&self,df_start_date:f64) -> f64 {
        let year_fraction = self.get_year_fraction(self.start_date);
        let discount_factor = df_start_date / (1.0 + self.fix_rate * year_fraction);
        discount_factor
    }
}

impl Rates for FRA{
    fn get_implied_rates(&self) -> f64 {
        let curve = self.term_structure.as_ref().expect("Term structure is not set");
        let df = curve.interpolate_log_linear(self.valuation_date,self.maturity_date);
        let implied_rate = (1.0/df - 1.0)/self.get_year_fraction(self.valuation_date);
        return implied_rate;
    }
    fn get_maturity_date(&self) -> NaiveDate {
        self.maturity_date
    }
    fn get_rate(&self) -> f64 {
        let df = self.get_maturity_discount_factor();
        let time = self.get_year_fraction(self.valuation_date);
        -df.ln() / time
    }
    fn get_maturity_discount_factor(&self) -> f64 {
        let curve = self.term_structure.as_ref().expect("Term structure is not set");
        let df = curve.interpolate_log_linear(self.valuation_date,self.start_date);
        self.get_discount_factor(df)
    }
    fn get_day_count(&self) -> &DayCountConvention {
        &self.day_count
    }
    fn set_term_structure(&mut self,term_structure:TermStructure) {
        self.term_structure = Some(term_structure);
    }
}
