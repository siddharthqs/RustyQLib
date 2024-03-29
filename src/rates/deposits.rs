use chrono::{Local, NaiveDate};
use crate::core::traits::{Instrument,Rates};
use crate::rates::utils::{DayCountConvention,TermStructure};
/*
"" An deposit is an agreement to borrow money interbank at the Ibor fixing rate starting on the start
    date and repaid on the maturity date with the interest amount calculated according to a day
    count convention and dates calculated according to a calendar and business day adjustment rule.
 */
pub struct Deposit {
    pub start_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub notional: f64,
    pub fix_rate: f64,
    pub day_count: DayCountConvention,
    pub business_day_adjustment: i8,
    pub term_structure: Option<TermStructure>,
}
impl Deposit {
    pub fn new(start_date: NaiveDate, maturity_date: NaiveDate, valuation_date: NaiveDate,
               notional: f64, fix_rate: f64, day_count: DayCountConvention,
               business_day_adjustment: i8) -> Deposit {
        Deposit {
            start_date,
            maturity_date,
            valuation_date,
            notional,
            fix_rate,
            day_count,
            business_day_adjustment,
            term_structure: None,
        }
    }
    pub fn builder(start_date: String,maturity_date:String,notional: f64, fix_rate: f64,day_count: String) ->Deposit{

        let today = Local::today();
        let start_date = NaiveDate::parse_from_str(&start_date, "%Y-%m-%d").expect("Invalid date format");
        let maturity_date = NaiveDate::parse_from_str(&maturity_date, "%Y-%m-%d").expect("Invalid date format");
        let mut deposit = Deposit {
            start_date: start_date,
            maturity_date: maturity_date,
            valuation_date: today.naive_utc(),
            notional: 1000000.0,
            fix_rate: 0.05,
            day_count: DayCountConvention::Act360,
            business_day_adjustment: 0,
            term_structure: None,
        };
        match day_count.as_str() {
            "Act360" |"A360" => {
                deposit.day_count = DayCountConvention::Act360;
            }
            "Act365" |"A365" => {
                deposit.day_count = DayCountConvention::Act365;
            }
            "Thirty360" |"30/360" => {
                deposit.day_count = DayCountConvention::Thirty360;
            }
            _ => {}
        }
        return deposit;
    }
    pub fn get_start_date(&self) -> NaiveDate {
        self.start_date
    }

    pub fn get_notional(&self) -> f64 {
        self.notional
    }
    pub fn get_rate(&self) -> f64 {
        let df = self.get_discount_factor();
        let year_fraction = self.get_year_fraction(self.start_date);
        -df.ln() / year_fraction
    }
    pub fn get_business_day_adjustment(&self) -> i8 {
        self.business_day_adjustment
    }
    pub fn get_year_fraction(&self,date:NaiveDate) -> f64 {
        let duration = self.maturity_date.signed_duration_since(date);
        let year_fraction = duration.num_days() as f64 / self.day_count.num_of_days() as f64;
        year_fraction
    }
    pub fn get_discount_factor(&self) -> f64 {
        let year_fraction = self.get_year_fraction(self.start_date);
        let discount_factor = 1.0 / (1.0 + self.fix_rate * year_fraction);
        discount_factor
    }
    pub fn get_remaining_interest_amount(&self) -> f64 {
        let year_fraction = self.get_year_fraction(self.valuation_date);
        let interest_amount = self.notional * self.fix_rate * year_fraction;
        interest_amount
    }
    pub fn get_value(&self) -> f64 {
        //let discount_factor = self.get_discount_factor();
        //
        //let pv = self.notional * discount_factor + interest_amount;
        let value = (1.0 + self.fix_rate * self.get_year_fraction(self.start_date)) * self.notional;
        value
    }

    pub fn get_pv(&self,curve:&TermStructure) -> f64 {
        let df = curve.interpolate_log_linear(self.valuation_date,self.maturity_date);
        let value = self.get_value() * df;
        return value;
    }
}
impl Rates for Deposit{
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
        self.fix_rate
    }
    fn get_maturity_discount_factor(&self) -> f64 {
        self.get_discount_factor()
    }
    fn get_day_count(&self) -> &DayCountConvention {
        &self.day_count
    }
    fn set_term_structure(&mut self,term_structure:TermStructure) {
        self.term_structure = Some(term_structure);
    }
}
// impl Instrument for Deposit {
//     fn npv(&self) -> f64 {
//         self.get_pv()
//     }
// }
