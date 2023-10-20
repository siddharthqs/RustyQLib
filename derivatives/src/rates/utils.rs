use chrono::{Local, NaiveDate, Weekday};
use chrono::Datelike;
use crate::rates::deposits::Deposit;
use crate::core::traits::Rates;
use serde::{Serialize,Deserialize};
fn is_weekend(date: NaiveDate) -> bool {
    // Check if the day of the week is Saturday (6) or Sunday (7)
    let day_of_week = date.weekday();
    day_of_week == Weekday::Sat || day_of_week == Weekday::Sun
}

fn is_holiday(date: NaiveDate) -> bool {
    // Check if the date is Christmas
    if date.month() == 12 && date.day() == 25{
        return true;
    }
    else if date.month() == 1 && date.day() == 1{ // Check if the date is New Year's Day
        return true;
    }
    else if date.month() == 1 && date.weekday() == Weekday::Mon
        && date.day() > 14 && date.day() <= 21 {
        // Check if the date is Martin Luther King Jr. Day
        return true;
    }
    else if date.month() == 11 &&
        date.weekday() == Weekday::Thu && date.day() > 21 && date.day() <= 28{
        // Check if the date is in November and falls on the fourth Thursday Thanksgiving Day
        return true;
    }
    else if date.month() == 9 && date.weekday() == Weekday::Mon && date.day() <= 7{
        // Check if the date is in September and falls on the first Monday Labor Day
        return true;
    }
    return false;
}

fn adjust_for_weekend(mut date: NaiveDate) -> NaiveDate {
    // Increment the date until it's not a weekend
    while is_holiday(date) || is_holiday(date) {
        date = date.succ();

    }
    date
}
#[derive(Clone,Debug)]
pub enum DayCountConvention{
    Act365,
    Act360,
    Thirty360,
}
impl DayCountConvention{
    pub fn num_of_days(&self) -> usize
    {
        match self {
            DayCountConvention::Act365 => 365,
            DayCountConvention::Act360 => 360,
            DayCountConvention::Thirty360 => 360,
        }
    }
    pub fn get_year_fraction(&self,start_date:NaiveDate,maturity_date:NaiveDate) -> f64 {
        let duration = maturity_date.signed_duration_since(start_date);
        let year_fraction = duration.num_days() as f64 / self.num_of_days() as f64;
        year_fraction
    }
}

#[derive(Clone,Debug)]
pub struct TermStructure {
    pub date: Vec<NaiveDate>,
    pub discount_factor: Vec<f64>,
    pub rate: Vec<f64>,
    pub day_count: DayCountConvention,
}

impl TermStructure {
    pub fn new(date: Vec<NaiveDate>, discount_factor: Vec<f64>,rate:Vec<f64>,day_count:DayCountConvention) -> TermStructure {
        TermStructure {
            date,
            discount_factor,
            rate,
            day_count
        }
    }

    pub fn interpolate_log_linear(&self,val_date:NaiveDate,maturity_date:NaiveDate)-> f64{
        let year_fraction = self.get_year_fraction(val_date);
        let target_yf = maturity_date.signed_duration_since(val_date).num_days() as f64
            / self.day_count.num_of_days() as f64;
        let mut df1 = 1.0;
        let mut df2 = 1.0;
        let mut t1 = 0.0;
        let mut t2 = 0.0;
        for (i, time) in year_fraction.iter().enumerate() {
            if time==&target_yf{
                 return self.discount_factor[i];
            }
            else if time< &target_yf {
                t1 = *time;
                df1 = self.discount_factor[i];
            }
            else if time> &target_yf {
                t2 = *time;
                df2 = self.discount_factor[i];
                break;
            }

        }
        let log_df1 = f64::ln(df1);
        let log_df2 = f64::ln(df2);
        let w = (target_yf - t1) / (t2 - t1);
        let log_df = log_df1 + w * (log_df2 - log_df1);
        let df = f64::exp(log_df);
        return df;
        //let dfs  = self.discount_factor;

    }
    pub fn get_year_fraction(&self,val_date:NaiveDate) -> Vec<f64> {
        let mut year_fraction_vec:Vec<f64> = Vec::new();
        for time in self.date.iter() {
            let duration = time.signed_duration_since(val_date);
            let year_fraction = duration.num_days() as f64 / self.day_count.num_of_days() as f64;
            year_fraction_vec.push(year_fraction);
        }
        year_fraction_vec
    }
    pub fn rates(&self,val_date:NaiveDate) -> Vec<f64> {
        let mut rates:Vec<f64> = Vec::new();
        for i in 0..self.discount_factor.len() {
            let rate = (1.0 / self.discount_factor[i] - 1.0) / self.day_count.get_year_fraction(val_date,self.date[i]);
            rates.push(rate);
        }
        return rates;
    }
    pub fn build_term_structure(&self,valuation_date:NaiveDate,deposits:Vec<Deposit>) -> TermStructure {
        let mut discount_factor:Vec<f64> = Vec::new();
        let mut rate:Vec<f64> = Vec::new();
        let mut dates:Vec<NaiveDate> = Vec::new();
        for deposit in deposits.iter() {
            discount_factor.push(deposit.get_discount_factor());
            dates.push(deposit.get_maturity_date());
            rate.push(deposit.get_rate());
        }
        let day_count = deposits[0].day_count.clone();
        let mut term_structure = TermStructure::new(dates,discount_factor,rate,day_count);
        return term_structure;
    }
}

pub fn convert_mm_to_date(mut date: String) -> NaiveDate {
    let current_date = Local::today();
    date.pop();
    let month = date.parse::<u32>().unwrap();

    let (new_year, new_month) = if current_date.month() + month > 12 {
        let year = ((current_date.month() + month) / 12) as i32;
        let m:u32 = (year * 12) as u32;
        let new_month = current_date.month() + month;
        (current_date.year() + year, new_month-m)
    } else {
        (current_date.year(), current_date.month() + month)
    };
    let date_in_months = current_date.with_year(new_year).unwrap_or(current_date)
        .with_month(new_month).unwrap_or(current_date);
    let mut maturity_date = date_in_months.naive_utc();
    maturity_date = adjust_for_weekend(maturity_date);
    return maturity_date;
}

