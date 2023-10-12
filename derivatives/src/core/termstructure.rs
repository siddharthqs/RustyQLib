use chrono::{DateTime, Local, NaiveDate};
pub struct YieldTermStructure<T> {
    pub date: Vec<T>,
    pub rates: Vec<f64>
}
impl<T> YieldTermStructure<T> {
    pub fn new(date: Vec<T>, rates: Vec<f64>) -> YieldTermStructure<T> {
        YieldTermStructure {
            date,
            rates
        }
    }
}

