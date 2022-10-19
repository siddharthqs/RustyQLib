use chrono::{DateTime,Local};
pub struct YieldTermStructure<T> {
    pub date: Vec<T>,
    pub rates: Vec<f64>
}