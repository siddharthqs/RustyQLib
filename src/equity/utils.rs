use serde::Deserialize;
use crate::equity::vanila_option::EquityOptionBase;
use std::str::FromStr;
use std::error::Error;
use crate::core::trade::{PutOrCall};
use std::fmt::Debug;
use crate::core::utils::ContractStyle;

///Enum for different engines to price options
#[derive(PartialEq,Clone,Debug)]
pub enum Engine{
    BlackScholes,
    MonteCarlo,
    Binomial,
    FiniteDifference
}
#[derive(Debug)]
pub enum LongShort{
    LONG,
    SHORT
}
#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PayoffType {
    Vanilla,
    Binary,
    Barrier,
    Asian,
    // etc.
}
impl FromStr for PayoffType {
    type Err = Box<dyn Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vanilla" => Ok(PayoffType::Vanilla),
            "binary" => Ok(PayoffType::Binary),
            "barrier" => Ok(PayoffType::Barrier),
            "asian" => Ok(PayoffType::Asian),
            _ => Err("Invalid payoff type".into()),
        }
    }
}


pub trait Payoff: Debug {
    fn payoff_amount(&self, base: &EquityOptionBase) -> f64;
    fn payoff_kind(&self) -> PayoffType;
    fn put_or_call(&self) -> &PutOrCall;
    fn exercise_style(&self)->&ContractStyle;

    // possibly other methods for payoff logic
}