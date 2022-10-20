extern crate probability;
use chrono::{Local,DateTime,NaiveDate,NaiveTime};
mod equity;
mod core;
use std::{io, thread};
//use std::env::{args,Args};

pub trait Contract{
    fn npv(&self);
}
pub struct Swap;
impl Contract for Swap {
    fn npv(&self) {
        println!("This is npv of swap")
    }
}
pub struct Swaption;
impl Contract for Swaption {
    fn npv(&self) {
        println!("This is npv of Swaption")
    }
}
pub struct Trade{
    contract: Box<dyn Contract>
}
impl Trade {
    pub fn new() -> Self {
        Trade {
            contract: Box::new(Swap),
        }
    }
    pub fn npv(&self) {
        self.contract.npv();
    }
    pub fn set_contract(&mut self, contract: Box<dyn Contract>){
        self.contract = contract;
    }
}

fn main() {
    let mut trade = Trade::new();
    trade.npv();
    println!("Setting new contract");
    trade.set_contract(Box::new(Swaption));
    trade.npv();
    println!("----------");
}
