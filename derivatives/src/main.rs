// extern crate probability;
// extern crate rand_chacha;
// extern crate rand_pcg;
#![allow(dead_code)]
#![allow(unused_variables)]
use rand;
use rand::{SeedableRng};
use chrono::{Local,DateTime,NaiveDate,NaiveTime};
use rand::distributions::{Standard,Uniform};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
mod equity;
mod core;
mod utils;
use rand::prelude::*;

use std::{io, thread};
use std::collections::HashMap;
use std::error::Error;
use csv;
//use std::env::{args,Args};
use utils::read_csv;
use utils::RNG;

use std::env::args;
use rand::Rng;
use equity::blackscholes;
use crate::equity::montecarlo;
use clap::{App,Arg};
fn main() {
    let _matches = App::new("qsLib").version("0.1.0").author("Siddharthqs.com")
        .about("Quant Library for retail traders using Rust").get_matches();
    println!("Welcome to Option pricing CLI");
    println!(" Do you want to price option (1) or calculate implied volatility (2)?");

    let mut input = String::new();
    print!("{}", input);
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    let input: u8 = input.trim().parse::<u8>().unwrap();
    if input == 1 {
        println!("Do you want to use the Black-Sholes (1) or Monte-Carlo (2) option pricing model?");
        let mut model = String::new();
        io::stdin()
            .read_line(&mut model)
            .expect("Failed to read line");
        let model: u8 = model.trim().parse::<u8>().unwrap();
        if model == 1 {
            blackscholes::option_pricing();
        }
        else if model == 2{
            montecarlo::option_pricing();
        }
        else {
            println!("You gave a wrong number! The accepted arguments are 1 and 2.")
        }
    } else if input == 2 {
        print!("Not Available yet");
    } else {
        println!("You gave a wrong number! The accepted arguments are 1 and 2.")
    }




}
