extern crate probability;
extern crate rand_chacha;
extern crate rand_pcg;

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

fn main() {

    // let input = args().collect::<Vec<String>>();
    // let filepath = &input[1];
    // read_csv::read_ts(filepath);
    // let mut rng = rand::thread_rng();
    // let r = Uniform::new(0.0,1.0).sample(&mut rng);
    // let val: f64 = rand::thread_rng().sample(StandardNormal);
    // let t = generate_standard_normal1();
    // println!("{}", t.0);
    // println!("{}",t.1);
    // prepare a deterministic generator:

    println!("Do you want to use the Binomial (1), Black-Sholes (2) or Monte-Carlo (3) option pricing model?");
    let mut model = String::new();
    print!("{}", model);
    io::stdin()
        .read_line(&mut model)
        .expect("Failed to read line");
    let model: u8 = model.trim().parse::<u8>().unwrap();
    if model == 1 {
        print!("{}", model);
        blackscholes::option_pricing();
    } else if model == 2 {
        print!("{}", model);
        montecarlo::option_pricing();
    } else if model == 3 {
        print!("{}", model);
        blackscholes::option_pricing();
    } else {
        println!("You gave a wrong number! The accepted arguments are 1, 2 and 3.")
    }




}
