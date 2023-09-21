// extern crate probability;
// extern crate rand_chacha;
// extern crate rand_pcg;
#![allow(dead_code)]
#![allow(unused_variables)]
use rand;
use rand::{SeedableRng};
use chrono::{Local,DateTime,NaiveDate,NaiveTime,Datelike, Duration};
use rand::distributions::{Standard,Uniform};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
mod equity;
mod core;
mod utils;
mod cmdty;


use rand::prelude::*;
use serde::Deserialize;

use std::fs::File;
use std::io::Read;
use std::{io, thread};
use std::collections::HashMap;
use std::error::Error;
use csv;
//use std::env::{args,Args};
use utils::read_csv;
use utils::RNG;

use std::env::{args, temp_dir};
use rand::Rng;
use equity::blackscholes;
use crate::equity::montecarlo;
use clap::{App,Arg};
use std::env;
use utils::parse_json;


fn main() {
    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    if num_args < 2 {
        println!("Usage: {} <argument>", args[0]);
        return;
    }
    let argument = &args[1];
    let output_filename = &args[2];
    println!("You provided the argument: {}", argument);
    let mut file = File::open(argument).expect("Failed to open JSON file");
    parse_json::parse_contract(&mut file,output_filename);

}









