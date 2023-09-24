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
use std::time::{Instant};

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    if num_args < 2 {
        println!("Usage: {} <argument>", args[0]);
        return;
    }
    let flag = &args[1];
    let argument = &args[2];
    let output_filename = &args[3];
    let flag = &args[1];
    println!("You provided the argument: {}", argument);
    match flag.as_str() {
        "-f" => {
            let mut file = File::open(argument).expect("Failed to open JSON file");
            let start_time = Instant::now();
            parse_json::parse_contract(&mut file,output_filename);
            let end_time = Instant::now();
            let elapsed_time = end_time - start_time;
            println!("Time taken: {:?}", elapsed_time);
        },
        "-d" => println!("Found flag -d"),
        _ => println!("No flag found"),
    }

}









