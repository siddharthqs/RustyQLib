// extern crate probability;
// extern crate rand_chacha;
// extern crate rand_pcg;

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

#[allow(dead_code)]
#[allow(unused_variables)]
fn main() {
    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    if num_args < 2 {
        println!("Usage: {} <argument>", args[0]);
        return;
    }
    let flag = &args[1];
    //let argument = &args[2];
    // make argument optional
    let binding = String::from("default");
    let argument = args.get(2).expect(" ok");
    //let argument = args.get(2).unwrap_or_else(|| &String::from("default"));


    let output_filename = &args[3];
    let output_filename = args.get(3).expect("ok ");


    //let output_filename = args.get(3).unwrap_or_else(|| &String::from("default"));
    let flag = &args[1];
    println!("You provided the argument: {}", flag);
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
        "-i" => {
            println!("Welcome to Option pricing CLI");
            loop {
                println!(" Do you want to price option (1) or calculate implied volatility (2)? or (3) to exit");
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
                        } else if model == 2 {
                            montecarlo::option_pricing();
                    } else {
                        println!("You gave a wrong number! The accepted arguments are 1 and 2.")
                    }
                } else if input == 2 {
                    blackscholes::implied_volatility();
                }
                else if input == 3 {
                    break;
                }
                else {
                    println!("You gave a wrong number! The accepted arguments are 1 and 2.")
                }

            }
        },
        _ => println!("No flag found"),
    }

}









