// extern crate probability;
// extern crate rand_chacha;
// extern crate rand_pcg;

use rand;
//use rand::{SeedableRng};
//use chrono::{Local,DateTime,NaiveDate,NaiveTime,Datelike, Duration};
//use rand::distributions::{Standard,Uniform};
//use rand::distributions::Distribution;
//use rand_distr::StandardNormal;
mod equity;
mod core;
mod utils;
mod cmdty;
mod rates;

use std::fs;
use std::path::Path;

use rand::prelude::*;
use serde::Deserialize;

use std::fs::File;
use std::io::Read;
use std::{io, thread};
use std::collections::HashMap;
use std::error::Error;
//use csv;
//use std::env::{args,Args};
//use utils::read_csv;
//use utils::RNG;

//use std::env::{args, temp_dir};
//use rand::Rng;
use equity::blackscholes;
use crate::equity::montecarlo;
use clap::{App, Arg, ArgMatches, SubCommand};
//use std::env;
use utils::parse_json;
use std::time::{Instant};

#[allow(dead_code)]
#[allow(unused_variables)]
fn main() {
    let matches = App::new("RustyQLib Quant Library for Option Pricing")
        .version("0.0.1")
        .author("Siddharth Singh <siddharth_qs@outlook.com>")
        .about("Pricing and risk management of financial derivatives")
        .subcommand(
            SubCommand::with_name("build")
                .about("Building the curve / Vol surface")
                .arg(
                    Arg::with_name("input")
                        .short("i")
                        .long("input")
                        .value_name("FILE")
                        .help("input financial contracts to use in constuction ")
                        .required(true)
                        .takes_value(true)
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .required(true)
                        .takes_value(true)
                )
        )
        .subcommand(
        SubCommand::with_name("file")
            .about("Pricing a single contract")
            .arg(
                Arg::with_name("input")
                    .short("i")
                    .long("input")
                    .value_name("FILE")
                    .help("Pricing a single contract")
                    .required(true)
                    .takes_value(true)
            )
            .arg(
                Arg::with_name("output")
                    .short("o")
                    .long("output")
                    .value_name("FILE")
                    .help("Output file name")
                    .required(true)
                    .takes_value(true)
            )
    )
        .subcommand(
            SubCommand::with_name("dir")
                .about("Pricing all contracts in a directory")
                .arg(
                    Arg::with_name("input")
                        .short("i")
                        .long("input")
                        .value_name("DIR")
                        .help("Pricing all contracts in a directory")
                        .required(true)
                        .takes_value(true)
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .help("Output priced contracts to a directory")
                        .required(true)
                        .takes_value(true)
                )
        )
        .subcommand(
            SubCommand::with_name("interactive")
                .about("Interactive mode")
        )
        .get_matches();
    let build_matches = matches.subcommand_matches("build");
    let input_matches = matches.subcommand_matches("file");
    let dir_matches = matches.subcommand_matches("dir");
    let interactive_matches = matches.subcommand_matches("interactive");
    match matches.subcommand(){
        ("build",Some(build_matches)) => {

            let input_file = build_matches.value_of("input").unwrap();
            let output_file = build_matches.value_of("output").unwrap();
            let mut file = File::open(input_file).expect("Failed to open JSON file");
            let start_time = Instant::now();
            parse_json::build_curve(&mut file,output_file);
            let end_time = Instant::now();
            let elapsed_time = end_time - start_time;
            println!("Time taken: {:?}", elapsed_time);
        }
        ("file",Some(input_matches)) => {
            let input_file = input_matches.value_of("input").unwrap();
            let output_file = input_matches.value_of("output").unwrap();
            let mut file = File::open(input_file).expect("Failed to open JSON file");
            let start_time = Instant::now();
            parse_json::parse_contract(&mut file,output_file);
            let end_time = Instant::now();
            let elapsed_time = end_time - start_time;
            println!("Time taken: {:?}", elapsed_time);
        }
        ("dir",Some(dir_matches)) => {
            let input_dir = dir_matches.value_of("input").unwrap();
            let output_dir = dir_matches.value_of("output").unwrap();
            let start_time = Instant::now();
            let output_vec:Vec<String> = Vec::new();
            let files = fs::read_dir(input_dir).unwrap();
            for ifile in files {
                let ifile = ifile.unwrap();
                let path = ifile.path();
                if path.is_file(){
                    // Check if the file has a ".json" extension
                    if let Some(extension) = path.extension() {
                        if extension == "json" {
                            let mut file = File::open(ifile.path()).expect("Failed to open JSON file");
                            let output_file_i = output_dir.to_owned() + "\\" + &ifile.path().file_name().unwrap().to_str().unwrap();
                            parse_json::parse_contract(&mut file,&output_file_i);
                        }
                    }
                }
            }
            let end_time = Instant::now();
            let elapsed_time = end_time - start_time;
            println!("Time taken to process the dir: {:?}", elapsed_time);
        }
        ("interactive",Some(interactive_matches)) => {

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
            }
        (_, _) => {
            println!("No mode specified. Please use --help to see the available options.");
        }
        }

}









