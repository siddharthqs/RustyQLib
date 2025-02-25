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


use std::io::Read;
use std::collections::HashMap;
use std::error::Error;
//use csv;
//use std::env::{args,Args};
//use utils::read_csv;
//use utils::RNG;

//use std::env::{args, temp_dir};
//use rand::Rng;

use clap::{App, Arg, ArgMatches, SubCommand};
//use std::env;
use utils::{parse_json,build_cli};
use std::time::{Instant};
#[allow(dead_code)]
#[allow(unused_variables)]
fn main() {
    let matches = build_cli::build_cli().get_matches();
    // Build the CLI

    // Match and dispatch subcommand
    match matches.subcommand() {
        ("build", Some(build_matches)) => build_cli::handle_build(build_matches),
        ("file", Some(file_matches)) => build_cli::handle_file(file_matches),
        ("dir", Some(dir_matches)) => build_cli::handle_dir(dir_matches),
        ("interactive", Some(_)) => build_cli::handle_interactive(),
        _ => {
            // No mode specified or unknown mode
            println!("No valid subcommand specified. Use --help to see available options.");
        }
    }



}











