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

#[derive(Deserialize)]
struct MyData {
    field1: String,
    field2: i32,
}
fn main0() {
    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    if num_args < 2 {
        println!("Usage: {} <argument>", args[0]);
        return;
    }
    let argument = &args[1];
    println!("You provided the argument: {}", argument);



    // let _matches = App::new("qsLib").version("0.1.0").author("Siddharthqs.com")
    //     .about("Quant Library for retail traders").get_matches();
    // println!("Welcome to Option pricing CLI");
    // println!(" Do you want to price option (1) or calculate implied volatility (2)?");

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
        blackscholes::implied_volatility();
    } else {
        println!("You gave a wrong number! The accepted arguments are 1 and 2.")
    }




}
fn main2() {
    struct Point {
        x: f64,
        y: f64,
    }
    impl Point {
        fn in_circles(&self) -> bool {
            let distance = (self.x.powi(2)   + self.y.powi(2)).sqrt();
            if distance<=1.0{
                return true;
            }
            else{
                return false;
            }
        }
    }
    let mut rng = rand::thread_rng();
    let mut point = Point{
        x:0.0, y: 0.0,
    };
    let number_of_simulation = 10000000;
    let mut in_circle_count = 0;
    for i in 0..number_of_simulation{
        let x:f64 = Uniform::new(-1.0,1.0).sample(&mut rng);
        let y:f64 = Uniform::new(-1.0,1.0).sample(&mut rng);
        point.x = x;
        point.y = y;
        if point.in_circles(){
            in_circle_count+=1;
        }
    }
    let pi = (in_circle_count as f64) / (number_of_simulation as f64)*4.0;
    println!("{:?}",pi);


}
fn main3() {
    let mut rng = rand::thread_rng();
    let number_of_simulation = 100000;
    let mut sum_of_simulation =0.0;
    for i in 0..number_of_simulation{
        let mut x:f64 = Uniform::new(0.0,1.0).sample(&mut rng);
        sum_of_simulation+=x.powi(2);
    }
    let avg = sum_of_simulation / (number_of_simulation as f64);
    println!("{}",avg);
}
fn main(){


    let args: Vec<String> = env::args().collect();
    let num_args = args.len();

    if num_args < 2 {
        println!("Usage: {} <argument>", args[0]);
        return;
    }
    let argument = &args[1];
    println!("You provided the argument: {}", argument);
    let mut file = File::open(argument).expect("Failed to open JSON file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
         .expect("Failed to read JSON file");
    //
    // // Deserialize the JSON into a Rust struct
    let data: MyData = serde_json::from_str(&contents).expect("Failed to deserialize JSON");
    //
    // // Now you can work with the data
    println!("field1: {}", data.field1);
    println!("field2: {}", data.field2);
}






