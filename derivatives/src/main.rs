extern crate probability;

//mod equity_options;
mod Eq;

use std::{thread,io};
//use std::env::{args,Args};
fn main() {
    //let a = equity_options::OptionType
    println!("Do you want to use the Binomial (1), Black-Sholes (2) or Monte-Carlo (3) option pricing model?");
    let mut model = String::new();
    print!("{}",model);
    io::stdin().read_line(&mut model).expect("Failed to read line");
    let model: u8= model.trim().parse::<u8>().unwrap();

    if model == 1 {
        print!("{}",model);
        Eq::blackscholes::option_pricing();
    } else if model == 2 {
        print!("{}",model);
        Eq::blackscholes::option_pricing();
    } else if model == 3 {
        print!("{}",model);
        Eq::blackscholes::option_pricing();
    } else {
        println!("You gave a wrong number! The accepted arguments are 1, 2 and 3.")
    }

}