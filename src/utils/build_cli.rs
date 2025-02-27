use std::time::Instant;
use clap::{App, Arg, SubCommand};
use std::fs::File;
use utils::parse_json;
use crate::utils;
use std::fs;
use std::path::Path;
use std::{io, thread};
use std::io::Write;
use crate::equity::blackscholes;
use crate::equity::montecarlo;
pub fn build_cli() -> App<'static, 'static> {
    App::new("RustyQLib Quant Library for Option Pricing")
        .version("0.0.2")
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
                        .help("Input financial contracts to use in construction")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .required(true)
                        .takes_value(true),
                ),
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
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .required(true)
                        .takes_value(true),
                ),
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
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("DIR")
                        .help("Output priced contracts to a directory")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("interactive").about("Interactive mode"),
        )
}

/// Handle the "build" subcommand.
pub fn handle_build(matches: &clap::ArgMatches<'_>) {
    let input_file = matches.value_of("input").unwrap();
    let output_file = matches.value_of("output").unwrap();

    // We measure the time of the operation
    measure_time("build_curve", || {
        let mut file = File::open(input_file).expect("Failed to open JSON file");
        parse_json::build_curve(&mut file, output_file);
        println!("(Stub) build_curve from {}", input_file);
    });

    // Save or do something with output_file if needed
}

/// Handle the "file" subcommand.
pub fn handle_file(matches: &clap::ArgMatches<'_>) {
    let input_file = matches.value_of("input").unwrap();
    let output_file = matches.value_of("output").unwrap();

    measure_time("parse_contract (single file)", || {
        let mut file = File::open(input_file).expect("Failed to open JSON file");
        parse_json::parse_contract(&mut file, output_file);
        println!("(Stub) parse_contract from {}", input_file);
    });
}

/// Handle the "dir" subcommand.
pub fn handle_dir(matches: &clap::ArgMatches<'_>) {
    let input_dir = matches.value_of("input").unwrap();
    let output_dir = matches.value_of("output").unwrap();

    let input_path = Path::new(input_dir);
    let output_path = Path::new(output_dir);

    measure_time("parse_contract (directory)", || {
        // Read the directory
        let files = fs::read_dir(input_path).expect("Failed to read input directory");

        for file_result in files {
            let dir_entry = file_result.expect("Failed to read entry");
            let path = dir_entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("json") {
                let mut file = File::open(&path).expect("Failed to open JSON file");

                // Construct the corresponding output file path
                let output_file_path = output_path.join(
                    path.file_name().expect("Failed to get file name"),
                );

                parse_json::parse_contract(&mut file, output_file_path.to_str().unwrap());
                println!(
                    "(Stub) parse_contract from {:?} -> {:?}",
                    path, output_file_path
                );
            }
        }
    });
}

/// Handle the "interactive" subcommand.
pub fn handle_interactive() {
    println!("Welcome to Option pricing CLI");
    loop {
        println!("Do you want to price an option (1), calculate implied volatility (2), or exit (3)?");

        // Prompt user
        print!("> ");
        io::stdout().flush().expect("Failed to flush stdout");

        // Read user input
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .expect("Failed to read line");

        let selection: u8 = match input.trim().parse() {
            Ok(num) => num,
            Err(_) => {
                eprintln!("Please enter a valid number!");
                continue;
            }
        };

        match selection {
            1 => {
                println!("Do you want to use the Black-Scholes (1) or Monte-Carlo (2) model?");
                print!("> ");
                io::stdout().flush().expect("Failed to flush stdout");

                let mut model_input = String::new();
                io::stdin()
                    .read_line(&mut model_input)
                    .expect("Failed to read line");

                let model_num: u8 = match model_input.trim().parse() {
                    Ok(num) => num,
                    Err(_) => {
                        eprintln!("Please enter a valid number!");
                        continue;
                    }
                };

                match model_num {
                    1 => {
                        blackscholes::option_pricing();
                        println!("(Stub) blackscholes::option_pricing()");
                    }
                    2 => {
                        montecarlo::option_pricing();
                        println!("(Stub) montecarlo::option_pricing()");
                    }
                    _ => println!("You gave a wrong number! Accepted arguments are 1 and 2."),
                }
            }
            2 => {
                blackscholes::implied_volatility();
                println!("(Stub) blackscholes::implied_volatility()");
            }
            3 => {
                println!("Exiting interactive mode...");
                break;
            }
            _ => println!("You gave a wrong number! Accepted arguments are 1, 2, or 3."),
        }
    }
}

/// Helper function to measure the time taken by a closure.
fn measure_time<F: FnOnce()>(label: &str, f: F) {
    let start_time = Instant::now();
    f();
    let elapsed_time = start_time.elapsed();
    println!("Time taken for {}: {:?}", label, elapsed_time);
}
