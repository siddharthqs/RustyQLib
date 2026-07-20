use std::time::Instant;
use clap::{Arg, ArgMatches, Command};
use std::fs::File;
use crate::utils::parse_contracts;
use std::fs;
use std::path::Path;
use std::io;
use std::io::Write;
use crate::equity::blackscholes;
use crate::equity::montecarlo;
pub fn build_cli() -> Command {
    Command::new("RustyQLib Quant Library for Option Pricing")
        .version("0.0.2")
        .author("Siddharth Singh <siddharth_qs@outlook.com>")
        .about("Pricing and risk management of financial derivatives")
        .subcommand(
            Command::new("build")
                .about("Building the curve / Vol surface")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Input financial contracts to use in construction")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("file")
                .about("Pricing a single contract")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("FILE")
                        .help("Pricing a single contract")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output file name")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("dir")
                .about("Pricing all contracts in a directory")
                .arg(
                    Arg::new("input")
                        .short('i')
                        .long("input")
                        .value_name("DIR")
                        .help("Pricing all contracts in a directory")
                        .required(true),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("DIR")
                        .help("Output priced contracts to a directory")
                        .required(true),
                ),
        )
        .subcommand(
            Command::new("interactive").about("Interactive mode"),
        )
}

/// Handle the "build" subcommand.
pub fn handle_build(matches: &ArgMatches) {
    let input_file = matches.get_one::<String>("input").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();

    // We measure the time of the operation
    measure_time("build_curve", || {
        let mut file = File::open(input_file).expect("Failed to open JSON file");
        parse_contracts::build_curve(&mut file, output_file);
        println!("(Stub) build_curve from {}", input_file);
    });

    // Save or do something with output_file if needed
}

/// Handle the "file" subcommand.
pub fn handle_file(matches: &ArgMatches) {
    let input_file = matches.get_one::<String>("input").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();

    measure_time("parse_contract (single file)", || {
        let mut file = File::open(input_file).expect("Failed to open JSON file");
        parse_contracts::parse_contract(&mut file, output_file);
        println!("(Stub) parse_contract from {}", input_file);
    });
}

/// Handle the "dir" subcommand.
pub fn handle_dir(matches: &ArgMatches) {
    let input_dir = matches.get_one::<String>("input").unwrap();
    let output_dir = matches.get_one::<String>("output").unwrap();

    let input_path = Path::new(input_dir);
    let output_path = Path::new(output_dir);

    measure_time("parse_contract (directory)", || {
        // Read the directory
        let files = fs::read_dir(input_path).expect("Failed to read input directory");

        for file_result in files {
            let dir_entry = file_result.expect("Failed to read entry");
            let path = dir_entry.path();

            let is_contract_file = path.is_file()
                && matches!(
                    path.extension().and_then(|s| s.to_str()).map(|e| e.to_lowercase()).as_deref(),
                    Some("json") | Some("xml")
                );
            if is_contract_file {
                let mut file = File::open(&path).expect("Failed to open contract file");

                // Construct the corresponding output file path
                let output_file_path = output_path.join(
                    path.file_name().expect("Failed to get file name"),
                );

                parse_contracts::parse_contract(&mut file, output_file_path.to_str().unwrap());
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
