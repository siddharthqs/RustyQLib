extern crate probability;
use chrono::{Local,DateTime,NaiveDate,NaiveTime};
mod equity;
mod core;
mod utils;

use std::{io, thread};
use std::collections::HashMap;
use std::error::Error;
use csv;
//use std::env::{args,Args};
use utils::read_csv;


use std::env::args;
fn main() {

    let input = args().collect::<Vec<String>>();
    let filepath = &input[1];
    read_csv::read_ts(filepath);


}
