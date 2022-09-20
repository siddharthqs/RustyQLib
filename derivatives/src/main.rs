extern crate probability;

mod equity_options;


//use std::env::{args,Args};
fn main() {
    //let a = equity_options::OptionType
    let o = equity_options::EquityOption{
        option_type: equity_options::OptionType::Call,
        transection : equity_options::Transection::Buy,
        current_price: 1.0,
        strike_price: 1.0,
        dividend_yield: 0.01,
        volatility: 0.2,
        time_to_maturity: 0.12,
        risk_free_rate: 0.05,
        transection_price: 0.9
    };
    o.calulate_price();
    let d = o.d1();
    println!("{},d1 of the EQ option",d);

}