#[derive(Debug,Clone)]
pub enum Transection {
    Buy,
    Sell,
}

#[derive(PartialEq, Debug,Clone)]
pub enum OptionType {
    Call,
    Put,
    Straddle,
}