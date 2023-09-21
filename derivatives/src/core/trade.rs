pub enum Transection {
    Buy,
    Sell,
}

#[derive(PartialEq, Debug)]
pub enum OptionType {
    Call,
    Put,
    Straddle,
}