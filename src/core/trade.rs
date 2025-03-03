#[derive(PartialEq,Debug,Clone)]
pub enum Transection {
    Buy,
    Sell,
}

#[derive(PartialEq, Debug,Clone)]
pub enum PutOrCall {
    Call,
    Put,

}