#[derive(PartialEq,Debug,Clone)]
pub enum Transection {
    Buy,
    Sell,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum PutOrCall {
    Call,
    Put,

}