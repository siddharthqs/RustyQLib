///Enum for different engines to price options
#[derive(PartialEq,Clone,Debug)]
pub enum Engine{
    BlackScholes,
    MonteCarlo,
    Binomial,
    FiniteDifference
}