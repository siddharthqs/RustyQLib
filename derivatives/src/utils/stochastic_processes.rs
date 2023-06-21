pub trait StochasticProcess{
    fn drift(&self)-> f64;
    fn diffusion(&self)-> f64;
}
