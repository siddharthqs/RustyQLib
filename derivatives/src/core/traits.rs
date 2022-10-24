pub trait Instrument{
    fn npv(&self)-> f64;
}

pub trait Greeks{
    fn delta(&self) -> f64;
    fn gamma(&self) -> f64;
    fn vega(&self) -> f64;
    fn theta(&self) -> f64;
    fn rho(&self) -> f64;

}