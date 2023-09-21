//mod dis{
use libm::{exp, log};
use probability;
use probability::distribution::Distribution;
use std::f64::consts::{PI, SQRT_2};

pub fn dN(x: f64) -> f64 {
    // Probability density function of standard normal random variable x.
    let t = -0.5 * x * x;
    return t.exp() / (SQRT_2 * PI.sqrt());
}

pub fn N(x: f64) -> f64 {
    //umulative density function of standard normal random variable x.
    let m = probability::distribution::Gaussian::new(0.0, 1.0);
    let cdf = m.distribution(x);
    return cdf;
}
//}
