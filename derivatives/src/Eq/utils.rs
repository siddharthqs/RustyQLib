use std::f64::consts::{PI, SQRT_2};
use libm::{log, exp};
use probability;
use probability::distribution::Distribution;
pub fn dN(x:f64)->f64 {
    // Probability density function of standard normal random variable x.
    let t = -0.5 * x * x;
    return t.exp() / (SQRT_2 * PI.sqrt());
}

pub fn N(x: f64) -> f64 {
    //umulative density function of standard normal random variable x.
    let m = probability::distribution::Gaussian::new(0.0,1.0);
    let cdf = m.distribution(x);
    return cdf
 }