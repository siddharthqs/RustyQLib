use rand::Rng;
use rand::distributions::{Standard,Uniform};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use libm::cos;
use libm::sin;

fn generate_standard_normal_marsaglia_polar() -> (f64, f64) {
    let mut rng = rand::thread_rng();
    let mut X = 0.0;
    let mut Y = 0.0;
    let mut S = 0.0f64;

    while(true) {
        X = Uniform::new(0.0,1.0).sample(&mut rng)*2.0 -1.0;
        Y = Uniform::new(0.0,1.0).sample(&mut rng)*2.0 -1.0;
        S = X*X + Y*Y;
        if S<1.0f64 && S != 0.0f64 {
            break;
        }
    }

    let I = ((-2.0 * S.ln()) / S).sqrt();
    (I*X,I*Y)

}

fn generate_standard_normal_box() -> (f64, f64) {
    let mut rng = rand::thread_rng();

    let r:f64 = Uniform::new(0.0,1.0).sample(&mut rng);
    let p:f64 = Uniform::new(0.0,1.0).sample(&mut rng);

    let tmp:f64 = (-2.0*r.ln()).sqrt();
    (tmp*cos(p*2.0*PI),tmp*sin(p*2.0*PI))

}

pub fn get_vector_standard_normal(size:u64)-> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut rn_vec:Vec<f64> = Vec::new();
    for i in 0..size{
        rn_vec.push(rng.sample(StandardNormal));
    }
    rn_vec
}
pub fn get_matrix_standard_normal(size_n:u64,size_m:u64)-> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let mut rn_vec_n:Vec<Vec<f64>> = Vec::new();
    for i in 0..size_n{
        let mut rn_vec_m:Vec<f64> = Vec::new();
        for j in 0..size_m{
            rn_vec_m.push(rng.sample(StandardNormal));
        }
        rn_vec_n.push(rn_vec_m);
    }
    rn_vec_n
}