use rand::Rng;
use rand::distributions::{Standard,Uniform};
use rand::distributions::Distribution;
use rand_distr::StandardNormal;
use std::f64::consts::PI;
use libm::cos;
use libm::sin;
use std::io::Write; // bring trait into scope
use byteorder::{ByteOrder, LittleEndian,BigEndian};
use byteorder::WriteBytesExt;
use byteorder::ReadBytesExt;
use std::fs::File;
use std::io::prelude::*;
use std::fs;
use std::path::Path;
use std::env::temp_dir;
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

pub struct MonteCarloSimulation{
    pub antithetic:bool,
    pub moment_matching:bool,
    pub dimentation: u64,
    pub size: u64,
    pub standard_normal_vector: Vec<f64>,
    pub standard_normal_matrix: Vec<Vec<f64>>
}
impl MonteCarloSimulation{
    pub fn set_standard_normal_vector(&mut self) {
        let mut dir = temp_dir();
        dir.push("rng1d");
        let rng_dir = dir.as_path();
        if !rng_dir.exists() {
            let _ = fs::create_dir(rng_dir);
        }
        dir.push("1dt.bin");
        let path = dir.as_path();
        let mut rn_vec:Vec<f64> = Vec::new();
        if path.exists() {
            rn_vec = read_from_file_byteorder(path).unwrap();
            self.standard_normal_vector = rn_vec;
        }
        else{
            let mut rng = rand::thread_rng();
            let mut rn_vec:Vec<f64> = Vec::new();

            for i in 0..self.size{
                let rn = rng.sample(StandardNormal);
                rn_vec.push(rn);
                if self.antithetic{
                    rn_vec.push(-rn)
                }
            }
            if self.moment_matching{
                let sum = rn_vec.iter().sum::<f64>() as f64;
                let mean = sum / rn_vec.len() as f64;
                let variance = rn_vec.iter().map(|x| {
                    let diff = mean -(*x as f64);
                    diff*diff
                }).sum::<f64>()/rn_vec.len() as f64;
                let std_dev = variance.sqrt();
                let mut mo_rn_vec = vec![];
                for i in 0..rn_vec.len() {
                    let mo_rn = (rn_vec[i]-mean)/std_dev as f64;
                    mo_rn_vec.push(mo_rn)
                }
                rn_vec = mo_rn_vec;
            }
            self.standard_normal_vector = rn_vec.clone();
            write_to_file_byteorder(&rn_vec, path).unwrap();
        }
    }
    pub fn get_standard_normal_vector(&self) ->&Vec<f64>{
        let ptr = &self.standard_normal_vector;
        ptr
    }

}

pub fn get_matrix_standard_normal(size_n:u64,size_m:u64)-> Vec<Vec<f64>> {
    // let mut dir = temp_dir();
    // dir.push("rng2d");
    // dir.push("1dt.bin");
    // let path = dir.as_path();

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
fn write_to_file_byteorder<P: AsRef<Path>>(data: &[f64], path: P) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    for f in data {
        file.write_f64::<BigEndian>(*f)?;
    }
    Ok(())
}
fn read_from_file_byteorder<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<f64>> {
    let mut file = File::open(path)?;
    let buf_len = file.metadata()?.len() / 8; // 8 bytes for one f64
    let mut buf: Vec<f64> = vec![0.0; buf_len.try_into().unwrap()];
    file.read_f64_into::<BigEndian>(&mut buf)?;
    Ok(buf)
}