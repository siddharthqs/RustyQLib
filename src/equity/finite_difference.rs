use super::vanila_option::{EquityOption};
use super::utils::{Engine};

use crate::core::trade::{OptionType,Transection};
use crate::core::utils::{ContractStyle};
use ndarray::{Array, Array2,Array1, ArrayBase, Ix1, OwnedRepr, s};
//use num_integer::Integer;
/// finite difference model for European and American options
pub fn npv(option: &&EquityOption) -> f64 {
    assert!(option.volatility >= 0.0);
    assert!(option.time_to_maturity() >= 0.0);
    assert!(option.underlying_price.value >= 0.0);
    let strike_price = option.strike_price;
    let time_to_maturity = option.time_to_maturity();
    let underlying_price = option.underlying_price.value;
    let volatility = option.volatility;
    let risk_free_rate = option.risk_free_rate;
    let dividend_yield = option.dividend_yield;
    let time_steps:f64 = 1000.0;
    //let time_steps:f64 = time_to_maturity/0.001 as f64;

    let mut spot_steps = (time_steps / 50.0) as usize; //should be even number
    //let spot_steps:usize = 20;
    if spot_steps % 2 != 0{
        spot_steps = spot_steps + 1;
    }// should be even number

    if option.option_type == OptionType::Call {
        return fd(underlying_price,strike_price,risk_free_rate,dividend_yield,volatility,
              time_to_maturity,spot_steps,time_steps);
    } else {
        //TODO implement  for put option
        //println!("Not implemented"");
        return 0.0;
    }

}
fn fd(s0:f64,k:f64,risk_free_rate:f64,dividend_yield:f64,sigma:f64,time_to_mat:f64,spot_steps:usize,time_steps:f64)->f64{
    let ds = 2.0*s0 / spot_steps as f64;

    let M:i32 = (spot_steps as f64+(spot_steps as f64)/2.0) as i32; // 40 +20 = 60-20 = 40
    //let ds = 2.0*s0 / spot_steps as f64;
    //let M:i32 = spot_steps as i32;
    let dt = time_to_mat/(time_steps as f64);
    let time_steps = time_steps as i32;
    // convert float to nearest integer

    //println!(" h {:?}",dt / (ds*ds));
    //let sigma:f64 = 0.3;
    //let r = 0.05;
    //let q = 0.01;
    let r = risk_free_rate-dividend_yield;
    let mut price_grid:Array2<f64> = Array2::zeros((M as usize +1,time_steps as usize+1));
    // Underlying price Grid
    for j in 0..time_steps+1 {
        for i in 0..M+1{
            price_grid[[(M-i) as usize,j as usize]] = (i as f64)*ds as f64;
        }
    }
    let mm = M as usize - ii as usize;
    println!("price_ {:?}",price_grid[[mm as usize as usize,0]]);
    let mut v_grid:Array2<f64> = Array2::zeros((M as usize +1,time_steps as usize+1));
    // Boundary condition
    // for j in 0..time_steps+1 {
    //     for i in 0..M+1{
    //         v_grid[[(M-i) as usize,j as usize]] = (price_grid[[(M-i) as usize,j as usize]]-k).max(0.0);
    //     }
    // }
    // Boundary condition
    for i in 0..M+1{
        v_grid[[(M-i) as usize,time_steps as usize]] = (price_grid[[(M-i) as usize,time_steps as usize]]-k).max(0.0);
    }

    let mut pd:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    let mut b:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    let mut x_:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    let mut y_:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    let mut pu:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    //let ss = price_grid.slice(s![0..M+1,time_steps]).to_vec();
    //let mut xx_:Vec<f64> = Vec::with_capacity((M + 1) as usize);
    for j in 0..M + 1 {
        // ssj = (j as f64)*ds;
        //let x = r * ss[j as usize] / ds;
        let x = r*((M-j) as f64);
        //let y = sigma.powi(2) * (ss[j as usize].powi(2)) / ds.powi(2);
        let y = sigma.powi(2) * ((M-j) as f64).powi(2);
        x_.push(x);
        y_.push(y);
        //xx_.push(ss[j as usize] / ds);
        b.push(1.0 + dt * r + y * dt*0.5); //0.5 * dt * (j as f64)*((r-q) - sigma.powi(2)*(j as f64));
        pu.push(-0.25 * dt * (x + y)); //0.5 * dt * (j as f64)*((r-q) + sigma.powi(2)*(j as f64))
        pd.push(0.25 * dt * (x - y)); //-0.5*dt*(j as f64)*((r-q) + sigma.powi(2)*(j as f64))
    }

    for i in (1..time_steps+1).rev(){
        let mut d = v_grid.slice(s![0..M as usize+1,i]).to_vec();
        d[0] = d[0]*(1.0-pu[0]);
        for j in 1..M{
            d[j as usize] = d[j as usize] +0.25*x_[j as usize]*dt*(d[(j-1) as usize]-d[(j+1) as usize]) +
                0.25*y_[j as usize]*dt*(d[(j-1) as usize]+d[(j+1) as usize] - 2.0*d[j as usize] );
        }
        let x = thomas_algorithm(&pu[0..M as usize], &b, &pd[1..(M+1) as usize], &d);
        for j in 0..M+1{
            v_grid[[j as usize,(i-1) as usize]] = x[j as usize];
        }
    }

    return v_grid[[spot_steps as usize,0]];

}
pub fn thomas_algorithm(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    ///https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    /// Solves Ax = d where A is a tridiagonal matrix consisting of vectors a, b, c
    let n = d.len();
    let mut c_ = c.to_vec();
    let mut d_ = d.to_vec();
    let mut x: Vec<f64> = vec![0.0; n];

    // Adjust for the upper boundary condition
    //d_[0] = d_[0]*(1.0-a[0]);

    c_[0] = c_[0] / b[0];
    d_[0] = d_[0] / b[0];
    for i in 1..n-1 {
        let id = 1.0 / (b[i] - a[i-1] * c_[i - 1]);
        c_[i] = c_[i] * id;
        d_[i] = (d_[i] - a[i-1] * d_[i - 1]) * id;
    }
    d_[n - 1] = (d_[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2]);

    x[n - 1] = d_[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_[i] - c_[i] * x[i + 1];
    }
    x
}
