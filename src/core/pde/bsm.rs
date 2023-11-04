/// This is Black Scholes Merton PDE solver using finite difference method
/// $ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0 $
/// $ V(S,T) = max(S-K,0) $
/// $ V(0,t) = 0 $
/// $ V(S,t) \rightarrow S $ as $ S \rightarrow \infty $
///https://de.wikipedia.org/wiki/Thomas-Algorithmus
// pub fn blackscholes_pde(spot:f64,strike:f64,rate:f64,volatility:f64,time_to_maturity:f64,steps:u64,option_type:OptionType) -> f64{
//     let mut grid = Grid::new(spot,strike,rate,volatility,time_to_maturity,steps,option_type);
//     grid.solve();
//     let value = grid.get_value();
//     value
// }
// pub struct Grid {
//     spot: f64,
//     strike: f64,
//     rate: f64,
//     dividend: f64,
//     //volatility:f64,
//     time_to_maturity: f64,
//     spot_steps: u64,
//     time_steps: u64
// }
// impl Grid{
//     pub fn payoff(&self,spot:f64) -> f64{
//         let payoff = (spot - self.strike).max(0.0);
//         payoff
//     }
//     pub fn build_grid(&self) -> Vec<Vec<f64>>{
//         let mut grid:Array2<f64> = Array2::zeros((self.time_steps as usize,self.spot_steps as usize));
//         //let mut grid = vec![vec![0.0;self.spot_steps as usize];self.time_steps as usize];
//         for i in 0..self.spot_steps as usize{
//             grid[0][i] = self.payoff(i as f64);
//         }
//         grid
//     }
// }



