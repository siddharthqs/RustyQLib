// use super::super::core::termstructure::YieldTermStructure;
// use super::super::core::quotes::Quote;
// use super::super::core::traits::{Instrument,Greeks};
// use crate::core::trade;
// use crate::cmdty::black76;
//
// pub enum Engine{
//     Black76,
//     MonteCarlo
// }
//
//
// pub struct CmdtyOption {
//     pub option_type: trade::PutOrCall,
//     pub transection: trade::Transection,
//     pub current_price: Quote,
//     pub strike_price: f64,
//     pub volatility: f64,
//     pub time_to_maturity: f64,
//     pub time_to_future_maturity: Option<f64>,
//     pub term_structure: YieldTermStructure<f64>,
//     pub risk_free_rate: Option<f64>,
//     pub transection_price: f64,
//     pub engine: Engine,
//     pub simulation:Option<u64>
// }
//
// impl Instrument for CmdtyOption  {
//     fn npv(&self) -> f64 {
//         match self.engine{
//             Engine::Black76 => {
//                 let value = black76::npv(&self);
//                 value
//             }
//             _ => {
//                 0.0
//             }
//         }
//     }
// }