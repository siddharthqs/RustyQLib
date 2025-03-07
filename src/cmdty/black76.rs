// use libm::{exp, log};
// use std::f64::consts::{PI, SQRT_2};
// use crate::core::utils::{dN, N};
// use crate::core::trade;
//
// use super::cmdty_option::{CmdtyOption,Engine};
// use super::super::core::termstructure::YieldTermStructure;
// use super::super::core::traits::{Instrument,Greeks};
// use super::super::core::interpolation;
//
// pub fn npv(bsd_option: &&CmdtyOption) -> f64 {
//     assert!(bsd_option.volatility >= 0.0);
//     assert!(bsd_option.time_to_maturity >= 0.0);
//     assert!(bsd_option.current_price.value >= 0.0);
//     if bsd_option.option_type == trade::PutOrCall::Call {
//         let option_price = bsd_option.current_price.value() * N(bsd_option.d1())
//             - bsd_option.strike_price * N(bsd_option.d2());
//         return option_price;
//     } else {
//         let option_price = -bsd_option.current_price.value()
//             * N(-bsd_option.d1())
//             + bsd_option.strike_price * N(-bsd_option.d2());
//         return option_price;
//     }
// }
//
//
//
// impl CmdtyOption {
//     pub fn set_risk_free_rate(&mut self){
//        let model = interpolation::CubicSpline::new(&self.term_structure.date, &self.term_structure.rates);
//        let r = model.interpolation(self.time_to_maturity);
//        self.risk_free_rate = Some(r);
//     }
//     pub fn get_premium_at_risk(&self) -> f64 {
//         let value = self.npv();
//         let mut pay_off = 0.0;
//         if self.option_type == trade::OptionType::Call {
//             pay_off = self.current_price.value() - self.strike_price;
//         } else if self.option_type == trade::OptionType::Put {
//             pay_off = self.strike_price - self.current_price.value();
//         }
//         if pay_off > 0.0 {
//             return value - pay_off;
//         } else {
//             return value;
//         }
//     }
//     pub fn d1(&self) -> f64 {
//         //Black76 d1 function Parameters
//         let tmp1 = (self.current_price.value() / self.strike_price).ln()
//             + (0.5 * self.volatility.powi(2))
//             * self.time_to_maturity;
//
//         let tmp2 = self.volatility * (self.time_to_maturity.sqrt());
//         return tmp1 / tmp2;
//     }
//     pub fn d2(&self) -> f64 {
//         let d2 = self.d1() - self.volatility * self.time_to_maturity.powf(0.5);
//         return d2;
//     }
//     // pub fn imp_vol(&mut self,option_price:f64) -> f64 {
//     //     for i in 0..100{
//     //         let d_sigma = (self.npv()-option_price)/self.vega();
//     //         self.volatility -= d_sigma
//     //     }
//     //     self.volatility
//     // }
// }
// impl Greeks for CmdtyOption{
//      fn delta(&self) -> f64 {
//         let mut delta = N(self.d1());
//         if self.option_type == trade::OptionType::Call {
//             delta = delta;
//         } else if self.option_type == trade::OptionType::Put {
//             delta = delta - 1.0;
//         }
//         return delta;
//     }
//     fn gamma(&self) -> f64 {
//         let gamma = dN(self.d1());
//
//         let var_sqrt = self.volatility * (self.time_to_maturity.sqrt());
//         return gamma / (self.current_price.value() * var_sqrt);
//     }
//     fn vega(&self) -> f64 {
//         //St * dN(d1) * math.sqrt(T - t)
//         let vega = self.current_price.value() * dN(self.d1()) * self.time_to_maturity.sqrt();
//         return vega;
//     }
//     fn theta(&self) -> f64 {
//         let mut theta = 0.0;
//         if self.option_type == trade::OptionType::Call {
//             //-(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) + r * K * math.exp(-r * (T - t)) * N(d2))
//             let t1 = -self.current_price.value() * dN(self.d1()) * self.volatility
//                 / (2.0 * self.time_to_maturity.sqrt());
//
//             theta = t1;
//         } else if self.option_type == trade::OptionType::Put {
//             //-(St * dN(d1) * sigma / (2 * math.sqrt(T - t)) - r * K * math.exp(-r * (T - t)) * N(d2))
//             let t1 = -self.current_price.value() * dN(self.d1()) * self.volatility
//                 / (2.0 * self.time_to_maturity.sqrt());
//
//             theta = t1;
//         }
//
//         return theta;
//     }
//     fn rho(&self) -> f64 {
//         //rho K * (T - t) * math.exp(-r * (T - t)) * N(d2)
//         let mut rho = 0.0;
//         if self.option_type == trade::OptionType::Call {
//             rho = self.strike_price
//                 * self.time_to_maturity
//
//                 * N(self.d2());
//         } else if self.option_type == trade::OptionType::Put {
//             //put_rho = -K * (T - t) * math.exp(-r * (T - t)) * N(-d2)
//             rho = -self.strike_price
//                 * self.time_to_maturity
//                 * N(-self.d2());
//         }
//
//         return rho;
//     }
// }