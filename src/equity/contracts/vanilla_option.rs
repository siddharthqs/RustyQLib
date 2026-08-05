//! The vanilla payoff — and, for backward compatibility, re-exports of
//! everything that historically lived in this module: the central
//! instrument types (now in [`equity_option`](super::equity_option)) and
//! the asian/barrier/binary/lookback payoffs (now with their products).

use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::utils::{Payoff, PayoffType};

pub use super::asian::AsianPayoff;
pub use super::barrier::BarrierPayoff;
pub use super::binary_option::{BinaryPayoff, BinaryType};
pub use super::equity_option::{EquityMarketData, EquityOption, EquityOptionBase};
pub use super::lookback::{LookbackPayoff, LookbackType};

#[derive(Debug, Clone)]
pub struct VanillaPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}

impl Payoff for VanillaPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match &self.put_or_call {
            PutOrCall::Call => (spot - strike).max(0.0),
            PutOrCall::Put => (strike - spot).max(0.0),
        }
    }
    fn path_payoff_var<'t>(
        &self,
        path: &[crate::core::aad::Var<'t>],
        strike: f64,
    ) -> Option<crate::core::aad::Var<'t>> {
        let terminal = *path.last().expect("empty path");
        Some(match self.put_or_call {
            PutOrCall::Call => (terminal - strike).maxf(0.0),
            PutOrCall::Put => (strike - terminal).maxf(0.0),
        })
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Vanilla
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}
