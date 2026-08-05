//! Binary (digital) options: the payoff definition. Analytic pricing
//! lives in the Black-Scholes engine's payoff-aware pricer; path pricing
//! goes through the generic `payoff` evaluation at the terminal spot.

use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::utils::{Payoff, PayoffType};

/// Binary (digital) settlement style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryType {
    /// Pays a fixed cash amount when in the money.
    CashOrNothing,
    /// Delivers the underlying (pays its level) when in the money.
    AssetOrNothing,
}

#[derive(Debug, Clone)]
pub struct BinaryPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    pub binary_type: BinaryType,
    /// Amount paid by a cash-or-nothing binary (ignored for asset-or-nothing).
    pub cash: f64,
}

/// Binary (digital) payoff, strictly in the money beyond the strike:
/// cash-or-nothing pays `cash`, asset-or-nothing pays the underlying level.
impl Payoff for BinaryPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        let in_the_money = match &self.put_or_call {
            PutOrCall::Call => spot > strike,
            PutOrCall::Put => spot < strike,
        };
        if !in_the_money {
            return 0.0;
        }
        match self.binary_type {
            BinaryType::CashOrNothing => self.cash,
            BinaryType::AssetOrNothing => spot,
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Binary
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
