//! Forward-start options: the strike is fixed at a future date `t_f` as a
//! fraction `k` of the then-prevailing spot; the payoff at expiry `T` is
//! `(S_T - k * S_{t_f})^+` (call) or the mirrored put.
//!
//! - **Black-Scholes closed form** (Rubinstein 1991) via homogeneity:
//!   `price = S_0 e^{-q t_f} * BS(1, k, r, q, sigma, T - t_f)`.
//! - **Monte Carlo** through [`Payoff::path_payoff`]: the payoff reads
//!   `S_{t_f}` off the simulated path, so the option prices under GBM,
//!   local vol and — the reason this product exists — **Heston stochastic
//!   vol**, whose forward smile differs materially from Black-Scholes.

use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::blackscholes::bs_price;
use crate::equity::utils::{Payoff, PayoffType};

#[derive(Debug)]
pub struct ForwardStartPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    /// Strike as a fraction of the spot on the fixing date (1.0 = at the money).
    pub strike_fraction: f64,
    /// Fixing time as a fraction of the option life, in (0, 1).
    pub start_fraction: f64,
}

impl Payoff for ForwardStartPayoff {
    /// Degenerate single-point value (strike not yet fixed): zero.
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, path: &[f64], _strike: f64) -> f64 {
        let n = path.len();
        // step i covers time (i+1) * T/n: the fixing index for t_f
        let idx = ((self.start_fraction * n as f64).round() as usize).clamp(1, n - 1) - 1;
        let strike = self.strike_fraction * path[idx];
        let terminal = path[n - 1];
        match self.put_or_call {
            PutOrCall::Call => (terminal - strike).max(0.0),
            PutOrCall::Put => (strike - terminal).max(0.0),
        }
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::ForwardStart
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
}

/// Rubinstein closed form under Black-Scholes: by homogeneity the value at
/// the fixing date is `S_{t_f} * BS(1, k, r, q, sigma, T - t_f)`, so the
/// time-0 value replaces `S_{t_f}` by its dividend-discounted spot.
#[allow(clippy::too_many_arguments)]
pub fn forward_start_price(
    s: f64,
    strike_fraction: f64,
    r: f64,
    q: f64,
    sigma: f64,
    start_t: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(start_t > 0.0 && start_t < t, "fixing must lie inside the option life");
    s * (-q * start_t).exp() * bs_price(1.0, strike_fraction, r, q, sigma, t - start_t, put_or_call)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduces_to_vanilla_when_fixing_is_immediate() {
        // t_f -> 0: strike ~= k * S_0, so the price approaches a vanilla
        // struck at k * S_0
        let fs = forward_start_price(100.0, 1.0, 0.05, 0.02, 0.3, 1e-6, 1.0, PutOrCall::Call);
        let vanilla = bs_price(100.0, 100.0, 0.05, 0.02, 0.3, 1.0, PutOrCall::Call);
        assert!((fs - vanilla).abs() < 1e-3, "{fs} vs {vanilla}");
    }

    #[test]
    fn price_is_homogeneous_in_spot() {
        let p1 = forward_start_price(100.0, 1.0, 0.05, 0.02, 0.3, 0.5, 1.0, PutOrCall::Call);
        let p2 = forward_start_price(200.0, 1.0, 0.05, 0.02, 0.3, 0.5, 1.0, PutOrCall::Call);
        assert!((p2 - 2.0 * p1).abs() < 1e-12);
    }

    #[test]
    fn path_payoff_reads_fixing_off_the_path() {
        let payoff = ForwardStartPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: ContractStyle::European,
            strike_fraction: 1.0,
            start_fraction: 0.5,
        };
        // 4-step path: fixing at step index 1 (time 0.5T), terminal 120
        let path = [100.0, 90.0, 110.0, 120.0];
        assert!((payoff.path_payoff(&path, 0.0) - 30.0).abs() < 1e-12); // 120 - 90
    }
}
