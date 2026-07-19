//! Autocallable notes (single underlying) with an autocall coupon (rebate)
//! and knock-in capital protection.
//!
//! Mechanics (classic "Athena" structure) on equally spaced observation
//! dates `t_1 .. t_n` (with `t_n = T`):
//! - if `S(t_m) >= autocall_barrier`, the note redeems early at `t_m`
//!   paying `notional + m * coupon` (the accrued coupon is the rebate);
//! - if never called: at `T`, if the path never breached
//!   `protection_barrier` (discretely monitored on the simulation grid),
//!   the holder receives the notional back; otherwise the protection is
//!   knocked in and the holder receives `notional * S_T / S_initial`
//!   (1:1 downside participation from the *contractual* initial fixing).
//!
//! Cash flows occur at different dates, so pricing is a dedicated Monte
//! Carlo route that discounts each call date on the option's curve. The
//! route runs under GBM, **Dupire local volatility** (the market-standard
//! model for these notes — the skew drives the knock-in value) and Heston.

use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::equity::utils::{Payoff, PayoffType};

#[derive(Debug)]
pub struct AutocallablePayoff {
    pub exercise_style: ContractStyle,
    /// Early-redemption trigger level (absolute).
    pub autocall_barrier: f64,
    /// Knock-in barrier for the capital protection (absolute).
    pub protection_barrier: f64,
    /// Coupon (rebate) accrued per observation period, paid at call.
    pub coupon: f64,
    /// Number of equally spaced observations over the life (last = expiry).
    pub observations: usize,
    pub notional: f64,
    /// Contractual initial fixing for the downside participation ratio.
    pub initial_fixing: f64,
}

impl AutocallablePayoff {
    /// Value of one simulated path: redemption cash flow times the discount
    /// factor of its payment date. `obs_idx` maps observation m to its path
    /// step; `dfs[m]` is the discount factor to that date.
    pub fn path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64]) -> f64 {
        for (m, &idx) in obs_idx.iter().enumerate() {
            if path[idx] >= self.autocall_barrier {
                return (self.notional + self.coupon * (m + 1) as f64) * dfs[m];
            }
        }
        // never called: capital protection at maturity
        let df_final = *dfs.last().unwrap();
        let knocked_in = path.iter().any(|&s| s <= self.protection_barrier);
        if knocked_in {
            let terminal = *path.last().unwrap();
            self.notional * (terminal / self.initial_fixing) * df_final
        } else {
            self.notional * df_final
        }
    }
}

impl Payoff for AutocallablePayoff {
    /// Degenerate single-point value: zero (all value is path- and
    /// schedule-dependent).
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, _path: &[f64], _strike: f64) -> f64 {
        panic!(
            "Autocallables pay at multiple dates and cannot be valued through \
             path_payoff; the Monte Carlo engine prices them via path_value"
        );
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Autocallable
    }
    fn put_or_call(&self) -> &PutOrCall {
        // the embedded optionality is put-like; the field is not used by
        // the pricing routes
        &PutOrCall::Call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note() -> AutocallablePayoff {
        AutocallablePayoff {
            exercise_style: ContractStyle::European,
            autocall_barrier: 100.0,
            protection_barrier: 70.0,
            coupon: 5.0,
            observations: 4,
            notional: 100.0,
            initial_fixing: 100.0,
        }
    }

    #[test]
    fn calls_at_first_breach_with_accrued_coupon() {
        let payoff = note();
        let obs_idx = [1, 3, 5, 7];
        let dfs = [0.99, 0.98, 0.97, 0.96];
        // second observation (index 3) is the first at/above the barrier
        let path = [90.0, 95.0, 99.0, 101.0, 50.0, 50.0, 50.0, 50.0];
        let value = payoff.path_value(&path, &obs_idx, &dfs);
        assert!((value - (100.0 + 2.0 * 5.0) * 0.98).abs() < 1e-12);
    }

    #[test]
    fn protected_redemption_when_never_called_nor_knocked() {
        let payoff = note();
        let path = [90.0, 92.0, 91.0, 95.0, 93.0, 92.0, 94.0, 96.0];
        let value = payoff.path_value(&path, &[1, 3, 5, 7], &[0.99, 0.98, 0.97, 0.96]);
        assert!((value - 100.0 * 0.96).abs() < 1e-12);
    }

    #[test]
    fn downside_participation_after_knock_in() {
        let payoff = note();
        // dips through the 70 protection barrier, finishes at 80
        let path = [90.0, 65.0, 75.0, 80.0, 78.0, 82.0, 79.0, 80.0];
        let value = payoff.path_value(&path, &[1, 3, 5, 7], &[0.99, 0.98, 0.97, 0.96]);
        assert!((value - 100.0 * (80.0 / 100.0) * 0.96).abs() < 1e-12);
    }
}
