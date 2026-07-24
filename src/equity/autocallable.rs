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
    /// Phoenix feature: when set, the coupon is paid **at each
    /// observation** with `S >= coupon_barrier` (independently of the
    /// autocall), instead of accruing as a rebate paid only at call.
    pub coupon_barrier: Option<f64>,
    /// Phoenix memory feature: missed coupons are recovered at the next
    /// observation above the coupon barrier.
    pub memory: bool,
}

impl AutocallablePayoff {
    /// Value of one simulated path: redemption cash flow times the discount
    /// factor of its payment date. `obs_idx` maps observation m to its path
    /// step; `dfs[m]` is the discount factor to that date.
    pub fn path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64]) -> f64 {
        match self.coupon_barrier {
            None => self.athena_path_value(path, obs_idx, dfs),
            Some(cb) => self.phoenix_path_value(path, obs_idx, dfs, cb),
        }
    }

    /// Classic Athena: the coupon accrues and is paid only at call.
    fn athena_path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64]) -> f64 {
        for (m, &idx) in obs_idx.iter().enumerate() {
            if path[idx] >= self.autocall_barrier {
                return (self.notional + self.coupon * (m + 1) as f64) * dfs[m];
            }
        }
        self.redemption_at_maturity(path) * dfs.last().unwrap()
    }

    /// Phoenix: conditional coupons at every observation above the coupon
    /// barrier (with optional memory), redemption logic unchanged.
    fn phoenix_path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64], cb: f64) -> f64 {
        let mut value = 0.0;
        let mut missed = 0usize;
        for (m, &idx) in obs_idx.iter().enumerate() {
            let s = path[idx];
            if s >= cb {
                let units = if self.memory { 1 + missed } else { 1 };
                value += self.coupon * units as f64 * dfs[m];
                missed = 0;
            } else {
                missed += 1;
            }
            if s >= self.autocall_barrier {
                return value + self.notional * dfs[m];
            }
        }
        value + self.redemption_at_maturity(path) * dfs.last().unwrap()
    }

    /// Never called: knock-in protection at maturity.
    fn redemption_at_maturity(&self, path: &[f64]) -> f64 {
        let knocked_in = path.iter().any(|&s| s <= self.protection_barrier);
        if knocked_in {
            self.notional * (path.last().unwrap() / self.initial_fixing)
        } else {
            self.notional
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
            coupon_barrier: None,
            memory: false,
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

    fn phoenix(memory: bool) -> AutocallablePayoff {
        let mut p = note();
        p.coupon_barrier = Some(80.0);
        p.memory = memory;
        p
    }

    #[test]
    fn phoenix_pays_conditional_coupons_and_redeems_at_call() {
        let payoff = phoenix(false);
        let obs_idx = [1, 3, 5, 7];
        let dfs = [0.99, 0.98, 0.97, 0.96];
        // obs1: 85 >= 80 -> coupon; obs2: 101 -> coupon + autocall
        let path = [90.0, 85.0, 99.0, 101.0, 50.0, 50.0, 50.0, 50.0];
        let value = payoff.path_value(&path, &obs_idx, &dfs);
        let expect = 5.0 * 0.99 + 5.0 * 0.98 + 100.0 * 0.98;
        assert!((value - expect).abs() < 1e-12, "{value} vs {expect}");
    }

    #[test]
    fn phoenix_memory_recovers_missed_coupons() {
        let no_memory = phoenix(false);
        let with_memory = phoenix(true);
        let obs_idx = [1, 3, 5, 7];
        let dfs = [0.99, 0.98, 0.97, 0.96];
        // obs1 below coupon barrier (75 < 80), obs2 above (85): memory
        // pays 2 coupons there; never autocalled, never knocked in (>70)
        let path = [90.0, 75.0, 78.0, 85.0, 90.0, 88.0, 90.0, 95.0];
        let v_plain = no_memory.path_value(&path, &obs_idx, &dfs);
        let v_memory = with_memory.path_value(&path, &obs_idx, &dfs);
        // plain: coupons at obs2, obs3, obs4 + notional at maturity
        let plain = 5.0 * (0.98 + 0.97 + 0.96) + 100.0 * 0.96;
        // memory: obs2 pays the missed obs1 coupon too
        assert!((v_plain - plain).abs() < 1e-12, "{v_plain} vs {plain}");
        assert!((v_memory - (plain + 5.0 * 0.98)).abs() < 1e-12, "{v_memory}");
    }

    #[test]
    fn phoenix_with_zero_coupon_equals_athena_with_zero_coupon() {
        // no coupons anywhere: both structures are pure autocall + protection
        let mut athena = note();
        athena.coupon = 0.0;
        let mut phx = phoenix(true);
        phx.coupon = 0.0;
        let obs_idx = [1, 3, 5, 7];
        let dfs = [0.99, 0.98, 0.97, 0.96];
        for path in [
            [90.0, 85.0, 99.0, 101.0, 50.0, 50.0, 50.0, 50.0],
            [90.0, 65.0, 75.0, 80.0, 78.0, 82.0, 79.0, 80.0],
            [90.0, 92.0, 91.0, 95.0, 93.0, 92.0, 94.0, 96.0],
        ] {
            let a = athena.path_value(&path, &obs_idx, &dfs);
            let p = phx.path_value(&path, &obs_idx, &dfs);
            assert!((a - p).abs() < 1e-12);
        }
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
