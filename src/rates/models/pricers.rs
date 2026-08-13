//! Generic pricers on top of [`OneFactorAffine`] models.
//!
//! Everything here reduces to European options on zero-coupon bonds:
//!
//! - **Coupon-bond options** by Jamshidian (1989): in a one-factor
//!   model where bond prices move monotonically in the short rate, an
//!   option on a coupon bond decomposes exactly into a portfolio of
//!   zero-bond options struck at the critical rate `r*` at which the
//!   coupon bond is worth the strike.
//! - **European swaptions** via the bond-option equivalence: a payer
//!   swaption is a put on the fixed leg (coupons plus redemption)
//!   struck at the notional; a receiver swaption is the call.
//! - **Caplets / floorlets**: a caplet paying `N tau (L - K)^+` equals
//!   `N (1 + K tau)` zero-bond puts struck at `1/(1 + K tau)`.
//!
//! Times are year fractions from the model's anchor; the caller
//! converts dates (see the `short_rate_models` example, which builds a
//! swaption schedule from swap accrual periods).

use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::rates::models::OneFactorAffine;
use crate::rates::PayerReceiver;

/// Bracket for the Jamshidian critical-rate solve.
const RATE_BRACKET: (f64, f64) = (-2.0, 10.0);

fn validate_flows(expiry: f64, flows: &[(f64, f64)]) -> Result<(), RustyQLibError> {
    if flows.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            "no cash flows after the expiry",
        ));
    }
    for &(time, amount) in flows {
        if !(time > expiry && time.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "coupon bond option",
                format!("flow time {time} must lie strictly after the expiry {expiry}"),
            ));
        }
        if !(amount > 0.0 && amount.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "coupon bond option",
                format!("flow amounts must be positive, got {amount} at {time}"),
            ));
        }
    }
    Ok(())
}

/// European option, valued today, on a bond paying `flows`
/// (`(time, amount)` pairs after `expiry`), struck at `strike` — the
/// Jamshidian decomposition.
pub fn coupon_bond_option(
    model: &impl OneFactorAffine,
    expiry: f64,
    flows: &[(f64, f64)],
    strike: f64,
    put_or_call: PutOrCall,
) -> Result<f64, RustyQLibError> {
    if !(expiry > 0.0 && expiry.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            format!("expiry must be positive, got {expiry}"),
        ));
    }
    if !(strike > 0.0 && strike.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            format!("strike must be positive, got {strike}"),
        ));
    }
    validate_flows(expiry, flows)?;

    // the critical short rate r* at which the coupon bond is worth the
    // strike at expiry; bond value is strictly decreasing in the rate,
    // so bisection on a wide bracket is safe
    let bond_value = |rate: f64| -> f64 {
        flows
            .iter()
            .map(|&(time, amount)| {
                amount
                    * model
                        .zero_bond(expiry, time, rate)
                        .expect("flow times were validated against the expiry")
            })
            .sum()
    };
    // normalized by the strike so the tolerance is scale-free (a
    // million-notional leg solves as precisely as a unit one)
    let objective = |rate: f64| 1.0 - bond_value(rate) / strike;
    let root = Solver1d::new(1e-12, 200).bisection(objective, RATE_BRACKET.0, RATE_BRACKET.1)?;
    if !root.converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations: root.iterations,
            residual: objective(root.x).abs(),
            reason: "Jamshidian critical-rate solve did not converge".to_string(),
        });
    }
    let critical_rate = root.x;

    // decompose: each flow's strike is its zero-bond price at r*
    let mut value = 0.0;
    for &(time, amount) in flows {
        let flow_strike = model.zero_bond(expiry, time, critical_rate)?;
        value += amount * model.zero_bond_option(expiry, time, flow_strike, put_or_call)?;
    }
    Ok(value)
}

/// European swaption on a swap starting at `expiry`: the fixed leg pays
/// `notional * strike_rate * tau` at each `(payment_time, tau)` and the
/// notional at the last payment. Priced as a coupon-bond option struck
/// at the notional (payer = put, receiver = call).
pub fn european_swaption(
    model: &impl OneFactorAffine,
    expiry: f64,
    fixed_leg: &[(f64, f64)],
    strike_rate: f64,
    notional: f64,
    payer_receiver: PayerReceiver,
) -> Result<f64, RustyQLibError> {
    if !(notional > 0.0 && notional.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "swaption",
            format!("notional must be positive, got {notional}"),
        ));
    }
    if !(strike_rate > 0.0 && strike_rate.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "swaption",
            format!("the bond-option equivalence needs a positive fixed rate, got {strike_rate}"),
        ));
    }
    let mut flows: Vec<(f64, f64)> = fixed_leg
        .iter()
        .map(|&(payment_time, tau)| (payment_time, notional * strike_rate * tau))
        .collect();
    let last = flows.last_mut().ok_or_else(|| {
        RustyQLibError::invalid_input("swaption", "the fixed leg has no payments")
    })?;
    last.1 += notional;

    let put_or_call = match payer_receiver {
        // paying fixed profits when the fixed leg is worth less than par
        PayerReceiver::Payer => PutOrCall::Put,
        PayerReceiver::Receiver => PutOrCall::Call,
    };
    coupon_bond_option(model, expiry, &flows, notional, put_or_call)
}

/// Caplet on the simple rate over `[start, end]` with accrual `tau`
/// (usually `end - start` on the leg's day count): pays
/// `notional * tau * (L - strike)^+` at `end`. Equal to
/// `notional * (1 + strike * tau)` zero-bond puts expiring at `start`
/// on the `end` bond, struck at `1 / (1 + strike * tau)`.
pub fn caplet(
    model: &impl OneFactorAffine,
    start: f64,
    end: f64,
    tau: f64,
    strike: f64,
    notional: f64,
) -> Result<f64, RustyQLibError> {
    caplet_floorlet(model, start, end, tau, strike, notional, PutOrCall::Put)
}

/// Floorlet: pays `notional * tau * (strike - L)^+` at `end`.
pub fn floorlet(
    model: &impl OneFactorAffine,
    start: f64,
    end: f64,
    tau: f64,
    strike: f64,
    notional: f64,
) -> Result<f64, RustyQLibError> {
    caplet_floorlet(model, start, end, tau, strike, notional, PutOrCall::Call)
}

fn caplet_floorlet(
    model: &impl OneFactorAffine,
    start: f64,
    end: f64,
    tau: f64,
    strike: f64,
    notional: f64,
    bond_option_side: PutOrCall,
) -> Result<f64, RustyQLibError> {
    if !(tau > 0.0 && tau.is_finite() && strike > 0.0 && strike.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "caplet",
            format!("need positive tau and strike, got tau={tau}, strike={strike}"),
        ));
    }
    if !(notional > 0.0 && notional.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "caplet",
            format!("notional must be positive, got {notional}"),
        ));
    }
    let scale = 1.0 + strike * tau;
    let bond_strike = 1.0 / scale;
    Ok(notional * scale * model.zero_bond_option(start, end, bond_strike, bond_option_side)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::models::{HullWhite, ShortRateModel, Vasicek};
    use chrono::NaiveDate;

    fn hull_white() -> HullWhite {
        let curve = YieldCurve::flat(
            0.04,
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        HullWhite::new(0.1, 0.01, curve).unwrap()
    }

    /// A 1y-into-5y annual fixed leg.
    fn fixed_leg() -> Vec<(f64, f64)> {
        (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect()
    }

    #[test]
    fn single_flow_jamshidian_collapses_to_the_zero_bond_option() {
        let m = hull_white();
        // an option on a single zero-coupon flow must equal the direct
        // zero-bond option: amount * zbo with strike scaled by amount
        let (expiry, time, amount, strike) = (1.0, 4.0, 100.0, 88.0);
        let via_jamshidian =
            coupon_bond_option(&m, expiry, &[(time, amount)], strike, PutOrCall::Call).unwrap();
        let direct = amount
            * m.zero_bond_option(expiry, time, strike / amount, PutOrCall::Call)
                .unwrap();
        assert!(
            (via_jamshidian - direct).abs() < 1e-10,
            "{via_jamshidian} vs {direct}"
        );
    }

    #[test]
    fn swaption_parity_recovers_the_forward_swap_value() {
        // payer - receiver = value of the forward-starting payer swap,
        // computable from the curve alone — an exact identity
        let m = hull_white();
        let leg = fixed_leg();
        let (strike, notional) = (0.045, 1_000_000.0);
        let payer =
            european_swaption(&m, 1.0, &leg, strike, notional, PayerReceiver::Payer).unwrap();
        let receiver =
            european_swaption(&m, 1.0, &leg, strike, notional, PayerReceiver::Receiver).unwrap();
        let curve = m.curve();
        let float_pv = notional * (curve.df(1.0) - curve.df(6.0));
        let fixed_pv: f64 = leg
            .iter()
            .map(|&(t, tau)| notional * strike * tau * curve.df(t))
            .sum();
        assert!(
            (payer - receiver - (float_pv - fixed_pv)).abs() < 1e-6,
            "parity: {payer} - {receiver} vs {}",
            float_pv - fixed_pv
        );
        assert!(payer > 0.0 && receiver > 0.0);
    }

    #[test]
    fn atm_swaption_is_symmetric_and_grows_with_volatility() {
        let m = hull_white();
        let leg = fixed_leg();
        let curve = m.curve();
        // ATM forward swap rate from the curve
        let annuity: f64 = leg.iter().map(|&(t, tau)| tau * curve.df(t)).sum();
        let atm = (curve.df(1.0) - curve.df(6.0)) / annuity;
        let payer = european_swaption(&m, 1.0, &leg, atm, 1e6, PayerReceiver::Payer).unwrap();
        let receiver = european_swaption(&m, 1.0, &leg, atm, 1e6, PayerReceiver::Receiver).unwrap();
        // at the forward rate the two sides are worth the same
        assert!(
            (payer - receiver).abs() < 1e-6 * payer.max(1.0),
            "{payer} vs {receiver}"
        );
        // doubling sigma raises the ATM price
        let hotter = HullWhite::new(0.1, 0.02, curve.clone()).unwrap();
        let payer_hot =
            european_swaption(&hotter, 1.0, &leg, atm, 1e6, PayerReceiver::Payer).unwrap();
        assert!(payer_hot > payer);
    }

    #[test]
    fn cap_floor_parity_prices_the_fra() {
        // caplet - floorlet = tau * (F - K) discounted: exactly
        // N * (P(0,start) - P(0,end)) - N * K * tau * P(0,end)
        let m = hull_white();
        let (start, end, tau, strike, notional) = (1.0, 1.5, 0.5, 0.042, 1_000_000.0);
        let cap = caplet(&m, start, end, tau, strike, notional).unwrap();
        let floor = floorlet(&m, start, end, tau, strike, notional).unwrap();
        let curve = m.curve();
        let fra =
            notional * (curve.df(start) - curve.df(end)) - notional * strike * tau * curve.df(end);
        assert!((cap - floor - fra).abs() < 1e-8, "{cap} - {floor} vs {fra}");
        assert!(cap > 0.0 && floor > 0.0);
    }

    #[test]
    fn jamshidian_works_for_vasicek_too() {
        let m = Vasicek::new(0.15, 0.045, 0.008, 0.04).unwrap();
        let flows: Vec<(f64, f64)> = (1..=4).map(|i| (1.0 + i as f64, 5.0)).collect();
        let mut with_redemption = flows.clone();
        with_redemption.last_mut().unwrap().1 += 100.0;
        let strike = 100.0;
        let call = coupon_bond_option(&m, 1.0, &with_redemption, strike, PutOrCall::Call).unwrap();
        let put = coupon_bond_option(&m, 1.0, &with_redemption, strike, PutOrCall::Put).unwrap();
        // parity against the model's own discount factors
        let bond_pv: f64 = with_redemption
            .iter()
            .map(|&(t, c)| c * m.zero_bond(0.0, t, m.r0).unwrap())
            .sum();
        let parity = bond_pv - strike * m.zero_bond(0.0, 1.0, m.r0).unwrap();
        assert!((call - put - parity).abs() < 1e-9, "{call} - {put}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let m = hull_white();
        assert!(coupon_bond_option(&m, 0.0, &[(2.0, 1.0)], 1.0, PutOrCall::Call).is_err());
        assert!(coupon_bond_option(&m, 1.0, &[], 1.0, PutOrCall::Call).is_err());
        // flow at or before the expiry
        assert!(coupon_bond_option(&m, 1.0, &[(1.0, 1.0)], 1.0, PutOrCall::Call).is_err());
        assert!(coupon_bond_option(&m, 1.0, &[(2.0, -1.0)], 1.0, PutOrCall::Call).is_err());
        assert!(
            european_swaption(&m, 1.0, &fixed_leg(), -0.01, 1e6, PayerReceiver::Payer).is_err()
        );
        assert!(european_swaption(&m, 1.0, &[], 0.04, 1e6, PayerReceiver::Payer).is_err());
        assert!(caplet(&m, 1.0, 1.5, 0.0, 0.04, 1e6).is_err());
    }
}
