//! Jamshidian's engine: generic pricers on top of [`OneFactorAffine`]
//! models, on year-fraction inputs. The date-aware products in
//! [`contracts`](crate::rates::contracts) — [`Swaption`] and
//! [`CapFloor`] — build their schedules and call down into here.
//!
//! Everything here reduces to European options on zero-coupon bonds:
//!
//! - **Coupon-bond options** by Jamshidian (1989): in a one-factor
//!   model where bond prices move monotonically in the short rate, an
//!   option on a coupon bond decomposes exactly into a portfolio of
//!   zero-bond options struck at the critical rate `r*` at which the
//!   coupon bond is worth the strike. The strike may be paid at a
//!   `settlement` after the `expiry` (the `_settled` variants): then
//!   `r*` equates the bond to `strike` settlement bonds and each piece
//!   is an exchange option between two zero bonds — still closed form
//!   in the Gaussian models.
//! - **European swaptions** via the bond-option equivalence: a payer
//!   swaption is a put on the fixed leg (coupons plus redemption)
//!   struck at the notional; a receiver swaption is the call. The
//!   notional is exchanged at the swap start, which is the settlement
//!   of the bond option — exactly the exercise date, or a settlement
//!   lag after it.
//! - **Caplets / floorlets**: a caplet paying `N tau (L - K)^+` equals
//!   `N (1 + K tau)` zero-bond puts struck at `1/(1 + K tau)`.
//!
//! Times are year fractions from the model's anchor; the caller
//! converts dates; the products in `contracts` do that from swap
//! accrual periods.
//!
//! [`Swaption`]: crate::rates::contracts::swaption::Swaption
//! [`CapFloor`]: crate::rates::contracts::cap_floor::CapFloor

use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::rates::models::OneFactorAffine;
use crate::rates::PayerReceiver;

/// Bracket for the Jamshidian critical-rate solve.
const RATE_BRACKET: (f64, f64) = (-2.0, 10.0);

fn validate_flows(settlement: f64, flows: &[(f64, f64)]) -> Result<(), RustyQLibError> {
    if flows.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            "no cash flows after the settlement",
        ));
    }
    for &(time, amount) in flows {
        if !(time > settlement && time.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "coupon bond option",
                format!("flow time {time} must lie strictly after the settlement {settlement}"),
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
    coupon_bond_option_settled(model, expiry, expiry, flows, strike, put_or_call)
}

/// [`coupon_bond_option`] with the strike paid at `settlement >= expiry`
/// rather than at exercise: the payoff at `expiry` is
/// `(bond - strike * P(expiry, settlement))^+` for a call. Jamshidian's
/// decomposition still applies — the bond measured in settlement bonds
/// is monotone in the rate — and each piece is a zero-bond exchange
/// option. `flows` must lie strictly after the settlement.
pub fn coupon_bond_option_settled(
    model: &impl OneFactorAffine,
    expiry: f64,
    settlement: f64,
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
    if !(settlement >= expiry && settlement.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            format!("settlement {settlement} must not precede the expiry {expiry}"),
        ));
    }
    if !(strike > 0.0 && strike.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "coupon bond option",
            format!("strike must be positive, got {strike}"),
        ));
    }
    validate_flows(settlement, flows)?;

    // the critical short rate r* at which the coupon bond is worth
    // `strike` settlement bonds at expiry; that ratio is strictly
    // decreasing in the rate, so bisection on a wide bracket is safe
    let settlement_bond = |rate: f64| -> f64 {
        model
            .zero_bond(expiry, settlement, rate)
            .expect("settlement was validated against the expiry")
    };
    let bond_value = |rate: f64| -> f64 {
        flows
            .iter()
            .map(|&(time, amount)| {
                amount
                    * model
                        .zero_bond(expiry, time, rate)
                        .expect("flow times were validated against the settlement")
            })
            .sum()
    };
    // normalized by the strike so the tolerance is scale-free (a
    // million-notional leg solves as precisely as a unit one)
    let objective = |rate: f64| 1.0 - bond_value(rate) / (strike * settlement_bond(rate));
    // square-root models admit no rate below their floor at expiry
    let lo = RATE_BRACKET.0.max(model.short_rate_floor(expiry) + 1e-12);
    let root = Solver1d::new(1e-12, 200).bisection(objective, lo, RATE_BRACKET.1)?;
    if !root.converged {
        return Err(RustyQLibError::CalibrationFailed {
            iterations: root.iterations,
            residual: objective(root.x).abs(),
            reason: "Jamshidian critical-rate solve did not converge".to_string(),
        });
    }
    let critical_rate = root.x;

    // decompose: each flow's strike is its zero-bond price at r*, in
    // units of the settlement bond
    let settlement_at_critical = settlement_bond(critical_rate);
    let mut value = 0.0;
    for &(time, amount) in flows {
        let flow_strike = model.zero_bond(expiry, time, critical_rate)? / settlement_at_critical;
        value += amount
            * model.zero_bond_exchange_option(
                expiry,
                settlement,
                time,
                flow_strike,
                put_or_call,
            )?;
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
    european_swaption_settled(
        model,
        expiry,
        expiry,
        fixed_leg,
        strike_rate,
        notional,
        payer_receiver,
    )
}

/// [`european_swaption`] on a swap starting at `swap_start >= expiry`
/// (exercise a settlement lag before the swap's effective date): the
/// notional is exchanged at the swap start, so the fixed leg is struck
/// at `notional` settlement bonds.
#[allow(clippy::too_many_arguments)]
pub fn european_swaption_settled(
    model: &impl OneFactorAffine,
    expiry: f64,
    swap_start: f64,
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
    coupon_bond_option_settled(model, expiry, swap_start, &flows, notional, put_or_call)
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
    fn settlement_lag_keeps_parity_against_the_lagged_forward_swap() {
        let m = hull_white();
        let curve = m.curve();
        let leg = fixed_leg();
        let (expiry, strike, notional) = (1.0, 0.045, 1_000_000.0);
        let start = expiry + 2.0 / 365.0;
        // no lag: the settled variant is the plain one, bit for bit
        let plain =
            european_swaption(&m, expiry, &leg, strike, notional, PayerReceiver::Payer).unwrap();
        let same = european_swaption_settled(
            &m,
            expiry,
            expiry,
            &leg,
            strike,
            notional,
            PayerReceiver::Payer,
        )
        .unwrap();
        assert_eq!(plain, same);
        // with a lag: payer - receiver is the forward swap that starts
        // at the swap start, not at the exercise
        let payer = european_swaption_settled(
            &m,
            expiry,
            start,
            &leg,
            strike,
            notional,
            PayerReceiver::Payer,
        )
        .unwrap();
        let receiver = european_swaption_settled(
            &m,
            expiry,
            start,
            &leg,
            strike,
            notional,
            PayerReceiver::Receiver,
        )
        .unwrap();
        let fixed_leg_pv: f64 = leg
            .iter()
            .map(|&(t, tau)| strike * tau * curve.df(t))
            .sum::<f64>()
            + curve.df(6.0);
        let forward_swap = notional * (curve.df(start) - fixed_leg_pv);
        assert!(
            (payer - receiver - forward_swap).abs() < 1e-6 * notional,
            "{payer} - {receiver} vs {forward_swap}"
        );
        // the lag is a small, non-zero correction
        assert!(payer != plain && (payer - plain).abs() < 0.05 * plain);
        // a settlement before the exercise is refused
        assert!(european_swaption_settled(
            &m,
            expiry,
            0.9,
            &leg,
            strike,
            notional,
            PayerReceiver::Payer
        )
        .is_err());
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
