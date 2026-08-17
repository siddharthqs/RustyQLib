//! Bachelier (normal) model: European options on a future/forward `F`
//! whose increments are arithmetic, `dF = sigma dW`.
//!
//! The model of choice when the underlying can be negative or sit near
//! zero — commodity basis (Waha, AECO), spreads, and outrights in
//! dislocations (CME switched energy options to Bachelier when WTI went
//! negative in April 2020). The vol is quoted in **price units per
//! √year**, and neither `F` nor `K` needs to be positive.
//!
//! Mirrors [`crate::equity::black76`]: same argument order, same two
//! [`FuturesSettlement`] styles (up-front discounted premium, or
//! futures-style margined with no discounting), Greeks with respect to
//! `F`, `sigma`, `r` and calendar time.

use crate::core::trade::PutOrCall;
use crate::core::utils::{norm_cdf, norm_pdf};
use crate::equity::black76::FuturesSettlement;

fn d(f: f64, k: f64, sigma: f64, t: f64) -> f64 {
    (f - k) / (sigma * t.sqrt())
}

fn intrinsic(f: f64, k: f64, put_or_call: PutOrCall) -> f64 {
    match put_or_call {
        PutOrCall::Call => (f - k).max(0.0),
        PutOrCall::Put => (k - f).max(0.0),
    }
}

/// Bachelier price of a European option on a future.
pub fn price(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    if t <= 0.0 || sigma <= 0.0 {
        return df * intrinsic(f, k, put_or_call);
    }
    let st = sigma * t.sqrt();
    let d = d(f, k, sigma, t);
    match put_or_call {
        PutOrCall::Call => df * ((f - k) * norm_cdf(d) + st * norm_pdf(d)),
        PutOrCall::Put => df * ((k - f) * norm_cdf(-d) + st * norm_pdf(d)),
    }
}

/// Delta with respect to the futures price `F`.
pub fn delta(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let d = d(f, k, sigma, t);
    match put_or_call {
        PutOrCall::Call => df * norm_cdf(d),
        PutOrCall::Put => -df * norm_cdf(-d),
    }
}

/// Gamma with respect to `F` (same for calls and puts).
pub fn gamma(f: f64, k: f64, r: f64, sigma: f64, t: f64, settlement: FuturesSettlement) -> f64 {
    let df = settlement.discount_factor(r, t);
    df * norm_pdf(d(f, k, sigma, t)) / (sigma * t.sqrt())
}

/// Vega per unit of normal vol (same for calls and puts).
pub fn vega(f: f64, k: f64, r: f64, sigma: f64, t: f64, settlement: FuturesSettlement) -> f64 {
    let df = settlement.discount_factor(r, t);
    df * norm_pdf(d(f, k, sigma, t)) * t.sqrt()
}

/// Theta (calendar time decay, `dV/dt = -dV/dT`).
pub fn theta(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    // volatility bleed sigma dN(d) / (2 sqrt(T)), common to calls and puts
    let bleed = df * sigma * norm_pdf(d(f, k, sigma, t)) / (2.0 * t.sqrt());
    match settlement {
        FuturesSettlement::Margined => -bleed,
        FuturesSettlement::Discounted => {
            r * price(f, k, r, sigma, t, put_or_call, settlement) - bleed
        }
    }
}

/// Rho: zero when margined; `-T * price` when the premium is discounted
/// (`F` is exogenous, so `r` enters only through the discount).
pub fn rho(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    match settlement {
        FuturesSettlement::Margined => 0.0,
        FuturesSettlement::Discounted => -t * price(f, k, r, sigma, t, put_or_call, settlement),
    }
}

/// Vanna, the change in delta per unit change in normal vol (same for
/// calls and puts).
pub fn vanna(f: f64, k: f64, r: f64, sigma: f64, t: f64, settlement: FuturesSettlement) -> f64 {
    let df = settlement.discount_factor(r, t);
    let d = d(f, k, sigma, t);
    -df * norm_pdf(d) * d / sigma
}

/// Charm, the change in delta per year of calendar time.
pub fn charm(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let df = settlement.discount_factor(r, t);
    let d = d(f, k, sigma, t);
    let d_dt = -d / (2.0 * t);
    let delta_component = match put_or_call {
        PutOrCall::Call => norm_cdf(d),
        PutOrCall::Put => norm_cdf(d) - 1.0,
    };
    let discount_decay = match settlement {
        FuturesSettlement::Discounted => r * df * delta_component,
        FuturesSettlement::Margined => 0.0,
    };
    discount_decay - df * norm_pdf(d) * d_dt
}

/// Zomma, the change in gamma per unit change in normal vol (same for
/// calls and puts).
pub fn zomma(f: f64, k: f64, r: f64, sigma: f64, t: f64, settlement: FuturesSettlement) -> f64 {
    let df = settlement.discount_factor(r, t);
    let d = d(f, k, sigma, t);
    df * norm_pdf(d) * (d * d - 1.0) / (sigma * sigma * t.sqrt())
}

/// Delta elasticity, `F * gamma / delta`, `NaN` when delta is zero.
pub fn gamma_p(
    f: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> f64 {
    let del = delta(f, k, r, sigma, t, put_or_call, settlement);
    if del == 0.0 {
        f64::NAN
    } else {
        f * gamma(f, k, r, sigma, t, settlement) / del
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::black76;

    const R: f64 = 0.05;
    const SIG: f64 = 22.5; // $/sqrt(year)
    const T: f64 = 0.75;

    #[test]
    fn put_call_parity_any_sign_of_forward() {
        for (s, factor) in [
            (FuturesSettlement::Discounted, (-R * T).exp()),
            (FuturesSettlement::Margined, 1.0),
        ] {
            // parity must hold for positive, zero and negative forwards
            for f in [75.0, 0.0, -5.25, -40.0] {
                let k = -2.0;
                let c = price(f, k, R, SIG, T, PutOrCall::Call, s);
                let p = price(f, k, R, SIG, T, PutOrCall::Put, s);
                assert!((c - p - factor * (f - k)).abs() < 1e-10, "{s:?} F={f}");
                assert!(c >= 0.0 && p >= 0.0);
            }
        }
    }

    #[test]
    fn margined_is_the_discounted_price_grossed_up() {
        let disc = price(
            -5.0,
            -2.0,
            R,
            SIG,
            T,
            PutOrCall::Call,
            FuturesSettlement::Discounted,
        );
        let marg = price(
            -5.0,
            -2.0,
            R,
            SIG,
            T,
            PutOrCall::Call,
            FuturesSettlement::Margined,
        );
        assert!((marg - disc * (R * T).exp()).abs() < 1e-10);
        assert_eq!(
            rho(
                -5.0,
                -2.0,
                R,
                SIG,
                T,
                PutOrCall::Call,
                FuturesSettlement::Margined
            ),
            0.0
        );
    }

    #[test]
    fn greeks_match_central_bumps() {
        use FuturesSettlement::Discounted as D;
        let (f, k) = (-5.0, -2.0);
        let h = 1e-5;
        let dv = |x: f64| price(x, k, R, SIG, T, PutOrCall::Call, D);
        let bumped_delta = (dv(f + h) - dv(f - h)) / (2.0 * h);
        assert!((delta(f, k, R, SIG, T, PutOrCall::Call, D) - bumped_delta).abs() < 1e-8);
        let bumped_gamma = (dv(f + h) - 2.0 * dv(f) + dv(f - h)) / (h * h);
        assert!((gamma(f, k, R, SIG, T, D) - bumped_gamma).abs() < 1e-4);
        let vv = |s: f64| price(f, k, R, s, T, PutOrCall::Call, D);
        let bumped_vega = (vv(SIG + h) - vv(SIG - h)) / (2.0 * h);
        assert!((vega(f, k, R, SIG, T, D) - bumped_vega).abs() < 1e-8);
        let tv = |t: f64| price(f, k, R, SIG, t, PutOrCall::Call, D);
        let bumped_theta = -(tv(T + h) - tv(T - h)) / (2.0 * h);
        assert!((theta(f, k, R, SIG, T, PutOrCall::Call, D) - bumped_theta).abs() < 1e-6);
        let bumped_vanna = (delta(f, k, R, SIG + h, T, PutOrCall::Call, D)
            - delta(f, k, R, SIG - h, T, PutOrCall::Call, D))
            / (2.0 * h);
        assert!((vanna(f, k, R, SIG, T, D) - bumped_vanna).abs() < 1e-8);
        let bumped_charm = -(delta(f, k, R, SIG, T + h, PutOrCall::Call, D)
            - delta(f, k, R, SIG, T - h, PutOrCall::Call, D))
            / (2.0 * h);
        assert!((charm(f, k, R, SIG, T, PutOrCall::Call, D) - bumped_charm).abs() < 1e-8);
        let bumped_zomma =
            (gamma(f, k, R, SIG + h, T, D) - gamma(f, k, R, SIG - h, T, D)) / (2.0 * h);
        assert!((zomma(f, k, R, SIG, T, D) - bumped_zomma).abs() < 1e-8);
        let bumped_rho = (price(f, k, R + h, SIG, T, PutOrCall::Call, D)
            - price(f, k, R - h, SIG, T, PutOrCall::Call, D))
            / (2.0 * h);
        assert!((rho(f, k, R, SIG, T, PutOrCall::Call, D) - bumped_rho).abs() < 1e-8);
    }

    #[test]
    fn small_vol_at_the_money_agrees_with_black76() {
        // for sigma_N = F * sigma_LN and small total vol the two models
        // coincide to O((sigma sqrt(T))^3)
        let (f, k, sig_ln) = (100.0, 100.0, 0.01);
        let b76 = black76::price(
            f,
            k,
            R,
            sig_ln,
            1.0,
            PutOrCall::Call,
            FuturesSettlement::Discounted,
        );
        let bach = price(
            f,
            k,
            R,
            sig_ln * f,
            1.0,
            PutOrCall::Call,
            FuturesSettlement::Discounted,
        );
        assert!((b76 - bach).abs() < 1e-4, "{b76} vs {bach}");
    }

    #[test]
    fn degenerate_inputs_price_intrinsic() {
        let df = (-R * T as f64).exp();
        assert_eq!(
            price(
                -2.0,
                -5.0,
                R,
                0.0,
                T,
                PutOrCall::Call,
                FuturesSettlement::Discounted
            ),
            df * 3.0
        );
        assert_eq!(
            price(
                -2.0,
                -5.0,
                R,
                SIG,
                0.0,
                PutOrCall::Put,
                FuturesSettlement::Discounted
            ),
            0.0
        );
    }
}
