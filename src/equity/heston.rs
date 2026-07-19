//! Heston (1993) stochastic volatility model.
//!
//! Dynamics under the risk-neutral measure:
//! ```text
//! dS = (r - q) S dt + sqrt(v) S dW_s
//! dv = kappa (theta - v) dt + vol_of_vol * sqrt(v) dW_v,   d<W_s, W_v> = rho dt
//! ```
//!
//! Semi-analytic pricing uses the characteristic function in the
//! "little Heston trap" formulation (Albrecher et al. 2007), which is
//! branch-cut stable under the principal complex logarithm, integrated
//! with composite Simpson. Vanilla calls/puts and both binary types come
//! from the same two probabilities:
//! `call = S e^{-qT} P1 - K e^{-rT} P2`, cash-or-nothing `= e^{-rT} P2`,
//! asset-or-nothing `= S e^{-qT} P1`.
//!
//! Monte Carlo simulation lives in the Monte Carlo engine (full-truncation
//! Euler; the Andersen QE scheme is the planned upgrade).

use serde::{Deserialize, Serialize};

use crate::core::trade::PutOrCall;

// ── Minimal complex arithmetic (principal branches) ─────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
struct Cpx {
    re: f64,
    im: f64,
}

const I: Cpx = Cpx { re: 0.0, im: 1.0 };

impl Cpx {
    fn new(re: f64, im: f64) -> Self {
        Cpx { re, im }
    }
    fn real(re: f64) -> Self {
        Cpx { re, im: 0.0 }
    }
    fn add(self, o: Cpx) -> Cpx {
        Cpx::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Cpx) -> Cpx {
        Cpx::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Cpx) -> Cpx {
        Cpx::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn div(self, o: Cpx) -> Cpx {
        let denom = o.re * o.re + o.im * o.im;
        Cpx::new(
            (self.re * o.re + self.im * o.im) / denom,
            (self.im * o.re - self.re * o.im) / denom,
        )
    }
    fn scale(self, x: f64) -> Cpx {
        Cpx::new(self.re * x, self.im * x)
    }
    fn exp(self) -> Cpx {
        let m = self.re.exp();
        Cpx::new(m * self.im.cos(), m * self.im.sin())
    }
    fn ln(self) -> Cpx {
        Cpx::new(self.norm().ln(), self.im.atan2(self.re))
    }
    fn sqrt(self) -> Cpx {
        let m = self.norm().sqrt();
        let half_arg = 0.5 * self.im.atan2(self.re);
        Cpx::new(m * half_arg.cos(), m * half_arg.sin())
    }
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

// ── Model parameters ────────────────────────────────────────────────────

/// Heston parameters. `theta` is the long-run *variance*, `v0` the initial
/// variance, `vol_of_vol` the volatility of variance (often written xi or
/// sigma), `rho` the spot-variance correlation.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct HestonParams {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    #[serde(alias = "sigma", alias = "xi")]
    pub vol_of_vol: f64,
    pub rho: f64,
}

impl HestonParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.v0 <= 0.0 || self.theta <= 0.0 || self.kappa <= 0.0 || self.vol_of_vol <= 0.0 {
            return Err("Heston v0, kappa, theta, vol_of_vol must be positive".to_string());
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err("Heston rho must be in [-1, 1]".to_string());
        }
        Ok(())
    }

    /// Whether the Feller condition `2 kappa theta >= vol_of_vol^2` holds
    /// (if not, the variance process can touch zero; pricing still works).
    pub fn feller_condition_holds(&self) -> bool {
        2.0 * self.kappa * self.theta >= self.vol_of_vol * self.vol_of_vol
    }

    /// Parameters with a parallel shift applied to the instantaneous and
    /// long-run vol (used for vega bump-and-reprice).
    pub fn with_vol_shift(&self, shift: f64) -> HestonParams {
        let bump = |var: f64| {
            let vol = (var.sqrt() + shift).max(1e-6);
            vol * vol
        };
        HestonParams { v0: bump(self.v0), theta: bump(self.theta), ..*self }
    }
}

// ── Characteristic function and pricing ─────────────────────────────────

/// Characteristic function of ln(S_T) in the trap-free formulation.
fn characteristic_fn(u: Cpx, s: f64, r: f64, q: f64, t: f64, hp: &HestonParams) -> Cpx {
    let kappa = Cpx::real(hp.kappa);
    let eps = hp.vol_of_vol;
    let eps2 = eps * eps;
    let iu = I.mul(u);
    let rho_eps_iu = iu.scale(hp.rho * eps);

    // d = sqrt((rho*eps*iu - kappa)^2 + eps^2 (iu + u^2))
    let a = rho_eps_iu.sub(kappa);
    let d = a.mul(a).add(iu.add(u.mul(u)).scale(eps2)).sqrt();
    // g2 = (kappa - rho*eps*iu - d) / (kappa - rho*eps*iu + d)  (trap-free)
    let kmr = kappa.sub(rho_eps_iu);
    let g2 = kmr.sub(d).div(kmr.add(d));

    let exp_mdt = d.scale(-t).exp();
    let one = Cpx::real(1.0);
    // A = iu (ln S + (r-q) T)
    let a_term = iu.scale(s.ln() + (r - q) * t);
    // B = theta*kappa/eps^2 * ((kappa - rho eps iu - d) T - 2 ln((1 - g2 e^{-dT})/(1 - g2)))
    let log_term = one.sub(g2.mul(exp_mdt)).div(one.sub(g2)).ln();
    let b_term = kmr
        .sub(d)
        .scale(t)
        .sub(log_term.scale(2.0))
        .scale(hp.theta * hp.kappa / eps2);
    // C = v0/eps^2 * (kappa - rho eps iu - d) (1 - e^{-dT}) / (1 - g2 e^{-dT})
    let c_term = kmr
        .sub(d)
        .mul(one.sub(exp_mdt))
        .div(one.sub(g2.mul(exp_mdt)))
        .scale(hp.v0 / eps2);

    a_term.add(b_term).add(c_term).exp()
}

/// The two Heston probabilities: P2 = P(S_T > K) under the risk-neutral
/// measure, P1 the same under the spot measure.
fn probabilities(s: f64, k: f64, r: f64, q: f64, t: f64, hp: &HestonParams) -> (f64, f64) {
    let ln_k = k.ln();
    let forward = s * ((r - q) * t).exp();
    // integrands: Re[ e^{-iu lnK} phi_j(u) / (iu) ]
    let integrand = |u: f64, shifted: bool| -> f64 {
        let uc = Cpx::real(u);
        let phi = if shifted {
            // phi1(u) = phi(u - i) / phi(-i), phi(-i) = forward
            characteristic_fn(uc.sub(I), s, r, q, t, hp).scale(1.0 / forward)
        } else {
            characteristic_fn(uc, s, r, q, t, hp)
        };
        let num = I.scale(-u * ln_k).exp().mul(phi);
        num.div(I.scale(u)).re
    };
    let p = |shifted: bool| 0.5 + simpson(|u| integrand(u, shifted), 1e-9, 250.0, 4000) / std::f64::consts::PI;
    (p(true), p(false))
}

fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

/// Semi-analytic Heston price of a European vanilla option.
#[allow(clippy::too_many_arguments)]
pub fn heston_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && t > 0.0);
    hp.validate().expect("invalid Heston parameters");
    let (p1, p2) = probabilities(s, k, r, q, t, hp);
    let call = s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2;
    match put_or_call {
        PutOrCall::Call => call,
        // put-call parity
        PutOrCall::Put => call - s * (-q * t).exp() + k * (-r * t).exp(),
    }
}

/// Semi-analytic Heston price of a cash-or-nothing binary
/// (`cash * e^{-rT} * P(S_T beyond K)`).
#[allow(clippy::too_many_arguments)]
pub fn heston_binary_cash_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    cash: f64,
    put_or_call: PutOrCall,
) -> f64 {
    let (_, p2) = probabilities(s, k, r, q, t, hp);
    let df = (-r * t).exp();
    match put_or_call {
        PutOrCall::Call => cash * df * p2,
        PutOrCall::Put => cash * df * (1.0 - p2),
    }
}

/// Semi-analytic Heston price of an asset-or-nothing binary
/// (`S e^{-qT} P1` for a call).
pub fn heston_binary_asset_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    put_or_call: PutOrCall,
) -> f64 {
    let (p1, _) = probabilities(s, k, r, q, t, hp);
    let leg = s * (-q * t).exp();
    match put_or_call {
        PutOrCall::Call => leg * p1,
        PutOrCall::Put => leg * (1.0 - p1),
    }
}

// ── Option-level analytic pricing and bump Greeks ───────────────────────

use crate::equity::utils::PayoffType;
use crate::equity::vanila_option::{BinaryPayoff, BinaryType, EquityOption};

/// Reprice the option under Heston with additive bumps to
/// (spot, vol shift, rate, expiry). The vol bump shifts sqrt(v0) and
/// sqrt(theta) in parallel.
fn price_with(option: &EquityOption, ds: f64, dvol: f64, dr: f64, dt_shift: f64) -> f64 {
    let hp = option
        .heston
        .expect("heston parameters are required for the Heston model")
        .with_vol_shift(dvol);
    let s = option.base.effective_spot() + ds;
    let k = option.base.strike_price;
    let r = option.base.risk_free_rate() + dr;
    let q = option.base.carry_yield();
    let t = option.time_to_maturity() + dt_shift;
    let pc = *option.payoff.put_or_call();
    match option.payoff.payoff_kind() {
        PayoffType::Vanilla => heston_price(s, k, r, q, t, &hp, pc),
        PayoffType::Binary => {
            let payoff = option
                .payoff
                .as_any()
                .downcast_ref::<BinaryPayoff>()
                .expect("payoff of kind Binary must be a BinaryPayoff");
            match payoff.binary_type {
                BinaryType::CashOrNothing => {
                    heston_binary_cash_price(s, k, r, q, t, &hp, payoff.cash, pc)
                }
                BinaryType::AssetOrNothing => heston_binary_asset_price(s, k, r, q, t, &hp, pc),
            }
        }
        _ => panic!(
            "The Heston analytic pricer supports vanilla and binary payoffs; \
             use the MonteCarlo engine for path-dependent payoffs"
        ),
    }
}

pub fn analytic_npv(option: &EquityOption) -> f64 {
    price_with(option, 0.0, 0.0, 0.0, 0.0)
}
pub fn analytic_delta(option: &EquityOption) -> f64 {
    let h = option.base.underlying_price.value() * 1e-4;
    (price_with(option, h, 0.0, 0.0, 0.0) - price_with(option, -h, 0.0, 0.0, 0.0)) / (2.0 * h)
}
pub fn analytic_gamma(option: &EquityOption) -> f64 {
    let h = option.base.underlying_price.value() * 1e-3;
    (price_with(option, h, 0.0, 0.0, 0.0) - 2.0 * price_with(option, 0.0, 0.0, 0.0, 0.0)
        + price_with(option, -h, 0.0, 0.0, 0.0))
        / (h * h)
}
/// Sensitivity to a parallel shift of the instantaneous and long-run vol.
pub fn analytic_vega(option: &EquityOption) -> f64 {
    let h = 1e-4;
    (price_with(option, 0.0, h, 0.0, 0.0) - price_with(option, 0.0, -h, 0.0, 0.0)) / (2.0 * h)
}
pub fn analytic_theta(option: &EquityOption) -> f64 {
    let h = (1.0 / 365.0_f64).min(0.5 * option.time_to_maturity());
    -(price_with(option, 0.0, 0.0, 0.0, h) - price_with(option, 0.0, 0.0, 0.0, -h)) / (2.0 * h)
}
pub fn analytic_rho(option: &EquityOption) -> f64 {
    let h = 1e-5;
    (price_with(option, 0.0, 0.0, h, 0.0) - price_with(option, 0.0, 0.0, -h, 0.0)) / (2.0 * h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    fn params() -> HestonParams {
        HestonParams { v0: 0.09, kappa: 2.0, theta: 0.09, vol_of_vol: 0.4, rho: -0.7 }
    }

    #[test]
    fn complex_arithmetic_sanity() {
        let z = Cpx::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-14);
        let e = Cpx::new(0.0, std::f64::consts::PI).exp();
        assert!((e.re + 1.0).abs() < 1e-12 && e.im.abs() < 1e-12, "e^{{i pi}} = -1");
        let s = Cpx::new(-1.0, 0.0).sqrt();
        assert!(s.re.abs() < 1e-12 && (s.im - 1.0).abs() < 1e-12, "sqrt(-1) = i");
        let l = z.ln().exp();
        assert!((l.re - z.re).abs() < 1e-12 && (l.im - z.im).abs() < 1e-12);
    }

    #[test]
    fn degenerates_to_black_scholes_when_vol_of_vol_vanishes() {
        // v0 = theta and vol_of_vol -> 0: variance is constant, so the
        // price must match Black-Scholes at sigma = sqrt(v0)
        let hp = HestonParams { v0: 0.09, kappa: 1.0, theta: 0.09, vol_of_vol: 1e-4, rho: 0.0 };
        for k in [80.0, 100.0, 120.0] {
            let heston = heston_price(100.0, k, 0.05, 0.02, 1.0, &hp, PutOrCall::Call);
            let bs = bs_price(100.0, k, 0.05, 0.02, 0.3, 1.0, PutOrCall::Call);
            assert!((heston - bs).abs() < 1e-4, "K={k}: heston {heston} vs bs {bs}");
        }
    }

    #[test]
    fn put_call_parity() {
        let hp = params();
        let (s, k, r, q, t) = (100.0, 95.0, 0.05, 0.02, 1.0);
        let c = heston_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let p = heston_price(s, k, r, q, t, &hp, PutOrCall::Put);
        let parity = s * (-q * t as f64).exp() - k * (-r * t as f64).exp();
        assert!((c - p - parity).abs() < 1e-10);
    }

    #[test]
    fn probabilities_are_probabilities() {
        let hp = params();
        for k in [50.0, 100.0, 200.0] {
            let (p1, p2) = probabilities(100.0, k, 0.05, 0.0, 1.0, &hp);
            assert!((0.0..=1.0).contains(&p1), "P1 {p1} at K={k}");
            assert!((0.0..=1.0).contains(&p2), "P2 {p2} at K={k}");
        }
        // deep ITM call: both probabilities near 1; deep OTM: near 0
        let (p1, p2) = probabilities(100.0, 1.0, 0.05, 0.0, 1.0, &hp);
        assert!(p1 > 0.999 && p2 > 0.999);
        let (p1, p2) = probabilities(100.0, 10_000.0, 0.05, 0.0, 1.0, &hp);
        assert!(p1 < 1e-3 && p2 < 1e-3);
    }

    #[test]
    fn binaries_replicate_vanilla() {
        // vanilla call = asset-or-nothing - K * cash-or-nothing, under any
        // model with these probabilities
        let hp = params();
        let (s, k, r, q, t) = (100.0, 100.0, 0.05, 0.02, 1.0);
        let vanilla = heston_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let asset = heston_binary_asset_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let cash = heston_binary_cash_price(s, k, r, q, t, &hp, k, PutOrCall::Call);
        assert!((vanilla - (asset - cash)).abs() < 1e-10);
    }

    #[test]
    fn negative_correlation_creates_skew() {
        // rho < 0 fattens the left tail: OTM puts gain value relative to
        // the symmetric case
        let hp_neg = params();
        let hp_zero = HestonParams { rho: 0.0, ..params() };
        let otm_put_neg = heston_price(100.0, 80.0, 0.05, 0.0, 1.0, &hp_neg, PutOrCall::Put);
        let otm_put_zero = heston_price(100.0, 80.0, 0.05, 0.0, 1.0, &hp_zero, PutOrCall::Put);
        assert!(otm_put_neg > otm_put_zero);
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(HestonParams { v0: -0.1, ..params() }.validate().is_err());
        assert!(HestonParams { rho: -1.5, ..params() }.validate().is_err());
        assert!(params().validate().is_ok());
        assert!(params().feller_condition_holds()); // 2*2*0.09 = 0.36 >= 0.4^2
        assert!(!HestonParams { vol_of_vol: 0.9, ..params() }.feller_condition_holds());
    }
}
