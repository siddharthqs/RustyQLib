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
//! Monte Carlo simulation lives in the Monte Carlo engine through
//! [`HestonProcess`](crate::equity::processes::HestonProcess): Andersen
//! QE with martingale correction by default, full-truncation Euler on
//! request.

use serde::{Deserialize, Serialize};

use crate::core::quadrature::simpson;
use crate::core::trade::PutOrCall;

// ── Complex arithmetic ──────────────────────────────────────────────────

/// The characteristic-function pricers' complex type: the core's shared
/// [`Complex`](crate::core::complex::Complex) (principal branches for
/// `ln` / `sqrt`, which the trap-free formulation relies on), under the
/// short name the Heston/Bates/COS code is written in.
pub(crate) type Cpx = crate::core::complex::Complex;

/// The imaginary unit.
pub(crate) const I: Cpx = Cpx { re: 0.0, im: 1.0 };

// ── Model parameters ────────────────────────────────────────────────────

/// Heston parameters. `theta` is the long-run *variance*, `v0` the initial
/// variance, `vol_of_vol` the volatility of variance (often written xi or
/// sigma), `rho` the spot-variance correlation.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct HestonParams {
    pub v0: f64,
    pub kappa: f64,
    pub theta: f64,
    #[serde(alias = "sigma", alias = "xi")]
    pub vol_of_vol: f64,
    pub rho: f64,
}

impl HestonParams {
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        // written as `!(finite && > 0)` so NaN fails too: a NaN v0 would
        // otherwise price to a finite wrong number through the
        // `max(1e-6)` floor in `with_vol_shift`
        let positive = |x: f64| x.is_finite() && x > 0.0;
        if !(positive(self.v0)
            && positive(self.theta)
            && positive(self.kappa)
            && positive(self.vol_of_vol))
        {
            return Err(RustyQLibError::invalid_input(
                "heston params",
                "Heston v0, kappa, theta, vol_of_vol must be positive and finite".to_string(),
            ));
        }
        if !(-1.0..=1.0).contains(&self.rho) {
            return Err(RustyQLibError::invalid_input(
                "heston params",
                "Heston rho must be in [-1, 1]".to_string(),
            ));
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
        HestonParams {
            v0: bump(self.v0),
            theta: bump(self.theta),
            ..*self
        }
    }
}

/// `ln` for the positive parameters and `atanh` for the correlation,
/// so any point the optimizer visits maps back to a valid parameter
/// set.
impl crate::equity::models::calibration::TransformSpace for HestonParams {
    fn to_unconstrained(&self) -> Vec<f64> {
        vec![
            self.v0.ln(),
            self.kappa.ln(),
            self.theta.ln(),
            self.vol_of_vol.ln(),
            // atanh, clamped away from the +-1 poles
            self.rho.clamp(-0.999, 0.999).atanh(),
        ]
    }

    fn from_unconstrained(u: &[f64]) -> HestonParams {
        HestonParams {
            v0: u[0].exp(),
            kappa: u[1].exp(),
            theta: u[2].exp(),
            vol_of_vol: u[3].exp(),
            rho: u[4].tanh(),
        }
    }
}

// ── Calibration ─────────────────────────────────────────────────────────

/// One vanilla market quote for calibration.
#[derive(Debug, Clone, Copy)]
pub struct HestonQuote {
    pub strike: f64,
    /// Year fraction to expiry.
    pub maturity: f64,
    /// Market price of the option.
    pub price: f64,
    pub put_or_call: PutOrCall,
}

/// Calibration outcome: fitted parameters plus fit diagnostics
/// (`rmse` in price units).
pub type HestonFit = crate::equity::models::calibration::Fit<HestonParams>;

/// Calibrate Heston parameters to European vanilla quotes.
///
/// A Levenberg-Marquardt least-squares fit (the calibration workhorse in
/// [`core::optimization`](crate::core::optimization)) on price residuals,
/// run in an unconstrained transform space (`ln` for the positive
/// parameters, `atanh` for rho) so every trial parameter set is valid —
/// the same pattern any parametric fit (SABR, Nelson-Siegel) should use.
/// `start` seeds the search; a poor start on a multimodal quote set can
/// be globalized first with
/// [`Method::DifferentialEvolution`](crate::core::optimization::Method).
///
/// Invalid input — an invalid `start`, no quotes, or a quote with a
/// non-positive / non-finite strike or maturity or a negative /
/// non-finite price — is a returned
/// [`invalid_input`](RustyQLibError::invalid_input) error.
pub fn calibrate(
    s: f64,
    r: f64,
    q: f64,
    quotes: &[HestonQuote],
    start: &HestonParams,
) -> Result<HestonFit, RustyQLibError> {
    start.validate()?;
    crate::equity::models::calibration::calibrate_generic(quotes, start, r, 1e-12, |p, u, t| {
        characteristic_fn(u, s, r, q, t, p)
    })
}

// ── Characteristic function and pricing ─────────────────────────────────

/// Characteristic function of ln(S_T) in the trap-free formulation.
pub(crate) fn characteristic_fn(u: Cpx, s: f64, r: f64, q: f64, t: f64, hp: &HestonParams) -> Cpx {
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
    let forward = s * ((r - q) * t).exp();
    probabilities_with_cf(&|u| characteristic_fn(u, s, r, q, t, hp), forward, k)
}

/// P1/P2 from any log-price characteristic function whose martingale
/// property gives `phi(-i) = forward` — shared by Heston and the Bates
/// jump-diffusion extensions.
pub(crate) fn probabilities_with_cf(cf: &dyn Fn(Cpx) -> Cpx, forward: f64, k: f64) -> (f64, f64) {
    let ln_k = k.ln();
    // integrands: Re[ e^{-iu lnK} phi_j(u) / (iu) ]
    let integrand = |u: f64, shifted: bool| -> f64 {
        let uc = Cpx::real(u);
        let phi = if shifted {
            // phi1(u) = phi(u - i) / phi(-i), phi(-i) = forward
            cf(uc.sub(I)).scale(1.0 / forward)
        } else {
            cf(uc)
        };
        let num = I.scale(-u * ln_k).exp().mul(phi);
        num.div(I.scale(u)).re
    };
    // integrate [0, 250] and extend block-wise while the tail still
    // contributes: short-dated / high vol-of-vol CFs decay slowly and a
    // fixed truncation silently loses ~1e-4 of probability mass
    let p = |shifted: bool| {
        let mut total = simpson(|u| integrand(u, shifted), 1e-9, 250.0, 4000);
        let mut lo = 250.0;
        while lo < 16_000.0 {
            let hi = lo * 2.0;
            let block = simpson(
                |u| integrand(u, shifted),
                lo,
                hi,
                (64.0 * (hi - lo)) as usize,
            );
            total += block;
            if block.abs() < 1e-12 {
                break;
            }
            lo = hi;
        }
        // integration noise on extreme parameter sets can push the
        // probability marginally outside [0, 1]; clamp so downstream
        // prices (binaries in particular) stay arbitrage-free
        (0.5 + total / std::f64::consts::PI).clamp(0.0, 1.0)
    };
    (p(true), p(false))
}

/// The input gate every analytic Heston route passes through: positive
/// spot, strike and expiry (a zero expiry makes the tail integration
/// crawl and a non-positive strike feeds `ln K` garbage into the
/// integrand), and valid parameters — panicking, as [`heston_price`]
/// always has, since these are programming errors at this level (the
/// option builder validates market data before it gets here).
fn check_inputs(s: f64, k: f64, t: f64, hp: &HestonParams) {
    assert!(
        s > 0.0 && k > 0.0 && t > 0.0,
        "Heston pricing needs positive spot, strike and expiry (got s={s}, k={k}, t={t})"
    );
    hp.validate().expect("invalid Heston parameters");
}

/// Price a whole strike strip in one COS pass: the characteristic
/// function is swept once for the expiry and every strike reuses it, so
/// a 20-strike smile costs about the same as one option. Agrees with
/// [`heston_price`] (the independent P1/P2 integration oracle) to ~1e-6.
/// Panics on non-positive spot, expiry or strikes, or invalid
/// parameters, exactly as [`heston_price`] does.
pub fn cos_smile(
    s: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    strikes: &[f64],
    put_or_call: crate::core::trade::PutOrCall,
) -> Vec<f64> {
    // the strip's smallest strike stands in for `k`: it is positive iff
    // every strike is (a NaN strike is kept, so it fails the gate too);
    // an empty strip still validates spot, expiry and parameters
    let k_min =
        strikes.iter().copied().fold(
            f64::INFINITY,
            |m, k| if k < m || k.is_nan() { k } else { m },
        );
    check_inputs(s, k_min, t, hp);
    let pricer = crate::equity::cos::CosPricer::new(
        &|u| characteristic_fn(u, s, r, q, t, hp),
        r,
        t,
        crate::equity::cos::DEFAULT_TERMS,
    );
    strikes
        .iter()
        .map(|&k| pricer.price(k, put_or_call))
        .collect()
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
    check_inputs(s, k, t, hp);
    let (p1, p2) = probabilities(s, k, r, q, t, hp);
    let call = s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2;
    match put_or_call {
        PutOrCall::Call => call,
        // put-call parity
        PutOrCall::Put => call - s * (-q * t).exp() + k * (-r * t).exp(),
    }
}

/// Semi-analytic Heston price of a cash-or-nothing binary
/// (`cash * e^{-rT} * P(S_T beyond K)`). Same input contract (and
/// panics) as [`heston_price`].
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
    check_inputs(s, k, t, hp);
    let (_, p2) = probabilities(s, k, r, q, t, hp);
    let df = (-r * t).exp();
    match put_or_call {
        PutOrCall::Call => cash * df * p2,
        PutOrCall::Put => cash * df * (1.0 - p2),
    }
}

/// Semi-analytic Heston price of an asset-or-nothing binary
/// (`S e^{-qT} P1` for a call). Same input contract (and panics) as
/// [`heston_price`].
pub fn heston_binary_asset_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    put_or_call: PutOrCall,
) -> f64 {
    check_inputs(s, k, t, hp);
    let (p1, _) = probabilities(s, k, r, q, t, hp);
    let leg = s * (-q * t).exp();
    match put_or_call {
        PutOrCall::Call => leg * p1,
        PutOrCall::Put => leg * (1.0 - p1),
    }
}

// ── Option-level analytic pricing and bump Greeks ───────────────────────

use crate::core::errors::RustyQLibError;
use crate::equity::bump::BumpedMarket;
use crate::equity::utils::PayoffType;
use crate::equity::vanilla_option::{BinaryPayoff, BinaryType, EquityOption};

/// Analytic Heston price through a market view; `None` prices the base
/// market. There is no implied surface to shift here, so the view's vol
/// bump is *interpreted*: it shifts sqrt(v0) and sqrt(theta) in parallel
/// (the model's analog of a parallel implied-vol move).
pub(crate) fn analytic_npv(option: &EquityOption, bumped_market: Option<&BumpedMarket>) -> f64 {
    let base = BumpedMarket::base(&option.market);
    let m = bumped_market.unwrap_or(&base);
    let maturity = option.base.maturity_date;
    let hp = option.heston_params().with_vol_shift(m.bump().d_vol);
    let s = m.effective_spot(maturity);
    let k = option.base.strike_price;
    let r = m.risk_free_rate(maturity);
    let q = m.carry_yield();
    let t = m.time_to_maturity(maturity);
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
        // `check_engine_support` refuses every other payoff on this
        // route before pricing (EquityOption::check_engine_support)
        _ => unreachable!(
            "check_engine_support admits only vanilla and binary payoffs on the Heston \
             analytic route"
        ),
    }
}

// Greeks: central-difference bumped-market reprices of [`analytic_npv`]
// (a vol bump shifts sqrt(v0) and sqrt(theta) in parallel), produced by
// the central sensitivity engine (`crate::equity::greeks`) — except
// vanilla delta, which falls out of the price integration for free below.

/// Vanilla delta directly from the characteristic-function integration:
/// the call price is `S e^{-qT} P1 - K e^{-rT} P2` and the homogeneity
/// identity cancels the `dP/dS` terms, so delta is `e^{-qT} P1` for calls
/// and `e^{-qT} (P1 - 1)` for puts — the probability the pricer already
/// integrates, no bump and no extra integration. `None` for non-vanilla
/// payoffs (binaries keep the central bump stencils).
pub(crate) fn native_vanilla_delta(option: &EquityOption) -> Option<f64> {
    if option.payoff.payoff_kind() != PayoffType::Vanilla {
        return None;
    }
    let hp = option.heston_params();
    let s = option.effective_spot();
    let k = option.base.strike_price;
    let r = option.risk_free_rate();
    let q = option.carry_yield();
    let t = option.time_to_maturity();
    let (p1, _) = probabilities(s, k, r, q, t, hp);
    let dfq = (-q * t).exp();
    Some(match option.payoff.put_or_call() {
        PutOrCall::Call => dfq * p1,
        PutOrCall::Put => dfq * (p1 - 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    fn params() -> HestonParams {
        HestonParams {
            v0: 0.09,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        }
    }

    #[test]
    fn calibration_recovers_the_generating_parameters() {
        // price a strike ladder from known parameters, perturb the start,
        // and calibrate back: the fit must reprice the quotes to sub-cent
        // accuracy and land near the generating v0 / vol_of_vol / rho
        let truth = HestonParams {
            v0: 0.04,
            kappa: 1.5,
            theta: 0.05,
            vol_of_vol: 0.5,
            rho: -0.7,
        };
        let (s, r, q, t) = (100.0, 0.03, 0.01, 1.0);
        let quotes: Vec<HestonQuote> = [80.0, 90.0, 100.0, 110.0, 120.0]
            .iter()
            .map(|&k| HestonQuote {
                strike: k,
                maturity: t,
                price: heston_price(s, k, r, q, t, &truth, PutOrCall::Call),
                put_or_call: PutOrCall::Call,
            })
            .collect();

        let start = HestonParams {
            v0: 0.06,
            kappa: 1.0,
            theta: 0.04,
            vol_of_vol: 0.3,
            rho: -0.3,
        };
        let fit = calibrate(s, r, q, &quotes, &start).unwrap();

        assert!(
            fit.rmse < 1e-3,
            "price rmse {} too large: {:?}",
            fit.rmse,
            fit.params
        );
        // the smile pins v0 (level), vol_of_vol (curvature) and rho (skew)
        assert!(
            (fit.params.v0 - truth.v0).abs() < 0.01,
            "v0 {}",
            fit.params.v0
        );
        assert!(
            (fit.params.rho - truth.rho).abs() < 0.1,
            "rho {}",
            fit.params.rho
        );
        assert!(fit.params.validate().is_ok());
    }

    #[test]
    fn calibration_rejects_bad_input_instead_of_panicking() {
        let (s, r, q, t) = (100.0, 0.03, 0.01, 1.0);
        let truth = params();
        let good: Vec<HestonQuote> = [90.0, 100.0, 110.0]
            .iter()
            .map(|&k| HestonQuote {
                strike: k,
                maturity: t,
                price: heston_price(s, k, r, q, t, &truth, PutOrCall::Call),
                put_or_call: PutOrCall::Call,
            })
            .collect();
        // no quotes
        assert!(calibrate(s, r, q, &[], &truth).is_err());
        // an invalid start
        let bad_start = HestonParams {
            v0: f64::NAN,
            ..truth
        };
        assert!(calibrate(s, r, q, &good, &bad_start).is_err());
        assert!(calibrate(s, r, q, &good, &HestonParams { rho: 1.5, ..truth }).is_err());
        // a poisoned quote
        for (strike, maturity, price) in [
            (0.0, t, 5.0),
            (f64::NAN, t, 5.0),
            (100.0, 0.0, 5.0),
            (100.0, f64::INFINITY, 5.0),
            (100.0, t, -1.0),
            (100.0, t, f64::NAN),
        ] {
            let mut quotes = good.clone();
            quotes[1] = HestonQuote {
                strike,
                maturity,
                price,
                put_or_call: PutOrCall::Call,
            };
            assert!(
                calibrate(s, r, q, &quotes, &truth).is_err(),
                "K={strike} t={maturity} price={price} should be refused"
            );
        }
        // and the untouched set still calibrates
        assert!(calibrate(s, r, q, &good, &truth).is_ok());
    }

    #[test]
    fn degenerates_to_black_scholes_when_vol_of_vol_vanishes() {
        // v0 = theta and vol_of_vol -> 0: variance is constant, so the
        // price must match Black-Scholes at sigma = sqrt(v0)
        let hp = HestonParams {
            v0: 0.09,
            kappa: 1.0,
            theta: 0.09,
            vol_of_vol: 1e-4,
            rho: 0.0,
        };
        for k in [80.0, 100.0, 120.0] {
            let heston = heston_price(100.0, k, 0.05, 0.02, 1.0, &hp, PutOrCall::Call);
            let bs = bs_price(100.0, k, 0.05, 0.02, 0.3, 1.0, PutOrCall::Call);
            assert!(
                (heston - bs).abs() < 1e-4,
                "K={k}: heston {heston} vs bs {bs}"
            );
        }
    }

    #[test]
    fn put_call_parity() {
        let hp = params();
        let (s, k, r, q, t) = (100.0, 95.0, 0.05, 0.02, 1.0);
        let c = heston_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let p = heston_price(s, k, r, q, t, &hp, PutOrCall::Put);
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
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
        let hp_zero = HestonParams {
            rho: 0.0,
            ..params()
        };
        let otm_put_neg = heston_price(100.0, 80.0, 0.05, 0.0, 1.0, &hp_neg, PutOrCall::Put);
        let otm_put_zero = heston_price(100.0, 80.0, 0.05, 0.0, 1.0, &hp_zero, PutOrCall::Put);
        assert!(otm_put_neg > otm_put_zero);
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(HestonParams {
            v0: -0.1,
            ..params()
        }
        .validate()
        .is_err());
        assert!(HestonParams {
            rho: -1.5,
            ..params()
        }
        .validate()
        .is_err());
        assert!(params().validate().is_ok());
        assert!(params().feller_condition_holds()); // 2*2*0.09 = 0.36 >= 0.4^2
        assert!(!HestonParams {
            vol_of_vol: 0.9,
            ..params()
        }
        .feller_condition_holds());
    }

    #[test]
    fn validation_rejects_nan_and_infinite_params() {
        // plain comparisons let NaN through; every positive parameter
        // and the correlation must refuse NaN and both infinities
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                HestonParams {
                    v0: bad,
                    ..params()
                }
                .validate()
                .is_err(),
                "v0 = {bad}"
            );
            assert!(
                HestonParams {
                    kappa: bad,
                    ..params()
                }
                .validate()
                .is_err(),
                "kappa = {bad}"
            );
            assert!(
                HestonParams {
                    theta: bad,
                    ..params()
                }
                .validate()
                .is_err(),
                "theta = {bad}"
            );
            assert!(
                HestonParams {
                    vol_of_vol: bad,
                    ..params()
                }
                .validate()
                .is_err(),
                "vol_of_vol = {bad}"
            );
            assert!(
                HestonParams {
                    rho: bad,
                    ..params()
                }
                .validate()
                .is_err(),
                "rho = {bad}"
            );
        }
        // the finding this closes: a NaN v0 used to price to a finite,
        // wrong number through with_vol_shift's max(1e-6) floor — the
        // shifted set is "valid", so the gate has to be on the input
        let nan_v0 = HestonParams {
            v0: f64::NAN,
            ..params()
        };
        assert!(nan_v0.with_vol_shift(0.0).validate().is_ok());
        assert!(nan_v0.validate().is_err());
    }

    #[test]
    fn analytic_routes_share_the_input_gate() {
        // every analytic entry point must refuse what heston_price
        // refuses: non-positive spot / strike / expiry, invalid params
        let hp = params();
        let bad_hp = HestonParams {
            v0: -0.1,
            ..params()
        };
        let cases: Vec<(&str, Box<dyn Fn() -> f64>)> = vec![
            (
                "cash binary, t = 0",
                Box::new(move || {
                    heston_binary_cash_price(
                        100.0,
                        100.0,
                        0.05,
                        0.0,
                        0.0,
                        &hp,
                        1.0,
                        PutOrCall::Call,
                    )
                }),
            ),
            (
                "cash binary, k <= 0",
                Box::new(move || {
                    heston_binary_cash_price(100.0, 0.0, 0.05, 0.0, 1.0, &hp, 1.0, PutOrCall::Call)
                }),
            ),
            (
                "asset binary, s <= 0",
                Box::new(move || {
                    heston_binary_asset_price(0.0, 100.0, 0.05, 0.0, 1.0, &hp, PutOrCall::Call)
                }),
            ),
            (
                "asset binary, bad params",
                Box::new(move || {
                    heston_binary_asset_price(100.0, 100.0, 0.05, 0.0, 1.0, &bad_hp, PutOrCall::Put)
                }),
            ),
            (
                "cos smile, negative strike in the strip",
                Box::new(move || {
                    cos_smile(
                        100.0,
                        0.05,
                        0.0,
                        1.0,
                        &hp,
                        &[90.0, -100.0, 110.0],
                        PutOrCall::Call,
                    )[0]
                }),
            ),
            (
                "cos smile, NaN strike in the strip",
                Box::new(move || {
                    cos_smile(
                        100.0,
                        0.05,
                        0.0,
                        1.0,
                        &hp,
                        &[90.0, f64::NAN],
                        PutOrCall::Call,
                    )[0]
                }),
            ),
            (
                "cos smile, t = 0",
                Box::new(move || {
                    cos_smile(100.0, 0.05, 0.0, 0.0, &hp, &[90.0, 110.0], PutOrCall::Call)[0]
                }),
            ),
            (
                "cos smile, bad params",
                Box::new(move || {
                    cos_smile(
                        100.0,
                        0.05,
                        0.0,
                        1.0,
                        &bad_hp,
                        &[90.0, 110.0],
                        PutOrCall::Call,
                    )[0]
                }),
            ),
        ];
        for (name, price) in cases {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(price));
            assert!(outcome.is_err(), "{name}: must panic like heston_price");
        }
        // valid input is untouched: the binaries still replicate the
        // vanilla and an empty strip prices to nothing
        let (s, k, r, q, t) = (100.0, 100.0, 0.05, 0.02, 1.0);
        let vanilla = heston_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let asset = heston_binary_asset_price(s, k, r, q, t, &hp, PutOrCall::Call);
        let cash = heston_binary_cash_price(s, k, r, q, t, &hp, k, PutOrCall::Call);
        assert!((vanilla - (asset - cash)).abs() < 1e-10);
        assert!(cos_smile(s, r, q, t, &hp, &[], PutOrCall::Call).is_empty());
    }
}
