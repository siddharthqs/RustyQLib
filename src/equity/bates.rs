//! Bates stochastic-volatility jump-diffusion models: Heston dynamics
//! plus a compound-Poisson jump in the log-price.
//!
//! - **Bates (1996)**, `BatesParams`: lognormal (Merton) jump sizes —
//!   the classic SVJ model, adding the short-dated skew and smile that
//!   pure Heston cannot produce;
//! - **Bates double-exponential**, `BatesDoubleExpParams`: Kou (2002)
//!   asymmetric double-exponential jump sizes — separate up/down tail
//!   decay rates, giving independent control of the two wings.
//!
//! Both price semi-analytically through the characteristic function:
//! the log-price CF is the Heston CF **times** an independent jump
//! factor `exp(lambda t (E[e^{iuY}] - 1) - iu lambda t kbar)` with
//! `kbar = E[e^Y] - 1` the martingale compensator, so the P1/P2
//! machinery of [`heston`](crate::equity::heston) is reused unchanged
//! (the compensator keeps `phi(-i) = forward` exactly). With
//! `intensity = 0` both models collapse to Heston to machine precision
//! (tested); with the vol-of-vol collapsed they reduce to Merton / Kou
//! jump-diffusion, which the tests verify against independent oracles.

use serde::{Deserialize, Serialize};

use crate::core::trade::PutOrCall;
use crate::equity::heston::{characteristic_fn, probabilities_with_cf, Cpx, HestonParams, I};

// ── Jump specifications ─────────────────────────────────────────────────

/// Lognormal (Merton) jumps: `ln(1 + J) ~ N(ln(1 + mean_jump) -
/// jump_vol^2/2, jump_vol^2)`, arriving at `intensity` per year.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct MertonJumps {
    /// Expected number of jumps per year (`lambda >= 0`).
    pub intensity: f64,
    /// Mean relative jump size `E[J] = E[e^Y] - 1 > -1` (negative =
    /// downward jumps).
    pub mean_jump: f64,
    /// Volatility of the log jump size (`> 0`).
    pub jump_vol: f64,
}

impl MertonJumps {
    pub fn validate(&self) -> Result<(), String> {
        if self.intensity < 0.0 {
            return Err("jump intensity must be non-negative".into());
        }
        if self.mean_jump <= -1.0 {
            return Err("mean jump must be greater than -100%".into());
        }
        if self.jump_vol <= 0.0 {
            return Err("jump vol must be positive".into());
        }
        Ok(())
    }

    /// Martingale compensator `kbar = E[e^Y] - 1`.
    fn kbar(&self) -> f64 {
        self.mean_jump
    }

    /// `E[e^{iuY}]` for complex `u`: `exp(iu nu - u^2 delta^2 / 2)`.
    fn cf(&self, u: Cpx) -> Cpx {
        let nu = (1.0 + self.mean_jump).ln() - 0.5 * self.jump_vol * self.jump_vol;
        I.mul(u)
            .scale(nu)
            .sub(u.mul(u).scale(0.5 * self.jump_vol * self.jump_vol))
            .exp()
    }
}

/// Kou (2002) double-exponential jumps: upward moves with probability
/// `p_up` and decay `eta_up`, downward with decay `eta_down` —
/// independent control of the two smile wings.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct KouJumps {
    /// Expected number of jumps per year (`lambda >= 0`).
    pub intensity: f64,
    /// Probability a jump is upward (`0..=1`).
    pub p_up: f64,
    /// Upward tail decay (`> 1` so the compensator is finite; the mean
    /// up-jump in log space is `1/eta_up`).
    pub eta_up: f64,
    /// Downward tail decay (`> 0`; mean down-jump `1/eta_down`).
    pub eta_down: f64,
}

impl KouJumps {
    pub fn validate(&self) -> Result<(), String> {
        if self.intensity < 0.0 {
            return Err("jump intensity must be non-negative".into());
        }
        if !(0.0..=1.0).contains(&self.p_up) {
            return Err("p_up must be in [0, 1]".into());
        }
        if self.eta_up <= 1.0 {
            return Err("eta_up must exceed 1 (finite expected up-jump)".into());
        }
        if self.eta_down <= 0.0 {
            return Err("eta_down must be positive".into());
        }
        Ok(())
    }

    /// `kbar = E[e^Y] - 1 = p eta1/(eta1 - 1) + (1-p) eta2/(eta2 + 1) - 1`.
    fn kbar(&self) -> f64 {
        self.p_up * self.eta_up / (self.eta_up - 1.0)
            + (1.0 - self.p_up) * self.eta_down / (self.eta_down + 1.0)
            - 1.0
    }

    /// `E[e^{iuY}] = p eta1/(eta1 - iu) + (1-p) eta2/(eta2 + iu)`.
    fn cf(&self, u: Cpx) -> Cpx {
        let iu = I.mul(u);
        let up = Cpx::real(self.eta_up).div(Cpx::real(self.eta_up).sub(iu)).scale(self.p_up);
        let down = Cpx::real(self.eta_down)
            .div(Cpx::real(self.eta_down).add(iu))
            .scale(1.0 - self.p_up);
        up.add(down)
    }
}

// ── Models ──────────────────────────────────────────────────────────────

/// Bates (1996): Heston diffusion plus lognormal jumps.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct BatesParams {
    pub heston: HestonParams,
    pub jumps: MertonJumps,
}

/// Heston diffusion plus Kou double-exponential jumps.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct BatesDoubleExpParams {
    pub heston: HestonParams,
    pub jumps: KouJumps,
}

impl BatesParams {
    pub fn validate(&self) -> Result<(), String> {
        self.heston.validate()?;
        self.jumps.validate()
    }
}

impl BatesDoubleExpParams {
    pub fn validate(&self) -> Result<(), String> {
        self.heston.validate()?;
        self.jumps.validate()
    }
}

/// The compensated compound-Poisson factor
/// `exp(lambda t (cf_jump(u) - 1) - iu lambda t kbar)`.
fn jump_factor(u: Cpx, t: f64, intensity: f64, kbar: f64, jump_cf: Cpx) -> Cpx {
    let one = Cpx::real(1.0);
    jump_cf
        .sub(one)
        .scale(intensity * t)
        .sub(I.mul(u).scale(intensity * t * kbar))
        .exp()
}

#[allow(clippy::too_many_arguments)]
fn price_with_jumps(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    hp: &HestonParams,
    intensity: f64,
    kbar: f64,
    jump_cf: &dyn Fn(Cpx) -> Cpx,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0 && t > 0.0);
    let cf = |u: Cpx| -> Cpx {
        characteristic_fn(u, s, r, q, t, hp)
            .mul(jump_factor(u, t, intensity, kbar, jump_cf(u)))
    };
    let forward = s * ((r - q) * t).exp();
    let (p1, p2) = probabilities_with_cf(&cf, forward, k);
    let call = s * (-q * t).exp() * p1 - k * (-r * t).exp() * p2;
    match put_or_call {
        PutOrCall::Call => call,
        PutOrCall::Put => call - s * (-q * t).exp() + k * (-r * t).exp(),
    }
}

/// Semi-analytic Bates (SVJ) price of a European vanilla option.
pub fn bates_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    params: &BatesParams,
    put_or_call: PutOrCall,
) -> f64 {
    params.validate().expect("invalid Bates parameters");
    let jumps = params.jumps;
    price_with_jumps(
        s, k, r, q, t,
        &params.heston,
        jumps.intensity,
        jumps.kbar(),
        &|u| jumps.cf(u),
        put_or_call,
    )
}

/// Semi-analytic Bates double-exponential (Heston + Kou jumps) price of
/// a European vanilla option.
pub fn bates_double_exp_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    params: &BatesDoubleExpParams,
    put_or_call: PutOrCall,
) -> f64 {
    params.validate().expect("invalid Bates double-exponential parameters");
    let jumps = params.jumps;
    price_with_jumps(
        s, k, r, q, t,
        &params.heston,
        jumps.intensity,
        jumps.kbar(),
        &|u| jumps.cf(u),
        put_or_call,
    )
}

// ── Calibration ─────────────────────────────────────────────────────────

pub use crate::equity::heston::HestonQuote;

/// Calibration outcome for the lognormal-jump Bates model.
#[derive(Debug, Clone)]
pub struct BatesFit {
    pub params: BatesParams,
    /// Root-mean-square price error over the quotes.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Calibration outcome for the double-exponential-jump Bates model.
#[derive(Debug, Clone)]
pub struct BatesDoubleExpFit {
    pub params: BatesDoubleExpParams,
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl BatesParams {
    /// Unconstrained space: Heston's five transforms plus
    /// `[ln lambda, ln(1 + mean_jump), ln jump_vol]`.
    fn to_unconstrained(&self) -> Vec<f64> {
        let mut u = self.heston.to_unconstrained();
        u.push(self.jumps.intensity.max(1e-8).ln());
        u.push((1.0 + self.jumps.mean_jump).ln());
        u.push(self.jumps.jump_vol.ln());
        u
    }

    fn from_unconstrained(u: &[f64]) -> BatesParams {
        BatesParams {
            heston: HestonParams::from_unconstrained(&u[..5]),
            jumps: MertonJumps {
                intensity: u[5].exp(),
                mean_jump: u[6].exp() - 1.0,
                jump_vol: u[7].exp(),
            },
        }
    }
}

impl BatesDoubleExpParams {
    /// Unconstrained space: Heston's five transforms plus
    /// `[ln lambda, logit p_up, ln(eta_up - 1), ln eta_down]`.
    fn to_unconstrained(&self) -> Vec<f64> {
        let p = self.jumps.p_up.clamp(1e-6, 1.0 - 1e-6);
        let mut u = self.heston.to_unconstrained();
        u.push(self.jumps.intensity.max(1e-8).ln());
        u.push((p / (1.0 - p)).ln());
        u.push((self.jumps.eta_up - 1.0).max(1e-8).ln());
        u.push(self.jumps.eta_down.ln());
        u
    }

    fn from_unconstrained(u: &[f64]) -> BatesDoubleExpParams {
        BatesDoubleExpParams {
            heston: HestonParams::from_unconstrained(&u[..5]),
            jumps: KouJumps {
                intensity: u[5].exp(),
                p_up: 1.0 / (1.0 + (-u[6]).exp()),
                eta_up: 1.0 + u[7].exp(),
                eta_down: u[8].exp(),
            },
        }
    }
}

fn calibrate_generic<P>(
    quotes: &[HestonQuote],
    x0: Vec<f64>,
    unpack: impl Fn(&[f64]) -> P,
    price: impl Fn(&P, f64, f64, crate::core::trade::PutOrCall) -> f64,
) -> (P, f64, usize, bool) {
    use crate::core::optimization::{levenberg_marquardt, OptimConfig};
    assert!(!quotes.is_empty(), "calibration needs at least one quote");
    let residuals = |u: &[f64]| -> Vec<f64> {
        let p = unpack(u);
        quotes
            .iter()
            .map(|q| price(&p, q.strike, q.maturity, q.put_or_call) - q.price)
            .collect()
    };
    let fit = levenberg_marquardt(&OptimConfig::new(1e-10, 100), &residuals, None, &x0);
    let params = unpack(&fit.x);
    let rmse = (fit.value / quotes.len() as f64).sqrt();
    (params, rmse, fit.iterations, fit.converged)
}

/// Calibrate all eight Bates parameters to European vanilla quotes —
/// the same Levenberg-Marquardt-in-transform-space pattern as
/// [`heston::calibrate`](crate::equity::heston::calibrate). Short-dated
/// quotes are what identify the jump parameters against the diffusion.
pub fn calibrate(
    s: f64,
    r: f64,
    q: f64,
    quotes: &[HestonQuote],
    start: &BatesParams,
) -> BatesFit {
    start.validate().expect("invalid starting parameters");
    let (params, rmse, iterations, converged) = calibrate_generic(
        quotes,
        start.to_unconstrained(),
        BatesParams::from_unconstrained,
        |p, k, t, pc| bates_price(s, k, r, q, t, p, pc),
    );
    BatesFit { params, rmse, iterations, converged }
}

/// Calibrate all nine double-exponential Bates parameters to European
/// vanilla quotes.
pub fn calibrate_double_exp(
    s: f64,
    r: f64,
    q: f64,
    quotes: &[HestonQuote],
    start: &BatesDoubleExpParams,
) -> BatesDoubleExpFit {
    start.validate().expect("invalid starting parameters");
    let (params, rmse, iterations, converged) = calibrate_generic(
        quotes,
        start.to_unconstrained(),
        BatesDoubleExpParams::from_unconstrained,
        |p, k, t, pc| bates_double_exp_price(s, k, r, q, t, p, pc),
    );
    BatesDoubleExpFit { params, rmse, iterations, converged }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;
    use crate::equity::heston::heston_price;

    const S: f64 = 100.0;
    const K: f64 = 100.0;
    const R: f64 = 0.03;
    const Q: f64 = 0.01;
    const T: f64 = 1.0;

    fn heston() -> HestonParams {
        HestonParams { v0: 0.04, kappa: 1.5, theta: 0.05, vol_of_vol: 0.5, rho: -0.7 }
    }

    fn bates() -> BatesParams {
        BatesParams {
            heston: heston(),
            jumps: MertonJumps { intensity: 0.6, mean_jump: -0.08, jump_vol: 0.15 },
        }
    }

    fn kou() -> BatesDoubleExpParams {
        BatesDoubleExpParams {
            heston: heston(),
            jumps: KouJumps { intensity: 0.6, p_up: 0.35, eta_up: 20.0, eta_down: 12.0 },
        }
    }

    #[test]
    fn zero_intensity_collapses_to_heston() {
        let mut b = bates();
        b.jumps.intensity = 0.0;
        let mut d = kou();
        d.jumps.intensity = 0.0;
        for k in [80.0, 100.0, 120.0] {
            let h = heston_price(S, k, R, Q, T, &heston(), PutOrCall::Call);
            let bp = bates_price(S, k, R, Q, T, &b, PutOrCall::Call);
            let dp = bates_double_exp_price(S, k, R, Q, T, &d, PutOrCall::Call);
            assert!((bp - h).abs() < 1e-10, "bates k={k}: {bp} vs {h}");
            assert!((dp - h).abs() < 1e-10, "kou k={k}: {dp} vs {h}");
        }
    }

    #[test]
    fn put_call_parity_holds_for_both_models() {
        let parity = S * (-Q * T).exp() - K * (-R * T).exp();
        let b = bates();
        let c = bates_price(S, K, R, Q, T, &b, PutOrCall::Call);
        let p = bates_price(S, K, R, Q, T, &b, PutOrCall::Put);
        assert!((c - p - parity).abs() < 1e-8, "bates parity {}", c - p - parity);
        let d = kou();
        let c2 = bates_double_exp_price(S, K, R, Q, T, &d, PutOrCall::Call);
        let p2 = bates_double_exp_price(S, K, R, Q, T, &d, PutOrCall::Put);
        assert!((c2 - p2 - parity).abs() < 1e-8, "kou parity {}", c2 - p2 - parity);
    }

    #[test]
    fn merton_limit_matches_the_independent_series_solution() {
        // collapse the vol-of-vol: Bates -> Merton jump-diffusion, which
        // has the classic Poisson-weighted Black-Scholes series
        let sigma = 0.2_f64;
        let flat = HestonParams {
            v0: sigma * sigma,
            kappa: 1.0,
            theta: sigma * sigma,
            vol_of_vol: 1e-4,
            rho: 0.0,
        };
        let jumps = MertonJumps { intensity: 0.8, mean_jump: -0.1, jump_vol: 0.2 };
        let b = BatesParams { heston: flat, jumps };

        let merton_series = |k: f64, pc: PutOrCall| -> f64 {
            let lam_bar = jumps.intensity * (1.0 + jumps.mean_jump);
            let mut price = 0.0;
            for n in 0..60 {
                let nf = n as f64;
                let weight =
                    (-lam_bar * T).exp() * (lam_bar * T).powi(n) / (1..=n).map(|i| i as f64).product::<f64>().max(1.0);
                let sigma_n =
                    (sigma * sigma + nf * jumps.jump_vol * jumps.jump_vol / T).sqrt();
                let r_n = R - jumps.intensity * jumps.mean_jump
                    + nf * (1.0 + jumps.mean_jump).ln() / T;
                price += weight * bs_price(S, k, r_n, Q, sigma_n, T, pc);
            }
            price
        };
        for k in [85.0, 100.0, 115.0] {
            let via_cf = bates_price(S, k, R, Q, T, &b, PutOrCall::Call);
            let via_series = merton_series(k, PutOrCall::Call);
            assert!(
                (via_cf - via_series).abs() < 2e-3,
                "k = {k}: cf {via_cf} vs series {via_series}"
            );
        }
    }

    #[test]
    fn kou_limit_matches_a_seeded_monte_carlo_oracle() {
        // collapse the vol-of-vol: the double-exp model becomes Kou
        // jump-diffusion, simulated directly and deterministically
        use crate::core::montecarlo::path_rng;
        use rand::Rng;

        let sigma = 0.2_f64;
        let flat = HestonParams {
            v0: sigma * sigma,
            kappa: 1.0,
            theta: sigma * sigma,
            vol_of_vol: 1e-4,
            rho: 0.0,
        };
        let jumps = KouJumps { intensity: 0.7, p_up: 0.3, eta_up: 25.0, eta_down: 10.0 };
        let d = BatesDoubleExpParams { heston: flat, jumps };

        let n_paths = 400_000;
        let kbar = jumps.kbar();
        let drift = (R - Q - 0.5 * sigma * sigma - jumps.intensity * kbar) * T;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let strike = 95.0;
        for path in 0..n_paths {
            let mut rng = path_rng(20260724, path);
            let z: f64 = rng.sample(rand_distr::StandardNormal);
            // Poisson(lambda T) by exponential inter-arrival products
            let mut jumps_sum = 0.0;
            let threshold = (-jumps.intensity * T).exp();
            let mut product: f64 = rng.gen();
            while product > threshold {
                let up: f64 = rng.gen();
                let e: f64 = -(rng.gen::<f64>().max(1e-300)).ln();
                jumps_sum += if up < jumps.p_up { e / jumps.eta_up } else { -e / jumps.eta_down };
                product *= rng.gen::<f64>();
            }
            let s_t = S * (drift + sigma * T.sqrt() * z + jumps_sum).exp();
            let payoff = (s_t - strike).max(0.0) * (-R * T).exp();
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        let mc = sum / n_paths as f64;
        let se = ((sum_sq / n_paths as f64 - mc * mc) / n_paths as f64).sqrt();
        let via_cf = bates_double_exp_price(S, strike, R, Q, T, &d, PutOrCall::Call);
        assert!(
            (via_cf - mc).abs() < (3.0 * se).max(0.02),
            "cf {via_cf} vs mc {mc} +/- {se}"
        );
    }

    #[test]
    fn jumps_raise_option_values_and_shape_the_skew() {
        let base = heston_price(S, K, R, Q, T, &heston(), PutOrCall::Call);
        let with_jumps = bates_price(S, K, R, Q, T, &bates(), PutOrCall::Call);
        assert!(with_jumps > base, "{with_jumps} vs {base}");
        // jump direction moves the wings the right way: down-biased jumps
        // price OTM puts above what up-biased jumps do, and vice versa
        // for OTM calls (same intensity and jump vol, so a clean contrast)
        let with_mean = |mean: f64| BatesParams {
            heston: heston(),
            jumps: MertonJumps { intensity: 0.6, mean_jump: mean, jump_vol: 0.15 },
        };
        let put_down = bates_price(S, 85.0, R, Q, T, &with_mean(-0.08), PutOrCall::Put);
        let put_up = bates_price(S, 85.0, R, Q, T, &with_mean(0.08), PutOrCall::Put);
        assert!(put_down > put_up, "put: down {put_down} vs up {put_up}");
        let call_up = bates_price(S, 115.0, R, Q, T, &with_mean(0.08), PutOrCall::Call);
        let call_down = bates_price(S, 115.0, R, Q, T, &with_mean(-0.08), PutOrCall::Call);
        assert!(call_up > call_down, "call: up {call_up} vs down {call_down}");
        // the same asymmetry holds for Kou jumps through p_up
        let kou_with = |p_up: f64| BatesDoubleExpParams {
            heston: heston(),
            jumps: KouJumps { intensity: 0.6, p_up, eta_up: 20.0, eta_down: 12.0 },
        };
        let kp_down = bates_double_exp_price(S, 85.0, R, Q, T, &kou_with(0.1), PutOrCall::Put);
        let kp_up = bates_double_exp_price(S, 85.0, R, Q, T, &kou_with(0.9), PutOrCall::Put);
        assert!(kp_down > kp_up, "kou put: down-heavy {kp_down} vs up-heavy {kp_up}");
    }

    #[test]
    fn calibration_recovers_bates_prices() {
        // quotes across two expiries (the short one identifies the jumps);
        // perturbed start; the fit must reprice to sub-cent accuracy
        let truth = bates();
        let mut quotes = Vec::new();
        for (t, strikes) in [(0.25_f64, [90.0, 100.0, 110.0]), (1.0, [85.0, 100.0, 115.0])] {
            for k in strikes {
                quotes.push(HestonQuote {
                    strike: k,
                    maturity: t,
                    price: bates_price(S, k, R, Q, t, &truth, PutOrCall::Call),
                    put_or_call: PutOrCall::Call,
                });
            }
        }
        let start = BatesParams {
            heston: HestonParams { v0: 0.05, kappa: 1.5, theta: 0.04, vol_of_vol: 0.4, rho: -0.5 },
            jumps: MertonJumps { intensity: 0.4, mean_jump: -0.04, jump_vol: 0.2 },
        };
        let fit = calibrate(S, R, Q, &quotes, &start);
        assert!(fit.rmse < 1e-3, "price rmse {} params {:?}", fit.rmse, fit.params);
        assert!(fit.params.validate().is_ok());
    }

    #[test]
    fn calibration_recovers_double_exp_prices() {
        let truth = kou();
        let quotes: Vec<HestonQuote> = [85.0, 95.0, 100.0, 105.0, 115.0]
            .iter()
            .map(|&k| HestonQuote {
                strike: k,
                maturity: 0.5,
                price: bates_double_exp_price(S, k, R, Q, 0.5, &truth, PutOrCall::Call),
                put_or_call: PutOrCall::Call,
            })
            .collect();
        let start = BatesDoubleExpParams {
            heston: HestonParams { v0: 0.05, kappa: 1.5, theta: 0.04, vol_of_vol: 0.4, rho: -0.5 },
            jumps: KouJumps { intensity: 0.4, p_up: 0.5, eta_up: 15.0, eta_down: 15.0 },
        };
        let fit = calibrate_double_exp(S, R, Q, &quotes, &start);
        assert!(fit.rmse < 1e-3, "price rmse {} params {:?}", fit.rmse, fit.params);
        assert!(fit.params.validate().is_ok());
    }

    #[test]
    fn parameter_validation_rejects_bad_inputs() {
        let mut b = bates();
        b.jumps.mean_jump = -1.5;
        assert!(b.validate().is_err());
        let mut d = kou();
        d.jumps.eta_up = 0.9; // infinite expected up-jump
        assert!(d.validate().is_err());
        d = kou();
        d.jumps.p_up = 1.4;
        assert!(d.validate().is_err());
    }
}
