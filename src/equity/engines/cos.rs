//! COS method: Fourier-cosine series pricing of European vanillas from a
//! characteristic function (Fang & Oosterlee, 2008).
//!
//! The risk-neutral density of `y = ln S_T` is unknown for Heston/Bates
//! models but its Fourier transform — the characteristic function `phi` —
//! is closed-form. The density's cosine-series coefficients on a
//! truncated interval `[a, b]` are values of `phi`, and the payoff's
//! cosine coefficients have closed forms, so the price collapses to a
//! dot product with **exponential** convergence in the number of terms:
//!
//! ```text
//! V ~ e^{-rT} sum_k' Re{ phi(u_k) e^{-i u_k a} } * V_k(K),   u_k = k pi/(b-a)
//! ```
//!
//! The `phi(u_k)` sweep depends only on the model and expiry — not the
//! strike — so one [`CosPricer`] prices an entire smile for the cost of
//! one option. That is what makes characteristic-function **calibration**
//! fast: the Levenberg-Marquardt objective revalues the whole quote grid
//! thousands of times, and with COS each revaluation is one CF sweep per
//! expiry instead of two 4000-point integrations per quote.
//!
//! The truncation range is set from the distribution's cumulants,
//! estimated numerically from `ln phi` near zero — so any model with a CF
//! (Heston, both Bates variants, GBM) works without per-model formulas.
//! The legacy P1/P2 integration ([`heston_price`]/[`bates_price`]) is kept
//! as the independent cross-check oracle in the tests.
//!
//! [`heston_price`]: crate::equity::heston::heston_price
//! [`bates_price`]: crate::equity::bates::bates_price

use std::f64::consts::PI;

use crate::core::trade::PutOrCall;
use crate::equity::heston::Cpx;

/// Series terms used by the calibration objectives: enough for ~1e-8
/// vanilla accuracy on market-typical parameters.
pub const CALIBRATION_TERMS: usize = 160;

/// Series terms for full-accuracy pricing and cross-checks. Exponential
/// convergence makes the sweep cheap; 2048 converges even the slowly
/// decaying CFs of high vol-of-vol (Feller-violated) parameter sets.
pub const DEFAULT_TERMS: usize = 2048;

/// A COS pricer for one (model, expiry): the CF sweep is done once at
/// construction, every strike after that is an `O(N)` dot product with
/// closed-form payoff coefficients.
pub(crate) struct CosPricer {
    a: f64,
    b: f64,
    /// `Re{ phi(u_k) e^{-i u_k a} }`, `k = 0..N`, `k = 0` term halved.
    weights: Vec<f64>,
    df: f64,
    /// Forward from the martingale property `phi(-i) = E[S_T]`.
    forward: f64,
}

impl CosPricer {
    /// Build from the characteristic function of `y = ln S_T` (drift
    /// included, as the Heston/Bates CFs are written).
    pub(crate) fn new(cf: &dyn Fn(Cpx) -> Cpx, r: f64, t: f64, n: usize) -> Self {
        // cumulants of y from psi(u) = ln phi(u):
        //   Im psi(h) =  c1 h + O(h^3),   Re psi(h) = -c2 h^2/2 + c4 h^4/24
        // two passes: a rough c2 fixes the step to the distribution scale,
        // then psi at h and 2h separate c2 from c4
        let psi = |h: f64| cf(Cpx::real(h)).ln();
        let rough = psi(1e-2);
        let c2_rough = (-2.0 * rough.re / 1e-4).abs().max(1e-10);
        let h = (0.05 / c2_rough.sqrt()).clamp(1e-4, 1e-1);
        let p1 = psi(h);
        let p2 = psi(2.0 * h);
        let c1 = p1.im / h;
        let c4 = (2.0 * (p2.re - 4.0 * p1.re) / h.powi(4)).max(0.0);
        let c2 = (-2.0 * p1.re / (h * h) + c4 * h * h / 12.0)
            .abs()
            .max(1e-10);

        // Fang-Oosterlee range with the kurtosis term: L = 10 covers the
        // fat tails of short-dated / high vol-of-vol Heston-class models
        let width = 10.0 * (c2 + c4.sqrt()).sqrt();
        let (a, b) = (c1 - width, c1 + width);

        let bma = b - a;
        let weights = (0..n)
            .map(|k| {
                let u = k as f64 * PI / bma;
                // Re{ phi(u) e^{-i u a} }
                let e = Cpx::new((u * a).cos(), -(u * a).sin());
                let w = cf(Cpx::real(u)).mul(e).re;
                if k == 0 {
                    0.5 * w
                } else {
                    w
                }
            })
            .collect();

        let forward = cf(Cpx::new(0.0, -1.0)).re;
        CosPricer {
            a,
            b,
            weights,
            df: (-r * t).exp(),
            forward,
        }
    }

    /// European call price at `strike`. The series is evaluated on the
    /// out-of-the-money side (small payoff coefficients, best relative
    /// precision) and in-the-money prices recover through put-call
    /// parity with the CF-implied forward.
    pub(crate) fn call(&self, strike: f64) -> f64 {
        if strike < self.forward {
            return self.raw_put(strike) + self.df * (self.forward - strike);
        }
        self.raw_call(strike)
    }

    /// European put price at `strike` (OTM series + parity, as for calls).
    pub(crate) fn put(&self, strike: f64) -> f64 {
        if strike > self.forward {
            return self.raw_call(strike) + self.df * (strike - self.forward);
        }
        self.raw_put(strike)
    }

    fn raw_call(&self, strike: f64) -> f64 {
        let lnk = strike.ln();
        if lnk >= self.b {
            return 0.0; // beyond the truncation range the tail mass is ~0
        }
        self.sum(strike, lnk.max(self.a), self.b, true)
    }

    fn raw_put(&self, strike: f64) -> f64 {
        let lnk = strike.ln();
        if lnk <= self.a {
            return 0.0;
        }
        self.sum(strike, self.a, lnk.min(self.b), false)
    }

    pub(crate) fn price(&self, strike: f64, put_or_call: PutOrCall) -> f64 {
        match put_or_call {
            PutOrCall::Call => self.call(strike),
            PutOrCall::Put => self.put(strike),
        }
    }

    /// `e^{-rT} sum_k' w_k V_k` with the closed-form cosine coefficients
    /// of the vanilla payoff over `[c, d]`:
    ///   chi_k = int e^y cos(k pi (y-a)/(b-a)) dy,
    ///   psi_k = int     cos(k pi (y-a)/(b-a)) dy.
    fn sum(&self, strike: f64, c: f64, d: f64, call: bool) -> f64 {
        let bma = self.b - self.a;
        let (ec, ed) = (c.exp(), d.exp());
        let (yc, yd) = (c - self.a, d - self.a);
        let mut total = 0.0;
        for (k, w) in self.weights.iter().enumerate() {
            let omega = k as f64 * PI / bma;
            let (sin_c, cos_c) = (omega * yc).sin_cos();
            let (sin_d, cos_d) = (omega * yd).sin_cos();
            let chi = (cos_d * ed - cos_c * ec + omega * (sin_d * ed - sin_c * ec))
                / (1.0 + omega * omega);
            let psi = if k == 0 {
                d - c
            } else {
                (sin_d - sin_c) / omega
            };
            let payoff_coeff = if call {
                chi - strike * psi
            } else {
                strike * psi - chi
            };
            total += w * payoff_coeff;
        }
        self.df * total * 2.0 / bma
    }
}

/// Group quote indices by (exact) maturity, preserving first-seen order —
/// the shape the calibration objectives iterate over so each expiry pays
/// for one CF sweep regardless of its strike count.
pub(crate) fn group_by_maturity(maturities: impl Iterator<Item = f64>) -> Vec<(f64, Vec<usize>)> {
    let mut groups: Vec<(f64, Vec<usize>)> = Vec::new();
    for (i, t) in maturities.enumerate() {
        match groups.iter_mut().find(|(gt, _)| *gt == t) {
            Some((_, idxs)) => idxs.push(i),
            None => groups.push((t, vec![i])),
        }
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::bates::{
        bates_double_exp_price, bates_price, BatesDoubleExpParams, BatesParams, KouJumps,
        MertonJumps,
    };
    use crate::equity::blackscholes::bs_price;
    use crate::equity::heston::{heston_price, HestonParams};

    const S: f64 = 100.0;
    const R: f64 = 0.03;
    const Q: f64 = 0.01;

    /// CF of ln S_T under GBM: the exact analytic benchmark.
    fn gbm_cf(u: Cpx, sigma: f64, t: f64) -> Cpx {
        let drift = S.ln() + (R - Q - 0.5 * sigma * sigma) * t;
        crate::equity::heston::I
            .mul(u)
            .scale(drift)
            .sub(u.mul(u).scale(0.5 * sigma * sigma * t))
            .exp()
    }

    fn heston() -> HestonParams {
        HestonParams {
            v0: 0.09,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        }
    }

    #[test]
    fn cos_reproduces_black_scholes() {
        for t in [0.1, 1.0, 3.0] {
            for sigma in [0.1, 0.3, 0.6] {
                let pricer = CosPricer::new(&|u| gbm_cf(u, sigma, t), R, t, DEFAULT_TERMS);
                for k in [60.0, 90.0, 100.0, 110.0, 150.0] {
                    let call = bs_price(S, k, R, Q, sigma, t, PutOrCall::Call);
                    let put = bs_price(S, k, R, Q, sigma, t, PutOrCall::Put);
                    assert!(
                        (pricer.call(k) - call).abs() < 1e-8,
                        "call K={k} t={t} sigma={sigma}: cos {} bs {call}",
                        pricer.call(k)
                    );
                    assert!(
                        (pricer.put(k) - put).abs() < 1e-8,
                        "put K={k} t={t} sigma={sigma}: cos {} bs {put}",
                        pricer.put(k)
                    );
                }
            }
        }
    }

    #[test]
    fn cos_agrees_with_the_heston_integration_oracle() {
        let params = [
            heston(),
            // high vol-of-vol, Feller violated: the stress case
            HestonParams {
                v0: 0.04,
                kappa: 1.0,
                theta: 0.04,
                vol_of_vol: 0.9,
                rho: -0.9,
            },
            HestonParams {
                v0: 0.16,
                kappa: 3.0,
                theta: 0.09,
                vol_of_vol: 0.2,
                rho: 0.3,
            },
        ];
        for hp in &params {
            for t in [0.25, 1.0, 2.0] {
                let pricer = CosPricer::new(
                    &|u| crate::equity::heston::characteristic_fn(u, S, R, Q, t, hp),
                    R,
                    t,
                    DEFAULT_TERMS,
                );
                for k in [70.0, 90.0, 100.0, 110.0, 140.0] {
                    let oracle = heston_price(S, k, R, Q, t, hp, PutOrCall::Call);
                    let cos = pricer.call(k);
                    assert!(
                        (cos - oracle).abs() < 5e-6,
                        "K={k} t={t} hp={hp:?}: cos {cos} oracle {oracle}"
                    );
                }
            }
        }
    }

    #[test]
    fn cos_agrees_with_both_bates_oracles() {
        let merton = BatesParams {
            heston: heston(),
            jumps: MertonJumps {
                intensity: 0.5,
                mean_jump: -0.1,
                jump_vol: 0.2,
            },
        };
        let kou = BatesDoubleExpParams {
            heston: heston(),
            jumps: KouJumps {
                intensity: 0.7,
                p_up: 0.4,
                eta_up: 12.0,
                eta_down: 8.0,
            },
        };
        let t = 1.0;
        let merton_pricer = CosPricer::new(
            &|u| crate::equity::bates::ln_price_cf(u, S, R, Q, t, &merton),
            R,
            t,
            DEFAULT_TERMS,
        );
        let kou_pricer = CosPricer::new(
            &|u| crate::equity::bates::ln_price_cf_double_exp(u, S, R, Q, t, &kou),
            R,
            t,
            DEFAULT_TERMS,
        );
        for k in [80.0, 100.0, 120.0] {
            let m_oracle = bates_price(S, k, R, Q, t, &merton, PutOrCall::Call);
            let k_oracle = bates_double_exp_price(S, k, R, Q, t, &kou, PutOrCall::Call);
            assert!(
                (merton_pricer.call(k) - m_oracle).abs() < 5e-6,
                "merton K={k}: cos {} oracle {m_oracle}",
                merton_pricer.call(k)
            );
            assert!(
                (kou_pricer.call(k) - k_oracle).abs() < 5e-6,
                "kou K={k}: cos {} oracle {k_oracle}",
                kou_pricer.call(k)
            );
        }
    }

    #[test]
    fn put_call_parity_holds_to_high_precision() {
        let t = 1.0;
        let hp = heston();
        let pricer = CosPricer::new(
            &|u| crate::equity::heston::characteristic_fn(u, S, R, Q, t, &hp),
            R,
            t,
            DEFAULT_TERMS,
        );
        let df = (-R * t).exp();
        let forward = S * ((R - Q) * t).exp();
        for k in [80.0, 100.0, 125.0] {
            let parity = pricer.call(k) - pricer.put(k) - df * (forward - k);
            assert!(parity.abs() < 1e-9, "parity violation {parity} at K={k}");
        }
    }

    #[test]
    fn maturity_grouping_preserves_indices() {
        let groups = group_by_maturity([1.0, 0.5, 1.0, 2.0, 0.5].into_iter());
        assert_eq!(
            groups,
            vec![(1.0, vec![0, 2]), (0.5, vec![1, 4]), (2.0, vec![3])]
        );
    }
}
