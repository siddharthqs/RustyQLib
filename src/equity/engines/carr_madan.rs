//! Carr-Madan FFT pricing: a whole log-strike grid of European vanilla
//! prices from one FFT of the characteristic function (Carr & Madan,
//! 1999).
//!
//! The call price is not integrable in log-strike `k = ln K` (it tends
//! to the spot as `k -> -inf`), so it is damped first: `c(k) = e^{alpha
//! k} C(k)` has the closed-form Fourier transform
//!
//! ```text
//! psi(v) = e^{-rT} phi(v - (alpha + 1) i)
//!          / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)
//! ```
//!
//! with `phi` the characteristic function of `ln S_T` (drift included,
//! the library's CF convention). Inverting on a uniform `v`-grid with
//! Simpson weights and evaluating all log-strikes at once through
//! [`fft`](crate::core::fft::fft) gives `N` prices on a uniform
//! log-strike grid for `N` CF evaluations plus one `O(N log N)`
//! transform; requested strikes are read off the grid by 4-point
//! Lagrange interpolation.
//!
//! Complements the COS engine ([`cos`](crate::equity::cos)): COS
//! converges exponentially per strike list and drives the calibration
//! objectives; Carr-Madan delivers a **dense strike grid in one shot**
//! (thousands of strikes for free), which is the natural tool for
//! building whole implied-vol surfaces from Heston / Bates parameters.
//! The damping requires `E[S_T^{alpha+1}]` to be finite — keep `alpha`
//! moderate (the default 1.25) for parameter sets near moment
//! explosion.

use std::f64::consts::PI;

use crate::core::fft::{fft, Complex};
use crate::core::trade::PutOrCall;
use crate::equity::bates::{BatesDoubleExpParams, BatesParams};
use crate::equity::heston::{Cpx, HestonParams};

/// Transform settings. The defaults refine the classic Carr-Madan
/// choices considerably — `n = 32768` frequency points at spacing
/// `eta = 0.05` — because each parameter guards a different error
/// source: the damped transform peaks at `v = 0` with a width of order
/// `alpha`, so `eta` must sit well below `alpha` (and below the decay
/// scale of slowly-decaying Feller-violated CFs), while the log-strike
/// spacing `2 pi / (n eta) ~ 0.38%` keeps the interpolation error
/// negligible even for short-dated, low-vol smiles. At ~30k CF
/// evaluations a full smile still builds in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CarrMadanConfig {
    /// Frequency grid points (a power of two keeps the FFT on the
    /// radix-2 path).
    pub n: usize,
    /// Frequency grid spacing; keep well below `alpha`.
    pub eta: f64,
    /// Damping exponent `alpha > 0`; needs `E[S_T^{alpha+1}] < inf`.
    pub alpha: f64,
}

impl Default for CarrMadanConfig {
    fn default() -> Self {
        CarrMadanConfig {
            n: 32_768,
            eta: 0.05,
            alpha: 1.25,
        }
    }
}

/// One (model, expiry) smile priced by the Carr-Madan transform: the
/// FFT is done at construction, every strike after that is an `O(1)`
/// interpolation on the log-strike grid.
pub struct CarrMadanSmile {
    /// First log-strike of the grid.
    k0: f64,
    /// Log-strike spacing `2 pi / (n eta)`.
    dk: f64,
    /// Discounted call prices on the grid.
    calls: Vec<f64>,
    df: f64,
    /// Forward from the martingale identity `phi(-i) = E[S_T]`.
    forward: f64,
}

impl CarrMadanSmile {
    /// Build from the characteristic function of `ln S_T` (drift
    /// included). The log-strike grid is centered on the CF-implied
    /// forward, so at-the-money sits mid-grid where the transform is
    /// most accurate.
    pub(crate) fn from_cf(
        cf: &dyn Fn(Cpx) -> Cpx,
        r: f64,
        t: f64,
        cfg: CarrMadanConfig,
    ) -> CarrMadanSmile {
        let (n, eta, alpha) = (cfg.n, cfg.eta, cfg.alpha);
        assert!(n >= 4 && n.is_multiple_of(2), "n must be even and >= 4");
        assert!(eta > 0.0 && alpha > 0.0);
        let df = (-r * t).exp();
        let forward = cf(Cpx::new(0.0, -1.0)).re;
        let dk = 2.0 * PI / (n as f64 * eta);
        let k0 = forward.ln() - 0.5 * n as f64 * dk;

        let mut x = vec![Complex::ZERO; n];
        for (j, slot) in x.iter_mut().enumerate() {
            let v = j as f64 * eta;
            // psi(v) = df * phi(v - (alpha+1) i) / (alpha^2 + alpha - v^2 + i (2 alpha + 1) v)
            let phi = cf(Cpx::new(v, -(alpha + 1.0)));
            let denom = Cpx::new(alpha * alpha + alpha - v * v, (2.0 * alpha + 1.0) * v);
            let psi = phi.div(denom).scale(df);
            // rotate onto the shifted strike grid and apply the Simpson
            // weight (1/3, 4/3, 2/3, 4/3, ...)
            let theta = -v * k0;
            let rot = Cpx::new(theta.cos(), theta.sin());
            let w = if j == 0 {
                1.0 / 3.0
            } else if j.is_multiple_of(2) {
                2.0 / 3.0
            } else {
                4.0 / 3.0
            };
            let val = psi.mul(rot).scale(eta * w);
            *slot = Complex::new(val.re, val.im);
        }
        fft(&mut x);

        let calls = x
            .iter()
            .enumerate()
            .map(|(u, xu)| {
                let k = k0 + u as f64 * dk;
                ((-alpha * k).exp() / PI * xu.re).max(0.0)
            })
            .collect();
        CarrMadanSmile {
            k0,
            dk,
            calls,
            df,
            forward,
        }
    }

    /// Smile under Heston dynamics.
    pub fn heston(
        s: f64,
        r: f64,
        q: f64,
        t: f64,
        hp: &HestonParams,
        cfg: CarrMadanConfig,
    ) -> CarrMadanSmile {
        Self::from_cf(
            &|u| crate::equity::heston::characteristic_fn(u, s, r, q, t, hp),
            r,
            t,
            cfg,
        )
    }

    /// Smile under Bates (Heston + lognormal Merton jumps) dynamics.
    pub fn bates(
        s: f64,
        r: f64,
        q: f64,
        t: f64,
        params: &BatesParams,
        cfg: CarrMadanConfig,
    ) -> CarrMadanSmile {
        Self::from_cf(
            &|u| crate::equity::bates::ln_price_cf(u, s, r, q, t, params),
            r,
            t,
            cfg,
        )
    }

    /// Smile under Bates with Kou double-exponential jumps.
    pub fn bates_double_exp(
        s: f64,
        r: f64,
        q: f64,
        t: f64,
        params: &BatesDoubleExpParams,
        cfg: CarrMadanConfig,
    ) -> CarrMadanSmile {
        Self::from_cf(
            &|u| crate::equity::bates::ln_price_cf_double_exp(u, s, r, q, t, params),
            r,
            t,
            cfg,
        )
    }

    /// The CF-implied forward `E[S_T]`.
    pub fn forward(&self) -> f64 {
        self.forward
    }

    /// European call price at `strike`: 4-point Lagrange interpolation
    /// of the transform's log-strike grid.
    pub fn call(&self, strike: f64) -> f64 {
        assert!(strike > 0.0);
        let x = (strike.ln() - self.k0) / self.dk;
        let i = (x.floor() as isize).clamp(1, self.calls.len() as isize - 3) as usize;
        let s = x - i as f64;
        let (p0, p1, p2, p3) = (
            self.calls[i - 1],
            self.calls[i],
            self.calls[i + 1],
            self.calls[i + 2],
        );
        let w0 = -s * (s - 1.0) * (s - 2.0) / 6.0;
        let w1 = (s + 1.0) * (s - 1.0) * (s - 2.0) / 2.0;
        let w2 = -(s + 1.0) * s * (s - 2.0) / 2.0;
        let w3 = (s + 1.0) * s * (s - 1.0) / 6.0;
        (w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3).max(0.0)
    }

    /// European put price by put-call parity with the CF-implied
    /// forward.
    pub fn put(&self, strike: f64) -> f64 {
        (self.call(strike) - self.df * (self.forward - strike)).max(0.0)
    }

    pub fn price(&self, strike: f64, put_or_call: PutOrCall) -> f64 {
        match put_or_call {
            PutOrCall::Call => self.call(strike),
            PutOrCall::Put => self.put(strike),
        }
    }

    /// Prices for a strike list — the whole smile off one transform,
    /// mirroring [`cos_smile`](crate::equity::heston::cos_smile).
    pub fn prices(&self, strikes: &[f64], put_or_call: PutOrCall) -> Vec<f64> {
        strikes
            .iter()
            .map(|&k| self.price(k, put_or_call))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::bates::{
        bates_double_exp_price, bates_price, KouJumps, MertonJumps,
    };
    use crate::equity::blackscholes::bs_price;
    use crate::equity::heston::heston_price;

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
    fn carr_madan_reproduces_black_scholes() {
        for t in [0.1, 1.0, 3.0] {
            for sigma in [0.1, 0.3, 0.6] {
                let smile = CarrMadanSmile::from_cf(
                    &|u| gbm_cf(u, sigma, t),
                    R,
                    t,
                    CarrMadanConfig::default(),
                );
                for k in [60.0, 90.0, 100.0, 110.0, 150.0] {
                    let call = bs_price(S, k, R, Q, sigma, t, PutOrCall::Call);
                    let put = bs_price(S, k, R, Q, sigma, t, PutOrCall::Put);
                    assert!(
                        (smile.call(k) - call).abs() < 1e-5,
                        "call K={k} t={t} sigma={sigma}: fft {} bs {call}",
                        smile.call(k)
                    );
                    assert!(
                        (smile.put(k) - put).abs() < 1e-5,
                        "put K={k} t={t} sigma={sigma}: fft {} bs {put}",
                        smile.put(k)
                    );
                }
            }
        }
    }

    #[test]
    fn carr_madan_agrees_with_the_heston_integration_oracle() {
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
                let smile = CarrMadanSmile::heston(S, R, Q, t, hp, CarrMadanConfig::default());
                for k in [70.0, 90.0, 100.0, 110.0, 140.0] {
                    let oracle = heston_price(S, k, R, Q, t, hp, PutOrCall::Call);
                    let got = smile.call(k);
                    assert!(
                        (got - oracle).abs() < 5e-5,
                        "K={k} t={t} hp={hp:?}: fft {got} oracle {oracle}"
                    );
                }
            }
        }
    }

    #[test]
    fn carr_madan_agrees_with_both_bates_oracles() {
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
        let m_smile = CarrMadanSmile::bates(S, R, Q, t, &merton, CarrMadanConfig::default());
        let k_smile =
            CarrMadanSmile::bates_double_exp(S, R, Q, t, &kou, CarrMadanConfig::default());
        for k in [80.0, 100.0, 120.0] {
            let m_oracle = bates_price(S, k, R, Q, t, &merton, PutOrCall::Call);
            let k_oracle = bates_double_exp_price(S, k, R, Q, t, &kou, PutOrCall::Call);
            assert!(
                (m_smile.call(k) - m_oracle).abs() < 5e-5,
                "merton K={k}: fft {} oracle {m_oracle}",
                m_smile.call(k)
            );
            assert!(
                (k_smile.call(k) - k_oracle).abs() < 5e-5,
                "kou K={k}: fft {} oracle {k_oracle}",
                k_smile.call(k)
            );
        }
    }

    #[test]
    fn cf_forward_matches_the_analytic_forward() {
        let t = 1.5;
        let smile = CarrMadanSmile::heston(S, R, Q, t, &heston(), CarrMadanConfig::default());
        let want = S * ((R - Q) * t).exp();
        assert!(
            (smile.forward() - want).abs() < 1e-8 * want,
            "{} vs {want}",
            smile.forward()
        );
    }

    #[test]
    fn damping_choice_does_not_move_the_price() {
        // the price is alpha-independent in exact arithmetic; any
        // material drift across dampings would flag a transform bug
        let t = 1.0;
        let hp = heston();
        let oracle = heston_price(S, 110.0, R, Q, t, &hp, PutOrCall::Call);
        for alpha in [0.75, 1.25, 1.75] {
            let smile = CarrMadanSmile::heston(
                S,
                R,
                Q,
                t,
                &hp,
                CarrMadanConfig {
                    alpha,
                    ..Default::default()
                },
            );
            assert!(
                (smile.call(110.0) - oracle).abs() < 5e-5,
                "alpha={alpha}: {} vs {oracle}",
                smile.call(110.0)
            );
        }
    }

    #[test]
    fn dense_smile_agrees_with_the_cos_engine() {
        // two independent Fourier methods, one dense strike ladder:
        // Carr-Madan's grid against per-strike COS series pricing
        let t = 1.0;
        let hp = heston();
        let smile = CarrMadanSmile::heston(S, R, Q, t, &hp, CarrMadanConfig::default());
        let strikes: Vec<f64> = (0..30).map(|i| 70.0 + 3.0 * i as f64).collect();
        let cm = smile.prices(&strikes, PutOrCall::Call);
        let cos = crate::equity::heston::cos_smile(S, R, Q, t, &hp, &strikes, PutOrCall::Call);
        for ((k, a), b) in strikes.iter().zip(&cm).zip(&cos) {
            assert!((a - b).abs() < 5e-5, "K={k}: carr-madan {a} cos {b}");
        }
    }
}
