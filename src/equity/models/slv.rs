//! Stochastic Local Volatility (SLV): Heston-style stochastic variance
//! multiplied by a **leverage function** calibrated so the model
//! reprices the market's vanilla surface exactly (in the limit):
//!
//! ```text
//! dS/S = (r - q) dt + L(S, t) sqrt(v) dW1
//! dv   = kappa (theta - v) dt + xi sqrt(v) dW2,   d<W1, W2> = rho dt
//! ```
//!
//! By Gyongy's theorem the model matches the market when
//! `L^2(S, t) = sigma_LV^2(S, t) / E[v_t | S_t = S]`, with `sigma_LV`
//! the Dupire local vol. The conditional expectation is estimated by
//! the standard particle / binning method: simulate forward, bin paths
//! by spot at each step, average the variance per bin, and use the
//! resulting leverage for the next step. Both the calibration and the
//! pricing simulations step the variance with the Andersen QE transition
//! and the spot through the Broadie-Kaya decomposition (see
//! [`SlvStepper`]), so the particle distribution carries no
//! variance-truncation bias.
//!
//! SLV interpolates between the two pure models: `xi -> 0` recovers
//! pure local vol (`E[v|S] -> v0`, `L -> sigma_LV / sqrt(v0)`), while a
//! flat market surface makes `L` collapse the stochastic vol back to
//! flat vanilla prices — but forward smiles and path-dependent payoffs
//! keep genuine stochastic-vol dynamics. Vanilla repricing is the
//! calibration test, forward-smile richness the reason to use it.

use crate::core::interpolation::interp_pairs;
use crate::core::montecarlo::path_rng;
use crate::core::trade::PutOrCall;
use crate::equity::heston::HestonParams;
use crate::equity::local_vol::LocalVol;
use crate::equity::processes::qe_variance_step;
use rand::Rng;
use rand_distr::StandardNormal;

/// One SLV time step, shared by calibration and pricing: the variance
/// leg is sampled with the Andersen QE transition (exact CIR conditional
/// moments — no truncation bias), and the spot moves through the
/// Broadie-Kaya decomposition of the integrated variance shock with the
/// leverage frozen over the step,
///
/// ```text
/// int sqrt(v) dW1 = (rho/xi)(v' - v - kappa theta dt + kappa vbar dt)
///                 + sqrt(1 - rho^2) sqrt(vbar dt) Z_perp
/// ln S += (r - q - 1/2 L^2 vbar) dt + L * int sqrt(v) dW1
/// ```
///
/// with `vbar = (v + v')/2`. Spot-variance correlation enters through
/// the sampled `v'`, so the two normals fed in are **independent**; at
/// `L = 1` the scheme is exactly Andersen's K-coefficient QE spot step.
struct SlvStepper {
    hp: HestonParams,
    rho_bar: f64,
    drift: f64,
    dt: f64,
}

impl SlvStepper {
    fn new(hp: &HestonParams, r: f64, q: f64, dt: f64) -> Self {
        SlvStepper {
            hp: *hp,
            rho_bar: (1.0 - hp.rho * hp.rho).sqrt(),
            drift: r - q,
            dt,
        }
    }

    /// Advance `(s, v)` one step under leverage `lev` with independent
    /// standard normals `z_perp` (spot) and `z_v` (variance).
    fn step(&self, lev: f64, s: f64, v: f64, z_perp: f64, z_v: f64) -> (f64, f64) {
        let hp = &self.hp;
        let vp = v.max(0.0);
        let v_next = qe_variance_step(hp, vp, self.dt, z_v);
        let vbar = 0.5 * (vp + v_next);
        let integrated = (hp.rho / hp.vol_of_vol)
            * (v_next - vp - hp.kappa * hp.theta * self.dt + hp.kappa * vbar * self.dt)
            + self.rho_bar * (vbar * self.dt).sqrt() * z_perp;
        let s_next = s * ((self.drift - 0.5 * lev * lev * vbar) * self.dt + lev * integrated).exp();
        (s_next, v_next)
    }
}

/// Simulation / calibration controls.
#[derive(Debug, Clone, Copy)]
pub struct SlvConfig {
    /// Calibration paths (also the default pricing paths).
    pub paths: usize,
    /// Time steps to the calibration horizon.
    pub steps: usize,
    /// Equal-count spot bins for the conditional expectation.
    pub bins: usize,
    pub seed: u64,
}

impl Default for SlvConfig {
    fn default() -> Self {
        Self {
            paths: 20_000,
            steps: 50,
            bins: 20,
            seed: 42,
        }
    }
}

/// The binned conditional-variance curves `E[v_t | S_t]`, one per time
/// step — together with the Dupire surface they define the leverage.
#[derive(Debug, Clone)]
pub struct ConditionalVariance {
    times: Vec<f64>,
    /// Per time: `(mean spot, mean variance)` per bin, sorted by spot.
    slices: Vec<Vec<(f64, f64)>>,
}

impl ConditionalVariance {
    /// `E[v_t | S_t = s]`: linear in spot with flat wings, from the
    /// slice nearest below `t` (piecewise-constant in time, matching
    /// how the calibration used it).
    pub fn value(&self, s: f64, t: f64) -> f64 {
        let idx = self
            .times
            .iter()
            .rposition(|&ti| ti <= t + 1e-12)
            .unwrap_or(0);
        let slice = &self.slices[idx];
        if slice.len() == 1 {
            return slice[0].1;
        }
        interp_pairs(slice, s)
    }
}

/// A calibrated SLV model (borrows the Dupire local vol it was built on).
pub struct Slv<'a> {
    pub heston: HestonParams,
    pub cond_var: ConditionalVariance,
    local_vol: &'a LocalVol<'a>,
    s0: f64,
    r: f64,
    q: f64,
    dt: f64,
}

impl<'a> Slv<'a> {
    /// Leverage `L(s, t) = sigma_LV(s, t) / sqrt(E[v_t | S_t = s])`.
    pub fn leverage(&self, s: f64, t: f64) -> f64 {
        self.local_vol.vol(s, t) / self.cond_var.value(s, t).max(1e-8).sqrt()
    }

    /// Price a European vanilla by simulating the calibrated dynamics
    /// (same step size as the calibration, deterministic per seed).
    pub fn price_vanilla(
        &self,
        strike: f64,
        t: f64,
        put_or_call: PutOrCall,
        paths: usize,
        seed: u64,
    ) -> f64 {
        let steps = (t / self.dt).round().max(1.0) as usize;
        let dt = t / steps as f64;
        let stepper = SlvStepper::new(&self.heston, self.r, self.q, dt);
        let mut sum = 0.0;
        for i in 0..paths {
            let mut rng = path_rng(seed, i as u64);
            let mut s = self.s0;
            let mut v: f64 = self.heston.v0;
            for k in 0..steps {
                let tk = k as f64 * dt;
                let z1: f64 = rng.sample(StandardNormal);
                let z2: f64 = rng.sample(StandardNormal);
                let lev = self.leverage(s, tk);
                (s, v) = stepper.step(lev, s, v, z1, z2);
            }
            sum += match put_or_call {
                PutOrCall::Call => (s - strike).max(0.0),
                PutOrCall::Put => (strike - s).max(0.0),
            };
        }
        (-self.r * t).exp() * sum / paths as f64
    }
}

/// Calibrate the leverage function to `local_vol` out to `horizon`
/// years: forward simulation with per-step binning of `E[v | S]`.
/// Deterministic for a given config.
pub fn calibrate<'a>(
    local_vol: &'a LocalVol<'a>,
    heston: &HestonParams,
    s0: f64,
    r: f64,
    q: f64,
    horizon: f64,
    cfg: &SlvConfig,
) -> Slv<'a> {
    heston.validate().expect("invalid Heston parameters");
    assert!(horizon > 0.0 && cfg.steps > 0 && cfg.bins >= 2 && cfg.paths >= cfg.bins * 10);
    let dt = horizon / cfg.steps as f64;
    let n = cfg.paths;
    let stepper = SlvStepper::new(heston, r, q, dt);

    let mut spots = vec![s0; n];
    let mut vars = vec![heston.v0; n];
    let mut times = Vec::with_capacity(cfg.steps);
    let mut slices = Vec::with_capacity(cfg.steps);

    for k in 0..cfg.steps {
        let t = k as f64 * dt;
        // conditional expectation E[v | S] by equal-count spot bins
        let slice: Vec<(f64, f64)> = if k == 0 {
            vec![(s0, heston.v0)]
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| spots[a].total_cmp(&spots[b]));
            let per_bin = n / cfg.bins;
            (0..cfg.bins)
                .map(|b| {
                    let lo = b * per_bin;
                    let hi = if b == cfg.bins - 1 { n } else { lo + per_bin };
                    let members = &order[lo..hi];
                    let ms = members.iter().map(|&i| spots[i]).sum::<f64>() / members.len() as f64;
                    let mv = members.iter().map(|&i| vars[i]).sum::<f64>() / members.len() as f64;
                    (ms, mv)
                })
                .collect()
        };
        times.push(t);
        slices.push(slice.clone());

        // step every path with the freshly-fitted leverage
        let cond = |s: f64| -> f64 {
            if slice.len() == 1 {
                slice[0].1
            } else {
                interp_pairs(&slice, s)
            }
        };
        for i in 0..n {
            let mut rng = path_rng(cfg.seed.wrapping_add(0x51_1e * k as u64 + 1), i as u64);
            let z1: f64 = rng.sample(StandardNormal);
            let z2: f64 = rng.sample(StandardNormal);
            let lev = local_vol.vol(spots[i], t) / cond(spots[i]).max(1e-8).sqrt();
            (spots[i], vars[i]) = stepper.step(lev, spots[i], vars[i], z1, z2);
        }
    }

    Slv {
        heston: *heston,
        cond_var: ConditionalVariance { times, slices },
        local_vol,
        s0,
        r,
        q,
        dt,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::curves::YieldCurve;
    use crate::core::daycount::DayCountConvention;
    use crate::core::vols::VolSurface;
    use crate::equity::blackscholes::{bs_price, implied_vol_from_price};
    use chrono::NaiveDate;

    const S0: f64 = 100.0;
    const R: f64 = 0.02;

    fn reference() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
    }

    fn curve() -> YieldCurve {
        YieldCurve::flat(
            R,
            reference(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    fn mixing_heston() -> HestonParams {
        // normalized variance process (v0 = theta = 1) supplying the
        // stochasticity the leverage must neutralize for vanillas
        HestonParams {
            v0: 1.0,
            kappa: 2.0,
            theta: 1.0,
            vol_of_vol: 0.8,
            rho: -0.6,
        }
    }

    #[test]
    fn conditional_variance_lookup_interpolates_and_clamps() {
        let cv = ConditionalVariance {
            times: vec![0.0, 0.5],
            slices: vec![vec![(100.0, 1.0)], vec![(90.0, 1.2), (110.0, 0.8)]],
        };
        assert_eq!(cv.value(50.0, 0.1), 1.0); // single-point slice: flat
        assert!((cv.value(100.0, 0.7) - 1.0).abs() < 1e-12); // midpoint
        assert_eq!(cv.value(50.0, 0.7), 1.2); // flat wings
        assert_eq!(cv.value(150.0, 0.7), 0.8);
    }

    #[test]
    fn flat_surface_slv_reprices_black_scholes() {
        // a flat 20% market: after leverage calibration the stochastic
        // vol must wash out of vanilla prices
        let surface = VolSurface::flat(0.2, reference(), DayCountConvention::Act365).unwrap();
        let yc = curve();
        let lv = LocalVol::new(&surface, &yc, S0, 0.0, 0.0);
        let cfg = SlvConfig {
            paths: 16_000,
            steps: 40,
            bins: 20,
            seed: 7,
        };
        let slv = calibrate(&lv, &mixing_heston(), S0, R, 0.0, 1.0, &cfg);

        let mc = slv.price_vanilla(100.0, 1.0, PutOrCall::Call, 32_000, 11);
        let bs = bs_price(S0, 100.0, R, 0.0, 0.2, 1.0, PutOrCall::Call);
        assert!((mc - bs).abs() < 0.02 * bs, "slv {mc} vs bs {bs}");
    }

    #[test]
    fn skewed_surface_slv_reprices_the_smile() {
        // a skewed market smile (same vols at each expiry, so total
        // variance grows in t): the calibrated SLV must give back the
        // input implied vols across strikes
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let smile = |k: f64| 0.2 - 0.1 * (k / S0 - 1.0) + 0.15 * (k / S0 - 1.0).powi(2);
        let smiles: Vec<Vec<(f64, f64)>> = (0..3)
            .map(|_| strikes.iter().map(|&k| (k, smile(k))).collect())
            .collect();
        let surface = VolSurface::from_strike_smiles(
            &[
                crate::core::curves::Tenor::YearFraction(0.25),
                crate::core::curves::Tenor::YearFraction(0.75),
                crate::core::curves::Tenor::YearFraction(1.5),
            ],
            &smiles,
            reference(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let yc = curve();
        let lv = LocalVol::new(&surface, &yc, S0, 0.0, 0.0);
        let cfg = SlvConfig {
            paths: 16_000,
            steps: 40,
            bins: 20,
            seed: 3,
        };
        let slv = calibrate(&lv, &mixing_heston(), S0, R, 0.0, 1.0, &cfg);

        for k in [90.0, 100.0, 110.0] {
            let t = 0.75;
            let mc = slv.price_vanilla(k, t, PutOrCall::Call, 32_000, 5);
            let iv = implied_vol_from_price(S0, k, R, 0.0, t, mc, PutOrCall::Call)
                .expect("inverting the SLV price");
            let market = smile(k);
            assert!(
                (iv - market).abs() < 0.01,
                "K = {k}: slv implied {iv:.4} vs market {market:.4}"
            );
        }
    }

    #[test]
    fn zero_vol_of_vol_degenerates_to_pure_local_vol() {
        // xi -> 0 with v0 = theta = 1: leverage becomes sigma_LV itself
        let surface = VolSurface::flat(0.25, reference(), DayCountConvention::Act365).unwrap();
        let yc = curve();
        let lv = LocalVol::new(&surface, &yc, S0, 0.0, 0.0);
        let degenerate = HestonParams {
            v0: 1.0,
            kappa: 1.0,
            theta: 1.0,
            vol_of_vol: 1e-6,
            rho: 0.0,
        };
        let cfg = SlvConfig {
            paths: 4_000,
            steps: 20,
            bins: 10,
            seed: 9,
        };
        let slv = calibrate(&lv, &degenerate, S0, R, 0.0, 1.0, &cfg);
        // leverage equals the local vol (E[v|S] = 1)
        for s in [80.0, 100.0, 125.0] {
            let lev = slv.leverage(s, 0.5);
            let sigma = lv.vol(s, 0.5);
            assert!((lev - sigma).abs() < 1e-3, "S = {s}: {lev} vs {sigma}");
        }
    }
}
