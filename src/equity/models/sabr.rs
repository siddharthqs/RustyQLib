//! SABR stochastic volatility (Hagan, Kumar, Lesniewski, Woodward 2002)
//! for equity underlyings.
//!
//! Dynamics on the **forward** `F` to the option's expiry, under its
//! expiry measure:
//!
//! ```text
//! dF     = alpha F^beta dW_f
//! dalpha = nu alpha dW_a,        d<W_f, W_a> = rho dt
//! ```
//!
//! four parameters: initial vol level `alpha`, CEV backbone exponent
//! `beta` in [0, 1] (1 = lognormal, the usual equity choice), spot-vol
//! correlation `rho`, and vol-of-vol `nu`.
//!
//! SABR is **pluggable dynamics, not a default**: like Heston and rough
//! Bergomi it rides inside [`Model`](crate::equity::utils::Model)
//! (`mc_model: "sabr"` plus a `sabr` parameter block), so any option can
//! swap it in for the other stochastic-vol models without touching
//! engine or payoff.
//!
//! Three layers live here:
//!
//! - **Pricing** — Hagan's lognormal implied-vol expansion
//!   ([`SabrParams::vol`]) fed into the Black-Scholes closed forms:
//!   vanillas ([`sabr_price`]) and binaries with the smile-slope
//!   correction (`digital = -dC/dK` includes the `vega * dsigma/dK`
//!   term, [`sabr_binary_cash_price`]). Monte Carlo simulation of the
//!   actual two-factor dynamics lives in the MC engine through
//!   [`SabrProcess`](crate::equity::processes::SabrProcess) and covers
//!   the path-dependent payoffs.
//! - **Calibration** — [`SabrParams::calibrate`] fits `(alpha, rho, nu)`
//!   to one expiry's `(strike, vol)` quotes at fixed `beta` (the market
//!   convention: `beta` is chosen, not identified, because it is nearly
//!   collinear with `rho`); [`SabrParams::calibrate_all`] frees `beta`
//!   too. Both are Levenberg-Marquardt fits in an unconstrained
//!   transform space (`ln` / `tanh` / logistic), the same pattern as
//!   [`heston::calibrate`](crate::equity::heston::calibrate) and SVI.
//! - **Smile smoothing** — [`SabrSurfaceFit`] fits one SABR smile per
//!   pillar expiry of a quoted [`VolSurface`] and interpolates total
//!   variance linearly in time (with a calendar floor), turning a noisy
//!   chain-implied surface into a smooth C^2 parameterization that
//!   samples back into a pricing surface via
//!   [`SabrSurfaceFit::to_vol_surface`] — the SABR sibling of
//!   [`SviSurfaceFit`](crate::equity::svi::SviSurfaceFit).

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::core::trade::PutOrCall;
use crate::core::utils::{norm_cdf, norm_pdf};
use crate::core::vols::{VolError, VolSurface};

// ── Model parameters ────────────────────────────────────────────────────

/// SABR parameters. `alpha` is the initial vol level (in the backbone's
/// units: at `beta = 1` it is the ATM lognormal vol, at `beta < 1` the
/// ATM vol is approximately `alpha / F^(1-beta)`), `beta` the CEV
/// exponent, `rho` the spot-vol correlation, `nu` the vol-of-vol.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct SabrParams {
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    #[serde(alias = "vol_of_vol", alias = "volvol")]
    pub nu: f64,
}

impl SabrParams {
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.alpha <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "sabr params",
                "SABR alpha must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.beta) {
            return Err(RustyQLibError::invalid_input(
                "sabr params",
                "SABR beta must lie in [0, 1]".to_string(),
            ));
        }
        if !(-1.0..1.0).contains(&self.rho) || self.rho == -1.0 {
            return Err(RustyQLibError::invalid_input(
                "sabr params",
                "SABR rho must lie in (-1, 1)".to_string(),
            ));
        }
        if self.nu < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "sabr params",
                "SABR nu (vol-of-vol) must be non-negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Parameters with a parallel implied-vol shift applied — the
    /// library's vega bump convention, mirroring
    /// [`HestonParams::with_vol_shift`](crate::equity::heston::HestonParams::with_vol_shift).
    /// `alpha` is scaled so the leading-order ATM vol
    /// `alpha / forward^(1-beta)` moves by exactly `shift` (the whole
    /// smile scales with it); `forward` anchors the backbone conversion
    /// and is irrelevant at `beta = 1`.
    pub fn with_vol_shift(&self, shift: f64, forward: f64) -> SabrParams {
        if shift == 0.0 {
            return *self;
        }
        let atm = self.alpha * forward.powf(self.beta - 1.0);
        let scale = (atm + shift).max(1e-6) / atm;
        SabrParams {
            alpha: self.alpha * scale,
            ..*self
        }
    }

    /// Hagan's lognormal (Black) implied vol for strike `k` given the
    /// forward `f` and expiry `t` — the 2002 "Managing Smile Risk"
    /// singular-perturbation expansion (eq. A.69), with the standard
    /// numerically stable evaluation of `z / x(z)` on both wings.
    pub fn vol(&self, f: f64, k: f64, t: f64) -> f64 {
        assert!(f > 0.0 && k > 0.0, "SABR vol needs positive forward and strike");
        let (alpha, beta, rho, nu) = (self.alpha, self.beta, self.rho, self.nu);
        let omb = 1.0 - beta;
        let l = (f / k).ln();
        let l2 = l * l;
        // (FK)^((1-beta)/2) — the backbone factor
        let fk_pow = (f * k).powf(0.5 * omb);
        let denom = fk_pow * (1.0 + omb * omb / 24.0 * l2 + omb.powi(4) / 1920.0 * l2 * l2);
        let z = nu / alpha * fk_pow * l;
        // time-order correction c: all three O(t) terms of the expansion
        let c = omb * omb * alpha * alpha / (24.0 * fk_pow * fk_pow)
            + 0.25 * rho * beta * nu * alpha / fk_pow
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0;
        (alpha / denom) * z_over_x(z, rho) * (1.0 + c * t)
    }

    /// Strike derivative of the Hagan vol, `dsigma/dK` (central
    /// difference) — what the smile-consistent digital pricing needs.
    pub fn vol_dk(&self, f: f64, k: f64, t: f64) -> f64 {
        let h = k * 1e-4;
        (self.vol(f, k + h, t) - self.vol(f, k - h, t)) / (2.0 * h)
    }

    /// Total Black variance `sigma^2 t` at log-moneyness `k = ln(K/F)`.
    pub fn total_variance(&self, f: f64, k: f64, t: f64) -> f64 {
        let v = self.vol(f, f * k.exp(), t);
        v * v * t
    }

    /// The Gatheral butterfly function `g(k)` (the density up to a
    /// positive factor) evaluated with numerical total-variance
    /// derivatives — negative values flag butterfly arbitrage, which
    /// Hagan's expansion is known to produce on far wings at long
    /// expiries. Same convention as
    /// [`SviParams::butterfly_g`](crate::equity::svi::SviParams::butterfly_g).
    pub fn butterfly_g(&self, f: f64, k: f64, t: f64) -> f64 {
        let h = 1e-4;
        let w = self.total_variance(f, k, t);
        let wp = self.total_variance(f, k + h, t);
        let wm = self.total_variance(f, k - h, t);
        let w1 = (wp - wm) / (2.0 * h);
        let w2 = (wp - 2.0 * w + wm) / (h * h);
        (1.0 - k * w1 / (2.0 * w)).powi(2) - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    }
}

/// `z / x(z)` with `x(z) = ln((sqrt(1 - 2 rho z + z^2) + z - rho)/(1 - rho))`:
/// series near zero, and the algebraically equivalent
/// `x(z) = ln((1 + rho)/(sqrt(..) - z + rho))` on the negative wing where
/// the direct form cancels catastrophically.
fn z_over_x(z: f64, rho: f64) -> f64 {
    if z.abs() < 1e-6 {
        return 1.0 + 0.5 * rho * z;
    }
    let y = (1.0 - 2.0 * rho * z + z * z).sqrt();
    let x = if z > 0.0 {
        ((y + z - rho) / (1.0 - rho)).ln()
    } else {
        // (y + z - rho)(y - z + rho) = 1 - rho^2
        ((1.0 + rho) / (y - z + rho)).ln()
    };
    z / x
}

// ── Calibration ─────────────────────────────────────────────────────────

/// Result of a SABR smile calibration.
#[derive(Debug, Clone)]
pub struct SabrFit {
    pub params: SabrParams,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl SabrParams {
    /// Calibrate `(alpha, rho, nu)` at fixed `beta` to one expiry's
    /// `(strike, implied vol)` quotes given the `forward`, by
    /// Levenberg-Marquardt on vol residuals with `alpha` and `nu` in log
    /// space and `rho` through `tanh` — every trial parameter set is
    /// admissible by construction.
    ///
    /// `beta` is fixed because it is what the trader chooses (backbone
    /// dynamics), not what the smile identifies: `beta` and `rho` are
    /// nearly collinear over any quoted strike range. Equity desks
    /// typically run `beta = 1`.
    pub fn calibrate(quotes: &[(f64, f64)], forward: f64, t: f64, beta: f64) -> SabrFit {
        assert!(
            quotes.len() >= 3,
            "SABR has three free parameters at fixed beta; need at least three quotes"
        );
        assert!(forward > 0.0 && t > 0.0);
        assert!((0.0..=1.0).contains(&beta), "beta must lie in [0, 1]");
        let x0 = start_point(quotes, forward, beta);
        let unpack = |u: &[f64]| SabrParams {
            alpha: u[0].exp(),
            beta,
            rho: u[1].tanh(),
            nu: u[2].exp(),
        };
        Self::run_fit(quotes, forward, t, &x0, unpack)
    }

    /// Calibrate all four parameters, `beta` through a logistic
    /// transform onto [0, 1]. Prefer [`calibrate`](Self::calibrate) with
    /// a chosen `beta` unless the quote set genuinely spans enough of
    /// the backbone to identify it.
    pub fn calibrate_all(quotes: &[(f64, f64)], forward: f64, t: f64) -> SabrFit {
        assert!(
            quotes.len() >= 4,
            "free-beta SABR has four parameters; need at least four quotes"
        );
        assert!(forward > 0.0 && t > 0.0);
        let beta0: f64 = 0.9;
        let mut x0 = start_point(quotes, forward, beta0);
        x0.push((beta0 / (1.0 - beta0)).ln());
        let unpack = |u: &[f64]| SabrParams {
            alpha: u[0].exp(),
            beta: 1.0 / (1.0 + (-u[3]).exp()),
            rho: u[1].tanh(),
            nu: u[2].exp(),
        };
        Self::run_fit(quotes, forward, t, &x0, unpack)
    }

    fn run_fit(
        quotes: &[(f64, f64)],
        forward: f64,
        t: f64,
        x0: &[f64],
        unpack: impl Fn(&[f64]) -> SabrParams,
    ) -> SabrFit {
        let residuals = |u: &[f64]| -> Vec<f64> {
            let p = unpack(u);
            quotes.iter().map(|&(k, v)| p.vol(forward, k, t) - v).collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, x0);
        let params = unpack(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(k, v)| (params.vol(forward, k, t) - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SabrFit {
            params,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        }
    }
}

/// Starting point `[ln alpha, atanh rho, ln nu]`: `alpha` backs out of
/// the quote nearest the money (leading order `sigma_atm = alpha
/// F^(beta-1)`), gentle negative skew and moderate vol-of-vol.
fn start_point(quotes: &[(f64, f64)], forward: f64, beta: f64) -> Vec<f64> {
    let atm_vol = quotes
        .iter()
        .min_by(|a, b| {
            (a.0 - forward)
                .abs()
                .partial_cmp(&(b.0 - forward).abs())
                .unwrap()
        })
        .map(|&(_, v)| v)
        .unwrap();
    let alpha0 = (atm_vol * forward.powf(1.0 - beta)).max(1e-4);
    vec![
        alpha0.ln(),
        (-0.3_f64).atanh(),
        0.5_f64.ln(),
    ]
}

// ── Analytic pricing (Hagan vol into Black-Scholes) ─────────────────────

/// SABR price of a European vanilla: Hagan implied vol at the option's
/// forward, fed into the Black-Scholes closed form.
pub fn sabr_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    sp: &SabrParams,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0);
    if t <= 0.0 {
        return crate::equity::blackscholes::bs_price(s, k, r, q, 0.0, t, put_or_call);
    }
    let forward = s * ((r - q) * t).exp();
    let sigma = sp.vol(forward, k, t);
    crate::equity::blackscholes::bs_price(s, k, r, q, sigma, t, put_or_call)
}

/// SABR price of a cash-or-nothing binary, **smile-consistent**: the
/// digital is the strike derivative of the vanilla, so alongside the
/// Black `N(d2)` term it carries the skew correction
/// `- vega * dsigma/dK` (a negative equity skew makes digital calls
/// *richer* than the flat-vol value). Clamped to the no-arbitrage band
/// `[0, e^{-rT}]` per unit cash.
#[allow(clippy::too_many_arguments)]
pub fn sabr_binary_cash_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    sp: &SabrParams,
    cash: f64,
    put_or_call: PutOrCall,
) -> f64 {
    assert!(s > 0.0 && k > 0.0);
    if t <= 0.0 {
        let itm = match put_or_call {
            PutOrCall::Call => s > k,
            PutOrCall::Put => s < k,
        };
        return if itm { cash } else { 0.0 };
    }
    let df = (-r * t).exp();
    let forward = s * ((r - q) * t).exp();
    let sigma = sp.vol(forward, k, t);
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let vega = s * (-q * t).exp() * norm_pdf(d1) * sqrt_t;
    let digital_call = (df * norm_cdf(d2) - vega * sp.vol_dk(forward, k, t)).clamp(0.0, df);
    match put_or_call {
        PutOrCall::Call => cash * digital_call,
        PutOrCall::Put => cash * (df - digital_call),
    }
}

/// SABR price of an asset-or-nothing binary, through the replication
/// `asset-or-nothing call = vanilla call + K * cash-or-nothing call`
/// (which keeps it consistent with the smile-corrected digital).
pub fn sabr_binary_asset_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    sp: &SabrParams,
    put_or_call: PutOrCall,
) -> f64 {
    let call = sabr_price(s, k, r, q, t, sp, PutOrCall::Call);
    let digital_call = sabr_binary_cash_price(s, k, r, q, t, sp, 1.0, PutOrCall::Call);
    let asset_call = call + k * digital_call;
    match put_or_call {
        PutOrCall::Call => asset_call,
        PutOrCall::Put => s * (-q * t).exp() - asset_call,
    }
}

// ── Option-level analytic pricing ───────────────────────────────────────

use crate::equity::bump::BumpedMarket;
use crate::equity::utils::PayoffType;
use crate::equity::vanilla_option::{BinaryPayoff, BinaryType, EquityOption};

/// Analytic SABR price through a market view; `None` prices the base
/// market. The view's vol bump is *interpreted* on the model (there is
/// no implied surface to shift): `alpha` is scaled so the ATM vol moves
/// in parallel ([`SabrParams::with_vol_shift`]) — the same convention as
/// the Heston and rough Bergomi analytic routes.
pub(crate) fn analytic_npv(option: &EquityOption, bumped_market: Option<&BumpedMarket>) -> f64 {
    let base = BumpedMarket::base(&option.market);
    let m = bumped_market.unwrap_or(&base);
    let maturity = option.base.maturity_date;
    let s = m.effective_spot(maturity);
    let k = option.base.strike_price;
    let r = m.risk_free_rate(maturity);
    let q = m.carry_yield();
    let t = m.time_to_maturity(maturity);
    let forward = s * ((r - q) * t).exp();
    let sp = option.sabr_params().with_vol_shift(m.bump().d_vol, forward);
    let pc = *option.payoff.put_or_call();
    match option.payoff.payoff_kind() {
        PayoffType::Vanilla => sabr_price(s, k, r, q, t, &sp, pc),
        PayoffType::Binary => {
            let payoff = option
                .payoff
                .as_any()
                .downcast_ref::<BinaryPayoff>()
                .expect("payoff of kind Binary must be a BinaryPayoff");
            match payoff.binary_type {
                BinaryType::CashOrNothing => {
                    sabr_binary_cash_price(s, k, r, q, t, &sp, payoff.cash, pc)
                }
                BinaryType::AssetOrNothing => sabr_binary_asset_price(s, k, r, q, t, &sp, pc),
            }
        }
        _ => panic!(
            "The SABR analytic pricer supports vanilla and binary payoffs; \
             use the MonteCarlo engine for path-dependent payoffs"
        ),
    }
}

// ── Per-expiry SABR surface fit (smile smoothing) ───────────────────────

/// One fitted expiry slice of a [`SabrSurfaceFit`].
#[derive(Debug, Clone)]
pub struct SabrSlice {
    /// Expiry time (year fraction).
    pub t: f64,
    /// Forward the slice is calibrated against.
    pub forward: f64,
    pub params: SabrParams,
    /// Fit error in implied vol against the input pillars.
    pub rmse: f64,
    pub converged: bool,
    /// The quoted log-moneyness span the fit is anchored on.
    pub k_range: (f64, f64),
    /// Minimum of Gatheral's `g(k)` over the quoted span; negative
    /// means the fitted smile carries butterfly arbitrage there.
    pub min_g: f64,
}

/// A per-expiry SABR fit of an implied surface: one [`SabrParams`] smile
/// per pillar expiry (at a common fixed `beta`), linear total variance
/// in time between slices at fixed log-moneyness with a floor against
/// calendar crossings — the SABR sibling of
/// [`SviSurfaceFit`](crate::equity::svi::SviSurfaceFit).
///
/// This is a **smoother**: every quoted point moves a little (by the fit
/// RMSE) in exchange for a C^2 smile whose wings extrapolate with the
/// model's dynamics rather than a spline's whim. Fit it to the *cleaned*
/// surface (after
/// [`repair_arbitrage`](crate::equity::surface_repair::repair_arbitrage))
/// so outright arbitrage is gone before smoothing; sample it back into a
/// pricing [`VolSurface`] with [`Self::to_vol_surface`].
#[derive(Debug, Clone)]
pub struct SabrSurfaceFit {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    /// The common backbone exponent every slice was fitted at.
    pub beta: f64,
    /// Slices in increasing expiry order.
    pub slices: Vec<SabrSlice>,
    /// Input expiries skipped for having fewer than three pillar quotes.
    pub skipped_slices: usize,
    /// Largest total-variance decrease between adjacent fitted slices
    /// over the quoted span (0 = calendar-clean fit); evaluation floors
    /// the later variance, so this measures fit tension, not arbitrage
    /// in the output.
    pub max_calendar_crossing: f64,
}

impl SabrSurfaceFit {
    /// Fit one SABR smile per pillar expiry of `surface` at fixed
    /// `beta`. `forward` maps expiry time to the underlying's forward,
    /// exactly as for
    /// [`SviSurfaceFit::fit`](crate::equity::svi::SviSurfaceFit::fit).
    /// Expiries with fewer than three pillars are skipped and counted.
    pub fn fit(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
        beta: f64,
    ) -> Result<SabrSurfaceFit, RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "sabr fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };
        let mut slices = Vec::new();
        let mut skipped = 0usize;
        for (tenor, smile) in expiries.iter().zip(&smiles) {
            let t = match tenor {
                Tenor::YearFraction(t) => *t,
                Tenor::Date(_) => continue, // to_input never emits dates
            };
            if smile.len() < 3 {
                skipped += 1;
                continue;
            }
            let f = forward(t);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let strike = match coordinate {
                        SmileCoordinate::Strike => x,
                        SmileCoordinate::Moneyness => x * f,
                        SmileCoordinate::LogMoneyness => f * x.exp(),
                    };
                    (strike, vol)
                })
                .collect();
            let fit = SabrParams::calibrate(&quotes, f, t, beta);
            let (k_lo, k_hi) = quotes
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &(strike, _)| {
                    let k = (strike / f).ln();
                    (lo.min(k), hi.max(k))
                });
            let min_g = (0..=200)
                .map(|i| {
                    fit.params
                        .butterfly_g(f, k_lo + (k_hi - k_lo) * i as f64 / 200.0, t)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(SabrSlice {
                t,
                forward: f,
                params: fit.params,
                rmse: fit.rmse,
                converged: fit.converged,
                k_range: (k_lo, k_hi),
                min_g,
            });
        }
        if slices.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "sabr fit",
                format!("no expiry has the three quotes a SABR fit needs ({skipped} skipped)"),
            ));
        }
        slices.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());

        // fit tension: does total variance ever fall between slices?
        let mut max_crossing: f64 = 0.0;
        for pair in slices.windows(2) {
            let (lo, hi) = (
                pair[0].k_range.0.min(pair[1].k_range.0),
                pair[0].k_range.1.max(pair[1].k_range.1),
            );
            for i in 0..=100 {
                let k = lo + (hi - lo) * i as f64 / 100.0;
                let crossing = slice_variance(&pair[0], k) - slice_variance(&pair[1], k);
                max_crossing = max_crossing.max(crossing);
            }
        }
        Ok(SabrSurfaceFit {
            reference_date: surface.reference_date(),
            day_count: surface.day_count(),
            beta,
            slices,
            skipped_slices: skipped,
            max_calendar_crossing: max_crossing,
        })
    }

    /// The slice pair bracketing `t`, with the interpolation weight on
    /// the later slice (0 at or below the earlier, 1 at or beyond the
    /// later; a single-slice surface brackets with itself).
    fn bracket(&self, t: f64) -> (&SabrSlice, &SabrSlice, f64) {
        let n = self.slices.len();
        if n == 1 || t <= self.slices[0].t {
            return (&self.slices[0], &self.slices[0], 0.0);
        }
        if t >= self.slices[n - 1].t {
            return (&self.slices[n - 1], &self.slices[n - 1], 0.0);
        }
        let idx = self.slices.partition_point(|s| s.t < t);
        let (a, b) = (&self.slices[idx - 1], &self.slices[idx]);
        (a, b, (t - a.t) / (b.t - a.t))
    }

    /// Forward at `t`: linear between the slice forwards, flat outside.
    pub fn forward(&self, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        a.forward + (b.forward - a.forward) * weight
    }

    /// Total variance at log-moneyness `k`: each slice contributes its
    /// own smile at its own forward, linear in time between slices at
    /// fixed `k` with the later slice floored at the earlier (calendar
    /// safety); proportional to `t` below the first slice.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        let wa = slice_variance(a, k);
        let wb = slice_variance(b, k).max(wa);
        let w = if t <= a.t {
            wa * (t / a.t).min(1.0)
        } else {
            wa + (wb - wa) * weight
        };
        w.max(0.0)
    }

    /// Implied vol for an absolute `strike` at `t`.
    pub fn vol(&self, strike: f64, t: f64) -> f64 {
        let k = (strike / self.forward(t)).ln();
        (self.total_variance(k, t).max(1e-12) / t.max(1e-8)).sqrt()
    }

    /// Sample the fit into the canonical pricing [`VolSurface`]: per
    /// slice, `samples` strikes across its own quoted log-moneyness
    /// span, through the floored [`Self::total_variance`] so the
    /// calendar floor is baked into the artifact. The sampled surface
    /// serializes, plots and prices like any other.
    pub fn to_vol_surface(&self, samples: usize) -> Result<VolSurface, VolError> {
        let expiries: Vec<Tenor> = self
            .slices
            .iter()
            .map(|s| Tenor::YearFraction(s.t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = self
            .slices
            .iter()
            .map(|slice| {
                let (lo, hi) = slice.k_range;
                let n = samples.max(5);
                (0..n)
                    .map(|i| {
                        let k = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                        let vol = (self.total_variance(k, slice.t).max(1e-12) / slice.t).sqrt();
                        (slice.forward * k.exp(), vol)
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, self.reference_date, self.day_count)
    }

    /// Fit-quality metadata for the surface document: per-slice params,
    /// RMSE in vol basis points, convergence, `min g`, and the global
    /// calendar-tension figure.
    pub fn metadata(&self) -> serde_json::Value {
        let slices: Vec<serde_json::Value> = self
            .slices
            .iter()
            .map(|s| {
                serde_json::json!({
                    "t": s.t,
                    "forward": s.forward,
                    "params": {
                        "alpha": s.params.alpha, "beta": s.params.beta,
                        "rho": s.params.rho, "nu": s.params.nu,
                    },
                    "rmse_vol_bps": s.rmse * 1e4,
                    "converged": s.converged,
                    "min_butterfly_g": s.min_g,
                    "k_range": [s.k_range.0, s.k_range.1],
                })
            })
            .collect();
        serde_json::json!({
            "model": "per-expiry SABR (Hagan 2002 lognormal vol), linear total variance in time",
            "beta": self.beta,
            "slices": slices,
            "skipped_slices": self.skipped_slices,
            "max_calendar_crossing": self.max_calendar_crossing,
        })
    }
}

/// One slice's total variance at log-moneyness `k` (strike measured
/// against the slice's own forward).
fn slice_variance(slice: &SabrSlice, k: f64) -> f64 {
    slice.params.total_variance(slice.forward, k, slice.t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::blackscholes::bs_price;

    fn params() -> SabrParams {
        SabrParams {
            alpha: 0.2,
            beta: 1.0,
            rho: -0.4,
            nu: 0.6,
        }
    }

    #[test]
    fn atm_vol_matches_the_closed_form_limit() {
        // at K = F the expansion collapses to alpha/F^(1-beta) (1 + c t)
        let sp = SabrParams {
            alpha: 1.8,
            beta: 0.5,
            rho: -0.3,
            nu: 0.5,
        };
        let (f, t) = (100.0_f64, 1.5);
        let fk = f.powf(1.0 - sp.beta);
        let c = (1.0 - sp.beta).powi(2) * sp.alpha * sp.alpha / (24.0 * fk * fk)
            + 0.25 * sp.rho * sp.beta * sp.nu * sp.alpha / fk
            + (2.0 - 3.0 * sp.rho * sp.rho) * sp.nu * sp.nu / 24.0;
        let expected = sp.alpha / fk * (1.0 + c * t);
        assert!((sp.vol(f, f, t) - expected).abs() < 1e-12);
    }

    #[test]
    fn zero_vol_of_vol_beta_one_is_flat_black_scholes() {
        let sp = SabrParams {
            alpha: 0.25,
            beta: 1.0,
            rho: 0.0,
            nu: 0.0,
        };
        for k in [60.0, 100.0, 150.0] {
            assert!((sp.vol(100.0, k, 2.0) - 0.25).abs() < 1e-12, "K={k}");
            let sabr = sabr_price(100.0, k, 0.03, 0.01, 2.0, &sp, PutOrCall::Call);
            let bs = bs_price(100.0, k, 0.03, 0.01, 0.25, 2.0, PutOrCall::Call);
            assert!((sabr - bs).abs() < 1e-12);
        }
    }

    #[test]
    fn negative_rho_skews_the_smile_and_nu_bends_it() {
        let sp = params();
        // negative correlation: put wing above call wing
        assert!(sp.vol(100.0, 80.0, 1.0) > sp.vol(100.0, 125.0, 1.0));
        // vol-of-vol curvature, isolated at rho = 0: both wings of the
        // symmetric smile sit above the zero-nu (flat) smile
        let smile = SabrParams { rho: 0.0, ..sp };
        let flat = SabrParams {
            rho: 0.0,
            nu: 1e-12,
            ..sp
        };
        assert!(smile.vol(100.0, 70.0, 1.0) > flat.vol(100.0, 70.0, 1.0));
        assert!(smile.vol(100.0, 140.0, 1.0) > flat.vol(100.0, 140.0, 1.0));
    }

    #[test]
    fn wings_are_finite_and_continuous_across_the_money() {
        // the z/x(z) evaluation must stay stable on both wings and glue
        // smoothly through K = F (where z -> 0)
        let sp = SabrParams {
            alpha: 1.5,
            beta: 0.6,
            rho: -0.7,
            nu: 0.9,
        };
        let f = 100.0;
        let mut prev = sp.vol(f, f * (-2.5_f64).exp(), 1.0);
        for i in 1..=1000 {
            let k = f * (-2.5 + i as f64 * 0.005).exp();
            let v = sp.vol(f, k, 1.0);
            assert!(v.is_finite() && v > 0.0, "K={k}");
            assert!((v - prev).abs() < 0.02, "jump at K={k}: {prev} -> {v}");
            prev = v;
        }
    }

    #[test]
    fn put_call_parity_and_binary_replication() {
        let sp = params();
        let (s, k, r, q, t) = (100.0, 92.0, 0.04, 0.015, 0.9);
        let c = sabr_price(s, k, r, q, t, &sp, PutOrCall::Call);
        let p = sabr_price(s, k, r, q, t, &sp, PutOrCall::Put);
        let parity = s * (-q * t).exp() - k * (-r * t).exp();
        // parity holds exactly: both legs read the same smile vol at K
        assert!((c - p - parity).abs() < 1e-10);
        // asset-or-nothing minus K * cash-or-nothing rebuilds the vanilla
        let asset = sabr_binary_asset_price(s, k, r, q, t, &sp, PutOrCall::Call);
        let cash = sabr_binary_cash_price(s, k, r, q, t, &sp, k, PutOrCall::Call);
        assert!((c - (asset - cash)).abs() < 1e-10);
        // cash-or-nothing call + put pay the discount factor
        let dc = sabr_binary_cash_price(s, k, r, q, t, &sp, 1.0, PutOrCall::Call);
        let dp = sabr_binary_cash_price(s, k, r, q, t, &sp, 1.0, PutOrCall::Put);
        assert!((dc + dp - (-r * t).exp()).abs() < 1e-12);
    }

    #[test]
    fn digital_matches_the_strike_derivative_of_the_vanilla() {
        // the smile-corrected digital is -dC/dK; check against a bump of
        // the full smile-read vanilla (which repriced at sigma(K +- h))
        let sp = params();
        let (s, r, q, t) = (100.0, 0.03, 0.01, 1.0);
        for k in [85.0, 100.0, 115.0] {
            let h = k * 1e-4;
            let up = sabr_price(s, k + h, r, q, t, &sp, PutOrCall::Call);
            let down_call = sabr_price(s, k - h, r, q, t, &sp, PutOrCall::Call);
            let numeric = -(up - down_call) / (2.0 * h);
            let analytic = sabr_binary_cash_price(s, k, r, q, t, &sp, 1.0, PutOrCall::Call);
            assert!(
                (numeric - analytic).abs() < 1e-5,
                "K={k}: numeric {numeric} vs analytic {analytic}"
            );
        }
    }

    #[test]
    fn skew_makes_digital_calls_richer_than_flat_vol() {
        // with dsigma/dK < 0 the digital call gains vega * |slope|
        let sp = params();
        let (s, k, r, q, t) = (100.0, 100.0, 0.03, 0.0, 1.0);
        let smile = sabr_binary_cash_price(s, k, r, q, t, &sp, 1.0, PutOrCall::Call);
        let sigma = sp.vol(s * ((r - q) * t).exp(), k, t);
        let flat = SabrParams {
            alpha: sigma,
            beta: 1.0,
            rho: 0.0,
            nu: 0.0,
        };
        let plain = sabr_binary_cash_price(s, k, r, q, t, &flat, 1.0, PutOrCall::Call);
        assert!(smile > plain, "{smile} vs {plain}");
    }

    #[test]
    fn vol_shift_moves_the_atm_vol_in_parallel() {
        for beta in [1.0, 0.5] {
            let sp = SabrParams {
                alpha: 0.2 * 100.0_f64.powf(1.0 - beta),
                beta,
                rho: -0.4,
                nu: 0.5,
            };
            let f = 100.0;
            let bumped = sp.with_vol_shift(0.01, f);
            // leading-order ATM vol (strip the O(t) correction with t = 0)
            let base_atm = sp.vol(f, f, 0.0);
            let bumped_atm = bumped.vol(f, f, 0.0);
            assert!(
                (bumped_atm - base_atm - 0.01).abs() < 1e-10,
                "beta={beta}: {base_atm} -> {bumped_atm}"
            );
        }
    }

    #[test]
    fn calibration_round_trips_at_fixed_beta() {
        let truth = SabrParams {
            alpha: 0.22,
            beta: 1.0,
            rho: -0.55,
            nu: 0.75,
        };
        let (f, t) = (100.0, 0.75);
        let quotes: Vec<(f64, f64)> = (0..13)
            .map(|i| f * (-0.3 + i as f64 * 0.05_f64).exp())
            .map(|k| (k, truth.vol(f, k, t)))
            .collect();
        let fit = SabrParams::calibrate(&quotes, f, t, 1.0);
        assert!(fit.rmse < 1e-7, "vol rmse {} params {:?}", fit.rmse, fit.params);
        assert!((fit.params.alpha - truth.alpha).abs() < 1e-3, "alpha {}", fit.params.alpha);
        assert!((fit.params.rho - truth.rho).abs() < 1e-2, "rho {}", fit.params.rho);
        assert!((fit.params.nu - truth.nu).abs() < 1e-2, "nu {}", fit.params.nu);
        assert!(fit.params.validate().is_ok());
        // off-grid strikes match too
        for i in 0..=20 {
            let k = f * (-0.35 + i as f64 * 0.035_f64).exp();
            assert!((fit.params.vol(f, k, t) - truth.vol(f, k, t)).abs() < 1e-5);
        }
    }

    #[test]
    fn free_beta_calibration_reprices_the_smile() {
        // beta is barely identified, so judge the fit by repricing error,
        // not by parameter recovery
        let truth = SabrParams {
            alpha: 1.1,
            beta: 0.7,
            rho: -0.35,
            nu: 0.55,
        };
        let (f, t) = (100.0, 1.0);
        let quotes: Vec<(f64, f64)> = (0..15)
            .map(|i| f * (-0.35 + i as f64 * 0.05_f64).exp())
            .map(|k| (k, truth.vol(f, k, t)))
            .collect();
        let fit = SabrParams::calibrate_all(&quotes, f, t);
        assert!(fit.rmse < 5e-4, "vol rmse {} params {:?}", fit.rmse, fit.params);
        assert!(fit.params.validate().is_ok());
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(SabrParams { alpha: 0.0, ..params() }.validate().is_err());
        assert!(SabrParams { beta: 1.2, ..params() }.validate().is_err());
        assert!(SabrParams { rho: -1.0, ..params() }.validate().is_err());
        assert!(SabrParams { nu: -0.1, ..params() }.validate().is_err());
        assert!(params().validate().is_ok());
    }

    // ── Surface fit ─────────────────────────────────────────────────────

    fn surface_from(slices: &[(f64, SabrParams, f64)]) -> VolSurface {
        let expiries: Vec<Tenor> = slices
            .iter()
            .map(|&(t, _, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = slices
            .iter()
            .map(|&(t, p, f)| {
                (0..11)
                    .map(|i| {
                        let k = -0.3 + i as f64 * 0.06;
                        let strike = f * k.exp();
                        (strike, p.vol(f, strike, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap()
    }

    #[test]
    fn surface_fit_recovers_generating_smiles() {
        let front = SabrParams {
            alpha: 0.24,
            beta: 1.0,
            rho: -0.5,
            nu: 0.8,
        };
        let back = SabrParams {
            alpha: 0.26,
            beta: 1.0,
            rho: -0.45,
            nu: 0.6,
        };
        let surface = surface_from(&[(0.5, front, 101.0), (1.0, back, 102.0)]);
        let forward = |t: f64| if t < 0.75 { 101.0 } else { 102.0 };
        let fit = SabrSurfaceFit::fit(&surface, forward, 1.0).unwrap();
        assert_eq!(fit.slices.len(), 2);
        assert_eq!(fit.skipped_slices, 0);
        for slice in &fit.slices {
            assert!(slice.rmse < 1e-5, "rmse {}", slice.rmse);
        }
        assert!(fit.max_calendar_crossing <= 1e-8);
        // fitted vols agree with the generators off the pillar grid
        for i in 0..=12 {
            let k: f64 = -0.28 + i as f64 * 0.05;
            let strike = 101.0 * k.exp();
            let want = front.vol(101.0, strike, 0.5);
            assert!(
                (fit.vol(strike, 0.5) - want).abs() < 5e-4,
                "k = {k}: {} vs {want}",
                fit.vol(strike, 0.5)
            );
        }
        // sampled surface matches the fit at its own nodes
        let sampled = fit.to_vol_surface(41).unwrap();
        assert_eq!(sampled.expiry_times().len(), 2);
        let probe = 101.0;
        assert!((sampled.vol(probe, probe, 0.5) - fit.vol(probe, 0.5)).abs() < 1e-3);
        // metadata carries the fit quality
        let meta = fit.metadata();
        assert_eq!(meta["slices"].as_array().unwrap().len(), 2);
        assert_eq!(meta["beta"].as_f64().unwrap(), 1.0);
        assert!(meta["slices"][0]["rmse_vol_bps"].as_f64().unwrap() < 0.5);
    }

    #[test]
    fn surface_fit_smooths_noisy_quotes() {
        // perturb a clean smile with +-30bp sawtooth noise: the fit must
        // land between the noise, not on it — that is the smoothing claim
        let truth = SabrParams {
            alpha: 0.22,
            beta: 1.0,
            rho: -0.5,
            nu: 0.7,
        };
        let (f, t) = (100.0, 0.5);
        let expiries = [Tenor::YearFraction(t)];
        let smiles = vec![(0..13)
            .map(|i| {
                let k = -0.3 + i as f64 * 0.05;
                let noise = if i % 2 == 0 { 0.003 } else { -0.003 };
                (f * k.exp(), truth.vol(f, f * k.exp(), t) + noise)
            })
            .collect::<Vec<_>>()];
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let fit = SabrSurfaceFit::fit(&surface, |_| f, 1.0).unwrap();
        let slice = &fit.slices[0];
        // the fit absorbs the sawtooth: recovered smile within 15bp of
        // the noise-free truth everywhere (noise amplitude is 30bp)
        for i in 0..=24 {
            let k = -0.3 + i as f64 * 0.025;
            let strike = f * k.exp();
            let got = slice.params.vol(f, strike, t);
            let want = truth.vol(f, strike, t);
            assert!(
                (got - want).abs() < 1.5e-3,
                "k={k}: fitted {got} vs truth {want}"
            );
        }
        // and the fit RMSE is on the order of the injected noise
        assert!(slice.rmse < 4e-3, "rmse {}", slice.rmse);
    }

    #[test]
    fn sparse_slices_are_skipped_not_fatal() {
        let p = params();
        let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
        let smiles = vec![
            // two quotes: below SABR's three-parameter minimum
            vec![(90.0, 0.25), (110.0, 0.23)],
            (0..9)
                .map(|i| {
                    let k = -0.2 + i as f64 * 0.05;
                    let strike = 100.0 * k.exp();
                    (strike, p.vol(100.0, strike, 1.0))
                })
                .collect(),
        ];
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let fit = SabrSurfaceFit::fit(&surface, |_| 100.0, 1.0).unwrap();
        assert_eq!(fit.slices.len(), 1);
        assert_eq!(fit.skipped_slices, 1);
    }
}
