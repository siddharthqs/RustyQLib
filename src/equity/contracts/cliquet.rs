//! Cliquet (ratchet) options: a strip of forward-start performance
//! periods with local and global caps/floors.
//!
//! The payoff observes period returns `R_i = S_{t_i}/S_{t_{i-1}} - 1`
//! over an equally-spaced reset schedule and pays at maturity
//!
//! ```text
//! N * clamp( sum_i clamp(R_i, local_floor, local_cap),
//!            global_floor, global_cap )
//! ```
//!
//! A **ratchet** is the `local_floor = 0` special case: each period
//! locks in its gain and losses are forgiven. Because the payoff is
//! built from returns it is spot-homogeneous — the classic product
//! whose value is all **forward smile**: under Black-Scholes each
//! period is an independent lognormal and the price collapses to a
//! closed form (a strip of forward-start call spreads); under Heston
//! the forward smile is model-generated and the price genuinely
//! differs, which is the reason desks price cliquets on stochastic-vol
//! models.
//!
//! Engines: `Analytical` (Black-Scholes closed form; requires no
//! global cap/floor, which break the per-period independence) and
//! `MonteCarlo` (GBM per-period sampling, or full Heston paths when
//! parameters are supplied). Under homogeneous dynamics the pure
//! cliquet has zero spot delta; the output reports Monte Carlo
//! standard errors instead of spot Greeks.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::errors::RustyQLibError;
use crate::core::montecarlo::{mean_std_err, path_rng};
use crate::core::traits::Instrument;
use crate::core::utils::norm_cdf;
use crate::equity::heston::HestonParams;
use rand::Rng;
use rand_distr::StandardNormal;

/// Pricing engine choice for a cliquet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CliquetPricer {
    Analytical,
    MonteCarlo,
}

/// Payoff family on the reset schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CliquetStyle {
    /// Sum of locally clamped returns (the classic cliquet; local
    /// floor 0 = ratchet).
    Standard,
    /// Reverse cliquet: a headline `coupon` eroded by the negative
    /// period returns, `coupon + sum_i min(R_i, 0)` — conventionally
    /// sold with `global_floor = 0`. Ignores the local clamp fields.
    Reverse { coupon: f64 },
    /// Napoleon: a `coupon` plus the **worst** period return,
    /// `coupon + min_i R_i` — conventionally `global_floor = 0`.
    /// Ignores the local clamp fields.
    Napoleon { coupon: f64 },
}

/// JSON contract data (`"product_type": "cliquet_option"`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CliquetOptionData {
    pub symbol: String,
    /// Number of equally-spaced reset periods.
    pub resets: usize,
    /// Maturity date, `YYYY-MM-DD`.
    pub maturity: String,
    /// Per-period floor on the return (e.g. `0.0` for a ratchet).
    pub local_floor: f64,
    /// Per-period cap on the return.
    pub local_cap: Option<f64>,
    pub global_floor: Option<f64>,
    pub global_cap: Option<f64>,
    pub notional: Option<f64>,
    pub risk_free_rate: f64,
    pub dividend: Option<f64>,
    /// Flat Black-Scholes volatility (GBM engine and the closed form).
    pub volatility: f64,
    /// Optional Heston parameters: when present the Monte Carlo engine
    /// simulates Heston dynamics instead of GBM.
    pub heston: Option<HestonParams>,
    pub pricer: Option<String>,
    pub simulation: Option<u64>,
    pub mc_seed: Option<u64>,
    /// "standard" (default) | "reverse" | "napoleon".
    pub style: Option<String>,
    /// Headline coupon for reverse / napoleon styles.
    pub coupon: Option<f64>,
    /// Pricing as-of date (`YYYY-MM-DD`); defaults to today.
    pub valuation_date: Option<String>,
}

/// A cliquet/ratchet option on equally-spaced resets.
#[derive(Debug, Clone)]
pub struct Cliquet {
    pub resets: usize,
    /// Year fraction to maturity.
    pub t: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub local_floor: f64,
    pub local_cap: Option<f64>,
    pub global_floor: Option<f64>,
    pub global_cap: Option<f64>,
    pub notional: f64,
    pub heston: Option<HestonParams>,
    pub style: CliquetStyle,
    pub pricer: CliquetPricer,
    pub paths: usize,
    pub seed: u64,
}

/// Euler substeps per reset period for the Heston path engine.
const HESTON_SUBSTEPS: usize = 32;

/// Per-period clamp, shared by the standalone product and
/// [`CliquetPayoff`].
fn clamp_local(ret: f64, floor: f64, cap: Option<f64>) -> f64 {
    let mut x = ret.max(floor);
    if let Some(cap) = cap {
        x = x.min(cap);
    }
    x
}

/// Terminal clamp of the accumulated units.
fn clamp_global(sum: f64, floor: Option<f64>, cap: Option<f64>) -> f64 {
    let mut x = sum;
    if let Some(floor) = floor {
        x = x.max(floor);
    }
    if let Some(cap) = cap {
        x = x.min(cap);
    }
    x
}

/// Fold one period return into the running style state
/// (`total` for standard/reverse, `worst` for the Napoleon).
fn accumulate_return(
    style: &CliquetStyle,
    local_floor: f64,
    local_cap: Option<f64>,
    ret: f64,
    total: &mut f64,
    worst: &mut f64,
) {
    match style {
        CliquetStyle::Standard => *total += clamp_local(ret, local_floor, local_cap),
        CliquetStyle::Reverse { .. } => *total += ret.min(0.0),
        CliquetStyle::Napoleon { .. } => *worst = worst.min(ret),
    }
}

/// Combine the accumulated state into payoff units per the style.
fn style_units(
    style: &CliquetStyle,
    total: f64,
    worst: f64,
    global_floor: Option<f64>,
    global_cap: Option<f64>,
) -> f64 {
    match style {
        CliquetStyle::Standard => clamp_global(total, global_floor, global_cap),
        CliquetStyle::Reverse { coupon } => clamp_global(coupon + total, global_floor, global_cap),
        CliquetStyle::Napoleon { coupon } => clamp_global(coupon + worst, global_floor, global_cap),
    }
}

impl Cliquet {
    fn accumulate(&self, ret: f64, total: &mut f64, worst: &mut f64) {
        accumulate_return(
            &self.style,
            self.local_floor,
            self.local_cap,
            ret,
            total,
            worst,
        );
    }

    /// Black-Scholes closed form: each period's clamped return is a
    /// forward-start call spread on the lognormal period ratio, and the
    /// periods are independent, so the sum prices term by term. Errs
    /// when a global cap/floor is present (it couples the periods) or
    /// Heston dynamics are requested.
    pub fn analytic_npv(&self) -> Result<f64, RustyQLibError> {
        if self.global_floor.is_some() || self.global_cap.is_some() {
            return Err(RustyQLibError::UnsupportedEngine(
                "global cap/floor couples the periods: use Monte Carlo".to_string(),
            ));
        }
        if self.heston.is_some() {
            return Err(RustyQLibError::UnsupportedEngine(
                "Heston cliquets price by Monte Carlo".to_string(),
            ));
        }
        let dt = self.t / self.resets as f64;
        let fwd = ((self.r - self.q) * dt).exp();
        let sd = self.sigma * dt.sqrt();
        // undiscounted E[(X - k)+] for the lognormal period ratio X
        let ratio_call = |k: f64| -> f64 {
            let d1 = ((fwd / k).ln() + 0.5 * sd * sd) / sd;
            fwd * norm_cdf(d1) - k * norm_cdf(d1 - sd)
        };
        let df_n = self.notional * (-self.r * self.t).exp();
        match self.style {
            CliquetStyle::Standard => {
                // E[clamp(R, lf, lc)] = lf + call(1 + lf) - call(1 + lc)
                let mut period = self.local_floor + ratio_call(1.0 + self.local_floor);
                if let Some(cap) = self.local_cap {
                    period -= ratio_call(1.0 + cap);
                }
                Ok(df_n * self.resets as f64 * period)
            }
            CliquetStyle::Reverse { coupon } => {
                // E[min(R, 0)] = -E[(1 - X)+], a put on the period ratio
                // struck at 1, via parity: put(1) = call(1) - (F - 1)
                let ratio_put_at_one = ratio_call(1.0) - (fwd - 1.0);
                Ok(df_n * (coupon - self.resets as f64 * ratio_put_at_one))
            }
            CliquetStyle::Napoleon { .. } => Err(RustyQLibError::UnsupportedEngine(
                "the Napoleon's worst-of statistic prices by Monte Carlo".to_string(),
            )),
        }
    }

    /// Monte Carlo price with standard error: per-period lognormal
    /// sampling under GBM, full Euler paths under Heston. Deterministic
    /// per seed.
    pub fn mc_npv(&self) -> (f64, f64) {
        let dt = self.t / self.resets as f64;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..self.paths {
            let mut rng = path_rng(self.seed, i as u64);
            let mut total = 0.0;
            let mut worst = f64::INFINITY;
            match &self.heston {
                None => {
                    let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt;
                    let sd = self.sigma * dt.sqrt();
                    for _ in 0..self.resets {
                        let z: f64 = rng.sample(StandardNormal);
                        let ret = (drift + sd * z).exp() - 1.0;
                        self.accumulate(ret, &mut total, &mut worst);
                    }
                }
                Some(hp) => {
                    let sub = dt / HESTON_SUBSTEPS as f64;
                    let rho_bar = (1.0 - hp.rho * hp.rho).sqrt();
                    let mut v: f64 = hp.v0;
                    for _ in 0..self.resets {
                        let mut log_ret = 0.0;
                        for _ in 0..HESTON_SUBSTEPS {
                            let z1: f64 = rng.sample(StandardNormal);
                            let z2: f64 = rng.sample(StandardNormal);
                            let zv = hp.rho * z1 + rho_bar * z2;
                            let vp = v.max(0.0);
                            log_ret += (self.r - self.q - 0.5 * vp) * sub + (vp * sub).sqrt() * z1;
                            v += hp.kappa * (hp.theta - vp) * sub
                                + hp.vol_of_vol * (vp * sub).sqrt() * zv;
                        }
                        self.accumulate(log_ret.exp() - 1.0, &mut total, &mut worst);
                    }
                }
            }
            let units = style_units(&self.style, total, worst, self.global_floor, self.global_cap);
            let payoff = self.notional * (-self.r * self.t).exp() * units;
            sum += payoff;
            sum_sq += payoff * payoff;
        }
        mean_std_err(sum, sum_sq, self.paths)
    }

    /// Price with the configured engine (analytic falls back to Monte
    /// Carlo when global constraints or Heston dynamics require it).
    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`Cliquet::try_from_json`].
    pub fn from_json(data: &CliquetOptionData) -> Box<Cliquet> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &CliquetOptionData) -> Result<Box<Cliquet>, RustyQLibError> {
        let today = crate::core::data_models::parse_valuation_date(data.valuation_date.as_deref())?;
        let maturity = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").map_err(|_| {
            RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            )
        })?;
        let t = crate::equity::conventions::year_fraction(today, maturity);
        if t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "maturity",
                "cliquet is expired",
            ));
        }
        if data.resets < 1 {
            return Err(RustyQLibError::invalid_input(
                "resets",
                "need at least one reset period",
            ));
        }
        if !(data.volatility.is_finite() && data.volatility > 0.0) {
            // sigma = 0 divides by sd = 0 in the per-period closed form
            return Err(RustyQLibError::invalid_input(
                "volatility",
                format!(
                    "volatility must be positive and finite, got {}",
                    data.volatility
                ),
            ));
        }
        crate::equity::conventions::check_vol_band("volatility", data.volatility)?;
        crate::equity::conventions::check_rate_band("risk_free_rate", data.risk_free_rate)?;
        if let Some(notional) = data.notional {
            if !(notional.is_finite() && notional > 0.0) {
                return Err(RustyQLibError::invalid_input(
                    "notional",
                    format!("notional must be positive and finite, got {notional}"),
                ));
            }
        }
        if let Some(hp) = &data.heston {
            hp.validate()?;
        }
        let pricer = match data.pricer.as_deref().map(str::trim) {
            None | Some("Analytical") | Some("analytical") | Some("bs") => {
                CliquetPricer::Analytical
            }
            Some("MonteCarlo") | Some("montecarlo") | Some("MC") | Some("mc") => {
                CliquetPricer::MonteCarlo
            }
            Some(other) => {
                return Err(RustyQLibError::invalid_input(
                    "pricer",
                    format!("invalid cliquet pricer '{other}' (use Analytical or MonteCarlo)"),
                ))
            }
        };
        let need_coupon = |style: &str| {
            data.coupon.ok_or_else(|| {
                RustyQLibError::invalid_input("coupon", format!("{style} cliquet needs a coupon"))
            })
        };
        let style = match data.style.as_deref().map(str::trim) {
            None | Some("standard") | Some("Standard") => CliquetStyle::Standard,
            Some("reverse") | Some("Reverse") => CliquetStyle::Reverse {
                coupon: need_coupon("reverse")?,
            },
            Some("napoleon") | Some("Napoleon") => CliquetStyle::Napoleon {
                coupon: need_coupon("napoleon")?,
            },
            Some(other) => {
                return Err(RustyQLibError::invalid_input(
                    "style",
                    format!("invalid cliquet style '{other}' (use standard, reverse or napoleon)"),
                ))
            }
        };
        Ok(Box::new(Cliquet {
            resets: data.resets,
            t,
            r: data.risk_free_rate,
            q: data.dividend.unwrap_or(0.0),
            sigma: data.volatility,
            local_floor: data.local_floor,
            local_cap: data.local_cap,
            global_floor: data.global_floor,
            global_cap: data.global_cap,
            notional: data.notional.unwrap_or(1.0),
            heston: data.heston,
            style,
            pricer,
            paths: data.simulation.unwrap_or(100_000) as usize,
            seed: data.mc_seed.unwrap_or(42),
        }))
    }
}

/// The cliquet as a mainline [`Payoff`], pricing inside
/// [`EquityOption`](crate::equity::vanilla_option::EquityOption) on the
/// shared Monte Carlo engine (GBM, local vol, Heston via QE-M, SABR,
/// rough Bergomi) — which gives it the market context for free: curve
/// discounting, dividend handling, `with_market`/`npv_in` rebinding,
/// portfolio membership and the stress runner. Spot, curve and surface
/// come from the bound market; the strike is unused (the payoff is
/// built from period returns and is spot-homogeneous). The standalone
/// [`Cliquet`] remains the flat-market closed-form / per-period-MC
/// validation reference.
///
/// Period returns observe the path on an equally spaced reset grid,
/// with the first return measured against `initial_fixing` (the spot
/// the option was built at). The payoff pays once, at maturity, so
/// like the other schedule products it is valued per path through
/// [`path_value`](Self::path_value) with per-date discount factors,
/// not through `path_payoff`.
#[derive(Debug, Clone)]
pub struct CliquetPayoff {
    pub exercise_style: crate::core::utils::ContractStyle,
    /// Number of equally spaced reset periods (last reset = maturity).
    pub resets: usize,
    pub local_floor: f64,
    pub local_cap: Option<f64>,
    pub global_floor: Option<f64>,
    pub global_cap: Option<f64>,
    pub notional: f64,
    pub style: CliquetStyle,
    /// Spot at inception — the denominator of the first period return.
    pub initial_fixing: f64,
}

impl CliquetPayoff {
    /// Value of one simulated path: period returns between consecutive
    /// observation spots (the first against `initial_fixing`), folded
    /// per the style, globally clamped, paid at maturity. `obs_idx`
    /// maps reset m to its path step; `dfs[m]` discounts its date
    /// (only the final, maturity discount factor is used — the grid
    /// contract is shared with the other schedule payoffs).
    pub fn path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64]) -> f64 {
        let mut total = 0.0;
        let mut worst = f64::INFINITY;
        let mut s_prev = self.initial_fixing;
        for &idx in obs_idx {
            let s = path[idx];
            accumulate_return(
                &self.style,
                self.local_floor,
                self.local_cap,
                s / s_prev - 1.0,
                &mut total,
                &mut worst,
            );
            s_prev = s;
        }
        let units = style_units(&self.style, total, worst, self.global_floor, self.global_cap);
        self.notional * units * dfs.last().copied().unwrap_or(1.0)
    }
}

impl crate::equity::utils::Payoff for CliquetPayoff {
    /// Degenerate single-point value: zero (all value lives in the
    /// period returns of the path).
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, _path: &[f64], _strike: f64) -> f64 {
        panic!(
            "Cliquets observe scheduled resets and cannot be valued through \
             path_payoff; the Monte Carlo engine prices them via path_value"
        );
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> crate::equity::utils::PayoffType {
        crate::equity::utils::PayoffType::Cliquet
    }
    fn put_or_call(&self) -> &crate::core::trade::PutOrCall {
        // by convention: the standard cliquet is a strip of forward-start
        // call spreads; not used by pricing
        &crate::core::trade::PutOrCall::Call
    }
    fn exercise_style(&self) -> &crate::core::utils::ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn crate::equity::utils::Payoff> {
        Box::new(self.clone())
    }
}

impl Instrument for Cliquet {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.price()?.pv)
    }

    /// Analytic where the payoff permits (falling back to Monte Carlo
    /// otherwise, mirroring the documented pricing policy); the standard
    /// error is reported whenever a simulation produced the value.
    fn price(&self) -> Result<crate::core::results::PricingResult, RustyQLibError> {
        let (pv, std_err) = match self.pricer {
            CliquetPricer::Analytical => match self.analytic_npv() {
                Ok(v) => (v, None),
                Err(_) => {
                    let (pv, se) = self.mc_npv();
                    (pv, Some(se))
                }
            },
            CliquetPricer::MonteCarlo => {
                let (pv, se) = self.mc_npv();
                (pv, Some(se))
            }
        };
        // the return-based payoff is spot-homogeneous: no spot Greeks
        Ok(crate::core::results::PricingResult {
            pv,
            greeks: Default::default(),
            std_err,
            asset_greeks: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::blackscholes::bs_price;

    fn base() -> Cliquet {
        Cliquet {
            resets: 12,
            t: 1.0,
            r: 0.03,
            q: 0.01,
            sigma: 0.2,
            local_floor: 0.0,
            local_cap: Some(0.03),
            global_floor: None,
            global_cap: None,
            notional: 1.0,
            heston: None,
            style: CliquetStyle::Standard,
            pricer: CliquetPricer::Analytical,
            paths: 60_000,
            seed: 42,
        }
    }

    #[test]
    fn single_period_ratchet_is_a_scaled_atm_call() {
        // one reset, floor 0, no cap: pays (S_T/S_0 - 1)+ = ATM call / S0
        let mut c = base();
        c.resets = 1;
        c.local_cap = None;
        let analytic = c.analytic_npv().unwrap();
        let atm = bs_price(100.0, 100.0, c.r, c.q, c.sigma, c.t, PutOrCall::Call) / 100.0;
        assert!((analytic - atm).abs() < 1e-12, "{analytic} vs {atm}");
    }

    #[test]
    fn monte_carlo_agrees_with_the_closed_form() {
        let c = base();
        let analytic = c.analytic_npv().unwrap();
        let (mc, se) = c.mc_npv();
        assert!(
            (mc - analytic).abs() < 3.0 * se + 1e-4,
            "mc {mc} +/- {se} vs analytic {analytic}"
        );
        // deterministic per seed
        assert_eq!(c.mc_npv().0, mc);
    }

    #[test]
    fn caps_and_floors_move_the_price_the_right_way() {
        let c = base();
        let baseline = c.analytic_npv().unwrap();
        // a tighter local cap must cheapen the strip
        let mut tight = base();
        tight.local_cap = Some(0.01);
        assert!(tight.analytic_npv().unwrap() < baseline);
        // a global floor adds value, a global cap removes it (MC only)
        let mut floored = base();
        floored.global_floor = Some(0.06);
        assert!(floored.analytic_npv().is_err());
        assert!(floored.mc_npv().0 > c.mc_npv().0 - 1e-12);
        let mut capped = base();
        capped.global_cap = Some(0.10);
        assert!(capped.mc_npv().0 < c.mc_npv().0);
        // local floor above the cap is degenerate but bounded
        let mut sunk = base();
        sunk.local_floor = -0.02;
        assert!(sunk.analytic_npv().unwrap() < baseline);
    }

    #[test]
    fn heston_with_tiny_vol_of_vol_matches_the_gbm_closed_form() {
        let mut c = base();
        c.heston = Some(HestonParams {
            v0: c.sigma * c.sigma,
            kappa: 1.0,
            theta: c.sigma * c.sigma,
            vol_of_vol: 1e-4,
            rho: 0.0,
        });
        c.paths = 40_000;
        let (mc, se) = c.mc_npv();
        let analytic = base().analytic_npv().unwrap();
        assert!(
            (mc - analytic).abs() < 4.0 * se + 5e-4,
            "heston-degenerate {mc} +/- {se} vs analytic {analytic}"
        );
    }

    #[test]
    fn heston_forward_smile_moves_the_cliquet_off_black_scholes() {
        // same total variance, real vol-of-vol and negative rho: the
        // capped/floored strip prices differently from flat-vol GBM —
        // the whole reason cliquets are priced on stochastic vol
        let mut c = base();
        c.heston = Some(HestonParams {
            v0: 0.04,
            kappa: 1.5,
            theta: 0.04,
            vol_of_vol: 0.7,
            rho: -0.7,
        });
        c.paths = 60_000;
        let (mc, se) = c.mc_npv();
        let flat = base().analytic_npv().unwrap();
        assert!(
            (mc - flat).abs() > 3.0 * se,
            "expected a forward-smile effect: heston {mc} +/- {se} vs bs {flat}"
        );
    }

    #[test]
    fn reverse_cliquet_closed_form_matches_monte_carlo() {
        // unfloored reverse: coupon minus a strip of forward-start puts
        let mut c = base();
        c.style = CliquetStyle::Reverse { coupon: 0.20 };
        c.local_cap = None;
        let analytic = c.analytic_npv().unwrap();
        let (mc, se) = c.mc_npv();
        assert!(
            (mc - analytic).abs() < 3.0 * se + 1e-4,
            "mc {mc} +/- {se} vs analytic {analytic}"
        );
        // the conventional 0% floor only adds value
        let mut floored = c.clone();
        floored.global_floor = Some(0.0);
        assert!(floored.analytic_npv().is_err());
        assert!(floored.mc_npv().0 >= mc - 1e-12);
    }

    #[test]
    fn single_period_napoleon_is_a_forward_start_call() {
        // one reset, floor 0: max(0, C + R) = (X - (1 - C))+ on the ratio
        let coupon = 0.10;
        let mut c = base();
        c.resets = 1;
        c.local_cap = None;
        c.style = CliquetStyle::Napoleon { coupon };
        c.global_floor = Some(0.0);
        c.paths = 200_000;
        let (mc, se) = c.mc_npv();
        // a call on the unit-spot period ratio struck at 1 - C, already
        // discounted by bs_price
        let analytic = bs_price(1.0, 1.0 - coupon, c.r, c.q, c.sigma, c.t, PutOrCall::Call);
        assert!(
            (mc - analytic).abs() < 3.0 * se + 1e-4,
            "mc {mc} +/- {se} vs analytic {analytic}"
        );
    }

    #[test]
    fn napoleon_worsens_with_more_resets_and_higher_vol() {
        let napoleon = |resets: usize, sigma: f64| -> f64 {
            let mut c = base();
            c.resets = resets;
            c.t = 1.0;
            c.sigma = sigma;
            c.local_cap = None;
            c.style = CliquetStyle::Napoleon { coupon: 0.10 };
            c.global_floor = Some(0.0);
            c.paths = 30_000;
            c.mc_npv().0
        };
        // the minimum of more period returns is worse
        assert!(napoleon(1, 0.2) > napoleon(4, 0.2));
        assert!(napoleon(4, 0.2) > napoleon(12, 0.2));
        // and the structure is short volatility
        assert!(napoleon(12, 0.15) > napoleon(12, 0.30));
    }

    #[test]
    fn napoleon_vol_of_vol_exposure_is_large_and_directionally_convex() {
        // Same total variance, real vol-of-vol. Note the direction: the
        // FLOORED Napoleon is convex in the worst-month distribution (the
        // 0% floor truncates the fat left tail, while calm-vol regimes
        // improve the worst month), so vol-of-vol RAISES the buyer's
        // value here — the famous Napoleon blowups were the sellers'
        // short position in exactly this convexity. The unfloored
        // structure, by contrast, is hurt by vol clustering.
        let mut gbm = base();
        gbm.style = CliquetStyle::Napoleon { coupon: 0.10 };
        gbm.local_cap = None;
        gbm.global_floor = Some(0.0);
        gbm.paths = 60_000;
        let (flat, flat_se) = gbm.mc_npv();
        let hp = HestonParams {
            v0: 0.04,
            kappa: 1.5,
            theta: 0.04,
            vol_of_vol: 0.7,
            rho: -0.7,
        };
        let mut heston = gbm.clone();
        heston.heston = Some(hp);
        let (stoch, stoch_se) = heston.mc_npv();
        let noise = (flat_se * flat_se + stoch_se * stoch_se).sqrt();
        assert!(
            stoch > flat + 5.0 * noise,
            "floored: heston {stoch} vs gbm {flat}"
        );

        // without the floor two effects compete (worst-month concavity
        // vs the right-skewed CIR variance making the typical month
        // calmer), so no sign is asserted — only that the model choice
        // moves the price by far more than the Monte Carlo noise
        let mut gbm_unfloored = gbm.clone();
        gbm_unfloored.global_floor = None;
        let (flat_u, se_u) = gbm_unfloored.mc_npv();
        let mut heston_unfloored = gbm_unfloored.clone();
        heston_unfloored.heston = Some(hp);
        let (stoch_u, se_u2) = heston_unfloored.mc_npv();
        let noise_u = (se_u * se_u + se_u2 * se_u2).sqrt();
        assert!(
            (stoch_u - flat_u).abs() > 5.0 * noise_u,
            "unfloored: heston {stoch_u} vs gbm {flat_u} (noise {noise_u})"
        );
    }

    #[test]
    fn styled_json_contracts_parse() {
        let json = r#"{
            "symbol": "NAP", "resets": 12, "maturity": "2030-01-01",
            "local_floor": 0.0, "global_floor": 0.0,
            "risk_free_rate": 0.03, "volatility": 0.2,
            "style": "napoleon", "coupon": 0.08,
            "pricer": "MC", "simulation": 5000
        }"#;
        let data: CliquetOptionData = serde_json::from_str(json).unwrap();
        let napoleon = Cliquet::from_json(&data);
        assert_eq!(napoleon.style, CliquetStyle::Napoleon { coupon: 0.08 });
        let pv = napoleon.npv();
        assert!(pv > 0.0 && pv < 0.08, "{pv}"); // bounded by the coupon
    }

    #[test]
    fn json_contract_round_trip() {
        let json = r#"{
            "symbol": "CLIQ", "resets": 4, "maturity": "2030-01-01",
            "local_floor": 0.0, "local_cap": 0.05, "global_floor": 0.02,
            "notional": 1000000.0, "risk_free_rate": 0.03, "dividend": 0.01,
            "volatility": 0.25, "pricer": "MC", "simulation": 20000, "mc_seed": 7
        }"#;
        let data: CliquetOptionData = serde_json::from_str(json).unwrap();
        let cliquet = Cliquet::from_json(&data);
        assert_eq!(cliquet.resets, 4);
        assert_eq!(cliquet.pricer, CliquetPricer::MonteCarlo);
        let pv = cliquet.npv();
        // bounded by the discounted global-capped maximum
        assert!(pv > 0.0 && pv < 1_000_000.0 * 4.0 * 0.05, "{pv}");
    }

    // ── the mainline CliquetPayoff (Market-bound spine) ────────────────

    fn builder_cliquet() -> crate::equity::builder::EquityOptionBuilder {
        use crate::equity::builder::EquityOptionBuilder;
        use crate::equity::utils::Engine;
        EquityOptionBuilder::new()
            .symbol("CLIQ")
            .spot(100.0)
            .flat_vol(0.2)
            .flat_rate(0.03)
            .valuation_date(chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(chrono::NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .cliquet(4, 0.0, 100.0)
            .cliquet_local_cap(0.08)
            .engine(Engine::MonteCarlo)
            .paths(100_000)
            .seed(42)
    }

    fn standalone_like_builder() -> Cliquet {
        Cliquet {
            resets: 4,
            t: 1.0,
            r: 0.03,
            q: 0.0,
            sigma: 0.2,
            local_floor: 0.0,
            local_cap: Some(0.08),
            global_floor: None,
            global_cap: None,
            notional: 100.0,
            heston: None,
            style: CliquetStyle::Standard,
            pricer: CliquetPricer::Analytical,
            paths: 60_000,
            seed: 42,
        }
    }

    #[test]
    fn mainline_cliquet_matches_the_standalone_closed_form() {
        // the builder route simulates full paths on the shared MC engine
        // (exact GBM steps => period returns exactly lognormal), so it
        // must converge to the standalone strip-of-forward-starts closed
        // form on the identical flat market
        let analytic = standalone_like_builder().analytic_npv().unwrap();
        let mc = builder_cliquet().build().expect("cliquet must build").npv();
        assert!(
            (mc - analytic).abs() < 0.05,
            "mainline MC {mc} vs closed form {analytic}"
        );
    }

    #[test]
    fn mainline_cliquet_is_spot_homogeneous() {
        // period returns are scale-invariant path by path, so the price
        // is exactly independent of the spot level (same seed and paths)
        let at_100 = builder_cliquet().build().unwrap().npv();
        let at_250 = builder_cliquet().spot(250.0).build().unwrap().npv();
        assert!(
            (at_100 - at_250).abs() < 1e-9,
            "spot 100: {at_100}, spot 250: {at_250}"
        );
    }

    #[test]
    fn mainline_reverse_cliquet_matches_the_closed_form() {
        let mut standalone = standalone_like_builder();
        standalone.style = CliquetStyle::Reverse { coupon: 0.15 };
        standalone.local_cap = None;
        let analytic = standalone.analytic_npv().unwrap();
        let mc = builder_cliquet()
            .cliquet(4, 0.0, 100.0)
            .cliquet_style(CliquetStyle::Reverse { coupon: 0.15 })
            .build()
            .expect("reverse cliquet must build")
            .npv();
        assert!(
            (mc - analytic).abs() < 0.05,
            "mainline MC {mc} vs closed form {analytic}"
        );
    }

    #[test]
    fn mainline_global_cap_binds() {
        let uncapped = builder_cliquet().build().unwrap().npv();
        let capped = builder_cliquet()
            .cliquet_global_cap(0.05)
            .build()
            .unwrap()
            .npv();
        assert!(
            capped < uncapped,
            "global cap must reduce value: capped {capped} vs {uncapped}"
        );
        // capped by the discounted global maximum
        assert!(capped <= 100.0 * 0.05 * (-0.03_f64).exp() + 1e-9);
    }

    #[test]
    fn mainline_heston_route_degenerates_to_gbm() {
        // v0 = theta = 0.2^2 with vanishing vol-of-vol: the Heston route
        // (through the shared SV payoff router) must land on the GBM
        // route's value up to sampler differences
        let gbm = builder_cliquet().build().unwrap().npv();
        let heston = builder_cliquet()
            .heston(crate::equity::heston::HestonParams {
                v0: 0.04,
                kappa: 2.0,
                theta: 0.04,
                vol_of_vol: 1e-4,
                rho: 0.0,
            })
            .build()
            .expect("heston cliquet must build")
            .npv();
        assert!(
            (heston - gbm).abs() < 0.05 * gbm.abs().max(1.0),
            "heston {heston} vs gbm {gbm}"
        );
    }

    #[test]
    fn mainline_cliquet_engine_and_input_validation() {
        use crate::core::errors::RustyQLibError;
        use crate::equity::utils::Engine;
        // the analytic engine refuses the payoff at build()
        let result = builder_cliquet().engine(Engine::BlackScholes).build();
        assert!(
            matches!(result, Err(RustyQLibError::UnsupportedEngine(_))),
            "cliquet on the analytic engine must be refused at build()"
        );
        // a local cap at or below the floor is rejected with the field
        match builder_cliquet().cliquet_local_cap(-0.01).build() {
            Err(RustyQLibError::InvalidInput { field, .. }) => assert_eq!(field, "local_cap"),
            other => panic!("expected local_cap error, got {:?}", other.map(|_| "an option")),
        }
        // modifiers without .cliquet(...) report the misuse
        match crate::equity::builder::EquityOptionBuilder::new()
            .spot(100.0)
            .flat_vol(0.2)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Call)
            .cliquet_global_cap(0.1)
            .build()
        {
            Err(RustyQLibError::InvalidInput { field, .. }) => {
                assert_eq!(field, "cliquet_global_cap")
            }
            other => panic!("expected setter error, got {:?}", other.map(|_| "an option")),
        }
    }
}

