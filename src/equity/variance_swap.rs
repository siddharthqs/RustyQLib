//! Volatility derivatives: variance swaps (and the volatility-swap
//! strike under GBM).
//!
//! A variance swap pays `notional * (realized variance - strike)` at
//! maturity, with realized variance the annualized mean of squared log
//! returns (**no mean subtraction** — the market convention). Its fair
//! strike is model-free by the log-contract replication
//! (Demeterfi-Derman-Kamal-Zou 1999):
//!
//! ```text
//! K_var = (2/T) * [ int_0^F P(K)/K^2 dK + int_F^inf C(K)/K^2 dK ]
//! ```
//!
//! with undiscounted OTM option prices struck off the forward. On a
//! flat surface the integral collapses to `sigma^2` **exactly** (the
//! test checks to 1e-6); a skewed smile adds the convexity that makes
//! variance strikes trade above ATM vol squared.
//!
//! The volatility swap (paying realized vol) needs the distribution,
//! not just the expectation: under GBM the exact fair strike is the
//! chi-distribution mean `sigma * sqrt(2/n) * Gamma((n+1)/2) /
//! Gamma(n/2)` — below `sigma` for finite sampling by Jensen.

use chrono::{Local, NaiveDate};
use serde::{Deserialize, Serialize};

use crate::core::traits::Instrument;
use crate::core::utils::norm_cdf;
use crate::core::vols::{VolInput, VolSurface};
use crate::core::errors::RustyQLibError;

/// Annualized realized variance of a log-return series, market
/// convention (mean **not** subtracted).
pub fn realized_variance(log_returns: &[f64], periods_per_year: f64) -> f64 {
    assert!(!log_returns.is_empty());
    log_returns.iter().map(|r| r * r).sum::<f64>() / log_returns.len() as f64
        * periods_per_year
}

/// Model-free fair variance strike by the log-contract replication.
/// `smile(strike) -> implied vol`; integration over ten ATM standard
/// deviations of log-strike with a fine Simpson rule.
pub fn fair_variance_strike(forward: f64, t: f64, smile: impl Fn(f64) -> f64) -> f64 {
    assert!(forward > 0.0 && t > 0.0);
    let atm_vol = smile(forward).max(1e-4);
    let width = (10.0 * atm_vol * t.sqrt()).max(1.0);
    let steps = 4000usize;
    let du = 2.0 * width / steps as f64;
    // undiscounted Black OTM price at strike K = F e^u
    let otm = |u: f64| -> f64 {
        let k = forward * u.exp();
        let sigma = smile(k).max(1e-6);
        let st = sigma * t.sqrt();
        let d1 = ((forward / k).ln() + 0.5 * st * st) / st;
        let d2 = d1 - st;
        if k >= forward {
            forward * norm_cdf(d1) - k * norm_cdf(d2) // call
        } else {
            k * norm_cdf(-d2) - forward * norm_cdf(-d1) // put
        }
    };
    // int Q(K)/K^2 dK = int Q(F e^u) e^{-u} du / F  (Simpson)
    let mut sum = 0.0;
    for i in 0..=steps {
        let u = -width + i as f64 * du;
        let w = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * otm(u) * (-u).exp();
    }
    let integral = sum * du / 3.0 / forward;
    2.0 / t * integral
}

/// Fair **gamma-swap** strike: the spot-weighted variance
/// `(1/T) int (S_t/S_0) sigma_t^2 dt`, replicated by the `S ln S`
/// contract with a `1/K` strike kernel:
///
/// ```text
/// K_gamma = (2/(T S_0)) * [ int_0^F P(K)/K dK + int_F^inf C(K)/K dK ]
///           * phi(bT),   phi(x) = (1 - e^-x)/x
/// ```
///
/// The `phi` factor is the carry adjustment for the drift-weighting
/// interaction: it makes the flat-vol strike exactly
/// `sigma^2 (e^{bT} - 1)/(bT)` for any carry `b` (tested), and is exact
/// for any smile when `b = 0`. Gamma swaps weight down-moves by a low
/// `S/S_0`, so under a put skew the gamma strike sits **below** the
/// variance strike — the crash-discount that motivates the product.
pub fn fair_gamma_swap_strike(
    spot: f64,
    forward: f64,
    t: f64,
    smile: impl Fn(f64) -> f64,
) -> f64 {
    assert!(spot > 0.0 && forward > 0.0 && t > 0.0);
    let atm_vol = smile(forward).max(1e-4);
    let width = (10.0 * atm_vol * t.sqrt()).max(1.0);
    let steps = 4000usize;
    let du = 2.0 * width / steps as f64;
    let otm = |u: f64| -> f64 {
        let k = forward * u.exp();
        let sigma = smile(k).max(1e-6);
        let st = sigma * t.sqrt();
        let d1 = ((forward / k).ln() + 0.5 * st * st) / st;
        let d2 = d1 - st;
        if k >= forward { forward * norm_cdf(d1) - k * norm_cdf(d2) } else { k * norm_cdf(-d2) - forward * norm_cdf(-d1) }
    };
    // int Q(K)/K dK = int Q(F e^u) du  (log-strike substitution)
    let mut sum = 0.0;
    for i in 0..=steps {
        let u = -width + i as f64 * du;
        let w = if i == 0 || i == steps { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * otm(u);
    }
    let integral = sum * du / 3.0;
    let b_t = (forward / spot).ln();
    let phi = if b_t.abs() < 1e-12 { 1.0 } else { (1.0 - (-b_t).exp()) / b_t };
    2.0 / (t * spot) * integral * phi
}

/// Fair **corridor variance** strike: variance accrues only while the
/// spot is inside `[low, high]` (Carr-Lewis), which truncates the
/// replication integral to the corridor's strikes:
/// `K_corr = (2/T) int_low^high Q(K)/K^2 dK`. Corridors are exactly
/// additive: adjacent corridors sum to the full variance strike
/// (tested), and the full-line corridor reproduces
/// [`fair_variance_strike`].
pub fn fair_corridor_variance_strike(
    forward: f64,
    t: f64,
    low: f64,
    high: f64,
    smile: impl Fn(f64) -> f64,
) -> f64 {
    assert!(forward > 0.0 && t > 0.0 && low >= 0.0 && high > low);
    let atm_vol = smile(forward).max(1e-4);
    let width = (10.0 * atm_vol * t.sqrt()).max(1.0);
    // integrate in log-strike over the corridor clipped to the window
    let u_lo = if low <= 0.0 { -width } else { (low / forward).ln().max(-width) };
    let u_hi = if high.is_infinite() { width } else { (high / forward).ln().min(width) };
    if u_hi <= u_lo {
        return 0.0;
    }
    let steps = 4000usize;
    let du = (u_hi - u_lo) / steps as f64;
    let otm = |u: f64| -> f64 {
        let k = forward * u.exp();
        let sigma = smile(k).max(1e-6);
        let st = sigma * t.sqrt();
        let d1 = ((forward / k).ln() + 0.5 * st * st) / st;
        let d2 = d1 - st;
        if k >= forward { forward * norm_cdf(d1) - k * norm_cdf(d2) } else { k * norm_cdf(-d2) - forward * norm_cdf(-d1) }
    };
    let mut sum = 0.0;
    for i in 0..=steps {
        let u = u_lo + i as f64 * du;
        let w = if i == 0 || i == steps { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * otm(u) * (-u).exp();
    }
    let integral = sum * du / 3.0 / forward;
    2.0 / t * integral
}

/// Realized leg of a gamma swap over a spot path: the annualized
/// spot-weighted squared returns `(A/n) sum (S_i/S_0) ln(S_i/S_{i-1})^2`.
pub fn realized_gamma_variance(spots: &[f64], s0: f64, periods_per_year: f64) -> f64 {
    assert!(spots.len() >= 2 && s0 > 0.0);
    let n = spots.len() - 1;
    let sum: f64 = spots
        .windows(2)
        .map(|w| {
            let r = (w[1] / w[0]).ln();
            w[1] / s0 * r * r
        })
        .sum();
    sum / n as f64 * periods_per_year
}

/// Realized leg of a corridor variance swap: squared returns accrue
/// when the **previous** observation was inside `[low, high]` (the
/// standard convention).
pub fn realized_corridor_variance(
    spots: &[f64],
    low: f64,
    high: f64,
    periods_per_year: f64,
) -> f64 {
    assert!(spots.len() >= 2 && high > low);
    let n = spots.len() - 1;
    let sum: f64 = spots
        .windows(2)
        .map(|w| {
            if w[0] >= low && w[0] <= high {
                let r = (w[1] / w[0]).ln();
                r * r
            } else {
                0.0
            }
        })
        .sum();
    sum / n as f64 * periods_per_year
}

/// Exact fair **volatility**-swap strike under GBM with `observations`
/// sampling dates: `sigma sqrt(2/n) Gamma((n+1)/2)/Gamma(n/2)`, the
/// mean of the chi distribution — strictly below `sigma`, converging to
/// it as sampling densifies.
pub fn volatility_swap_strike_gbm(sigma: f64, observations: usize) -> f64 {
    assert!(sigma > 0.0 && observations >= 1);
    let n = observations as f64;
    let log_ratio = libm::lgamma((n + 1.0) / 2.0) - libm::lgamma(n / 2.0);
    sigma * (2.0 / n).sqrt() * log_ratio.exp()
}

/// JSON contract data (`"product_type": "variance_swap"`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VarianceSwapData {
    pub symbol: String,
    pub underlying_price: f64,
    /// Strike quoted in **volatility** units (0.20 = 20 vol);
    /// `K_var = strike_vol^2`.
    pub strike_vol: f64,
    /// Variance notional (payout per unit of annualized variance).
    pub notional: f64,
    /// Maturity date, `YYYY-MM-DD`.
    pub maturity: String,
    pub risk_free_rate: f64,
    pub dividend: Option<f64>,
    /// Flat implied vol, used when no surface is given.
    pub volatility: f64,
    /// Optional smile/surface: the fair strike integrates over it.
    pub vol_surface: Option<VolInput>,
    /// Seasoned swaps: annualized variance realized so far.
    pub accrued_variance: Option<f64>,
    /// Seasoned swaps: elapsed observation time in years.
    pub elapsed: Option<f64>,
    /// "variance" (default) | "gamma" | "corridor".
    pub swap_type: Option<String>,
    /// Corridor bounds (corridor swaps only; either may be omitted for
    /// a one-sided corridor).
    pub corridor_low: Option<f64>,
    pub corridor_high: Option<f64>,
}

/// A (possibly seasoned) variance swap.
#[derive(Debug, Clone)]
pub struct VarianceSwap {
    /// Variance notional.
    pub notional: f64,
    /// Strike in variance units.
    pub strike_variance: f64,
    /// Remaining time to maturity (years).
    pub t_remaining: f64,
    pub r: f64,
    /// Fair strike of the **remaining** variance (annualized).
    pub fair_remaining_variance: f64,
    /// (elapsed years, annualized variance realized over them).
    pub accrued: Option<(f64, f64)>,
}

impl VarianceSwap {
    /// Expected total-period annualized variance: the time-weighted
    /// blend of what has been realized and the fair value of the rest.
    pub fn expected_total_variance(&self) -> f64 {
        match self.accrued {
            None => self.fair_remaining_variance,
            Some((elapsed, accrued)) => {
                let total = elapsed + self.t_remaining;
                (elapsed * accrued + self.t_remaining * self.fair_remaining_variance) / total
            }
        }
    }

    /// Mark-to-market: discounted expected payoff.
    pub fn mtm(&self) -> f64 {
        self.notional
            * (-self.r * self.t_remaining).exp()
            * (self.expected_total_variance() - self.strike_variance)
    }

    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`VarianceSwap::try_from_json`].
    pub fn from_json(data: &VarianceSwapData) -> Box<VarianceSwap> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &VarianceSwapData) -> Result<Box<VarianceSwap>, RustyQLibError> {
        let today = Local::now().date_naive();
        let maturity = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d")
            .map_err(|_| RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            ))?;
        let t = (maturity - today).num_days() as f64 / 365.0;
        if t <= 0.0 {
            return Err(RustyQLibError::invalid_input("maturity", "variance swap is expired"));
        }
        let q = data.dividend.unwrap_or(0.0);
        let forward = data.underlying_price * ((data.risk_free_rate - q) * t).exp();
        let surface = data
            .vol_surface
            .as_ref()
            .map(|input| VolSurface::from_input(input, today))
            .transpose()?;
        let flat = data.volatility;
        let smile = |k: f64| match &surface {
            Some(s) => s.vol(k, forward, t),
            None => flat,
        };
        let fair = match data.swap_type.as_deref().map(str::trim) {
            None | Some("variance") => fair_variance_strike(forward, t, smile),
            Some("gamma") => {
                fair_gamma_swap_strike(data.underlying_price, forward, t, smile)
            }
            Some("corridor") => fair_corridor_variance_strike(
                forward,
                t,
                data.corridor_low.unwrap_or(0.0),
                data.corridor_high.unwrap_or(f64::INFINITY),
                smile,
            ),
            Some(other) => return Err(RustyQLibError::invalid_input(
                "swap_type",
                format!("invalid swap_type '{other}' (use variance, gamma or corridor)"),
            )),
        };
        let accrued = match (data.elapsed, data.accrued_variance) {
            (Some(e), Some(v)) => {
                if e < 0.0 || v < 0.0 {
                    return Err(RustyQLibError::invalid_input(
                        "accrued_variance",
                        "elapsed and accrued_variance must be non-negative",
                    ));
                }
                Some((e, v))
            }
            (None, None) => None,
            _ => return Err(RustyQLibError::invalid_input(
                "accrued_variance",
                "seasoned swaps need both elapsed and accrued_variance",
            )),
        };
        Ok(Box::new(VarianceSwap {
            notional: data.notional,
            strike_variance: data.strike_vol * data.strike_vol,
            t_remaining: t,
            r: data.risk_free_rate,
            fair_remaining_variance: fair,
            accrued,
        }))
    }
}

impl Instrument for VarianceSwap {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.mtm())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_surface_replication_recovers_sigma_squared_exactly() {
        for sigma in [0.1, 0.25, 0.6] {
            for t in [0.25, 1.0, 3.0] {
                let k_var = fair_variance_strike(100.0, t, |_| sigma);
                assert!(
                    (k_var - sigma * sigma).abs() < 1e-6,
                    "sigma {sigma} t {t}: {k_var} vs {}",
                    sigma * sigma
                );
            }
        }
    }

    #[test]
    fn skew_lifts_the_variance_strike_above_atm_squared() {
        // a put-skewed smile: OTM puts are priced richer than flat, and
        // the 1/K^2 weighting loads on them
        let atm = 0.2;
        let smile =
            |k: f64| atm - 0.15 * (k / 100.0 - 1.0) + 0.1 * (k / 100.0 - 1.0).powi(2);
        let k_var = fair_variance_strike(100.0, 1.0, smile);
        assert!(k_var > atm * atm * 1.02, "{k_var} vs {}", atm * atm);
        // and the SVI smile from the vol-model module plugs straight in
        let svi = crate::equity::svi::SviParams {
            a: 0.03,
            b: 0.12,
            rho: -0.4,
            m: -0.02,
            sigma: 0.3,
        };
        let k_svi = fair_variance_strike(100.0, 0.75, |k| svi.vol((k / 100.0_f64).ln(), 0.75));
        let atm_svi = svi.vol(0.0, 0.75);
        assert!(k_svi > atm_svi * atm_svi, "{k_svi} vs {}", atm_svi * atm_svi);
    }

    #[test]
    fn realized_leg_matches_convention_and_the_gbm_expectation() {
        // hand check: two returns of 1% at daily frequency
        let rv = realized_variance(&[0.01, -0.01], 252.0);
        assert!((rv - 252.0 * 0.0001).abs() < 1e-12);
        // under GBM the expected realized variance is sigma^2 (exactly,
        // including the drift-free convention); check by simulation
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let (sigma, n_days, n_paths) = (0.3, 252, 3000);
        let dt: f64 = 1.0 / 252.0;
        let mut sum = 0.0;
        for p in 0..n_paths {
            let mut rng = path_rng(11, p);
            let returns: Vec<f64> = (0..n_days)
                .map(|_| {
                    let z: f64 = rng.sample(rand_distr::StandardNormal);
                    (0.03 - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z
                })
                .collect();
            sum += realized_variance(&returns, 252.0);
        }
        let mean_rv = sum / n_paths as f64;
        // small positive drift bias of the convention is O(mu^2 dt)
        assert!((mean_rv - sigma * sigma).abs() < 0.002, "{mean_rv}");
    }

    #[test]
    fn seasoned_mtm_blends_accrued_and_remaining_variance() {
        let swap = VarianceSwap {
            notional: 1_000_000.0,
            strike_variance: 0.04,
            t_remaining: 0.5,
            r: 0.03,
            fair_remaining_variance: 0.05,
            accrued: Some((0.5, 0.09)), // a realized-vol spike of 30%
        };
        // blend: (0.5*0.09 + 0.5*0.05) / 1.0 = 0.07
        assert!((swap.expected_total_variance() - 0.07).abs() < 1e-12);
        let expected = 1_000_000.0 * (-0.03_f64 * 0.5).exp() * (0.07 - 0.04);
        assert!((swap.mtm() - expected).abs() < 1e-9);
        // a fresh swap struck at fair value has zero MtM
        let fresh = VarianceSwap {
            notional: 1_000_000.0,
            strike_variance: 0.05,
            t_remaining: 1.0,
            r: 0.03,
            fair_remaining_variance: 0.05,
            accrued: None,
        };
        assert!(fresh.mtm().abs() < 1e-9);
    }

    #[test]
    fn volatility_swap_strike_shows_the_jensen_gap_and_converges() {
        let sigma = 0.25;
        // finite sampling: strictly below sigma
        let k21 = volatility_swap_strike_gbm(sigma, 21);
        assert!(k21 < sigma, "{k21}");
        // exact chi-mean vs simulation at n = 21
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let mut sum = 0.0;
        let paths = 200_000;
        for p in 0..paths {
            let mut rng = path_rng(5, p);
            let mean_sq: f64 = (0..21)
                .map(|_| {
                    let z: f64 = rng.sample(rand_distr::StandardNormal);
                    z * z
                })
                .sum::<f64>()
                / 21.0;
            sum += sigma * mean_sq.sqrt();
        }
        let mc = sum / paths as f64;
        assert!((k21 - mc).abs() < 5e-4, "chi mean {k21} vs mc {mc}");
        // dense sampling converges up to sigma
        let k_dense = volatility_swap_strike_gbm(sigma, 100_000);
        assert!(sigma - k_dense < 1e-5 && k_dense < sigma);
        assert!(volatility_swap_strike_gbm(sigma, 252) > k21);
    }

    #[test]
    fn gamma_swap_flat_vol_matches_the_carry_closed_form() {
        // flat vol: K_gamma = sigma^2 (e^{bT} - 1)/(bT), exactly
        for (sigma, b, t) in [(0.2, 0.04, 1.0), (0.3, -0.02, 0.5), (0.25, 0.0, 2.0)] {
            let spot = 100.0_f64;
            let forward = spot * (b * t as f64).exp();
            let k_gamma = fair_gamma_swap_strike(spot, forward, t, |_| sigma);
            let expect = if b == 0.0 {
                sigma * sigma
            } else {
                sigma * sigma * ((b * t).exp() - 1.0) / (b * t)
            };
            assert!(
                (k_gamma - expect).abs() < 1e-6,
                "sigma {sigma} b {b} t {t}: {k_gamma} vs {expect}"
            );
        }
    }

    #[test]
    fn gamma_swap_matches_monte_carlo_and_discounts_the_crash_leg() {
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        // MC oracle: E[(A/n) sum (S_i/S0) r_i^2] under GBM
        let (sigma, b, t, s0) = (0.3, 0.04, 1.0, 100.0);
        let n_days = 252;
        let dt = t / n_days as f64;
        let mut sum = 0.0;
        let paths = 4000;
        for p in 0..paths {
            let mut rng = path_rng(23, p);
            let mut spots = vec![s0];
            for _ in 0..n_days {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                let prev = *spots.last().unwrap();
                spots.push(prev * ((b - 0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z).exp());
            }
            sum += realized_gamma_variance(&spots, s0, 252.0);
        }
        let mc = sum / paths as f64;
        let analytic = fair_gamma_swap_strike(s0, s0 * (b * t as f64).exp(), t, |_| sigma);
        assert!((mc - analytic).abs() < 0.004, "mc {mc} vs analytic {analytic}");

        // under a put skew the spot-weighting discounts crash variance:
        // gamma strike < variance strike
        let smile = |k: f64| 0.2 - 0.15 * (k / 100.0_f64 - 1.0);
        let k_var = fair_variance_strike(100.0, 1.0, smile);
        let k_gam = fair_gamma_swap_strike(100.0, 100.0, 1.0, smile);
        assert!(k_gam < k_var, "gamma {k_gam} vs variance {k_var}");
    }

    #[test]
    fn corridor_strikes_are_additive_and_recover_the_full_swap() {
        let smile = |k: f64| 0.2 - 0.1 * (k / 100.0_f64 - 1.0) + 0.2 * (k / 100.0_f64 - 1.0).powi(2);
        let full = fair_variance_strike(100.0, 1.0, smile);
        let below = fair_corridor_variance_strike(100.0, 1.0, 0.0, 90.0, smile);
        let middle = fair_corridor_variance_strike(100.0, 1.0, 90.0, 115.0, smile);
        let above = fair_corridor_variance_strike(100.0, 1.0, 115.0, f64::INFINITY, smile);
        // adjacent corridors tile the line: strikes add to the full swap
        assert!(
            (below + middle + above - full).abs() < 1e-6,
            "{below} + {middle} + {above} vs {full}"
        );
        // every corridor is a strict subset of the full variance
        for part in [below, middle, above] {
            assert!(part > 0.0 && part < full);
        }
        // the full-line corridor IS the variance swap
        let line = fair_corridor_variance_strike(100.0, 1.0, 0.0, f64::INFINITY, smile);
        assert!((line - full).abs() < 1e-9);
        // under put skew the downside corridor carries more variance than
        // the mirrored upside one
        let down = fair_corridor_variance_strike(100.0, 1.0, 70.0, 90.0, smile);
        let up = fair_corridor_variance_strike(100.0, 1.0, 111.0, 143.0, smile);
        assert!(down > up, "down {down} vs up {up}");
    }

    #[test]
    fn corridor_realized_leg_matches_a_monte_carlo_of_the_strike() {
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let (sigma, t, s0) = (0.25_f64, 1.0, 100.0);
        let (low, high) = (90.0, 115.0);
        let n_days = 504; // dense monitoring shrinks the indicator bias
        let dt = t / n_days as f64;
        let mut sum = 0.0;
        let paths = 4000;
        for p in 0..paths {
            let mut rng = path_rng(31, p);
            let mut spots = vec![s0];
            for _ in 0..n_days {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                let prev = *spots.last().unwrap();
                spots.push(prev * ((-0.5 * sigma * sigma) * dt + sigma * dt.sqrt() * z).exp());
            }
            sum += realized_corridor_variance(&spots, low, high, n_days as f64);
        }
        let mc = sum / paths as f64;
        let analytic = fair_corridor_variance_strike(s0, t, low, high, |_| sigma);
        assert!((mc - analytic).abs() < 0.003, "mc {mc} vs analytic {analytic}");
        // hand check of the accrual convention: only the step leaving the
        // corridor from inside counts
        let path = [100.0, 120.0, 110.0, 80.0, 85.0];
        let expect = ((120.0_f64 / 100.0).ln().powi(2) + (80.0_f64 / 110.0).ln().powi(2))
            / 4.0
            * 252.0;
        assert!((realized_corridor_variance(&path, 90.0, 115.0, 252.0) - expect).abs() < 1e-12);
    }

    #[test]
    fn json_contract_round_trip() {
        let json = r#"{
            "symbol": "VSWAP", "underlying_price": 100.0,
            "strike_vol": 0.22, "notional": 1000000.0,
            "maturity": "2030-01-01", "risk_free_rate": 0.03,
            "volatility": 0.25
        }"#;
        let data: VarianceSwapData = serde_json::from_str(json).unwrap();
        let swap = VarianceSwap::from_json(&data);
        // flat 25% vol: fair variance 0.0625 vs strike 0.0484 -> positive MtM
        assert!((swap.fair_remaining_variance - 0.0625).abs() < 1e-5);
        assert!(swap.npv() > 0.0);
        // strike at fair vol prices to ~zero
        let atm = r#"{
            "symbol": "VSWAP", "underlying_price": 100.0,
            "strike_vol": 0.25, "notional": 1000000.0,
            "maturity": "2030-01-01", "risk_free_rate": 0.03,
            "volatility": 0.25
        }"#;
        let fair: VarianceSwapData = serde_json::from_str(atm).unwrap();
        assert!(VarianceSwap::from_json(&fair).npv().abs() < 50.0);

        // typed contracts: gamma (b > 0 lifts the strike above sigma^2 at
        // fair) and corridor (a sub-range strikes below the full swap)
        let gamma = r#"{
            "symbol": "GSWAP", "underlying_price": 100.0,
            "strike_vol": 0.25, "notional": 1000000.0,
            "maturity": "2030-01-01", "risk_free_rate": 0.03,
            "volatility": 0.25, "swap_type": "gamma"
        }"#;
        let g: VarianceSwapData = serde_json::from_str(gamma).unwrap();
        let g_swap = VarianceSwap::from_json(&g);
        assert!(g_swap.fair_remaining_variance > 0.0625, "{}", g_swap.fair_remaining_variance);
        let corridor = r#"{
            "symbol": "CSWAP", "underlying_price": 100.0,
            "strike_vol": 0.20, "notional": 1000000.0,
            "maturity": "2030-01-01", "risk_free_rate": 0.03,
            "volatility": 0.25, "swap_type": "corridor",
            "corridor_low": 80.0, "corridor_high": 120.0
        }"#;
        let c: VarianceSwapData = serde_json::from_str(corridor).unwrap();
        let c_swap = VarianceSwap::from_json(&c);
        assert!(
            c_swap.fair_remaining_variance < 0.0625,
            "{}",
            c_swap.fair_remaining_variance
        );
    }
}
