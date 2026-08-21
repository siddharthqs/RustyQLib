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

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::errors::RustyQLibError;
use crate::core::traits::Instrument;
use crate::core::utils::norm_cdf;
use crate::core::vols::{VolInput, VolSurface};

/// Annualized realized variance of a log-return series, market
/// convention (mean **not** subtracted).
pub fn realized_variance(log_returns: &[f64], periods_per_year: f64) -> f64 {
    assert!(!log_returns.is_empty());
    log_returns.iter().map(|r| r * r).sum::<f64>() / log_returns.len() as f64 * periods_per_year
}

/// Half-width of the replication window in log-strike: ten ATM
/// standard deviations, floored at one unit.
fn replication_width(forward: f64, t: f64, smile: &impl Fn(f64) -> f64) -> f64 {
    (10.0 * smile(forward).max(1e-4) * t.sqrt()).max(1.0)
}

/// The common core of the three replication strikes: a fine Simpson
/// integral (4000 intervals) of `Q(F e^u) * kernel(u)` over
/// `[u_lo, u_hi]`, where `Q` is the undiscounted Black OTM price on the
/// smile. The strike kernels: `e^{-u}` gives `int Q(K)/K^2 dK / F`
/// (variance, corridor), `1` gives `int Q(K)/K dK` (gamma) — both by
/// the log-strike substitution `K = F e^u`.
fn replication_integral(
    forward: f64,
    t: f64,
    u_lo: f64,
    u_hi: f64,
    kernel: impl Fn(f64) -> f64,
    smile: impl Fn(f64) -> f64,
) -> f64 {
    let steps = 4000usize;
    let du = (u_hi - u_lo) / steps as f64;
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
    let mut sum = 0.0;
    for i in 0..=steps {
        let u = u_lo + i as f64 * du;
        let w = if i == 0 || i == steps {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += w * otm(u) * kernel(u);
    }
    sum * du / 3.0
}

/// Model-free fair variance strike by the log-contract replication.
/// `smile(strike) -> implied vol`; integration over ten ATM standard
/// deviations of log-strike with a fine Simpson rule.
pub fn fair_variance_strike(forward: f64, t: f64, smile: impl Fn(f64) -> f64) -> f64 {
    assert!(forward > 0.0 && t > 0.0);
    let width = replication_width(forward, t, &smile);
    let integral =
        replication_integral(forward, t, -width, width, |u| (-u).exp(), smile) / forward;
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
pub fn fair_gamma_swap_strike(spot: f64, forward: f64, t: f64, smile: impl Fn(f64) -> f64) -> f64 {
    assert!(spot > 0.0 && forward > 0.0 && t > 0.0);
    let width = replication_width(forward, t, &smile);
    // int Q(K)/K dK = int Q(F e^u) du  (log-strike substitution)
    let integral = replication_integral(forward, t, -width, width, |_| 1.0, smile);
    let b_t = (forward / spot).ln();
    let phi = if b_t.abs() < 1e-12 {
        1.0
    } else {
        (1.0 - (-b_t).exp()) / b_t
    };
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
    let width = replication_width(forward, t, &smile);
    // integrate in log-strike over the corridor clipped to the window
    let u_lo = if low <= 0.0 {
        -width
    } else {
        (low / forward).ln().max(-width)
    };
    let u_hi = if high.is_infinite() {
        width
    } else {
        (high / forward).ln().min(width)
    };
    if u_hi <= u_lo {
        return 0.0;
    }
    let integral =
        replication_integral(forward, t, u_lo, u_hi, |u| (-u).exp(), smile) / forward;
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
    /// Pricing as-of date (`YYYY-MM-DD`); defaults to today.
    pub valuation_date: Option<String>,
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
                "variance swap is expired",
            ));
        }
        crate::equity::conventions::check_vol_band("volatility", data.volatility)?;
        crate::equity::conventions::check_vol_band("strike_vol", data.strike_vol)?;
        crate::equity::conventions::check_rate_band("risk_free_rate", data.risk_free_rate)?;
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
            Some("gamma") => fair_gamma_swap_strike(data.underlying_price, forward, t, smile),
            Some("corridor") => fair_corridor_variance_strike(
                forward,
                t,
                data.corridor_low.unwrap_or(0.0),
                data.corridor_high.unwrap_or(f64::INFINITY),
                smile,
            ),
            Some(other) => {
                return Err(RustyQLibError::invalid_input(
                    "swap_type",
                    format!("invalid swap_type '{other}' (use variance, gamma or corridor)"),
                ))
            }
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
            _ => {
                return Err(RustyQLibError::invalid_input(
                    "accrued_variance",
                    "seasoned swaps need both elapsed and accrued_variance",
                ))
            }
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

// ── The Market-bound payoff ─────────────────────────────────────────────

/// Which realized-variance statistic the swap pays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarianceSwapKind {
    /// Plain annualized variance of log returns.
    Variance,
    /// Spot-weighted (gamma) variance, `(A/n) sum (S_i/S_0) r_i^2`.
    Gamma,
    /// Variance accruing only while the previous observation lies
    /// inside `[low, high]`.
    Corridor { low: f64, high: f64 },
}

/// The variance swap as a mainline [`Payoff`], pricing inside
/// [`EquityOption`](crate::equity::vanilla_option::EquityOption) —
/// which gives it the market context the standalone lacks: the bound
/// vol surface feeds the replication on the **Analytical** engine
/// (term-curve discounting, cash-dividend-consistent forwards), and
/// the **MonteCarlo** engine prices the discretely monitored contract
/// under every model the engine carries (GBM, local vol, Heston, SABR,
/// rough Bergomi) — model-based valuation the standalone never had.
/// The standalone [`VarianceSwap`] remains the flat-market validation
/// reference.
///
/// The realized leg annualizes by the observation schedule's own
/// frequency (`observations / t`), so a flat-vol simulation converges
/// to `sigma^2` for any observation count and agrees with the
/// (continuous) replication strike up to the discrete-monitoring bias
/// — deliberately avoiding the 252-vs-365 annualization blend the
/// caller-supplied convention invited (review finding B15).
#[derive(Debug, Clone)]
pub struct VarianceSwapPayoff {
    pub exercise_style: crate::core::utils::ContractStyle,
    pub kind: VarianceSwapKind,
    /// Strike in **variance** units (`strike_vol^2`).
    pub strike_variance: f64,
    /// Variance notional (payout per unit of annualized variance).
    pub notional: f64,
    /// Observation count of the Monte Carlo (discrete) route; the
    /// analytic route replicates continuous monitoring.
    pub observations: usize,
    /// Seasoned swaps: (elapsed years, annualized variance realized
    /// over them), blended time-weighted with the remaining leg.
    pub accrued: Option<(f64, f64)>,
    /// Spot at inception — denominator of the first return and the
    /// `S_0` of the gamma weighting.
    pub initial_fixing: f64,
}

impl VarianceSwapPayoff {
    /// Time-weighted blend of the accrued and remaining annualized
    /// variance (the standalone's convention).
    pub fn blend(&self, remaining: f64, t_remaining: f64) -> f64 {
        match self.accrued {
            None => remaining,
            Some((elapsed, accrued)) => {
                (elapsed * accrued + t_remaining * remaining) / (elapsed + t_remaining)
            }
        }
    }

    /// Value of one simulated path: the kind's realized statistic over
    /// the observation spots (first return against `initial_fixing`),
    /// annualized by the schedule frequency, seasoned-blended, and the
    /// strike difference paid at maturity. `t` is the remaining life
    /// the observation grid spans (bumped views pass the bumped life).
    pub fn path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64], t: f64) -> f64 {
        let mut sum = 0.0;
        let mut s_prev = self.initial_fixing;
        for &idx in obs_idx {
            let s = path[idx];
            let r = (s / s_prev).ln();
            match self.kind {
                VarianceSwapKind::Variance => sum += r * r,
                VarianceSwapKind::Gamma => sum += s / self.initial_fixing * r * r,
                VarianceSwapKind::Corridor { low, high } => {
                    // standard convention: the step accrues when the
                    // *previous* observation was inside the corridor
                    if s_prev >= low && s_prev <= high {
                        sum += r * r;
                    }
                }
            }
            s_prev = s;
        }
        // annualization by the schedule's own frequency: (A/n) sum r^2
        // with A = n/t collapses to sum / t
        let realized = sum / t;
        let total = self.blend(realized, t);
        self.notional * (total - self.strike_variance) * dfs.last().copied().unwrap_or(1.0)
    }
}

impl crate::equity::utils::Payoff for VarianceSwapPayoff {
    /// Degenerate single-point value: zero (the value is the realized
    /// statistic of the whole path).
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, _path: &[f64], _strike: f64) -> f64 {
        panic!(
            "Variance swaps observe a fixing schedule and cannot be valued \
             through path_payoff; the Monte Carlo engine prices them via \
             path_value and the analytic engine by replication"
        );
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> crate::equity::utils::PayoffType {
        crate::equity::utils::PayoffType::VarianceSwap
    }
    fn put_or_call(&self) -> &crate::core::trade::PutOrCall {
        // by convention: long realized variance is long volatility,
        // call-shaped in variance; not used by pricing
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
        let smile = |k: f64| atm - 0.15 * (k / 100.0 - 1.0) + 0.1 * (k / 100.0 - 1.0).powi(2);
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
        assert!(
            k_svi > atm_svi * atm_svi,
            "{k_svi} vs {}",
            atm_svi * atm_svi
        );
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
        for (sigma, b, t) in [
            (0.2_f64, 0.04_f64, 1.0_f64),
            (0.3, -0.02, 0.5),
            (0.25, 0.0, 2.0),
        ] {
            let spot = 100.0_f64;
            let forward = spot * (b * t).exp();
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
        let analytic = fair_gamma_swap_strike(s0, s0 * (b * t).exp(), t, |_| sigma);
        assert!(
            (mc - analytic).abs() < 0.004,
            "mc {mc} vs analytic {analytic}"
        );

        // under a put skew the spot-weighting discounts crash variance:
        // gamma strike < variance strike
        let smile = |k: f64| 0.2 - 0.15 * (k / 100.0_f64 - 1.0);
        let k_var = fair_variance_strike(100.0, 1.0, smile);
        let k_gam = fair_gamma_swap_strike(100.0, 100.0, 1.0, smile);
        assert!(k_gam < k_var, "gamma {k_gam} vs variance {k_var}");
    }

    #[test]
    fn corridor_strikes_are_additive_and_recover_the_full_swap() {
        let smile =
            |k: f64| 0.2 - 0.1 * (k / 100.0_f64 - 1.0) + 0.2 * (k / 100.0_f64 - 1.0).powi(2);
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
        assert!(
            (mc - analytic).abs() < 0.003,
            "mc {mc} vs analytic {analytic}"
        );
        // hand check of the accrual convention: only the step leaving the
        // corridor from inside counts
        let path = [100.0, 120.0, 110.0, 80.0, 85.0];
        let expect =
            ((120.0_f64 / 100.0).ln().powi(2) + (80.0_f64 / 110.0).ln().powi(2)) / 4.0 * 252.0;
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
        assert!(
            g_swap.fair_remaining_variance > 0.0625,
            "{}",
            g_swap.fair_remaining_variance
        );
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

    // ── the mainline VarianceSwapPayoff (Market-bound spine) ───────────

    fn builder_vswap(strike_vol: f64) -> crate::equity::builder::EquityOptionBuilder {
        use crate::equity::builder::EquityOptionBuilder;
        EquityOptionBuilder::new()
            .symbol("VSWAP")
            .spot(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(chrono::NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(chrono::NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .variance_swap(strike_vol, 100.0)
            .seed(42)
    }

    #[test]
    fn mainline_analytic_matches_the_standalone_replication() {
        use crate::equity::utils::Engine;
        // identical flat market: the bound-surface replication must equal
        // the standalone product (same integral, same forward, same df)
        let mainline = builder_vswap(0.22)
            .engine(Engine::BlackScholes)
            .build()
            .expect("variance swap must build")
            .npv();
        let standalone = VarianceSwap {
            notional: 100.0,
            strike_variance: 0.22 * 0.22,
            t_remaining: 1.0,
            r: 0.03,
            fair_remaining_variance: fair_variance_strike(
                100.0 * (0.03_f64).exp(),
                1.0,
                |_| 0.25,
            ),
            accrued: None,
        }
        .mtm();
        assert!(
            (mainline - standalone).abs() < 1e-9,
            "mainline {mainline} vs standalone {standalone}"
        );
        // struck at fair vol the swap is worth ~zero
        let fair = builder_vswap(0.25)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        assert!(fair.abs() < 1e-3, "at-fair pv {fair}");
    }

    #[test]
    fn mainline_mc_agrees_with_the_replication_on_flat_vol() {
        use crate::equity::utils::Engine;
        // GBM Monte Carlo of the discrete contract vs the continuous
        // replication: the discrete-monitoring gap under GBM is O(dt)
        // drift terms, far below the tolerance
        let analytic = builder_vswap(0.25)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        let mc = builder_vswap(0.25)
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build()
            .expect("MC variance swap must build")
            .npv();
        assert!(
            (mc - analytic).abs() < 0.03,
            "mc {mc} vs replication {analytic} (notional 100)"
        );
    }

    #[test]
    fn mainline_heston_mc_recovers_the_integrated_variance_expectation() {
        use crate::equity::utils::Engine;
        // under Heston the fair variance strike is the closed-form
        // expected integrated variance
        // theta + (v0 - theta)(1 - e^{-kappa T})/(kappa T)
        let (v0, kappa, theta, t) = (0.09_f64, 2.0_f64, 0.04_f64, 1.0_f64);
        let expected = theta + (v0 - theta) * (1.0 - (-kappa * t).exp()) / (kappa * t);
        let pv = builder_vswap(expected.sqrt())
            .heston(crate::equity::heston::HestonParams {
                v0,
                kappa,
                theta,
                vol_of_vol: 0.4,
                rho: -0.7,
            })
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build()
            .expect("heston variance swap must build")
            .npv();
        // struck at the model's own expectation the swap is ~worthless
        assert!(pv.abs() < 0.15, "heston var swap at model-fair strike: {pv}");
    }

    #[test]
    fn mainline_corridor_and_gamma_track_their_replication_strikes() {
        use crate::equity::utils::Engine;
        // corridor: MC discrete accrual vs truncated replication
        let corridor_analytic = builder_vswap(0.2)
            .corridor(90.0, 115.0)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        let corridor_mc = builder_vswap(0.2)
            .corridor(90.0, 115.0)
            .variance_swap_observations(504)
            .engine(Engine::MonteCarlo)
            .paths(30_000)
            .build()
            .unwrap()
            .npv();
        assert!(
            (corridor_mc - corridor_analytic).abs() < 0.4,
            "corridor mc {corridor_mc} vs replication {corridor_analytic}"
        );
        // gamma: spot-weighted MC vs the S ln S replication with carry
        let gamma_analytic = builder_vswap(0.25)
            .gamma_swap()
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        let gamma_mc = builder_vswap(0.25)
            .gamma_swap()
            .engine(Engine::MonteCarlo)
            .paths(50_000)
            .build()
            .unwrap()
            .npv();
        assert!(
            (gamma_mc - gamma_analytic).abs() < 0.1,
            "gamma mc {gamma_mc} vs replication {gamma_analytic}"
        );
    }

    #[test]
    fn mainline_seasoning_blends_and_greeks_report() {
        use crate::equity::utils::Engine;
        // seasoned blend against the hand formula, on the analytic engine
        let fresh = builder_vswap(0.2)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap();
        let fair = fair_variance_strike(100.0 * (0.03_f64).exp(), 1.0, |_| 0.25);
        let seasoned = builder_vswap(0.2)
            .seasoned_variance(0.5, 0.09)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap()
            .npv();
        let blend = (0.5 * 0.09 + 1.0 * fair) / 1.5;
        let expect = 100.0 * (-0.03_f64).exp() * (blend - 0.04);
        assert!(
            (seasoned - expect).abs() < 1e-9,
            "seasoned {seasoned} vs {expect}"
        );
        // the batch result carries the replication vega (long variance =
        // long vol) and a near-zero delta on a flat smile
        let result = fresh.price().expect("greeks must evaluate");
        assert!(result.greeks.vega > 0.0, "vega {}", result.greeks.vega);
        assert!(
            result.greeks.delta.abs() < 0.05,
            "flat-smile variance swap is ~delta-neutral: {}",
            result.greeks.delta
        );
    }

    #[test]
    fn mainline_variance_swap_engine_and_input_validation() {
        use crate::core::errors::RustyQLibError;
        use crate::equity::utils::Engine;
        // lattice/PDE engines are refused at build()
        for (engine, name) in [
            (Engine::Binomial, "Binomial"),
            (Engine::FiniteDifference, "FiniteDifference"),
        ] {
            let result = builder_vswap(0.2).engine(engine).build();
            assert!(
                matches!(result, Err(RustyQLibError::UnsupportedEngine(_))),
                "variance swap must be refused on {name}"
            );
        }
        // Heston + Analytical is refused (replication is a GBM-engine
        // route; Heston variance swaps price on MC)
        let result = builder_vswap(0.2)
            .heston(crate::equity::heston::HestonParams {
                v0: 0.04,
                kappa: 2.0,
                theta: 0.04,
                vol_of_vol: 0.4,
                rho: -0.5,
            })
            .engine(Engine::BlackScholes)
            .build();
        assert!(matches!(result, Err(RustyQLibError::UnsupportedEngine(_))));
        // corridor bounds must be ordered
        match builder_vswap(0.2).corridor(115.0, 90.0).build() {
            Err(RustyQLibError::InvalidInput { field, .. }) => assert_eq!(field, "corridor"),
            other => panic!("expected corridor error, got {:?}", other.map(|_| "an option")),
        }
        // modifiers without .variance_swap(...) report the misuse
        match crate::equity::builder::EquityOptionBuilder::new()
            .spot(100.0)
            .flat_vol(0.2)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .vanilla(crate::core::trade::PutOrCall::Call)
            .gamma_swap()
            .build()
        {
            Err(RustyQLibError::InvalidInput { field, .. }) => assert_eq!(field, "gamma_swap"),
            other => panic!("expected setter error, got {:?}", other.map(|_| "an option")),
        }
    }
}

