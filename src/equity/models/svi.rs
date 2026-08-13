//! SVI and SSVI implied-volatility parameterizations (Gatheral 2004;
//! Gatheral & Jacquier 2014).
//!
//! **SVI** (raw form) parameterizes one expiry's total variance in
//! log-moneyness `k = ln(K/F)`:
//!
//! ```text
//! w(k) = a + b [ rho (k - m) + sqrt((k - m)^2 + sigma^2) ]
//! ```
//!
//! five parameters per smile: level `a`, wing slope `b`, skew `rho`,
//! shift `m`, ATM curvature `sigma`. Wings are asymptotically linear
//! with slopes `b(1 - rho)` (put side) and `b(1 + rho)` (call side).
//!
//! **SSVI** parameterizes the whole surface from the ATM total-variance
//! term structure `theta_t` and three global parameters `(rho, eta,
//! gamma)` through the power-law curvature
//! `phi(theta) = eta / (theta^gamma (1 + theta)^(1-gamma))`:
//!
//! ```text
//! w(k, t) = theta_t/2 [ 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) ]
//! ```
//!
//! Both calibrate by Levenberg-Marquardt
//! ([`core::optimization`](crate::core::optimization)) in transformed
//! parameter spaces, the same pattern as
//! [`heston::calibrate`](crate::equity::heston::calibrate). Butterfly
//! arbitrage is checked through the Gatheral-Jacquier `g(k)` density
//! condition (SVI) and the power-law sufficient conditions (SSVI), and
//! fitted smiles sample into the pricing
//! [`VolSurface`](crate::core::vols::VolSurface) via
//! [`Ssvi::to_vol_surface`].

use chrono::NaiveDate;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::core::vols::{VolError, VolSurface};

// ── SVI: one expiry ─────────────────────────────────────────────────────

/// Raw SVI parameters for a single expiry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SviParams {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

/// Result of an SVI smile calibration.
#[derive(Debug, Clone)]
pub struct SviFit {
    pub params: SviParams,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl SviParams {
    /// Total variance `w(k)` at log-moneyness `k = ln(K/F)`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let d = k - self.m;
        self.a + self.b * (self.rho * d + (d * d + self.sigma * self.sigma).sqrt())
    }

    /// Implied vol at log-moneyness `k` for expiry `t`.
    pub fn vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k).max(0.0) / t).sqrt()
    }

    /// Static parameter constraints: `b >= 0`, `|rho| < 1`, `sigma > 0`
    /// and non-negative minimum variance `a + b sigma sqrt(1 - rho^2)`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.b < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "b must be non-negative",
            ));
        }
        if !(-1.0..1.0).contains(&self.rho) && self.rho != -1.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "rho must be in (-1, 1)",
            ));
        }
        if self.sigma <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "sigma must be positive",
            ));
        }
        if self.a + self.b * self.sigma * (1.0 - self.rho * self.rho).sqrt() < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "minimum total variance is negative",
            ));
        }
        Ok(())
    }

    /// Total variance and its first two log-moneyness derivatives at
    /// `k` — all closed-form, which is what makes SVI-based Dupire
    /// local vol smooth by construction.
    pub fn variance_derivatives(&self, k: f64) -> (f64, f64, f64) {
        let d = k - self.m;
        let root = (d * d + self.sigma * self.sigma).sqrt();
        let w = self.a + self.b * (self.rho * d + root);
        let w1 = self.b * (self.rho + d / root);
        let w2 = self.b * self.sigma * self.sigma / (root * root * root);
        (w, w1, w2)
    }

    /// The Gatheral-Jacquier butterfly function
    /// `g(k) = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2`,
    /// which must stay non-negative for an arbitrage-free density.
    pub fn butterfly_g(&self, k: f64) -> f64 {
        let (w, w1, w2) = self.variance_derivatives(k);
        (1.0 - k * w1 / (2.0 * w)).powi(2) - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0
    }

    /// Minimum of `g(k)` over a wide log-moneyness scan; negative means
    /// the smile carries butterfly arbitrage.
    pub fn min_butterfly_g(&self) -> f64 {
        (0..=800)
            .map(|i| self.butterfly_g(-2.0 + i as f64 * 0.005))
            .fold(f64::INFINITY, f64::min)
    }

    pub fn has_butterfly_arbitrage(&self) -> bool {
        self.min_butterfly_g() < 0.0
    }

    /// Calibrate to one expiry's quotes `(k, implied vol)` by
    /// Levenberg-Marquardt on total-variance residuals, with `b` and
    /// `sigma` in log space and `rho` through `tanh` so every trial is
    /// admissible.
    pub fn calibrate(quotes: &[(f64, f64)], t: f64) -> SviFit {
        assert!(
            quotes.len() >= 5,
            "SVI has five parameters; need at least five quotes"
        );
        assert!(t > 0.0);
        let w_target: Vec<(f64, f64)> = quotes.iter().map(|&(k, v)| (k, v * v * t)).collect();
        let (w_min, w_max) = w_target
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &(_, w)| {
                (lo.min(w), hi.max(w))
            });
        let k_at_min = w_target
            .iter()
            .fold(
                (0.0, f64::INFINITY),
                |acc, &(k, w)| if w < acc.1 { (k, w) } else { acc },
            )
            .0;
        let k_span = quotes.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max)
            - quotes.iter().map(|q| q.0).fold(f64::INFINITY, f64::min);
        // start: level at the observed floor, gentle wings, no skew
        let x0 = vec![
            0.5 * w_min,                                          // a
            (((w_max - w_min) / k_span.max(0.1)).max(1e-3)).ln(), // ln b
            0.0,                                                  // atanh rho
            k_at_min,                                             // m
            0.2_f64.ln(),                                         // ln sigma
        ];
        let unpack = |u: &[f64]| SviParams {
            a: u[0],
            b: u[1].exp(),
            rho: u[2].tanh(),
            m: u[3],
            sigma: u[4].exp(),
        };
        let residuals = |u: &[f64]| -> Vec<f64> {
            let p = unpack(u);
            w_target
                .iter()
                .map(|&(k, w)| p.total_variance(k) - w)
                .collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let params = unpack(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(k, v)| (params.vol(k, t) - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SviFit {
            params,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        }
    }
}

// ── Per-expiry SVI surface fit ──────────────────────────────────────────

/// One fitted expiry slice of a [`SviSurfaceFit`].
#[derive(Debug, Clone)]
pub struct SviSlice {
    /// Expiry time (year fraction).
    pub t: f64,
    /// Forward the slice's log-moneyness is measured against.
    pub forward: f64,
    pub params: SviParams,
    /// Fit error in implied vol against the input pillars.
    pub rmse: f64,
    pub converged: bool,
    /// The quoted log-moneyness span the fit is anchored on.
    pub k_range: (f64, f64),
    /// Minimum of Gatheral's `g(k)` over the quoted span; negative
    /// means the *fit itself* carries butterfly arbitrage there.
    pub min_g: f64,
}

/// A per-expiry SVI fit of an implied surface: one [`SviParams`] smile
/// per pillar expiry, linear total variance in time between slices at
/// fixed log-moneyness (with a forward-variance floor for calendar
/// safety), and **analytic** Dupire local vol from SVI's closed-form
/// derivatives.
///
/// This is the smoother, where
/// [`repair_arbitrage`](crate::equity::surface_repair::repair_arbitrage)
/// is the repair: every point moves a little (by the fit RMSE), in
/// exchange for a C^2 smile that Dupire can differentiate without the
/// spikes piecewise-linear interpolation produces. Fit it to the
/// *cleaned* surface so outright arbitrage is gone before smoothing.
#[derive(Debug, Clone)]
pub struct SviSurfaceFit {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    /// Slices in increasing expiry order.
    pub slices: Vec<SviSlice>,
    /// Input expiries skipped for having fewer than five pillar quotes.
    pub skipped_slices: usize,
    /// Largest total-variance decrease between adjacent fitted slices
    /// over the quoted span (0 = calendar-clean fit); evaluation floors
    /// forward variance, so this measures fit tension, not arbitrage in
    /// the output.
    pub max_calendar_crossing: f64,
}

/// Forward-variance floor: `dw/dt` never drops below this, so local
/// variance stays positive even where fitted slices graze.
const MIN_FORWARD_VARIANCE: f64 = 1e-8;
/// Local vol clamps, matching
/// [`LocalVol`](crate::equity::local_vol::LocalVol).
const MIN_LOCAL_VOL: f64 = 0.01;
const MAX_LOCAL_VOL: f64 = 3.0;

impl SviSurfaceFit {
    /// Fit one SVI smile per pillar expiry of `surface` (its per-expiry
    /// point smiles, on any coordinate). `forward` maps expiry time to
    /// the underlying's forward, exactly as for
    /// [`VolSurface::diagnostics`](crate::core::vols::VolSurface::diagnostics).
    /// Expiries with fewer than five pillars (SVI has five parameters)
    /// are skipped and counted.
    pub fn fit(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
    ) -> Result<SviSurfaceFit, RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "svi fit",
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
            if smile.len() < 5 {
                skipped += 1;
                continue;
            }
            let f = forward(t);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let k = match coordinate {
                        SmileCoordinate::Strike => (x / f).ln(),
                        SmileCoordinate::Moneyness => x.ln(),
                        SmileCoordinate::LogMoneyness => x,
                    };
                    (k, vol)
                })
                .collect();
            let fit = SviParams::calibrate(&quotes, t);
            let (k_lo, k_hi) = quotes
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &(k, _)| {
                    (lo.min(k), hi.max(k))
                });
            let min_g = (0..=200)
                .map(|i| {
                    fit.params
                        .butterfly_g(k_lo + (k_hi - k_lo) * i as f64 / 200.0)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(SviSlice {
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
                "svi fit",
                format!("no expiry has the five quotes an SVI fit needs ({skipped} skipped)"),
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
                let crossing = pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
                max_crossing = max_crossing.max(crossing);
            }
        }
        Ok(SviSurfaceFit {
            reference_date: surface.reference_date(),
            day_count: surface.day_count(),
            slices,
            skipped_slices: skipped,
            max_calendar_crossing: max_crossing,
        })
    }

    /// The slice pair bracketing `t`, with the interpolation weight on
    /// the later slice (0 at or below the earlier, 1 at or beyond the
    /// later; a single-slice surface brackets with itself).
    fn bracket(&self, t: f64) -> (&SviSlice, &SviSlice, f64) {
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

    /// Total variance at log-moneyness `k`: linear in time between
    /// slices at fixed `k`, floored to be non-decreasing; proportional
    /// to `t` below the first slice (variance accrues from zero).
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        let (wa, wb) = (
            a.params.total_variance(k),
            b.params.total_variance(k).max(a.params.total_variance(k)),
        );
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

    /// Analytic Dupire local vol at underlying `level` and time `t`:
    /// Gatheral's formula with `w`, `w_k`, `w_kk` in closed form from
    /// the bracketing SVI slices (interpolated linearly in time) and
    /// `dw/dt` as the floored forward variance between them. Clamped to
    /// the same `[1%, 300%]` band as the numerical
    /// [`LocalVol`](crate::equity::local_vol::LocalVol).
    pub fn local_vol(&self, level: f64, t: f64) -> f64 {
        self.local_vol_checked(level, t).0
    }

    /// [`local_vol`](Self::local_vol) plus whether a guard fired
    /// (`true` = implied vol was returned instead of the Dupire value:
    /// vanishing variance or a non-positive density denominator).
    pub fn local_vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        let t = t.max(1e-4);
        let k = (level / self.forward(t)).ln();
        let (a, b, weight) = self.bracket(t);
        let (wa, wa1, wa2) = a.params.variance_derivatives(k);
        let (mut wb, mut wb1, mut wb2) = b.params.variance_derivatives(k);
        if wb < wa {
            // grazing slices: floor the later variance (flat forward)
            (wb, wb1, wb2) = (wa, wa1, wa2);
        }
        let (w, w1, w2, dwdt) = if t <= a.t {
            // below the first pillar variance accrues proportionally
            let scale = (t / a.t).min(1.0);
            (wa * scale, wa1 * scale, wa2 * scale, wa / a.t)
        } else if a.t == b.t {
            // at or beyond the last pillar: flat-extrapolated smile,
            // forward variance from the last inter-slice segment
            let dwdt = self.last_segment_dwdt(k);
            (wa, wa1, wa2, dwdt)
        } else {
            let dwdt = (wb - wa) / (b.t - a.t);
            (
                wa + (wb - wa) * weight,
                wa1 + (wb1 - wa1) * weight,
                wa2 + (wb2 - wa2) * weight,
                dwdt,
            )
        };
        let dwdt = dwdt.max(MIN_FORWARD_VARIANCE);
        if w < 1e-8 {
            return (
                (w.max(1e-12) / t)
                    .sqrt()
                    .clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL),
                true,
            );
        }
        let denominator =
            (1.0 - k * w1 / (2.0 * w)).powi(2) - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0;
        if denominator <= 1e-4 {
            return ((w / t).sqrt().clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL), true);
        }
        (
            (dwdt / denominator)
                .sqrt()
                .clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL),
            false,
        )
    }

    fn last_segment_dwdt(&self, k: f64) -> f64 {
        let n = self.slices.len();
        if n == 1 {
            let s = &self.slices[0];
            return s.params.total_variance(k) / s.t;
        }
        let (prev, last) = (&self.slices[n - 2], &self.slices[n - 1]);
        (last.params.total_variance(k) - prev.params.total_variance(k)) / (last.t - prev.t)
    }

    /// Sample the fit into the canonical pricing [`VolSurface`]: per
    /// slice, `samples` strikes across its own quoted log-moneyness
    /// span, through the floored [`Self::total_variance`] so the
    /// calendar floor is baked into the artifact. The sampled surface
    /// serializes, plots and prices like any other; Dupire should use
    /// [`Self::local_vol`] directly, which stays analytic. (Sub-basis-
    /// point calendar crossings can survive in the sampled wings where
    /// grazing weekly fits overlap — the diagnostics in the build
    /// metadata report them; the analytic path floors them.)
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
                        // sample through the floored accessor, so the
                        // calendar floor between grazing fitted slices
                        // is baked into the sampled artifact too
                        let vol = (self.total_variance(k, slice.t).max(1e-12) / slice.t).sqrt();
                        (slice.forward * k.exp(), vol)
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, self.reference_date, self.day_count)
    }

    /// Sample [`Self::local_vol`] on a `levels` x `times` grid
    /// (`grid[i][j]` = level i, time j — the plotting layout).
    pub fn local_vol_grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        levels
            .iter()
            .map(|&level| times.iter().map(|&t| self.local_vol(level, t)).collect())
            .collect()
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
                        "a": s.params.a, "b": s.params.b, "rho": s.params.rho,
                        "m": s.params.m, "sigma": s.params.sigma,
                    },
                    "rmse_vol_bps": s.rmse * 1e4,
                    "converged": s.converged,
                    "min_butterfly_g": s.min_g,
                    "k_range": [s.k_range.0, s.k_range.1],
                })
            })
            .collect();
        serde_json::json!({
            "model": "per-expiry raw SVI (Gatheral), linear total variance in time",
            "slices": slices,
            "skipped_slices": self.skipped_slices,
            "max_calendar_crossing": self.max_calendar_crossing,
        })
    }
}

// ── SSVI: the whole surface ─────────────────────────────────────────────

/// SSVI surface: ATM total-variance pillars plus global `(rho, eta,
/// gamma)` with the power-law curvature.
#[derive(Debug, Clone)]
pub struct Ssvi {
    pub rho: f64,
    pub eta: f64,
    /// Power-law exponent in `(0, 1]`.
    pub gamma: f64,
    /// `(t, theta_t)` pillars, `t` and `theta` strictly increasing.
    pub theta_pillars: Vec<(f64, f64)>,
}

/// Result of an SSVI calibration.
#[derive(Debug, Clone)]
pub struct SsviFit {
    pub surface: Ssvi,
    /// Root-mean-square error in implied vol.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

impl Ssvi {
    /// ATM total variance at `t`: proportional below the first pillar
    /// (variance accrues from zero), linear between pillars, and
    /// continued with the last segment's slope beyond.
    pub fn theta(&self, t: f64) -> f64 {
        let p = &self.theta_pillars;
        let n = p.len();
        if t <= 0.0 {
            return 0.0;
        }
        if t <= p[0].0 {
            return p[0].1 * t / p[0].0;
        }
        if t >= p[n - 1].0 {
            if n == 1 {
                return p[0].1 * t / p[0].0;
            }
            let slope = (p[n - 1].1 - p[n - 2].1) / (p[n - 1].0 - p[n - 2].0);
            return p[n - 1].1 + slope * (t - p[n - 1].0);
        }
        let idx = p.partition_point(|&(ti, _)| ti < t);
        let (t0, w0) = p[idx - 1];
        let (t1, w1) = p[idx];
        w0 + (w1 - w0) * (t - t0) / (t1 - t0)
    }

    /// Power-law curvature `phi(theta)`.
    pub fn phi(&self, theta: f64) -> f64 {
        self.eta / (theta.powf(self.gamma) * (1.0 + theta).powf(1.0 - self.gamma))
    }

    /// Total variance `w(k, t)`.
    pub fn total_variance(&self, k: f64, t: f64) -> f64 {
        let theta = self.theta(t);
        if theta <= 0.0 {
            return 0.0;
        }
        let phi = self.phi(theta);
        let pk = phi * k;
        0.5 * theta
            * (1.0 + self.rho * pk + ((pk + self.rho).powi(2) + 1.0 - self.rho * self.rho).sqrt())
    }

    /// Implied vol for `strike` given the `forward` at expiry `t`.
    pub fn vol(&self, strike: f64, forward: f64, t: f64) -> f64 {
        (self.total_variance((strike / forward).ln(), t) / t).sqrt()
    }

    /// Static no-arbitrage checks (Gatheral-Jacquier): admissible
    /// parameters, nondecreasing `theta` (calendar), the power-law
    /// sufficient condition `eta (1 + |rho|) <= 2`, and the per-pillar
    /// butterfly bounds `theta phi (1 + |rho|) <= 4` and
    /// `theta phi^2 (1 + |rho|) <= 4`.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if !(-1.0..1.0).contains(&self.rho) {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "rho must be in (-1, 1)",
            ));
        }
        if self.eta <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "eta must be positive",
            ));
        }
        if !(0.0..=1.0).contains(&self.gamma) || self.gamma == 0.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "gamma must be in (0, 1]",
            ));
        }
        if self.theta_pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "need at least one theta pillar",
            ));
        }
        if self
            .theta_pillars
            .iter()
            .any(|&(t, w)| t <= 0.0 || w <= 0.0)
        {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "theta pillars must have positive times and variances",
            ));
        }
        if self
            .theta_pillars
            .windows(2)
            .any(|p| p[1].0 <= p[0].0 || p[1].1 < p[0].1)
        {
            return Err(RustyQLibError::invalid_input("svi params", "theta pillars must be increasing in time and nondecreasing in variance (calendar arbitrage)"));
        }
        if self.eta * (1.0 + self.rho.abs()) > 2.0 {
            return Err(RustyQLibError::invalid_input(
                "svi params",
                "eta (1 + |rho|) must not exceed 2 (static arbitrage)",
            ));
        }
        for &(_, theta) in &self.theta_pillars {
            let phi = self.phi(theta);
            if theta * phi * (1.0 + self.rho.abs()) > 4.0
                || theta * phi * phi * (1.0 + self.rho.abs()) > 4.0
            {
                return Err(RustyQLibError::invalid_input(
                    "svi params",
                    "butterfly bound violated at a theta pillar",
                ));
            }
        }
        Ok(())
    }

    /// Calibrate `(rho, eta, gamma)` to surface quotes `(t, k, vol)`
    /// given the ATM total-variance pillars, by Levenberg-Marquardt on
    /// total-variance residuals (`tanh` / `exp` / logistic transforms
    /// keep every trial admissible).
    pub fn calibrate(
        quotes: &[(f64, f64, f64)],
        theta_pillars: &[(f64, f64)],
        start: (f64, f64, f64),
    ) -> SsviFit {
        assert!(
            quotes.len() >= 3,
            "need at least three quotes for three parameters"
        );
        let make = |u: &[f64]| Ssvi {
            rho: u[0].tanh(),
            eta: u[1].exp(),
            gamma: 1.0 / (1.0 + (-u[2]).exp()),
            theta_pillars: theta_pillars.to_vec(),
        };
        let (rho0, eta0, gamma0) = start;
        let x0 = vec![
            rho0.clamp(-0.999, 0.999).atanh(),
            eta0.ln(),
            (gamma0.clamp(1e-3, 1.0 - 1e-9) / (1.0 - gamma0.clamp(1e-3, 1.0 - 1e-9))).ln(),
        ];
        let residuals = |u: &[f64]| -> Vec<f64> {
            let s = make(u);
            quotes
                .iter()
                .map(|&(t, k, v)| s.total_variance(k, t) - v * v * t)
                .collect()
        };
        let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
        let surface = make(&fit.x);
        let rmse = (quotes
            .iter()
            .map(|&(t, k, v)| ((surface.total_variance(k, t) / t).sqrt() - v).powi(2))
            .sum::<f64>()
            / quotes.len() as f64)
            .sqrt();
        SsviFit {
            surface,
            rmse,
            iterations: fit.iterations,
            converged: fit.converged,
        }
    }

    /// Sample the SSVI surface into the canonical pricing
    /// [`VolSurface`]: per expiry `(t, forward)`, strikes are placed at
    /// `forward * exp(k)` over the log-moneyness grid.
    pub fn to_vol_surface(
        &self,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        expiry_forwards: &[(f64, f64)],
        log_moneyness_grid: &[f64],
    ) -> Result<VolSurface, VolError> {
        let expiries: Vec<Tenor> = expiry_forwards
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = expiry_forwards
            .iter()
            .map(|&(t, forward)| {
                log_moneyness_grid
                    .iter()
                    .map(|&k| {
                        let strike = forward * k.exp();
                        (strike, self.vol(strike, forward, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&expiries, &smiles, reference_date, day_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sane() -> SviParams {
        SviParams {
            a: 0.03,
            b: 0.12,
            rho: -0.4,
            m: -0.02,
            sigma: 0.3,
        }
    }

    #[test]
    fn svi_shape_matches_the_closed_form_structure() {
        let p = sane();
        p.validate().unwrap();
        // total variance at k = m is a + b sigma
        assert!((p.total_variance(p.m) - (p.a + p.b * p.sigma)).abs() < 1e-14);
        // asymptotic wing slopes b (1 +- rho), measured per unit of |k|
        let far = 60.0;
        let call_slope = p.total_variance(far + 1.0) - p.total_variance(far);
        let put_slope = p.total_variance(-far - 1.0) - p.total_variance(-far);
        assert!(
            (call_slope - p.b * (1.0 + p.rho)).abs() < 1e-3,
            "{call_slope}"
        );
        assert!(
            (put_slope - p.b * (1.0 - p.rho)).abs() < 1e-3,
            "{put_slope}"
        );
    }

    #[test]
    fn vogt_example_carries_butterfly_arbitrage_and_sane_params_do_not() {
        // the classic arbitrageable SVI smile (Gatheral-Jacquier 2014 §3)
        let vogt = SviParams {
            a: -0.0410,
            b: 0.1331,
            rho: 0.3060,
            m: 0.3586,
            sigma: 0.4153,
        };
        assert!(
            vogt.has_butterfly_arbitrage(),
            "min g = {}",
            vogt.min_butterfly_g()
        );
        assert!((vogt.min_butterfly_g() - -0.0329).abs() < 2e-3);
        assert!(
            !sane().has_butterfly_arbitrage(),
            "min g = {}",
            sane().min_butterfly_g()
        );
    }

    #[test]
    fn svi_calibration_round_trips() {
        let truth = sane();
        let t = 0.75;
        let quotes: Vec<(f64, f64)> = (0..15)
            .map(|i| -0.42 + i as f64 * 0.06)
            .map(|k| (k, truth.vol(k, t)))
            .collect();
        let fit = SviParams::calibrate(&quotes, t);
        assert!(
            fit.rmse < 1e-6,
            "vol rmse {} params {:?}",
            fit.rmse,
            fit.params
        );
        assert!(fit.params.validate().is_ok());
        // the fitted smile matches off the quote grid too
        for i in 0..=20 {
            let k = -0.5 + i as f64 * 0.05;
            assert!(
                (fit.params.vol(k, t) - truth.vol(k, t)).abs() < 1e-4,
                "k = {k}"
            );
        }
    }

    fn surface_from(slices: &[(f64, SviParams, f64)]) -> VolSurface {
        // sample each known smile onto pillar strikes, as a chain would
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
                        (f * k.exp(), p.vol(k, t))
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
    fn per_expiry_fit_recovers_generating_smiles() {
        let front = sane();
        let back = SviParams { a: 0.055, ..sane() };
        let surface = surface_from(&[(0.5, front, 101.0), (1.0, back, 102.0)]);
        let forward = |t: f64| if t < 0.75 { 101.0 } else { 102.0 };
        let fit = SviSurfaceFit::fit(&surface, forward).unwrap();
        assert_eq!(fit.slices.len(), 2);
        assert_eq!(fit.skipped_slices, 0);
        for slice in &fit.slices {
            assert!(slice.rmse < 1e-5, "rmse {}", slice.rmse);
            assert!(slice.min_g > 0.0, "min g {}", slice.min_g);
        }
        assert!(fit.max_calendar_crossing <= 1e-10);
        // fitted vols agree with the generators off the pillar grid too
        for i in 0..=12 {
            let k = -0.28 + i as f64 * 0.05;
            let strike = 101.0 * k.exp();
            assert!(
                (fit.vol(strike, 0.5) - front.vol((strike / 101.0_f64).ln(), 0.5)).abs() < 5e-4,
                "k = {k}"
            );
        }
        // sampled surface matches the fit at its own nodes
        let sampled = fit.to_vol_surface(41).unwrap();
        assert_eq!(sampled.expiry_times().len(), 2);
        let probe = 101.0;
        assert!((sampled.vol(probe, probe, 0.5) - fit.vol(probe, 0.5)).abs() < 1e-3);
        // metadata carries per-slice fit quality
        let meta = fit.metadata();
        assert_eq!(meta["slices"].as_array().unwrap().len(), 2);
        assert!(meta["slices"][0]["rmse_vol_bps"].as_f64().unwrap() < 0.5);
    }

    #[test]
    fn flat_svi_term_structure_gives_flat_local_vol() {
        // b = 0 collapses SVI to w(k) = a: constant vol per slice
        let vol = 0.3_f64;
        let slice = |t: f64| SviParams {
            a: vol * vol * t,
            b: 0.0,
            rho: 0.0,
            m: 0.0,
            sigma: 0.3,
        };
        let surface = surface_from(&[(0.5, slice(0.5), 100.0), (1.0, slice(1.0), 100.0)]);
        let fit = SviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        // sigma_loc = sigma_imp everywhere: interior, between slices,
        // below the first pillar and beyond the last
        for level in [80.0, 100.0, 120.0] {
            for t in [0.1, 0.5, 0.75, 1.0, 1.4] {
                let lv = fit.local_vol(level, t);
                assert!((lv - vol).abs() < 5e-3, "level {level} t {t}: {lv}");
            }
        }
    }

    #[test]
    fn sparse_slices_are_skipped_not_fatal() {
        let p = sane();
        let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
        let smiles = vec![
            // three quotes: below SVI's five-parameter minimum
            vec![(90.0, 0.25), (100.0, 0.24), (110.0, 0.23)],
            (0..9)
                .map(|i| {
                    let k = -0.2 + i as f64 * 0.05;
                    (100.0 * k.exp(), p.vol(k, 1.0))
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
        let fit = SviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        assert_eq!(fit.slices.len(), 1);
        assert_eq!(fit.skipped_slices, 1);
        // a surface with no fittable slice errors instead
        let tiny = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5)],
            &[vec![(100.0, 0.2), (105.0, 0.19)]],
            NaiveDate::from_ymd_opt(2026, 8, 11).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        assert!(SviSurfaceFit::fit(&tiny, |_| 100.0).is_err());
    }

    fn ssvi() -> Ssvi {
        Ssvi {
            rho: -0.55,
            eta: 0.9,
            gamma: 0.45,
            theta_pillars: vec![(0.25, 0.012), (0.5, 0.023), (1.0, 0.045), (2.0, 0.09)],
        }
    }

    #[test]
    fn ssvi_reproduces_the_atm_term_structure_and_skew_sign() {
        let s = ssvi();
        s.validate().unwrap();
        for &(t, theta) in &s.theta_pillars {
            assert!(
                (s.total_variance(0.0, t) - theta).abs() < 1e-14,
                "w(0, {t})"
            );
        }
        // negative rho: puts richer than calls
        assert!(s.total_variance(-0.2, 1.0) > s.total_variance(0.2, 1.0));
        // calendar: total variance nondecreasing in t at fixed k
        for i in 1..40 {
            let (t0, t1) = (i as f64 * 0.05, (i + 1) as f64 * 0.05);
            assert!(
                s.total_variance(0.15, t1) >= s.total_variance(0.15, t0),
                "t = {t0}"
            );
        }
    }

    #[test]
    fn ssvi_no_arbitrage_bounds_are_enforced() {
        let mut bad = ssvi();
        bad.eta = 1.5; // eta (1 + |rho|) = 2.325 > 2
        assert!(bad.validate().is_err());
        let mut decreasing = ssvi();
        decreasing.theta_pillars[2].1 = 0.01; // calendar violation
        assert!(decreasing.validate().is_err());
    }

    #[test]
    fn ssvi_calibration_round_trips() {
        let truth = ssvi();
        let mut quotes = Vec::new();
        for &(t, _) in &truth.theta_pillars {
            for i in 0..7 {
                let k = -0.3 + i as f64 * 0.1;
                quotes.push((t, k, (truth.total_variance(k, t) / t).sqrt()));
            }
        }
        let fit = Ssvi::calibrate(&quotes, &truth.theta_pillars, (-0.2, 0.5, 0.5));
        assert!(fit.rmse < 1e-8, "vol rmse {}", fit.rmse);
        assert!(
            (fit.surface.rho - truth.rho).abs() < 1e-4,
            "rho {}",
            fit.surface.rho
        );
        assert!(
            (fit.surface.eta - truth.eta).abs() < 1e-3,
            "eta {}",
            fit.surface.eta
        );
        assert!(fit.surface.validate().is_ok());
    }

    #[test]
    fn sampled_vol_surface_agrees_with_the_parametric_form() {
        use chrono::NaiveDate;
        let s = ssvi();
        let reference = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let forwards = [(0.25, 101.0), (1.0, 104.0), (2.0, 108.0)];
        let grid: Vec<f64> = (0..13).map(|i| -0.3 + i as f64 * 0.05).collect();
        let surface = s
            .to_vol_surface(reference, DayCountConvention::Act365, &forwards, &grid)
            .unwrap();
        // exact at the sampled nodes
        for &(t, f) in &forwards {
            for &k in &grid {
                let strike = f * k.exp();
                let sampled = surface.vol(strike, f, t);
                let parametric = s.vol(strike, f, t);
                assert!(
                    (sampled - parametric).abs() < 1e-10,
                    "t = {t}, k = {k}: {sampled} vs {parametric}"
                );
            }
        }
    }
}
