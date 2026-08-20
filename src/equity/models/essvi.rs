//! eSSVI — the extended SSVI surface (Hendriks & Martini 2019).
//!
//! SSVI ([`Ssvi`](crate::equity::svi::Ssvi)) shapes every expiry with
//! **one global** correlation `rho` and a curvature `phi(theta)` drawn
//! from a chosen family, so the skew term structure can only scale: one
//! `rho` fixes the sign and, up to `phi`, the shape at every tenor. Real
//! equity skew *twists* — steepening into short dates faster than any
//! single power law allows — and that rigidity is SSVI's main empirical
//! cost.
//!
//! eSSVI keeps SSVI's slice shape and frees the parameters per expiry.
//! Writing `psi = theta phi` (so the ATM skew is exactly `rho psi`), each
//! slice is
//!
//! ```text
//! w_t(k) = theta_t/2 [ 1 + rho_t psi_t k / theta_t
//!                        + sqrt((psi_t k / theta_t + rho_t)^2 + 1 - rho_t^2) ]
//! ```
//!
//! with `w_t(0) = theta_t` and `dw/dk|_0 = rho_t psi_t` by construction.
//! It therefore sits in the one gap the other parameterizations leave:
//! per-expiry flexibility **with** cross-expiry structure, where raw SVI
//! and SABR have flexibility and no structure, and SSVI has structure and
//! no flexibility.
//!
//! Three properties make it a distinctive smoother rather than "SVI with
//! fewer parameters":
//!
//! - **Two free parameters per slice.** `theta_t` is read off the ATM
//!   quote rather than fitted (as in
//!   [`SsviSurfaceFit`](crate::equity::svi::SsviSurfaceFit)), leaving
//!   `(psi_t, rho_t)` to the optimizer — fewer degrees of freedom than
//!   SABR, let alone SVI, which matters where quotes are thin.
//! - **Interpolation stays in the family.** The surface between pillars
//!   comes from interpolating the parameter triple, so every intermediate
//!   expiry is itself an eSSVI slice. Interpolating two SVI slices in
//!   total variance, by contrast, produces a smile that is not SVI.
//! - **Calendar structure is enforced during the fit.** Slices are
//!   calibrated in time order and each one carries a penalty against
//!   crossing its predecessor, so the fitted surface is pushed into the
//!   calendar-admissible region instead of being repaired afterwards.
//!
//! **Scope of the no-arbitrage checks.** The per-slice butterfly bounds
//! are the Gatheral-Jacquier sufficient conditions
//! (`psi (1 + |rho|) <= 4` and `psi^2/theta (1 + |rho|) <= 4`), the same
//! ones [`Ssvi::validate`](crate::equity::svi::Ssvi::validate) applies,
//! which carry over slice-wise unchanged. For **calendar** arbitrage this
//! module checks the definition directly and numerically — total variance
//! non-decreasing in `t` at fixed `k`, scanned on a dense grid — rather
//! than the sharp analytic pairwise conditions of Hendriks & Martini,
//! which are not implemented here. The numerical check is correct to grid
//! resolution and is what [`EssviSurfaceFit::max_calendar_crossing`]
//! reports; it is not a substitute for the analytic domain if you need
//! provable admissibility over the whole real line.

use chrono::NaiveDate;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::optimization::{levenberg_marquardt, OptimConfig};
use crate::core::vols::{VolError, VolSurface};
use crate::equity::smoothed_surface::{SmoothedSurface, VarianceDerivatives, MIN_TIME};

// ── One slice's shape ───────────────────────────────────────────────────

/// One eSSVI slice in the `(theta, psi, rho)` coordinates: ATM total
/// variance, skew scale (`ATM skew = rho psi`), and correlation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssviParams {
    /// ATM total variance `w(0)`.
    pub theta: f64,
    /// `theta * phi` — the skew scale.
    pub psi: f64,
    /// Correlation in `(-1, 1)`; sets the skew sign.
    pub rho: f64,
}

impl EssviParams {
    /// SSVI's curvature parameter `phi = psi / theta`.
    pub fn phi(&self) -> f64 {
        self.psi / self.theta
    }

    /// Total variance at log-moneyness `k`.
    pub fn total_variance(&self, k: f64) -> f64 {
        let phi = self.phi();
        let u = phi * k + self.rho;
        let r = (u * u + 1.0 - self.rho * self.rho).sqrt();
        0.5 * self.theta * (1.0 + self.rho * phi * k + r)
    }

    /// Implied vol at log-moneyness `k` for expiry `t`.
    pub fn vol(&self, k: f64, t: f64) -> f64 {
        (self.total_variance(k).max(1e-12) / t.max(MIN_TIME)).sqrt()
    }

    /// `[w, w_k, w_kk]` — all closed form, identically to SSVI:
    ///
    /// ```text
    /// w_k  = theta phi / 2 (rho + u/R)
    /// w_kk = theta phi^2 (1 - rho^2) / (2 R^3)
    /// ```
    ///
    /// `w_kk > 0` always, so a slice's own density never degenerates from
    /// curvature alone — only the `g(k)` combination can.
    pub fn k_derivatives(&self, k: f64) -> [f64; 3] {
        let (theta, rho) = (self.theta, self.rho);
        let phi = self.phi();
        let u = phi * k + rho;
        let r = (u * u + 1.0 - rho * rho).sqrt();
        [
            0.5 * theta * (1.0 + rho * phi * k + r),
            0.5 * theta * phi * (rho + u / r),
            0.5 * theta * phi * phi * (1.0 - rho * rho) / (r * r * r),
        ]
    }

    /// Derivatives with respect to the parameters themselves, which is
    /// what a time derivative needs once `(theta, psi, rho)` are
    /// functions of `t`:
    ///
    /// ```text
    /// dw/dtheta = (w - k w_k) / theta
    /// dw/dpsi   = k w_k / psi
    /// dw/drho   = psi k (1 + 1/R) / 2
    /// ```
    ///
    /// Holding `rho` constant and setting `psi = theta phi(theta)`
    /// recovers SSVI's `dw/dt = (dtheta/dt)[w/theta + k (phi'/phi) w_k]`
    /// exactly — the two derivations agree, which is the cheapest
    /// available check that both are right.
    pub fn param_derivatives(&self, k: f64) -> [f64; 3] {
        let phi = self.phi();
        let u = phi * k + self.rho;
        let r = (u * u + 1.0 - self.rho * self.rho).sqrt();
        let [w, w_k, _] = self.k_derivatives(k);
        [
            (w - k * w_k) / self.theta,
            k * w_k / self.psi,
            0.5 * self.psi * k * (1.0 + 1.0 / r),
        ]
    }

    /// Gatheral's butterfly function `g(k)` for this slice.
    pub fn butterfly_g(&self, k: f64) -> f64 {
        let [w, w_k, w_kk] = self.k_derivatives(k);
        crate::equity::smoothed_surface::butterfly_g(
            &VarianceDerivatives {
                w,
                dk: w_k,
                dkk: w_kk,
                dt: 0.0,
            },
            k,
        )
    }

    /// Admissible parameters plus the Gatheral-Jacquier butterfly
    /// bounds `psi (1 + |rho|) <= 4` and `psi^2/theta (1 + |rho|) <= 4`
    /// — sufficient (not necessary) for a non-negative density.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        if self.theta <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "essvi params",
                "theta (ATM total variance) must be positive",
            ));
        }
        if self.psi <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "essvi params",
                "psi must be positive",
            ));
        }
        // strictly inside: at |rho| = 1 the curvature term
        // theta phi^2 (1 - rho^2) vanishes and R = |phi k + rho| can
        // reach zero, so the slice degenerates and w_kk divides by zero.
        // (Note a Rust range is half-open, so `(-1.0..1.0).contains`
        // would admit -1.0.)
        if !(self.rho > -1.0 && self.rho < 1.0) {
            return Err(RustyQLibError::invalid_input(
                "essvi params",
                "rho must lie strictly inside (-1, 1)",
            ));
        }
        let one_rho = 1.0 + self.rho.abs();
        if self.psi * one_rho > 4.0 || self.psi * self.psi / self.theta * one_rho > 4.0 {
            return Err(RustyQLibError::invalid_input(
                "essvi params",
                "butterfly bound violated (psi (1+|rho|) <= 4 and psi^2/theta (1+|rho|) <= 4)",
            ));
        }
        Ok(())
    }
}

// ── A fitted surface ────────────────────────────────────────────────────

/// One fitted expiry of an [`EssviSurfaceFit`].
#[derive(Debug, Clone)]
pub struct EssviSlice {
    /// Expiry time (year fraction).
    pub t: f64,
    /// Forward this slice's log-moneyness is measured against.
    pub forward: f64,
    pub params: EssviParams,
    /// Fit error in implied vol against the pillar quotes.
    pub rmse: f64,
    pub converged: bool,
    /// The quoted log-moneyness span the fit is anchored on.
    pub k_range: (f64, f64),
    /// Minimum of `g(k)` over the quoted span; negative means the fitted
    /// slice carries butterfly arbitrage there.
    pub min_g: f64,
}

/// A per-expiry eSSVI fit: one `(theta, psi, rho)` triple per pillar,
/// calibrated in time order with a calendar penalty, and interpolated
/// **in the parameters** between pillars so every intermediate expiry is
/// itself an eSSVI slice.
#[derive(Debug, Clone)]
pub struct EssviSurfaceFit {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    /// Slices in increasing expiry order.
    pub slices: Vec<EssviSlice>,
    /// Input expiries skipped for having too few quotes.
    pub skipped_slices: usize,
    /// Largest total-variance decrease in time found by a dense scan over
    /// the quoted region (0 = calendar-clean). This is the definition
    /// checked numerically; see the module docs on scope.
    pub max_calendar_crossing: f64,
}

/// Calibration weights. The two penalties are what make an eSSVI fit a
/// *constrained* fit rather than a per-slice one, so they are the knob
/// that trades in-sample accuracy against admissibility of the fitted
/// surface — worth sweeping rather than assuming.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssviFitConfig {
    /// Weight on the calendar-crossing penalty. Crossings are
    /// total-variance scale (O(1e-3)), so the default makes a 1e-3
    /// crossing dominate a smile's worth of vol residuals. Set to zero to
    /// fit each slice independently and measure what the constraint costs.
    pub calendar_penalty: f64,
    /// Weight on the butterfly-bound penalties (the bounded quantities
    /// are O(1), vol residuals O(0.01)).
    pub butterfly_penalty: f64,
}

impl Default for EssviFitConfig {
    fn default() -> Self {
        EssviFitConfig {
            calendar_penalty: 1000.0,
            butterfly_penalty: 10.0,
        }
    }
}
/// Log-moneyness grid used for the calendar penalty and the crossing scan.
const CALENDAR_GRID: usize = 40;
/// Largest total-variance crossing [`EssviSurfaceFit::validate`] tolerates.
///
/// A penalized fit drives the crossing down until the optimizer's own
/// convergence tolerance stops it, not to zero, so the bar has to be
/// economic rather than exact. At a typical `theta = 0.04` and one year,
/// a crossing of `1e-6` in total variance is a move of
/// `1e-6 / (2 sigma t) ~ 0.025` vol basis points — far below anything
/// quotable, while the crossings that matter (SPY at zero penalty
/// reaches `4.6e-3`, i.e. over a vol point) remain flagged.
const CALENDAR_TOLERANCE: f64 = 1e-6;
/// Plausible range for the skew scale `psi`. Equity smiles live near the
/// bottom of it: the ATM skew is `rho psi = 2 sigma t (dsigma/dk)`, so a
/// steep one-year name at 21% vol with `dsigma/dk = -0.10` implies
/// `psi ~ 0.07`. Values of order one would breach the butterfly bound
/// `psi^2/theta (1 + |rho|) <= 4` outright.
const PSI_MIN: f64 = 0.005;
const PSI_MAX: f64 = 3.0;

impl EssviSurfaceFit {
    /// Fit one eSSVI slice per pillar expiry of `surface`.
    ///
    /// `theta_t` is read off the surface at each expiry's forward rather
    /// than fitted — so the ATM term structure is reproduced exactly, and
    /// the only difference from
    /// [`SsviSurfaceFit`](crate::equity::svi::SsviSurfaceFit) is that
    /// `(psi, rho)` are free per slice instead of tied to three global
    /// numbers. That makes the SSVI/eSSVI comparison a clean experiment
    /// in exactly one variable.
    ///
    /// Slices are fitted in time order, each warm-started from its
    /// predecessor and penalized for crossing it, so calendar structure
    /// is built in rather than repaired.
    pub fn fit(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
    ) -> Result<EssviSurfaceFit, RustyQLibError> {
        Self::fit_with(surface, forward, &EssviFitConfig::default())
    }

    /// [`fit`](Self::fit) with explicit calibration weights — the entry
    /// point for measuring what the calendar constraint costs in fit.
    pub fn fit_with(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
        config: &EssviFitConfig,
    ) -> Result<EssviSurfaceFit, RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "essvi fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };

        // ── gather the pillars in time order ────────────────────────────
        struct Pillar {
            t: f64,
            forward: f64,
            theta: f64,
            quotes: Vec<(f64, f64)>,
            k_range: (f64, f64),
        }
        let mut pillars: Vec<Pillar> = Vec::new();
        let mut skipped = 0usize;
        for (tenor, smile) in expiries.iter().zip(&smiles) {
            let t = match tenor {
                Tenor::YearFraction(t) => *t,
                Tenor::Date(_) => continue, // to_input never emits dates
            };
            // two free parameters per slice; three quotes keeps it honest
            if smile.len() < 3 || t <= 0.0 {
                skipped += 1;
                continue;
            }
            let f = forward(t);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let k = match coordinate {
                        SmileCoordinate::Strike => (x / f).ln(),
                        SmileCoordinate::Moneyness => x.ln(),
                        SmileCoordinate::LogMoneyness => x,
                    };
                    lo = lo.min(k);
                    hi = hi.max(k);
                    (k, vol)
                })
                .collect();
            let atm = surface.vol(f, f, t);
            pillars.push(Pillar {
                t,
                forward: f,
                theta: atm * atm * t,
                quotes,
                k_range: (lo, hi),
            });
        }
        if pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "essvi fit",
                format!("no expiry has the three quotes an eSSVI fit needs ({skipped} skipped)"),
            ));
        }
        pillars.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        // calendar floor on the ATM term structure: theta cannot decrease
        for i in 1..pillars.len() {
            if pillars[i].theta < pillars[i - 1].theta {
                pillars[i].theta = pillars[i - 1].theta;
            }
        }

        // ── fit slice by slice, each anchored on the previous ───────────
        let mut slices: Vec<EssviSlice> = Vec::with_capacity(pillars.len());
        for pillar in &pillars {
            let previous = slices.last().map(|s: &EssviSlice| s.params);
            // the calendar penalty is evaluated over the union of this
            // slice's span and the previous one's, so a crossing cannot
            // hide just outside the current quotes
            let (lo, hi) = match slices.last() {
                Some(prev) => (
                    pillar.k_range.0.min(prev.k_range.0),
                    pillar.k_range.1.max(prev.k_range.1),
                ),
                None => pillar.k_range,
            };
            let cal_grid: Vec<f64> = (0..=CALENDAR_GRID)
                .map(|i| lo + (hi - lo) * i as f64 / CALENDAR_GRID as f64)
                .collect();
            let fit = fit_slice(
                pillar.theta,
                &pillar.quotes,
                pillar.t,
                previous,
                &cal_grid,
                config,
            );
            let min_g = (0..=200)
                .map(|i| {
                    let k = pillar.k_range.0
                        + (pillar.k_range.1 - pillar.k_range.0) * i as f64 / 200.0;
                    fit.0.butterfly_g(k)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(EssviSlice {
                t: pillar.t,
                forward: pillar.forward,
                params: fit.0,
                rmse: fit.1,
                converged: fit.2,
                k_range: pillar.k_range,
                min_g,
            });
        }

        // ── residual calendar tension, measured not assumed ─────────────
        let mut max_crossing: f64 = 0.0;
        for pair in slices.windows(2) {
            let (lo, hi) = (
                pair[0].k_range.0.min(pair[1].k_range.0),
                pair[0].k_range.1.max(pair[1].k_range.1),
            );
            for i in 0..=CALENDAR_GRID {
                let k = lo + (hi - lo) * i as f64 / CALENDAR_GRID as f64;
                let crossing =
                    pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
                max_crossing = max_crossing.max(crossing);
            }
        }

        Ok(EssviSurfaceFit {
            reference_date: surface.reference_date(),
            day_count: surface.day_count(),
            slices,
            skipped_slices: skipped,
            max_calendar_crossing: max_crossing,
        })
    }

    /// The slice pair bracketing `t`, with the interpolation weight on
    /// the later slice.
    fn bracket(&self, t: f64) -> (&EssviSlice, &EssviSlice, f64) {
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

    /// Slope of the ATM term structure over the last fitted segment,
    /// used to continue `theta` beyond the final pillar.
    fn last_theta_slope(&self) -> f64 {
        let n = self.slices.len();
        if n == 1 {
            let s = &self.slices[0];
            return s.params.theta / s.t;
        }
        let (prev, last) = (&self.slices[n - 2], &self.slices[n - 1]);
        ((last.params.theta - prev.params.theta) / (last.t - prev.t)).max(0.0)
    }

    /// The parameter triple at `t` and its time derivative, under the
    /// interpolation and extrapolation conventions:
    ///
    /// - **below the first pillar**, `phi` and `rho` are held and
    ///   `theta` accrues proportionally, so total variance scales like
    ///   `t` — the same rule the per-expiry smoothers use;
    /// - **between pillars**, `(theta, psi, rho)` are linear in `t`, so
    ///   every intermediate expiry is a genuine eSSVI slice;
    /// - **beyond the last pillar**, the smile shape `(psi, rho)` is held
    ///   and `theta` continues at the last segment's slope, so variance
    ///   keeps accruing.
    fn params_at(&self, t: f64) -> (EssviParams, [f64; 3]) {
        let (a, b, weight) = self.bracket(t);
        if t <= a.t {
            // proportional accrual: phi and rho fixed, theta ~ t
            let scale = (t / a.t).min(1.0);
            let p = EssviParams {
                theta: a.params.theta * scale,
                psi: a.params.psi * scale,
                rho: a.params.rho,
            };
            return (p, [a.params.theta / a.t, a.params.psi / a.t, 0.0]);
        }
        if a.t == b.t {
            // beyond the last pillar
            let slope = self.last_theta_slope();
            let p = EssviParams {
                theta: a.params.theta + slope * (t - a.t),
                psi: a.params.psi,
                rho: a.params.rho,
            };
            return (p, [slope, 0.0, 0.0]);
        }
        let dt = b.t - a.t;
        let (pa, pb) = (&a.params, &b.params);
        let p = EssviParams {
            theta: pa.theta + (pb.theta - pa.theta) * weight,
            psi: pa.psi + (pb.psi - pa.psi) * weight,
            rho: pa.rho + (pb.rho - pa.rho) * weight,
        };
        (
            p,
            [
                (pb.theta - pa.theta) / dt,
                (pb.psi - pa.psi) / dt,
                (pb.rho - pa.rho) / dt,
            ],
        )
    }

    /// The eSSVI slice in force at `t` (interpolated between pillars).
    pub fn params(&self, t: f64) -> EssviParams {
        self.params_at(t.max(MIN_TIME)).0
    }

    /// Implied vol for an absolute `strike` at `t`.
    pub fn vol(&self, strike: f64, t: f64) -> f64 {
        <Self as SmoothedSurface>::vol(self, strike, t)
    }

    /// Dupire local vol at underlying `level` and time `t`.
    pub fn local_vol(&self, level: f64, t: f64) -> f64 {
        <Self as SmoothedSurface>::local_vol(self, level, t)
    }

    /// [`local_vol`](Self::local_vol) plus whether a guard fired.
    pub fn local_vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        <Self as SmoothedSurface>::local_vol_checked(self, level, t)
    }

    /// Sample [`Self::local_vol`] on a `levels` x `times` grid.
    pub fn local_vol_grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        <Self as SmoothedSurface>::local_vol_grid(self, levels, times)
    }

    /// Per-slice butterfly bounds and the measured calendar crossing.
    pub fn validate(&self) -> Result<(), RustyQLibError> {
        for slice in &self.slices {
            slice.params.validate()?;
        }
        if self.max_calendar_crossing > CALENDAR_TOLERANCE {
            return Err(RustyQLibError::invalid_input(
                "essvi fit",
                format!(
                    "total variance decreases in time by up to {:.3e} over the quoted region",
                    self.max_calendar_crossing
                ),
            ));
        }
        Ok(())
    }

    /// Sample the fit into the canonical pricing [`VolSurface`].
    pub fn to_vol_surface(&self, samples: usize) -> Result<VolSurface, VolError> {
        let n = samples.max(5);
        let tenors: Vec<Tenor> = self
            .slices
            .iter()
            .map(|s| Tenor::YearFraction(s.t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = self
            .slices
            .iter()
            .map(|slice| {
                let (lo, hi) = slice.k_range;
                (0..n)
                    .map(|i| {
                        let k = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                        (slice.forward * k.exp(), slice.params.vol(k, slice.t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(&tenors, &smiles, self.reference_date, self.day_count)
    }

    /// Fit-quality metadata for the surface document.
    pub fn metadata(&self) -> serde_json::Value {
        let slices: Vec<serde_json::Value> = self
            .slices
            .iter()
            .map(|s| {
                serde_json::json!({
                    "t": s.t,
                    "forward": s.forward,
                    "params": {
                        "theta": s.params.theta,
                        "psi": s.params.psi,
                        "rho": s.params.rho,
                        "atm_skew": s.params.rho * s.params.psi,
                    },
                    "rmse_vol_bps": s.rmse * 1e4,
                    "converged": s.converged,
                    "min_butterfly_g": s.min_g,
                    "k_range": [s.k_range.0, s.k_range.1],
                })
            })
            .collect();
        serde_json::json!({
            "model": "per-expiry eSSVI (Hendriks-Martini), parameters interpolated in time",
            "slices": slices,
            "skipped_slices": self.skipped_slices,
            "max_calendar_crossing": self.max_calendar_crossing,
            "calendar_check": "numerical: total variance non-decreasing in t on a dense \
                               log-moneyness grid (not the analytic Hendriks-Martini domain)",
            "static_arbitrage_free": self.validate().is_ok(),
        })
    }
}

/// Calibrate one slice's `(psi, rho)` at fixed `theta`: vol residuals,
/// plus penalties for the butterfly bounds and for crossing `previous`.
/// Returns `(params, vol rmse, converged)`.
fn fit_slice(
    theta: f64,
    quotes: &[(f64, f64)],
    t: f64,
    previous: Option<EssviParams>,
    calendar_grid: &[f64],
    config: &EssviFitConfig,
) -> (EssviParams, f64, bool) {
    // warm start from the previous slice where there is one; otherwise a
    // generic equity smile (negative skew, moderate curvature)
    let (psi0, rho0) = match previous {
        Some(p) => (p.psi.clamp(PSI_MIN, PSI_MAX), p.rho.clamp(-0.95, 0.95)),
        None => (atm_skew_scale(quotes, theta, t), -0.6),
    };
    debug_assert!(rho0.abs() < RHO_CAP, "start point must be inside the cap");
    let x0 = vec![psi0.ln(), (rho0 / RHO_CAP).atanh()];
    let unpack = |u: &[f64]| EssviParams {
        theta,
        psi: u[0].exp().max(PSI_MIN),
        rho: RHO_CAP * u[1].tanh(),
    };
    let residuals = |u: &[f64]| -> Vec<f64> {
        let p = unpack(u);
        let mut out: Vec<f64> = quotes
            .iter()
            .map(|&(k, vol)| p.vol(k, t) - vol)
            .collect();
        // Gatheral-Jacquier butterfly bounds, as one-sided penalties
        let one_rho = 1.0 + p.rho.abs();
        out.push(config.butterfly_penalty * (p.psi * one_rho - 4.0).max(0.0));
        out.push(config.butterfly_penalty * (p.psi * p.psi / theta * one_rho - 4.0).max(0.0));
        // calendar: this slice must not dip below its predecessor
        if config.calendar_penalty > 0.0 {
            if let Some(prev) = previous {
                for &k in calendar_grid {
                    out.push(
                        config.calendar_penalty
                            * (prev.total_variance(k) - p.total_variance(k)).max(0.0),
                    );
                }
            }
        }
        out
    };
    let fit = levenberg_marquardt(&OptimConfig::new(1e-14, 200), &residuals, None, &x0);
    let params = unpack(&fit.x);
    let rmse = (quotes
        .iter()
        .map(|&(k, vol)| (params.vol(k, t) - vol).powi(2))
        .sum::<f64>()
        / quotes.len() as f64)
        .sqrt();
    (params, rmse, fit.converged)
}

/// A starting `psi` from the quoted ATM skew: `dw/dk|_0 = rho psi`, and
/// `dw/dk = 2 sigma t dsigma/dk`, so with a nominal `|rho| = 0.6` the
/// scale follows from a finite difference of the two quotes straddling
/// the money. Falls back to a generic value when the smile is too sparse
/// or flat to say anything.
fn atm_skew_scale(quotes: &[(f64, f64)], theta: f64, t: f64) -> f64 {
    let mut sorted: Vec<(f64, f64)> = quotes.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let idx = sorted.partition_point(|&(k, _)| k < 0.0);
    if idx == 0 || idx >= sorted.len() {
        return DEFAULT_PSI;
    }
    let (k0, v0) = sorted[idx - 1];
    let (k1, v1) = sorted[idx];
    if (k1 - k0).abs() < 1e-9 {
        return DEFAULT_PSI;
    }
    let dsigma_dk = (v1 - v0) / (k1 - k0);
    let sigma_atm = (theta / t.max(MIN_TIME)).sqrt();
    let dw_dk = 2.0 * sigma_atm * t * dsigma_dk;
    let scale = dw_dk.abs() / 0.6;
    if scale < PSI_MIN {
        DEFAULT_PSI
    } else {
        scale.min(PSI_MAX)
    }
}

/// Fallback skew scale when the quotes say nothing usable about the ATM
/// slope — a mid-range equity value.
const DEFAULT_PSI: f64 = 0.1;
/// Ceiling on `|rho|` in the calibration transform.
///
/// Plain `tanh` saturates to exactly `1.0` in `f64` for arguments beyond
/// about 19, and the optimizer will happily go there: the ATM skew is
/// `rho psi`, so on a nearly symmetric smile the pair is only identified
/// through its product and `(rho, psi) -> (-1, 0)` is a free direction
/// along that ridge. At `|rho| = 1` the slice degenerates — the
/// curvature `theta phi^2 (1 - rho^2)` vanishes and `R` can reach zero —
/// so the transform is capped strictly inside instead.
const RHO_CAP: f64 = 0.9999;

/// eSSVI reaches local volatility with **closed-form** derivatives in
/// both strike and time. The time derivative runs through the parameter
/// paths, `dw/dt = w_theta theta' + w_psi psi' + w_rho rho'`, which is
/// what lets the skew term structure twist — the thing SSVI's single
/// global `rho` cannot do.
impl SmoothedSurface for EssviSurfaceFit {
    fn forward(&self, t: f64) -> f64 {
        let (a, b, weight) = self.bracket(t);
        a.forward + (b.forward - a.forward) * weight
    }

    fn total_variance(&self, k: f64, t: f64) -> f64 {
        self.params_at(t.max(MIN_TIME)).0.total_variance(k)
    }

    fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives {
        let (p, slopes) = self.params_at(t.max(MIN_TIME));
        let [w, dk, dkk] = p.k_derivatives(k);
        let [w_theta, w_psi, w_rho] = p.param_derivatives(k);
        VarianceDerivatives {
            w,
            dk,
            dkk,
            dt: w_theta * slopes[0] + w_psi * slopes[1] + w_rho * slopes[2],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equity::smoothed_surface::numeric_k_derivatives;
    use crate::equity::svi::Ssvi;

    /// A realistic one-year equity slice: 21% ATM vol (`theta = 0.045`)
    /// and an ATM skew of `rho psi = -0.048` in total variance, i.e.
    /// about `-11` vol points per unit log-moneyness.
    fn params() -> EssviParams {
        EssviParams {
            theta: 0.045,
            psi: 0.08,
            rho: -0.6,
        }
    }

    #[test]
    fn atm_level_and_skew_are_exact_by_construction() {
        let p = params();
        assert!((p.total_variance(0.0) - p.theta).abs() < 1e-15);
        let [_, w_k, _] = p.k_derivatives(0.0);
        assert!(
            (w_k - p.rho * p.psi).abs() < 1e-14,
            "ATM skew {w_k} vs rho psi {}",
            p.rho * p.psi
        );
    }

    #[test]
    fn strike_derivatives_match_finite_differences() {
        let p = params();
        for k in [-0.4, -0.1, 0.0, 0.25, 0.6] {
            let [w, dk, dkk] = p.k_derivatives(k);
            let [wn, dkn, dkkn] = numeric_k_derivatives(|kk| p.total_variance(kk), k);
            assert!((w - wn).abs() < 1e-15, "w at {k}");
            // the numeric reference carries O(h^2) truncation error on
            // the shared 1e-4 stencil, so 1e-8 would test the stencil
            assert!((dk - dkn).abs() < 1e-6, "w_k at {k}: {dk} vs {dkn}");
            assert!((dkk - dkkn).abs() < 1e-4, "w_kk at {k}: {dkk} vs {dkkn}");
        }
    }

    #[test]
    fn parameter_derivatives_match_finite_differences() {
        // the analytic dw/dtheta, dw/dpsi, dw/drho drive the whole time
        // derivative, so bump each parameter and check
        let p = params();
        let h = 1e-7;
        for k in [-0.35, 0.0, 0.3] {
            let [d_theta, d_psi, d_rho] = p.param_derivatives(k);
            let bump = |dt: f64, dp: f64, dr: f64| {
                EssviParams {
                    theta: p.theta + dt,
                    psi: p.psi + dp,
                    rho: p.rho + dr,
                }
                .total_variance(k)
            };
            let n_theta = (bump(h, 0.0, 0.0) - bump(-h, 0.0, 0.0)) / (2.0 * h);
            let n_psi = (bump(0.0, h, 0.0) - bump(0.0, -h, 0.0)) / (2.0 * h);
            let n_rho = (bump(0.0, 0.0, h) - bump(0.0, 0.0, -h)) / (2.0 * h);
            assert!((d_theta - n_theta).abs() < 1e-6, "dw/dtheta at {k}");
            assert!((d_psi - n_psi).abs() < 1e-6, "dw/dpsi at {k}");
            assert!((d_rho - n_rho).abs() < 1e-6, "dw/drho at {k}");
        }
    }

    #[test]
    fn ssvi_is_the_special_case_with_frozen_rho() {
        // pin rho and set psi = theta phi(theta) from the power law: eSSVI
        // must reproduce SSVI's total variance *and* its time derivative
        // to machine precision. This is the regression test the whole
        // design was arranged to make available.
        let ssvi = Ssvi {
            rho: -0.55,
            eta: 0.9,
            gamma: 0.45,
            theta_pillars: vec![(0.25, 0.012), (0.5, 0.023), (1.0, 0.045), (2.0, 0.09)],
        };
        for t in [0.3, 0.6, 1.3] {
            let theta = ssvi.theta(t);
            let psi = theta * ssvi.phi(theta);
            let p = EssviParams {
                theta,
                psi,
                rho: ssvi.rho,
            };
            for k in [-0.5, -0.2, 0.0, 0.3] {
                assert!(
                    (p.total_variance(k) - ssvi.total_variance(k, t)).abs() < 1e-15,
                    "w at k={k} t={t}"
                );
                // dw/dt through the parameter paths must equal SSVI's
                // own closed form: theta' from the pillars, psi' by the
                // chain rule through phi(theta)
                let d = ssvi.variance_derivatives(k, t);
                let theta_slope = ssvi.theta_slope(t);
                let h = 1e-7;
                let psi_of = |th: f64| th * ssvi.phi(th);
                let dpsi_dtheta = (psi_of(theta + h) - psi_of(theta - h)) / (2.0 * h);
                let [w_theta, w_psi, _] = p.param_derivatives(k);
                let dt_essvi = w_theta * theta_slope + w_psi * dpsi_dtheta * theta_slope;
                assert!(
                    (dt_essvi - d.dt).abs() < 1e-6,
                    "dw/dt at k={k} t={t}: {dt_essvi} vs {}",
                    d.dt
                );
            }
        }
    }

    fn surface_from(slices: &[(f64, EssviParams)], forward: f64) -> VolSurface {
        let expiries: Vec<Tenor> = slices
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = slices
            .iter()
            .map(|&(t, p)| {
                (0..13)
                    .map(|i| {
                        let k = -0.3 + i as f64 * 0.05;
                        (forward * k.exp(), p.vol(k, t))
                    })
                    .collect()
            })
            .collect();
        VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap()
    }

    #[test]
    fn fit_recovers_a_twisting_skew_term_structure() {
        // the case SSVI structurally cannot match: rho steepening into
        // the short end while the level rises
        // 6m at 20% vol with a steep skew, 18m at 22% with a flat one
        let front = EssviParams {
            theta: 0.02,
            psi: 0.067,
            rho: -0.75,
        };
        let back = EssviParams {
            theta: 0.0726,
            psi: 0.151,
            rho: -0.35,
        };
        let surface = surface_from(&[(0.5, front), (1.5, back)], 100.0);
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        assert_eq!(fit.slices.len(), 2);
        assert_eq!(fit.skipped_slices, 0);
        for (slice, truth) in fit.slices.iter().zip([front, back]) {
            assert!(slice.rmse < 1e-5, "rmse {} at t={}", slice.rmse, slice.t);
            assert!(
                (slice.params.psi - truth.psi).abs() < 5e-3,
                "psi {} vs {}",
                slice.params.psi,
                truth.psi
            );
            assert!(
                (slice.params.rho - truth.rho).abs() < 2e-2,
                "rho {} vs {}",
                slice.params.rho,
                truth.rho
            );
            assert!(slice.min_g > 0.0, "min g {}", slice.min_g);
        }
        // the twist survived the fit: the front slice is more negatively
        // skewed than the back one
        assert!(fit.slices[0].params.rho < fit.slices[1].params.rho - 0.2);
        assert!(fit.max_calendar_crossing <= 1e-10, "{}", fit.max_calendar_crossing);
        fit.validate().unwrap();
    }

    #[test]
    fn local_vol_is_clean_and_time_derivative_matches_finite_differences() {
        let front = EssviParams {
            theta: 0.02,
            psi: 0.07,
            rho: -0.7,
        };
        let back = EssviParams {
            theta: 0.06,
            psi: 0.14,
            rho: -0.45,
        };
        let surface = surface_from(&[(0.5, front), (1.5, back)], 100.0);
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        let mut guarded = 0;
        for i in 0..=10 {
            let k: f64 = -0.28 + i as f64 * 0.056;
            // strictly inside a segment, where the parameter paths are smooth
            for t in [0.7, 1.0, 1.3] {
                let d = fit.variance_derivatives(k, t);
                let h = 1e-6;
                let dt_num = (fit.total_variance(k, t + h) - fit.total_variance(k, t - h))
                    / (2.0 * h);
                assert!(
                    (d.dt - dt_num).abs() < 1e-5,
                    "dw/dt at k={k} t={t}: {} vs {dt_num}",
                    d.dt
                );
                let (lv, g) = fit.local_vol_checked(100.0 * k.exp(), t);
                assert!(lv.is_finite() && lv > 0.0);
                guarded += g as usize;
            }
        }
        assert_eq!(guarded, 0, "a clean eSSVI fit needs no guards");
    }

    #[test]
    fn flat_essvi_gives_flat_local_vol() {
        // psi -> 0 collapses the smile to w = theta; with theta = sigma^2 t
        // the local vol must be sigma everywhere
        let vol = 0.26_f64;
        let slice = |t: f64| EssviParams {
            theta: vol * vol * t,
            psi: 1e-8,
            rho: -0.3,
        };
        let surface = surface_from(&[(0.5, slice(0.5)), (1.0, slice(1.0)), (2.0, slice(2.0))], 100.0);
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        for level in [80.0, 100.0, 125.0] {
            for t in [0.1, 0.5, 0.75, 1.0, 2.0, 2.5] {
                let (lv, guarded) = fit.local_vol_checked(level, t);
                assert!(!guarded, "level {level} t {t}");
                assert!((lv - vol).abs() < 5e-3, "level {level} t {t}: {lv}");
            }
        }
    }

    #[test]
    fn calendar_penalty_pushes_a_crossing_fit_apart() {
        // quotes that would cross if each slice were fitted alone: the
        // back expiry is quoted *below* the front one in the left wing
        let front = EssviParams {
            theta: 0.03,
            psi: 0.1,
            rho: -0.5,
        };
        let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
        let smiles = vec![
            (0..13)
                .map(|i| {
                    let k = -0.3 + i as f64 * 0.05;
                    (100.0 * k.exp(), front.vol(k, 0.5))
                })
                .collect::<Vec<_>>(),
            (0..13)
                .map(|i| {
                    let k: f64 = -0.3 + i as f64 * 0.05;
                    // deliberately too cheap in the left wing at 1y
                    let w = front.total_variance(k) * if k < -0.1 { 0.9 } else { 1.4 };
                    (100.0 * k.exp(), (w / 1.0_f64).sqrt())
                })
                .collect::<Vec<_>>(),
        ];
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        // the penalty cannot make the quotes consistent, but it must keep
        // the *fitted* surface from carrying a material crossing
        assert!(
            fit.max_calendar_crossing < 1e-3,
            "crossing {} survived the penalty",
            fit.max_calendar_crossing
        );
    }

    #[test]
    fn a_symmetric_smile_does_not_saturate_rho() {
        // when the ATM skew is ~0 the pair (rho, psi) is identified only
        // through its product, so the optimizer can slide toward
        // (rho, psi) = (-1, 0) — where the slice degenerates. The capped
        // transform has to keep every fitted slice strictly admissible.
        let flat = EssviParams {
            theta: 0.02,
            psi: 1e-4,
            rho: -0.05,
        };
        let surface = surface_from(&[(0.25, flat), (0.75, EssviParams { theta: 0.06, ..flat })], 100.0);
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        for slice in &fit.slices {
            assert!(
                slice.params.rho.abs() < 1.0,
                "rho saturated at t={}: {}",
                slice.t,
                slice.params.rho
            );
            assert!(slice.params.psi > 0.0, "psi collapsed at t={}", slice.t);
            slice.params.validate().unwrap_or_else(|e| {
                panic!("inadmissible slice at t={}: {e}", slice.t);
            });
        }
    }

    #[test]
    fn sparse_slices_are_skipped_not_fatal() {
        let p = params();
        let expiries = [Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)];
        let smiles = vec![
            vec![(95.0, 0.22), (105.0, 0.21)], // two quotes: below the minimum
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
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let fit = EssviSurfaceFit::fit(&surface, |_| 100.0).unwrap();
        assert_eq!(fit.slices.len(), 1);
        assert_eq!(fit.skipped_slices, 1);
    }

    #[test]
    fn validation_rejects_bad_params() {
        assert!(EssviParams {
            theta: -0.01,
            ..params()
        }
        .validate()
        .is_err());
        assert!(EssviParams {
            psi: 0.0,
            ..params()
        }
        .validate()
        .is_err());
        assert!(EssviParams {
            rho: -1.0,
            ..params()
        }
        .validate()
        .is_err());
        // butterfly bounds: psi (1 + |rho|) > 4 breaches the first,
        // and a psi of order one breaches psi^2/theta (1 + |rho|) <= 4
        // long before that — which is why equity psi is O(0.1)
        assert!(EssviParams {
            psi: 3.0,
            ..params()
        }
        .validate()
        .is_err());
        assert!(EssviParams {
            psi: 0.55,
            ..params()
        }
        .validate()
        .is_err());
        assert!(params().validate().is_ok());
    }
}
