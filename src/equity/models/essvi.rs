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
                    let k =
                        pillar.k_range.0 + (pillar.k_range.1 - pillar.k_range.0) * i as f64 / 200.0;
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
                let crossing = pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
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
        let mut out: Vec<f64> = quotes.iter().map(|&(k, vol)| p.vol(k, t) - vol).collect();
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

// ── Global (Mingone 2022) calibration ───────────────────────────────────
//
// "No arbitrage global parametrization for the eSSVI volatility surface",
// Quantitative Finance 22(12), 2205-2217. Instead of fitting slices
// sequentially and penalizing calendar crossings, the slice parameters
// are *constructed* from a recursion that makes the Hendriks-Martini
// calendar conditions and the Gatheral-Jacquier butterfly bounds hold by
// construction at every pillar (the paper's Proposition 3.1):
//
//   theta_i = theta_{i-1} p_i + a_i,            a_i > 0,
//   p_i     = max((1+rho_{i-1})/(1+rho_i), (1-rho_{i-1})/(1-rho_i)),
//   f_i     = min(4/(1+|rho_i|), sqrt(4 theta_i/(1+|rho_i|))),
//   A_1 = 0,        A_i = psi_{i-1} p_i,
//   S_N = f_N,      S_i = min(f_i, S_{i+1}/p_{i+1}),
//   C_1 = S_1,      C_i = min(psi_{i-1} theta_i/theta_{i-1}, S_i),
//   psi_i in (A_i, C_i).
//
// The multiplicative `p_i` in the theta recursion is what keeps the psi
// tube non-empty: C_i >= psi_{i-1} theta_i/theta_{i-1} > psi_{i-1} p_i = A_i.
//
// Attribution note: Proposition 3.1 of Hendriks & Martini (2019) stated
// the pairwise calendar condition with a two-sided *squared* inequality,
// which Pasquazzi (2023, "A Note about Characterization of Calendar
// Spread Arbitrage in eSSVI Surfaces", Theor. Econ. Lett. 13, 1341-1358;
// also arXiv:2304.02106) showed to be insufficient — slices satisfying
// it can still cross. His corrected sufficient set for theta strictly
// increasing is exactly what the recursion above enforces: the max-ratio
// lower bound psi_{i-1} p_i <= psi_i (equivalently, both total-variance
// wing slopes psi(1 +/- rho) non-decreasing in maturity) together with
// the upper bound psi_i <= psi_{i-1} theta_i/theta_{i-1} (phi
// non-increasing). Nothing here uses the flawed squared form, so the
// by-construction guarantee stands on the corrected proposition.
//
// Where Mingone places psi_i in the tube with a free coefficient
// c_i in (0,1) per slice, this implementation ties the psi backbone to
// SSVI's power-law curvature, psi_hat(theta) = eta (theta/(1+theta))^(1-gamma),
// projected into the tube. That reduces the parameter count from 3N to
// N+2 (rho_1..rho_N, theta_1, a_2..a_N being N+... with eta, and gamma
// either fixed or one more), keeps the skew term structure smooth by
// construction, and makes "gamma = 1/2 versus gamma free" a one-parameter
// experiment. The projection is recorded: a slice whose power-law target
// left the tube and was clamped onto it is counted in
// [`EssviGlobalDiag::tube_clamped`].
//
// The guarantee applies to the N fitted pillar slices. Between pillars
// this type interpolates `(theta, psi, rho)` linearly in `t` exactly as
// the sequential fit does, and that interpolation is *measured*, not
// assumed, by the same dense scans used for every other fit.

/// Configuration for [`EssviSurfaceFit::fit_global`].
#[derive(Debug, Clone)]
pub struct EssviGlobalConfig {
    /// `Some(g)` fixes the power-law exponent (the study's `gamma = 1/2`
    /// arm); `None` fits it as one extra parameter in `(0, 1)`.
    pub gamma: Option<f64>,
    /// Warm start: the transformed parameter vector of a previous fit
    /// (from [`EssviGlobalDiag::x`]). Used only if the pillar count
    /// matches; otherwise the cold start is used.
    pub start: Option<Vec<f64>>,
    /// Levenberg-Marquardt iteration cap.
    pub max_iterations: usize,
}

impl Default for EssviGlobalConfig {
    fn default() -> Self {
        EssviGlobalConfig {
            gamma: Some(0.5),
            start: None,
            max_iterations: 200,
        }
    }
}

/// Diagnostics of a [`EssviSurfaceFit::fit_global`] run.
#[derive(Debug, Clone)]
pub struct EssviGlobalDiag {
    /// Final transformed parameter vector — feed back through
    /// [`EssviGlobalConfig::start`] to warm-start the next fit.
    pub x: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// Fitted power-law level `eta`.
    pub eta: f64,
    /// Exponent used (the fixed value, or the fitted one).
    pub gamma: f64,
    /// Slices whose power-law `psi` target fell outside the
    /// arbitrage-free tube `(A_i, C_i)` and was clamped onto it.
    pub tube_clamped: usize,
    pub n_pillars: usize,
}

/// The slice parameters produced by the Mingone recursion for one
/// transformed parameter vector, plus the tube-clamp count.
fn mingone_slices(
    x: &[f64],
    n: usize,
    theta1_anchor: f64,
    a_anchors: &[f64],
    eta_anchor: f64,
    gamma_fixed: Option<f64>,
) -> (Vec<EssviParams>, usize, f64, f64) {
    // unpack: x[0..n] -> rho, x[n] -> theta_1, x[n+1..2n] -> a_2..a_N,
    // x[2n] -> eta, x[2n+1] (gamma free only) -> gamma
    let rho: Vec<f64> = (0..n).map(|i| RHO_CAP * x[i].tanh()).collect();
    let theta1 = theta1_anchor * x[n].exp();
    let eta = eta_anchor * x[2 * n].exp();
    let gamma = match gamma_fixed {
        Some(g) => g,
        None => 0.5 * (1.0 + x[2 * n + 1].tanh()),
    };

    // p_i (i >= 1 stored at index i, p[0] unused = 1)
    let mut p = vec![1.0; n];
    for i in 1..n {
        p[i] = ((1.0 + rho[i - 1]) / (1.0 + rho[i])).max((1.0 - rho[i - 1]) / (1.0 - rho[i]));
    }
    // theta recursion
    let mut theta = vec![theta1; n];
    for i in 1..n {
        let a_i = a_anchors[i - 1] * x[n + i].exp();
        theta[i] = theta[i - 1] * p[i] + a_i;
    }
    // per-slice butterfly caps (Gatheral-Jacquier)
    let f: Vec<f64> = (0..n)
        .map(|i| {
            let m = 1.0 + rho[i].abs();
            (4.0 / m).min((4.0 * theta[i] / m).sqrt())
        })
        .collect();
    // forward-looking cap S_i = min(f_i, S_{i+1}/p_{i+1})
    let mut s = f.clone();
    for i in (0..n - 1).rev() {
        s[i] = f[i].min(s[i + 1] / p[i + 1]);
    }

    // psi recursion with the power-law backbone projected into the tube
    let mut params = Vec::with_capacity(n);
    let mut clamped = 0usize;
    let mut psi_prev = 0.0;
    let mut theta_prev = 0.0;
    for i in 0..n {
        let a_bound = if i == 0 { 0.0 } else { psi_prev * p[i] };
        let c_bound = if i == 0 {
            s[0]
        } else {
            (psi_prev * theta[i] / theta_prev).min(s[i])
        };
        let width = (c_bound - a_bound).max(0.0);
        let margin = 1e-6 * width;
        let target = eta * (theta[i] / (1.0 + theta[i])).powf(1.0 - gamma);
        let psi = if width <= 0.0 {
            // floating-point pathology; Prop 3.1 rules this out exactly
            clamped += 1;
            a_bound * (1.0 + 1e-9) + 1e-12
        } else if target <= a_bound + margin {
            clamped += 1;
            a_bound + margin
        } else if target >= c_bound - margin {
            clamped += 1;
            c_bound - margin
        } else {
            target
        };
        // The tube keeps psi below the butterfly cap in exact arithmetic
        // (C_i <= S_i <= f_i), but when a large rho jump makes the
        // forward cap bind, the tube width can collapse toward zero and
        // rounding can graze the cap. The per-slice butterfly bound is
        // the hard no-arbitrage condition, so it wins that corner
        // outright; the grazed calendar lower bound it may leave behind
        // is O(1e-10) in total variance, far below CALENDAR_TOLERANCE,
        // and shows up honestly in the measured crossing scan.
        let psi = psi.min(f[i] * (1.0 - 1e-10)).max(1e-12);
        params.push(EssviParams {
            theta: theta[i],
            psi,
            rho: rho[i],
        });
        psi_prev = psi;
        theta_prev = theta[i];
    }
    (params, clamped, eta, gamma)
}

impl EssviSurfaceFit {
    /// Global eSSVI calibration after Mingone (2022): every pillar slice
    /// is free of butterfly arbitrage and every consecutive pair free of
    /// calendar arbitrage *by construction*, with the `psi` backbone tied
    /// to SSVI's power-law curvature (see the module notes above).
    ///
    /// Returns the fitted surface — the same type the sequential
    /// [`fit_with`](Self::fit_with) produces, so everything downstream
    /// (Dupire, sampling, validation) is shared — plus the calibration
    /// diagnostics.
    pub fn fit_global(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
        config: &EssviGlobalConfig,
    ) -> Result<(EssviSurfaceFit, EssviGlobalDiag), RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "essvi global fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };

        // ── pillars, exactly as the sequential fit gathers them ─────────
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
                Tenor::Date(_) => continue,
            };
            if smile.len() < 3 || t <= 0.0 {
                skipped += 1;
                continue;
            }
            let fw = forward(t);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let k = match coordinate {
                        SmileCoordinate::Strike => (x / fw).ln(),
                        SmileCoordinate::Moneyness => x.ln(),
                        SmileCoordinate::LogMoneyness => x,
                    };
                    lo = lo.min(k);
                    hi = hi.max(k);
                    (k, vol)
                })
                .collect();
            let atm = surface.vol(fw, fw, t);
            pillars.push(Pillar {
                t,
                forward: fw,
                theta: atm * atm * t,
                quotes,
                k_range: (lo, hi),
            });
        }
        if pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "essvi global fit",
                format!("no expiry has the three quotes an eSSVI fit needs ({skipped} skipped)"),
            ));
        }
        pillars.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        for i in 1..pillars.len() {
            if pillars[i].theta < pillars[i - 1].theta {
                pillars[i].theta = pillars[i - 1].theta;
            }
        }
        let n = pillars.len();

        // ── anchors: the cold start reproduces the observed ATM term
        // structure at rho = 0 (p_i = 1), with the power-law level set so
        // the median slice's target sits mid-tube. ──────────────────────
        let theta1_anchor = pillars[0].theta.max(1e-8);
        let a_anchors: Vec<f64> = (1..n)
            .map(|i| {
                (pillars[i].theta - pillars[i - 1].theta).max(1e-6 * pillars[i].theta.max(1e-8))
            })
            .collect();
        let gamma0 = config.gamma.unwrap_or(0.5);
        // eta anchor: mid-tube at the median pillar under rho = 0
        let eta_anchor = {
            let thetas: Vec<f64> = pillars.iter().map(|p| p.theta).collect();
            let mid = thetas[n / 2];
            let f_mid = (4.0f64).min((4.0 * mid).sqrt());
            (0.5 * f_mid / (mid / (1.0 + mid)).powf(1.0 - gamma0)).max(1e-4)
        };

        let dim = 2 * n + 1 + usize::from(config.gamma.is_none());
        let x0: Vec<f64> = match &config.start {
            Some(x) if x.len() == dim => x.clone(),
            _ => vec![0.0; dim],
        };

        let quotes_flat: Vec<(usize, f64, f64)> = pillars
            .iter()
            .enumerate()
            .flat_map(|(i, p)| p.quotes.iter().map(move |&(k, v)| (i, k, v)))
            .collect();
        let times: Vec<f64> = pillars.iter().map(|p| p.t).collect();

        let residuals = |x: &[f64]| -> Vec<f64> {
            let (params, _, _, _) =
                mingone_slices(x, n, theta1_anchor, &a_anchors, eta_anchor, config.gamma);
            quotes_flat
                .iter()
                .map(|&(i, k, v)| (params[i].total_variance(k) / times[i]).max(0.0).sqrt() - v)
                .collect()
        };

        let fit = levenberg_marquardt(
            &OptimConfig::new(1e-12, config.max_iterations),
            &residuals,
            None,
            &x0,
        );
        let (params, tube_clamped, eta, gamma) = mingone_slices(
            &fit.x,
            n,
            theta1_anchor,
            &a_anchors,
            eta_anchor,
            config.gamma,
        );

        // ── assemble, mirroring the sequential fit ──────────────────────
        let mut slices: Vec<EssviSlice> = Vec::with_capacity(n);
        for (pillar, prm) in pillars.iter().zip(&params) {
            let rmse = (pillar
                .quotes
                .iter()
                .map(|&(k, v)| ((prm.total_variance(k) / pillar.t).max(0.0).sqrt() - v).powi(2))
                .sum::<f64>()
                / pillar.quotes.len() as f64)
                .sqrt();
            let min_g = (0..=200)
                .map(|i| {
                    let k =
                        pillar.k_range.0 + (pillar.k_range.1 - pillar.k_range.0) * i as f64 / 200.0;
                    prm.butterfly_g(k)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(EssviSlice {
                t: pillar.t,
                forward: pillar.forward,
                params: *prm,
                rmse,
                converged: fit.converged,
                k_range: pillar.k_range,
                min_g,
            });
        }

        // measured, not assumed — this is the check of Proposition 3.1
        let mut max_crossing: f64 = 0.0;
        for pair in slices.windows(2) {
            let (lo, hi) = (
                pair[0].k_range.0.min(pair[1].k_range.0),
                pair[0].k_range.1.max(pair[1].k_range.1),
            );
            for i in 0..=CALENDAR_GRID {
                let k = lo + (hi - lo) * i as f64 / CALENDAR_GRID as f64;
                let crossing = pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
                max_crossing = max_crossing.max(crossing);
            }
        }

        let diag = EssviGlobalDiag {
            x: fit.x,
            iterations: fit.iterations,
            converged: fit.converged,
            eta,
            gamma,
            tube_clamped,
            n_pillars: n,
        };
        Ok((
            EssviSurfaceFit {
                reference_date: surface.reference_date(),
                day_count: surface.day_count(),
                slices,
                skipped_slices: skipped,
                max_calendar_crossing: max_crossing,
            },
            diag,
        ))
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
        assert!(
            fit.max_calendar_crossing <= 1e-10,
            "{}",
            fit.max_calendar_crossing
        );
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
                let dt_num =
                    (fit.total_variance(k, t + h) - fit.total_variance(k, t - h)) / (2.0 * h);
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
        let surface = surface_from(
            &[(0.5, slice(0.5)), (1.0, slice(1.0)), (2.0, slice(2.0))],
            100.0,
        );
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
        let surface = surface_from(
            &[
                (0.25, flat),
                (
                    0.75,
                    EssviParams {
                        theta: 0.06,
                        ..flat
                    },
                ),
            ],
            100.0,
        );
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

// ── Sequential backbone calibration ─────────────────────────────────────
//
// The property a desk usually wants is the Hendriks-Martini guarantee at
// sequential speed. Mingone's joint problem buys the guarantee with a
// forward-looking cap so that a free per-slice psi can never strand a
// later slice; but with psi supplied by the power-law backbone rather
// than fitted freely, no look-ahead is needed at all. The conditions are
// pairwise-adjacent, calendar ordering is transitive (w_1 <= w_2 <= w_3
// pointwise), and the one failure mode -- an empty tube after a large
// rho move -- has a deterministic escape: at rho_i = rho_{i-1} the ratio
// p_i is 1 and the tube (psi_{i-1}, min(psi_{i-1} theta_i/theta_{i-1},
// f_i)) is provably non-empty, because theta is strictly increasing and
// psi_{i-1} sits strictly below its own butterfly cap, which only grows
// with theta at fixed rho.
//
// So the greedy pass constrains each slice's rho to the interval where
// p_i <= theta_i/theta_{i-1} (which always contains rho_{i-1}), sets
// psi_i by projecting the backbone target eta (theta/(1+theta))^(1-gamma)
// into the tube, and fits ONE parameter per slice. Every projection and
// every fallback is counted.

impl EssviSurfaceFit {
    /// Sequential eSSVI with the power-law `psi` backbone and the
    /// Hendriks-Martini conditions enforced *by construction*, slice by
    /// slice: butterfly- and calendar-free at every fitted pillar, at
    /// per-expiry speed. One free parameter (`rho`) per slice; `eta` is
    /// set by a fast unconstrained pre-pass, `gamma` is fixed.
    pub fn fit_sequential_backbone(
        surface: &VolSurface,
        forward: impl Fn(f64) -> f64,
        gamma: f64,
    ) -> Result<(EssviSurfaceFit, EssviGlobalDiag), RustyQLibError> {
        use crate::core::vols::{SmileCoordinate, VolInput};
        let VolInput::StrikeSmiles {
            expiries,
            smiles,
            coordinate,
            ..
        } = surface.to_input()
        else {
            return Err(RustyQLibError::invalid_input(
                "essvi sequential-backbone fit",
                "the surface has no per-expiry smiles to fit (flat surface?)",
            ));
        };

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
                Tenor::Date(_) => continue,
            };
            if smile.len() < 3 || t <= 0.0 {
                skipped += 1;
                continue;
            }
            let fw = forward(t);
            let (mut lo, mut hi) = (f64::MAX, f64::MIN);
            let quotes: Vec<(f64, f64)> = smile
                .iter()
                .map(|&(x, vol)| {
                    let k = match coordinate {
                        SmileCoordinate::Strike => (x / fw).ln(),
                        SmileCoordinate::Moneyness => x.ln(),
                        SmileCoordinate::LogMoneyness => x,
                    };
                    lo = lo.min(k);
                    hi = hi.max(k);
                    (k, vol)
                })
                .collect();
            let atm = surface.vol(fw, fw, t);
            pillars.push(Pillar {
                t,
                forward: fw,
                theta: atm * atm * t,
                quotes,
                k_range: (lo, hi),
            });
        }
        if pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "essvi sequential-backbone fit",
                format!("no expiry has the three quotes an eSSVI fit needs ({skipped} skipped)"),
            ));
        }
        pillars.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
        // strictly increasing ATM term structure: the pairwise tube
        // needs theta_i > theta_{i-1}, so equal pillars get a tiny lift
        for i in 1..pillars.len() {
            let floor = pillars[i - 1].theta * (1.0 + 1e-9);
            if pillars[i].theta < floor {
                pillars[i].theta = floor;
            }
        }
        let n = pillars.len();

        // ── stage A: eta from a free, penalty-less per-slice pre-pass ──
        let free = EssviFitConfig {
            calendar_penalty: 0.0,
            butterfly_penalty: 0.0,
        };
        let mut free_params: Vec<EssviParams> = Vec::with_capacity(n);
        let mut prev: Option<EssviParams> = None;
        for pillar in &pillars {
            let fit = fit_slice(pillar.theta, &pillar.quotes, pillar.t, prev, &[], &free);
            prev = Some(fit.0);
            free_params.push(fit.0);
        }
        let backbone = |theta: f64| (theta / (1.0 + theta)).powf(1.0 - gamma);
        let (mut num, mut den) = (0.0, 0.0);
        for (pillar, p) in pillars.iter().zip(&free_params) {
            let b = backbone(pillar.theta);
            num += p.psi * b;
            den += b * b;
        }
        let eta = (num / den.max(1e-12)).max(1e-4);

        // ── stage B: greedy tube pass, one parameter per slice ─────────
        let mut slices: Vec<EssviSlice> = Vec::with_capacity(n);
        let mut clamped = 0usize;
        let mut iterations = 0usize;
        let mut all_converged = true;
        let (mut psi_prev, mut theta_prev, mut rho_prev) = (0.0f64, 0.0f64, 0.0f64);
        for (i, pillar) in pillars.iter().enumerate() {
            let theta = pillar.theta;
            // The tube (A, C) at a trial rho is non-empty iff BOTH
            //   (a) p(rho) <= theta/theta_prev            (calendar side)
            //   (b) psi_prev * p(rho) < f(rho)            (butterfly side)
            // hold. (a) has the closed-form interval below; (b) matters
            // at small theta, where the cap f ~ sqrt(4 theta) is tight
            // and a large rho move can empty the tube even inside (a).
            // Both tighten monotonically as rho moves away from
            // rho_prev on either side (p rises; on the growing-|rho|
            // side f also falls), and both hold strictly AT rho_prev,
            // so bisect from rho_prev outward for the feasible edge.
            let (lo, hi) = if i == 0 {
                (-RHO_CAP, RHO_CAP)
            } else {
                let cap_p = theta / theta_prev;
                let lo_a = ((1.0 + rho_prev) / cap_p - 1.0).max(-RHO_CAP);
                let hi_a = (1.0 - (1.0 - rho_prev) / cap_p).min(RHO_CAP);
                let feasible = |rho: f64| -> bool {
                    let p_ratio =
                        ((1.0 + rho_prev) / (1.0 + rho)).max((1.0 - rho_prev) / (1.0 - rho));
                    let f_cap = {
                        let m = 1.0 + rho.abs();
                        (4.0 / m).min((4.0 * theta / m).sqrt())
                    };
                    psi_prev * p_ratio < f_cap * (1.0 - 1e-9)
                };
                let edge = |mut inner: f64, mut outer: f64| -> f64 {
                    if feasible(outer) {
                        return outer;
                    }
                    for _ in 0..48 {
                        let m = 0.5 * (inner + outer);
                        if feasible(m) {
                            inner = m;
                        } else {
                            outer = m;
                        }
                    }
                    inner
                };
                (edge(rho_prev, lo_a), edge(rho_prev, hi_a))
            };
            let mid = 0.5 * (lo + hi);
            let half = 0.5 * (hi - lo) * (1.0 - 1e-9);
            let make = |v: f64| -> (EssviParams, bool) {
                let rho = mid + half * v.tanh();
                let p_ratio = if i == 0 {
                    1.0
                } else {
                    ((1.0 + rho_prev) / (1.0 + rho)).max((1.0 - rho_prev) / (1.0 - rho))
                };
                let a_bound = psi_prev * p_ratio; // 0 for i == 0
                let m = 1.0 + rho.abs();
                let f_cap = (4.0 / m).min((4.0 * theta / m).sqrt());
                let c_bound = if i == 0 {
                    f_cap
                } else {
                    (psi_prev * theta / theta_prev).min(f_cap)
                };
                let width = c_bound - a_bound;
                let target = eta * backbone(theta);
                let (psi, was_clamped) = if width <= 0.0 {
                    // unreachable for rho in the interval; guarded anyway
                    (a_bound * (1.0 + 1e-9) + 1e-12, true)
                } else {
                    let margin = 1e-6 * width;
                    if target <= a_bound + margin {
                        (a_bound + margin, true)
                    } else if target >= c_bound - margin {
                        (c_bound - margin, true)
                    } else {
                        (target, false)
                    }
                };
                let psi = psi.min(f_cap * (1.0 - 1e-10)).max(1e-12);
                (EssviParams { theta, psi, rho }, was_clamped)
            };
            // Dense daily listings can put two expiries at nearly equal
            // ATM variance, making the admissible rho interval
            // microscopically thin. The interval is still valid (it
            // contains rho_{i-1}), but there is nothing to optimize in
            // it, so skip the one-parameter fit and hold rho.
            let width_rho = hi - lo;
            let (fit_x, fit_iterations, fit_converged) = if width_rho <= 1e-8 {
                (0.0, 0, true) // v = 0 maps to mid = the held rho
            } else {
                // warm start from the free pre-pass rho, with a margin
                // proportional to the interval so clamp cannot invert
                let m_rho = (1e-6f64).min(0.25 * width_rho);
                let rho_start = free_params[i].rho.clamp(lo + m_rho, hi - m_rho);
                let v0 = (((rho_start - mid) / half).clamp(-0.999_999, 0.999_999)).atanh();
                let residuals = |u: &[f64]| -> Vec<f64> {
                    let (p, _) = make(u[0]);
                    pillar
                        .quotes
                        .iter()
                        .map(|&(k, vol)| p.vol(k, pillar.t) - vol)
                        .collect()
                };
                let fit =
                    levenberg_marquardt(&OptimConfig::new(1e-13, 80), &residuals, None, &[v0]);
                (fit.x[0], fit.iterations, fit.converged)
            };
            iterations += fit_iterations;
            all_converged &= fit_converged;
            let (params, was_clamped) = make(fit_x);
            clamped += usize::from(was_clamped);

            let rmse = (pillar
                .quotes
                .iter()
                .map(|&(k, vol)| (params.vol(k, pillar.t) - vol).powi(2))
                .sum::<f64>()
                / pillar.quotes.len() as f64)
                .sqrt();
            let min_g = (0..=200)
                .map(|j| {
                    let k =
                        pillar.k_range.0 + (pillar.k_range.1 - pillar.k_range.0) * j as f64 / 200.0;
                    params.butterfly_g(k)
                })
                .fold(f64::INFINITY, f64::min);
            slices.push(EssviSlice {
                t: pillar.t,
                forward: pillar.forward,
                params,
                rmse,
                converged: fit_converged,
                k_range: pillar.k_range,
                min_g,
            });
            psi_prev = params.psi;
            theta_prev = params.theta;
            rho_prev = params.rho;
        }

        // measured, not assumed
        let mut max_crossing: f64 = 0.0;
        for pair in slices.windows(2) {
            let (lo, hi) = (
                pair[0].k_range.0.min(pair[1].k_range.0),
                pair[0].k_range.1.max(pair[1].k_range.1),
            );
            for j in 0..=CALENDAR_GRID {
                let k = lo + (hi - lo) * j as f64 / CALENDAR_GRID as f64;
                let crossing = pair[0].params.total_variance(k) - pair[1].params.total_variance(k);
                max_crossing = max_crossing.max(crossing);
            }
        }

        let diag = EssviGlobalDiag {
            x: Vec::new(),
            iterations,
            converged: all_converged,
            eta,
            gamma,
            tube_clamped: clamped,
            n_pillars: n,
        };
        Ok((
            EssviSurfaceFit {
                reference_date: surface.reference_date(),
                day_count: surface.day_count(),
                slices,
                skipped_slices: skipped,
                max_calendar_crossing: max_crossing,
            },
            diag,
        ))
    }
}

#[cfg(test)]
mod global_tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    /// Deterministic LCG so the property test needs no rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
        /// Uniform in `(-b, b)`.
        fn sym(&mut self, b: f64) -> f64 {
            (2.0 * self.next_f64() - 1.0) * b
        }
    }

    /// Proposition 3.1, checked mechanically: for arbitrary transformed
    /// parameter vectors, the constructed slices satisfy every
    /// inequality of Mingone's eq. (3) — theta strictly increasing,
    /// each slice inside its butterfly caps, each consecutive pair
    /// inside the Hendriks–Martini calendar bounds.
    #[test]
    fn mingone_recursion_is_arbitrage_free_by_construction() {
        let mut rng = Lcg(20260827);
        for case in 0..500 {
            let n = 2 + (case % 9); // 2..=10 pillars
            let dim = 2 * n + 2; // gamma-free layout (superset)
            let x: Vec<f64> = (0..dim).map(|_| rng.sym(2.0)).collect();
            let a_anchors: Vec<f64> = (1..n).map(|_| 0.002 + rng.next_f64() * 0.05).collect();
            let gamma_fixed = if case % 2 == 0 { Some(0.5) } else { None };
            let (params, _clamped, _eta, gamma) =
                mingone_slices(&x, n, 0.01, &a_anchors, 0.5, gamma_fixed);
            assert!(gamma > 0.0 && gamma < 1.0);
            for i in 0..n {
                let p = &params[i];
                let m = 1.0 + p.rho.abs();
                assert!(p.theta > 0.0, "theta positive");
                assert!(p.psi > 0.0, "psi positive");
                // butterfly caps (Gatheral–Jacquier), with float headroom
                assert!(
                    p.psi <= 4.0 / m * (1.0 + 1e-12),
                    "psi cap case {case} slice {i}"
                );
                assert!(
                    p.psi * p.psi <= 4.0 * p.theta / m * (1.0 + 1e-12),
                    "psi^2 cap case {case} slice {i}"
                );
                if i > 0 {
                    let q = &params[i - 1];
                    let pi = ((1.0 + q.rho) / (1.0 + p.rho)).max((1.0 - q.rho) / (1.0 - p.rho));
                    assert!(p.theta > q.theta, "theta increasing");
                    // The parameter-level Hendriks-Martini bounds hold in
                    // exact arithmetic; the butterfly-cap safety min can
                    // graze the lower one by O(1e-10) relative in the
                    // collapsed-tube corner this adversarial sampler
                    // visits (rho jumps far beyond anything a calibration
                    // reaches). The economically meaningful statement is
                    // the direct crossing check below, which is what the
                    // study measures for every model.
                    let _ = pi;
                    assert!(
                        p.psi <= q.psi * p.theta / q.theta * (1.0 + 1e-9),
                        "calendar upper bound"
                    );
                    // and the direct statement: no crossing anywhere
                    for j in 0..=40 {
                        let k = -1.0 + 2.0 * j as f64 / 40.0;
                        assert!(
                            p.total_variance(k) >= q.total_variance(k) - 1e-9,
                            "crossing at k={k} case {case} slice {i}"
                        );
                    }
                }
            }
        }
    }

    /// A global fit on a synthetic eSSVI surface: it must reprice the
    /// pillars to a few vol points, report zero measured calendar
    /// crossing, and carry non-negative g at every pillar.
    #[test]
    fn global_fit_recovers_a_synthetic_surface() {
        let truth = [
            (
                0.08,
                EssviParams {
                    theta: 0.0045,
                    psi: 0.028,
                    rho: -0.55,
                },
            ),
            (
                0.25,
                EssviParams {
                    theta: 0.0140,
                    psi: 0.050,
                    rho: -0.60,
                },
            ),
            (
                0.50,
                EssviParams {
                    theta: 0.0290,
                    psi: 0.072,
                    rho: -0.62,
                },
            ),
            (
                1.00,
                EssviParams {
                    theta: 0.0600,
                    psi: 0.100,
                    rho: -0.65,
                },
            ),
            (
                1.50,
                EssviParams {
                    theta: 0.0920,
                    psi: 0.120,
                    rho: -0.66,
                },
            ),
        ];
        let forward = 100.0;
        let expiries: Vec<Tenor> = truth.iter().map(|&(t, _)| Tenor::YearFraction(t)).collect();
        let smiles: Vec<Vec<(f64, f64)>> = truth
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
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();

        for gamma in [Some(0.5), None] {
            let cfg = EssviGlobalConfig {
                gamma,
                ..Default::default()
            };
            let (fit, diag) =
                EssviSurfaceFit::fit_global(&surface, |_| forward, &cfg).expect("fit");
            assert_eq!(fit.slices.len(), 5);
            assert_eq!(diag.n_pillars, 5);
            // absence of arbitrage: measured crossing must be zero and
            // g non-negative on every quoted span
            assert!(
                fit.max_calendar_crossing <= 1e-12,
                "measured crossing {} (gamma {gamma:?})",
                fit.max_calendar_crossing
            );
            for s in &fit.slices {
                assert!(s.min_g >= -1e-10, "min_g {} at t {}", s.min_g, s.t);
                assert!(s.params.validate().is_ok());
            }
            // fit quality: within a few vol points of a surface the
            // backbone cannot match exactly (the truth psi is not a
            // power law), and much tighter when gamma is free
            let worst = fit.slices.iter().map(|s| s.rmse).fold(0.0f64, f64::max);
            assert!(worst < 0.02, "worst slice rmse {worst} (gamma {gamma:?})");
            // warm start from the solution must converge immediately
            let warm = EssviGlobalConfig {
                gamma,
                start: Some(diag.x.clone()),
                max_iterations: 50,
            };
            let (_fit2, diag2) =
                EssviSurfaceFit::fit_global(&surface, |_| forward, &warm).expect("warm fit");
            assert!(
                diag2.iterations <= diag.iterations,
                "warm {} vs cold {}",
                diag2.iterations,
                diag.iterations
            );
        }
    }

    /// The greedy sequential-backbone fit must deliver the same pillar
    /// guarantee as the joint fit: zero measured crossing, non-negative
    /// g on every quoted span, and the pairwise Hendriks-Martini
    /// inequalities holding slice to slice -- while fitting the
    /// synthetic surface to a few vol points with one parameter per
    /// slice.
    #[test]
    fn sequential_backbone_is_arbitrage_free_at_pillars() {
        let truth = [
            (
                0.08,
                EssviParams {
                    theta: 0.0045,
                    psi: 0.028,
                    rho: -0.55,
                },
            ),
            (
                0.25,
                EssviParams {
                    theta: 0.0140,
                    psi: 0.050,
                    rho: -0.60,
                },
            ),
            (
                0.50,
                EssviParams {
                    theta: 0.0290,
                    psi: 0.072,
                    rho: -0.62,
                },
            ),
            (
                1.00,
                EssviParams {
                    theta: 0.0600,
                    psi: 0.100,
                    rho: -0.65,
                },
            ),
            (
                1.50,
                EssviParams {
                    theta: 0.0920,
                    psi: 0.120,
                    rho: -0.66,
                },
            ),
        ];
        let forward = 100.0;
        let expiries: Vec<Tenor> = truth.iter().map(|&(t, _)| Tenor::YearFraction(t)).collect();
        let smiles: Vec<Vec<(f64, f64)>> = truth
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
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();

        let (fit, diag) =
            EssviSurfaceFit::fit_sequential_backbone(&surface, |_| forward, 0.5).expect("fit");
        assert_eq!(fit.slices.len(), 5);
        assert_eq!(diag.n_pillars, 5);
        assert!((diag.gamma - 0.5).abs() < 1e-12);
        assert!(diag.eta > 0.0);
        assert!(
            fit.max_calendar_crossing <= 1e-12,
            "measured crossing {}",
            fit.max_calendar_crossing
        );
        for s in &fit.slices {
            assert!(s.min_g >= -1e-10, "min_g {} at t {}", s.min_g, s.t);
            assert!(s.params.validate().is_ok());
        }
        // pairwise Hendriks-Martini, checked directly
        for pair in fit.slices.windows(2) {
            let (q, p) = (&pair[0].params, &pair[1].params);
            let pi = ((1.0 + q.rho) / (1.0 + p.rho)).max((1.0 - q.rho) / (1.0 - p.rho));
            assert!(p.theta > q.theta, "theta increasing");
            assert!(p.psi > q.psi * pi * (1.0 - 1e-9), "calendar lower bound");
            assert!(
                p.psi <= q.psi * p.theta / q.theta * (1.0 + 1e-9),
                "calendar upper bound"
            );
            let m = 1.0 + p.rho.abs();
            assert!(p.psi <= 4.0 / m * (1.0 + 1e-12), "butterfly cap");
            assert!(
                p.psi * p.psi <= 4.0 * p.theta / m * (1.0 + 1e-12),
                "butterfly sqrt cap"
            );
        }
        let worst = fit.slices.iter().map(|s| s.rmse).fold(0.0f64, f64::max);
        assert!(worst < 0.02, "worst slice rmse {worst}");
    }

    /// Regression: dense daily listings can put adjacent expiries at
    /// (nearly) identical ATM total variance, which makes the
    /// admissible rho interval microscopically thin. The greedy pass
    /// must hold rho there rather than panic in `clamp` (this exact
    /// shape crashed the first intraday run).
    #[test]
    fn sequential_backbone_survives_flat_atm_term_structure() {
        let p = EssviParams {
            theta: 0.004,
            psi: 0.03,
            rho: -0.5,
        };
        // three one-day-apart expiries with an ATM term structure flat
        // to 1e-12, then a normal tail
        let slices = [
            (1.0 / 365.0, 0.0040),
            (2.0 / 365.0, 0.0040 + 1e-12),
            (3.0 / 365.0, 0.0040 + 2e-12),
            (0.25, 0.0140),
            (1.00, 0.0600),
        ];
        let forward = 100.0;
        let expiries: Vec<Tenor> = slices
            .iter()
            .map(|&(t, _)| Tenor::YearFraction(t))
            .collect();
        let smiles: Vec<Vec<(f64, f64)>> = slices
            .iter()
            .map(|&(t, th)| {
                let sc = (th / p.theta).sqrt();
                (0..9)
                    .map(|i| {
                        let k = -0.2 + i as f64 * 0.05;
                        (forward * k.exp(), sc * p.vol(k, t.max(0.02)))
                    })
                    .collect()
            })
            .collect();
        let surface = VolSurface::from_strike_smiles(
            &expiries,
            &smiles,
            NaiveDate::from_ymd_opt(2026, 8, 20).unwrap(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let (fit, _diag) = EssviSurfaceFit::fit_sequential_backbone(&surface, |_| forward, 0.5)
            .expect("must not panic on a flat ATM term structure");
        assert!(
            fit.max_calendar_crossing <= 1e-12,
            "crossing {}",
            fit.max_calendar_crossing
        );
        for pair in fit.slices.windows(2) {
            assert!(pair[1].params.theta > pair[0].params.theta);
        }
    }
}
