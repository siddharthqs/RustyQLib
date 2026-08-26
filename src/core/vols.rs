//! Volatility surface infrastructure, mirroring [`crate::core::curves`].
//!
//! Design invariants:
//! - Every input form is canonicalized at construction into per-expiry
//!   smiles on a strike-like coordinate, so pricing has one query path:
//!   [`VolSurface::vol`]`(strike, forward, t)`.
//! - **Time interpolation is in total variance** (`w = sigma^2 * t`) at a
//!   fixed smile coordinate: linear by default (the industry-standard
//!   baseline), or monotone cubic (PCHIP) via
//!   [`VolSurface::with_time_interpolation`] for a C^1 forward variance.
//! - Strike interpolation is linear in vol on the smile coordinate, with
//!   flat wing extrapolation; flat vol extrapolation before the first and
//!   after the last expiry.
//!
//! Quoting conventions per axis:
//! - `strikes` — absolute strikes (equity listed convention). Time
//!   interpolation at fixed strike (sticky strike).
//! - `moneyness` — forward moneyness `K/F` (relative strikes). Sticky
//!   moneyness behavior.
//! - `deltas` — **forward call deltas** in (0, 1) (FX convention). Quote a
//!   25-delta put as `0.75` (`= 1 + forward put delta`); ATM-delta-neutral is
//!   approximately `0.5`. Pillars are converted to log-moneyness at
//!   construction using each pillar's own quoted vol
//!   (`ln(K/F) = 0.5*sigma^2*t - sigma*sqrt(t)*inv_N(delta)`), so queries are
//!   sticky delta.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::utils::{inv_norm_cdf, norm_cdf};

/// The accepted input forms for a volatility surface. Deserializes from
/// JSON; canonicalized at construction ([`VolSurface::from_input`]).
///
/// For the grid forms, `vols[i][j]` is the vol at `expiries[i]` and the
/// j-th strike/moneyness/delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VolInput {
    /// A single constant volatility for all strikes and expiries.
    Flat {
        vol: f64,
        #[serde(default)]
        day_count: DayCountConvention,
    },
    /// Absolute strike x expiry grid (equity convention).
    StrikeExpiry {
        expiries: Vec<Tenor>,
        strikes: Vec<f64>,
        vols: Vec<Vec<f64>>,
        #[serde(default)]
        day_count: DayCountConvention,
    },
    /// Forward moneyness (K/F) x expiry grid.
    MoneynessExpiry {
        expiries: Vec<Tenor>,
        moneyness: Vec<f64>,
        vols: Vec<Vec<f64>>,
        #[serde(default)]
        day_count: DayCountConvention,
    },
    /// Forward call delta x expiry grid (FX convention).
    DeltaExpiry {
        expiries: Vec<Tenor>,
        deltas: Vec<f64>,
        vols: Vec<Vec<f64>>,
        #[serde(default)]
        day_count: DayCountConvention,
    },
    /// Per-expiry smiles, each with its own point list (as quoted option
    /// chains are, no rectangular grid required): `smiles[i]` is a list
    /// of `[coordinate, vol]` pairs for `expiries[i]`, sorted by
    /// coordinate. `coordinate` names what the first element means
    /// (default: absolute strike). This is also the lossless save/load
    /// form of a built surface ([`VolSurface::to_input`]).
    StrikeSmiles {
        expiries: Vec<Tenor>,
        smiles: Vec<Vec<(f64, f64)>>,
        #[serde(default)]
        coordinate: SmileCoordinate,
        #[serde(default)]
        day_count: DayCountConvention,
    },
}

/// The x-coordinate a smile's points are quoted on (the public,
/// serializable face of the surface's internal coordinate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SmileCoordinate {
    /// Absolute strike (equity listed convention).
    #[default]
    Strike,
    /// Forward moneyness `K/F`.
    Moneyness,
    /// Log forward moneyness `ln(K/F)`.
    LogMoneyness,
}

/// Errors from surface construction.
#[derive(Debug, Clone, PartialEq)]
pub enum VolError {
    Empty,
    LengthMismatch { expected: usize, got: usize },
    NonPositiveVol(f64),
    NonPositiveTime(f64),
    NonIncreasingTimes,
    NonIncreasingAxis,
    DeltaOutOfRange(f64),
}

impl fmt::Display for VolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VolError::Empty => write!(f, "vol surface needs at least one pillar"),
            VolError::LengthMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            VolError::NonPositiveVol(v) => write!(f, "volatility must be > 0, got {v}"),
            VolError::NonPositiveTime(t) => write!(f, "expiry time must be > 0, got {t}"),
            VolError::NonIncreasingTimes => write!(f, "expiry times must be strictly increasing"),
            VolError::NonIncreasingAxis => {
                write!(f, "strike/moneyness/delta axis must be strictly increasing")
            }
            VolError::DeltaOutOfRange(d) => {
                write!(f, "forward call delta must be in (0,1), got {d}")
            }
        }
    }
}

impl std::error::Error for VolError {}

/// One expiry's smile: `(coordinate, vol)` points sorted by coordinate.
#[derive(Debug, Clone, Serialize)]
struct Smile {
    points: Vec<(f64, f64)>,
}

impl Smile {
    /// Linear in vol on the coordinate; flat beyond the wings
    /// (via the shared [`interp_pairs`](crate::core::interpolation::interp_pairs)).
    fn vol(&self, x: f64) -> f64 {
        if self.points.len() == 1 {
            return self.points[0].1;
        }
        crate::core::interpolation::interp_pairs(&self.points, x)
    }
}

#[derive(Debug, Clone, Serialize)]
enum SurfaceData {
    Flat(f64),
    Term {
        times: Vec<f64>,
        smiles: Vec<Smile>,
        coord: SmileCoordinate,
    },
}

/// Time-dimension interpolation of total variance between expiry pillars
/// at a fixed smile coordinate. A runtime pricing choice
/// ([`VolSurface::with_time_interpolation`]), not part of the saved
/// document — a rebuilt surface starts at the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeInterpolation {
    /// Piecewise-linear `w(t)`: exact at pillars, but the forward
    /// variance `dw/dt` is piecewise constant — Dupire local vol gets a
    /// staircase in time, one step per pillar.
    #[default]
    Linear,
    /// Monotone cubic Hermite (Fritsch-Carlson / PCHIP) on `w(t)`
    /// through a virtual `w(0) = 0` anchor: C^1, so `dw/dt` is
    /// continuous through the pillars, and the slope limiter cannot
    /// create a calendar violation between monotone pillars (an
    /// unconstrained cubic spline can — that is why it is not offered).
    Pchip,
}

/// A shift applied to a whole surface by [`VolSurface::bumped`]. The
/// surface owns the semantics: shifts move every quoted vol, preserving
/// the smile shape and the surface's coordinate system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolShift {
    /// Add `d` vol points to every quote (e.g. `0.01` = +1 vol point).
    ParallelAbsolute(f64),
    /// Scale every quote by `1 + r` (e.g. `0.10` = vols up 10%).
    ParallelRelative(f64),
}

/// One butterfly violation: negative call-price convexity across three
/// adjacent quoted strikes at one expiry pillar.
#[derive(Debug, Clone, Serialize)]
pub struct ButterflyViolation {
    /// Expiry pillar time (year fraction).
    pub time: f64,
    /// The three adjacent strikes whose butterfly prices negative.
    pub strikes: (f64, f64, f64),
    /// The butterfly's (negative) undiscounted price.
    pub magnitude: f64,
}

/// One calendar violation: total variance `sigma^2 t` decreasing between
/// two adjacent expiry pillars at a fixed forward moneyness.
#[derive(Debug, Clone, Serialize)]
pub struct CalendarViolation {
    pub earlier_time: f64,
    pub later_time: f64,
    /// The forward moneyness `K/F` where the decrease occurs.
    pub moneyness: f64,
    /// How much total variance falls (`w_earlier - w_later`, positive).
    pub magnitude: f64,
}

/// Static-arbitrage findings on a surface — a report, not a gate (see
/// [`VolSurface::diagnostics`]).
#[derive(Debug, Clone, Default, Serialize)]
pub struct SurfaceDiagnostics {
    pub butterfly: Vec<ButterflyViolation>,
    pub calendar: Vec<CalendarViolation>,
}

impl SurfaceDiagnostics {
    pub fn is_clean(&self) -> bool {
        self.butterfly.is_empty() && self.calendar.is_empty()
    }

    /// Compact metadata summary: violation counts plus the single worst
    /// finding of each kind (the full listings stay on the struct).
    pub fn to_metadata(&self) -> serde_json::Value {
        let worst_butterfly = self
            .butterfly
            .iter()
            .min_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap());
        let worst_calendar = self
            .calendar
            .iter()
            .max_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap());
        serde_json::json!({
            "butterfly_violations": self.butterfly.len(),
            "calendar_violations": self.calendar.len(),
            "worst_butterfly": worst_butterfly,
            "worst_calendar": worst_calendar,
        })
    }
}

/// Butterflies more negative than this (undiscounted price units) are
/// violations; anything smaller is numerical noise.
const BUTTERFLY_TOL: f64 = 1e-9;
/// Total-variance decreases beyond this are calendar violations.
const CALENDAR_TOL: f64 = 1e-12;

/// Undiscounted Black call price (the forward-measure price the
/// butterfly check needs; discounting cancels out of the convexity
/// comparison).
fn black_call(f: f64, k: f64, sigma: f64, t: f64) -> f64 {
    if sigma <= 0.0 || t <= 0.0 {
        return (f - k).max(0.0);
    }
    let sq = sigma * t.sqrt();
    let d1 = ((f / k).ln() + 0.5 * sq * sq) / sq;
    f * norm_cdf(d1) - k * norm_cdf(d1 - sq)
}

/// A canonical Black volatility surface anchored at `reference_date`.
#[derive(Debug, Clone, Serialize)]
pub struct VolSurface {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    data: SurfaceData,
    time_interp: TimeInterpolation,
}

/// A saved volatility surface: a [`VolInput`] payload plus the anchor
/// date and free-form provenance metadata (data source, forwards used,
/// build settings, ...). Because the payload is an *input* form, a saved
/// document doubles as a valid `vol` block for pricing contracts — save
/// once, price against it later, no separate loader path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolSurfaceDocument {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    pub reference_date: NaiveDate,
    pub surface: VolInput,
}

impl VolSurfaceDocument {
    /// Rebuild the canonical surface this document describes.
    pub fn build(&self) -> Result<VolSurface, VolError> {
        VolSurface::from_input(&self.surface, self.reference_date)
    }
}

impl VolSurface {
    /// The same market re-anchored at a later reference date: each
    /// pillar smile stays attached to its expiry (the quotes are
    /// unchanged in date space), while the pillar year fractions are
    /// re-measured from the new anchor. Pillars expiring at or before
    /// the new reference drop out. Errors if `new_reference` precedes
    /// the current reference or if no pillar survives.
    pub fn rolled(&self, new_reference: NaiveDate) -> Result<VolSurface, VolError> {
        let tau = self
            .day_count
            .year_fraction(self.reference_date, new_reference);
        if tau < 0.0 {
            return Err(VolError::NonPositiveTime(tau));
        }
        if tau == 0.0 {
            return Ok(self.clone());
        }
        let data = match &self.data {
            SurfaceData::Flat(v) => SurfaceData::Flat(*v),
            SurfaceData::Term {
                times,
                smiles,
                coord,
            } => {
                let mut new_times = Vec::with_capacity(times.len());
                let mut new_smiles = Vec::with_capacity(smiles.len());
                for (t, smile) in times.iter().zip(smiles) {
                    if *t > tau + 1e-12 {
                        new_times.push(t - tau);
                        new_smiles.push(smile.clone());
                    }
                }
                if new_times.is_empty() {
                    return Err(VolError::Empty);
                }
                SurfaceData::Term {
                    times: new_times,
                    smiles: new_smiles,
                    coord: *coord,
                }
            }
        };
        Ok(VolSurface {
            reference_date: new_reference,
            day_count: self.day_count,
            data,
            time_interp: self.time_interp,
        })
    }

    // ── Constructors ────────────────────────────────────────────────────

    /// Constant volatility for all strikes and expiries.
    pub fn flat(
        vol: f64,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        if vol <= 0.0 {
            return Err(VolError::NonPositiveVol(vol));
        }
        Ok(VolSurface {
            reference_date,
            day_count,
            data: SurfaceData::Flat(vol),
            time_interp: TimeInterpolation::default(),
        })
    }

    /// Absolute strike x expiry grid.
    pub fn from_strike_grid(
        expiries: &[Tenor],
        strikes: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        Self::from_grid(
            expiries,
            strikes,
            vols,
            reference_date,
            day_count,
            SmileCoordinate::Strike,
        )
    }

    /// Forward moneyness (K/F) x expiry grid.
    pub fn from_moneyness_grid(
        expiries: &[Tenor],
        moneyness: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        Self::from_grid(
            expiries,
            moneyness,
            vols,
            reference_date,
            day_count,
            SmileCoordinate::Moneyness,
        )
    }

    /// Forward call delta x expiry grid (FX convention). Each pillar is
    /// converted to log-moneyness with its own quoted vol:
    /// `ln(K/F) = 0.5*sigma^2*t - sigma*sqrt(t)*inv_N(delta)`.
    pub fn from_delta_grid(
        expiries: &[Tenor],
        deltas: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        for &d in deltas {
            if !(d > 0.0 && d < 1.0) {
                return Err(VolError::DeltaOutOfRange(d));
            }
        }
        let times = Self::resolve_expiries(expiries, reference_date, day_count)?;
        Self::validate_grid(&times, deltas, vols)?;
        let smiles = times
            .iter()
            .zip(vols)
            .map(|(&t, row)| {
                let mut points: Vec<(f64, f64)> = deltas
                    .iter()
                    .zip(row)
                    .map(|(&delta, &sigma)| {
                        let k = 0.5 * sigma * sigma * t - sigma * t.sqrt() * inv_norm_cdf(delta);
                        (k, sigma)
                    })
                    .collect();
                points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                Smile { points }
            })
            .collect();
        Ok(VolSurface {
            reference_date,
            day_count,
            data: SurfaceData::Term {
                times,
                smiles,
                coord: SmileCoordinate::LogMoneyness,
            },
            time_interp: TimeInterpolation::default(),
        })
    }

    /// Per-expiry smiles on absolute strikes, where each expiry may have its
    /// own strike list (as quoted option chains do): `smiles[i]` is a list of
    /// `(strike, vol)` points for `expiries[i]`, sorted by strike.
    pub fn from_strike_smiles(
        expiries: &[Tenor],
        smiles: &[Vec<(f64, f64)>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        Self::from_smiles(
            expiries,
            smiles,
            SmileCoordinate::Strike,
            reference_date,
            day_count,
        )
    }

    /// [`Self::from_strike_smiles`] generalized to any smile coordinate:
    /// each point's first element is read as `coordinate` says (absolute
    /// strike, forward moneyness `K/F`, or `ln(K/F)`).
    pub fn from_smiles(
        expiries: &[Tenor],
        smiles: &[Vec<(f64, f64)>],
        coordinate: SmileCoordinate,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        let times = Self::resolve_expiries(expiries, reference_date, day_count)?;
        if smiles.len() != times.len() {
            return Err(VolError::LengthMismatch {
                expected: times.len(),
                got: smiles.len(),
            });
        }
        for smile in smiles {
            if smile.is_empty() {
                return Err(VolError::Empty);
            }
            for &(_, v) in smile {
                if v <= 0.0 {
                    return Err(VolError::NonPositiveVol(v));
                }
            }
            if smile.windows(2).any(|w| w[1].0 <= w[0].0) {
                return Err(VolError::NonIncreasingAxis);
            }
        }
        let smiles = smiles
            .iter()
            .map(|points| Smile {
                points: points.clone(),
            })
            .collect();
        Ok(VolSurface {
            reference_date,
            day_count,
            data: SurfaceData::Term {
                times,
                smiles,
                coord: coordinate,
            },
            time_interp: TimeInterpolation::default(),
        })
    }

    /// Build from a deserialized [`VolInput`], anchored at `reference_date`.
    pub fn from_input(input: &VolInput, reference_date: NaiveDate) -> Result<Self, VolError> {
        match input {
            VolInput::Flat { vol, day_count } => Self::flat(*vol, reference_date, *day_count),
            VolInput::StrikeExpiry {
                expiries,
                strikes,
                vols,
                day_count,
            } => Self::from_strike_grid(expiries, strikes, vols, reference_date, *day_count),
            VolInput::MoneynessExpiry {
                expiries,
                moneyness,
                vols,
                day_count,
            } => Self::from_moneyness_grid(expiries, moneyness, vols, reference_date, *day_count),
            VolInput::DeltaExpiry {
                expiries,
                deltas,
                vols,
                day_count,
            } => Self::from_delta_grid(expiries, deltas, vols, reference_date, *day_count),
            VolInput::StrikeSmiles {
                expiries,
                smiles,
                coordinate,
                day_count,
            } => Self::from_smiles(expiries, smiles, *coordinate, reference_date, *day_count),
        }
    }

    /// This surface's data as the input form that reconstructs it exactly
    /// (per-expiry smiles on the surface's own coordinate). Expiries come
    /// back as year fractions — pillar times are what the surface stores;
    /// calendar dates used at construction are not retained. A
    /// delta-quoted surface returns its converted log-moneyness smiles.
    pub fn to_input(&self) -> VolInput {
        match &self.data {
            SurfaceData::Flat(vol) => VolInput::Flat {
                vol: *vol,
                day_count: self.day_count,
            },
            SurfaceData::Term {
                times,
                smiles,
                coord,
            } => VolInput::StrikeSmiles {
                expiries: times.iter().map(|&t| Tenor::YearFraction(t)).collect(),
                smiles: smiles.iter().map(|s| s.points.clone()).collect(),
                coordinate: *coord,
                day_count: self.day_count,
            },
        }
    }

    /// Static-arbitrage diagnostics: **butterfly** violations (negative
    /// call-price convexity across adjacent quoted strikes within one
    /// expiry, priced undiscounted off the pillar vols) and **calendar**
    /// violations (total variance `sigma^2 t` decreasing between
    /// adjacent expiries at fixed forward moneyness). `forward` maps an
    /// expiry time to the underlying's forward price.
    ///
    /// A report, not a gate: quoted market snapshots are noisy and the
    /// surface stays usable regardless — but violations mean the smile
    /// carries static arbitrage, and the Dupire transformation falls
    /// back to implied vol wherever they bite. Checks run at the quoted
    /// pillars, so each finding names the strikes responsible;
    /// interpolation between pillars is not separately scanned. A flat
    /// surface is clean by construction.
    pub fn diagnostics(&self, forward: impl Fn(f64) -> f64) -> SurfaceDiagnostics {
        let mut report = SurfaceDiagnostics::default();
        let SurfaceData::Term {
            times,
            smiles,
            coord,
        } = &self.data
        else {
            return report;
        };
        let to_strike = |x: f64, f: f64| match coord {
            SmileCoordinate::Strike => x,
            SmileCoordinate::Moneyness => x * f,
            SmileCoordinate::LogMoneyness => x.exp() * f,
        };

        // butterfly: convexity of undiscounted calls at the pillar strikes
        for (&t, smile) in times.iter().zip(smiles) {
            let f = forward(t);
            let prices: Vec<(f64, f64)> = smile
                .points
                .iter()
                .map(|&(x, vol)| {
                    let k = to_strike(x, f);
                    (k, black_call(f, k, vol, t))
                })
                .collect();
            for window in prices.windows(3) {
                let [(k1, c1), (k2, c2), (k3, c3)] = [window[0], window[1], window[2]];
                let weight = (k3 - k2) / (k3 - k1);
                let butterfly = weight * c1 + (1.0 - weight) * c3 - c2;
                if butterfly < -BUTTERFLY_TOL {
                    report.butterfly.push(ButterflyViolation {
                        time: t,
                        strikes: (k1, k2, k3),
                        magnitude: butterfly,
                    });
                }
            }
        }

        // calendar: total variance across adjacent expiries at the fixed
        // forward moneyness of both pillars' quoted strikes
        for i in 1..times.len() {
            let (t1, t2) = (times[i - 1], times[i]);
            let (f1, f2) = (forward(t1), forward(t2));
            let mut moneyness: Vec<f64> = smiles[i - 1]
                .points
                .iter()
                .map(|&(x, _)| to_strike(x, f1) / f1)
                .chain(smiles[i].points.iter().map(|&(x, _)| to_strike(x, f2) / f2))
                .collect();
            moneyness.sort_by(|a, b| a.partial_cmp(b).unwrap());
            moneyness.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            for m in moneyness {
                let v1 = self.vol(m * f1, f1, t1);
                let v2 = self.vol(m * f2, f2, t2);
                let (w1, w2) = (v1 * v1 * t1, v2 * v2 * t2);
                if w2 < w1 - CALENDAR_TOL {
                    report.calendar.push(CalendarViolation {
                        earlier_time: t1,
                        later_time: t2,
                        moneyness: m,
                        magnitude: w1 - w2,
                    });
                }
            }
        }
        report
    }

    /// This surface as a self-contained document: the input form plus the
    /// anchor date and optional provenance metadata.
    pub fn to_document(&self, metadata: Option<serde_json::Value>) -> VolSurfaceDocument {
        VolSurfaceDocument {
            metadata,
            reference_date: self.reference_date,
            surface: self.to_input(),
        }
    }

    /// Serialize as a pretty-printed JSON surface document (without
    /// metadata; use [`Self::to_document`] to attach provenance first).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.to_document(None))
            .expect("a validated surface always serializes (finite times and vols)")
    }

    /// Load a surface from a JSON document written by [`Self::to_json`]
    /// (or any hand-written [`VolSurfaceDocument`]). Metadata is carried
    /// by the document, not the surface — parse a [`VolSurfaceDocument`]
    /// directly if you need it.
    pub fn from_json(text: &str) -> Result<VolSurface, RustyQLibError> {
        let document: VolSurfaceDocument = serde_json::from_str(text).map_err(|e| {
            RustyQLibError::ParseError(format!("invalid vol surface document: {e}"))
        })?;
        document.build().map_err(RustyQLibError::from)
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Black volatility for an option with the given absolute `strike`,
    /// `forward` price of the underlying at expiry, and year fraction `t`.
    ///
    /// Strike dimension: linear in vol, flat wings. Time dimension: total
    /// variance at the fixed smile coordinate per
    /// [`Self::time_interpolation`], flat vol before the first and after
    /// the last expiry pillar.
    pub fn vol(&self, strike: f64, forward: f64, t: f64) -> f64 {
        match &self.data {
            SurfaceData::Flat(v) => *v,
            SurfaceData::Term {
                times,
                smiles,
                coord,
            } => {
                let x = match coord {
                    SmileCoordinate::Strike => strike,
                    SmileCoordinate::Moneyness => strike / forward,
                    SmileCoordinate::LogMoneyness => (strike / forward).ln(),
                };
                let n = times.len();
                if t <= times[0] {
                    return smiles[0].vol(x);
                }
                if t >= times[n - 1] {
                    return smiles[n - 1].vol(x);
                }
                let idx = times.partition_point(|&ti| ti < t);
                let (t0, t1) = (times[idx - 1], times[idx]);
                let (v0, v1) = (smiles[idx - 1].vol(x), smiles[idx].vol(x));
                // total variance in time at fixed coordinate
                let (w0, w1) = (v0 * v0 * t0, v1 * v1 * t1);
                let h = t1 - t0;
                let w = match self.time_interp {
                    TimeInterpolation::Linear => w0 + (w1 - w0) * (t - t0) / h,
                    TimeInterpolation::Pchip => {
                        use crate::core::interpolation::pchip;
                        // secants of the segments around [t0, t1]. Flat vol
                        // below the first pillar is exactly the chord from a
                        // virtual (0, 0) anchor, which supplies the left
                        // secant for the first interval.
                        let s_mid = (w1 - w0) / h;
                        let (h_prev, s_prev) = if idx >= 2 {
                            let tp = times[idx - 2];
                            let vp = smiles[idx - 2].vol(x);
                            (t0 - tp, (w0 - vp * vp * tp) / (t0 - tp))
                        } else {
                            (t0, w0 / t0)
                        };
                        let d0 = pchip::interior_slope(h_prev, s_prev, h, s_mid);
                        let d1 = if idx + 1 < n {
                            let tn = times[idx + 1];
                            let vn = smiles[idx + 1].vol(x);
                            let s_next = (vn * vn * tn - w1) / (tn - t1);
                            pchip::interior_slope(h, s_mid, tn - t1, s_next)
                        } else {
                            pchip::end_slope(h, h_prev, s_mid, s_prev)
                        };
                        pchip::hermite(t, t0, t1, w0, w1, d0, d1)
                    }
                };
                (w / t).sqrt()
            }
        }
    }

    /// This surface with the given time-dimension interpolation (see
    /// [`TimeInterpolation`]). A runtime pricing choice: it is not part
    /// of the saved document, so a surface rebuilt from one starts at
    /// the default (linear).
    pub fn with_time_interpolation(mut self, interp: TimeInterpolation) -> Self {
        self.time_interp = interp;
        self
    }

    /// The time-dimension interpolation queries use.
    pub fn time_interpolation(&self) -> TimeInterpolation {
        self.time_interp
    }

    /// This surface with `shift` applied to every quoted vol — the smile
    /// shape, coordinate system and expiry pillars are preserved. Errors
    /// when any bumped vol would be non-positive (a shock that large is a
    /// data problem, not a market).
    pub fn bumped(&self, shift: VolShift) -> Result<VolSurface, VolError> {
        let apply = |v: f64| match shift {
            VolShift::ParallelAbsolute(d) => v + d,
            VolShift::ParallelRelative(r) => v * (1.0 + r),
        };
        let mut bumped = self.clone();
        match &mut bumped.data {
            SurfaceData::Flat(v) => {
                *v = apply(*v);
                if *v <= 0.0 {
                    return Err(VolError::NonPositiveVol(*v));
                }
            }
            SurfaceData::Term { smiles, .. } => {
                for smile in smiles {
                    for point in &mut smile.points {
                        point.1 = apply(point.1);
                        if point.1 <= 0.0 {
                            return Err(VolError::NonPositiveVol(point.1));
                        }
                    }
                }
            }
        }
        Ok(bumped)
    }

    pub fn reference_date(&self) -> NaiveDate {
        self.reference_date
    }
    pub fn day_count(&self) -> DayCountConvention {
        self.day_count
    }
    /// Expiry pillar times (empty for a flat surface).
    pub fn expiry_times(&self) -> &[f64] {
        match &self.data {
            SurfaceData::Flat(_) => &[],
            SurfaceData::Term { times, .. } => times,
        }
    }

    // ── Internals ───────────────────────────────────────────────────────

    fn from_grid(
        expiries: &[Tenor],
        axis: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        coord: SmileCoordinate,
    ) -> Result<Self, VolError> {
        let times = Self::resolve_expiries(expiries, reference_date, day_count)?;
        Self::validate_grid(&times, axis, vols)?;
        if axis.windows(2).any(|w| w[1] <= w[0]) {
            return Err(VolError::NonIncreasingAxis);
        }
        let smiles = vols
            .iter()
            .map(|row| Smile {
                points: axis.iter().copied().zip(row.iter().copied()).collect(),
            })
            .collect();
        Ok(VolSurface {
            reference_date,
            day_count,
            data: SurfaceData::Term {
                times,
                smiles,
                coord,
            },
            time_interp: TimeInterpolation::default(),
        })
    }

    fn resolve_expiries(
        expiries: &[Tenor],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Vec<f64>, VolError> {
        if expiries.is_empty() {
            return Err(VolError::Empty);
        }
        let times: Vec<f64> = expiries
            .iter()
            .map(|tenor| match tenor {
                Tenor::Date(d) => day_count.year_fraction(reference_date, *d),
                Tenor::YearFraction(t) => *t,
            })
            .collect();
        for &t in &times {
            if t <= 0.0 {
                return Err(VolError::NonPositiveTime(t));
            }
        }
        if times.windows(2).any(|w| w[1] <= w[0]) {
            return Err(VolError::NonIncreasingTimes);
        }
        Ok(times)
    }

    fn validate_grid(times: &[f64], axis: &[f64], vols: &[Vec<f64>]) -> Result<(), VolError> {
        if axis.is_empty() {
            return Err(VolError::Empty);
        }
        if vols.len() != times.len() {
            return Err(VolError::LengthMismatch {
                expected: times.len(),
                got: vols.len(),
            });
        }
        for row in vols {
            if row.len() != axis.len() {
                return Err(VolError::LengthMismatch {
                    expected: axis.len(),
                    got: row.len(),
                });
            }
            for &v in row {
                if v <= 0.0 {
                    return Err(VolError::NonPositiveVol(v));
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for VolSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "VolSurface (ref {}, {:?})",
            self.reference_date, self.day_count
        )?;
        match &self.data {
            SurfaceData::Flat(v) => writeln!(f, "  flat vol: {v}"),
            SurfaceData::Term {
                times,
                smiles,
                coord,
            } => {
                writeln!(f, "  smile coordinate: {coord:?}")?;
                for (t, smile) in times.iter().zip(smiles) {
                    write!(f, "  t={t:<8.4}")?;
                    for (x, v) in &smile.points {
                        write!(f, " ({x:.4}, {v:.4})")?;
                    }
                    writeln!(f)?;
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::norm_cdf;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 16).unwrap()
    }

    #[test]
    fn flat_surface_is_constant() {
        let surface = VolSurface::flat(0.3, asof(), DayCountConvention::Act365).unwrap();
        assert_eq!(surface.vol(50.0, 100.0, 0.1), 0.3);
        assert_eq!(surface.vol(200.0, 100.0, 5.0), 0.3);
    }

    fn strike_grid() -> VolSurface {
        // expiries 1y and 2y; strikes 90 / 100 / 110
        VolSurface::from_strike_grid(
            &[Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
            &[90.0, 100.0, 110.0],
            &[vec![0.22, 0.20, 0.19], vec![0.27, 0.25, 0.24]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap()
    }

    #[test]
    fn strike_grid_exact_at_pillars() {
        let s = strike_grid();
        assert!((s.vol(100.0, 100.0, 1.0) - 0.20).abs() < 1e-14);
        assert!((s.vol(90.0, 100.0, 2.0) - 0.27).abs() < 1e-14);
    }

    #[test]
    fn strike_interpolation_linear_with_flat_wings() {
        let s = strike_grid();
        // midway between 90 and 100 at t=1: (0.22+0.20)/2
        assert!((s.vol(95.0, 100.0, 1.0) - 0.21).abs() < 1e-14);
        // wings are flat
        assert!((s.vol(50.0, 100.0, 1.0) - 0.22).abs() < 1e-14);
        assert!((s.vol(500.0, 100.0, 1.0) - 0.19).abs() < 1e-14);
    }

    #[test]
    fn bumped_shifts_every_quote_and_preserves_the_smile() {
        let flat = VolSurface::flat(0.30, asof(), DayCountConvention::Act365).unwrap();
        let up = flat.bumped(VolShift::ParallelAbsolute(0.05)).unwrap();
        assert!((up.vol(100.0, 100.0, 1.0) - 0.35).abs() < 1e-14);
        let scaled = flat.bumped(VolShift::ParallelRelative(0.10)).unwrap();
        assert!((scaled.vol(100.0, 100.0, 1.0) - 0.33).abs() < 1e-14);

        let s = strike_grid();
        let up = s.bumped(VolShift::ParallelAbsolute(0.01)).unwrap();
        // every pillar moves by the shift; the skew (differences) is intact
        assert!((up.vol(100.0, 100.0, 1.0) - 0.21).abs() < 1e-14);
        assert!((up.vol(90.0, 100.0, 2.0) - 0.28).abs() < 1e-14);
        let skew_base = s.vol(90.0, 100.0, 1.0) - s.vol(110.0, 100.0, 1.0);
        let skew_up = up.vol(90.0, 100.0, 1.0) - up.vol(110.0, 100.0, 1.0);
        assert!(
            (skew_base - skew_up).abs() < 1e-14,
            "parallel shift must keep the skew"
        );
        // the original is untouched
        assert!((s.vol(100.0, 100.0, 1.0) - 0.20).abs() < 1e-14);
    }

    #[test]
    fn bumped_rejects_non_positive_vols() {
        let flat = VolSurface::flat(0.20, asof(), DayCountConvention::Act365).unwrap();
        assert!(matches!(
            flat.bumped(VolShift::ParallelAbsolute(-0.20)),
            Err(VolError::NonPositiveVol(_))
        ));
        assert!(matches!(
            strike_grid().bumped(VolShift::ParallelRelative(-1.0)),
            Err(VolError::NonPositiveVol(_))
        ));
    }

    #[test]
    fn time_interpolation_is_linear_total_variance() {
        let s = strike_grid();
        // at K=100: w1 = 0.2^2*1 = 0.04, w2 = 0.25^2*2 = 0.125
        // at t=1.5: w = 0.0825 -> vol = sqrt(0.0825/1.5)
        let expected = (0.0825_f64 / 1.5).sqrt();
        assert!((s.vol(100.0, 100.0, 1.5) - expected).abs() < 1e-12);
    }

    #[test]
    fn time_extrapolation_is_flat_vol() {
        let s = strike_grid();
        assert!((s.vol(100.0, 100.0, 0.25) - 0.20).abs() < 1e-14); // before first
        assert!((s.vol(100.0, 100.0, 5.0) - 0.25).abs() < 1e-14); // after last
    }

    /// Flat vols make `w(t)` collinear with the virtual `(0, 0)` anchor,
    /// and PCHIP reproduces collinear data exactly — so the two modes
    /// must agree to machine precision.
    #[test]
    fn pchip_equals_linear_on_flat_vols() {
        let flat = VolSurface::from_strike_grid(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
            ],
            &[90.0, 100.0, 110.0],
            &[vec![0.2; 3], vec![0.2; 3], vec![0.2; 3]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap()
        .with_time_interpolation(TimeInterpolation::Pchip);
        for t in [0.6, 0.75, 1.0, 1.3, 1.9] {
            assert!((flat.vol(100.0, 100.0, t) - 0.2).abs() < 1e-14, "t={t}");
        }
    }

    #[test]
    fn pchip_is_exact_at_pillars() {
        let s = strike_grid().with_time_interpolation(TimeInterpolation::Pchip);
        assert!((s.vol(100.0, 100.0, 1.0) - 0.20).abs() < 1e-14);
        assert!((s.vol(100.0, 100.0, 2.0) - 0.25).abs() < 1e-14);
    }

    /// The reason PCHIP exists here: `dw/dt` must be continuous through
    /// an interior pillar, where linear total variance jumps.
    #[test]
    fn pchip_forward_variance_is_continuous_at_pillars() {
        let surface = |interp: TimeInterpolation| {
            VolSurface::from_strike_grid(
                &[
                    Tenor::YearFraction(0.5),
                    Tenor::YearFraction(1.0),
                    Tenor::YearFraction(1.5),
                ],
                &[100.0],
                &[vec![0.20], vec![0.25], vec![0.26]],
                asof(),
                DayCountConvention::Act365,
            )
            .unwrap()
            .with_time_interpolation(interp)
        };
        let w = |s: &VolSurface, t: f64| s.vol(100.0, 100.0, t).powi(2) * t;
        let jump = |s: &VolSurface| {
            let eps = 1e-5;
            let left = (w(s, 1.0) - w(s, 1.0 - eps)) / eps;
            let right = (w(s, 1.0 + eps) - w(s, 1.0)) / eps;
            (right - left).abs()
        };
        let linear = surface(TimeInterpolation::Linear);
        let pchip = surface(TimeInterpolation::Pchip);
        // linear: forward variance steps from 0.085 to ~0.0778 at t=1
        assert!(jump(&linear) > 5e-3, "linear jump {}", jump(&linear));
        assert!(jump(&pchip) < 1e-3, "pchip jump {}", jump(&pchip));
    }

    /// Steep-then-flat total variance: an unconstrained cubic spline
    /// would overshoot and manufacture a calendar violation; PCHIP's
    /// limiter must keep `w(t)` non-decreasing between monotone pillars.
    #[test]
    fn pchip_preserves_monotone_total_variance() {
        let s = VolSurface::from_strike_grid(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(1.1),
            ],
            &[100.0],
            // w = 0.02, 0.0625, 0.0630 — slope 0.085 then 0.005
            &[vec![0.2], vec![0.25], vec![(0.0630_f64 / 1.1).sqrt()]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap()
        .with_time_interpolation(TimeInterpolation::Pchip);
        let mut prev = 0.0;
        for i in 0..=200 {
            let t = 0.5 + 0.6 * i as f64 / 200.0;
            let w = s.vol(100.0, 100.0, t).powi(2) * t;
            assert!(w >= prev - 1e-12, "w decreasing at t={t}: {w} < {prev}");
            prev = w;
        }
    }

    #[test]
    fn moneyness_grid_uses_forward() {
        let s = VolSurface::from_moneyness_grid(
            &[Tenor::YearFraction(1.0)],
            &[0.9, 1.0, 1.1],
            &[vec![0.22, 0.20, 0.19]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        // strike 105 vs forward 105 -> K/F = 1.0 -> ATM vol
        assert!((s.vol(105.0, 105.0, 1.0) - 0.20).abs() < 1e-14);
        // strike 94.5 vs forward 105 -> K/F = 0.9
        assert!((s.vol(94.5, 105.0, 1.0) - 0.22).abs() < 1e-12);
    }

    #[test]
    fn delta_grid_round_trips_pillar_quotes() {
        // FX-style smile at t=1: 25d call 19%, ATM 20%, 25d put (0.75) 23%
        let t = 1.0_f64;
        let deltas = [0.25, 0.5, 0.75];
        let vols = [0.19, 0.20, 0.23];
        let s = VolSurface::from_delta_grid(
            &[Tenor::YearFraction(t)],
            &deltas,
            &[vols.to_vec()],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let forward = 100.0;
        for (&delta, &sigma) in deltas.iter().zip(&vols) {
            // strike implied by the pillar's own quote
            let k = 0.5 * sigma * sigma * t - sigma * t.sqrt() * inv_norm_cdf(delta);
            let strike = forward * k.exp();
            assert!(
                (s.vol(strike, forward, t) - sigma).abs() < 1e-10,
                "delta {delta}: {} vs {sigma}",
                s.vol(strike, forward, t)
            );
            // and the strike really has that forward delta under its vol
            let d1 = ((forward / strike).ln() + 0.5 * sigma * sigma * t) / (sigma * t.sqrt());
            assert!((norm_cdf(d1) - delta).abs() < 1e-10);
        }
        // put wing (low strike = high call delta) has the higher vol
        assert!(s.vol(80.0, forward, t) > s.vol(120.0, forward, t));
    }

    #[test]
    fn inv_norm_cdf_round_trip() {
        for i in -60..=60 {
            let x = i as f64 / 10.0;
            let p = norm_cdf(x);
            if p > 0.0 && p < 1.0 {
                // in the far tails a 1-ulp error in p maps to ~1e-8 in x
                // (dp/dx = phi(x) is tiny there) — that is the attainable
                // double-precision accuracy, not an approximation error
                let tol = if x.abs() <= 4.5 { 1e-9 } else { 5e-8 };
                assert!(
                    (inv_norm_cdf(p) - x).abs() < tol,
                    "x={x}: inv_norm_cdf(norm_cdf(x))={}",
                    inv_norm_cdf(p)
                );
            }
        }
        assert!(inv_norm_cdf(0.0).is_nan());
        assert!(inv_norm_cdf(1.0).is_nan());
    }

    #[test]
    fn vol_input_deserializes_from_json() {
        let flat: VolInput = serde_json::from_str(r#"{"type": "flat", "vol": 0.3}"#).unwrap();
        let s = VolSurface::from_input(&flat, asof()).unwrap();
        assert_eq!(s.vol(100.0, 100.0, 1.0), 0.3);

        let grid: VolInput = serde_json::from_str(
            r#"{
                "type": "strike_expiry",
                "expiries": [0.5, "2028-07-16"],
                "strikes": [90.0, 100.0, 110.0],
                "vols": [[0.22, 0.20, 0.19], [0.26, 0.24, 0.23]],
                "day_count": "Act365"
            }"#,
        )
        .unwrap();
        let s = VolSurface::from_input(&grid, asof()).unwrap();
        assert!((s.vol(100.0, 100.0, 0.5) - 0.20).abs() < 1e-14);

        let delta: VolInput = serde_json::from_str(
            r#"{
                "type": "delta_expiry",
                "expiries": [1.0],
                "deltas": [0.25, 0.5, 0.75],
                "vols": [[0.19, 0.20, 0.23]]
            }"#,
        )
        .unwrap();
        assert!(VolSurface::from_input(&delta, asof()).is_ok());
    }

    #[test]
    fn validation_errors() {
        let dc = DayCountConvention::Act365;
        assert_eq!(
            VolSurface::flat(0.0, asof(), dc).unwrap_err(),
            VolError::NonPositiveVol(0.0)
        );
        assert_eq!(
            VolSurface::from_strike_grid(&[], &[100.0], &[], asof(), dc).unwrap_err(),
            VolError::Empty
        );
        assert!(matches!(
            VolSurface::from_strike_grid(
                &[Tenor::YearFraction(1.0)],
                &[90.0, 100.0],
                &[vec![0.2]],
                asof(),
                dc
            )
            .unwrap_err(),
            VolError::LengthMismatch { .. }
        ));
        assert_eq!(
            VolSurface::from_strike_grid(
                &[Tenor::YearFraction(1.0)],
                &[100.0, 90.0],
                &[vec![0.2, 0.2]],
                asof(),
                dc
            )
            .unwrap_err(),
            VolError::NonIncreasingAxis
        );
        assert_eq!(
            VolSurface::from_delta_grid(
                &[Tenor::YearFraction(1.0)],
                &[1.5],
                &[vec![0.2]],
                asof(),
                dc
            )
            .unwrap_err(),
            VolError::DeltaOutOfRange(1.5)
        );
    }

    #[test]
    fn strike_smiles_input_builds_ragged_surfaces() {
        // the design-doc document shape: per-expiry point lists, no grid
        let input: VolInput = serde_json::from_str(
            r#"{
                "type": "strike_smiles",
                "expiries": [0.5, 1.0],
                "smiles": [[[95.0, 0.31], [100.0, 0.28]],
                           [[90.0, 0.30], [100.0, 0.27], [110.0, 0.25]]],
                "day_count": "Act365"
            }"#,
        )
        .unwrap();
        let s = VolSurface::from_input(&input, asof()).unwrap();
        assert!((s.vol(95.0, 100.0, 0.5) - 0.31).abs() < 1e-14);
        assert!((s.vol(110.0, 100.0, 1.0) - 0.25).abs() < 1e-14);
        // coordinate defaults to strike; an explicit moneyness coordinate
        // reads the same numbers as K/F
        let m: VolInput = serde_json::from_str(
            r#"{
                "type": "strike_smiles",
                "coordinate": "moneyness",
                "expiries": [1.0],
                "smiles": [[[0.9, 0.22], [1.0, 0.20], [1.1, 0.19]]]
            }"#,
        )
        .unwrap();
        let s = VolSurface::from_input(&m, asof()).unwrap();
        assert!((s.vol(94.5, 105.0, 1.0) - 0.22).abs() < 1e-12);
    }

    #[test]
    fn surface_documents_round_trip_through_json() {
        let probes: [(f64, f64, f64); 4] = [
            (90.0, 100.0, 0.5),
            (100.0, 100.0, 1.0),
            (104.0, 98.0, 1.5),
            (130.0, 105.0, 3.0),
        ];
        let ragged = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::Date(d(2028, 7, 16))],
            &[
                vec![(95.0, 0.31), (100.0, 0.28)],
                vec![(90.0, 0.30), (100.0, 0.27), (110.0, 0.25)],
            ],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let moneyness = VolSurface::from_moneyness_grid(
            &[Tenor::YearFraction(1.0)],
            &[0.9, 1.0, 1.1],
            &[vec![0.22, 0.20, 0.19]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let delta = VolSurface::from_delta_grid(
            &[Tenor::YearFraction(1.0)],
            &[0.25, 0.5, 0.75],
            &[vec![0.19, 0.20, 0.23]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let flat = VolSurface::flat(0.3, asof(), DayCountConvention::Act365).unwrap();

        for (name, surface) in [
            ("ragged", &ragged),
            ("moneyness", &moneyness),
            ("delta", &delta),
            ("flat", &flat),
        ] {
            let loaded = VolSurface::from_json(&surface.to_json()).unwrap();
            for &(k, f, t) in &probes {
                assert!(
                    (loaded.vol(k, f, t) - surface.vol(k, f, t)).abs() < 1e-14,
                    "{name} surface changed after a JSON round trip at ({k}, {f}, {t})"
                );
            }
        }

        // a document with metadata survives parsing, and build() ignores it
        let mut document = ragged.to_document(Some(serde_json::json!({"symbol": "ACME"})));
        let text = serde_json::to_string(&document).unwrap();
        document = serde_json::from_str(&text).unwrap();
        assert_eq!(document.metadata.as_ref().unwrap()["symbol"], "ACME");
        assert!(document.build().is_ok());

        assert!(VolSurface::from_json("{ not a document").is_err());
    }

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn diagnostics_pass_clean_surfaces() {
        // gentle skew, total variance increasing in time: no arbitrage
        let clean = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[
                vec![(90.0, 0.22), (100.0, 0.20), (110.0, 0.19)],
                vec![(90.0, 0.24), (100.0, 0.22), (110.0, 0.21)],
            ],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let report = clean.diagnostics(|_| 100.0);
        assert!(report.is_clean(), "{report:?}");
        // flat surfaces are clean by construction
        let flat = VolSurface::flat(0.2, asof(), DayCountConvention::Act365).unwrap();
        assert!(flat.diagnostics(|_| 100.0).is_clean());
    }

    #[test]
    fn diagnostics_flag_butterfly_arbitrage() {
        // a vol spike at the middle strike prices the middle call above
        // the convex hull of its neighbors
        let spiked = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(1.0)],
            &[vec![(90.0, 0.20), (100.0, 0.50), (110.0, 0.20)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let report = spiked.diagnostics(|_| 100.0);
        assert_eq!(report.butterfly.len(), 1);
        assert_eq!(report.butterfly[0].strikes, (90.0, 100.0, 110.0));
        assert!(report.butterfly[0].magnitude < -1.0, "clearly negative");
        assert!(report.calendar.is_empty());
    }

    #[test]
    fn diagnostics_flag_calendar_arbitrage() {
        // total variance falls: 0.4^2 * 0.5 = 0.08 -> 0.2^2 * 1.0 = 0.04
        let falling = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.40)], vec![(100.0, 0.20)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let report = falling.diagnostics(|_| 100.0);
        assert!(report.butterfly.is_empty(), "one strike per expiry");
        assert_eq!(report.calendar.len(), 1);
        let violation = &report.calendar[0];
        assert_eq!(violation.moneyness, 1.0);
        assert!((violation.magnitude - 0.04).abs() < 1e-12);

        let meta = report.to_metadata();
        assert_eq!(meta["butterfly_violations"], 0);
        assert_eq!(meta["calendar_violations"], 1);
        assert!(meta["worst_butterfly"].is_null());
        assert!(meta["worst_calendar"]["magnitude"].as_f64().unwrap() > 0.03);
    }
}
