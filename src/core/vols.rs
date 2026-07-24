//! Volatility surface infrastructure, mirroring [`crate::core::curves`].
//!
//! Design invariants:
//! - Every input form is canonicalized at construction into per-expiry
//!   smiles on a strike-like coordinate, so pricing has one query path:
//!   [`VolSurface::vol`]`(strike, forward, t)`.
//! - **Time interpolation is linear in total variance** (`w = sigma^2 * t`)
//!   at a fixed smile coordinate — the industry-standard baseline.
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
use crate::core::utils::inv_norm_cdf;

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

/// How query `(strike, forward)` maps onto the stored smile coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SmileCoord {
    /// coordinate = absolute strike (forward unused)
    Strike,
    /// coordinate = K/F
    Moneyness,
    /// coordinate = ln(K/F)
    LogMoneyness,
}

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
    Term { times: Vec<f64>, smiles: Vec<Smile>, coord: SmileCoord },
}

// manual impl so SmileCoord needn't be public-serializable
impl Serialize for SmileCoord {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(match self {
            SmileCoord::Strike => "strike",
            SmileCoord::Moneyness => "moneyness",
            SmileCoord::LogMoneyness => "log_moneyness",
        })
    }
}

/// A canonical Black volatility surface anchored at `reference_date`.
#[derive(Debug, Clone, Serialize)]
pub struct VolSurface {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    data: SurfaceData,
}

impl VolSurface {
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
        Ok(VolSurface { reference_date, day_count, data: SurfaceData::Flat(vol) })
    }

    /// Absolute strike x expiry grid.
    pub fn from_strike_grid(
        expiries: &[Tenor],
        strikes: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        Self::from_grid(expiries, strikes, vols, reference_date, day_count, SmileCoord::Strike)
    }

    /// Forward moneyness (K/F) x expiry grid.
    pub fn from_moneyness_grid(
        expiries: &[Tenor],
        moneyness: &[f64],
        vols: &[Vec<f64>],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, VolError> {
        Self::from_grid(expiries, moneyness, vols, reference_date, day_count, SmileCoord::Moneyness)
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
            data: SurfaceData::Term { times, smiles, coord: SmileCoord::LogMoneyness },
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
        let times = Self::resolve_expiries(expiries, reference_date, day_count)?;
        if smiles.len() != times.len() {
            return Err(VolError::LengthMismatch { expected: times.len(), got: smiles.len() });
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
        let smiles = smiles.iter().map(|points| Smile { points: points.clone() }).collect();
        Ok(VolSurface {
            reference_date,
            day_count,
            data: SurfaceData::Term { times, smiles, coord: SmileCoord::Strike },
        })
    }

    /// Build from a deserialized [`VolInput`], anchored at `reference_date`.
    pub fn from_input(input: &VolInput, reference_date: NaiveDate) -> Result<Self, VolError> {
        match input {
            VolInput::Flat { vol, day_count } => Self::flat(*vol, reference_date, *day_count),
            VolInput::StrikeExpiry { expiries, strikes, vols, day_count } => {
                Self::from_strike_grid(expiries, strikes, vols, reference_date, *day_count)
            }
            VolInput::MoneynessExpiry { expiries, moneyness, vols, day_count } => {
                Self::from_moneyness_grid(expiries, moneyness, vols, reference_date, *day_count)
            }
            VolInput::DeltaExpiry { expiries, deltas, vols, day_count } => {
                Self::from_delta_grid(expiries, deltas, vols, reference_date, *day_count)
            }
        }
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Black volatility for an option with the given absolute `strike`,
    /// `forward` price of the underlying at expiry, and year fraction `t`.
    ///
    /// Strike dimension: linear in vol, flat wings. Time dimension: linear
    /// in total variance at the fixed smile coordinate, flat vol before the
    /// first and after the last expiry pillar.
    pub fn vol(&self, strike: f64, forward: f64, t: f64) -> f64 {
        match &self.data {
            SurfaceData::Flat(v) => *v,
            SurfaceData::Term { times, smiles, coord } => {
                let x = match coord {
                    SmileCoord::Strike => strike,
                    SmileCoord::Moneyness => strike / forward,
                    SmileCoord::LogMoneyness => (strike / forward).ln(),
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
                // linear total variance in time at fixed coordinate
                let (w0, w1) = (v0 * v0 * t0, v1 * v1 * t1);
                let w = w0 + (w1 - w0) * (t - t0) / (t1 - t0);
                (w / t).sqrt()
            }
        }
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
        coord: SmileCoord,
    ) -> Result<Self, VolError> {
        let times = Self::resolve_expiries(expiries, reference_date, day_count)?;
        Self::validate_grid(&times, axis, vols)?;
        if axis.windows(2).any(|w| w[1] <= w[0]) {
            return Err(VolError::NonIncreasingAxis);
        }
        let smiles = vols
            .iter()
            .map(|row| Smile { points: axis.iter().copied().zip(row.iter().copied()).collect() })
            .collect();
        Ok(VolSurface { reference_date, day_count, data: SurfaceData::Term { times, smiles, coord } })
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
            return Err(VolError::LengthMismatch { expected: times.len(), got: vols.len() });
        }
        for row in vols {
            if row.len() != axis.len() {
                return Err(VolError::LengthMismatch { expected: axis.len(), got: row.len() });
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
        writeln!(f, "VolSurface (ref {}, {:?})", self.reference_date, self.day_count)?;
        match &self.data {
            SurfaceData::Flat(v) => writeln!(f, "  flat vol: {v}"),
            SurfaceData::Term { times, smiles, coord } => {
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
        assert_eq!(VolSurface::flat(0.0, asof(), dc).unwrap_err(), VolError::NonPositiveVol(0.0));
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
}
