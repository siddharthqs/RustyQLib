//! Yield / discount curve infrastructure shared by all asset classes.
//!
//! Design invariant: **discount factors are state, rates are views.**
//! A [`YieldCurve`] stores only pillar times and raw discount factors;
//! zero and forward rates are derived on demand. The [`Compounding`] and
//! [`DayCountConvention`] fields are *conventions* — they control how rates
//! are converted in (at construction) and out (rate queries), never what
//! `df(t)` returns.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::core::daycount::DayCountConvention;

/// Convention used to convert between rates and discount factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Compounding {
    /// df = exp(-z*t)
    #[default]
    Continuous,
    /// df = (1+z)^(-t)
    Annual,
    /// df = 1/(1+z*t)  (money-market style, e.g. deposits)
    Simple,
}

impl Compounding {
    /// Discount factor over year fraction `t` for rate `z` under this convention.
    pub fn df(&self, z: f64, t: f64) -> f64 {
        match self {
            Compounding::Continuous => (-z * t).exp(),
            Compounding::Annual => (1.0 + z).powf(-t),
            Compounding::Simple => 1.0 / (1.0 + z * t),
        }
    }
    /// Rate over year fraction `t` implied by discount factor `df` under this convention.
    pub fn rate(&self, df: f64, t: f64) -> f64 {
        match self {
            Compounding::Continuous => -df.ln() / t,
            Compounding::Annual => df.powf(-1.0 / t) - 1.0,
            Compounding::Simple => (1.0 / df - 1.0) / t,
        }
    }
}

/// Interpolation scheme between curve pillars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum InterpolationMethod {
    /// Linear in ln(df) — market-standard default; piecewise-flat forwards.
    #[default]
    LogLinearDf,
    /// Linear in the continuously compounded zero rate.
    LinearZero,
}

/// A curve pillar location: either an absolute date or a year fraction
/// relative to the curve's reference date (e.g. `"2027-07-16"` or `0.25`).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Tenor {
    Date(NaiveDate),
    YearFraction(f64),
}

/// The accepted *input forms* for a curve. This is what deserializes from
/// JSON; every form is canonicalized to discount factors at construction
/// ([`YieldCurve::from_input`]), so pricing code sees a single representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CurveInput {
    /// A single constant rate for all maturities.
    Flat {
        rate: f64,
        #[serde(default)]
        compounding: Compounding,
        #[serde(default)]
        day_count: DayCountConvention,
    },
    /// Zero rates quoted in `compounding` at the given tenors.
    ZeroRates {
        tenors: Vec<Tenor>,
        rates: Vec<f64>,
        #[serde(default)]
        compounding: Compounding,
        #[serde(default)]
        day_count: DayCountConvention,
        #[serde(default)]
        interpolation: InterpolationMethod,
    },
    /// Discount factors at the given tenors (`compounding` only sets the
    /// quoting convention for rate queries on the resulting curve).
    DiscountFactors {
        tenors: Vec<Tenor>,
        dfs: Vec<f64>,
        #[serde(default)]
        compounding: Compounding,
        #[serde(default)]
        day_count: DayCountConvention,
        #[serde(default)]
        interpolation: InterpolationMethod,
    },
    /// Forward rates, each applying from the previous tenor (or the
    /// reference date for the first) to its own tenor.
    ForwardRates {
        tenors: Vec<Tenor>,
        forwards: Vec<f64>,
        #[serde(default)]
        compounding: Compounding,
        #[serde(default)]
        day_count: DayCountConvention,
        #[serde(default)]
        interpolation: InterpolationMethod,
    },
}

/// Errors from curve construction or queries.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveError {
    Empty,
    LengthMismatch { tenors: usize, values: usize },
    NonPositiveDf(f64),
    NonPositiveTime(f64),
    NonIncreasingTimes,
    InvalidForwardPeriod { t1: f64, t2: f64 },
    /// Two key-rate bump tenors resolve to the same curve pillar (they are
    /// closer together than twice [`KEY_RATE_TENOR_TOLERANCE`]).
    TenorCollision { t1: f64, t2: f64 },
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CurveError::Empty => write!(f, "curve needs at least one pillar"),
            CurveError::LengthMismatch { tenors, values } => {
                write!(f, "tenors ({tenors}) and values ({values}) differ in length")
            }
            CurveError::NonPositiveDf(df) => write!(f, "discount factor must be > 0, got {df}"),
            CurveError::NonPositiveTime(t) => write!(f, "pillar time must be > 0, got {t}"),
            CurveError::NonIncreasingTimes => write!(f, "pillar times must be strictly increasing"),
            CurveError::InvalidForwardPeriod { t1, t2 } => {
                write!(f, "forward period requires t2 > t1 >= 0, got t1={t1}, t2={t2}")
            }
            CurveError::TenorCollision { t1, t2 } => {
                write!(f, "bump tenors {t1} and {t2} resolve to the same curve pillar")
            }
        }
    }
}

impl std::error::Error for CurveError {}

/// One pillar of the curve, with the zero rate derived for inspection.
#[derive(Debug, Clone, Copy)]
pub struct CurvePillar {
    /// Original pillar date, when the curve was built from date tenors.
    pub date: Option<NaiveDate>,
    pub time: f64,
    pub df: f64,
    /// Continuously compounded zero rate at this pillar.
    pub zero_rate: f64,
}

/// One inter-pillar segment and its discrete continuously compounded
/// forward rate, as reported by [`YieldCurve::min_forward`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardSegment {
    pub t1: f64,
    pub t2: f64,
    pub forward: f64,
}

/// A shift applied to a curve by [`YieldCurve::bumped`]. Shifts act on the
/// **continuously compounded zero rates** (the curve's discount factors
/// are re-derived exactly), independent of the curve's quoting convention.
#[derive(Debug, Clone, PartialEq)]
pub enum RateShift {
    /// Add `d` to every continuous zero rate (e.g. `0.0001` = +1bp).
    ParallelAbsolute(f64),
    /// Scale every continuous zero rate by `1 + r`.
    ParallelRelative(f64),
    /// Key-rate bump: add `shifts[i]` to the continuous zero rate at pillar
    /// `tenors[i]` (year fractions, strictly increasing). Each tenor reuses
    /// the nearest existing pillar within [`KEY_RATE_TENOR_TOLERANCE`];
    /// otherwise a pillar is inserted on the base curve first, so the bump
    /// is represented exactly. Off-pillar shape follows the curve's own
    /// interpolation — no separate shift interpolation exists — which makes
    /// node bumps exactly additive: single-tenor bumps over the full pillar
    /// set sum to the parallel bump to machine precision.
    KeyRateAbsolute { tenors: Vec<f64>, shifts: Vec<f64> },
}

/// A bump tenor within this distance (in years, ~3.7 days) of an existing
/// pillar reuses that pillar; farther away, a new pillar is inserted.
/// Prevents needle-thin tents when date-built pillars sit at times like
/// `1.0027` and the bump asks for `1.0`.
pub const KEY_RATE_TENOR_TOLERANCE: f64 = 0.01;

/// A canonical discount curve anchored at `reference_date`.
///
/// State is the pillar `(times, dfs)` vectors only — `dfs[0] = 1.0` at
/// `times[0] = 0.0` always. `compounding` is the quoting convention used by
/// [`zero_rate`](Self::zero_rate) / [`forward_rate`](Self::forward_rate);
/// changing it never changes discounting.
#[derive(Debug, Clone, Serialize)]
pub struct YieldCurve {
    reference_date: NaiveDate,
    day_count: DayCountConvention,
    compounding: Compounding,
    interpolation: InterpolationMethod,
    times: Vec<f64>,
    dfs: Vec<f64>,
    dates: Vec<Option<NaiveDate>>,
}

/// Pillar grid used to materialize a flat curve. Log-linear interpolation is
/// exact between these pillars for continuous and annual compounding; for
/// simple compounding the curve is exact at the pillars.
const FLAT_CURVE_GRID: [f64; 13] = [
    1.0 / 365.0,
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
];

impl YieldCurve {
    // ── Constructors ────────────────────────────────────────────────────

    /// Flat curve at a single `rate` quoted in `compounding`.
    pub fn flat(
        rate: f64,
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        compounding: Compounding,
    ) -> Result<Self, CurveError> {
        let tenors: Vec<Tenor> = FLAT_CURVE_GRID.iter().map(|&t| Tenor::YearFraction(t)).collect();
        let rates = vec![rate; tenors.len()];
        Self::from_zero_rates(
            &tenors,
            &rates,
            reference_date,
            day_count,
            compounding,
            InterpolationMethod::LogLinearDf,
        )
    }

    /// Curve from zero rates quoted in `compounding`.
    pub fn from_zero_rates(
        tenors: &[Tenor],
        rates: &[f64],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        compounding: Compounding,
        interpolation: InterpolationMethod,
    ) -> Result<Self, CurveError> {
        let (times, dates) = Self::resolve_tenors(tenors, reference_date, day_count)?;
        if rates.len() != times.len() {
            return Err(CurveError::LengthMismatch { tenors: times.len(), values: rates.len() });
        }
        let dfs: Vec<f64> = times.iter().zip(rates).map(|(&t, &z)| compounding.df(z, t)).collect();
        Self::from_parts(reference_date, day_count, compounding, interpolation, times, dfs, dates)
    }

    /// Curve directly from discount factors.
    pub fn from_discount_factors(
        tenors: &[Tenor],
        dfs: &[f64],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        compounding: Compounding,
        interpolation: InterpolationMethod,
    ) -> Result<Self, CurveError> {
        let (times, dates) = Self::resolve_tenors(tenors, reference_date, day_count)?;
        if dfs.len() != times.len() {
            return Err(CurveError::LengthMismatch { tenors: times.len(), values: dfs.len() });
        }
        Self::from_parts(
            reference_date,
            day_count,
            compounding,
            interpolation,
            times,
            dfs.to_vec(),
            dates,
        )
    }

    /// Curve from forward rates: `forwards[i]` applies between tenor `i-1`
    /// (or the reference date for `i = 0`) and tenor `i`, quoted in
    /// `compounding`.
    pub fn from_forward_rates(
        tenors: &[Tenor],
        forwards: &[f64],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        compounding: Compounding,
        interpolation: InterpolationMethod,
    ) -> Result<Self, CurveError> {
        let (times, dates) = Self::resolve_tenors(tenors, reference_date, day_count)?;
        if forwards.len() != times.len() {
            return Err(CurveError::LengthMismatch { tenors: times.len(), values: forwards.len() });
        }
        let mut dfs = Vec::with_capacity(times.len());
        let mut prev_t = 0.0;
        let mut prev_df = 1.0;
        for (&t, &fwd) in times.iter().zip(forwards) {
            let df = prev_df * compounding.df(fwd, t - prev_t);
            dfs.push(df);
            prev_t = t;
            prev_df = df;
        }
        Self::from_parts(reference_date, day_count, compounding, interpolation, times, dfs, dates)
    }

    /// Build from a deserialized [`CurveInput`], anchored at `reference_date`.
    pub fn from_input(input: &CurveInput, reference_date: NaiveDate) -> Result<Self, CurveError> {
        match input {
            CurveInput::Flat { rate, compounding, day_count } => {
                Self::flat(*rate, reference_date, *day_count, *compounding)
            }
            CurveInput::ZeroRates { tenors, rates, compounding, day_count, interpolation } => {
                Self::from_zero_rates(tenors, rates, reference_date, *day_count, *compounding, *interpolation)
            }
            CurveInput::DiscountFactors { tenors, dfs, compounding, day_count, interpolation } => {
                Self::from_discount_factors(tenors, dfs, reference_date, *day_count, *compounding, *interpolation)
            }
            CurveInput::ForwardRates { tenors, forwards, compounding, day_count, interpolation } => {
                Self::from_forward_rates(tenors, forwards, reference_date, *day_count, *compounding, *interpolation)
            }
        }
    }

    /// This curve with `shift` applied to its continuous zero rates.
    /// The discount factors are re-derived exactly at the affected pillars:
    /// `z -> z + d` gives `df -> df * exp(-d*t)`, `z -> z*(1+r)` gives
    /// `df -> df^(1+r)`. Day count, quoting convention and interpolation
    /// are unchanged; `df(0) = 1` is preserved. A key-rate shift may add
    /// pillars (see [`RateShift::KeyRateAbsolute`]); parallel shifts never
    /// do. Errors only on a malformed key-rate shift.
    pub fn bumped(&self, shift: &RateShift) -> Result<YieldCurve, CurveError> {
        let mut bumped = self.clone();
        match shift {
            RateShift::ParallelAbsolute(d) => {
                for (df, &t) in bumped.dfs.iter_mut().zip(self.times.iter()) {
                    *df *= (-d * t).exp();
                }
            }
            RateShift::ParallelRelative(r) => {
                for df in bumped.dfs.iter_mut() {
                    *df = df.powf(1.0 + r);
                }
            }
            RateShift::KeyRateAbsolute { tenors, shifts } => {
                Self::validate_key_rate(tenors, shifts)?;
                // Pass 1: give every bump tenor a pillar. Missing ones are
                // inserted with the *base* curve's df — both interpolation
                // methods are piecewise linear in a transform, so inserting
                // an on-curve point leaves df(t) unchanged everywhere.
                let mut targets: Vec<usize> = Vec::with_capacity(tenors.len());
                let mut prev: Option<(usize, f64)> = None;
                for &tenor in tenors {
                    let idx = match bumped.nearest_pillar(tenor) {
                        Some(i) => i,
                        None => bumped.insert_pillar(tenor, self.df(tenor)),
                    };
                    if let Some((prev_idx, prev_tenor)) = prev {
                        if idx <= prev_idx {
                            return Err(CurveError::TenorCollision { t1: prev_tenor, t2: tenor });
                        }
                    }
                    prev = Some((idx, tenor));
                    targets.push(idx);
                }
                // Pass 2: shift each target pillar's continuous zero.
                for (&idx, &d) in targets.iter().zip(shifts) {
                    bumped.dfs[idx] *= (-d * bumped.times[idx]).exp();
                }
            }
        }
        Ok(bumped)
    }

    /// The inter-pillar segment with the smallest discrete continuous
    /// forward `ln(df(t1)/df(t2)) / (t2 - t1)` — the no-arbitrage
    /// diagnostic: a value below zero means the discount factors increase
    /// somewhere. Checking consecutive pillars suffices: under
    /// [`InterpolationMethod::LogLinearDf`] this *is* the instantaneous
    /// forward on the segment; under `LinearZero` it is the segment
    /// average. The first segment starts at the `t = 0` anchor.
    pub fn min_forward(&self) -> ForwardSegment {
        let mut worst = ForwardSegment { t1: 0.0, t2: 0.0, forward: f64::INFINITY };
        for i in 0..self.times.len() - 1 {
            let (t1, t2) = (self.times[i], self.times[i + 1]);
            let forward = (self.dfs[i] / self.dfs[i + 1]).ln() / (t2 - t1);
            if forward < worst.forward {
                worst = ForwardSegment { t1, t2, forward };
            }
        }
        worst
    }

    // ── Queries ─────────────────────────────────────────────────────────

    /// Discount factor at year fraction `t` from the reference date.
    /// `t <= 0` returns 1.0; beyond the last pillar the last continuously
    /// compounded zero rate is extrapolated flat.
    pub fn df(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        let n = self.times.len();
        let t_last = self.times[n - 1];
        if t >= t_last {
            // flat extrapolation of the last zero rate
            let z_last = -self.dfs[n - 1].ln() / t_last;
            return (-z_last * t).exp();
        }
        // shared pillar bracketing; idx >= 1 because times[0] = 0 < t
        let (idx, w) = crate::core::interpolation::bracket(&self.times, t);
        let (df0, df1) = (self.dfs[idx - 1], self.dfs[idx]);
        match self.interpolation {
            InterpolationMethod::LogLinearDf => {
                crate::core::interpolation::lerp(df0.ln(), df1.ln(), w).exp()
            }
            InterpolationMethod::LinearZero => {
                let z0 = self.pillar_zero(idx - 1);
                let z1 = self.pillar_zero(idx);
                let z = crate::core::interpolation::lerp(z0, z1, w);
                (-z * t).exp()
            }
        }
    }

    /// Discount factor at an absolute date (via the curve's day count).
    pub fn df_date(&self, date: NaiveDate) -> f64 {
        self.df(self.day_count.year_fraction(self.reference_date, date))
    }

    /// Zero rate at `t` in the curve's quoting convention.
    pub fn zero_rate(&self, t: f64) -> f64 {
        self.zero_rate_with(t, self.compounding)
    }

    /// Zero rate at `t` in an explicit convention.
    pub fn zero_rate_with(&self, t: f64, compounding: Compounding) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        compounding.rate(self.df(t), t)
    }

    /// Forward rate between `t1` and `t2` in the curve's quoting convention.
    pub fn forward_rate(&self, t1: f64, t2: f64) -> Result<f64, CurveError> {
        self.forward_rate_with(t1, t2, self.compounding)
    }

    /// Forward rate between `t1` and `t2` in an explicit convention
    /// (`Simple` gives the FRA-style forward).
    pub fn forward_rate_with(
        &self,
        t1: f64,
        t2: f64,
        compounding: Compounding,
    ) -> Result<f64, CurveError> {
        if !(t2 > t1 && t1 >= 0.0) {
            return Err(CurveError::InvalidForwardPeriod { t1, t2 });
        }
        let df12 = self.df(t2) / self.df(t1);
        Ok(compounding.rate(df12, t2 - t1))
    }

    pub fn reference_date(&self) -> NaiveDate {
        self.reference_date
    }
    pub fn day_count(&self) -> DayCountConvention {
        self.day_count
    }
    pub fn compounding(&self) -> Compounding {
        self.compounding
    }

    /// The curve's pillars (excluding the synthetic t=0 node) with derived
    /// continuously compounded zero rates — for inspection and display;
    /// always computed fresh from the stored dfs so it cannot disagree with
    /// what `df(t)` returns.
    pub fn pillars(&self) -> Vec<CurvePillar> {
        (1..self.times.len())
            .map(|i| CurvePillar {
                date: self.dates[i],
                time: self.times[i],
                df: self.dfs[i],
                zero_rate: self.pillar_zero(i),
            })
            .collect()
    }

    // ── Internals ───────────────────────────────────────────────────────

    /// Continuously compounded zero at pillar `i` (internal interpolation
    /// math is always continuous, independent of the quoting convention).
    fn pillar_zero(&self, i: usize) -> f64 {
        if self.times[i] <= 0.0 {
            // flat short end: use the first real pillar's zero
            return -self.dfs[1].ln() / self.times[1];
        }
        -self.dfs[i].ln() / self.times[i]
    }

    fn validate_key_rate(tenors: &[f64], shifts: &[f64]) -> Result<(), CurveError> {
        if tenors.is_empty() {
            return Err(CurveError::Empty);
        }
        if tenors.len() != shifts.len() {
            return Err(CurveError::LengthMismatch { tenors: tenors.len(), values: shifts.len() });
        }
        for &t in tenors {
            if t <= 0.0 {
                return Err(CurveError::NonPositiveTime(t));
            }
        }
        if tenors.windows(2).any(|w| w[1] <= w[0]) {
            return Err(CurveError::NonIncreasingTimes);
        }
        Ok(())
    }

    /// The real pillar (index >= 1, never the t=0 anchor) nearest to `t`
    /// within [`KEY_RATE_TENOR_TOLERANCE`], if any.
    fn nearest_pillar(&self, t: f64) -> Option<usize> {
        let mut best: Option<usize> = None;
        for i in 1..self.times.len() {
            let dist = (self.times[i] - t).abs();
            if dist <= KEY_RATE_TENOR_TOLERANCE
                && best.map_or(true, |j| dist < (self.times[j] - t).abs())
            {
                best = Some(i);
            }
        }
        best
    }

    /// Insert a pillar at time `t` with discount factor `df`, keeping the
    /// grids sorted; returns its index. The date is unknown (`None`).
    fn insert_pillar(&mut self, t: f64, df: f64) -> usize {
        let idx = self.times.partition_point(|&x| x < t);
        self.times.insert(idx, t);
        self.dfs.insert(idx, df);
        self.dates.insert(idx, None);
        idx
    }

    fn resolve_tenors(
        tenors: &[Tenor],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<(Vec<f64>, Vec<Option<NaiveDate>>), CurveError> {
        if tenors.is_empty() {
            return Err(CurveError::Empty);
        }
        let mut times = Vec::with_capacity(tenors.len());
        let mut dates = Vec::with_capacity(tenors.len());
        for tenor in tenors {
            match tenor {
                Tenor::Date(d) => {
                    times.push(day_count.year_fraction(reference_date, *d));
                    dates.push(Some(*d));
                }
                Tenor::YearFraction(t) => {
                    times.push(*t);
                    dates.push(None);
                }
            }
        }
        Ok((times, dates))
    }

    fn from_parts(
        reference_date: NaiveDate,
        day_count: DayCountConvention,
        compounding: Compounding,
        interpolation: InterpolationMethod,
        mut times: Vec<f64>,
        mut dfs: Vec<f64>,
        mut dates: Vec<Option<NaiveDate>>,
    ) -> Result<Self, CurveError> {
        for &t in &times {
            if t <= 0.0 {
                return Err(CurveError::NonPositiveTime(t));
            }
        }
        for &df in &dfs {
            // dfs > 1 are allowed (negative rates); dfs <= 0 are not
            if df <= 0.0 {
                return Err(CurveError::NonPositiveDf(df));
            }
        }
        if times.windows(2).any(|w| w[1] <= w[0]) {
            return Err(CurveError::NonIncreasingTimes);
        }
        // synthetic anchor node at t = 0
        times.insert(0, 0.0);
        dfs.insert(0, 1.0);
        dates.insert(0, Some(reference_date));
        Ok(YieldCurve { reference_date, day_count, compounding, interpolation, times, dfs, dates })
    }
}

impl fmt::Display for YieldCurve {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "YieldCurve (ref {}, {:?}, {:?}, {:?})",
            self.reference_date, self.day_count, self.compounding, self.interpolation
        )?;
        writeln!(f, "{:>12} {:>12} {:>12} {:>12}", "date", "time", "df", "zero(cont)")?;
        for p in self.pillars() {
            let date = p.date.map_or_else(|| "-".to_string(), |d| d.to_string());
            writeln!(f, "{:>12} {:>12.6} {:>12.8} {:>12.6}", date, p.time, p.df, p.zero_rate)?;
        }
        let worst = self.min_forward();
        writeln!(
            f,
            "min forward (cont): {:.6} on [{:.4}, {:.4}]",
            worst.forward, worst.t1, worst.t2
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 7, 16).unwrap()
    }

    fn flat_5pct() -> YieldCurve {
        YieldCurve::flat(0.05, asof(), DayCountConvention::Act365, Compounding::Continuous).unwrap()
    }

    #[test]
    fn bumped_shifts_continuous_zeros_exactly() {
        let curve = flat_5pct();
        let up = curve.bumped(&RateShift::ParallelAbsolute(0.01)).unwrap();
        for t in [0.1, 1.0, 4.2, 10.0, 30.0, 60.0] {
            // +100bp on a 5% flat curve = a 6% flat curve, including extrapolation
            assert!(
                (up.df(t) - (-0.06_f64 * t).exp()).abs() < 1e-12,
                "df({t}) = {}",
                up.df(t)
            );
            assert!((up.zero_rate_with(t, Compounding::Continuous) - 0.06).abs() < 1e-12);
        }
        // relative: zeros scale, 5% * 1.2 = 6%
        let scaled = curve.bumped(&RateShift::ParallelRelative(0.20)).unwrap();
        assert!((scaled.zero_rate_with(1.0, Compounding::Continuous) - 0.06).abs() < 1e-12);
        // df(0) = 1 preserved, original untouched
        assert_eq!(up.df(0.0), 1.0);
        assert!((curve.zero_rate_with(1.0, Compounding::Continuous) - 0.05).abs() < 1e-12);
    }

    fn key_rate(tenors: &[f64], shifts: &[f64]) -> RateShift {
        RateShift::KeyRateAbsolute { tenors: tenors.to_vec(), shifts: shifts.to_vec() }
    }

    #[test]
    fn key_rate_bump_moves_target_pillar_and_decays_to_neighbours() {
        let curve = flat_5pct();
        let up = curve.bumped(&key_rate(&[2.0], &[0.01])).unwrap();
        let z = |c: &YieldCurve, t: f64| c.zero_rate_with(t, Compounding::Continuous);
        // full bump at the target pillar, neighbours (1y, 3y pillars) untouched
        assert!((z(&up, 2.0) - 0.06).abs() < 1e-12);
        assert!((z(&up, 1.0) - 0.05).abs() < 1e-12);
        assert!((z(&up, 3.0) - 0.05).abs() < 1e-12);
        // strictly between: partial bump, shaped by the curve interpolation
        let mid = z(&up, 2.5);
        assert!(mid > 0.05 + 1e-6 && mid < 0.06 - 1e-6, "mid-tent zero {mid}");
        // pillar count unchanged: 2.0 is an existing grid point
        assert_eq!(up.pillars().len(), curve.pillars().len());
    }

    #[test]
    fn key_rate_plateau_between_equally_bumped_tenors() {
        let curve = flat_5pct();
        let up = curve.bumped(&key_rate(&[1.0, 2.0], &[0.005, 0.005])).unwrap();
        // interior of [1y, 2y] carries exactly the full bump (log-linear df
        // interpolation is linear in z*t, so equal node bumps lerp exactly)
        for t in [1.0, 1.25, 1.5, 1.75, 2.0] {
            assert!(
                (up.zero_rate_with(t, Compounding::Continuous) - 0.055).abs() < 1e-12,
                "plateau broken at t={t}"
            );
        }
        // decays outside toward the adjacent unbumped pillars (0.5y, 3y)
        assert!((up.zero_rate_with(0.5, Compounding::Continuous) - 0.05).abs() < 1e-12);
        assert!((up.zero_rate_with(3.0, Compounding::Continuous) - 0.05).abs() < 1e-12);
    }

    #[test]
    fn key_rate_bumps_sum_exactly_to_parallel() {
        let curve = flat_5pct();
        let d = 0.0025;
        let pillar_times: Vec<f64> = curve.pillars().iter().map(|p| p.time).collect();
        // apply single-tenor bumps successively: composition = sum, since
        // each bump touches only its own pillar
        let mut laddered = curve.clone();
        for &t in &pillar_times {
            laddered = laddered.bumped(&key_rate(&[t], &[d])).unwrap();
        }
        let parallel = curve.bumped(&RateShift::ParallelAbsolute(d)).unwrap();
        for t in [0.1, 0.7, 1.0, 2.5, 9.0, 30.0, 55.0] {
            assert!(
                (laddered.df(t) - parallel.df(t)).abs() < 1e-14,
                "ladder != parallel at t={t}: {} vs {}",
                laddered.df(t),
                parallel.df(t)
            );
        }
    }

    #[test]
    fn key_rate_tenor_off_grid_inserts_a_pillar_exactly() {
        let curve = flat_5pct();
        let up = curve.bumped(&key_rate(&[1.5], &[0.01])).unwrap();
        assert_eq!(up.pillars().len(), curve.pillars().len() + 1);
        // the inserted pillar carries base + bump exactly; grid pillars around
        // it are untouched
        assert!((up.zero_rate_with(1.5, Compounding::Continuous) - 0.06).abs() < 1e-12);
        assert!((up.zero_rate_with(1.0, Compounding::Continuous) - 0.05).abs() < 1e-12);
        assert!((up.zero_rate_with(2.0, Compounding::Continuous) - 0.05).abs() < 1e-12);
        // a zero-size bump at an off-grid tenor reproduces the base curve
        let noop = curve.bumped(&key_rate(&[1.5], &[0.0])).unwrap();
        for t in [0.3, 1.2, 1.5, 1.9, 4.0] {
            assert!((noop.df(t) - curve.df(t)).abs() < 1e-15, "insertion changed df({t})");
        }
    }

    #[test]
    fn key_rate_tolerance_matches_nearby_pillar_instead_of_inserting() {
        // date-built pillar at 367/365 ≈ 1.0055; a 1.0 bump tenor must reuse it
        let pillar_date = NaiveDate::from_ymd_opt(2027, 7, 18).unwrap(); // 367 days
        let curve = YieldCurve::from_zero_rates(
            &[Tenor::Date(pillar_date), Tenor::YearFraction(2.0)],
            &[0.05, 0.05],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let up = curve.bumped(&key_rate(&[1.0], &[0.01])).unwrap();
        assert_eq!(up.pillars().len(), curve.pillars().len(), "must not insert");
        let t_pillar = 367.0 / 365.0;
        assert!((up.zero_rate_with(t_pillar, Compounding::Continuous) - 0.06).abs() < 1e-12);
    }

    #[test]
    fn key_rate_validation_errors() {
        let curve = flat_5pct();
        assert_eq!(curve.bumped(&key_rate(&[], &[])).unwrap_err(), CurveError::Empty);
        assert!(matches!(
            curve.bumped(&key_rate(&[1.0], &[0.01, 0.02])).unwrap_err(),
            CurveError::LengthMismatch { .. }
        ));
        assert!(matches!(
            curve.bumped(&key_rate(&[-1.0], &[0.01])).unwrap_err(),
            CurveError::NonPositiveTime(_)
        ));
        assert_eq!(
            curve.bumped(&key_rate(&[2.0, 1.0], &[0.01, 0.01])).unwrap_err(),
            CurveError::NonIncreasingTimes
        );
        // 1.0 and 1.005 both resolve to the 1y pillar
        assert!(matches!(
            curve.bumped(&key_rate(&[1.0, 1.005], &[0.01, 0.01])).unwrap_err(),
            CurveError::TenorCollision { .. }
        ));
    }

    #[test]
    fn min_forward_flags_negative_forwards_from_a_hard_down_bump() {
        let curve = flat_5pct();
        assert!((curve.min_forward().forward - 0.05).abs() < 1e-10, "flat curve forward");
        // -200bp at 10y: the zero bump amplifies into the preceding forward
        // by t/dt = 10/3 => forward 5% - 2%*10/3 < 0 on [7, 10]
        let down = curve.bumped(&key_rate(&[10.0], &[-0.02])).unwrap();
        let worst = down.min_forward();
        assert!(worst.forward < 0.0, "expected negative forward, got {}", worst.forward);
        assert!((worst.t1 - 7.0).abs() < 1e-12 && (worst.t2 - 10.0).abs() < 1e-12);
        // the same size at the short end stays arbitrage-free: entering the
        // [1y, 2y] plateau costs only d * t/dt = 2% * 1/0.5 = 4% < 5%
        let gentle = curve.bumped(&key_rate(&[1.0, 2.0], &[-0.02, -0.02])).unwrap();
        assert!(gentle.min_forward().forward > 0.0, "got {:?}", gentle.min_forward());
    }

    #[test]
    fn flat_curve_matches_closed_form() {
        let curve = flat_5pct();
        for t in [0.1, 0.5, 1.0, 1.7, 4.2, 10.0, 30.0, 60.0] {
            let expected = (-0.05_f64 * t).exp();
            assert!(
                (curve.df(t) - expected).abs() < 1e-12,
                "t={t}: {} vs {expected}",
                curve.df(t)
            );
        }
        assert_eq!(curve.df(0.0), 1.0);
        assert_eq!(curve.df(-1.0), 1.0);
    }

    #[test]
    fn flat_curve_annual_compounding() {
        let curve =
            YieldCurve::flat(0.04, asof(), DayCountConvention::Act365, Compounding::Annual).unwrap();
        // exact everywhere under log-linear interpolation
        assert!((curve.df(2.0) - 0.924556213018).abs() < 1e-10);
        assert!((curve.df(1.3) - 1.04_f64.powf(-1.3)).abs() < 1e-12);
        // zero rate reported back in the curve's own convention
        assert!((curve.zero_rate(2.0) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn simple_compounding_exact_at_pillars() {
        let tenors = [Tenor::YearFraction(0.5), Tenor::YearFraction(2.0)];
        let curve = YieldCurve::from_zero_rates(
            &tenors,
            &[0.04, 0.04],
            asof(),
            DayCountConvention::Act365,
            Compounding::Simple,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        assert!((curve.df(2.0) - 0.925925925926).abs() < 1e-10);
        assert!((curve.zero_rate(2.0) - 0.04).abs() < 1e-12);
    }

    #[test]
    fn zero_rate_round_trip_all_compoundings() {
        for comp in [Compounding::Continuous, Compounding::Annual, Compounding::Simple] {
            for (z, t) in [(0.03, 0.5), (0.05, 1.0), (-0.005, 2.0), (0.07, 10.0)] {
                let df = comp.df(z, t);
                assert!(
                    (comp.rate(df, t) - z).abs() < 1e-12,
                    "{comp:?} z={z} t={t}"
                );
            }
        }
    }

    #[test]
    fn input_forms_agree_on_flat_curve() {
        // the same flat 5% (continuous) curve expressed four ways
        let tenors = [Tenor::YearFraction(1.0), Tenor::YearFraction(2.0), Tenor::YearFraction(5.0)];
        let dc = DayCountConvention::Act365;
        let comp = Compounding::Continuous;
        let interp = InterpolationMethod::LogLinearDf;

        let from_flat = YieldCurve::flat(0.05, asof(), dc, comp).unwrap();
        let from_zeros =
            YieldCurve::from_zero_rates(&tenors, &[0.05; 3], asof(), dc, comp, interp).unwrap();
        let dfs: Vec<f64> = [1.0_f64, 2.0, 5.0].iter().map(|t| (-0.05 * t).exp()).collect();
        let from_dfs =
            YieldCurve::from_discount_factors(&tenors, &dfs, asof(), dc, comp, interp).unwrap();
        let from_fwds =
            YieldCurve::from_forward_rates(&tenors, &[0.05; 3], asof(), dc, comp, interp).unwrap();

        for t in [0.3, 1.0, 1.7, 4.9] {
            let reference = from_flat.df(t);
            for (name, curve) in
                [("zeros", &from_zeros), ("dfs", &from_dfs), ("fwds", &from_fwds)]
            {
                assert!(
                    (curve.df(t) - reference).abs() < 1e-12,
                    "{name} disagrees at t={t}"
                );
            }
        }
    }

    #[test]
    fn date_and_yearfraction_tenors_agree() {
        let one_year_date = NaiveDate::from_ymd_opt(2027, 7, 16).unwrap(); // 365 days from asof
        let by_date = YieldCurve::from_zero_rates(
            &[Tenor::Date(one_year_date)],
            &[0.05],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let by_time = YieldCurve::from_zero_rates(
            &[Tenor::YearFraction(1.0)],
            &[0.05],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        assert!((by_date.df(1.0) - by_time.df(1.0)).abs() < 1e-14);
        assert!((by_date.df_date(one_year_date) - (-0.05_f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn log_linear_interpolation_between_pillars() {
        let tenors = [Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)];
        let dfs = [(-0.05_f64).exp(), (-0.12_f64).exp()];
        let curve = YieldCurve::from_discount_factors(
            &tenors,
            &dfs,
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        // ln df linear: at t=1.4, ln df = 0.6*(-0.05) + 0.4*(-0.12)
        assert!((curve.df(1.4) - 0.924964426544).abs() < 1e-10);
    }

    #[test]
    fn forward_rate_on_flat_curve_equals_rate() {
        let curve = flat_5pct();
        let fwd = curve.forward_rate_with(1.0, 2.0, Compounding::Continuous).unwrap();
        assert!((fwd - 0.05).abs() < 1e-10);
        // FRA-style simple forward over 6M on a flat 5% cc curve
        let fwd_simple = curve.forward_rate_with(1.0, 1.5, Compounding::Simple).unwrap();
        let expected = ((0.05_f64 * 0.5).exp() - 1.0) / 0.5;
        assert!((fwd_simple - expected).abs() < 1e-12);
        assert!(curve.forward_rate_with(2.0, 1.0, Compounding::Simple).is_err());
    }

    #[test]
    fn extrapolation_is_flat_in_zero_rate() {
        let tenors = [Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)];
        let curve = YieldCurve::from_zero_rates(
            &tenors,
            &[0.03, 0.05],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        assert!((curve.zero_rate_with(7.0, Compounding::Continuous) - 0.05).abs() < 1e-12);
        assert!((curve.df(7.0) - (-0.05_f64 * 7.0).exp()).abs() < 1e-12);
    }

    #[test]
    fn negative_rates_allowed() {
        let curve =
            YieldCurve::flat(-0.005, asof(), DayCountConvention::Act365, Compounding::Continuous)
                .unwrap();
        assert!(curve.df(2.0) > 1.0);
        assert!((curve.zero_rate(2.0) + 0.005).abs() < 1e-12);
    }

    #[test]
    fn validation_errors() {
        let dc = DayCountConvention::Act365;
        let comp = Compounding::Continuous;
        let interp = InterpolationMethod::LogLinearDf;
        // empty
        assert_eq!(
            YieldCurve::from_zero_rates(&[], &[], asof(), dc, comp, interp).unwrap_err(),
            CurveError::Empty
        );
        // length mismatch
        assert!(matches!(
            YieldCurve::from_zero_rates(
                &[Tenor::YearFraction(1.0)],
                &[0.05, 0.06],
                asof(),
                dc,
                comp,
                interp
            )
            .unwrap_err(),
            CurveError::LengthMismatch { .. }
        ));
        // non-increasing times
        assert_eq!(
            YieldCurve::from_zero_rates(
                &[Tenor::YearFraction(2.0), Tenor::YearFraction(1.0)],
                &[0.05, 0.05],
                asof(),
                dc,
                comp,
                interp
            )
            .unwrap_err(),
            CurveError::NonIncreasingTimes
        );
        // non-positive time
        assert!(matches!(
            YieldCurve::from_zero_rates(&[Tenor::YearFraction(0.0)], &[0.05], asof(), dc, comp, interp)
                .unwrap_err(),
            CurveError::NonPositiveTime(_)
        ));
        // non-positive df
        assert!(matches!(
            YieldCurve::from_discount_factors(
                &[Tenor::YearFraction(1.0)],
                &[0.0],
                asof(),
                dc,
                comp,
                interp
            )
            .unwrap_err(),
            CurveError::NonPositiveDf(_)
        ));
    }

    #[test]
    fn curve_input_deserializes_from_json() {
        // flat, minimal
        let flat: CurveInput = serde_json::from_str(r#"{"type": "flat", "rate": 0.05}"#).unwrap();
        let curve = YieldCurve::from_input(&flat, asof()).unwrap();
        assert!((curve.df(1.0) - (-0.05_f64).exp()).abs() < 1e-12);

        // zero rates with mixed date / year-fraction tenors and explicit conventions
        let zeros: CurveInput = serde_json::from_str(
            r#"{
                "type": "zero_rates",
                "tenors": [0.5, "2027-07-16", 5.0],
                "rates": [0.03, 0.04, 0.05],
                "compounding": "annual",
                "day_count": "Act365"
            }"#,
        )
        .unwrap();
        let curve = YieldCurve::from_input(&zeros, asof()).unwrap();
        assert!((curve.df(1.0) - 1.04_f64.powf(-1.0)).abs() < 1e-12);
        assert!((curve.zero_rate(1.0) - 0.04).abs() < 1e-12);

        // discount factors
        let dfs: CurveInput = serde_json::from_str(
            r#"{"type": "discount_factors", "tenors": [1.0, 2.0], "dfs": [0.95, 0.90]}"#,
        )
        .unwrap();
        let curve = YieldCurve::from_input(&dfs, asof()).unwrap();
        assert!((curve.df(1.0) - 0.95).abs() < 1e-12);
    }

    #[test]
    fn display_prints_pillar_table() {
        let text = format!("{}", flat_5pct());
        assert!(text.contains("zero(cont)"));
        assert!(text.contains("0.05000")); // zero column shows the flat rate
    }
}
