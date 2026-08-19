//! Dupire local volatility calibrated from an implied vol surface.
//!
//! Uses Gatheral's formulation of the Dupire equation in total variance
//! `w(y, t) = sigma_imp(K, t)^2 * t`, `y = ln(K / F(t))`:
//!
//! ```text
//! sigma_loc^2(K, t) = (dw/dt) /
//!   [ 1 - (y/w) w_y + 1/4 (-1/4 - 1/w + y^2/w^2) w_y^2 + 1/2 w_yy ]
//! ```
//!
//! Derivatives are taken numerically on the implied surface: the time
//! derivative at fixed moneyness `y`, the strike derivatives at fixed `t`.
//! The "calibration" is therefore non-parametric — the local vol function
//! is the exact transformation of whatever implied surface it is given.
//!
//! Guards: at very short times the implied vol is returned directly; where
//! interpolation noise makes the denominator or numerator non-positive
//! (butterfly / calendar violations in the inputs) the implied vol is used
//! as a fallback; the result is clamped to `[1%, 300%]`.

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::vols::VolSurface;

const TIME_BUMP: f64 = 1.0 / 365.0;
const LOG_STRIKE_BUMP: f64 = 0.01;
const MIN_LOCAL_VOL: f64 = 0.01;
const MAX_LOCAL_VOL: f64 = 3.0;

/// Default pricing-grid resolution and level span (in ATM standard
/// deviations around the spot) for [`LocalVol::to_grid`].
const GRID_LEVELS: usize = 61;
const GRID_TIMES: usize = 31;
const GRID_STDEVS: f64 = 4.0;

/// Local volatility function `sigma_loc(level, t)`, frozen at construction
/// from an implied surface, a discount curve (for forwards) and a dividend
/// yield.
pub struct LocalVol<'a> {
    surface: &'a VolSurface,
    curve: &'a YieldCurve,
    spot: f64,
    dividend_yield: f64,
    /// Parallel shift added to every implied vol before the Dupire
    /// transformation — used by vega bump-and-reprice.
    vol_shift: f64,
}

impl<'a> LocalVol<'a> {
    pub fn new(
        surface: &'a VolSurface,
        curve: &'a YieldCurve,
        spot: f64,
        dividend_yield: f64,
        vol_shift: f64,
    ) -> Self {
        LocalVol {
            surface,
            curve,
            spot,
            dividend_yield,
            vol_shift,
        }
    }

    fn forward(&self, t: f64) -> f64 {
        let r = self.curve.zero_rate_with(t, Compounding::Continuous);
        self.spot * ((r - self.dividend_yield) * t).exp()
    }

    fn implied(&self, strike: f64, t: f64) -> f64 {
        self.surface.vol(strike, self.forward(t), t) + self.vol_shift
    }

    /// Total variance at absolute strike `k` and expiry `t`.
    fn total_variance(&self, strike: f64, t: f64) -> f64 {
        let v = self.implied(strike, t);
        v * v * t
    }

    /// Local volatility at underlying level `level` and time `t`.
    pub fn vol(&self, level: f64, t: f64) -> f64 {
        self.vol_checked(level, t).0
    }

    /// [`vol`](Self::vol) plus whether a guard fired: `true` means the
    /// Dupire transformation was *not* used at this point — the implied
    /// vol was returned instead (very short time, vanishing total
    /// variance, or a calendar/butterfly violation in the inputs). The
    /// usability report's raw material.
    pub fn vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        let implied = self.implied(level, t.max(1e-4));
        if t < 1e-3 {
            return (implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL), true);
        }

        let f = self.forward(t);
        let y = (level / f).ln();
        let w = self.total_variance(level, t);
        if w < 1e-8 {
            return (implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL), true);
        }

        // dw/dt at fixed moneyness y: strike moves with the forward
        let ht = TIME_BUMP.min(0.5 * t);
        let w_up = self.total_variance(self.forward(t + ht) * y.exp(), t + ht);
        let w_dn = self.total_variance(self.forward(t - ht) * y.exp(), t - ht);
        let dw_dt = (w_up - w_dn) / (2.0 * ht);

        // strike derivatives at fixed t (central, multiplicative bump)
        let hy = LOG_STRIKE_BUMP;
        let w_plus = self.total_variance(level * hy.exp(), t);
        let w_minus = self.total_variance(level * (-hy).exp(), t);
        let dw_dy = (w_plus - w_minus) / (2.0 * hy);
        let d2w_dy2 = (w_plus - 2.0 * w + w_minus) / (hy * hy);

        let denom = 1.0 - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + (y * y) / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;

        if dw_dt <= 0.0 || denom <= 1e-4 {
            // calendar / butterfly violation in the interpolated inputs:
            // fall back to the implied vol at this point
            return (implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL), true);
        }
        (
            (dw_dt / denom).sqrt().clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL),
            false,
        )
    }

    /// The local vol function sampled on a `levels` x `times` grid:
    /// `grid[i][j]` is [`vol`](Self::vol)`(levels[i], times[j])` — the
    /// layout the plotting and document-writing code consumes.
    pub fn grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        levels
            .iter()
            .map(|&level| times.iter().map(|&t| self.vol(level, t)).collect())
            .collect()
    }

    /// This function sampled and repaired on a default grid covering
    /// `[0, horizon]` — the pricing engines' default (see
    /// [`LocalVolGrid`]). Levels are log-spaced [`GRID_STDEVS`] ATM
    /// standard deviations around the spot (plus the forward drift);
    /// times run from one week (or half the horizon) out to `horizon`.
    pub fn to_grid(&self, horizon: f64) -> LocalVolGrid {
        let t_max = horizon.max(1.0 / 52.0);
        let t_lo = (1.0 / 52.0_f64).min(0.5 * t_max);
        let times: Vec<f64> = (0..GRID_TIMES)
            .map(|j| t_lo + (t_max - t_lo) * j as f64 / (GRID_TIMES - 1) as f64)
            .collect();
        // the ATM vol only sets the level span's scale, so clamp it away
        // from degenerate surfaces
        let f = self.forward(t_max);
        let atm = self.implied(f, t_max).clamp(0.05, MAX_LOCAL_VOL);
        let half_width = GRID_STDEVS * atm * t_max.sqrt() + (f / self.spot).ln().abs();
        let levels: Vec<f64> = (0..GRID_LEVELS)
            .map(|i| {
                let x = -half_width + 2.0 * half_width * i as f64 / (GRID_LEVELS - 1) as f64;
                self.spot * x.exp()
            })
            .collect();
        LocalVolGrid::sample(self, levels, times)
    }
}

/// [`LocalVol`] sampled once on a levels x times grid. A query is one
/// bilinear lookup instead of the lazy version's several surface
/// interpolations per point — the difference between the definition and
/// what a Monte Carlo path can afford to call every step. Nodes where a
/// Dupire guard fired are bridged from the surrounding valid nodes
/// rather than keeping the pointwise implied-vol fallback, which is
/// discontinuous against its neighbours (a repair only a grid can do:
/// the lazy function evaluates each point in isolation). Queries beyond
/// an axis clamp to its edge, matching the implied surface's flat
/// wings. The lazy [`LocalVol`] remains the right form for diagnostics,
/// artifacts and SLV leverage calibration.
pub struct LocalVolGrid {
    levels: Vec<f64>,
    times: Vec<f64>,
    /// `vols[i][j]` is the local vol at `(levels[i], times[j])`.
    vols: Vec<Vec<f64>>,
    /// `valid[i][j]` is false where the Dupire guard fired during
    /// sampling — the node was repaired, or kept its fallback value.
    valid: Vec<Vec<bool>>,
    /// Nodes whose guard fired and were bridged from valid neighbours.
    repaired: usize,
}

impl LocalVolGrid {
    /// Sample `local_vol` at every node, then repair guarded nodes one
    /// time slice at a time (along the level axis, where the valid
    /// neighbours live). Axes must be strictly increasing and non-empty.
    pub fn sample(local_vol: &LocalVol, levels: Vec<f64>, times: Vec<f64>) -> Self {
        let mut vols: Vec<Vec<f64>> = Vec::with_capacity(levels.len());
        let mut valid: Vec<Vec<bool>> = Vec::with_capacity(levels.len());
        for &level in &levels {
            let mut row = Vec::with_capacity(times.len());
            let mut ok = Vec::with_capacity(times.len());
            for &t in &times {
                let (v, guarded) = local_vol.vol_checked(level, t);
                row.push(v);
                ok.push(!guarded);
            }
            vols.push(row);
            valid.push(ok);
        }
        let mut repaired = 0;
        for j in 0..times.len() {
            let mut slice: Vec<f64> = vols.iter().map(|row| row[j]).collect();
            let ok: Vec<bool> = valid.iter().map(|row| row[j]).collect();
            repaired += repair_slice(&mut slice, &ok);
            for (row, v) in vols.iter_mut().zip(slice) {
                row[j] = v;
            }
        }
        LocalVolGrid {
            levels,
            times,
            vols,
            valid,
            repaired,
        }
    }

    /// Local volatility at underlying level `level` and time `t`:
    /// bilinear between nodes, clamped to the grid edges outside.
    pub fn vol(&self, level: f64, t: f64) -> f64 {
        let (i0, i1, wi) = bracket(&self.levels, level);
        let (j0, j1, wj) = bracket(&self.times, t);
        let lo = self.vols[i0][j0] + wj * (self.vols[i0][j1] - self.vols[i0][j0]);
        let hi = self.vols[i1][j0] + wj * (self.vols[i1][j1] - self.vols[i1][j0]);
        lo + wi * (hi - lo)
    }

    /// [`vol`](Self::vol) plus whether any node the lookup interpolates
    /// between had its Dupire guard fire during sampling — the value
    /// leans on repaired (or fallback) nodes rather than pure Dupire
    /// values. Mirrors [`LocalVol::vol_checked`], so the artifact
    /// writers can instrument either form the same way.
    pub fn vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        let (i0, i1, _) = bracket(&self.levels, level);
        let (j0, j1, _) = bracket(&self.times, t);
        let guarded = !(self.valid[i0][j0]
            && self.valid[i0][j1]
            && self.valid[i1][j0]
            && self.valid[i1][j1]);
        (self.vol(level, t), guarded)
    }

    /// How many nodes the neighbour repair changed — the grid analogue
    /// of the lazy version's per-query guard flag.
    pub fn repaired_nodes(&self) -> usize {
        self.repaired
    }
}

/// Interval and weight locating `x` on the (strictly increasing) axis:
/// `xs[lo] <= x <= xs[hi]` after clamping to the ends.
fn bracket(xs: &[f64], x: f64) -> (usize, usize, f64) {
    if xs.len() < 2 {
        return (0, 0, 0.0);
    }
    let x = x.clamp(xs[0], xs[xs.len() - 1]);
    let hi = xs.partition_point(|&v| v < x).clamp(1, xs.len() - 1);
    let lo = hi - 1;
    (lo, hi, (x - xs[lo]) / (xs[hi] - xs[lo]))
}

/// Replace guarded entries by bridging from the nearest valid ones:
/// linear across an interior gap, flat past the first/last valid entry.
/// A slice with no valid entries keeps its fallback values, which are at
/// least internally consistent. Returns how many entries changed.
fn repair_slice(values: &mut [f64], valid: &[bool]) -> usize {
    let anchors: Vec<usize> = (0..values.len()).filter(|&i| valid[i]).collect();
    if anchors.is_empty() {
        return 0;
    }
    let mut repaired = 0;
    for i in 0..values.len() {
        if valid[i] {
            continue;
        }
        let next = anchors.partition_point(|&a| a < i);
        values[i] = if next == 0 {
            values[anchors[0]]
        } else if next == anchors.len() {
            values[anchors[anchors.len() - 1]]
        } else {
            let (a, b) = (anchors[next - 1], anchors[next]);
            let w = (i - a) as f64 / (b - a) as f64;
            values[a] + w * (values[b] - values[a])
        };
        repaired += 1;
    }
    repaired
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Tenor;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
    }

    fn flat_curve() -> YieldCurve {
        YieldCurve::flat(
            0.05,
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn flat_surface_gives_flat_local_vol() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        for level in [60.0, 90.0, 100.0, 130.0] {
            for t in [0.05, 0.5, 1.0, 2.0] {
                let v = lv.vol(level, t);
                assert!((v - 0.25).abs() < 1e-6, "level={level} t={t}: {v}");
            }
        }
    }

    #[test]
    fn grid_matches_pointwise_queries_in_level_time_layout() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let levels = [80.0, 100.0, 120.0];
        let times = [0.25, 1.0];
        let grid = lv.grid(&levels, &times);
        assert_eq!(grid.len(), levels.len());
        for (i, row) in grid.iter().enumerate() {
            assert_eq!(row.len(), times.len());
            for (j, v) in row.iter().enumerate() {
                assert_eq!(*v, lv.vol(levels[i], times[j]), "grid[{i}][{j}]");
            }
        }
    }

    #[test]
    fn term_structure_gives_forward_variance() {
        // sigma(0.5) = 20%, sigma(1.0) = 25% (flat in strike): between the
        // pillars the local variance is the forward variance
        // (w2 - w1)/(t2 - t1) = (0.0625 - 0.02)/0.5 = 0.085
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let expected = (0.085_f64).sqrt();
        let v = lv.vol(100.0, 0.75);
        assert!((v - expected).abs() < 1e-3, "{v} vs {expected}");
    }

    #[test]
    fn vol_shift_moves_local_vol() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.01);
        assert!((lv.vol(100.0, 1.0) - 0.26).abs() < 1e-6);
    }

    #[test]
    fn default_grid_matches_lazy_on_flat_surface() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let grid = lv.to_grid(2.0);
        for level in [70.0, 100.0, 133.7] {
            for t in [0.05, 0.5, 1.9] {
                let (v, guarded) = grid.vol_checked(level, t);
                assert!((v - 0.25).abs() < 1e-6, "level={level} t={t}: {v}");
                assert!(!guarded, "level={level} t={t}");
            }
        }
        assert_eq!(grid.repaired_nodes(), 0);
    }

    #[test]
    fn grid_node_matches_lazy_value() {
        // the forward-variance surface of the term-structure test: an
        // unguarded node must carry the exact lazy Dupire value
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let grid = LocalVolGrid::sample(&lv, vec![90.0, 100.0, 110.0], vec![0.4, 0.75]);
        assert!((grid.vol(100.0, 0.75) - lv.vol(100.0, 0.75)).abs() < 1e-12);
    }

    #[test]
    fn grid_clamps_beyond_axes() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let grid = lv.to_grid(1.0);
        for (level, t) in [(1e6, 50.0), (1e-6, 1e-9)] {
            let v = grid.vol(level, t);
            assert!((v - 0.25).abs() < 1e-6, "level={level} t={t}: {v}");
        }
    }

    #[test]
    fn repair_bridges_gaps_and_extends_flat() {
        // interior gap: linear bridge between the bounding valid values
        let mut v = [1.0, 9.0, 9.0, 4.0];
        assert_eq!(repair_slice(&mut v, &[true, false, false, true]), 2);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
        // one-sided: flat from the nearest valid value
        let mut v = [7.0, 2.0, 7.0];
        assert_eq!(repair_slice(&mut v, &[false, true, false]), 2);
        assert_eq!(v, [2.0, 2.0, 2.0]);
        // nothing valid: left untouched
        let mut v = [5.0, 6.0];
        assert_eq!(repair_slice(&mut v, &[false, false]), 0);
        assert_eq!(v, [5.0, 6.0]);
    }
}
