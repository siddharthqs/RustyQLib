//! Minimal-change static-arbitrage repair for implied vol surfaces.
//!
//! Two deterministic passes, iterated until the surface passes its own
//! [`diagnostics`](VolSurface::diagnostics):
//!
//! 1. **Butterfly** (within each expiry): undiscounted call prices at
//!    the pillar strikes are projected onto their **lower convex hull**
//!    — only quotes priced above the hull move, and only *down* to the
//!    no-arbitrage boundary — then implied vols are re-solved from the
//!    repaired prices.
//! 2. **Calendar** (across expiries): a forward sweep in total variance
//!    lifts any pillar whose `sigma^2 t` falls below the previous
//!    expiry's total variance at the same forward moneyness, up to
//!    exactly that bound.
//!
//! This is the pragmatic desk repair: transparent, solver-free, and
//! *minimal* — points that carry no arbitrage are untouched, so the
//! cleaned surface stays as close to the market as the constraints
//! allow. Every change is counted in the [`RepairReport`]. The repair
//! operates at the quoted pillars (like the diagnostics); for a smooth
//! globally arbitrage-free surface, fit a parameterization such as
//! SVI/SSVI ([`svi`](crate::equity::models::svi)) instead — that is a
//! smoother, which moves every point, where this is a repair, which
//! moves only the indefensible ones.

use serde::Serialize;

use crate::core::curves::Tenor;
use crate::core::errors::RustyQLibError;
use crate::core::interpolation::interp_pairs;
use crate::core::trade::PutOrCall;
use crate::core::vols::{SmileCoordinate, VolInput, VolSurface};
use crate::equity::engines::blackscholes::{bs_price, implied_vol_from_price};

/// What [`repair_arbitrage`] did to the surface.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RepairReport {
    /// Pillar vols lowered by the convex-hull price projection.
    pub butterfly_adjustments: usize,
    /// Pillar vols raised by the total-variance sweep.
    pub calendar_adjustments: usize,
    /// Pillars dropped because their repaired price no longer implies a
    /// vol (pinned to intrinsic).
    pub dropped_points: usize,
    /// Largest absolute vol change across all adjustments.
    pub max_vol_change: f64,
    /// Butterfly+calendar passes run.
    pub iterations: usize,
    /// Whether the result passes the diagnostics (it should; `false`
    /// means the passes stopped converging and the caller is warned).
    pub clean: bool,
}

/// Passes to attempt before giving up (each pass fixes butterflies then
/// calendars; a calendar lift can re-bend a smile, hence the loop).
const MAX_PASSES: usize = 8;
/// Prices above the hull by less than this are left alone (numerical
/// noise, not arbitrage; the diagnostics tolerance is looser).
const PRICE_TOL: f64 = 1e-12;

/// Repair static arbitrage in `surface` with the minimal-change hull /
/// variance-sweep passes (see the module docs). `forward` maps expiry
/// time to the underlying's forward, exactly as for
/// [`VolSurface::diagnostics`]. Returns the cleaned surface — on
/// absolute strikes, same reference date and day count — with the
/// repair report. A surface that is already clean comes back unchanged
/// with a zeroed report.
pub fn repair_arbitrage(
    surface: &VolSurface,
    forward: impl Fn(f64) -> f64,
) -> Result<(VolSurface, RepairReport), RustyQLibError> {
    let VolInput::StrikeSmiles {
        expiries,
        smiles,
        coordinate,
        ..
    } = surface.to_input()
    else {
        // flat surfaces are arbitrage-free by construction
        let report = RepairReport {
            clean: true,
            ..RepairReport::default()
        };
        return Ok((surface.clone(), report));
    };
    let mut times: Vec<f64> = expiries
        .iter()
        .map(|tenor| match tenor {
            Tenor::YearFraction(t) => *t,
            Tenor::Date(_) => 0.0, // to_input never emits dates
        })
        .collect();

    // work on absolute strikes regardless of the stored coordinate
    let mut slices: Vec<Vec<(f64, f64)>> = times
        .iter()
        .zip(&smiles)
        .map(|(&t, smile)| {
            let f = forward(t);
            smile
                .iter()
                .map(|&(x, vol)| {
                    let strike = match coordinate {
                        SmileCoordinate::Strike => x,
                        SmileCoordinate::Moneyness => x * f,
                        SmileCoordinate::LogMoneyness => x.exp() * f,
                    };
                    (strike, vol)
                })
                .collect()
        })
        .collect();

    let mut report = RepairReport::default();
    let mut cleaned = surface.clone();
    for pass in 1..=MAX_PASSES {
        report.iterations = pass;
        let mut changed = false;

        // pass 1: butterfly, per expiry
        for (&t, slice) in times.iter().zip(slices.iter_mut()) {
            changed |= convexify_slice(slice, forward(t), t, &mut report)?;
        }
        // dropped pillars can empty a slice; keep times aligned
        let kept: Vec<bool> = slices.iter().map(|s| !s.is_empty()).collect();
        if kept.iter().any(|k| !k) {
            slices.retain(|s| !s.is_empty());
            times = times
                .iter()
                .zip(&kept)
                .filter(|(_, &k)| k)
                .map(|(&t, _)| t)
                .collect();
        }
        if slices.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "surface repair",
                "every pillar was dropped — the surface cannot be repaired",
            ));
        }

        // pass 2: calendar, forward sweep in total variance. The
        // diagnostics check the union of both slices' moneyness points,
        // and a piecewise-linear smile can dip below the earlier
        // variance *between* its own knots — so the sweep checks the
        // same union, lifting existing pillars and inserting a knot at
        // the variance floor where none exists.
        for i in 1..slices.len() {
            let (t0, t1) = (times[i - 1], times[i]);
            let (f0, f1) = (forward(t0), forward(t1));
            let previous = slices[i - 1].clone();
            let mut moneyness: Vec<f64> = previous
                .iter()
                .map(|&(k, _)| k / f0)
                .chain(slices[i].iter().map(|&(k, _)| k / f1))
                .collect();
            moneyness.sort_by(|a, b| a.partial_cmp(b).unwrap());
            moneyness.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
            let current = slices[i].clone();
            for m in moneyness {
                let prev_vol = smile_vol(&previous, m * f0);
                let floor = prev_vol * prev_vol * t0;
                let strike = m * f1;
                let vol_here = smile_vol(&current, strike);
                if vol_here * vol_here * t1 < floor {
                    let lifted = (floor / t1).sqrt();
                    report.max_vol_change = report.max_vol_change.max(lifted - vol_here);
                    report.calendar_adjustments += 1;
                    upsert(&mut slices[i], strike, lifted);
                    changed = true;
                }
            }
        }

        let tenors: Vec<Tenor> = times.iter().map(|&t| Tenor::YearFraction(t)).collect();
        cleaned = VolSurface::from_strike_smiles(
            &tenors,
            &slices,
            surface.reference_date(),
            surface.day_count(),
        )?;
        report.clean = cleaned.diagnostics(&forward).is_clean();
        if report.clean || !changed {
            break;
        }
    }
    Ok((cleaned, report))
}

/// Replace the pillar at `strike` (within rounding) or insert a new one
/// keeping the slice sorted.
fn upsert(points: &mut Vec<(f64, f64)>, strike: f64, vol: f64) {
    if let Some(position) = points.iter().position(|&(k, _)| (k - strike).abs() < 1e-9) {
        points[position].1 = vol;
    } else {
        let index = points.partition_point(|&(k, _)| k < strike);
        points.insert(index, (strike, vol));
    }
}

/// Linear-in-vol smile lookup with flat wings — the same behavior the
/// surface itself has on a strike coordinate, so the sweep and the
/// diagnostics agree exactly.
fn smile_vol(points: &[(f64, f64)], strike: f64) -> f64 {
    if points.len() == 1 {
        return points[0].1;
    }
    interp_pairs(points, strike)
}

/// Project one expiry's undiscounted call prices onto their lower
/// convex hull and re-imply the vols of the points that moved. Returns
/// whether anything changed; unpriceable repaired quotes (pinned to
/// intrinsic) are dropped and counted.
fn convexify_slice(
    slice: &mut Vec<(f64, f64)>,
    f: f64,
    t: f64,
    report: &mut RepairReport,
) -> Result<bool, RustyQLibError> {
    if slice.len() < 3 {
        return Ok(false);
    }
    let prices: Vec<(f64, f64)> = slice
        .iter()
        .map(|&(k, vol)| (k, bs_price(f, k, 0.0, 0.0, vol, t, PutOrCall::Call)))
        .collect();

    // Andrew's monotone chain, lower hull (strikes already ascending)
    let mut hull: Vec<(f64, f64)> = Vec::with_capacity(prices.len());
    for &p in &prices {
        while hull.len() >= 2 {
            let (o, a) = (hull[hull.len() - 2], hull[hull.len() - 1]);
            let cross = (a.0 - o.0) * (p.1 - o.1) - (a.1 - o.1) * (p.0 - o.0);
            if cross <= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }

    let mut changed = false;
    let mut repaired: Vec<(f64, f64)> = Vec::with_capacity(slice.len());
    for (&(k, vol), &(_, price)) in slice.iter().zip(&prices) {
        let hull_price = interp_pairs(&hull, k);
        if price - hull_price <= PRICE_TOL {
            repaired.push((k, vol));
            continue;
        }
        changed = true;
        match implied_vol_from_price(f, k, 0.0, 0.0, t, hull_price, PutOrCall::Call) {
            Ok(new_vol) => {
                report.butterfly_adjustments += 1;
                report.max_vol_change = report.max_vol_change.max((vol - new_vol).abs());
                repaired.push((k, new_vol));
            }
            Err(_) => {
                // the hull pinned this quote to intrinsic: no vol exists
                report.dropped_points += 1;
            }
        }
    }
    *slice = repaired;
    Ok(changed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, 5).unwrap()
    }

    fn strike_surface(expiries: &[f64], smiles: &[Vec<(f64, f64)>]) -> VolSurface {
        let tenors: Vec<Tenor> = expiries.iter().map(|&t| Tenor::YearFraction(t)).collect();
        VolSurface::from_strike_smiles(&tenors, smiles, asof(), DayCountConvention::Act365)
            .unwrap()
    }

    #[test]
    fn clean_surfaces_come_back_unchanged() {
        let surface = strike_surface(
            &[0.5, 1.0],
            &[
                vec![(90.0, 0.22), (100.0, 0.20), (110.0, 0.19)],
                vec![(90.0, 0.24), (100.0, 0.22), (110.0, 0.21)],
            ],
        );
        let (cleaned, report) = repair_arbitrage(&surface, |_| 100.0).unwrap();
        assert!(report.clean);
        assert_eq!(report.butterfly_adjustments, 0);
        assert_eq!(report.calendar_adjustments, 0);
        assert_eq!(report.max_vol_change, 0.0);
        for (k, t) in [(90.0, 0.5), (100.0, 1.0), (95.0, 0.75)] {
            assert_eq!(cleaned.vol(k, 100.0, t), surface.vol(k, 100.0, t));
        }
        // flat surfaces short-circuit
        let flat = VolSurface::flat(0.2, asof(), DayCountConvention::Act365).unwrap();
        let (_, report) = repair_arbitrage(&flat, |_| 100.0).unwrap();
        assert!(report.clean);
    }

    #[test]
    fn butterfly_spike_is_pulled_down_to_the_hull() {
        let spiked = strike_surface(&[1.0], &[vec![(90.0, 0.20), (100.0, 0.50), (110.0, 0.20)]]);
        assert!(!spiked.diagnostics(|_| 100.0).is_clean());
        let (cleaned, report) = repair_arbitrage(&spiked, |_| 100.0).unwrap();
        assert!(report.clean, "{report:?}");
        assert!(cleaned.diagnostics(|_| 100.0).is_clean());
        assert_eq!(report.butterfly_adjustments, 1);
        // the spike came down; its neighbors did not move
        assert!(cleaned.vol(100.0, 100.0, 1.0) < 0.30);
        assert_eq!(cleaned.vol(90.0, 100.0, 1.0), 0.20);
        assert_eq!(cleaned.vol(110.0, 100.0, 1.0), 0.20);
        assert!(report.max_vol_change > 0.15);
    }

    #[test]
    fn falling_total_variance_is_lifted_to_the_floor() {
        let falling = strike_surface(&[0.5, 1.0], &[vec![(100.0, 0.40)], vec![(100.0, 0.20)]]);
        let (cleaned, report) = repair_arbitrage(&falling, |_| 100.0).unwrap();
        assert!(report.clean, "{report:?}");
        assert_eq!(report.calendar_adjustments, 1);
        assert_eq!(report.butterfly_adjustments, 0);
        // lifted to sqrt(0.4^2 * 0.5 / 1.0), the exact variance floor
        let expected = (0.4_f64 * 0.4 * 0.5).sqrt();
        assert!((cleaned.vol(100.0, 100.0, 1.0) - expected).abs() < 1e-12);
        // the earlier pillar is untouched
        assert_eq!(cleaned.vol(100.0, 100.0, 0.5), 0.40);
    }

    #[test]
    fn combined_violations_converge_to_a_clean_surface() {
        // a spiked front smile and a collapsing back smile together
        let messy = strike_surface(
            &[0.5, 1.0],
            &[
                vec![(90.0, 0.20), (100.0, 0.55), (110.0, 0.20)],
                vec![(90.0, 0.12), (100.0, 0.10), (110.0, 0.11)],
            ],
        );
        assert!(!messy.diagnostics(|_| 100.0).is_clean());
        let (cleaned, report) = repair_arbitrage(&messy, |_| 100.0).unwrap();
        assert!(report.clean, "{report:?}");
        assert!(cleaned.diagnostics(|_| 100.0).is_clean());
        assert!(report.butterfly_adjustments >= 1);
        assert!(report.calendar_adjustments >= 3, "{report:?}");
        assert!(report.iterations <= MAX_PASSES);
    }
}
