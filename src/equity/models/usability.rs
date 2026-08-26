//! Local-vol usability: "is this surface safe to price with?" as
//! numbers instead of a squint at the 3-D plot.
//!
//! [`usability_report`] measures three things for a local vol function
//! derived from an implied surface:
//!
//! 1. **Round-trip repricing** (the gold standard): interior vanillas
//!    are priced under the local vol model on the finite-difference
//!    engine, a Black-Scholes vol is re-implied from each price, and
//!    the gap to the surface's own implied vol is reported in **vol
//!    basis points**. Desk rule of thumb: interior errors under ~20-50
//!    vol bps mean the calibration is usable for vanillas.
//! 2. **Clamp and fallback fractions** over the published grid: the
//!    share of points pinned at the `[1%, 300%]` clamps, and the share
//!    where Dupire's guards fired and implied vol was silently
//!    substituted (see `vol_checked` /
//!    `local_vol_checked`).
//! 3. **The trusted region**: the strike/time box actually backed by
//!    quotes — the intersection of the per-expiry quoted strike ranges,
//!    from the first to the last pillar. Outside it, wings and
//!    extrapolation own the numbers.
//! 4. **Forward recovery** (optional, off by default because it costs a
//!    simulation): the martingale check of
//!    [`validation::martingale`](crate::validation::martingale) —
//!    simulate the local-vol dynamics and compare the mean underlying at
//!    each expiry against the chain's parity forwards, in z-scores.
//!    Round-trip repricing (1) grades the *strike* dimension; forward
//!    recovery grades the *drift and the wings*, where clamped or badly
//!    extrapolated local vol leaks probability mass.

use chrono::Days;
use serde::Serialize;

use crate::core::curves::{Compounding, Tenor, YieldCurve};
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::vols::{SmileCoordinate, VolInput, VolSurface};
use crate::equity::builder::EquityOptionBuilder;
use crate::equity::engines::blackscholes::implied_vol_from_price;
use crate::equity::smoothed_surface::{MAX_LOCAL_VOL, MIN_LOCAL_VOL};
use crate::equity::utils::{Engine, Model};
use crate::validation::martingale::{martingale_report, MartingaleConfig, MartingaleReport};

/// Sampling choices for [`usability_report`].
#[derive(Debug, Clone)]
pub struct UsabilityConfig {
    /// Outer cap on the round-trip moneyness band. Within it, strikes
    /// are drawn at up to +/- 1.5 ATM standard deviations
    /// (`1.5 sigma_atm sqrt(t)`), so short expiries are tested at
    /// comparable deltas instead of vega-dead far wings.
    pub moneyness: (f64, f64),
    /// At most this many pillar expiries are repriced (evenly sampled).
    pub max_expiries: usize,
    /// Strikes per repriced expiry, spread over the moneyness band.
    pub strikes_per_expiry: usize,
    /// Forward-recovery check, or `None` to skip it. Off by default: it
    /// runs a Monte Carlo simulation, which is orders of magnitude more
    /// expensive than the rest of the report.
    pub martingale: Option<MartingaleSpec>,
}

/// What the optional forward-recovery check needs: the target forwards
/// to hit, and how hard to simulate.
///
/// The targets are supplied rather than derived, and that is the whole
/// point — see
/// [`martingale_report`](crate::validation::martingale::martingale_report).
/// Fed the chain's **parity forwards**, the check measures the local-vol
/// dynamics against where the market actually puts the forward, so it
/// also prices in the pipeline's zero-dividend assumption; fed
/// curve-grown forwards, it validates the scheme alone.
#[derive(Debug, Clone)]
pub struct MartingaleSpec {
    /// `(expiry time, target forward)` pairs.
    pub targets: Vec<(f64, f64)>,
    pub config: MartingaleConfig,
}

impl Default for UsabilityConfig {
    fn default() -> Self {
        UsabilityConfig {
            moneyness: (0.85, 1.15),
            max_expiries: 8,
            strikes_per_expiry: 3,
            martingale: None,
        }
    }
}

/// Round-trip repricing statistics, in implied-vol basis points.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RoundTrip {
    pub points: usize,
    /// Vanillas that failed to build, price, or re-imply.
    pub failures: usize,
    pub mean_vol_bps: f64,
    pub max_vol_bps: f64,
}

/// The strike/time box backed by quotes on every pillar.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TrustedRegion {
    pub strike_lo: f64,
    pub strike_hi: f64,
    pub t_lo: f64,
    pub t_hi: f64,
}

/// The usability verdict attached to every local-vol document.
#[derive(Debug, Clone, Default, Serialize)]
pub struct UsabilityReport {
    pub roundtrip: RoundTrip,
    /// Share of published grid points pinned at the [1%, 300%] clamps.
    pub clamped_fraction: f64,
    /// Share of published grid points where Dupire's guards substituted
    /// the implied vol.
    pub fallback_fraction: f64,
    pub trusted_region: TrustedRegion,
    /// The desk rule of thumb applied: mean round-trip error <= 20 vol
    /// bps, max <= 50, and under 1% of the grid clamped or guarded.
    ///
    /// Deliberately **not** a function of [`Self::martingale`]. The two
    /// answer different questions, and against parity forwards the
    /// martingale check also carries the pipeline's zero-dividend
    /// assumption: a dividend-paying name can miss its forwards by many
    /// standard errors while its local vol reprices vanillas perfectly.
    /// Folding that into one verdict would blame the calibration for a
    /// modelling choice.
    pub within_desk_tolerance: bool,
    /// Forward recovery, when
    /// [`UsabilityConfig::martingale`] asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub martingale: Option<MartingaleReport>,
}

/// Measure a local vol function against the implied `surface` it was
/// calibrated from. `local_vol_checked` is the instrumented sampler
/// (`(vol, guard_fired)`); `levels`/`times` are the published grid the
/// clamp/fallback fractions describe; `curve` supplies the rate for the
/// round-trip vanillas and, when
/// [`UsabilityConfig::martingale`] is set, the drift for the
/// forward-recovery simulation (zero dividends throughout, matching the
/// build pipeline).
pub fn usability_report(
    surface: &VolSurface,
    local_vol_checked: &dyn Fn(f64, f64) -> (f64, bool),
    levels: &[f64],
    times: &[f64],
    curve: &YieldCurve,
    spot: f64,
    config: &UsabilityConfig,
) -> UsabilityReport {
    let mut report = UsabilityReport {
        trusted_region: trusted_region(surface, spot),
        ..UsabilityReport::default()
    };

    // clamp / fallback fractions over the published grid
    let mut clamped = 0usize;
    let mut fallback = 0usize;
    let mut total = 0usize;
    for &level in levels {
        for &t in times {
            let (vol, guarded) = local_vol_checked(level, t);
            total += 1;
            // detect values pinned at the shared clamp bounds (with a 1%
            // margin for interpolation between a clamped and a free node)
            if vol <= MIN_LOCAL_VOL * 1.01 || vol >= MAX_LOCAL_VOL * 0.999 {
                clamped += 1;
            }
            if guarded {
                fallback += 1;
            }
        }
    }
    if total > 0 {
        report.clamped_fraction = clamped as f64 / total as f64;
        report.fallback_fraction = fallback as f64 / total as f64;
    }

    // round-trip: reprice interior vanillas under the local vol model
    let pillar_times: Vec<f64> = surface.expiry_times().to_vec();
    let selected: Vec<f64> = if pillar_times.len() <= config.max_expiries {
        pillar_times
    } else {
        let step = (pillar_times.len() - 1) as f64 / (config.max_expiries - 1) as f64;
        (0..config.max_expiries)
            .map(|i| pillar_times[(i as f64 * step).round() as usize])
            .collect()
    };
    let reference = surface.reference_date();
    let mut errors: Vec<f64> = Vec::new();
    for &t in &selected {
        let maturity = reference + Days::new((t * 365.0).round().max(3.0) as u64);
        let t_eff = crate::equity::conventions::year_fraction(reference, maturity);
        let rate = curve.zero_rate_with(t_eff, Compounding::Continuous);
        let forward = spot * (rate * t_eff).exp();
        let n = config.strikes_per_expiry.max(1);
        // delta-comparable band: +/- 1.5 ATM standard deviations,
        // capped by the configured moneyness box
        let atm_vol = surface.vol(forward, forward, t_eff);
        let cap = (config.moneyness.1.ln()).min(-config.moneyness.0.ln());
        let half_width = (1.5 * atm_vol * t_eff.sqrt()).min(cap.abs());
        for i in 0..n {
            let k = -half_width + 2.0 * half_width * i as f64 / (n - 1).max(1) as f64;
            let strike = forward * k.exp();
            if strike < report.trusted_region.strike_lo || strike > report.trusted_region.strike_hi
            {
                continue;
            }
            let side = if strike >= forward {
                PutOrCall::Call
            } else {
                PutOrCall::Put
            };
            let implied = surface.vol(strike, forward, t_eff);
            let repriced = EquityOptionBuilder::new()
                .spot(spot)
                .strike(strike)
                .vol_surface(surface.clone())
                .flat_rate(rate)
                .valuation_date(reference)
                .maturity_date(maturity)
                .vanilla(side)
                .engine(Engine::FiniteDifference)
                .model(Model::LocalVol)
                .build()
                .and_then(|option| option.price())
                .map(|result| result.pv)
                .and_then(|pv| implied_vol_from_price(spot, strike, rate, 0.0, t_eff, pv, side));
            match repriced {
                Ok(re_implied) => errors.push((re_implied - implied).abs() * 1e4),
                Err(e) => {
                    log::debug!("usability round trip failed at K={strike:.2} t={t_eff:.3}: {e}");
                    report.roundtrip.failures += 1;
                }
            }
        }
    }
    report.roundtrip.points = errors.len();
    if !errors.is_empty() {
        report.roundtrip.mean_vol_bps = errors.iter().sum::<f64>() / errors.len() as f64;
        report.roundtrip.max_vol_bps = errors.iter().copied().fold(0.0, f64::max);
    }

    report.within_desk_tolerance = report.roundtrip.points > 0
        && report.roundtrip.mean_vol_bps <= 20.0
        && report.roundtrip.max_vol_bps <= 50.0
        && report.clamped_fraction + report.fallback_fraction < 0.01;

    // the expensive one, last and only on request: simulate the dynamics
    // this local vol defines and see whether they land on the forwards
    if let Some(spec) = &config.martingale {
        report.martingale = Some(martingale_report(
            &|level, t| local_vol_checked(level, t).0,
            curve,
            spot,
            &spec.targets,
            &spec.config,
        ));
    }
    report
}

/// The strike/time box quoted on every pillar: the intersection of the
/// per-expiry strike ranges (falling back to the union when disjoint
/// expiries share no strikes), from the first to the last pillar time.
fn trusted_region(surface: &VolSurface, spot: f64) -> TrustedRegion {
    let VolInput::StrikeSmiles {
        expiries,
        smiles,
        coordinate,
        ..
    } = surface.to_input()
    else {
        return TrustedRegion {
            strike_lo: spot * 0.5,
            strike_hi: spot * 1.5,
            t_lo: 0.0,
            t_hi: f64::MAX,
        };
    };
    let times: Vec<f64> = expiries
        .iter()
        .map(|tenor| match tenor {
            Tenor::YearFraction(t) => *t,
            Tenor::Date(_) => 0.0,
        })
        .collect();
    let ranges: Vec<(f64, f64)> = smiles
        .iter()
        .map(|smile| {
            smile
                .iter()
                .fold((f64::MAX, f64::MIN), |(lo, hi), &(x, _)| {
                    // coordinate-to-strike with the spot as the scale for
                    // relative coordinates: a bounds box, not a repricing
                    let strike = match coordinate {
                        SmileCoordinate::Strike => x,
                        SmileCoordinate::Moneyness => x * spot,
                        SmileCoordinate::LogMoneyness => x.exp() * spot,
                    };
                    (lo.min(strike), hi.max(strike))
                })
        })
        .collect();
    let intersection = ranges.iter().fold((f64::MIN, f64::MAX), |(lo, hi), r| {
        (lo.max(r.0), hi.min(r.1))
    });
    let (strike_lo, strike_hi) = if intersection.0 < intersection.1 {
        intersection
    } else {
        ranges.iter().fold((f64::MAX, f64::MIN), |(lo, hi), r| {
            (lo.min(r.0), hi.max(r.1))
        })
    };
    TrustedRegion {
        strike_lo,
        strike_hi,
        t_lo: times.iter().copied().fold(f64::MAX, f64::min),
        t_hi: times.iter().copied().fold(0.0, f64::max),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::daycount::DayCountConvention;
    use crate::equity::local_vol::LocalVol;
    use crate::validation::martingale::MartingaleConfig;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, 11).unwrap()
    }

    fn flat_curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn martingale_leg_is_opt_in_and_recovers_curve_forwards() {
        // a flat surface under a flat curve: the local vol is constant,
        // so the simulated underlying must land on the curve-grown
        // forwards. Fed those as targets the check isolates the scheme
        // (no modelling gap to measure), which is exactly the case where
        // it should pass.
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve(0.03);
        let spot = 100.0;
        let local_vol = LocalVol::new(&surface, &curve, spot, 0.0, 0.0);
        let levels: Vec<f64> = (0..15).map(|i| 80.0 + 3.0 * i as f64).collect();
        let times: Vec<f64> = (0..8).map(|i| 0.15 + 0.1 * i as f64).collect();
        let sampler = |level: f64, t: f64| local_vol.vol_checked(level, t);

        // off by default: the simulation is expensive, so it must be asked for
        let plain = usability_report(
            &surface,
            &sampler,
            &levels,
            &times,
            &curve,
            spot,
            &UsabilityConfig::default(),
        );
        assert!(plain.martingale.is_none(), "must be opt-in");

        // curve-grown targets under zero dividends
        let targets: Vec<(f64, f64)> = [0.25, 0.5, 1.0]
            .iter()
            .map(|&t| (t, spot * (0.03_f64 * t).exp()))
            .collect();
        let checked = usability_report(
            &surface,
            &sampler,
            &levels,
            &times,
            &curve,
            spot,
            &UsabilityConfig {
                martingale: Some(MartingaleSpec {
                    targets: targets.clone(),
                    config: MartingaleConfig {
                        paths: 4096,
                        ..Default::default()
                    },
                }),
                ..UsabilityConfig::default()
            },
        );
        let m = checked.martingale.expect("requested, so present");
        assert_eq!(m.checks.len(), targets.len());
        // the z-score is the assertion, not a hand-picked bp tolerance:
        // the recovery error is a Monte Carlo estimate, so any absolute
        // bound is either vacuous or flaky as the path count changes
        assert!(
            m.within_threshold,
            "max |z| = {:.1} (worst error {:.1} bp)",
            m.max_abs_z,
            m.worst_relative_error * 1e4
        );
        for c in &m.checks {
            assert!(c.standard_error > 0.0, "t={} has no dispersion", c.t);
        }
        // the round-trip verdict is deliberately independent of it
        assert_eq!(plain.within_desk_tolerance, checked.within_desk_tolerance);
    }

    #[test]
    fn clean_surface_passes_the_desk_tolerance() {
        // gentle skew, healthy term structure — Dupire behaves
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[
                (0..13)
                    .map(|i| {
                        (
                            70.0 + 5.0 * i as f64,
                            0.25 - 0.0005 * (5.0 * i as f64 - 30.0),
                        )
                    })
                    .collect(),
                (0..13)
                    .map(|i| {
                        (
                            70.0 + 5.0 * i as f64,
                            0.28 - 0.0005 * (5.0 * i as f64 - 30.0),
                        )
                    })
                    .collect(),
            ],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let curve = flat_curve(0.03);
        let local_vol = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let levels: Vec<f64> = (0..20).map(|i| 75.0 + 2.5 * i as f64).collect();
        let times: Vec<f64> = (0..10).map(|i| 0.1 + 0.09 * i as f64).collect();
        let report = usability_report(
            &surface,
            &|level, t| local_vol.vol_checked(level, t),
            &levels,
            &times,
            &curve,
            100.0,
            &UsabilityConfig::default(),
        );
        assert!(report.roundtrip.points >= 4, "{report:?}");
        assert_eq!(report.roundtrip.failures, 0, "{report:?}");
        assert!(report.roundtrip.mean_vol_bps < 20.0, "{report:?}");
        assert!(report.clamped_fraction == 0.0, "{report:?}");
        assert!(report.within_desk_tolerance, "{report:?}");
        // trusted region = the quoted box
        assert_eq!(report.trusted_region.strike_lo, 70.0);
        assert_eq!(report.trusted_region.strike_hi, 130.0);
        assert!((report.trusted_region.t_hi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn guarded_points_are_counted_as_fallback() {
        // falling total variance: the Dupire guard fires everywhere
        // between the pillars
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.40)], vec![(100.0, 0.20)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let curve = flat_curve(0.0);
        let local_vol = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let (_, guarded) = local_vol.vol_checked(100.0, 0.75);
        assert!(guarded, "calendar violation must trip the guard");
        let report = usability_report(
            &surface,
            &|level, t| local_vol.vol_checked(level, t),
            &[90.0, 100.0, 110.0],
            &[0.6, 0.75, 0.9],
            &curve,
            100.0,
            &UsabilityConfig::default(),
        );
        assert!(report.fallback_fraction > 0.5, "{report:?}");
        assert!(!report.within_desk_tolerance);
    }
}
