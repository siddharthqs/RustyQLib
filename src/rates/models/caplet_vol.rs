//! Caplet volatilities stripped from cap quotes.
//!
//! The market quotes a cap by one **flat** vol for the whole strip;
//! the caplets inside it do not share that vol. Stripping recovers a
//! term structure of caplet vols, piecewise constant in the caplet's
//! fixing time, from a ladder of caps of increasing maturity at one
//! strike: the shortest cap's caplets all take its flat vol; each
//! longer cap adds the caplets beyond the previous maturity, and their
//! common vol is the one that makes the cap reprice given the caplets
//! already stripped (a 1-D root, bisection). Any product priced caplet
//! by caplet then uses the stripped curve
//! ([`CapFloor::npv_with_caplet_vols`]).
//!
//! [`CapFloor::npv_with_caplet_vols`]: crate::rates::contracts::cap_floor::CapFloor::npv_with_caplet_vols

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::rates::contracts::cap_floor::CapFloor;
use crate::rates::engines::black::{RateVol, RateVolKind};

const FIELD: &str = "caplet vols";

/// Caplet vols, piecewise constant in fixing time: `vols[i]` applies to
/// fixings in `(ends[i-1], ends[i]]`, the last beyond `ends[-1]`.
#[derive(Debug, Clone)]
pub struct CapletVolCurve {
    ends: Vec<f64>,
    vols: Vec<f64>,
    kind: RateVolKind,
}

impl CapletVolCurve {
    /// A curve from interval ends (ascending, positive) and one vol per
    /// interval.
    pub fn new(ends: Vec<f64>, vols: Vec<f64>, kind: RateVolKind) -> Result<Self, RustyQLibError> {
        if ends.is_empty() || ends.len() != vols.len() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need one vol per interval end, at least one interval",
            ));
        }
        if ends[0] <= 0.0 || ends.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("interval ends must be positive and increasing, got {ends:?}"),
            ));
        }
        if vols.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "vols must be finite and non-negative",
            ));
        }
        Ok(CapletVolCurve { ends, vols, kind })
    }

    pub fn ends(&self) -> &[f64] {
        &self.ends
    }

    pub fn vols(&self) -> &[f64] {
        &self.vols
    }

    pub fn kind(&self) -> RateVolKind {
        self.kind
    }

    /// The vol of a caplet fixing at `fixing_time`.
    pub fn vol(&self, fixing_time: f64) -> f64 {
        let i = self
            .ends
            .iter()
            .position(|&end| fixing_time <= end + 1e-12)
            .unwrap_or(self.ends.len() - 1);
        self.vols[i]
    }

    /// The quote of a caplet fixing at `fixing_time`.
    pub fn quote(&self, fixing_time: f64) -> RateVol {
        self.kind.with_vol(self.vol(fixing_time))
    }
}

/// One cap of the ladder and its flat vol.
#[derive(Debug, Clone)]
pub struct CapQuote {
    pub cap: CapFloor,
    pub flat_vol: f64,
}

/// Strip caplet vols from a ladder of caps at one strike with a shared
/// schedule (same effective date, frequency, day count and conventions;
/// maturities increasing), each quoted at a flat vol of `kind`. The
/// stripped curve prices every cap of the ladder back to its flat-vol
/// price.
pub fn strip_caplet_vols(
    quotes: &[CapQuote],
    discount: &YieldCurve,
    forecast: &YieldCurve,
    kind: RateVolKind,
) -> Result<CapletVolCurve, RustyQLibError> {
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(FIELD, "no cap quotes"));
    }
    let mut order: Vec<usize> = (0..quotes.len()).collect();
    order.sort_by_key(|&i| quotes[i].cap.maturity_date);
    let first = &quotes[order[0]].cap;
    for &i in &order[1..] {
        let cap = &quotes[i].cap;
        let same = cap.strike == first.strike
            && cap.cap_or_floor == first.cap_or_floor
            && cap.effective_date == first.effective_date
            && cap.frequency == first.frequency
            && cap.day_count == first.day_count
            && cap.convention == first.convention;
        if !same {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the cap maturing {} does not share the ladder's strike and schedule",
                    cap.maturity_date
                ),
            ));
        }
    }
    let mut ends: Vec<f64> = Vec::with_capacity(quotes.len());
    let mut vols: Vec<f64> = Vec::with_capacity(quotes.len());
    for (k, &i) in order.iter().enumerate() {
        let quote = &quotes[i];
        if !(quote.flat_vol.is_finite() && quote.flat_vol >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("flat vol must be non-negative, got {}", quote.flat_vol),
            ));
        }
        let target = quote
            .cap
            .npv_black(discount, forecast, kind.with_vol(quote.flat_vol))?;
        let previous_end = ends.last().copied().unwrap_or(0.0);
        // the cap's last fixing time closes this interval
        let fixings = quote
            .cap
            .caplet_black_values(discount, forecast, |_| kind.with_vol(0.0))?;
        let Some(&(last_fixing, _)) = fixings.last() else {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the cap maturing {} has no unfixed caplets",
                    quote.cap.maturity_date
                ),
            ));
        };
        if last_fixing <= previous_end {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the cap maturing {} adds no caplets beyond the previous cap",
                    quote.cap.maturity_date
                ),
            ));
        }
        // price the cap with the stripped vols so far and a trial vol on
        // the new interval
        let price_at = |trial: f64| -> Result<f64, RustyQLibError> {
            let mut trial_ends = ends.clone();
            trial_ends.push(last_fixing);
            let mut trial_vols = vols.clone();
            trial_vols.push(trial);
            let curve = CapletVolCurve::new(trial_ends, trial_vols, kind)?;
            quote.cap.npv_with_caplet_vols(discount, forecast, &curve)
        };
        let hi = match kind {
            RateVolKind::Normal => 10.0 * (quote.flat_vol + 0.01),
            _ => 10.0,
        };
        let (lo_err, hi_err) = (price_at(0.0)? - target, price_at(hi)? - target);
        if lo_err > 1e-12 || hi_err < 0.0 {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: k,
                residual: lo_err.abs().min(hi_err.abs()),
                reason: format!(
                    "the cap maturing {} cannot be repriced with a caplet vol in [0, {hi}] \
                     given the shorter caps (its flat vol is inconsistent with the ladder)",
                    quote.cap.maturity_date
                ),
            });
        }
        let root = Solver1d::new(1e-14, 200).bisection(
            |v| price_at(v).unwrap_or(f64::NAN) - target,
            0.0,
            hi,
        )?;
        ends.push(last_fixing);
        vols.push(root.x);
    }
    CapletVolCurve::new(ends, vols, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::contracts::cap_floor::CapOrFloor;
    use chrono::NaiveDate;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045],
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    /// Spot-starting USD caps at 4.5% maturing in 1, 2, 3 and 5 years.
    fn ladder(strike: f64) -> Vec<CapFloor> {
        [
            d(2027, 8, 13),
            d(2028, 8, 14),
            d(2029, 8, 13),
            d(2031, 8, 13),
        ]
        .iter()
        .map(|&maturity| {
            CapFloor::usd_standard(1.0, strike, CapOrFloor::Cap, d(2026, 8, 13), maturity).unwrap()
        })
        .collect()
    }

    #[test]
    fn stripping_recovers_a_generating_caplet_term_structure() {
        // caplet vols by interval -> flat cap vols -> strip -> the same
        // caplet vols, and every cap of the ladder reprices
        let curve = market_curve();
        let caps = ladder(0.045);
        let ends: Vec<f64> = caps
            .iter()
            .map(|c| {
                c.caplet_black_values(&curve, &curve, |_| RateVol::Normal(0.0))
                    .unwrap()
                    .last()
                    .unwrap()
                    .0
            })
            .collect();
        let truth = CapletVolCurve::new(
            ends.clone(),
            vec![0.0070, 0.0085, 0.0090, 0.0080],
            RateVolKind::Normal,
        )
        .unwrap();
        let quotes: Vec<CapQuote> = caps
            .iter()
            .map(|cap| {
                let price = cap.npv_with_caplet_vols(&curve, &curve, &truth).unwrap();
                CapQuote {
                    cap: cap.clone(),
                    flat_vol: cap.implied_flat_normal_vol(&curve, &curve, price).unwrap(),
                }
            })
            .collect();
        // flat vols are averages of the caplet vols, so they lie between
        assert!(quotes[0].flat_vol > 0.0069 && quotes[0].flat_vol < 0.0071);
        assert!(quotes[3].flat_vol > 0.0075 && quotes[3].flat_vol < 0.0090);
        let stripped = strip_caplet_vols(&quotes, &curve, &curve, RateVolKind::Normal).unwrap();
        assert_eq!(stripped.ends().len(), 4);
        for (got, want) in stripped.vols().iter().zip(truth.vols()) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
        for q in &quotes {
            let flat = q
                .cap
                .npv_black(&curve, &curve, RateVol::Normal(q.flat_vol))
                .unwrap();
            let strip = q
                .cap
                .npv_with_caplet_vols(&curve, &curve, &stripped)
                .unwrap();
            assert!((flat - strip).abs() < 1e-12, "{flat} vs {strip}");
        }
        // order does not matter
        let mut reversed = quotes.clone();
        reversed.reverse();
        let again = strip_caplet_vols(&reversed, &curve, &curve, RateVolKind::Normal).unwrap();
        assert_eq!(again.vols(), stripped.vols());
        // the curve is piecewise constant in fixing time
        assert_eq!(stripped.vol(0.1), stripped.vols()[0]);
        assert_eq!(stripped.vol(1.5), stripped.vols()[1]);
        assert_eq!(stripped.vol(40.0), stripped.vols()[3]);
    }

    #[test]
    fn a_falling_flat_vol_ladder_strips_to_lower_forward_caplet_vols() {
        let curve = market_curve();
        let caps = ladder(0.045);
        let flats = [0.0090, 0.0085, 0.0080, 0.0075];
        let quotes: Vec<CapQuote> = caps
            .iter()
            .zip(flats)
            .map(|(cap, flat_vol)| CapQuote {
                cap: cap.clone(),
                flat_vol,
            })
            .collect();
        let stripped = strip_caplet_vols(&quotes, &curve, &curve, RateVolKind::Normal).unwrap();
        // the first interval is the first cap's flat vol; later intervals
        // must sit below the falling flats to pull the averages down
        assert!((stripped.vols()[0] - 0.0090).abs() < 1e-10);
        for k in 1..4 {
            assert!(
                stripped.vols()[k] < flats[k],
                "interval {k}: {}",
                stripped.vols()[k]
            );
        }
    }

    #[test]
    fn validation_rejects_bad_ladders() {
        let curve = market_curve();
        assert!(strip_caplet_vols(&[], &curve, &curve, RateVolKind::Normal).is_err());
        // mixed strikes
        let mut caps = ladder(0.045);
        caps[1] =
            CapFloor::usd_standard(1.0, 0.05, CapOrFloor::Cap, d(2026, 8, 13), d(2028, 8, 14))
                .unwrap();
        let quotes: Vec<CapQuote> = caps
            .iter()
            .map(|cap| CapQuote {
                cap: cap.clone(),
                flat_vol: 0.008,
            })
            .collect();
        assert!(strip_caplet_vols(&quotes, &curve, &curve, RateVolKind::Normal).is_err());
        // an inconsistent ladder: a longer cap cheaper than the caplets
        // already stripped can be refused
        let caps = ladder(0.045);
        let quotes = vec![
            CapQuote {
                cap: caps[0].clone(),
                flat_vol: 0.02,
            },
            CapQuote {
                cap: caps[1].clone(),
                flat_vol: 0.001,
            },
        ];
        assert!(strip_caplet_vols(&quotes, &curve, &curve, RateVolKind::Normal).is_err());
        assert!(CapletVolCurve::new(vec![1.0], vec![], RateVolKind::Normal).is_err());
        assert!(
            CapletVolCurve::new(vec![2.0, 1.0], vec![0.01, 0.01], RateVolKind::Normal).is_err()
        );
    }
}
