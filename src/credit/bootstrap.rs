//! Bootstrapping the hazard curve from CDS par spreads.
//!
//! The credit analogue of the discount-curve bootstrap: the quotes are
//! taken in maturity order and each pins the hazard on the segment from
//! the previous maturity to its own, chosen so that a standard contract
//! to that maturity, paying the quoted spread as its coupon, is worth
//! zero on the curve built so far. Pillar times run from the valuation
//! date on the discount curve's day count, as [`CreditCurve`] expects.

use chrono::NaiveDate;

use super::cds::CreditDefaultSwap;
use super::curve::CreditCurve;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// One quoted maturity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CdsQuote {
    pub maturity: NaiveDate,
    /// Par spread as a decimal.
    pub par_spread: f64,
}

/// Bootstrap a piecewise-constant hazard curve from par-spread quotes
/// of one name, all with the same recovery and the standard contract
/// terms, valued on `valuation` (the protection start of every
/// contract).
pub fn bootstrap_cds_curve(
    quotes: &[CdsQuote],
    recovery_rate: f64,
    curve: &YieldCurve,
    valuation: NaiveDate,
) -> Result<CreditCurve, RustyQLibError> {
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input("cds bootstrap", "no quotes"));
    }
    let mut order: Vec<usize> = (0..quotes.len()).collect();
    order.sort_by_key(|&i| quotes[i].maturity);
    for pair in order.windows(2) {
        if quotes[pair[0]].maturity == quotes[pair[1]].maturity {
            return Err(RustyQLibError::invalid_input(
                "cds bootstrap",
                format!("two quotes share the maturity {}", quotes[pair[0]].maturity),
            ));
        }
    }
    let day_count = curve.day_count();
    let mut pillars: Vec<(f64, f64)> = Vec::with_capacity(quotes.len());
    for &index in &order {
        let quote = &quotes[index];
        if !(quote.par_spread > 0.0 && quote.par_spread.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "cds bootstrap",
                format!("par spreads must be positive, got {}", quote.par_spread),
            ));
        }
        let pillar_time = day_count.year_fraction(valuation, quote.maturity);
        if pillar_time <= pillars.last().map_or(0.0, |&(t, _)| t) {
            return Err(RustyQLibError::invalid_input(
                "cds bootstrap",
                format!(
                    "maturity {} is not after the valuation date",
                    quote.maturity
                ),
            ));
        }
        let contract = CreditDefaultSwap::new(
            1.0,
            quote.par_spread,
            recovery_rate,
            valuation,
            quote.maturity,
        )?;
        let value_at = |hazard: f64| -> Result<f64, RustyQLibError> {
            let mut trial = pillars.clone();
            trial.push((pillar_time, hazard));
            contract.npv(curve, &CreditCurve::new(&trial)?, valuation)
        };
        // the buyer's value rises with the segment's hazard
        value_at(0.0)?;
        value_at(10.0)?;
        let objective = |hazard: f64| value_at(hazard).expect("the hazard bracket was priced");
        let root = Solver1d::new(1e-14, 200).bisection(objective, 0.0, 10.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: format!(
                    "the hazard for the {} quote did not converge",
                    quote.maturity
                ),
            });
        }
        pillars.push((pillar_time, root.x));
    }
    CreditCurve::new(&pillars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat_curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(2026, 8, 14),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    #[test]
    fn round_trips_a_piecewise_hazard_curve() {
        let curve = flat_curve(0.04);
        let valuation = d(2026, 8, 14);
        let maturities = [
            d(2027, 9, 20),
            d(2029, 9, 20),
            d(2031, 9, 20),
            d(2036, 9, 20),
        ];
        let hazards = [0.01, 0.02, 0.015, 0.03];
        let pillars: Vec<(f64, f64)> = maturities
            .iter()
            .zip(hazards)
            .map(|(&m, h)| (DayCountConvention::Act365.year_fraction(valuation, m), h))
            .collect();
        let truth = CreditCurve::new(&pillars).unwrap();
        // the quotes the true curve implies
        let quotes: Vec<CdsQuote> = maturities
            .iter()
            .map(|&maturity| {
                let contract = CreditDefaultSwap::new(1.0, 0.01, 0.4, valuation, maturity).unwrap();
                CdsQuote {
                    maturity,
                    par_spread: contract.par_spread(&curve, &truth, valuation).unwrap(),
                }
            })
            .collect();
        assert!(quotes
            .windows(2)
            .all(|w| w[0].par_spread != w[1].par_spread));
        // shuffled on input, recovered to the solver's tolerance
        let shuffled = [quotes[2], quotes[0], quotes[3], quotes[1]];
        let built = bootstrap_cds_curve(&shuffled, 0.4, &curve, valuation).unwrap();
        for ((t, h), (bt, bh)) in pillars.iter().zip(built.pillars()) {
            assert!(
                (t - bt).abs() < 1e-12 && (h - bh).abs() < 1e-8,
                "{h} vs {bh}"
            );
        }
        // and every quoted contract reprices to zero on it
        for quote in &quotes {
            let contract = CreditDefaultSwap::new(
                10_000_000.0,
                quote.par_spread,
                0.4,
                valuation,
                quote.maturity,
            )
            .unwrap();
            assert!(contract.npv(&curve, &built, valuation).unwrap().abs() < 1e-3);
        }
    }

    #[test]
    fn rejects_bad_quotes() {
        let curve = flat_curve(0.04);
        let valuation = d(2026, 8, 14);
        assert!(bootstrap_cds_curve(&[], 0.4, &curve, valuation).is_err());
        let quote = |maturity, par_spread| CdsQuote {
            maturity,
            par_spread,
        };
        assert!(bootstrap_cds_curve(
            &[quote(d(2029, 9, 20), 0.01), quote(d(2029, 9, 20), 0.012)],
            0.4,
            &curve,
            valuation
        )
        .is_err());
        assert!(
            bootstrap_cds_curve(&[quote(d(2029, 9, 20), -0.01)], 0.4, &curve, valuation).is_err()
        );
        assert!(
            bootstrap_cds_curve(&[quote(d(2025, 9, 20), 0.01)], 0.4, &curve, valuation).is_err()
        );
        assert!(
            bootstrap_cds_curve(&[quote(d(2029, 9, 20), 0.01)], 0.4, &curve, valuation).is_ok()
        );
    }
}
