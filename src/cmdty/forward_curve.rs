//! Commodity forward price curve.
//!
//! A strip of forward prices by delivery/pricing date — the term
//! structure a commodity swap's floating leg is projected from. Prices
//! are interpolated linearly in calendar time between pillars and held
//! flat outside them (before the first pillar and beyond the last).

use chrono::NaiveDate;
use serde::Serialize;

use crate::core::errors::RustyQLibError;

/// A commodity forward curve: pillar dates with their forward prices,
/// anchored at `reference_date` (the valuation date the curve was
/// observed on).
///
/// Prices may be any finite value — negative prices are legitimate
/// market data (WTI settled at −$37.63/bbl on 2020-04-20).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CommodityForwardCurve {
    reference_date: NaiveDate,
    /// Strictly increasing dates, none before `reference_date`.
    pillars: Vec<(NaiveDate, f64)>,
}

impl CommodityForwardCurve {
    /// A flat curve: the same forward price at every date.
    pub fn flat(price: f64, reference_date: NaiveDate) -> Result<Self, RustyQLibError> {
        Self::from_prices(reference_date, vec![(reference_date, price)])
    }

    /// Build a curve from `(date, price)` pillars. The pillars are
    /// sorted by date; duplicate dates, non-finite prices and dates
    /// before `reference_date` are rejected.
    pub fn from_prices(
        reference_date: NaiveDate,
        mut pillars: Vec<(NaiveDate, f64)>,
    ) -> Result<Self, RustyQLibError> {
        if pillars.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "forward curve",
                "at least one pillar is required",
            ));
        }
        pillars.sort_by_key(|&(date, _)| date);
        for pair in pillars.windows(2) {
            if pair[0].0 == pair[1].0 {
                return Err(RustyQLibError::invalid_input(
                    "forward curve",
                    format!("duplicate pillar date {}", pair[0].0),
                ));
            }
        }
        for &(date, price) in &pillars {
            if date < reference_date {
                return Err(RustyQLibError::invalid_input(
                    "forward curve",
                    format!("pillar {date} is before reference date {reference_date}"),
                ));
            }
            if !price.is_finite() {
                return Err(RustyQLibError::invalid_input(
                    "forward curve",
                    format!("price at {date} must be finite, got {price}"),
                ));
            }
        }
        Ok(CommodityForwardCurve {
            reference_date,
            pillars,
        })
    }

    /// The forward price at `date`: linear in calendar days between
    /// pillars, flat before the first pillar and beyond the last.
    pub fn price(&self, date: NaiveDate) -> f64 {
        let first = self.pillars.first().expect("validated non-empty");
        let last = self.pillars.last().expect("validated non-empty");
        if date <= first.0 {
            return first.1;
        }
        if date >= last.0 {
            return last.1;
        }
        let idx = self.pillars.partition_point(|&(d, _)| d <= date);
        let (d0, p0) = self.pillars[idx - 1];
        let (d1, p1) = self.pillars[idx];
        let w = (date - d0).num_days() as f64 / (d1 - d0).num_days() as f64;
        p0 + w * (p1 - p0)
    }

    /// A copy of the curve with every pillar price shifted by `amount`
    /// (a parallel bump, for scenario and delta work).
    pub fn bumped(&self, amount: f64) -> Result<Self, RustyQLibError> {
        if !amount.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "forward curve",
                format!("bump must be finite, got {amount}"),
            ));
        }
        Ok(CommodityForwardCurve {
            reference_date: self.reference_date,
            pillars: self
                .pillars
                .iter()
                .map(|&(date, price)| (date, price + amount))
                .collect(),
        })
    }

    pub fn reference_date(&self) -> NaiveDate {
        self.reference_date
    }

    /// The `(date, price)` pillars, sorted by date.
    pub fn pillars(&self) -> &[(NaiveDate, f64)] {
        &self.pillars
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn flat_curve_returns_the_price_everywhere() {
        let curve = CommodityForwardCurve::flat(72.5, d(2026, 8, 6)).unwrap();
        assert_eq!(curve.price(d(2026, 8, 6)), 72.5);
        assert_eq!(curve.price(d(2030, 1, 1)), 72.5);
    }

    #[test]
    fn interpolates_linearly_and_extrapolates_flat() {
        let reference = d(2026, 8, 1);
        let curve = CommodityForwardCurve::from_prices(
            reference,
            // deliberately unsorted: the constructor sorts
            vec![(d(2026, 10, 1), 74.0), (d(2026, 9, 1), 72.0)],
        )
        .unwrap();
        assert_eq!(curve.pillars()[0].0, d(2026, 9, 1));
        // flat before the first pillar and after the last
        assert_eq!(curve.price(d(2026, 8, 15)), 72.0);
        assert_eq!(curve.price(d(2027, 1, 1)), 74.0);
        // exactly on the pillars
        assert_eq!(curve.price(d(2026, 9, 1)), 72.0);
        assert_eq!(curve.price(d(2026, 10, 1)), 74.0);
        // halfway through the 30-day gap: 72 + 15/30 * 2
        assert!((curve.price(d(2026, 9, 16)) - 73.0).abs() < 1e-12);
    }

    #[test]
    fn bump_shifts_every_price() {
        let curve = CommodityForwardCurve::from_prices(
            d(2026, 8, 1),
            vec![(d(2026, 9, 1), 72.0), (d(2026, 10, 1), 74.0)],
        )
        .unwrap();
        let up = curve.bumped(1.5).unwrap();
        assert!((up.price(d(2026, 9, 16)) - curve.price(d(2026, 9, 16)) - 1.5).abs() < 1e-12);
        assert!(curve.bumped(f64::NAN).is_err());
    }

    #[test]
    fn negative_prices_are_allowed() {
        // WTI, 2020-04-20
        let curve = CommodityForwardCurve::flat(-37.63, d(2020, 4, 20)).unwrap();
        assert_eq!(curve.price(d(2020, 4, 21)), -37.63);
    }

    #[test]
    fn validation_rejects_bad_pillars() {
        let reference = d(2026, 8, 1);
        assert!(CommodityForwardCurve::from_prices(reference, vec![]).is_err());
        assert!(CommodityForwardCurve::from_prices(
            reference,
            vec![(d(2026, 9, 1), 72.0), (d(2026, 9, 1), 73.0)],
        )
        .is_err());
        assert!(
            CommodityForwardCurve::from_prices(reference, vec![(d(2026, 7, 1), 72.0)]).is_err()
        );
        assert!(
            CommodityForwardCurve::from_prices(reference, vec![(d(2026, 9, 1), f64::NAN)]).is_err()
        );
    }
}
