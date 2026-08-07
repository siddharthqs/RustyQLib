//! US Treasury bond futures: conversion factors, invoice prices, basis
//! analytics and the cheapest-to-deliver.
//!
//! The [`conversion_factor`] follows the CME method: the price per 1
//! face of the deliverable at a 6% semiannual yield, with the remaining
//! term measured from the first day of the delivery month and rounded
//! down — to whole quarters for bond / 10-year style contracts
//! ([`FactorRounding::Quarter`]), to whole months for the short-tenor
//! contracts ([`FactorRounding::Month`]).
//!
//! Basis analytics follow the standard cash-and-carry algebra on an
//! Act/360 money-market basis (Burghardt): the implied repo rate is the
//! financing rate that equates buying the bond and delivering it into
//! the future; the net basis is the loss of that trade at a given repo
//! rate; the cheapest-to-deliver maximises the implied repo.

use chrono::{Datelike, NaiveDate};

use crate::bonds::FixedRateBond;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// How the remaining term is rounded in the conversion-factor
/// calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactorRounding {
    /// Down to whole quarters — T-Bond, Ultra and 10-year contracts.
    Quarter,
    /// Down to whole months — 2-, 3- and 5-year note contracts.
    Month,
}

/// First calendar day of `date`'s month.
fn month_start(date: NaiveDate) -> NaiveDate {
    NaiveDate::from_ymd_opt(date.year(), date.month(), 1).expect("valid month start")
}

/// Whole calendar months from the first day of `from`'s month to `to`.
fn whole_months(from: NaiveDate, to: NaiveDate) -> i32 {
    (to.year() - from.year()) * 12 + to.month() as i32 - from.month() as i32
}

/// CME conversion factor: the clean price per 1 face of a bond with
/// `coupon_rate` maturing `maturity`, valued at a 6% semiannual yield
/// from the first day of `delivery_month` with the term rounded down
/// per `rounding`. A 6% coupon on a whole-year term gives exactly 1.
pub fn conversion_factor(
    coupon_rate: f64,
    maturity: NaiveDate,
    delivery_month: NaiveDate,
    rounding: FactorRounding,
) -> Result<f64, RustyQLibError> {
    if !coupon_rate.is_finite() || coupon_rate < 0.0 {
        return Err(RustyQLibError::invalid_input(
            "conversion factor",
            format!("coupon rate must be non-negative, got {coupon_rate}"),
        ));
    }
    let reference = month_start(delivery_month);
    let months = whole_months(reference, maturity);
    if months <= 0 {
        return Err(RustyQLibError::invalid_input(
            "conversion factor",
            format!("maturity {maturity} is not after the delivery month start {reference}"),
        ));
    }
    let months = match rounding {
        FactorRounding::Quarter => months - months % 3,
        FactorRounding::Month => months,
    };
    let n = months / 12;
    let z = months % 12;
    // CME formula: front stub v months (capped into the half-year), one
    // extra half-period when z >= 7
    let (v, half_periods) = if z < 7 {
        (z, 2 * n)
    } else {
        (z - 6, 2 * n + 1)
    };
    let c = coupon_rate;
    let a = 1.03_f64.powf(-(v as f64) / 6.0);
    let b = (c / 2.0) * (6.0 - v as f64) / 6.0;
    let redemption = 1.03_f64.powi(-half_periods);
    let annuity = (c / 0.06) * (1.0 - redemption);
    Ok(a * (c / 2.0 + redemption + annuity) - b)
}

/// One bond in the deliverable basket with its conversion factor.
#[derive(Debug, Clone)]
pub struct DeliverableBond {
    pub bond: FixedRateBond,
    pub conversion_factor: f64,
}

/// A Treasury bond future over a deliverable basket, settling delivery
/// on a single `delivery_date` (the short's timing option within the
/// month is not modelled).
#[derive(Debug, Clone)]
pub struct BondFuture {
    pub delivery_date: NaiveDate,
    pub deliverables: Vec<DeliverableBond>,
}

impl BondFuture {
    pub fn new(
        delivery_date: NaiveDate,
        deliverables: Vec<DeliverableBond>,
    ) -> Result<Self, RustyQLibError> {
        if deliverables.is_empty() {
            return Err(RustyQLibError::invalid_input(
                "bond future",
                "the deliverable basket is empty",
            ));
        }
        for d in &deliverables {
            if !d.conversion_factor.is_finite() || d.conversion_factor <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "bond future",
                    format!(
                        "conversion factor must be positive, got {} for maturity {}",
                        d.conversion_factor, d.bond.maturity_date
                    ),
                ));
            }
            if d.bond.maturity_date <= delivery_date {
                return Err(RustyQLibError::invalid_input(
                    "bond future",
                    format!(
                        "deliverable maturing {} does not survive delivery {delivery_date}",
                        d.bond.maturity_date
                    ),
                ));
            }
        }
        Ok(BondFuture {
            delivery_date,
            deliverables,
        })
    }

    /// Build the basket computing each bond's conversion factor from the
    /// delivery month.
    pub fn with_computed_factors(
        delivery_date: NaiveDate,
        bonds: Vec<FixedRateBond>,
        rounding: FactorRounding,
    ) -> Result<Self, RustyQLibError> {
        let deliverables = bonds
            .into_iter()
            .map(|bond| {
                let conversion_factor = conversion_factor(
                    bond.coupon_rate,
                    bond.maturity_date,
                    delivery_date,
                    rounding,
                )?;
                Ok(DeliverableBond {
                    bond,
                    conversion_factor,
                })
            })
            .collect::<Result<Vec<_>, RustyQLibError>>()?;
        Self::new(delivery_date, deliverables)
    }

    /// Invoice price per 100 face received by the short on delivering
    /// `deliverable`: `futures_price * cf + accrued at delivery`.
    pub fn invoice_price(
        &self,
        deliverable: &DeliverableBond,
        futures_price: f64,
    ) -> Result<f64, RustyQLibError> {
        let accrued = deliverable.bond.accrued_interest(self.delivery_date)?;
        Ok(futures_price * deliverable.conversion_factor + accrued)
    }

    /// Gross basis per 100 face: `clean_price - futures_price * cf`.
    pub fn gross_basis(
        &self,
        deliverable: &DeliverableBond,
        futures_price: f64,
        clean_price: f64,
    ) -> f64 {
        clean_price - futures_price * deliverable.conversion_factor
    }

    /// Interim coupons per 100 face paid after `settlement` up to and
    /// including delivery, with their financing days to delivery
    /// (scheduled dates, Act/360 basis).
    fn interim_coupons(
        &self,
        deliverable: &DeliverableBond,
        settlement: NaiveDate,
    ) -> Vec<(f64, f64)> {
        let bond = &deliverable.bond;
        bond.cashflows()
            .iter()
            .filter(|cf| cf.accrual_end > settlement && cf.accrual_end <= self.delivery_date)
            .map(|cf| {
                let coupon_only = cf.amount
                    - if cf.accrual_end == bond.maturity_date {
                        bond.face_value
                    } else {
                        0.0
                    };
                (
                    coupon_only * 100.0 / bond.face_value,
                    (self.delivery_date - cf.accrual_end).num_days() as f64,
                )
            })
            .collect()
    }

    /// The implied repo rate (Act/360): the break-even financing rate of
    /// buying the bond at `clean_price` on `settlement` and delivering
    /// it into the future at `futures_price`.
    pub fn implied_repo(
        &self,
        deliverable: &DeliverableBond,
        futures_price: f64,
        clean_price: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let days = (self.delivery_date - settlement).num_days() as f64;
        if days <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "implied repo",
                format!(
                    "settlement {settlement} must precede delivery {}",
                    self.delivery_date
                ),
            ));
        }
        let full_start = clean_price + deliverable.bond.accrued_interest(settlement)?;
        let full_end = self.invoice_price(deliverable, futures_price)?;
        let coupons = self.interim_coupons(deliverable, settlement);
        let coupon_sum: f64 = coupons.iter().map(|(c, _)| c).sum();
        let denominator =
            full_start * days / 360.0 - coupons.iter().map(|(c, d_i)| c * d_i / 360.0).sum::<f64>();
        if denominator <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "degenerate implied-repo denominator {denominator}"
            )));
        }
        Ok((full_end + coupon_sum - full_start) / denominator)
    }

    /// Net basis per 100 face at a given `repo_rate` (Act/360): the cost
    /// of carrying the bond to delivery minus the invoice proceeds.
    /// Zero when the repo rate equals the implied repo.
    pub fn net_basis(
        &self,
        deliverable: &DeliverableBond,
        futures_price: f64,
        clean_price: f64,
        settlement: NaiveDate,
        repo_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        let days = (self.delivery_date - settlement).num_days() as f64;
        let full_start = clean_price + deliverable.bond.accrued_interest(settlement)?;
        let full_end = self.invoice_price(deliverable, futures_price)?;
        let reinvested_coupons: f64 = self
            .interim_coupons(deliverable, settlement)
            .iter()
            .map(|(c, d_i)| c * (1.0 + repo_rate * d_i / 360.0))
            .sum();
        Ok(full_start * (1.0 + repo_rate * days / 360.0) - reinvested_coupons - full_end)
    }

    /// The cheapest-to-deliver: the basket index with the highest
    /// implied repo, given each deliverable's clean price (same order as
    /// the basket). Returns `(index, implied_repo)`.
    pub fn cheapest_to_deliver(
        &self,
        futures_price: f64,
        clean_prices: &[f64],
        settlement: NaiveDate,
    ) -> Result<(usize, f64), RustyQLibError> {
        if clean_prices.len() != self.deliverables.len() {
            return Err(RustyQLibError::invalid_input(
                "cheapest to deliver",
                format!(
                    "{} clean prices for {} deliverables",
                    clean_prices.len(),
                    self.deliverables.len()
                ),
            ));
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, (deliverable, &clean)) in self.deliverables.iter().zip(clean_prices).enumerate() {
            let repo = self.implied_repo(deliverable, futures_price, clean, settlement)?;
            if best.is_none_or(|(_, r)| repo > r) {
                best = Some((i, repo));
            }
        }
        Ok(best.expect("basket is non-empty"))
    }

    /// Theoretical futures price off a discount curve: the smallest
    /// forward clean price over conversion factor across the basket (the
    /// short delivers whatever is cheapest).
    pub fn theoretical_price(&self, curve: &YieldCurve) -> Result<f64, RustyQLibError> {
        let mut best = f64::INFINITY;
        for deliverable in &self.deliverables {
            let forward_clean = deliverable
                .bond
                .clean_price_from_curve(curve, self.delivery_date)?;
            best = best.min(forward_clean / deliverable.conversion_factor);
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn six_percent_coupon_on_whole_years_has_factor_one() {
        for years in [2, 5, 10, 25] {
            let cf = conversion_factor(
                0.06,
                d(2026 + years, 9, 1),
                d(2026, 9, 1),
                FactorRounding::Quarter,
            )
            .unwrap();
            assert!((cf - 1.0).abs() < 1e-12, "{years}y: {cf}");
        }
    }

    #[test]
    fn twenty_year_four_percent_matches_the_annuity_arithmetic() {
        // 4% for 20 whole years at 6%: (0.02/0.03)(1 - 1.03^-40) + 1.03^-40
        let cf = conversion_factor(0.04, d(2046, 9, 1), d(2026, 9, 15), FactorRounding::Quarter)
            .unwrap();
        let redemption = 1.03_f64.powi(-40);
        let expected = (0.02 / 0.03) * (1.0 - redemption) + redemption;
        assert!((cf - expected).abs() < 1e-14, "{cf} vs {expected}");
        // below 6% coupon -> factor below 1; above -> above 1
        assert!(cf < 1.0);
        let rich =
            conversion_factor(0.08, d(2046, 9, 1), d(2026, 9, 1), FactorRounding::Quarter).unwrap();
        assert!(rich > 1.0);
    }

    #[test]
    fn whole_year_factor_agrees_with_the_bond_pricer_at_six_percent() {
        // CF with z = 0 is the clean price per 1 face at a 6% street
        // yield settling on the cycle: cross-check against FixedRateBond
        let delivery = d(2026, 9, 1);
        let maturity = d(2033, 9, 1);
        let cf = conversion_factor(0.045, maturity, delivery, FactorRounding::Quarter).unwrap();
        let bond = FixedRateBond::us_treasury(100.0, 0.045, delivery, maturity).unwrap();
        let clean = bond.clean_price_from_yield(0.06, delivery).unwrap();
        assert!(
            (cf - clean / 100.0).abs() < 1e-12,
            "{cf} vs {}",
            clean / 100.0
        );
    }

    #[test]
    fn quarter_rounding_floors_odd_months() {
        // maturity 2036-05-15 from 2026-09-01: 116 whole months -> 114
        // after quarter rounding -> n = 9, z = 6
        let a = conversion_factor(0.05, d(2036, 5, 15), d(2026, 9, 1), FactorRounding::Quarter)
            .unwrap();
        let b =
            conversion_factor(0.05, d(2036, 3, 1), d(2026, 9, 1), FactorRounding::Quarter).unwrap();
        assert!((a - b).abs() < 1e-15, "same rounded term must match");
        // monthly rounding keeps the extra 2 months
        let m =
            conversion_factor(0.05, d(2036, 5, 15), d(2026, 9, 1), FactorRounding::Month).unwrap();
        assert!((m - a).abs() > 1e-6);
    }

    /// A 2-bond basket around a Sep 2026 delivery.
    fn sample_future() -> (BondFuture, Vec<f64>) {
        let low_coupon =
            FixedRateBond::us_treasury(100.0, 0.04, d(2026, 5, 15), d(2033, 5, 15)).unwrap();
        let high_coupon =
            FixedRateBond::us_treasury(100.0, 0.055, d(2026, 2, 15), d(2034, 2, 15)).unwrap();
        let future = BondFuture::with_computed_factors(
            d(2026, 9, 30),
            vec![low_coupon, high_coupon],
            FactorRounding::Quarter,
        )
        .unwrap();
        (future, vec![97.50, 106.20])
    }

    #[test]
    fn invoice_and_gross_basis_arithmetic() {
        let (future, cleans) = sample_future();
        let deliverable = &future.deliverables[0];
        let futures_price = 110.0;
        let invoice = future.invoice_price(deliverable, futures_price).unwrap();
        let accrued = deliverable
            .bond
            .accrued_interest(future.delivery_date)
            .unwrap();
        assert!(
            (invoice - (futures_price * deliverable.conversion_factor + accrued)).abs() < 1e-12
        );
        let basis = future.gross_basis(deliverable, futures_price, cleans[0]);
        assert!(
            (basis - (cleans[0] - futures_price * deliverable.conversion_factor)).abs() < 1e-12
        );
    }

    #[test]
    fn implied_repo_round_trips_a_fair_forward() {
        // choose the futures price that makes carry at 4% break even:
        // the implied repo must then come back as exactly 4%
        let (future, cleans) = sample_future();
        let deliverable = &future.deliverables[0];
        let settlement = d(2026, 8, 6);
        let repo = 0.04;
        let days = (future.delivery_date - settlement).num_days() as f64;
        let full_start = cleans[0] + deliverable.bond.accrued_interest(settlement).unwrap();
        let accrued_delivery = deliverable
            .bond
            .accrued_interest(future.delivery_date)
            .unwrap();
        // no coupon between Aug 6 and Sep 30 on the May/Nov cycle
        let fair_futures = (full_start * (1.0 + repo * days / 360.0) - accrued_delivery)
            / deliverable.conversion_factor;
        let implied = future
            .implied_repo(deliverable, fair_futures, cleans[0], settlement)
            .unwrap();
        assert!((implied - repo).abs() < 1e-12, "implied {implied}");
        // and the net basis at that repo is zero
        let nb = future
            .net_basis(deliverable, fair_futures, cleans[0], settlement, repo)
            .unwrap();
        assert!(nb.abs() < 1e-10, "net basis {nb}");
        // a cheaper future makes delivery a losing carry: positive net basis
        let nb_low = future
            .net_basis(deliverable, fair_futures - 0.5, cleans[0], settlement, repo)
            .unwrap();
        assert!(nb_low > 0.0);
    }

    #[test]
    fn implied_repo_handles_an_interim_coupon() {
        // settlement before the Nov 15 coupon, delivery after it
        let bond =
            FixedRateBond::us_treasury(100.0, 0.045, d(2026, 5, 15), d(2033, 5, 15)).unwrap();
        let future =
            BondFuture::with_computed_factors(d(2026, 12, 15), vec![bond], FactorRounding::Quarter)
                .unwrap();
        let deliverable = &future.deliverables[0];
        let settlement = d(2026, 8, 6);
        let clean = 99.25;
        let repo = 0.045;
        // build the fair futures price including the reinvested coupon
        let days = (future.delivery_date - settlement).num_days() as f64;
        let coupon_days = (future.delivery_date - d(2026, 11, 15)).num_days() as f64;
        let full_start = clean + deliverable.bond.accrued_interest(settlement).unwrap();
        let accrued_delivery = deliverable
            .bond
            .accrued_interest(future.delivery_date)
            .unwrap();
        let fair_futures = (full_start * (1.0 + repo * days / 360.0)
            - 2.25 * (1.0 + repo * coupon_days / 360.0)
            - accrued_delivery)
            / deliverable.conversion_factor;
        let implied = future
            .implied_repo(deliverable, fair_futures, clean, settlement)
            .unwrap();
        assert!((implied - repo).abs() < 1e-12, "implied {implied}");
    }

    #[test]
    fn cheapest_to_deliver_picks_the_highest_implied_repo() {
        let (future, mut cleans) = sample_future();
        let settlement = d(2026, 8, 6);
        let futures_price = 108.0;
        let (index, best_repo) = future
            .cheapest_to_deliver(futures_price, &cleans, settlement)
            .unwrap();
        // verify it really is the max
        for (i, deliverable) in future.deliverables.iter().enumerate() {
            let repo = future
                .implied_repo(deliverable, futures_price, cleans[i], settlement)
                .unwrap();
            assert!(repo <= best_repo + 1e-15, "basket {i}");
        }
        // cheapening the other bond flips the CTD
        let other = 1 - index;
        cleans[other] -= 3.0;
        let (flipped, _) = future
            .cheapest_to_deliver(futures_price, &cleans, settlement)
            .unwrap();
        assert_eq!(flipped, other);
    }

    #[test]
    fn theoretical_price_is_the_cheapest_forward_over_factor() {
        use crate::core::curves::Compounding;
        use crate::core::daycount::DayCountConvention;
        let (future, _) = sample_future();
        let curve = YieldCurve::flat(
            0.04,
            d(2026, 8, 6),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let theo = future.theoretical_price(&curve).unwrap();
        let manual = future
            .deliverables
            .iter()
            .map(|del| {
                del.bond
                    .clean_price_from_curve(&curve, future.delivery_date)
                    .unwrap()
                    / del.conversion_factor
            })
            .fold(f64::INFINITY, f64::min);
        assert!((theo - manual).abs() < 1e-12);
        assert!(theo > 50.0 && theo < 150.0, "theo {theo}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(
            conversion_factor(-0.01, d(2033, 9, 1), d(2026, 9, 1), FactorRounding::Quarter)
                .is_err()
        );
        assert!(
            conversion_factor(0.05, d(2026, 9, 1), d(2026, 9, 1), FactorRounding::Quarter).is_err()
        );
        assert!(BondFuture::new(d(2026, 9, 30), vec![]).is_err());
        // deliverable matured before delivery
        let short =
            FixedRateBond::us_treasury(100.0, 0.04, d(2024, 5, 15), d(2026, 5, 15)).unwrap();
        assert!(BondFuture::new(
            d(2026, 9, 30),
            vec![DeliverableBond {
                bond: short,
                conversion_factor: 0.9,
            }],
        )
        .is_err());
        let (future, cleans) = sample_future();
        // wrong clean-price count
        assert!(future
            .cheapest_to_deliver(110.0, &cleans[..1], d(2026, 8, 6))
            .is_err());
        // settlement after delivery
        assert!(future
            .implied_repo(&future.deliverables[0], 110.0, cleans[0], d(2027, 1, 1))
            .is_err());
    }
}
