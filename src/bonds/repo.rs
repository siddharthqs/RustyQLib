//! Repurchase agreements on bond collateral, and the financing
//! analytics they imply for the bond.
//!
//! A repo is a collateralised loan dressed as a sale and repurchase:
//! the seller delivers collateral worth its dirty market value, receives
//! the **purchase price** — that value less a haircut — and on the
//! repurchase date buys the collateral back at the **repurchase
//! price**, the purchase price accrued at the repo rate (simple
//! interest, Act/360 in the US market). Coupons paid on the collateral
//! during the term are passed straight through to the seller
//! (manufactured payments under the GMRA), so they do not enter the
//! repo's own cash legs; they do move the collateral's value and hence
//! the margin.
//!
//! [`RepurchaseAgreement`] holds one such trade from either side and
//! gives its cash legs, the accrued repurchase price on any date, the
//! coupon pass-through, the collateral exposure and margin call as the
//! collateral reprices, and the mark-to-market of a term repo against
//! the current market rate (an open repo resets daily and has none).
//!
//! On the bond side, [`FixedRateBond::forward_dirty_price`] and
//! [`FixedRateBond::carry`] give the forward price of a position
//! financed in repo, with interim coupons reinvested at the repo rate —
//! the same convention as the futures implied repo in
//! [`futures`](crate::bonds::futures), so a future priced at the repo
//! forward implies exactly the repo rate.

use chrono::NaiveDate;

use crate::bonds::FixedRateBond;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

/// Which side of the trade the holder is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepoSide {
    /// Sells the collateral and borrows the cash (the "repo" side).
    Repo,
    /// Buys the collateral and lends the cash (the "reverse" side).
    ReverseRepo,
}

/// A repurchase agreement on a fixed-rate bond.
#[derive(Debug, Clone)]
pub struct RepurchaseAgreement {
    /// The collateral.
    pub collateral: FixedRateBond,
    /// Par amount of collateral delivered.
    pub par: f64,
    /// Collateral clean price per 100 at the start.
    pub start_clean_price: f64,
    /// Fraction of the collateral's market value not lent, in `[0, 1)`
    /// (a 2% haircut is `0.02`).
    pub haircut: f64,
    /// Simple annual repo rate.
    pub repo_rate: f64,
    /// Day count of the repo interest (Act/360 in the US market).
    pub day_count: DayCountConvention,
    /// Purchase (start) date.
    pub start: NaiveDate,
    /// Repurchase date, or `None` for an open repo that rolls daily
    /// until terminated.
    pub end: Option<NaiveDate>,
    pub side: RepoSide,
}

impl RepurchaseAgreement {
    /// A term (or, with `end = None`, open) repo at an Act/360 rate,
    /// seen from the reverse side (the cash lender). Change `side` and
    /// `day_count` on the struct.
    pub fn new(
        collateral: FixedRateBond,
        par: f64,
        start_clean_price: f64,
        haircut: f64,
        repo_rate: f64,
        start: NaiveDate,
        end: Option<NaiveDate>,
    ) -> Result<Self, RustyQLibError> {
        let repo = RepurchaseAgreement {
            collateral,
            par,
            start_clean_price,
            haircut,
            repo_rate,
            day_count: DayCountConvention::Act360,
            start,
            end,
            side: RepoSide::ReverseRepo,
        };
        repo.validate()?;
        Ok(repo)
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        let invalid = |what: String| Err(RustyQLibError::invalid_input("repo", what));
        if !(self.par > 0.0 && self.par.is_finite()) {
            return invalid(format!("par must be positive, got {}", self.par));
        }
        if !(self.start_clean_price > 0.0 && self.start_clean_price.is_finite()) {
            return invalid(format!(
                "start clean price must be positive, got {}",
                self.start_clean_price
            ));
        }
        if !(self.haircut.is_finite() && (0.0..1.0).contains(&self.haircut)) {
            return invalid(format!("haircut must be in [0, 1), got {}", self.haircut));
        }
        if !self.repo_rate.is_finite() {
            return invalid("repo rate must be finite".to_string());
        }
        if let Some(end) = self.end {
            if end <= self.start {
                return invalid(format!("end {end} must be after start {}", self.start));
            }
        }
        // the collateral must be alive over the term
        self.collateral.accrued_interest(self.start)?;
        Ok(())
    }

    /// The signed multiplier for this side: cash received is positive.
    fn sign(&self) -> f64 {
        match self.side {
            RepoSide::Repo => 1.0,
            RepoSide::ReverseRepo => -1.0,
        }
    }

    /// The date interest accrues to at the latest: the repurchase date
    /// of a term repo, the collateral's maturity for an open one.
    fn last_date(&self) -> NaiveDate {
        self.end.unwrap_or(self.collateral.maturity_date)
    }

    fn check_date(&self, date: NaiveDate) -> Result<(), RustyQLibError> {
        if date < self.start || date > self.last_date() {
            return Err(RustyQLibError::invalid_input(
                "repo",
                format!(
                    "{date} is outside the repo's life {} to {}",
                    self.start,
                    self.last_date()
                ),
            ));
        }
        Ok(())
    }

    /// Dirty market value of the collateral on `date` at a clean price
    /// per 100.
    pub fn collateral_value(
        &self,
        date: NaiveDate,
        clean_price: f64,
    ) -> Result<f64, RustyQLibError> {
        if !(clean_price > 0.0 && clean_price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "repo",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        let dirty = clean_price + self.collateral.accrued_interest(date)?;
        Ok(self.par * dirty / 100.0)
    }

    /// The purchase price: the collateral's start value less the
    /// haircut — the cash that changes hands at the start.
    pub fn purchase_price(&self) -> Result<f64, RustyQLibError> {
        Ok(self.collateral_value(self.start, self.start_clean_price)? * (1.0 - self.haircut))
    }

    /// The purchase price accrued at the repo rate to `date`: what the
    /// seller owes to unwind the trade that day.
    pub fn repurchase_price(&self, date: NaiveDate) -> Result<f64, RustyQLibError> {
        self.check_date(date)?;
        let accrual = self.repo_rate * self.day_count.year_fraction(self.start, date);
        Ok(self.purchase_price()? * (1.0 + accrual))
    }

    /// The repurchase price of a term repo; an open repo has none.
    pub fn end_cash(&self) -> Result<f64, RustyQLibError> {
        match self.end {
            Some(end) => self.repurchase_price(end),
            None => Err(RustyQLibError::invalid_input(
                "repo",
                "an open repo has no fixed repurchase price; use repurchase_price(date)",
            )),
        }
    }

    /// Repo interest over the full term.
    pub fn interest(&self) -> Result<f64, RustyQLibError> {
        Ok(self.end_cash()? - self.purchase_price()?)
    }

    /// The repo's own cash legs from this side, positive when received:
    /// the purchase price at the start and the repurchase price at the
    /// end (term repos only).
    pub fn cash_flows(&self) -> Result<Vec<(NaiveDate, f64)>, RustyQLibError> {
        let end = self.end.ok_or_else(|| {
            RustyQLibError::invalid_input("repo", "an open repo has no fixed cash legs")
        })?;
        let sign = self.sign();
        Ok(vec![
            (self.start, sign * self.purchase_price()?),
            (end, -sign * self.end_cash()?),
        ])
    }

    /// Coupons the collateral pays during the term, which the buyer
    /// passes to the seller on their payment dates: `(date, amount)` on
    /// the par delivered, positive from this side's point of view.
    pub fn manufactured_payments(&self) -> Vec<(NaiveDate, f64)> {
        let last = self.last_date();
        let scale = self.par / self.collateral.face_value;
        let sign = self.sign();
        self.collateral
            .cashflows()
            .iter()
            .filter(|cf| cf.payment_date > self.start && cf.payment_date <= last)
            .map(|cf| {
                let coupon = cf.amount - self.collateral.principal_at(cf.accrual_end);
                (cf.payment_date, sign * coupon * scale)
            })
            .collect()
    }

    /// The cash lender's uncovered exposure on `date` at a collateral
    /// clean price: the accrued repurchase price less the collateral's
    /// value after the haircut. Positive means the seller is under-
    /// collateralised and owes margin; negative, the buyer holds excess.
    pub fn exposure(&self, date: NaiveDate, clean_price: f64) -> Result<f64, RustyQLibError> {
        let covered = self.collateral_value(date, clean_price)? * (1.0 - self.haircut);
        Ok(self.repurchase_price(date)? - covered)
    }

    /// The variation margin due on `date`, from this side's point of
    /// view (positive = receive), once the exposure exceeds `threshold`
    /// in absolute value; zero inside the threshold.
    pub fn margin_call(
        &self,
        date: NaiveDate,
        clean_price: f64,
        threshold: f64,
    ) -> Result<f64, RustyQLibError> {
        if !(threshold >= 0.0 && threshold.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "repo",
                format!("the margin threshold must be non-negative, got {threshold}"),
            ));
        }
        let exposure = self.exposure(date, clean_price)?;
        if exposure.abs() <= threshold {
            return Ok(0.0);
        }
        // the seller (repo side) pays a positive exposure
        Ok(-self.sign() * exposure)
    }

    /// The value on `date` of the repurchase-price claim at the current
    /// market repo rate for the remaining term: the repurchase price
    /// discounted at simple interest. An open repo resets daily, so its
    /// value is the accrued repurchase price itself.
    pub fn value(&self, date: NaiveDate, market_repo_rate: f64) -> Result<f64, RustyQLibError> {
        if !market_repo_rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "repo",
                "the market repo rate must be finite",
            ));
        }
        self.check_date(date)?;
        match self.end {
            None => self.repurchase_price(date),
            Some(end) => {
                let remaining = self.day_count.year_fraction(date, end);
                Ok(self.end_cash()? / (1.0 + market_repo_rate * remaining))
            }
        }
    }

    /// Mark-to-market on `date` against the current market repo rate,
    /// from this side's point of view: the remaining claim repriced at
    /// the market rate less its value at the trade rate, so the lender
    /// gains when rates fall below the trade rate and the borrower when
    /// they rise, and the mark is exactly zero at the trade rate. Zero
    /// for an open repo.
    pub fn mark_to_market(
        &self,
        date: NaiveDate,
        market_repo_rate: f64,
    ) -> Result<f64, RustyQLibError> {
        let lender = self.value(date, market_repo_rate)? - self.value(date, self.repo_rate)?;
        Ok(-self.sign() * lender)
    }
}

impl FixedRateBond {
    /// Dirty forward price per 100 face on `forward_date` of a position
    /// bought at `clean_price` on `settlement` and financed in repo at
    /// `repo_rate` (simple interest on `day_count`), with interim
    /// coupons reinvested at the same rate — the futures implied-repo
    /// convention.
    pub fn forward_dirty_price(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        forward_date: NaiveDate,
        repo_rate: f64,
        day_count: DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        if forward_date <= settlement {
            return Err(RustyQLibError::invalid_input(
                "repo forward",
                format!("forward date {forward_date} must follow settlement {settlement}"),
            ));
        }
        if !(clean_price > 0.0 && clean_price.is_finite() && repo_rate.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "repo forward",
                "clean price must be positive and the repo rate finite",
            ));
        }
        let dirty = clean_price + self.accrued_interest(settlement)?;
        let financed =
            dirty * (1.0 + repo_rate * day_count.year_fraction(settlement, forward_date));
        let coupons: f64 = self
            .cashflows()
            .iter()
            .filter(|cf| cf.payment_date > settlement && cf.payment_date <= forward_date)
            .map(|cf| {
                let coupon =
                    (cf.amount - self.principal_at(cf.accrual_end)) * 100.0 / self.face_value;
                coupon * (1.0 + repo_rate * day_count.year_fraction(cf.payment_date, forward_date))
            })
            .sum();
        Ok(financed - coupons)
    }

    /// Clean forward price per 100 face: the dirty forward less the
    /// accrued on `forward_date`.
    pub fn forward_clean_price(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        forward_date: NaiveDate,
        repo_rate: f64,
        day_count: DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.forward_dirty_price(clean_price, settlement, forward_date, repo_rate, day_count)?
                - self.accrued_interest(forward_date)?,
        )
    }

    /// Carry per 100 face to `forward_date`: coupon income less the
    /// repo financing cost, which is the spot clean price less the
    /// clean forward. Positive when the coupon outruns the funding.
    pub fn carry(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        forward_date: NaiveDate,
        repo_rate: f64,
        day_count: DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        Ok(clean_price
            - self.forward_clean_price(
                clean_price,
                settlement,
                forward_date,
                repo_rate,
                day_count,
            )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::{BondFuture, DeliverableBond};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// A 4% ten-year Treasury.
    fn treasury() -> FixedRateBond {
        FixedRateBond::us_treasury(1000.0, 0.04, d(2026, 2, 15), d(2036, 2, 15)).unwrap()
    }

    /// 10mm par at 99 clean, 2% haircut, 5.3% for 30 days, starting
    /// after the 15 August coupon.
    fn repo() -> RepurchaseAgreement {
        RepurchaseAgreement::new(
            treasury(),
            10_000_000.0,
            99.0,
            0.02,
            0.053,
            d(2026, 8, 20),
            Some(d(2026, 9, 19)),
        )
        .unwrap()
    }

    #[test]
    fn cash_legs_follow_the_haircut_and_the_rate() {
        let r = repo();
        let accrued = r.collateral.accrued_interest(d(2026, 8, 20)).unwrap();
        let market_value = 10_000_000.0 * (99.0 + accrued) / 100.0;
        let purchase = r.purchase_price().unwrap();
        assert!((purchase - market_value * 0.98).abs() < 1e-6, "{purchase}");
        let end = r.end_cash().unwrap();
        assert!(
            (end - purchase * (1.0 + 0.053 * 30.0 / 360.0)).abs() < 1e-6,
            "{end}"
        );
        assert!((r.interest().unwrap() - (end - purchase)).abs() < 1e-9);
        // half way, half the interest
        let mid = r.repurchase_price(d(2026, 9, 4)).unwrap();
        assert!((mid - purchase * (1.0 + 0.053 * 15.0 / 360.0)).abs() < 1e-6);
        // the lender pays out at the start and gets the repurchase price back
        let flows = r.cash_flows().unwrap();
        assert_eq!(flows.len(), 2);
        assert!(flows[0].1 < 0.0 && (flows[0].1 + purchase).abs() < 1e-9);
        assert!((flows[1].1 - end).abs() < 1e-9);
        let mut borrower = repo();
        borrower.side = RepoSide::Repo;
        let flows = borrower.cash_flows().unwrap();
        assert!((flows[0].1 - purchase).abs() < 1e-9 && (flows[1].1 + end).abs() < 1e-9);
        // an Act/365 repo accrues less
        let mut sterling = repo();
        sterling.day_count = DayCountConvention::Act365;
        assert!(sterling.end_cash().unwrap() < end);
    }

    #[test]
    fn coupons_inside_the_term_are_passed_through() {
        // no coupon between 20 August and 19 September
        assert!(repo().manufactured_payments().is_empty());
        // a term over the 15 August coupon carries it, on the par
        let mut over_coupon = repo();
        over_coupon.start = d(2026, 8, 1);
        over_coupon.end = Some(d(2026, 8, 31));
        let payments = over_coupon.manufactured_payments();
        assert_eq!(payments.len(), 1);
        let (date, amount) = payments[0];
        assert!(date >= d(2026, 8, 15) && date <= d(2026, 8, 18), "{date}");
        // the lender passes 2% semiannual on 10mm through to the seller
        assert!((amount + 200_000.0).abs() < 1e-6, "{amount}");
        let mut seller = over_coupon.clone();
        seller.side = RepoSide::Repo;
        assert!((seller.manufactured_payments()[0].1 - 200_000.0).abs() < 1e-6);
    }

    #[test]
    fn exposure_and_margin_track_the_collateral() {
        let r = repo();
        let date = d(2026, 9, 4);
        // the haircut is netted into the purchase price, so the exposure
        // starts at zero and then drifts with the repo interest against
        // the collateral's own accrual
        assert!(r.exposure(r.start, 99.0).unwrap().abs() < 1e-9);
        let flat = r.exposure(date, 99.0).unwrap();
        let value = r.collateral_value(date, 99.0).unwrap();
        let expected = r.repurchase_price(date).unwrap() - value * 0.98;
        assert!((flat - expected).abs() < 1e-6, "{flat}");
        // a 5-point drop leaves the lender short, and the seller posts
        let dropped = r.exposure(date, 94.0).unwrap();
        assert!(dropped > 0.0, "{dropped}");
        assert!((dropped - flat - 10_000_000.0 * 5.0 / 100.0 * 0.98).abs() < 1e-6);
        // the lender receives the call; a threshold above it silences it
        assert!((r.margin_call(date, 94.0, 0.0).unwrap() - dropped).abs() < 1e-9);
        assert_eq!(r.margin_call(date, 94.0, 1e9).unwrap(), 0.0);
        let mut seller = repo();
        seller.side = RepoSide::Repo;
        assert!((seller.margin_call(date, 94.0, 0.0).unwrap() + dropped).abs() < 1e-9);
        assert!(r.margin_call(date, 94.0, -1.0).is_err());
        assert!(r.exposure(d(2027, 1, 1), 99.0).is_err());
    }

    #[test]
    fn mark_to_market_moves_against_the_market_rate() {
        let r = repo();
        let date = d(2026, 9, 4);
        // at the trade rate the claim is worth its accrual
        assert_eq!(r.mark_to_market(date, 0.053).unwrap(), 0.0);
        // rates up: the lender's fixed claim is worth less, the borrower gains
        let lender = r.mark_to_market(date, 0.06).unwrap();
        assert!(lender < 0.0, "{lender}");
        let mut borrower = repo();
        borrower.side = RepoSide::Repo;
        assert!((borrower.mark_to_market(date, 0.06).unwrap() + lender).abs() < 1e-9);
        // hand check: repurchase price discounted at 6% for the 15 days left
        let value = r.value(date, 0.06).unwrap();
        let expected = r.end_cash().unwrap() / (1.0 + 0.06 * 15.0 / 360.0);
        assert!((value - expected).abs() < 1e-6);
        // an open repo has no rate risk
        let mut open = repo();
        open.end = None;
        assert_eq!(open.mark_to_market(date, 0.06).unwrap(), 0.0);
        assert!(open.end_cash().is_err() && open.cash_flows().is_err());
        assert!(open.repurchase_price(d(2027, 8, 20)).unwrap() > open.purchase_price().unwrap());
    }

    #[test]
    fn repo_forward_agrees_with_the_futures_implied_repo() {
        let bond = treasury();
        let settlement = d(2026, 8, 20);
        let forward = d(2026, 12, 15); // no coupon until 2027-02-15
        let dc = DayCountConvention::Act360;
        // no coupon in the window: the dirty forward is the financed dirty price
        let dirty = 99.0 + bond.accrued_interest(settlement).unwrap();
        let fwd_dirty = bond
            .forward_dirty_price(99.0, settlement, forward, 0.053, dc)
            .unwrap();
        let days = (forward - settlement).num_days() as f64;
        assert!((fwd_dirty - dirty * (1.0 + 0.053 * days / 360.0)).abs() < 1e-9);
        let fwd_clean = bond
            .forward_clean_price(99.0, settlement, forward, 0.053, dc)
            .unwrap();
        assert!((fwd_clean - (fwd_dirty - bond.accrued_interest(forward).unwrap())).abs() < 1e-9);
        // a future priced at the repo forward implies exactly the repo rate
        let deliverable = DeliverableBond {
            bond: bond.clone(),
            conversion_factor: 0.85,
        };
        let future = BondFuture::new(forward, vec![deliverable.clone()]).unwrap();
        let implied = future
            .implied_repo(&deliverable, fwd_clean / 0.85, 99.0, settlement)
            .unwrap();
        assert!((implied - 0.053).abs() < 1e-9, "{implied}");
        // over a coupon the same holds, with the coupon reinvested
        let far = d(2027, 3, 15);
        let fwd_clean = bond
            .forward_clean_price(99.0, settlement, far, 0.053, dc)
            .unwrap();
        let future = BondFuture::new(far, vec![deliverable.clone()]).unwrap();
        let implied = future
            .implied_repo(&deliverable, fwd_clean / 0.85, 99.0, settlement)
            .unwrap();
        // (to a few millionths: the future reinvests the coupon from its
        // accrual end, the forward from its business-day-adjusted payment)
        assert!((implied - 0.053).abs() < 2e-5, "{implied}");
        // carry: a 4% coupon funded at 5.3% is negative carry, so the
        // forward sits above the spot clean price
        let carry = bond.carry(99.0, settlement, forward, 0.053, dc).unwrap();
        assert!(carry < 0.0, "{carry}");
        assert!(bond.carry(99.0, settlement, forward, 0.02, dc).unwrap() > 0.0);
        assert!(bond
            .forward_dirty_price(99.0, forward, settlement, 0.053, dc)
            .is_err());
    }

    #[test]
    fn validation_rejects_bad_terms() {
        let bond = treasury();
        let ok = |par, price, haircut, rate, end| {
            RepurchaseAgreement::new(bond.clone(), par, price, haircut, rate, d(2026, 8, 14), end)
        };
        assert!(ok(0.0, 99.0, 0.02, 0.05, Some(d(2026, 9, 13))).is_err());
        assert!(ok(1e6, -1.0, 0.02, 0.05, Some(d(2026, 9, 13))).is_err());
        assert!(ok(1e6, 99.0, 1.0, 0.05, Some(d(2026, 9, 13))).is_err());
        assert!(ok(1e6, 99.0, 0.02, f64::NAN, Some(d(2026, 9, 13))).is_err());
        assert!(ok(1e6, 99.0, 0.02, 0.05, Some(d(2026, 8, 14))).is_err());
        assert!(ok(1e6, 99.0, 0.02, 0.05, None).is_ok());
        assert!(repo().collateral_value(d(2026, 8, 20), 0.0).is_err());
        assert!(repo().value(d(2026, 8, 20), f64::INFINITY).is_err());
    }
}
