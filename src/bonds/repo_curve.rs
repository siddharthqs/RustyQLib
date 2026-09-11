//! The term repo curve: general-collateral financing rates by term,
//! with a per-issue specialness overlay.
//!
//! A term repo quote is a deposit on the financing side — cash placed
//! against collateral from a start date to an end date at a simple
//! rate — so a strip of them (overnight, one week, one month, three
//! months, ...) bootstraps into an ordinary [`YieldCurve`] of financing
//! discount factors through the same sequential bootstrap the discount
//! curves use. [`RepoCurve`] wraps that curve and adds the specialness
//! spread: an issue in demand in the repo market (the on-the-run, the
//! cheapest-to-deliver) finances *below* general collateral, by a
//! spread that is the issue's, not the curve's.
//!
//! The curve feeds the financing analytics that otherwise take a typed-
//! in rate: the repo forward and carry of a bond
//! ([`FixedRateBond::forward_clean_price_on_curve`]), the futures net
//! basis ([`BondFuture::net_basis_on_curve`]) and the mark-to-market of
//! a term repo ([`RepurchaseAgreement::mark_to_market_on_curve`]). The
//! overnight point is what the DTCC GCF index or SOFR publishes; a
//! SOFR discount curve can also stand in for the general-collateral
//! curve directly through [`RepoCurve::from_curve`].

use chrono::{Days, NaiveDate};

use crate::bonds::futures::{BondFuture, DeliverableBond};
use crate::bonds::repo::RepurchaseAgreement;
use crate::bonds::{bootstrap_curve, df_to_start, CurveInstrument, FixedRateBond};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

/// A general-collateral term repo quote: simple `rate` on `day_count`
/// from `start` to `end`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TermRepoQuote {
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub rate: f64,
    pub day_count: DayCountConvention,
}

impl TermRepoQuote {
    /// A quote on Act/360.
    pub fn new(start: NaiveDate, end: NaiveDate, rate: f64) -> Result<Self, RustyQLibError> {
        if end <= start {
            return Err(RustyQLibError::invalid_input(
                "repo quote",
                format!("end {end} must be after start {start}"),
            ));
        }
        if !rate.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "repo quote",
                format!("rate must be finite, got {rate}"),
            ));
        }
        Ok(TermRepoQuote {
            start,
            end,
            rate,
            day_count: DayCountConvention::Act360,
        })
    }

    /// An overnight quote from `start` to the next calendar day, the
    /// form the GCF index and SOFR publish (a percent there: divide by
    /// 100).
    pub fn overnight(start: NaiveDate, rate: f64) -> Result<Self, RustyQLibError> {
        let end = start
            .checked_add_days(Days::new(1))
            .ok_or_else(|| RustyQLibError::invalid_input("repo quote", "date overflow"))?;
        Self::new(start, end, rate)
    }

    /// Year fraction of the term under the quote's day count.
    pub fn accrual(&self) -> f64 {
        self.day_count.year_fraction(self.start, self.end)
    }
}

impl CurveInstrument for TermRepoQuote {
    fn maturity_date(&self) -> NaiveDate {
        self.end
    }

    fn implied_df(
        &self,
        reference_date: NaiveDate,
        curve_so_far: Option<&YieldCurve>,
    ) -> Result<f64, RustyQLibError> {
        let df_start = df_to_start("repo quote", self.start, reference_date, curve_so_far)?;
        Ok(df_start / (1.0 + self.rate * self.accrual()))
    }
}

/// The financing curve: general-collateral discount factors plus the
/// specialness of the issue being financed.
#[derive(Debug, Clone)]
pub struct RepoCurve {
    general: YieldCurve,
    /// How far below general collateral the issue finances, as a rate.
    specialness: f64,
}

impl RepoCurve {
    /// Bootstrap the general-collateral curve from term repo quotes.
    /// The first quote must start on `reference_date`; later ones may
    /// start forward. Times run on `day_count`.
    pub fn bootstrap(
        quotes: &[TermRepoQuote],
        reference_date: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<Self, RustyQLibError> {
        let instruments: Vec<Box<dyn CurveInstrument>> = quotes
            .iter()
            .map(|q| Box::new(*q) as Box<dyn CurveInstrument>)
            .collect();
        Ok(RepoCurve {
            general: bootstrap_curve(&instruments, reference_date, day_count)?,
            specialness: 0.0,
        })
    }

    /// Use an existing curve (a SOFR curve, say) as general collateral.
    pub fn from_curve(general: YieldCurve) -> Self {
        RepoCurve {
            general,
            specialness: 0.0,
        }
    }

    /// The same curve for an issue trading `spread` special: its
    /// financing rate is the general-collateral rate less `spread`
    /// (`0.0025` for 25bp special).
    pub fn with_specialness(self, spread: f64) -> Result<Self, RustyQLibError> {
        if !spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "repo curve",
                format!("the specialness spread must be finite, got {spread}"),
            ));
        }
        Ok(RepoCurve {
            specialness: spread,
            ..self
        })
    }

    pub fn general(&self) -> &YieldCurve {
        &self.general
    }

    pub fn specialness(&self) -> f64 {
        self.specialness
    }

    /// The general-collateral simple rate from `start` to `end` on
    /// `day_count`, read off the curve's discount factors.
    pub fn general_rate(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        if end <= start {
            return Err(RustyQLibError::invalid_input(
                "repo curve",
                format!("end {end} must be after start {start}"),
            ));
        }
        let ratio = self.general.df_date(start) / self.general.df_date(end);
        Ok((ratio - 1.0) / day_count.year_fraction(start, end))
    }

    /// The issue's financing rate from `start` to `end`: general
    /// collateral less the specialness.
    pub fn rate(
        &self,
        start: NaiveDate,
        end: NaiveDate,
        day_count: DayCountConvention,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.general_rate(start, end, day_count)? - self.specialness)
    }
}

impl RepurchaseAgreement {
    /// [`mark_to_market`](Self::mark_to_market) against the market rate
    /// the curve gives for the remaining term. Zero for an open repo.
    pub fn mark_to_market_on_curve(
        &self,
        date: NaiveDate,
        curve: &RepoCurve,
    ) -> Result<f64, RustyQLibError> {
        match self.end {
            None => self.mark_to_market(date, self.repo_rate),
            Some(end) => {
                if date >= end {
                    return Err(RustyQLibError::invalid_input(
                        "repo",
                        format!("{date} is not before the repurchase date {end}"),
                    ));
                }
                let market = curve.rate(date, end, self.day_count)?;
                self.mark_to_market(date, market)
            }
        }
    }
}

impl FixedRateBond {
    /// [`forward_clean_price`](Self::forward_clean_price) financed at
    /// the curve's rate for the term, Act/360.
    pub fn forward_clean_price_on_curve(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        forward_date: NaiveDate,
        curve: &RepoCurve,
    ) -> Result<f64, RustyQLibError> {
        let day_count = DayCountConvention::Act360;
        let rate = curve.rate(settlement, forward_date, day_count)?;
        self.forward_clean_price(clean_price, settlement, forward_date, rate, day_count)
    }

    /// [`carry`](Self::carry) financed at the curve's rate for the
    /// term, Act/360.
    pub fn carry_on_curve(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        forward_date: NaiveDate,
        curve: &RepoCurve,
    ) -> Result<f64, RustyQLibError> {
        let day_count = DayCountConvention::Act360;
        let rate = curve.rate(settlement, forward_date, day_count)?;
        self.carry(clean_price, settlement, forward_date, rate, day_count)
    }
}

impl BondFuture {
    /// [`net_basis`](Self::net_basis) financed at the curve's rate from
    /// settlement to delivery, Act/360.
    pub fn net_basis_on_curve(
        &self,
        deliverable: &DeliverableBond,
        futures_price: f64,
        clean_price: f64,
        settlement: NaiveDate,
        curve: &RepoCurve,
    ) -> Result<f64, RustyQLibError> {
        let rate = curve.rate(settlement, self.delivery_date, DayCountConvention::Act360)?;
        self.net_basis(deliverable, futures_price, clean_price, settlement, rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// Overnight 5.30%, then a rising term strip out to six months.
    fn strip() -> Vec<TermRepoQuote> {
        let t0 = d(2026, 8, 20);
        vec![
            TermRepoQuote::overnight(t0, 0.0530).unwrap(),
            TermRepoQuote::new(t0, d(2026, 8, 27), 0.0532).unwrap(),
            TermRepoQuote::new(t0, d(2026, 9, 21), 0.0535).unwrap(),
            TermRepoQuote::new(t0, d(2026, 11, 20), 0.0540).unwrap(),
            TermRepoQuote::new(t0, d(2027, 2, 22), 0.0545).unwrap(),
        ]
    }

    fn curve() -> RepoCurve {
        RepoCurve::bootstrap(&strip(), d(2026, 8, 20), DayCountConvention::Act365).unwrap()
    }

    #[test]
    fn bootstrapped_curve_reprices_its_quotes_and_interpolates_between() {
        let curve = curve();
        for quote in strip() {
            let rate = curve
                .general_rate(quote.start, quote.end, quote.day_count)
                .unwrap();
            assert!(
                (rate - quote.rate).abs() < 1e-12,
                "{rate} vs {}",
                quote.rate
            );
        }
        // a two-month term sits between the one- and three-month quotes
        let two_months = curve
            .general_rate(d(2026, 8, 20), d(2026, 10, 20), DayCountConvention::Act360)
            .unwrap();
        assert!(two_months > 0.0535 && two_months < 0.0540, "{two_months}");
        // and a forward-starting term reads the forward rate: above the
        // three-month quote, though below the six-month one, since a
        // simple forward is deflated by the first period's accrual
        let forward = curve
            .general_rate(d(2026, 11, 20), d(2027, 2, 22), DayCountConvention::Act360)
            .unwrap();
        assert!(forward > 0.0540 && forward < 0.0545, "{forward}");
        assert!(curve
            .general_rate(d(2026, 9, 1), d(2026, 9, 1), DayCountConvention::Act360)
            .is_err());
    }

    #[test]
    fn specialness_lowers_the_financing_rate_and_the_forward() {
        let general = curve();
        let special = curve().with_specialness(0.0025).unwrap();
        let (s, e) = (d(2026, 8, 20), d(2026, 11, 20));
        let dc = DayCountConvention::Act360;
        assert!(
            (special.rate(s, e, dc).unwrap() - (general.rate(s, e, dc).unwrap() - 0.0025)).abs()
                < 1e-15
        );
        assert_eq!(special.specialness(), 0.0025);
        // cheaper financing: a lower forward and more carry
        let bond =
            FixedRateBond::us_treasury(1000.0, 0.04, d(2026, 2, 15), d(2036, 2, 15)).unwrap();
        let on_general = bond
            .forward_clean_price_on_curve(99.0, s, e, &general)
            .unwrap();
        let on_special = bond
            .forward_clean_price_on_curve(99.0, s, e, &special)
            .unwrap();
        assert!(on_special < on_general, "{on_special} vs {on_general}");
        assert!(
            bond.carry_on_curve(99.0, s, e, &special).unwrap()
                > bond.carry_on_curve(99.0, s, e, &general).unwrap()
        );
        // and the forward equals the scalar version at the curve's rate
        let rate = general.rate(s, e, dc).unwrap();
        let scalar = bond.forward_clean_price(99.0, s, e, rate, dc).unwrap();
        assert!((on_general - scalar).abs() < 1e-12);
        assert!(curve().with_specialness(f64::NAN).is_err());
    }

    #[test]
    fn futures_basis_and_repo_mark_read_the_curve() {
        let curve = curve();
        let bond =
            FixedRateBond::us_treasury(1000.0, 0.04, d(2026, 2, 15), d(2036, 2, 15)).unwrap();
        let settlement = d(2026, 8, 20);
        let delivery = d(2026, 12, 15);
        let deliverable = DeliverableBond {
            bond: bond.clone(),
            conversion_factor: 0.85,
        };
        let future = BondFuture::new(delivery, vec![deliverable.clone()]).unwrap();
        // a future at the curve's repo forward has no net basis
        let fair = bond
            .forward_clean_price_on_curve(99.0, settlement, delivery, &curve)
            .unwrap()
            / 0.85;
        let basis = future
            .net_basis_on_curve(&deliverable, fair, 99.0, settlement, &curve)
            .unwrap();
        assert!(basis.abs() < 1e-6, "{basis}");
        // a cheaper future is a positive net basis (the delivery is rich)
        assert!(
            future
                .net_basis_on_curve(&deliverable, fair - 0.5, 99.0, settlement, &curve)
                .unwrap()
                > 0.0
        );

        // a term repo marked on the curve: the scalar mark at the curve's
        // rate for the remaining term, positive for the lender when the
        // curve sits below the trade rate
        let repo = RepurchaseAgreement::new(
            bond,
            10_000_000.0,
            99.0,
            0.02,
            0.0560,
            settlement,
            Some(d(2026, 11, 20)),
        )
        .unwrap();
        let date = d(2026, 9, 20);
        let on_curve = repo.mark_to_market_on_curve(date, &curve).unwrap();
        let rate = curve
            .rate(date, d(2026, 11, 20), DayCountConvention::Act360)
            .unwrap();
        assert!((on_curve - repo.mark_to_market(date, rate).unwrap()).abs() < 1e-9);
        assert!(on_curve > 0.0, "{on_curve}");
        assert!(repo
            .mark_to_market_on_curve(d(2026, 11, 20), &curve)
            .is_err());
        let mut open = repo.clone();
        open.end = None;
        assert_eq!(open.mark_to_market_on_curve(date, &curve).unwrap(), 0.0);
    }

    #[test]
    fn a_sofr_curve_can_stand_in_for_general_collateral() {
        let sofr = YieldCurve::flat(
            0.053,
            d(2026, 8, 20),
            DayCountConvention::Act365,
            crate::core::curves::Compounding::Continuous,
        )
        .unwrap();
        let curve = RepoCurve::from_curve(sofr);
        let rate = curve
            .general_rate(d(2026, 8, 20), d(2026, 11, 20), DayCountConvention::Act360)
            .unwrap();
        // a 5.30% continuous Act/365 curve reads as a lower simple
        // Act/360 rate (the 360-day basis, less the compounding pickup)
        assert!(rate > 0.052 && rate < 0.053, "{rate}");
        assert!(TermRepoQuote::new(d(2026, 8, 20), d(2026, 8, 20), 0.05).is_err());
        assert!(TermRepoQuote::new(d(2026, 8, 20), d(2026, 8, 27), f64::NAN).is_err());
        assert!(RepoCurve::bootstrap(&[], d(2026, 8, 20), DayCountConvention::Act365).is_err());
    }
}
