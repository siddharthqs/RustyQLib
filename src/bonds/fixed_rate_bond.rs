//! Fixed-rate coupon bond with US-Treasury-style analytics.
//!
//! Prices, accrued interest, DV01 and durations are quoted **per 100
//! face** (the market convention); [`cashflows`](FixedRateBond::cashflows)
//! and [`pv`](FixedRateBond::pv) work in absolute amounts on
//! `face_value`.
//!
//! Yield analytics follow the street convention: the yield compounds
//! `frequency` times per year and every cash flow at scheduled date `d_k`
//! is discounted by `(1 + y/f)^-(w + k)`, where `w` is the Act/Act ICMA
//! fraction of the current coupon period remaining at settlement.
//! Accrual runs between *scheduled* (unadjusted) coupon dates; payment
//! dates are business-day adjusted separately and used for
//! curve discounting.

use chrono::NaiveDate;

use crate::bonds::schedule::{coupon_dates, is_end_of_month, CouponSchedule};
use crate::bonds::Frequency;
use crate::core::calendar::{BusinessDayConvention, Calendar};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// One bond cash flow. The final flow contains the redemption amount on
/// top of the last coupon.
#[derive(Debug, Clone, PartialEq)]
pub struct Cashflow {
    /// Scheduled accrual period start (unadjusted).
    pub accrual_start: NaiveDate,
    /// Scheduled accrual period end / coupon date (unadjusted).
    pub accrual_end: NaiveDate,
    /// Business-day adjusted payment date.
    pub payment_date: NaiveDate,
    /// Absolute amount on `face_value`.
    pub amount: f64,
}

/// A fixed-rate bullet bond.
#[derive(Debug, Clone)]
pub struct FixedRateBond {
    pub face_value: f64,
    /// Annual coupon rate (e.g. `0.04125` for 4 1/8s).
    pub coupon_rate: f64,
    pub frequency: Frequency,
    /// Interest accrual start (the dated date).
    pub dated_date: NaiveDate,
    pub maturity_date: NaiveDate,
    pub day_count: DayCountConvention,
    pub calendar: Calendar,
    /// Adjustment applied to payment dates (accrual stays unadjusted).
    pub payment_convention: BusinessDayConvention,
    /// Business days from trade to settlement (1 for US Treasuries).
    pub settlement_days: i64,
    /// Snap all coupon dates to month-ends (bonds maturing on one).
    pub end_of_month: bool,
    schedule: CouponSchedule,
}

/// One coupon accrual period with its ICMA reference period.
#[derive(Debug, Clone, Copy)]
struct Period {
    start: NaiveDate,
    end: NaiveDate,
    /// Reference (quasi) period start: differs from `start` only for a
    /// short front stub.
    ref_start: NaiveDate,
}

/// A remaining cash flow prepared for yield math: discount by
/// `(1 + y/f)^-tau`.
#[derive(Debug, Clone, Copy)]
struct Flow {
    /// Exponent in coupon periods: `w` for the next coupon, `w + 1` for
    /// the one after, ...
    tau: f64,
    /// Absolute amount (redemption folded into the last flow).
    amount: f64,
}

impl FixedRateBond {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        face_value: f64,
        coupon_rate: f64,
        frequency: Frequency,
        dated_date: NaiveDate,
        maturity_date: NaiveDate,
        day_count: DayCountConvention,
        calendar: Calendar,
        payment_convention: BusinessDayConvention,
        settlement_days: i64,
        end_of_month: bool,
    ) -> Result<Self, RustyQLibError> {
        if !face_value.is_finite() || face_value <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("face value must be positive, got {face_value}"),
            ));
        }
        if !coupon_rate.is_finite() || coupon_rate < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("coupon rate must be non-negative, got {coupon_rate}"),
            ));
        }
        if settlement_days < 0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("settlement days must be non-negative, got {settlement_days}"),
            ));
        }
        let schedule = coupon_dates(dated_date, maturity_date, frequency.months(), end_of_month)?;
        Ok(FixedRateBond {
            face_value,
            coupon_rate,
            frequency,
            dated_date,
            maturity_date,
            day_count,
            calendar,
            payment_convention,
            settlement_days,
            end_of_month,
            schedule,
        })
    }

    /// A US Treasury note/bond: semiannual Act/Act ICMA coupons on the
    /// SIFMA bond-market calendar, payments rolled forward, T+1
    /// settlement, and the end-of-month rule when maturity is a
    /// month-end.
    pub fn us_treasury(
        face_value: f64,
        coupon_rate: f64,
        dated_date: NaiveDate,
        maturity_date: NaiveDate,
    ) -> Result<Self, RustyQLibError> {
        Self::new(
            face_value,
            coupon_rate,
            Frequency::Semiannual,
            dated_date,
            maturity_date,
            DayCountConvention::ActActIcma,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::Following,
            1,
            is_end_of_month(maturity_date),
        )
    }

    /// Scheduled (unadjusted) coupon dates, ending at maturity.
    pub fn coupon_dates(&self) -> &[NaiveDate] {
        &self.schedule.dates
    }

    /// Settlement date for a trade done on `trade_date`.
    pub fn settlement_date(&self, trade_date: NaiveDate) -> NaiveDate {
        self.calendar
            .add_business_days(trade_date, self.settlement_days)
    }

    /// All cash flows of the bond, in order; the last one includes the
    /// redemption of `face_value`.
    pub fn cashflows(&self) -> Vec<Cashflow> {
        let periods = self.periods();
        let n = periods.len();
        periods
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut amount = self.coupon_amount(p);
                if i == n - 1 {
                    amount += self.face_value;
                }
                Cashflow {
                    accrual_start: p.start,
                    accrual_end: p.end,
                    payment_date: self.calendar.adjust(p.end, self.payment_convention),
                    amount,
                }
            })
            .collect()
    }

    /// Accrued interest per 100 face at `settlement` (street convention:
    /// the coupon prorated by Act/Act ICMA days within the current
    /// period). Zero at the dated date and on each coupon date.
    pub fn accrued_interest(&self, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        self.check_settlement(settlement)?;
        if settlement <= self.dated_date {
            return Ok(0.0);
        }
        let period = self
            .periods()
            .into_iter()
            .find(|p| settlement < p.end)
            .expect("settlement is before maturity");
        if settlement <= period.start {
            return Ok(0.0);
        }
        let fraction = self.day_count.year_fraction_icma(
            period.start,
            settlement,
            period.ref_start,
            period.end,
            self.frequency.per_year(),
        );
        Ok(100.0 * self.coupon_rate * fraction)
    }

    // ── Yield analytics (street convention) ─────────────────────────────

    /// Dirty (invoice) price per 100 face at `settlement` for a given
    /// street-convention yield.
    pub fn dirty_price_from_yield(
        &self,
        yield_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let flows = self.remaining_flows(settlement)?;
        Ok(self.per_100(self.pv_flows(&flows, yield_rate)?))
    }

    /// Clean (quoted) price per 100 face: dirty minus accrued.
    pub fn clean_price_from_yield(
        &self,
        yield_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.dirty_price_from_yield(yield_rate, settlement)?
            - self.accrued_interest(settlement)?)
    }

    /// Street-convention yield from a clean price per 100 face, solved
    /// with a safeguarded Newton iteration.
    pub fn yield_from_clean_price(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let target_dirty = clean_price + self.accrued_interest(settlement)?;
        if !target_dirty.is_finite() || target_dirty <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("dirty price must be positive, got {target_dirty}"),
            ));
        }
        let flows = self.remaining_flows(settlement)?;
        let f = self.frequency.per_year() as f64;
        // g(y) = target - dirty(y) is increasing in y; bracket wide:
        // y > -f keeps 1 + y/f positive.
        let (lo, hi) = (-0.99 * f, 10.0);
        let g = |y: f64| {
            target_dirty
                - self.per_100(
                    self.pv_flows(&flows, y)
                        .expect("bracket keeps 1 + y/f positive"),
                )
        };
        let dg = |y: f64| -self.per_100(self.dpv_dy(&flows, y));
        let root = Solver1d::new(1e-10, 100).newton_safeguarded(g, dg, lo, hi, self.coupon_rate);
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: g(root.x).abs(),
                reason: "yield solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }

    /// Macaulay duration in years at the given yield.
    pub fn macaulay_duration(
        &self,
        yield_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let flows = self.remaining_flows(settlement)?;
        let f = self.frequency.per_year() as f64;
        let base = Self::check_base(yield_rate, f)?;
        let pv = self.pv_flows(&flows, yield_rate)?;
        let weighted: f64 = flows
            .iter()
            .map(|flow| (flow.tau / f) * flow.amount * base.powf(-flow.tau))
            .sum();
        Ok(weighted / pv)
    }

    /// Modified duration: Macaulay / (1 + y/f). `-1/P dP/dy`.
    pub fn modified_duration(
        &self,
        yield_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let f = self.frequency.per_year() as f64;
        Ok(self.macaulay_duration(yield_rate, settlement)? / (1.0 + yield_rate / f))
    }

    /// Convexity in years²: `1/P d²P/dy²`.
    pub fn convexity(&self, yield_rate: f64, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        let flows = self.remaining_flows(settlement)?;
        let f = self.frequency.per_year() as f64;
        let base = Self::check_base(yield_rate, f)?;
        let pv = self.pv_flows(&flows, yield_rate)?;
        let second: f64 = flows
            .iter()
            .map(|flow| {
                flow.amount * flow.tau * (flow.tau + 1.0) / (f * f) * base.powf(-flow.tau - 2.0)
            })
            .sum();
        Ok(second / pv)
    }

    /// Price change per 100 face for a one-basis-point yield move
    /// (modified duration × dirty price / 10 000).
    pub fn dv01(&self, yield_rate: f64, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        let dirty = self.dirty_price_from_yield(yield_rate, settlement)?;
        let modified = self.modified_duration(yield_rate, settlement)?;
        Ok(modified * dirty / 10_000.0)
    }

    // ── Curve pricing ───────────────────────────────────────────────────

    /// Present value at the curve's reference date of all payments
    /// strictly after it (absolute, on `face_value`).
    pub fn pv(&self, curve: &YieldCurve) -> f64 {
        self.cashflows()
            .iter()
            .filter(|cf| cf.payment_date > curve.reference_date())
            .map(|cf| cf.amount * curve.df_date(cf.payment_date))
            .sum()
    }

    /// Dirty price per 100 face at `settlement`, discounting each
    /// remaining payment on `curve` and compounding the result forward to
    /// settlement.
    pub fn dirty_price_from_curve(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.check_settlement(settlement)?;
        let df_settlement = curve.df_date(settlement);
        if !df_settlement.is_finite() || df_settlement <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive discount factor {df_settlement} at settlement {settlement}"
            )));
        }
        let pv: f64 = self
            .cashflows()
            .iter()
            .filter(|cf| cf.accrual_end > settlement)
            .map(|cf| cf.amount * curve.df_date(cf.payment_date))
            .sum();
        Ok(self.per_100(pv / df_settlement))
    }

    /// Clean price per 100 face off a discount curve.
    pub fn clean_price_from_curve(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.dirty_price_from_curve(curve, settlement)? - self.accrued_interest(settlement)?)
    }

    // ── Internals ───────────────────────────────────────────────────────

    /// The accrual periods with their ICMA reference starts. Only the
    /// first period can be a stub (short first coupon).
    fn periods(&self) -> Vec<Period> {
        let dates = &self.schedule.dates;
        let mut periods = Vec::with_capacity(dates.len());
        periods.push(Period {
            start: self.dated_date,
            end: dates[0],
            ref_start: self.schedule.prev_anchor,
        });
        for w in dates.windows(2) {
            periods.push(Period {
                start: w[0],
                end: w[1],
                ref_start: w[0],
            });
        }
        periods
    }

    /// Coupon interest paid at the end of `period` (absolute).
    fn coupon_amount(&self, period: &Period) -> f64 {
        let fraction = self.day_count.year_fraction_icma(
            period.start,
            period.end,
            period.ref_start,
            period.end,
            self.frequency.per_year(),
        );
        self.face_value * self.coupon_rate * fraction
    }

    /// Cash flows after `settlement` with their discount exponents
    /// `tau = w + k` in coupon periods.
    fn remaining_flows(&self, settlement: NaiveDate) -> Result<Vec<Flow>, RustyQLibError> {
        self.check_settlement(settlement)?;
        let periods = self.periods();
        let next = periods
            .iter()
            .position(|p| p.end > settlement)
            .expect("settlement is before maturity");
        let current = &periods[next];
        // fraction of a coupon period remaining at settlement. Under
        // Act/Act ICMA this is days-to-coupon over the *reference* period
        // length (the street convention, also correct for a short first
        // coupon); other day counts use the ratio within the period.
        let f = self.frequency.per_year();
        let remaining = self.day_count.year_fraction_icma(
            settlement,
            current.end,
            current.ref_start,
            current.end,
            f,
        );
        let w = if self.day_count == DayCountConvention::ActActIcma {
            remaining * f as f64
        } else {
            let full = self.day_count.year_fraction(current.start, current.end);
            if full <= 0.0 {
                return Err(RustyQLibError::NumericalError(format!(
                    "degenerate coupon period ending {}",
                    current.end
                )));
            }
            remaining / full
        };

        let n = periods.len();
        let flows = periods[next..]
            .iter()
            .enumerate()
            .map(|(k, p)| {
                let mut amount = self.coupon_amount(p);
                if next + k == n - 1 {
                    amount += self.face_value;
                }
                Flow {
                    tau: w + k as f64,
                    amount,
                }
            })
            .collect();
        Ok(flows)
    }

    /// `1 + y/f`, rejecting yields at or below `-f`.
    fn check_base(yield_rate: f64, f: f64) -> Result<f64, RustyQLibError> {
        let base = 1.0 + yield_rate / f;
        if !yield_rate.is_finite() || base <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("yield {yield_rate} is out of range (needs 1 + y/f > 0)"),
            ));
        }
        Ok(base)
    }

    /// Present value of `flows` at the street-convention yield (absolute).
    fn pv_flows(&self, flows: &[Flow], yield_rate: f64) -> Result<f64, RustyQLibError> {
        let f = self.frequency.per_year() as f64;
        let base = Self::check_base(yield_rate, f)?;
        Ok(flows
            .iter()
            .map(|flow| flow.amount * base.powf(-flow.tau))
            .sum())
    }

    /// `d/dy` of [`pv_flows`](Self::pv_flows) (absolute). Callers ensure
    /// `1 + y/f > 0`.
    fn dpv_dy(&self, flows: &[Flow], yield_rate: f64) -> f64 {
        let f = self.frequency.per_year() as f64;
        let base = 1.0 + yield_rate / f;
        flows
            .iter()
            .map(|flow| -flow.amount * flow.tau / f * base.powf(-flow.tau - 1.0))
            .sum()
    }

    fn per_100(&self, absolute: f64) -> f64 {
        absolute * 100.0 / self.face_value
    }

    fn check_settlement(&self, settlement: NaiveDate) -> Result<(), RustyQLibError> {
        if settlement >= self.maturity_date {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!(
                    "settlement {settlement} is on or after maturity {}",
                    self.maturity_date
                ),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// 5% semiannual two-year note on the May 15 / Nov 15 cycle.
    fn five_pct_two_year() -> FixedRateBond {
        FixedRateBond::us_treasury(100.0, 0.05, d(2026, 5, 15), d(2028, 5, 15)).unwrap()
    }

    #[test]
    fn cashflows_are_regular_coupons_plus_redemption() {
        let bond = five_pct_two_year();
        let cfs = bond.cashflows();
        assert_eq!(cfs.len(), 4);
        for cf in &cfs[..3] {
            assert!((cf.amount - 2.5).abs() < 1e-12, "coupon {}", cf.amount);
        }
        assert!((cfs[3].amount - 102.5).abs() < 1e-12);
        assert_eq!(cfs[3].accrual_end, d(2028, 5, 15));
        // Nov 15 2026 is a Sunday: payment rolls to Monday Nov 16,
        // accrual date stays put
        assert_eq!(cfs[0].accrual_end, d(2026, 11, 15));
        assert_eq!(cfs[0].payment_date, d(2026, 11, 16));
    }

    #[test]
    fn accrued_interest_street_convention() {
        let bond = five_pct_two_year();
        // May 15 -> Nov 15 2026 is 184 days; settle Jul 15 = 61 days in
        let accrued = bond.accrued_interest(d(2026, 7, 15)).unwrap();
        assert!((accrued - 2.5 * 61.0 / 184.0).abs() < 1e-12, "{accrued}");
        // zero at the dated date and at a coupon date
        assert_eq!(bond.accrued_interest(d(2026, 5, 15)).unwrap(), 0.0);
        assert_eq!(bond.accrued_interest(d(2026, 11, 15)).unwrap(), 0.0);
        // nearly a full coupon the day before payment (183/184)
        let almost = bond.accrued_interest(d(2026, 11, 14)).unwrap();
        assert!((almost - 2.5 * 183.0 / 184.0).abs() < 1e-12);
    }

    #[test]
    fn par_bond_prices_at_par_on_a_coupon_date() {
        let bond = five_pct_two_year();
        for settle in [d(2026, 5, 15), d(2026, 11, 15), d(2027, 11, 15)] {
            let clean = bond.clean_price_from_yield(0.05, settle).unwrap();
            assert!((clean - 100.0).abs() < 1e-10, "{settle}: {clean}");
        }
    }

    #[test]
    fn one_year_bond_prices_by_hand() {
        // 4% semiannual, settle on the dated date, yield 6%:
        // P = 2/1.03 + 102/1.03^2
        let bond = FixedRateBond::us_treasury(100.0, 0.04, d(2026, 5, 15), d(2027, 5, 15)).unwrap();
        let dirty = bond.dirty_price_from_yield(0.06, d(2026, 5, 15)).unwrap();
        let expected = 2.0 / 1.03 + 102.0 / (1.03_f64 * 1.03);
        assert!((dirty - expected).abs() < 1e-12, "{dirty} vs {expected}");
        // clean = dirty on the dated date
        let clean = bond.clean_price_from_yield(0.06, d(2026, 5, 15)).unwrap();
        assert_eq!(clean, dirty);
    }

    #[test]
    fn yield_round_trips_through_price() {
        let bond = five_pct_two_year();
        let settle = d(2026, 8, 3);
        for y in [-0.005, 0.01, 0.045, 0.05, 0.08, 0.15] {
            let clean = bond.clean_price_from_yield(y, settle).unwrap();
            let back = bond.yield_from_clean_price(clean, settle).unwrap();
            assert!((back - y).abs() < 1e-9, "y={y}, back={back}");
        }
    }

    #[test]
    fn durations_and_convexity_match_finite_differences() {
        let bond = five_pct_two_year();
        let settle = d(2026, 8, 3);
        let y = 0.045;
        let h = 1e-6;
        let p = bond.dirty_price_from_yield(y, settle).unwrap();
        let p_up = bond.dirty_price_from_yield(y + h, settle).unwrap();
        let p_dn = bond.dirty_price_from_yield(y - h, settle).unwrap();

        let num_mod = -(p_up - p_dn) / (2.0 * h) / p;
        let ana_mod = bond.modified_duration(y, settle).unwrap();
        assert!((num_mod - ana_mod).abs() < 1e-6, "{num_mod} vs {ana_mod}");

        let num_cvx = (p_up - 2.0 * p + p_dn) / (h * h) / p;
        let ana_cvx = bond.convexity(y, settle).unwrap();
        assert!((num_cvx - ana_cvx).abs() < 1e-3, "{num_cvx} vs {ana_cvx}");

        // Macaulay = modified * (1 + y/2)
        let mac = bond.macaulay_duration(y, settle).unwrap();
        assert!((mac - ana_mod * (1.0 + y / 2.0)).abs() < 1e-12);

        // DV01 approximates the actual 1bp move
        let dv01 = bond.dv01(y, settle).unwrap();
        let actual = bond.dirty_price_from_yield(y - 1e-4, settle).unwrap() - p;
        assert!((dv01 - actual).abs() < 1e-4, "{dv01} vs {actual}");
    }

    #[test]
    fn zero_coupon_duration_equals_time_to_maturity() {
        let bond = FixedRateBond::us_treasury(100.0, 0.0, d(2026, 5, 15), d(2028, 5, 15)).unwrap();
        let settle = d(2026, 5, 15);
        let y = 0.05;
        // single flow at tau = 4 halves -> Macaulay = 2 years exactly
        let mac = bond.macaulay_duration(y, settle).unwrap();
        assert!((mac - 2.0).abs() < 1e-12);
        let modified = bond.modified_duration(y, settle).unwrap();
        assert!((modified - 2.0 / 1.025).abs() < 1e-12);
        // price is the pure discount 100 / 1.025^4
        let clean = bond.clean_price_from_yield(y, settle).unwrap();
        assert!((clean - 100.0 / 1.025_f64.powi(4)).abs() < 1e-10);
    }

    #[test]
    fn short_first_coupon_is_prorated_icma() {
        // dated Jul 1 inside the May 15 / Nov 15 cycle: first coupon
        // accrues Jul 1 -> Nov 15 (137 days) against a 184-day quasi period
        let bond = FixedRateBond::us_treasury(100.0, 0.04, d(2026, 7, 1), d(2027, 5, 15)).unwrap();
        let cfs = bond.cashflows();
        assert_eq!(cfs.len(), 2);
        assert!((cfs[0].amount - 100.0 * 0.04 * 137.0 / 368.0).abs() < 1e-12);
        // the second period is regular
        assert!((cfs[1].amount - (2.0 + 100.0)).abs() < 1e-12);
        // accrued mid-stub: Jul 1 -> Aug 1 is 31 days
        let accrued = bond.accrued_interest(d(2026, 8, 1)).unwrap();
        assert!((accrued - 100.0 * 0.04 * 31.0 / 368.0).abs() < 1e-12);
        // street discounting at the dated date: the first coupon sits
        // 137/184 of a quasi period away, the second one period later
        let w = 137.0 / 184.0;
        let v = 1.0 / 1.02_f64; // y = 4%, semiannual
        let expected = cfs[0].amount * v.powf(w) + cfs[1].amount * v.powf(w + 1.0);
        let dirty = bond.dirty_price_from_yield(0.04, d(2026, 7, 1)).unwrap();
        assert!((dirty - expected).abs() < 1e-12, "{dirty} vs {expected}");
    }

    #[test]
    fn curve_pricing_is_consistent_with_manual_discounting() {
        let bond = five_pct_two_year();
        let reference = d(2026, 8, 3);
        let curve = YieldCurve::flat(
            0.04,
            reference,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        // pv = sum of cf * df at the adjusted payment dates
        let manual: f64 = bond
            .cashflows()
            .iter()
            .map(|cf| cf.amount * curve.df_date(cf.payment_date))
            .sum();
        assert!((bond.pv(&curve) - manual).abs() < 1e-12);
        // settling on the reference date, dirty price is just pv per 100
        let dirty = bond.dirty_price_from_curve(&curve, reference).unwrap();
        assert!((dirty - manual).abs() < 1e-12);
        // clean + accrued = dirty
        let clean = bond.clean_price_from_curve(&curve, reference).unwrap();
        let accrued = bond.accrued_interest(reference).unwrap();
        assert!((clean + accrued - dirty).abs() < 1e-12);
    }

    #[test]
    fn us_treasury_settles_t_plus_1_on_the_bond_calendar() {
        let bond = five_pct_two_year();
        // trade Friday Oct 9 2026: Monday Oct 12 is Columbus Day (bond
        // market closed) -> settles Tuesday Oct 13
        assert_eq!(bond.settlement_date(d(2026, 10, 9)), d(2026, 10, 13));
    }

    #[test]
    fn eom_flag_follows_the_maturity_date() {
        let eom = FixedRateBond::us_treasury(100.0, 0.04, d(2026, 6, 30), d(2028, 6, 30)).unwrap();
        assert!(eom.end_of_month);
        assert_eq!(eom.coupon_dates()[0], d(2026, 12, 31));
        let mid = five_pct_two_year();
        assert!(!mid.end_of_month);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let ok = |face: f64, rate: f64, days: i64| {
            FixedRateBond::new(
                face,
                rate,
                Frequency::Semiannual,
                d(2026, 5, 15),
                d(2028, 5, 15),
                DayCountConvention::ActActIcma,
                Calendar::UsGovernmentBond,
                BusinessDayConvention::Following,
                days,
                false,
            )
        };
        assert!(ok(0.0, 0.05, 1).is_err());
        assert!(ok(100.0, -0.01, 1).is_err());
        assert!(ok(100.0, 0.05, -1).is_err());
        // settlement past maturity
        let bond = five_pct_two_year();
        assert!(bond.accrued_interest(d(2028, 5, 15)).is_err());
        assert!(bond.dirty_price_from_yield(0.05, d(2029, 1, 1)).is_err());
        // yield below -f
        assert!(bond.dirty_price_from_yield(-2.5, d(2026, 8, 3)).is_err());
    }
}
