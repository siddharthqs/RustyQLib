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

/// A fixed-rate bond: bullet by default, with optional step-up
/// coupons ([`with_coupon_steps`](FixedRateBond::with_coupon_steps))
/// and a sinking fund
/// ([`with_sinking_fund`](FixedRateBond::with_sinking_fund)).
///
/// Prices, accrued and yields are quoted per 100 of the **outstanding**
/// face at settlement (the factor-adjusted trading convention); for a
/// bullet that is simply per 100 face.
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
    /// `(from_date, annual_rate)` coupon steps: from each date the rate
    /// applying to periods **starting** on or after it. Empty = flat.
    coupon_steps: Vec<(NaiveDate, f64)>,
    /// `(coupon_date, fraction_of_original_face)` sinking-fund
    /// redemptions; the remainder redeems at maturity. Empty = bullet.
    sinking_fund: Vec<(NaiveDate, f64)>,
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

/// One entry of a call schedule: the issuer may redeem on `call_date`
/// at `call_price` per 100 face (plus accrued).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallOption {
    pub call_date: NaiveDate,
    pub call_price: f64,
}

/// One entry of a put schedule: the holder may demand redemption on
/// `put_date` at `put_price` per 100 face (plus accrued).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PutOption {
    pub put_date: NaiveDate,
    pub put_price: f64,
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
            coupon_steps: Vec::new(),
            sinking_fund: Vec::new(),
        })
    }

    /// Add step-up (or step-down) coupons: from each `(date, rate)` the
    /// annual rate applying to coupon periods **starting** on or after
    /// that date (the rate of a period is set at its start, so a step
    /// inside a period takes effect from the next one).
    pub fn with_coupon_steps(mut self, steps: &[(NaiveDate, f64)]) -> Result<Self, RustyQLibError> {
        for &(date, rate) in steps {
            if !rate.is_finite() || rate < 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "coupon_steps",
                    format!("step rates must be non-negative, got {rate} at {date}"),
                ));
            }
            if date <= self.dated_date || date >= self.maturity_date {
                return Err(RustyQLibError::invalid_input(
                    "coupon_steps",
                    format!(
                        "step date {date} must lie strictly between the dated date and maturity"
                    ),
                ));
            }
        }
        if steps.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(RustyQLibError::invalid_input(
                "coupon_steps",
                "step dates must be strictly increasing",
            ));
        }
        self.coupon_steps = steps.to_vec();
        Ok(self)
    }

    /// Add a sinking fund: on each `(coupon_date, fraction)` the issuer
    /// repays that fraction of the **original** face; the remainder
    /// redeems at maturity. Redemption dates must be scheduled coupon
    /// dates before maturity, and the fractions must sum to at most 1.
    pub fn with_sinking_fund(
        mut self,
        redemptions: &[(NaiveDate, f64)],
    ) -> Result<Self, RustyQLibError> {
        let mut total = 0.0;
        for &(date, fraction) in redemptions {
            if !fraction.is_finite() || fraction <= 0.0 {
                return Err(RustyQLibError::invalid_input(
                    "sinking_fund",
                    format!("fractions must be positive, got {fraction} at {date}"),
                ));
            }
            if date >= self.maturity_date || !self.schedule.dates.contains(&date) {
                return Err(RustyQLibError::invalid_input(
                    "sinking_fund",
                    format!(
                        "redemption date {date} must be a scheduled coupon date before maturity"
                    ),
                ));
            }
            total += fraction;
        }
        if redemptions.windows(2).any(|w| w[1].0 <= w[0].0) {
            return Err(RustyQLibError::invalid_input(
                "sinking_fund",
                "redemption dates must be strictly increasing",
            ));
        }
        if total > 1.0 + 1e-12 {
            return Err(RustyQLibError::invalid_input(
                "sinking_fund",
                format!("redemption fractions sum to {total}, above the face"),
            ));
        }
        self.sinking_fund = redemptions.to_vec();
        Ok(self)
    }

    /// The annual coupon rate applying to a period starting at `date`.
    fn rate_for(&self, date: NaiveDate) -> f64 {
        self.coupon_steps
            .iter()
            .rev()
            .find(|&&(from, _)| from <= date)
            .map_or(self.coupon_rate, |&(_, rate)| rate)
    }

    /// Outstanding face after all sinking-fund payments on or before
    /// `date` (the factor times the original face).
    pub fn outstanding_face(&self, date: NaiveDate) -> f64 {
        let repaid: f64 = self
            .sinking_fund
            .iter()
            .filter(|&&(d, _)| d <= date)
            .map(|&(_, fraction)| fraction)
            .sum();
        self.face_value * (1.0 - repaid).max(0.0)
    }

    /// Principal repaid at a scheduled coupon date (absolute): the sink
    /// amount, plus the remaining outstanding when the date is maturity.
    fn principal_at(&self, coupon_date: NaiveDate) -> f64 {
        let sink: f64 = self
            .sinking_fund
            .iter()
            .filter(|&&(d, _)| d == coupon_date)
            .map(|&(_, fraction)| fraction * self.face_value)
            .sum();
        if coupon_date == self.maturity_date {
            let repaid: f64 = self.sinking_fund.iter().map(|&(_, f)| f).sum();
            sink + self.face_value * (1.0 - repaid).max(0.0)
        } else {
            sink
        }
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

    /// A US investment-grade corporate bond: semiannual 30/360 coupons,
    /// T+2 settlement on the bond-market calendar, payments rolled
    /// forward, end-of-month rule when maturity is a month-end.
    pub fn us_corporate(
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
            DayCountConvention::Thirty360,
            Calendar::UsGovernmentBond,
            BusinessDayConvention::Following,
            2,
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
        periods
            .iter()
            .map(|p| {
                let amount = self.coupon_amount(p) + self.principal_at(p.end);
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
        Ok(100.0 * self.rate_for(period.start) * fraction)
    }

    /// Absolute dirty redemption amount at `date` for a price quoted
    /// per 100 face: the price plus the accrued interest, scaled by the
    /// outstanding face — the strike an embedded call or put settles at.
    pub(crate) fn dirty_redemption_amount(
        &self,
        date: NaiveDate,
        price_per_100: f64,
    ) -> Result<f64, RustyQLibError> {
        let outstanding = self.outstanding_face(date);
        let accrued = self.accrued_interest(date)?;
        Ok(outstanding * price_per_100 / 100.0 + outstanding * accrued / 100.0)
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
        Ok(self.pv_flows(&flows, yield_rate)? * 100.0 / self.outstanding_face(settlement))
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
        let flows = self.remaining_flows(settlement)?;
        self.solve_yield(&flows, target_dirty, self.outstanding_face(settlement))
    }

    /// Yield to a call: the street yield assuming the bond is redeemed
    /// on `call.call_date` at `call.call_price` (per 100). Coupons up to
    /// the call are paid; a call between coupon dates pays the accrued
    /// to the call date.
    pub fn yield_to_call(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        call: &CallOption,
    ) -> Result<f64, RustyQLibError> {
        let target_dirty = clean_price + self.accrued_interest(settlement)?;
        let flows = self.flows_to_redemption(settlement, call.call_date, call.call_price)?;
        self.solve_yield(&flows, target_dirty, self.outstanding_face(settlement))
    }

    /// Yield to a put: the street yield assuming the holder redeems on
    /// `put.put_date` at `put.put_price` (per 100) — the same truncated
    /// cash-flow math as a call, exercised by the other side.
    pub fn yield_to_put(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        put: &PutOption,
    ) -> Result<f64, RustyQLibError> {
        let target_dirty = clean_price + self.accrued_interest(settlement)?;
        let flows = self.flows_to_redemption(settlement, put.put_date, put.put_price)?;
        self.solve_yield(&flows, target_dirty, self.outstanding_face(settlement))
    }

    /// Yield to worst: the lowest of the yield to maturity and the
    /// yields to every call still alive at `settlement`.
    pub fn yield_to_worst(
        &self,
        clean_price: f64,
        settlement: NaiveDate,
        calls: &[CallOption],
    ) -> Result<f64, RustyQLibError> {
        let mut worst = self.yield_from_clean_price(clean_price, settlement)?;
        for call in calls {
            if call.call_date <= settlement || call.call_date >= self.maturity_date {
                continue;
            }
            worst = worst.min(self.yield_to_call(clean_price, settlement, call)?);
        }
        Ok(worst)
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
        Ok(pv / df_settlement * 100.0 / self.outstanding_face(settlement))
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

    /// Coupon interest paid at the end of `period` (absolute): the
    /// period's stepped rate on the outstanding face at its start.
    fn coupon_amount(&self, period: &Period) -> f64 {
        let fraction = self.day_count.year_fraction_icma(
            period.start,
            period.end,
            period.ref_start,
            period.end,
            self.frequency.per_year(),
        );
        self.outstanding_face(period.start) * self.rate_for(period.start) * fraction
    }

    /// Fraction of a coupon period remaining at `at`, in period units.
    /// Under Act/Act ICMA this is days-to-coupon over the *reference*
    /// period length (the street convention, also correct for a short
    /// first coupon); other day counts use the ratio within the period.
    fn fraction_remaining(&self, at: NaiveDate, period: &Period) -> Result<f64, RustyQLibError> {
        let f = self.frequency.per_year();
        let remaining =
            self.day_count
                .year_fraction_icma(at, period.end, period.ref_start, period.end, f);
        if self.day_count == DayCountConvention::ActActIcma {
            Ok(remaining * f as f64)
        } else {
            let full = self.day_count.year_fraction(period.start, period.end);
            if full <= 0.0 {
                return Err(RustyQLibError::NumericalError(format!(
                    "degenerate coupon period ending {}",
                    period.end
                )));
            }
            Ok(remaining / full)
        }
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
        let w = self.fraction_remaining(settlement, &periods[next])?;

        let flows = periods[next..]
            .iter()
            .enumerate()
            .map(|(k, p)| Flow {
                tau: w + k as f64,
                amount: self.coupon_amount(p) + self.principal_at(p.end),
            })
            .collect();
        Ok(flows)
    }

    /// Cash flows assuming redemption at `redemption_date` instead of
    /// maturity (a call or a put): coupons through the redemption date,
    /// then the redemption price plus accrued, at the date's fractional
    /// period position.
    fn flows_to_redemption(
        &self,
        settlement: NaiveDate,
        redemption_date: NaiveDate,
        redemption_price: f64,
    ) -> Result<Vec<Flow>, RustyQLibError> {
        self.check_settlement(settlement)?;
        if !redemption_price.is_finite() || redemption_price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "redemption",
                format!("redemption price must be positive, got {redemption_price}"),
            ));
        }
        if redemption_date <= settlement || redemption_date > self.maturity_date {
            return Err(RustyQLibError::invalid_input(
                "redemption",
                format!(
                    "redemption date {redemption_date} must lie after settlement \
                     {settlement} and at or before maturity {}",
                    self.maturity_date
                ),
            ));
        }
        let redemption = self.outstanding_face(redemption_date) * redemption_price / 100.0;
        let periods = self.periods();
        let next = periods
            .iter()
            .position(|p| p.end > settlement)
            .expect("settlement is before maturity");
        let w = self.fraction_remaining(settlement, &periods[next])?;

        let mut flows = Vec::new();
        for (k, p) in periods[next..].iter().enumerate() {
            if p.end <= redemption_date {
                let mut amount = self.coupon_amount(p);
                if p.end == redemption_date {
                    amount += redemption;
                } else {
                    // scheduled sink payments before the redemption
                    amount += self.principal_at(p.end);
                }
                flows.push(Flow {
                    tau: w + k as f64,
                    amount,
                });
                if p.end == redemption_date {
                    break;
                }
            } else {
                // redemption strictly inside this period: the price plus
                // the accrued coupon, at the elapsed fraction of the period
                let elapsed = 1.0 - self.fraction_remaining(redemption_date, p)?;
                let accrued = self.outstanding_face(p.start)
                    * self.rate_for(p.start)
                    * self.day_count.year_fraction_icma(
                        p.start,
                        redemption_date,
                        p.ref_start,
                        p.end,
                        self.frequency.per_year(),
                    );
                flows.push(Flow {
                    tau: w + k as f64 - 1.0 + elapsed,
                    amount: redemption + accrued,
                });
                break;
            }
        }
        Ok(flows)
    }

    /// Solve the street yield hitting `target_dirty` (per 100 of
    /// `quote_base` outstanding face) over the given flows.
    fn solve_yield(
        &self,
        flows: &[Flow],
        target_dirty: f64,
        quote_base: f64,
    ) -> Result<f64, RustyQLibError> {
        if !target_dirty.is_finite() || target_dirty <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "bond",
                format!("dirty price must be positive, got {target_dirty}"),
            ));
        }
        let f = self.frequency.per_year() as f64;
        // g(y) = target - dirty(y) is increasing in y; bracket wide:
        // y > -f keeps 1 + y/f positive.
        let (lo, hi) = (-0.99 * f, 10.0);
        let g = |y: f64| {
            target_dirty
                - self
                    .pv_flows(flows, y)
                    .expect("bracket keeps 1 + y/f positive")
                    * 100.0
                    / quote_base
        };
        let dg = |y: f64| -self.dpv_dy(flows, y) * 100.0 / quote_base;
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

    #[test]
    fn us_corporate_conventions() {
        let bond =
            FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        assert_eq!(bond.day_count, DayCountConvention::Thirty360);
        assert_eq!(bond.settlement_days, 2);
        // trade Wednesday Aug 5 2026: T+2 settles Friday Aug 7
        assert_eq!(bond.settlement_date(d(2026, 8, 5)), d(2026, 8, 7));
        // 30/360 accrued: May 15 -> Aug 7 is 82 thirty-day-count days
        let accrued = bond.accrued_interest(d(2026, 8, 7)).unwrap();
        assert!(
            (accrued - 100.0 * 0.055 * 82.0 / 360.0).abs() < 1e-12,
            "accrued {accrued}"
        );
        // par identity holds under 30/360 exactly like Act/Act
        let clean = bond.clean_price_from_yield(0.055, d(2026, 11, 15)).unwrap();
        assert!((clean - 100.0).abs() < 1e-10, "clean {clean}");
    }

    #[test]
    fn yield_to_call_on_a_coupon_date_preserves_the_par_identity() {
        // a par-priced bond called at 100 on any coupon date still
        // yields the coupon
        let bond =
            FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        let settlement = d(2026, 11, 15); // coupon date
        for call_date in [d(2028, 5, 15), d(2029, 11, 15)] {
            let call = CallOption {
                call_date,
                call_price: 100.0,
            };
            let ytc = bond.yield_to_call(100.0, settlement, &call).unwrap();
            assert!((ytc - 0.055).abs() < 1e-10, "{call_date}: {ytc}");
        }
        // a premium call price raises the yield to that call
        let premium = CallOption {
            call_date: d(2028, 5, 15),
            call_price: 102.0,
        };
        let ytc = bond.yield_to_call(100.0, settlement, &premium).unwrap();
        assert!(ytc > 0.055, "premium call ytc {ytc}");
    }

    #[test]
    fn yield_to_worst_picks_the_binding_scenario() {
        let bond =
            FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        let settlement = d(2026, 8, 7);
        let calls = [
            CallOption {
                call_date: d(2028, 5, 15),
                call_price: 100.0,
            },
            CallOption {
                call_date: d(2029, 5, 15),
                call_price: 100.0,
            },
        ];
        // priced at a premium: early redemption at par is the worst
        let premium_clean = 104.0;
        let ytm = bond
            .yield_from_clean_price(premium_clean, settlement)
            .unwrap();
        let ytw = bond
            .yield_to_worst(premium_clean, settlement, &calls)
            .unwrap();
        let first_call = bond
            .yield_to_call(premium_clean, settlement, &calls[0])
            .unwrap();
        assert!(ytw < ytm, "{ytw} vs ytm {ytm}");
        assert!((ytw - first_call).abs() < 1e-12, "worst is the first call");
        // priced at a discount: holding to maturity is the worst
        let discount_clean = 95.0;
        let ytm = bond
            .yield_from_clean_price(discount_clean, settlement)
            .unwrap();
        let ytw = bond
            .yield_to_worst(discount_clean, settlement, &calls)
            .unwrap();
        assert!((ytw - ytm).abs() < 1e-12, "worst is maturity");
        // dead or maturity-dated calls are ignored
        let stale = [CallOption {
            call_date: d(2026, 6, 1),
            call_price: 100.0,
        }];
        let same = bond
            .yield_to_worst(discount_clean, settlement, &stale)
            .unwrap();
        assert!((same - ytm).abs() < 1e-15);
    }

    #[test]
    fn mid_period_call_approaches_the_coupon_date_call() {
        let bond =
            FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        let settlement = d(2026, 8, 7);
        let on_coupon = bond
            .yield_to_call(
                101.0,
                settlement,
                &CallOption {
                    call_date: d(2028, 5, 15),
                    call_price: 100.0,
                },
            )
            .unwrap();
        // one day earlier, mid-period: accrued to call replaces the
        // final coupon, so the yield barely moves
        let mid_period = bond
            .yield_to_call(
                101.0,
                settlement,
                &CallOption {
                    call_date: d(2028, 5, 14),
                    call_price: 100.0,
                },
            )
            .unwrap();
        assert!(
            (mid_period - on_coupon).abs() < 5e-4,
            "{mid_period} vs {on_coupon}"
        );
        // call before settlement or after maturity is rejected
        assert!(bond
            .yield_to_call(
                101.0,
                settlement,
                &CallOption {
                    call_date: d(2026, 8, 7),
                    call_price: 100.0,
                },
            )
            .is_err());
        assert!(bond
            .yield_to_call(
                101.0,
                settlement,
                &CallOption {
                    call_date: d(2032, 1, 1),
                    call_price: 100.0,
                },
            )
            .is_err());
    }

    #[test]
    fn step_up_coupons_change_the_flows_and_accrued_by_hand() {
        // 4% stepping to 6% from the 2027-05-15 coupon period onward
        let bond = FixedRateBond::us_corporate(100.0, 0.04, d(2026, 5, 15), d(2029, 5, 15))
            .unwrap()
            .with_coupon_steps(&[(d(2027, 5, 15), 0.06)])
            .unwrap();
        let amounts: Vec<f64> = bond.cashflows().iter().map(|cf| cf.amount).collect();
        // periods starting Nov 15 2026 and May 15 2027... the period
        // *starting* 2027-05-15 is the first at 6%: flows 2, 2, 3, 3, 3, 103
        assert_eq!(amounts.len(), 6);
        for (i, expected) in [2.0, 2.0, 3.0, 3.0, 3.0, 103.0].iter().enumerate() {
            assert!(
                (amounts[i] - expected).abs() < 1e-12,
                "flow {i}: {} vs {expected}",
                amounts[i]
            );
        }
        // accrued before the step uses 4%, after it 6% (30/360)
        let before = bond.accrued_interest(d(2026, 8, 15)).unwrap();
        assert!((before - 100.0 * 0.04 * 90.0 / 360.0).abs() < 1e-12);
        let after = bond.accrued_interest(d(2027, 8, 15)).unwrap();
        assert!((after - 100.0 * 0.06 * 90.0 / 360.0).abs() < 1e-12);
        // yield round trip still holds on the stepped flows
        let settle = d(2026, 8, 7);
        let clean = bond.clean_price_from_yield(0.05, settle).unwrap();
        let back = bond.yield_from_clean_price(clean, settle).unwrap();
        assert!((back - 0.05).abs() < 1e-9);
    }

    #[test]
    fn step_up_bond_prices_between_the_flat_bounds() {
        let settle = d(2026, 8, 7);
        let flat_low =
            FixedRateBond::us_corporate(100.0, 0.04, d(2026, 5, 15), d(2029, 5, 15)).unwrap();
        let flat_high =
            FixedRateBond::us_corporate(100.0, 0.06, d(2026, 5, 15), d(2029, 5, 15)).unwrap();
        let stepped = FixedRateBond::us_corporate(100.0, 0.04, d(2026, 5, 15), d(2029, 5, 15))
            .unwrap()
            .with_coupon_steps(&[(d(2027, 5, 15), 0.06)])
            .unwrap();
        let y = 0.05;
        let low = flat_low.clean_price_from_yield(y, settle).unwrap();
        let high = flat_high.clean_price_from_yield(y, settle).unwrap();
        let step = stepped.clean_price_from_yield(y, settle).unwrap();
        assert!(low < step && step < high, "{low} < {step} < {high}");
        // validation: out-of-range and unordered steps rejected
        let base =
            FixedRateBond::us_corporate(100.0, 0.04, d(2026, 5, 15), d(2029, 5, 15)).unwrap();
        assert!(base
            .clone()
            .with_coupon_steps(&[(d(2026, 5, 15), 0.06)])
            .is_err());
        assert!(base
            .clone()
            .with_coupon_steps(&[(d(2028, 5, 15), 0.06), (d(2027, 5, 15), 0.05)])
            .is_err());
        assert!(base.with_coupon_steps(&[(d(2027, 5, 15), -0.01)]).is_err());
    }

    #[test]
    fn sinking_fund_flows_and_outstanding_by_hand() {
        // 5.5% 2031, 25% sinks in 2028 and 2030 (coupon dates)
        let bond = FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15))
            .unwrap()
            .with_sinking_fund(&[(d(2028, 5, 15), 0.25), (d(2030, 5, 15), 0.25)])
            .unwrap();
        assert_eq!(bond.outstanding_face(d(2027, 1, 1)), 100.0);
        assert_eq!(bond.outstanding_face(d(2028, 5, 15)), 75.0);
        assert_eq!(bond.outstanding_face(d(2030, 5, 15)), 50.0);
        let flows = bond.cashflows();
        // coupons: 2.75 on 100 until May 2028, then 2.0625 on 75, then
        // 1.375 on 50; principals 25, 25, and 50 at maturity
        let expect = |cf: &crate::bonds::Cashflow, coupon: f64, principal: f64| {
            assert!(
                (cf.amount - coupon - principal).abs() < 1e-12,
                "{}: {} vs {} + {}",
                cf.accrual_end,
                cf.amount,
                coupon,
                principal
            );
        };
        expect(&flows[0], 2.75, 0.0); // Nov 2026
        expect(&flows[3], 2.75, 25.0); // May 2028: coupon on 100 + sink
        expect(&flows[4], 75.0 * 0.055 / 2.0, 0.0); // Nov 2028 on 75
        expect(&flows[7], 75.0 * 0.055 / 2.0, 25.0); // May 2030 + sink
        expect(&flows[8], 50.0 * 0.055 / 2.0, 0.0); // Nov 2030 on 50
        expect(&flows[9], 50.0 * 0.055 / 2.0, 50.0); // maturity remainder
    }

    #[test]
    fn sinker_prices_yield_round_trips_and_shortens_duration() {
        let settle = d(2026, 8, 7);
        let bullet =
            FixedRateBond::us_corporate(100.0, 0.055, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        let sinker = bullet
            .clone()
            .with_sinking_fund(&[(d(2028, 5, 15), 0.25), (d(2030, 5, 15), 0.25)])
            .unwrap();
        // par identity: at y = coupon, a sinker still prices at par
        // (every principal tranche is a par bond at its own horizon)
        let clean = sinker
            .clean_price_from_yield(0.055, d(2026, 11, 15))
            .unwrap();
        assert!((clean - 100.0).abs() < 1e-9, "sinker par {clean}");
        // yield round trip
        let quoted = sinker.clean_price_from_yield(0.05, settle).unwrap();
        let back = sinker.yield_from_clean_price(quoted, settle).unwrap();
        assert!((back - 0.05).abs() < 1e-9);
        // early principal return shortens duration vs the bullet
        let d_sinker = sinker.macaulay_duration(0.05, settle).unwrap();
        let d_bullet = bullet.macaulay_duration(0.05, settle).unwrap();
        assert!(d_sinker < d_bullet, "{d_sinker} vs {d_bullet}");
        // curve pricing consistent with manual discounting per 100 of
        // outstanding
        let curve = YieldCurve::flat(
            0.04,
            settle,
            DayCountConvention::Act365,
            crate::core::curves::Compounding::Continuous,
        )
        .unwrap();
        let manual: f64 = sinker
            .cashflows()
            .iter()
            .map(|cf| cf.amount * curve.df_date(cf.payment_date))
            .sum();
        let dirty = sinker.dirty_price_from_curve(&curve, settle).unwrap();
        assert!(
            (dirty - manual / curve.df_date(settle) * 100.0 / 100.0).abs() < 1e-10,
            "{dirty} vs {manual}"
        );
        // validation: off-schedule dates and over-redemption rejected
        assert!(bullet
            .clone()
            .with_sinking_fund(&[(d(2028, 5, 16), 0.25)])
            .is_err());
        assert!(bullet
            .clone()
            .with_sinking_fund(&[(d(2028, 5, 15), 0.7), (d(2030, 5, 15), 0.5)])
            .is_err());
        assert!(bullet.with_sinking_fund(&[(d(2031, 5, 15), 0.25)]).is_err());
    }
}
