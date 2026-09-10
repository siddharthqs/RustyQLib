//! Convertible bonds under the Tsiveriotis-Fernandes and the
//! jump-to-default models.
//!
//! **Why this lives in `bonds`**: a convertible is a corporate bond
//! with an embedded equity option, not an equity derivative with
//! coupons. It is quoted per 100 face with accrued interest, carries
//! the issuer's credit, and shares its call/put mechanics with the
//! straight corporates in this module — so it reuses [`FixedRateBond`]
//! wholesale (schedule, accrued, conventions) plus [`CallOption`] /
//! [`PutOption`]. The equity leg enters only as market inputs.
//!
//! Two credit treatments, behind the [`CreditModel`] trait, share one
//! CRR binomial tree ([`tree`]) and one finite-difference engine
//! ([`fd`]); the market struct passed in selects the model. The
//! instrument side is the [`ConvertibleInstrument`] trait — the bond
//! here and the preferred in [`preferred`](crate::bonds::preferred) —
//! each mapping its schedule onto the event grid ([`events`]), and the
//! whole pricing API comes through the blanket [`ConvertiblePricing`]
//! trait, which callers bring into scope. The contractual extras live
//! in [`features`].
//!
//! # Tsiveriotis-Fernandes ([`ConvertibleMarket`])
//!
//! The Tsiveriotis-Fernandes (1998) split, the industry-standard
//! single-factor treatment (QuantLib's convertible engine): the node
//! value is decomposed as `V = E + B`, where `E` is the part that ends
//! as shares (discounted **risk-free** — delivering your own stock
//! carries no default risk) and `B` the part that ends as cash
//! (discounted at **risk-free + credit spread**). The tree uses CRR
//! spacing with the risk-neutral drift taken from the discount curve's
//! own forward factors per step, so a never-converted bond reprices the
//! analytic risky bond exactly. The stock itself never defaults; credit
//! is a flat spread on the cash leg, independent of the share price.
//!
//! # Jump to default ([`JumpToDefaultMarket`])
//!
//! The reduced-form model (Andersen-Buffum 2004; Ayache-Forsyth-Vetzal
//! 2003 with total loss of the stock): default arrives at a hazard rate
//! `lambda`, and the share price follows
//!
//! ```text
//! dS / S = (r + lambda - q - b) dt + sigma dW - dP
//! ```
//!
//! where `dP` is the default indicator (`E[dP] = lambda dt`), `q` the
//! dividend yield and `b` the stock borrow cost. At default the stock
//! jumps to zero and stays there (`dS = -S`, an absorbing state); the
//! drift carries `lambda` on top of the risk-free carry so that the
//! stock still earns `r - q - b` unconditionally. On the tree each step
//! survives with probability `exp(-lambda dt)`, in which case the CRR
//! branches (with the survival-conditional drift) apply, or defaults,
//! in which case the holder receives the recovery claim: `recovery *
//! outstanding face`, paid at the end of the coupon period in which
//! default happens — the same convention as
//! [`credit`](crate::bonds::credit), so a busted convertible reprices
//! [`FixedRateBond::risky_dirty_price`] and the credit-triangle
//! identity `spread ~ lambda * (1 - recovery)` links the two engines.
//! Coupons are received only on survival to their payment date.
//!
//! Unlike Tsiveriotis-Fernandes there is no equity/cash split: every
//! claim, shares included, is lost on default, so one risky value is
//! carried per node and the whole convertible sees the hazard. The
//! equity option is worth *more* per unit of survival than under a
//! plain spread (the `lambda` in the drift is the compensation for the
//! jump), which is the model's well-known signature: a European call
//! under jump to default is Black-Scholes at rate `r + lambda`.
//!
//! A consequence worth knowing: the price is **not monotone in the
//! hazard rate**. The conversion claim is invariant to `lambda` (the
//! drift compensation hands the survival branch the whole equity
//! forward), the survival-only coupons and face fall with it, and the
//! recovery leg rises with it — so the price drops from the
//! zero-hazard value, bottoms out, and climbs back towards the equity
//! forward plus the recovery. In the money, or with a high recovery,
//! the minimum sits at a modest hazard. Only the falling branch reads
//! as credit; [`ConvertiblePricing::implied_hazard_rate`] searches it
//! alone.
//!
//! # Exercise logic
//!
//! Per node, inside the conversion window, for both models:
//!
//! - issuer call (optionally gated by a **soft-call trigger** on the
//!   share price): the holder responds by converting when parity beats
//!   the call price — `V = max(call_dirty, ratio * S)` if that improves
//!   the issuer's position;
//! - holder put: `V = max(V, put_dirty)`, all cash;
//! - voluntary conversion: `V = max(V, ratio * S)`, all equity.
//!
//! The contractual extras ride on the same projection, in both engines:
//!
//! - [`ContingentConversion`] ("CoCo"): conversion, voluntary or in
//!   answer to a call, needs the spot at or above a trigger until a
//!   given date;
//! - [`CouponMakeWhole`]: a call before a given date also pays the
//!   present value of the coupons up to that date, whether the holder
//!   takes cash or converts;
//! - [`FundamentalChangeMakeWhole`]: a takeover-type event arriving at
//!   a given intensity lets the holder convert with the indenture
//!   table's additional shares, put at par plus accrued, or carry on;
//! - [`MandatoryConversion`]: the terminal payoff is the share schedule
//!   between a maximum and a minimum ratio with no cash principal, and
//!   any early conversion is at the minimum ratio;
//! - [`CashDividend`]s on the underlying: the exact ex-date jump
//!   `V(S) = V(S - D)`, by interpolation on the engine's spot ladder.

pub mod credit;
pub mod events;
pub mod fd;
pub mod features;
pub mod implied;
pub mod instrument;
pub mod pricing;
pub mod tree;

#[cfg(test)]
mod fd_tests;
#[cfg(test)]
mod tests;

pub use credit::{
    ConvertibleMarket, CreditModel, EquityInputs, JumpToDefaultMarket, NodeValue, Split,
};
pub use events::EventGrid;
pub use fd::{ConvertibleFdGreeks, ConvertibleFdGrid, ConvertibleFdValuation, FdVolModel};
pub use features::{
    ContingentConversion, CouponMakeWhole, FundamentalChangeMakeWhole, MandatoryConversion,
};
pub use instrument::{CashDividend, ConvertibleInstrument};
pub use pricing::{ConvertiblePricing, DEFAULT_TREE_STEPS};

use chrono::NaiveDate;

use crate::bonds::{CallOption, FixedRateBond, PutOption};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

/// A convertible bond: a straight bond chassis plus the conversion
/// right, an optional (soft-)call schedule and an optional put
/// schedule, with the usual contractual extras: contingent
/// conversion, a coupon make-whole on calls, a fundamental-change
/// make-whole, and mandatory conversion.
#[derive(Debug, Clone)]
pub struct ConvertibleBond {
    pub bond: FixedRateBond,
    /// Shares received on converting one bond of `bond.face_value`.
    pub conversion_ratio: f64,
    /// First date conversion is allowed (default: the dated date).
    pub convert_from: Option<NaiveDate>,
    /// Last date conversion is allowed (default: maturity).
    pub convert_until: Option<NaiveDate>,
    /// Issuer calls (dirty strike = price + accrued, as for straights).
    pub calls: Vec<CallOption>,
    /// Soft-call trigger: calls are exercisable only when the share
    /// price is at or above this level (`None` = hard calls).
    pub soft_call_trigger: Option<f64>,
    /// Holder puts.
    pub puts: Vec<PutOption>,
    /// Contingent conversion trigger (`None` = unconditional).
    pub contingent_conversion: Option<ContingentConversion>,
    /// Coupon make-whole paid on issuer calls.
    pub coupon_make_whole: Option<CouponMakeWhole>,
    /// Fundamental-change make-whole and par put.
    pub fundamental_change: Option<FundamentalChangeMakeWhole>,
    /// Mandatory conversion at maturity (`None` = optional conversion
    /// against the cash redemption).
    pub mandatory: Option<MandatoryConversion>,
    /// Discrete cash dividends on the underlying share over the bond's
    /// life, on top of the market's continuous yield.
    pub cash_dividends: Vec<CashDividend>,
}

impl ConvertibleBond {
    pub fn new(bond: FixedRateBond, conversion_ratio: f64) -> Result<Self, RustyQLibError> {
        if !(conversion_ratio > 0.0 && conversion_ratio.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("conversion ratio must be positive, got {conversion_ratio}"),
            ));
        }
        Ok(ConvertibleBond {
            bond,
            conversion_ratio,
            convert_from: None,
            convert_until: None,
            calls: Vec::new(),
            soft_call_trigger: None,
            puts: Vec::new(),
            contingent_conversion: None,
            coupon_make_whole: None,
            fundamental_change: None,
            mandatory: None,
            cash_dividends: Vec::new(),
        })
    }

    /// Shares per bond delivered at maturity at a terminal share price:
    /// the mandatory schedule, or the plain ratio when not mandatory.
    pub fn maturity_shares(&self, spot: f64) -> f64 {
        let min_ratio = self.conversion_ratio;
        match &self.mandatory {
            None => min_ratio,
            Some(m) => {
                let face = self.bond.face_value;
                if spot <= face / m.max_ratio {
                    m.max_ratio
                } else if spot >= face / min_ratio {
                    min_ratio
                } else {
                    face / spot
                }
            }
        }
    }

    /// Validates the contractual extras (the chassis validates itself).
    fn validate_features(&self) -> Result<(), RustyQLibError> {
        instrument::validate_cash_dividends(&self.cash_dividends)?;
        if let Some(coco) = &self.contingent_conversion {
            if !(coco.trigger > 0.0 && coco.trigger.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "convertible",
                    format!(
                        "the conversion trigger must be positive, got {}",
                        coco.trigger
                    ),
                ));
            }
        }
        if let Some(mw) = &self.coupon_make_whole {
            if !mw.spread.is_finite() {
                return Err(RustyQLibError::invalid_input(
                    "convertible",
                    "the make-whole spread must be finite",
                ));
            }
        }
        if let Some(fc) = &self.fundamental_change {
            fc.validate()?;
        }
        if let Some(m) = &self.mandatory {
            if !(m.max_ratio.is_finite() && m.max_ratio > self.conversion_ratio) {
                return Err(RustyQLibError::invalid_input(
                    "convertible",
                    format!(
                        "the mandatory maximum ratio {} must exceed the minimum ratio {}",
                        m.max_ratio, self.conversion_ratio
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The share price at which conversion breaks even against face.
    pub fn conversion_price(&self) -> f64 {
        self.bond.face_value / self.conversion_ratio
    }

    /// Conversion (parity) value per 100 face at a share price.
    pub fn parity(&self, spot: f64) -> f64 {
        self.conversion_ratio * spot * 100.0 / self.bond.face_value
    }

    /// Conversion premium of a clean price over parity, as a fraction
    /// (`0.15` = 15% premium).
    pub fn conversion_premium(&self, clean_price: f64, spot: f64) -> Result<f64, RustyQLibError> {
        let parity = self.parity(spot);
        if parity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive parity {parity}"
            )));
        }
        Ok(clean_price / parity - 1.0)
    }

    /// Present value per 100 face at `settlement` of the principal
    /// repayments after it, discounted on `curve` and weighted by
    /// `survival` (of the time from settlement).
    pub(crate) fn principal_pv(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        survival: impl Fn(f64) -> f64,
    ) -> f64 {
        let day_count = curve.day_count();
        let df_settlement = curve.df_date(settlement);
        let pv: f64 = self
            .bond
            .cashflows()
            .iter()
            .filter(|cf| cf.accrual_end > settlement)
            .map(|cf| {
                let t = day_count.year_fraction(settlement, cf.payment_date.max(settlement));
                self.bond.principal_at(cf.accrual_end)
                    * survival(t)
                    * curve.df_date(cf.payment_date)
                    / df_settlement
            })
            .sum();
        pv * 100.0 / self.bond.outstanding_face(settlement)
    }

    /// The straight-bond floor under the market's credit model: the
    /// chassis ignoring the conversion right (and, for a mandatory,
    /// the principal). The engines collapse onto this when the shares
    /// are worthless. The same as
    /// [`straight_floor`](ConvertiblePricing::straight_floor).
    pub fn bond_floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        market.bond_floor(self, curve, settlement)
    }
}

impl ConvertibleInstrument for ConvertibleBond {
    fn conversion_ratio(&self) -> f64 {
        self.conversion_ratio
    }

    fn maturity_shares(&self, spot: f64) -> f64 {
        ConvertibleBond::maturity_shares(self, spot)
    }

    fn is_mandatory(&self) -> bool {
        self.mandatory.is_some()
    }

    fn soft_call_trigger(&self) -> Option<f64> {
        self.soft_call_trigger
    }

    fn conversion_price(&self) -> f64 {
        ConvertibleBond::conversion_price(self)
    }

    fn accrued(&self, settlement: NaiveDate) -> Result<f64, RustyQLibError> {
        self.bond.accrued_interest(settlement)
    }

    fn final_payment_date(&self, settlement: NaiveDate) -> Result<NaiveDate, RustyQLibError> {
        self.bond.accrued_interest(settlement)?;
        Ok(self
            .bond
            .cashflows()
            .last()
            .map(|cf| cf.payment_date)
            .unwrap_or(self.bond.maturity_date))
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        self.validate_features()
    }

    fn event_grid(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
        credit_rate: f64,
    ) -> Result<EventGrid, RustyQLibError> {
        events::event_grid(self, curve, settlement, steps, credit_rate)
    }

    fn floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        market.bond_floor(self, curve, settlement)
    }
}
