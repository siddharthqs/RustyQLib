//! The credit treatments of the convertible, behind one trait.
//!
//! A [`CreditModel`] is a market snapshot — the equity inputs plus the
//! credit inputs — together with the rules for carrying value back one
//! step: how the share price drifts conditional on survival, how a
//! node's components discount, and what the straight-bond floor is.
//! Its associated [`Node`](CreditModel::Node) is the per-node state,
//! with just enough algebra ([`NodeValue`]) for the shared exercise
//! logic to work on either shape:
//!
//! - [`ConvertibleMarket`] is Tsiveriotis-Fernandes: the node is a
//!   [`Split`] into an equity part discounted risk-free and a cash part
//!   discounted at risk-free plus the spread;
//! - [`JumpToDefaultMarket`] is jump to default: the node is one risky
//!   value, discounted at risk-free plus the hazard, with the recovery
//!   claim added on the default branch.
//!
//! The pricing methods on [`ConvertibleBond`] are generic over the
//! trait, so the market type selects the model and the engines
//! (tree, finite differences) are written once.

use chrono::NaiveDate;

use super::ConvertibleBond;
use crate::core::curves::{RateShift, YieldCurve};
use crate::core::errors::RustyQLibError;

/// The equity inputs every credit model carries.
#[derive(Debug, Clone, Copy)]
pub struct EquityInputs {
    pub spot: f64,
    pub volatility: f64,
    pub dividend_yield: f64,
    /// Stock borrow cost (zero where the model has none).
    pub borrow_cost: f64,
}

/// The per-node state of a credit model through the backward
/// induction, with the algebra the shared exercise logic needs.
pub trait NodeValue: Copy {
    /// A claim settled entirely in shares worth `x`.
    fn equity(x: f64) -> Self;
    /// A claim settled entirely in cash `x`.
    fn cash(x: f64) -> Self;
    /// The node's total value.
    fn total(&self) -> f64;
    /// `(1 - w) * a + w * b`, component by component.
    fn blend(a: Self, b: Self, w: f64) -> Self;
    /// The node with cash `x` added.
    fn plus_cash(self, x: f64) -> Self;
}

/// One risky value: every claim shares the same discounting.
impl NodeValue for f64 {
    fn equity(x: f64) -> Self {
        x
    }
    fn cash(x: f64) -> Self {
        x
    }
    fn total(&self) -> f64 {
        *self
    }
    fn blend(a: Self, b: Self, w: f64) -> Self {
        (1.0 - w) * a + w * b
    }
    fn plus_cash(self, x: f64) -> Self {
        self + x
    }
}

/// The Tsiveriotis-Fernandes split of a node into the part that ends
/// as shares and the part that ends as cash.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Split {
    pub equity: f64,
    pub cash: f64,
}

impl NodeValue for Split {
    fn equity(x: f64) -> Self {
        Split {
            equity: x,
            cash: 0.0,
        }
    }
    fn cash(x: f64) -> Self {
        Split {
            equity: 0.0,
            cash: x,
        }
    }
    fn total(&self) -> f64 {
        self.equity + self.cash
    }
    fn blend(a: Self, b: Self, w: f64) -> Self {
        Split {
            equity: (1.0 - w) * a.equity + w * b.equity,
            cash: (1.0 - w) * a.cash + w * b.cash,
        }
    }
    fn plus_cash(self, x: f64) -> Self {
        Split {
            equity: self.equity,
            cash: self.cash + x,
        }
    }
}

/// A credit treatment of the convertible: the market snapshot and the
/// per-step rules. Implemented by [`ConvertibleMarket`] and
/// [`JumpToDefaultMarket`]; the pricing API is generic over it.
pub trait CreditModel: Copy {
    /// The per-node state.
    type Node: NodeValue;

    fn validate(&self) -> Result<(), RustyQLibError>;

    /// The equity inputs.
    fn equity(&self) -> EquityInputs;

    /// The rate over the risk-free forward that carries the credit:
    /// the spread, or the hazard rate. Coupons at stake discount at the
    /// forward plus this, and `exp(-credit_rate * dt)` is a step's
    /// credit factor (the cash leg's extra discount, or the survival).
    fn credit_rate(&self) -> f64;

    /// The share price's drift over `r` conditional on survival:
    /// `-q` under a spread, `lambda - q - b` under jump to default.
    fn survival_drift(&self) -> f64;

    /// Tree: the node at a step's start from the risk-neutral
    /// expectation `expected` of the later nodes, given the step's
    /// risk-free discount factor, credit factor, and the face-recovery
    /// claim (per unit recovery rate) should default be observed.
    fn discount_step(
        &self,
        expected: Self::Node,
        riskfree_df: f64,
        credit_df: f64,
        default_claim: f64,
    ) -> Self::Node;

    /// Finite differences: the nodes at a step's start, given
    /// `diffuse`, which carries one component vector back over the
    /// step at a discount rate, plus the step's risk-free rate and
    /// factors as for [`discount_step`](Self::discount_step).
    fn pde_step(
        &self,
        nodes: &[Self::Node],
        diffuse: &dyn Fn(Vec<f64>, f64) -> Vec<f64>,
        r: f64,
        riskfree_df: f64,
        credit_df: f64,
        default_claim: f64,
    ) -> Vec<Self::Node>;

    /// The straight-bond floor per 100 face: the chassis under this
    /// credit treatment ignoring the conversion right (and, for a
    /// mandatory, the principal that is never paid in cash).
    fn bond_floor(
        &self,
        convertible: &ConvertibleBond,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError>;

    /// The value at `settlement` of promised cash flows `(time from
    /// settlement on the curve's day count, amount)`, sorted by time,
    /// under this credit treatment, with `claim` the amount recovered
    /// (per unit recovery rate) should default happen — the analytic
    /// floor of any straight schedule.
    fn value_of_flows(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        flows: &[(f64, f64)],
        claim: f64,
    ) -> f64;

    fn with_spot(&self, spot: f64) -> Self;
    fn with_volatility(&self, volatility: f64) -> Self;
    /// The same market with the credit input shifted by `bump`.
    fn with_credit_bump(&self, bump: f64) -> Self;
}

fn validate_equity_inputs(spot: f64, volatility: f64) -> Result<(), RustyQLibError> {
    if !(spot > 0.0 && spot.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            format!("spot must be positive, got {spot}"),
        ));
    }
    if !(volatility > 0.0 && volatility.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            format!("volatility must be positive, got {volatility}"),
        ));
    }
    Ok(())
}

/// Equity and credit inputs for Tsiveriotis-Fernandes pricing.
#[derive(Debug, Clone, Copy)]
pub struct ConvertibleMarket {
    /// Share price.
    pub spot: f64,
    /// Flat lognormal equity volatility.
    pub volatility: f64,
    /// Continuous dividend yield.
    pub dividend_yield: f64,
    /// Issuer credit spread applied to the cash-only part
    /// (continuously compounded, e.g. from the issuer's z-spread).
    pub credit_spread: f64,
}

impl ConvertibleMarket {
    pub(crate) fn validate(&self) -> Result<(), RustyQLibError> {
        validate_equity_inputs(self.spot, self.volatility)?;
        if !self.dividend_yield.is_finite() || !self.credit_spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                "dividend yield and credit spread must be finite",
            ));
        }
        Ok(())
    }
}

impl CreditModel for ConvertibleMarket {
    type Node = Split;

    fn validate(&self) -> Result<(), RustyQLibError> {
        ConvertibleMarket::validate(self)
    }

    fn equity(&self) -> EquityInputs {
        EquityInputs {
            spot: self.spot,
            volatility: self.volatility,
            dividend_yield: self.dividend_yield,
            borrow_cost: 0.0,
        }
    }

    fn credit_rate(&self) -> f64 {
        self.credit_spread
    }

    fn survival_drift(&self) -> f64 {
        -self.dividend_yield
    }

    /// The equity part discounts risk-free, the cash part at the
    /// spread on top; the stock never defaults.
    fn discount_step(
        &self,
        expected: Split,
        riskfree_df: f64,
        credit_df: f64,
        _default_claim: f64,
    ) -> Split {
        Split {
            equity: riskfree_df * expected.equity,
            cash: riskfree_df * credit_df * expected.cash,
        }
    }

    fn pde_step(
        &self,
        nodes: &[Split],
        diffuse: &dyn Fn(Vec<f64>, f64) -> Vec<f64>,
        r: f64,
        _riskfree_df: f64,
        _credit_df: f64,
        _default_claim: f64,
    ) -> Vec<Split> {
        let equity = diffuse(nodes.iter().map(|n| n.equity).collect(), r);
        let cash = diffuse(
            nodes.iter().map(|n| n.cash).collect(),
            r + self.credit_spread,
        );
        equity
            .into_iter()
            .zip(cash)
            .map(|(equity, cash)| Split { equity, cash })
            .collect()
    }

    /// The chassis priced at the credit spread.
    fn bond_floor(
        &self,
        convertible: &ConvertibleBond,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let chassis = convertible.bond.clean_price_from_curve_with_spread(
            curve,
            self.credit_spread,
            settlement,
        )?;
        if convertible.mandatory.is_none() {
            return Ok(chassis);
        }
        // a mandatory pays no principal: the floor is the coupons alone
        let spread_curve = curve.bumped(&RateShift::ParallelAbsolute(self.credit_spread))?;
        Ok(chassis - convertible.principal_pv(&spread_curve, settlement, |_| 1.0))
    }

    /// Every flow discounts at the curve plus the spread.
    fn value_of_flows(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        flows: &[(f64, f64)],
        _claim: f64,
    ) -> f64 {
        let t0 = curve
            .day_count()
            .year_fraction(curve.reference_date(), settlement);
        flows
            .iter()
            .map(|&(t, amount)| {
                amount * curve.df(t0 + t) / curve.df(t0) * (-self.credit_spread * t).exp()
            })
            .sum()
    }

    fn with_spot(&self, spot: f64) -> Self {
        ConvertibleMarket { spot, ..*self }
    }

    fn with_volatility(&self, volatility: f64) -> Self {
        ConvertibleMarket {
            volatility,
            ..*self
        }
    }

    fn with_credit_bump(&self, bump: f64) -> Self {
        ConvertibleMarket {
            credit_spread: self.credit_spread + bump,
            ..*self
        }
    }
}

/// Equity and credit inputs for jump-to-default pricing (see the
/// module docs for the process).
#[derive(Debug, Clone, Copy)]
pub struct JumpToDefaultMarket {
    /// Share price.
    pub spot: f64,
    /// Flat lognormal equity volatility of the pre-default diffusion.
    pub volatility: f64,
    /// Continuous dividend yield `q`.
    pub dividend_yield: f64,
    /// Continuous stock borrow cost `b` (the repo/lending fee the
    /// hedger pays to short the shares; lowers the forward).
    pub borrow_cost: f64,
    /// Default intensity `lambda` (continuously compounded, per year).
    pub hazard_rate: f64,
    /// Fraction of the outstanding face recovered on default, in
    /// `[0, 1]`.
    pub recovery_rate: f64,
}

impl JumpToDefaultMarket {
    pub(crate) fn validate(&self) -> Result<(), RustyQLibError> {
        validate_equity_inputs(self.spot, self.volatility)?;
        if !self.dividend_yield.is_finite() || !self.borrow_cost.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                "dividend yield and borrow cost must be finite",
            ));
        }
        if !(self.hazard_rate >= 0.0 && self.hazard_rate.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("hazard rate must be non-negative, got {}", self.hazard_rate),
            ));
        }
        if !(self.recovery_rate.is_finite() && (0.0..=1.0).contains(&self.recovery_rate)) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!(
                    "recovery rate must be in [0, 1], got {}",
                    self.recovery_rate
                ),
            ));
        }
        Ok(())
    }
}

impl CreditModel for JumpToDefaultMarket {
    type Node = f64;

    fn validate(&self) -> Result<(), RustyQLibError> {
        JumpToDefaultMarket::validate(self)
    }

    fn equity(&self) -> EquityInputs {
        EquityInputs {
            spot: self.spot,
            volatility: self.volatility,
            dividend_yield: self.dividend_yield,
            borrow_cost: self.borrow_cost,
        }
    }

    fn credit_rate(&self) -> f64 {
        self.hazard_rate
    }

    /// Conditional on surviving the step the stock earns the hazard on
    /// top of the risk-free carry: with the default branch worth
    /// nothing this leaves the unconditional drift at `r - q - b`.
    fn survival_drift(&self) -> f64 {
        self.hazard_rate - self.dividend_yield - self.borrow_cost
    }

    /// Survive with the credit factor (here the survival probability)
    /// or default and collect the recovery on the face.
    fn discount_step(
        &self,
        expected: f64,
        riskfree_df: f64,
        credit_df: f64,
        default_claim: f64,
    ) -> f64 {
        let recovery = self.recovery_rate * default_claim;
        riskfree_df * (credit_df * expected + (1.0 - credit_df) * recovery)
    }

    fn pde_step(
        &self,
        nodes: &[f64],
        diffuse: &dyn Fn(Vec<f64>, f64) -> Vec<f64>,
        r: f64,
        riskfree_df: f64,
        credit_df: f64,
        default_claim: f64,
    ) -> Vec<f64> {
        // the survival branch discounts at r + lambda; the default
        // branch adds the recovery on the face, as on the tree
        let recovery = riskfree_df * (1.0 - credit_df) * self.recovery_rate * default_claim;
        diffuse(nodes.to_vec(), r + self.hazard_rate)
            .into_iter()
            .map(|v| v + recovery)
            .collect()
    }

    /// The chassis as a hazard-rate risky bond with the market's
    /// recovery ([`FixedRateBond::risky_clean_price`](crate::bonds::FixedRateBond::risky_clean_price)).
    fn bond_floor(
        &self,
        convertible: &ConvertibleBond,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let chassis = convertible.bond.risky_clean_price(
            curve,
            self.hazard_rate,
            self.recovery_rate,
            settlement,
        )?;
        if convertible.mandatory.is_none() {
            return Ok(chassis);
        }
        // a mandatory pays no principal on survival; the recovery claim
        // on the face in default stays
        let hazard = self.hazard_rate;
        Ok(chassis - convertible.principal_pv(curve, settlement, |t| (-hazard * t).exp()))
    }

    /// Flows weighted by survival, plus the recovery on the claim paid
    /// at the end of the period in which default happens — the credit
    /// module's convention, which the tree's default branch mirrors.
    fn value_of_flows(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        flows: &[(f64, f64)],
        claim: f64,
    ) -> f64 {
        let t0 = curve
            .day_count()
            .year_fraction(curve.reference_date(), settlement);
        let survival = |t: f64| (-self.hazard_rate * t).exp();
        let mut value = 0.0;
        let mut survival_start = 1.0;
        for &(t, amount) in flows {
            let df = curve.df(t0 + t) / curve.df(t0);
            let survival_end = survival(t);
            value += amount * survival_end * df;
            value += self.recovery_rate * claim * (survival_start - survival_end) * df;
            survival_start = survival_end;
        }
        value
    }

    fn with_spot(&self, spot: f64) -> Self {
        JumpToDefaultMarket { spot, ..*self }
    }

    fn with_volatility(&self, volatility: f64) -> Self {
        JumpToDefaultMarket {
            volatility,
            ..*self
        }
    }

    fn with_credit_bump(&self, bump: f64) -> Self {
        JumpToDefaultMarket {
            hazard_rate: self.hazard_rate + bump,
            ..*self
        }
    }
}
