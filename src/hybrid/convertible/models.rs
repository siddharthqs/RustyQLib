//! The credit models of the convertible, behind one trait.
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
//!   claim added on the default branch; the hazard is a
//!   [`HazardLevel`], flat or a term structure on a [`CreditCurve`];
//! - [`EquityLinkedHazardMarket`] is the same jump with a hazard that
//!   depends on the share price, `lambda(S, t) = a(t) (S0 / S)^p`, so
//!   the credit weakens as the stock falls; the node-aware hooks
//!   ([`credit_rate_at`](CreditModel::credit_rate_at),
//!   [`survival_drift_at`](CreditModel::survival_drift_at)) carry the
//!   dependence into both engines, and the scalar models leave them at
//!   their defaults. Its level `a(t)` is calibrated through the model
//!   to a CDS curve by
//!   [`calibrated_to`](EquityLinkedHazardMarket::calibrated_to).
//!
//! The pricing methods on [`ConvertibleBond`] are generic over the
//! trait, so the market type selects the model and the engines
//! (tree, finite differences) are written once.

use chrono::NaiveDate;

use super::fd::{valuation, ConvertibleFdGrid, FdVolModel};
use super::ConvertibleBond;
use crate::core::curves::{RateShift, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::vols::VolSurface;
use crate::credit::CreditCurve;

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

/// A finite-difference carry-back of one component vector over a step,
/// at a discount rate per node: what [`CreditModel::pde_step`] is handed.
pub type Diffusion<'a> = dyn Fn(Vec<f64>, &[f64]) -> Vec<f64> + 'a;

/// One grid step's context for [`CreditModel::pde_step`].
#[derive(Debug, Clone, Copy)]
pub struct StepContext<'a> {
    /// The step's continuously compounded risk-free rate.
    pub r: f64,
    /// The step's risk-free discount factor.
    pub riskfree_df: f64,
    pub dt: f64,
    /// The share price at every node.
    pub spots: &'a [f64],
    /// Time from settlement at the step's midpoint.
    pub t: f64,
    /// The face-recovery claim (per unit recovery rate) should default
    /// be observed in the step.
    pub default_claim: f64,
}

/// A credit treatment of the convertible: the market snapshot and the
/// per-step rules. Implemented by [`ConvertibleMarket`],
/// [`JumpToDefaultMarket`] and [`EquityLinkedHazardMarket`]; the
/// pricing API is generic over it.
pub trait CreditModel: Clone {
    /// The per-node state.
    type Node: NodeValue;

    fn validate(&self) -> Result<(), RustyQLibError>;

    /// The equity inputs.
    fn equity(&self) -> EquityInputs;

    /// The rate over the risk-free forward that carries the credit
    /// where one number is needed — the spread, or the hazard level at
    /// the long end of its term structure: coupons at stake discount at
    /// the forward plus this within a step, and a perpetual tail prices
    /// at it.
    fn credit_rate(&self) -> f64;

    /// The share price's drift over `r` conditional on survival at the
    /// [`credit_rate`](Self::credit_rate): `-q` under a spread,
    /// `lambda - q - b` under jump to default.
    fn survival_drift(&self) -> f64;

    /// The credit rate at a share price and time from settlement, for
    /// models whose hazard depends on the state or the term; the scalar
    /// rate by default.
    fn credit_rate_at(&self, _spot: f64, _t: f64) -> f64 {
        self.credit_rate()
    }

    /// The survival drift at a share price and time from settlement;
    /// the scalar drift by default.
    fn survival_drift_at(&self, _spot: f64, _t: f64) -> f64 {
        self.survival_drift()
    }

    /// Whether the credit rate and drift vary with the share price, so
    /// the engines evaluate them node by node (a term structure alone
    /// is evaluated step by step and needs no per-node work).
    fn is_state_dependent(&self) -> bool {
        false
    }

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
    /// step at a discount rate per node, and the step's context.
    fn pde_step(
        &self,
        nodes: &[Self::Node],
        diffuse: &Diffusion<'_>,
        step: &StepContext,
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

// ---------------------------------------------------------------------
// Tsiveriotis-Fernandes
// ---------------------------------------------------------------------

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

    fn pde_step(&self, nodes: &[Split], diffuse: &Diffusion<'_>, step: &StepContext) -> Vec<Split> {
        let n = nodes.len();
        let equity = diffuse(nodes.iter().map(|n| n.equity).collect(), &vec![step.r; n]);
        let cash = diffuse(
            nodes.iter().map(|n| n.cash).collect(),
            &vec![step.r + self.credit_spread; n],
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

// ---------------------------------------------------------------------
// The hazard level: flat or a term structure
// ---------------------------------------------------------------------

/// The default intensity by term: a single rate, or the piecewise-
/// constant hazards of a [`CreditCurve`] (times from settlement on the
/// discount curve's day count, as the curve is built — from a CDS
/// bootstrap, say).
#[derive(Debug, Clone)]
pub enum HazardLevel {
    Flat(f64),
    Term(CreditCurve),
}

impl HazardLevel {
    /// The instantaneous hazard at time `t` from settlement.
    pub fn at(&self, t: f64) -> f64 {
        match self {
            HazardLevel::Flat(h) => *h,
            HazardLevel::Term(curve) => curve.hazard_at(t),
        }
    }

    /// Survival to `t` from settlement.
    pub fn survival(&self, t: f64) -> f64 {
        match self {
            HazardLevel::Flat(h) => (-h * t).exp(),
            HazardLevel::Term(curve) => curve.survival(t),
        }
    }

    /// The average hazard to `t`: `-ln(survival) / t`, the flat rate
    /// with the same survival.
    pub fn average_to(&self, t: f64) -> f64 {
        match self {
            HazardLevel::Flat(h) => *h,
            HazardLevel::Term(curve) => {
                if t <= 0.0 {
                    curve.hazard_at(0.0)
                } else {
                    -curve.survival(t).ln() / t
                }
            }
        }
    }

    /// The level where one number is needed: the flat rate, or the
    /// last pillar of the term structure (which extends flat).
    pub fn long_end(&self) -> f64 {
        match self {
            HazardLevel::Flat(h) => *h,
            HazardLevel::Term(curve) => curve.pillars().last().map_or(0.0, |&(_, h)| h),
        }
    }

    /// The level with `bump` added at every term (floored at zero).
    pub fn shifted(&self, bump: f64) -> Result<Self, RustyQLibError> {
        Ok(match self {
            HazardLevel::Flat(h) => HazardLevel::Flat((h + bump).max(0.0)),
            HazardLevel::Term(curve) => {
                let pillars: Vec<(f64, f64)> = curve
                    .pillars()
                    .iter()
                    .map(|&(t, h)| (t, (h + bump).max(0.0)))
                    .collect();
                HazardLevel::Term(CreditCurve::new(&pillars)?)
            }
        })
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        if let HazardLevel::Flat(h) = self {
            if !(*h >= 0.0 && h.is_finite()) {
                return Err(RustyQLibError::invalid_input(
                    "convertible",
                    format!("hazard rate must be non-negative, got {h}"),
                ));
            }
        }
        Ok(())
    }
}

fn validate_recovery(recovery_rate: f64) -> Result<(), RustyQLibError> {
    if !(recovery_rate.is_finite() && (0.0..=1.0).contains(&recovery_rate)) {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            format!("recovery rate must be in [0, 1], got {recovery_rate}"),
        ));
    }
    Ok(())
}

/// The value of a survival-weighted schedule with recovery on `claim`
/// paid at each flow date for default in the preceding period — the
/// credit module's convention, which the tree's default branch mirrors.
fn value_of_flows_with_survival(
    curve: &YieldCurve,
    settlement: NaiveDate,
    flows: &[(f64, f64)],
    claim: f64,
    recovery_rate: f64,
    survival: impl Fn(f64) -> f64,
) -> f64 {
    let t0 = curve
        .day_count()
        .year_fraction(curve.reference_date(), settlement);
    let mut value = 0.0;
    let mut survival_start = 1.0;
    for &(t, amount) in flows {
        let df = curve.df(t0 + t) / curve.df(t0);
        let survival_end = survival(t);
        value += amount * survival_end * df;
        value += recovery_rate * claim * (survival_start - survival_end) * df;
        survival_start = survival_end;
    }
    value
}

// ---------------------------------------------------------------------
// Jump to default
// ---------------------------------------------------------------------

/// Equity and credit inputs for jump-to-default pricing (see the
/// module docs of [`convertible`](super) for the process). The hazard
/// is flat or a term structure ([`HazardLevel`]).
#[derive(Debug, Clone)]
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
    /// Default intensity `lambda`, flat or by term.
    pub hazard: HazardLevel,
    /// Fraction of the outstanding face recovered on default, in
    /// `[0, 1]`.
    pub recovery_rate: f64,
}

impl JumpToDefaultMarket {
    /// The flat hazard at `hazard_rate`.
    pub fn flat(
        spot: f64,
        volatility: f64,
        dividend_yield: f64,
        borrow_cost: f64,
        hazard_rate: f64,
        recovery_rate: f64,
    ) -> Self {
        JumpToDefaultMarket {
            spot,
            volatility,
            dividend_yield,
            borrow_cost,
            hazard: HazardLevel::Flat(hazard_rate),
            recovery_rate,
        }
    }

    /// The same market on a hazard term structure (a CDS-bootstrapped
    /// [`CreditCurve`], say).
    pub fn with_hazard_curve(self, curve: CreditCurve) -> Self {
        JumpToDefaultMarket {
            hazard: HazardLevel::Term(curve),
            ..self
        }
    }

    pub(crate) fn validate(&self) -> Result<(), RustyQLibError> {
        validate_equity_inputs(self.spot, self.volatility)?;
        if !self.dividend_yield.is_finite() || !self.borrow_cost.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                "dividend yield and borrow cost must be finite",
            ));
        }
        self.hazard.validate()?;
        validate_recovery(self.recovery_rate)
    }

    /// `surface` de-jumped for this market's hazard, carry and spot,
    /// sampled on a strike x expiry grid: the input for a Dupire local
    /// vol that is consistent with the jump (see
    /// [`dejump_surface`](super::dejump_surface)). A term structure
    /// de-jumps each expiry at its average hazard.
    pub fn dejump_surface(
        &self,
        surface: &VolSurface,
        curve: &YieldCurve,
        strikes: &[f64],
        expiries: &[f64],
    ) -> Result<VolSurface, RustyQLibError> {
        super::dejump::dejump_surface_with(
            surface,
            curve,
            self.spot,
            self.dividend_yield,
            self.borrow_cost,
            |t| self.hazard.average_to(t),
            strikes,
            expiries,
        )
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
        self.hazard.long_end()
    }

    /// Conditional on surviving the step the stock earns the hazard on
    /// top of the risk-free carry: with the default branch worth
    /// nothing this leaves the unconditional drift at `r - q - b`.
    fn survival_drift(&self) -> f64 {
        self.hazard.long_end() - self.dividend_yield - self.borrow_cost
    }

    fn credit_rate_at(&self, _spot: f64, t: f64) -> f64 {
        self.hazard.at(t)
    }

    fn survival_drift_at(&self, _spot: f64, t: f64) -> f64 {
        self.hazard.at(t) - self.dividend_yield - self.borrow_cost
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

    fn pde_step(&self, nodes: &[f64], diffuse: &Diffusion<'_>, step: &StepContext) -> Vec<f64> {
        // the survival branch discounts at r + lambda; the default
        // branch adds the recovery on the face, as on the tree
        let hazard = self.hazard.at(step.t);
        let credit_df = (-hazard * step.dt).exp();
        let recovery =
            step.riskfree_df * (1.0 - credit_df) * self.recovery_rate * step.default_claim;
        diffuse(nodes.to_vec(), &vec![step.r + hazard; nodes.len()])
            .into_iter()
            .map(|v| v + recovery)
            .collect()
    }

    /// The chassis as a hazard-rate risky bond with the market's
    /// recovery, on the flat hazard or the term structure
    /// ([`FixedRateBond::risky_clean_price`](crate::bonds::FixedRateBond::risky_clean_price)
    /// and its curve variant).
    fn bond_floor(
        &self,
        convertible: &ConvertibleBond,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let chassis = match &self.hazard {
            HazardLevel::Flat(h) => {
                convertible
                    .bond
                    .risky_clean_price(curve, *h, self.recovery_rate, settlement)?
            }
            HazardLevel::Term(credit) => convertible.bond.risky_clean_price_on_curve(
                curve,
                credit,
                self.recovery_rate,
                settlement,
            )?,
        };
        if convertible.mandatory.is_none() {
            return Ok(chassis);
        }
        // a mandatory pays no principal on survival; the recovery claim
        // on the face in default stays
        let hazard = self.hazard.clone();
        Ok(chassis - convertible.principal_pv(curve, settlement, |t| hazard.survival(t)))
    }

    /// Flows weighted by survival, plus the recovery on the claim paid
    /// at the end of the period in which default happens.
    fn value_of_flows(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        flows: &[(f64, f64)],
        claim: f64,
    ) -> f64 {
        value_of_flows_with_survival(curve, settlement, flows, claim, self.recovery_rate, |t| {
            self.hazard.survival(t)
        })
    }

    fn with_spot(&self, spot: f64) -> Self {
        JumpToDefaultMarket {
            spot,
            ..self.clone()
        }
    }

    fn with_volatility(&self, volatility: f64) -> Self {
        JumpToDefaultMarket {
            volatility,
            ..self.clone()
        }
    }

    /// Every term of the hazard shifted by `bump`.
    fn with_credit_bump(&self, bump: f64) -> Self {
        JumpToDefaultMarket {
            hazard: self
                .hazard
                .shifted(bump)
                .expect("a shifted hazard level keeps its pillars"),
            ..self.clone()
        }
    }
}

// ---------------------------------------------------------------------
// Equity-linked hazard
// ---------------------------------------------------------------------

/// Equity and credit inputs for the equity-linked hazard model: jump to
/// default with a default intensity that rises as the share price
/// falls,
///
/// ```text
/// lambda(S, t) = a(t) (S0 / S)^p
/// ```
///
/// with `a(t)` the level ([`hazard`](Self::hazard), flat or a term
/// structure: the intensity at the reference price), `S0` the reference
/// price and `p` the elasticity in `[0, 2]`. At `p = 0` the model is
/// exactly [`JumpToDefaultMarket`]; at `p = 1` the hazard doubles when
/// the stock halves; at `p = 2` it quadruples. The conditional drift
/// and the survival factor follow the node's hazard, so the bond floor
/// is no longer flat in the stock: a busted convertible prices lower
/// and carries a credit delta, which the flat models miss. Above the
/// reference the current hazard is below the level, yet the price still
/// sits at or below jump to default: the credit exposure lives in the
/// states where the stock has fallen, where the hazard exceeds the
/// level, so the gap narrows with the share price rather than changing
/// sign.
///
/// `a(t)` is not the CDS hazard: for `p > 0` the model's unconditional
/// survival sits below `exp(-integral of a)`, because the hazard is
/// convex in the share price, so the level that reproduces a CDS curve
/// sits *below* its hazards. [`calibrated_to`](Self::calibrated_to)
/// solves the term structure through the model so that its survival
/// probabilities match the curve's.
///
/// On the tree the hazard is capped at the lattice's admissible
/// maximum, roughly the volatility over the square root of the step,
/// which binds only at deep out-of-the-money nodes; the grid applies
/// no cap and switches to upwind differences where the drift dominates.
/// The straight floor is priced on the grid with the conversion right
/// removed, so it moves with the stock (at zero elasticity it is the
/// analytic hazard floor); the analytic floor of the preferred uses
/// the level `a(t)`.
#[derive(Debug, Clone)]
pub struct EquityLinkedHazardMarket {
    /// Share price.
    pub spot: f64,
    /// Flat lognormal equity volatility of the pre-default diffusion.
    pub volatility: f64,
    /// Continuous dividend yield `q`.
    pub dividend_yield: f64,
    /// Continuous stock borrow cost `b`.
    pub borrow_cost: f64,
    /// The hazard level `a(t)`: the intensity at the reference price.
    pub hazard: HazardLevel,
    /// Fraction of the outstanding face recovered on default, in
    /// `[0, 1]`.
    pub recovery_rate: f64,
    /// The reference price `S0` at which the hazard equals `a(t)`.
    pub reference_spot: f64,
    /// The elasticity `p` in `[0, 2]`; zero is jump to default.
    pub elasticity: f64,
}

impl EquityLinkedHazardMarket {
    /// The model around a jump-to-default snapshot: the same inputs,
    /// the snapshot's spot as the reference price, and `elasticity`.
    pub fn new(base: JumpToDefaultMarket, elasticity: f64) -> Result<Self, RustyQLibError> {
        let market = EquityLinkedHazardMarket {
            spot: base.spot,
            volatility: base.volatility,
            dividend_yield: base.dividend_yield,
            borrow_cost: base.borrow_cost,
            hazard: base.hazard,
            recovery_rate: base.recovery_rate,
            reference_spot: base.spot,
            elasticity,
        };
        market.validate()?;
        Ok(market)
    }

    /// The hazard at a share price and time from settlement.
    pub fn hazard_at(&self, spot: f64, t: f64) -> f64 {
        self.hazard.at(t)
            * (self.reference_spot / spot.max(f64::MIN_POSITIVE)).powf(self.elasticity)
    }

    /// The jump-to-default model at the level `a(t)`, for the analytic
    /// floors.
    pub(crate) fn at_reference(&self) -> JumpToDefaultMarket {
        JumpToDefaultMarket {
            spot: self.spot,
            volatility: self.volatility,
            dividend_yield: self.dividend_yield,
            borrow_cost: self.borrow_cost,
            hazard: self.hazard.clone(),
            recovery_rate: self.recovery_rate,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), RustyQLibError> {
        self.at_reference().validate()?;
        if !(self.reference_spot > 0.0 && self.reference_spot.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!(
                    "the reference spot must be positive, got {}",
                    self.reference_spot
                ),
            ));
        }
        if !(self.elasticity.is_finite() && (0.0..=2.0).contains(&self.elasticity)) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!(
                    "the hazard elasticity must be in [0, 2], got {}",
                    self.elasticity
                ),
            ));
        }
        Ok(())
    }
}

impl CreditModel for EquityLinkedHazardMarket {
    type Node = f64;

    fn validate(&self) -> Result<(), RustyQLibError> {
        EquityLinkedHazardMarket::validate(self)
    }

    fn equity(&self) -> EquityInputs {
        EquityInputs {
            spot: self.spot,
            volatility: self.volatility,
            dividend_yield: self.dividend_yield,
            borrow_cost: self.borrow_cost,
        }
    }

    /// The level `a` at the long end: what at-stake coupons discount at
    /// and a perpetual tail prices at.
    fn credit_rate(&self) -> f64 {
        self.hazard.long_end()
    }

    fn survival_drift(&self) -> f64 {
        self.hazard.long_end() - self.dividend_yield - self.borrow_cost
    }

    fn credit_rate_at(&self, spot: f64, t: f64) -> f64 {
        self.hazard_at(spot, t)
    }

    fn survival_drift_at(&self, spot: f64, t: f64) -> f64 {
        self.hazard_at(spot, t) - self.dividend_yield - self.borrow_cost
    }

    fn is_state_dependent(&self) -> bool {
        self.elasticity > 0.0
    }

    /// As jump to default, with the node's own survival factor.
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

    /// The survival branch discounts at `r + lambda(S, t)` and the
    /// default branch pays the recovery with the node's default
    /// probability.
    fn pde_step(&self, nodes: &[f64], diffuse: &Diffusion<'_>, step: &StepContext) -> Vec<f64> {
        let hazards: Vec<f64> = step
            .spots
            .iter()
            .map(|&s| self.hazard_at(s, step.t))
            .collect();
        let rho: Vec<f64> = hazards.iter().map(|h| step.r + h).collect();
        diffuse(nodes.to_vec(), &rho)
            .into_iter()
            .zip(&hazards)
            .map(|(v, h)| {
                v + step.riskfree_df
                    * (1.0 - (-h * step.dt).exp())
                    * self.recovery_rate
                    * step.default_claim
            })
            .collect()
    }

    /// The chassis with the conversion right, calls, puts and extras
    /// removed, priced on the grid under the state-dependent hazard —
    /// so the floor moves with the stock. At zero elasticity, and for a
    /// mandatory, it is the level's analytic floor.
    fn bond_floor(
        &self,
        convertible: &ConvertibleBond,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !self.is_state_dependent() || convertible.mandatory.is_some() {
            return self
                .at_reference()
                .bond_floor(convertible, curve, settlement);
        }
        let mut straight = convertible.clone();
        straight.calls.clear();
        straight.puts.clear();
        straight.soft_call_trigger = None;
        straight.contingent_conversion = None;
        straight.coupon_make_whole = None;
        straight.fundamental_change = None;
        // a window that never opens
        straight.convert_from = Some(convertible.bond.maturity_date);
        straight.convert_until = Some(convertible.bond.dated_date);
        Ok(valuation(
            &straight,
            self,
            curve,
            settlement,
            ConvertibleFdGrid::default(),
            &FdVolModel::Flat,
        )?
        .clean_price)
    }

    /// At the level `a(t)` (the flows carry no share-price state of
    /// their own).
    fn value_of_flows(
        &self,
        curve: &YieldCurve,
        settlement: NaiveDate,
        flows: &[(f64, f64)],
        claim: f64,
    ) -> f64 {
        self.at_reference()
            .value_of_flows(curve, settlement, flows, claim)
    }

    /// The reference price stays put: a spot bump moves the hazard
    /// along the curve, which is the credit delta.
    fn with_spot(&self, spot: f64) -> Self {
        EquityLinkedHazardMarket {
            spot,
            ..self.clone()
        }
    }

    fn with_volatility(&self, volatility: f64) -> Self {
        EquityLinkedHazardMarket {
            volatility,
            ..self.clone()
        }
    }

    /// Every term of the level shifted by `bump`.
    fn with_credit_bump(&self, bump: f64) -> Self {
        EquityLinkedHazardMarket {
            hazard: self
                .hazard
                .shifted(bump)
                .expect("a shifted hazard level keeps its pillars"),
            ..self.clone()
        }
    }
}
