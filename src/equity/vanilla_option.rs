use std::sync::Arc;
use chrono::NaiveDate;
use crate::equity::{baw,bjerksund_stensland,binomial,finite_difference,greeks,montecarlo};
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::vols::VolSurface;
use crate::equity::asian::{AsianStrikeType, AveragingType};
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::builder::EquityOptionBuilder;
use crate::equity::heston;
use super::super::core::quotes::Quote;
use super::super::core::traits::Instrument;
use super::blackscholes;
use crate::equity::utils::{Engine, Model, Payoff, PayoffType, PricingEngine, LongShort};
use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use blackscholes::BlackScholesPricer;
use crate::core::data_models::EquityOptionData;
use crate::core::errors::RustyQLibError;
use crate::core::results::PricingResult;

#[derive(Debug, Clone)]
pub struct VanillaPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
}

/// Binary (digital) settlement style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryType {
    /// Pays a fixed cash amount when in the money.
    CashOrNothing,
    /// Delivers the underlying (pays its level) when in the money.
    AssetOrNothing,
}

/// Lookback flavor: floating strike pays against the path extremum,
/// fixed strike pays the extremum against a fixed strike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookbackType {
    FloatingStrike,
    FixedStrike,
}

/// Lookback payoff: the call watches the minimum (floating) or maximum
/// (fixed), the put the mirror image. Discretely monitored on the path
/// grid under Monte Carlo; the analytic engine prices the continuous-
/// monitoring closed forms.
#[derive(Debug, Clone)]
pub struct LookbackPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    pub lookback_type: LookbackType,
}

impl Payoff for LookbackPayoff {
    /// Degenerate single-point value (fresh option): zero.
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, path: &[f64], strike: f64) -> f64 {
        let terminal = *path.last().expect("empty path");
        let max = path.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = path.iter().cloned().fold(f64::INFINITY, f64::min);
        match (self.lookback_type, &self.put_or_call) {
            (LookbackType::FloatingStrike, PutOrCall::Call) => terminal - min,
            (LookbackType::FloatingStrike, PutOrCall::Put) => max - terminal,
            (LookbackType::FixedStrike, PutOrCall::Call) => (max - strike).max(0.0),
            (LookbackType::FixedStrike, PutOrCall::Put) => (strike - min).max(0.0),
        }
    }
    fn path_payoff_var<'t>(
        &self,
        path: &[crate::core::aad::Var<'t>],
        strike: f64,
    ) -> Option<crate::core::aad::Var<'t>> {
        let terminal = *path.last().expect("empty path");
        let mut max = path[0];
        let mut min = path[0];
        for s in &path[1..] {
            max = max.max(*s);
            min = min.min(*s);
        }
        Some(match (self.lookback_type, &self.put_or_call) {
            (LookbackType::FloatingStrike, PutOrCall::Call) => terminal - min,
            (LookbackType::FloatingStrike, PutOrCall::Put) => max - terminal,
            (LookbackType::FixedStrike, PutOrCall::Call) => (max - strike).maxf(0.0),
            (LookbackType::FixedStrike, PutOrCall::Put) => (strike - min).maxf(0.0),
        })
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Lookback
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}

#[derive(Debug, Clone)]
pub struct BinaryPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    pub binary_type: BinaryType,
    /// Amount paid by a cash-or-nothing binary (ignored for asset-or-nothing).
    pub cash: f64,
}
#[derive(Debug, Clone)]
pub struct BarrierPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    pub direction: BarrierDirection,
    pub knock: KnockType,
    pub barrier: f64,
    /// Second barrier level: `Some` makes this a **double** barrier (the
    /// corridor `[min, max]` of the two levels; `direction` is ignored).
    pub barrier2: Option<f64>,
    /// Rebate paid on the knock event (knock-out) or at expiry when the
    /// option never knocks in (knock-in). Zero = no rebate.
    pub rebate: f64,
    /// Knock-out rebate timing: at the touch (`true`, analytic engine
    /// only) or at expiry (`false`; also the Monte Carlo convention).
    pub rebate_at_hit: bool,
}

/// Barrier payoff: `payoff` is the underlying vanilla leg (used by the
/// analytic building blocks and as the terminal leg of path pricing);
/// `path_payoff` applies discretely monitored knock logic to a full path.
/// The Monte Carlo engine additionally applies a Brownian-bridge crossing
/// correction, so its effective monitoring is continuous.
impl Payoff for BarrierPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match &self.put_or_call {
            PutOrCall::Call => (spot - strike).max(0.0),
            PutOrCall::Put => (strike - spot).max(0.0),
        }
    }
    fn path_payoff(&self, path: &[f64], strike: f64) -> f64 {
        let crossed = match self.barrier2 {
            Some(b2) => {
                let (lo, hi) = (self.barrier.min(b2), self.barrier.max(b2));
                path.iter().any(|&s| s <= lo || s >= hi)
            }
            None => path.iter().any(|&s| match self.direction {
                BarrierDirection::Up => s >= self.barrier,
                BarrierDirection::Down => s <= self.barrier,
            }),
        };
        let alive = match self.knock {
            KnockType::Out => !crossed,
            KnockType::In => crossed,
        };
        // rebate legs pay at expiry under path pricing: a knocked-out
        // path collects the rebate, a never-in path of a knock-in does
        let rebate = match self.knock {
            KnockType::Out if crossed => self.rebate,
            KnockType::In if !crossed => self.rebate,
            _ => 0.0,
        };
        let payoff = if alive {
            self.payoff(*path.last().expect("empty path"), strike)
        } else {
            0.0
        };
        payoff + rebate
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Barrier
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}
#[derive(Debug, Clone)]
pub struct AsianPayoff {
    pub put_or_call: PutOrCall,
    pub exercise_style: ContractStyle,
    pub averaging: AveragingType,
    pub strike_type: AsianStrikeType,
}

/// Asian payoff: the average is taken over the monitored path points
/// (equally spaced, spot excluded). Fixed strike pays on the average
/// against the strike; floating strike pays on the terminal spot against
/// the average.
impl Payoff for AsianPayoff {
    /// Degenerate single-point average (used for intrinsic display only;
    /// engines route Asians through `path_payoff`).
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match &self.put_or_call {
            PutOrCall::Call => (spot - strike).max(0.0),
            PutOrCall::Put => (strike - spot).max(0.0),
        }
    }
    fn path_payoff(&self, path: &[f64], strike: f64) -> f64 {
        let n = path.len() as f64;
        let average = match self.averaging {
            AveragingType::Arithmetic => path.iter().sum::<f64>() / n,
            AveragingType::Geometric => (path.iter().map(|s| s.ln()).sum::<f64>() / n).exp(),
        };
        let terminal = *path.last().expect("empty path");
        let (long_leg, short_leg) = match self.strike_type {
            AsianStrikeType::FixedStrike => (average, strike),
            AsianStrikeType::FloatingStrike => (terminal, average),
        };
        match &self.put_or_call {
            PutOrCall::Call => (long_leg - short_leg).max(0.0),
            PutOrCall::Put => (short_leg - long_leg).max(0.0),
        }
    }
    fn path_payoff_var<'t>(
        &self,
        path: &[crate::core::aad::Var<'t>],
        strike: f64,
    ) -> Option<crate::core::aad::Var<'t>> {
        let n = path.len() as f64;
        let average = match self.averaging {
            AveragingType::Arithmetic => {
                let mut sum = path[0];
                for s in &path[1..] {
                    sum = sum + *s;
                }
                sum / n
            }
            AveragingType::Geometric => {
                let mut sum = path[0].ln();
                for s in &path[1..] {
                    sum = sum + s.ln();
                }
                (sum / n).exp()
            }
        };
        let terminal = *path.last().expect("empty path");
        Some(match (self.strike_type, &self.put_or_call) {
            (AsianStrikeType::FixedStrike, PutOrCall::Call) => (average - strike).maxf(0.0),
            (AsianStrikeType::FixedStrike, PutOrCall::Put) => (strike - average).maxf(0.0),
            (AsianStrikeType::FloatingStrike, PutOrCall::Call) => (terminal - average).maxf(0.0),
            (AsianStrikeType::FloatingStrike, PutOrCall::Put) => (average - terminal).maxf(0.0),
        })
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Asian
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}
impl Payoff for VanillaPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        match &self.put_or_call {
            PutOrCall::Call => (spot - strike).max(0.0),
            PutOrCall::Put => (strike - spot).max(0.0),
        }
    }
    fn path_payoff_var<'t>(
        &self,
        path: &[crate::core::aad::Var<'t>],
        strike: f64,
    ) -> Option<crate::core::aad::Var<'t>> {
        let terminal = *path.last().expect("empty path");
        Some(match self.put_or_call {
            PutOrCall::Call => (terminal - strike).maxf(0.0),
            PutOrCall::Put => (strike - terminal).maxf(0.0),
        })
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Vanilla
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}

/// Binary (digital) payoff, strictly in the money beyond the strike:
/// cash-or-nothing pays `cash`, asset-or-nothing pays the underlying level.
impl Payoff for BinaryPayoff {
    fn payoff(&self, spot: f64, strike: f64) -> f64 {
        let in_the_money = match &self.put_or_call {
            PutOrCall::Call => spot > strike,
            PutOrCall::Put => spot < strike,
        };
        if !in_the_money {
            return 0.0;
        }
        match self.binary_type {
            BinaryType::CashOrNothing => self.cash,
            BinaryType::AssetOrNothing => spot,
        }
    }
    fn payoff_kind(&self) -> PayoffType {
        PayoffType::Binary
    }
    fn put_or_call(&self) -> &PutOrCall {
        &self.put_or_call
    }
    fn exercise_style(&self) -> &ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Payoff> {
        Box::new(self.clone())
    }
}

/// Contract terms and trade identity — **no market state**. Immutable
/// for the life of the trade; everything that moves with the market
/// lives in [`EquityMarketData`].
#[derive(Debug, Clone)]
pub struct EquityOptionBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    pub strike_price: f64,
    pub maturity_date: NaiveDate,
    /// When set, the underlying is a future priced with Black-76
    /// (the market spot is the futures price `F`), settled either with an
    /// up-front discounted premium or futures-style margined. European
    /// vanilla only, on the Analytical engine.
    pub futures_settlement: Option<crate::equity::black76::FuturesSettlement>,
    pub multiplier: f64,

    // trade info (candidate for a future Trade struct)
    pub current_price: Quote,
    pub entry_price: f64,
    pub long_short: LongShort,
}

/// The market state one equity instrument is currently **bound to** — the
/// pricing-view companion of the contract (QuantLib's process, Strata's
/// provider). Resolved from / snapshotted to the typed
/// [`Market`](crate::core::market::Market) store; swapped wholesale by
/// [`EquityOption::with_market`].
#[derive(Debug, Clone)]
pub struct EquityMarketData {
    /// The as-of date of this market snapshot; anchors every year fraction.
    pub valuation_date: NaiveDate,
    pub spot: Quote,
    pub dividend_yield: f64,
    /// Continuous stock borrow (repo) cost; part of the carry alongside
    /// the dividend yield.
    pub borrow_cost: f64,
    /// Discrete cash dividend forecasts (ex-date, amount per share).
    /// Analytic, tree and terminal Monte Carlo engines use the escrowed
    /// model (spot minus PV of dividends); path-wise Monte Carlo and
    /// finite difference apply the jumps at the ex-dates.
    pub cash_dividends: Vec<(NaiveDate, f64)>,
    /// Volatility surface; a flat surface represents a single constant vol.
    /// `Arc`-shared with the [`Market`](crate::core::market::Market) store:
    /// rebinding an instrument is a refcount bump, and replacing the
    /// surface means installing a **new** `Arc` (copy-on-write), never
    /// mutating through it.
    pub vol_surface: Arc<VolSurface>,
    /// Discounting curve anchored at `valuation_date`; discount factors are
    /// the source of truth, rates are derived views. `Arc`-shared and
    /// copy-on-write, like `vol_surface`.
    pub discount_curve: Arc<YieldCurve>,
}

#[derive(Debug)]
pub struct EquityOption {
    /// The contract (and trade identity): pure data, never market state.
    pub base: EquityOptionBase,
    /// The market this instrument is currently bound to.
    pub market: EquityMarketData,
    pub payoff: Box<dyn Payoff>,
    /// The numerical method, carrying its own settings.
    pub engine: PricingEngine,
    /// The dynamics of the underlying (GBM, local vol, or Heston with
    /// its parameters); consulted by the MC, FD and analytic engines.
    pub model: Model,
}

// manual impl: `payoff` clones through the trait object, engine and model
// are Copy
impl Clone for EquityOption {
    fn clone(&self) -> Self {
        EquityOption {
            base: self.base.clone(),
            market: self.market.clone(),
            payoff: self.payoff.clone_box(),
            engine: self.engine,
            model: self.model,
        }
    }
}

impl EquityOption {
    /// The Monte Carlo settings. Invariant: only called on the Monte
    /// Carlo engine's code paths (the dispatch guarantees it).
    pub(crate) fn mc_cfg(&self) -> &montecarlo::MonteCarloConfig {
        match &self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    pub(crate) fn fd_cfg(&self) -> &finite_difference::FdConfig {
        match &self.engine {
            PricingEngine::FiniteDifference(cfg) => cfg,
            _ => unreachable!("finite-difference code path reached on a non-FD engine"),
        }
    }

    /// Heston parameters. Invariant: only called on Heston-model code
    /// paths (the model dispatch guarantees it).
    pub(crate) fn heston_params(&self) -> &crate::equity::heston::HestonParams {
        match &self.model {
            Model::Heston(hp) => hp,
            _ => unreachable!("Heston code path reached on a non-Heston model"),
        }
    }

    pub(crate) fn lattice_cfg(&self) -> &crate::core::lattice::LatticeConfig {
        match &self.engine {
            PricingEngine::Binomial(cfg) => cfg,
            _ => unreachable!("lattice code path reached on a non-Binomial engine"),
        }
    }

    /// Test-only shortcuts for tweaking engine configuration in place;
    /// production code configures engines through the builder.
    #[cfg(test)]
    pub(crate) fn mc_cfg_mut(&mut self) -> &mut montecarlo::MonteCarloConfig {
        match &mut self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    #[cfg(test)]
    pub(crate) fn fd_cfg_mut(&mut self) -> &mut finite_difference::FdConfig {
        match &mut self.engine {
            PricingEngine::FiniteDifference(cfg) => cfg,
            _ => unreachable!("finite-difference code path reached on a non-FD engine"),
        }
    }
}
impl EquityOption {

    /// Build an option from contract data, panicking on any invalid field.
    /// Fallible callers (batch pricing, services) should use
    /// [`EquityOption::try_from_json`].
    pub fn from_json(data: &EquityOptionData) -> Box<EquityOption> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build an option from contract data, reporting the offending field in
    /// the error instead of panicking.
    ///
    /// This is a thin translation layer: it parses JSON-level fields
    /// (dates, enum strings) into typed values and feeds them through
    /// [`EquityOptionBuilder`], which owns all domain validation and
    /// assembly — both construction paths share one set of checks.
    pub fn try_from_json(data: &EquityOptionData) -> Result<Box<EquityOption>, RustyQLibError> {
        let valuation_date =
            crate::core::data_models::parse_valuation_date(data.base.valuation_date.as_deref())?;
        let maturity_date = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d")
            .map_err(|_| RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            ))?;
        let payoff_type = data.payoff_type.parse::<PayoffType>()
            .map_err(|_| RustyQLibError::invalid_input(
                "payoff_type",
                format!("unknown payoff_type '{}'", data.payoff_type),
            ))?;
        let strike_price = match payoff_type {
            // strike is set by the contract mechanics for these payoffs
            PayoffType::ForwardStart | PayoffType::Autocallable => {
                data.strike_price.unwrap_or(0.0)
            }
            _ => data.strike_price.ok_or_else(|| RustyQLibError::invalid_input(
                "strike_price",
                "strike_price is required for this payoff",
            ))?,
        };

        let mut builder = EquityOptionBuilder::new()
            .symbol(&data.base.symbol)
            .spot(data.base.underlying_price)
            .strike(strike_price)
            .valuation_date(valuation_date)
            .maturity_date(maturity_date)
            .dividend_yield(data.dividend.unwrap_or(0.0))
            .borrow_cost(data.base.borrow_cost.unwrap_or(0.0));

        // ── market objects ──────────────────────────────────────────────
        builder = match &data.discount_curve {
            Some(input) => builder.discount_curve(YieldCurve::from_input(input, valuation_date)?),
            None => builder.flat_rate(data.base.risk_free_rate.unwrap_or(0.0)),
        };
        builder = match &data.vol_surface {
            Some(input) => builder.vol_surface(VolSurface::from_input(input, valuation_date)?),
            None => builder.flat_vol(data.volatility.ok_or_else(|| {
                RustyQLibError::invalid_input(
                    "volatility",
                    "either volatility or vol_surface must be provided",
                )
            })?),
        };
        for d in data.cash_dividends.as_deref().unwrap_or(&[]) {
            let date = NaiveDate::parse_from_str(&d.date, "%Y-%m-%d")
                .map_err(|_| RustyQLibError::invalid_input(
                    "cash_dividends",
                    format!("invalid dividend date '{}' (expected YYYY-MM-DD)", d.date),
                ))?;
            builder = builder.cash_dividend(date, d.amount);
        }
        if let Some(s) = data.futures_settlement.as_deref() {
            let settlement = s
                .parse::<crate::equity::black76::FuturesSettlement>()
                .map_err(|_| RustyQLibError::invalid_input(
                    "futures_settlement",
                    format!("invalid futures_settlement '{s}' (use 'discounted' or 'margined')"),
                ))?;
            builder = builder.on_future(settlement);
        }

        // ── exercise style ──────────────────────────────────────────────
        builder = match data.exercise_style.as_deref().unwrap_or("European").trim() {
            "American" | "american" => builder.american(),
            "Bermudan" | "bermudan" => {
                let dates = data.exercise_dates.as_deref().ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "exercise_dates",
                        "exercise_dates is required when exercise_style is Bermudan",
                    )
                })?;
                builder.bermudan(parse_date_list("exercise_dates", dates)?)
            }
            // unknown styles fall back to European, as before
            _ => builder,
        };

        let side = match data.put_or_call.trim() {
            "C" | "c" | "Call" | "call" => PutOrCall::Call,
            "P" | "p" | "Put" | "put" => PutOrCall::Put,
            other => return Err(RustyQLibError::invalid_input(
                "put_or_call",
                format!("invalid side '{other}' (use 'C' or 'P')"),
            )),
        };

        // ── payoff ──────────────────────────────────────────────────────
        builder = match payoff_type {
            // not reachable from JSON yet: PayoffType::from_str does not
            // produce Accumulator; build through EquityOptionBuilder
            PayoffType::Accumulator => {
                return Err(RustyQLibError::invalid_input(
                    "payoff_type",
                    "accumulators are built through EquityOptionBuilder::accumulator, \
                     not JSON contract data",
                ));
            }
            PayoffType::Vanilla => builder.vanilla(side),
            PayoffType::Binary => {
                let binary_type = match data
                    .binary_type
                    .as_deref()
                    .unwrap_or("cash")
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "cash" | "cash_or_nothing" | "cash-or-nothing" => BinaryType::CashOrNothing,
                    "asset" | "asset_or_nothing" | "asset-or-nothing" => BinaryType::AssetOrNothing,
                    other => return Err(RustyQLibError::invalid_input(
                        "binary_type",
                        format!("invalid binary_type '{other}' (use 'cash' or 'asset')"),
                    )),
                };
                builder.binary(side, binary_type, data.cash_amount.unwrap_or(1.0))
            }
            PayoffType::Lookback => {
                let lookback_type = match data
                    .lookback_type
                    .as_deref()
                    .unwrap_or("floating")
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "floating" | "floating_strike" => LookbackType::FloatingStrike,
                    "fixed" | "fixed_strike" => LookbackType::FixedStrike,
                    other => return Err(RustyQLibError::invalid_input(
                        "lookback_type",
                        format!("invalid lookback_type '{other}' (use 'floating' or 'fixed')"),
                    )),
                };
                builder.lookback(side, lookback_type)
            }
            PayoffType::Barrier => {
                let barrier = data
                    .barrier_level
                    .ok_or_else(|| RustyQLibError::invalid_input(
                        "barrier_level",
                        "barrier_level is required for barrier options",
                    ))?;
                let (direction, knock) = match data
                    .barrier_type
                    .as_deref()
                    .unwrap_or("")
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "up_in" | "up-in" | "ui" => (BarrierDirection::Up, KnockType::In),
                    "up_out" | "up-out" | "uo" => (BarrierDirection::Up, KnockType::Out),
                    "down_in" | "down-in" | "di" => (BarrierDirection::Down, KnockType::In),
                    "down_out" | "down-out" | "do" => (BarrierDirection::Down, KnockType::Out),
                    other => return Err(RustyQLibError::invalid_input(
                        "barrier_type",
                        format!("barrier_type must be up_in/up_out/down_in/down_out, got '{other}'"),
                    )),
                };
                // a second level makes it a double barrier: the corridor
                // between the two levels (direction is then ignored)
                let b = match data.barrier_level2 {
                    Some(b2) => builder.double_barrier(
                        side,
                        knock,
                        barrier.min(b2),
                        barrier.max(b2),
                    ),
                    None => builder.barrier(side, direction, knock, barrier),
                };
                b.barrier_rebate(
                    data.rebate.unwrap_or(0.0),
                    data.rebate_at_hit.unwrap_or(false),
                )
            }
            PayoffType::Asian => {
                let averaging = match data
                    .averaging_type
                    .as_deref()
                    .unwrap_or("arithmetic")
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "arithmetic" | "arith" => AveragingType::Arithmetic,
                    "geometric" | "geo" => AveragingType::Geometric,
                    other => return Err(RustyQLibError::invalid_input(
                        "averaging_type",
                        format!("averaging_type must be arithmetic or geometric, got '{other}'"),
                    )),
                };
                let strike_type = match data
                    .asian_strike_type
                    .as_deref()
                    .unwrap_or("fixed")
                    .trim()
                    .to_lowercase()
                    .as_str()
                {
                    "fixed" | "average_price" => AsianStrikeType::FixedStrike,
                    "floating" | "average_strike" => AsianStrikeType::FloatingStrike,
                    other => return Err(RustyQLibError::invalid_input(
                        "asian_strike_type",
                        format!("asian_strike_type must be fixed or floating, got '{other}'"),
                    )),
                };
                builder.asian(side, averaging, strike_type)
            }
            PayoffType::ForwardStart => {
                let start_date_str = data
                    .forward_start_date
                    .as_ref()
                    .ok_or_else(|| RustyQLibError::invalid_input(
                        "forward_start_date",
                        "forward_start_date is required for forward-start options",
                    ))?;
                let start_date = NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d")
                    .map_err(|_| RustyQLibError::invalid_input(
                        "forward_start_date",
                        format!("invalid date '{start_date_str}' (expected YYYY-MM-DD)"),
                    ))?;
                if !(start_date > valuation_date && start_date < maturity_date) {
                    return Err(RustyQLibError::invalid_input(
                        "forward_start_date",
                        "forward_start_date must lie between valuation and maturity",
                    ));
                }
                let start_fraction = (start_date - valuation_date).num_days() as f64
                    / (maturity_date - valuation_date).num_days() as f64;
                builder.forward_start(
                    side,
                    data.strike_fraction.unwrap_or(1.0),
                    start_fraction,
                )
            }
            PayoffType::Autocallable => {
                let autocall_barrier = data
                    .autocall_barrier
                    .ok_or_else(|| RustyQLibError::invalid_input(
                        "autocall_barrier",
                        "autocall_barrier is required for autocallables",
                    ))?;
                let protection_barrier = data
                    .protection_barrier
                    .ok_or_else(|| RustyQLibError::invalid_input(
                        "protection_barrier",
                        "protection_barrier is required for autocallables",
                    ))?;
                let coupon = data.autocall_coupon.unwrap_or(0.0);
                let observations = data.autocall_observations.unwrap_or(4).max(1);
                let notional = data.notional.unwrap_or(100.0);
                // a coupon barrier makes it a phoenix; memory is inert
                // without one
                let mut b = match data.coupon_barrier {
                    Some(coupon_barrier) => builder.phoenix(
                        autocall_barrier,
                        coupon_barrier,
                        protection_barrier,
                        coupon,
                        observations,
                        notional,
                        data.coupon_memory.unwrap_or(false),
                    ),
                    None => builder.autocallable(
                        autocall_barrier,
                        protection_barrier,
                        coupon,
                        observations,
                        notional,
                    ),
                };
                if let Some(dates) = data.autocall_observation_dates.as_deref() {
                    b = b.autocall_observation_dates(parse_date_list(
                        "autocall_observation_dates",
                        dates,
                    )?);
                }
                b
            }
        };

        // ── engine (it carries only its own configuration) ──────────────
        let engine_kind = match data.pricer.as_ref().map_or("Analytical", |v| v).trim() {
            "Analytical" | "analytical" | "bs" => Engine::BlackScholes,
            "MonteCarlo" | "montecarlo" | "MC" | "mc" => Engine::MonteCarlo,
            "Binomial" | "binomial" | "bino" => Engine::Binomial,
            "FiniteDifference" | "finitdifference" | "FD" | "fd" => Engine::FiniteDifference,
            "BaroneAdesiWhaley" | "baw" | "BAW" => Engine::BaroneAdesiWhaley,
            "BjerksundStensland" | "bjerksund_stensland" | "bs2002" | "BS2002" => {
                Engine::BjerksundStensland
            }
            other => {
                return Err(RustyQLibError::invalid_input(
                    "pricer",
                    format!(
                        "unknown pricer '{other}' (use Analytical, MonteCarlo, Binomial, \
                         FiniteDifference, BAW or BS2002)"
                    ),
                ));
            }
        };
        builder = match &engine_kind {
            Engine::MonteCarlo => {
                builder.mc_config(montecarlo::MonteCarloConfig::from_data(data)?)
            }
            Engine::FiniteDifference => {
                builder.fd_config(finite_difference::FdConfig::from_data(data))
            }
            Engine::Binomial => {
                let defaults = crate::core::lattice::LatticeConfig::default();
                builder.lattice_config(crate::core::lattice::LatticeConfig {
                    tree_type: match data.tree_type.as_deref() {
                        Some(s) => s.parse()?,
                        None => defaults.tree_type,
                    },
                    steps: data.tree_steps.unwrap_or(defaults.steps),
                    term_structure: data.tree_term_structure.unwrap_or(false),
                })
            }
            _ => builder,
        };
        builder = builder
            .engine(engine_kind)
            .model(Model::from_contract(data.mc_model.as_deref(), data.heston)?);

        let mut option = builder.build()?;

        // trade and reporting metadata the builder does not model
        option.base.currency = data.base.currency.clone();
        option.base.exchange = data.base.exchange.clone();
        option.base.name = data.base.name.clone();
        option.base.cusip = data.base.cusip.clone();
        option.base.isin = data.base.isin.clone();
        option.base.settlement_type = data.base.settlement_type.clone();
        option.base.multiplier = data.multiplier.unwrap_or(1.0);
        option.base.current_price = Quote::new(data.current_price.unwrap_or(0.0));
        option.base.entry_price = data.entry_price.unwrap_or(0.0);
        Ok(Box::new(option))
    }
}

/// Parse a JSON date-string list into `NaiveDate`s. Ordering and range
/// validation happens in [`EquityOptionBuilder::build`]; this only
/// handles the string format, naming `field` in errors.
fn parse_date_list(field: &str, dates: &[String]) -> Result<Vec<NaiveDate>, RustyQLibError> {
    dates
        .iter()
        .map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    field,
                    format!("invalid date '{s}' (expected YYYY-MM-DD)"),
                )
            })
        })
        .collect()
}

impl EquityOptionBase {
    /// True when the underlying is a future priced with Black-76.
    pub fn is_futures_option(&self) -> bool {
        self.futures_settlement.is_some()
    }
    /// The contract's currency code, falling back to
    /// [`DEFAULT_CURRENCY`](crate::core::market::DEFAULT_CURRENCY) when the
    /// contract does not state one — used as the [`Discount`]
    /// (crate::core::market::Discount) key into a [`Market`]
    /// (crate::core::market::Market).
    pub fn currency_code(&self) -> &str {
        self.currency.as_deref().unwrap_or(crate::core::market::DEFAULT_CURRENCY)
    }
}

// Contract-and-market quantities: these straddle the boundary (a year
// fraction needs the contract's maturity AND the market's valuation
// date), so they live on the pairing — the option — not on either half.
impl EquityOption {
    pub fn time_to_maturity(&self) -> f64 {
        (self.base.maturity_date - self.market.valuation_date).num_days() as f64 / 365.0
    }
    /// Discount factor from the valuation date to maturity, off the curve.
    pub fn maturity_discount_factor(&self) -> f64 {
        self.market.discount_curve.df(self.time_to_maturity())
    }
    /// Continuously compounded zero rate to maturity implied by the curve.
    /// This is the `r` that enters d1/d2; it is consistent with
    /// [`maturity_discount_factor`](Self::maturity_discount_factor) by construction.
    pub fn risk_free_rate(&self) -> f64 {
        self.market
            .discount_curve
            .zero_rate_with(self.time_to_maturity(), Compounding::Continuous)
    }
    /// Total continuous carry on the underlying: dividend yield plus
    /// borrow cost. This is the "q" every pricing formula uses.
    pub fn carry_yield(&self) -> f64 {
        self.market.dividend_yield + self.market.borrow_cost
    }
    /// Escrow value of the cash dividends with ex-dates inside the option's
    /// life: the amount to carve out of spot so the risky stub reproduces
    /// the jump-model forward.
    ///
    /// Each dividend is discounted at the **net carry rate** `r - carry`,
    /// not the risk-free rate, so that the escrow accretes at the same rate
    /// the risky stub grows (`effective_spot` is grown at `r - carry` in
    /// [`forward_price`](Self::forward_price)). This makes the analytic
    /// forward match the well-defined jump model
    /// `F = (S - D e^{-(r-carry)t}) e^{(r-carry)T}` used by the FD and
    /// path-wise Monte Carlo engines. With no continuous carry this
    /// reduces to plain risk-free discounting.
    pub fn pv_cash_dividends(&self) -> f64 {
        let carry = self.carry_yield();
        self.market
            .cash_dividends
            .iter()
            .filter(|(date, _)| {
                *date > self.market.valuation_date && *date <= self.base.maturity_date
            })
            .map(|(date, amount)| {
                let t = (*date - self.market.valuation_date).num_days() as f64 / 365.0;
                // df(t) = e^{-r t}; multiplying by e^{carry t} discounts at
                // the net carry (r - carry), generalizing to any curve shape.
                amount * self.market.discount_curve.df(t) * (carry * t).exp()
            })
            .sum()
    }
    /// Escrowed-model spot: the quoted spot minus the PV of cash dividends
    /// paid over the option's life. This is the lognormal driver for the
    /// analytic and terminal-simulation engines.
    pub fn effective_spot(&self) -> f64 {
        let s = self.market.spot.value() - self.pv_cash_dividends();
        assert!(s > 0.0, "cash dividends exceed the spot price");
        s
    }
    /// Forward price of the underlying at maturity: escrowed spot grown at
    /// the carry-adjusted rate, `(S - PV(divs)) * exp((r - q - b) * T)`.
    pub fn forward_price(&self) -> f64 {
        let t = self.time_to_maturity();
        self.effective_spot() * ((self.risk_free_rate() - self.carry_yield()) * t).exp()
    }
    /// Black volatility for this option's strike and expiry, read off the
    /// surface (a flat surface returns its single vol).
    pub fn volatility(&self) -> f64 {
        self.market
            .vol_surface
            .vol(self.base.strike_price, self.forward_price(), self.time_to_maturity())
    }
    pub fn d1(&self) -> f64 {
        // Black-Scholes-Merton d1 on the escrowed spot and total carry
        let volatility = self.volatility();
        let d1_numerator = (self.effective_spot() / self.base.strike_price).ln()
            + (self.risk_free_rate() - self.carry_yield() + 0.5 * volatility.powi(2))
                * self.time_to_maturity();
        let d1_denominator = volatility * (self.time_to_maturity().sqrt());
        d1_numerator / d1_denominator
    }
    pub fn d2(&self) -> f64 {
        self.d1() - self.volatility() * self.time_to_maturity().sqrt()
    }
}
impl EquityOption {
    pub fn get_premium_at_risk(&self) -> f64 {
        let value = self.npv();
        let pay_off =
            self.payoff.payoff_amount(self.market.spot.value(), self.base.strike_price);
        if pay_off > 0.0 {
            return value - pay_off;
        } else {
            return value;
        }
    }
    
    /// Implied Black-Scholes volatility for `option_price` (safeguarded
    /// Newton with arbitrage-bound checks); does not modify the option.
    pub fn try_imp_vol(&self, option_price: f64) -> Result<f64, RustyQLibError> {
        blackscholes::implied_vol_from_price(
            self.effective_spot(),
            self.base.strike_price,
            self.risk_free_rate(),
            self.carry_yield(),
            self.time_to_maturity(),
            option_price,
            *self.payoff.put_or_call(),
        )
    }
    /// Implied vol for `option_price`; leaves the option holding a flat
    /// surface at the solved vol. Panics on arbitrage-violating prices —
    /// use [`try_imp_vol`](Self::try_imp_vol) to handle those gracefully.
    pub fn imp_vol(&mut self,option_price:f64) -> f64 {
        let vol = self.try_imp_vol(option_price).expect("implied vol solve failed");
        self.set_flat_vol(vol.max(1e-8));
        vol
    }
    pub fn get_imp_vol(&mut self) -> f64 {
        let target = self.base.current_price.mid();
        self.imp_vol(target)
    }
    fn set_flat_vol(&mut self, vol: f64) {
        self.market.vol_surface = Arc::new(
            VolSurface::flat(
                vol,
                self.market.vol_surface.reference_date(),
                self.market.vol_surface.day_count(),
            )
            .expect("vol must be positive"),
        );
    }
}


impl EquityOption {
    /// Reject engine/model/payoff combinations the library cannot price,
    /// with an error naming the engine that can.
    pub(crate) fn check_engine_support(&self) -> Result<(), RustyQLibError> {
        let unsupported = |msg: &str| Err(RustyQLibError::UnsupportedEngine(msg.to_string()));
        let bermudan = matches!(self.payoff.exercise_style(), ContractStyle::Bermudan(_));
        // American and Bermudan share the early-exercise engine rules
        let american =
            matches!(self.payoff.exercise_style(), ContractStyle::American) || bermudan;
        if self.base.is_futures_option() {
            if !matches!(self.engine, PricingEngine::BlackScholes) {
                return unsupported(
                    "Options on futures (Black-76) price on the Analytical engine only",
                );
            }
            if american {
                return unsupported("Black-76 supports European exercise only");
            }
        }
        if self.payoff.is_path_dependent() {
            if american {
                return unsupported(
                    "early-exercise (American/Bermudan) path-dependent options are not supported yet",
                );
            }
            if matches!(self.engine, PricingEngine::Binomial(_)) {
                return unsupported(
                    "Path-dependent payoffs are not supported on the Binomial engine",
                );
            }
            if matches!(self.engine, PricingEngine::FiniteDifference(_))
                && !matches!(self.payoff.payoff_kind(), PayoffType::Barrier)
            {
                return unsupported(
                    "Of the path-dependent payoffs only barriers price on the FD \
                     engine; use MonteCarlo",
                );
            }
            if matches!(self.engine, PricingEngine::BlackScholes)
                && matches!(
                    self.payoff.payoff_kind(),
                    PayoffType::Autocallable | PayoffType::Accumulator
                )
            {
                return unsupported(
                    "Autocallables and accumulators price on the MonteCarlo engine only",
                );
            }
        }
        let heston = self.model.is_heston();
        if heston && matches!(self.engine, PricingEngine::Binomial(_)) {
            return unsupported(
                "The Heston model is supported on the Analytical, MonteCarlo and \
                 FiniteDifference (2-D ADI) engines, not Binomial",
            );
        }
        if heston
            && matches!(self.engine, PricingEngine::FiniteDifference(_))
            && !matches!(self.payoff.payoff_kind(), PayoffType::Vanilla | PayoffType::Binary)
        {
            return unsupported(
                "The Heston ADI engine prices vanilla and binary payoffs; \
                 use MonteCarlo for path-dependent payoffs",
            );
        }
        match self.engine {
            PricingEngine::BlackScholes if american => unsupported(
                "Analytical engine cannot price early exercise; \
                 use Binomial, FiniteDifference or MonteCarlo",
            ),
            PricingEngine::BaroneAdesiWhaley | PricingEngine::BjerksundStensland => {
                let name = match self.engine {
                    PricingEngine::BaroneAdesiWhaley => "Barone-Adesi-Whaley",
                    _ => "Bjerksund-Stensland",
                };
                if !matches!(self.payoff.payoff_kind(), PayoffType::Vanilla) {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} approximates vanilla options only"
                    )));
                }
                if heston {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} assumes constant-vol Black-Scholes dynamics, not Heston"
                    )));
                }
                if bermudan {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} approximates American exercise only; Bermudan prices on \
                         Binomial, FiniteDifference or MonteCarlo"
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Instrument for EquityOption  {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        self.check_engine_support()?;
        let heston = self.model.is_heston();
        Ok(match self.engine {
            PricingEngine::BlackScholes if heston => heston::analytic_npv(&self),
            PricingEngine::BlackScholes => BlackScholesPricer::new().npv(&self),
            PricingEngine::MonteCarlo(_) => montecarlo::npv(&self),
            PricingEngine::Binomial(_) => binomial::npv(&self),
            PricingEngine::FiniteDifference(_) => finite_difference::npv(&self),
            PricingEngine::BaroneAdesiWhaley => baw::npv(&self),
            PricingEngine::BjerksundStensland => bjerksund_stensland::npv(&self),
        })
    }

    /// Value, all nine Greeks, and (on the Monte Carlo engine) the
    /// standard error, from one call — batched through the central
    /// sensitivity engine ([`crate::equity::greeks`]), which shares
    /// solves and reprices across the Greeks.
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        self.check_engine_support()?;
        Ok(crate::equity::greeks::pricing_result(self))
    }
}

/// Greeks route through the central sensitivity engine
/// ([`crate::equity::greeks`]): the FD and Binomial engines read
/// delta/gamma/theta off their own grid/tree with higher orders from
/// bumped solutions; the analytic engine uses the payoff-aware
/// Black-Scholes closed forms (including Black-76 futures); the bump
/// engines (Monte Carlo with common random numbers, BAW,
/// Bjerksund-Stensland, analytic Heston) share one set of
/// central-difference stencils with per-engine bump sizes.
impl EquityOption {
    pub(crate) fn analytic_heston(&self) -> bool {
        matches!(self.engine, PricingEngine::BlackScholes | PricingEngine::Binomial(_))
            && self.model.is_heston()
    }
    pub fn delta(&self) -> f64 {
        greeks::delta(self)
    }
    pub fn gamma(&self) -> f64 {
        greeks::gamma(self)
    }
    pub fn vega(&self) -> f64 {
        greeks::vega(self)
    }
    pub fn theta(&self) -> f64 {
        greeks::theta(self)
    }
    pub fn rho(&self) -> f64 {
        greeks::rho(self)
    }
    /// Vanna: change in delta per unit change in implied volatility.
    pub fn vanna(&self) -> f64 {
        greeks::vanna(self)
    }
    /// Charm: change in delta per year of calendar time.
    pub fn charm(&self) -> f64 {
        greeks::charm(self)
    }
    /// Delta elasticity (`S * gamma / delta`), also called percentage gamma.
    pub fn gamma_p(&self) -> f64 {
        greeks::gamma_p(self)
    }
    /// Zomma: change in gamma per unit change in implied volatility.
    pub fn zomma(&self) -> f64 {
        greeks::zomma(self)
    }
    /// Volga (vomma): change in vega per unit change in implied volatility.
    pub fn volga(&self) -> f64 {
        greeks::volga(self)
    }
    /// Reprice under a shifted market: spot `+ d_spot`, a parallel implied
    /// vol shift `+ d_vol`, rate `+ d_rate`, and `d_time` years of elapsed
    /// calendar time. `price_with(0, 0, 0, 0)` is the base price; the
    /// portfolio PnL attribution uses the difference of the two.
    ///
    /// Monte Carlo repricing uses common random numbers, so the difference is
    /// free of sampling noise.
    pub fn price_with(&self, d_spot: f64, d_vol: f64, d_rate: f64, d_time: f64) -> f64 {
        if self.base.is_futures_option() {
            let f = self.market.spot.value();
            let k = self.base.strike_price;
            let t = self.time_to_maturity();
            let sigma = self.market.vol_surface.vol(k, f, t);
            return crate::equity::black76::price(
                f + d_spot,
                k,
                self.risk_free_rate() + d_rate,
                sigma + d_vol,
                (t - d_time).max(1e-6),
                *self.payoff.put_or_call(),
                self.base.futures_settlement.expect("futures option must carry a settlement"),
            );
        }
        match self.engine {
            PricingEngine::MonteCarlo(_) => montecarlo::npv_with(&self, d_spot, d_vol, d_rate, d_time),
            PricingEngine::FiniteDifference(_) => {
                finite_difference::npv_with(&self, d_spot, d_vol, d_rate, d_time)
            }
            PricingEngine::BaroneAdesiWhaley => baw::price_with(&self, d_spot, d_vol, d_rate, d_time),
            PricingEngine::BjerksundStensland => {
                bjerksund_stensland::price_with(&self, d_spot, d_vol, d_rate, d_time)
            }
            // price_with shifts the maturity, so elapsed calendar time enters
            // with the opposite sign
            _ if self.analytic_heston() => {
                heston::price_with(&self, d_spot, d_vol, d_rate, -d_time)
            }
            PricingEngine::Binomial(_) => binomial::npv_with(&self, d_spot, d_vol, d_rate, d_time),
            _ => BlackScholesPricer::price_with(&self, d_spot, d_vol, d_rate, -d_time),
        }
    }
}
// #[cfg(test)]
// mod tests {
//     //write a unit test for from_json
//     use super::*;
//     use crate::core::utils::{Contract,MarketData};
//     use crate::core::trade::OptionType;
//     use crate::core::trade::Transection;
//     use crate::core::utils::ContractStyle;
//     use crate::core::termstructure::YieldTermStructure;
//     use crate::core::quotes::Quote;
//     use chrono::{Datelike, Local, NaiveDate};
//     #[test]
//     fn test_from_json() {
//         let data = Contract {
//             action: "PV".to_string(),
//             market_data: Some(MarketData {
//                 underlying_price: 100.0,
//                 strike_price: 100.0,
//                 volatility: None,
//                 option_price: Some(10.0),
//                 risk_free_rate: Some(0.05),
//                 dividend: Some(0.0),
//                 maturity: "2024-01-01".to_string(),
//                 option_type: "C".to_string(),
//                 simulation: None
//             }),
//             pricer: "Analytical".to_string(),
//             asset: "".to_string(),
//             style: Some("European".to_string()),
//             rate_data: None
//         };
//         let option = EquityOption::from_json(&data);
//         assert_eq!(option.option_type, OptionType::Call);
//         assert_eq!(option.transection, Transection::Buy);
//         assert_eq!(option.underlying_price.value, 100.0);
//         assert_eq!(option.strike_price, 100.0);
//         assert_eq!(option.current_price.value, 10.0);
//         assert_eq!(option.dividend_yield, 0.0);
//         assert_eq!(option.volatility, 0.2);
//         assert_eq!(option.maturity_date, NaiveDate::from_ymd(2024, 1, 1));
//         assert_eq!(option.valuation_date, Local::today().naive_utc());
//         assert_eq!(option.engine, Engine::BlackScholes);
//         assert_eq!(option.style, ContractStyle::European);
//     }
// }
