//! Ergonomic construction of [`EquityOption`] from Rust code.
//!
//! The JSON path ([`EquityOption::from_json`]) is the primary interface for
//! the CLI; this builder is the equivalent for library users and for the
//! runnable examples in `examples/`.
//!
//! ```no_run
//! use rustyqlib::equity::builder::EquityOptionBuilder;
//! use rustyqlib::equity::utils::Engine;
//! use rustyqlib::core::trade::PutOrCall;
//! use rustyqlib::Instrument;
//!
//! let option = EquityOptionBuilder::new()
//!     .spot(100.0)
//!     .strike(100.0)
//!     .flat_vol(0.30)
//!     .flat_rate(0.05)
//!     .years_to_maturity(1.0)
//!     .vanilla(PutOrCall::Call)
//!     .engine(Engine::BlackScholes)
//!     .build().expect("option must build");
//! println!("{}", option.npv());
//! ```

use chrono::{Duration, Local, NaiveDate};

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::daycount::DayCountConvention;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::asian::{AsianStrikeType, AveragingType};
use crate::equity::autocallable::AutocallablePayoff;
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::finite_difference::FdConfig;
use crate::equity::forward_start_option::ForwardStartPayoff;
use crate::equity::heston::HestonParams;
use crate::equity::montecarlo::{McModel, MonteCarloConfig};
use crate::equity::utils::{Engine, LongShort, Payoff};
use crate::equity::vanilla_option::{
    AsianPayoff, BarrierPayoff, BinaryPayoff, BinaryType, EquityOption, EquityOptionBase,
    VanillaPayoff,
};

/// What to price, recorded as data and materialized into a [`Payoff`] at
/// [`EquityOptionBuilder::build`] time — so setter order never matters:
/// `.vanilla(..).american()` and `.american().vanilla(..)` are equivalent,
/// and an autocallable's initial fixing always uses the final spot.
enum PayoffSpec {
    Vanilla {
        put_or_call: PutOrCall,
    },
    Binary {
        put_or_call: PutOrCall,
        binary_type: BinaryType,
        cash: f64,
    },
    Barrier {
        put_or_call: PutOrCall,
        direction: BarrierDirection,
        knock: KnockType,
        barrier: f64,
        barrier2: Option<f64>,
        rebate: f64,
        rebate_at_hit: bool,
    },
    Asian {
        put_or_call: PutOrCall,
        averaging: AveragingType,
        strike_type: AsianStrikeType,
    },
    Lookback {
        put_or_call: PutOrCall,
        lookback_type: crate::equity::vanilla_option::LookbackType,
    },
    ForwardStart {
        put_or_call: PutOrCall,
        strike_fraction: f64,
        start_fraction: f64,
    },
    Autocallable {
        autocall_barrier: f64,
        protection_barrier: f64,
        coupon: f64,
        observations: usize,
        notional: f64,
        coupon_barrier: Option<f64>,
        memory: bool,
    },
    /// Escape hatch: a caller-supplied payoff is used as given (its own
    /// exercise style included).
    Custom(Box<dyn Payoff>),
}

pub struct EquityOptionBuilder {
    symbol: String,
    spot: f64,
    strike: f64,
    vol_surface: Option<VolSurface>,
    flat_vol: f64,
    discount_curve: Option<YieldCurve>,
    flat_rate: f64,
    dividend_yield: f64,
    borrow_cost: f64,
    cash_dividends: Vec<(NaiveDate, f64)>,
    futures_settlement: Option<crate::equity::black76::FuturesSettlement>,
    valuation_date: NaiveDate,
    maturity_date: Option<NaiveDate>,
    exercise_style: ContractStyle,
    payoff: Option<PayoffSpec>,
    engine: Engine,
    mc: MonteCarloConfig,
    fd: FdConfig,
    heston: Option<HestonParams>,
    /// Misuse detected in a setter (e.g. `barrier_rebate` without a
    /// barrier); reported by `build()` so setters stay chainable.
    setter_error: Option<RustyQLibError>,
}

impl Default for EquityOptionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EquityOptionBuilder {
    pub fn new() -> Self {
        EquityOptionBuilder {
            symbol: "TEST".to_string(),
            spot: 100.0,
            strike: 100.0,
            vol_surface: None,
            flat_vol: 0.2,
            discount_curve: None,
            flat_rate: 0.0,
            dividend_yield: 0.0,
            borrow_cost: 0.0,
            cash_dividends: Vec::new(),
            futures_settlement: None,
            valuation_date: Local::now().date_naive(),
            maturity_date: None,
            exercise_style: ContractStyle::European,
            payoff: None,
            engine: Engine::BlackScholes,
            mc: MonteCarloConfig::default(),
            fd: FdConfig::default(),
            heston: None,
            setter_error: None,
        }
    }

    // ── Market data ─────────────────────────────────────────────────────

    pub fn symbol(mut self, symbol: &str) -> Self {
        self.symbol = symbol.to_string();
        self
    }
    pub fn spot(mut self, spot: f64) -> Self {
        self.spot = spot;
        self
    }
    pub fn strike(mut self, strike: f64) -> Self {
        self.strike = strike;
        self
    }
    pub fn flat_vol(mut self, vol: f64) -> Self {
        self.flat_vol = vol;
        self.vol_surface = None;
        self
    }
    pub fn vol_surface(mut self, surface: VolSurface) -> Self {
        self.vol_surface = Some(surface);
        self
    }
    pub fn flat_rate(mut self, rate: f64) -> Self {
        self.flat_rate = rate;
        self.discount_curve = None;
        self
    }
    pub fn discount_curve(mut self, curve: YieldCurve) -> Self {
        self.discount_curve = Some(curve);
        self
    }
    pub fn dividend_yield(mut self, q: f64) -> Self {
        self.dividend_yield = q;
        self
    }
    /// Continuous stock borrow (repo) cost; part of the carry.
    pub fn borrow_cost(mut self, b: f64) -> Self {
        self.borrow_cost = b;
        self
    }
    pub fn cash_dividend(mut self, date: NaiveDate, amount: f64) -> Self {
        self.cash_dividends.push((date, amount));
        self
    }
    /// Price the option on a future with Black-76: `spot` is then the
    /// futures price `F`. European vanilla, Analytical engine only.
    pub fn on_future(
        mut self,
        settlement: crate::equity::black76::FuturesSettlement,
    ) -> Self {
        self.futures_settlement = Some(settlement);
        self
    }

    // ── Dates ───────────────────────────────────────────────────────────

    pub fn valuation_date(mut self, date: NaiveDate) -> Self {
        self.valuation_date = date;
        self
    }
    pub fn maturity_date(mut self, date: NaiveDate) -> Self {
        self.maturity_date = Some(date);
        self
    }
    /// Convenience for examples: maturity = valuation + `years * 365` days.
    pub fn years_to_maturity(mut self, years: f64) -> Self {
        self.maturity_date =
            Some(self.valuation_date + Duration::days((years * 365.0).round() as i64));
        self
    }

    // ── Payoffs ─────────────────────────────────────────────────────────

    pub fn american(mut self) -> Self {
        self.exercise_style = ContractStyle::American;
        self
    }
    pub fn exercise_style(mut self, style: ContractStyle) -> Self {
        self.exercise_style = style;
        self
    }
    pub fn payoff(mut self, payoff: Box<dyn Payoff>) -> Self {
        self.payoff = Some(PayoffSpec::Custom(payoff));
        self
    }
    pub fn vanilla(mut self, put_or_call: PutOrCall) -> Self {
        self.payoff = Some(PayoffSpec::Vanilla { put_or_call });
        self
    }
    pub fn binary(mut self, put_or_call: PutOrCall, binary_type: BinaryType, cash: f64) -> Self {
        self.payoff = Some(PayoffSpec::Binary { put_or_call, binary_type, cash });
        self
    }
    pub fn barrier(
        mut self,
        put_or_call: PutOrCall,
        direction: BarrierDirection,
        knock: KnockType,
        barrier: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Barrier {
            put_or_call,
            direction,
            knock,
            barrier,
            barrier2: None,
            rebate: 0.0,
            rebate_at_hit: false,
        });
        self
    }
    /// Double-barrier option on the corridor between the two levels.
    pub fn double_barrier(
        mut self,
        put_or_call: PutOrCall,
        knock: crate::equity::barrier::KnockType,
        lower: f64,
        upper: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Barrier {
            put_or_call,
            direction: crate::equity::barrier::BarrierDirection::Down,
            knock,
            barrier: lower,
            barrier2: Some(upper),
            rebate: 0.0,
            rebate_at_hit: false,
        });
        self
    }

    /// Rebate on the most recently configured barrier payoff
    /// (`at_hit = true` pays the knock-out rebate at the touch;
    /// analytic engine only).
    pub fn barrier_rebate(mut self, rebate: f64, at_hit: bool) -> Self {
        match &mut self.payoff {
            Some(PayoffSpec::Barrier { rebate: r, rebate_at_hit: h, .. }) => {
                *r = rebate;
                *h = at_hit;
            }
            _ => {
                self.setter_error = Some(RustyQLibError::invalid_input(
                    "barrier_rebate",
                    "barrier_rebate must follow .barrier(...) or .double_barrier(...)",
                ));
            }
        }
        self
    }

    pub fn asian(
        mut self,
        put_or_call: PutOrCall,
        averaging: AveragingType,
        strike_type: AsianStrikeType,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Asian { put_or_call, averaging, strike_type });
        self
    }
    /// Lookback on the path extremum: floating strike pays against the
    /// min (call) / max (put); fixed strike pays the max (call) / min
    /// (put) against the built strike.
    pub fn lookback(
        mut self,
        put_or_call: PutOrCall,
        lookback_type: crate::equity::vanilla_option::LookbackType,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Lookback { put_or_call, lookback_type });
        self
    }
    /// `start_fraction` is the strike-fixing time as a fraction of the
    /// option's life, in (0, 1).
    pub fn forward_start(
        mut self,
        put_or_call: PutOrCall,
        strike_fraction: f64,
        start_fraction: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::ForwardStart {
            put_or_call,
            strike_fraction,
            start_fraction,
        });
        self
    }
    pub fn autocallable(
        mut self,
        autocall_barrier: f64,
        protection_barrier: f64,
        coupon: f64,
        observations: usize,
        notional: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Autocallable {
            autocall_barrier,
            protection_barrier,
            coupon,
            observations,
            notional,
            coupon_barrier: None,
            memory: false,
        });
        self
    }

    /// Phoenix certificate: an autocallable whose coupon is paid at every
    /// observation with `S >= coupon_barrier` (with optional memory),
    /// rather than accruing as an at-call rebate.
    #[allow(clippy::too_many_arguments)]
    pub fn phoenix(
        mut self,
        autocall_barrier: f64,
        coupon_barrier: f64,
        protection_barrier: f64,
        coupon: f64,
        observations: usize,
        notional: f64,
        memory: bool,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Autocallable {
            autocall_barrier,
            protection_barrier,
            coupon,
            observations,
            notional,
            coupon_barrier: Some(coupon_barrier),
            memory,
        });
        self
    }

    // ── Engine and model ────────────────────────────────────────────────

    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }
    pub fn model(mut self, model: McModel) -> Self {
        self.mc.model = model;
        self
    }
    pub fn heston(mut self, params: HestonParams) -> Self {
        self.heston = Some(params);
        self.mc.model = McModel::Heston;
        self
    }
    pub fn mc_config(mut self, cfg: MonteCarloConfig) -> Self {
        self.mc = cfg;
        self
    }
    pub fn paths(mut self, paths: usize) -> Self {
        self.mc.paths = paths;
        self
    }
    pub fn mc_time_steps(mut self, steps: usize) -> Self {
        self.mc.time_steps = steps;
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.mc.seed = seed;
        self
    }
    pub fn fd_config(mut self, cfg: FdConfig) -> Self {
        self.fd = cfg;
        self
    }
    pub fn fd_grid(mut self, spot_steps: usize, time_steps: usize) -> Self {
        self.fd.spot_steps = spot_steps;
        self.fd.time_steps = time_steps;
        self
    }

    /// Validate every input and construct the option.
    ///
    /// The invariant after a successful `build()` is that the option
    /// prices: field domains are checked (positive spot, positive vol,
    /// maturity after valuation, ...), payoff-specific parameters are
    /// checked, and the engine/model/payoff combination is verified, so
    /// [`Instrument::price`](crate::core::traits::Instrument::price) on
    /// the result cannot fail with `InvalidInput` or `UnsupportedEngine`.
    pub fn build(self) -> Result<EquityOption, RustyQLibError> {
        if let Some(e) = self.setter_error {
            return Err(e);
        }
        let invalid = |field: &str, reason: String| {
            Err(RustyQLibError::InvalidInput { field: field.to_string(), reason })
        };

        // ── market data domains ─────────────────────────────────────────
        if !(self.spot.is_finite() && self.spot > 0.0) {
            return invalid("spot", format!("spot must be positive and finite, got {}", self.spot));
        }
        if self.vol_surface.is_none() && !(self.flat_vol.is_finite() && self.flat_vol > 0.0) {
            return invalid(
                "flat_vol",
                format!("volatility must be positive and finite, got {}", self.flat_vol),
            );
        }
        for (name, x) in [
            ("flat_rate", self.flat_rate),
            ("dividend_yield", self.dividend_yield),
            ("borrow_cost", self.borrow_cost),
        ] {
            if !x.is_finite() {
                return invalid(name, format!("{name} must be finite, got {x}"));
            }
        }
        for (date, amount) in &self.cash_dividends {
            if !(amount.is_finite() && *amount >= 0.0) {
                return invalid(
                    "cash_dividends",
                    format!("dividend on {date} must be non-negative and finite, got {amount}"),
                );
            }
        }

        // ── dates ───────────────────────────────────────────────────────
        let maturity_date = match self.maturity_date {
            Some(d) => d,
            None => {
                return invalid(
                    "maturity_date",
                    "set maturity_date() or years_to_maturity() before build()".to_string(),
                )
            }
        };
        if maturity_date <= self.valuation_date {
            return invalid(
                "maturity_date",
                format!(
                    "maturity {maturity_date} must be after the valuation date {}",
                    self.valuation_date
                ),
            );
        }

        // ── payoff-specific domains ─────────────────────────────────────
        let spec = match self.payoff {
            Some(spec) => spec,
            None => {
                return invalid(
                    "payoff",
                    "set a payoff (vanilla(), barrier(), ...) before build()".to_string(),
                )
            }
        };
        let strike_based = matches!(
            spec,
            PayoffSpec::Vanilla { .. }
                | PayoffSpec::Binary { .. }
                | PayoffSpec::Barrier { .. }
                | PayoffSpec::Asian { .. }
                | PayoffSpec::Lookback { .. }
        );
        if strike_based && !(self.strike.is_finite() && self.strike > 0.0) {
            return invalid(
                "strike",
                format!("strike must be positive and finite, got {}", self.strike),
            );
        }
        match &spec {
            PayoffSpec::Binary { cash, .. } => {
                if !(cash.is_finite() && *cash >= 0.0) {
                    return invalid(
                        "cash",
                        format!("binary cash amount must be non-negative and finite, got {cash}"),
                    );
                }
            }
            PayoffSpec::Barrier { barrier, barrier2, rebate, .. } => {
                if !(barrier.is_finite() && *barrier > 0.0) {
                    return invalid(
                        "barrier",
                        format!("barrier level must be positive and finite, got {barrier}"),
                    );
                }
                if let Some(upper) = barrier2 {
                    if !(upper.is_finite() && upper > barrier) {
                        return invalid(
                            "double_barrier",
                            format!(
                                "the upper barrier ({upper}) must exceed the lower ({barrier})"
                            ),
                        );
                    }
                }
                if !(rebate.is_finite() && *rebate >= 0.0) {
                    return invalid(
                        "rebate",
                        format!("rebate must be non-negative and finite, got {rebate}"),
                    );
                }
            }
            PayoffSpec::ForwardStart { strike_fraction, start_fraction, .. } => {
                if !(strike_fraction.is_finite() && *strike_fraction > 0.0) {
                    return invalid(
                        "strike_fraction",
                        format!("strike_fraction must be positive and finite, got {strike_fraction}"),
                    );
                }
                if !(*start_fraction > 0.0 && *start_fraction < 1.0) {
                    return invalid(
                        "start_fraction",
                        format!("start_fraction must lie in (0, 1), got {start_fraction}"),
                    );
                }
            }
            PayoffSpec::Autocallable {
                autocall_barrier,
                protection_barrier,
                coupon,
                observations,
                notional,
                coupon_barrier,
                ..
            } => {
                for (name, x) in [
                    ("autocall_barrier", *autocall_barrier),
                    ("protection_barrier", *protection_barrier),
                    ("notional", *notional),
                ] {
                    if !(x.is_finite() && x > 0.0) {
                        return invalid(name, format!("{name} must be positive and finite, got {x}"));
                    }
                }
                if let Some(cb) = coupon_barrier {
                    if !(cb.is_finite() && *cb > 0.0) {
                        return invalid(
                            "coupon_barrier",
                            format!("coupon_barrier must be positive and finite, got {cb}"),
                        );
                    }
                }
                if !(coupon.is_finite() && *coupon >= 0.0) {
                    return invalid(
                        "coupon",
                        format!("coupon must be non-negative and finite, got {coupon}"),
                    );
                }
                if *observations < 1 {
                    return invalid("observations", "need at least one observation".to_string());
                }
            }
            _ => {}
        }
        if self.futures_settlement.is_some() {
            if !matches!(spec, PayoffSpec::Vanilla { .. }) {
                return invalid(
                    "on_future",
                    "options on futures (Black-76) support the vanilla payoff only".to_string(),
                );
            }
            if matches!(self.exercise_style, ContractStyle::American) {
                return invalid(
                    "on_future",
                    "Black-76 supports European exercise only".to_string(),
                );
            }
        }

        // ── model configuration ─────────────────────────────────────────
        if self.mc.model == McModel::Heston {
            match &self.heston {
                Some(params) => params.validate()?,
                None => {
                    return invalid(
                        "heston",
                        "heston parameters are required when the model is Heston".to_string(),
                    )
                }
            }
        }
        if self.mc.paths == 0 {
            return invalid("paths", "Monte Carlo needs at least one path".to_string());
        }
        if self.mc.time_steps == 0 {
            return invalid("mc_time_steps", "Monte Carlo needs at least one time step".to_string());
        }
        if self.fd.spot_steps < 3 || self.fd.time_steps < 1 {
            return invalid(
                "fd_grid",
                format!(
                    "the FD grid needs at least 3 spot steps and 1 time step, got {} x {}",
                    self.fd.spot_steps, self.fd.time_steps
                ),
            );
        }

        // ── market objects (curve/surface errors convert via From) ──────
        let discount_curve = match self.discount_curve {
            Some(c) => c,
            None => YieldCurve::flat(
                self.flat_rate,
                self.valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )?,
        };
        let vol_surface = match self.vol_surface {
            Some(s) => s,
            None => {
                VolSurface::flat(self.flat_vol, self.valuation_date, DayCountConvention::Act365)?
            }
        };

        // ── materialize the payoff with the final exercise style ────────
        let style = self.exercise_style.clone();
        let payoff: Box<dyn Payoff> = match spec {
            PayoffSpec::Vanilla { put_or_call } => {
                Box::new(VanillaPayoff { put_or_call, exercise_style: style })
            }
            PayoffSpec::Binary { put_or_call, binary_type, cash } => Box::new(BinaryPayoff {
                put_or_call,
                exercise_style: style,
                binary_type,
                cash,
            }),
            PayoffSpec::Barrier {
                put_or_call,
                direction,
                knock,
                barrier,
                barrier2,
                rebate,
                rebate_at_hit,
            } => Box::new(BarrierPayoff {
                put_or_call,
                exercise_style: style,
                direction,
                knock,
                barrier,
                barrier2,
                rebate,
                rebate_at_hit,
            }),
            PayoffSpec::Asian { put_or_call, averaging, strike_type } => Box::new(AsianPayoff {
                put_or_call,
                exercise_style: style,
                averaging,
                strike_type,
            }),
            PayoffSpec::Lookback { put_or_call, lookback_type } => {
                Box::new(crate::equity::vanilla_option::LookbackPayoff {
                    put_or_call,
                    exercise_style: style,
                    lookback_type,
                })
            }
            PayoffSpec::ForwardStart { put_or_call, strike_fraction, start_fraction } => {
                Box::new(ForwardStartPayoff {
                    put_or_call,
                    exercise_style: style,
                    strike_fraction,
                    start_fraction,
                })
            }
            PayoffSpec::Autocallable {
                autocall_barrier,
                protection_barrier,
                coupon,
                observations,
                notional,
                coupon_barrier,
                memory,
            } => Box::new(AutocallablePayoff {
                exercise_style: style,
                autocall_barrier,
                protection_barrier,
                coupon,
                observations,
                notional,
                initial_fixing: self.spot,
                coupon_barrier,
                memory,
            }),
            PayoffSpec::Custom(p) => p,
        };
        let base = EquityOptionBase {
            symbol: self.symbol,
            currency: None,
            exchange: None,
            name: None,
            cusip: None,
            isin: None,
            settlement_type: None,
            underlying_price: Quote::new(self.spot),
            current_price: Quote::new(0.0),
            strike_price: self.strike,
            dividend_yield: self.dividend_yield,
            borrow_cost: self.borrow_cost,
            cash_dividends: self.cash_dividends,
            futures_settlement: self.futures_settlement,
            vol_surface,
            maturity_date,
            valuation_date: self.valuation_date,
            discount_curve,
            entry_price: 0.0,
            long_short: LongShort::LONG,
            multiplier: 1.0,
        };
        let option = EquityOption {
            base,
            payoff,
            engine: self.engine,
            mc: self.mc,
            fd: self.fd,
            heston: self.heston,
        };
        // "builds => prices": refuse engine/model/payoff combinations here
        // rather than at pricing time
        option.check_engine_support()?;
        Ok(option)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::Instrument;

    #[test]
    fn builder_reproduces_black_scholes_golden() {
        let option = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build().expect("option must build");
        assert!((option.npv() - 14.2312547860).abs() < 1e-8);
        assert!((option.delta() - 0.6242517279).abs() < 1e-8);
    }

    #[test]
    fn builder_carries_dividends_and_borrow() {
        let option = EquityOptionBuilder::new()
            .spot(100.0)
            .dividend_yield(0.01)
            .borrow_cost(0.02)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Call)
            .build().expect("option must build");
        assert!((option.base.carry_yield() - 0.03).abs() < 1e-12);
    }

    #[test]
    fn american_flag_applies_to_the_payoff_in_either_order() {
        // the payoff is materialized at build() time, so the exercise
        // style applies regardless of setter order
        for build_order_reversed in [false, true] {
            let b = EquityOptionBuilder::new()
                .spot(100.0)
                .years_to_maturity(1.0)
                .engine(Engine::Binomial);
            let b = if build_order_reversed {
                b.vanilla(PutOrCall::Put).american()
            } else {
                b.american().vanilla(PutOrCall::Put)
            };
            let option = b.build().expect("option must build");
            assert!(matches!(option.payoff.exercise_style(), ContractStyle::American));
        }
    }

    #[test]
    fn build_rejects_bad_inputs_with_the_offending_field() {
        use crate::core::errors::RustyQLibError;
        let field = |r: Result<EquityOption, RustyQLibError>| match r {
            Err(RustyQLibError::InvalidInput { field, .. }) => field,
            other => panic!("expected InvalidInput, got {:?}", other.map(|_| "an option")),
        };

        let base = || EquityOptionBuilder::new().years_to_maturity(1.0).vanilla(PutOrCall::Call);

        assert_eq!(field(base().spot(-1.0).build()), "spot");
        assert_eq!(field(base().flat_vol(0.0).build()), "flat_vol");
        assert_eq!(field(base().strike(f64::NAN).build()), "strike");
        assert_eq!(
            field(EquityOptionBuilder::new().vanilla(PutOrCall::Call).build()),
            "maturity_date"
        );
        assert_eq!(field(base().years_to_maturity(-1.0).build()), "maturity_date");
        assert_eq!(
            field(EquityOptionBuilder::new().years_to_maturity(1.0).build()),
            "payoff"
        );
        assert_eq!(field(base().barrier_rebate(5.0, false).build()), "barrier_rebate");
        assert_eq!(
            field(base().model(McModel::Heston).engine(Engine::MonteCarlo).build()),
            "heston"
        );
        assert_eq!(
            field(
                base()
                    .forward_start(PutOrCall::Call, 1.0, 1.5)
                    .engine(Engine::MonteCarlo)
                    .build()
            ),
            "start_fraction"
        );
        assert_eq!(
            field(
                base()
                    .double_barrier(PutOrCall::Call, KnockType::Out, 120.0, 80.0)
                    .engine(Engine::MonteCarlo)
                    .build()
            ),
            "double_barrier"
        );
    }

    #[test]
    fn build_rejects_unsupported_engine_combinations() {
        use crate::core::errors::RustyQLibError;
        // an option that builds must price: engine support is checked here
        let result = EquityOptionBuilder::new()
            .spot(100.0)
            .years_to_maturity(1.0)
            .vanilla(PutOrCall::Call)
            .american()
            .engine(Engine::BlackScholes)
            .build();
        assert!(
            matches!(result, Err(RustyQLibError::UnsupportedEngine(_))),
            "American exercise on the analytic engine must be refused at build()"
        );
    }

    #[test]
    fn autocallable_initial_fixing_uses_the_final_spot() {
        // spot() after autocallable() must still set the initial fixing
        let option = EquityOptionBuilder::new()
            .years_to_maturity(1.0)
            .autocallable(1.0, 0.7, 0.05, 4, 100.0)
            .spot(250.0)
            .engine(Engine::MonteCarlo)
            .build().expect("option must build");
        let payoff = option
            .payoff
            .as_any()
            .downcast_ref::<AutocallablePayoff>()
            .expect("autocallable payoff");
        assert_eq!(payoff.initial_fixing, 250.0);
    }
}
