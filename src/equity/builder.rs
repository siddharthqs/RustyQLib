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
//!     .build();
//! println!("{}", option.npv());
//! ```

use chrono::{Duration, Local, NaiveDate};

use crate::core::curves::{Compounding, YieldCurve};
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
use crate::equity::vanila_option::{
    AsianPayoff, BarrierPayoff, BinaryPayoff, BinaryType, EquityOption, EquityOptionBase,
    VanillaPayoff,
};

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
    valuation_date: NaiveDate,
    maturity_date: Option<NaiveDate>,
    exercise_style: ContractStyle,
    payoff: Option<Box<dyn Payoff>>,
    engine: Engine,
    mc: MonteCarloConfig,
    fd: FdConfig,
    heston: Option<HestonParams>,
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
            valuation_date: Local::now().date_naive(),
            maturity_date: None,
            exercise_style: ContractStyle::European,
            payoff: None,
            engine: Engine::BlackScholes,
            mc: MonteCarloConfig::default(),
            fd: FdConfig::default(),
            heston: None,
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
        self.payoff = Some(payoff);
        self
    }
    pub fn vanilla(mut self, put_or_call: PutOrCall) -> Self {
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(VanillaPayoff { put_or_call, exercise_style: style }));
        self
    }
    pub fn binary(mut self, put_or_call: PutOrCall, binary_type: BinaryType, cash: f64) -> Self {
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(BinaryPayoff {
            put_or_call,
            exercise_style: style,
            binary_type,
            cash,
        }));
        self
    }
    pub fn barrier(
        mut self,
        put_or_call: PutOrCall,
        direction: BarrierDirection,
        knock: KnockType,
        barrier: f64,
    ) -> Self {
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(BarrierPayoff {
            put_or_call,
            exercise_style: style,
            direction,
            knock,
            barrier,
        }));
        self
    }
    pub fn asian(
        mut self,
        put_or_call: PutOrCall,
        averaging: AveragingType,
        strike_type: AsianStrikeType,
    ) -> Self {
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(AsianPayoff {
            put_or_call,
            exercise_style: style,
            averaging,
            strike_type,
        }));
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
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(ForwardStartPayoff {
            put_or_call,
            exercise_style: style,
            strike_fraction,
            start_fraction,
        }));
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
        let style = self.exercise_style.clone();
        self.payoff = Some(Box::new(AutocallablePayoff {
            exercise_style: style,
            autocall_barrier,
            protection_barrier,
            coupon,
            observations,
            notional,
            initial_fixing: self.spot,
        }));
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

    pub fn build(self) -> EquityOption {
        let maturity_date = self
            .maturity_date
            .expect("set maturity_date() or years_to_maturity() before build()");
        let discount_curve = self.discount_curve.unwrap_or_else(|| {
            YieldCurve::flat(
                self.flat_rate,
                self.valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )
            .expect("invalid flat rate")
        });
        let vol_surface = self.vol_surface.unwrap_or_else(|| {
            VolSurface::flat(self.flat_vol, self.valuation_date, DayCountConvention::Act365)
                .expect("invalid flat vol")
        });
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
            vol_surface,
            maturity_date,
            valuation_date: self.valuation_date,
            discount_curve,
            entry_price: 0.0,
            long_short: LongShort::LONG,
            multiplier: 1.0,
        };
        EquityOption {
            base,
            payoff: self.payoff.expect("set a payoff (vanilla(), barrier(), ...) before build()"),
            engine: self.engine,
            mc: self.mc,
            fd: self.fd,
            heston: self.heston,
        }
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
            .build();
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
            .build();
        assert!((option.base.carry_yield() - 0.03).abs() < 1e-12);
    }

    #[test]
    fn american_flag_applies_to_the_payoff() {
        let option = EquityOptionBuilder::new()
            .spot(100.0)
            .years_to_maturity(1.0)
            .american()
            .vanilla(PutOrCall::Put)
            .build();
        assert!(matches!(option.payoff.exercise_style(), ContractStyle::American));
    }
}
