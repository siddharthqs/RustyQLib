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
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::accumulator::{AccumulatorPayoff, AccumulatorSide};
use crate::equity::asian::{AsianStrikeType, AveragingType};
use crate::equity::autocallable::AutocallablePayoff;
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::finite_difference::FdConfig;
use crate::equity::forward_start_option::ForwardStartPayoff;
use crate::equity::heston::HestonParams;
use crate::equity::montecarlo::MonteCarloConfig;
use crate::equity::utils::Model;
use crate::equity::utils::PricingEngine;
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
    Chooser {
        choice_fraction: f64,
        legs: Option<crate::equity::chooser::ChooserLegs>,
    },
    Autocallable {
        autocall_barrier: f64,
        protection_barrier: f64,
        coupon: f64,
        observations: usize,
        notional: f64,
        coupon_barrier: Option<f64>,
        memory: bool,
        observation_dates: Option<Vec<NaiveDate>>,
    },
    Accumulator {
        side: AccumulatorSide,
        barrier: f64,
        observations: usize,
        shares_per_day: f64,
        gearing: f64,
    },
    /// Escape hatch: a caller-supplied payoff is used as given (its own
    /// exercise style included).
    Custom(Box<dyn Payoff>),
}

/// The builder inputs a payoff needs beyond its own parameters, shared by
/// [`PayoffSpec::validate`] and [`PayoffSpec::materialize`].
struct BuildContext {
    spot: f64,
    strike: f64,
    valuation_date: NaiveDate,
    maturity_date: NaiveDate,
}

/// Check that a date list is non-empty, strictly increasing after the
/// valuation date and not past maturity; convert to Act/365 year fractions.
fn date_list_to_times(
    field: &str,
    dates: &[NaiveDate],
    valuation_date: NaiveDate,
    maturity_date: NaiveDate,
) -> Result<Vec<f64>, RustyQLibError> {
    if dates.is_empty() {
        return Err(RustyQLibError::invalid_input(
            field,
            "the date list must not be empty",
        ));
    }
    let mut prev = valuation_date;
    let mut times = Vec::with_capacity(dates.len());
    for date in dates {
        if *date <= prev {
            return Err(RustyQLibError::invalid_input(
                field,
                format!("dates must be strictly increasing after valuation; {date} is not"),
            ));
        }
        if *date > maturity_date {
            return Err(RustyQLibError::invalid_input(
                field,
                format!("date {date} lies after the maturity {maturity_date}"),
            ));
        }
        prev = *date;
        times.push((*date - valuation_date).num_days() as f64 / 365.0);
    }
    Ok(times)
}

impl PayoffSpec {
    /// Payoffs whose value reads the built strike; the rest set their
    /// strike through the contract mechanics.
    fn requires_strike(&self) -> bool {
        matches!(
            self,
            PayoffSpec::Vanilla { .. }
                | PayoffSpec::Binary { .. }
                | PayoffSpec::Barrier { .. }
                | PayoffSpec::Asian { .. }
                | PayoffSpec::Lookback { .. }
        )
    }

    /// Domain checks for this payoff's own parameters — one-dimensional
    /// checks only; engine/model/payoff combination rules stay centralized
    /// in `EquityOption::check_engine_support`.
    fn validate(&self, ctx: &BuildContext) -> Result<(), RustyQLibError> {
        let invalid = |field: &str, reason: String| {
            Err(RustyQLibError::InvalidInput {
                field: field.to_string(),
                reason,
            })
        };
        if self.requires_strike() && !(ctx.strike.is_finite() && ctx.strike > 0.0) {
            return invalid(
                "strike",
                format!("strike must be positive and finite, got {}", ctx.strike),
            );
        }
        match self {
            PayoffSpec::Vanilla { .. }
            | PayoffSpec::Asian { .. }
            | PayoffSpec::Lookback { .. }
            | PayoffSpec::Custom(_) => Ok(()),
            PayoffSpec::Binary { cash, .. } => {
                if !(cash.is_finite() && *cash >= 0.0) {
                    return invalid(
                        "cash",
                        format!("binary cash amount must be non-negative and finite, got {cash}"),
                    );
                }
                Ok(())
            }
            PayoffSpec::Barrier {
                barrier,
                barrier2,
                rebate,
                ..
            } => {
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
                Ok(())
            }
            PayoffSpec::ForwardStart {
                strike_fraction,
                start_fraction,
                ..
            } => {
                if !(strike_fraction.is_finite() && *strike_fraction > 0.0) {
                    return invalid(
                        "strike_fraction",
                        format!(
                            "strike_fraction must be positive and finite, got {strike_fraction}"
                        ),
                    );
                }
                if !(*start_fraction > 0.0 && *start_fraction < 1.0) {
                    return invalid(
                        "start_fraction",
                        format!("start_fraction must lie in (0, 1), got {start_fraction}"),
                    );
                }
                Ok(())
            }
            PayoffSpec::Chooser {
                choice_fraction,
                legs,
            } => {
                if !(*choice_fraction > 0.0 && *choice_fraction < 1.0) {
                    return invalid(
                        "choice_fraction",
                        format!("choice_fraction must lie in (0, 1), got {choice_fraction}"),
                    );
                }
                if let Some(legs) = legs {
                    for (name, strike) in [
                        ("call_strike", legs.call_strike),
                        ("put_strike", legs.put_strike),
                    ] {
                        if !(strike.is_finite() && strike > 0.0) {
                            return invalid(
                                name,
                                format!("{name} must be positive and finite, got {strike}"),
                            );
                        }
                    }
                    for (name, fraction) in [
                        ("call_expiry_fraction", legs.call_expiry_fraction),
                        ("put_expiry_fraction", legs.put_expiry_fraction),
                    ] {
                        if !(fraction > *choice_fraction && fraction <= 1.0) {
                            return invalid(
                                name,
                                format!(
                                    "{name} must lie in (choice_fraction, 1], got {fraction}                                      with choice_fraction {choice_fraction}"
                                ),
                            );
                        }
                    }
                }
                Ok(())
            }
            PayoffSpec::Autocallable {
                autocall_barrier,
                protection_barrier,
                coupon,
                observations,
                notional,
                coupon_barrier,
                observation_dates,
                ..
            } => {
                for (name, x) in [
                    ("autocall_barrier", *autocall_barrier),
                    ("protection_barrier", *protection_barrier),
                    ("notional", *notional),
                ] {
                    if !(x.is_finite() && x > 0.0) {
                        return invalid(
                            name,
                            format!("{name} must be positive and finite, got {x}"),
                        );
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
                if let Some(dates) = observation_dates {
                    date_list_to_times(
                        "autocall_observation_dates",
                        dates,
                        ctx.valuation_date,
                        ctx.maturity_date,
                    )?;
                }
                Ok(())
            }
            PayoffSpec::Accumulator {
                side,
                barrier,
                observations,
                shares_per_day,
                gearing,
            } => {
                if !(barrier.is_finite() && *barrier > 0.0) {
                    return invalid(
                        "barrier",
                        format!("barrier must be positive and finite, got {barrier}"),
                    );
                }
                match side {
                    AccumulatorSide::Accumulator if *barrier <= ctx.spot => {
                        return invalid(
                            "barrier",
                            "accumulator knock-out must be above the spot".to_string(),
                        );
                    }
                    AccumulatorSide::Decumulator if *barrier >= ctx.spot => {
                        return invalid(
                            "barrier",
                            "decumulator knock-out must be below the spot".to_string(),
                        );
                    }
                    _ => {}
                }
                if *observations < 1 {
                    return invalid("observations", "need at least one observation".to_string());
                }
                if !(shares_per_day.is_finite() && *shares_per_day > 0.0) {
                    return invalid(
                        "shares_per_day",
                        format!("shares_per_day must be positive and finite, got {shares_per_day}"),
                    );
                }
                if !(gearing.is_finite() && *gearing >= 0.0) {
                    return invalid(
                        "gearing",
                        format!("gearing must be non-negative and finite, got {gearing}"),
                    );
                }
                Ok(())
            }
        }
    }

    /// Construct the runtime payoff with the final exercise style; assumes
    /// [`validate`](Self::validate) has passed.
    fn materialize(self, style: ContractStyle, ctx: &BuildContext) -> Box<dyn Payoff> {
        match self {
            PayoffSpec::Vanilla { put_or_call } => Box::new(VanillaPayoff {
                put_or_call,
                exercise_style: style,
            }),
            PayoffSpec::Binary {
                put_or_call,
                binary_type,
                cash,
            } => Box::new(BinaryPayoff {
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
            PayoffSpec::Asian {
                put_or_call,
                averaging,
                strike_type,
            } => Box::new(AsianPayoff {
                put_or_call,
                exercise_style: style,
                averaging,
                strike_type,
            }),
            PayoffSpec::Lookback {
                put_or_call,
                lookback_type,
            } => Box::new(crate::equity::vanilla_option::LookbackPayoff {
                put_or_call,
                exercise_style: style,
                lookback_type,
            }),
            PayoffSpec::ForwardStart {
                put_or_call,
                strike_fraction,
                start_fraction,
            } => Box::new(ForwardStartPayoff {
                put_or_call,
                exercise_style: style,
                strike_fraction,
                start_fraction,
            }),
            PayoffSpec::Chooser {
                choice_fraction,
                legs,
            } => Box::new(crate::equity::chooser::ChooserPayoff {
                exercise_style: style,
                choice_fraction,
                legs,
            }),
            PayoffSpec::Autocallable {
                autocall_barrier,
                protection_barrier,
                coupon,
                observations,
                notional,
                coupon_barrier,
                memory,
                observation_dates,
            } => {
                let observation_times = observation_dates.as_ref().map(|dates| {
                    dates
                        .iter()
                        .map(|d| (*d - ctx.valuation_date).num_days() as f64 / 365.0)
                        .collect::<Vec<f64>>()
                });
                let observations = observation_dates
                    .as_ref()
                    .map_or(observations, |dates| dates.len());
                Box::new(AutocallablePayoff {
                    exercise_style: style,
                    autocall_barrier,
                    protection_barrier,
                    coupon,
                    observations,
                    notional,
                    initial_fixing: ctx.spot,
                    coupon_barrier,
                    memory,
                    observation_times,
                })
            }
            PayoffSpec::Accumulator {
                side,
                barrier,
                observations,
                shares_per_day,
                gearing,
            } => Box::new(AccumulatorPayoff {
                exercise_style: style,
                side,
                barrier,
                observations,
                shares_per_day,
                gearing,
            }),
            PayoffSpec::Custom(p) => p,
        }
    }
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
    lattice: crate::core::lattice::LatticeConfig,
    model: Model,
    /// Autocall schedule request: (months between observations, calendar),
    /// resolved into dates at `build()` from valuation and maturity.
    autocall_schedule: Option<(u32, crate::core::calendar::Calendar)>,
    /// Bermudan exercise dates; converted to year fractions at `build()`.
    bermudan_dates: Option<Vec<NaiveDate>>,
    /// Bermudan schedule request, resolved like `autocall_schedule`.
    bermudan_schedule: Option<(u32, crate::core::calendar::Calendar)>,
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
            lattice: crate::core::lattice::LatticeConfig::default(),
            model: Model::Gbm,
            autocall_schedule: None,
            bermudan_dates: None,
            bermudan_schedule: None,
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
    pub fn on_future(mut self, settlement: crate::equity::black76::FuturesSettlement) -> Self {
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
    /// Bermudan exercise on the given dates (expiry is always exercisable
    /// through the terminal payoff). Overrides `american()` /
    /// `exercise_style()`; applies to built-in payoffs, not `payoff()`.
    pub fn bermudan(mut self, dates: Vec<NaiveDate>) -> Self {
        self.bermudan_dates = Some(dates);
        self
    }
    /// Bermudan exercise every `months` months on business-day adjusted
    /// dates (modified following) from valuation to maturity, generated at
    /// `build()` time.
    pub fn bermudan_schedule(
        mut self,
        months: u32,
        calendar: crate::core::calendar::Calendar,
    ) -> Self {
        self.bermudan_schedule = Some((months, calendar));
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
        self.payoff = Some(PayoffSpec::Binary {
            put_or_call,
            binary_type,
            cash,
        });
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
            Some(PayoffSpec::Barrier {
                rebate: r,
                rebate_at_hit: h,
                ..
            }) => {
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
        self.payoff = Some(PayoffSpec::Asian {
            put_or_call,
            averaging,
            strike_type,
        });
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
        self.payoff = Some(PayoffSpec::Lookback {
            put_or_call,
            lookback_type,
        });
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
    /// Simple chooser: the holder picks call or put (same strike and
    /// expiry) at the choice date, given as a fraction of the option's
    /// life in (0, 1). Prices on the Analytical engine.
    pub fn chooser(mut self, choice_fraction: f64) -> Self {
        self.payoff = Some(PayoffSpec::Chooser {
            choice_fraction,
            legs: None,
        });
        self
    }
    /// Complex chooser: the holder picks at the choice date between a
    /// call and a put with their own strikes and expiries. Expiries are
    /// fractions of the option's life (1.0 = the maturity date), strictly
    /// after the choice fraction. Prices on the Analytical engine.
    pub fn complex_chooser(
        mut self,
        choice_fraction: f64,
        call_strike: f64,
        call_expiry_fraction: f64,
        put_strike: f64,
        put_expiry_fraction: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Chooser {
            choice_fraction,
            legs: Some(crate::equity::chooser::ChooserLegs {
                call_strike,
                put_strike,
                call_expiry_fraction,
                put_expiry_fraction,
            }),
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
            observation_dates: None,
        });
        self
    }

    /// Explicit autocall observation dates (e.g. from a
    /// [`Schedule`](crate::core::calendar::Schedule)); must follow
    /// `.autocallable(...)` or `.phoenix(...)`. Overrides the equally
    /// spaced observation count.
    pub fn autocall_observation_dates(mut self, dates: Vec<NaiveDate>) -> Self {
        match &mut self.payoff {
            Some(PayoffSpec::Autocallable {
                observation_dates, ..
            }) => {
                *observation_dates = Some(dates);
            }
            _ => {
                self.setter_error = Some(RustyQLibError::invalid_input(
                    "autocall_observation_dates",
                    "autocall_observation_dates must follow .autocallable(...) or .phoenix(...)",
                ));
            }
        }
        self
    }

    /// Generate business-day adjusted autocall observation dates every
    /// `months` months from valuation to maturity on the given calendar
    /// (modified following); must follow `.autocallable(...)` or
    /// `.phoenix(...)`. The schedule is built at `build()` time from the
    /// final valuation and maturity dates.
    pub fn autocall_schedule(
        mut self,
        months: u32,
        calendar: crate::core::calendar::Calendar,
    ) -> Self {
        if !matches!(self.payoff, Some(PayoffSpec::Autocallable { .. })) {
            self.setter_error = Some(RustyQLibError::invalid_input(
                "autocall_schedule",
                "autocall_schedule must follow .autocallable(...) or .phoenix(...)",
            ));
            return self;
        }
        self.autocall_schedule = Some((months, calendar));
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
            observation_dates: None,
        });
        self
    }

    /// Accumulator: the holder buys `shares_per_day` at the strike (set
    /// via `.strike(...)`, below spot) on every equally spaced
    /// observation day, knocked out when the spot reaches `barrier`
    /// (above spot), with `gearing`x the quantity on days the spot closes
    /// below the strike. Prices on the MonteCarlo engine.
    pub fn accumulator(
        mut self,
        barrier: f64,
        observations: usize,
        shares_per_day: f64,
        gearing: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Accumulator {
            side: AccumulatorSide::Accumulator,
            barrier,
            observations,
            shares_per_day,
            gearing,
        });
        self
    }

    /// Decumulator: the mirror of [`accumulator`](Self::accumulator) —
    /// sell at the strike (above spot), knocked out at `barrier` (below
    /// spot), geared on days the spot closes above the strike.
    pub fn decumulator(
        mut self,
        barrier: f64,
        observations: usize,
        shares_per_day: f64,
        gearing: f64,
    ) -> Self {
        self.payoff = Some(PayoffSpec::Accumulator {
            side: AccumulatorSide::Decumulator,
            barrier,
            observations,
            shares_per_day,
            gearing,
        });
        self
    }

    // ── Engine and model ────────────────────────────────────────────────

    pub fn engine(mut self, engine: Engine) -> Self {
        self.engine = engine;
        self
    }
    pub fn model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }
    pub fn heston(mut self, params: HestonParams) -> Self {
        self.model = Model::Heston(params);
        self
    }
    pub fn rbergomi(mut self, params: crate::equity::rbergomi::RBergomiParams) -> Self {
        self.model = Model::RBergomi(params);
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
    pub fn lattice_config(mut self, cfg: crate::core::lattice::LatticeConfig) -> Self {
        self.lattice = cfg;
        self
    }
    /// Binomial tree parameterization (default Leisen-Reimer).
    pub fn tree_type(mut self, tree_type: crate::core::lattice::BinomialTreeType) -> Self {
        self.lattice.tree_type = tree_type;
        self
    }
    /// Binomial tree steps (default 1000).
    pub fn tree_steps(mut self, steps: usize) -> Self {
        self.lattice.steps = steps;
        self
    }
    /// Price the binomial tree with term structures of rates and
    /// volatility applied per step (`tree_type` is then ignored).
    pub fn tree_term_structure(mut self) -> Self {
        self.lattice.term_structure = true;
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
    ///
    /// Only the configuration of the *selected* engine is validated: an
    /// out-of-domain Monte Carlo or grid setting is ignored when that
    /// engine is not the one pricing the option.
    pub fn build(mut self) -> Result<EquityOption, RustyQLibError> {
        if let Some(e) = self.setter_error.take() {
            return Err(e);
        }
        let invalid = |field: &str, reason: String| {
            Err(RustyQLibError::InvalidInput {
                field: field.to_string(),
                reason,
            })
        };

        // ── market data domains (shared by every payoff) ────────────────
        self.validate_market_data()?;

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

        // ── payoff spec: resolve schedules, then check its own domains ──
        let mut spec = match self.payoff.take() {
            Some(spec) => spec,
            None => {
                return invalid(
                    "payoff",
                    "set a payoff (vanilla(), barrier(), ...) before build()".to_string(),
                )
            }
        };
        if let Some((months, calendar)) = &self.autocall_schedule {
            match &mut spec {
                PayoffSpec::Autocallable {
                    observation_dates, ..
                } => {
                    let schedule = crate::core::calendar::Schedule::generate(
                        self.valuation_date,
                        maturity_date,
                        *months,
                        calendar,
                        crate::core::calendar::BusinessDayConvention::ModifiedFollowing,
                        crate::core::calendar::DateGeneration::Backward,
                    )?;
                    *observation_dates = Some(schedule.dates);
                }
                _ => {
                    return invalid(
                        "autocall_schedule",
                        "autocall_schedule set but the payoff is not an autocallable".to_string(),
                    )
                }
            }
        }
        let ctx = BuildContext {
            spot: self.spot,
            strike: self.strike,
            valuation_date: self.valuation_date,
            maturity_date,
        };
        spec.validate(&ctx)?;

        // ── exercise style (Bermudan dates → year fractions) ────────────
        let mut bermudan_dates = self.bermudan_dates.take();
        if let Some((months, calendar)) = &self.bermudan_schedule {
            let schedule = crate::core::calendar::Schedule::generate(
                self.valuation_date,
                maturity_date,
                *months,
                calendar,
                crate::core::calendar::BusinessDayConvention::ModifiedFollowing,
                crate::core::calendar::DateGeneration::Backward,
            )?;
            bermudan_dates = Some(schedule.dates);
        }
        let style = match &bermudan_dates {
            Some(dates) => {
                if matches!(spec, PayoffSpec::Custom(_)) {
                    return invalid(
                        "bermudan",
                        "bermudan dates apply to built-in payoffs; embed the exercise style \
                         in the custom payoff instead"
                            .to_string(),
                    );
                }
                ContractStyle::Bermudan(date_list_to_times(
                    "bermudan",
                    dates,
                    self.valuation_date,
                    maturity_date,
                )?)
            }
            None => self.exercise_style.clone(),
        };
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
        match &self.model {
            Model::Heston(params) => params.validate()?,
            Model::RBergomi(params) => params.validate()?,
            Model::Gbm | Model::LocalVol => {}
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
            None => VolSurface::flat(
                self.flat_vol,
                self.valuation_date,
                DayCountConvention::Act365,
            )?,
        };

        // ── materialize the payoff with the final exercise style ────────
        let payoff: Box<dyn Payoff> = spec.materialize(style, &ctx);
        let base = EquityOptionBase {
            symbol: self.symbol,
            currency: None,
            exchange: None,
            name: None,
            cusip: None,
            isin: None,
            settlement_type: None,
            strike_price: self.strike,
            maturity_date,
            futures_settlement: self.futures_settlement,
            multiplier: 1.0,
            current_price: Quote::new(0.0),
            entry_price: 0.0,
            long_short: LongShort::LONG,
        };
        let market = crate::equity::vanilla_option::EquityMarketData {
            valuation_date: self.valuation_date,
            spot: Quote::new(self.spot),
            dividend_yield: self.dividend_yield,
            borrow_cost: self.borrow_cost,
            cash_dividends: self.cash_dividends,
            vol_surface: std::sync::Arc::new(vol_surface),
            discount_curve: std::sync::Arc::new(discount_curve),
        };
        // only the selected engine's configuration is validated: the
        // others never influence the built option
        let engine = match self.engine {
            Engine::BlackScholes => PricingEngine::BlackScholes,
            Engine::MonteCarlo => {
                self.mc.validate()?;
                PricingEngine::MonteCarlo(self.mc)
            }
            Engine::Binomial => {
                self.lattice.validate()?;
                PricingEngine::Binomial(self.lattice)
            }
            Engine::FiniteDifference => {
                self.fd.validate()?;
                PricingEngine::FiniteDifference(self.fd)
            }
            Engine::BaroneAdesiWhaley => PricingEngine::BaroneAdesiWhaley,
            Engine::BjerksundStensland => PricingEngine::BjerksundStensland,
        };
        let option = EquityOption {
            base,
            market,
            payoff,
            engine,
            model: self.model,
        };
        // "builds => prices": refuse engine/model/payoff combinations here
        // rather than at pricing time
        option.check_engine_support()?;
        Ok(option)
    }

    /// Domain checks on the market data fields shared by every payoff.
    fn validate_market_data(&self) -> Result<(), RustyQLibError> {
        let invalid = |field: &str, reason: String| {
            Err(RustyQLibError::InvalidInput {
                field: field.to_string(),
                reason,
            })
        };
        if !(self.spot.is_finite() && self.spot > 0.0) {
            return invalid(
                "spot",
                format!("spot must be positive and finite, got {}", self.spot),
            );
        }
        if self.vol_surface.is_none() && !(self.flat_vol.is_finite() && self.flat_vol > 0.0) {
            return invalid(
                "flat_vol",
                format!(
                    "volatility must be positive and finite, got {}",
                    self.flat_vol
                ),
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
        Ok(())
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
            .build()
            .expect("option must build");
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
            .build()
            .expect("option must build");
        assert!((option.carry_yield() - 0.03).abs() < 1e-12);
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
            assert!(matches!(
                option.payoff.exercise_style(),
                ContractStyle::American
            ));
        }
    }

    #[test]
    fn build_rejects_bad_inputs_with_the_offending_field() {
        use crate::core::errors::RustyQLibError;
        let field = |r: Result<EquityOption, RustyQLibError>| match r {
            Err(RustyQLibError::InvalidInput { field, .. }) => field,
            other => panic!(
                "expected InvalidInput, got {:?}",
                other.map(|_| "an option")
            ),
        };

        let base = || {
            EquityOptionBuilder::new()
                .years_to_maturity(1.0)
                .vanilla(PutOrCall::Call)
        };

        assert_eq!(field(base().spot(-1.0).build()), "spot");
        assert_eq!(field(base().flat_vol(0.0).build()), "flat_vol");
        assert_eq!(field(base().strike(f64::NAN).build()), "strike");
        assert_eq!(
            field(EquityOptionBuilder::new().vanilla(PutOrCall::Call).build()),
            "maturity_date"
        );
        assert_eq!(
            field(base().years_to_maturity(-1.0).build()),
            "maturity_date"
        );
        assert_eq!(
            field(EquityOptionBuilder::new().years_to_maturity(1.0).build()),
            "payoff"
        );
        assert_eq!(
            field(base().barrier_rebate(5.0, false).build()),
            "barrier_rebate"
        );
        // Heston params travel inside the model, so "params missing" is
        // unrepresentable; invalid params are still rejected at build()
        let bad_heston = crate::equity::heston::HestonParams {
            v0: -0.1,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        };
        assert_eq!(
            field(base().heston(bad_heston).engine(Engine::MonteCarlo).build()),
            "heston params"
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

        // at-hit rebates need the touch time, which the Monte Carlo path
        // grid does not carry; formerly a pricing-time panic
        let result = EquityOptionBuilder::new()
            .spot(100.0)
            .years_to_maturity(1.0)
            .barrier(PutOrCall::Call, BarrierDirection::Up, KnockType::Out, 120.0)
            .barrier_rebate(5.0, true)
            .engine(Engine::MonteCarlo)
            .build();
        assert!(
            matches!(result, Err(RustyQLibError::UnsupportedEngine(_))),
            "at-hit rebate on Monte Carlo must be refused at build()"
        );
    }

    #[test]
    fn autocall_schedule_generates_business_day_observations() {
        use crate::core::calendar::Calendar;
        let option = EquityOptionBuilder::new()
            .spot(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .autocallable(105.0, 70.0, 0.02, 4, 100.0)
            .autocall_schedule(3, Calendar::UsNyse)
            .engine(Engine::MonteCarlo)
            .paths(20_000)
            .build()
            .expect("option must build");
        let auto = option
            .payoff
            .as_any()
            .downcast_ref::<AutocallablePayoff>()
            .expect("autocallable payoff");
        let times = auto
            .observation_times
            .as_ref()
            .expect("schedule must set times");
        assert_eq!(auto.observations, times.len());
        assert!(times.windows(2).all(|w| w[0] < w[1]), "times must increase");
        // the same schedule regenerated must be all NYSE business days
        let schedule = crate::core::calendar::Schedule::generate(
            NaiveDate::from_ymd_opt(2026, 1, 5).unwrap(),
            NaiveDate::from_ymd_opt(2027, 1, 4).unwrap(),
            3,
            &Calendar::UsNyse,
            crate::core::calendar::BusinessDayConvention::ModifiedFollowing,
            crate::core::calendar::DateGeneration::Backward,
        )
        .unwrap();
        for d in &schedule.dates {
            assert!(
                Calendar::UsNyse.is_business_day(*d),
                "{d} not a business day"
            );
        }
        // calendar-adjusted observations price close to the equally
        // spaced approximation (same seed, same paths)
        let baseline = EquityOptionBuilder::new()
            .spot(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .autocallable(105.0, 70.0, 0.02, 4, 100.0)
            .engine(Engine::MonteCarlo)
            .paths(20_000)
            .build()
            .expect("option must build");
        let a = option.npv();
        let b = baseline.npv();
        assert!(a.is_finite() && a > 0.0);
        assert!(
            (a - b).abs() < 1.0,
            "dates ~quarterly: {a} vs equally spaced {b}"
        );
    }

    #[test]
    fn autocall_observation_dates_are_validated() {
        use crate::core::errors::RustyQLibError;
        let field = |r: Result<EquityOption, RustyQLibError>| match r {
            Err(RustyQLibError::InvalidInput { field, .. }) => field,
            other => panic!(
                "expected InvalidInput, got {:?}",
                other.map(|_| "an option")
            ),
        };
        let base = || {
            EquityOptionBuilder::new()
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
                .autocallable(105.0, 70.0, 0.02, 4, 100.0)
                .engine(Engine::MonteCarlo)
        };
        // unsorted dates
        let unsorted = vec![
            NaiveDate::from_ymd_opt(2026, 7, 6).unwrap(),
            NaiveDate::from_ymd_opt(2026, 4, 6).unwrap(),
        ];
        assert_eq!(
            field(base().autocall_observation_dates(unsorted).build()),
            "autocall_observation_dates"
        );
        // date past maturity
        let late = vec![NaiveDate::from_ymd_opt(2027, 6, 1).unwrap()];
        assert_eq!(
            field(base().autocall_observation_dates(late).build()),
            "autocall_observation_dates"
        );
        // schedule on a non-autocallable payoff
        assert_eq!(
            field(
                EquityOptionBuilder::new()
                    .years_to_maturity(1.0)
                    .vanilla(PutOrCall::Call)
                    .autocall_observation_dates(vec![NaiveDate::from_ymd_opt(2026, 9, 1).unwrap()])
                    .build()
            ),
            "autocall_observation_dates"
        );
    }

    fn bermudan_put(dates: Vec<NaiveDate>, engine: Engine) -> EquityOption {
        EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .bermudan(dates)
            .vanilla(PutOrCall::Put)
            .engine(engine)
            .build()
            .expect("option must build")
    }

    fn put_with_style(
        style: fn(EquityOptionBuilder) -> EquityOptionBuilder,
        engine: Engine,
    ) -> EquityOption {
        let b = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap());
        style(b)
            .vanilla(PutOrCall::Put)
            .engine(engine)
            .build()
            .expect("option must build")
    }

    fn quarterly_dates() -> Vec<NaiveDate> {
        vec![
            NaiveDate::from_ymd_opt(2026, 4, 6).unwrap(),
            NaiveDate::from_ymd_opt(2026, 7, 6).unwrap(),
            NaiveDate::from_ymd_opt(2026, 10, 5).unwrap(),
        ]
    }

    #[test]
    fn bermudan_with_no_interior_dates_is_european() {
        // a single exercise date at expiry adds nothing beyond the
        // terminal payoff: the tree must reproduce the European value
        let euro = put_with_style(|b| b, Engine::Binomial).npv();
        let berm = bermudan_put(
            vec![NaiveDate::from_ymd_opt(2027, 1, 4).unwrap()],
            Engine::Binomial,
        )
        .npv();
        assert!((berm - euro).abs() < 1e-10, "berm {berm} vs euro {euro}");
    }

    #[test]
    fn bermudan_value_sits_between_european_and_american() {
        let euro = put_with_style(|b| b, Engine::Binomial).npv();
        let amer = put_with_style(|b| b.american(), Engine::Binomial).npv();
        let quarterly = bermudan_put(quarterly_dates(), Engine::Binomial).npv();
        // monthly rights dominate quarterly rights
        let monthly: Vec<NaiveDate> = (1..12)
            .map(|m| NaiveDate::from_ymd_opt(2026, 1, 5).unwrap() + chrono::Months::new(m))
            .collect();
        let monthly_pv = bermudan_put(monthly, Engine::Binomial).npv();
        let eps = 1e-9;
        assert!(euro <= quarterly + eps, "euro {euro} quarterly {quarterly}");
        assert!(
            quarterly <= monthly_pv + eps,
            "quarterly {quarterly} monthly {monthly_pv}"
        );
        assert!(
            monthly_pv <= amer + eps,
            "monthly {monthly_pv} american {amer}"
        );
        // quarterly rights must be worth something on an ITM-prone put
        assert!(quarterly > euro + 1e-4, "quarterly rights must add value");
    }

    #[test]
    fn dense_bermudan_converges_to_american() {
        let amer = put_with_style(|b| b.american(), Engine::Binomial).npv();
        // weekly exercise rights
        let weekly: Vec<NaiveDate> = (1..52)
            .map(|w| NaiveDate::from_ymd_opt(2026, 1, 5).unwrap() + chrono::Duration::weeks(w))
            .collect();
        let dense = bermudan_put(weekly, Engine::Binomial).npv();
        assert!(
            (amer - dense).abs() < 0.05,
            "weekly Bermudan {dense} must approach American {amer}"
        );
    }

    #[test]
    fn bermudan_prices_agree_across_engines() {
        let tree = bermudan_put(quarterly_dates(), Engine::Binomial).npv();
        let fd = bermudan_put(quarterly_dates(), Engine::FiniteDifference).npv();
        assert!((tree - fd).abs() < 0.05, "binomial {tree} vs FD {fd}");
        let mc = bermudan_put(quarterly_dates(), Engine::MonteCarlo)
            .price()
            .unwrap();
        let se = mc.std_err.expect("MC std err");
        assert!(
            (mc.pv - tree).abs() < (3.0 * se).max(0.15),
            "MC {} +- {se} vs tree {tree}",
            mc.pv
        );
    }

    #[test]
    fn bermudan_schedule_and_validation() {
        use crate::core::calendar::Calendar;
        use crate::core::errors::RustyQLibError;
        // schedule-generated quarterly rights price like explicit ones
        let scheduled = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .bermudan_schedule(3, Calendar::UsNyse)
            .vanilla(PutOrCall::Put)
            .engine(Engine::Binomial)
            .build()
            .expect("option must build");
        let explicit = bermudan_put(quarterly_dates(), Engine::Binomial);
        assert!((scheduled.npv() - explicit.npv()).abs() < 0.05);

        // analytic American approximations refuse Bermudan at build()
        let r = bermudan_put_result(quarterly_dates(), Engine::BaroneAdesiWhaley);
        assert!(matches!(r, Err(RustyQLibError::UnsupportedEngine(_))));
        // a date after maturity is rejected with the offending field
        let r = bermudan_put_result(
            vec![NaiveDate::from_ymd_opt(2028, 1, 1).unwrap()],
            Engine::Binomial,
        );
        assert!(
            matches!(r, Err(RustyQLibError::InvalidInput { field, .. }) if field == "bermudan")
        );
    }

    fn bermudan_put_result(
        dates: Vec<NaiveDate>,
        engine: Engine,
    ) -> Result<EquityOption, crate::core::errors::RustyQLibError> {
        EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .bermudan(dates)
            .vanilla(PutOrCall::Put)
            .engine(engine)
            .build()
    }

    #[test]
    fn tree_type_selects_the_lattice_scheme() {
        use crate::core::lattice::BinomialTreeType;
        use crate::equity::blackscholes::bs_price;
        let build = |tree: BinomialTreeType, steps: usize| {
            EquityOptionBuilder::new()
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.3)
                .flat_rate(0.05)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 5).unwrap())
                .vanilla(PutOrCall::Call)
                .engine(Engine::Binomial)
                .tree_type(tree)
                .tree_steps(steps)
                .build()
                .expect("option must build")
        };
        let reference = bs_price(100.0, 100.0, 0.05, 0.0, 0.3, 1.0, PutOrCall::Call);
        // Leisen-Reimer at 101 steps beats CRR at 101 steps by an order
        // of magnitude on the same contract
        let lr_err = (build(BinomialTreeType::LeisenReimer, 101).npv() - reference).abs();
        let crr_err = (build(BinomialTreeType::CoxRossRubinstein, 101).npv() - reference).abs();
        assert!(
            lr_err * 10.0 < crr_err,
            "LR err {lr_err} vs CRR err {crr_err}"
        );
        // diagnostics agree with the fast engine
        let option = build(BinomialTreeType::LeisenReimer, 101);
        let diag = crate::equity::binomial::npv_with_diagnostics(&option);
        assert_eq!(diag.price, option.npv());
        assert_eq!(diag.steps, 101);
    }

    #[test]
    fn term_structure_tree_prices_rate_timing_into_early_exercise() {
        use crate::core::curves::{Compounding, CurveInput, InterpolationMethod, Tenor};
        // steep upward curve: 1% short rate, 9% long rate
        let curve_input = CurveInput::ZeroRates {
            tenors: vec![Tenor::YearFraction(0.25), Tenor::YearFraction(1.0)],
            rates: vec![0.01, 0.09],
            compounding: Compounding::Continuous,
            day_count: DayCountConvention::Act365,
            interpolation: InterpolationMethod::LinearZero,
        };
        let build = |term: bool, american: bool| {
            let curve =
                YieldCurve::from_input(&curve_input, NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
                    .unwrap();
            let mut b = EquityOptionBuilder::new()
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.3)
                .discount_curve(curve)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 5).unwrap());
            if american {
                b = b.american();
            }
            let mut b = b
                .vanilla(PutOrCall::Put)
                .engine(Engine::Binomial)
                .tree_steps(801);
            if term {
                b = b.tree_term_structure();
            }
            b.build().expect("option must build")
        };
        // European: only df(T) matters, so term and uniform trees agree
        let (euro_term, euro_uniform) = (build(true, false).npv(), build(false, false).npv());
        assert!(
            (euro_term - euro_uniform).abs() < 0.05,
            "European must agree: term {euro_term} uniform {euro_uniform}"
        );
        // American put: early rates are 1%, not the 8.8%-ish zero rate to
        // maturity — cheap short-dated carry makes waiting cheaper, so
        // rate timing must move the early-exercise value visibly
        let (amer_term, amer_uniform) = (build(true, true).npv(), build(false, true).npv());
        assert!(
            (amer_term - amer_uniform).abs() > 0.02,
            "rate timing must matter for early exercise: term {amer_term} uniform {amer_uniform}"
        );
        assert!(amer_term >= euro_term - 1e-9);
    }

    #[test]
    fn autocallable_initial_fixing_uses_the_final_spot() {
        // spot() after autocallable() must still set the initial fixing
        let option = EquityOptionBuilder::new()
            .years_to_maturity(1.0)
            .autocallable(1.0, 0.7, 0.05, 4, 100.0)
            .spot(250.0)
            .engine(Engine::MonteCarlo)
            .build()
            .expect("option must build");
        let payoff = option
            .payoff
            .as_any()
            .downcast_ref::<AutocallablePayoff>()
            .expect("autocallable payoff");
        assert_eq!(payoff.initial_fixing, 250.0);
    }
}
