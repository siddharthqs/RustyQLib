//! JSON → [`EquityOption`] translation layer: parses JSON-level fields
//! (dates, enum strings) into typed values and feeds them through
//! [`EquityOptionBuilder`], which owns all domain validation and assembly.

use super::equity_option::EquityOption;
use crate::core::curves::YieldCurve;
use crate::core::data_models::EquityOptionData;
use crate::core::errors::RustyQLibError;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::vols::VolSurface;
use crate::equity::asian::{AsianStrikeType, AveragingType};
use crate::equity::barrier::{BarrierDirection, KnockType};
use crate::equity::binary_option::BinaryType;
use crate::equity::builder::EquityOptionBuilder;
use crate::equity::lookback::LookbackType;
use crate::equity::utils::{Engine, Model, PayoffType};
use crate::equity::{finite_difference, montecarlo};
use chrono::NaiveDate;

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
        let maturity_date =
            NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    "maturity",
                    format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
                )
            })?;
        let payoff_type = data.payoff_type.parse::<PayoffType>().map_err(|_| {
            RustyQLibError::invalid_input(
                "payoff_type",
                format!("unknown payoff_type '{}'", data.payoff_type),
            )
        })?;
        let strike_price = match payoff_type {
            // strike is set by the contract mechanics for these payoffs
            PayoffType::ForwardStart | PayoffType::Autocallable => data.strike_price.unwrap_or(0.0),
            _ => data.strike_price.ok_or_else(|| {
                RustyQLibError::invalid_input(
                    "strike_price",
                    "strike_price is required for this payoff",
                )
            })?,
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
            let date = NaiveDate::parse_from_str(&d.date, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    "cash_dividends",
                    format!("invalid dividend date '{}' (expected YYYY-MM-DD)", d.date),
                )
            })?;
            builder = builder.cash_dividend(date, d.amount);
        }
        if let Some(s) = data.futures_settlement.as_deref() {
            let settlement = s
                .parse::<crate::equity::black76::FuturesSettlement>()
                .map_err(|_| {
                    RustyQLibError::invalid_input(
                        "futures_settlement",
                        format!(
                            "invalid futures_settlement '{s}' (use 'discounted' or 'margined')"
                        ),
                    )
                })?;
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
            other => {
                return Err(RustyQLibError::invalid_input(
                    "put_or_call",
                    format!("invalid side '{other}' (use 'C' or 'P')"),
                ))
            }
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
                    other => {
                        return Err(RustyQLibError::invalid_input(
                            "binary_type",
                            format!("invalid binary_type '{other}' (use 'cash' or 'asset')"),
                        ))
                    }
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
                    other => {
                        return Err(RustyQLibError::invalid_input(
                            "lookback_type",
                            format!("invalid lookback_type '{other}' (use 'floating' or 'fixed')"),
                        ))
                    }
                };
                builder.lookback(side, lookback_type)
            }
            PayoffType::Chooser => {
                let date_str = data.choice_date.as_ref().ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "choice_date",
                        "choice_date is required for chooser options",
                    )
                })?;
                let choice_date =
                    NaiveDate::parse_from_str(date_str, "%Y-%m-%d").map_err(|_| {
                        RustyQLibError::invalid_input(
                            "choice_date",
                            format!("invalid date '{date_str}' (expected YYYY-MM-DD)"),
                        )
                    })?;
                if !(choice_date > valuation_date && choice_date < maturity_date) {
                    return Err(RustyQLibError::invalid_input(
                        "choice_date",
                        "choice_date must lie between valuation and maturity",
                    ));
                }
                let choice_fraction = (choice_date - valuation_date).num_days() as f64
                    / (maturity_date - valuation_date).num_days() as f64;
                let complex = data.chooser_call_strike.is_some()
                    || data.chooser_put_strike.is_some()
                    || data.chooser_call_expiry.is_some()
                    || data.chooser_put_expiry.is_some();
                if complex {
                    // absent leg expiries default to the maturity (1.0)
                    let leg_fraction = |field: &str, date: &Option<String>| match date {
                        None => Ok(1.0),
                        Some(sd) => {
                            let leg = NaiveDate::parse_from_str(sd, "%Y-%m-%d").map_err(|_| {
                                RustyQLibError::invalid_input(
                                    field,
                                    format!("invalid date '{sd}' (expected YYYY-MM-DD)"),
                                )
                            })?;
                            if !(leg > choice_date && leg <= maturity_date) {
                                return Err(RustyQLibError::invalid_input(
                                    field,
                                    "leg expiry must lie after the choice date and at or \
                                     before maturity",
                                ));
                            }
                            Ok((leg - valuation_date).num_days() as f64
                                / (maturity_date - valuation_date).num_days() as f64)
                        }
                    };
                    builder.complex_chooser(
                        choice_fraction,
                        data.chooser_call_strike.unwrap_or(strike_price),
                        leg_fraction("chooser_call_expiry", &data.chooser_call_expiry)?,
                        data.chooser_put_strike.unwrap_or(strike_price),
                        leg_fraction("chooser_put_expiry", &data.chooser_put_expiry)?,
                    )
                } else {
                    builder.chooser(choice_fraction)
                }
            }
            PayoffType::Barrier => {
                let barrier = data.barrier_level.ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "barrier_level",
                        "barrier_level is required for barrier options",
                    )
                })?;
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
                    other => {
                        return Err(RustyQLibError::invalid_input(
                            "barrier_type",
                            format!(
                                "barrier_type must be up_in/up_out/down_in/down_out, got '{other}'"
                            ),
                        ))
                    }
                };
                // a second level makes it a double barrier: the corridor
                // between the two levels (direction is then ignored)
                let b = match data.barrier_level2 {
                    Some(b2) => {
                        builder.double_barrier(side, knock, barrier.min(b2), barrier.max(b2))
                    }
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
                    other => {
                        return Err(RustyQLibError::invalid_input(
                            "averaging_type",
                            format!(
                                "averaging_type must be arithmetic or geometric, got '{other}'"
                            ),
                        ))
                    }
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
                    other => {
                        return Err(RustyQLibError::invalid_input(
                            "asian_strike_type",
                            format!("asian_strike_type must be fixed or floating, got '{other}'"),
                        ))
                    }
                };
                builder.asian(side, averaging, strike_type)
            }
            PayoffType::ForwardStart => {
                let start_date_str = data.forward_start_date.as_ref().ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "forward_start_date",
                        "forward_start_date is required for forward-start options",
                    )
                })?;
                let start_date =
                    NaiveDate::parse_from_str(start_date_str, "%Y-%m-%d").map_err(|_| {
                        RustyQLibError::invalid_input(
                            "forward_start_date",
                            format!("invalid date '{start_date_str}' (expected YYYY-MM-DD)"),
                        )
                    })?;
                if !(start_date > valuation_date && start_date < maturity_date) {
                    return Err(RustyQLibError::invalid_input(
                        "forward_start_date",
                        "forward_start_date must lie between valuation and maturity",
                    ));
                }
                let start_fraction = (start_date - valuation_date).num_days() as f64
                    / (maturity_date - valuation_date).num_days() as f64;
                builder.forward_start(side, data.strike_fraction.unwrap_or(1.0), start_fraction)
            }
            PayoffType::Autocallable => {
                let autocall_barrier = data.autocall_barrier.ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "autocall_barrier",
                        "autocall_barrier is required for autocallables",
                    )
                })?;
                let protection_barrier = data.protection_barrier.ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "protection_barrier",
                        "protection_barrier is required for autocallables",
                    )
                })?;
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
            Engine::MonteCarlo => builder.mc_config(montecarlo::MonteCarloConfig::from_data(data)?),
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
