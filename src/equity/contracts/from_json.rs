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
            None => builder.flat_rate(data.base.risk_free_rate.ok_or_else(|| {
                RustyQLibError::invalid_input(
                    "risk_free_rate",
                    "either risk_free_rate or discount_curve must be provided",
                )
            })?),
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
        // enum strings are matched case-insensitively, like the payoff
        // sub-type strings below
        let exercise_style = data.exercise_style.as_deref().unwrap_or("European").trim();
        builder = match exercise_style.to_lowercase().as_str() {
            "american" | "european" => {
                // a date list on a non-Bermudan contract is a mistake in
                // the document, not something to ignore
                if data.exercise_dates.is_some() {
                    return Err(RustyQLibError::invalid_input(
                        "exercise_dates",
                        format!(
                            "exercise_dates is only meaningful when exercise_style is \
                             Bermudan (got '{exercise_style}')"
                        ),
                    ));
                }
                if exercise_style.eq_ignore_ascii_case("american") {
                    builder.american()
                } else {
                    builder
                }
            }
            "bermudan" => {
                let dates = data.exercise_dates.as_deref().ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "exercise_dates",
                        "exercise_dates is required when exercise_style is Bermudan",
                    )
                })?;
                builder.bermudan(parse_date_list("exercise_dates", dates)?)
            }
            _ => {
                return Err(RustyQLibError::invalid_input(
                    "exercise_style",
                    format!(
                        "unknown exercise_style '{exercise_style}' (use 'European', \
                         'American' or 'Bermudan')"
                    ),
                ))
            }
        };

        let put_or_call = data.put_or_call.trim();
        let side = match put_or_call.to_lowercase().as_str() {
            "c" | "call" => PutOrCall::Call,
            "p" | "put" => PutOrCall::Put,
            _ => {
                return Err(RustyQLibError::invalid_input(
                    "put_or_call",
                    format!("invalid side '{put_or_call}' (use 'C' or 'P')"),
                ))
            }
        };

        // ── payoff ──────────────────────────────────────────────────────
        builder = match payoff_type {
            PayoffType::Accumulator => {
                return Err(RustyQLibError::invalid_input(
                    "payoff_type",
                    "accumulators are built through EquityOptionBuilder::accumulator, \
                     not JSON contract data",
                ));
            }
            PayoffType::Cliquet => {
                return Err(RustyQLibError::invalid_input(
                    "payoff_type",
                    "cliquets are built through EquityOptionBuilder::cliquet, or as the \
                     standalone 'cliquet_option' product contract",
                ));
            }
            PayoffType::VarianceSwap => {
                return Err(RustyQLibError::invalid_input(
                    "payoff_type",
                    "variance swaps are built through EquityOptionBuilder::variance_swap, \
                     or as the standalone 'variance_swap' product contract",
                ));
            }
            PayoffType::Rainbow => {
                return Err(RustyQLibError::invalid_input(
                    "payoff_type",
                    "rainbow payoffs are multi-asset: build through \
                     MultiAssetEquityOption::builder, or as the 'rainbow_option' \
                     product contract",
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
                let choice_date = parse_date("choice_date", date_str)?;
                let choice_fraction = life_fraction(
                    "choice_date",
                    choice_date,
                    valuation_date,
                    maturity_date,
                    valuation_date,
                    false,
                    "choice_date must lie between valuation and maturity",
                )?;
                let complex = data.chooser_call_strike.is_some()
                    || data.chooser_put_strike.is_some()
                    || data.chooser_call_expiry.is_some()
                    || data.chooser_put_expiry.is_some();
                if complex {
                    // absent leg expiries default to the maturity (1.0)
                    let leg_fraction = |field: &str, date: &Option<String>| match date {
                        None => Ok(1.0),
                        Some(sd) => life_fraction(
                            field,
                            parse_date(field, sd)?,
                            valuation_date,
                            maturity_date,
                            choice_date,
                            true,
                            "leg expiry must lie after the choice date and at or before \
                             maturity",
                        ),
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
                let start_fraction = life_fraction(
                    "forward_start_date",
                    parse_date("forward_start_date", start_date_str)?,
                    valuation_date,
                    maturity_date,
                    valuation_date,
                    false,
                    "forward_start_date must lie between valuation and maturity",
                )?;
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
                // an explicit 0 reaches the builder, which rejects it
                let observations = data.autocall_observations.unwrap_or(4);
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
        let pricer = data.pricer.as_deref().unwrap_or("Analytical").trim();
        let engine_kind = match pricer.to_lowercase().as_str() {
            "analytical" | "bs" => Engine::BlackScholes,
            "montecarlo" | "mc" => Engine::MonteCarlo,
            "binomial" | "bino" => Engine::Binomial,
            // "finitdifference" is a historical misspelling that existing
            // documents rely on
            "finitedifference" | "finitdifference" | "fd" => Engine::FiniteDifference,
            "baroneadesiwhaley" | "baw" => Engine::BaroneAdesiWhaley,
            "bjerksundstensland" | "bjerksund_stensland" | "bs2002" => Engine::BjerksundStensland,
            _ => {
                return Err(RustyQLibError::invalid_input(
                    "pricer",
                    format!(
                        "unknown pricer '{pricer}' (use Analytical, MonteCarlo, Binomial, \
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
        builder = builder.engine(engine_kind).model(Model::from_contract(
            data.mc_model.as_deref(),
            data.heston,
            data.rbergomi,
            data.sabr,
        )?);

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

/// Parse one `YYYY-MM-DD` contract date, naming `field` in the error.
fn parse_date(field: &str, s: &str) -> Result<NaiveDate, RustyQLibError> {
    NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| {
        RustyQLibError::invalid_input(field, format!("invalid date '{s}' (expected YYYY-MM-DD)"))
    })
}

/// Parse a JSON date-string list into `NaiveDate`s. Ordering and range
/// validation happens in [`EquityOptionBuilder::build`]; this only
/// handles the string format, naming `field` in errors.
fn parse_date_list(field: &str, dates: &[String]) -> Result<Vec<NaiveDate>, RustyQLibError> {
    dates.iter().map(|s| parse_date(field, s)).collect()
}

/// Calendar-day fraction of the option life at `date`:
/// `(date - valuation) / (maturity - valuation)`, the time argument the
/// chooser, chooser-leg and forward-start payoffs take. The date must
/// lie strictly after `lower` (the valuation date, or the choice date
/// for a chooser leg) and strictly before maturity — at maturity too
/// when `upper_inclusive` — else `field` is rejected with `out_of_range`
/// as the reason.
#[allow(clippy::too_many_arguments)]
fn life_fraction(
    field: &str,
    date: NaiveDate,
    valuation: NaiveDate,
    maturity: NaiveDate,
    lower: NaiveDate,
    upper_inclusive: bool,
    out_of_range: &str,
) -> Result<f64, RustyQLibError> {
    let below_upper = if upper_inclusive {
        date <= maturity
    } else {
        date < maturity
    };
    if !(date > lower && below_upper) {
        return Err(RustyQLibError::invalid_input(field, out_of_range));
    }
    Ok((date - valuation).num_days() as f64 / (maturity - valuation).num_days() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::ContractStyle;

    fn data(json: &str) -> EquityOptionData {
        serde_json::from_str(json).expect("contract must parse")
    }

    const VANILLA: &str = r#"{
        "symbol": "ACME", "underlying_price": 100.0, "put_or_call": "C",
        "payoff_type": "vanilla", "strike_price": 100.0, "volatility": 0.25,
        "risk_free_rate": 0.03, "valuation_date": "2026-01-05",
        "maturity": "2027-01-04", "pricer": "Analytical"
    }"#;

    fn with(field: &str, value: &str) -> EquityOptionData {
        let mut json = VANILLA.trim_end_matches(['}', ' ', '\n']).to_string();
        json.push_str(&format!(", \"{field}\": {value} }}"));
        data(&json)
    }

    fn set_pricer(name: &str) -> EquityOptionData {
        data(&VANILLA.replace(
            "\"pricer\": \"Analytical\"",
            &format!("\"pricer\": \"{name}\""),
        ))
    }

    #[test]
    fn engine_names_parse_case_insensitively_with_every_legacy_spelling() {
        let cases = [
            ("Analytical", Engine::BlackScholes),
            ("analytical", Engine::BlackScholes),
            ("ANALYTICAL", Engine::BlackScholes),
            ("bs", Engine::BlackScholes),
            ("MonteCarlo", Engine::MonteCarlo),
            ("MONTECARLO", Engine::MonteCarlo),
            ("MC", Engine::MonteCarlo),
            ("mc", Engine::MonteCarlo),
            ("Binomial", Engine::Binomial),
            ("bino", Engine::Binomial),
            ("FiniteDifference", Engine::FiniteDifference),
            ("finitedifference", Engine::FiniteDifference),
            ("finitdifference", Engine::FiniteDifference),
            ("FD", Engine::FiniteDifference),
            ("fd", Engine::FiniteDifference),
            (" fd ", Engine::FiniteDifference),
        ];
        for (name, expected) in cases {
            let option = EquityOption::try_from_json(&set_pricer(name))
                .unwrap_or_else(|e| panic!("pricer '{name}' must parse: {e}"));
            assert_eq!(option.engine.kind(), expected, "pricer '{name}'");
        }
        // the American approximations, on an American vanilla
        let american = VANILLA.replace(
            "\"pricer\": \"Analytical\"",
            "\"exercise_style\": \"American\", \"pricer\": \"PRICER\"",
        );
        for (name, expected) in [
            ("BaroneAdesiWhaley", Engine::BaroneAdesiWhaley),
            ("baw", Engine::BaroneAdesiWhaley),
            ("BAW", Engine::BaroneAdesiWhaley),
            ("BjerksundStensland", Engine::BjerksundStensland),
            ("bjerksund_stensland", Engine::BjerksundStensland),
            ("bs2002", Engine::BjerksundStensland),
            ("BS2002", Engine::BjerksundStensland),
        ] {
            let option = EquityOption::try_from_json(&data(&american.replace("PRICER", name)))
                .unwrap_or_else(|e| panic!("pricer '{name}' must parse: {e}"));
            assert_eq!(option.engine.kind(), expected, "pricer '{name}'");
        }
        // unknown names still name the field and echo the input spelling
        let err = EquityOption::try_from_json(&set_pricer("NoSuchEngine"))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("pricer") && err.contains("NoSuchEngine"),
            "{err}"
        );
    }

    #[test]
    fn side_and_exercise_style_parse_case_insensitively() {
        for side in ["C", "c", "Call", "call", "CALL"] {
            let option = EquityOption::try_from_json(&data(&VANILLA.replace(
                "\"put_or_call\": \"C\"",
                &format!("\"put_or_call\": \"{side}\""),
            )))
            .unwrap_or_else(|e| panic!("side '{side}' must parse: {e}"));
            assert_eq!(
                *option.payoff.put_or_call(),
                PutOrCall::Call,
                "side '{side}'"
            );
        }
        for side in ["P", "p", "Put", "put", "PUT"] {
            let option = EquityOption::try_from_json(&data(&VANILLA.replace(
                "\"put_or_call\": \"C\"",
                &format!("\"put_or_call\": \"{side}\""),
            )))
            .unwrap_or_else(|e| panic!("side '{side}' must parse: {e}"));
            assert_eq!(
                *option.payoff.put_or_call(),
                PutOrCall::Put,
                "side '{side}'"
            );
        }
        let err = EquityOption::try_from_json(&data(
            &VANILLA.replace("\"put_or_call\": \"C\"", "\"put_or_call\": \"X\""),
        ))
        .unwrap_err()
        .to_string();
        assert!(err.contains("put_or_call"), "{err}");

        let styled = |style: &str| {
            let mut d = set_pricer("Binomial");
            d.exercise_style = Some(style.to_string());
            EquityOption::try_from_json(&d)
        };
        for style in ["American", "american", "AMERICAN", " American "] {
            let option = styled(style).unwrap_or_else(|e| panic!("style '{style}': {e}"));
            assert_eq!(*option.payoff.exercise_style(), ContractStyle::American);
        }
        for style in ["European", "EUROPEAN"] {
            let option = styled(style).unwrap_or_else(|e| panic!("style '{style}': {e}"));
            assert_eq!(*option.payoff.exercise_style(), ContractStyle::European);
        }
        let err = styled("Asian").unwrap_err().to_string();
        assert!(err.contains("exercise_style"), "{err}");
    }

    #[test]
    fn exercise_dates_outside_bermudan_style_are_rejected() {
        let mut d = set_pricer("Binomial");
        d.exercise_dates = Some(vec!["2026-07-06".to_string()]);
        // default (European) and explicit American both refuse the list
        let err = EquityOption::try_from_json(&d).unwrap_err().to_string();
        assert!(err.contains("exercise_dates"), "{err}");
        d.exercise_style = Some("American".to_string());
        let err = EquityOption::try_from_json(&d).unwrap_err().to_string();
        assert!(err.contains("exercise_dates"), "{err}");
        // Bermudan consumes it (case-insensitively)
        d.exercise_style = Some("BERMUDAN".to_string());
        let option = EquityOption::try_from_json(&d).expect("Bermudan must build");
        assert!(matches!(
            option.payoff.exercise_style(),
            ContractStyle::Bermudan(_)
        ));
    }

    #[test]
    fn zero_autocall_observations_reach_the_builder_rejection() {
        let autocall = r#"{
            "symbol": "ACME", "underlying_price": 100.0, "put_or_call": "C",
            "payoff_type": "autocallable", "autocall_barrier": 100.0,
            "protection_barrier": 70.0, "autocall_coupon": 0.05,
            "autocall_observations": 0, "volatility": 0.25,
            "risk_free_rate": 0.03, "valuation_date": "2026-01-05",
            "maturity": "2027-01-04", "pricer": "MC", "simulation": 1000
        }"#;
        let err = EquityOption::try_from_json(&data(autocall))
            .unwrap_err()
            .to_string();
        assert!(err.contains("observations"), "{err}");
        // and the default still applies when the field is absent
        let defaulted = autocall.replace("\"autocall_observations\": 0,", "");
        assert!(EquityOption::try_from_json(&data(&defaulted)).is_ok());
    }

    #[test]
    fn life_fraction_preserves_each_sites_range_rule() {
        let d = |y, m, day| NaiveDate::from_ymd_opt(y, m, day).unwrap();
        let (valuation, maturity) = (d(2026, 1, 1), d(2027, 1, 1));
        // strict on both ends (chooser choice date, forward-start fixing)
        let mid = life_fraction(
            "f",
            d(2026, 7, 2),
            valuation,
            maturity,
            valuation,
            false,
            "bad",
        )
        .unwrap();
        assert!((mid - 182.0 / 365.0).abs() < 1e-15, "{mid}");
        for date in [valuation, maturity, d(2025, 12, 31), d(2027, 1, 2)] {
            let err = life_fraction("f", date, valuation, maturity, valuation, false, "bad")
                .unwrap_err()
                .to_string();
            assert!(err.contains("f") && err.contains("bad"), "{err}");
        }
        // chooser legs: after the choice date, maturity itself allowed
        let choice = d(2026, 7, 1);
        assert_eq!(
            life_fraction("leg", maturity, valuation, maturity, choice, true, "bad").unwrap(),
            1.0
        );
        assert!(life_fraction("leg", choice, valuation, maturity, choice, true, "bad").is_err());
        assert!(life_fraction(
            "leg",
            d(2026, 6, 1),
            valuation,
            maturity,
            choice,
            true,
            "bad"
        )
        .is_err());
        assert!(life_fraction(
            "leg",
            d(2026, 7, 2),
            valuation,
            maturity,
            choice,
            true,
            "bad"
        )
        .is_ok());
        // the JSON sites report their own wording
        let mut chooser = with("choice_date", "\"2027-06-01\"");
        chooser.payoff_type = "chooser".to_string();
        let err = EquityOption::try_from_json(&chooser)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("choice_date must lie between valuation and maturity"),
            "{err}"
        );
        let mut fs = with("forward_start_date", "\"2027-01-04\"");
        fs.payoff_type = "forward_start".to_string();
        fs.pricer = Some("MC".to_string());
        let err = EquityOption::try_from_json(&fs).unwrap_err().to_string();
        assert!(
            err.contains("forward_start_date must lie between valuation and maturity"),
            "{err}"
        );
        let mut leg = with("choice_date", "\"2026-07-05\"");
        leg.payoff_type = "chooser".to_string();
        leg.chooser_put_expiry = Some("2026-07-05".to_string());
        let err = EquityOption::try_from_json(&leg).unwrap_err().to_string();
        assert!(
            err.contains("chooser_put_expiry") && err.contains("leg expiry must lie after"),
            "{err}"
        );
        leg.chooser_put_expiry = Some("2027-01-04".to_string());
        assert!(EquityOption::try_from_json(&leg).is_ok());
    }
}
