//! Interactive terminal wizards for the `interactive` CLI subcommand.
//!
//! Built on `inquire`: every prompt validates its input and re-asks on a
//! bad entry instead of aborting, Esc cancels the current wizard, and
//! Ctrl-C leaves the session. These read from stdin and print to stdout,
//! so they live behind the `cli` feature rather than in the pricing
//! modules.

use anyhow::{Context, Result};
use chrono::{Local, NaiveDate};
use inquire::validator::Validation;
use inquire::{CustomType, InquireError, Select};

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::blackscholes::implied_vol_from_price;
use crate::equity::montecarlo::{stats, MonteCarloConfig};
use crate::equity::utils::{LongShort, Model, PricingEngine};
use crate::equity::vanilla_option::{
    EquityMarketData, EquityOption, EquityOptionBase, VanillaPayoff,
};

/// True when the error is the user backing out (Esc / Ctrl-C) rather
/// than a real failure.
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<InquireError>(),
        Some(InquireError::OperationCanceled | InquireError::OperationInterrupted)
    )
}

/// The terms of a European vanilla option collected from the terminal.
struct OptionTerms {
    spot: f64,
    side: PutOrCall,
    strike: f64,
    vol: f64,
    rate: f64,
    dividend: f64,
    maturity: NaiveDate,
}

fn positive_f64(message: &str) -> Result<f64> {
    Ok(CustomType::<f64>::new(message)
        .with_error_message("please enter a number, e.g. 100 or 0.25")
        .with_validator(|value: &f64| {
            if *value > 0.0 {
                Ok(Validation::Valid)
            } else {
                Ok(Validation::Invalid("must be positive".into()))
            }
        })
        .prompt()?)
}

fn f64_with_default(message: &str, default: f64) -> Result<f64> {
    Ok(CustomType::<f64>::new(message)
        .with_error_message("please enter a number, e.g. 0.05")
        .with_default(default)
        .prompt()?)
}

fn prompt_side() -> Result<PutOrCall> {
    let side = Select::new("Call or put?", vec!["Call", "Put"]).prompt()?;
    Ok(match side {
        "Put" => PutOrCall::Put,
        _ => PutOrCall::Call,
    })
}

fn future_date(message: &str) -> Result<NaiveDate> {
    let today = Local::now().date_naive();
    Ok(CustomType::<NaiveDate>::new(message)
        .with_error_message("enter a date as YYYY-MM-DD, e.g. 2027-06-30")
        .with_validator(move |date: &NaiveDate| {
            if *date > today {
                Ok(Validation::Valid)
            } else {
                Ok(Validation::Invalid("maturity must be after today".into()))
            }
        })
        .prompt()?)
}

fn prompt_option_terms(with_vol: bool) -> Result<OptionTerms> {
    Ok(OptionTerms {
        spot: positive_f64("Spot price of the underlying:")?,
        side: prompt_side()?,
        strike: positive_f64("Strike price:")?,
        vol: if with_vol {
            positive_f64("Annualized volatility (e.g. 0.25 for 25%):")?
        } else {
            f64::NAN // implied-vol flow solves for it
        },
        rate: f64_with_default("Continuously compounded risk-free rate:", 0.0)?,
        dividend: f64_with_default("Continuous dividend yield:", 0.0)?,
        maturity: future_date("Maturity date (YYYY-MM-DD):")?,
    })
}

/// Assemble a European vanilla option from prompted terms with a flat
/// curve and flat vol, ready to price on `engine`.
fn build_option(terms: &OptionTerms, engine: PricingEngine) -> Result<EquityOption> {
    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        terms.rate,
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .context("invalid risk-free rate")?;
    let vol_surface = VolSurface::flat(terms.vol, valuation_date, DayCountConvention::Act365)
        .context("invalid volatility")?;
    let base = EquityOptionBase {
        symbol: "TERMINAL".to_string(),
        currency: None,
        exchange: None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: None,
        strike_price: terms.strike,
        maturity_date: terms.maturity,
        futures_settlement: None,
        multiplier: 1.0,
        current_price: Quote::new(0.0),
        entry_price: 0.0,
        long_short: LongShort::LONG,
    };
    let market = EquityMarketData {
        valuation_date,
        spot: Quote::new(terms.spot),
        dividend_yield: terms.dividend,
        borrow_cost: 0.0,
        cash_dividends: vec![],
        vol_surface: std::sync::Arc::new(vol_surface),
        discount_curve: std::sync::Arc::new(discount_curve),
    };
    let payoff = Box::new(VanillaPayoff {
        put_or_call: terms.side,
        exercise_style: ContractStyle::European,
    });
    Ok(EquityOption {
        base,
        market,
        payoff,
        engine,
        model: Model::Gbm,
    })
}

/// Prompt for the terms of a European vanilla option and price it on the
/// chosen engine, printing the price (and Greeks for Black-Scholes).
pub fn price_option_wizard() -> Result<()> {
    let engine_choice =
        Select::new("Pricing engine:", vec!["Black-Scholes", "Monte Carlo"]).prompt()?;
    let use_monte_carlo = engine_choice == "Monte Carlo";
    let terms = prompt_option_terms(true)?;
    let engine = if use_monte_carlo {
        PricingEngine::MonteCarlo(MonteCarloConfig::default())
    } else {
        PricingEngine::BlackScholes
    };
    let option = build_option(&terms, engine)?;

    if use_monte_carlo {
        let result = stats(&option, None);
        println!("Theoretical price  ${:.6} (std err {:.6})", result.pv, result.std_err);
    } else {
        println!("Theoretical price  ${:.6}", option.npv());
        println!("Premium at risk    ${:.6}", option.get_premium_at_risk());
        println!("Delta  {:.6}", option.delta());
        println!("Gamma  {:.6}", option.gamma());
        println!("Vega   {:.6}  (per 1% vol)", option.vega() * 0.01);
        println!("Theta  {:.6}  (per day)", option.theta() / 365.0);
        println!("Rho    {:.6}  (per 1% rate)", option.rho() * 0.01);
    }
    Ok(())
}

/// Prompt for a quoted option price and back out its implied
/// Black-Scholes volatility.
pub fn implied_vol_wizard() -> Result<()> {
    let terms = prompt_option_terms(false)?;
    let price = positive_f64("Observed option price:")?;
    let t = (terms.maturity - Local::now().date_naive()).num_days() as f64 / 365.0;
    let vol = implied_vol_from_price(
        terms.spot,
        terms.strike,
        terms.rate,
        terms.dividend,
        t,
        price,
        terms.side,
    )
    .context("implied vol solve failed")?;
    println!("Implied volatility  {:.6}  ({:.2}%)", vol, vol * 100.0);
    Ok(())
}
