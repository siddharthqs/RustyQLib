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
use inquire::{CustomType, InquireError, Select, Text};

use crate::bonds::schedule::{coupon_dates, is_end_of_month};
use crate::bonds::{FixedRateBond, TreasuryBill};
use crate::core::calendar::Calendar;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::lattice::LatticeConfig;
use crate::core::quotes::Quote;
use crate::core::trade::PutOrCall;
use crate::core::traits::Instrument;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::blackscholes::implied_vol_from_price;
use crate::equity::contracts::asian::{AsianStrikeType, AveragingType};
use crate::equity::contracts::barrier::{BarrierDirection, KnockType};
use crate::equity::contracts::chooser::ChooserPayoff;
use crate::equity::finite_difference::FdConfig;
use crate::equity::montecarlo::MonteCarloConfig;
use crate::equity::utils::{LongShort, Model, Payoff, PricingEngine};
use crate::equity::vanilla_option::{
    AsianPayoff, BarrierPayoff, BinaryPayoff, BinaryType, EquityMarketData, EquityOption,
    EquityOptionBase, LookbackPayoff, LookbackType, VanillaPayoff,
};

/// True when the error is the user backing out (Esc / Ctrl-C) rather
/// than a real failure.
pub fn is_cancelled(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<InquireError>(),
        Some(InquireError::OperationCanceled | InquireError::OperationInterrupted)
    )
}

// ── Prompt helpers ──────────────────────────────────────────────────────

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

fn fraction_with_default(message: &str, default: f64) -> Result<f64> {
    Ok(CustomType::<f64>::new(message)
        .with_error_message("please enter a fraction strictly between 0 and 1, e.g. 0.5")
        .with_default(default)
        .with_validator(|value: &f64| {
            if *value > 0.0 && *value < 1.0 {
                Ok(Validation::Valid)
            } else {
                Ok(Validation::Invalid(
                    "must be strictly between 0 and 1".into(),
                ))
            }
        })
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

/// The terms of a European vanilla option collected from the terminal
/// for the implied-vol flow (the volatility is what gets solved for).
struct OptionTerms {
    spot: f64,
    side: PutOrCall,
    strike: f64,
    rate: f64,
    dividend: f64,
    maturity: NaiveDate,
}

fn prompt_option_terms() -> Result<OptionTerms> {
    Ok(OptionTerms {
        spot: positive_f64("Spot price of the underlying:")?,
        side: prompt_side()?,
        strike: positive_f64("Strike price:")?,
        rate: f64_with_default("Continuously compounded risk-free rate:", 0.0)?,
        dividend: f64_with_default("Continuous dividend yield:", 0.0)?,
        maturity: future_date("Maturity date (YYYY-MM-DD):")?,
    })
}

// ── Product / style / engine matrix ─────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum Product {
    Vanilla,
    FuturesOption,
    Binary,
    Barrier,
    Asian,
    Lookback,
    Chooser,
}

const PRODUCTS: [(&str, Product); 7] = [
    ("Vanilla", Product::Vanilla),
    ("Option on future (Black-76)", Product::FuturesOption),
    ("Binary (digital)", Product::Binary),
    ("Barrier", Product::Barrier),
    ("Asian (average)", Product::Asian),
    ("Lookback", Product::Lookback),
    ("Chooser", Product::Chooser),
];

fn prompt_product() -> Result<Product> {
    let labels: Vec<&str> = PRODUCTS.iter().map(|(label, _)| *label).collect();
    let choice = Select::new("Product:", labels).prompt()?;
    Ok(PRODUCTS
        .iter()
        .find(|(label, _)| *label == choice)
        .expect("selection comes from the list")
        .1)
}

/// Exercise style, where the product supports a choice. Path-dependent
/// products, options on futures and the chooser price European-only in
/// this library.
fn prompt_style(product: Product) -> Result<ContractStyle> {
    match product {
        Product::Vanilla | Product::Binary => {
            let style = Select::new("Exercise style:", vec!["European", "American"]).prompt()?;
            Ok(if style == "American" {
                ContractStyle::American
            } else {
                ContractStyle::European
            })
        }
        _ => Ok(ContractStyle::European),
    }
}

/// Engine menu for a product/style combination — only combinations the
/// pricing library accepts are offered.
fn engine_choices(product: Product, style: &ContractStyle) -> Vec<&'static str> {
    let american = matches!(style, ContractStyle::American);
    match (product, american) {
        (Product::Vanilla, false) | (Product::Binary, false) => vec![
            "Analytical (closed form)",
            "Monte Carlo",
            "Binomial tree",
            "Finite difference",
        ],
        (Product::Vanilla, true) => vec![
            "Binomial tree",
            "Finite difference",
            "Monte Carlo (Longstaff-Schwartz)",
            "Barone-Adesi-Whaley (approximation)",
            "Bjerksund-Stensland (approximation)",
        ],
        (Product::Binary, true) => vec![
            "Binomial tree",
            "Finite difference",
            "Monte Carlo (Longstaff-Schwartz)",
        ],
        // path-dependent: no binomial; FD covers barriers only
        (Product::Barrier, _) => vec![
            "Analytical (Reiner-Rubinstein)",
            "Monte Carlo",
            "Finite difference",
        ],
        (Product::Asian, _) => vec![
            "Analytical (geometric exact / Turnbull-Wakeman)",
            "Monte Carlo",
        ],
        (Product::Lookback, _) => vec!["Analytical (closed form)", "Monte Carlo"],
        (Product::FuturesOption, _) => vec!["Analytical (Black-76 closed form)"],
        (Product::Chooser, _) => vec!["Analytical (closed form)"],
    }
}

/// Map a menu label to a pricing engine with its default configuration.
fn engine_from_label(label: &str) -> PricingEngine {
    if label.starts_with("Analytical") {
        PricingEngine::BlackScholes
    } else if label.starts_with("Monte Carlo") {
        PricingEngine::MonteCarlo(MonteCarloConfig::default())
    } else if label.starts_with("Binomial") {
        PricingEngine::Binomial(LatticeConfig::default())
    } else if label.starts_with("Finite difference") {
        PricingEngine::FiniteDifference(FdConfig::default())
    } else if label.starts_with("Barone-Adesi-Whaley") {
        PricingEngine::BaroneAdesiWhaley
    } else {
        PricingEngine::BjerksundStensland
    }
}

fn prompt_engine(product: Product, style: &ContractStyle) -> Result<PricingEngine> {
    let choices = engine_choices(product, style);
    let choice = if choices.len() == 1 {
        println!(
            "Pricing engine: {} (the only engine for this product)",
            choices[0]
        );
        choices[0]
    } else {
        Select::new("Pricing engine:", choices).prompt()?
    };
    Ok(engine_from_label(choice))
}

/// Product-specific payoff prompts. Returns the payoff and whether the
/// strike entered is actually used by the product (floating-strike
/// products derive their strike from the path).
fn prompt_payoff(product: Product, style: ContractStyle) -> Result<Box<dyn Payoff>> {
    match product {
        Product::Vanilla | Product::FuturesOption => Ok(Box::new(VanillaPayoff {
            put_or_call: prompt_side()?,
            exercise_style: style,
        })),
        Product::Binary => {
            let side = prompt_side()?;
            let kind = Select::new(
                "Binary type:",
                vec![
                    "Cash-or-nothing (pays a fixed amount)",
                    "Asset-or-nothing (pays the underlying level)",
                ],
            )
            .prompt()?;
            let binary_type = if kind.starts_with("Asset") {
                BinaryType::AssetOrNothing
            } else {
                BinaryType::CashOrNothing
            };
            let cash = if binary_type == BinaryType::CashOrNothing {
                positive_f64("Cash amount paid when in the money:")?
            } else {
                0.0
            };
            Ok(Box::new(BinaryPayoff {
                put_or_call: side,
                exercise_style: style,
                binary_type,
                cash,
            }))
        }
        Product::Barrier => {
            let side = prompt_side()?;
            let direction = Select::new("Barrier direction:", vec!["Up", "Down"]).prompt()?;
            let knock = Select::new(
                "Knock type:",
                vec![
                    "Out (dies at the barrier)",
                    "In (comes alive at the barrier)",
                ],
            )
            .prompt()?;
            let barrier = positive_f64("Barrier level:")?;
            let rebate = f64_with_default("Rebate (paid at expiry, 0 for none):", 0.0)?;
            Ok(Box::new(BarrierPayoff {
                put_or_call: side,
                exercise_style: style,
                direction: if direction == "Up" {
                    BarrierDirection::Up
                } else {
                    BarrierDirection::Down
                },
                knock: if knock.starts_with("In") {
                    KnockType::In
                } else {
                    KnockType::Out
                },
                barrier,
                barrier2: None,
                rebate,
                rebate_at_hit: false,
            }))
        }
        Product::Asian => {
            let side = prompt_side()?;
            let averaging = Select::new("Averaging:", vec!["Arithmetic", "Geometric"]).prompt()?;
            let strike_kind = Select::new(
                "Strike type:",
                vec![
                    "Fixed strike (average vs strike)",
                    "Floating strike (terminal spot vs average)",
                ],
            )
            .prompt()?;
            Ok(Box::new(AsianPayoff {
                put_or_call: side,
                exercise_style: style,
                averaging: if averaging == "Geometric" {
                    AveragingType::Geometric
                } else {
                    AveragingType::Arithmetic
                },
                strike_type: if strike_kind.starts_with("Floating") {
                    AsianStrikeType::FloatingStrike
                } else {
                    AsianStrikeType::FixedStrike
                },
            }))
        }
        Product::Lookback => {
            let side = prompt_side()?;
            let kind = Select::new(
                "Lookback type:",
                vec![
                    "Floating strike (buy the low / sell the high)",
                    "Fixed strike (extremum vs strike)",
                ],
            )
            .prompt()?;
            Ok(Box::new(LookbackPayoff {
                put_or_call: side,
                exercise_style: style,
                lookback_type: if kind.starts_with("Fixed") {
                    LookbackType::FixedStrike
                } else {
                    LookbackType::FloatingStrike
                },
            }))
        }
        Product::Chooser => {
            let choice_fraction = fraction_with_default(
                "Choice time as a fraction of the option life (e.g. 0.5):",
                0.5,
            )?;
            Ok(Box::new(ChooserPayoff {
                exercise_style: style,
                choice_fraction,
                legs: None,
            }))
        }
    }
}

/// Whether the product actually exercises against an entered strike;
/// floating-strike products derive it from the path.
fn uses_strike(payoff: &dyn Payoff) -> bool {
    if let Some(asian) = payoff.as_any().downcast_ref::<AsianPayoff>() {
        return asian.strike_type == AsianStrikeType::FixedStrike;
    }
    if let Some(lookback) = payoff.as_any().downcast_ref::<LookbackPayoff>() {
        return lookback.lookback_type == LookbackType::FixedStrike;
    }
    true
}

/// Market terms shared by every product in the pricing wizard.
struct MarketTerms {
    spot: f64,
    strike: f64,
    vol: f64,
    rate: f64,
    dividend: f64,
    maturity: NaiveDate,
}

/// How an option on a future settles its premium, prompted for
/// [`Product::FuturesOption`]; `None` for every other product.
fn prompt_futures_settlement() -> Result<crate::equity::black76::FuturesSettlement> {
    use crate::equity::black76::FuturesSettlement;
    let choice = Select::new(
        "Premium settlement:",
        vec![
            "Discounted (premium paid up front)",
            "Margined (futures-style, undiscounted)",
        ],
    )
    .prompt()?;
    Ok(if choice.starts_with("Margined") {
        FuturesSettlement::Margined
    } else {
        FuturesSettlement::Discounted
    })
}

/// Assemble an option from prompted terms with a flat curve and flat
/// vol, ready to price on `engine`.
fn build_option(
    payoff: Box<dyn Payoff>,
    engine: PricingEngine,
    terms: &MarketTerms,
    futures_settlement: Option<crate::equity::black76::FuturesSettlement>,
) -> Result<EquityOption> {
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
        futures_settlement,
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
    Ok(EquityOption {
        base,
        market,
        payoff,
        engine,
        model: Model::Gbm,
    })
}

/// Price `option` and print the value with its Greeks (and the Monte
/// Carlo standard error when applicable).
fn print_pricing(option: &EquityOption, product: Product) -> Result<()> {
    let result = option.price()?;
    println!("Theoretical price  ${:.6}", result.pv);
    if let Some(std_err) = result.std_err {
        println!("MC standard error  ${:.6}", std_err);
    }
    if product == Product::Vanilla && matches!(option.engine, PricingEngine::BlackScholes) {
        println!("Premium at risk    ${:.6}", option.get_premium_at_risk());
    }
    println!("Delta  {:.6}", result.greeks.delta);
    println!("Gamma  {:.6}", result.greeks.gamma);
    println!("Vega   {:.6}  (per 1% vol)", result.greeks.vega * 0.01);
    println!("Theta  {:.6}  (per day)", result.greeks.theta / 365.0);
    println!("Rho    {:.6}  (per 1% rate)", result.greeks.rho * 0.01);
    Ok(())
}

/// After the first pricing, let the user reprice the same option with a
/// changed strike, volatility, spot, rate or maturity — without walking
/// the whole wizard again. Esc (or "Back") leaves the loop.
fn reprice_loop(option: &mut EquityOption, product: Product) -> Result<()> {
    let strike_editable = uses_strike(option.payoff.as_ref());
    let underlying = if product == Product::FuturesOption {
        "futures price"
    } else {
        "spot"
    };
    loop {
        let mut choices = Vec::new();
        if strike_editable {
            choices.push("Reprice with a different strike".to_string());
        }
        choices.extend([
            "Reprice with a different volatility".to_string(),
            format!("Reprice with a different {underlying}"),
            "Reprice with a different rate".to_string(),
            "Reprice with a different maturity".to_string(),
            "Back to the main menu".to_string(),
        ]);
        let choice = match Select::new("What next?", choices).prompt() {
            Ok(choice) => choice,
            // Esc backs out of the reprice loop, not the whole session
            Err(InquireError::OperationCanceled) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        if choice == "Back to the main menu" {
            return Ok(());
        }
        if choice.ends_with("strike") {
            option.base.strike_price = positive_f64("New strike price:")?;
        } else if choice.ends_with("volatility") {
            let vol = positive_f64("New annualized volatility:")?;
            let surface = VolSurface::flat(
                vol,
                option.market.valuation_date,
                DayCountConvention::Act365,
            )
            .context("invalid volatility")?;
            option.market.vol_surface = std::sync::Arc::new(surface);
        } else if choice.ends_with(underlying) {
            option.market.spot = Quote::new(positive_f64(&format!("New {underlying}:"))?);
        } else if choice.ends_with("rate") {
            let rate = f64_with_default("New continuously compounded risk-free rate:", 0.0)?;
            let curve = YieldCurve::flat(
                rate,
                option.market.valuation_date,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )
            .context("invalid risk-free rate")?;
            option.market.discount_curve = std::sync::Arc::new(curve);
        } else {
            option.base.maturity_date = future_date("New maturity date (YYYY-MM-DD):")?;
        }
        print_pricing(option, product)?;
    }
}

/// Prompt for a product, exercise style, engine and terms, price the
/// option, then offer quick reprices with individual terms changed.
pub fn price_option_wizard() -> Result<()> {
    let product = prompt_product()?;
    let style = prompt_style(product)?;
    let engine = prompt_engine(product, &style)?;
    let payoff = prompt_payoff(product, style)?;
    let on_future = product == Product::FuturesOption;
    let futures_settlement = if on_future {
        Some(prompt_futures_settlement()?)
    } else {
        None
    };

    let spot = if on_future {
        positive_f64("Futures price:")?
    } else {
        positive_f64("Spot price of the underlying:")?
    };
    let strike = if uses_strike(payoff.as_ref()) {
        positive_f64("Strike price:")?
    } else {
        println!("(floating strike: derived from the path, no strike to enter)");
        spot
    };
    let terms = MarketTerms {
        spot,
        strike,
        vol: positive_f64("Annualized volatility (e.g. 0.25 for 25%):")?,
        rate: f64_with_default("Continuously compounded risk-free rate:", 0.0)?,
        // Black-76 prices off the futures price directly; no carry input
        dividend: if on_future {
            0.0
        } else {
            f64_with_default("Continuous dividend yield:", 0.0)?
        },
        maturity: future_date("Maturity date (YYYY-MM-DD):")?,
    };

    let mut option = build_option(payoff, engine, &terms, futures_settlement)?;
    print_pricing(&option, product)?;
    reprice_loop(&mut option, product)
}

/// Prompt for a quoted option price and back out its implied
/// Black-Scholes volatility.
pub fn implied_vol_wizard() -> Result<()> {
    let terms = prompt_option_terms()?;
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

// ── Bond wizard ─────────────────────────────────────────────────────────

fn nonnegative_f64(message: &str) -> Result<f64> {
    Ok(CustomType::<f64>::new(message)
        .with_error_message("please enter a number, e.g. 0.045")
        .with_validator(|value: &f64| {
            if *value >= 0.0 {
                Ok(Validation::Valid)
            } else {
                Ok(Validation::Invalid("must not be negative".into()))
            }
        })
        .prompt()?)
}

fn date_with_default(message: &str, default: NaiveDate) -> Result<NaiveDate> {
    Ok(CustomType::<NaiveDate>::new(message)
        .with_error_message("enter a date as YYYY-MM-DD, e.g. 2026-08-05")
        .with_default(default)
        .prompt()?)
}

/// Optional ISO date: Enter on an empty line means "not provided".
fn optional_date(message: &str, help: &str) -> Result<Option<NaiveDate>> {
    let text = Text::new(message)
        .with_help_message(help)
        .with_validator(|input: &str| {
            if input.trim().is_empty()
                || NaiveDate::parse_from_str(input.trim(), "%Y-%m-%d").is_ok()
            {
                Ok(Validation::Valid)
            } else {
                Ok(Validation::Invalid(
                    "enter a date as YYYY-MM-DD, or leave empty".into(),
                ))
            }
        })
        .prompt()?;
    let text = text.trim();
    Ok(if text.is_empty() {
        None
    } else {
        Some(NaiveDate::parse_from_str(text, "%Y-%m-%d").expect("validated above"))
    })
}

/// Build a US Treasury note/bond from wizard terms: T+1 settlement on
/// the SIFMA calendar, and a missing dated date inferred as the coupon
/// anchor at or before settlement.
fn treasury_from_terms(
    coupon_rate: f64,
    dated: Option<NaiveDate>,
    maturity: NaiveDate,
    trade_date: NaiveDate,
) -> Result<(FixedRateBond, NaiveDate)> {
    let settlement = Calendar::UsGovernmentBond.add_business_days(trade_date, 1);
    let dated = match dated {
        Some(date) => date,
        None => {
            coupon_dates(settlement, maturity, 6, is_end_of_month(maturity))
                .context("maturity must be after settlement")?
                .prev_anchor
        }
    };
    let bond = FixedRateBond::us_treasury(100.0, coupon_rate, dated, maturity)
        .context("invalid bond terms")?;
    Ok((bond, settlement))
}

/// Print the full analytics set for a bond at a clean price / yield pair.
fn print_bond_analytics(
    bond: &FixedRateBond,
    settlement: NaiveDate,
    clean: f64,
    yield_rate: f64,
) -> Result<()> {
    let accrued = bond.accrued_interest(settlement)?;
    println!("Settlement          {settlement} (T+1)");
    println!("Clean price         {clean:.6}");
    println!("Accrued interest    {accrued:.6}");
    println!("Dirty price         {:.6}", clean + accrued);
    println!("Street yield        {:.4} %", yield_rate * 100.0);
    println!(
        "Macaulay duration   {:.4} years",
        bond.macaulay_duration(yield_rate, settlement)?
    );
    println!(
        "Modified duration   {:.4} years",
        bond.modified_duration(yield_rate, settlement)?
    );
    println!(
        "Convexity           {:.4} years^2",
        bond.convexity(yield_rate, settlement)?
    );
    println!(
        "DV01                {:.6} per 100 face / bp",
        bond.dv01(yield_rate, settlement)?
    );
    Ok(())
}

/// Price a bond from one of the three quote forms and print the result.
fn price_bond_quote(
    bond: &FixedRateBond,
    settlement: NaiveDate,
    quote: &str,
    value: f64,
) -> Result<()> {
    let (clean, yield_rate) = if quote.starts_with("Clean") {
        (value, bond.yield_from_clean_price(value, settlement)?)
    } else if quote.starts_with("Street") {
        (bond.clean_price_from_yield(value, settlement)?, value)
    } else {
        let curve = YieldCurve::flat(
            value,
            settlement,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .context("invalid curve rate")?;
        let clean = bond.clean_price_from_curve(&curve, settlement)?;
        (clean, bond.yield_from_clean_price(clean, settlement)?)
    };
    print_bond_analytics(bond, settlement, clean, yield_rate)
}

const BOND_QUOTES: [&str; 3] = [
    "Clean price (per 100 face)",
    "Street yield (e.g. 0.045)",
    "Flat curve rate (continuously compounded)",
];

fn prompt_bond_quote_value(quote: &str) -> Result<f64> {
    if quote.starts_with("Clean") {
        positive_f64("Clean price per 100 face:")
    } else if quote.starts_with("Street") {
        f64_with_default("Street yield (e.g. 0.045 for 4.5%):", 0.04)
    } else {
        f64_with_default("Flat curve rate (continuously compounded):", 0.04)
    }
}

fn treasury_note_wizard() -> Result<()> {
    let coupon_rate = nonnegative_f64("Annual coupon rate (e.g. 0.045 for 4.5%):")?;
    let maturity = future_date("Maturity date (YYYY-MM-DD):")?;
    let trade_date = date_with_default("Trade date:", Local::now().date_naive())?;
    let dated = optional_date(
        "Dated date (interest accrual start):",
        "Enter to infer the previous coupon date from the maturity cycle",
    )?;
    let (bond, settlement) = treasury_from_terms(coupon_rate, dated, maturity, trade_date)?;

    let quote = Select::new("Quote by:", BOND_QUOTES.to_vec()).prompt()?;
    let value = prompt_bond_quote_value(quote)?;
    price_bond_quote(&bond, settlement, quote, value)?;

    // reprice loop: switch quote form or value without re-entering terms
    loop {
        let mut choices: Vec<String> = BOND_QUOTES
            .iter()
            .map(|q| format!("Reprice by {}", q.to_lowercase()))
            .collect();
        choices.push("Back to the main menu".to_string());
        let choice = match Select::new("What next?", choices).prompt() {
            Ok(choice) => choice,
            Err(InquireError::OperationCanceled) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        if choice == "Back to the main menu" {
            return Ok(());
        }
        let quote = BOND_QUOTES
            .iter()
            .find(|q| choice.contains(&q.to_lowercase()))
            .expect("choice comes from the list");
        let value = prompt_bond_quote_value(quote)?;
        price_bond_quote(&bond, settlement, quote, value)?;
    }
}

fn treasury_bill_wizard() -> Result<()> {
    let maturity = future_date("Maturity date (YYYY-MM-DD):")?;
    let trade_date = date_with_default("Trade date:", Local::now().date_naive())?;
    let settlement = Calendar::UsGovernmentBond.add_business_days(trade_date, 1);
    let bill = TreasuryBill::new(100.0, maturity).context("invalid bill terms")?;

    let print_bill = |discount_rate: f64| -> Result<()> {
        let price = bill.price_from_discount_rate(discount_rate, settlement)?;
        println!("Settlement          {settlement} (T+1)");
        println!("Price               {price:.6} per 100 face");
        println!("Discount rate       {:.4} %", discount_rate * 100.0);
        println!(
            "Bond-equiv yield    {:.4} %",
            bill.bond_equivalent_yield(discount_rate, settlement)? * 100.0
        );
        println!("Days to maturity    {}", (maturity - settlement).num_days());
        Ok(())
    };

    let quote_bill = |quote: &str| -> Result<f64> {
        if quote.starts_with("Discount") {
            positive_f64("Act/360 discount rate (e.g. 0.048 for 4.8%):")
        } else {
            let price = positive_f64("Price per 100 face:")?;
            Ok(bill.discount_rate_from_price(price, settlement)?)
        }
    };

    let quotes = ["Discount rate (Act/360)", "Price (per 100 face)"];
    let quote = Select::new("Quote by:", quotes.to_vec()).prompt()?;
    print_bill(quote_bill(quote)?)?;

    loop {
        let mut choices: Vec<String> = quotes
            .iter()
            .map(|q| format!("Reprice by {}", q.to_lowercase()))
            .collect();
        choices.push("Back to the main menu".to_string());
        let choice = match Select::new("What next?", choices).prompt() {
            Ok(choice) => choice,
            Err(InquireError::OperationCanceled) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        if choice == "Back to the main menu" {
            return Ok(());
        }
        let quote = quotes
            .iter()
            .find(|q| choice.contains(&q.to_lowercase()))
            .expect("choice comes from the list");
        print_bill(quote_bill(quote)?)?;
    }
}

/// Prompt for a US Treasury note/bond or bill and price it with full
/// street-convention analytics.
pub fn price_bond_wizard() -> Result<()> {
    let kind = Select::new(
        "Instrument:",
        vec![
            "Treasury note/bond (fixed coupon)",
            "Treasury bill (discount)",
        ],
    )
    .prompt()?;
    if kind.starts_with("Treasury bill") {
        treasury_bill_wizard()
    } else {
        treasury_note_wizard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exercise styles the wizard offers for each product.
    fn styles_for(product: Product) -> Vec<ContractStyle> {
        match product {
            Product::Vanilla | Product::Binary => {
                vec![ContractStyle::European, ContractStyle::American]
            }
            _ => vec![ContractStyle::European],
        }
    }

    /// A payoff mirroring what `prompt_payoff` builds, without prompts.
    fn payoff_for(product: Product, style: ContractStyle) -> Box<dyn Payoff> {
        match product {
            Product::Vanilla | Product::FuturesOption => Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: style,
            }),
            Product::Binary => Box::new(BinaryPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: style,
                binary_type: BinaryType::CashOrNothing,
                cash: 10.0,
            }),
            Product::Barrier => Box::new(BarrierPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: style,
                direction: BarrierDirection::Up,
                knock: KnockType::Out,
                barrier: 130.0,
                barrier2: None,
                rebate: 0.0,
                rebate_at_hit: false,
            }),
            Product::Asian => Box::new(AsianPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: style,
                averaging: AveragingType::Arithmetic,
                strike_type: AsianStrikeType::FixedStrike,
            }),
            Product::Lookback => Box::new(LookbackPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: style,
                lookback_type: LookbackType::FloatingStrike,
            }),
            Product::Chooser => Box::new(ChooserPayoff {
                exercise_style: style,
                choice_fraction: 0.5,
                legs: None,
            }),
        }
    }

    /// Every product/style/engine combination the wizard offers must pass
    /// the pricing library's engine-support gate — the menu must never
    /// lead the user into an "unsupported engine" error.
    #[test]
    fn every_offered_combination_is_accepted_by_the_pricer() {
        let terms = MarketTerms {
            spot: 100.0,
            strike: 100.0,
            vol: 0.25,
            rate: 0.03,
            dividend: 0.01,
            maturity: Local::now().date_naive() + chrono::Duration::days(365),
        };
        for &(label, product) in &PRODUCTS {
            // the wizard sets a settlement convention on futures options,
            // which is what routes them to the Black-76 pricer
            let settlement = if product == Product::FuturesOption {
                Some(crate::equity::black76::FuturesSettlement::Discounted)
            } else {
                None
            };
            for style in styles_for(product) {
                for engine_label in engine_choices(product, &style) {
                    let option = build_option(
                        payoff_for(product, style.clone()),
                        engine_from_label(engine_label),
                        &terms,
                        settlement,
                    )
                    .expect("wizard terms build");
                    assert!(
                        option.check_engine_support().is_ok(),
                        "menu offers unsupported combination: {label} / {style:?} / {engine_label}"
                    );
                }
            }
        }
    }

    /// The two closed-form engines must broadly agree with the lattice on
    /// a European vanilla, so menu default configs are sane.
    #[test]
    fn menu_default_engines_agree_on_a_european_vanilla() {
        let terms = MarketTerms {
            spot: 100.0,
            strike: 100.0,
            vol: 0.25,
            rate: 0.03,
            dividend: 0.01,
            maturity: Local::now().date_naive() + chrono::Duration::days(365),
        };
        let price = |engine_label: &str| {
            build_option(
                payoff_for(Product::Vanilla, ContractStyle::European),
                engine_from_label(engine_label),
                &terms,
                None,
            )
            .unwrap()
            .npv()
        };
        let analytic = price("Analytical (closed form)");
        let binomial = price("Binomial tree");
        let fd = price("Finite difference");
        assert!(
            (analytic - binomial).abs() < 1e-2,
            "{analytic} vs {binomial}"
        );
        assert!((analytic - fd).abs() < 1e-2, "{analytic} vs {fd}");
    }

    /// The reprice loop mutates the option in place; each mutation must
    /// actually move the price (no stale cached state), in the direction
    /// no-arbitrage dictates for a call.
    #[test]
    fn in_place_reprice_mutations_move_the_price() {
        let terms = MarketTerms {
            spot: 100.0,
            strike: 100.0,
            vol: 0.25,
            rate: 0.03,
            dividend: 0.01,
            maturity: Local::now().date_naive() + chrono::Duration::days(365),
        };
        let mut option = build_option(
            payoff_for(Product::Vanilla, ContractStyle::European),
            engine_from_label("Analytical (closed form)"),
            &terms,
            None,
        )
        .unwrap();
        let base_price = option.npv();

        // higher strike -> cheaper call
        option.base.strike_price = 110.0;
        let higher_strike = option.npv();
        assert!(
            higher_strike < base_price,
            "{higher_strike} vs {base_price}"
        );
        option.base.strike_price = 100.0;

        // higher vol (surface swapped in place) -> dearer call
        let surface = VolSurface::flat(
            0.40,
            option.market.valuation_date,
            DayCountConvention::Act365,
        )
        .unwrap();
        option.market.vol_surface = std::sync::Arc::new(surface);
        let higher_vol = option.npv();
        assert!(higher_vol > base_price, "{higher_vol} vs {base_price}");

        // higher spot -> dearer call
        option.market.spot = Quote::new(120.0);
        let higher_spot = option.npv();
        assert!(higher_spot > higher_vol, "{higher_spot} vs {higher_vol}");
    }

    /// The futures-option path must satisfy Black-76 put-call parity:
    /// C - P = df * (F - K) when the premium is discounted, F - K when
    /// it is margined (undiscounted).
    #[test]
    fn futures_option_satisfies_black76_put_call_parity() {
        use crate::equity::black76::FuturesSettlement;
        let terms = MarketTerms {
            spot: 105.0, // futures price
            strike: 100.0,
            vol: 0.25,
            rate: 0.03,
            dividend: 0.0,
            maturity: Local::now().date_naive() + chrono::Duration::days(365),
        };
        let price = |side: PutOrCall, settlement: FuturesSettlement| {
            build_option(
                Box::new(VanillaPayoff {
                    put_or_call: side,
                    exercise_style: ContractStyle::European,
                }),
                engine_from_label("Analytical (Black-76 closed form)"),
                &terms,
                Some(settlement),
            )
            .unwrap()
            .npv()
        };
        let t: f64 = 365.0 / 365.0;
        let df = (-terms.rate * t).exp();

        let discounted = price(PutOrCall::Call, FuturesSettlement::Discounted)
            - price(PutOrCall::Put, FuturesSettlement::Discounted);
        assert!(
            (discounted - df * (terms.spot - terms.strike)).abs() < 1e-9,
            "discounted parity: {discounted}"
        );

        let margined = price(PutOrCall::Call, FuturesSettlement::Margined)
            - price(PutOrCall::Put, FuturesSettlement::Margined);
        assert!(
            (margined - (terms.spot - terms.strike)).abs() < 1e-9,
            "margined parity: {margined}"
        );
    }

    /// The bond wizard's term builder: T+1 settlement on the bond
    /// calendar and an inferred dated date matching the explicit one.
    #[test]
    fn treasury_from_terms_infers_dated_date_and_settles_t_plus_1() {
        let d = |y, m, day| NaiveDate::from_ymd_opt(y, m, day).unwrap();
        // trade Friday Oct 9 2026: Monday Oct 12 is Columbus Day, so
        // T+1 settlement lands on Tuesday Oct 13
        let (bond, settlement) =
            treasury_from_terms(0.045, None, d(2028, 5, 15), d(2026, 10, 9)).unwrap();
        assert_eq!(settlement, d(2026, 10, 13));
        // inferred dated date = previous coupon anchor on the May/Nov cycle
        assert_eq!(bond.dated_date, d(2026, 5, 15));
        // identical to passing the dated date explicitly
        let (explicit, _) =
            treasury_from_terms(0.045, Some(d(2026, 5, 15)), d(2028, 5, 15), d(2026, 10, 9))
                .unwrap();
        assert_eq!(bond.coupon_dates(), explicit.coupon_dates());
        assert!(
            (bond.accrued_interest(settlement).unwrap()
                - explicit.accrued_interest(settlement).unwrap())
            .abs()
                < 1e-15
        );
        // pricing works end to end through the wizard's quote path
        let clean = bond.clean_price_from_yield(0.045, settlement).unwrap();
        let back = bond.yield_from_clean_price(clean, settlement).unwrap();
        assert!((back - 0.045).abs() < 1e-9);
    }
}
