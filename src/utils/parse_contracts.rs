use crate::core::data_models::ProductData;
use crate::core::traits::Rates;
use crate::core::utils::{CombinedContract, Contract, ContractOutput, Contracts};
use crate::equity::build_contracts::build_eq_contracts_from_json;
use crate::equity::handle_equity_contracts::handle_equity_contract;
use crate::equity::portfolio::EquityPortfolio;
use crate::equity::vanilla_option::EquityOption;
use crate::rates;
use crate::rates::build_contracts::{build_ir_contracts_from_json, build_term_structure};
use crate::rates::deposits::Deposit;
use anyhow::{bail, Context, Result};
use chrono::Local;
use std::fs;
use std::path::{Path, PathBuf};

use crate::core::serialization::{self, Format};
use rayon::prelude::*;
use serde_json::Value;

/// Confirm a build artifact on stdout, in green when it's a terminal.
fn saved_note(what: &str, path: &Path) {
    let success = crate::utils::style::SUCCESS;
    anstream::println!("{success}{what} saved to {}{success:#}", path.display());
}

/// Write `output` to `output_folder[/subfolder]/filename`, creating the
/// directories as needed, and return the path written to.
pub fn save_to_file(
    output_folder: &Path,
    subfolder: &str,
    filename: &str,
    output: &str,
) -> Result<PathBuf> {
    let mut dir = output_folder.to_path_buf();
    if !subfolder.is_empty() {
        dir.push(subfolder);
    }
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create output directory {}", dir.display()))?;
    dir.push(filename);
    fs::write(&dir, output).with_context(|| format!("failed to write {}", dir.display()))?;
    Ok(dir)
}

/// Build a curve (term structure, volatility surface, ...) from a
/// contracts document and save it under `output_folder`. `source` names
/// the input (a path or "stdin") in error messages.
pub fn build_curve(contents: &str, source: &str, output_folder: &Path) -> Result<()> {
    let format = Format::detect(contents);
    let list_contracts: Contracts = serialization::parse(contents, format)
        .with_context(|| format!("failed to parse {format:?} curve definition {source}"))?;
    if list_contracts.contracts.is_empty() {
        bail!("no contracts found in {source}");
    }
    match list_contracts.asset.as_str() {
        "EQ" => {
            log::info!("building implied volatility surface");
            let contracts: Vec<Box<EquityOption>> =
                build_eq_contracts_from_json(list_contracts.contracts);
            let vol_surface = crate::equity::vol_surface::build_implied_vol_surface(&contracts)
                .context("failed to build implied vol surface")?;
            log::debug!("implied vol surface:\n{}", vol_surface);
            let vol_value =
                serde_json::to_value(&vol_surface).context("failed to serialize vol surface")?;
            let serialized_vol_surface =
                serialization::render_value(&vol_value, format, "vol_surface");
            let filename = format!("vol_surface.{}", format.extension());
            let out_path = save_to_file(
                output_folder,
                "vol_surface",
                &filename,
                &serialized_vol_surface,
            )?;
            saved_note("Volatility surface", &out_path);
        }
        "IR" => {
            let contracts: Vec<Box<dyn Rates>> =
                build_ir_contracts_from_json(list_contracts.contracts);
            let ts = build_term_structure(contracts);
            let mut output: String = String::new();
            for i in 0..ts.date.len() {
                output.push_str(&format!(
                    "{},{},{}\n",
                    ts.date[i], ts.discount_factor[i], ts.rate[i]
                ));
            }

            let out_path = save_to_file(
                output_folder,
                "term_structure",
                "term_structure.csv",
                &output,
            )?;
            saved_note("Term structure", &out_path);
        }
        "CO" => bail!("commodity curve building is not supported yet"),
        other => bail!("unsupported asset class `{other}` (expected EQ or IR)"),
    }
    Ok(())
}

/// Price every contract in a document and return the rendered results.
/// The input format is detected from the content (JSON or XML); the
/// output format defaults to the input format unless overridden. Returns
/// `Ok(None)` when the document holds no contracts.
pub fn price_document(contents: &str, out_format: Option<Format>) -> Result<Option<String>> {
    let in_format = Format::detect(contents);
    let out_format = out_format.unwrap_or(in_format);

    let list_contracts: Contracts = serialization::parse(contents, in_format)
        .with_context(|| format!("failed to parse {in_format:?} contracts"))?;

    if list_contracts.contracts.is_empty() {
        return Ok(None);
    }
    // parallel processing of each contract using rayon
    let mut output_vec: Vec<_> = list_contracts
        .contracts
        .par_iter()
        .enumerate()
        .map(|(index, data)| (index, process_contract(data)))
        .collect();
    output_vec.sort_by_key(|k| k.0);

    let results: Vec<Value> = output_vec.into_iter().map(|(_, v)| v).collect();
    Ok(Some(serialization::render_results(&results, out_format)))
}

/// Price every contract in `input_file` into `output_file`. The output
/// format comes from `format_override`, then the output file extension,
/// then the input format.
pub fn parse_contract(
    input_file: &Path,
    output_file: &Path,
    format_override: Option<Format>,
) -> Result<()> {
    let contents = fs::read_to_string(input_file)
        .with_context(|| format!("failed to read contract file {}", input_file.display()))?;

    let out_format = format_override.or_else(|| Format::from_path(output_file));
    let rendered = price_document(&contents, out_format)
        .with_context(|| format!("failed to price {}", input_file.display()))?;
    match rendered {
        Some(output_str) => fs::write(output_file, output_str)
            .with_context(|| format!("failed to write {}", output_file.display()))?,
        None => log::warn!(
            "no contracts found in {}; nothing written",
            input_file.display()
        ),
    }
    Ok(())
}

/// Load an equity options book from a contracts document for risk and
/// stress runs. Every contract must be an option on the same underlying;
/// the signed position quantity is taken from each contract's
/// `long_short` field (default 1).
pub fn build_portfolio(contents: &str) -> Result<EquityPortfolio> {
    let format = Format::detect(contents);
    let list_contracts: Contracts = serialization::parse(contents, format)
        .with_context(|| format!("failed to parse {format:?} portfolio document"))?;
    if list_contracts.contracts.is_empty() {
        bail!("no contracts found in portfolio document");
    }
    let mut book = EquityPortfolio::new();
    for (index, contract) in list_contracts.contracts.iter().enumerate() {
        let ProductData::Option(data) = &contract.product_type else {
            bail!("contract {index}: only option contracts can go into a risk/stress portfolio");
        };
        let option = *EquityOption::try_from_json(data)
            .with_context(|| format!("contract {index} failed to build"))?;
        if let Some(first) = book.positions.first() {
            if first.option.base.symbol != option.base.symbol {
                bail!(
                    "contract {index}: the portfolio must share one underlying \
                     (book is '{}', contract is '{}')",
                    first.option.base.symbol,
                    option.base.symbol
                );
            }
        }
        let quantity = data.base.long_short.unwrap_or(1) as f64;
        book.add(option, quantity);
    }
    Ok(book)
}

/// Price one contract, always producing one result `Value` per contract.
/// Failures are reported in the result's `error` field rather than by
/// panicking, so one bad contract cannot abort the batch (this runs on a
/// rayon worker thread).
pub fn process_contract(data: &Contract) -> Value {
    match (data.action.as_str(), data.asset.as_str()) {
        ("PV", "EQ") => handle_equity_contract(data),
        ("PV", "IR") => {
            price_ir_contract(data).unwrap_or_else(|e| error_result(data, format!("{e:#}")))
        }
        (action, asset) => error_result(
            data,
            format!("unsupported action/asset combination `{action}`/`{asset}`"),
        ),
    }
}

fn price_ir_contract(data: &Contract) -> Result<Value> {
    let rate_data = data
        .rate_data
        .clone()
        .context("IR contract is missing `rate_data`")?;
    let start_date_str = rate_data.start_date; // Only for 0M case
    let maturity_date_str = rate_data.maturity_date;
    let current_date = Local::now().date_naive();
    let maturity_date = rates::utils::convert_mm_to_date(maturity_date_str);
    let start_date = rates::utils::convert_mm_to_date(start_date_str);
    log::debug!("deposit maturity date {:?}", maturity_date);
    let mut deposit = Deposit {
        start_date,
        maturity_date,
        valuation_date: current_date,
        notional: rate_data.notional,
        fix_rate: rate_data.fix_rate,
        day_count: rates::utils::DayCountConvention::Act360,
        business_day_adjustment: 0,
        term_structure: None,
    };
    match rate_data.day_count.as_str() {
        "Act360" | "A360" => {
            deposit.day_count = rates::utils::DayCountConvention::Act360;
        }
        "Act365" | "A365" => {
            deposit.day_count = rates::utils::DayCountConvention::Act365;
        }
        "Thirty360" | "30/360" => {
            deposit.day_count = rates::utils::DayCountConvention::Thirty360;
        }
        other => {
            log::warn!("unrecognized day count `{other}`; defaulting to Act360");
        }
    }
    let df = deposit.get_discount_factor();
    log::debug!("deposit discount factor {:?}", df);
    Ok(Value::String("Work in progress".to_string()))
}

/// Render a failed contract in the same `{contract, output}` shape as a
/// priced one, with the message in `output.error`.
fn error_result(data: &Contract, msg: String) -> Value {
    log::warn!("contract error: {msg}");
    let combined = CombinedContract {
        contract: data.clone(),
        output: ContractOutput::from_error(msg),
    };
    serde_json::to_value(&combined)
        .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }))
}
