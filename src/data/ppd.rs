//! DTCC Public Price Dissemination (pddata.dtcc.com): credit derivative
//! prints reported in real time under CFTC Part 43 and SEC Regulation
//! SBSR.
//!
//! DTCC's swap data repositories publish every reportable trade shortly
//! after execution, and once a day roll the whole day's dissemination log
//! into one cumulative CSV per asset class and jurisdiction, served as a
//! zip archive with no login or key. The `CREDITS` files are the ones
//! this module reads:
//!
//! - **CFTC** — index CDS (CDX, iTraxx, CMBX), index tranches and index
//!   swaptions;
//! - **SEC** — single-name corporate and sovereign CDS (security-based
//!   swaps).
//!
//! Each row is one dissemination event: a new trade (`NEWT`), or a
//! modification, correction or termination of an earlier one (`MODI`,
//! `CORR`, `TERM`) that names the original row. The file is the log as
//! published — nothing here collapses that chain, because which state
//! is "current" is a consumer's decision; the ids needed to do it are
//! carried on every print.
//!
//! The traded level lives in two places, and the feed's `Price` column
//! is empty for credit: index prints carry the conventional spread as a
//! decimal (`Spread-Leg 1`) and the upfront fee in currency (`Other
//! payment amount`, type `UFRO`); single-name prints almost always carry
//! only the upfront and the fixed coupon. Notionals above the public
//! cap are published as the cap with a trailing `+`. No side is
//! published, so an upfront has no sign. A few columns hold aligned
//! `;`-separated lists (several other payments on one trade, several
//! identifiers for one reference obligation); those become arrays.
//!
//! [`parse_csv`] maps the 110-column regulatory schema to a
//! [`CdsPrint`] by header name (never by position) and keeps every value
//! as published; [`to_document`] records the column each field came
//! from so the mapping is auditable. [`unzip_csv`], [`parse_csv`] and
//! [`to_document`] are pure, so everything after the download is
//! testable offline.
//!
//! The data is published under DTCC's terms of use: internal and
//! personal use, no redistribution.

use std::io::Read;

use chrono::NaiveDate;
use serde::Serialize;

use crate::core::errors::RustyQLibError;

/// Human-readable name of the source, recorded in document metadata.
pub const SOURCE: &str = "DTCC Public Price Dissemination (pddata.dtcc.com): real-time public \
                          reporting under CFTC Part 43 and SEC Regulation SBSR";

/// Which repository's daily file: the CFTC one carries index CDS and
/// index swaptions, the SEC one single-name security-based swaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Jurisdiction {
    Cftc,
    Sec,
}

impl Jurisdiction {
    /// Both repositories, in the order the daily fetch reads them.
    pub const ALL: [Jurisdiction; 2] = [Jurisdiction::Cftc, Jurisdiction::Sec];

    /// The name as it appears in the file name (`CFTC`, `SEC`).
    pub fn name(self) -> &'static str {
        match self {
            Jurisdiction::Cftc => "CFTC",
            Jurisdiction::Sec => "SEC",
        }
    }

    fn path(self) -> &'static str {
        match self {
            Jurisdiction::Cftc => "cftc",
            Jurisdiction::Sec => "sec",
        }
    }

    /// Infer the jurisdiction from a downloaded file's name
    /// (`CFTC_CUMULATIVE_CREDITS_...` / `SEC_CUMULATIVE_CREDITS_...`).
    pub fn from_file_name(name: &str) -> Option<Jurisdiction> {
        let upper = name.to_ascii_uppercase();
        Jurisdiction::ALL
            .into_iter()
            .find(|j| upper.contains(&format!("{}_CUMULATIVE", j.name())))
    }
}

/// Recover the report day from a downloaded file's name
/// (`..._CREDITS_2026_09_09.zip`).
pub fn date_from_file_name(name: &str) -> Option<NaiveDate> {
    let bytes = name.as_bytes();
    (0..bytes.len().saturating_sub(9))
        .filter(|&i| name.is_char_boundary(i) && name.is_char_boundary(i + 10))
        .find_map(|i| NaiveDate::parse_from_str(&name[i..i + 10], "%Y_%m_%d").ok())
}

/// URL of the cumulative `CREDITS` zip for one repository and day.
pub fn cumulative_url(jurisdiction: Jurisdiction, date: NaiveDate) -> String {
    format!(
        "https://pddata.dtcc.com/ppd/api/report/cumulative/{}/{}_CUMULATIVE_CREDITS_{}.zip",
        jurisdiction.path(),
        jurisdiction.name(),
        date.format("%Y_%m_%d")
    )
}

/// One dissemination event, with every value as published. Fields are
/// `None` where the feed left the cell blank. Rates and spreads are the
/// feed's decimals (`0.00504` is 50.4 bp); amounts are in the currency
/// alongside them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CdsPrint {
    /// Which repository published the row, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jurisdiction: Option<Jurisdiction>,
    pub dissemination_id: String,
    /// For `MODI` / `CORR` / `TERM`: the row this one amends.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_dissemination_id: Option<String>,
    /// `NEWT`, `MODI`, `CORR` or `TERM`.
    pub action: String,
    /// `TRAD`, `NOVA`, `ETRM`, ... — why the row was published.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    pub event_timestamp: String,
    pub execution_timestamp: String,
    /// UPI short name, e.g. `NA/CDS Corp Idx`, `NA/CDS Corp SN Sr`,
    /// `NA/CDS Idx Swt`.
    pub product: String,
    pub upi: String,
    /// Index name (`CDX.NA.IG`) or, for single names, the reference
    /// obligation's seniority label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub underlier: Option<String>,
    /// Reference entity name, when the feed carries it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity_name: Option<String>,
    /// Reference obligation / entity identifiers as published (ISIN,
    /// CUSIP, Markit RED code, ...), one entry per `;`-separated value.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub underlier_ids: Vec<UnderlierId>,
    pub effective_date: NaiveDate,
    pub expiration_date: NaiveDate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notional: Option<f64>,
    /// The published notional is the public cap, not the traded size.
    pub notional_capped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
    /// Fixed coupon as a decimal (`0.01` = 100 bp running).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_rate: Option<f64>,
    /// Traded (conventional) spread as published; see `spread_notation`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread: Option<f64>,
    /// Feed notation code for `spread` (`3` = decimal, `1` = monetary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spread_notation: Option<String>,
    /// Upfront fee in `upfront_currency`, unsigned: the `UFRO` entry of
    /// `other_payments`, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upfront_amount: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upfront_currency: Option<String>,
    /// Every other-payment entry as published: upfront fees (`UFRO`),
    /// unwind payments (`UWIN`), partial exercises (`PEXH`), ...
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub other_payments: Vec<OtherPayment>,
    /// Index swaptions: strike spread, expiry and premium.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strike: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strike_notation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_exercise_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub option_premium: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub option_premium_currency: Option<String>,
    /// Index factor after defaults (`0.98` = two names removed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_rate_day_count: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_rate_payment_period: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixed_rate_payment_multiplier: Option<String>,
    /// `I` intended to clear, `Y` cleared, `N` uncleared.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cleared: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub block_trade: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_price: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_spread: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_standard_terms: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,
}

/// One entry of a print's other-payment list.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OtherPayment {
    /// Amount in `currency`, unsigned.
    pub amount: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub currency: Option<String>,
    /// Feed payment type (`UFRO` upfront fee, `UWIN` unwind, `PEXH`
    /// partial exercise, ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_type: Option<String>,
}

/// One identifier of a print's reference obligation or entity.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnderlierId {
    pub id: String,
    /// `ISIN`, `CUSIP`, `REDID`, `Bloomberg`, ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// `(document field, feed column)` for every column the parser reads —
/// the mapping recorded in document metadata.
const COLUMNS: [(&str, &str); 42] = [
    ("dissemination_id", "Dissemination Identifier"),
    (
        "original_dissemination_id",
        "Original Dissemination Identifier",
    ),
    ("action", "Action type"),
    ("event", "Event type"),
    ("event_timestamp", "Event timestamp"),
    ("execution_timestamp", "Execution Timestamp"),
    ("product", "UPI FISN"),
    ("upi", "Unique Product Identifier"),
    ("underlier", "UPI Underlier Name"),
    ("entity_name", "Underlying Asset Name"),
    ("underlier_ids.id", "Underlier ID-Leg 1"),
    ("underlier_ids.source", "Underlier ID source-Leg 1"),
    ("effective_date", "Effective Date"),
    ("expiration_date", "Expiration Date"),
    ("notional", "Notional amount-Leg 1"),
    ("currency", "Notional currency-Leg 1"),
    ("fixed_rate", "Fixed rate-Leg 1"),
    ("spread", "Spread-Leg 1"),
    ("spread_notation", "Spread notation-Leg 1"),
    ("upfront_amount", "Other payment amount"),
    ("upfront_currency", "Other payment currency"),
    ("other_payments.amount", "Other payment amount"),
    ("other_payments.currency", "Other payment currency"),
    ("other_payments.payment_type", "Other payment type"),
    ("strike", "Strike Price"),
    ("strike_notation", "Strike price notation"),
    ("first_exercise_date", "First exercise date"),
    ("option_premium", "Option Premium Amount"),
    ("option_premium_currency", "Option Premium Currency"),
    ("index_factor", "Index factor"),
    (
        "fixed_rate_day_count",
        "Fixed rate day count convention-leg 1",
    ),
    (
        "fixed_rate_payment_period",
        "Fixed rate payment frequency period-Leg 1",
    ),
    (
        "fixed_rate_payment_multiplier",
        "Fixed rate payment frequency period multiplier-Leg 1",
    ),
    ("cleared", "Cleared"),
    ("block_trade", "Block trade election indicator"),
    ("package", "Package indicator"),
    ("package_price", "Package transaction price"),
    ("package_spread", "Package transaction spread"),
    ("non_standard_terms", "Non-standardized term indicator"),
    ("platform", "Platform identifier"),
    ("asset_class", "Asset Class"),
    ("notional_capped", "Notional amount-Leg 1 (trailing `+`)"),
];

/// Columns without which the file cannot be a PPD `CREDITS` report.
const REQUIRED: [&str; 7] = [
    "Dissemination Identifier",
    "Action type",
    "Execution Timestamp",
    "Effective Date",
    "Expiration Date",
    "Unique Product Identifier",
    "UPI FISN",
];

/// Extract the one CSV inside a PPD cumulative zip archive.
pub fn unzip_csv(bytes: &[u8]) -> Result<String, RustyQLibError> {
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
        .map_err(|e| RustyQLibError::ParseError(format!("not a zip archive: {e}")))?;
    let csv_index = (0..archive.len())
        .find(|&i| {
            archive
                .by_index(i)
                .map(|f| f.name().to_ascii_lowercase().ends_with(".csv"))
                .unwrap_or(false)
        })
        .ok_or_else(|| {
            RustyQLibError::ParseError("the zip archive contains no .csv file".to_string())
        })?;
    let file = archive
        .by_index(csv_index)
        .map_err(|e| RustyQLibError::ParseError(format!("unreadable zip entry: {e}")))?;
    let mut text = String::new();
    file.take(super::MAX_BODY_BYTES as u64 + 1)
        .read_to_string(&mut text)
        .map_err(|e| RustyQLibError::ParseError(format!("could not read the CSV entry: {e}")))?;
    if text.len() > super::MAX_BODY_BYTES {
        return Err(RustyQLibError::ParseError(format!(
            "the CSV inside the archive exceeds the {}-byte limit",
            super::MAX_BODY_BYTES
        )));
    }
    Ok(text)
}

/// Parse a cumulative `CREDITS` CSV (the zip's contents, or the CSV
/// itself) into prints, in file order. Columns are matched by header
/// name; the seven columns that define the schema must be present.
/// `jurisdiction` labels every print when the caller knows which file
/// it is reading.
pub fn parse_csv(
    text: &str,
    jurisdiction: Option<Jurisdiction>,
) -> Result<Vec<CdsPrint>, RustyQLibError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|e| RustyQLibError::ParseError(format!("invalid CSV header: {e}")))?
        .clone();
    let position = |name: &str| headers.iter().position(|h| h.eq_ignore_ascii_case(name));
    for name in REQUIRED {
        if position(name).is_none() {
            return Err(RustyQLibError::ParseError(format!(
                "no `{name}` column — this does not look like a DTCC PPD CREDITS report \
                 ({} columns, first: {:?})",
                headers.len(),
                headers.get(0).unwrap_or("")
            )));
        }
    }
    let column_index: Vec<Option<usize>> =
        COLUMNS.iter().map(|(_, column)| position(column)).collect();
    let field = |record: &csv::StringRecord, name: &str| -> Option<String> {
        let slot = COLUMNS.iter().position(|(f, _)| *f == name)?;
        let value = record.get(column_index[slot]?)?.trim();
        (!value.is_empty()).then(|| value.to_string())
    };

    let mut prints = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let line = line + 2;
        let record = record
            .map_err(|e| RustyQLibError::ParseError(format!("invalid CSV row {line}: {e}")))?;
        let required = |name: &str| {
            field(&record, name)
                .ok_or_else(|| RustyQLibError::ParseError(format!("row {line}: `{name}` is blank")))
        };
        let date = |name: &str| -> Result<NaiveDate, RustyQLibError> {
            let value = required(name)?;
            NaiveDate::parse_from_str(&value, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {line}: {name} `{value}` is not a YYYY-MM-DD date"
                ))
            })
        };
        let number = |name: &str| -> Result<Option<f64>, RustyQLibError> {
            field(&record, name)
                .map(|value| parse_number(&value, name, line).map(|(n, _)| n))
                .transpose()
        };
        let flag = |name: &str| -> Result<Option<bool>, RustyQLibError> {
            field(&record, name)
                .map(|value| match value.to_ascii_uppercase().as_str() {
                    "TRUE" | "Y" => Ok(true),
                    "FALSE" | "N" => Ok(false),
                    _ => Err(RustyQLibError::ParseError(format!(
                        "row {line}: {name} `{value}` is not a boolean"
                    ))),
                })
                .transpose()
        };
        if let Some(class) = field(&record, "asset_class") {
            if !class.eq_ignore_ascii_case("CR") {
                return Err(RustyQLibError::ParseError(format!(
                    "row {line}: asset class `{class}` is not credit (CR) — wrong PPD file?"
                )));
            }
        }
        // aligned `;`-separated lists: several other payments on one
        // trade, several identifiers for one reference obligation
        let amounts = split_list(field(&record, "other_payments.amount"));
        let currencies = split_list(field(&record, "other_payments.currency"));
        let types = split_list(field(&record, "other_payments.payment_type"));
        let mut other_payments = Vec::with_capacity(amounts.len());
        for (i, amount) in amounts.iter().enumerate() {
            let (amount, _) = parse_number(amount, "other payment amount", line)?;
            other_payments.push(OtherPayment {
                amount,
                currency: aligned(&currencies, i),
                payment_type: aligned(&types, i),
            });
        }
        let upfront = other_payments
            .iter()
            .find(|p| p.payment_type.as_deref() == Some("UFRO"));
        let upfront_amount = upfront.map(|p| p.amount);
        let upfront_currency = upfront.and_then(|p| p.currency.clone());
        let ids = split_list(field(&record, "underlier_ids.id"));
        let sources = split_list(field(&record, "underlier_ids.source"));
        let underlier_ids = ids
            .iter()
            .enumerate()
            .map(|(i, id)| UnderlierId {
                id: id.clone(),
                source: aligned(&sources, i),
            })
            .collect();
        let (notional, notional_capped) = match field(&record, "notional") {
            Some(value) => {
                let (n, capped) = parse_number(&value, "notional", line)?;
                (Some(n), capped)
            }
            None => (None, false),
        };
        prints.push(CdsPrint {
            jurisdiction,
            dissemination_id: required("dissemination_id")?,
            original_dissemination_id: field(&record, "original_dissemination_id"),
            action: required("action")?,
            event: field(&record, "event"),
            event_timestamp: required("event_timestamp")?,
            execution_timestamp: required("execution_timestamp")?,
            product: required("product")?,
            upi: required("upi")?,
            underlier: field(&record, "underlier"),
            entity_name: field(&record, "entity_name"),
            underlier_ids,
            effective_date: date("effective_date")?,
            expiration_date: date("expiration_date")?,
            notional,
            notional_capped,
            currency: field(&record, "currency"),
            fixed_rate: number("fixed_rate")?,
            spread: number("spread")?,
            spread_notation: field(&record, "spread_notation"),
            upfront_amount,
            upfront_currency,
            other_payments,
            strike: number("strike")?,
            strike_notation: field(&record, "strike_notation"),
            first_exercise_date: field(&record, "first_exercise_date"),
            option_premium: number("option_premium")?,
            option_premium_currency: field(&record, "option_premium_currency"),
            index_factor: number("index_factor")?,
            fixed_rate_day_count: field(&record, "fixed_rate_day_count"),
            fixed_rate_payment_period: field(&record, "fixed_rate_payment_period"),
            fixed_rate_payment_multiplier: field(&record, "fixed_rate_payment_multiplier"),
            cleared: field(&record, "cleared"),
            block_trade: flag("block_trade")?,
            package: flag("package")?,
            package_price: number("package_price")?,
            package_spread: number("package_spread")?,
            non_standard_terms: flag("non_standard_terms")?,
            platform: field(&record, "platform"),
        });
    }
    Ok(prints)
}

/// Parse a feed number: thousands separators are dropped and a trailing
/// `+` (the notional cap marker) is reported as `capped`.
fn parse_number(value: &str, name: &str, line: usize) -> Result<(f64, bool), RustyQLibError> {
    let capped = value.ends_with('+');
    let digits: String = value
        .trim_end_matches('+')
        .chars()
        .filter(|c| *c != ',')
        .collect();
    let number: f64 = digits.parse().map_err(|_| {
        RustyQLibError::ParseError(format!("row {line}: {name} `{value}` is not a number"))
    })?;
    if !number.is_finite() {
        return Err(RustyQLibError::ParseError(format!(
            "row {line}: {name} `{value}` is not finite"
        )));
    }
    Ok((number, capped))
}

/// Split a `;`-separated feed cell into its non-empty entries.
fn split_list(value: Option<String>) -> Vec<String> {
    value
        .map(|v| {
            v.split(';')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Entry `i` of a list aligned with another: a single value applies to
/// every position, otherwise positions match.
fn aligned(list: &[String], i: usize) -> Option<String> {
    match list {
        [only] => Some(only.clone()),
        _ => list.get(i).cloned(),
    }
}

/// Keep the prints whose index name, entity name or reference id
/// contains `needle`, case-insensitively.
pub fn filter_underlier(prints: Vec<CdsPrint>, needle: &str) -> Vec<CdsPrint> {
    let needle = needle.to_ascii_uppercase();
    prints
        .into_iter()
        .filter(|p| {
            [&p.underlier, &p.entity_name]
                .into_iter()
                .flatten()
                .chain(p.underlier_ids.iter().map(|u| &u.id))
                .any(|s| s.to_ascii_uppercase().contains(&needle))
        })
        .collect()
}

/// Render the prints as a plain document under a `metadata` block that
/// says what the report is, which day it covers, how many rows of each
/// action it carries, and which feed column each field came from.
pub fn to_document(
    prints: &[CdsPrint],
    report_date: NaiveDate,
    underlier_filter: Option<&str>,
) -> serde_json::Value {
    let mut actions = serde_json::Map::new();
    for p in prints {
        let count = actions
            .entry(p.action.clone())
            .or_insert_with(|| serde_json::Value::from(0u64));
        *count = serde_json::Value::from(count.as_u64().unwrap_or(0) + 1);
    }
    let mut jurisdictions: Vec<&str> = prints
        .iter()
        .filter_map(|p| p.jurisdiction.map(Jurisdiction::name))
        .collect();
    jurisdictions.sort_unstable();
    jurisdictions.dedup();
    let columns: serde_json::Map<String, serde_json::Value> = COLUMNS
        .iter()
        .map(|(f, c)| ((*f).to_string(), serde_json::Value::from(*c)))
        .collect();
    serde_json::json!({
        "metadata": {
            "source": SOURCE,
            "report": "cumulative CREDITS dissemination log",
            "report_date": report_date.to_string(),
            "jurisdictions": jurisdictions,
            "description": "every credit derivative dissemination event of the day as \
                            published: new trades (NEWT) and the modifications, \
                            corrections and terminations (MODI/CORR/TERM) of earlier \
                            ones, which name the row they amend in \
                            original_dissemination_id; the log is not collapsed",
            "underlier_filter": underlier_filter,
            "prints": prints.len(),
            "actions": actions,
            "units": {
                "fixed_rate": "decimal per annum as published (0.01 = 100 bp)",
                "spread": "as published; spread_notation 3 = decimal (0.00504 = 50.4 bp)",
                "upfront_amount": "upfront_currency, unsigned — PPD publishes no side",
                "notional": "currency; the public cap with notional_capped = true",
                "other_payments": "the feed's `;`-separated aligned lists, one entry each; upfront_amount is the UFRO entry",
                "underlier_ids": "the feed's `;`-separated aligned lists, one entry each",
            },
            "columns": columns,
        },
        "prints": prints,
    })
}

/// Download one repository's cumulative `CREDITS` zip for `date`.
pub fn fetch_cumulative(
    jurisdiction: Jurisdiction,
    date: NaiveDate,
) -> Result<Vec<u8>, RustyQLibError> {
    super::http_get_bytes(&cumulative_url(jurisdiction, date))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real 110-column header with illustrative rows (the data is
    /// licensed for internal use and may not be redistributed).
    const SAMPLE: &str = include_str!("../../tests/fixtures/ppd_cds_sample.csv");

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn parses_the_real_schema_by_column_name() {
        let prints = parse_csv(SAMPLE, Some(Jurisdiction::Cftc)).unwrap();
        assert_eq!(prints.len(), 8);
        let ig = &prints[0];
        assert_eq!(ig.jurisdiction, Some(Jurisdiction::Cftc));
        assert_eq!(ig.underlier.as_deref(), Some("CDX.NA.IG"));
        assert_eq!(ig.product, "NA/CDS Corp Idx");
        assert_eq!(ig.action, "NEWT");
        assert_eq!(ig.expiration_date, d(2031, 6, 20));
        assert_eq!(ig.effective_date, d(2026, 9, 10));
        assert_eq!(ig.notional, Some(25_000_000.0));
        assert!(!ig.notional_capped);
        assert_eq!(ig.fixed_rate, Some(0.01));
        assert_eq!(ig.spread, Some(0.00504));
        assert_eq!(ig.spread_notation.as_deref(), Some("3"));
        assert_eq!(ig.upfront_amount, Some(586_206.73));
        assert_eq!(ig.other_payments.len(), 1);
        assert_eq!(ig.other_payments[0].payment_type.as_deref(), Some("UFRO"));
        assert_eq!(ig.index_factor, Some(1.0));
        assert_eq!(ig.block_trade, Some(false));
        assert_eq!(ig.cleared.as_deref(), Some("I"));
        // block above the cap: notional is the cap, marked
        let block = &prints[1];
        assert_eq!(block.notional, Some(250_000_000.0));
        assert!(block.notional_capped);
        assert_eq!(block.block_trade, Some(true));
        // a correction names the row it amends
        let corr = &prints[3];
        assert_eq!(corr.action, "CORR");
        assert_eq!(
            corr.original_dissemination_id.as_deref(),
            Some("5200000000000000103")
        );
        // swaption fields
        let swaption = &prints[4];
        assert_eq!(swaption.product, "NA/CDS Idx Swt");
        assert_eq!(swaption.strike, Some(0.0055));
        assert_eq!(swaption.option_premium, Some(210_000.0));
        assert_eq!(swaption.first_exercise_date.as_deref(), Some("2026-10-21"));
        // single name: upfront only, ISIN reference obligation
        let sn = &prints[5];
        assert_eq!(
            sn.entity_name.as_deref(),
            Some("Example Semiconductor Inc.")
        );
        assert_eq!(sn.underlier_ids.len(), 1);
        assert_eq!(sn.underlier_ids[0].id, "US11135FBR10");
        assert_eq!(sn.underlier_ids[0].source.as_deref(), Some("ISIN"));
        assert!(sn.spread.is_none());
        assert_eq!(sn.upfront_amount, Some(45_507.745));
        assert!(sn.notional_capped);
        // termination
        assert_eq!(prints[6].action, "TERM");
        assert_eq!(prints[6].event.as_deref(), Some("ETRM"));
        // `;`-separated lists: two payments, two identifiers; the upfront
        // is the UFRO entry; a `;` inside a name is not a separator
        let multi = &prints[7];
        assert_eq!(multi.other_payments.len(), 2);
        assert_eq!(
            multi.other_payments[1].payment_type.as_deref(),
            Some("UWIN")
        );
        assert_eq!(multi.other_payments[1].amount, 115_213_480.0);
        assert_eq!(multi.other_payments[1].currency.as_deref(), Some("JPY"));
        assert_eq!(multi.upfront_amount, Some(78_410_107.0));
        assert_eq!(multi.upfront_currency.as_deref(), Some("JPY"));
        assert_eq!(multi.underlier_ids.len(), 2);
        assert_eq!(multi.underlier_ids[1].id, "123456AB7");
        assert_eq!(multi.underlier_ids[1].source.as_deref(), Some("CUSIP"));
        assert_eq!(
            multi.entity_name.as_deref(),
            Some("Example Trading Co;Ltd.")
        );
    }

    #[test]
    fn lists_split_and_align() {
        assert_eq!(split_list(None), Vec::<String>::new());
        assert_eq!(split_list(Some("a; b;;c".to_string())), ["a", "b", "c"]);
        let one = vec!["USD".to_string()];
        assert_eq!(aligned(&one, 3).as_deref(), Some("USD"));
        let two = vec!["UFRO".to_string(), "UWIN".to_string()];
        assert_eq!(aligned(&two, 1).as_deref(), Some("UWIN"));
        assert_eq!(aligned(&two, 2), None);
    }

    #[test]
    fn malformed_files_are_rejected() {
        let err = parse_csv("Date,1 Mo\n08/03/2026,4.1\n", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Dissemination Identifier"), "{err}");
        // a rates file has the same schema but a different asset class
        let rates = SAMPLE.replacen(",CR,", ",IR,", 1);
        let err = parse_csv(&rates, None).unwrap_err().to_string();
        assert!(err.contains("not credit"), "{err}");
        // a bad date and a bad number are named by row
        let bad_date = SAMPLE.replacen("2031-06-20", "06/20/2031", 1);
        let err = parse_csv(&bad_date, None).unwrap_err().to_string();
        assert!(err.contains("row 2") && err.contains("YYYY-MM-DD"), "{err}");
        let bad_number = SAMPLE.replacen("0.00504", "n/a", 1);
        let err = parse_csv(&bad_number, None).unwrap_err().to_string();
        assert!(err.contains("row 2") && err.contains("spread"), "{err}");
        // header only is fine: an empty day
        let header_only = SAMPLE.lines().next().unwrap().to_string() + "\n";
        assert!(parse_csv(&header_only, None).unwrap().is_empty());
    }

    #[test]
    fn numbers_drop_separators_and_report_the_cap() {
        assert_eq!(parse_number("250,000,000+", "n", 2).unwrap(), (250e6, true));
        assert_eq!(parse_number("5000000+", "n", 2).unwrap(), (5e6, true));
        assert_eq!(parse_number("0.00504", "n", 2).unwrap(), (0.00504, false));
        assert_eq!(parse_number("-0.00735", "n", 2).unwrap(), (-0.00735, false));
        assert!(parse_number("9.9+9", "n", 2).is_err());
        assert!(parse_number("NaN", "n", 2).is_err());
    }

    #[test]
    fn filter_matches_index_entity_or_reference_id() {
        let prints = parse_csv(SAMPLE, None).unwrap();
        assert_eq!(filter_underlier(prints.clone(), "cdx.na.ig").len(), 2);
        assert_eq!(filter_underlier(prints.clone(), "CDX.NA").len(), 4);
        assert_eq!(filter_underlier(prints.clone(), "semiconductor").len(), 1);
        assert_eq!(filter_underlier(prints.clone(), "9AAA4I").len(), 1);
        assert_eq!(
            filter_underlier(prints.clone(), "123456AB7").len(),
            1,
            "CUSIP"
        );
        assert!(filter_underlier(prints, "ITRAXX").is_empty());
    }

    #[test]
    fn document_counts_actions_and_records_the_column_mapping() {
        let mut prints = parse_csv(SAMPLE, Some(Jurisdiction::Cftc)).unwrap();
        prints[5].jurisdiction = Some(Jurisdiction::Sec);
        let doc = to_document(&prints, d(2026, 9, 9), Some("cdx"));
        let meta = &doc["metadata"];
        assert_eq!(meta["report_date"], "2026-09-09");
        assert_eq!(meta["prints"], 8);
        assert_eq!(meta["actions"]["NEWT"], 6);
        assert_eq!(meta["actions"]["CORR"], 1);
        assert_eq!(meta["actions"]["TERM"], 1);
        assert_eq!(meta["jurisdictions"], serde_json::json!(["CFTC", "SEC"]));
        assert_eq!(meta["underlier_filter"], "cdx");
        assert_eq!(meta["columns"]["spread"], "Spread-Leg 1");
        assert_eq!(
            meta["columns"]["other_payments.amount"],
            "Other payment amount"
        );
        assert_eq!(
            doc["prints"][7]["other_payments"][1]["payment_type"],
            "UWIN"
        );
        assert_eq!(doc["prints"][7]["underlier_ids"][0]["source"], "ISIN");
        let first = &doc["prints"][0];
        assert_eq!(first["jurisdiction"], "CFTC");
        assert_eq!(first["spread"], 0.00504);
        assert_eq!(first["expiration_date"], "2031-06-20");
        // blanks are absent, not null
        assert!(first.get("entity_name").is_none());
        assert!(first.get("strike").is_none());
    }

    #[test]
    fn unzip_finds_the_csv_entry() {
        use std::io::Write;
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        writer.start_file("README.txt", options).unwrap();
        writer.write_all(b"not the data").unwrap();
        writer
            .start_file("CFTC_CUMULATIVE_CREDITS_2026_09_09.csv", options)
            .unwrap();
        writer.write_all(SAMPLE.as_bytes()).unwrap();
        let bytes = writer.finish().unwrap().into_inner();
        let text = unzip_csv(&bytes).unwrap();
        assert_eq!(text, SAMPLE);
        assert_eq!(parse_csv(&text, None).unwrap().len(), 8);
        assert!(unzip_csv(b"PK not really").is_err());
        let mut empty = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        empty.start_file("notes.txt", options).unwrap();
        let bytes = empty.finish().unwrap().into_inner();
        assert!(unzip_csv(&bytes)
            .unwrap_err()
            .to_string()
            .contains("no .csv"));
    }

    #[test]
    fn urls_and_file_names_name_the_jurisdiction() {
        assert_eq!(
            cumulative_url(Jurisdiction::Cftc, d(2026, 9, 9)),
            "https://pddata.dtcc.com/ppd/api/report/cumulative/cftc/CFTC_CUMULATIVE_CREDITS_2026_09_09.zip"
        );
        assert_eq!(
            cumulative_url(Jurisdiction::Sec, d(2026, 1, 2)),
            "https://pddata.dtcc.com/ppd/api/report/cumulative/sec/SEC_CUMULATIVE_CREDITS_2026_01_02.zip"
        );
        assert_eq!(
            Jurisdiction::from_file_name("downloads/SEC_CUMULATIVE_CREDITS_2026_09_09.zip"),
            Some(Jurisdiction::Sec)
        );
        assert_eq!(
            Jurisdiction::from_file_name("cftc_cumulative_credits_2026_09_09.csv"),
            Some(Jurisdiction::Cftc)
        );
        assert_eq!(Jurisdiction::from_file_name("prints.csv"), None);
        assert_eq!(
            date_from_file_name("out/SEC_CUMULATIVE_CREDITS_2026_09_09.zip"),
            Some(d(2026, 9, 9))
        );
        assert_eq!(date_from_file_name("ppd_cds_sample.csv"), None);
    }

    /// Live check that the endpoint and schema still exist. Excluded from
    /// normal runs: `cargo test --features fetch -- --ignored` to run it.
    #[test]
    #[ignore = "hits pddata.dtcc.com"]
    fn live_feed_still_parses() {
        // the previous business day's file is complete; walk back over a weekend
        let mut date = chrono::Local::now().date_naive() - chrono::Days::new(1);
        let bytes = loop {
            match fetch_cumulative(Jurisdiction::Cftc, date) {
                Ok(bytes) => break bytes,
                Err(_) if date > chrono::Local::now().date_naive() - chrono::Days::new(6) => {
                    date = date - chrono::Days::new(1);
                }
                Err(e) => panic!("fetch failed: {e}"),
            }
        };
        let prints = parse_csv(&unzip_csv(&bytes).unwrap(), Some(Jurisdiction::Cftc)).unwrap();
        assert!(!prints.is_empty());
        assert!(prints
            .iter()
            .any(|p| p.spread.is_some() || p.upfront_amount.is_some()));
        to_document(&prints, date, None);
    }
}
