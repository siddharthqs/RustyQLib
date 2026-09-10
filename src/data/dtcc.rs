//! DTCC GCF Repo Index® (dtcc.com): overnight general-collateral repo
//! rates and par volume cleared at FICC's Government Securities Division.
//!
//! DTCC publishes, once per business day around 3:30pm ET, the
//! par-weighted average rate and total par value of overnight GCF Repo
//! trades for the two most-traded GCF CUSIPs: US Treasury (< 30-year
//! maturity, CUSIP 371487AE9) and Fannie Mae / Freddie Mac fixed-rate
//! MBS (CUSIP 371487AL3). Term trades, forward-start repos and the
//! inter-dealer-broker leg are excluded. The feed is a rolling one-year
//! CSV, free and keyless, published under DTCC's terms of use (personal
//! and informational use; no redistribution or commercial exploitation).
//!
//! Nothing here interprets the data: each daily row is passed through
//! as published — rates in percent, par value in US dollars — and
//! [`to_document`] wraps the selected row in a `metadata` block saying
//! what the index is and where it came from. [`parse_csv`],
//! [`select_row`] and [`to_document`] are pure functions, so everything
//! after the download is testable offline.
//!
//! The feed's own web chart reads the same CSV, validates the `TRAILER`
//! record count and rejects any row with a missing par or rate; the
//! parser here applies the same checks so a truncated or reshuffled file
//! cannot pass through silently.

use chrono::NaiveDate;

use crate::core::errors::RustyQLibError;

/// Human-readable name of the source, recorded in document metadata.
pub const SOURCE: &str = "DTCC GCF Repo Index (dtcc.com, FICC Government Securities Division)";

/// The rolling one-year CSV behind DTCC's GCF Repo Index chart. The chart
/// page loads it relative to its own host, and that host has moved once
/// already (`www.dtcc.com` → `cms-prod.dtcc.com`), so [`fetch`] tries
/// each of these in order and reports every failure if none answers.
pub const CSV_URLS: [&str; 2] = [
    "https://cms-prod.dtcc.com/data/gcfindex.csv",
    "https://www.dtcc.com/data/gcfindex.csv",
];

/// The five columns of the feed, in the order and spelling it publishes
/// them. Matched by header name (case-insensitively), never by position.
const COLUMNS: [&str; 5] = [
    "Date",
    "MBS Total PAR Value",
    "MBS Weighted Average",
    "Treasury Total PAR Value",
    "Treasury Weighted Average",
];

/// One collateral class on one day: par-weighted average overnight rate
/// in percent and total par value in US dollars, exactly as published.
/// `None` when the feed left both cells blank for that class (no
/// eligible trades that day).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcfComponent {
    /// Par-weighted average overnight GCF repo rate, percent.
    pub weighted_average_rate: f64,
    /// Total par value of overnight GCF repo cleared that day, USD.
    pub total_par_value: f64,
}

/// One business day of the index.
#[derive(Debug, Clone, PartialEq)]
pub struct GcfIndexRow {
    pub date: NaiveDate,
    /// Fannie Mae / Freddie Mac fixed-rate MBS collateral.
    pub mbs: Option<GcfComponent>,
    /// US Treasury (< 30-year) collateral.
    pub treasury: Option<GcfComponent>,
}

/// Parse the feed CSV into rows in file order (oldest first as published;
/// use [`select_row`] rather than relying on order). The header must
/// carry the five known columns; the trailing `TRAILER,<n>` record, when
/// present, must count the data rows exactly. Rates are validated to be
/// plausible percents and par values to be non-negative; a row with one
/// of a class's two cells blank is rejected, as the feed's own chart
/// rejects it. Values are otherwise kept as published.
pub fn parse_csv(text: &str) -> Result<Vec<GcfIndexRow>, RustyQLibError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|e| RustyQLibError::ParseError(format!("invalid CSV header: {e}")))?
        .clone();
    // column index of each known header, resolved by name
    let mut index = [None; 5];
    for (i, header) in headers.iter().enumerate() {
        if let Some(slot) = COLUMNS.iter().position(|c| c.eq_ignore_ascii_case(header)) {
            index[slot] = Some(i);
        } else {
            log::warn!("skipping unrecognized column `{header}` in the GCF Repo Index CSV");
        }
    }
    let index = index
        .iter()
        .zip(COLUMNS)
        .map(|(slot, name)| {
            slot.ok_or_else(|| {
                RustyQLibError::ParseError(format!(
                    "no `{name}` column — this does not look like the DTCC GCF Repo Index CSV \
                     (header: {})",
                    headers.iter().collect::<Vec<_>>().join(",")
                ))
            })
        })
        .collect::<Result<Vec<usize>, _>>()?;
    let cell = |record: &csv::StringRecord, slot: usize| -> String {
        record.get(index[slot]).unwrap_or("").to_string()
    };

    let mut rows = Vec::new();
    let mut trailer: Option<usize> = None;
    for (line, record) in reader.records().enumerate() {
        let line = line + 2;
        let record = record
            .map_err(|e| RustyQLibError::ParseError(format!("invalid CSV row {line}: {e}")))?;
        if trailer.is_some() {
            return Err(RustyQLibError::ParseError(format!(
                "row {line} comes after the TRAILER record"
            )));
        }
        let date_field = cell(&record, 0);
        if date_field.is_empty() {
            continue;
        }
        if date_field.eq_ignore_ascii_case("TRAILER") {
            let count = cell(&record, 1);
            trailer = Some(count.parse().map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {line}: TRAILER count `{count}` is not an integer"
                ))
            })?);
            continue;
        }
        let date = NaiveDate::parse_from_str(&date_field, "%m/%d/%Y")
            .or_else(|_| NaiveDate::parse_from_str(&date_field, "%Y-%m-%d"))
            .map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {line}: `{date_field}` is not a MM/DD/YYYY or YYYY-MM-DD date"
                ))
            })?;
        let mbs = parse_component(&cell(&record, 1), &cell(&record, 2), date, "MBS", line)?;
        let treasury =
            parse_component(&cell(&record, 3), &cell(&record, 4), date, "Treasury", line)?;
        if mbs.is_none() && treasury.is_none() {
            return Err(RustyQLibError::ParseError(format!(
                "row {line} ({date}) carries no rate for either collateral class"
            )));
        }
        rows.push(GcfIndexRow {
            date,
            mbs,
            treasury,
        });
    }
    if let Some(count) = trailer {
        if count != rows.len() {
            return Err(RustyQLibError::ParseError(format!(
                "TRAILER says {count} records but the file has {} — truncated download?",
                rows.len()
            )));
        }
    }
    if rows.is_empty() {
        return Err(RustyQLibError::ParseError(
            "no GCF Repo Index rows in the CSV".to_string(),
        ));
    }
    Ok(rows)
}

fn parse_component(
    par: &str,
    rate: &str,
    date: NaiveDate,
    class: &str,
    line: usize,
) -> Result<Option<GcfComponent>, RustyQLibError> {
    match (par.is_empty(), rate.is_empty()) {
        (true, true) => Ok(None),
        (false, false) => {
            let total_par_value: f64 = par.parse().map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {line} ({date}): {class} par value `{par}` is not a number"
                ))
            })?;
            if !(total_par_value.is_finite() && total_par_value >= 0.0) {
                return Err(RustyQLibError::ParseError(format!(
                    "row {line} ({date}): {class} par value {total_par_value} is negative"
                )));
            }
            let weighted_average_rate: f64 = rate.parse().map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {line} ({date}): {class} rate `{rate}` is not a number"
                ))
            })?;
            super::plausible_percent(
                weighted_average_rate,
                &format!("{date}: {class} weighted average"),
            )?;
            Ok(Some(GcfComponent {
                weighted_average_rate,
                total_par_value,
            }))
        }
        _ => Err(RustyQLibError::ParseError(format!(
            "row {line} ({date}): {class} has a par value or a rate but not both"
        ))),
    }
}

/// Pick the row for `date`, or the latest published one when `date` is
/// `None`. A missing date reports the nearest earlier published date so
/// weekends and holidays fail with an actionable message.
pub fn select_row(
    rows: &[GcfIndexRow],
    date: Option<NaiveDate>,
) -> Result<&GcfIndexRow, RustyQLibError> {
    super::select_dated(rows, date, |r| r.date, "GCF Repo Index")
}

fn component_json(component: Option<GcfComponent>) -> serde_json::Value {
    match component {
        Some(c) => serde_json::json!({
            "weighted_average_rate": c.weighted_average_rate,
            "total_par_value": c.total_par_value,
        }),
        None => serde_json::Value::Null,
    }
}

/// Render one day as a plain document: both collateral classes exactly
/// as published (percent rate, USD par) under a `metadata` block saying
/// what the index measures. The feed's own column names are recorded so
/// the mapping to the document's keys is auditable.
pub fn to_document(row: &GcfIndexRow) -> serde_json::Value {
    serde_json::json!({
        "metadata": {
            "source": SOURCE,
            "index": "DTCC GCF Repo Index",
            "description": "par-weighted average rate and total par value of overnight \
                            GCF Repo trades cleared at FICC GSD, per collateral class; \
                            term and forward-start trades and the inter-dealer-broker \
                            leg excluded",
            "index_date": row.date.to_string(),
            "tenor": "overnight",
            "unit": "percent",
            "par_unit": "USD",
            "collateral": {
                "mbs": "Fannie Mae / Freddie Mac fixed-rate MBS (GCF CUSIP 371487AL3)",
                "treasury": "US Treasury < 30-year maturity (GCF CUSIP 371487AE9)",
            },
            "columns": {
                "mbs.total_par_value": COLUMNS[1],
                "mbs.weighted_average_rate": COLUMNS[2],
                "treasury.total_par_value": COLUMNS[3],
                "treasury.weighted_average_rate": COLUMNS[4],
            },
        },
        "index": {
            "date": row.date.to_string(),
            "mbs": component_json(row.mbs),
            "treasury": component_json(row.treasury),
        },
    })
}

/// Download the rolling one-year CSV, trying each of [`CSV_URLS`] in
/// order. Returns the text and the URL that served it.
pub fn fetch() -> Result<(String, &'static str), RustyQLibError> {
    let mut failures = Vec::with_capacity(CSV_URLS.len());
    for url in CSV_URLS {
        match super::http_get(url) {
            Ok(text) => return Ok((text, url)),
            Err(e) => {
                log::warn!("{e}");
                failures.push(e.to_string());
            }
        }
    }
    Err(RustyQLibError::Network(format!(
        "the DTCC GCF Repo Index CSV is not reachable at any known URL: {}",
        failures.join("; ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// The feed's exact shape: MM/DD/YYYY dates, integer USD par, rates
    /// padded with spaces, a TRAILER record with the row count. Values
    /// are illustrative, not DTCC's (the data is licensed for personal
    /// use only and may not be redistributed).
    const SAMPLE: &str = "\
Date,MBS Total PAR Value,MBS Weighted Average,Treasury Total PAR Value,Treasury Weighted Average
08/03/2026,38000000000,  4.410,52000000000,  4.395
08/04/2026,41500000000,  4.402,49800000000,  4.388
08/05/2026,36900000000,  4.415,55100000000,  4.401
TRAILER,3,,,,,
";

    #[test]
    fn parses_the_real_feed_shape() {
        let rows = parse_csv(SAMPLE).unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].date, d(2026, 8, 3));
        let mbs = rows[2].mbs.unwrap();
        assert_eq!(mbs.weighted_average_rate, 4.415);
        assert_eq!(mbs.total_par_value, 36_900_000_000.0);
        let ust = rows[2].treasury.unwrap();
        assert_eq!(ust.weighted_average_rate, 4.401);
        assert_eq!(ust.total_par_value, 55_100_000_000.0);
    }

    #[test]
    fn columns_are_matched_by_name_not_position() {
        let shuffled = "\
Treasury Weighted Average,Date,MBS Weighted Average,MBS Total PAR Value,Treasury Total PAR Value
4.395,08/03/2026,4.410,38000000000,52000000000
";
        let rows = parse_csv(shuffled).unwrap();
        assert_eq!(rows[0].treasury.unwrap().weighted_average_rate, 4.395);
        assert_eq!(rows[0].mbs.unwrap().total_par_value, 38_000_000_000.0);
    }

    #[test]
    fn blank_class_is_null_but_half_blank_is_an_error() {
        let blank = "\
Date,MBS Total PAR Value,MBS Weighted Average,Treasury Total PAR Value,Treasury Weighted Average
08/03/2026,,,52000000000,4.395
";
        let rows = parse_csv(blank).unwrap();
        assert!(rows[0].mbs.is_none());
        assert!(rows[0].treasury.is_some());
        let half = "\
Date,MBS Total PAR Value,MBS Weighted Average,Treasury Total PAR Value,Treasury Weighted Average
08/03/2026,38000000000,,52000000000,4.395
";
        let err = parse_csv(half).unwrap_err().to_string();
        assert!(err.contains("not both"), "{err}");
    }

    #[test]
    fn malformed_files_are_rejected() {
        // wrong feed entirely
        let err = parse_csv("Date,1 Mo,2 Mo\n08/03/2026,4.1,4.2\n")
            .unwrap_err()
            .to_string();
        assert!(err.contains("MBS Total PAR Value"), "{err}");
        // trailer count disagrees with the rows: truncated download
        let truncated = SAMPLE.replace("TRAILER,3", "TRAILER,251");
        let err = parse_csv(&truncated).unwrap_err().to_string();
        assert!(err.contains("251") && err.contains("truncated"), "{err}");
        // rows after the trailer
        let tail = format!("{SAMPLE}08/06/2026,1,4.4,1,4.4\n");
        assert!(parse_csv(&tail)
            .unwrap_err()
            .to_string()
            .contains("after the TRAILER"));
        // units changed: 441 "percent"
        let bad_units = SAMPLE.replace("4.410", "441.0");
        assert!(parse_csv(&bad_units)
            .unwrap_err()
            .to_string()
            .contains("units"));
        // negative par
        let neg = SAMPLE.replace("38000000000", "-1");
        assert!(parse_csv(&neg)
            .unwrap_err()
            .to_string()
            .contains("negative"));
        // no rows at all
        assert!(parse_csv(&SAMPLE.lines().next().unwrap().to_string()).is_err());
    }

    #[test]
    fn select_row_picks_latest_or_exact_and_reports_gaps() {
        let rows = parse_csv(SAMPLE).unwrap();
        assert_eq!(select_row(&rows, None).unwrap().date, d(2026, 8, 5));
        assert_eq!(
            select_row(&rows, Some(d(2026, 8, 4))).unwrap().date,
            d(2026, 8, 4)
        );
        let err = select_row(&rows, Some(d(2026, 8, 9)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("2026-08-05"), "{err}");
    }

    #[test]
    fn document_carries_both_classes_verbatim_plus_metadata() {
        let rows = parse_csv(SAMPLE).unwrap();
        let doc = to_document(&rows[2]);
        assert_eq!(doc["metadata"]["index_date"], "2026-08-05");
        assert_eq!(doc["metadata"]["unit"], "percent");
        assert_eq!(doc["metadata"]["par_unit"], "USD");
        assert_eq!(doc["metadata"]["tenor"], "overnight");
        assert!(doc["metadata"]["source"].as_str().unwrap().contains("DTCC"));
        assert_eq!(
            doc["metadata"]["columns"]["treasury.weighted_average_rate"],
            "Treasury Weighted Average"
        );
        assert_eq!(doc["index"]["date"], "2026-08-05");
        assert_eq!(doc["index"]["mbs"]["weighted_average_rate"], 4.415);
        assert_eq!(doc["index"]["mbs"]["total_par_value"], 36_900_000_000.0);
        assert_eq!(doc["index"]["treasury"]["weighted_average_rate"], 4.401);
        assert_eq!(
            doc["index"]["treasury"]["total_par_value"],
            55_100_000_000.0
        );
        // a class with no trades that day is null, not invented
        let mut row = rows[2].clone();
        row.mbs = None;
        assert!(to_document(&row)["index"]["mbs"].is_null());
    }

    /// Live check that the endpoint and schema still exist. Excluded from
    /// normal runs: `cargo test --features fetch -- --ignored` to run it.
    #[test]
    #[ignore = "hits dtcc.com"]
    fn live_feed_still_parses() {
        let (text, url) = fetch().expect("fetch failed");
        let rows = parse_csv(&text).expect("parse failed");
        let latest = select_row(&rows, None).expect("no rows");
        assert!(latest.treasury.is_some() || latest.mbs.is_some(), "{url}");
        to_document(latest);
    }
}
