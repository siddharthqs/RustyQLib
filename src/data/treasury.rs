//! US Treasury daily par yield curve rates (home.treasury.gov).
//!
//! The Treasury publishes a fitted end-of-day curve: par yields at fixed
//! tenors (1 Mo ... 30 Yr), read off a monotone-convex spline through the
//! on-the-run quotes. The feed is official, free, keyless and public
//! domain, served as one CSV file per calendar year with the most recent
//! business day first.
//!
//! Nothing here interprets the data: [`to_document`] emits the curve
//! exactly as published — tenor labels and percent yields — under a
//! `metadata` block recording what the curve is and where it came from.
//! [`parse_csv`], [`select_row`] and [`to_document`] are pure functions,
//! so everything after the download is testable offline.

use chrono::{Days, Months, NaiveDate};

use crate::bonds::{bootstrap_curve, BillQuote, BondQuote, CurveInstrument, FixedRateBond, TreasuryBill};
use crate::core::calendar::Calendar;
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

/// Human-readable name of the source, recorded in document metadata.
pub const SOURCE: &str = "US Treasury daily par yield curve rates (home.treasury.gov)";

/// URL of the par-yield-curve CSV for one calendar year.
pub fn csv_url(year: i32) -> String {
    format!(
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/\
         daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve\
         &field_tdr_date_value={year}&page&_format=csv"
    )
}

/// One published tenor point, as it appears in the feed. `months` and
/// `extra_days` are the label resolved for programmatic use (the feed has
/// fractional-month columns like `"1.5 Month"`); the yield stays in the
/// feed's own percent units.
#[derive(Debug, Clone, PartialEq)]
pub struct ParYieldPoint {
    /// Column header as published, e.g. `"2 Yr"`.
    pub label: String,
    /// Whole months from the curve date to the tenor point.
    pub months: u32,
    /// Days on top of `months` for fractional-month tenors.
    pub extra_days: u32,
    /// Par yield in percent, exactly as published (e.g. `4.63`).
    pub yield_pct: f64,
}

/// One business day's fitted curve.
#[derive(Debug, Clone, PartialEq)]
pub struct ParYieldRow {
    pub date: NaiveDate,
    pub points: Vec<ParYieldPoint>,
}

/// Reject percent yields outside this band: a value above 50 almost
/// certainly means the feed changed units and must not pass through.
const PERCENT_BOUNDS: (f64, f64) = (-5.0, 50.0);

/// Parse a tenor column header (`"1 Mo"`, `"1.5 Month"`, `"30 Yr"`) into
/// whole months plus leftover days. `None` for anything unrecognized.
fn parse_tenor_header(header: &str) -> Option<(u32, u32)> {
    let mut parts = header.split_whitespace();
    let value: f64 = parts.next()?.parse().ok()?;
    let unit = parts.next()?.to_ascii_lowercase();
    if parts.next().is_some() {
        return None;
    }
    let months = match unit.as_str() {
        "mo" | "month" | "months" => value,
        "yr" | "year" | "years" => value * 12.0,
        _ => return None,
    };
    if !(months > 0.0 && months <= 1200.0) {
        return None;
    }
    // 30.4375 = average Gregorian month; only fractional tenors use it
    let whole = months.floor() as u32;
    let extra_days = ((months - months.floor()) * 30.4375).round() as u32;
    if whole == 0 && extra_days == 0 {
        return None;
    }
    Some((whole, extra_days))
}

/// Parse the yearly CSV into rows, in file order (the feed publishes the
/// most recent date first — use [`select_row`] rather than relying on
/// order). Columns are matched by header name, never by position, so the
/// Treasury adding or dropping a tenor cannot silently shift values; an
/// unrecognized column is skipped with a warning. Empty and `N/A` cells
/// are skipped. Yields are kept in percent, as published.
pub fn parse_csv(text: &str) -> Result<Vec<ParYieldRow>, RustyQLibError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .trim(csv::Trim::All)
        .from_reader(text.as_bytes());
    let headers = reader
        .headers()
        .map_err(|e| RustyQLibError::ParseError(format!("invalid CSV header: {e}")))?
        .clone();
    let date_ok = headers
        .get(0)
        .is_some_and(|h| h.eq_ignore_ascii_case("date"));
    if !date_ok {
        return Err(RustyQLibError::ParseError(format!(
            "expected the first column to be `Date`, got {:?} — \
             this does not look like the Treasury par yield curve CSV",
            headers.get(0).unwrap_or("")
        )));
    }
    let tenors: Vec<Option<(u32, u32)>> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let tenor = parse_tenor_header(h);
            if i > 0 && tenor.is_none() {
                log::warn!("skipping unrecognized column `{h}` in the par yield CSV");
            }
            tenor
        })
        .collect();

    let mut rows = Vec::new();
    for (line, record) in reader.records().enumerate() {
        let record = record.map_err(|e| {
            RustyQLibError::ParseError(format!("invalid CSV row {}: {e}", line + 2))
        })?;
        let Some(date_field) = record.get(0).filter(|s| !s.is_empty()) else {
            continue;
        };
        let date = NaiveDate::parse_from_str(date_field, "%m/%d/%Y")
            .or_else(|_| NaiveDate::parse_from_str(date_field, "%Y-%m-%d"))
            .map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {}: `{date_field}` is not a MM/DD/YYYY or YYYY-MM-DD date",
                    line + 2
                ))
            })?;
        let mut points = Vec::new();
        for (i, cell) in record.iter().enumerate() {
            let Some(&Some((months, extra_days))) = tenors.get(i) else {
                continue;
            };
            if cell.is_empty() || cell.eq_ignore_ascii_case("n/a") {
                continue;
            }
            let yield_pct: f64 = cell.parse().map_err(|_| {
                RustyQLibError::ParseError(format!(
                    "row {} ({date}): `{cell}` in column `{}` is not a number",
                    line + 2,
                    &headers[i]
                ))
            })?;
            if !(yield_pct.is_finite()
                && yield_pct > PERCENT_BOUNDS.0
                && yield_pct < PERCENT_BOUNDS.1)
            {
                return Err(RustyQLibError::ParseError(format!(
                    "row {} ({date}): {yield_pct} in column `{}` is outside the plausible \
                     percent range — refusing to guess the feed's units",
                    line + 2,
                    &headers[i]
                )));
            }
            points.push(ParYieldPoint {
                label: headers[i].to_string(),
                months,
                extra_days,
                yield_pct,
            });
        }
        if !points.is_empty() {
            rows.push(ParYieldRow { date, points });
        }
    }
    Ok(rows)
}

/// Pick the row for `date`, or the latest published row when `date` is
/// `None`. A missing date reports the nearest earlier published date so
/// weekends and holidays fail with an actionable message.
pub fn select_row(
    rows: &[ParYieldRow],
    date: Option<NaiveDate>,
) -> Result<&ParYieldRow, RustyQLibError> {
    let latest = rows
        .iter()
        .max_by_key(|r| r.date)
        .ok_or_else(|| RustyQLibError::ParseError("the CSV contains no data rows".to_string()))?;
    let Some(date) = date else {
        return Ok(latest);
    };
    if let Some(row) = rows.iter().find(|r| r.date == date) {
        return Ok(row);
    }
    let nearest_earlier = rows.iter().filter(|r| r.date < date).max_by_key(|r| r.date);
    Err(match nearest_earlier {
        Some(row) => RustyQLibError::invalid_input(
            "date",
            format!(
                "no par yields published for {date} (weekend or holiday?); \
                 the nearest earlier published date is {}",
                row.date
            ),
        ),
        None => RustyQLibError::invalid_input(
            "date",
            format!(
                "no par yields published for {date}; this file starts at {}",
                rows.iter().map(|r| r.date).min().unwrap_or(latest.date)
            ),
        ),
    })
}

/// Render one day's curve as a plain document: the points exactly as
/// published (tenor label, percent yield) under a `metadata` block saying
/// what the curve is. No instruments, no conventions, no interpretation —
/// downstream consumers decide how to use the fitted curve.
pub fn to_document(row: &ParYieldRow) -> serde_json::Value {
    let points: Vec<serde_json::Value> = row
        .points
        .iter()
        .map(|p| {
            serde_json::json!({
                "tenor": p.label,
                "yield": p.yield_pct,
            })
        })
        .collect();
    serde_json::json!({
        "metadata": {
            "source": SOURCE,
            "curve_type": "par yield curve (fitted by the Treasury from on-the-run quotes)",
            "curve_date": row.date.to_string(),
            "unit": "percent",
            "quote_basis": "bond-equivalent yield, semiannual coupon convention",
        },
        "points": points,
    })
}

/// Download the par-yield-curve CSV for one calendar year (one GET via
/// [`http_get`](super::http_get); everything downstream — [`parse_csv`],
/// [`select_row`], [`to_document`] — is pure).
pub fn fetch_year_csv(year: i32) -> Result<String, RustyQLibError> {
    super::http_get(&csv_url(year))
}

/// Read a fetched par-yield curve document (the [`to_document`] output,
/// after any JSON/XML round trip) back into a [`ParYieldRow`]. Tenor
/// labels are re-parsed from the document; unrecognized labels are
/// skipped with a warning, and yields are bounds-checked exactly like
/// the CSV path.
pub fn row_from_document(value: &serde_json::Value) -> Result<ParYieldRow, RustyQLibError> {
    let date_field = value["metadata"]["curve_date"].as_str().ok_or_else(|| {
        RustyQLibError::ParseError(
            "no `metadata.curve_date` — this does not look like a fetched \
             Treasury par yield curve document"
                .to_string(),
        )
    })?;
    let date = NaiveDate::parse_from_str(date_field, "%Y-%m-%d").map_err(|_| {
        RustyQLibError::ParseError(format!("`{date_field}` is not a YYYY-MM-DD curve date"))
    })?;
    let entries = value["points"].as_array().ok_or_else(|| {
        RustyQLibError::ParseError("the curve document has no `points` array".to_string())
    })?;
    let mut points = Vec::with_capacity(entries.len());
    for entry in entries {
        let label = entry["tenor"].as_str().ok_or_else(|| {
            RustyQLibError::ParseError(format!("curve point without a `tenor`: {entry}"))
        })?;
        let Some((months, extra_days)) = parse_tenor_header(label) else {
            log::warn!("skipping unrecognized tenor `{label}` in the curve document");
            continue;
        };
        let yield_pct = entry["yield"].as_f64().ok_or_else(|| {
            RustyQLibError::ParseError(format!("tenor `{label}` has no numeric `yield`"))
        })?;
        if !(yield_pct.is_finite()
            && yield_pct > PERCENT_BOUNDS.0
            && yield_pct < PERCENT_BOUNDS.1)
        {
            return Err(RustyQLibError::ParseError(format!(
                "tenor `{label}`: {yield_pct} is outside the plausible percent range — \
                 refusing to guess the document's units"
            )));
        }
        points.push(ParYieldPoint {
            label: label.to_string(),
            months,
            extra_days,
            yield_pct,
        });
    }
    if points.is_empty() {
        return Err(RustyQLibError::ParseError(
            "the curve document has no usable points".to_string(),
        ));
    }
    Ok(ParYieldRow { date, points })
}

/// Bootstrap a discount curve from one day's fitted par yields — the
/// explicit interpretation step the pass-through documents deliberately
/// omit. Conventions:
///
/// - settlement is T+1 on the US government-bond calendar; maturities
///   are settlement advanced by the tenor with plain calendar
///   arithmetic (fractional-month tenors add their leftover days for
///   bills, and are rounded to whole months for bonds);
/// - **sub-year tenors** become [`BillQuote`]s: the published
///   bond-equivalent par yield is restated as the Act/360 discount rate
///   (simple Act/365 up to 182 days, one compounding period beyond,
///   inverting [`TreasuryBill::bond_equivalent_yield`]);
/// - **1Y+ tenors** become synthetic par bonds: coupon equal to the par
///   yield, clean price 100, dated at settlement so the par identity is
///   exact ([`BondQuote`] on [`FixedRateBond::us_treasury`]).
///
/// The result is an ordinary [`YieldCurve`] (continuous zeros,
/// log-linear discount factors, Act/365) that reprices every input
/// exactly.
pub fn bootstrap_par_yield_curve(row: &ParYieldRow) -> Result<YieldCurve, RustyQLibError> {
    let valuation = row.date;
    let settlement = Calendar::UsGovernmentBond.add_business_days(valuation, 1);
    let mut instruments: Vec<Box<dyn CurveInstrument>> = Vec::with_capacity(row.points.len());
    for point in &row.points {
        let months = settlement
            .checked_add_months(Months::new(point.months))
            .ok_or_else(|| {
                RustyQLibError::invalid_input(
                    "tenor",
                    format!("cannot advance {settlement} by `{}`", point.label),
                )
            })?;
        let yield_rate = point.yield_pct / 100.0;
        if point.months < 12 {
            let maturity = months
                .checked_add_days(Days::new(u64::from(point.extra_days)))
                .ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "tenor",
                        format!("cannot advance {settlement} by `{}`", point.label),
                    )
                })?;
            let n = (maturity - settlement).num_days() as f64;
            let price = if n <= 182.0 {
                100.0 / (1.0 + yield_rate * n / 365.0)
            } else {
                100.0 / ((1.0 + yield_rate / 2.0) * (1.0 + (n / 365.0 - 0.5) * yield_rate))
            };
            let bill = TreasuryBill::new(100.0, maturity)?;
            let discount_rate = bill.discount_rate_from_price(price, settlement)?;
            instruments.push(Box::new(BillQuote::new(bill, discount_rate, settlement)?));
        } else {
            // whole coupon periods keep the par identity exact
            let bond = FixedRateBond::us_treasury(100.0, yield_rate, settlement, months)?;
            instruments.push(Box::new(BondQuote::new(bond, 100.0, settlement)?));
        }
    }
    bootstrap_curve(&instruments, valuation, DayCountConvention::Act365)
}

/// [`row_from_document`] + [`bootstrap_par_yield_curve`] in one step,
/// returning the curve together with a metadata description of where it
/// came from (for embedding in whatever the curve is used to build).
pub fn bootstrap_from_document(
    value: &serde_json::Value,
) -> Result<(YieldCurve, serde_json::Value), RustyQLibError> {
    let row = row_from_document(value)?;
    let curve = bootstrap_par_yield_curve(&row)?;
    let description = serde_json::json!({
        "type": "bootstrapped US Treasury par yield curve",
        "curve_date": row.date.to_string(),
        "pillars": row.points.len(),
        "source": value["metadata"]["source"],
        "note": "sub-year par yields as bills (bond-equivalent restated on the Act/360 \
                 discount basis); 1Y+ as synthetic par bonds dated at T+1 settlement",
    });
    Ok((curve, description))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// Real feed shape: quoted fractional-month header included.
    const SAMPLE: &str = "\
Date,\"1 Mo\",\"1.5 Month\",\"2 Mo\",\"3 Mo\",\"4 Mo\",\"6 Mo\",\"1 Yr\",\"2 Yr\",\"3 Yr\",\"5 Yr\",\"7 Yr\",\"10 Yr\",\"20 Yr\",\"30 Yr\"
08/05/2026,3.77,3.79,3.84,3.89,3.91,3.98,4.03,4.18,4.24,4.33,4.47,4.63,5.18,5.17
08/04/2026,3.78,3.80,3.85,3.89,3.91,4.00,4.04,4.20,4.25,4.33,4.47,4.63,5.18,5.18
";

    #[test]
    fn tenor_headers_parse_by_name() {
        assert_eq!(parse_tenor_header("1 Mo"), Some((1, 0)));
        assert_eq!(parse_tenor_header("1.5 Month"), Some((1, 15)));
        assert_eq!(parse_tenor_header("6 Mo"), Some((6, 0)));
        assert_eq!(parse_tenor_header("1 Yr"), Some((12, 0)));
        assert_eq!(parse_tenor_header("30 Yr"), Some((360, 0)));
        assert_eq!(parse_tenor_header("Date"), None);
        assert_eq!(parse_tenor_header("BC_10YEAR"), None);
        assert_eq!(parse_tenor_header("0 Mo"), None);
        assert_eq!(parse_tenor_header("-1 Yr"), None);
    }

    #[test]
    fn parses_the_real_feed_shape() {
        let rows = parse_csv(SAMPLE).unwrap();
        assert_eq!(rows.len(), 2);
        let row = &rows[0];
        assert_eq!(row.date, d(2026, 8, 5));
        assert_eq!(row.points.len(), 14);
        // yields stay in percent, exactly as published
        assert_eq!(row.points[0].yield_pct, 3.77);
        let thirty = row.points.last().unwrap();
        assert_eq!(thirty.label, "30 Yr");
        assert_eq!(thirty.months, 360);
        assert_eq!(thirty.yield_pct, 5.17);
    }

    #[test]
    fn empty_na_cells_and_unknown_columns_are_skipped() {
        let text = "\
Date,\"1 Mo\",\"Mystery\",\"10 Yr\"
08/05/2026,3.77,9.99,4.63
08/04/2026,,extra,N/A
08/03/2026,3.79,,4.70
";
        let rows = parse_csv(text).unwrap();
        // the all-empty row disappears, the Mystery column never parses
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].points.len(), 2);
        assert!(rows[0].points.iter().all(|p| p.label != "Mystery"));
        assert_eq!(rows[1].points.len(), 2);
    }

    #[test]
    fn implausible_units_and_alien_headers_are_rejected() {
        // 463 "percent" — the feed would have changed units
        let bad_units = "Date,\"10 Yr\"\n08/05/2026,463\n";
        assert!(parse_csv(bad_units).is_err());
        // first column is not Date: not this feed
        let alien = "NEW_DATE,\"10 Yr\"\n08/05/2026,4.63\n";
        assert!(parse_csv(alien).is_err());
        // garbage cell in a recognized column
        let garbage = "Date,\"10 Yr\"\n08/05/2026,four\n";
        assert!(parse_csv(garbage).is_err());
    }

    #[test]
    fn select_row_picks_latest_or_exact_and_reports_gaps() {
        let rows = parse_csv(SAMPLE).unwrap();
        // no date: the latest row regardless of file order
        assert_eq!(select_row(&rows, None).unwrap().date, d(2026, 8, 5));
        assert_eq!(
            select_row(&rows, Some(d(2026, 8, 4))).unwrap().date,
            d(2026, 8, 4)
        );
        // a gap names the nearest earlier published date
        let err = select_row(&rows, Some(d(2026, 8, 9)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("2026-08-05"), "{err}");
        assert!(select_row(&rows, Some(d(2026, 1, 1))).is_err());
        assert!(select_row(&[], None).is_err());
    }

    #[test]
    fn document_carries_the_feed_verbatim_plus_metadata() {
        let rows = parse_csv(SAMPLE).unwrap();
        let doc = to_document(&rows[0]);
        assert_eq!(doc["metadata"]["curve_date"], "2026-08-05");
        assert_eq!(doc["metadata"]["unit"], "percent");
        assert!(doc["metadata"]["source"]
            .as_str()
            .unwrap()
            .contains("Treasury"));
        let points = doc["points"].as_array().unwrap();
        assert_eq!(points.len(), 14);
        assert_eq!(points[0]["tenor"], "1 Mo");
        assert_eq!(points[0]["yield"], 3.77);
        assert_eq!(points[13]["tenor"], "30 Yr");
        // no float artifacts: the value is the feed's own number
        assert_eq!(points[13]["yield"], 5.17);
        // nothing invented: a point is exactly {tenor, yield}
        assert_eq!(points[0].as_object().unwrap().len(), 2);
    }

    #[test]
    fn curve_documents_round_trip_into_rows() {
        let rows = parse_csv(SAMPLE).unwrap();
        let document = to_document(&rows[0]);
        let back = row_from_document(&document).unwrap();
        assert_eq!(back, rows[0], "labels re-parse to the same tenors");
        // malformed documents are rejected with context
        assert!(row_from_document(&serde_json::json!({})).is_err());
        assert!(row_from_document(&serde_json::json!({
            "metadata": {"curve_date": "2026-08-05"}, "points": []
        }))
        .is_err());
        assert!(row_from_document(&serde_json::json!({
            "metadata": {"curve_date": "2026-08-05"},
            "points": [{"tenor": "10 Yr", "yield": 463.0}]
        }))
        .is_err());
    }

    #[test]
    fn bootstrapped_par_curve_reprices_its_inputs() {
        let rows = parse_csv(SAMPLE).unwrap();
        let curve = bootstrap_par_yield_curve(&rows[0]).unwrap();
        assert_eq!(curve.reference_date(), d(2026, 8, 5));
        assert_eq!(curve.pillars().len(), 14);
        let dfs: Vec<f64> = curve.pillars().iter().map(|p| p.df).collect();
        assert!(dfs.windows(2).all(|w| w[1] < w[0]), "{dfs:?}");
        // zeros stay in the vicinity of the 3.8-5.2% par inputs
        for pillar in curve.pillars() {
            assert!(
                pillar.zero_rate > 0.03 && pillar.zero_rate < 0.06,
                "zero {} at t={}",
                pillar.zero_rate,
                pillar.time
            );
        }
        // the curve must reprice a synthetic par bond at exactly 100
        // (10 Yr published at 4.63: Aug 5 is a Wednesday, T+1 Thursday)
        let settlement = d(2026, 8, 6);
        let ten_year =
            FixedRateBond::us_treasury(100.0, 0.0463, settlement, d(2036, 8, 6)).unwrap();
        let clean = ten_year.clean_price_from_curve(&curve, settlement).unwrap();
        assert!((clean - 100.0).abs() < 1e-6, "10Y par bond repriced at {clean}");
        // and the short end reprices the bill implied by the 3M BEY
        let bill = TreasuryBill::new(100.0, d(2026, 11, 6)).unwrap();
        let n = (d(2026, 11, 6) - settlement).num_days() as f64;
        let price = 100.0 / (1.0 + 0.0389 * n / 365.0);
        let repriced = 100.0 * curve.df_date(d(2026, 11, 6)) / curve.df_date(settlement);
        assert!(
            (repriced - price).abs() < 1e-7,
            "3M bill: {repriced} vs {price}"
        );
        let _ = bill;
    }

    /// Live check that the endpoint and schema still exist. Excluded from
    /// normal runs: `cargo test --features fetch -- --ignored` to run it.
    #[test]
    #[ignore = "hits home.treasury.gov"]
    fn live_feed_still_parses() {
        use chrono::Datelike;
        let year = chrono::Local::now().date_naive().year();
        let text = fetch_year_csv(year).expect("fetch failed");
        let rows = parse_csv(&text).expect("parse failed");
        let row = select_row(&rows, None).expect("no rows");
        assert!(row.points.len() >= 10, "only {} points", row.points.len());
        to_document(row);
    }
}
