//! Federal Reserve Bank of New York reference rates
//! (markets.newyorkfed.org): SOFR and EFFR.
//!
//! The NY Fed publishes each rate as one observation per business day —
//! the rate in percent with its distribution percentiles and underlying
//! volume (EFFR also carries the FOMC target range). The API is
//! official, free, keyless, and returns JSON.
//!
//! Nothing here interprets the data: each `refRates` record is passed
//! through verbatim, and [`to_document`] wraps the selected observation
//! in a `metadata` block recording what the rate is and where it came
//! from. [`parse_response`], [`select_observation`] and [`to_document`]
//! are pure functions, so everything after the download is testable
//! offline.

use chrono::NaiveDate;

use crate::core::errors::RustyQLibError;

/// Human-readable name of the source, recorded in document metadata.
pub const SOURCE: &str =
    "Federal Reserve Bank of New York reference rates (markets.newyorkfed.org)";

/// A reference rate the NY Fed publishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceRate {
    /// Secured Overnight Financing Rate.
    Sofr,
    /// Effective Federal Funds Rate.
    Effr,
}

impl ReferenceRate {
    /// The rate's name as the feed spells it (`"SOFR"`, `"EFFR"`).
    pub fn name(self) -> &'static str {
        match self {
            ReferenceRate::Sofr => "SOFR",
            ReferenceRate::Effr => "EFFR",
        }
    }

    /// API path segment: SOFR lives under `secured`, EFFR under
    /// `unsecured`.
    fn path(self) -> &'static str {
        match self {
            ReferenceRate::Sofr => "secured/sofr",
            ReferenceRate::Effr => "unsecured/effr",
        }
    }

    fn describe(self) -> &'static str {
        match self {
            ReferenceRate::Sofr => {
                "Secured Overnight Financing Rate: overnight Treasury repo, volume-weighted median"
            }
            ReferenceRate::Effr => {
                "Effective Federal Funds Rate: overnight unsecured fed funds, volume-weighted median"
            }
        }
    }
}

/// URL returning the last `n` published observations.
pub fn last_url(rate: ReferenceRate, n: u32) -> String {
    format!(
        "https://markets.newyorkfed.org/api/rates/{}/last/{n}.json",
        rate.path()
    )
}

/// URL returning the observations in `[start, end]`.
pub fn search_url(rate: ReferenceRate, start: NaiveDate, end: NaiveDate) -> String {
    format!(
        "https://markets.newyorkfed.org/api/rates/{}/search.json?startDate={start}&endDate={end}",
        rate.path()
    )
}

/// One published observation: the effective date, and the feed's record
/// exactly as it arrived (rate, percentiles, volume, target range, ...).
#[derive(Debug, Clone, PartialEq)]
pub struct RateObservation {
    pub date: NaiveDate,
    /// The `refRates` entry verbatim.
    pub record: serde_json::Value,
}

/// Reject percent rates outside this band: a value above 50 almost
/// certainly means the feed changed units and must not pass through.
const PERCENT_BOUNDS: (f64, f64) = (-5.0, 50.0);

/// Parse an API response (a `{"refRates": [...]}` document) into
/// observations, validating only what selection and sanity require:
/// each record must carry a parseable `effectiveDate` and a plausible
/// `percentRate`. Records are otherwise kept verbatim.
pub fn parse_response(text: &str) -> Result<Vec<RateObservation>, RustyQLibError> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| RustyQLibError::ParseError(format!("invalid JSON: {e}")))?;
    let records = value
        .get("refRates")
        .and_then(|r| r.as_array())
        .ok_or_else(|| {
            RustyQLibError::ParseError(
                "no `refRates` array — this does not look like a NY Fed rates response".to_string(),
            )
        })?;
    let mut observations = Vec::with_capacity(records.len());
    for record in records {
        let date_field = record
            .get("effectiveDate")
            .and_then(|d| d.as_str())
            .ok_or_else(|| {
                RustyQLibError::ParseError(format!(
                    "a refRates record has no `effectiveDate`: {record}"
                ))
            })?;
        let date = NaiveDate::parse_from_str(date_field, "%Y-%m-%d").map_err(|_| {
            RustyQLibError::ParseError(format!("`{date_field}` is not a YYYY-MM-DD effective date"))
        })?;
        let rate = record
            .get("percentRate")
            .and_then(|r| r.as_f64())
            .ok_or_else(|| {
                RustyQLibError::ParseError(format!(
                    "the {date} record has no numeric `percentRate`"
                ))
            })?;
        if !(rate > PERCENT_BOUNDS.0 && rate < PERCENT_BOUNDS.1) {
            return Err(RustyQLibError::ParseError(format!(
                "{date}: percentRate {rate} is outside the plausible percent range — \
                 refusing to guess the feed's units"
            )));
        }
        observations.push(RateObservation {
            date,
            record: record.clone(),
        });
    }
    Ok(observations)
}

/// Pick the observation for `date`, or the latest published one when
/// `date` is `None`. A missing date reports the nearest earlier
/// published date so weekends and holidays fail with an actionable
/// message.
pub fn select_observation(
    observations: &[RateObservation],
    date: Option<NaiveDate>,
) -> Result<&RateObservation, RustyQLibError> {
    let latest = observations.iter().max_by_key(|o| o.date).ok_or_else(|| {
        RustyQLibError::ParseError("the response contains no observations".to_string())
    })?;
    let Some(date) = date else {
        return Ok(latest);
    };
    if let Some(observation) = observations.iter().find(|o| o.date == date) {
        return Ok(observation);
    }
    let nearest_earlier = observations
        .iter()
        .filter(|o| o.date < date)
        .max_by_key(|o| o.date);
    Err(match nearest_earlier {
        Some(observation) => RustyQLibError::invalid_input(
            "date",
            format!(
                "no rate published for {date} (weekend or holiday?); \
                 the nearest earlier published date is {}",
                observation.date
            ),
        ),
        None => RustyQLibError::invalid_input(
            "date",
            format!("no rate published for {date} in the fetched window"),
        ),
    })
}

/// Render one observation as a plain document: the feed's record
/// verbatim under a `metadata` block saying what the rate is.
pub fn to_document(observation: &RateObservation, rate: ReferenceRate) -> serde_json::Value {
    serde_json::json!({
        "metadata": {
            "source": SOURCE,
            "rate_type": rate.name(),
            "description": rate.describe(),
            "effective_date": observation.date.to_string(),
            "unit": "percent",
        },
        "rate": observation.record,
    })
}

/// Download the last `n` published observations.
pub fn fetch_last(rate: ReferenceRate, n: u32) -> Result<String, RustyQLibError> {
    super::http_get(&last_url(rate, n))
}

/// Download the observations in `[start, end]`.
pub fn fetch_range(
    rate: ReferenceRate,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<String, RustyQLibError> {
    super::http_get(&search_url(rate, start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// Real response shape, including EFFR's target-range fields.
    const SAMPLE: &str = r#"{ "refRates": [
        { "effectiveDate": "2026-08-05", "type": "EFFR", "percentRate": 3.63,
          "percentPercentile1": 3.60, "percentPercentile25": 3.62,
          "percentPercentile75": 3.63, "percentPercentile99": 3.65,
          "targetRateFrom": 3.50, "targetRateTo": 3.75,
          "volumeInBillions": 114, "revisionIndicator": "" },
        { "effectiveDate": "2026-08-04", "type": "EFFR", "percentRate": 3.63,
          "percentPercentile1": 3.60, "percentPercentile25": 3.62,
          "percentPercentile75": 3.63, "percentPercentile99": 3.68,
          "targetRateFrom": 3.50, "targetRateTo": 3.75,
          "volumeInBillions": 117, "revisionIndicator": "" }
    ] }"#;

    #[test]
    fn parses_the_real_response_shape() {
        let observations = parse_response(SAMPLE).unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].date, d(2026, 8, 5));
        // the record is verbatim: percent value, target range, volume
        assert_eq!(observations[0].record["percentRate"], 3.63);
        assert_eq!(observations[0].record["targetRateTo"], 3.75);
        assert_eq!(observations[0].record["volumeInBillions"], 114);
    }

    #[test]
    fn malformed_responses_are_rejected() {
        assert!(parse_response("not json").is_err());
        assert!(parse_response(r#"{"rates": []}"#).is_err());
        // record without a date, record without a rate
        assert!(parse_response(r#"{"refRates": [{"percentRate": 3.6}]}"#).is_err());
        assert!(parse_response(r#"{"refRates": [{"effectiveDate": "2026-08-05"}]}"#).is_err());
        // 363 "percent" — the feed would have changed units
        assert!(parse_response(
            r#"{"refRates": [{"effectiveDate": "2026-08-05", "percentRate": 363}]}"#
        )
        .is_err());
    }

    #[test]
    fn select_observation_picks_latest_or_exact_and_reports_gaps() {
        let observations = parse_response(SAMPLE).unwrap();
        assert_eq!(
            select_observation(&observations, None).unwrap().date,
            d(2026, 8, 5)
        );
        assert_eq!(
            select_observation(&observations, Some(d(2026, 8, 4)))
                .unwrap()
                .date,
            d(2026, 8, 4)
        );
        let err = select_observation(&observations, Some(d(2026, 8, 9)))
            .unwrap_err()
            .to_string();
        assert!(err.contains("2026-08-05"), "{err}");
        assert!(select_observation(&observations, Some(d(2026, 1, 1))).is_err());
        assert!(select_observation(&[], None).is_err());
    }

    #[test]
    fn document_carries_the_record_verbatim_plus_metadata() {
        let observations = parse_response(SAMPLE).unwrap();
        let doc = to_document(&observations[0], ReferenceRate::Effr);
        assert_eq!(doc["metadata"]["rate_type"], "EFFR");
        assert_eq!(doc["metadata"]["effective_date"], "2026-08-05");
        assert_eq!(doc["metadata"]["unit"], "percent");
        assert!(doc["metadata"]["source"]
            .as_str()
            .unwrap()
            .contains("New York"));
        // nothing invented, nothing dropped: the record as the feed sent it
        assert_eq!(doc["rate"], observations[0].record);
    }

    #[test]
    fn urls_cover_both_rates() {
        assert_eq!(
            last_url(ReferenceRate::Sofr, 5),
            "https://markets.newyorkfed.org/api/rates/secured/sofr/last/5.json"
        );
        assert_eq!(
            search_url(ReferenceRate::Effr, d(2026, 7, 27), d(2026, 8, 3)),
            "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json\
             ?startDate=2026-07-27&endDate=2026-08-03"
        );
    }

    /// Live check that the endpoint and schema still exist. Excluded from
    /// normal runs: `cargo test --features fetch -- --ignored` to run it.
    #[test]
    #[ignore = "hits markets.newyorkfed.org"]
    fn live_feed_still_parses() {
        for rate in [ReferenceRate::Sofr, ReferenceRate::Effr] {
            let text = fetch_last(rate, 2).expect("fetch failed");
            let observations = parse_response(&text).expect("parse failed");
            let latest = select_observation(&observations, None).expect("no observations");
            assert_eq!(latest.record["type"], rate.name());
            to_document(latest, rate);
        }
    }
}
