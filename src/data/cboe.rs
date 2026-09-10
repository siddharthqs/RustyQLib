//! Cboe delayed option quotes (cdn.cboe.com).
//!
//! Cboe publishes the full listed chain per underlying as free, keyless
//! JSON, delayed 15 minutes: bid/ask, volume, open interest and the
//! underlying's current price, with each contract identified by its OCC
//! symbol (`AAPL261218C00310000`). Exchange data on personal-use terms —
//! provenance is recorded in the emitted metadata, and redistribution is
//! the user's own question.
//!
//! Two pure functions sit behind the one [`fetch`]: [`to_document`]
//! wraps the response verbatim under a `metadata` block (boundary 1 —
//! nothing reinterpreted), and [`to_chain`] normalizes it into the
//! unified [`OptionChain`] (boundary 2) that
//! [`implied_vol_surface_from_chain`](crate::equity::option_chain::implied_vol_surface_from_chain)
//! consumes.

use chrono::NaiveDate;

use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::equity::option_chain::{OptionChain, OptionQuote};

/// Human-readable name of the source, recorded in document metadata.
pub const SOURCE: &str = "Cboe delayed quotes (cdn.cboe.com), 15-minute delayed";

/// URL of the delayed-quotes chain for one underlying. Index symbols
/// keep their underscore prefix (`_SPX`).
pub fn url(symbol: &str) -> String {
    format!(
        "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json",
        symbol.trim().to_ascii_uppercase()
    )
}

/// Basic symbol sanity: letters, digits and the few characters listed
/// tickers actually use. Refuses anything that could mangle the URL.
fn validate_symbol(symbol: &str) -> Result<(), RustyQLibError> {
    let s = symbol.trim();
    let ok = !s.is_empty()
        && s.len() <= 12
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '^'));
    if ok {
        Ok(())
    } else {
        Err(RustyQLibError::invalid_input(
            "symbol",
            format!("`{symbol}` is not a plausible ticker"),
        ))
    }
}

/// Parse an OCC option symbol — root, `YYMMDD`, `C`/`P`, strike in
/// thousandths of a dollar (`AAPL261218C00310000` = strike 310) — into
/// `(root, expiry, right, strike)`.
pub fn parse_occ(symbol: &str) -> Option<(&str, NaiveDate, PutOrCall, f64)> {
    if symbol.len() < 16 || !symbol.is_ascii() {
        return None;
    }
    let (head, strike_digits) = symbol.split_at(symbol.len() - 8);
    let (head, right_char) = head.split_at(head.len() - 1);
    let (root, date_digits) = head.split_at(head.len() - 6);
    if root.is_empty() || !strike_digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let right = match right_char {
        "C" => PutOrCall::Call,
        "P" => PutOrCall::Put,
        _ => return None,
    };
    let expiry = NaiveDate::parse_from_str(date_digits, "%y%m%d").ok()?;
    let strike = strike_digits.parse::<u64>().ok()? as f64 / 1000.0;
    if strike <= 0.0 {
        return None;
    }
    Some((root, expiry, right, strike))
}

/// The response's `data.options` array — the one shape check every
/// entry point shares, owning both the error and the borrow so no
/// caller re-checks and then unwraps.
fn options_of(value: &serde_json::Value) -> Result<&Vec<serde_json::Value>, RustyQLibError> {
    value["data"]["options"].as_array().ok_or_else(|| {
        RustyQLibError::ParseError(
            "no `data.options` array — this does not look like a Cboe delayed-quotes response"
                .to_string(),
        )
    })
}

fn parse_response(text: &str) -> Result<serde_json::Value, RustyQLibError> {
    let value: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| RustyQLibError::ParseError(format!("invalid JSON: {e}")))?;
    options_of(&value)?;
    Ok(value)
}

/// Wrap a raw Cboe response verbatim under a `metadata` block.
pub fn to_document(text: &str, symbol: &str) -> Result<serde_json::Value, RustyQLibError> {
    let response = parse_response(text)?;
    Ok(serde_json::json!({
        "metadata": {
            "source": SOURCE,
            "symbol": symbol.trim().to_ascii_uppercase(),
            "note": "response passed through verbatim; quotes are 15-minute delayed exchange data",
        },
        "response": response,
    }))
}

/// Normalize a raw Cboe response into the unified [`OptionChain`].
///
/// Verbatim where it matters: bids and asks are taken as sent (a `0.0`
/// bid stays `Some(0.0)` — the cleaning rules, not the converter, decide
/// what is usable). The snapshot date comes from the response's own
/// timestamp; records whose OCC symbol does not parse are skipped and
/// counted in the chain metadata.
pub fn to_chain(text: &str) -> Result<OptionChain, RustyQLibError> {
    chain_from_value(&parse_response(text)?)
}

/// [`to_chain`] for an already-parsed response value (the CLI `build`
/// command arrives here from either JSON or XML documents).
pub fn chain_from_value(value: &serde_json::Value) -> Result<OptionChain, RustyQLibError> {
    let records = options_of(value)?;
    let timestamp = value["timestamp"].as_str().unwrap_or_default().to_string();
    // `get` respects char boundaries: a multibyte timestamp from an
    // unexpected feed must fail as a parse error, not panic on a byte
    // slice through the middle of a code point
    let head = timestamp.get(..10).unwrap_or(&timestamp);
    let as_of = NaiveDate::parse_from_str(head, "%Y-%m-%d").map_err(|_| {
        RustyQLibError::ParseError(format!(
            "cannot read a snapshot date from timestamp `{timestamp}`"
        ))
    })?;
    let symbol = value["data"]["symbol"]
        .as_str()
        .or(value["symbol"].as_str())
        .unwrap_or_default()
        .to_string();
    let spot = value["data"]["current_price"].as_f64().filter(|p| *p > 0.0);

    let mut quotes = Vec::with_capacity(records.len());
    let mut unparsed = 0usize;
    for record in records {
        let Some((_, expiry, right, strike)) = record["option"].as_str().and_then(parse_occ) else {
            unparsed += 1;
            continue;
        };
        quotes.push(OptionQuote {
            expiry,
            strike,
            right,
            bid: record["bid"].as_f64(),
            ask: record["ask"].as_f64(),
            last: record["last_trade_price"].as_f64(),
            volume: record["volume"].as_f64(),
            open_interest: record["open_interest"].as_f64(),
        });
    }
    if unparsed > 0 {
        log::warn!("skipped {unparsed} Cboe records with unparseable OCC symbols");
    }
    Ok(OptionChain {
        symbol,
        as_of,
        timestamp: Some(timestamp),
        spot,
        quotes,
        metadata: Some(serde_json::json!({
            "source": SOURCE,
            "unparsed_records": unparsed,
        })),
    })
}

/// Download the delayed chain for one underlying (one GET via
/// [`http_get`](super::http_get); [`to_document`] and [`to_chain`] are
/// pure).
pub fn fetch(symbol: &str) -> Result<String, RustyQLibError> {
    validate_symbol(symbol)?;
    super::http_get(&url(symbol))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::equity::option_chain::{implied_vol_surface_from_chain, FilterConfig};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn fixture() -> String {
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/cboe_chain_sample.json"
        ))
        .expect("fixture exists")
    }

    #[test]
    fn occ_symbols_parse_by_position_from_the_right() {
        assert_eq!(
            parse_occ("AAPL261218C00310000"),
            Some(("AAPL", d(2026, 12, 18), PutOrCall::Call, 310.0))
        );
        assert_eq!(
            parse_occ("BRKB260918P00447500"),
            Some(("BRKB", d(2026, 9, 18), PutOrCall::Put, 447.5))
        );
        // 1-char root, fractional strike
        assert_eq!(
            parse_occ("F270115C00012500"),
            Some(("F", d(2027, 1, 15), PutOrCall::Call, 12.5))
        );
        assert_eq!(parse_occ("AAPL261218X00310000"), None);
        assert_eq!(parse_occ("261218C00310000"), None, "no root");
        assert_eq!(parse_occ("AAPL26121C0031000"), None, "too short");
        assert_eq!(parse_occ("AAPL269918C00310000"), None, "month 99");
    }

    #[test]
    fn real_response_normalizes_into_a_chain() {
        let chain = to_chain(&fixture()).unwrap();
        assert_eq!(chain.symbol, "AAPL");
        assert_eq!(chain.as_of, d(2026, 8, 8));
        assert_eq!(chain.quotes.len(), 56);
        assert_eq!(chain.spot, Some(313.15));
        let expiries: std::collections::BTreeSet<NaiveDate> =
            chain.quotes.iter().map(|q| q.expiry).collect();
        assert_eq!(
            expiries.into_iter().collect::<Vec<_>>(),
            vec![d(2026, 9, 18), d(2026, 12, 18)]
        );
        assert_eq!(chain.metadata.as_ref().unwrap()["unparsed_records"], 0);
    }

    #[test]
    fn real_chain_builds_a_sane_surface() {
        let chain = to_chain(&fixture()).unwrap();
        let curve = YieldCurve::flat(
            0.037,
            chain.as_of,
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let (surface, report) =
            implied_vol_surface_from_chain(&chain, &curve, &FilterConfig::default()).unwrap();
        assert_eq!(report.forwards.len(), 2);
        for (expiry, forward) in &report.forwards {
            // parity forward within a few percent of the delayed spot
            assert!(
                (forward / 313.15 - 1.0).abs() < 0.05,
                "{expiry}: forward {forward}"
            );
            let t = DayCountConvention::Act365.year_fraction(chain.as_of, *expiry);
            for strike in [280.0, 310.0, 340.0] {
                let vol = surface.vol(strike, *forward, t);
                assert!(
                    vol > 0.05 && vol < 1.0,
                    "{expiry} K={strike}: implausible vol {vol}"
                );
            }
        }
        assert!(report.quotes_used >= 20, "used {}", report.quotes_used);
    }

    #[test]
    fn verbatim_document_wraps_the_response() {
        let document = to_document(&fixture(), "aapl").unwrap();
        assert_eq!(document["metadata"]["symbol"], "AAPL");
        assert_eq!(
            document["response"]["data"]["options"]
                .as_array()
                .unwrap()
                .len(),
            56
        );
        assert!(to_document("{}", "AAPL").is_err());
    }

    /// A non-ASCII timestamp used to byte-slice through a code point and
    /// panic; it must come back as a parse error like any other garbage.
    #[test]
    fn odd_timestamps_are_parse_errors_not_panics() {
        let with_timestamp = |timestamp: serde_json::Value| {
            serde_json::json!({
                "timestamp": timestamp,
                "data": { "symbol": "AAPL", "options": [] },
            })
        };
        for timestamp in [
            // multibyte: the 10-byte prefix lands mid-character
            serde_json::json!("2026-08-0\u{4e2d}\u{6587}"),
            serde_json::json!("\u{1f4c8}\u{1f4c9}"),
            serde_json::json!("short"),
            serde_json::json!(""),
            serde_json::json!(20260808),
        ] {
            let err = chain_from_value(&with_timestamp(timestamp.clone()))
                .expect_err("must not parse")
                .to_string();
            assert!(err.contains("snapshot date"), "{timestamp}: {err}");
        }
        // exactly ten ASCII characters still parse
        let ok =
            chain_from_value(&with_timestamp(serde_json::json!("2026-08-08 14:30:00"))).unwrap();
        assert_eq!(ok.as_of, d(2026, 8, 8));
    }

    #[test]
    fn symbols_are_validated_before_touching_the_url() {
        assert!(validate_symbol("AAPL").is_ok());
        assert!(validate_symbol("_SPX").is_ok());
        assert!(validate_symbol("BRK.B").is_ok());
        assert!(validate_symbol("").is_err());
        assert!(validate_symbol("AAPL/../etc").is_err());
        assert!(validate_symbol("A B").is_err());
    }

    /// Live check that the endpoint and schema still exist. Excluded from
    /// normal runs: `cargo test --features fetch -- --ignored` to run it.
    #[test]
    #[ignore = "hits cdn.cboe.com"]
    fn live_feed_still_parses() {
        let text = fetch("AAPL").expect("fetch failed");
        let chain = to_chain(&text).expect("normalize failed");
        assert!(
            chain.quotes.len() > 100,
            "only {} quotes",
            chain.quotes.len()
        );
        assert!(chain.spot.is_some());
        to_document(&text, "AAPL").expect("document failed");
    }
}
