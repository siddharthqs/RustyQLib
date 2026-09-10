//! Free official end-of-day market data (feature `fetch`).
//!
//! This module is the library's only bridge to the network, and it is a
//! deliberately thin one: each source is split into a *fetch* function
//! (one HTTP GET returning the raw document) and pure *parse* functions.
//! The data is passed through as published — labeled with provenance
//! metadata, never reinterpreted — and pricing never touches the network,
//! so every downstream computation stays reproducible from a file.

pub mod cboe;
pub mod dtcc;
pub mod nyfed;
pub mod treasury;

use std::io::Read;

use chrono::NaiveDate;

use crate::core::errors::RustyQLibError;

/// Largest response body this module will read into memory. A full
/// listed option chain is the biggest document any source here serves
/// and runs to a few MiB; 64 MiB leaves headroom for the widest index
/// chain while still bounding a runaway or hostile response.
///
/// `ureq`'s own `into_string` caps at 10 MiB and reports the truncation
/// as an opaque I/O error, so bodies are read through the reader with
/// this explicit limit and a message that names it.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// One HTTP GET with a 30-second timeout and a real User-Agent, no
/// retries. Every fetch in this module goes through here — it is the
/// library's entire network surface. The body is read up to
/// [`MAX_BODY_BYTES`] and must be valid UTF-8.
pub(crate) fn http_get(url: &str) -> Result<String, RustyQLibError> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(30))
        .user_agent(concat!(
            "rustyqlib/",
            env!("CARGO_PKG_VERSION"),
            " (+https://github.com/siddharthqs/RustyQLib)"
        ))
        .build();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| RustyQLibError::Network(format!("GET {url} failed: {e}")))?;
    // read one byte past the limit: a full buffer then means "too large"
    let mut body = Vec::new();
    response
        .into_reader()
        .take(MAX_BODY_BYTES as u64 + 1)
        .read_to_end(&mut body)
        .map_err(|e| {
            RustyQLibError::Network(format!("could not read the response body of {url}: {e}"))
        })?;
    if body.len() > MAX_BODY_BYTES {
        return Err(RustyQLibError::Network(format!(
            "the response body of {url} exceeds the {MAX_BODY_BYTES}-byte limit \
             (MAX_BODY_BYTES); refusing to buffer it"
        )));
    }
    String::from_utf8(body).map_err(|e| {
        RustyQLibError::ParseError(format!(
            "the response body of {url} is not valid UTF-8: {e}"
        ))
    })
}

/// Pick the item dated `date`, or the latest one when `date` is `None`.
/// A missing date reports the nearest earlier dated item, so weekends
/// and holidays fail with an actionable message. `what` names the thing
/// being selected in those messages (e.g. `"par yields"`), and
/// `date_of` reads an item's date.
///
/// Shared by the Treasury curve rows and the NY Fed rate observations —
/// same selection rule, one implementation.
pub(crate) fn select_dated<'a, T>(
    items: &'a [T],
    date: Option<NaiveDate>,
    date_of: impl Fn(&T) -> NaiveDate,
    what: &str,
) -> Result<&'a T, RustyQLibError> {
    let latest = items
        .iter()
        .max_by_key(|item| date_of(item))
        .ok_or_else(|| RustyQLibError::ParseError(format!("no {what} in the response")))?;
    let Some(date) = date else {
        return Ok(latest);
    };
    if let Some(item) = items.iter().find(|item| date_of(item) == date) {
        return Ok(item);
    }
    let nearest_earlier = items
        .iter()
        .filter(|item| date_of(item) < date)
        .max_by_key(|item| date_of(item));
    Err(RustyQLibError::invalid_input(
        "date",
        match nearest_earlier {
            Some(item) => format!(
                "no {what} published for {date} (weekend or holiday?); \
                 the nearest earlier published date is {}",
                date_of(item)
            ),
            None => format!(
                "no {what} published for {date}; this response starts at {}",
                items
                    .iter()
                    .map(&date_of)
                    .min()
                    .unwrap_or_else(|| date_of(latest))
            ),
        },
    ))
}

/// Reject a percent-quoted rate outside the plausible band: a value
/// above 50 (or below -5) almost certainly means the feed changed units
/// and must not pass through. `what` names the value in the message.
pub(crate) fn plausible_percent(value: f64, what: &str) -> Result<(), RustyQLibError> {
    const PERCENT_BOUNDS: (f64, f64) = (-5.0, 50.0);
    if !(value.is_finite() && value > PERCENT_BOUNDS.0 && value < PERCENT_BOUNDS.1) {
        return Err(RustyQLibError::ParseError(format!(
            "{what}: {value} is outside the plausible percent range — \
             refusing to guess the feed's units"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn select_dated_picks_latest_exact_or_names_the_nearest() {
        let items = [d(2026, 8, 4), d(2026, 8, 5), d(2026, 7, 31)];
        let by = |x: &NaiveDate| *x;
        // no date: the latest regardless of order
        assert_eq!(
            *select_dated(&items, None, by, "rows").unwrap(),
            d(2026, 8, 5)
        );
        assert_eq!(
            *select_dated(&items, Some(d(2026, 8, 4)), by, "rows").unwrap(),
            d(2026, 8, 4)
        );
        let err = select_dated(&items, Some(d(2026, 8, 9)), by, "rows")
            .unwrap_err()
            .to_string();
        assert!(err.contains("2026-08-05") && err.contains("rows"), "{err}");
        // before the window: names where the response starts
        let err = select_dated(&items, Some(d(2026, 1, 1)), by, "rows")
            .unwrap_err()
            .to_string();
        assert!(err.contains("2026-07-31"), "{err}");
        assert!(select_dated::<NaiveDate>(&[], None, by, "rows").is_err());
    }

    #[test]
    fn plausible_percent_bounds_the_units() {
        assert!(plausible_percent(3.77, "1 Mo").is_ok());
        assert!(plausible_percent(-1.0, "1 Mo").is_ok(), "negative rates");
        for bad in [363.0, -5.0, 50.0, f64::NAN, f64::INFINITY] {
            let err = plausible_percent(bad, "10 Yr").unwrap_err().to_string();
            assert!(err.contains("units") && err.contains("10 Yr"), "{err}");
        }
    }
}
