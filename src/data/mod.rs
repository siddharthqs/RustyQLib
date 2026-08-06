//! Free official end-of-day market data (feature `fetch`).
//!
//! This module is the library's only bridge to the network, and it is a
//! deliberately thin one: each source is split into a *fetch* function
//! (one HTTP GET returning the raw document) and pure *parse* functions.
//! The data is passed through as published — labeled with provenance
//! metadata, never reinterpreted — and pricing never touches the network,
//! so every downstream computation stays reproducible from a file.

pub mod nyfed;
pub mod treasury;

use crate::core::errors::RustyQLibError;

/// One HTTP GET with a 30-second timeout and a real User-Agent, no
/// retries. Every fetch in this module goes through here — it is the
/// library's entire network surface.
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
    response
        .into_string()
        .map_err(|e| RustyQLibError::Network(format!("could not read the response body: {e}")))
}
