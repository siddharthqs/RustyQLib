//! Listed option chains: the unified market-quote structure every data
//! source normalizes into, and the pipeline that turns a chain into an
//! implied volatility surface.
//!
//! An [`OptionChain`] is a flat, possibly ragged list of quotes — real
//! chains have different strikes per expiry and missing sides, so there
//! is no grid here. Sources (a broker download, the CBOE delayed feed, a
//! reshaped CSV) each convert *into* this one type; everything
//! downstream is source-agnostic. The serde derives make the struct its
//! own JSON file format.
//!
//! [`implied_vol_surface_from_chain`] is the pipeline:
//!
//! 1. **Clean** ([`FilterConfig`]): mid prices from two-sided quotes, a
//!    relative-spread cap, expiry and moneyness windows, out-of-the-money
//!    quotes only per side (the liquid side, free of the American
//!    early-exercise premium that contaminates ITM call vols).
//! 2. **Imply forwards** from put-call parity per expiry —
//!    `F = K + (C - P) / df(T)` — as the median over the strike pairs
//!    nearest the money, discounting off the supplied [`YieldCurve`].
//!    No dividend assumptions: the chain itself says where the forward
//!    is. (Parity is exact for European options; for American equity
//!    options it is an excellent approximation near the money, which is
//!    exactly where it is sampled.)
//! 3. **Solve Black-76 implied vols** from the mids against each
//!    expiry's forward, and assemble per-expiry smiles into a
//!    [`VolSurface`] on absolute strikes.
//!
//! Every dropped quote is counted by reason in the
//! [`SurfaceBuildReport`], so a surface built from a noisy chain says
//! what it ignored and which forwards it used.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::curves::{Tenor, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::interpolation::interp_pairs;
use crate::core::trade::PutOrCall;
use crate::core::vols::{SurfaceDiagnostics, VolSurface};
use crate::equity::engines::blackscholes::implied_vol_from_price;

/// One side of the book for one listed option.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptionQuote {
    pub expiry: NaiveDate,
    pub strike: f64,
    pub right: PutOrCall,
    /// Best bid. `None` = not quoted (distinct from `Some(0.0)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bid: Option<f64>,
    /// Best ask. `None` = not quoted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ask: Option<f64>,
    /// Optional color, kept when the source provides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_interest: Option<f64>,
}

impl OptionQuote {
    /// Mid price when the quote is two-sided and sane (`bid > 0`,
    /// `ask >= bid`); `None` otherwise.
    pub fn mid(&self) -> Option<f64> {
        match (self.bid, self.ask) {
            (Some(bid), Some(ask)) if bid > 0.0 && ask >= bid => Some(0.5 * (bid + ask)),
            _ => None,
        }
    }
}

/// An option chain snapshot for one underlying, normalized from any
/// source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptionChain {
    pub symbol: String,
    /// The date the snapshot represents (valuation date for the surface).
    pub as_of: NaiveDate,
    /// Snapshot timestamp when the source provides one (RFC 3339-ish,
    /// kept verbatim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Underlying price at snapshot time, when the source provides it.
    /// Used to anchor the near-the-money strike selection and as a
    /// zero-dividend forward fallback for expiries without put-call
    /// pairs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot: Option<f64>,
    pub quotes: Vec<OptionQuote>,
    /// Provenance: source name, delayed/live, file/url, fetch time, ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl OptionChain {
    /// Serialize as pretty-printed JSON (the chain document format).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("a chain of plain values always serializes")
    }

    /// Parse a chain document written by [`Self::to_json`] (or any JSON
    /// matching the [`OptionChain`] schema).
    pub fn from_json(text: &str) -> Result<OptionChain, RustyQLibError> {
        serde_json::from_str(text)
            .map_err(|e| RustyQLibError::ParseError(format!("invalid option chain: {e}")))
    }
}

/// Cleaning rules for [`implied_vol_surface_from_chain`], with defaults
/// tuned for delayed retail data. Every rule's casualties are counted by
/// name in the [`SurfaceBuildReport`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Reject quotes with `(ask - bid) / mid` above this.
    pub max_relative_spread: f64,
    /// Keep strikes with `K/F` inside `[min_moneyness, max_moneyness]`.
    pub min_moneyness: f64,
    pub max_moneyness: f64,
    /// Skip expiries closer than this many calendar days.
    pub min_days_to_expiry: i64,
    /// Skip expiries beyond this many years (Act/365).
    pub max_years_to_expiry: f64,
    /// Drop an expiry that ends up with fewer usable vols than this.
    pub min_quotes_per_expiry: usize,
    /// Keep only out-of-the-money quotes per side (calls above the
    /// forward, puts below), merged into one smile. When `false` both
    /// sides are used and same-strike vols are averaged.
    pub otm_only: bool,
    /// How many nearest-the-money put-call pairs the forward median uses.
    pub forward_pairs: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        FilterConfig {
            max_relative_spread: 0.25,
            min_moneyness: 0.5,
            max_moneyness: 2.0,
            min_days_to_expiry: 7,
            max_years_to_expiry: 2.0,
            min_quotes_per_expiry: 3,
            otm_only: true,
            forward_pairs: 5,
        }
    }
}

/// Everything worth knowing about how a surface was built from a chain:
/// the forwards actually used, and where every dropped quote went.
#[derive(Clone, Debug, Default, Serialize)]
pub struct SurfaceBuildReport {
    /// Parity-implied forward per surviving expiry.
    pub forwards: Vec<(NaiveDate, f64)>,
    /// Quotes whose implied vols made it into the surface.
    pub quotes_used: usize,
    /// Dropped-quote counts keyed by the rule that dropped them.
    pub dropped: BTreeMap<String, usize>,
    /// Static-arbitrage findings on the built surface (butterfly and
    /// calendar violations at the quoted pillars) — reported, not
    /// enforced.
    pub diagnostics: SurfaceDiagnostics,
}

impl SurfaceBuildReport {
    fn drop_quotes(&mut self, reason: &str, count: usize) {
        if count > 0 {
            *self.dropped.entry(reason.to_string()).or_insert(0) += count;
        }
    }

    /// Total quotes dropped, over all reasons.
    pub fn quotes_dropped(&self) -> usize {
        self.dropped.values().sum()
    }

    /// This report as surface-document metadata (see
    /// [`VolSurface::to_document`]), including the chain's identity.
    pub fn to_metadata(&self, chain: &OptionChain) -> serde_json::Value {
        let forwards: BTreeMap<String, f64> = self
            .forwards
            .iter()
            .map(|(date, f)| (date.to_string(), *f))
            .collect();
        serde_json::json!({
            "symbol": chain.symbol,
            "as_of": chain.as_of.to_string(),
            "chain_source": chain.metadata,
            "forwards": forwards,
            "quotes_used": self.quotes_used,
            "quotes_dropped": self.dropped,
            "diagnostics": self.diagnostics.to_metadata(),
        })
    }
}

/// Solved vols outside this band are treated as data problems, not
/// market: the quote is dropped and counted, in the same
/// refuse-to-guess spirit as the market-data parsers.
const VOL_BOUNDS: (f64, f64) = (0.005, 5.0);

/// Build an implied volatility surface from an option chain (see the
/// module docs for the pipeline). `discount` supplies the discount
/// factors for parity forwards and de-discounting mids — a bootstrapped
/// Treasury curve or a [`YieldCurve::flat`] rate. Returns the surface on
/// absolute strikes together with the build report; attach the report to
/// the saved surface via
/// [`SurfaceBuildReport::to_metadata`] + [`VolSurface::to_document`].
pub fn implied_vol_surface_from_chain(
    chain: &OptionChain,
    discount: &YieldCurve,
    filter: &FilterConfig,
) -> Result<(VolSurface, SurfaceBuildReport), RustyQLibError> {
    if chain.quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "option chain",
            format!("chain for {} has no quotes", chain.symbol),
        ));
    }
    let mut report = SurfaceBuildReport::default();
    let day_count = DayCountConvention::Act365;

    // group usable quotes by expiry, applying the quote-level filters
    let mut by_expiry: BTreeMap<NaiveDate, Vec<(&OptionQuote, f64)>> = BTreeMap::new();
    for quote in &chain.quotes {
        let Some(mid) = quote.mid() else {
            report.drop_quotes("unquoted_or_crossed", 1);
            continue;
        };
        let spread = quote.ask.unwrap_or(mid) - quote.bid.unwrap_or(mid);
        if spread / mid > filter.max_relative_spread {
            report.drop_quotes("wide_spread", 1);
            continue;
        }
        let days = (quote.expiry - chain.as_of).num_days();
        let years = day_count.year_fraction(chain.as_of, quote.expiry);
        if days < filter.min_days_to_expiry || years > filter.max_years_to_expiry {
            report.drop_quotes("expiry_window", 1);
            continue;
        }
        by_expiry
            .entry(quote.expiry)
            .or_default()
            .push((quote, mid));
    }

    let mut tenors: Vec<Tenor> = Vec::new();
    let mut smiles: Vec<Vec<(f64, f64)>> = Vec::new();
    for (expiry, quotes) in by_expiry {
        let t = day_count.year_fraction(chain.as_of, expiry);
        let df = discount.df_date(expiry) / discount.df_date(chain.as_of);

        // parity pairs: strikes quoted usable on both sides
        let mut calls: BTreeMap<u64, f64> = BTreeMap::new();
        let mut puts: BTreeMap<u64, f64> = BTreeMap::new();
        for (quote, mid) in &quotes {
            let side = match quote.right {
                PutOrCall::Call => &mut calls,
                PutOrCall::Put => &mut puts,
            };
            side.insert(quote.strike.to_bits(), *mid);
        }
        let mut pair_forwards: Vec<(f64, f64)> = calls
            .iter()
            .filter_map(|(bits, call_mid)| {
                let put_mid = puts.get(bits)?;
                let strike = f64::from_bits(*bits);
                Some((strike, strike + (call_mid - put_mid) / df))
            })
            .collect();

        let forward = if pair_forwards.is_empty() {
            match chain.spot {
                // zero-dividend fallback: grow the spot at the curve
                Some(spot) => spot / df,
                None => {
                    report.drop_quotes("no_forward", quotes.len());
                    continue;
                }
            }
        } else {
            // median of the pairs nearest the money
            let anchor = chain.spot.unwrap_or_else(|| {
                let mut implied: Vec<f64> = pair_forwards.iter().map(|&(_, f)| f).collect();
                implied.sort_by(|a, b| a.partial_cmp(b).unwrap());
                implied[implied.len() / 2]
            });
            pair_forwards.sort_by(|a, b| {
                (a.0 - anchor)
                    .abs()
                    .partial_cmp(&(b.0 - anchor).abs())
                    .unwrap()
            });
            let mut nearest: Vec<f64> = pair_forwards
                .iter()
                .take(filter.forward_pairs.max(1))
                .map(|&(_, f)| f)
                .collect();
            nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
            nearest[nearest.len() / 2]
        };

        // moneyness window, OTM selection, and the vol solve
        let mut points: Vec<(f64, f64)> = Vec::new();
        for (quote, mid) in &quotes {
            let moneyness = quote.strike / forward;
            if moneyness < filter.min_moneyness || moneyness > filter.max_moneyness {
                report.drop_quotes("moneyness_window", 1);
                continue;
            }
            let otm = match quote.right {
                PutOrCall::Call => quote.strike >= forward,
                PutOrCall::Put => quote.strike <= forward,
            };
            if filter.otm_only && !otm {
                report.drop_quotes("in_the_money", 1);
                continue;
            }
            // Black-76: undiscounted mid against the forward
            match implied_vol_from_price(forward, quote.strike, 0.0, 0.0, t, mid / df, quote.right)
            {
                Ok(vol) if vol > VOL_BOUNDS.0 && vol < VOL_BOUNDS.1 => {
                    points.push((quote.strike, vol));
                }
                Ok(_) => report.drop_quotes("vol_out_of_bounds", 1),
                Err(e) => {
                    log::debug!(
                        "chain {} {expiry} K={}: implied vol solve failed: {e}",
                        chain.symbol,
                        quote.strike
                    );
                    report.drop_quotes("solver_failed", 1);
                }
            }
        }

        // merge same-strike duplicates (both sides usable) by averaging
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut merged: Vec<(f64, f64)> = Vec::with_capacity(points.len());
        for (strike, vol) in points {
            match merged.last_mut() {
                Some(last) if (last.0 - strike).abs() < 1e-9 => last.1 = 0.5 * (last.1 + vol),
                _ => merged.push((strike, vol)),
            }
        }

        if merged.len() < filter.min_quotes_per_expiry {
            report.drop_quotes("sparse_expiry", merged.len());
            continue;
        }
        report.quotes_used += merged.len();
        report.forwards.push((expiry, forward));
        tenors.push(Tenor::Date(expiry));
        smiles.push(merged);
    }

    if tenors.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "option chain",
            format!(
                "no expiry of {} survived cleaning (dropped: {:?})",
                chain.symbol, report.dropped
            ),
        ));
    }
    let surface = VolSurface::from_strike_smiles(&tenors, &smiles, chain.as_of, day_count)?;

    // arbitrage diagnostics at the quoted pillars, with the forward
    // curve the surface was built against (linear between pillars)
    let forward_points: Vec<(f64, f64)> = report
        .forwards
        .iter()
        .map(|(expiry, forward)| (day_count.year_fraction(chain.as_of, *expiry), *forward))
        .collect();
    report.diagnostics = surface.diagnostics(|t| match forward_points.len() {
        1 => forward_points[0].1,
        _ => interp_pairs(&forward_points, t),
    });
    if !report.diagnostics.is_clean() {
        log::warn!(
            "{}: the implied surface carries static arbitrage at the quotes \
             ({} butterfly, {} calendar violations — see the build report)",
            chain.symbol,
            report.diagnostics.butterfly.len(),
            report.diagnostics.calendar.len()
        );
    }
    Ok((surface, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::equity::engines::blackscholes::bs_price;

    const SPOT: f64 = 100.0;
    const RATE: f64 = 0.04;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn as_of() -> NaiveDate {
        d(2026, 8, 5)
    }

    fn curve() -> YieldCurve {
        YieldCurve::flat(
            RATE,
            as_of(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// The known skew the synthetic chain is generated from.
    fn true_vol(strike: f64, base: f64) -> f64 {
        base - 0.001 * (strike - 100.0)
    }

    /// Both sides quoted at a 2% relative spread around the exact
    /// Black-Scholes mid, strikes 70..130, two expiries.
    fn synthetic_chain() -> OptionChain {
        let mut quotes = Vec::new();
        for (expiry, base) in [(d(2027, 2, 5), 0.23), (d(2027, 8, 5), 0.25)] {
            let t = DayCountConvention::Act365.year_fraction(as_of(), expiry);
            for i in 0..13 {
                let strike = 70.0 + 5.0 * i as f64;
                let vol = true_vol(strike, base);
                for right in [PutOrCall::Call, PutOrCall::Put] {
                    let mid = bs_price(SPOT, strike, RATE, 0.0, vol, t, right);
                    quotes.push(OptionQuote {
                        expiry,
                        strike,
                        right,
                        bid: Some(mid * 0.99),
                        ask: Some(mid * 1.01),
                        last: None,
                        volume: None,
                        open_interest: None,
                    });
                }
            }
        }
        OptionChain {
            symbol: "ACME".to_string(),
            as_of: as_of(),
            timestamp: None,
            spot: Some(SPOT),
            quotes,
            metadata: Some(serde_json::json!({"source": "synthetic"})),
        }
    }

    #[test]
    fn parity_forwards_are_recovered_exactly() {
        let (_, report) =
            implied_vol_surface_from_chain(&synthetic_chain(), &curve(), &FilterConfig::default())
                .unwrap();
        assert_eq!(report.forwards.len(), 2);
        for (expiry, forward) in &report.forwards {
            let t = DayCountConvention::Act365.year_fraction(as_of(), *expiry);
            let expected = SPOT * (RATE * t).exp();
            assert!(
                (forward - expected).abs() < 1e-9,
                "{expiry}: forward {forward} vs {expected}"
            );
        }
    }

    #[test]
    fn surface_recovers_the_generating_skew() {
        let chain = synthetic_chain();
        let (surface, report) =
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).unwrap();
        assert_eq!(surface.reference_date(), as_of());
        for (expiry, base) in [(d(2027, 2, 5), 0.23), (d(2027, 8, 5), 0.25)] {
            let t = DayCountConvention::Act365.year_fraction(as_of(), expiry);
            let forward = SPOT * (RATE * t).exp();
            for strike in [80.0, 90.0, 100.0, 110.0, 120.0] {
                let recovered = surface.vol(strike, forward, t);
                let expected = true_vol(strike, base);
                assert!(
                    (recovered - expected).abs() < 5e-4,
                    "T={t:.3} K={strike}: {recovered} vs {expected}"
                );
            }
        }
        // every strike contributes exactly one smile point per expiry
        assert_eq!(report.quotes_used, 26);
        assert!(report.dropped.contains_key("in_the_money"));
        // a chain generated from a smooth skew carries no static arbitrage
        assert!(
            report.diagnostics.is_clean(),
            "diagnostics: {:?}",
            report.diagnostics
        );
    }

    #[test]
    fn junk_quotes_are_dropped_and_counted() {
        let mut chain = synthetic_chain();
        let expiry = d(2027, 2, 5);
        let junk = [
            // one-sided, zero bid, crossed, wide
            (Some(0.0), Some(1.0)),
            (None, Some(1.0)),
            (Some(2.0), Some(1.0)),
            (Some(1.0), Some(2.0)),
        ];
        for (bid, ask) in junk {
            chain.quotes.push(OptionQuote {
                expiry,
                strike: 100.0,
                right: PutOrCall::Call,
                bid,
                ask,
                last: None,
                volume: None,
                open_interest: None,
            });
        }
        // far outside the moneyness window
        chain.quotes.push(OptionQuote {
            expiry,
            strike: 500.0,
            right: PutOrCall::Call,
            bid: Some(0.01),
            ask: Some(0.011),
            last: None,
            volume: None,
            open_interest: None,
        });
        let (_, report) =
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).unwrap();
        assert_eq!(report.dropped["unquoted_or_crossed"], 3);
        assert_eq!(report.dropped["wide_spread"], 1);
        assert_eq!(report.dropped["moneyness_window"], 1);
    }

    #[test]
    fn expiry_window_and_sparse_expiries_are_enforced() {
        let mut chain = synthetic_chain();
        // tomorrow: inside min_days_to_expiry
        chain.quotes.push(OptionQuote {
            expiry: d(2026, 8, 6),
            strike: 100.0,
            right: PutOrCall::Call,
            bid: Some(1.0),
            ask: Some(1.01),
            last: None,
            volume: None,
            open_interest: None,
        });
        // a lonely far expiry (OTM call): survives the window but is too sparse
        chain.quotes.push(OptionQuote {
            expiry: d(2028, 2, 5),
            strike: 120.0,
            right: PutOrCall::Call,
            bid: Some(6.0),
            ask: Some(6.1),
            last: None,
            volume: None,
            open_interest: None,
        });
        let (surface, report) =
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).unwrap();
        assert_eq!(
            surface.expiry_times().len(),
            2,
            "only the two real expiries"
        );
        assert_eq!(report.dropped["expiry_window"], 1);
        assert_eq!(report.dropped["sparse_expiry"], 1);
    }

    #[test]
    fn nothing_usable_is_an_error_not_a_panic() {
        let mut chain = synthetic_chain();
        for quote in &mut chain.quotes {
            quote.bid = None;
        }
        assert!(
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).is_err()
        );
        chain.quotes.clear();
        assert!(
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).is_err()
        );
    }

    #[test]
    fn chain_documents_round_trip_through_json() {
        let chain = synthetic_chain();
        let back = OptionChain::from_json(&chain.to_json()).unwrap();
        assert_eq!(back, chain);
        // the wire format spells sides as C/P and accepts long forms too
        let text = chain.to_json();
        assert!(text.contains("\"right\": \"C\""), "sides serialize as C/P");
        let lenient: OptionQuote = serde_json::from_str(
            r#"{"expiry": "2027-02-05", "strike": 100.0, "right": "put", "bid": 1.0, "ask": 1.1}"#,
        )
        .unwrap();
        assert_eq!(lenient.right, PutOrCall::Put);
        assert!(OptionChain::from_json("[]").is_err());
    }

    #[test]
    fn surface_document_metadata_carries_the_build_report() {
        let chain = synthetic_chain();
        let (surface, report) =
            implied_vol_surface_from_chain(&chain, &curve(), &FilterConfig::default()).unwrap();
        let document = surface.to_document(Some(report.to_metadata(&chain)));
        let text = serde_json::to_string_pretty(&document).unwrap();
        let parsed: crate::core::vols::VolSurfaceDocument = serde_json::from_str(&text).unwrap();
        assert!(parsed.build().is_ok(), "the surface itself still rebuilds");
        let meta = parsed.metadata.unwrap();
        assert_eq!(meta["symbol"], "ACME");
        assert_eq!(meta["quotes_used"], 26);
        assert_eq!(meta["chain_source"]["source"], "synthetic");
        assert_eq!(meta["diagnostics"]["butterfly_violations"], 0);
        assert_eq!(meta["diagnostics"]["calendar_violations"], 0);
    }
}
