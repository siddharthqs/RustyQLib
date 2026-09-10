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
//!
//! For American chains (US equity and ETF options), run
//! [`de_americanize_chain`] first: it strips each quote's
//! early-exercise premium (CRR tree at the quote's own implied vol,
//! carry implied from the chain's parity forwards — measured, never
//! assumed) so that parity holds as an identity on the corrected chain
//! and the forwards above lose their American bias. On index ETFs the
//! bias is sub-basis-point at the short end and a few tens of basis
//! points of forward at 1--1.5 years.

use std::collections::BTreeMap;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::curves::{Tenor, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::interpolation::interp_pairs;
use crate::core::trade::PutOrCall;
use crate::core::vols::{SurfaceDiagnostics, VolSurface};
use crate::equity::engines::blackscholes::{bs_price, implied_vol_from_price};

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

// ═══════════════════════════════════════════════════════════════════════
//  De-Americanization
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for [`de_americanize_chain`].
#[derive(Clone, Debug)]
pub struct DeAmericanizeConfig {
    /// CRR binomial steps for the early-exercise premium. The EEP is a
    /// *difference* of two prices with identical discretization error to
    /// leading order, so it converges much faster than either price.
    pub tree_steps: usize,
    /// Correction rounds (forward → implied carry → EEP → forward).
    /// Two suffice: the EEP barely depends on the residual carry error.
    pub rounds: usize,
    /// Quotes with `strike / spot` outside this window are left
    /// untouched — they fall outside any sensible smile window
    /// downstream, and deep wings are where the trees cost the most.
    pub strike_window: (f64, f64),
    /// Nearest-the-money put-call pairs the parity forward medians over
    /// — match [`FilterConfig::forward_pairs`] so the corrected chain's
    /// forwards line up with the surface builder's.
    pub forward_pairs: usize,
    /// Leave expiries beyond this horizon untouched (`None` = correct
    /// everything). Set it to the downstream
    /// [`FilterConfig::max_years_to_expiry`] to skip tree work on
    /// expiries the surface builder will drop anyway.
    pub max_years_to_expiry: Option<f64>,
}

impl Default for DeAmericanizeConfig {
    fn default() -> Self {
        DeAmericanizeConfig {
            tree_steps: 201,
            rounds: 2,
            strike_window: (0.4, 2.5),
            forward_pairs: 5,
            max_years_to_expiry: None,
        }
    }
}

/// Per-expiry outcome of [`de_americanize_chain`]. Premium magnitudes
/// are in basis points of spot.
#[derive(Clone, Debug, Serialize)]
pub struct DeaExpiry {
    pub expiry: NaiveDate,
    pub t: f64,
    /// Parity forward implied from the raw (American) mids.
    pub forward_raw: f64,
    /// Parity forward implied from the de-Americanized mids.
    pub forward_dea: f64,
    /// Continuous carry `q` implied by the final forward
    /// (`F = S·e^{(r−q)t}`) — dividends + borrow + basis, measured, not
    /// assumed.
    pub implied_carry: f64,
    /// Quotes whose early-exercise premium came out strictly positive.
    pub corrected: usize,
    pub median_eep_call_bp: f64,
    pub median_eep_put_bp: f64,
    pub max_eep_bp: f64,
}

/// What [`de_americanize_chain`] did, per expiry and overall.
#[derive(Clone, Debug, Default, Serialize)]
pub struct DeAmericanizeReport {
    pub rounds: usize,
    pub expiries: Vec<DeaExpiry>,
    /// `max |F_dea − F_raw| / F_raw` across expiries, in basis points.
    pub max_forward_shift_bp: f64,
    /// Quotes left untouched because subtracting the premium would have
    /// crossed the bid through zero.
    pub skipped_floor: usize,
}

/// Strip the early-exercise premium from an American option chain,
/// returning a European-equivalent chain (same strikes, expiries and
/// spreads; bid/ask/last each lowered by the premium) plus a report.
///
/// Put–call parity is an *inequality* for American options, so the
/// parity forwards of [`implied_vol_surface_from_chain`] carry a small
/// bias — the difference of the near-the-money call and put
/// early-exercise premia. This routine removes it without assuming a
/// dividend yield, by the standard fixed-point iteration:
///
/// 1. Imply the parity forward `F̂` per expiry from the current mids
///    (round one: the raw American mids, exactly as the surface builder
///    would).
/// 2. Read the carry off it: `q̂ = r − ln(F̂/S)/t`, with `r` from the
///    supplied discount curve — the carry is *measured*, never assumed.
/// 3. For each quote, solve the Black-76 vol of the current mid and
///    price the quote both American (CRR binomial) and European
///    (Black-Scholes) at that vol under `(r, q̂)`; the difference,
///    floored at zero, is the early-exercise premium. Subtract it from
///    the *original* mid.
/// 4. Repeat: the corrected pairs satisfy parity as an identity, so the
///    forward tightens; the premium barely moves after one round.
///
/// The premium is computed with a flat per-expiry carry, so discrete
/// dividend timing *within* an expiry is approximated; across expiries
/// it is captured by the per-expiry forwards. For out-of-the-money
/// quotes — the only ones the surface builder keeps — the premium is
/// second-order small and this approximation is well inside it.
pub fn de_americanize_chain(
    chain: &OptionChain,
    discount: &YieldCurve,
    cfg: &DeAmericanizeConfig,
) -> Result<(OptionChain, DeAmericanizeReport), RustyQLibError> {
    let Some(spot) = chain.spot.filter(|s| s.is_finite() && *s > 0.0) else {
        return Err(RustyQLibError::invalid_input(
            "option chain",
            format!(
                "de-americanize: chain for {} has no usable spot",
                chain.symbol
            ),
        ));
    };
    let day_count = DayCountConvention::Act365;

    // Working set: every two-sided quote inside the strike window with a
    // positive expiry. (Expiry/moneyness/OTM filtering stays downstream;
    // pairs need both sides, including the in-the-money leg.)
    struct Work {
        idx: usize,
        strike: f64,
        right: PutOrCall,
        mid: f64,
        eep: f64,
    }
    let mut by_expiry: BTreeMap<NaiveDate, Vec<Work>> = BTreeMap::new();
    for (idx, quote) in chain.quotes.iter().enumerate() {
        let Some(mid) = quote.mid() else { continue };
        let m = quote.strike / spot;
        if m < cfg.strike_window.0 || m > cfg.strike_window.1 {
            continue;
        }
        if (quote.expiry - chain.as_of).num_days() < 1 {
            continue;
        }
        if let Some(max_t) = cfg.max_years_to_expiry {
            if day_count.year_fraction(chain.as_of, quote.expiry) > max_t {
                continue;
            }
        }
        by_expiry.entry(quote.expiry).or_default().push(Work {
            idx,
            strike: quote.strike,
            right: quote.right,
            mid,
            eep: 0.0,
        });
    }

    let parity = |work: &[Work], df: f64| -> Option<f64> {
        let mut calls: BTreeMap<u64, f64> = BTreeMap::new();
        let mut puts: BTreeMap<u64, f64> = BTreeMap::new();
        for w in work {
            let side = match w.right {
                PutOrCall::Call => &mut calls,
                PutOrCall::Put => &mut puts,
            };
            side.insert(w.strike.to_bits(), (w.mid - w.eep).max(0.0));
        }
        let mut pair_forwards: Vec<(f64, f64)> = calls
            .iter()
            .filter_map(|(bits, call)| {
                let put = puts.get(bits)?;
                let strike = f64::from_bits(*bits);
                Some((strike, strike + (call - put) / df))
            })
            .collect();
        if pair_forwards.is_empty() {
            return Some(spot / df); // zero-carry fallback, as the builder's
        }
        pair_forwards.sort_by(|a, b| (a.0 - spot).abs().partial_cmp(&(b.0 - spot).abs()).unwrap());
        let mut nearest: Vec<f64> = pair_forwards
            .iter()
            .take(cfg.forward_pairs.max(1))
            .map(|&(_, f)| f)
            .collect();
        nearest.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(nearest[nearest.len() / 2])
    };

    let mut raw_forwards: BTreeMap<NaiveDate, f64> = BTreeMap::new();
    for round in 0..cfg.rounds {
        for (expiry, work) in by_expiry.iter_mut() {
            let t = day_count.year_fraction(chain.as_of, *expiry);
            let df = discount.df_date(*expiry) / discount.df_date(chain.as_of);
            let Some(fwd) = parity(work, df) else {
                continue;
            };
            if round == 0 {
                raw_forwards.insert(*expiry, fwd);
            }
            let r = -df.ln() / t;
            let q = r - (fwd / spot).ln() / t;
            for w in work.iter_mut() {
                let price = (w.mid - w.eep).max(1e-10);
                let Ok(sigma) =
                    implied_vol_from_price(fwd, w.strike, 0.0, 0.0, t, price / df, w.right)
                else {
                    continue;
                };
                if sigma <= VOL_BOUNDS.0 || sigma >= VOL_BOUNDS.1 {
                    continue;
                }
                let american =
                    crr_american(spot, w.strike, r, q, sigma, t, w.right, cfg.tree_steps);
                let european = bs_price(spot, w.strike, r, q, sigma, t, w.right);
                w.eep = (american - european).max(0.0);
            }
        }
    }

    // Final forwards from the fully corrected mids, and the report.
    let mut report = DeAmericanizeReport {
        rounds: cfg.rounds,
        ..DeAmericanizeReport::default()
    };
    let mut corrected_chain = chain.clone();
    for (expiry, work) in &by_expiry {
        let t = day_count.year_fraction(chain.as_of, *expiry);
        let df = discount.df_date(*expiry) / discount.df_date(chain.as_of);
        let Some(fwd) = parity(work, df) else {
            continue;
        };
        let forward_raw = raw_forwards.get(expiry).copied().unwrap_or(fwd);
        let r = -df.ln() / t;

        let mut eep_calls: Vec<f64> = Vec::new();
        let mut eep_puts: Vec<f64> = Vec::new();
        let mut corrected = 0usize;
        let mut max_eep = 0.0f64;
        for w in work {
            if w.eep <= 0.0 {
                continue;
            }
            let quote = &mut corrected_chain.quotes[w.idx];
            // never cross the bid through zero — leave such quotes alone
            if quote.bid.map_or(false, |b| b - w.eep <= 0.0) {
                report.skipped_floor += 1;
                continue;
            }
            if let Some(b) = quote.bid.as_mut() {
                *b -= w.eep;
            }
            if let Some(a) = quote.ask.as_mut() {
                *a -= w.eep;
            }
            if let Some(l) = quote.last.as_mut() {
                *l = (*l - w.eep).max(0.0);
            }
            corrected += 1;
            let bp = w.eep / spot * 1e4;
            max_eep = max_eep.max(bp);
            match w.right {
                PutOrCall::Call => eep_calls.push(bp),
                PutOrCall::Put => eep_puts.push(bp),
            }
        }
        let median = |v: &mut Vec<f64>| -> f64 {
            if v.is_empty() {
                return 0.0;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        report.max_forward_shift_bp = report
            .max_forward_shift_bp
            .max(((fwd - forward_raw) / forward_raw).abs() * 1e4);
        report.expiries.push(DeaExpiry {
            expiry: *expiry,
            t,
            forward_raw,
            forward_dea: fwd,
            implied_carry: r - (fwd / spot).ln() / t,
            corrected,
            median_eep_call_bp: median(&mut eep_calls),
            median_eep_put_bp: median(&mut eep_puts),
            max_eep_bp: max_eep,
        });
    }
    Ok((corrected_chain, report))
}

/// American option price on a Cox–Ross–Rubinstein lattice with flat
/// continuous rate and carry — deliberately minimal: it exists only to
/// measure the early-exercise premium in [`de_americanize_chain`],
/// where its discretization error cancels against the matching European
/// limit to leading order.
fn crr_american(
    spot: f64,
    strike: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    right: PutOrCall,
    steps: usize,
) -> f64 {
    if !(t > 0.0 && sigma > 0.0 && steps >= 1) {
        return bs_price(spot, strike, r, q, sigma, t, right);
    }
    let intrinsic = |s: f64| -> f64 {
        match right {
            PutOrCall::Call => (s - strike).max(0.0),
            PutOrCall::Put => (strike - s).max(0.0),
        }
    };
    let dt = t / steps as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((((r - q) * dt).exp() - d) / (u - d)).clamp(1e-12, 1.0 - 1e-12);
    let disc = (-r * dt).exp();

    // node (n, j): S = spot · d^n · u^{2j}
    let mut values: Vec<f64> = {
        let mut s = spot * d.powi(steps as i32);
        (0..=steps)
            .map(|_| {
                let v = intrinsic(s);
                s *= u * u;
                v
            })
            .collect()
    };
    for n in (0..steps).rev() {
        let mut s = spot * d.powi(n as i32);
        for j in 0..=n {
            let cont = disc * (p * values[j + 1] + (1.0 - p) * values[j]);
            values[j] = cont.max(intrinsic(s));
            s *= u * u;
        }
    }
    values[0]
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

    /// American chain generated on a fine tree from known `(r, q, vol)`,
    /// with a different step count than the corrector uses.
    fn synthetic_american_chain(q: f64) -> OptionChain {
        let mut quotes = Vec::new();
        for expiry in [d(2027, 8, 5), d(2028, 2, 5)] {
            let t = DayCountConvention::Act365.year_fraction(as_of(), expiry);
            for i in 0..17 {
                let strike = 80.0 + 2.5 * i as f64;
                let vol = true_vol(strike, 0.20);
                for right in [PutOrCall::Call, PutOrCall::Put] {
                    let mid = crr_american(SPOT, strike, RATE, q, vol, t, right, 501);
                    quotes.push(OptionQuote {
                        expiry,
                        strike,
                        right,
                        bid: Some(mid * 0.9975),
                        ask: Some(mid * 1.0025),
                        last: None,
                        volume: None,
                        open_interest: None,
                    });
                }
            }
        }
        OptionChain {
            symbol: "AMER".to_string(),
            as_of: as_of(),
            timestamp: None,
            spot: Some(SPOT),
            quotes,
            metadata: None,
        }
    }

    #[test]
    fn crr_matches_european_when_exercise_is_never_optimal() {
        // an American call on a zero-carry underlying is never exercised
        // early, so the lattice must reproduce Black-Scholes
        for (t, vol, strike) in [(0.25, 0.2, 95.0), (1.0, 0.3, 110.0), (1.5, 0.15, 100.0)] {
            let tree = crr_american(SPOT, strike, RATE, 0.0, vol, t, PutOrCall::Call, 801);
            let bs = bs_price(SPOT, strike, RATE, 0.0, vol, t, PutOrCall::Call);
            assert!(
                (tree - bs).abs() / bs < 5e-3,
                "t={t} K={strike}: tree {tree} vs bs {bs}"
            );
        }
    }

    #[test]
    fn de_americanization_tightens_the_parity_forward() {
        let q = 0.015;
        let chain = synthetic_american_chain(q);
        let filter = FilterConfig::default();

        let (_, raw) = implied_vol_surface_from_chain(&chain, &curve(), &filter).unwrap();
        let (fixed_chain, dea) =
            de_americanize_chain(&chain, &curve(), &DeAmericanizeConfig::default()).unwrap();
        let (_, fixed) = implied_vol_surface_from_chain(&fixed_chain, &curve(), &filter).unwrap();

        assert_eq!(raw.forwards.len(), fixed.forwards.len());
        for ((expiry, f_raw), (_, f_dea)) in raw.forwards.iter().zip(&fixed.forwards) {
            let t = DayCountConvention::Act365.year_fraction(as_of(), *expiry);
            let f_true = SPOT * ((RATE - q) * t).exp();
            let e_raw = (f_raw / f_true - 1.0).abs();
            let e_dea = (f_dea / f_true - 1.0).abs();
            // the raw American parity forward must be visibly biased on
            // this configuration, and the correction must remove most of
            // it and land within 10 bp of the truth
            assert!(
                e_raw > 5e-4,
                "{expiry}: raw bias only {e_raw:.2e} — test has no teeth"
            );
            assert!(
                e_dea < e_raw / 3.0,
                "{expiry}: dea error {e_dea:.2e} not well under raw {e_raw:.2e}"
            );
            assert!(e_dea < 1e-3, "{expiry}: dea error {e_dea:.2e} above 10 bp");
        }
        assert!(dea.max_forward_shift_bp > 0.0);

        // premia are non-negative: no corrected quote got more expensive
        for (orig, fixed) in chain.quotes.iter().zip(&fixed_chain.quotes) {
            assert!(fixed.bid.unwrap() <= orig.bid.unwrap() + 1e-12);
            assert!(fixed.ask.unwrap() <= orig.ask.unwrap() + 1e-12);
        }
    }

    #[test]
    fn de_americanized_vols_are_closer_to_the_generating_skew() {
        let q = 0.015;
        let chain = synthetic_american_chain(q);
        let filter = FilterConfig::default();
        let (fixed_chain, _) =
            de_americanize_chain(&chain, &curve(), &DeAmericanizeConfig::default()).unwrap();

        let worst = |c: &OptionChain| -> f64 {
            let (surface, report) = implied_vol_surface_from_chain(c, &curve(), &filter).unwrap();
            let mut w = 0.0f64;
            for (expiry, forward) in &report.forwards {
                let t = DayCountConvention::Act365.year_fraction(as_of(), *expiry);
                for i in 0..17 {
                    let strike = 80.0 + 2.5 * i as f64;
                    let m = strike / forward;
                    if !(0.9..=1.1).contains(&m) {
                        continue; // near the money, where both sides quote
                    }
                    w = w.max((surface.vol(strike, *forward, t) - true_vol(strike, 0.20)).abs());
                }
            }
            w
        };
        let (w_raw, w_dea) = (worst(&chain), worst(&fixed_chain));
        assert!(
            w_dea < w_raw,
            "de-americanized worst vol error {w_dea:.5} not below raw {w_raw:.5}"
        );
        assert!(w_dea < 2e-3, "residual vol error {w_dea:.5} above 20 bp");
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
