# Design: option chains → implied vol surface

Status: **draft for review** — nothing in this document is implemented yet.
Scope: the option-chain data model, chain cleaning, parity-implied
forwards, vol surface construction/serialization, broker connectors
(IBKR first), and the 3D surface plot. Local vol is sequenced but out of
scope here.

## Goals

1. A user with an IBKR account (or a saved CBOE delayed-quotes file, or
   any CSV they can reshape) gets from *option chain* to *implied vol
   surface* with the CLI, prices with it, and views it as an interactive
   3D plot.
2. The pricing library stays clean: no broker SDK, no credentials, no
   network outside `fetch`. Connectors normalize into one typed chain
   structure; everything downstream is broker-agnostic and offline-testable.
3. Surfaces are **files**: built once, saved with provenance metadata,
   reloaded later — same auditability contract as the fetched Fed data.

## Architecture: three layers, two boundaries

```
IBKR gateway ──┐
CBOE delayed ──┼─► fetched document (verbatim, per-source)     [fetch feature]
user's CSV  ───┘         │
                         ▼  normalize (per-source converter)
                  OptionChain (unified Rust struct, serde)      [always compiled]
                         │
                         ▼  clean → imply forwards → solve vols
                  VolSurface (+ to/from JSON)                   [always compiled]
                         │
                         ├─► pricing (existing engines, Dupire later)
                         └─► 3D HTML plot
```

- **Boundary 1 — the fetched document**: whatever the source sent,
  verbatim, plus `metadata` (source, symbol, timestamps, delayed/live).
  Same philosophy as `fetch ust`/`sofr`: never reinterpret at the fetch
  layer.
- **Boundary 2 — `OptionChain`**: the unified structure the user asked
  for. Every source converts *into* it; the surface builder consumes
  *only* it. A user with a broker we don't support writes one converter
  to this type (or its JSON form) and everything else works.

## The unified `OptionChain`

New module `src/equity/option_chain.rs` (always compiled — it is a data
model, not a network feature). Serde derives give the JSON file format
for free; the struct *is* the schema.

```rust
/// One side of the book for one listed option.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionQuote {
    pub expiry: NaiveDate,
    pub strike: f64,
    pub right: PutOrCall,           // existing core::trade::PutOrCall
    /// Best bid/ask. `None` = not quoted (distinct from 0.0 = quoted at zero).
    pub bid: Option<f64>,
    pub ask: Option<f64>,
    /// Optional color, kept when the source provides it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub volume: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_interest: Option<f64>,
}

/// An option chain snapshot for one underlying, normalized from any source.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptionChain {
    pub symbol: String,
    /// Trade/valuation date the snapshot represents.
    pub as_of: NaiveDate,
    /// Snapshot timestamp when the source provides one (RFC 3339).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Underlying price at snapshot time (needed for OTM classification
    /// when parity forwards cannot be implied; optional because some
    /// sources omit it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spot: Option<f64>,
    pub quotes: Vec<OptionQuote>,
    /// Provenance: source name, delayed/live, file/url, fetch time, ...
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}
```

Deliberate choices:

- **Quotes are a flat list, not a grid.** Real chains are ragged
  (different strikes per expiry, missing sides); grids come later, if ever.
- **`bid`/`ask` as `Option`** so "no quote" survives normalization
  honestly. Cleaning rules, not the data model, decide what is usable.
- **No greeks/IV fields.** Sources like CBOE publish their own IVs; they
  stay in the verbatim fetched document. This library derives its own
  vols from prices — one solver, one set of conventions, comparable
  across sources. (A later `compare` can diff our IVs against the
  source's.)
- **No prices-per-source variants.** A connector's entire job is:
  fetched document → `OptionChain`.

## Cleaning rules (chain → usable quotes)

Explicit, documented, and applied in one place
(`OptionChain::cleaned(&FilterConfig)`), with defaults tuned for delayed
retail data. Each rule logs what it drops; the count of dropped quotes
goes into the surface metadata.

| # | Rule | Default |
|---|------|---------|
| 1 | Price = mid of bid/ask; both sides required | `bid > 0` and `ask >= bid` |
| 2 | Relative spread cap: `(ask-bid)/mid` | ≤ 25% |
| 3 | OTM-only per side (calls above forward, puts below), vols merged into one smile | on |
| 4 | Moneyness window `K/F` | [0.5, 2.0] |
| 5 | Expiry window | 7 days ≤ T ≤ 2 years |
| 6 | Minimum surviving quotes per expiry, else the expiry is dropped | 3 |

Rule 3 is the standard practice (OTM options are the liquid side and
avoid the American-early-exercise premium contaminating ITM call vols);
worth stating in the docs since it surprises people seeing half their
chain "missing".

## Forwards by put-call parity

Implied vol needs the forward; the forward needs dividends — which we
refuse to guess. Instead, per expiry:

`F = K + df(T)^-1 * (C_mid(K) - P_mid(K))`

evaluated at every strike where both sides survive cleaning, taking the
**median** across the strikes nearest the spot (default: 5 pairs). The
discount factor comes from a `YieldCurve` argument — the bootstrapped
Treasury curve once roadmap #1 lands, a flat-rate curve until then (the
function signature takes `&YieldCurve` either way; `YieldCurve::flat`
already exists). This is where `fetch ust` composes with the chain work.

Output: `Vec<(NaiveDate, f64)>` of per-expiry forwards, recorded in the
surface metadata. Caveat documented: parity holds exactly for European
options; for American equity options it is an approximation that is
excellent near-ATM, which is exactly where we sample it.

## Implied vols and surface assembly

For each cleaned quote: Black-76 implied vol from the mid price against
the expiry's implied forward and discount factor (the existing
safeguarded-Newton solver; `equity::black76` already prices off
forwards). Failures (below intrinsic, no solution) are skipped with a
warning, consistent with `build_implied_vol_surface` today.

Per-expiry `(strike, vol)` points then feed the existing
`VolSurface::from_strike_smiles` — no new surface math needed.

The current `build_implied_vol_surface(contracts)` path (spot + rate +
dividend per contract) stays; the chain path is a second constructor:

```rust
pub fn implied_vol_surface_from_chain(
    chain: &OptionChain,
    discount: &YieldCurve,
    filter: &FilterConfig,
) -> Result<(VolSurface, SurfaceBuildReport), RustyQLibError>;
```

`SurfaceBuildReport` carries what the metadata needs: per-expiry
forwards, quotes used/dropped per rule, solver failures.

## `VolSurface` serialization (to_json / from_json)

**Current state (the gap):** `VolSurface` derives `Serialize` only — a
saved surface cannot be read back. The deserializable form, `VolInput`,
has only rectangular grids, which cannot represent the per-expiry ragged
smiles that `from_strike_smiles` (and therefore any chain-built surface)
produces. So today the canonical surface is write-only.

**Proposal — make the saved form an input form:**

1. New `VolInput::StrikeSmiles` variant mirroring `from_strike_smiles`:

   ```json
   { "type": "strike_smiles",
     "expiries": ["2026-09-18", "2026-12-18"],
     "smiles": [[[95.0, 0.31], [100.0, 0.28]],
                [[90.0, 0.30], [100.0, 0.27], [110.0, 0.25]]],
     "day_count": "Act365" }
   ```

   This also benefits hand-written contracts: ragged quotes no longer
   need padding into a grid.

2. `VolSurface::to_input(&self) -> VolInput` — exact, lossless (the
   canonical data *is* per-expiry point lists; delta-grid surfaces
   round-trip as their converted log-moneyness smiles, documented).

3. A surface document wrapping it, symmetric with the fetch documents:

   ```json
   { "metadata": { "symbol": "AAPL", "built_from": "...", "curve": "...",
                   "forwards": {"2026-09-18": 231.4}, "quotes_used": 412,
                   "quotes_dropped": {"wide_spread": 31}, "fetched_at": "..." },
     "reference_date": "2026-08-05",
     "surface": { "type": "strike_smiles", ... } }
   ```

   With `VolSurface::to_json(&self) -> String` /
   `VolSurface::from_json(&str) -> Result<Self, ...>` (JSON and, through
   the existing serialization layer, XML for free). Because the payload
   is a `VolInput`, a saved surface is *also* a valid `vol` block for a
   pricing contract — save once, price against it later, no separate
   loader path.

## Other `VolSurface` enhancements (recommended, small)

- **Public pillar access**: `smile_points(&self) -> impl Iterator<...>`
  or a `pillars()` view (times + (coord, vol) points + coordinate kind).
  Needed by serialization and the 3D plot anyway; today the internals
  are private and only `Display` exposes them.
- **Arbitrage diagnostics, not gates**: `fn diagnostics(&self, forward:
  impl Fn(f64) -> f64) -> SurfaceDiagnostics` reporting butterfly
  violations (call-price convexity in strike per expiry) and calendar
  violations (total variance decreasing in time at fixed moneyness).
  Market snapshots are noisy — construction should not hard-fail, but
  the report belongs in the surface metadata and the CLI output, and it
  is a prerequisite sanity layer for Dupire later.
- **Vol bounds on input** (e.g. reject solved vols outside (0.5%, 500%))
  — same refuse-to-guess-units stance as the fetch parsers.
- Already right, keep as is: linear-total-variance time interpolation,
  flat wings, the canonical one-query-path design mirroring `curves`.

## CLI and plotting

- `fetch chain --symbol AAPL [--source cboe | --broker ibkr --settings broker.toml]
  [--from-file saved.json]` → verbatim fetched document (boundary 1).
- `build` recognizes a chain document (or `OptionChain` JSON) and writes
  `vol_surface/vol_surface.json` — the surface document above — next to
  the existing vol-surface output path, plus `vol_surface.html`: the 3D
  plot. `--rate` flat fallback or `--curve ust.json` once #1 lands.
- The plot reuses the `examples/common/plot3d.rs` approach (self-contained
  Plotly HTML, figure spec via `serde_json`): x = strike or moneyness,
  y = expiry, z = implied vol, one marker trace for the actual quote
  pillars over the interpolated surface — seeing raw points vs
  interpolation is the honest view. Promote the helper from
  `examples/common` into the `cli` feature (it is ~150 lines, no new deps).

## Sources

### CBOE delayed quotes (first, keyless)

`https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json` —
free, no key, 15-min delayed, whole chain with bid/ask/OI/volume, spot
included. Option symbols are OCC-format (`AAPL251219C00230000`); the
converter parses expiry/strike/right from them. Terms: exchange data for
personal/research use — recorded in metadata, noted in docs, same
stance as for all fetched data. Fixture-tested offline; one `#[ignore]`d
live test.

### IBKR (the broker connector)

- **Auth boundary**: the user runs TWS or IB Gateway, logged in; we
  connect to `host:port` as an API client. The library never sees
  credentials. `broker.toml` (pattern: existing `stress-config` TOML):

  ```toml
  [ibkr]
  host = "127.0.0.1"
  port = 4002            # gateway paper; 4001 live, 7497/7496 TWS
  client_id = 7
  market_data = "delayed" # default; "live" needs the account's OPRA sub
  ```

- **Flow**: resolve underlying → `reqSecDefOptParams` (expirations ×
  strikes) → filter to the requested expiry/moneyness windows *before*
  quoting → snapshot quotes per contract, batched under IBKR pacing
  (~50 msg/s, limited concurrent lines) → normalize → verbatim-ish
  fetched document + `OptionChain`. A full board is minutes; a filtered
  window (default: expiries ≤ 1y, K/F in [0.7, 1.3]) is tens of seconds.
- **Packaging**: feature `broker-ibkr` using the community `ibapi`
  crate, **not** included in `cli` (heavy dependency; install is
  `--features cli,broker-ibkr`). If dependency churn proves painful it
  moves to a companion crate later; same code either way.
- **Testing**: converters and pacing logic against recorded fixtures in
  CI; end-to-end only via `#[ignore]`d tests + a smoke run with a live
  Gateway (user-side; cannot run in CI or by the assistant).

## Sequencing

1. `OptionChain` + cleaning + parity forwards + Black-76 solve +
   `StrikeSmiles`/`to_json`/`from_json` + diagnostics (offline, the bulk).
2. CBOE fetch + converter (small; proves boundary 1 with a keyless source).
3. 3D plot + `build` integration.
4. IBKR connector (needs a Gateway session to verify — coordinated with
   the user).
5. Later, separate designs: SVI fit → Dupire local vol from the fitted
   surface; `compare` (our IVs vs source IVs slots in here too).

Steps 1–3 are CI-testable end-to-end from fixtures. Step 4 rides on the
same `OptionChain` boundary, so its blast radius is one module.

## Open questions for review

1. Chain-surface coordinate: build on **absolute strikes** (matches
   `from_strike_smiles`, simplest, sticky-strike) — or convert to
   forward moneyness at build time (better for cross-expiry
   comparability and the later SVI fit)? Proposal: strikes now,
   revisit at SVI time; the forwards are saved either way.
2. Should the surface document also embed the discount curve used, or
   only reference it in metadata? Proposal: metadata reference only —
   curves are their own files.
3. `FilterConfig` defaults above — any you would tune?
