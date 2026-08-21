# `src/equity` — Equity Derivatives

Contracts, pricing engines, volatility models, and the spine that binds them.

```
equity/
├── contracts/   what you can trade   (payoffs and product logic)
├── engines/     how it gets priced   (numerical and analytic methods)
├── models/      dynamics             (volatility structure)
├── service/     JSON/XML batch entry points
└── *.rs         the spine: builder, market, greeks, portfolio, option_chain
```

Any contract prices on any engine that supports it — the combination is
validated at build time, so **an option that builds is guaranteed to price**.

---

## Engines (`engines/`)

| Engine | File | Use it for |
|---|---|---|
| Black-Scholes analytic | `blackscholes.rs` | European vanillas, binaries, closed-form Greeks |
| Black-76 | `black76.rs` | Options on futures — discounted or futures-style margined |
| Barone-Adesi-Whaley | `baw.rs` | American approximation, microseconds |
| Bjerksund-Stensland 2002 | `bjerksund_stensland.rs` | American approximation, better in the wings |
| Binomial tree | `binomial.rs` | American/Bermudan exercise; CRR and Leisen-Reimer |
| Finite difference | `finite_difference.rs` | Log-spot Crank-Nicolson + Rannacher; barriers, American via PSOR |
| Heston ADI | `heston_adi.rs` | Two-factor `(S, v)` grid, Douglas splitting |
| Monte Carlo | `montecarlo.rs` | Path-dependent and multi-asset; parallel, per-path RNG streams |
| COS | `cos.rs` | Fourier-cosine pricing from a characteristic function |
| Carr-Madan | `carr_madan.rs` | FFT pricing of a whole strike grid at once |

The COS method prices a 20-strike Heston smile ~90× faster than per-strike
integration — the ratio every calibration loop inherits.

## Contracts (`contracts/`)

- **Vanilla** — European and American calls/puts (`vanilla_option.rs`,
  `equity_option.rs`), forwards and futures.
- **Binary** — cash-or-nothing and asset-or-nothing, with the smile-slope
  correction (`binary_option.rs`).
- **Barrier** — all eight knock-in/out × up/down types, rebates (at hit or at
  expiry), double-barrier corridors via Ikeda-Kunitomo (`barrier.rs`).
- **Asian** — arithmetic/geometric, fixed/floating strike (`asian.rs`).
- **Lookback** — floating and fixed strike; Goldman-Sosin-Gatto and
  Conze-Viswanathan closed forms (`lookback.rs`).
- **Chooser** — simple (Rubinstein parity) and complex, per-leg strikes and
  expiries (`chooser.rs`).
- **Autocallable** — coupons, knock-in protection, Phoenix certificates with
  conditional and memory coupons (`autocallable.rs`).
- **Cliquet** — ratchets, reverse cliquets, Napoleons (`cliquet.rs`).
- **Accumulator / decumulator** — daily geared accrual with knock-out, priced
  as a strip of barrier pairs or by Monte Carlo (`accumulator.rs`).
- **Variance swaps** — variance, gamma and corridor; model-free replication
  over any smile, seasoned MtM with accrued realized variance
  (`variance_swap.rs`).
- **Multi-asset** — rainbow best-of/worst-of/spread/basket/exchange on *n*
  correlated assets (`rainbow.rs`, `multi_asset.rs`, `worst_of.rs`).
- **Forward-start** and **perpetual** American.

## Models (`models/`)

| Model | File | Notes |
|---|---|---|
| Dupire local vol | `local_vol.rs` | Non-parametric, from an implied surface; Gatheral's total-variance form |
| Heston | `heston.rs` | Semi-analytic CF pricing, trap-free formulation, LM calibration |
| Bates | `bates.rs` | Heston + Merton lognormal jumps, or + Kou double-exponential |
| SABR | `sabr.rs` | Hagan expansion, two-factor MC, per-expiry LM calibration, surface smoother |
| SLV | `slv.rs` | Heston variance × leverage function, particle/binning calibration |
| Rough Bergomi | `rbergomi.rs` | Volterra variance; exact Gaussian scheme + FFT hybrid scheme |
| SVI / SSVI | `svi.rs` | Gatheral / Gatheral-Jacquier; butterfly `g(k)` and calendar checks |
| eSSVI | `essvi.rs` | Hendriks-Martini `(θ, ψ, ρ)` per slice, calendar penalty during the fit |
| Surface tools | `vol_surface.rs`, `surface_repair.rs`, `smoothed_surface.rs`, `usability.rs` | Build, repair, smooth, and score a surface |
| Processes | `processes.rs` | `StochasticProcess` implementations feeding the MC engine |

## The spine

- **`builder.rs`** — fluent `EquityOptionBuilder`; validates every input and
  the engine/payoff combination.
- **`market.rs`** — spot, curves, vol surfaces, dividends and borrow.
- **`greeks.rs`** — one module for every Greek request. Four routes: grid
  (read off the solved grid), tree (one backward pass), analytic (closed
  forms), and bump (shared central-difference stencils for everything else).
  Caches shared repricings — one MC `price()` costs 17 simulations, not 28.
- **`option_chain.rs`** — listed chain → cleaned quotes → parity forwards →
  Black-76 implied vols → surface, with a `SurfaceBuildReport` counting every
  dropped quote by reason.
- **`portfolio.rs`** — book-level aggregation.
- **`service/`** — JSON/XML batch entry points.

## Example

```rust
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::{Engine, Model};
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::Instrument;

let option = EquityOptionBuilder::new()
    .spot(100.0).strike(100.0)
    .flat_vol(0.30).flat_rate(0.05)
    .years_to_maturity(1.0)
    .vanilla(PutOrCall::Call)
    .engine(Engine::FiniteDifference)
    .model(Model::LocalVol)
    .build()?;

let r = option.price()?;
println!("pv {:.6}  delta {:.4}  vega {:.4}", r.pv, r.greeks.delta, r.greeks.vega);
```

See [`examples/`](../../examples) for runnable end-to-end programs, and the
[main README](../../README.md) for the CLI.
