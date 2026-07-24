[![Build and Tests](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml/badge.svg)](https://github.com/siddharthqs/RustyQLib/actions/workflows/rust.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Crates.io](https://img.shields.io/crates/dr/rustyqlib)
![Crates.io](https://img.shields.io/crates/v/rustyqlib)
[![codecov](https://codecov.io/gh/siddharthqs/RustyQLib/graph/badge.svg?token=879K6LTTR4)](https://codecov.io/gh/siddharthqs/RustyQLib)

# RustyQLib — Pricing Options using JSON or XML

RustyQLib is a lightweight quantitative finance library written entirely in Rust.
It prices equity derivatives through JSON or XML contracts (a stateless pricing service
in a single binary) or as a Rust library, with an emphasis on numerically validated
implementations: every pricer is cross-checked against independent oracles,
put-call parity, replication identities and cross-engine agreement in the test suite.

## Highlights

- **Six pricing engines** — analytic closed forms, two analytic American
  approximations (Barone-Adesi-Whaley and Bjerksund-Stensland 2002), binomial
  tree, finite difference (log-spot Crank-Nicolson with Rannacher smoothing),
  and parallel Monte Carlo — behind one dispatch, so the same contract prices
  on any suitable engine.
- **Four volatility frameworks** — Black-Scholes, **Dupire local volatility**
  (calibrated non-parametrically from an implied vol surface), **Heston
  stochastic volatility** (semi-analytic characteristic-function pricing +
  Monte Carlo) with **Bates jump-diffusion extensions** (Heston + lognormal
  Merton jumps, and Heston + Kou double-exponential jumps, both priced
  semi-analytically through the shared characteristic-function machinery and
  calibrated by the same Levenberg-Marquardt transform-space pattern),
  **stochastic local volatility** (Heston-style variance times a leverage
  function calibrated to the Dupire surface by the particle / binning
  method, so vanillas reprice while forward smiles stay stochastic),
  and **SVI / SSVI parametric implied surfaces** (Gatheral,
  Gatheral-Jacquier) — Levenberg-Marquardt smile and surface calibration,
  Gatheral-Jacquier butterfly (`g(k)`) and calendar no-arbitrage checks, and
  sampling into the pricing vol surface.
- **JSON and XML** contracts and results, over a single shared schema.
- **Market-standard infrastructure** — discount curves with discount factors as the
  source of truth (flat / zero rates / discount factors / forward rates in, any
  compounding), volatility surfaces (flat, strike x expiry, moneyness, FX-style
  delta quotes), robust implied vol, day counts, term-structure-consistent PDE
  discounting.
- **Options on futures**: European vanillas priced with Black-76, both
  standard (discounted premium) and futures-style (margined, undiscounted).
- **Payoffs**: European & American vanillas, cash- and asset-or-nothing binaries,
  all eight barrier types (knock-in/out, up/down) with **rebates** (at hit or
  at expiry) and **double-barrier corridors** (Ikeda-Kunitomo), Asian options (arithmetic /
  geometric, fixed / floating strike), forward-start options, autocallable
  notes with coupons and knock-in protection (incl. **Phoenix certificates**
  with conditional and memory coupons), **lookback options** (floating and
  fixed strike, Goldman-Sosin-Gatto / Conze-Viswanathan closed forms),
  **accumulators / decumulators** (daily geared accrual with knock-out,
  priced as a strip of barrier-option pairs or by Monte Carlo),
  **variance, gamma and corridor variance swaps** (model-free replication
  over any smile: DDKZ log contract, spot-weighted `S ln S` contract with
  exact carry adjustment, Carr-Lewis corridor truncation with exact corridor
  additivity; seasoned MtM with accrued realized variance and the exact GBM
  volatility-swap strike), **cliquets / ratchets, reverse
  cliquets and Napoleons** (capped-floored return strips, coupon-minus-losses
  and coupon-plus-worst-month structures; closed forms under Black-Scholes
  where the payoff permits, Monte Carlo under GBM or Heston), and **multi-asset rainbow
  options** (best-of, worst-of, spread, basket, exchange) on n correlated
  assets. Carry handles dividend yield, discrete cash dividends and stock
  borrow cost.

## Products and engines

| Payoff | Analytic | Binomial | Finite difference | Monte Carlo |
|---|---|---|---|---|
| Vanilla European | Black-Scholes / Heston CF | yes | yes (grid Greeks) | yes (+ stderr) |
| Vanilla on a future | Black-76 (discounted / margined) | — | — | — |
| Vanilla American | Barone-Adesi-Whaley / Bjerksund-Stensland 2002 (approx.) | yes | Brennan-Schwartz | two-pass Longstaff-Schwartz |
| Perpetual American | Merton closed form (exact) | — | — | — |
| Binary (cash / asset) | closed form / Heston CF | yes | yes (Rannacher + cell averaging) | yes |
| Barrier (8 types, + rebates at hit / at expiry) | Reiner-Rubinstein + E/F rebate terms | — | absorbing boundary / parity | Brownian-bridge corrected (rebate at expiry) |
| Double barrier (knock-in / knock-out corridor) | Ikeda-Kunitomo image series | — | — | discrete corridor monitoring |
| Asian (arith / geo, fixed / floating) | Turnbull-Wakeman (fixed + Henderson-Wojakowski average-strike) / exact geometric (fixed + average-strike) | — | — | geometric control variate |
| Forward-start | Rubinstein (BS) | — | — | yes (incl. Heston forward smile) |
| Autocallable / Phoenix (conditional + memory coupons) | — | — | — | multi-date discounting; GBM / local vol / Heston |
| Rainbow (best/worst-of, spread, basket, exchange) | Margrabe / Kirk / moment matching | — | — | correlated terminal GBM |
| Accumulator / decumulator (geared, knock-out) | strip of Reiner-Rubinstein knock-out pairs | — | — | discrete daily knockout |
| Lookback (floating / fixed strike) | Goldman-Sosin-Gatto / Conze-Viswanathan (continuous) | — | — | discrete monitoring |
| Variance / gamma / corridor variance swap (+ GBM vol-swap strike) | model-free replication: DDKZ 1/K^2 kernel, S ln S contract with carry adjustment, Carr-Lewis corridor truncation | — | — | — |
| Cliquet / ratchet / reverse cliquet / Napoleon | forward-start call/put spreads (no global clamp; Napoleon MC-only) | — | — | per-period GBM or Heston paths |

Model availability: local vol runs on the FD and MC engines; Heston runs on the
analytic (vanilla + binary) and MC engines (all payoffs above except American
and rainbow). Rainbow options are a separate product type
(`"product_type": "rainbow_option"`) with per-asset spots/vols/dividends and a
correlation matrix; outputs include per-asset `deltas` and `vegas`.

### Engine details

- **Finite difference**: theta-scheme in log-spot with per-node, per-step
  coefficients (local vol ready), forward rates from the discount curve per time
  step, cell-averaged terminal conditions for digitals, barrier-aligned absorbing
  boundaries, and delta/gamma/theta read directly off the grid. Grid sizes are
  configurable per contract. The numerical kernels live in a reusable
  `core::fd_solvers` toolkit covering 1-D to 3-D problems: Thomas tridiagonal,
  Brennan-Schwartz and PSOR obstacle solvers (one- and two-sided), tensor-grid
  axis operators, and Douglas / Hundsdorfer-Verwer ADI time steppers with
  explicit mixed-derivative (correlation) terms — the machinery a 2-D Heston or
  hybrid three-factor PDE needs.
- **Monte Carlo**: deterministic per-path RNG streams (bit-reproducible under
  rayon parallelism), low-discrepancy sampling through a Brownian bridge,
  exact/Euler/Milstein stepping, antithetic + moment matching, geometric control
  variates for Asians, Brownian-bridge barrier monitoring, and standard errors
  reported with every price. Greeks via common-random-number bumps.
- **Analytic American approximations**: Barone-Adesi-Whaley (`pricer: "BAW"`,
  quadratic approximation with a Newton solve for the exercise boundary) and
  Bjerksund-Stensland 2002 (`pricer: "BS2002"`, two-step flat exercise
  boundary priced in closed form via the cumulative bivariate normal — a lower
  bound on the true price, generally the tighter of the two). Both price in
  under a microsecond, within a few cents of a fine tree, and return true
  American Greeks (unlike the tree, whose Greeks fall back to the European
  closed form). Use them when speed matters more than the last basis point.
  **Perpetual American** calls and puts have exact Merton closed forms
  (`equity::perpetual`), verified against the stationary pricing ODE, smooth
  pasting and the finite-maturity limit.
- **Calibration workflow**: quoted option prices -> robust implied vols
  (safeguarded Newton with arbitrage bounds) -> implied surface -> Dupire local
  vol -> reprice anything, including barriers under smile dynamics. **Heston
  calibration** fits all five parameters to vanilla quotes by
  Levenberg-Marquardt in an unconstrained transform space (log/atanh).
- **Risk analytics** (`risk`): Value-at-Risk and Expected Shortfall in the
  standard flavors — historical, parametric normal, Cornish-Fisher
  higher-moment corrected, and delta-normal multi-asset VaR with the exact
  Euler component / marginal decomposition — plus scenario **VaR/ES for an
  options book** (delta-gamma-vega-theta from aggregated Greeks and full
  revaluation through the portfolio repricer, on shared scenarios so the
  difference isolates the Taylor error), EWMA / realized volatility,
  max drawdown, Sharpe / Sortino, the Kupiec VaR backtest, and
  **TOML-configured stress MtM**: named shock scenarios (relative / absolute
  bumps on spot, vol, rates and time, with per-underlying filters) prepared
  into bumped market data and fully revalued, reported per trade and
  aggregated per scenario.
- **Adjoint Algorithmic Differentiation** (`core::aad`): a tape-based
  reverse-mode differentiator with operator overloading — write a pricer over
  `Var` and one backward sweep returns every input sensitivity at a fixed
  small multiple of pricing cost. Ships with the Black-Scholes closed form
  on tape (all six first-order Greeks from one sweep, validated to the
  library's closed forms) and pathwise Monte Carlo Greeks (delta / vega /
  rho differentiated straight through the simulation).
- **Numerical toolkits** in `core`: 1-D root finding (`solvers`: bisection,
  Newton-Raphson, secant, Halley, safeguarded Newton — pluggable by enum),
  multi-dimensional optimization (`optimization`: Levenberg-Marquardt, BFGS,
  conjugate gradient, steepest descent, Nelder-Mead, differential evolution —
  the fitting layer for Heston today, SABR / Nelson-Siegel tomorrow), FD
  linear solvers (`fd_solvers`, 1-D to 3-D), asset-agnostic Monte Carlo
  machinery (`montecarlo`: reproducible per-path RNG streams, Sobol with
  digital-shift scrambling, Halton with Cranley-Patterson rotation, Brownian
  bridge, stratified / Latin-hypercube sampling, control variates, moment
  matching, Welford statistics), an interpolation toolkit (`interpolation`:
  linear, cubic splines with natural / clamped / not-a-knot boundaries,
  shape-preserving PCHIP and Akima, bilinear grids and thin-plate splines for
  2-D vol surfaces), and linear algebra (`linalg`: Cholesky / QR / SVD /
  symmetric-eigen decompositions with least-squares and pseudo-inverse
  solves, plus correlation repair — PSD-tolerant Cholesky and Higham's
  nearest-correlation projection, auto-applied to non-PSD rainbow
  correlation inputs).

## Running the CLI

```bash
cargo build --release
# price a single JSON file of contracts
rustyqlib file --input contracts.json --output results.json
# price every JSON file in a directory (parallel)
rustyqlib dir --input contracts/ --output results/
# build an implied vol surface from quoted options
rustyqlib build --input quotes.json --output out/
# guided pricing in the terminal
rustyqlib interactive
```

### Contract examples

Vanilla European call priced analytically (a flat rate builds a flat curve):

```json
{
  "asset": "EQ",
  "contracts": [{
    "action": "PV", "asset": "EQ",
    "product_type": {
      "product_type": "option", "symbol": "ABC",
      "underlying_price": 100.0, "put_or_call": "C", "payoff_type": "vanilla",
      "strike_price": 100.0, "volatility": 0.3, "maturity": "2027-07-17",
      "risk_free_rate": 0.05, "dividend": 0.0, "pricer": "Analytical"
    }
  }]
}
```

The same contract can carry richer market data and model choices:

```json
{
  "discount_curve": { "type": "zero_rates", "tenors": [0.25, 1.0, "2029-07-17"],
                      "rates": [0.045, 0.05, 0.055], "compounding": "continuous" },
  "vol_surface":    { "type": "strike_expiry", "expiries": [0.5, 1.0],
                      "strikes": [90.0, 100.0, 110.0],
                      "vols": [[0.32, 0.30, 0.28], [0.33, 0.31, 0.30]] },
  "mc_model": "heston",
  "heston": { "v0": 0.09, "kappa": 2.0, "theta": 0.09, "vol_of_vol": 0.4, "rho": -0.7 },
  "pricer": "MC", "simulation": 100000
}
```

Selected fields (all optional unless noted):

| Field | Meaning |
|---|---|
| `pricer` | `Analytical`, `Binomial`, `FD`, `MC`, `BAW`, `BS2002` (analytic American) |
| `payoff_type` | `vanilla`, `binary`, `barrier`, `asian`, `forward_start`, `autocallable` |
| `exercise_style` | `European` (default), `American` |
| `binary_type`, `cash_amount` | `cash` / `asset`, cash paid when ITM |
| `barrier_type`, `barrier_level` | `up_in`, `up_out`, `down_in`, `down_out` |
| `averaging_type`, `asian_strike_type` | `arithmetic`/`geometric`, `fixed`/`floating` |
| `rainbow_type`, `assets`, `correlations`, `weights` | rainbow options: `best_of`, `worst_of`, `spread`, `basket`, `exchange` |
| `forward_start_date`, `strike_fraction` | forward-start options |
| `autocall_barrier`, `protection_barrier`, `autocall_coupon`, `autocall_observations`, `notional` | autocallable notes |
| `borrow_cost` | continuous stock borrow (repo) cost, part of the carry |
| `futures_settlement` | option on a future (Black-76): `discounted` (standard) or `margined` (futures-style); `underlying_price` is then the futures price |
| `cash_dividends` | discrete dividends `[{"date", "amount"}]`; escrowed model on analytic/tree/terminal-MC, jumps on path-MC and FD |
| `discount_curve` | `flat`, `zero_rates`, `discount_factors`, `forward_rates` |
| `vol_surface` | `flat`, `strike_expiry`, `moneyness_expiry`, `delta_expiry` |
| `mc_model` | `gbm` (default), `local_vol`, `heston` (needs `heston` params) |
| `simulation`, `mc_time_steps`, `mc_scheme`, `mc_sampler`, `mc_seed` | Monte Carlo controls |
| `fd_spot_steps`, `fd_time_steps` | finite difference grid |

Working examples for every product live in [`src/examples/EQ/`](src/examples/EQ/).
Monte Carlo outputs include the standard error (`std_err`) alongside price and Greeks.

### XML contracts

Every contract can equally be written in XML — the input format is detected from the
content and the output format from the output file extension, so `-o results.xml`
writes XML and `-o results.json` writes JSON regardless of the input:

```xml
<contracts>
  <asset>EQ</asset>
  <contracts>
    <item>
      <action>PV</action>
      <asset>EQ</asset>
      <product_type product_type="option">
        <symbol>ABC</symbol>
        <underlying_price>100.0</underlying_price>
        <put_or_call>C</put_or_call>
        <payoff_type>vanilla</payoff_type>
        <strike_price>100.0</strike_price>
        <volatility>0.3</volatility>
        <maturity>2027-07-17</maturity>
        <risk_free_rate>0.05</risk_free_rate>
        <pricer>Analytical</pricer>
      </product_type>
    </item>
  </contracts>
</contracts>
```

Conventions: elements are object fields; **attributes are fields too** (convenient for
enum tags such as `type="flat"`); `<item>` children make an array, including
single-element ones; scalars are inferred, so numbers become numbers while
`2027-07-17` and `C` stay strings. See
[`src/examples/EQ/equity_option.xml`](src/examples/EQ/equity_option.xml), and convert
between formats with `cargo run --example convert_format -- in.json out.xml`.

XML is a *syntax* over the same data model — documents are transcoded to
`serde_json::Value` and deserialized with the same derives, so both formats share one
schema, one set of defaults and one set of validation rules, and every new product
supports both automatically.

## Runnable examples

One file per product under [`examples/`](examples/), each pricing across every
applicable engine and model with identity checks:

```bash
cargo run --release --example vanilla_option     # all four engines, European + American
cargo run --release --example barrier_option     # eight barrier types, in-out parity
cargo run --release --example heston_option      # char. function vs MC, smile shape
cargo run --release --example local_vol_calibration  # quotes -> surface -> Dupire -> reprice
```

See [`examples/README.md`](examples/README.md) for the full list.

## Using it as a library

Build contracts with the fluent builder:

```rust
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::Instrument;

let option = EquityOptionBuilder::new()
    .spot(100.0)
    .strike(100.0)
    .flat_vol(0.30)
    .flat_rate(0.05)
    .dividend_yield(0.02)
    .years_to_maturity(1.0)
    .vanilla(PutOrCall::Call)
    .engine(Engine::FiniteDifference)
    .build();

println!(
    "pv {:.6}  delta {:.4}  vanna {:.4}  charm {:.4}",
    option.npv(), option.delta(), option.vanna(), option.charm()
);
```

...or deserialize the same JSON the CLI consumes:

```rust
use rustyqlib::equity::vanilla_option::EquityOption;
use rustyqlib::core::data_models::EquityOptionData;

let contract: EquityOptionData = serde_json::from_str(json)?;
let option = EquityOption::from_json(&contract);
```

Lower-level building blocks are exported directly: `YieldCurve`, `VolSurface`,
`DayCountConvention`, the `Payoff` trait, Dupire `LocalVol`, `HestonParams`, and
the engine modules (`blackscholes`, `binomial`, `finite_difference`, `montecarlo`).

## Design principles

- **Discount factors are state, rates are views** — curves store pillar dfs;
  zero/forward rates in any compounding are derived on demand. Vol surfaces
  canonicalize every quoting style into per-expiry smiles with total-variance
  time interpolation.
- **One payoff trait, every engine** — payoffs implement `payoff(spot, strike)`
  and (for path dependence) `path_payoff(path, strike)`; adding a payoff makes it
  price on every compatible engine without engine changes.
- **Validated numerics** — golden values against independently coded oracles,
  parity and replication identities at 1e-10, cross-engine agreement tests, and
  bit-reproducible Monte Carlo. Engines refuse unsupported combinations with a
  clear error instead of silently mispricing.

## Roadmap

- Andersen QE scheme and American exercise (LSMC) under Heston; 2-D ADI finite
  difference for stochastic vol
- Barrier rebates, double/window barriers, seasoned Asians
- Rates: curve bootstrapping from deposits/FRAs/swaps onto the core curve type,
  swaps and swaptions; FX (Garman-Kohlhagen)
- Stulz closed forms for two-asset best-of/worst-of; per-asset smiles and
  path-dependent multi-asset payoffs; SVI smile parameterization with
  no-arbitrage checks; pathwise / likelihood-ratio Greeks
- `Result`-based error API for the library surface

## License

MIT — see [License](License).
