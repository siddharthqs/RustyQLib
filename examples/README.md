# Runnable examples

One file per equity product. Each builds the product in Rust (no JSON),
prices it on every applicable engine and model, prints NPV and Greeks in a
single table, and verifies the identities that must hold for that product.

```bash
cargo run --release --example vanilla_option
```

Release mode matters: the Monte Carlo examples run 50k–100k paths.

| Example | Product | What it demonstrates |
|---|---|---|
| `vanilla_option` | European / American vanilla | all four engines side by side; American early-exercise premium; put-call parity; implied vol round trip; analytic vs bumped Greeks |
| `binary_option` | Cash- and asset-or-nothing digitals | both settlement types; cash scaling; the replication identity across all Greeks; digital risk blowing up near expiry |
| `barrier_option` | All eight barrier types | analytic vs FD vs bridge-corrected MC; in-out parity; barrier sweep; skew effect under local vol |
| `asian_option` | Asian options | geometric (exact) vs arithmetic (Turnbull-Wakeman); the geometric control variate cutting MC variance ~15x; floating strike; AM-GM ordering |
| `forward_start_option` | Forward-start options | Rubinstein closed form vs MC; **the forward smile** (Heston vs Black-Scholes); strike-fraction and fixing-date sweeps |
| `autocallable_option` | Autocallable note with coupon | GBM vs local vol vs Heston; coupon / barrier / frequency sensitivity; exact degenerate cases |
| `heston_option` | Heston stochastic vol | semi-analytic characteristic function vs MC; binaries and barriers; **how rho and vol-of-vol shape the smile** |
| `rainbow_option` | Multi-asset rainbows | best-of, worst-of, spread (Kirk), basket (moment matching), exchange (Margrabe); correlation sweep; per-asset Greeks |
| `local_vol_calibration` | Local vol workflow | quotes -> implied vols -> surface -> Dupire -> reprice, end to end with checks at each step |
| `dividends_and_borrow` | Carry inputs | borrow cost as carry; escrowed vs jump dividend models per engine; where the difference matters |

## Reading the output

- Engines that refuse a combination by design (analytic + American, tree +
  barrier, FD + Heston) print `unsupported` with the reason instead of
  aborting — the tables double as a support matrix.
- `std err` is populated for Monte Carlo rows only.
- `[OK ]` / `[BAD]` lines are identity checks with an explicit tolerance.

## Building your own

All ten use [`EquityOptionBuilder`](../src/equity/builder.rs):

```rust
let option = EquityOptionBuilder::new()
    .spot(100.0)
    .strike(100.0)
    .flat_vol(0.30)
    .flat_rate(0.05)
    .dividend_yield(0.02)
    .years_to_maturity(1.0)
    .vanilla(PutOrCall::Call)
    .engine(Engine::MonteCarlo)
    .paths(100_000)
    .build();

println!("{} +/- {}", option.npv(), montecarlo::npv_with_stats(&option).std_err);
```

Swap `.vanilla(...)` for `.binary(...)`, `.barrier(...)`, `.asian(...)`,
`.forward_start(...)` or `.autocallable(...)`; swap `.engine(...)` and
`.model(...)` to change pricer and dynamics. JSON contract equivalents for
the CLI live in [`../src/examples/`](../src/examples/).
