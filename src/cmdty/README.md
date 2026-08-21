# `src/cmdty` — Commodity Derivatives

Swaps, average-price options, spread options and swaptions on a forward curve
— with the distribution model chosen by the vol quote, so negative prices are
a supported case rather than a crash.

---

## Products

| Product | File | What it is |
|---|---|---|
| `CommoditySwap` | `swap.rs` | Fixed-for-floating. Each period cash-settles the **arithmetic average** of the daily index over the period's business days against a fixed price |
| `CommodityBasisSwap` | `basis_swap.rs` | Floating-for-floating differential (WTI–Brent, location or quality basis); each leg averages over its own pricing calendar, with a spread on the received leg |
| `CommodityOption` | `option.rs` | European option on a future — Black-76, premium up front or futures-style margined |
| `AveragePriceOption` | `apo.rs` | Asian-style APO on the same average as the swap; discrete moment matching (Levy), with realized fixings folded into an adjusted strike |
| `CommoditySwaption` | `swaption.rs` | European option to enter a swap; Black-on-par × the settlement annuity, exact under one-factor flat-vol dynamics |
| `CommoditySpreadOption` | `spread_option.rs` | Crack, spark, location or calendar spread; Kirk's approximation for lognormal legs, **exact Bachelier** for normal legs |
| `CommodityForwardCurve` | `forward_curve.rs` | The strip of forward prices everything projects from, linearly interpolated between pillars |

Partially realized swap periods blend published `PriceFixings` with curve
forwards. Discounting comes from a `YieldCurve` in [`core`](../core).

## Distribution models

This is what makes the module different from equity. Commodity underlyings can
print **negative** — Waha and AECO basis routinely, WTI in the April 2020
dislocation — so lognormality is not a safe default. The model is selected by
the vol quote (`CommodityVol` in `vol.rs`):

- **Black-76 lognormal** — the default; a bare `f64` vol.
- **Shifted lognormal** — a displacement that admits prices above `-shift`.
- **Bachelier normal** (`bachelier.rs`) — for underlyings that genuinely go
  negative.

Two smile/term-structure models feed that dispatch:

- **`ShiftedSabr`** (`sabr.rs`) — generates the shifted-lognormal quote per
  strike, so a whole smile prices consistently through one code path.
- **`ClewlowStrickland`** (`clewlow_strickland.rs`) — does the same across
  maturities, capturing the **Samuelson effect** (volatility rising as a
  contract approaches expiry). It additionally supplies the cross-maturity
  covariances that the APO and swaption use in their `price_cs` variants.

## Example

```rust
use rustyqlib::cmdty::option::CommodityOption;
use rustyqlib::cmdty::vol::CommodityVol;
use rustyqlib::core::trade::PutOrCall;

// a normal-model option — safe if the underlying can print through zero
let opt = CommodityOption::new(
    /* forward */ 2.85,
    /* strike  */ 3.00,
    /* expiry  */ 0.5,
    PutOrCall::Call,
    CommodityVol::Bachelier(0.85),   // absolute vol, price units
);

println!("pv {:.6}", opt.price(&discount_curve)?);
```

Swap, APO and spread-option pricing are demonstrated in
[`examples/`](../../examples). See the [main README](../../README.md) for the
CLI and JSON contract schema.
