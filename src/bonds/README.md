# `src/bonds` — Fixed Income

Bond instruments, credit, and discount-curve construction.

Conventions follow the rest of the library: each instrument carries its own
day-count convention for accrual, the curve carries its own for date-to-time
conversion, and **valuation dates are always explicit inputs** — nothing here
reads the wall clock.

---

## Curve construction

The short end is pinned by money-market instruments, the long end by bonds,
and everything bootstraps into the library-wide `YieldCurve`.

| Piece | File | Notes |
|---|---|---|
| `Deposit` | `deposit.rs` | Money-market deposit |
| `Fra` | `fra.rs` | Forward rate agreement |
| `CurveInstrument` | `bootstrap.rs` | The abstraction they share |
| `bootstrap_curve` | `bootstrap.rs` | Sequential bootstrapper |
| `BillQuote`, `BondQuote` | `quotes.rs` | Quoted pillars; the bootstrapped Treasury curve **exactly reprices its input quotes** |

## Instruments

- **`FixedRateBond`** (`fixed_rate_bond.rs`) — US-Treasury street analytics:
  accrued interest, price/yield, duration, convexity, DV01, and curve pricing.
  `us_corporate()` switches to 30/360 with T+2 settlement. Supports step-up
  coupons and sinking funds.
- **`TreasuryBill`** (`bills.rs`) — discount-quoted, with the bank-discount
  and bond-equivalent yield conventions.
- **`FloatingRateNote`** (`frn.rs`) — projected off the curve, with discount
  margins.
- **`BondFuture`** (`futures.rs`) — CME conversion factors, invoice prices,
  gross and net basis, implied repo, and the cheapest-to-deliver.
- **`ConvertibleBond`** (`convertible/`) and **`ConvertiblePreferred`**
  (`preferred.rs`) — both `ConvertibleInstrument`s priced through the
  blanket `ConvertiblePricing` trait; two credit treatments behind one
  `CreditModel` trait, on one equity tree: Tsiveriotis-Fernandes (equity/cash split, flat credit
  spread) and jump to default (hazard rate, stock absorbed at zero on
  default, recovery on face, borrow cost in the drift), with soft calls,
  puts, contingent conversion, a coupon make-whole on calls, a
  fundamental-change make-whole table with the par put, mandatory
  conversion (PEPS/DECS share schedules), discrete cash dividends as the
  exact ex-date jump on either engine with threshold dividend protection,
  parity/premium
  analytics, and implied credit spreads, hazard rates and volatilities. `convertible/fd.rs` solves the same two models by finite
  differences (Crank-Nicolson in log-spot on the tree's event grid), giving
  price, delta and gamma from one solve, and vega, theta, parallel and
  key-rate DV01s and the spread or hazard DV01 by bumping. The volatility
  is pluggable: flat, a Dupire local-vol grid from an implied surface, or
  any custom function of share price and time; `dejump.rs` strips the
  default jump out of listed implied vols so a surface can feed the
  jump-to-default model consistently.

## Credit and optionality

- **`spreads.rs`** — z-spread, spread DV01, G-spread, asset-swap spread.
- **`credit.rs`** — hazard-rate pricing and credit-curve bootstrapping.
- **`callable.rs`** — Hull-White option model for calls, puts and
  make-wholes; yield-to-call, yield-to-put and yield-to-worst.
- **`schedule.rs`** — coupon schedule generation (forward/backward with
  stubs), business-day adjusted against the holiday calendars in
  [`core`](../core).

## Example

```rust
use rustyqlib::bonds::fixed_rate_bond::FixedRateBond;
use chrono::NaiveDate;

let bond = FixedRateBond::us_treasury(
    NaiveDate::from_ymd_opt(2030, 5, 15).unwrap(),  // maturity
    0.0425,                                          // coupon
    100.0,                                           // face
);

let asof = NaiveDate::from_ymd_opt(2026, 8, 21).unwrap();
let ytm  = bond.yield_from_price(asof, 98.75)?;
println!("ytm {:.4}%  dur {:.3}  dv01 {:.4}",
         100.0 * ytm, bond.duration(asof, ytm)?, bond.dv01(asof, ytm)?);
```

Curve bootstrapping, Treasury analytics and futures basis are demonstrated in
[`examples/`](../../examples) (`treasury_curve.rs`, `us_treasury_bond.rs`,
`fed_funds_future.rs`, `swaps.rs`). The CLI can fetch the live US Treasury par
yield curve — see the [main README](../../README.md).
