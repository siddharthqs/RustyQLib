# Examples

One runnable JSON file per product. Price any of them with:

```bash
rustyqlib file --input src/examples/EQ/<file>.json --output out.json
```

## Equity (`EQ/`)

| File | Product | Shows |
|---|---|---|
| `equity_option.json` | European vanilla call/put + future | flat rate vs full zero curve, strike x expiry vol surface |
| `binary_option.json` | Cash- and asset-or-nothing digitals | `binary_type`, `cash_amount`, analytic / MC / FD engines |
| `barrier_option.json` | Knock-in / knock-out barriers | Reiner-Rubinstein vs bridge-corrected MC, local vol barrier |
| `asian_option.json` | Asian options | arithmetic (Turnbull-Wakeman + control-variate MC), geometric, floating strike |
| `forward_start_option.json` | Forward-start options | strike fixed at a future date; **Heston** vs Black-Scholes forward smile |
| `autocallable_option.json` | Autocallable note with coupon (rebate) | observation schedule, knock-in protection, **local vol** vs GBM |
| `heston_option.json` | Vanillas + barrier under Heston | semi-analytic CF pricing vs Monte Carlo |
| `rainbow_option.json` | Multi-asset rainbows | worst-of (MC), exchange (Margrabe), spread (Kirk), basket (moment matching) |
| `dividends_borrow.json` | Carry inputs | cash dividends + borrow cost across analytic, FD (American) and MC (barrier) |
| `eq2.json`, `equity_forward.json` | Forward / future contracts | linear products |

## Rates (`IR/`) and commodities (`CO/`)

| File | Product |
|---|---|
| `IR/ir1.json` | Deposit pricing |
| `build/build_ir_curve.json` | Curve bootstrap input (build mode) |
| `CO/cmdty_option.json` | Commodity option (legacy schema) |

Monte Carlo outputs include a `std_err` field; rainbow outputs report
per-asset `deltas` and `vegas`.
