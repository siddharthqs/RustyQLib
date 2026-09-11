# `src/rates` — Interest Rates

Contracts, pricing engines, short-rate models, and the leg machinery that
binds them — the same layout as `src/equity`.

```
rates/
├── contracts/   what you can trade   (swaps, futures, swaptions, caps/floors)
├── engines/     how it gets priced   (Jamshidian on one-factor affine models)
├── models/      dynamics             (Vasicek, Hull-White, CIR; HW calibration)
└── *.rs         the spine: leg (schedules, leg PVs), overnight (fixings)
```

Linear products discount off `YieldCurve`s directly. Optional products
take a short-rate model and an **anchor** curve that maps their dates to
the year fractions the models work in — for Hull-White, the curve it was
fitted to, so `npv_hull_white(&model)` needs nothing else.

---

## Contracts (`contracts/`)

| Product | File | Prices as |
|---|---|---|
| `VanillaSwap` | `vanilla_swap.rs` | Fixed vs floating; dual-curve, seasoned with a fixing, par rate, DV01 |
| `OvernightIndexSwap` | `ois.rs` | Fixed vs daily-compounded overnight, payment lag |
| `BasisSwap` | `basis_swap.rs` | Float vs float with a spread; fair spread |
| `FedFundsFuture` | `fed_funds_future.rs` | ZQ: arithmetic average of EFFR, FOMC step analytics |
| `SofrFuture` | `sofr_future.rs` | SR1 / SR3, convexity helper |
| `Swaption` | `swaption.rs` | European option to enter a `VanillaSwap`, under a short-rate model |
| `CapFloor` | `cap_floor.rs` | Strip of caplets / floorlets over a schedule, under a short-rate model |

**Swaption.** Bond-option equivalence: a payer is a put on the fixed leg
(coupons plus redemption) struck at the notional, a receiver the call. The
settlement lag between exercise and the swap's adjusted effective date is
carried exactly — the notional is exchanged at the swap start, so each
Jamshidian piece is an exchange option between two zero bonds. Helpers:
`forward_swap_rate` (the ATM strike), `annuity`, and `to_quote` to feed the
calibrator with a market price.

**CapFloor.** Owns the schedule like a floating leg (frequency, day
count, calendar, roll convention). Each period's caplet is `1 + K tau`
zero-bond puts struck at `1/(1 + K tau)`; `npv` sums the strip,
`caplet_values` returns the per-period breakdown with forwards. A period
that has already fixed is excluded (market convention for a spot-starting
cap's first caplet) unless its realized rate is supplied through
`npv_with_fixing`. `atm_strike` is the schedule's forward swap rate, where
cap and floor are worth the same. Cap minus floor equals the payer swap on
the same schedule exactly.

**Leg conventions.** Every product builds its legs through
`LegSchedule` (`schedule.rs`): a `StubConvention` (short or long, front
or back) and a `RollConvention` (the roll origin's day, month-end, a
fixed day of month, or IMM) on top of the calendar, business-day rule
and payment lag. Constructors give the market defaults and `with_stub`
/ `with_roll` change them. A floating leg that resets more often than
it pays (`with_float_reset` on a swap, `BasisSwapLeg::with_reset`)
combines its sub-periods by a `CompoundingMethod` — ISDA straight, flat
or spread-exclusive compounding, or none — and prices seasoned off a
`RateFixings` history keyed by each reset's accrual start
(`pv_with_fixings` on the swap, `pv_with_fixing_history` on the basis
swap). An OIS carries an `OvernightConvention` (lookback, lockout,
observation shift); with one set, or once fixings are supplied through
`pv_with_fixings`, each period compounds day by day off the fixings and
the curve's overnight forwards.

**Market vols.** Both optional products also price straight from a
quoted vol and read a premium back as one. `RateVol` is the quote:
`Normal` (Bachelier, absolute rate units per √year — the market
standard), `Lognormal` (Black-76) or `ShiftedLognormal`. On a
`Swaption`, `npv_black` is `annuity * kernel`, `implied_normal_vol` and
`implied_black_vol` invert a premium, `implied_normal_vol_hull_white`
reads the model price as the normal vol it implies, and
`to_quote_from_vol` turns a screen vol into a calibration quote. On a
`CapFloor` the same at one flat vol for the strip: `npv_black`,
`implied_flat_normal_vol`, `implied_flat_black_vol`,
`implied_normal_vol_hull_white`.

## Engines (`engines/`)

| Engine | File | Use it for |
|---|---|---|
| Jamshidian | `jamshidian.rs` | Coupon-bond options, European swaptions, caplets and floorlets on any `OneFactorAffine` model |
| Black / Bachelier | `black.rs` | Market-quote formulas on the forward rate times the annuity, and the implied-vol inversions (normal, Black, shifted Black) |

Everything reduces to the model's closed-form zero-bond option: a
coupon-bond option decomposes at the critical rate `r*` where the bond is
worth the strike; a swaption is a coupon-bond option struck at the
notional; a caplet is a scaled zero-bond put. The `_settled` variants
take a settlement after the expiry (the swaption's settlement lag) and
use zero-bond exchange options instead.

## Models (`models/`)

| Model | File | Dynamics |
|---|---|---|
| `Vasicek` | `vasicek.rs` | `dr = a(b - r)dt + sigma dW`; closed forms, exact transition, endogenous curve |
| `HullWhite` | `hull_white.rs` | `dr = (theta(t) - a r)dt + sigma dW`; fitted **exactly** to a `YieldCurve`, exact transition |
| `CoxIngersollRoss` | `cir.rs` | `dr = a(b - r)dt + sigma sqrt(r) dW`; affine bonds, full-truncation Euler step, no option layer |

Two traits: `ShortRateModel` is the simulation contract (initial rate,
one-step transition from a normal draw, zero-bond reconstitution from the
state) — what a Monte Carlo engine or an equity-rate hybrid needs.
`OneFactorAffine` adds the analytic zero-bond (exchange) option the
Jamshidian engine builds on.

**Calibration** (`calibration.rs`). Hull-White to European swaption
prices: the QuantLib swaption-helper workflow. `calibrate_hull_white` fits
`(a, sigma)` jointly, `calibrate_hull_white_sigma` fits `sigma` with the
mean reversion fixed (the usual desk setup — coterminal swaptions identify
`a` weakly). Nelder-Mead over the log parameters on the sum of squared
relative price errors; the fit reports the price RMSE and refuses to
return a non-converged model. Quotes are prices per unit notional, built
by hand or from `Swaption::to_quote`.

## Multi-curve calibration (`multicurve.rs`)

`MultiCurveBuilder` solves a discount curve and any number of forecast
curves **together** from quoted instruments — `RateInstrument`:
deposits, FRAs, SOFR futures (price less a convexity), par swaps, par
OIS and basis swaps. Each instrument pins one pillar (its maturity) on
one named curve and prices off the full set, discounting on the
builder's discount curve; the unknowns are the continuous zero rates at
every pillar of every curve, solved by Newton with a finite-difference
Jacobian and a backtracking line search until every quote reprices to
1e-12. A single curve is the case where discount and forecast share a
name; on deposits and FRAs it reproduces the sequential
`bonds::bootstrap_curve` exactly.

The result, `MultiCurve`, keeps the Jacobian. `quote_sensitivities`
turns any pricing closure into the trade's PV01 to each market quote
(`(J^-1)^T dPV/dz`, one basis point per instrument with every curve
re-solved) — the bucketed risk a desk hedges with. A calibration
instrument's own sensitivity is exactly its annuity to its own quote and
zero to the others; an off-pillar trade spreads over its neighbours.

## Where the models are used

- `bonds/callable.rs` — callable, puttable and make-whole bonds by
  backward induction on the Hull-White state grid; OAS, option value,
  effective duration and convexity.
- `examples/short_rate_models.rs` — fit, swaption and cap products, and
  the simulation contract; `examples/corporate_bond.rs` — calibrate to
  ATM swaptions, then price a callable.

## Not yet here

Cap/floor quotes in the Hull-White calibrator, a SABR smile on swaption
vols, time-dependent `sigma`, two-factor (G2++) and lognormal
(Black-Karasinski) models, CIR bond options, cross-currency and
inflation instruments in the multi-curve set, and a hybrid equity-rate
consumer of the simulation contract.
