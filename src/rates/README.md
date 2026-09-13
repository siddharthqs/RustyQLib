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
| `BermudanSwaption` | `bermudan_swaption.rs` | The same right on any of several dates, exercising into the remaining swap, on the Hull-White grid |
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
| Hull-White grid | `hw_grid.rs` | Backward induction on the `x = r - alpha(t)` state grid between event dates, exact forward-measure transitions; Bermudan swaptions and callable bonds share it |
| Gaussian1d | `gaussian1d.rs` | European swaptions by quadrature and Bermudans by deflated backward induction on any `Gaussian1dModel` (Hull-White, Markov functional) |
| FD Hull-White | `fd_hull_white.rs` | The state PDE by a Crank-Nicolson theta scheme with Rannacher steps after events; same event-closure API as the grid, so Europeans, Bermudans and callables cross-check the integration grid |
| FD G2++ | `fd_g2pp.rs` | The two-factor PDE by Hundsdorfer-Verwer ADI with the explicit mixed term — the two-factor model's Bermudan engine; collapses to the Hull-White PDE when a factor has no vol |
| MC Hull-White | `mc_hull_white.rs` | Monte Carlo with exact bivariate steps of the state and its integral between event dates (no time grid, no discretization bias), antithetic paths and standard errors; `simulate` takes any payoff on the path's rates and discount factors, `cap_floor` (and `CapFloor::npv_mc_hull_white`) prices the caplet strip |

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
| `HullWhite` | `hull_white.rs` | `dr = (theta(t) - a(t) r)dt + sigma(t) dW`; fitted **exactly** to a `YieldCurve`, exact transition; constant coefficients (`new`), piecewise sigma (`with_piecewise_sigma`, QuantLib's **GSR**, alias `Gsr`) or piecewise `a` and sigma (`generalized`, **GeneralizedHullWhite**) |
| `CoxIngersollRoss` | `cir.rs` | `dr = a(b - r)dt + sigma sqrt(r) dW`; affine bonds, noncentral chi-square bond options (so Jamshidian swaptions and caps), full-truncation Euler step |
| `ExtendedCir` | `cir.rs` | **CIR++**: a CIR process plus the deterministic shift that fits the curve exactly; bonds and options inherited from CIR by a deterministic rescaling |
| `BlackKarasinski` | `black_karasinski.rs` | Lognormal `d ln r = (theta(t) - a ln r)dt + sigma dW` on a curve-fitted trinomial tree (Hull-White two-stage); bond options, European and Bermudan swaptions by tree induction |
| `G2pp` | `g2pp.rs` | Two-factor Gaussian `r = x + y + phi(t)` with correlated factors; closed-form bonds and bond options, exact 2-D transition, European swaptions by the one-dimensional integral of Brigo-Mercurio 4.2.3, Bermudans by the FD engine; `calibrate_g2pp_vols` fits the two vols with `a`, `b`, `rho` fixed, `calibrate_g2pp` all five |
| `MarkovFunctional` | `markov_functional.rs` | Hunt-Kennedy-Pelsser with a terminal-bond numeraire, calibrated backward to a coterminal column — flat normal vols (`calibrate`) or SABR smiles (`calibrate_with_smiles`) — so every calibrating swaption reprices at every strike |
| `RateSabr`, `SabrSwaptionCube` | `sabr.rs` | Hagan SABR for rates: normal (any rate sign at `beta = 0`) and shifted-lognormal vols, per-node calibration of `(alpha, rho, nu)` at chosen `beta` and shift, and the expiry-tenor cube with bilinear parameter interpolation; `Swaption::npv_sabr` and `CapFloor::npv_with_smile` price off it |
| `ZabrSmile` | `zabr.rs` | Andreasen-Huge ZABR: SABR with a vol-of-vol exponent `gamma`, priced arbitrage-free — the short-maturity smile from the geodesic distance of the two-factor metric (shooting on the Hamiltonian flow), the local vol it implies, and an implicit Dupire solve in strike; `calibrate` fits `(alpha, rho, eps)` at chosen `beta`, `gamma`, shift to normal-vol quotes |

**Gaussian1d framework** (`gaussian1d.rs`). QuantLib's `Gaussian1dModel`
as a trait: a Gaussian Markov driver under the terminal-bond measure,
described by its numeraire and zero bonds as functions of the state.
Hull-White implements it through `gaussian1d(numeraire_time)`, the
Markov functional model natively; the `engines/gaussian1d.rs` engines
price European (quadrature) and Bermudan (deflated backward induction)
swaptions on either.

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
by hand or from `Swaption::to_quote`. `calibrate_hull_white_piecewise`
bootstraps a **piecewise-constant** `sigma` to a column of expiries with
`a` fixed: each expiry pins the sigma of the interval ending there, one
quote per expiry hit to tolerance, several fitted in least squares — the
Bermudan desk's setup.

**Vol surfaces** (`vol_surface.rs`, `caplet_vol.rs`).
`SwaptionVolSurface` holds ATM vols on an expiry × tenor grid (normal,
Black or shifted), bilinear inside and flat outside, and
`calibration_quotes` / `usd_standard_quotes` turn every node into a
dated ATM swaption priced at its vol — a screen straight into the
Hull-White calibrators. `strip_caplet_vols` turns a ladder of flat cap
vols at one strike into a `CapletVolCurve`, piecewise constant in
fixing time, each longer cap's new caplets solved by bisection given
the shorter caps; `CapFloor::npv_with_caplet_vols` prices off it and
every cap of the ladder reprices.

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

Cap/floor quotes in the Hull-White calibrator, SABR parameter
sensitivities (vega by node), cash-settled swaptions and CMS,
cross-currency and inflation instruments in the multi-curve set, and a
hybrid equity-rate consumer of the simulation contract.
