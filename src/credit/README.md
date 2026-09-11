# credit

- **`curve.rs`** — `CreditCurve`: piecewise-constant hazard rates with
  survival and default probabilities, the term structure the risky bond
  pricers, the jump-to-default convertible and the CDS price off.
- **`cds.rs`** — `CreditDefaultSwap`: premium and protection legs with the
  ISDA standard model's closed-form segment integrals (accrual on default
  included), par spread, NPV from either side, points upfront, risky PV01,
  and the flat-hazard conversion between par spread and upfront.
- **`bootstrap.rs`** — the hazard curve from par-spread quotes, one segment
  per maturity, so every quoted contract reprices to zero.

Cross-checked against QuantLib 1.43's `IsdaCdsEngine` (see the test in
`cds.rs`). Depends on `core` only; `bonds` and `hybrid` depend on it.
