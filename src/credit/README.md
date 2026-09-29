# credit

- **`curve.rs`** — `CreditCurve`: piecewise-constant hazard rates with
  survival and default probabilities, the term structure the risky bond
  pricers, the jump-to-default convertible and the CDS price off.
- **`cds.rs`** — `CreditDefaultSwap`: premium and protection legs with the
  ISDA standard model's closed-form segment integrals (accrual on default
  included), par spread, NPV from either side, points upfront, risky PV01,
  and the flat-hazard conversion between par spread and upfront.
- **`cds_option.rs`** — `CdsOption`: the European payer / receiver option on
  a forward-starting CDS. The forward risky annuity and protection come off
  the same segment integrals read at a valuation date before the contract
  starts; front-end protection covers the option's own life, so the
  no-knockout market standard prices as Black (or Bachelier) on the
  loss-adjusted forward spread `(P + FEP) / A`. Implied vol inverts a
  premium in any of the three quote conventions.
- **`bootstrap.rs`** — the hazard curve from par-spread quotes, one segment
  per maturity, so every quoted contract reprices to zero.

Cross-checked against QuantLib 1.43's `IsdaCdsEngine` (see the test in
`cds.rs`). Depends on `core`, plus `rates::engines::black` for the shared
rate-option kernel and its implied-vol inversion; `bonds` and `hybrid`
depend on it.
