# hybrid

Instruments that are a bond, an equity claim and a credit exposure at once.

- **`ConvertibleBond`** (`convertible/`) and **`ConvertiblePreferred`**
  (`preferred.rs`) — both `ConvertibleInstrument`s priced through the
  blanket `ConvertiblePricing` trait; two credit models behind one
  `CreditModel` trait (Tsiveriotis-Fernandes with an equity/cash split and a
  flat spread; jump to default with a hazard rate, the stock absorbed at
  zero, recovery on face and borrow in the drift) on one CRR tree and one
  Crank-Nicolson grid. Soft calls and puts, contingent conversion, coupon
  and fundamental-change make-wholes, mandatory conversion, discrete cash
  dividends with threshold protection, parity/premium analytics, implied
  credit spreads, hazard rates and volatilities.
- **`convertible/fd.rs`** — the grid engine: price, delta and gamma from one
  solve; vega, theta, parallel and key-rate DV01s and the spread or hazard
  DV01 by bumping; pluggable volatility (flat, Dupire local vol, custom).
- **`convertible/dejump.rs`** — strips the default jump out of listed
  implied vols so a surface can feed the jump-to-default model.

The module sits above `bonds` (the fixed-rate chassis), `equity` (implied
and local volatility) and `credit` (hazard curves). A QuantLib cross-check
of the Tsiveriotis-Fernandes tree is in `docs/benchmarks/`.
