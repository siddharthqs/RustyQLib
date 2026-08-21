# `src/validation` — Model Validation

Runtime checks that measure **model quality on given data**, complementing the
unit tests that pin implementation correctness.

The distinction is the point of the module:

|  | Unit test | Validation check |
|---|---|---|
| Asks | "does this code compute what it claims, exactly, forever?" | "is this calibration usable, on *these* inputs, *today*?" |
| Inputs | fixed | live |
| Answer | binary pass/fail | numbers with tolerances and context |
| Runs | in CI | at runtime |
| Output | a test result | a serializable report |

Statistical checks report **z-scores rather than naked pass/fail** — they say
how surprised you should be. Reports are serializable, so a validation run is
an auditable artifact, the same document philosophy as the rest of the library.

---

## Checks

### `martingale.rs`

Forward recovery under simulated dynamics, against externally supplied target
forwards, in z-scores. If a Monte Carlo engine cannot reproduce the forward it
was handed, nothing else it computes is trustworthy — this is the cheapest
possible test of that, and the first one to run on a new model or scheme.

```rust
use rustyqlib::validation::{martingale_report, MartingaleConfig};

let report = martingale_report(&paths, &target_forwards, MartingaleConfig::default());
for check in &report.checks {
    println!("t={:.3}  z={:+.2}  {}", check.t, check.z_score,
             if check.passed { "ok" } else { "SUSPECT" });
}
```

### Re-exported from where they grew up

These belong to this family and are indexed here, so there is one answer to
"how do I know these numbers are right":

| Report | Source | Answers |
|---|---|---|
| `SurfaceDiagnostics` | `core::vols::VolSurface::diagnostics` | Static-arbitrage findings on an implied surface (butterfly / calendar) |
| `RepairReport` | `equity::models::surface_repair::repair_arbitrage` | What the repair changed, and by how much |
| `UsabilityReport` | `equity::models::usability` | Local-vol round-trip repricing, clamp and guard-fallback fractions, trusted region |

VaR backtesting lives in [`risk`](../risk).

## Planned

- **`convergence`** — empirical convergence order of each engine, measured
  rather than assumed.
- **`stability`** — input-bump amplification ratios.
- **`sensitivity`** — Greeks and parity consistency across engines.

---

## Why this exists

Every pricer in this library is cross-checked in the test suite against
independent oracles, put-call parity, replication identities and cross-engine
agreement. That proves the *code* is right. It does not prove the *calibration
you ran this morning* is usable, that your surface is arbitrage-free, or that
your simulation respects its martingale property on today's inputs. This
module is for the second question.

See the [main README](../../README.md) for the validation philosophy in
context.
