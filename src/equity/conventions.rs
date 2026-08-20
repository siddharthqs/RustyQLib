//! Shared equity-side conventions.
//!
//! Two families of constants that were previously scattered as
//! literals across the engines and contracts:
//!
//! - the day count behind every ad-hoc `(end - start).num_days() /
//!   365.0` year fraction, now a single function so an Act/360 book
//!   would be a one-line change instead of a hunt;
//! - the central-difference Greek bump sizes shared by the lattice and
//!   finite-difference engines (each previously kept its own copies,
//!   with comments claiming they were shared).

use crate::core::daycount::DayCountConvention;
use chrono::NaiveDate;

/// The day count for equity time measures — maturities, dividend
/// ex-dates, observation schedules — wherever the contract does not
/// carry an explicit convention. Flat curves and surfaces built by the
/// equity builder use the same convention, which is what keeps the
/// hardcoded fractions and the curve objects consistent today.
pub const EQUITY_DAY_COUNT: DayCountConvention = DayCountConvention::Act365;

/// Year fraction from `start` to `end` under [`EQUITY_DAY_COUNT`].
/// Negative if `end` is before `start`.
pub fn year_fraction(start: NaiveDate, end: NaiveDate) -> f64 {
    EQUITY_DAY_COUNT.year_fraction(start, end)
}

// ── Greek bump sizes (lattice + finite-difference engines) ─────────────
// Central differences of already-smooth grid/tree solutions: small
// enough to be local, large enough that the solve's own discretization
// error does not dominate the difference.

/// Parallel vol bump for vega / vanna / zomma.
pub const VOL_BUMP: f64 = 1e-3;
/// Parallel rate bump for rho.
pub const RATE_BUMP: f64 = 1e-4;
/// Vol bump for volga: a larger step tempers the roundoff
/// amplification of a second difference against the solve's own
/// discretization error.
pub const VOLGA_BUMP: f64 = 1e-2;
/// Relative spot bump (fraction of spot) for charm.
pub const SPOT_REL_BUMP: f64 = 1e-3;

// ── Degenerate-bump floors ─────────────────────────────────────────────

/// Floor for the volatility read through a [`BumpedMarket`]
/// (crate::equity::bump::BumpedMarket): a vega down-bump on a tiny-vol
/// option would otherwise hand the engines a non-positive vol and
/// panic layers under `try_npv`'s `Result` promise. Repricing at the
/// floor slightly biases the affected stencil leg, but only in the
/// region where a central difference was already ill-posed.
pub const MIN_BUMPED_VOL: f64 = 1e-4;

/// Relative floor (fraction of the base spot) for spot values read
/// through a bumped view: a deep spot-down stress on a cash-dividend
/// name values the option at an (escrowed) spot pinned just above zero
/// instead of feeding the engines a negative spot.
pub const MIN_BUMPED_SPOT_FRAC: f64 = 1e-6;

// ── Unit sanity bands ──────────────────────────────────────────────────
// The classic percent-vs-decimal slip (`volatility: 20` meaning 20%,
// `risk_free_rate: 5` meaning 5%) parses cleanly and prices absurdly.
// These bands are far outside anything a real market produces, so a
// violation is a unit error, not an exotic input.

/// Largest plausible implied volatility as a decimal (300%).
pub const MAX_SANE_VOL: f64 = 3.0;
/// Largest plausible magnitude for a rate-like input as a decimal (50%).
pub const MAX_SANE_RATE: f64 = 0.5;

/// Reject a volatility outside `(0, MAX_SANE_VOL]` with a
/// percent-vs-decimal hint. Positivity/finiteness is the caller's
/// domain check; this only screens the unit slip.
pub fn check_vol_band(field: &str, vol: f64) -> Result<(), crate::core::errors::RustyQLibError> {
    if vol > MAX_SANE_VOL {
        return Err(crate::core::errors::RustyQLibError::invalid_input(
            field,
            format!(
                "{field} = {vol} exceeds {MAX_SANE_VOL} (300%): volatilities are decimals \
                 (0.2 = 20%), not percentages"
            ),
        ));
    }
    Ok(())
}

/// Reject a rate-like input (risk-free rate, dividend yield, borrow
/// cost) with magnitude above [`MAX_SANE_RATE`], with a
/// percent-vs-decimal hint.
pub fn check_rate_band(field: &str, rate: f64) -> Result<(), crate::core::errors::RustyQLibError> {
    if rate.abs() > MAX_SANE_RATE {
        return Err(crate::core::errors::RustyQLibError::invalid_input(
            field,
            format!(
                "{field} = {rate} has magnitude above {MAX_SANE_RATE} (50%): rates are \
                 decimals (0.05 = 5%), not percentages"
            ),
        ));
    }
    Ok(())
}
