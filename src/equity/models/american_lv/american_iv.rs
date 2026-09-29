//! The American implied volatility `sigma^A`, the vega ratio
//! `nu^A / nu^E` of the engine-independence proposition, and the industry
//! de-Americanization fixed point.
//!
//! ```text
//! sigma^A(K, T):  the constant vol at which the constant-vol AMERICAN price on the
//!                 quote's own mesh equals the market (or model) American price;
//!                 defined on N = { P > psi(S0) }, where nu^A = dP^A/dsigma > 0.
//! Newton:         sigma <- sigma - (P^A(sigma) - P) / nu^A(sigma), nu^A = sum g (adjoint)
//!                 or a central bump, safeguarded by a bracket [lo, hi] inside
//!                 [SIGMA_LO, SIGMA_HI] = [0.01, 3.0] with bisection when a Newton
//!                 step leaves the bracket; price tolerance PRICE_TOL_REL * S0.
//! Start:          the European implied vol of the price with the expiry's effective
//!                 flat rates (r_eff, q_eff) (the crate inverter's silent 1e-4 floor
//!                 and its errors are mapped to the bracket ends / a default start).
//! Fixed point:    g(sigma') = IV^E( P_mkt - P^A_{sigma'} + P^E_{sigma'} ),
//!                 P^A, P^E from the same backward solver, IV^E the Black-Scholes
//!                 closed-form inversion at (r_eff, q_eff); g(sigma) = sigma iff
//!                 sigma = sigma^A up to the European PDE-vs-closed-form
//!                 discretization difference (of order 1e-4 relative in price);
//!                 g'(sigma^A) = 1 - nu^A/nu^E.
//! ```
//!
//! Outcomes that are not numbers are reported as [`IvOutcome`] variants
//! rather than errors: a price at or below the intrinsic value carries no
//! volatility information (`AtIntrinsic`, `BelowIntrinsic`); a price the
//! constant-vol American model cannot reach even at `SIGMA_HI` is
//! `AboveRange`. A root below `SIGMA_LO` is reported as `AtIntrinsic` too:
//! the price sits within the penalty/discretization floor of the intrinsic.
//!
//! Every function here takes an [`IvWorkspace`] so that the study's
//! per-quote loops (`par_iter().map_init(IvWorkspace::new, ...)`) allocate
//! nothing per quote.

use super::adjoint::{vega_from_adjoint, AdjointWorkspace};
use super::grid::{effective_rates, MarketSlice, Mesh};
use super::solver::{price_only, solve_into, Mode, QuoteSpec, Solution, Workspace};
use super::vol_field::NodeField;
use crate::core::errors::RustyQLibError;
use crate::equity::blackscholes::implied_vol_from_price;

/// Lower end of the volatility bracket.
pub const SIGMA_LO: f64 = 0.01;
/// Upper end of the volatility bracket.
pub const SIGMA_HI: f64 = 3.0;
/// Price tolerance of the root search, relative to the spot.
pub const PRICE_TOL_REL: f64 = 1e-10;
/// Cap on Newton/bisection evaluations of [`american_implied_vol`].
pub const MAX_IV_EVALUATIONS: usize = 60;
/// Default central-difference bump for [`VegaMethod::Bump`] and
/// [`vega_ratio_bump`].
pub const BUMP_H: f64 = 1e-3;
/// Start used when the European inversion of the price fails outright.
pub const FALLBACK_START: f64 = 0.3;
/// The crate inverter's silent floor.
const IV_FLOOR: f64 = 1e-4;

// ── Outcomes and results ─────────────────────────────────────────────────

/// Why a price has no American implied volatility (or the search failed).
#[derive(Debug)]
pub enum IvOutcome {
    /// `price <= psi(S0) + PRICE_TOL_REL * S0`, or the root lies below
    /// [`SIGMA_LO`]: the quote carries no volatility information.
    AtIntrinsic,
    /// `price < psi(S0) - PRICE_TOL_REL * S0`: the quote is below its
    /// intrinsic value (crossed or stale).
    BelowIntrinsic,
    /// `P^A(SIGMA_HI) < price`: no constant volatility in the bracket
    /// reproduces the price.
    AboveRange,
    /// The solver failed (non-finite price, invalid inputs) or the search
    /// did not converge within [`MAX_IV_EVALUATIONS`].
    Error(RustyQLibError),
}

impl std::fmt::Display for IvOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IvOutcome::AtIntrinsic => f.write_str("price at intrinsic: no American implied vol"),
            IvOutcome::BelowIntrinsic => f.write_str("price below intrinsic"),
            IvOutcome::AboveRange => {
                write!(f, "price above the American price at sigma = {SIGMA_HI}")
            }
            IvOutcome::Error(e) => write!(f, "American implied vol failed: {e}"),
        }
    }
}

impl From<RustyQLibError> for IvOutcome {
    fn from(e: RustyQLibError) -> Self {
        IvOutcome::Error(e)
    }
}

/// How the derivative `dP^A/dsigma` of the Newton iteration is obtained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VegaMethod {
    /// `sum g` from the discrete adjoint of the retained solve (exact for
    /// the frozen active set; one adjoint march per evaluation).
    Adjoint,
    /// Central difference of two extra price-only solves with the given
    /// bump `h` (in vol units).
    Bump(f64),
}

/// A converged American implied volatility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmericanIv {
    /// `sigma^A`.
    pub sigma: f64,
    /// The American price at the root (differs from the target by at most
    /// `PRICE_TOL_REL * S0`).
    pub price: f64,
    /// `dP^A/dsigma` at the root, from the method used.
    pub vega: f64,
    /// Backward solves used (price evaluations; an adjoint march counts as
    /// one more solve, a bump vega as two).
    pub solves: usize,
    /// Newton steps taken (the remaining evaluations were bisections or
    /// bracket-end checks).
    pub newton_steps: usize,
    /// European implied vol the search started from.
    pub start: f64,
}

/// The American and European vegas of one quote at a flat volatility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VegaRatio {
    /// `nu^A`.
    pub vega_a: f64,
    /// `nu^E`.
    pub vega_e: f64,
    /// `nu^A / nu^E` (`NaN` when `nu^E` is not positive).
    pub ratio: f64,
    /// American price at the volatility.
    pub price_a: f64,
    /// European price at the volatility.
    pub price_e: f64,
}

/// Outcome of the industry fixed-point iteration [`dea_fixed_point`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeaFixedPoint {
    /// The last iterate `sigma_k`.
    pub sigma: f64,
    /// Rounds performed (one American solve, one European solve and one
    /// closed-form inversion each).
    pub rounds: usize,
    /// `|sigma_{k} - sigma_{k-1}| < tol` was reached.
    pub converged: bool,
    /// The de-Americanized target `P_mkt - P^A + P^E` left the domain of
    /// the European inversion (below the arbitrage bound / at the vol
    /// floor / above `5.0`); the iteration stopped at the previous
    /// iterate.
    pub left_domain: bool,
    /// Every iterate, starting with `sigma_start`.
    pub iterates: Vec<f64>,
}

// ── Workspace ────────────────────────────────────────────────────────────

/// Scratch of the root searches: solver and adjoint workspaces, one
/// retained solution and one flat node field, all reused across quotes.
#[derive(Debug, Default)]
pub struct IvWorkspace {
    /// Backward-solver scratch.
    pub ws: Workspace,
    /// Adjoint scratch.
    pub aws: AdjointWorkspace,
    /// Retained solution of the last evaluation.
    pub sol: Solution,
    field: Option<NodeField>,
    field_key: (usize, usize, f64),
}

impl IvWorkspace {
    /// An empty workspace (allocates on first use).
    pub fn new() -> Self {
        IvWorkspace::default()
    }
}

/// Fill (or build, when the mesh shape changed) the flat field `sigma`
/// on `mesh` inside `slot`; returns it. A free function over the two
/// fields so that the solver scratch of the same workspace can be
/// borrowed mutably alongside the returned reference.
fn flat_field<'a>(
    slot: &'a mut Option<NodeField>,
    key: &mut (usize, usize, f64),
    mesh: &Mesh,
    sigma: f64,
) -> &'a NodeField {
    let want = (mesh.n_steps(), mesh.n_nodes(), mesh.t_expiry);
    let rebuild = match slot {
        Some(f) => *key != want || f.t_mid.len() != mesh.n_steps(),
        None => true,
    };
    if rebuild {
        *slot = Some(mesh.node_field(&super::vol_field::FlatVol(sigma)));
        *key = want;
    } else if let Some(f) = slot.as_mut() {
        f.values.iter_mut().for_each(|v| *v = sigma);
    }
    slot.as_ref().expect("flat field is set")
}

// ── American implied volatility ──────────────────────────────────────────

/// The American implied volatility of `price` for `spec` on `mesh` under
/// `market`, with the penalty rate `rho`.
///
/// Pre-checks the price against the intrinsic value `psi(S0)` (tolerance
/// `PRICE_TOL_REL * S0`), starts from the European implied volatility of
/// the price at the expiry's effective flat rates, then runs a
/// safeguarded Newton iteration with the derivative from `vega` inside
/// the bracket `[SIGMA_LO, SIGMA_HI]`, falling back to bisection whenever
/// a Newton step leaves the current bracket or the derivative is not
/// positive. Converges when `|P^A(sigma) - price| <= PRICE_TOL_REL * S0`.
pub fn american_implied_vol(
    price: f64,
    spec: &QuoteSpec,
    market: &MarketSlice,
    mesh: &Mesh,
    rho: f64,
    vega: VegaMethod,
    ws: &mut IvWorkspace,
) -> Result<AmericanIv, IvOutcome> {
    if !price.is_finite() {
        return Err(IvOutcome::Error(RustyQLibError::invalid_input(
            "price",
            format!("option price must be finite, got {price}"),
        )));
    }
    spec.validate(mesh)?;
    market.validate(mesh)?;
    let s0 = market.s0;
    let tol = PRICE_TOL_REL * s0;
    let intrinsic = spec.payoff(s0);
    if price < intrinsic - tol {
        return Err(IvOutcome::BelowIntrinsic);
    }
    if price <= intrinsic + tol {
        return Err(IvOutcome::AtIntrinsic);
    }
    let (r_eff, q_eff, _, _) = effective_rates(mesh, market);
    let start = match implied_vol_from_price(
        s0,
        spec.strike,
        r_eff,
        q_eff,
        mesh.t_expiry,
        price,
        spec.right,
    ) {
        Ok(v) => v.clamp(SIGMA_LO, SIGMA_HI),
        Err(e) => {
            log::debug!(
                "american_implied_vol: European start failed for K = {} T = {} ({e}); \
                 starting from {FALLBACK_START}",
                spec.strike,
                spec.t_expiry
            );
            FALLBACK_START
        }
    };
    let mode = Mode::American { rho };

    // one evaluation: price and derivative at sigma
    let mut solves = 0usize;
    let mut eval = |sigma: f64, ws: &mut IvWorkspace| -> Result<(f64, f64), RustyQLibError> {
        match vega {
            VegaMethod::Adjoint => {
                let IvWorkspace {
                    ws: sws,
                    aws,
                    sol,
                    field,
                    field_key,
                } = ws;
                let field = flat_field(field, field_key, mesh, sigma);
                solve_into(mesh, spec, field, market, mode, true, sws, sol)?;
                let nu = vega_from_adjoint(mesh, spec, sol, market, field, aws)?;
                solves += 2;
                Ok((sol.price - price, nu))
            }
            VegaMethod::Bump(h) => {
                let h = h.abs().max(1e-6);
                let p0 = flat_price(mesh, spec, market, mode, sigma, ws)?;
                let lo = (sigma - h).max(SIGMA_LO * 0.5);
                let hi = sigma + h;
                let p_lo = flat_price(mesh, spec, market, mode, lo, ws)?;
                let p_hi = flat_price(mesh, spec, market, mode, hi, ws)?;
                solves += 3;
                Ok((p0 - price, (p_hi - p_lo) / (hi - lo)))
            }
        }
    };

    let (mut lo, mut hi) = (SIGMA_LO, SIGMA_HI);
    let (mut f_lo_known, mut f_hi_known) = (false, false);
    let mut sigma = start;
    let mut newton_steps = 0usize;
    for _ in 0..MAX_IV_EVALUATIONS {
        let (f, nu) = eval(sigma, ws)?;
        if f.abs() <= tol {
            return Ok(AmericanIv {
                sigma,
                price: f + price,
                vega: nu,
                solves,
                newton_steps,
                start,
            });
        }
        if f < 0.0 {
            // the model price is below the target: the root is above sigma
            if sigma >= SIGMA_HI {
                return Err(IvOutcome::AboveRange);
            }
            lo = sigma;
            f_lo_known = true;
        } else {
            if sigma <= SIGMA_LO {
                return Err(IvOutcome::AtIntrinsic);
            }
            hi = sigma;
            f_hi_known = true;
        }
        if hi - lo <= 1e-13 * hi.max(1.0) {
            // bracket collapsed without meeting the price tolerance: the
            // price is not attained within round-off (a kink of the
            // penalized map); report the midpoint with its residual
            return Err(IvOutcome::Error(RustyQLibError::NumericalError(format!(
                "American implied vol bracket collapsed at sigma = {sigma} with residual {f} \
                 (K = {}, T = {})",
                spec.strike, spec.t_expiry
            ))));
        }
        let candidate = if nu.is_finite() && nu > 0.0 {
            sigma - f / nu
        } else {
            f64::NAN
        };
        sigma = if candidate.is_finite() && candidate > lo && candidate < hi {
            newton_steps += 1;
            candidate
        } else if !f_lo_known {
            lo
        } else if !f_hi_known {
            hi
        } else {
            0.5 * (lo + hi)
        };
    }
    Err(IvOutcome::Error(RustyQLibError::CalibrationFailed {
        iterations: MAX_IV_EVALUATIONS,
        residual: f64::NAN,
        reason: format!(
            "American implied vol did not converge for K = {} T = {} (bracket [{lo}, {hi}])",
            spec.strike, spec.t_expiry
        ),
    }))
}

/// Price-only flat-vol solve through the workspace's field buffer.
fn flat_price(
    mesh: &Mesh,
    spec: &QuoteSpec,
    market: &MarketSlice,
    mode: Mode,
    sigma: f64,
    ws: &mut IvWorkspace,
) -> Result<f64, RustyQLibError> {
    let IvWorkspace {
        ws: sws,
        field,
        field_key,
        ..
    } = ws;
    let field = flat_field(field, field_key, mesh, sigma);
    price_only(mesh, spec, field, market, mode, sws)
}

// ── Vega ratio ───────────────────────────────────────────────────────────

/// `nu^A / nu^E` of `spec` at the flat volatility `sigma` from the
/// adjoint vegas of one retained American and one retained European
/// solve (two solves, two adjoint marches).
pub fn vega_ratio(
    sigma: f64,
    spec: &QuoteSpec,
    market: &MarketSlice,
    mesh: &Mesh,
    rho: f64,
    ws: &mut IvWorkspace,
) -> Result<VegaRatio, RustyQLibError> {
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "sigma",
            format!("flat volatility must be positive and finite, got {sigma}"),
        ));
    }
    let IvWorkspace {
        ws: sws,
        aws,
        sol,
        field,
        field_key,
    } = ws;
    let field = flat_field(field, field_key, mesh, sigma);
    solve_into(
        mesh,
        spec,
        field,
        market,
        Mode::American { rho },
        true,
        sws,
        sol,
    )?;
    let price_a = sol.price;
    let vega_a = vega_from_adjoint(mesh, spec, sol, market, field, aws)?;
    solve_into(mesh, spec, field, market, Mode::European, true, sws, sol)?;
    let price_e = sol.price;
    let vega_e = vega_from_adjoint(mesh, spec, sol, market, field, aws)?;
    let ratio = if vega_e > 0.0 {
        vega_a / vega_e
    } else {
        f64::NAN
    };
    Ok(VegaRatio {
        vega_a,
        vega_e,
        ratio,
        price_a,
        price_e,
    })
}

/// [`vega_ratio`] by central bumps of `h` (four price-only solves): the
/// cross-check of the adjoint vegas. The American bump vega straddles
/// active-set switches, so agreement with the adjoint is `O(h)` relative
/// near the free boundary and `~1e-6` elsewhere.
pub fn vega_ratio_bump(
    sigma: f64,
    spec: &QuoteSpec,
    market: &MarketSlice,
    mesh: &Mesh,
    rho: f64,
    h: f64,
    ws: &mut IvWorkspace,
) -> Result<VegaRatio, RustyQLibError> {
    if !(sigma.is_finite() && sigma > 0.0 && h.is_finite() && h > 0.0 && h < sigma) {
        return Err(RustyQLibError::invalid_input(
            "sigma",
            format!("need 0 < h < sigma, got sigma = {sigma}, h = {h}"),
        ));
    }
    let am = Mode::American { rho };
    let price_a = flat_price(mesh, spec, market, am, sigma, ws)?;
    let a_hi = flat_price(mesh, spec, market, am, sigma + h, ws)?;
    let a_lo = flat_price(mesh, spec, market, am, sigma - h, ws)?;
    let price_e = flat_price(mesh, spec, market, Mode::European, sigma, ws)?;
    let e_hi = flat_price(mesh, spec, market, Mode::European, sigma + h, ws)?;
    let e_lo = flat_price(mesh, spec, market, Mode::European, sigma - h, ws)?;
    let vega_a = (a_hi - a_lo) / (2.0 * h);
    let vega_e = (e_hi - e_lo) / (2.0 * h);
    let ratio = if vega_e > 0.0 {
        vega_a / vega_e
    } else {
        f64::NAN
    };
    Ok(VegaRatio {
        vega_a,
        vega_e,
        ratio,
        price_a,
        price_e,
    })
}

// ── The industry fixed point ─────────────────────────────────────────────

/// The de-Americanization iteration `sigma' <- IV^E(P_mkt - P^A_{sigma'} +
/// P^E_{sigma'})` from `sigma_start`, at most `max_rounds` rounds,
/// stopping when consecutive iterates differ by less than `tol` or when
/// the de-Americanized price leaves the domain of the European inversion
/// (`left_domain`). Both model prices come from the backward solver on
/// `mesh`; the inversion is the Black-Scholes closed form at the expiry's
/// effective flat rates.
#[allow(clippy::too_many_arguments)]
pub fn dea_fixed_point(
    price: f64,
    spec: &QuoteSpec,
    market: &MarketSlice,
    mesh: &Mesh,
    rho: f64,
    sigma_start: f64,
    max_rounds: usize,
    tol: f64,
    ws: &mut IvWorkspace,
) -> Result<DeaFixedPoint, RustyQLibError> {
    if !(price.is_finite() && price > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "price",
            format!("market price must be positive and finite, got {price}"),
        ));
    }
    if !(sigma_start.is_finite() && sigma_start > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "sigma_start",
            format!("start volatility must be positive and finite, got {sigma_start}"),
        ));
    }
    spec.validate(mesh)?;
    market.validate(mesh)?;
    let (r_eff, q_eff, _, _) = effective_rates(mesh, market);
    let am = Mode::American { rho };
    let mut sigma = sigma_start;
    let mut iterates = vec![sigma];
    let mut converged = false;
    let mut left_domain = false;
    let mut rounds = 0usize;
    for _ in 0..max_rounds {
        let p_a = flat_price(mesh, spec, market, am, sigma, ws)?;
        let p_e = flat_price(mesh, spec, market, Mode::European, sigma, ws)?;
        let target = price - p_a + p_e;
        rounds += 1;
        let next = match implied_vol_from_price(
            market.s0,
            spec.strike,
            r_eff,
            q_eff,
            mesh.t_expiry,
            target,
            spec.right,
        ) {
            Ok(v) if v > IV_FLOOR * (1.0 + 1e-9) => v,
            Ok(_) => {
                left_domain = true;
                break;
            }
            Err(e) => {
                log::debug!(
                    "dea_fixed_point: round {rounds} left the domain (target {target}): {e}"
                );
                left_domain = true;
                break;
            }
        };
        iterates.push(next);
        let done = (next - sigma).abs() < tol;
        sigma = next;
        if done {
            converged = true;
            break;
        }
    }
    Ok(DeaFixedPoint {
        sigma,
        rounds,
        converged,
        left_domain,
        iterates,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::blackscholes::bs_price;
    use crate::equity::models::american_lv::grid::half_width;
    use crate::equity::models::american_lv::solver::RHO_DEFAULT;
    use crate::equity::models::american_lv::vol_field::FlatVol;

    const S0: f64 = 100.0;

    fn mesh_for(t: f64, n_x: usize, n_t: usize) -> Mesh {
        let l = half_width(0.3, t, 0.08, S0, 0.6 * S0, 1.5 * S0);
        Mesh::new(S0, l, n_x, t, n_t, &[]).unwrap()
    }

    #[test]
    fn american_iv_round_trips_flat_vol_prices_to_1e_8() {
        let mesh = mesh_for(0.5, 200, 60);
        let market = MarketSlice::flat(&mesh, S0, 0.04, 0.02, &[]);
        let mut ws = IvWorkspace::new();
        let mut solver_ws = Workspace::new();
        for &(k, right) in &[
            (90.0, PutOrCall::Put),
            (100.0, PutOrCall::Put),
            (110.0, PutOrCall::Put),
            (95.0, PutOrCall::Call),
            (110.0, PutOrCall::Call),
        ] {
            for &sigma in &[0.15, 0.25, 0.6] {
                let spec = QuoteSpec::new(k, 0.5, right);
                let field = mesh.node_field(&FlatVol(sigma));
                let price = price_only(
                    &mesh,
                    &spec,
                    &field,
                    &market,
                    Mode::American { rho: RHO_DEFAULT },
                    &mut solver_ws,
                )
                .unwrap();
                for method in [VegaMethod::Adjoint, VegaMethod::Bump(BUMP_H)] {
                    let iv = american_implied_vol(
                        price,
                        &spec,
                        &market,
                        &mesh,
                        RHO_DEFAULT,
                        method,
                        &mut ws,
                    )
                    .unwrap_or_else(|o| panic!("K = {k} sigma = {sigma} {method:?}: {o}"));
                    assert!(
                        (iv.sigma - sigma).abs() < 1e-8,
                        "K = {k} {right:?} sigma = {sigma} {method:?}: recovered {} in {} solves",
                        iv.sigma,
                        iv.solves
                    );
                    assert!(
                        (iv.price - price).abs() <= PRICE_TOL_REL * S0,
                        "price at root"
                    );
                    assert!(
                        iv.solves <= 24,
                        "solves {} for K = {k} sigma = {sigma}",
                        iv.solves
                    );
                }
            }
        }
    }

    #[test]
    fn american_iv_reports_intrinsic_and_range_outcomes() {
        let mesh = mesh_for(0.5, 120, 40);
        let market = MarketSlice::flat(&mesh, S0, 0.06, 0.0, &[]);
        let mut ws = IvWorkspace::new();
        let deep_put = QuoteSpec::new(140.0, 0.5, PutOrCall::Put);
        let intrinsic = 40.0;
        let r = american_implied_vol(
            intrinsic,
            &deep_put,
            &market,
            &mesh,
            RHO_DEFAULT,
            VegaMethod::Adjoint,
            &mut ws,
        );
        assert!(
            matches!(r, Err(IvOutcome::AtIntrinsic)),
            "at intrinsic: {r:?}"
        );
        let r = american_implied_vol(
            intrinsic - 0.05,
            &deep_put,
            &market,
            &mesh,
            RHO_DEFAULT,
            VegaMethod::Adjoint,
            &mut ws,
        );
        assert!(
            matches!(r, Err(IvOutcome::BelowIntrinsic)),
            "below intrinsic: {r:?}"
        );
        // an ATM put with r = 0 has P^A(sigma) ~ 0.4 sigma sqrt(T) S0 = 28 sigma:
        // a price of 0.1 needs sigma ~ 0.0036 < SIGMA_LO -> AtIntrinsic
        let zero_rate = MarketSlice::flat(&mesh, S0, 0.0, 0.0, &[]);
        let atm_put = QuoteSpec::new(100.0, 0.5, PutOrCall::Put);
        let r = american_implied_vol(
            0.1,
            &atm_put,
            &zero_rate,
            &mesh,
            RHO_DEFAULT,
            VegaMethod::Adjoint,
            &mut ws,
        );
        assert!(
            matches!(r, Err(IvOutcome::AtIntrinsic)),
            "root below the bracket is AtIntrinsic: {r:?}"
        );
        // above the price at sigma = 3.0
        let call = QuoteSpec::new(100.0, 0.5, PutOrCall::Call);
        let r = american_implied_vol(
            99.0,
            &call,
            &market,
            &mesh,
            RHO_DEFAULT,
            VegaMethod::Adjoint,
            &mut ws,
        );
        assert!(
            matches!(r, Err(IvOutcome::AboveRange)),
            "above range: {r:?}"
        );
        let r = american_implied_vol(
            f64::NAN,
            &call,
            &market,
            &mesh,
            RHO_DEFAULT,
            VegaMethod::Adjoint,
            &mut ws,
        );
        assert!(
            matches!(r, Err(IvOutcome::Error(_))),
            "NaN price is an error: {r:?}"
        );
    }

    #[test]
    fn vega_ratio_from_the_adjoint_matches_bumps_and_is_one_for_a_no_carry_call() {
        let mesh = mesh_for(0.5, 200, 60);
        let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
        let mut ws = IvWorkspace::new();
        let put = QuoteSpec::new(110.0, 0.5, PutOrCall::Put);
        let adj = vega_ratio(0.25, &put, &market, &mesh, RHO_DEFAULT, &mut ws).unwrap();
        let bump =
            vega_ratio_bump(0.25, &put, &market, &mesh, RHO_DEFAULT, BUMP_H, &mut ws).unwrap();
        assert!(
            (adj.vega_e - bump.vega_e).abs() < 1e-5 * adj.vega_e,
            "European vega adjoint {} vs bump {}",
            adj.vega_e,
            bump.vega_e
        );
        assert!(
            (adj.vega_a - bump.vega_a).abs() < 2e-3 * adj.vega_a,
            "American vega adjoint {} vs bump {}",
            adj.vega_a,
            bump.vega_a
        );
        assert!(
            adj.ratio > 0.3 && adj.ratio < 1.0,
            "ITM put with r > q has 0 < nu^A/nu^E < 1: {}",
            adj.ratio
        );
        // a call with q = 0 is never exercised early: nu^A = nu^E
        let call = QuoteSpec::new(90.0, 0.5, PutOrCall::Call);
        let c = vega_ratio(0.25, &call, &market, &mesh, RHO_DEFAULT, &mut ws).unwrap();
        assert!(
            (c.ratio - 1.0).abs() < 1e-6,
            "no-carry call vega ratio {}",
            c.ratio
        );
        assert!(
            (c.price_a - c.price_e).abs() < 1e-8 * S0,
            "no-carry call prices agree"
        );
        let bs = bs_price(S0, 90.0, 0.04, 0.0, 0.25, 0.5, PutOrCall::Call);
        assert!(
            (c.price_e - bs).abs() < 2e-4 * bs,
            "European price vs BS {} {}",
            c.price_e,
            bs
        );
    }

    #[test]
    fn dea_fixed_point_contracts_at_rate_one_minus_vega_ratio_on_an_otm_put() {
        // an OTM put with a high rate: nu^A/nu^E well below one, so the
        // contraction rate |1 - nu^A/nu^E| is measurable over several rounds
        let mesh = mesh_for(1.0, 200, 100);
        let market = MarketSlice::flat(&mesh, S0, 0.08, 0.0, &[]);
        let mut ws = IvWorkspace::new();
        let spec = QuoteSpec::new(95.0, 1.0, PutOrCall::Put);
        let sigma_true = 0.25;
        let price = flat_price(
            &mesh,
            &spec,
            &market,
            Mode::American { rho: RHO_DEFAULT },
            sigma_true,
            &mut ws,
        )
        .unwrap();
        // start from the naive European IV of the American price (above sigma^A)
        let (r_eff, q_eff, _, _) = effective_rates(&mesh, &market);
        let start =
            implied_vol_from_price(S0, 95.0, r_eff, q_eff, 1.0, price, PutOrCall::Put).unwrap();
        assert!(
            start > sigma_true,
            "naive start {start} above sigma^A {sigma_true}"
        );
        let fp = dea_fixed_point(
            price,
            &spec,
            &market,
            &mesh,
            RHO_DEFAULT,
            start,
            30,
            1e-9,
            &mut ws,
        )
        .unwrap();
        assert!(fp.converged && !fp.left_domain, "fixed point: {fp:?}");
        assert!(
            (fp.sigma - sigma_true).abs() < 5e-4,
            "fixed point {} vs sigma^A {sigma_true} (PDE-vs-closed-form floor)",
            fp.sigma
        );
        let vr = vega_ratio(fp.sigma, &spec, &market, &mesh, RHO_DEFAULT, &mut ws).unwrap();
        let rate = (1.0 - vr.ratio).abs();
        // monotone convergence (nu^A < nu^E) and the observed contraction rate
        let it = &fp.iterates;
        let n = it.len();
        assert!(n >= 4, "needs several rounds: {n}");
        let star = fp.sigma;
        let mut rates = Vec::new();
        for k in 1..n - 1 {
            let e0 = (it[k - 1] - star).abs();
            let e1 = (it[k] - star).abs();
            if e0 > 1e-6 && e1 > 1e-8 {
                rates.push(e1 / e0);
            }
        }
        assert!(!rates.is_empty(), "no usable rate samples: {it:?}");
        assert!(
            rate > 0.01,
            "the case should have a measurable rate, got {rate}"
        );
        let observed = rates[rates.len() - 1];
        assert!(
            (observed - rate).abs() < 0.15 * rate + 0.005,
            "observed contraction {observed} vs |1 - nu^A/nu^E| = {rate} (iterates {it:?})"
        );
        // the sign pattern of g'(sigma^A) = 1 - nu^A/nu^E: monotone when
        // nu^A < nu^E, alternating when nu^E < nu^A < 2 nu^E (here the OTM
        // put's early-exercise premium adds vega, so the ratio exceeds one)
        let signs: Vec<f64> = it
            .iter()
            .filter(|&&s| (s - star).abs() > 1e-7)
            .map(|&s| (s - star).signum())
            .collect();
        assert!(signs.len() >= 3, "too few sign samples: {it:?}");
        let alternating = vr.ratio > 1.0;
        for w in signs.windows(2) {
            let ok = if alternating {
                w[1] == -w[0]
            } else {
                w[1] == w[0]
            };
            assert!(
                ok,
                "sign pattern (ratio {}, alternating {alternating}) broken: {it:?}",
                vr.ratio
            );
        }
    }

    #[test]
    fn dea_fixed_point_leaves_the_domain_on_a_deep_itm_call_with_carry_below_rate() {
        // K/S0 = 0.8, T = 0.5, r = 8%, q = 6%, sigma = 10%: nu^A/nu^E ~ 3.4,
        // so the iteration repels from sigma^A
        let t = 0.5;
        let l = half_width(0.3, t, 0.08, S0, 0.6 * S0, 1.5 * S0);
        let mesh = Mesh::new(S0, l, 200, t, 100, &[]).unwrap();
        let market = MarketSlice::flat(&mesh, S0, 0.08, 0.06, &[]);
        let mut ws = IvWorkspace::new();
        let spec = QuoteSpec::new(80.0, t, PutOrCall::Call);
        let sigma_true = 0.10;
        let price = flat_price(
            &mesh,
            &spec,
            &market,
            Mode::American { rho: RHO_DEFAULT },
            sigma_true,
            &mut ws,
        )
        .unwrap();
        let vr = vega_ratio(sigma_true, &spec, &market, &mesh, RHO_DEFAULT, &mut ws).unwrap();
        assert!(vr.ratio > 2.0, "vega ratio {} should exceed 2", vr.ratio);
        let fp = dea_fixed_point(
            price,
            &spec,
            &market,
            &mesh,
            RHO_DEFAULT,
            sigma_true + 0.01,
            30,
            1e-6,
            &mut ws,
        )
        .unwrap();
        assert!(
            fp.left_domain || (!fp.converged && (fp.sigma - sigma_true).abs() > 0.01),
            "repelling fixed point should leave the domain or diverge: {fp:?}"
        );
    }
}
