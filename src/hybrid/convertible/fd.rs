//! Finite-difference pricing of convertible instruments, generic over
//! the [`ConvertibleInstrument`] and the [`CreditModel`].
//!
//! The pricing PDE is solved on a uniform log-spot grid with a
//! theta scheme (Crank-Nicolson after a fully-implicit Rannacher
//! start), the same machinery as the equity FD engine. Time runs on
//! the event grid the trees use, so coupons, calls, puts and the
//! conversion window land on exactly the same steps and the two
//! engines converge to the same value. The exercise constraints are
//! applied by projection after every step, as on the tree.
//!
//! Under **Tsiveriotis-Fernandes** two coupled equations march
//! together, the equity part `E` discounted at `r` and the cash part
//! `B` at `r + spread`, both driven by the risk-neutral diffusion
//! with drift `r - q`; the split is reassigned at every exercise, as
//! on the tree. Under **jump to default** one risky value solves
//!
//! ```text
//! V_t + 1/2 sigma^2 S^2 V_SS + (r + lambda - q - b) S V_S
//!     - (r + lambda) V + lambda * recovery(t) = 0
//! ```
//!
//! with the recovery claim integrated per step exactly as on the tree
//! (default observed at the step's end pays on the coupon period's
//! payment date), so the busted limit again reprices the credit
//! module's hazard-rate bond. The model supplies both steps through
//! [`CreditModel::pde_step`].
//!
//! Both grid edges carry the linearity condition `V_SS = 0`: at the
//! bottom the convertible is the straight bond, flat in `S`; at the top
//! it is parity, linear in `S`. The grid is centred on the spot with
//! the spot on a node, so the value, delta and gamma read straight off
//! the solution — one solve yields all three, where the tree needs
//! three solves for delta alone.
//!
//! The volatility is pluggable ([`FdVolModel`]): the market struct's
//! flat vol, a Dupire local-vol grid sampled from an implied surface
//! (the same [`LocalVolGrid`] the equity engines use), or any function
//! of share price and time. Local vol enters the PDE node by node, so
//! the skew reshapes the conversion option while the credit treatment
//! is untouched. Under jump to default the field is the diffusion
//! *conditional on survival*, so a listed surface must be de-jumped
//! first ([`dejump_surface`](super::dejump_surface), or
//! [`JumpToDefaultMarket::dejump_surface`](super::JumpToDefaultMarket::dejump_surface))
//! before its Dupire local vol is sampled.
//! With a non-flat model the market struct's `volatility` only sizes the
//! grid and anchors the vega bump, which shifts the whole field in
//! parallel.
//!
//! The bump greeks ([`ConvertibleFdGreeks`]) are built on this engine
//! for the same reason: a lattice's bump-and-reprice sensitivities carry
//! odd/even noise of a few hundredths, the grid's are smooth. Rate
//! sensitivities follow the straight-bond DV01 convention (the price
//! *drop* per unit for a one-basis-point rise), and the key-rate DV01s
//! use the curve's own key-rate bump, so over a tenor set covering the
//! curve's pillars they sum to the parallel DV01.

use chrono::NaiveDate;

use super::models::{CreditModel, NodeValue};
use super::events::{apply_cash_dividend, EventGrid};
use super::instrument::ConvertibleInstrument;
use crate::core::curves::{RateShift, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::fd_solvers::tridiagonal::thomas_algorithm;
use crate::core::vols::VolSurface;
use crate::equity::local_vol::{LocalVol, LocalVolGrid};

/// Fully-implicit starting layers of the Rannacher start, damping the
/// terminal kink at parity = redemption.
const RANNACHER_STEPS: usize = 4;

/// Grid for the finite-difference engine.
#[derive(Debug, Clone, Copy)]
pub struct ConvertibleFdGrid {
    /// Time steps from settlement to the final payment.
    pub time_steps: usize,
    /// Log-spot intervals (rounded up to even so the spot sits on a
    /// node).
    pub space_steps: usize,
    /// Half-width of the log-spot grid in terminal standard deviations,
    /// on top of the drift and the distance to the conversion price.
    pub grid_stdevs: f64,
}

impl Default for ConvertibleFdGrid {
    fn default() -> Self {
        ConvertibleFdGrid {
            time_steps: 400,
            space_steps: 400,
            grid_stdevs: 5.0,
        }
    }
}

impl ConvertibleFdGrid {
    fn validate(&self) -> Result<usize, RustyQLibError> {
        if self.time_steps < 10 {
            return Err(RustyQLibError::invalid_input(
                "convertible fd",
                format!(
                    "the grid needs at least 10 time steps, got {}",
                    self.time_steps
                ),
            ));
        }
        if self.space_steps < 20 {
            return Err(RustyQLibError::invalid_input(
                "convertible fd",
                format!(
                    "the grid needs at least 20 space steps, got {}",
                    self.space_steps
                ),
            ));
        }
        if !(self.grid_stdevs > 0.0 && self.grid_stdevs.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible fd",
                format!("grid_stdevs must be positive, got {}", self.grid_stdevs),
            ));
        }
        Ok(self.space_steps + self.space_steps % 2)
    }
}

/// Value and spot greeks in price units from one finite-difference
/// solve.
#[derive(Debug, Clone, Copy)]
pub struct ConvertibleFdValuation {
    /// Dirty price.
    pub dirty_price: f64,
    /// Clean price.
    pub clean_price: f64,
    /// Price change per unit share move.
    pub delta: f64,
    /// Delta change per unit share move.
    pub gamma: f64,
}

/// The volatility the finite-difference engine diffuses with.
pub enum FdVolModel<'a> {
    /// The market struct's flat volatility.
    Flat,
    /// Dupire local volatility sampled on a grid, from
    /// [`local_vol_grid`](super::ConvertiblePricing::local_vol_grid) or
    /// the equity module. Time is on the curve's axis, as for the
    /// implied surface.
    Local(&'a LocalVolGrid),
    /// Any volatility function of `(share price, curve time)`.
    Custom(&'a dyn Fn(f64, f64) -> f64),
}

/// A volatility lookup by share price and curve time.
trait VolField {
    fn vol(&self, spot: f64, t: f64) -> f64;
}

struct FlatVol(f64);

impl VolField for FlatVol {
    fn vol(&self, _spot: f64, _t: f64) -> f64 {
        self.0
    }
}

impl VolField for LocalVolGrid {
    fn vol(&self, spot: f64, t: f64) -> f64 {
        LocalVolGrid::vol(self, spot, t)
    }
}

struct CustomVol<'a>(&'a dyn Fn(f64, f64) -> f64);

impl VolField for CustomVol<'_> {
    fn vol(&self, spot: f64, t: f64) -> f64 {
        (self.0)(spot, t)
    }
}

/// A field shifted in parallel, floored away from zero: the vega bump.
struct ShiftedVol<'a> {
    inner: &'a dyn VolField,
    bump: f64,
}

const MIN_VOL: f64 = 1e-4;

impl VolField for ShiftedVol<'_> {
    fn vol(&self, spot: f64, t: f64) -> f64 {
        (self.inner.vol(spot, t) + self.bump).max(MIN_VOL)
    }
}

impl FdVolModel<'_> {
    fn field(&self, flat: f64) -> Box<dyn VolField + '_> {
        match self {
            FdVolModel::Flat => Box::new(FlatVol(flat)),
            FdVolModel::Local(grid) => Box::new(ShiftedVol {
                inner: *grid,
                bump: 0.0,
            }),
            FdVolModel::Custom(f) => Box::new(CustomVol(*f)),
        }
    }
}

/// The bump greeks in price units from the finite-difference engine.
#[derive(Debug, Clone)]
pub struct ConvertibleFdGreeks {
    /// The base valuation: dirty and clean price, delta and gamma.
    pub valuation: ConvertibleFdValuation,
    /// Price change per one volatility point (`+0.01` of flat vol),
    /// from a symmetric one-point bump.
    pub vega: f64,
    /// Dirty price change from holding one calendar day, spot and
    /// curve unchanged (accrued carry included).
    pub theta: f64,
    /// Price drop for a one-basis-point parallel rise of the curve.
    pub rate_dv01: f64,
    /// Price drop for a one-basis-point rise of the credit input: the
    /// spread under Tsiveriotis-Fernandes, the hazard rate under jump
    /// to default.
    pub credit_dv01: f64,
    /// The tenors (year fractions on the curve) of the key-rate bumps.
    pub key_rate_tenors: Vec<f64>,
    /// Price drop for a one-basis-point rise at each key-rate tenor.
    pub key_rate_dv01: Vec<f64>,
}

const BASIS_POINT: f64 = 1e-4;
const VOL_POINT: f64 = 0.01;

/// Dupire local volatility from an implied surface, sampled over the
/// instrument's life from `settlement`.
pub(crate) fn local_vol_grid<I: ConvertibleInstrument + ?Sized>(
    instrument: &I,
    surface: &VolSurface,
    curve: &YieldCurve,
    settlement: NaiveDate,
    spot: f64,
    dividend_yield: f64,
) -> Result<LocalVolGrid, RustyQLibError> {
    let last = instrument.final_payment_date(settlement)?;
    let horizon = curve
        .day_count()
        .year_fraction(curve.reference_date(), last);
    Ok(LocalVol::new(surface, curve, spot, dividend_yield, 0.0).to_grid(horizon))
}

/// Value and spot greeks under a chosen volatility model.
pub(crate) fn valuation<I: ConvertibleInstrument + ?Sized, M: CreditModel>(
    instrument: &I,
    market: &M,
    curve: &YieldCurve,
    settlement: NaiveDate,
    grid: ConvertibleFdGrid,
    vol: &FdVolModel,
) -> Result<ConvertibleFdValuation, RustyQLibError> {
    market.validate()?;
    instrument.validate()?;
    let field = vol.field(market.equity().volatility);
    value_on(instrument, market, curve, settlement, grid, field.as_ref())
}

fn value_on<I: ConvertibleInstrument + ?Sized, M: CreditModel>(
    instrument: &I,
    market: &M,
    curve: &YieldCurve,
    settlement: NaiveDate,
    grid: ConvertibleFdGrid,
    field: &dyn VolField,
) -> Result<ConvertibleFdValuation, RustyQLibError> {
    let space_steps = grid.validate()?;
    let events = instrument.event_grid(curve, settlement, grid.time_steps, market.credit_rate())?;
    let equity = market.equity();
    let space = SpaceGrid::new(
        instrument.conversion_price(),
        equity.spot,
        equity.volatility,
        &events,
        &grid,
        space_steps,
    );
    let value = solve(instrument, market, &events, &space, field);
    read_off(instrument, &value, &space, &events, settlement)
}

/// Reads value, delta and gamma at the spot node, scaled to the
/// instrument's price units.
fn read_off<I: ConvertibleInstrument + ?Sized>(
    instrument: &I,
    value: &[f64],
    space: &SpaceGrid,
    events: &EventGrid,
    settlement: NaiveDate,
) -> Result<ConvertibleFdValuation, RustyQLibError> {
    let j = space.spot_index;
    let h = space.h;
    let spot = space.spot[j];
    let v_x = (value[j + 1] - value[j - 1]) / (2.0 * h);
    let v_xx = (value[j + 1] - 2.0 * value[j] + value[j - 1]) / (h * h);
    let scale = 100.0 / events.outstanding;
    let dirty_price = value[j] * scale;
    if !dirty_price.is_finite() {
        return Err(RustyQLibError::NumericalError(
            "the finite-difference solve produced a non-finite value; refine the grid".to_string(),
        ));
    }
    Ok(ConvertibleFdValuation {
        dirty_price,
        clean_price: dirty_price - instrument.accrued(settlement)?,
        delta: v_x / spot * scale,
        gamma: (v_xx - v_x) / (spot * spot) * scale,
    })
}

/// The bump greeks under a chosen volatility model.
pub(crate) fn greeks<I: ConvertibleInstrument + ?Sized, M: CreditModel>(
    instrument: &I,
    market: &M,
    curve: &YieldCurve,
    settlement: NaiveDate,
    grid: ConvertibleFdGrid,
    key_rate_tenors: &[f64],
    vol: &FdVolModel,
) -> Result<ConvertibleFdGreeks, RustyQLibError> {
    market.validate()?;
    instrument.validate()?;
    let volatility = market.equity().volatility;
    let field = vol.field(volatility);
    let field = field.as_ref();
    let value = |m: &M, c: &YieldCurve, s: NaiveDate, f: &dyn VolField| {
        value_on(instrument, m, c, s, grid, f)
    };
    let wider = market.with_credit_bump(BASIS_POINT);
    assemble_greeks(
        value(market, curve, settlement, field)?,
        volatility,
        curve,
        settlement,
        key_rate_tenors,
        |bumped_vol| {
            let shifted = ShiftedVol {
                inner: field,
                bump: bumped_vol - volatility,
            };
            value(market, curve, settlement, &shifted)
        },
        |c| value(market, c, settlement, field),
        || value(&wider, curve, settlement, field),
        |s| value(market, curve, s, field),
    )
}

/// The bump greeks around `base` from the four repricing closures:
/// at a volatility, on a curve, with the credit input one basis point
/// wider, and at a settlement date.
#[allow(clippy::too_many_arguments)]
fn assemble_greeks(
    base: ConvertibleFdValuation,
    volatility: f64,
    curve: &YieldCurve,
    settlement: NaiveDate,
    key_rate_tenors: &[f64],
    at_vol: impl Fn(f64) -> Result<ConvertibleFdValuation, RustyQLibError>,
    on_curve: impl Fn(&YieldCurve) -> Result<ConvertibleFdValuation, RustyQLibError>,
    credit_wider: impl Fn() -> Result<ConvertibleFdValuation, RustyQLibError>,
    at_date: impl Fn(NaiveDate) -> Result<ConvertibleFdValuation, RustyQLibError>,
) -> Result<ConvertibleFdGreeks, RustyQLibError> {
    let dirty = base.dirty_price;
    let vega = (at_vol(volatility + VOL_POINT)?.dirty_price
        - at_vol(volatility - VOL_POINT)?.dirty_price)
        / 2.0;
    let tomorrow = settlement.succ_opt().ok_or_else(|| {
        RustyQLibError::invalid_input("convertible fd", "settlement has no next day")
    })?;
    let theta = at_date(tomorrow)?.dirty_price - dirty;
    let parallel = curve.bumped(&RateShift::ParallelAbsolute(BASIS_POINT))?;
    let rate_dv01 = dirty - on_curve(&parallel)?.dirty_price;
    let credit_dv01 = dirty - credit_wider()?.dirty_price;
    let mut key_rate_dv01 = Vec::with_capacity(key_rate_tenors.len());
    for i in 0..key_rate_tenors.len() {
        // every tenor gets its pillar and one is shifted, so the bumps
        // partition the parallel bump
        let mut shifts = vec![0.0; key_rate_tenors.len()];
        shifts[i] = BASIS_POINT;
        let bumped = curve.bumped(&RateShift::KeyRateAbsolute {
            tenors: key_rate_tenors.to_vec(),
            shifts,
        })?;
        key_rate_dv01.push(dirty - on_curve(&bumped)?.dirty_price);
    }
    Ok(ConvertibleFdGreeks {
        valuation: base,
        vega,
        theta,
        rate_dv01,
        credit_dv01,
        key_rate_tenors: key_rate_tenors.to_vec(),
        key_rate_dv01,
    })
}

/// The uniform log-spot grid, centred on the spot.
struct SpaceGrid {
    h: f64,
    spot: Vec<f64>,
    spot_index: usize,
}

impl SpaceGrid {
    fn new(
        conversion_price: f64,
        spot: f64,
        volatility: f64,
        events: &EventGrid,
        grid: &ConvertibleFdGrid,
        space_steps: usize,
    ) -> Self {
        let horizon = events.times[events.times.len() - 1] - events.times[0];
        let x0 = spot.ln();
        // the drift over the life is bounded by the curve's average
        // forward; the conversion price must sit inside the grid so the
        // parity kink is resolved
        let average_rate = -(events.riskfree_df.iter().map(|df| df.ln()).sum::<f64>()) / horizon;
        let drift_width = ((average_rate - 0.5 * volatility * volatility) * horizon).abs();
        let half_width = grid.grid_stdevs * volatility * horizon.sqrt()
            + drift_width
            + (conversion_price / spot).ln().abs().max(1e-2);
        let h = 2.0 * half_width / space_steps as f64;
        let spot_index = space_steps / 2;
        let spot = (0..=space_steps)
            .map(|j| (x0 + (j as f64 - spot_index as f64) * h).exp())
            .collect();
        SpaceGrid {
            h,
            spot,
            spot_index,
        }
    }

    /// The volatility at every node for the step from `times[step]` to
    /// `times[step + 1]`, sampled at the step's midpoint.
    fn sigma_at(&self, field: &dyn VolField, events: &EventGrid, step: usize) -> Vec<f64> {
        let t = 0.5 * (events.times[step] + events.times[step + 1]);
        self.spot.iter().map(|&s| field.vol(s, t)).collect()
    }
}

/// One backward theta step of `u_t + 1/2 sigma^2 u_xx + (mu - 1/2
/// sigma^2) u_x - rho u = 0` over `dt`, with `sigma` given per node and
/// the linearity condition `u_xx = u_x` (flat in the share price) at
/// both edges. Returns the values at the earlier time.
fn theta_step(
    u: &[f64],
    h: f64,
    dt: f64,
    theta: f64,
    sigma: &[f64],
    mu: f64,
    rho: f64,
) -> Vec<f64> {
    let n = u.len();
    let m = n - 1;
    // operator coefficients: lower, diagonal, upper per row
    let coefficients = |j: usize| -> (f64, f64, f64) {
        if j == 0 {
            (0.0, -mu / h - rho, mu / h)
        } else if j == m {
            (-mu / h, mu / h - rho, 0.0)
        } else {
            let s2 = sigma[j] * sigma[j];
            let nu = mu - 0.5 * s2;
            (
                0.5 * s2 / (h * h) - nu / (2.0 * h),
                -s2 / (h * h) - rho,
                0.5 * s2 / (h * h) + nu / (2.0 * h),
            )
        }
    };
    let mut sub = Vec::with_capacity(m);
    let mut diag = Vec::with_capacity(n);
    let mut sup = Vec::with_capacity(m);
    let mut rhs = Vec::with_capacity(n);
    for j in 0..n {
        let (lo, mid, hi) = coefficients(j);
        // explicit part
        let mut lu = mid * u[j];
        if j > 0 {
            lu += lo * u[j - 1];
        }
        if j < m {
            lu += hi * u[j + 1];
        }
        rhs.push(u[j] + (1.0 - theta) * dt * lu);
        // implicit part
        if j > 0 {
            sub.push(-theta * dt * lo);
        }
        diag.push(1.0 - theta * dt * mid);
        if j < m {
            sup.push(-theta * dt * hi);
        }
    }
    thomas_algorithm(&sub, &diag, &sup, &rhs)
}

fn rannacher_theta(backward_step: usize) -> f64 {
    if backward_step < RANNACHER_STEPS {
        1.0
    } else {
        0.5
    }
}

/// The backward march; returns the node totals at settlement over the
/// space grid.
fn solve<I: ConvertibleInstrument + ?Sized, M: CreditModel>(
    instrument: &I,
    market: &M,
    events: &EventGrid,
    space: &SpaceGrid,
    field: &dyn VolField,
) -> Vec<f64> {
    let steps = events.riskfree_df.len();
    let dt = events.dt;
    let credit_df = (-market.credit_rate() * dt).exp();
    // the log-price width of one grid cell, for a smoothed soft trigger
    let cell_width = space.h;

    let mut nodes: Vec<M::Node> = space
        .spot
        .iter()
        .map(|&spot| events.terminal(instrument, spot, cell_width))
        .collect();

    for step in (0..steps).rev() {
        let theta = rannacher_theta(steps - 1 - step);
        let riskfree_df = events.riskfree_df[step];
        let r = -riskfree_df.ln() / dt;
        let mu = r + market.survival_drift();
        let sigma = space.sigma_at(field, events, step);
        let diffuse = |u: Vec<f64>, rho: f64| theta_step(&u, space.h, dt, theta, &sigma, mu, rho);
        nodes = market.pde_step(
            &nodes,
            &diffuse,
            r,
            riskfree_df,
            credit_df,
            events.default_claim_at_step[step],
        );
        apply_cash_dividend(&mut nodes, &space.spot, events.cash_dividend_at_step[step]);
        for (j, node) in nodes.iter_mut().enumerate() {
            let continuation = node.plus_cash(events.coupon_at_step[step]);
            *node = events
                .exercise(instrument, step, space.spot[j], continuation, cell_width)
                .plus_cash(events.coupon_kept_at_step[step]);
        }
    }
    nodes.iter().map(|n| n.total()).collect()
}
