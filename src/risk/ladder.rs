//! Spot risk ladders: the desk risk slide.
//!
//! Point Greeks are a local Taylor expansion — valid near today's spot
//! and silent about what gamma does three percent away, which is exactly
//! where barrier products (knock-outs, autocallables, geared
//! accumulators) hide their risk. A **ladder** completes the picture by
//! full revaluation at a grid of spot levels: per rung, the book's MtM
//! and P&L, with ladder **delta** and **gamma** read off adjacent rungs
//! by (unevenly spaced) central differences — the non-local versions of
//! delta/gamma that a desk trusts where speed and higher-order Greeks
//! get noisy.
//!
//! Mechanically a ladder is a parametric family of relative spot
//! [`Shock`]s applied through the pricing context: snapshot once
//! ([`EquityPortfolio::snapshot_market`]), bump per rung
//! ([`Market::bumped`](crate::core::market::Market::bumped)), revalue
//! every position on its own engine. Monte Carlo positions keep their
//! seed through the rebind, so rung-to-rung differences are free of
//! sampling noise (common random numbers).

use crate::core::errors::RustyQLibError;
use crate::core::market::{BumpMode, RiskFactor, Shock, Spot};
use crate::equity::portfolio::EquityPortfolio;

/// One rung of the ladder.
#[derive(Debug, Clone)]
pub struct LadderPoint {
    /// Relative spot move of this rung (e.g. `-0.10` = spot down 10%).
    pub move_rel: f64,
    /// Absolute spot level at this rung, `S0 * (1 + move_rel)`.
    pub spot: f64,
    /// Book MtM under the bumped market (quantity-weighted).
    pub mtm: f64,
    /// `mtm - base_mtm`.
    pub pnl: f64,
    /// Per-position MtM at this rung, in book order — the drill-down for
    /// "which trade drives the flip".
    pub position_mtm: Vec<f64>,
    /// Ladder delta `dV/dS` at this rung from the neighbouring rungs
    /// (central differences, uneven spacing supported); `None` at the
    /// endpoints, which have only one neighbour.
    pub delta: Option<f64>,
    /// Ladder gamma `d²V/dS²` at this rung; `None` at the endpoints.
    pub gamma: Option<f64>,
}

/// A spot ladder over one book: base MtM plus one [`LadderPoint`] per
/// requested move, in ascending move order.
#[derive(Debug, Clone)]
pub struct SpotLadder {
    /// The book's underlying symbol (EquityPortfolio books are
    /// single-underlying).
    pub symbol: String,
    /// Unbumped spot the moves are relative to.
    pub base_spot: f64,
    /// Book MtM under the unbumped snapshot.
    pub base_mtm: f64,
    pub points: Vec<LadderPoint>,
}

/// A symmetric uniform move grid: `rungs_per_side` rungs each side of
/// zero in steps of `step`, zero included — e.g. `(0.05, 4)` gives
/// `[-0.20, -0.15, ..., 0.20]`.
pub fn symmetric_moves(step: f64, rungs_per_side: usize) -> Vec<f64> {
    let n = rungs_per_side as i64;
    (-n..=n).map(|k| k as f64 * step).collect()
}

/// Revalue `book` at every relative spot move in `moves` (strictly
/// increasing, each above -100%) and read ladder delta/gamma off the
/// rungs. Errors on an empty book, an invalid grid, or a position the
/// snapshot cannot reprice.
pub fn spot_ladder(
    book: &EquityPortfolio,
    moves: &[f64],
) -> Result<SpotLadder, RustyQLibError> {
    let symbol = match book.positions.first() {
        Some(p) => p.option.base.symbol.clone(),
        None => {
            return Err(RustyQLibError::invalid_input("book", "cannot ladder an empty book"));
        }
    };
    if moves.is_empty() {
        return Err(RustyQLibError::invalid_input("moves", "the ladder needs at least one rung"));
    }
    if moves.iter().any(|x| !x.is_finite() || *x <= -1.0) {
        return Err(RustyQLibError::invalid_input(
            "moves",
            "moves must be finite relative bumps above -100%",
        ));
    }
    if moves.windows(2).any(|w| w[1] <= w[0]) {
        return Err(RustyQLibError::invalid_input(
            "moves",
            "moves must be strictly increasing",
        ));
    }

    let base_market = book.snapshot_market();
    let base_spot = base_market.get(&Spot(symbol.clone()))?.mid();
    let base_values = book.position_values_in(&base_market)?;
    let base_mtm: f64 = base_values.iter().sum();

    let mut points = Vec::with_capacity(moves.len());
    for &x in moves {
        let shock = Shock {
            factor: RiskFactor::Spot,
            mode: BumpMode::Relative,
            size: x,
            underlying: None,
            tenors: None,
            shifts: None,
        };
        let bumped = base_market.bumped(std::slice::from_ref(&shock))?;
        let position_mtm = book.position_values_in(&bumped)?;
        let mtm: f64 = position_mtm.iter().sum();
        points.push(LadderPoint {
            move_rel: x,
            spot: base_spot * (1.0 + x),
            mtm,
            pnl: mtm - base_mtm,
            position_mtm,
            delta: None,
            gamma: None,
        });
    }

    let xs: Vec<f64> = points.iter().map(|p| p.spot).collect();
    let vs: Vec<f64> = points.iter().map(|p| p.mtm).collect();
    for (point, (d1, d2)) in points.iter_mut().zip(ladder_derivatives(&xs, &vs)) {
        point.delta = d1;
        point.gamma = d2;
    }

    Ok(SpotLadder { symbol, base_spot, base_mtm, points })
}

/// First and second derivatives of `vs` w.r.t. `xs` at every point by
/// three-point central differences on a possibly uneven grid (the
/// standard unequal-spacing stencil, second-order accurate); the
/// endpoints, having one neighbour, get `(None, None)`.
fn ladder_derivatives(xs: &[f64], vs: &[f64]) -> Vec<(Option<f64>, Option<f64>)> {
    let mut out = vec![(None, None); xs.len()];
    for i in 1..xs.len().saturating_sub(1) {
        let (h1, h2) = (xs[i] - xs[i - 1], xs[i + 1] - xs[i]);
        let (v_prev, v_mid, v_next) = (vs[i - 1], vs[i], vs[i + 1]);
        let d1 = -h2 / (h1 * (h1 + h2)) * v_prev + (h2 - h1) / (h1 * h2) * v_mid
            + h1 / (h2 * (h1 + h2)) * v_next;
        let d2 =
            2.0 * (v_prev / (h1 * (h1 + h2)) - v_mid / (h1 * h2) + v_next / (h2 * (h1 + h2)));
        out[i] = (Some(d1), Some(d2));
    }
    out
}

/// One rung of a [`vol_ladder`].
#[derive(Debug, Clone)]
pub struct VolLadderPoint {
    /// Absolute vol-point shift of this rung (e.g. `-0.05` = every
    /// implied vol down 5 points), applied as a parallel surface shift.
    pub shift: f64,
    /// Book MtM under the shifted surface (quantity-weighted).
    pub mtm: f64,
    /// `mtm - base_mtm`.
    pub pnl: f64,
    /// Per-position MtM at this rung, in book order.
    pub position_mtm: Vec<f64>,
    /// Ladder vega `dV/dσ` (per unit vol; divide by 100 for per-vol-point)
    /// from the neighbouring rungs; `None` at the endpoints.
    pub vega: Option<f64>,
    /// Ladder volga `d²V/dσ²`; `None` at the endpoints.
    pub volga: Option<f64>,
}

/// A vol ladder over one book: the vega profile across parallel shifts
/// of the implied surface — the non-local view of vega/volga, as
/// [`spot_ladder`] is for delta/gamma.
#[derive(Debug, Clone)]
pub struct VolLadder {
    pub symbol: String,
    /// Book MtM under the unshifted snapshot.
    pub base_mtm: f64,
    pub points: Vec<VolLadderPoint>,
}

/// Revalue `book` under parallel **absolute** vol-point shifts of every
/// implied surface (strictly increasing `shifts`, e.g. `-0.10..=0.10`)
/// and read ladder vega/volga off the rungs. A shift that drives any
/// vol non-positive surfaces as the surface's own bump error. Errors on
/// an empty book or an invalid grid.
pub fn vol_ladder(book: &EquityPortfolio, shifts: &[f64]) -> Result<VolLadder, RustyQLibError> {
    let symbol = match book.positions.first() {
        Some(p) => p.option.base.symbol.clone(),
        None => {
            return Err(RustyQLibError::invalid_input("book", "cannot ladder an empty book"));
        }
    };
    if shifts.is_empty() {
        return Err(RustyQLibError::invalid_input("shifts", "the ladder needs at least one rung"));
    }
    if shifts.iter().any(|x| !x.is_finite()) {
        return Err(RustyQLibError::invalid_input("shifts", "shifts must be finite vol points"));
    }
    if shifts.windows(2).any(|w| w[1] <= w[0]) {
        return Err(RustyQLibError::invalid_input(
            "shifts",
            "shifts must be strictly increasing",
        ));
    }

    let base_market = book.snapshot_market();
    let base_values = book.position_values_in(&base_market)?;
    let base_mtm: f64 = base_values.iter().sum();

    let mut points = Vec::with_capacity(shifts.len());
    for &shift in shifts {
        let shock = Shock {
            factor: RiskFactor::Vol,
            mode: BumpMode::Absolute,
            size: shift,
            underlying: None,
            tenors: None,
            shifts: None,
        };
        let bumped = base_market.bumped(std::slice::from_ref(&shock))?;
        let position_mtm = book.position_values_in(&bumped)?;
        let mtm: f64 = position_mtm.iter().sum();
        points.push(VolLadderPoint {
            shift,
            mtm,
            pnl: mtm - base_mtm,
            position_mtm,
            vega: None,
            volga: None,
        });
    }

    let vs: Vec<f64> = points.iter().map(|p| p.mtm).collect();
    for (point, (d1, d2)) in points.iter_mut().zip(ladder_derivatives(shifts, &vs)) {
        point.vega = d1;
        point.volga = d2;
    }

    Ok(VolLadder { symbol, base_mtm, points })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    fn call_book(quantity: f64) -> EquityPortfolio {
        let option = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .expect("option must build");
        let mut book = EquityPortfolio::new();
        book.add(option, quantity);
        book
    }

    #[test]
    fn symmetric_moves_span_zero_uniformly() {
        let moves = symmetric_moves(0.05, 4);
        assert_eq!(moves.len(), 9);
        assert!((moves[0] + 0.20).abs() < 1e-12);
        assert!((moves[4]).abs() < 1e-12);
        assert!((moves[8] - 0.20).abs() < 1e-12);
    }

    #[test]
    fn ladder_greeks_match_the_closed_forms_at_the_centre() {
        let quantity = 100.0;
        let book = call_book(quantity);
        let ladder = spot_ladder(&book, &symmetric_moves(0.02, 2)).unwrap();
        assert_eq!(ladder.symbol, "ACME");
        assert!((ladder.base_spot - 100.0).abs() < 1e-12);
        // the zero rung reprices to the base MtM exactly
        let centre = &ladder.points[2];
        assert!((centre.move_rel).abs() < 1e-12);
        assert!((centre.mtm - ladder.base_mtm).abs() < 1e-10);
        assert!((centre.pnl).abs() < 1e-10);
        // ladder delta/gamma at the centre against the analytic Greeks
        // (quantity-weighted); 2% spot steps keep the FD error small
        let greeks = book.positions[0].option.price().unwrap().greeks;
        let delta = centre.delta.expect("interior rung has delta");
        let gamma = centre.gamma.expect("interior rung has gamma");
        assert!(
            (delta - quantity * greeks.delta).abs() < 0.01 * quantity * greeks.delta.abs(),
            "ladder delta {delta} vs analytic {}",
            quantity * greeks.delta
        );
        assert!(
            (gamma - quantity * greeks.gamma).abs() < 0.01 * quantity * greeks.gamma.abs(),
            "ladder gamma {gamma} vs analytic {}",
            quantity * greeks.gamma
        );
        // endpoints have no neighbours on both sides
        assert!(ladder.points[0].delta.is_none() && ladder.points[4].gamma.is_none());
        // long call: P&L monotone in spot, positive gamma on every rung
        assert!(ladder.points.windows(2).all(|w| w[1].mtm > w[0].mtm));
        assert!(ladder.points[1].gamma.unwrap() > 0.0);
        assert!(ladder.points[3].gamma.unwrap() > 0.0);
        // per-position drill-down sums to the book at every rung
        for p in &ladder.points {
            let sum: f64 = p.position_mtm.iter().sum();
            assert!((sum - p.mtm).abs() < 1e-10);
        }
    }

    #[test]
    fn uneven_grids_reproduce_the_same_centre_greeks() {
        // desk-style uneven grid: the uneven-spacing stencil must agree
        // with the closed forms just like the uniform one
        let book = call_book(1.0);
        let ladder = spot_ladder(&book, &[-0.05, -0.02, 0.0, 0.02, 0.05]).unwrap();
        let greeks = book.positions[0].option.price().unwrap().greeks;
        let centre = &ladder.points[2];
        assert!((centre.delta.unwrap() - greeks.delta).abs() < 0.01 * greeks.delta.abs());
        assert!((centre.gamma.unwrap() - greeks.gamma).abs() < 0.015 * greeks.gamma.abs());
    }

    #[test]
    fn accumulator_ladder_shows_the_toxic_tail_and_the_knockout_relief() {
        // the product the ladder exists for: geared accumulator, KO 110
        let option = EquityOptionBuilder::new()
            .symbol("ACCU")
            .spot(100.0)
            .strike(95.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .accumulator(110.0, 12, 1.0, 2.0)
            .engine(Engine::MonteCarlo)
            .paths(4_000)
            .seed(42)
            .build()
            .expect("accumulator must build");
        let mut book = EquityPortfolio::new();
        book.add(option, 1.0);
        let ladder = spot_ladder(&book, &[-0.20, -0.10, 0.0, 0.10, 0.20]).unwrap();
        let down = ladder.points[0].pnl;
        let up = ladder.points[4].pnl;
        // down 20%: deep in the geared zone — the toxic tail
        assert!(down < 0.0, "toxic tail pnl {down}");
        // up 20%: spot starts above the KO, the structure dies almost
        // immediately — relief for the short-the-wings holder
        assert!(up > 0.0, "knock-out relief pnl {up}");
        assert!(up.abs() < down.abs(), "asymmetry: relief is capped, the tail is not");
    }

    #[test]
    fn vol_ladder_vega_matches_the_closed_form_and_shows_convexity() {
        let quantity = 100.0;
        let book = call_book(quantity);
        let ladder = vol_ladder(&book, &[-0.04, -0.02, 0.0, 0.02, 0.04]).unwrap();
        assert_eq!(ladder.symbol, "ACME");
        let centre = &ladder.points[2];
        assert!((centre.shift).abs() < 1e-12);
        assert!((centre.mtm - ladder.base_mtm).abs() < 1e-10);
        // ladder vega at the centre against the analytic vega
        let greeks = book.positions[0].option.price().unwrap().greeks;
        let vega = centre.vega.expect("interior rung has vega");
        assert!(
            (vega - quantity * greeks.vega).abs() < 0.01 * quantity * greeks.vega.abs(),
            "ladder vega {vega} vs analytic {}",
            quantity * greeks.vega
        );
        // a long option gains monotonically as vols rise
        assert!(ladder.points.windows(2).all(|w| w[1].mtm > w[0].mtm));
        // endpoints have no both-sided neighbours
        assert!(ladder.points[0].vega.is_none() && ladder.points[4].volga.is_none());
        // per-position drill-down sums to the book at every rung
        for p in &ladder.points {
            let sum: f64 = p.position_mtm.iter().sum();
            assert!((sum - p.mtm).abs() < 1e-10);
        }

        // volga: an OTM option is vol-convex (long volga), visibly so on
        // a coarser grid
        let otm = EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(140.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap();
        let mut otm_book = EquityPortfolio::new();
        otm_book.add(otm, 1.0);
        let otm_ladder = vol_ladder(&otm_book, &[-0.05, 0.0, 0.05]).unwrap();
        assert!(otm_ladder.points[1].volga.unwrap() > 0.0, "OTM option is long volga");
    }

    #[test]
    fn accumulator_vol_ladder_confirms_the_short_vol_holder() {
        // the geared holder is short the wings: vols down is relief,
        // vols up is pain — the vega profile the stress test sampled at
        // one point, now as a curve
        let option = EquityOptionBuilder::new()
            .symbol("ACCU")
            .spot(100.0)
            .strike(95.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .accumulator(110.0, 12, 1.0, 2.0)
            .engine(Engine::MonteCarlo)
            .paths(4_000)
            .seed(42)
            .build()
            .expect("accumulator must build");
        let mut book = EquityPortfolio::new();
        book.add(option, 1.0);
        let ladder = vol_ladder(&book, &[-0.05, 0.0, 0.05]).unwrap();
        assert!(ladder.points[0].pnl > 0.0, "vols down relieves the short-vol holder");
        assert!(ladder.points[2].pnl < 0.0, "vols up hurts the short-vol holder");
        assert!(ladder.points[1].vega.unwrap() < 0.0, "book vega is short");
    }

    #[test]
    fn vol_ladder_rejects_invalid_grids_and_impossible_shifts() {
        let book = call_book(1.0);
        assert!(vol_ladder(&book, &[]).is_err(), "empty grid");
        assert!(vol_ladder(&book, &[0.02, 0.01]).is_err(), "descending");
        assert!(vol_ladder(&book, &[f64::INFINITY]).is_err(), "non-finite");
        // a shift that drives the 25% surface negative surfaces the
        // surface's own bump error rather than pricing nonsense
        assert!(vol_ladder(&book, &[-0.30, 0.0]).is_err(), "negative vol");
        let empty = EquityPortfolio::new();
        assert!(vol_ladder(&empty, &[0.0]).is_err(), "empty book");
    }

    #[test]
    fn invalid_grids_and_empty_books_are_rejected() {
        let book = call_book(1.0);
        assert!(spot_ladder(&book, &[]).is_err(), "empty grid");
        assert!(spot_ladder(&book, &[-0.1, -0.1, 0.1]).is_err(), "not strictly increasing");
        assert!(spot_ladder(&book, &[0.1, -0.1]).is_err(), "descending");
        assert!(spot_ladder(&book, &[-1.5, 0.0]).is_err(), "below -100%");
        assert!(spot_ladder(&book, &[f64::NAN]).is_err(), "non-finite");
        let empty = EquityPortfolio::new();
        assert!(spot_ladder(&empty, &[0.0]).is_err(), "empty book");
    }
}
