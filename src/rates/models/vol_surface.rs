//! The swaption volatility surface: at-the-money vols on an expiry ×
//! tenor grid — the screen a rates desk quotes from — and the bridge
//! from that screen to short-rate calibration.
//!
//! Vols are bilinear in `(expiry, tenor)` inside the grid and flat
//! outside it. [`SwaptionVolSurface::calibration_quotes`] turns every
//! grid node into a [`SwaptionQuote`]: it builds the ATM swaption on
//! real dates and conventions (expiry rolled from the anchor in months,
//! the swap starting a spot lag later), prices it with the node's vol
//! through the market formula, and hands the price to the Hull-White
//! calibrators — so a normal-vol grid calibrates the model directly,
//! per column with [`calibrate_hull_white_piecewise`].
//!
//! [`calibrate_hull_white_piecewise`]: crate::rates::models::calibration::calibrate_hull_white_piecewise

use chrono::NaiveDate;

use crate::core::calendar::{BusinessDayConvention, Calendar, Period};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::contracts::swaption::Swaption;
use crate::rates::contracts::vanilla_swap::VanillaSwap;
use crate::rates::engines::black::{RateVol, RateVolKind};
use crate::rates::models::calibration::SwaptionQuote;
use crate::rates::PayerReceiver;

const FIELD: &str = "swaption vol surface";

/// ATM swaption vols on an expiry × tenor grid (years).
#[derive(Debug, Clone)]
pub struct SwaptionVolSurface {
    expiries: Vec<f64>,
    tenors: Vec<f64>,
    /// `vols[i][j]` at `expiries[i]`, `tenors[j]`.
    vols: Vec<Vec<f64>>,
    kind: RateVolKind,
}

impl SwaptionVolSurface {
    /// A surface from its grid; `vols[i][j]` is the vol at
    /// `expiries[i]` into `tenors[j]`, all in `kind`.
    pub fn new(
        expiries: Vec<f64>,
        tenors: Vec<f64>,
        vols: Vec<Vec<f64>>,
        kind: RateVolKind,
    ) -> Result<Self, RustyQLibError> {
        for (name, axis) in [("expiries", &expiries), ("tenors", &tenors)] {
            if axis.is_empty() {
                return Err(RustyQLibError::invalid_input(FIELD, format!("no {name}")));
            }
            if axis[0] <= 0.0 || axis.windows(2).any(|w| w[1] <= w[0]) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("{name} must be positive and strictly increasing, got {axis:?}"),
                ));
            }
        }
        if vols.len() != expiries.len() || vols.iter().any(|row| row.len() != tenors.len()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "need a {} x {} vol grid (expiries x tenors)",
                    expiries.len(),
                    tenors.len()
                ),
            ));
        }
        if vols.iter().flatten().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "vols must be finite and non-negative",
            ));
        }
        Ok(SwaptionVolSurface {
            expiries,
            tenors,
            vols,
            kind,
        })
    }

    pub fn expiries(&self) -> &[f64] {
        &self.expiries
    }

    pub fn tenors(&self) -> &[f64] {
        &self.tenors
    }

    pub fn kind(&self) -> RateVolKind {
        self.kind
    }

    /// The vol at `(expiry, tenor)`: bilinear inside the grid, flat
    /// beyond its edges.
    pub fn vol(&self, expiry: f64, tenor: f64) -> f64 {
        let (i, wi) = bracket(&self.expiries, expiry);
        let (j, wj) = bracket(&self.tenors, tenor);
        let at = |r: usize, c: usize| self.vols[r][c];
        let row0 = at(i, j) * (1.0 - wj) + at(i, j + 1) * wj;
        let row1 = at(i + 1, j) * (1.0 - wj) + at(i + 1, j + 1) * wj;
        row0 * (1.0 - wi) + row1 * wi
    }

    /// The quote at `(expiry, tenor)`.
    pub fn quote(&self, expiry: f64, tenor: f64) -> RateVol {
        self.kind.with_vol(self.vol(expiry, tenor))
    }

    /// One calibration quote per grid node: the ATM swaption on dates
    /// built by `swap` from `(effective, maturity, atm rate)`, expiry
    /// rolled `expiry` years (in whole months) from the curve's anchor
    /// on `calendar`, the swap starting `spot_lag` business days later
    /// and running `tenor` years; priced with the node's vol on `curve`
    /// (single-curve) and stored per unit notional.
    pub fn calibration_quotes(
        &self,
        curve: &YieldCurve,
        calendar: &Calendar,
        spot_lag: i64,
        swap: impl Fn(NaiveDate, NaiveDate, f64) -> Result<VanillaSwap, RustyQLibError>,
    ) -> Result<Vec<SwaptionQuote>, RustyQLibError> {
        let anchor = curve.reference_date();
        let mut quotes = Vec::with_capacity(self.expiries.len() * self.tenors.len());
        for (i, &expiry) in self.expiries.iter().enumerate() {
            let expiry_date = calendar.advance(
                anchor,
                Period::Months(months_of(expiry)?),
                BusinessDayConvention::ModifiedFollowing,
            );
            let effective = calendar.add_business_days(expiry_date, spot_lag);
            for (j, &tenor) in self.tenors.iter().enumerate() {
                let maturity = calendar.advance(
                    effective,
                    Period::Months(months_of(tenor)?),
                    BusinessDayConvention::Unadjusted,
                );
                let probe = swap(effective, maturity, 0.04)?;
                let atm = probe.par_rate(curve, curve)?;
                let underlying = swap(effective, maturity, atm)?;
                let swaption = Swaption::new(underlying, expiry_date)?;
                quotes
                    .push(swaption.to_quote_from_vol(curve, self.kind.with_vol(self.vols[i][j]))?);
            }
        }
        Ok(quotes)
    }

    /// [`calibration_quotes`](Self::calibration_quotes) on USD-standard
    /// swaps (semiannual 30/360 vs quarterly Act/360, US bond calendar,
    /// T+2 spot lag), payer side.
    pub fn usd_standard_quotes(
        &self,
        curve: &YieldCurve,
    ) -> Result<Vec<SwaptionQuote>, RustyQLibError> {
        self.calibration_quotes(
            curve,
            &Calendar::UsGovernmentBond,
            2,
            |effective, maturity, rate| {
                VanillaSwap::usd_standard(1.0, rate, PayerReceiver::Payer, effective, maturity)
            },
        )
    }
}

/// Whole months in `years`, rejecting fractions that are not months.
fn months_of(years: f64) -> Result<i32, RustyQLibError> {
    let months = years * 12.0;
    if (months - months.round()).abs() > 1e-9 || months < 1.0 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("{years} years is not a whole number of months"),
        ));
    }
    Ok(months.round() as i32)
}

/// `(index, weight)`: `x` sits between `axis[index]` and
/// `axis[index + 1]` with linear weight on the upper node; clamped to
/// the ends (weight 0 or 1) outside the axis. For a single-node axis
/// the index is 0 with weight 0 and the node is repeated.
pub(crate) fn bracket(axis: &[f64], x: f64) -> (usize, f64) {
    if axis.len() == 1 || x <= axis[0] {
        return (0, 0.0);
    }
    let last = axis.len() - 1;
    if x >= axis[last] {
        return (last - 1, 1.0);
    }
    let i = axis
        .iter()
        .rposition(|&a| a <= x)
        .unwrap_or(0)
        .min(last - 1);
    ((i), (x - axis[i]) / (axis[i + 1] - axis[i]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::models::calibration::calibrate_hull_white_piecewise;
    use crate::rates::models::HullWhite;

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn surface() -> SwaptionVolSurface {
        SwaptionVolSurface::new(
            vec![1.0, 2.0, 5.0],
            vec![2.0, 5.0, 10.0],
            vec![
                vec![0.0090, 0.0088, 0.0085],
                vec![0.0092, 0.0089, 0.0086],
                vec![0.0088, 0.0084, 0.0080],
            ],
            RateVolKind::Normal,
        )
        .unwrap()
    }

    #[test]
    fn interpolates_bilinearly_and_extrapolates_flat() {
        let s = surface();
        // nodes are hit exactly
        assert_eq!(s.vol(2.0, 5.0), 0.0089);
        // midpoint in expiry only
        assert!((s.vol(1.5, 5.0) - 0.5 * (0.0088 + 0.0089)).abs() < 1e-15);
        // midpoint in both
        let mid = 0.25 * (0.0090 + 0.0088 + 0.0092 + 0.0089);
        assert!((s.vol(1.5, 3.5) - mid).abs() < 1e-15);
        // beyond the grid: flat
        assert_eq!(s.vol(0.25, 1.0), 0.0090);
        assert_eq!(s.vol(20.0, 30.0), 0.0080);
        assert_eq!(s.quote(2.0, 5.0), RateVol::Normal(0.0089));
    }

    #[test]
    fn grid_nodes_become_calibration_quotes_that_reprice_their_vols() {
        let curve = market_curve();
        let s = surface();
        let quotes = s.usd_standard_quotes(&curve).unwrap();
        assert_eq!(quotes.len(), 9);
        // every quote is a positive price at a positive ATM strike, with
        // the swap starting two business days after expiry
        for q in &quotes {
            assert!(q.market_price > 0.0 && q.strike_rate > 0.03 && q.strike_rate < 0.06);
            assert!(q.swap_start > q.expiry && q.swap_start - q.expiry < 6.0 / 365.0);
        }
        // the 1y expiries come first and land a year out
        assert!((quotes[0].expiry - 1.0).abs() < 0.01);
    }

    #[test]
    fn a_surface_built_from_a_model_calibrates_back_to_it() {
        // implied vols of a piecewise Hull-White at the grid nodes make
        // a surface; its quotes bootstrap the model's sigmas back
        let curve = market_curve();
        let expiries = vec![1.0, 2.0, 5.0];
        let tenors = vec![3.0, 7.0];
        let flat = SwaptionVolSurface::new(
            expiries.clone(),
            tenors.clone(),
            vec![vec![0.01; 2]; 3],
            RateVolKind::Normal,
        )
        .unwrap();
        let nodes = flat
            .calibration_quotes(&curve, &Calendar::UsGovernmentBond, 2, |e, m, r| {
                VanillaSwap::usd_standard(1.0, r, PayerReceiver::Payer, e, m)
            })
            .unwrap();
        // the generating model breaks its sigma exactly at the quoted
        // expiries (the rolled dates, not the round years), which is
        // where the bootstrap will place its breakpoints
        let truth = HullWhite::with_piecewise_sigma(
            0.05,
            &[nodes[0].expiry, nodes[2].expiry],
            &[0.012, 0.009, 0.011],
            curve.clone(),
        )
        .unwrap();
        // read the model's implied normal vol at each node's swaption
        let mut vols = vec![vec![0.0; tenors.len()]; expiries.len()];
        for (k, q) in nodes.iter().enumerate() {
            let price = crate::rates::engines::jamshidian::european_swaption_settled(
                &truth,
                q.expiry,
                q.swap_start,
                &q.fixed_leg,
                q.strike_rate,
                1.0,
                q.payer_receiver,
            )
            .unwrap();
            let annuity: f64 = q.fixed_leg.iter().map(|&(t, tau)| tau * curve.df(t)).sum();
            let forward =
                (curve.df(q.swap_start) - curve.df(q.fixed_leg.last().unwrap().0)) / annuity;
            let vol = crate::rates::engines::black::implied_normal_vol(
                annuity,
                forward,
                q.strike_rate,
                q.expiry,
                crate::core::trade::PutOrCall::Call,
                price,
            )
            .unwrap();
            vols[k / tenors.len()][k % tenors.len()] = vol;
        }
        let s = SwaptionVolSurface::new(expiries, tenors, vols, RateVolKind::Normal).unwrap();
        let quotes = s.usd_standard_quotes(&curve).unwrap();
        let fit = calibrate_hull_white_piecewise(&curve, &quotes, 0.05).unwrap();
        assert!(fit.price_rmse < 1e-6, "rmse {}", fit.price_rmse);
        for (got, want) in fit.model.sigmas().iter().zip([0.012, 0.009, 0.011]) {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(SwaptionVolSurface::new(vec![], vec![1.0], vec![], RateVolKind::Normal).is_err());
        assert!(SwaptionVolSurface::new(
            vec![2.0, 1.0],
            vec![1.0],
            vec![vec![0.01], vec![0.01]],
            RateVolKind::Normal
        )
        .is_err());
        assert!(SwaptionVolSurface::new(
            vec![1.0],
            vec![1.0, 2.0],
            vec![vec![0.01]],
            RateVolKind::Normal
        )
        .is_err());
        assert!(SwaptionVolSurface::new(
            vec![1.0],
            vec![1.0],
            vec![vec![-0.01]],
            RateVolKind::Normal
        )
        .is_err());
        // a tenor that is not whole months cannot be scheduled
        let odd =
            SwaptionVolSurface::new(vec![1.0], vec![1.3], vec![vec![0.01]], RateVolKind::Normal)
                .unwrap();
        assert!(odd.usd_standard_quotes(&market_curve()).is_err());
    }
}
