//! Dupire local volatility calibrated from an implied vol surface.
//!
//! Uses Gatheral's formulation of the Dupire equation in total variance
//! `w(y, t) = sigma_imp(K, t)^2 * t`, `y = ln(K / F(t))`:
//!
//! ```text
//! sigma_loc^2(K, t) = (dw/dt) /
//!   [ 1 - (y/w) w_y + 1/4 (-1/4 - 1/w + y^2/w^2) w_y^2 + 1/2 w_yy ]
//! ```
//!
//! Derivatives are taken numerically on the implied surface: the time
//! derivative at fixed moneyness `y`, the strike derivatives at fixed `t`.
//! The "calibration" is therefore non-parametric — the local vol function
//! is the exact transformation of whatever implied surface it is given.
//!
//! Guards: at very short times the implied vol is returned directly; where
//! interpolation noise makes the denominator or numerator non-positive
//! (butterfly / calendar violations in the inputs) the implied vol is used
//! as a fallback; the result is clamped to `[1%, 300%]`.

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::vols::VolSurface;

const TIME_BUMP: f64 = 1.0 / 365.0;
const LOG_STRIKE_BUMP: f64 = 0.01;
const MIN_LOCAL_VOL: f64 = 0.01;
const MAX_LOCAL_VOL: f64 = 3.0;

/// Local volatility function `sigma_loc(level, t)`, frozen at construction
/// from an implied surface, a discount curve (for forwards) and a dividend
/// yield.
pub struct LocalVol<'a> {
    surface: &'a VolSurface,
    curve: &'a YieldCurve,
    spot: f64,
    dividend_yield: f64,
    /// Parallel shift added to every implied vol before the Dupire
    /// transformation — used by vega bump-and-reprice.
    vol_shift: f64,
}

impl<'a> LocalVol<'a> {
    pub fn new(
        surface: &'a VolSurface,
        curve: &'a YieldCurve,
        spot: f64,
        dividend_yield: f64,
        vol_shift: f64,
    ) -> Self {
        LocalVol { surface, curve, spot, dividend_yield, vol_shift }
    }

    fn forward(&self, t: f64) -> f64 {
        let r = self.curve.zero_rate_with(t, Compounding::Continuous);
        self.spot * ((r - self.dividend_yield) * t).exp()
    }

    fn implied(&self, strike: f64, t: f64) -> f64 {
        self.surface.vol(strike, self.forward(t), t) + self.vol_shift
    }

    /// Total variance at absolute strike `k` and expiry `t`.
    fn total_variance(&self, strike: f64, t: f64) -> f64 {
        let v = self.implied(strike, t);
        v * v * t
    }

    /// Local volatility at underlying level `level` and time `t`.
    pub fn vol(&self, level: f64, t: f64) -> f64 {
        let implied = self.implied(level, t.max(1e-4));
        if t < 1e-3 {
            return implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL);
        }

        let f = self.forward(t);
        let y = (level / f).ln();
        let w = self.total_variance(level, t);
        if w < 1e-8 {
            return implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL);
        }

        // dw/dt at fixed moneyness y: strike moves with the forward
        let ht = TIME_BUMP.min(0.5 * t);
        let w_up = self.total_variance(self.forward(t + ht) * y.exp(), t + ht);
        let w_dn = self.total_variance(self.forward(t - ht) * y.exp(), t - ht);
        let dw_dt = (w_up - w_dn) / (2.0 * ht);

        // strike derivatives at fixed t (central, multiplicative bump)
        let hy = LOG_STRIKE_BUMP;
        let w_plus = self.total_variance(level * hy.exp(), t);
        let w_minus = self.total_variance(level * (-hy).exp(), t);
        let dw_dy = (w_plus - w_minus) / (2.0 * hy);
        let d2w_dy2 = (w_plus - 2.0 * w + w_minus) / (hy * hy);

        let denom = 1.0 - (y / w) * dw_dy
            + 0.25 * (-0.25 - 1.0 / w + (y * y) / (w * w)) * dw_dy * dw_dy
            + 0.5 * d2w_dy2;

        if dw_dt <= 0.0 || denom <= 1e-4 {
            // calendar / butterfly violation in the interpolated inputs:
            // fall back to the implied vol at this point
            return implied.clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL);
        }
        (dw_dt / denom).sqrt().clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Tenor;
    use crate::core::daycount::DayCountConvention;
    use chrono::NaiveDate;

    fn asof() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
    }

    fn flat_curve() -> YieldCurve {
        YieldCurve::flat(0.05, asof(), DayCountConvention::Act365, Compounding::Continuous)
            .unwrap()
    }

    #[test]
    fn flat_surface_gives_flat_local_vol() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        for level in [60.0, 90.0, 100.0, 130.0] {
            for t in [0.05, 0.5, 1.0, 2.0] {
                let v = lv.vol(level, t);
                assert!((v - 0.25).abs() < 1e-6, "level={level} t={t}: {v}");
            }
        }
    }

    #[test]
    fn term_structure_gives_forward_variance() {
        // sigma(0.5) = 20%, sigma(1.0) = 25% (flat in strike): between the
        // pillars the local variance is the forward variance
        // (w2 - w1)/(t2 - t1) = (0.0625 - 0.02)/0.5 = 0.085
        let surface = VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
            asof(),
            DayCountConvention::Act365,
        )
        .unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.0);
        let expected = (0.085_f64).sqrt();
        let v = lv.vol(100.0, 0.75);
        assert!((v - expected).abs() < 1e-3, "{v} vs {expected}");
    }

    #[test]
    fn vol_shift_moves_local_vol() {
        let surface = VolSurface::flat(0.25, asof(), DayCountConvention::Act365).unwrap();
        let curve = flat_curve();
        let lv = LocalVol::new(&surface, &curve, 100.0, 0.0, 0.01);
        assert!((lv.vol(100.0, 1.0) - 0.26).abs() < 1e-6);
    }
}
