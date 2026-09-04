//! European spread option on two commodity futures, priced with Kirk's
//! approximation.
//!
//! The two-underlying workhorse of energy desks: crack spreads (product
//! versus crude), spark spreads (power versus gas), location spreads
//! (Brent versus WTI) and calendar spreads (two delivery dates on one
//! curve — pass the same forward curve for both legs). The payoff at
//! expiry is
//!
//! `quantity * max(F_a - F_b - K, 0)` (calls; reversed for puts),
//!
//! cash-settled, where `F_a`/`F_b` are the two futures read off their
//! forward curves at each leg's delivery date.
//!
//! # Model
//!
//! Each leg quotes a [`CommodityVol`] and the pair a correlation `rho`;
//! both legs must quote the same model family.
//!
//! - **Lognormal** (and **shifted lognormal**): Kirk (1995) — treat
//!   `F_a / (F_b + K)` as lognormal with composite vol
//!   `sigma^2 = sigma_a^2 - 2 rho sigma_a sigma_b w + (sigma_b w)^2`,
//!   `w = F_b / (F_b + K)`, which is exactly Black-76 on `F_a` struck
//!   at `F_b + K`. Exact at `K = 0` (Margrabe's exchange option),
//!   accurate for strikes small against the legs. Shifted legs run
//!   Kirk on the displaced forwards with the strike displaced by
//!   `shift_a - shift_b`.
//! - **Normal**: exact, no approximation — a difference of jointly
//!   normal legs is normal, so the spread prices in the
//!   [`bachelier`] kernel with spread vol
//!   `sqrt(sigma_a^2 - 2 rho sigma_a sigma_b + sigma_b^2)`. Any sign
//!   of leg, spread or strike — the natural quote for basis spreads.

use chrono::NaiveDate;

use crate::cmdty::bachelier;
use crate::cmdty::expiry_inputs;
use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::cmdty::vol::CommodityVol;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::equity::black76::{self, FuturesSettlement};

const FIELD: &str = "spread option";

/// A European option on the spread `F_a - F_b`. Premium is quoted for
/// the whole contract (`quantity` units).
#[derive(Debug, Clone)]
pub struct CommoditySpreadOption {
    /// Contract size in the legs' (common) units.
    pub quantity: f64,
    /// Strike on the spread, any sign.
    pub strike: f64,
    pub put_or_call: PutOrCall,
    /// Option expiry (last exercise date).
    pub expiry_date: NaiveDate,
    /// Delivery date of leg A (the received leg) on its forward curve.
    pub underlying_date_a: NaiveDate,
    /// Delivery date of leg B (the paid leg) on its forward curve.
    pub underlying_date_b: NaiveDate,
    pub settlement: FuturesSettlement,
}

impl CommoditySpreadOption {
    pub fn new(
        quantity: f64,
        strike: f64,
        put_or_call: PutOrCall,
        expiry_date: NaiveDate,
        underlying_date_a: NaiveDate,
        underlying_date_b: NaiveDate,
        settlement: FuturesSettlement,
    ) -> Result<Self, RustyQLibError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "spread option",
                format!("quantity must be positive, got {quantity}"),
            ));
        }
        if !strike.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "spread option",
                format!("strike must be finite, got {strike}"),
            ));
        }
        if underlying_date_a < expiry_date || underlying_date_b < expiry_date {
            return Err(RustyQLibError::invalid_input(
                "spread option",
                format!(
                    "underlying dates {underlying_date_a} / {underlying_date_b} must not \
                     precede expiry {expiry_date}"
                ),
            ));
        }
        Ok(CommoditySpreadOption {
            quantity,
            strike,
            put_or_call,
            expiry_date,
            underlying_date_a,
            underlying_date_b,
            settlement,
        })
    }

    /// The two legs' futures prices `(F_a, F_b)` off their curves.
    pub fn forward_prices(
        &self,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
    ) -> (f64, f64) {
        (
            forward_a.price(self.underlying_date_a),
            forward_b.price(self.underlying_date_b),
        )
    }

    /// Premium for the whole contract. Bare `f64` vols are Black
    /// (lognormal) leg vols priced with Kirk; both legs must quote the
    /// same [`CommodityVol`] family.
    pub fn price(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        vol_a: impl Into<CommodityVol>,
        vol_b: impl Into<CommodityVol>,
        rho: f64,
    ) -> Result<f64, RustyQLibError> {
        let quote_a = vol_a.into().validated("spread option")?;
        let quote_b = vol_b.into().validated("spread option")?;
        if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
            return Err(RustyQLibError::invalid_input(
                "spread option",
                format!("correlation must be in [-1, 1], got {rho}"),
            ));
        }
        let inputs = expiry_inputs(FIELD, self.expiry_date, discount)?;
        let (r, t) = (inputs.r, inputs.t);
        let (f_a, f_b) = self.forward_prices(forward_a, forward_b);
        let (k, pc, s) = (self.strike, self.put_or_call, self.settlement);
        let per_unit = match (quote_a, quote_b) {
            (CommodityVol::Lognormal(v_a), CommodityVol::Lognormal(v_b)) => {
                kirk(f_a, f_b, k, v_a, v_b, rho, r, t, pc, s)?
            }
            (
                CommodityVol::ShiftedLognormal {
                    vol: v_a,
                    shift: s_a,
                },
                CommodityVol::ShiftedLognormal {
                    vol: v_b,
                    shift: s_b,
                },
            ) => {
                // max(Fa - Fb - K) = max((Fa+sa) - (Fb+sb) - (K + sa - sb))
                kirk(
                    f_a + s_a,
                    f_b + s_b,
                    k + s_a - s_b,
                    v_a,
                    v_b,
                    rho,
                    r,
                    t,
                    pc,
                    s,
                )?
            }
            (CommodityVol::Normal(v_a), CommodityVol::Normal(v_b)) => {
                // a difference of joint normals is normal: exact
                let spread_vol = composite_vol(v_a, v_b, rho, 1.0);
                bachelier::price(f_a - f_b, k, r, spread_vol, t, pc, s)
            }
            _ => {
                return Err(RustyQLibError::invalid_input(
                    "spread option",
                    "both legs must quote the same model family \
                     (lognormal, shifted lognormal, or normal)",
                ));
            }
        };
        Ok(self.quantity * per_unit)
    }

    /// Delta of leg A (per $1 of its curve), by central bump.
    pub fn delta_a(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        vol_a: impl Into<CommodityVol>,
        vol_b: impl Into<CommodityVol>,
        rho: f64,
    ) -> Result<f64, RustyQLibError> {
        let (qa, qb) = (vol_a.into(), vol_b.into());
        let h = 1e-4;
        let up = self.price(discount, &forward_a.bumped(h)?, forward_b, qa, qb, rho)?;
        let down = self.price(discount, &forward_a.bumped(-h)?, forward_b, qa, qb, rho)?;
        Ok((up - down) / (2.0 * h))
    }

    /// Delta of leg B (per $1 of its curve), by central bump.
    pub fn delta_b(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        vol_a: impl Into<CommodityVol>,
        vol_b: impl Into<CommodityVol>,
        rho: f64,
    ) -> Result<f64, RustyQLibError> {
        let (qa, qb) = (vol_a.into(), vol_b.into());
        let h = 1e-4;
        let up = self.price(discount, forward_a, &forward_b.bumped(h)?, qa, qb, rho)?;
        let down = self.price(discount, forward_a, &forward_b.bumped(-h)?, qa, qb, rho)?;
        Ok((up - down) / (2.0 * h))
    }

    /// Correlation sensitivity (cega): PV change per unit of `rho`, by
    /// central bump clamped to `[-1, 1]`.
    pub fn cega(
        &self,
        discount: &YieldCurve,
        forward_a: &CommodityForwardCurve,
        forward_b: &CommodityForwardCurve,
        vol_a: impl Into<CommodityVol>,
        vol_b: impl Into<CommodityVol>,
        rho: f64,
    ) -> Result<f64, RustyQLibError> {
        let (qa, qb) = (vol_a.into(), vol_b.into());
        let h = 1e-4;
        let (up, down) = ((rho + h).min(1.0), (rho - h).max(-1.0));
        let v_up = self.price(discount, forward_a, forward_b, qa, qb, up)?;
        let v_down = self.price(discount, forward_a, forward_b, qa, qb, down)?;
        Ok((v_up - v_down) / (up - down))
    }
}

/// Kirk's approximation: Black-76 on `F_a` struck at `F_b + K` with the
/// composite vol. Needs both (displaced) legs and the effective strike
/// positive.
#[allow(clippy::too_many_arguments)]
fn kirk(
    f_a: f64,
    f_b: f64,
    k: f64,
    v_a: f64,
    v_b: f64,
    rho: f64,
    r: f64,
    t: f64,
    put_or_call: PutOrCall,
    settlement: FuturesSettlement,
) -> Result<f64, RustyQLibError> {
    if f_a <= 0.0 || f_b <= 0.0 || f_b + k <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "spread option",
            format!(
                "Kirk needs positive (shifted) legs and F_b + K > 0, got \
                 F_a = {f_a}, F_b = {f_b}, K = {k}; use larger shifts or normal vols"
            ),
        ));
    }
    let w = f_b / (f_b + k);
    let sigma = composite_vol(v_a, v_b, rho, w);
    Ok(black76::price(
        f_a,
        f_b + k,
        r,
        sigma,
        t,
        put_or_call,
        settlement,
    ))
}

/// The vol of `sigma_a dW_a - w sigma_b dW_b`: the composite vol Kirk
/// puts into Black-76 (and, at `w = 1`, the spread vol of two normal
/// legs).
///
/// Evaluated as `(v_a - rho v_b w)^2 + (v_b w)^2 (1 - rho^2)`, a sum of
/// squares, rather than the algebraically equal
/// `v_a^2 - 2 rho v_a v_b w + (v_b w)^2`. The two agree in exact
/// arithmetic, but the second cancels catastrophically at `rho = ±1`
/// with the legs' vols close: the three terms nearly annihilate and
/// rounding can leave the variance a few ulps **negative**, whose
/// square root is `NaN` — a silent one that propagates all the way out
/// as an `Ok(NaN)` premium. In this form each addend is non-negative by
/// construction, so the variance cannot go below zero by rounding; the
/// clamp only guards a `1 - rho^2` that rounds below zero just past
/// `|rho| = 1`.
fn composite_vol(v_a: f64, v_b: f64, rho: f64, w: f64) -> f64 {
    let bw = v_b * w;
    let var = (v_a - rho * bw).powi(2) + bw * bw * (1.0 - rho * rho);
    var.max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    // one Act/365 year from the curve reference, so t = 1.0 exactly
    const REF: (i32, u32, u32) = (2026, 9, 1);
    const EXPIRY: (i32, u32, u32) = (2027, 9, 1);

    fn flat_discount(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(REF.0, REF.1, REF.2),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    fn flat_curve(price: f64) -> CommodityForwardCurve {
        CommodityForwardCurve::flat(price, d(REF.0, REF.1, REF.2)).unwrap()
    }

    fn spread_call(strike: f64) -> CommoditySpreadOption {
        CommoditySpreadOption::new(
            1_000.0,
            strike,
            PutOrCall::Call,
            d(EXPIRY.0, EXPIRY.1, EXPIRY.2),
            d(2027, 9, 20),
            d(2027, 9, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap()
    }

    /// Bivariate GBM / arithmetic Monte Carlo for the spread payoff at
    /// t = 1, discounted on `df`, per unit. The market is the two
    /// forwards and the strike; the model is the two vols, their
    /// correlation, and whether the legs are lognormal.
    fn spread_mc(
        (f_a, f_b, k): (f64, f64, f64),
        (v_a, v_b, rho): (f64, f64, f64),
        df: f64,
        lognormal: bool,
    ) -> f64 {
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let paths = 200_000;
        let mut sum = 0.0;
        for _ in 0..paths {
            let z1: f64 = StandardNormal.sample(&mut rng);
            let e: f64 = StandardNormal.sample(&mut rng);
            for sign in [1.0, -1.0] {
                let (z_a, z_b) = (
                    sign * z1,
                    rho * sign * z1 + (1.0 - rho * rho).sqrt() * sign * e,
                );
                let (s_a, s_b) = if lognormal {
                    (
                        f_a * (v_a * z_a - 0.5 * v_a * v_a).exp(),
                        f_b * (v_b * z_b - 0.5 * v_b * v_b).exp(),
                    )
                } else {
                    (f_a + v_a * z_a, f_b + v_b * z_b)
                };
                sum += (s_a - s_b - k).max(0.0);
            }
        }
        df * sum / (2.0 * paths as f64)
    }

    #[test]
    fn margrabe_limit_matches_monte_carlo() {
        // K = 0: Kirk is exact (Margrabe's exchange option)
        let discount = flat_discount(0.04);
        let (brent, wti) = (flat_curve(76.0), flat_curve(72.0));
        let option = spread_call(0.0);
        let price = option
            .price(&discount, &brent, &wti, 0.32, 0.35, 0.85)
            .unwrap();
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let mc = 1_000.0 * spread_mc((76.0, 72.0, 0.0), (0.32, 0.35, 0.85), df, true);
        assert!((price - mc).abs() < 0.01 * price, "kirk {price} vs MC {mc}");
    }

    #[test]
    fn kirk_tracks_monte_carlo_at_nonzero_strike() {
        let discount = flat_discount(0.04);
        let (brent, wti) = (flat_curve(76.0), flat_curve(72.0));
        let option = spread_call(3.0);
        let price = option
            .price(&discount, &brent, &wti, 0.32, 0.35, 0.85)
            .unwrap();
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let mc = 1_000.0 * spread_mc((76.0, 72.0, 3.0), (0.32, 0.35, 0.85), df, true);
        // Kirk approximation error + MC noise, both well inside 1.5%
        assert!(
            (price - mc).abs() < 0.015 * price,
            "kirk {price} vs MC {mc}"
        );
    }

    #[test]
    fn normal_model_is_exact_against_monte_carlo() {
        let discount = flat_discount(0.04);
        // a negative-price leg: only the normal model can hold it
        let (waha, hub) = (flat_curve(-1.5), flat_curve(2.75));
        let option = spread_call(-4.0);
        let (v_a, v_b, rho) = (1.6, 1.1, 0.6);
        let price = option
            .price(
                &discount,
                &waha,
                &hub,
                CommodityVol::Normal(v_a),
                CommodityVol::Normal(v_b),
                rho,
            )
            .unwrap();
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let mc = 1_000.0 * spread_mc((-1.5, 2.75, -4.0), (v_a, v_b, rho), df, false);
        assert!(
            (price - mc).abs() < 0.005 * price,
            "normal {price} vs MC {mc}"
        );
    }

    #[test]
    fn put_call_parity_all_models() {
        let discount = flat_discount(0.04);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let parity = 1_000.0 * df * (76.0 - 72.0 - 3.0);
        let call = spread_call(3.0);
        let mut put = call.clone();
        put.put_or_call = PutOrCall::Put;
        for (qa, qb) in [
            (CommodityVol::Lognormal(0.32), CommodityVol::Lognormal(0.35)),
            (
                CommodityVol::ShiftedLognormal {
                    vol: 0.30,
                    shift: 8.0,
                },
                CommodityVol::ShiftedLognormal {
                    vol: 0.33,
                    shift: 5.0,
                },
            ),
            (CommodityVol::Normal(24.0), CommodityVol::Normal(25.0)),
        ] {
            let c = call.price(&discount, &a, &b, qa, qb, 0.85).unwrap();
            let p = put.price(&discount, &a, &b, qa, qb, 0.85).unwrap();
            assert!(
                (c - p - parity).abs() < 1e-8,
                "{qa:?}: {} vs {parity}",
                c - p
            );
        }
    }

    #[test]
    fn shifted_legs_equal_kirk_on_the_displaced_market() {
        let discount = flat_discount(0.04);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        let option = spread_call(3.0);
        let (s_a, s_b) = (8.0, 5.0);
        let via_shift = option
            .price(
                &discount,
                &a,
                &b,
                CommodityVol::ShiftedLognormal {
                    vol: 0.30,
                    shift: s_a,
                },
                CommodityVol::ShiftedLognormal {
                    vol: 0.33,
                    shift: s_b,
                },
                0.85,
            )
            .unwrap();
        // displaced market: curves + shifts, strike + (s_a - s_b)
        let displaced = spread_call(3.0 + s_a - s_b);
        let via_displacement = displaced
            .price(
                &discount,
                &flat_curve(76.0 + s_a),
                &flat_curve(72.0 + s_b),
                0.30,
                0.33,
                0.85,
            )
            .unwrap();
        assert!(
            (via_shift - via_displacement).abs() < 1e-8,
            "{via_shift} vs {via_displacement}"
        );
    }

    #[test]
    fn correlation_and_vol_shape_the_premium() {
        let discount = flat_discount(0.04);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        let option = spread_call(3.0);
        // higher correlation compresses the spread distribution
        let lo = option.price(&discount, &a, &b, 0.32, 0.35, 0.30).unwrap();
        let hi = option.price(&discount, &a, &b, 0.32, 0.35, 0.90).unwrap();
        assert!(hi < lo, "{hi} vs {lo}");
        let cega = option.cega(&discount, &a, &b, 0.32, 0.35, 0.85).unwrap();
        assert!(cega < 0.0);
        // perfectly correlated identical legs: the spread is frozen at 0
        let same = flat_curve(72.0);
        let frozen = spread_call(0.0)
            .price(&discount, &same, &same, 0.35, 0.35, 1.0)
            .unwrap();
        assert!(frozen.abs() < 1e-10, "{frozen}");
        // zero vol prices the discounted intrinsic
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let zero = option.price(&discount, &a, &b, 0.0, 0.0, 0.85).unwrap();
        assert!((zero - 1_000.0 * df * 1.0).abs() < 1e-8, "{zero}");
    }

    #[test]
    fn deltas_have_spread_signs() {
        let discount = flat_discount(0.04);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        let option = spread_call(3.0);
        let da = option.delta_a(&discount, &a, &b, 0.32, 0.35, 0.85).unwrap();
        let db = option.delta_b(&discount, &a, &b, 0.32, 0.35, 0.85).unwrap();
        // long the received leg, short the paid leg
        assert!(da > 0.0 && db < 0.0, "{da} / {db}");
    }

    #[test]
    fn perfect_correlation_never_rounds_the_variance_negative() {
        // at rho = +-1 the composite variance is a difference of nearly
        // equal terms: in the cancelling form it rounds negative and
        // sqrt gives a silent NaN premium. Walk vols either side of
        // equality, both models, both signs of rho.
        let discount = flat_discount(0.04);
        let option = spread_call(0.0);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        for rho in [1.0, -1.0] {
            for eps in [0.0, 1e-9, -1e-9, 1e-15, -1e-15] {
                let v_b = 0.35;
                let v_a = v_b * (1.0 + eps);
                let kirk = option.price(&discount, &a, &b, v_a, v_b, rho).unwrap();
                assert!(kirk.is_finite() && kirk >= 0.0, "kirk rho={rho} eps={eps}");
                let normal = option
                    .price(
                        &discount,
                        &a,
                        &b,
                        CommodityVol::Normal(25.0 * (1.0 + eps)),
                        CommodityVol::Normal(25.0),
                        rho,
                    )
                    .unwrap();
                assert!(
                    normal.is_finite() && normal >= 0.0,
                    "normal rho={rho} eps={eps}"
                );
            }
        }
        // the exactly-degenerate case: identical legs, rho = 1, K = 0 is
        // a frozen spread worth nothing
        let same = flat_curve(72.0);
        let frozen = option
            .price(&discount, &same, &same, 0.35, 0.35, 1.0)
            .unwrap();
        assert_eq!(frozen, 0.0);
        // and the composite vol itself is exact where it can be checked
        assert_eq!(composite_vol(0.35, 0.35, 1.0, 1.0), 0.0);
        assert!((composite_vol(0.3, 0.4, -1.0, 1.0) - 0.7).abs() < 1e-15);
        // uncorrelated legs: the plain quadrature
        let root = (0.3f64 * 0.3 + 0.4 * 0.4).sqrt();
        assert!((composite_vol(0.3, 0.4, 0.0, 1.0) - root).abs() < 1e-15);
    }

    #[test]
    fn validation_and_model_mixing_errors() {
        let discount = flat_discount(0.04);
        let (a, b) = (flat_curve(76.0), flat_curve(72.0));
        let expiry = d(EXPIRY.0, EXPIRY.1, EXPIRY.2);
        let s = FuturesSettlement::Discounted;
        assert!(
            CommoditySpreadOption::new(0.0, 3.0, PutOrCall::Call, expiry, expiry, expiry, s)
                .is_err()
        );
        // underlying before expiry
        assert!(CommoditySpreadOption::new(
            1e3,
            3.0,
            PutOrCall::Call,
            expiry,
            d(2027, 8, 1),
            expiry,
            s
        )
        .is_err());
        let option = spread_call(3.0);
        // mixed model families are rejected
        assert!(option
            .price(
                &discount,
                &a,
                &b,
                CommodityVol::Lognormal(0.32),
                CommodityVol::Normal(25.0),
                0.85
            )
            .is_err());
        // correlation outside [-1, 1]
        assert!(option.price(&discount, &a, &b, 0.32, 0.35, 1.5).is_err());
        // a negative leg under Kirk
        let negative = flat_curve(-1.5);
        assert!(option
            .price(&discount, &negative, &b, 0.32, 0.35, 0.85)
            .is_err());
        // valuing after expiry
        let late = YieldCurve::flat(
            0.04,
            d(2027, 9, 2),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        assert!(option.price(&late, &a, &b, 0.32, 0.35, 0.85).is_err());
    }
}
