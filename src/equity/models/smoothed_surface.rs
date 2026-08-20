//! One interface for every smoothed implied-volatility surface, and one
//! implementation of everything downstream of it.
//!
//! A *smoothed surface* is what a parametric fit produces from noisy
//! quotes: [`SviSurfaceFit`](crate::equity::svi::SviSurfaceFit),
//! [`SabrSurfaceFit`](crate::equity::sabr::SabrSurfaceFit), an SSVI fit.
//! Each parameterization differs in exactly one respect — how it supplies
//! total variance and its derivatives
//! ([`SmoothedSurface::variance_derivatives`]) — and in nothing else.
//! Dupire's formula, the butterfly function, the guards, the clamps and
//! the differentiation stencil all live here, once.
//!
//! That is a deliberate design constraint, not tidiness. Local volatility
//! is a second-order functional of the surface (Gatheral's form divides
//! `dw/dt` by a denominator containing `w_kk`), so it magnifies whatever
//! the smoother left behind — which makes it the sharpest way to compare
//! parameterizations, and also the easiest place to compare them
//! *unfairly*. If SVI were differentiated in closed form while SABR went
//! through a numerical stencil with a different step, a difference in
//! measured local-vol roughness would say more about the two code paths
//! than about the two models. Routing every contender through this module
//! makes that particular mistake unrepresentable.
//!
//! What each parameterization is then free to differ in — and what a
//! comparison legitimately measures — is whether its derivatives are
//! closed-form or numerical, and whether its term structure is continuous
//! or pieced together per expiry. Those are properties of the model.

// ── Shared guards ───────────────────────────────────────────────────────

/// Forward variance floor: `dw/dt` never drops below this, so local
/// variance stays positive even where fitted slices graze.
pub const MIN_FORWARD_VARIANCE: f64 = 1e-8;
/// Local vol clamps, matching [`LocalVol`](crate::equity::local_vol::LocalVol).
pub const MIN_LOCAL_VOL: f64 = 0.01;
pub const MAX_LOCAL_VOL: f64 = 3.0;
/// Below this total variance the Dupire quotient is meaningless and the
/// implied vol is returned instead (with the guard flagged).
pub const MIN_TOTAL_VARIANCE: f64 = 1e-8;
/// Below this density denominator the same fallback applies: a
/// non-positive `g` is a negative risk-neutral density, where Dupire's
/// formula has no admissible root.
pub const MIN_DENOMINATOR: f64 = 1e-4;
/// Time floor, so a zero-maturity query cannot divide by zero.
pub const MIN_TIME: f64 = 1e-4;
/// **The** central-difference step in log-moneyness, shared by every
/// parameterization without closed-form derivatives. Comparisons across
/// smoothers are only attributable if the stencil is identical, so this
/// constant is deliberately not a per-model tuning knob.
pub const K_DIFF_STEP: f64 = 1e-4;

// ── Total variance and its derivatives ──────────────────────────────────

/// Total implied variance `w = sigma^2 t` at one `(k, t)` point, with the
/// three derivatives Dupire's formula consumes: twice in log-moneyness,
/// once in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceDerivatives {
    /// `w(k, t)`.
    pub w: f64,
    /// `dw/dk`.
    pub dk: f64,
    /// `d2w/dk2`.
    pub dkk: f64,
    /// `dw/dt` — the forward variance.
    pub dt: f64,
}

/// Gatheral's butterfly function `g(k)`, the risk-neutral density up to a
/// positive factor and the denominator of Dupire's formula:
///
/// ```text
/// g = (1 - k w_k / (2w))^2 - (w_k^2 / 4)(1/w + 1/4) + w_kk / 2
/// ```
///
/// Negative values are a negative density. Sharing one implementation
/// between the fitted-surface diagnostics and the local-vol denominator
/// is what makes "this smile carries butterfly arbitrage here" and "the
/// local-vol guard fired there" the *same* statement rather than two
/// independently drifting ones.
pub fn butterfly_g(d: &VarianceDerivatives, k: f64) -> f64 {
    (1.0 - k * d.dk / (2.0 * d.w)).powi(2) - (d.dk * d.dk / 4.0) * (1.0 / d.w + 0.25)
        + d.dkk / 2.0
}

/// Dupire local volatility from total-variance derivatives:
/// `sigma_loc^2 = (dw/dt) / g(k)`, clamped to `[1%, 300%]`.
///
/// Returns `(sigma_loc, guard_fired)`. `guard_fired` is `true` when the
/// quotient was not usable — vanishing total variance, or a non-positive
/// density denominator — and the implied volatility was substituted
/// instead. Counting those substitutions is the point of returning the
/// flag: a silent fallback is indistinguishable from a good calibration
/// in the output, which is precisely how unusable surfaces reach
/// production.
pub fn dupire(d: &VarianceDerivatives, k: f64, t: f64) -> (f64, bool) {
    let t = t.max(MIN_TIME);
    let dwdt = d.dt.max(MIN_FORWARD_VARIANCE);
    if d.w < MIN_TOTAL_VARIANCE {
        return (
            (d.w.max(1e-12) / t)
                .sqrt()
                .clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL),
            true,
        );
    }
    let denominator = butterfly_g(d, k);
    if denominator <= MIN_DENOMINATOR {
        return ((d.w / t).sqrt().clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL), true);
    }
    (
        (dwdt / denominator)
            .sqrt()
            .clamp(MIN_LOCAL_VOL, MAX_LOCAL_VOL),
        false,
    )
}

/// Central-difference `[w, w_k, w_kk]` at `k` on the shared stencil, for
/// parameterizations whose strike derivatives have no usable closed form
/// (SABR through Hagan's expansion). Three evaluations, second-order
/// accurate in [`K_DIFF_STEP`].
pub fn numeric_k_derivatives<F>(total_variance: F, k: f64) -> [f64; 3]
where
    F: Fn(f64) -> f64,
{
    let h = K_DIFF_STEP;
    let w = total_variance(k);
    let up = total_variance(k + h);
    let down = total_variance(k - h);
    [w, (up - down) / (2.0 * h), (up - 2.0 * w + down) / (h * h)]
}

/// Total variance and derivatives at `t` from two bracketing per-expiry
/// slices, under the library's linear-total-variance-in-time rule.
///
/// `early`/`late` carry `(slice time, [w, w_k, w_kk])` evaluated at the
/// same log-moneyness; `weight` is the interpolation weight on the later
/// slice (as produced by each fit's `bracket`). A single-slice or
/// out-of-range bracket passes the same slice twice, which selects the
/// extrapolation branches:
///
/// - **below the first pillar**, variance accrues proportionally from
///   zero, so `w(t) = w(t_1) t / t_1` and `dw/dt = w(t_1) / t_1`;
/// - **at or beyond the last pillar**, the smile is flat-extrapolated and
///   the forward variance comes from `beyond_last_dwdt` (each fit's last
///   inter-slice segment);
/// - **between pillars**, everything is linear in `t` and the forward
///   variance is the segment's slope.
///
/// Grazing slices — where the later fit dips below the earlier at this
/// `k` — are floored to the earlier slice, so the interpolated forward
/// variance is never negative. Shared by every per-expiry smoother, so
/// that the one structural artifact of this rule (piecewise-constant
/// `dw/dt`, hence a local volatility that jumps in time at each pillar)
/// is identical across them rather than an accident of each
/// implementation.
pub fn interpolate_slices(
    early: (f64, [f64; 3]),
    late: (f64, [f64; 3]),
    t: f64,
    weight: f64,
    beyond_last_dwdt: impl FnOnce() -> f64,
) -> VarianceDerivatives {
    let (ta, da) = early;
    let (tb, mut db) = late;
    if db[0] < da[0] {
        db = da;
    }
    if t <= ta {
        let scale = (t / ta).min(1.0);
        VarianceDerivatives {
            w: da[0] * scale,
            dk: da[1] * scale,
            dkk: da[2] * scale,
            dt: da[0] / ta,
        }
    } else if ta == tb {
        VarianceDerivatives {
            w: da[0],
            dk: da[1],
            dkk: da[2],
            dt: beyond_last_dwdt(),
        }
    } else {
        VarianceDerivatives {
            w: da[0] + (db[0] - da[0]) * weight,
            dk: da[1] + (db[1] - da[1]) * weight,
            dkk: da[2] + (db[2] - da[2]) * weight,
            dt: (db[0] - da[0]) / (tb - ta),
        }
    }
}

// ── The interface ───────────────────────────────────────────────────────

/// A smoothed implied-volatility surface: forward curve, total variance,
/// and the derivatives everything downstream needs.
///
/// Implementors supply the first three methods; volatility, local
/// volatility and the butterfly diagnostic follow from them identically
/// for every parameterization.
pub trait SmoothedSurface {
    /// Forward at expiry `t`, the reference the surface's log-moneyness
    /// is measured against.
    fn forward(&self, t: f64) -> f64;

    /// Total variance at log-moneyness `k` and expiry `t`.
    fn total_variance(&self, k: f64, t: f64) -> f64;

    /// Total variance and its derivatives at `(k, t)` — the one method
    /// where parameterizations legitimately differ (closed-form versus
    /// numerical in `k`, continuous versus per-expiry in `t`).
    fn variance_derivatives(&self, k: f64, t: f64) -> VarianceDerivatives;

    /// Implied volatility for an absolute `strike` at expiry `t`.
    fn vol(&self, strike: f64, t: f64) -> f64 {
        let k = (strike / self.forward(t)).ln();
        (self.total_variance(k, t).max(1e-12) / t.max(MIN_TIME)).sqrt()
    }

    /// Dupire local volatility at underlying `level` and time `t`, with
    /// the guard flag: `true` means implied vol was substituted for an
    /// unusable Dupire quotient (see [`dupire`]).
    fn local_vol_checked(&self, level: f64, t: f64) -> (f64, bool) {
        let t = t.max(MIN_TIME);
        let k = (level / self.forward(t)).ln();
        dupire(&self.variance_derivatives(k, t), k, t)
    }

    /// Dupire local volatility, discarding the guard flag.
    fn local_vol(&self, level: f64, t: f64) -> f64 {
        self.local_vol_checked(level, t).0
    }

    /// Gatheral's `g(k)` for the fitted surface at `(k, t)` — negative
    /// means the smoothed smile carries butterfly arbitrage there.
    fn butterfly_g_at(&self, k: f64, t: f64) -> f64 {
        butterfly_g(&self.variance_derivatives(k, t.max(MIN_TIME)), k)
    }

    /// Sample [`local_vol`](Self::local_vol) on a `levels` x `times` grid
    /// (`grid[i][j]` = level i, time j — the plotting layout).
    fn local_vol_grid(&self, levels: &[f64], times: &[f64]) -> Vec<Vec<f64>> {
        levels
            .iter()
            .map(|&level| times.iter().map(|&t| self.local_vol(level, t)).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat smile: `w = sigma^2 t`, no strike dependence. Dupire must
    /// return `sigma` exactly, since `g = 1` and `dw/dt = sigma^2`.
    #[test]
    fn flat_surface_gives_back_its_own_volatility() {
        let sigma: f64 = 0.28;
        for t in [0.1, 1.0, 3.0] {
            for k in [-0.4, 0.0, 0.35] {
                let d = VarianceDerivatives {
                    w: sigma * sigma * t,
                    dk: 0.0,
                    dkk: 0.0,
                    dt: sigma * sigma,
                };
                assert!((butterfly_g(&d, k) - 1.0).abs() < 1e-14, "g at k={k}");
                let (lv, guarded) = dupire(&d, k, t);
                assert!(!guarded);
                assert!((lv - sigma).abs() < 1e-14, "t={t} k={k}: {lv}");
            }
        }
    }

    #[test]
    fn guards_fire_and_flag_themselves() {
        // vanishing total variance
        let (_, guarded) = dupire(
            &VarianceDerivatives {
                w: 1e-12,
                dk: 0.0,
                dkk: 0.0,
                dt: 0.04,
            },
            0.0,
            1.0,
        );
        assert!(guarded, "vanishing variance must flag");
        // negative density: a large negative w_kk drives g below zero
        let d = VarianceDerivatives {
            w: 0.04,
            dk: 0.0,
            dkk: -4.0,
            dt: 0.04,
        };
        assert!(butterfly_g(&d, 0.0) < 0.0);
        let (lv, guarded) = dupire(&d, 0.0, 1.0);
        assert!(guarded, "negative density must flag");
        // the fallback is the implied vol, clamped
        assert!((lv - 0.2).abs() < 1e-12, "fallback should be sqrt(w/t)");
    }

    #[test]
    fn local_vol_is_clamped_to_the_band() {
        // a tiny denominator would send the quotient to infinity
        let d = VarianceDerivatives {
            w: 0.04,
            dk: 0.0,
            dkk: 0.0,
            dt: 1e6,
        };
        let (lv, guarded) = dupire(&d, 0.0, 1.0);
        assert!(!guarded);
        assert_eq!(lv, MAX_LOCAL_VOL);
    }

    #[test]
    fn numeric_derivatives_match_a_known_quadratic() {
        // w(k) = 0.04 + 0.1 k + 0.5 k^2  =>  w_k = 0.1 + k, w_kk = 1
        let w = |k: f64| 0.04 + 0.1 * k + 0.5 * k * k;
        for k in [-0.3, 0.0, 0.45] {
            let [w0, dk, dkk] = numeric_k_derivatives(w, k);
            assert!((w0 - w(k)).abs() < 1e-15);
            assert!((dk - (0.1 + k)).abs() < 1e-8, "w_k at {k}: {dk}");
            assert!((dkk - 1.0).abs() < 1e-4, "w_kk at {k}: {dkk}");
        }
    }

    #[test]
    fn slice_interpolation_follows_the_documented_rules() {
        let early = (0.5, [0.02, 0.01, 0.30]);
        let late = (1.0, [0.05, 0.02, 0.40]);

        // between pillars: linear in t, forward variance = segment slope
        let mid = interpolate_slices(early, late, 0.75, 0.5, || panic!("not beyond"));
        assert!((mid.w - 0.035).abs() < 1e-14);
        assert!((mid.dk - 0.015).abs() < 1e-14);
        assert!((mid.dt - (0.05 - 0.02) / 0.5).abs() < 1e-14);

        // below the first pillar: proportional accrual
        let below = interpolate_slices(early, early, 0.25, 0.0, || panic!("not beyond"));
        assert!((below.w - 0.01).abs() < 1e-14, "{}", below.w);
        assert!((below.dt - 0.04).abs() < 1e-14);

        // beyond the last pillar: flat smile, supplied forward variance
        let beyond = interpolate_slices(late, late, 1.5, 0.0, || 0.06);
        assert!((beyond.w - 0.05).abs() < 1e-14);
        assert!((beyond.dt - 0.06).abs() < 1e-14);

        // grazing slices: the later variance is floored at the earlier,
        // so the forward variance cannot go negative
        let grazing = interpolate_slices(early, (1.0, [0.015, 0.0, 0.0]), 0.75, 0.5, || {
            panic!("not beyond")
        });
        assert!(grazing.dt >= 0.0, "dt {}", grazing.dt);
        assert!((grazing.w - 0.02).abs() < 1e-14);
    }
}
