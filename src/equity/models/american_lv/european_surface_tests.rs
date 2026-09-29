//! Tests of the dense European surface: the Dupire forward march against
//! Black-Scholes closed forms (flat, time-dependent and sloped-curve
//! cases), a cash-dividend case against a one-dimensional quadrature
//! reference and put-call parity, the implied-vol inversion with
//! effective flat rates, and the static-arbitrage scans.

use super::*;
use crate::core::fd_solvers::thomas_algorithm;
use crate::equity::american_lv::CallbackVol;
use crate::equity::blackscholes::{bs_price, bs_vega};

/// A tiny deterministic generator (LCG) so the tests need no `rand`.
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

const S0: f64 = 100.0;
const ONE_WEEK: f64 = 1.0 / 52.0;
const ONE_MONTH: f64 = 1.0 / 12.0;

/// The dense grid every surface test uses (the grid recommended for the
/// study's display surface): `ln 100` on a node, half-width 1.0 in
/// log-strike, `dk = 1/800`.
fn test_grid() -> Vec<f64> {
    log_strike_grid(S0, 1.0, 800)
}

/// The listed strikes of the synthetic study.
fn listed_strikes() -> Vec<f64> {
    (0..=24).map(|i| 70.0 + 2.5 * i as f64).collect()
}

/// A listed quote is 'informative' when its Black-Scholes vega exceeds
/// `5e-3 * S0` (half a cent per vol bp at `S0 = 100`): below that the
/// inversion amplifies the tail's discretization error beyond what a
/// half-bp vol comparison can say (at one week that keeps `|d| <= 2.3`).
fn informative(k: f64, r: f64, q: f64, sigma: f64, t: f64) -> bool {
    bs_vega(S0, k, r, q, sigma, t) >= 5e-3 * S0
}

/// Black-Scholes price under a general (r, q) term structure through the
/// effective flat rates.
fn bs_price_ts(market: &ForwardMarket, k: f64, sigma: f64, t: f64, right: PutOrCall) -> f64 {
    let (r, q) = market.effective_rates(t);
    bs_price(market.s0, k, r, q, sigma, t, right)
}

// ── Kernels ──────────────────────────────────────────────────────────────

#[test]
fn thomas_inplace_matches_the_crate_thomas_algorithm() {
    let mut rng = Lcg(12345);
    for &n in &[1usize, 2, 3, 17, 200] {
        let b: Vec<f64> = (0..n).map(|_| 4.0 + rng.next_f64()).collect();
        let a: Vec<f64> = (0..n.saturating_sub(1))
            .map(|_| rng.next_f64() - 0.5)
            .collect();
        let c: Vec<f64> = (0..n.saturating_sub(1))
            .map(|_| rng.next_f64() - 0.5)
            .collect();
        let d: Vec<f64> = (0..n).map(|_| 10.0 * (rng.next_f64() - 0.5)).collect();
        let reference = thomas_algorithm(&a, &b, &c, &d);
        let (mut cw, mut dw, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        assert!(
            thomas_inplace(&a, &b, &c, &d, &mut cw, &mut dw, &mut x),
            "diagonally dominant system must not break down (n = {n})"
        );
        for i in 0..n {
            assert!(
                (x[i] - reference[i]).abs() <= 1e-14 * (1.0 + reference[i].abs()),
                "n = {n}, row {i}: in-place {} vs crate {}",
                x[i],
                reference[i]
            );
        }
    }
    let (mut cw, mut dw, mut x) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    assert!(
        !thomas_inplace(
            &[1.0],
            &[1.0, 1.0],
            &[1.0],
            &[1.0, 1.0],
            &mut cw,
            &mut dw,
            &mut x
        ),
        "a singular system reports a breakdown instead of panicking"
    );
}

#[test]
fn lagrange4_reproduces_cubics_exactly_and_clamps_the_stencil() {
    let k0 = -0.3;
    let dk = 0.05;
    let f: Vec<f64> = (0..12)
        .map(|j| {
            let x = k0 + j as f64 * dk;
            1.0 - 2.0 * x + 0.5 * x * x - 0.25 * x * x * x
        })
        .collect();
    let cubic = |x: f64| 1.0 - 2.0 * x + 0.5 * x * x - 0.25 * x * x * x;
    for &x in &[-0.3, -0.29, -0.1234, 0.0, 0.17, 0.249, 0.25, -0.35, 0.3] {
        let v = lagrange4(k0, dk, &f, x);
        assert!(
            (v - cubic(x)).abs() < 1e-13,
            "cubic reproduced at x = {x}: {v} vs {}",
            cubic(x)
        );
    }
}

#[test]
fn cell_average_payoff_is_the_exact_integral_of_the_kink() {
    let dk = 0.01;
    for &k in &[
        S0.ln() - 0.2,
        S0.ln() - 0.004,
        S0.ln(),
        S0.ln() + 0.003,
        S0.ln() + 0.2,
    ] {
        for right in [PutOrCall::Call, PutOrCall::Put] {
            let payoff = |x: f64| match right {
                PutOrCall::Call => (S0 - x.exp()).max(0.0),
                PutOrCall::Put => (x.exp() - S0).max(0.0),
            };
            // Simpson with a very fine sub-grid inside the cell
            let m = 20000;
            let a = k - 0.5 * dk;
            let h = dk / m as f64;
            let mut sum = payoff(a) + payoff(a + dk);
            for i in 1..m {
                let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                sum += w * payoff(a + i as f64 * h);
            }
            let reference = sum * h / 3.0 / dk;
            let v = cell_average_payoff(k, dk, S0, right);
            assert!(
                (v - reference).abs() < 1e-6,
                "{right:?} cell average at k = {k}: {v} vs {reference}"
            );
        }
    }
}

// ── Term structures, effective rates and the time grid ───────────────────

#[test]
fn term_structure_integral_and_average_are_exact() {
    let ts = TermStructure::piecewise(vec![0.25, 0.5, 1.0], vec![0.02, 0.03, 0.04, 0.05]).unwrap();
    assert_eq!(ts.at(0.0), 0.02);
    assert_eq!(ts.at(0.25), 0.03, "right-continuous at a break");
    assert_eq!(ts.at(0.9), 0.04);
    assert_eq!(ts.at(7.0), 0.05, "flat beyond the last break");
    let i = ts.integral(0.1, 0.8);
    let expected = 0.02 * 0.15 + 0.03 * 0.25 + 0.04 * 0.3;
    assert!((i - expected).abs() < 1e-15, "integral {i} vs {expected}");
    assert!(
        (ts.integral(0.8, 0.1) + expected).abs() < 1e-15,
        "antisymmetric"
    );
    assert!(
        (ts.average(0.1, 0.8) - expected / 0.7).abs() < 1e-15,
        "average"
    );
    assert_eq!(ts.average(0.3, 0.3), 0.03, "empty interval reads the value");
    assert!(
        TermStructure::piecewise(vec![0.5, 0.25], vec![1.0, 1.0, 1.0]).is_err(),
        "non-increasing breaks rejected"
    );
    assert!(
        TermStructure::piecewise(vec![0.5], vec![1.0]).is_err(),
        "value count must be breaks + 1"
    );
}

#[test]
fn term_structure_from_discount_factors_reproduces_the_pillars() {
    let times: [f64; 5] = [0.1, 0.25, 0.5, 1.0, 2.0];
    let rates: [f64; 5] = [0.03, 0.035, 0.04, 0.045, 0.05];
    let dfs: Vec<f64> = times
        .iter()
        .zip(&rates)
        .map(|(&t, &r)| (-r * t).exp())
        .collect();
    let ts = TermStructure::from_discount_factors(&times, &dfs).unwrap();
    for (&t, &d) in times.iter().zip(&dfs) {
        let model = (-ts.integral(0.0, t)).exp();
        assert!(
            (model - d).abs() < 1e-14,
            "discount factor at {t}: {model} vs {d}"
        );
    }
    assert_eq!(ts.breaks().len(), 4);
    assert_eq!(ts.values().len(), 5);
    assert!(
        TermStructure::from_discount_factors(&[0.5, 0.25], &[0.99, 0.98]).is_err(),
        "non-increasing times rejected"
    );
    assert!(
        TermStructure::from_discount_factors(&[0.5], &[-1.0]).is_err(),
        "negative discount factor rejected"
    );
}

#[test]
fn effective_flat_rates_reproduce_discount_factor_and_forward() {
    let rate =
        TermStructure::piecewise(vec![0.1, 0.25, 0.5], vec![0.03, 0.04, 0.045, 0.05]).unwrap();
    let carry = TermStructure::piecewise(vec![0.3], vec![0.01, 0.02]).unwrap();
    let market = ForwardMarket::new(S0, rate, carry, vec![(0.2, 1.5), (0.7, 1.5)]).unwrap();
    for &t in &[ONE_WEEK, 0.25, 0.5, 1.0] {
        let (r, q) = market.effective_rates(t);
        let df = market.df(t);
        let fwd = market.forward(t);
        assert!(
            ((-r * t).exp() - df).abs() < 1e-14,
            "df at {t}: {} vs {df}",
            (-r * t).exp()
        );
        assert!(
            (S0 * ((r - q) * t).exp() - fwd).abs() < 1e-11,
            "forward at {t}: {} vs {fwd}",
            S0 * ((r - q) * t).exp()
        );
        // the Black-Scholes price with (r_eff, q_eff) equals df x Black-76
        let k = 105.0;
        let b76 = df * bs_price(fwd, k, 0.0, 0.0, 0.2, t, PutOrCall::Call);
        let bs = bs_price(S0, k, r, q, 0.2, t, PutOrCall::Call);
        assert!((b76 - bs).abs() < 1e-11, "Black-76 {b76} vs BS {bs} at {t}");
    }
    // dividends strictly before t reduce the forward by their grown amount
    let f_before = market.forward(0.2);
    let f_after = market.forward(0.2 + 1e-9);
    assert!(
        f_before > f_after + 1.4,
        "the dividend at 0.2 is excluded at t = 0.2 and included just after: {f_before} vs {f_after}"
    );
}

#[test]
fn graded_time_grid_has_the_required_resolution_and_hits_every_anchor() {
    let cfg = DupireConfig::default();
    let anchors = [ONE_WEEK, ONE_MONTH, 0.2, 1.0];
    let t = graded_time_grid(&anchors, &cfg).unwrap();
    assert_eq!(t[0], 0.0);
    assert!(t.windows(2).all(|w| w[1] > w[0]), "strictly increasing");
    let before_first = t.iter().filter(|&&v| v < ONE_WEEK).count();
    assert!(
        before_first >= cfg.min_steps_first,
        "{before_first} steps before the first maturity (need {})",
        cfg.min_steps_first
    );
    assert!(
        t[1] <= 2.0 * cfg.first_step,
        "first step {} is of the order of first_step",
        t[1]
    );
    for &a in &anchors {
        assert!(t.contains(&a), "anchor {a} is a node (bit-exact)");
    }
    let max_step = t.windows(2).map(|w| w[1] - w[0]).fold(0.0, f64::max);
    assert!(
        max_step <= cfg.max_dt * (1.0 + 1e-9),
        "largest step {max_step} <= max_dt {}",
        cfg.max_dt
    );
    // a long first anchor: the geometric part is capped at max_dt and
    // scaled to land exactly
    let t2 = graded_time_grid(&[1.0], &cfg).unwrap();
    assert_eq!(*t2.last().unwrap(), 1.0);
    assert!(
        t2.len() > 300,
        "roughly daily steps over one year: {}",
        t2.len()
    );
    assert!(
        graded_time_grid(&[], &cfg).is_err(),
        "empty anchors rejected"
    );
    assert!(
        graded_time_grid(&[0.5, 0.25], &cfg).is_err(),
        "unsorted anchors rejected"
    );
    let steps = geometric_steps(1e-4, 0.02, 40);
    let sum: f64 = steps.iter().sum();
    assert!(
        (sum - 0.02).abs() < 1e-15,
        "geometric steps sum to the total"
    );
    assert!(
        steps.windows(2).all(|w| w[1] > w[0]),
        "geometric steps grow"
    );
}

// ── Dupire forward vs Black-Scholes ──────────────────────────────────────

#[test]
fn flat_vol_reproduces_black_scholes_at_nodes_for_1w_1m_1y() {
    let (r, q, sigma) = (0.04, 0.02, 0.2);
    let market = ForwardMarket::flat(S0, r, q);
    let k = test_grid();
    let maturities = [ONE_WEEK, ONE_MONTH, 1.0];
    let cfg = DupireConfig::default();
    let mut ws = DupireWorkspace::new(k.len());
    for right in [PutOrCall::Call, PutOrCall::Put] {
        let surface = dupire_forward_with(
            &FlatVol(sigma),
            &market,
            &k,
            &maturities,
            &cfg,
            right,
            &mut ws,
        )
        .unwrap();
        assert_eq!(surface.upwinded_rows, 0, "no upwinding at sigma = 0.2");
        let mut worst = 0.0f64;
        for (i, &t) in maturities.iter().enumerate() {
            for (j, &kj) in k.iter().enumerate() {
                let strike = kj.exp();
                if !(70.0..=130.0).contains(&strike) {
                    continue;
                }
                let reference = bs_price(S0, strike, r, q, sigma, t, right);
                let err = (surface.price(i, j) - reference).abs();
                worst = worst.max(err);
                assert!(
                    err < 1e-5 * S0,
                    "{right:?} T = {t}, K = {strike}: {} vs BS {reference}",
                    surface.price(i, j)
                );
            }
        }
        log::debug!("{right:?} worst node error {worst}");
    }
}

#[test]
fn flat_vol_matches_black_scholes_at_listed_strikes_to_half_a_vol_bp() {
    let (r, q, sigma) = (0.04, 0.02, 0.2);
    let market = ForwardMarket::flat(S0, r, q);
    let k = test_grid();
    let maturities = [ONE_WEEK, ONE_MONTH, 0.25, 1.0];
    let surface = dupire_forward(
        &FlatVol(sigma),
        &market,
        &k,
        &maturities,
        &DupireConfig::default(),
    )
    .unwrap();
    let strikes = listed_strikes();
    let vols = implied_vols(&surface, &strikes);
    let mut checked = 0;
    for (i, &t) in maturities.iter().enumerate() {
        for (m, &strike) in strikes.iter().enumerate() {
            if !informative(strike, r, q, sigma, t) {
                continue;
            }
            let iv = vols[i][m]
                .as_ref()
                .unwrap_or_else(|e| panic!("T = {t}, K = {strike}: {e}"));
            assert!(
                (iv - sigma).abs() < 0.5e-4,
                "T = {t}, K = {strike}: implied vol {iv} vs {sigma} ({} bp)",
                (iv - sigma).abs() * 1e4
            );
            checked += 1;
        }
    }
    assert!(
        checked >= 30,
        "enough informative quotes were checked: {checked}"
    );
    // the one-week expiry must be part of the check
    assert!(
        strikes
            .iter()
            .filter(|&&s| informative(s, r, q, sigma, ONE_WEEK))
            .count()
            >= 5,
        "at least five informative one-week strikes"
    );
}

#[test]
fn time_dependent_vol_reproduces_black_scholes_at_the_rms_vol() {
    // sigma(t) = 0.35 -> 0.05 over one year (RMS 0.25 at T = 1) and the
    // reverse ramp: the European price depends on int sigma^2 only
    let (r, q) = (0.04, 0.01);
    let market = ForwardMarket::flat(S0, r, q);
    let k = test_grid();
    let maturities = [ONE_MONTH, 0.5, 1.0];
    let cfg = DupireConfig::default();
    let down = CallbackVol::new(|_x, t| 0.35 - 0.30 * t);
    let up = CallbackVol::new(|_x, t| 0.05 + 0.30 * t);
    let rms = |a: f64, b: f64, t: f64| -> f64 {
        // int_0^t (a + b s)^2 ds / t
        ((a * a * t + a * b * t * t + b * b * t * t * t / 3.0) / t).sqrt()
    };
    for (field, a, b) in [(&down, 0.35, -0.30), (&up, 0.05, 0.30)] {
        let surface = dupire_forward(field, &market, &k, &maturities, &cfg).unwrap();
        for (i, &t) in maturities.iter().enumerate() {
            let sigma_rms = rms(a, b, t);
            for (j, &kj) in k.iter().enumerate() {
                let strike = kj.exp();
                if !(70.0..=130.0).contains(&strike) {
                    continue;
                }
                let reference = bs_price(S0, strike, r, q, sigma_rms, t, PutOrCall::Call);
                assert!(
                    (surface.price(i, j) - reference).abs() < 1e-5 * S0,
                    "ramp ({a}, {b}) T = {t}, K = {strike}: {} vs BS({sigma_rms}) {reference}",
                    surface.price(i, j)
                );
            }
        }
    }
}

#[test]
fn implied_vols_with_a_sloped_rate_curve_return_the_flat_vol() {
    let sigma = 0.2;
    let rate =
        TermStructure::piecewise(vec![0.1, 0.25, 0.5], vec![0.03, 0.04, 0.045, 0.05]).unwrap();
    let carry = TermStructure::piecewise(vec![0.3], vec![0.01, 0.025]).unwrap();
    let market = ForwardMarket::new(S0, rate, carry, Vec::new()).unwrap();
    let k = test_grid();
    let maturities = [ONE_WEEK, ONE_MONTH, 0.25, 0.5, 1.0];
    let strikes = listed_strikes();

    // (1) exact prices under the term structure, hand-built surface: the
    // inversion with the effective flat rates must return sigma to 1e-6
    let mut prices = Vec::with_capacity(maturities.len() * k.len());
    let mut df = Vec::new();
    let mut forward = Vec::new();
    for &t in &maturities {
        df.push(market.df(t));
        forward.push(market.forward(t));
        for &kj in &k {
            prices.push(bs_price_ts(&market, kj.exp(), sigma, t, PutOrCall::Put));
        }
    }
    let exact = DenseSurface::new(
        PutOrCall::Put,
        S0,
        k.clone(),
        maturities.to_vec(),
        df,
        forward,
        prices,
    )
    .unwrap();
    for (i, &t) in maturities.iter().enumerate() {
        let (r, q) = market.effective_rates(t);
        for (j, &kj) in k.iter().enumerate() {
            if !informative(kj.exp(), r, q, sigma, t) {
                continue;
            }
            let iv = exact.implied_vol(i, j).unwrap();
            assert!(
                (iv - sigma).abs() < 1e-6,
                "exact surface T = {t}, K = {}: {iv} vs {sigma}",
                kj.exp()
            );
        }
        for &strike in &strikes {
            // the interpolated price of a smooth exact surface is exact to
            // O(dk^4); the vol must still come back to 1e-6
            if !informative(strike, r, q, sigma, t) {
                continue;
            }
            let iv = exact.implied_vol_at(i, strike).unwrap();
            assert!(
                (iv - sigma).abs() < 1e-6,
                "exact surface T = {t}, K = {strike} (interpolated): {iv} vs {sigma}"
            );
        }
    }

    // (2) the Dupire march under the sloped curve: the same inversion is
    // within the discretization budget (0.5 vol bp) at listed strikes
    let surface = dupire_forward(
        &FlatVol(sigma),
        &market,
        &k,
        &maturities,
        &DupireConfig::default(),
    )
    .unwrap();
    let vols = implied_vols(&surface, &strikes);
    for (i, &t) in maturities.iter().enumerate() {
        let (r, q) = market.effective_rates(t);
        assert!(
            (surface.df[i] - market.df(t)).abs() < 1e-15
                && (surface.forward[i] - market.forward(t)).abs() < 1e-12,
            "surface records the market df and forward"
        );
        for (m, &strike) in strikes.iter().enumerate() {
            if !informative(strike, r, q, sigma, t) {
                continue;
            }
            let iv = vols[i][m].clone().unwrap();
            assert!(
                (iv - sigma).abs() < 0.5e-4,
                "Dupire under sloped curve T = {t}, K = {strike}: {iv} vs {sigma}"
            );
        }
    }
}

/// Reference price of a European option with one cash dividend `delta`
/// at `t_ex` under Black-Scholes dynamics: integrate the post-dividend
/// Black-Scholes price over the lognormal density of the cum-dividend
/// spot at `t_ex` (composite Simpson over 8 standard deviations).
#[allow(clippy::too_many_arguments)]
fn dividend_reference(
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t_ex: f64,
    delta: f64,
    t: f64,
    right: PutOrCall,
) -> f64 {
    let m = 8000usize;
    let (lo, hi) = (-8.0, 8.0);
    let h = (hi - lo) / m as f64;
    let drift = (r - q - 0.5 * sigma * sigma) * t_ex;
    let vol = sigma * t_ex.sqrt();
    let integrand = |z: f64| {
        let s_pre = S0 * (drift + vol * z).exp();
        let s_post = s_pre - delta;
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        if s_post <= 1e-12 {
            match right {
                PutOrCall::Call => 0.0,
                PutOrCall::Put => pdf * k * (-r * (t - t_ex)).exp(),
            }
        } else {
            pdf * bs_price(s_post, k, r, q, sigma, t - t_ex, right)
        }
    };
    let mut sum = integrand(lo) + integrand(hi);
    for i in 1..m {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * integrand(lo + i as f64 * h);
    }
    (-r * t_ex).exp() * sum * h / 3.0
}

#[test]
fn dividend_surface_matches_the_quadrature_reference_and_put_call_parity() {
    let (r, q, sigma) = (0.04, 0.0, 0.25);
    let (t_ex, delta) = (0.2, 1.5);
    let market = ForwardMarket::new(
        S0,
        TermStructure::flat(r),
        TermStructure::flat(q),
        vec![(t_ex, delta)],
    )
    .unwrap();
    let k = test_grid();
    let maturities = [0.1, t_ex, 0.5, 1.0];
    let cfg = DupireConfig::default();
    let mut ws = DupireWorkspace::new(k.len());
    let call = dupire_forward_with(
        &FlatVol(sigma),
        &market,
        &k,
        &maturities,
        &cfg,
        PutOrCall::Call,
        &mut ws,
    )
    .unwrap();
    let put = dupire_forward_with(
        &FlatVol(sigma),
        &market,
        &k,
        &maturities,
        &cfg,
        PutOrCall::Put,
        &mut ws,
    )
    .unwrap();

    // the dividend-adjusted forward and parity at every maturity
    for (i, &t) in maturities.iter().enumerate() {
        let expected_f = if t > t_ex + NODE_TOL {
            S0 * ((r - q) * t).exp() - delta * ((r - q) * (t - t_ex)).exp()
        } else {
            S0 * ((r - q) * t).exp()
        };
        assert!(
            (call.forward[i] - expected_f).abs() < 1e-10,
            "forward at {t}: {} vs {expected_f}",
            call.forward[i]
        );
        let df = (-r * t).exp();
        for (j, &kj) in k.iter().enumerate() {
            let strike = kj.exp();
            if !(70.0..=130.0).contains(&strike) {
                continue;
            }
            let parity = call.price(i, j) - put.price(i, j) - df * (expected_f - strike);
            assert!(
                parity.abs() < 1e-5 * S0,
                "parity at T = {t}, K = {strike}: C - P - df (F - K) = {parity}"
            );
        }
    }

    // before the ex-date the surface is plain Black-Scholes; after it, the
    // quadrature reference
    for (i, &t) in maturities.iter().enumerate() {
        for &strike in &listed_strikes() {
            if !(85.0..=115.0).contains(&strike) {
                continue;
            }
            let reference = if t <= t_ex + NODE_TOL {
                bs_price(S0, strike, r, q, sigma, t, PutOrCall::Call)
            } else {
                dividend_reference(strike, r, q, sigma, t_ex, delta, t, PutOrCall::Call)
            };
            let model = call.price_at(i, strike);
            assert!(
                (model - reference).abs() < 1e-5 * S0,
                "call T = {t}, K = {strike}: {model} vs reference {reference}"
            );
        }
    }
    // a maturity on the ex-date reads the cum-dividend layer (BS exactly)
    let i_ex = 1;
    let bs_atm = bs_price(S0, S0, r, q, sigma, t_ex, PutOrCall::Call);
    assert!(
        (call.price_at(i_ex, S0) - bs_atm).abs() < 1e-5 * S0,
        "maturity on the ex-date settles cum-dividend: {} vs {bs_atm}",
        call.price_at(i_ex, S0)
    );
    // a put converted by parity on the dividend-adjusted forward gives the
    // same implied vol as the call
    let i = 3;
    for &strike in &[90.0, 100.0, 110.0] {
        let iv_c = call.implied_vol_at(i, strike).unwrap();
        let iv_p = put.implied_vol_at(i, strike).unwrap();
        assert!(
            (iv_c - iv_p).abs() < 1e-5,
            "K = {strike}: call vol {iv_c} vs put vol {iv_p}"
        );
    }
}

#[test]
fn dupire_rejects_bad_inputs_and_flags_non_finite_vol() {
    let market = ForwardMarket::flat(S0, 0.04, 0.0);
    let cfg = DupireConfig::default();
    let k = test_grid();
    assert!(
        dupire_forward(&FlatVol(0.2), &market, &k[..3], &[0.5], &cfg).is_err(),
        "too few strike nodes"
    );
    let mut bad = k.clone();
    bad[5] += 1e-6;
    assert!(
        dupire_forward(&FlatVol(0.2), &market, &bad, &[0.5], &cfg).is_err(),
        "non-uniform grid"
    );
    assert!(
        dupire_forward(&FlatVol(0.2), &market, &k, &[0.5, 0.25], &cfg).is_err(),
        "unsorted maturities"
    );
    assert!(
        dupire_forward(&FlatVol(0.2), &market, &k, &[], &cfg).is_err(),
        "no maturities"
    );
    assert!(
        dupire_forward(&FlatVol(f64::NAN), &market, &k, &[0.5], &cfg).is_err(),
        "NaN volatility"
    );
    let bad_cfg = DupireConfig {
        first_step: 0.0,
        ..cfg
    };
    assert!(
        dupire_forward(&FlatVol(0.2), &market, &k, &[0.5], &bad_cfg).is_err(),
        "invalid configuration"
    );
    assert!(ForwardMarket::new(
        -1.0,
        TermStructure::flat(0.0),
        TermStructure::flat(0.0),
        vec![]
    )
    .is_err());
    assert!(ForwardMarket::new(
        S0,
        TermStructure::flat(0.0),
        TermStructure::flat(0.0),
        vec![(-0.1, 1.0)]
    )
    .is_err());
}

#[test]
fn low_vol_high_drift_rows_are_upwinded_and_remain_monotone() {
    // sigma = 0.01 with |r - q| = 0.1 violates the central-difference
    // bound |b| dk / 2 <= sigma^2 / 2 at dk = 1/800 (6.3e-5 > 5e-5): the
    // rows fall back to upwinding, the count is reported and the call
    // price stays monotone in strike
    let market = ForwardMarket::flat(S0, 0.10, 0.0);
    let k = test_grid();
    let surface = dupire_forward(
        &FlatVol(0.01),
        &market,
        &k,
        &[0.5],
        &DupireConfig::default(),
    )
    .unwrap();
    assert!(surface.upwinded_rows > 0, "upwinding was needed");
    let row = surface.row(0);
    assert!(
        row.windows(2).all(|w| w[1] <= w[0] + 1e-12),
        "call prices nonincreasing in strike"
    );
    assert!(
        row.iter().all(|c| c.is_finite() && *c >= -1e-12),
        "finite, nonnegative prices"
    );
}

// ── Implied-vol inversion ────────────────────────────────────────────────

#[test]
fn implied_vol_wrapper_maps_floor_errors_and_parity_correctly() {
    let (f, t, sigma) = (100.0, 0.5, 0.3);
    let c = bs_price(f, 110.0, 0.0, 0.0, sigma, t, PutOrCall::Call);
    let p = bs_price(f, 110.0, 0.0, 0.0, sigma, t, PutOrCall::Put);
    let iv_c = otm_implied_vol(f, 110.0, t, c, PutOrCall::Call).unwrap();
    let iv_p = otm_implied_vol(f, 110.0, t, p, PutOrCall::Put).unwrap();
    assert!((iv_c - sigma).abs() < 1e-9, "OTM call {iv_c}");
    assert!(
        (iv_p - sigma).abs() < 1e-9,
        "ITM put converted to the OTM call by parity {iv_p}"
    );
    let c_itm = bs_price(f, 80.0, 0.0, 0.0, sigma, t, PutOrCall::Call);
    let iv = otm_implied_vol(f, 80.0, t, c_itm, PutOrCall::Call).unwrap();
    assert!((iv - sigma).abs() < 1e-9, "ITM call via the OTM put {iv}");
    assert_eq!(
        black76_implied_vol(f, 100.0, t, 0.0, PutOrCall::Call),
        Err(IvError::NonPositivePrice)
    );
    assert_eq!(
        black76_implied_vol(f, 100.0, t, -1.0, PutOrCall::Call),
        Err(IvError::NonPositivePrice)
    );
    // an ITM call at its intrinsic (undiscounted F - K) converts to a
    // zero-valued put -> NonPositivePrice; an at-the-money price below the
    // sigma = 1e-4 price (about 2.8e-3) hits the crate's silent floor
    assert_eq!(
        otm_implied_vol(f, 80.0, t, 20.0, PutOrCall::Call),
        Err(IvError::NonPositivePrice)
    );
    assert_eq!(
        black76_implied_vol(f, 100.0, t, 1e-3, PutOrCall::Call),
        Err(IvError::AtFloor)
    );
    assert!(
        matches!(
            black76_implied_vol(f, 100.0, t, 150.0, PutOrCall::Call),
            Err(IvError::Rejected(_))
        ),
        "price above the forward is rejected"
    );
    assert!(
        matches!(
            black76_implied_vol(f, 100.0, 0.0, 5.0, PutOrCall::Call),
            Err(IvError::Rejected(_))
        ),
        "expired option is rejected"
    );
    assert_eq!(
        format!("{}", IvError::AtFloor),
        "implied vol at the 1e-4 floor"
    );
}

// ── Arbitrage scans ──────────────────────────────────────────────────────

#[test]
fn black_scholes_surface_is_clean_at_the_calibrated_tolerance() {
    let (r, q) = (0.04, 0.02);
    let market = ForwardMarket::flat(S0, r, q);
    let k = log_strike_grid(S0, 0.8, 200);
    let maturities = [ONE_WEEK, ONE_MONTH, 0.25, 0.5, 1.0];
    let cfg = DupireConfig::default();
    let control = flat_vol_control(0.2, &market, &k, &maturities, &cfg, 2.0).unwrap();
    assert!(
        control.worst_butterfly > -1e-6 * S0,
        "flat control butterfly noise is at the discretization floor: {}",
        control.worst_butterfly
    );
    assert!(
        control.worst_calendar < 1e-6,
        "flat control calendar noise is at the discretization floor: {}",
        control.worst_calendar
    );
    assert!(control.tol_butterfly >= 1e-12 * S0 && control.tol_calendar >= 1e-12);
    assert!(control.report.n_checked > 0);

    // a different flat vol through the same pipeline is clean at those
    // tolerances
    let surface = dupire_forward(&FlatVol(0.3), &market, &k, &maturities, &cfg).unwrap();
    let report = arbitrage_scan(&surface, control.tol_butterfly, control.tol_calendar);
    assert_eq!(report.butterfly_count, 0, "{report:?}");
    assert_eq!(report.calendar_count, 0, "{report:?}");
    assert_eq!(
        report.n_checked,
        report.n_checked_butterfly + report.n_checked_calendar
    );
    assert!(report.n_checked_calendar > 0, "calendar pairs were checked");

    // the exact Black-Scholes surface is convex and calendar-monotone at
    // (almost) zero tolerance
    let mut prices = Vec::new();
    let (mut df, mut fwd) = (Vec::new(), Vec::new());
    for &t in &maturities {
        df.push(market.df(t));
        fwd.push(market.forward(t));
        for &kj in &k {
            prices.push(bs_price(S0, kj.exp(), r, q, 0.2, t, PutOrCall::Call));
        }
    }
    let exact = DenseSurface::new(
        PutOrCall::Call,
        S0,
        k.clone(),
        maturities.to_vec(),
        df,
        fwd,
        prices,
    )
    .unwrap();
    let exact_report = arbitrage_scan(&exact, 1e-12 * S0, 1e-9);
    assert_eq!(exact_report.butterfly_count, 0, "{exact_report:?}");
    assert_eq!(exact_report.calendar_count, 0, "{exact_report:?}");
    assert!(
        exact_report.worst_butterfly >= -1e-12,
        "every triple convex to round-off: {}",
        exact_report.worst_butterfly
    );
    assert!(
        exact_report.worst_calendar < 0.0,
        "total variance strictly increasing: {}",
        exact_report.worst_calendar
    );
}

#[test]
fn hand_built_violating_surface_is_caught_by_the_scan() {
    let (r, q) = (0.04, 0.02);
    let market = ForwardMarket::flat(S0, r, q);
    let k = log_strike_grid(S0, 0.8, 200);
    let n_k = k.len();
    let maturities = [0.25, 0.3];
    let (mut df, mut fwd, mut prices) = (Vec::new(), Vec::new(), Vec::new());
    // the second maturity carries a much lower vol: total variance falls
    // from 0.04 * 0.25 = 0.01 to 0.15^2 * 0.3 = 0.00675
    for (i, &t) in maturities.iter().enumerate() {
        df.push(market.df(t));
        fwd.push(market.forward(t));
        let sigma = if i == 0 { 0.2 } else { 0.15 };
        for &kj in &k {
            prices.push(bs_price(S0, kj.exp(), r, q, sigma, t, PutOrCall::Call));
        }
    }
    // one butterfly violation: lift the ATM node of the first maturity
    let j0 = n_k / 2;
    let bump = 0.05;
    prices[j0] += bump;
    let surface = DenseSurface::new(
        PutOrCall::Call,
        S0,
        k.clone(),
        maturities.to_vec(),
        df.clone(),
        fwd,
        prices,
    )
    .unwrap();
    let report = arbitrage_scan(&surface, 1e-9 * S0, 1e-9);
    assert_eq!(
        report.butterfly_count, 1,
        "exactly the lifted node fails convexity: {report:?}"
    );
    // the lifted triple's value is the bump (undiscounted) net of the
    // smile's own small convexity at that node
    assert!(
        report.worst_butterfly > -bump / df[0] && report.worst_butterfly < -0.9 * bump / df[0],
        "worst butterfly is the bump in undiscounted units: {} vs {}",
        report.worst_butterfly,
        -bump / df[0]
    );
    // every checkable calendar pair fails (the wings below the price floor
    // are skipped: at sigma = 0.15, T = 0.3 the +-0.8 log-strike range is
    // about ten standard deviations wide)
    assert_eq!(
        report.calendar_count, report.n_checked_calendar,
        "every checked calendar pair is a violation: {report:?}"
    );
    assert!(
        report.n_checked_calendar >= n_k / 4 && report.n_skipped > 0,
        "the body of the smile was checked and the wings skipped: {report:?}"
    );
    // the worst decrease is at the bumped ATM node: the bump raises the
    // first maturity's total variance above 0.01, the smile elsewhere gives
    // exactly 0.01 - 0.00675 = 0.00325
    let w1_bumped = {
        let iv = surface.implied_vol(0, j0).unwrap();
        iv * iv * maturities[0]
    };
    assert!(
        w1_bumped > 0.01 + 1e-4,
        "the bump lifted the ATM total variance: {w1_bumped}"
    );
    assert!(
        report.worst_calendar > 0.00325 + 1e-4
            && report.worst_calendar < w1_bumped - 0.00675 + 1e-9,
        "worst calendar decrease {} lies between the smile's 0.00325 and the bumped node's {}",
        report.worst_calendar,
        w1_bumped - 0.00675
    );
    // the same surface passes at a tolerance above the violations
    let lenient = arbitrage_scan(&surface, bump, 0.01);
    assert_eq!(lenient.butterfly_count, 0);
    assert_eq!(lenient.calendar_count, 0);
}

#[test]
fn beyond_spread_scan_ignores_violations_a_within_band_price_removes() {
    let (r, q, sigma) = (0.04, 0.02, 0.2);
    let market = ForwardMarket::flat(S0, r, q);
    let t1 = 0.25;
    let df1 = market.df(t1);
    let f1 = market.forward(t1);
    let strikes = [90.0, 95.0, 100.0, 105.0, 110.0];
    let mid = |k: f64, t: f64, s: f64| bs_price(S0, k, r, q, s, t, PutOrCall::Call);
    // lift the 100 strike by 0.02 (discounted), a butterfly violation of
    // the mids; a half-spread of 0.05 lets the bands absorb it, 0.005 not
    let build = |half: f64, lift: f64| -> ExpiryQuotes {
        ExpiryQuotes {
            t: t1,
            df: df1,
            forward: f1,
            calls: strikes
                .iter()
                .map(|&k| {
                    let m = mid(k, t1, sigma) + if k == 100.0 { lift } else { 0.0 };
                    QuoteBand::call(k, m - half, m + half)
                })
                .collect(),
        }
    };
    let convexity_gap =
        0.5 * (mid(95.0, t1, sigma) + mid(105.0, t1, sigma)) - mid(100.0, t1, sigma);
    let lift = convexity_gap + 0.02;
    let wide = arbitrage_scan_quotes(&[build(0.05, lift)], 1e-9, 1e-9);
    assert_eq!(wide.n_checked_butterfly, 3);
    assert_eq!(wide.butterfly_raw, 1, "{wide:?}");
    assert_eq!(
        wide.butterfly_beyond_spread, 0,
        "bands of +-0.05 absorb a 0.02 violation: {wide:?}"
    );
    assert!(
        (wide.worst_butterfly + 0.02 / df1).abs() < 1e-9,
        "worst mid butterfly {} vs {}",
        wide.worst_butterfly,
        -0.02 / df1
    );
    let narrow = arbitrage_scan_quotes(&[build(0.005, lift)], 1e-9, 1e-9);
    assert_eq!(narrow.butterfly_raw, 1, "{narrow:?}");
    assert_eq!(
        narrow.butterfly_beyond_spread, 1,
        "bands of +-0.005 cannot absorb a 0.02 violation: {narrow:?}"
    );
    let clean = arbitrage_scan_quotes(&[build(0.05, 0.0)], 1e-9, 1e-9);
    assert_eq!(clean.butterfly_raw, 0, "{clean:?}");
    assert!(clean.worst_butterfly > 0.0);

    // calendar: a later expiry priced at a lower vol; wide bands on the
    // later expiry (ask vol above the earlier bid vol) remove it
    let t2 = 0.3;
    let df2 = market.df(t2);
    let f2 = market.forward(t2);
    let later = |half: f64| -> ExpiryQuotes {
        ExpiryQuotes {
            t: t2,
            df: df2,
            forward: f2,
            calls: strikes
                .iter()
                .map(|&k| {
                    let m = mid(k, t2, 0.17);
                    QuoteBand::call(k, m - half, m + half)
                })
                .collect(),
        }
    };
    let earlier = build(0.0005, 0.0);
    // the later expiry's mids: total variance 0.17^2 * 0.3 = 0.00867 < 0.01
    let raw = arbitrage_scan_quotes(&[later(0.0005), earlier.clone()], 1e-9, 1e-9);
    assert!(raw.n_checked_calendar >= 3, "{raw:?}");
    assert_eq!(
        raw.calendar_raw, raw.n_checked_calendar,
        "every strike fails: {raw:?}"
    );
    assert_eq!(
        raw.calendar_beyond_spread, raw.calendar_raw,
        "narrow bands keep every violation: {raw:?}"
    );
    assert!(
        raw.worst_calendar > 0.001 && raw.worst_calendar < 0.002,
        "worst decrease about 0.00133: {}",
        raw.worst_calendar
    );
    // the vega of the later expiry near the money is about 0.2 * S0 *
    // sqrt(0.3) * 0.4 = 4.4 per unit vol; a half-spread of 0.4 moves the
    // ask vol by ~9 vol points, far above the earlier expiry's 20%
    let wide_later = arbitrage_scan_quotes(&[earlier.clone(), later(0.4)], 1e-9, 1e-9);
    assert_eq!(
        wide_later.calendar_raw, raw.calendar_raw,
        "raw counts use the mids"
    );
    assert_eq!(
        wide_later.calendar_beyond_spread, 0,
        "wide bands on the later expiry absorb the calendar violation: {wide_later:?}"
    );
    // put bands are converted by parity on the forward
    let from_put = QuoteBand::from_put(100.0, 1.0, 1.2, df1, f1);
    assert!((from_put.bid - (1.0 + df1 * (f1 - 100.0))).abs() < 1e-14);
    assert!((from_put.ask - (1.2 + df1 * (f1 - 100.0))).abs() < 1e-14);
}

#[test]
fn dense_surface_accessors_and_validation() {
    let k = log_strike_grid(S0, 0.5, 10);
    assert_eq!(k.len(), 21);
    assert!((k[10] - S0.ln()).abs() < 1e-15, "ln S0 is the middle node");
    let n = k.len();
    let prices = vec![1.0; 2 * n];
    let s = DenseSurface::new(
        PutOrCall::Call,
        S0,
        k.clone(),
        vec![0.5, 1.0],
        vec![0.98, 0.96],
        vec![101.0, 102.0],
        prices,
    )
    .unwrap();
    assert_eq!(s.n_k(), n);
    assert_eq!(s.n_t(), 2);
    assert_eq!(s.row(1).len(), n);
    assert!((s.undiscounted_call(1, 10) - 1.0 / 0.96).abs() < 1e-15);
    let p = DenseSurface::new(
        PutOrCall::Put,
        S0,
        k.clone(),
        vec![0.5],
        vec![0.98],
        vec![101.0],
        vec![1.0; n],
    )
    .unwrap();
    assert!(
        (p.undiscounted_call(0, 10) - (1.0 / 0.98 + 101.0 - S0)).abs() < 1e-12,
        "put converted by parity"
    );
    assert!(
        DenseSurface::new(
            PutOrCall::Call,
            S0,
            k.clone(),
            vec![1.0, 0.5],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0; 2 * n]
        )
        .is_err(),
        "unsorted maturities"
    );
    assert!(
        DenseSurface::new(
            PutOrCall::Call,
            S0,
            k.clone(),
            vec![0.5],
            vec![1.0],
            vec![1.0],
            vec![1.0; n + 1]
        )
        .is_err(),
        "price count mismatch"
    );
    assert!(
        DenseSurface::new(
            PutOrCall::Call,
            S0,
            k,
            vec![0.5],
            vec![1.0, 1.0],
            vec![1.0],
            vec![1.0; n]
        )
        .is_err(),
        "df count mismatch"
    );
}

#[test]
fn backward_european_prices_at_quotes_match_black_scholes_and_the_single_solver() {
    use crate::equity::american_lv::{
        price_only, FlatVol, MarketSlice, Mesh, Mode, QuoteSpec, Workspace,
    };
    use crate::equity::models::american_lv::grid::{half_width, n_t_default};
    let (s0, r, q, sigma, t) = (100.0, 0.04, 0.02, 0.25, 0.25);
    let l = half_width(sigma, t, 0.05, s0, 70.0, 130.0);
    let mesh = Mesh::new(s0, l, 400, t, n_t_default(t), &[]).unwrap();
    let market = MarketSlice::flat(&mesh, s0, r, q, &[]);
    let specs: Vec<QuoteSpec> = [80.0, 100.0, 120.0]
        .iter()
        .flat_map(|&k| {
            [PutOrCall::Put, PutOrCall::Call]
                .into_iter()
                .map(move |right| QuoteSpec::new(k, t, right))
        })
        .collect();
    let prices = european_prices_backward(&mesh, &specs, &FlatVol(sigma), &market).unwrap();
    assert_eq!(prices.len(), specs.len());
    let field = mesh.node_field(&FlatVol(sigma));
    let mut ws = Workspace::new();
    for (spec, &p) in specs.iter().zip(&prices) {
        // the working-mesh tolerance of the solver's own Black-Scholes test
        let bs = bs_price(s0, spec.strike, r, q, sigma, t, spec.right);
        let tol = (2.5e-4 * bs).max(8e-6 * s0);
        assert!(
            (p - bs).abs() < tol,
            "K = {} {:?}: backward {p} vs BS {bs}",
            spec.strike,
            spec.right
        );
        let single = price_only(&mesh, spec, &field, &market, Mode::European, &mut ws).unwrap();
        assert_eq!(p, single, "identical to the single-quote European solve");
    }
    // a quote at another expiry is rejected by the solver's validation
    let bad = [QuoteSpec::new(100.0, 0.5, PutOrCall::Put)];
    assert!(european_prices_backward(&mesh, &bad, &FlatVol(sigma), &market).is_err());
}
