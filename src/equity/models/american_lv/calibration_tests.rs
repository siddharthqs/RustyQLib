//! Tests of the Tikhonov calibration: quote conventions, capture meshes,
//! the warm-start identity (the first Gauss–Newton step at `alpha_lin`
//! from `theta_0` equals `theta_lin`), a small end-to-end run with
//! one-sided intrinsic quotes, carry recovery, the SPEC 1.8 synthetic
//! recovery of case (c), and the wall-clock of one 500-quote Jacobian.
//!
//! Profile note: the recovery test uses `n_x = 200` for the working mesh
//! and `n_x = 800` for the truth in the debug (test) profile to stay
//! under ~70 s, with the sigma^E and 5%-support reconstruction tolerances
//! relaxed (20 bp, 1.0 vol pt) to match the coarser mesh; in `--release`
//! it uses the study's `n_x = 400` and the fine reference `n_x = 1600`
//! with the SPEC 1.8 tolerances (5 bp, 0.5 vol pt) — the release run is
//! the correctness gate. Wall-clocks are printed (visible with
//! `--nocapture`).

use super::*;
use crate::equity::models::american_lv::grid::half_width;
use crate::equity::models::american_lv::vol_field::{CallbackVol, FlatVol};

const S0: f64 = 100.0;

/// A flat-rate capture with `q` on every interval and `n_x` intervals,
/// `n_t_of` steps per expiry.
fn capture_with(
    expiries: &[f64],
    r: f64,
    q: f64,
    n_x: usize,
    n_t_of: &dyn Fn(f64) -> usize,
    dividends: &[(f64, f64)],
) -> CaptureMeshes {
    let t_max = *expiries.last().unwrap();
    let l = half_width(0.35, t_max, 0.1, S0, 0.6 * S0, 1.5 * S0);
    let qs = vec![q; expiries.len()];
    CaptureMeshes::new(
        S0,
        l,
        n_x,
        expiries,
        dividends,
        RateInput::Flat(r),
        &qs,
        n_t_of,
    )
    .unwrap()
}

/// Half-spread templates of the synthetic study.
#[derive(Clone, Copy)]
enum Spread {
    /// `0.01 + 0.004 mid` for every quote (the SPEC 1.8 recovery test).
    Uniform,
    /// "spy": `0.01 + 0.004 mid` OTM, `0.03 mid` ITM.
    Spy,
}

impl Spread {
    fn half(self, mid: f64, intrinsic: f64) -> f64 {
        match self {
            Spread::Uniform => 0.01 + 0.004 * mid,
            Spread::Spy if intrinsic > 0.0 => 0.03 * mid,
            Spread::Spy => 0.01 + 0.004 * mid,
        }
    }
}

/// Two-sided quotes of both rights at every strike and expiry from the
/// American prices of `truth` on `pricing` (the mesh whose prices are the
/// data), half-spreads from `spread`, classified against intrinsic with
/// `tick`.
fn quotes_from_truth(
    pricing: &CaptureMeshes,
    strikes: &[f64],
    truth: &dyn VolField,
    tick: f64,
    spread: Spread,
) -> Vec<Quote> {
    let mut quotes = Vec::new();
    for (e, &t) in pricing.expiries.iter().enumerate() {
        for &k in strikes {
            for right in [PutOrCall::Put, PutOrCall::Call] {
                quotes.push(Quote::from_mid(k, t, e, right, 1.0, 0.01));
            }
        }
    }
    let prices = american_prices(&quotes, pricing, truth, RHO_DEFAULT).unwrap();
    quotes
        .into_iter()
        .zip(prices)
        .map(|(q, mid)| {
            let intrinsic = q.intrinsic(S0);
            let s = spread.half(mid, intrinsic);
            Quote::from_mid(q.strike, q.t, q.expiry_index, q.right, mid, s.max(0.005))
                .classify_intrinsic(intrinsic, tick)
        })
        // the data layer keeps two-sided quotes only (bid > 0)
        .filter(|q| q.bid > 0.0)
        .collect()
}

/// Case (c) of the synthetic study in log-spot coordinates:
/// `sigma(y, t) = (0.20 + 0.08 e^{-t/0.3}) - 0.25 y + 0.60 y^2` clipped to
/// `[0.08, 0.80]`, `y = ln(S / F_true(t))`.
fn case_c(r: f64, q: f64) -> CallbackVol {
    CallbackVol::new(move |x: f64, t: f64| {
        let ln_f = S0.ln() + (r - q) * t;
        let y = x - ln_f;
        ((0.20 + 0.08 * (-t / 0.3).exp()) - 0.25 * y + 0.60 * y * y).clamp(0.08, 0.80)
    })
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[test]
fn capture_meshes_build_intervals_and_rebuild_carry_per_step() {
    let expiries = [0.25, 0.5, 1.0];
    let mut cap = capture_with(&expiries, 0.04, 0.0, 20, &|_| 8, &[(0.4, 1.0)]);
    assert_eq!(cap.n_expiries(), 3);
    assert_eq!(cap.intervals, vec![(0.0, 0.25), (0.25, 0.5), (0.5, 1.0)]);
    assert_eq!(cap.interval_of(0.1), 0);
    assert_eq!(
        cap.interval_of(0.25),
        1,
        "half-open: T_1 belongs to the second interval"
    );
    assert_eq!(cap.interval_of(0.7), 2);
    assert_eq!(
        cap.interval_of(5.0),
        2,
        "beyond the last expiry -> last interval"
    );
    cap.set_q(&[0.01, 0.02, 0.03]).unwrap();
    for e in 0..3 {
        let mesh = &cap.meshes[e];
        let mk = &cap.markets[e];
        assert_eq!(mk.carry.len(), mesh.n_steps());
        for (n, &tm) in mesh.t_mid.iter().enumerate() {
            let expect = [0.01, 0.02, 0.03][cap.interval_of(tm)];
            assert_eq!(mk.carry[n], expect, "carry of expiry {e} step {n}");
            assert_eq!(mk.rate[n], 0.04);
        }
        let si = cap.step_intervals(e);
        assert!(
            si.iter().all(|&k| k <= e),
            "steps of expiry {e} map to intervals <= {e}"
        );
        assert!(mk.validate(mesh).is_ok());
    }
    // the 1y mesh spans the dividend, the 3m mesh does not
    assert_eq!(cap.meshes[2].div_steps.len(), 1);
    assert_eq!(cap.meshes[0].div_steps.len(), 0);
    let (r_eff, q_eff, f, df) = cap.effective_rates(0);
    assert!(
        (r_eff - 0.04).abs() < 1e-12 && (q_eff - 0.01).abs() < 1e-10,
        "{r_eff} {q_eff}"
    );
    assert!(
        (f - S0 * (0.03f64 * 0.25).exp()).abs() < 1e-9 && (df - (-0.01f64).exp()).abs() < 1e-12
    );
    let table = cap.f_ref_table();
    assert_eq!(table.len(), 4);
    assert_eq!(table[0], (0.0, S0.ln()));
    assert!(
        table.windows(2).all(|w| w[1].0 > w[0].0),
        "increasing times"
    );
    assert!(
        cap.markets_for(&[0.0]).is_err(),
        "wrong carry length is rejected"
    );
    assert!(
        CaptureMeshes::new(
            S0,
            1.0,
            20,
            &[0.5, 0.25],
            &[],
            RateInput::Flat(0.0),
            &[0.0, 0.0],
            &|_| 4
        )
        .is_err(),
        "non-increasing expiries are rejected"
    );
}

#[test]
fn quote_residual_conventions_two_sided_one_sided_and_held_out() {
    let q = Quote::two_sided(100.0, 0.5, 0, PutOrCall::Put, 4.0, 4.2);
    assert!((q.mid - 4.1).abs() < 1e-15 && (q.half_spread - 0.1).abs() < 1e-15);
    assert!((q.residual(4.3) - 2.0).abs() < 1e-12, "two-sided (F - m)/s");
    assert!((q.residual(3.9) + 2.0).abs() < 1e-12, "two-sided is signed");
    assert!(q.active_at(0.0) && q.two_sided_in_fit());
    // one-sided: bid at intrinsic
    let deep =
        Quote::two_sided(140.0, 0.5, 0, PutOrCall::Put, 40.0, 40.4).classify_intrinsic(40.0, 0.01);
    assert!(deep.at_intrinsic, "bid <= intrinsic + tick");
    assert!(
        (deep.one_sided_target - 40.4).abs() < 1e-12,
        "target max(mid + s, intrinsic + tick)"
    );
    assert_eq!(deep.residual(40.3), 0.0, "below the target: no penalty");
    assert!(
        (deep.residual(40.6) - 1.0).abs() < 1e-12,
        "above: (F - target)/s"
    );
    assert!(!deep.active_at(40.3) && deep.active_at(40.6));
    assert!(!deep.two_sided_in_fit() && deep.in_fit());
    let tiny =
        Quote::two_sided(140.0, 0.5, 0, PutOrCall::Put, 39.95, 40.0).classify_intrinsic(40.0, 0.01);
    assert!(
        (tiny.one_sided_target - 40.01).abs() < 1e-12,
        "intrinsic + tick dominates mid + s = 40.0"
    );
    let held = q.clone().held_out();
    assert!(!held.in_fit() && !held.two_sided_in_fit());
    assert!(
        (held.residual(4.3) - 2.0).abs() < 1e-12,
        "held-out residuals use the two-sided formula"
    );
    let weighted = q.with_weight(0.5);
    assert!(
        (weighted.residual(4.6) - 1.0).abs() < 1e-12,
        "weight override replaces s"
    );
    let cap = capture_with(&[0.5], 0.0, 0.0, 10, &|_| 4, &[]);
    assert!(Quote::two_sided(100.0, 0.4, 0, PutOrCall::Put, 4.0, 4.2)
        .validate(&cap)
        .is_err());
    assert!(Quote::two_sided(100.0, 0.5, 1, PutOrCall::Put, 4.0, 4.2)
        .validate(&cap)
        .is_err());
    assert!(
        Quote::two_sided(100.0, 0.5, 0, PutOrCall::Put, 4.2, 4.2)
            .validate(&cap)
            .is_err(),
        "zero spread"
    );
    let (target, n_eff) = discrepancy_target(
        &[deep.clone(), tiny.clone(), held.clone(), weighted.clone()],
        &CalibrationConfig::american(),
    );
    assert_eq!(n_eff, 1, "only the weighted two-sided in-fit quote counts");
    assert!((target - 1.0).abs() < 1e-15, "tau N_eff");
    let mut cfg = CalibrationConfig::synthetic();
    cfg.noise_floor = Some(0.01);
    let (target, _) = discrepancy_target(&[weighted.clone()], &cfg);
    assert!(
        (target - (0.01f64 / 0.5).powi(2)).abs() < 1e-15,
        "noise-floor target sum (delta/s)^2"
    );
}

#[test]
fn identification_quotes_convert_to_european_prices_with_propagated_weights() {
    let cap = capture_with(&[0.5], 0.04, 0.02, 10, &|_| 4, &[]);
    let quotes = vec![
        Quote::two_sided(100.0, 0.5, 0, PutOrCall::Put, 5.0, 5.2),
        Quote::two_sided(140.0, 0.5, 0, PutOrCall::Put, 40.0, 40.4).classify_intrinsic(40.0, 0.01),
        Quote::two_sided(90.0, 0.5, 0, PutOrCall::Call, 12.0, 12.4).held_out(),
    ];
    let sigma_a = [0.25, f64::NAN, 0.3];
    let ratio = [0.8, f64::NAN, 1.0];
    let id = identification_quotes(&quotes, &cap, &sigma_a, &ratio, true);
    let (r_eff, q_eff, _, _) = cap.effective_rates(0);
    let bs = bs_price(S0, 100.0, r_eff, q_eff, 0.25, 0.5, PutOrCall::Put);
    assert!((id[0].mid - bs).abs() < 1e-12, "P_hat^E = BS(sigma^A)");
    assert!(
        (id[0].half_spread - 0.1 / 0.8).abs() < 1e-12,
        "propagated s nu^E/nu^A"
    );
    assert!(!id[0].at_intrinsic && !id[0].held_out);
    assert!(
        id[1].held_out && (id[1].mid - 40.2).abs() < 1e-12,
        "no sigma^A -> held out, mid kept"
    );
    assert!(id[2].held_out, "input held-out flag preserved");
    let plain = identification_quotes(&quotes, &cap, &sigma_a, &ratio, false);
    assert!(
        (plain[0].half_spread - 0.1).abs() < 1e-12,
        "plain weights keep s"
    );
}

/// The small skewed problem shared by the identity and end-to-end tests.
struct Small {
    cap: CaptureMeshes,
    quotes: Vec<Quote>,
    surface: BSplineLocalVol,
}

fn small_problem(with_intrinsic_and_held_out: bool) -> Small {
    let expiries = [1.0 / 12.0, 0.25, 0.5];
    let cap = capture_with(
        &expiries,
        0.04,
        0.01,
        100,
        &|t| ((t * 120.0).ceil() as usize).max(12),
        &[],
    );
    let truth = CallbackVol::new(|x: f64, t: f64| {
        let y = x - (S0.ln() + 0.03 * t);
        (0.22 - 0.30 * y + 0.02 * (-t / 0.2).exp()).clamp(0.08, 0.8)
    });
    let mut strikes: Vec<f64> = (0..7).map(|i| 85.0 + 5.0 * i as f64).collect();
    if with_intrinsic_and_held_out {
        strikes.push(130.0);
        strikes.push(140.0);
    }
    let mut quotes = quotes_from_truth(&cap, &strikes, &truth, 0.01, Spread::Spy);
    if with_intrinsic_and_held_out {
        // hold out one quote per expiry
        for e in 0..expiries.len() {
            let i = quotes
                .iter()
                .position(|q| q.expiry_index == e && q.strike == 95.0 && q.right == PutOrCall::Call)
                .unwrap();
            quotes[i].held_out = true;
        }
        assert!(
            quotes.iter().any(|q| q.at_intrinsic),
            "the deep ITM puts are at intrinsic"
        );
    }
    let surface = BSplineLocalVol::with_defaults(0.22, cap.f_ref_table()).unwrap();
    Small {
        cap,
        quotes,
        surface,
    }
}

#[test]
fn first_gauss_newton_step_at_alpha_lin_equals_theta_lin_to_1e_8() {
    let Small {
        cap,
        quotes,
        surface,
    } = small_problem(false);
    let mut cfg = CalibrationConfig::american();
    cfg.noise_floor = Some(1e-4 * S0);
    let mode = Mode::American { rho: cfg.rho };
    let lin = linearized_step(&quotes, &cap, &surface, &cfg, mode).unwrap();
    assert_eq!(lin.jacobians, 2);
    assert_eq!(lin.n_theta, surface.len());
    assert_eq!(
        lin.n_cols,
        surface.len() + cap.n_expiries(),
        "carry free: E extra columns"
    );
    assert!(lin.path.len() >= 2, "linear path {:?}", lin.path);
    assert!(
        lin.path.windows(2).all(|w| w[0].alpha > w[1].alpha),
        "path sorted by decreasing alpha"
    );
    let step = max_abs_diff(&lin.theta_lin, &lin.theta0);
    assert!(step > 1e-3, "the skew moves theta: max step {step}");
    let (theta_gn, q_gn) =
        gauss_newton_step(&quotes, &cap, &surface, &cfg, mode, lin.alpha_lin).unwrap();
    let scale = max_abs(&lin.theta_lin);
    let diff = max_abs_diff(&theta_gn, &lin.theta_lin);
    assert!(
        diff <= 1e-8 * scale,
        "theta: GN step vs theta_lin differ by {diff} (scale {scale})"
    );
    let dq = max_abs_diff(&q_gn, &lin.q_lin);
    assert!(dq <= 1e-8 * max_abs(&lin.q_lin).max(1e-3), "carry: {dq}");
    // the same identity through the LM driver: one level, one undamped
    // iteration, no polish
    let mut cfg_lm = cfg.clone();
    cfg_lm.lm_lambda0 = 0.0;
    cfg_lm.max_alpha_levels = 1;
    cfg_lm.lm_iters_intermediate = 1;
    cfg_lm.lm_iters_final = 0;
    let res =
        calibrate_american(&quotes, &cap, &surface, &cfg_lm, Some(lin.alpha_lin), None).unwrap();
    assert_eq!(res.path.len(), 1);
    assert_eq!(
        res.path[0].iterations, 1,
        "the undamped step must be accepted: {:?}",
        res.flags
    );
    let diff = max_abs_diff(&res.theta, &lin.theta_lin);
    assert!(diff <= 1e-8 * scale, "LM first step vs theta_lin: {diff}");
    // the linearized rows give a finite first-order prediction
    let pred = lin.prediction(&lin.theta_lin);
    assert!(pred.iter().all(|p| p.is_finite()), "pred finite: {pred:?}");
    let fo = lin.first_order_sigma_e(0.22, &lin.theta_lin);
    assert!(
        fo.iter().all(|v| v.is_finite() && *v > 0.05 && *v < 0.6),
        "first-order sigma^E: {fo:?}"
    );
    // the European rows are the kernels: prices match the European solver
    let pe = european_prices(&quotes, &cap, &surface).unwrap();
    assert!(
        max_abs_diff(&pe, &lin.prices_e) < 1e-12,
        "European prices from the pass equal the backward solver"
    );
}

#[test]
fn tiny_end_to_end_calibration_with_one_sided_intrinsic_and_held_out_quotes() {
    let Small {
        cap,
        quotes,
        surface,
    } = small_problem(true);
    let mut cfg = CalibrationConfig::american();
    cfg.noise_floor = Some(1e-4 * S0);
    cfg.carry_frozen = true;
    cfg.jacobian_budget = 30;
    let grid = SupportGrid {
        y: (0..26).map(|i| -0.6 + 0.04 * i as f64).collect(),
        t: vec![1.0 / 12.0, 1.0 / 6.0, 0.25, 0.5],
    };
    let lin = linearized_step(
        &quotes,
        &cap,
        &surface,
        &cfg,
        Mode::American { rho: cfg.rho },
    )
    .unwrap();
    let warm = surface.clone().with_theta(lin.theta_lin.clone()).unwrap();
    let res = calibrate_american(
        &quotes,
        &cap,
        &warm,
        &cfg,
        Some(16.0 * lin.alpha_lin),
        Some(&grid),
    )
    .unwrap();
    assert!(
        res.jacobians <= cfg.jacobian_budget,
        "budget respected: {}",
        res.jacobians
    );
    assert!(!res.path.is_empty() && res.prices.iter().all(|p| p.is_finite()));
    assert_eq!(res.q, cap.q, "carry frozen keeps the prior");
    assert!(res.q_se.is_empty() && res.n_cols == surface.len());
    for (q, (&r, &p)) in quotes.iter().zip(res.residuals.iter().zip(&res.prices)) {
        assert!(r.is_finite(), "residual finite");
        if q.at_intrinsic {
            assert!(
                r >= 0.0 && r < 1.0,
                "one-sided quote K = {} residual {r} (price {p}, target {})",
                q.strike,
                q.one_sided_target
            );
        }
        if q.held_out {
            assert!(
                r.abs() < 5.0,
                "held-out quote K = {} evaluated: {r}",
                q.strike
            );
        }
    }
    let in_sample: Vec<f64> = quotes
        .iter()
        .zip(&res.residuals)
        .filter(|(q, _)| q.two_sided_in_fit())
        .map(|(_, &r)| r)
        .collect();
    let rms = (in_sample.iter().map(|r| r * r).sum::<f64>() / in_sample.len() as f64).sqrt();
    assert!(
        rms < 1.0,
        "in-sample RMS residual {rms} half-spreads (path {:?}, flags {:?})",
        res.path,
        res.flags
    );
    let sm = res.support_map.as_ref().expect("support map requested");
    assert_eq!(sm.values.len(), 26 * 4);
    assert!(sm.values.iter().all(|v| *v >= 0.0) && sm.values.iter().any(|v| *v > 0.0));
    let mask = sm.mask(1e-3);
    assert!(
        mask.iter().filter(|&&m| m).count() > 10,
        "support covers the quoted region"
    );
    // the surface reproduces the American prices it was fitted to
    let check = american_prices(
        &quotes,
        &cap,
        &surface.clone().with_theta(res.theta.clone()).unwrap(),
        cfg.rho,
    )
    .unwrap();
    assert!(
        max_abs_diff(&check, &res.prices) < 1e-10 * S0,
        "result prices equal a fresh reprice"
    );
    // discrepancy monotone along the path
    for w in res.path.windows(2) {
        assert!(
            w[1].discrepancy <= w[0].discrepancy * (1.0 + 1e-6),
            "monotone path: {:?}",
            res.path
        );
    }
    eprintln!(
        "tiny end-to-end: {} Jacobians, {} iterations, alpha* = {:.3e}, discrepancy {:.3} (target {:.3}), {:.0} ms",
        res.jacobians, res.iterations, res.alpha, res.discrepancy, res.discrepancy_target, res.wall_ms
    );
}

#[test]
fn free_carry_recovers_a_two_percent_dividend_yield_from_a_zero_prior_to_20_bp() {
    let expiries = [0.25, 0.5, 1.0];
    let n_t = |t: f64| ((t * 120.0).ceil() as usize).max(20);
    let truth_cap = capture_with(&expiries, 0.04, 0.02, 100, &n_t, &[]);
    let strikes: Vec<f64> = (0..9).map(|i| 80.0 + 5.0 * i as f64).collect();
    let truth = FlatVol(0.25);
    let quotes = quotes_from_truth(&truth_cap, &strikes, &truth, 0.01, Spread::Spy);
    // the calibration starts from a zero carry prior
    let cap = capture_with(&expiries, 0.04, 0.0, 100, &n_t, &[]);
    let surface = BSplineLocalVol::with_defaults(0.25, cap.f_ref_table()).unwrap();
    let mut cfg = CalibrationConfig::american();
    cfg.noise_floor = Some(1e-4 * S0);
    cfg.carry_frozen = false;
    cfg.jacobian_budget = 40;
    let mode = Mode::American { rho: cfg.rho };
    let lin = linearized_step(&quotes, &cap, &surface, &cfg, mode).unwrap();
    assert!(
        lin.q_lin.iter().all(|q| *q > 0.005),
        "the linear step already moves the carry toward 2%: {:?}",
        lin.q_lin
    );
    let mut warm_cap = cap.clone();
    warm_cap.set_q(&cap.q).unwrap();
    let warm = surface.clone().with_theta(lin.theta_lin.clone()).unwrap();
    let res = calibrate_american(
        &quotes,
        &warm_cap,
        &warm,
        &cfg,
        Some(16.0 * lin.alpha_lin),
        None,
    )
    .unwrap();
    eprintln!(
        "carry recovery: q* = {:?}, se = {:?}, corr = {:?}, {} Jacobians, {:.0} ms, flags {:?}",
        res.q, res.q_se, res.q_theta_correlation, res.jacobians, res.wall_ms, res.flags
    );
    for (k, &qk) in res.q.iter().enumerate() {
        assert!(
            (qk - 0.02).abs() < 20e-4,
            "carry {k} recovered to {qk} (true 0.02)"
        );
    }
    assert!(
        res.q_se.iter().all(|s| s.is_finite() && *s > 0.0),
        "carry standard errors {:?}",
        res.q_se
    );
    assert!(
        res.q_theta_correlation
            .iter()
            .all(|c| (0.0..=1.0).contains(c)),
        "{:?}",
        res.q_theta_correlation
    );
    // the surface stays near flat where the data are
    let vol = surface.clone().with_theta(res.theta.clone()).unwrap();
    for &t in &[0.1, 0.4, 0.9] {
        for &k in &[85.0f64, 100.0, 115.0] {
            let v = vol.vol(k.ln(), t);
            assert!((v - 0.25).abs() < 0.01, "surface at K = {k}, t = {t}: {v}");
        }
    }
}

/// SPEC 1.8: synthetic case (c), noiseless, truth on the fine mesh,
/// M1 then M2 on the working mesh with the noise-floor rule.
#[test]
fn recovery_of_synthetic_case_c_from_noiseless_quotes() {
    let t0 = Instant::now();
    let release = !cfg!(debug_assertions);
    let (n_x_work, n_x_truth) = if release { (400, 1600) } else { (200, 800) };
    let (r, q) = (0.04, 0.02);
    let expiries = [
        1.0 / 52.0,
        2.0 / 52.0,
        1.0 / 12.0,
        2.0 / 12.0,
        0.25,
        4.0 / 12.0,
        0.5,
        0.75,
        1.0,
    ];
    let strikes: Vec<f64> = (0..25).map(|i| 70.0 + 2.5 * i as f64).collect();
    let truth = case_c(r, q);
    let truth_cap = capture_with(&expiries, r, q, n_x_truth, &n_t_fine, &[]);
    let quotes = quotes_from_truth(&truth_cap, &strikes, &truth, 0.01, Spread::Uniform);
    let t_truth = t0.elapsed().as_secs_f64();
    let n_two_sided = quotes.iter().filter(|q| q.two_sided_in_fit()).count();
    assert!(
        quotes.len() > 400 && quotes.len() <= 450,
        "{} quotes with bid > 0",
        quotes.len()
    );

    // working mesh, F_prior = F_true (control), carry frozen at the truth
    let cap = capture_with(&expiries, r, q, n_x_work, &n_t_default, &[]);
    let sigma0 = 0.25;
    let surface = BSplineLocalVol::with_defaults(sigma0, cap.f_ref_table()).unwrap();
    // the discretization floor, MEASURED: the truth priced on the working
    // mesh against the fine-mesh quotes; the noise-floor delta reproduces
    // 1.1x that discrepancy through sum (delta / s_i)^2
    let p_work = american_prices(&quotes, &cap, &truth, RHO_DEFAULT).unwrap();
    let (mut disc_floor, mut inv_s2) = (0.0, 0.0);
    for (qt, &p) in quotes.iter().zip(&p_work) {
        if qt.two_sided_in_fit() {
            let r = qt.residual(p);
            disc_floor += r * r;
            inv_s2 += 1.0 / (qt.scale() * qt.scale());
        }
    }
    let delta_disc = (1.1 * disc_floor / inv_s2).sqrt();
    // the basis floor: the least-squares projection of the truth
    let f_true: Vec<(f64, f64)> = std::iter::once((0.0, S0.ln()))
        .chain(expiries.iter().map(|&t| (t, S0.ln() + (r - q) * t)))
        .collect();
    let theta_proj = surface.projection_of(&truth, Some(&f_true)).unwrap();
    let proj = surface.clone().with_theta(theta_proj).unwrap();
    eprintln!(
        "  floors: working-vs-fine discrepancy {disc_floor:.2} (N_eff {n_two_sided}) -> delta_disc {delta_disc:.2e}          ({:.2e} S0)",
        delta_disc / S0
    );
    let mut cfg = CalibrationConfig::synthetic();
    cfg.n_x = n_x_work;
    cfg.noise_floor = Some(delta_disc);
    cfg.carry_frozen = true;
    let mode = Mode::American { rho: cfg.rho };
    let t1 = Instant::now();
    let lin = linearized_step(&quotes, &cap, &surface, &cfg, mode).unwrap();
    let t_m1 = t1.elapsed().as_secs_f64();
    let grid = SupportGrid {
        y: (0..26).map(|i| -0.6 + 0.04 * i as f64).collect(),
        t: vec![1.0 / 12.0, 1.0 / 6.0, 0.25, 0.5, 0.75, 1.0],
    };
    let warm = surface.clone().with_theta(lin.theta_lin.clone()).unwrap();
    let t2 = Instant::now();
    let res = calibrate_american(
        &quotes,
        &cap,
        &warm,
        &cfg,
        Some(16.0 * lin.alpha_lin),
        Some(&grid),
    )
    .unwrap();
    let t_m2 = t2.elapsed().as_secs_f64();
    eprintln!(
        "recovery (n_x work {n_x_work}, truth {n_x_truth}): truth {t_truth:.1} s, M1 {t_m1:.2} s ({} levels, alpha_lin {:.3e}), \
         M2 {t_m2:.1} s ({} Jacobians, {} iterations, alpha* {:.3e}, discrepancy {:.2} of target {:.2}, N_eff {n_two_sided}), flags {:?}",
        lin.path.len(), lin.alpha_lin, res.jacobians, res.iterations, res.alpha, res.discrepancy, res.discrepancy_target, res.flags
    );
    for p in &res.path {
        eprintln!(
            "  path: alpha {:.3e} disc {:.3} R {:.3e} it {} jac {} switches {} {:?}",
            p.alpha,
            p.discrepancy,
            p.regularizer,
            p.iterations,
            p.jacobians,
            p.active_set_switches,
            p.kind
        );
    }
    assert!(
        !res.flags.discrepancy_not_reached,
        "discrepancy reached: {:?}",
        res.flags
    );
    assert!(
        !res.flags.jacobian_budget_exhausted,
        "within budget: {:?}",
        res.flags
    );
    // alpha* strictly inside the path: the first level did not already
    // satisfy the target and alpha* lies below it
    assert!(res.path.len() >= 2);
    assert!(
        res.path[0].discrepancy > res.discrepancy_target,
        "first level above the target"
    );
    assert!(
        res.alpha < res.path[0].alpha,
        "alpha* {} below alpha_0 {}",
        res.alpha,
        res.path[0].alpha
    );
    for w in res.path.windows(2) {
        assert!(
            w[1].discrepancy <= w[0].discrepancy * (1.0 + 1e-6),
            "discrepancy monotone along the path: {:?}",
            res.path
        );
    }
    // sigma^E at the quotes vs the truth's sigma^E (both on the working
    // mesh, so the discretization cancels)
    let vol = surface.clone().with_theta(res.theta.clone()).unwrap();
    let pe_fit = european_prices(&quotes, &cap, &vol).unwrap();
    let pe_true = european_prices(&quotes, &cap, &truth).unwrap();
    let iv_fit = european_implied_vols(&quotes, &cap, &pe_fit);
    let iv_true = european_implied_vols(&quotes, &cap, &pe_true);
    // resolved quotes: the half-spread in vol units s_i / nu^E_i is below
    // 25 bp (elsewhere the price carries no vol information at this
    // precision and sigma^E of any surface is noise)
    let mut worst_bp = 0.0f64;
    let mut worst_resolved_bp = 0.0f64;
    let (mut n_iv, mut n_resolved) = (0, 0);
    for ((a, b), qt) in iv_fit.iter().zip(&iv_true).zip(&quotes) {
        if let (Ok(a), Ok(b)) = (a, b) {
            let err_bp = (a - b).abs() * 1e4;
            worst_bp = worst_bp.max(err_bp);
            n_iv += 1;
            let (r_eff, q_eff, _, _) = cap.effective_rates(qt.expiry_index);
            let vega = crate::equity::blackscholes::bs_vega(S0, qt.strike, r_eff, q_eff, *b, qt.t);
            if qt.half_spread / vega < 25e-4 {
                worst_resolved_bp = worst_resolved_bp.max(err_bp);
                n_resolved += 1;
            }
        }
    }
    eprintln!(
        "  sigma^E at {n_iv} quotes: max error {worst_bp:.2} bp; {n_resolved} quotes resolved to 25 bp by their spread: max error {worst_resolved_bp:.2} bp"
    );
    // the 5 bp criterion is stated for the study mesh (n_x = 400, truth
    // 1600, release); the debug profile's n_x = 200 carries a 4x larger
    // working-vs-truth discretization gap, hence 20 bp there
    let sigma_e_tol_bp = if release { 5.0 } else { 20.0 };
    assert!(n_iv > 350, "most quotes invert: {n_iv}");
    assert!(n_resolved > 150, "resolved quotes: {n_resolved}");
    assert!(
        worst_resolved_bp < sigma_e_tol_bp,
        "sigma^E error {worst_resolved_bp} bp exceeds {sigma_e_tol_bp} bp"
    );
    // reconstruction error at physical (S, t) points, S = F_true(t) e^y, by
    // support level: the 1e-3 mask reaches the unquoted wings (below K =
    // 70 the truth's quadratic wing is unidentified and the regularizer
    // flattens it), the assertion is made on the well-supported cells
    let sm = res.support_map.as_ref().unwrap();
    let sm_max = sm.values.iter().cloned().fold(0.0, f64::max);
    let mut worst_proj = 0.0f64;
    let mut worst_at = [0.0f64; 4];
    let mut cells_at = [0usize; 4];
    let thresholds = [1e-3, 1e-2, 5e-2, 1e-1];
    for (iy, &y) in grid.y.iter().enumerate() {
        for (it, &t) in grid.t.iter().enumerate() {
            let frac = sm.at(iy, it) / sm_max;
            if frac < thresholds[0] {
                continue;
            }
            let s = S0 * ((r - q) * t).exp() * y.exp();
            let x = s.ln();
            let err = (vol.vol(x, t) - truth.vol(x, t)).abs();
            worst_proj = worst_proj.max((proj.vol(x, t) - truth.vol(x, t)).abs());
            for (k, &th) in thresholds.iter().enumerate() {
                if frac >= th {
                    worst_at[k] = worst_at[k].max(err);
                    cells_at[k] += 1;
                }
            }
            if err > 0.004 {
                eprintln!(
                    "    cell y {y:+.2} t {t:.3} (S {s:.1}): support {frac:.2e} of max, fit {:.4} truth {:.4} projection {:.4}",
                    vol.vol(x, t),
                    truth.vol(x, t),
                    proj.vol(x, t)
                );
            }
        }
    }
    for (k, &th) in thresholds.iter().enumerate() {
        eprintln!(
            "  reconstruction on support >= {th:.0e} of max: {} cells, max |Sigma* - Sigma_true| = {:.2} vol pt",
            cells_at[k],
            100.0 * worst_at[k]
        );
    }
    eprintln!(
        "  basis floor on the 1e-3 mask: {:.2} vol pt; total {:.1} s",
        100.0 * worst_proj,
        t0.elapsed().as_secs_f64()
    );
    assert!(cells_at[2] >= 30, "well-supported cells: {}", cells_at[2]);
    assert!(
        cells_at[3] >= 25,
        "strongly supported cells: {}",
        cells_at[3]
    );
    // the 0.5 vol pt criterion is stated for the study mesh (release:
    // measured 0.23 vol pt on the 5% support, 0.07 on the 10% support).
    // In the debug profile the working mesh is n_x = 200: the working-vs-
    // truth discretization gap is 4x larger, the noise-floor discrepancy
    // rule (Morozov at the measured floor) selects a 16x larger alpha*,
    // and the recovery on the 5% support degrades to ~0.8 vol pt (0.3 on
    // the 10% support); the debug run keeps the same assertions at twice
    // the tolerance on the 5% support and the full 0.5 vol pt on the 10%
    // support
    let recon_tol_5pct = if release { 0.005 } else { 0.010 };
    assert!(
        worst_at[2] < recon_tol_5pct,
        "reconstruction error on the well-supported region {} exceeds {} vol pt",
        worst_at[2],
        100.0 * recon_tol_5pct
    );
    assert!(
        worst_at[3] < 0.005,
        "reconstruction error on the strongly supported region {} exceeds 0.5 vol pt",
        worst_at[3]
    );
    assert!(worst_proj < 0.005, "basis floor {worst_proj}");
}

/// Wall-clock of one Jacobian pass (SPEC step 5). The `setup_ms` and
/// `wall_ms` figures are inflated when the other tests of this module run
/// concurrently; for a clean measurement run this test alone
/// (`cargo test --release --lib one_jacobian_pass -- --nocapture`).
#[test]
fn one_jacobian_pass_of_500_quotes_on_the_working_mesh_is_timed() {
    let expiries = [
        1.0 / 52.0,
        2.0 / 52.0,
        1.0 / 12.0,
        2.0 / 12.0,
        0.25,
        4.0 / 12.0,
        0.5,
        0.75,
        1.0,
    ];
    let cap = capture_with(&expiries, 0.04, 0.01, N_X_DEFAULT, &n_t_default, &[]);
    let strikes: Vec<f64> = (0..28).map(|i| 66.0 + 2.5 * i as f64).collect();
    let mut quotes = Vec::new();
    for (e, &t) in expiries.iter().enumerate() {
        for &k in &strikes {
            for right in [PutOrCall::Put, PutOrCall::Call] {
                quotes.push(Quote::from_mid(k, t, e, right, 5.0, 0.05));
            }
        }
    }
    assert_eq!(quotes.len(), 504);
    let surface = BSplineLocalVol::with_defaults(0.25, cap.f_ref_table()).unwrap();
    let cfg = CalibrationConfig::american();
    let mode = Mode::American { rho: cfg.rho };
    // first pass warms the thread-local workspaces; the second is timed
    let first = jacobian_rows(&quotes, &cap, &surface, &cfg, mode, None).unwrap();
    let second = jacobian_rows(&quotes, &cap, &surface, &cfg, mode, None).unwrap();
    eprintln!(
        "Jacobian pass, 504 quotes, n_x = {}, {} threads: {:.0} ms (setup {:.0} ms; first pass {:.0} ms), {} columns",
        N_X_DEFAULT,
        rayon::current_num_threads(),
        second.wall_ms,
        second.setup_ms,
        first.wall_ms,
        second.n_cols
    );
    assert_eq!(second.n_cols, surface.len() + expiries.len());
    assert!(second.prices.iter().all(|p| p.is_finite() && *p > 0.0));
    let min_vega = second.vegas.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        second.vegas.iter().all(|v| v.is_finite()) && min_vega > -1e-3,
        "vegas finite and nonnegative up to the penalty floor (deep ITM puts sit at intrinsic): min {min_vega}"
    );
    assert!(
        second.vegas.iter().filter(|v| **v > 1e-3).count() > 400,
        "most quotes carry vega"
    );
    assert!(
        max_abs_diff(&first.rows, &second.rows) == 0.0,
        "passes are deterministic"
    );
    // the parallel-shift derivative equals the sum of the theta row (partition of unity)
    for i in 0..quotes.len() {
        let row_sum: f64 = second.rows[i * second.n_cols..i * second.n_cols + second.n_theta]
            .iter()
            .sum();
        assert!(
            (row_sum - second.vegas[i]).abs() < 1e-9 * second.vegas[i].abs().max(1e-3),
            "quote {i}: sum of dF/dtheta {row_sum} vs vega {}",
            second.vegas[i]
        );
    }
    let budget_ms = if cfg!(debug_assertions) {
        60_000.0
    } else {
        3_000.0
    };
    assert!(
        second.wall_ms < budget_ms,
        "Jacobian pass took {:.0} ms",
        second.wall_ms
    );
}
