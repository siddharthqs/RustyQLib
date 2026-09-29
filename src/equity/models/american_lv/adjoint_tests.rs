//! Tests of the discrete adjoint: the flat vega and directional derivatives
//! against frozen-active-set finite differences (the transpose must be
//! exact), free finite differences (semismooth consistency), the carry
//! sensitivities, the European kernel against the closed-form Brownian
//! bridge, the American kernel's positivity and front-loading, and the
//! absence of a Crank-Nicolson checkerboard in the kernel fields.
//!
//! Finite-difference stencil: the frozen map is smooth in `sigma`
//! (rational), so its central difference carries the truncation term
//! `h^2 f'''/(6 f')`. Measured on these quotes it is 1-3e-8 relative at
//! `h = 1e-4` and shrinks by exactly 4x per halving of `h`, i.e. a 2-point
//! stencil cannot certify the transpose to 1e-8. The tests therefore use
//! the Richardson-extrapolated stencil `(4 D(h/2) - D(h)) / 3` (truncation
//! `O(h^4)`), which agrees with the adjoint to ~1e-9, the roundoff floor of
//! the frozen solves, and they check the `h^2` scaling of the plain stencil
//! explicitly so that a genuine transpose error (which does not scale)
//! cannot hide behind truncation.

use super::*;
use crate::core::trade::PutOrCall;
use crate::equity::models::american_lv::grid::{half_width, n_t_default, RANNACHER_STEPS};
use crate::equity::models::american_lv::solver::{price_frozen, price_only, solve_backward};

const S0: f64 = 100.0;
const SIGMA0: f64 = 0.25;

fn mesh_for(t_expiry: f64, n_x: usize, n_t: usize, divs: &[(f64, f64)]) -> Mesh {
    let l = half_width(SIGMA0, t_expiry, 0.05, S0, 0.7 * S0, 1.3 * S0);
    Mesh::new(S0, l, n_x, t_expiry, n_t, divs).unwrap()
}

fn flat_field(mesh: &Mesh, sigma: f64) -> NodeField {
    mesh.node_field(&FlatVol(sigma))
}

/// `sigma + h eta` as a node field.
fn bumped(vol: &NodeField, eta: &[f64], h: f64) -> NodeField {
    let mut out = vol.clone();
    for (v, e) in out.values.iter_mut().zip(eta) {
        *v += h * e;
    }
    out
}

/// A Gaussian bump in `(x, t)` sampled on the mesh, layout of `g`.
fn gaussian_bump(mesh: &Mesh, xc: f64, wx: f64, tc: f64, wt: f64) -> Vec<f64> {
    let mut eta = Vec::with_capacity(mesh.n_steps() * mesh.n_nodes());
    for &t in &mesh.t_mid {
        let ft = (-((t - tc) / wt).powi(2)).exp();
        for &x in &mesh.x {
            eta.push(ft * (-((x - xc) / wx).powi(2)).exp());
        }
    }
    eta
}

/// Plain central difference `(f(h) - f(-h)) / (2h)`.
fn fd_central(f: &mut dyn FnMut(f64) -> f64, h: f64) -> f64 {
    (f(h) - f(-h)) / (2.0 * h)
}

/// Richardson-extrapolated central difference `(4 D(h/2) - D(h)) / 3`,
/// truncation `O(h^4)` (see the module doc).
fn fd_richardson(f: &mut dyn FnMut(f64) -> f64, h: f64) -> f64 {
    let d1 = fd_central(f, h);
    let d2 = fd_central(f, 0.5 * h);
    (4.0 * d2 - d1) / 3.0
}

/// One test case: mesh, market, spec, mode, flat level; `directional`
/// marks the cases whose kernel is wide enough for the Gaussian bumps of
/// the directional test to carry a non-negligible derivative.
struct Case {
    name: &'static str,
    mesh: Mesh,
    market: MarketSlice,
    spec: QuoteSpec,
    mode: Mode,
    sigma: f64,
    directional: bool,
    expect_upwinded: bool,
}

fn cases() -> Vec<Case> {
    let t = 0.5;
    let n_t = n_t_default(t);
    let mut v = Vec::new();
    for (name, right, k, r, q, mode) in [
        (
            "European put",
            PutOrCall::Put,
            100.0,
            0.04,
            0.0,
            Mode::European,
        ),
        (
            "European call",
            PutOrCall::Call,
            100.0,
            0.02,
            0.06,
            Mode::European,
        ),
        (
            "American put ATM",
            PutOrCall::Put,
            100.0,
            0.04,
            0.0,
            Mode::american(),
        ),
        (
            "American put ITM",
            PutOrCall::Put,
            110.0,
            0.04,
            0.0,
            Mode::american(),
        ),
        (
            "American call ITM (q > r)",
            PutOrCall::Call,
            90.0,
            0.02,
            0.06,
            Mode::american(),
        ),
        (
            "American call ATM (q > r)",
            PutOrCall::Call,
            100.0,
            0.02,
            0.06,
            Mode::american(),
        ),
    ] {
        let mesh = mesh_for(t, 400, n_t, &[]);
        let market = MarketSlice::flat(&mesh, S0, r, q, &[]);
        v.push(Case {
            name,
            mesh,
            market,
            spec: QuoteSpec::new(k, t, right),
            mode,
            sigma: SIGMA0,
            directional: true,
            expect_upwinded: false,
        });
    }
    // the dividend call of SPEC 1.3 (v): K = 90, r = 4%, 1.5 at t = 0.2, T = 0.5
    // (mixed mask at the ex-date), and a European put across the same jump
    let divs = [(0.2, 1.5)];
    for (name, right, k, mode) in [
        (
            "American call with a cash dividend",
            PutOrCall::Call,
            90.0,
            Mode::american(),
        ),
        (
            "European put with a cash dividend",
            PutOrCall::Put,
            100.0,
            Mode::European,
        ),
    ] {
        let mesh = mesh_for(t, 400, n_t, &divs);
        let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &divs);
        v.push(Case {
            name,
            mesh,
            market,
            spec: QuoteSpec::new(k, t, right),
            mode,
            sigma: SIGMA0,
            directional: true,
            expect_upwinded: false,
        });
    }
    // upwinded rows: sigma = 0.04, r - q = 0.15 on a coarse grid (dx = 0.017)
    // fails the Peclet bound |mu| dx <= sigma^2 on every interior row, at
    // sigma +- h as well, so the frozen-stencil derivative must still be exact
    for (name, right, mode) in [
        (
            "European call, upwinded rows",
            PutOrCall::Call,
            Mode::European,
        ),
        (
            "American put, upwinded rows",
            PutOrCall::Put,
            Mode::american(),
        ),
    ] {
        let mesh = mesh_for(t, 100, n_t, &[]);
        let market = MarketSlice::flat(&mesh, S0, 0.15, 0.0, &[]);
        v.push(Case {
            name,
            mesh,
            market,
            spec: QuoteSpec::new(100.0, t, right),
            mode,
            sigma: 0.04,
            directional: false,
            expect_upwinded: true,
        });
    }
    v
}

#[test]
fn field_sum_equals_the_frozen_finite_difference_vega_to_1e_8() {
    let mut ws = Workspace::new();
    let mut aws = AdjointWorkspace::new();
    for c in cases() {
        let vol = flat_field(&c.mesh, c.sigma);
        let sol = solve_backward(&c.mesh, &c.spec, &vol, &c.market, c.mode, true, &mut ws).unwrap();
        assert_eq!(
            sol.penalty_inconsistent_steps, 0,
            "{}: consistent active sets",
            c.name
        );
        assert_eq!(
            sol.upwinded_rows > 0,
            c.expect_upwinded,
            "{}: upwinded rows {}",
            c.name,
            sol.upwinded_rows
        );
        let adj = adjoint(&c.mesh, &c.spec, &sol, &c.market, &vol, &mut aws).unwrap();
        let mut price_at = |h: f64| {
            price_frozen(
                &c.mesh,
                &c.spec,
                &flat_field(&c.mesh, c.sigma + h),
                &c.market,
                &sol,
                &mut ws,
            )
            .unwrap()
        };
        let fd = fd_richardson(&mut price_at, 1e-4);
        assert!(fd > 0.0, "{}: positive vega {fd}", c.name);
        let rel = (adj.vega_from_field - fd).abs() / fd.abs();
        assert!(
            rel <= 1e-8,
            "{}: sum g = {} vs frozen FD vega {fd} (rel {rel:.2e})",
            c.name,
            adj.vega_from_field
        );
        // the plain stencil's discrepancy is truncation: it scales as h^2
        let e1 = (fd_central(&mut price_at, 4e-4) - adj.vega_from_field).abs();
        let e2 = (fd_central(&mut price_at, 2e-4) - adj.vega_from_field).abs();
        if e2 > 1e-9 * fd {
            let ratio = e1 / e2;
            assert!(
                (3.5..=4.5).contains(&ratio),
                "{}: plain central-difference error must scale as h^2: {e1:.3e} / {e2:.3e} = {ratio:.3}",
                c.name
            );
        }
        // edge rows carry no sigma
        for step in 0..c.mesh.n_steps() {
            let row = adj.g_row(step);
            assert_eq!(row[0], 0.0);
            assert_eq!(row[c.mesh.n_x], 0.0);
        }
        // the dyn path, the streaming path and vega_from_adjoint agree exactly
        let adj_dyn = adjoint_dyn(
            &c.mesh,
            &c.spec,
            &sol,
            &c.market,
            &FlatVol(c.sigma),
            &mut aws,
        )
        .unwrap();
        assert_eq!(
            adj_dyn.vega_from_field, adj.vega_from_field,
            "{}: dyn path",
            c.name
        );
        assert_eq!(adj_dyn.g, adj.g);
        let mut streamed = vec![0.0; adj.g.len()];
        let mut h_stream = vec![0.0; c.mesh.n_steps()];
        let stride = c.mesh.n_nodes();
        let vega = adjoint_stream(
            &c.mesh,
            &c.spec,
            &sol,
            &c.market,
            &vol,
            &mut aws,
            |step, row, hn| {
                streamed[step * stride..(step + 1) * stride].copy_from_slice(row);
                h_stream[step] = hn;
            },
        )
        .unwrap();
        assert_eq!(vega, adj.vega_from_field);
        assert_eq!(streamed, adj.g);
        assert_eq!(h_stream, adj.h);
        let v2 = vega_from_adjoint(&c.mesh, &c.spec, &sol, &c.market, &vol, &mut aws).unwrap();
        assert_eq!(v2, adj.vega_from_field);
        // buffer reuse
        let mut again = adj.clone();
        adjoint_into(
            &c.mesh, &c.spec, &sol, &c.market, &vol, &mut aws, &mut again,
        )
        .unwrap();
        assert_eq!(again.g, adj.g);
    }
}

#[test]
fn directional_derivatives_match_frozen_fd_to_1e_8_and_free_fd_to_1e_3() {
    let mut ws = Workspace::new();
    let mut aws = AdjointWorkspace::new();
    for c in cases().into_iter().filter(|c| c.directional) {
        let vol = flat_field(&c.mesh, c.sigma);
        let sol = solve_backward(&c.mesh, &c.spec, &vol, &c.market, c.mode, true, &mut ws).unwrap();
        let adj = adjoint(&c.mesh, &c.spec, &sol, &c.market, &vol, &mut aws).unwrap();
        let x0 = c.mesh.x0;
        let t = c.mesh.t_expiry;
        let mut bumps = vec![
            ("centre", gaussian_bump(&c.mesh, x0, 0.1, 0.5 * t, 0.15)),
            (
                "low early",
                gaussian_bump(&c.mesh, x0 - 0.15, 0.08, 0.1, 0.08),
            ),
            (
                "high late",
                gaussian_bump(&c.mesh, x0 + 0.1, 0.08, 0.4, 0.08),
            ),
            ("wide", gaussian_bump(&c.mesh, x0, 0.4, 0.5 * t, 0.5)),
        ];
        if !c.mesh.div_steps.is_empty() {
            // a pure time bump straddling the ex-date 0.2 (all x)
            bumps.push((
                "t-bump straddling the ex-date",
                gaussian_bump(&c.mesh, x0, 1e6, 0.2, 0.06),
            ));
            let mask = sol.jump_mask(0);
            let kept = mask.iter().filter(|&&m| m == 1).count();
            if c.mode.is_american() {
                assert!(
                    kept > 0 && kept < mask.len(),
                    "{}: the ex-date mask is mixed ({kept} kept)",
                    c.name
                );
            } else {
                assert_eq!(
                    kept,
                    mask.len(),
                    "{}: European mask is the identity",
                    c.name
                );
            }
        }
        for (bname, eta) in &bumps {
            let deriv = adj.directional(eta);
            assert!(
                deriv.abs() > 1e-4 * adj.vega_from_field,
                "{} / {bname}: derivative {deriv} is not negligible",
                c.name
            );
            // (b1) frozen sets: exact transpose
            let mut frozen_at = |h: f64| {
                price_frozen(
                    &c.mesh,
                    &c.spec,
                    &bumped(&vol, eta, h),
                    &c.market,
                    &sol,
                    &mut ws,
                )
                .unwrap()
            };
            let fd = fd_richardson(&mut frozen_at, 1e-4);
            assert!(
                (deriv - fd).abs() <= 1e-8 * fd.abs(),
                "{} / {bname}: <g, eta> = {deriv} vs frozen FD {fd} (rel {:.2e})",
                c.name,
                (deriv - fd).abs() / fd.abs()
            );
            // (b2) free penalty iteration: semismooth consistency
            let h = 1e-3;
            let up = price_only(
                &c.mesh,
                &c.spec,
                &bumped(&vol, eta, h),
                &c.market,
                c.mode,
                &mut ws,
            )
            .unwrap();
            let dn = price_only(
                &c.mesh,
                &c.spec,
                &bumped(&vol, eta, -h),
                &c.market,
                c.mode,
                &mut ws,
            )
            .unwrap();
            let fd_free = (up - dn) / (2.0 * h);
            assert!(
                (deriv - fd_free).abs() <= 1e-3 * fd_free.abs(),
                "{} / {bname}: <g, eta> = {deriv} vs free FD {fd_free} (rel {:.2e})",
                c.name,
                (deriv - fd_free).abs() / fd_free.abs()
            );
        }
    }
}

#[test]
fn carry_sensitivities_match_finite_differences() {
    let mut ws = Workspace::new();
    let mut aws = AdjointWorkspace::new();
    for c in cases() {
        let vol = flat_field(&c.mesh, c.sigma);
        let sol = solve_backward(&c.mesh, &c.spec, &vol, &c.market, c.mode, true, &mut ws).unwrap();
        let adj = adjoint(&c.mesh, &c.spec, &sol, &c.market, &vol, &mut aws).unwrap();
        let t = c.mesh.t_expiry;
        let intervals = [(0.0, 0.5 * t), (0.5 * t, t + 1e-9)];
        let dq = adj.carry_sensitivities(&c.mesh, &intervals);
        assert_eq!(dq.len(), 2);
        let mut dq_into = [0.0; 2];
        carry_sensitivities_into(&c.mesh, &adj.h, &intervals, &mut dq_into);
        assert_eq!(dq, dq_into, "{}: the in-place variant agrees", c.name);
        // every step belongs to exactly one interval
        let total: f64 = adj.h.iter().sum();
        assert!(
            (dq[0] + dq[1] + total).abs() < 1e-12 * total.abs().max(1.0),
            "{}: partition of the steps",
            c.name
        );
        for (k, &(a, b)) in intervals.iter().enumerate() {
            let bump = |h: f64| {
                let mut m = c.market.clone();
                for (n, &tm) in c.mesh.t_mid.iter().enumerate() {
                    if tm >= a && tm < b {
                        m.carry[n] += h;
                    }
                }
                m
            };
            let mut frozen_at =
                |h: f64| price_frozen(&c.mesh, &c.spec, &vol, &bump(h), &sol, &mut ws).unwrap();
            let fd = fd_richardson(&mut frozen_at, 1e-4);
            assert!(
                (dq[k] - fd).abs() <= 1e-8 * fd.abs().max(1e-3),
                "{} / interval {k}: dF/dq = {} vs frozen FD {fd} (rel {:.2e})",
                c.name,
                dq[k],
                (dq[k] - fd).abs() / fd.abs()
            );
            let h = 1e-4;
            let up = price_only(&c.mesh, &c.spec, &vol, &bump(h), c.mode, &mut ws).unwrap();
            let dn = price_only(&c.mesh, &c.spec, &vol, &bump(-h), c.mode, &mut ws).unwrap();
            let fd_free = (up - dn) / (2.0 * h);
            assert!(
                (dq[k] - fd_free).abs() <= 1e-3 * fd_free.abs().max(1e-3),
                "{} / interval {k}: dF/dq = {} vs free FD {fd_free} (rel {:.2e})",
                c.name,
                dq[k],
                (dq[k] - fd_free).abs() / fd_free.abs()
            );
        }
        // a call's price falls with carry, a put's rises
        let sign = match c.spec.right {
            PutOrCall::Call => -1.0,
            PutOrCall::Put => 1.0,
        };
        assert!(
            sign * dq[0] > 0.0 && sign * dq[1] > 0.0,
            "{}: sign of dF/dq {dq:?}",
            c.name
        );
    }
}

#[test]
fn european_kernel_matches_the_brownian_bridge_closed_form() {
    let t = 0.5;
    let mesh = mesh_for(t, 400, n_t_default(t), &[]);
    let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
    for &k in &[100.0, 110.0] {
        let spec = QuoteSpec::new(k, t, PutOrCall::Put);
        let ker = kernels_flat(&mesh, &spec, SIGMA0, &market, 1e7).unwrap();
        let dens = field_density(&mesh, &ker.kappa_e);
        let cf = brownian_bridge_kernel(&mesh, &spec, SIGMA0);
        let stride = mesh.n_nodes();
        let mut l1 = 0.0;
        let mut cf_mass = 0.0;
        for n in 0..mesh.n_steps() {
            let cell = mesh.dx * mesh.dt[n];
            for j in 0..stride {
                l1 += (dens[n * stride + j] - cf[n * stride + j]).abs() * cell;
                cf_mass += cf[n * stride + j] * cell;
            }
        }
        assert!(
            (cf_mass - 1.0).abs() < 1e-3,
            "closed form integrates to 1 on the mesh: {cf_mass}"
        );
        assert!(
            l1 < 2e-2,
            "K = {k}: L1 distance to the bridge kernel {l1:.3e}"
        );
        // uniform time marginal: w^E(t) T = 1 on every step
        for (n, &w) in ker.w_e.iter().enumerate() {
            assert!(
                (w * t - 1.0).abs() < 1e-2,
                "K = {k}: w^E T at step {n} (t = {:.4}) = {}",
                mesh.t_mid[n],
                w * t
            );
        }
        let kappa_sum: f64 = ker.kappa_e.iter().sum();
        assert!((kappa_sum - 1.0).abs() < 1e-12, "normalized");
        // the European kernel does not depend on r, q (only through the mesh error)
        let market2 = MarketSlice::flat(&mesh, S0, 0.0, 0.03, &[]);
        let ker2 = kernels_flat(&mesh, &spec, SIGMA0, &market2, 1e7).unwrap();
        let tv = total_variation(&ker.kappa_e, &ker2.kappa_e);
        assert!(
            tv < 1e-2,
            "K = {k}: European kernel drift-independence, TV {tv:.3e}"
        );
    }
}

#[test]
fn american_kernel_is_nonnegative_front_loaded_and_satisfies_the_endpoint_identity() {
    let t = 0.5;
    let mesh = mesh_for(t, 400, n_t_default(t), &[]);
    let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
    let spec = QuoteSpec::new(110.0, t, PutOrCall::Put);
    let ker = kernels_flat(&mesh, &spec, SIGMA0, &market, 1e7).unwrap();
    assert!(
        ker.price_a > ker.price_e && ker.vega_a > 0.0 && ker.vega_e > ker.vega_a,
        "prices {} > {}, vegas {} < {}",
        ker.price_a,
        ker.price_e,
        ker.vega_a,
        ker.vega_e
    );
    let max_a = ker.kappa_a.iter().cloned().fold(0.0, f64::max);
    let min_a = ker.kappa_a.iter().cloned().fold(0.0, f64::min);
    assert!(
        min_a >= -1e-3 * max_a,
        "kappa^A >= -1e-3 relative: min {min_a:.3e} vs max {max_a:.3e}"
    );
    // the kernel vanishes on the exercise region
    let stride = mesh.n_nodes();
    for n in 0..mesh.n_steps() {
        for j in 0..stride {
            if ker.active_a[n * stride + j] == 1 {
                assert!(
                    ker.kappa_a[n * stride + j].abs() <= 1e-3 * max_a,
                    "kappa^A on the exercise region at ({n}, {j})"
                );
            }
        }
    }
    // w^A nonincreasing (Theorem front-loading), up to a small tolerance
    let w = &ker.w_a;
    let mut max_uptick = 0.0f64;
    for k in 1..w.len() {
        max_uptick = max_uptick.max(w[k] - w[k - 1]);
    }
    assert!(
        max_uptick <= 2e-3 * w[0],
        "w^A nonincreasing: max uptick {max_uptick:.3e} vs w^A(0) = {}",
        w[0]
    );
    assert!(w[0] * t >= 1.0, "w^A(0) T = {} >= 1", w[0] * t);
    assert!(w[w.len() - 1] < w[0], "strictly decreasing over [0, T)");
    // smooth in time on the uniform interior (the Rannacher half-steps sit at the ends)
    let n_steps = mesh.n_steps();
    for k in 5..n_steps - 5 {
        let d2 = w[k + 1] - 2.0 * w[k] + w[k - 1];
        assert!(
            d2.abs() <= 2e-2 * w[k],
            "second difference of w^A at step {k}: {d2:.3e} vs w = {}",
            w[k]
        );
    }
    // sum_n w dt = 1
    let mass: f64 = w.iter().zip(&mesh.dt).map(|(a, b)| a * b).sum();
    assert!((mass - 1.0).abs() < 1e-12, "w^A integrates to 1: {mass}");
    // endpoint identity w^A(0) T = [Gamma^A / Gamma^E](S0, 0) nu^E / nu^A
    let lhs = w[0] * t;
    let rhs = ker.dollar_gamma_a0 / ker.dollar_gamma_e0 * ker.vega_e / ker.vega_a;
    assert!(
        (lhs - rhs).abs() < 0.02 * rhs,
        "w^A(0) T = {lhs} vs Gamma ratio x vega ratio {rhs}"
    );
    // the identification capacity is a distance in [0, 1]
    assert!(
        ker.tv_distance > 0.05 && ker.tv_distance < 1.0,
        "TV distance {}",
        ker.tv_distance
    );
    // the OTM put has a smaller identification gap than the ITM put
    let otm = kernels_flat(
        &mesh,
        &QuoteSpec::new(90.0, t, PutOrCall::Put),
        SIGMA0,
        &market,
        1e7,
    )
    .unwrap();
    assert!(
        otm.tv_distance < ker.tv_distance,
        "OTM {} < ITM {}",
        otm.tv_distance,
        ker.tv_distance
    );
    // the dividend call's w^A jumps down at the ex-date
    let divs = [(0.2, 1.5)];
    let mesh_d = mesh_for(t, 400, n_t_default(t), &divs);
    let market_d = MarketSlice::flat(&mesh_d, S0, 0.04, 0.0, &divs);
    let ker_d = kernels_flat(
        &mesh_d,
        &QuoteSpec::new(90.0, t, PutOrCall::Call),
        SIGMA0,
        &market_d,
        1e7,
    )
    .unwrap();
    let m = mesh_d.div_steps[0].0;
    assert!(
        ker_d.w_a[m] < 0.9 * ker_d.w_a[m - 1],
        "w^A drops across the ex-date: {} -> {}",
        ker_d.w_a[m - 1],
        ker_d.w_a[m]
    );
}

/// Largest row-wise alternating fraction `|sum_j (-1)^j kappa_{j,n}| /
/// sum_j kappa_{j,n}` over the steps selected by `include(n)`: the
/// projection of each time slice on the checkerboard mode. A slice that is
/// smooth on the scale of a few nodes projects to ~0 (a Gaussian of width
/// `w` nodes gives `~ exp(-pi^2 w^2 / 2)`), so the fraction detects a
/// checkerboard without confusing it with genuine curvature.
fn alternating_fraction(mesh: &Mesh, kappa: &[f64], include: impl Fn(usize) -> bool) -> f64 {
    let stride = mesh.n_nodes();
    kappa
        .chunks(stride)
        .enumerate()
        .filter(|(n, _)| include(*n))
        .map(|(_, row)| {
            let s: f64 = row.iter().sum();
            let a: f64 = row
                .iter()
                .enumerate()
                .map(|(j, v)| if j % 2 == 0 { *v } else { -*v })
                .sum();
            a.abs() / s
        })
        .fold(0.0, f64::max)
}

/// Crank-Nicolson with the implicit steps at `t = 0` (the adjoint's delta
/// seed) and at `T` versus a fully implicit march on the same grid. The two
/// time schemes converge to the same kernel and differ by their own
/// truncation errors: globally by 3e-3 in total variation; pointwise by
/// `O(dt / t)` where the bridge is not resolved (width below 8 dx at both
/// ends, where the fully implicit scheme's numerical diffusion is largest)
/// and by < 1e-2 of the peak on the resolved part. The checkerboard the
/// implicit seed steps are there to prevent is measured directly by the
/// row-wise alternating fraction, which is < 1e-3 for both schemes and
/// > 5e-3 (alternating in sign from step to step) for the control that runs
/// Crank-Nicolson from the first step.
#[test]
fn fully_implicit_and_crank_nicolson_kernels_agree_away_from_the_boundary() {
    let t = 0.5;
    let mesh = mesh_for(t, 400, n_t_default(t), &[]);
    let mut mesh_impl = mesh.clone();
    mesh_impl.theta.iter_mut().for_each(|th| *th = 1.0);
    let mut mesh_cn0 = mesh.clone();
    for th in mesh_cn0.theta.iter_mut().take(RANNACHER_STEPS) {
        *th = 0.5;
    }
    let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
    let spec = QuoteSpec::new(110.0, t, PutOrCall::Put);
    let cn = kernels_flat(&mesh, &spec, SIGMA0, &market, 1e7).unwrap();
    let im = kernels_flat(&mesh_impl, &spec, SIGMA0, &market, 1e7).unwrap();
    let cn0 = kernels_flat(&mesh_cn0, &spec, SIGMA0, &market, 1e7).unwrap();
    // bridge width sigma0 sqrt(t (T - t) / T) in nodes at step n
    let width = |n: usize| {
        let tm = mesh.t_mid[n];
        SIGMA0 * (tm * (t - tm) / t).sqrt() / mesh.dx
    };
    // (1) global agreement
    let tv_a = total_variation(&cn.kappa_a, &im.kappa_a);
    let tv_e = total_variation(&cn.kappa_e, &im.kappa_e);
    assert!(
        tv_a < 1e-2,
        "American kernels: TV(CN, implicit) = {tv_a:.3e}"
    );
    assert!(
        tv_e < 1e-2,
        "European kernels: TV(CN, implicit) = {tv_e:.3e}"
    );
    assert!(
        (cn.vega_a - im.vega_a).abs() < 5e-3 * cn.vega_a
            && (cn.vega_e - im.vega_e).abs() < 5e-3 * cn.vega_e,
        "vegas: A {} vs {}, E {} vs {}",
        cn.vega_a,
        im.vega_a,
        cn.vega_e,
        im.vega_e
    );
    // (2) pointwise on the resolved part of the mesh, away from the boundary
    let stride = mesh.n_nodes();
    let max_a = cn.kappa_a.iter().cloned().fold(0.0, f64::max);
    let max_e = cn.kappa_e.iter().cloned().fold(0.0, f64::max);
    let (mut worst_a, mut worst_e) = (0.0f64, 0.0f64);
    let mut resolved_steps = 0;
    for n in 0..mesh.n_steps() {
        if width(n) < 8.0 {
            continue;
        }
        resolved_steps += 1;
        let active = &cn.active_a[n * stride..(n + 1) * stride];
        for j in 0..stride {
            let near_boundary = (j.saturating_sub(4)..(j + 5).min(stride)).any(|i| active[i] == 1);
            if !near_boundary {
                worst_a =
                    worst_a.max((cn.kappa_a[n * stride + j] - im.kappa_a[n * stride + j]).abs());
            }
            worst_e = worst_e.max((cn.kappa_e[n * stride + j] - im.kappa_e[n * stride + j]).abs());
        }
    }
    assert!(
        resolved_steps > mesh.n_steps() * 3 / 4,
        "most steps are resolved: {resolved_steps} of {}",
        mesh.n_steps()
    );
    assert!(
        worst_a < 1e-2 * max_a,
        "American kernels: max |CN - implicit| {worst_a:.3e} vs max {max_a:.3e}"
    );
    assert!(
        worst_e < 1e-2 * max_e,
        "European kernels: max |CN - implicit| {worst_e:.3e} vs max {max_e:.3e}"
    );
    // (3) no checkerboard once the delta seed is wider than 4 nodes
    let include = |n: usize| width(n) >= 4.0;
    for (name, ker) in [("CN + implicit ends", &cn), ("fully implicit", &im)] {
        let fa = alternating_fraction(&mesh, &ker.kappa_a, include);
        let fe = alternating_fraction(&mesh, &ker.kappa_e, include);
        assert!(
            fa < 1e-3 && fe < 1e-3,
            "{name}: alternating fraction A {fa:.3e}, E {fe:.3e}"
        );
    }
    // the control without the implicit seed steps shows the checkerboard
    let fa0 = alternating_fraction(&mesh, &cn0.kappa_a, include);
    let fe0 = alternating_fraction(&mesh, &cn0.kappa_e, include);
    assert!(
        fa0 > 5e-3 && fe0 > 5e-3,
        "control (CN from the first step): alternating fraction A {fa0:.3e}, E {fe0:.3e}"
    );
    let first = (0..mesh.n_steps()).find(|&n| include(n)).unwrap();
    let alt = |n: usize| -> f64 {
        cn0.kappa_e[n * stride..(n + 1) * stride]
            .iter()
            .enumerate()
            .map(|(j, v)| if j % 2 == 0 { *v } else { -*v })
            .sum()
    };
    assert!(
        alt(first) * alt(first + 1) < 0.0 && alt(first + 1) * alt(first + 2) < 0.0,
        "the control's checkerboard alternates in sign from step to step"
    );
}

#[test]
fn adjoint_rejects_unretained_or_mismatched_inputs() {
    let t = 0.5;
    let mesh = mesh_for(t, 40, 20, &[]);
    let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
    let spec = QuoteSpec::new(100.0, t, PutOrCall::Put);
    let vol = flat_field(&mesh, SIGMA0);
    let mut ws = Workspace::new();
    let mut aws = AdjointWorkspace::new();
    let unretained = solve_backward(
        &mesh,
        &spec,
        &vol,
        &market,
        Mode::american(),
        false,
        &mut ws,
    )
    .unwrap();
    assert!(adjoint(&mesh, &spec, &unretained, &market, &vol, &mut aws).is_err());
    let sol = solve_backward(&mesh, &spec, &vol, &market, Mode::american(), true, &mut ws).unwrap();
    let other = mesh_for(t, 40, 21, &[]);
    assert!(adjoint(
        &other,
        &spec,
        &sol,
        &MarketSlice::flat(&other, S0, 0.04, 0.0, &[]),
        &flat_field(&other, SIGMA0),
        &mut aws
    )
    .is_err());
    assert!(kernels_flat(&mesh, &spec, -0.1, &market, 1e7).is_err());
    // deep-ITM put at intrinsic has no American vega: no kernel
    let deep = QuoteSpec::new(200.0, t, PutOrCall::Put);
    let mesh_w = mesh_for(t, 40, 20, &[]);
    let mk = MarketSlice::flat(&mesh_w, S0, 0.10, 0.0, &[]);
    let res = kernels_flat(&mesh_w, &deep, 0.05, &mk, 1e7);
    assert!(
        res.is_err() || res.unwrap().vega_a < 1e-6,
        "no kernel at intrinsic"
    );
}
