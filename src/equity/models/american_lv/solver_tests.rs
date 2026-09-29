//! Tests of the penalized backward solver: the in-place tridiagonal kernel
//! against the crate kernel, European prices against Black-Scholes,
//! American puts against a published Forsyth-Vetzal value and a CRR tree,
//! Richardson convergence, `O(1/rho)` penalty convergence, put-call parity
//! across a cash dividend, the dividend call against a Bermudan
//! quadrature reference, time-dependent volatility, effective flat rates
//! on a sloped curve, and the boundary dollar-gamma identity.

use super::*;
use crate::core::fd_solvers::thomas_algorithm;
use crate::core::utils::{norm_cdf, norm_pdf};
use crate::equity::blackscholes::{bs_price, implied_vol_from_price};
use crate::equity::models::american_lv::grid::{self, effective_rates, half_width};
use crate::equity::models::american_lv::vol_field::{CallbackVol, FlatVol};

const S0: f64 = 100.0;

/// A capture-style mesh: half-width from the spec's rule with the given
/// reference vol, longest expiry and strike range.
fn mesh_for(
    sigma_ref: f64,
    t_max: f64,
    t_expiry: f64,
    n_x: usize,
    n_t: usize,
    divs: &[(f64, f64)],
) -> Mesh {
    let l = half_width(sigma_ref, t_max, 0.05, S0, 0.7 * S0, 1.3 * S0);
    Mesh::new(S0, l, n_x, t_expiry, n_t, divs).unwrap()
}

fn flat_field(mesh: &Mesh, sigma: f64) -> NodeField {
    mesh.node_field(&FlatVol(sigma))
}

/// Tolerance rule of SPEC 1.3: `max(1e-4 relative, 1e-6 S0 absolute)`.
fn tol_of(reference: f64) -> f64 {
    (1e-4 * reference.abs()).max(1e-6 * S0)
}

/// CRR binomial American option with `n` steps.
fn crr_american(
    s0: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    right: PutOrCall,
    n: usize,
) -> f64 {
    let dt = t / n as f64;
    let u = (sigma * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = (((r - q) * dt).exp() - d) / (u - d);
    let disc = (-r * dt).exp();
    let payoff = |s: f64| match right {
        PutOrCall::Call => (s - k).max(0.0),
        PutOrCall::Put => (k - s).max(0.0),
    };
    let mut v: Vec<f64> = (0..=n)
        .map(|i| payoff(s0 * u.powi(i as i32) * d.powi((n - i) as i32)))
        .collect();
    for m in (0..n).rev() {
        for i in 0..=m {
            let s = s0 * u.powi(i as i32) * d.powi((m - i) as i32);
            let cont = disc * (p * v[i + 1] + (1.0 - p) * v[i]);
            v[i] = cont.max(payoff(s));
        }
    }
    v[0]
}

/// Simpson's rule on `[a, b]` with `n` (even) intervals.
fn simpson(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

#[test]
fn thomas_inplace_matches_the_crate_kernel_to_1e_14() {
    let n = 57;
    let a: Vec<f64> = (0..n - 1).map(|i| -0.3 + 0.01 * (i as f64).sin()).collect();
    let c: Vec<f64> = (0..n - 1)
        .map(|i| -0.4 + 0.02 * (i as f64 * 0.7).cos())
        .collect();
    let b: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.05 * (i as f64 * 0.3).sin())
        .collect();
    let d: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.11).sin() * 3.0 + 1.0)
        .collect();
    let reference = thomas_algorithm(&a, &b, &c, &d);
    let (mut cw, mut dw, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    thomas_inplace(&a, &b, &c, &d, &mut cw, &mut dw, &mut x);
    for i in 0..n {
        assert!(
            (x[i] - reference[i]).abs() <= 1e-14 * reference[i].abs().max(1.0),
            "row {i}: {} vs {}",
            x[i],
            reference[i]
        );
    }
    // the transpose solve (bands swapped) solves A^T y = d
    let mut y = vec![0.0; n];
    thomas_inplace(&c, &b, &a, &d, &mut cw, &mut dw, &mut y);
    for i in 0..n {
        let mut r = b[i] * y[i];
        if i > 0 {
            r += c[i - 1] * y[i - 1];
        }
        if i + 1 < n {
            r += a[i] * y[i + 1];
        }
        assert!(
            (r - d[i]).abs() < 1e-12,
            "transpose residual row {i}: {r} vs {}",
            d[i]
        );
    }
    // n = 1 and n = 2 edge cases
    let mut x1 = vec![0.0];
    thomas_inplace(&[], &[2.0], &[], &[4.0], &mut [0.0], &mut [0.0], &mut x1);
    assert_eq!(x1[0], 2.0);
    let mut x2 = vec![0.0; 2];
    thomas_inplace(
        &[1.0],
        &[2.0, 3.0],
        &[0.5],
        &[1.0, 2.0],
        &mut [0.0; 2],
        &mut [0.0; 2],
        &mut x2,
    );
    let r2 = thomas_algorithm(&[1.0], &[2.0, 3.0], &[0.5], &[1.0, 2.0]);
    assert!((x2[0] - r2[0]).abs() < 1e-15 && (x2[1] - r2[1]).abs() < 1e-15);
}

#[test]
fn operator_transpose_is_the_exact_transpose() {
    let mesh = mesh_for(0.3, 0.5, 0.5, 40, 10, &[]);
    let n = mesh.n_nodes();
    // a low vol near the bound forces some upwinded rows too
    let sigma: Vec<f64> = (0..n)
        .map(|j| if j % 7 == 0 { 0.02 } else { 0.25 })
        .collect();
    let (mut lo, mut di, mut up, mut uw) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0u8; n]);
    let count = assemble_operator(
        &sigma, 0.05, 0.0, mesh.dx, &mut lo, &mut di, &mut up, &mut uw,
    );
    assert!(count > 0, "the test needs upwinded rows, got {count}");
    let u: Vec<f64> = (0..n).map(|j| (j as f64 * 0.37).sin()).collect();
    let v: Vec<f64> = (0..n).map(|j| (j as f64 * 0.11).cos() + 0.3).collect();
    let (mut lu, mut ltv) = (vec![0.0; n], vec![0.0; n]);
    apply_operator(&u, &lo, &di, &up, 0.7, &mut lu);
    apply_operator_transpose(&v, &lo, &di, &up, 0.7, &mut ltv);
    let lhs: f64 = lu.iter().zip(&v).map(|(a, b)| a * b).sum();
    let rhs: f64 = u.iter().zip(&ltv).map(|(a, b)| a * b).sum();
    assert!(
        (lhs - rhs).abs() < 1e-9 * lhs.abs().max(1.0),
        "<Bu, v> = {lhs} vs <u, B^T v> = {rhs}"
    );
    // row sums of L are -r on every row (L 1 = -r 1), edge rows included
    for j in 0..n {
        let s = lo[j] + di[j] + up[j];
        assert!((s + 0.05).abs() < 1e-9, "row sum {j}: {s}");
    }
}

#[test]
fn european_prices_match_black_scholes_across_tenors_and_strikes() {
    // The spec rule max(1e-4 relative, 1e-6 S0) holds on the fine reference
    // mesh (N_X_FINE, n_t_fine). On the working mesh (400 nodes over the
    // capture-wide half-width, n_t_default) the measured worst errors are
    // 1.8e-4 relative (1w ATM: the eight implicit half-steps are first
    // order) and 5.8e-6 S0 absolute (1m 10%-OTM call: second-order spatial
    // error of an off-node kink, 0.4 bp of implied vol); both shrink 4x per
    // doubling of the resolution. Same-mesh differences (sigma^A - sigma^E)
    // cancel these to leading order.
    let (r, q, sigma) = (0.04, 0.02, 0.25);
    let mut ws = Workspace::new();
    for &t in &[1.0 / 52.0, 1.0 / 12.0, 1.0] {
        for &(n_x, n_t, rel, abs) in &[
            (400usize, grid::n_t_default(t), 2.5e-4, 8e-6 * S0),
            (grid::N_X_FINE, grid::n_t_fine(t), 1e-4, 1e-6 * S0),
        ] {
            let mesh = mesh_for(sigma, 1.0, t, n_x, n_t, &[]);
            let market = MarketSlice::flat(&mesh, S0, r, q, &[]);
            let vol = flat_field(&mesh, sigma);
            for &m in &[0.8, 0.9, 1.0, 1.1, 1.2] {
                let k = m * S0;
                for right in [PutOrCall::Put, PutOrCall::Call] {
                    let spec = QuoteSpec::new(k, t, right);
                    let price =
                        price_only(&mesh, &spec, &vol, &market, Mode::European, &mut ws).unwrap();
                    let bs = bs_price(S0, k, r, q, sigma, t, right);
                    let tol = (rel * bs.abs()).max(abs);
                    assert!(
                        (price - bs).abs() < tol,
                        "n_x = {n_x} n_t = {n_t} T = {t:.4} K = {k} {right:?}: pde {price} vs bs {bs} (diff {:.2e}, tol {tol:.2e})",
                        price - bs
                    );
                    if n_x == 400 {
                        // the dyn path and the retained path agree bit-for-bit
                        let dyn_price = price_only_dyn(
                            &mesh,
                            &spec,
                            &FlatVol(sigma),
                            &market,
                            Mode::European,
                            &mut ws,
                        )
                        .unwrap();
                        assert_eq!(dyn_price, price, "dyn path");
                        let sol = solve_backward(
                            &mesh,
                            &spec,
                            &vol,
                            &market,
                            Mode::European,
                            true,
                            &mut ws,
                        )
                        .unwrap();
                        assert_eq!(sol.price, price, "retained path");
                        assert_eq!(sol.penalty_iterations, mesh.n_steps());
                        assert_eq!(sol.upwinded_rows, 0, "no upwinding at sigma = 0.25");
                    }
                }
            }
        }
    }
}

#[test]
fn american_put_matches_forsyth_vetzal_and_a_crr_tree() {
    // Forsyth & Vetzal (2002), Table: K = 100, r = 0.10, sigma = 0.80, T = 0.25,
    // American put at S = 100; converged value 14.67882.
    let (k, r, sigma, t) = (100.0, 0.10, 0.80, 0.25);
    let l = half_width(sigma, t, 0.1, S0, 0.7 * S0, 1.3 * S0);
    let mesh = Mesh::new(S0, l, 800, t, 400, &[]).unwrap();
    let market = MarketSlice::flat(&mesh, S0, r, 0.0, &[]);
    let vol = flat_field(&mesh, sigma);
    let spec = QuoteSpec::new(k, t, PutOrCall::Put);
    let mut ws = Workspace::new();
    let sol = solve_backward(&mesh, &spec, &vol, &market, Mode::american(), true, &mut ws).unwrap();
    let published = 14.67882;
    assert!(
        (sol.price - published).abs() < 1e-3,
        "FV 2002 put: {} vs {published} (diff {:.2e})",
        sol.price,
        sol.price - published
    );
    let crr = crr_american(S0, k, r, 0.0, sigma, t, PutOrCall::Put, 4000);
    assert!(
        (sol.price - crr).abs() < 2e-4 * S0,
        "vs CRR(4000) {crr}: {} (diff {:.2e})",
        sol.price,
        sol.price - crr
    );
    assert_eq!(
        sol.penalty_inconsistent_steps, 0,
        "every step's active set is consistent"
    );
    assert!(
        sol.max_penalty_iterations_in_a_step <= 5,
        "{} penalty iterations in a step",
        sol.max_penalty_iterations_in_a_step
    );
    let european = bs_price(S0, k, r, 0.0, sigma, t, PutOrCall::Put);
    assert!(sol.price > european, "early-exercise premium is positive");
    // the active set is a lower half-line at every step (put) inside the
    // payoff-positive region (a far-OTM node may sit at u = -1e-6 < psi = 0
    // through the linear boundary row and count as active; that is not
    // exercise)
    let k_pos = mesh.x.partition_point(|&x| x.exp() < k);
    for step in 0..mesh.n_steps() {
        let row = &sol.active_row(step)[..k_pos];
        let first_inactive = row.iter().position(|&p| p == 0).unwrap();
        assert!(
            row[first_inactive..].iter().all(|&p| p == 0),
            "connected exercise region at step {step}"
        );
        assert!(
            first_inactive > 0,
            "exercise region is non-empty at step {step}"
        );
    }
    // the frozen re-solve reproduces the price exactly
    let frozen = price_frozen(&mesh, &spec, &vol, &market, &sol, &mut ws).unwrap();
    assert!(
        (frozen - sol.price).abs() < 1e-12,
        "frozen re-solve {frozen} vs {}",
        sol.price
    );
    // the solution buffers can be reused with solve_into
    let mut sol2 = sol.clone();
    solve_into(
        &mesh,
        &spec,
        &vol,
        &market,
        Mode::american(),
        true,
        &mut ws,
        &mut sol2,
    )
    .unwrap();
    assert_eq!(sol2.price, sol.price);
    assert!(
        sol.boundary_gamma_check.is_some(),
        "boundary gamma check is filled for a retained American solve"
    );
}

#[test]
fn american_put_converges_at_second_order_under_refinement() {
    let (k, r, sigma, t) = (100.0, 0.05, 0.30, 0.25);
    let l = half_width(sigma, t, 0.05, S0, 0.7 * S0, 1.3 * S0);
    let spec = QuoteSpec::new(k, t, PutOrCall::Put);
    let mut ws = Workspace::new();
    let mut price_at = |n_x: usize, n_t: usize| {
        let mesh = Mesh::new(S0, l, n_x, t, n_t, &[]).unwrap();
        let market = MarketSlice::flat(&mesh, S0, r, 0.0, &[]);
        let vol = flat_field(&mesh, sigma);
        price_only(&mesh, &spec, &vol, &market, Mode::american(), &mut ws).unwrap()
    };
    let reference = price_at(1600, 400);
    let e200 = (price_at(200, 50) - reference).abs();
    let e400 = (price_at(400, 100) - reference).abs();
    let e800 = (price_at(800, 200) - reference).abs();
    let ratio1 = e200 / e400;
    let ratio2 = e400 / e800;
    // pure second order against a reference with its own error gives ratios
    // (64 - 1)/(16 - 1) = 4.2 and (16 - 1)/(4 - 1) = 5
    assert!(
        (2.8..8.0).contains(&ratio1) && (2.8..8.0).contains(&ratio2),
        "errors {e200:.3e} {e400:.3e} {e800:.3e}: ratios {ratio1:.2} {ratio2:.2}"
    );
    assert!(e400 < 2e-5 * S0, "working-mesh error {e400:.3e}");
}

#[test]
fn penalty_error_is_first_order_in_one_over_rho() {
    let (k, r, sigma, t) = (105.0, 0.05, 0.25, 0.5);
    let mesh = mesh_for(sigma, t, t, 400, 100, &[]);
    let market = MarketSlice::flat(&mesh, S0, r, 0.0, &[]);
    let vol = flat_field(&mesh, sigma);
    let spec = QuoteSpec::new(k, t, PutOrCall::Put);
    let mut ws = Workspace::new();
    let prices: Vec<f64> = [1e5, 1e6, 1e7, 1e8]
        .iter()
        .map(|&rho| {
            price_only(&mesh, &spec, &vol, &market, Mode::American { rho }, &mut ws).unwrap()
        })
        .collect();
    // the penalized price increases with rho (0 <= u - u_rho <= r K / rho)
    for w in prices.windows(2) {
        assert!(w[1] >= w[0] - 1e-12, "monotone in rho: {prices:?}");
    }
    let d1 = prices[1] - prices[0];
    let d2 = prices[2] - prices[1];
    let d3 = prices[3] - prices[2];
    // successive differences of C/rho shrink by 10x
    assert!(
        (5.0..20.0).contains(&(d1 / d2)),
        "differences {d1:.3e} {d2:.3e} {d3:.3e}"
    );
    assert!(
        (5.0..20.0).contains(&(d2 / d3)),
        "differences {d1:.3e} {d2:.3e} {d3:.3e}"
    );
    assert!(d1 < r * k / 1e5, "penalty error bound r K / rho: {d1:.3e}");
}

#[test]
fn european_put_call_parity_holds_across_a_cash_dividend() {
    let (r, t, t_ex, delta) = (0.04, 0.5, 0.2037, 1.5);
    let sigma = 0.25;
    let mut ws = Workspace::new();
    for &n_x in &[400usize, 800] {
        let mesh = mesh_for(sigma, t, t, n_x, 125, &[(t_ex, delta)]);
        let market = MarketSlice::flat(&mesh, S0, r, 0.0, &[(t_ex, delta)]);
        let vol = flat_field(&mesh, sigma);
        for &k in &[90.0, 100.0, 110.0] {
            let c = price_only(
                &mesh,
                &QuoteSpec::new(k, t, PutOrCall::Call),
                &vol,
                &market,
                Mode::European,
                &mut ws,
            )
            .unwrap();
            let p = price_only(
                &mesh,
                &QuoteSpec::new(k, t, PutOrCall::Put),
                &vol,
                &market,
                Mode::European,
                &mut ws,
            )
            .unwrap();
            let parity = S0 - delta * (-r * t_ex).exp() - k * (-r * t).exp();
            // the cell-averaged terminal layer carries e^x (1 + dx^2/24) and the
            // linear-in-x jump interpolation of e^x has error <= dx^2/8 e^x, so
            // parity holds up to those two O(dx^2) biases of the spot term
            // (measured: 2.3e-4 at n_x = 400, 5.4e-5 at 800, 1.1e-5 at 1600)
            let bias = S0 * mesh.dx * mesh.dx * (1.0 / 24.0 + 1.0 / 8.0);
            let tol = 1e-6 * S0 + bias;
            assert!(
                ((c - p) - parity).abs() < tol,
                "n_x = {n_x} K = {k}: C - P = {} vs {parity} (diff {:.2e}, tol {tol:.2e})",
                c - p,
                (c - p) - parity
            );
        }
        // the mesh has the ex-date node and the jump is applied there
        assert_eq!(mesh.div_steps.len(), 1);
        let (m, amt) = mesh.div_steps[0];
        assert!((mesh.t[m] - t_ex).abs() < 1e-12 && amt == delta);
    }
}

#[test]
fn american_call_with_a_cash_dividend_matches_the_bermudan_reference_and_has_a_mixed_mask() {
    let (k, r, sigma, t, t_ex, delta) = (90.0, 0.04, 0.25, 0.5, 0.2, 1.5);
    let mesh = mesh_for(sigma, t, t, 400, 125, &[(t_ex, delta)]);
    let market = MarketSlice::flat(&mesh, S0, r, 0.0, &[(t_ex, delta)]);
    let vol = flat_field(&mesh, sigma);
    let spec = QuoteSpec::new(k, t, PutOrCall::Call);
    let mut ws = Workspace::new();
    let american =
        solve_backward(&mesh, &spec, &vol, &market, Mode::american(), true, &mut ws).unwrap();
    let european = price_only(&mesh, &spec, &vol, &market, Mode::European, &mut ws).unwrap();
    assert!(
        american.price > european + 1e-3,
        "early exercise before the ex-date is worth something: {} vs {european}",
        american.price
    );
    // Bermudan reference: with q = 0 the call is exercised only just before
    // the ex-date, so V = e^{-r t_ex} E[max(S - K, C_BS(S - delta, T - t_ex))]
    let tau = t - t_ex;
    let integrand = |z: f64| {
        let s = S0 * ((r - 0.5 * sigma * sigma) * t_ex + sigma * t_ex.sqrt() * z).exp();
        let hold = if s - delta > 0.0 {
            bs_price(s - delta, k, r, 0.0, sigma, tau, PutOrCall::Call)
        } else {
            0.0
        };
        (s - k).max(hold) * norm_pdf(z)
    };
    let reference = (-r * t_ex).exp() * simpson(&integrand, -9.0, 9.0, 20000);
    assert!(
        (american.price - reference).abs() < 2e-4 * S0,
        "American dividend call {} vs Bermudan reference {reference} (diff {:.2e})",
        american.price,
        american.price - reference
    );
    // the mask at the ex-date has both exercised (0) and kept (1) nodes
    assert_eq!(american.n_jumps, 1);
    let mask = american.jump_mask(0);
    let kept = mask.iter().filter(|&&m| m == 1).count();
    let exercised = mask.len() - kept;
    assert!(
        kept > 0 && exercised > 0,
        "mask kept {kept} exercised {exercised}"
    );
    // exercised nodes are the high-spot ones
    let first_ex = mask.iter().position(|&m| m == 0).unwrap();
    assert!(
        mask[first_ex..].iter().all(|&m| m == 0),
        "exercise region at the ex-date is an upper half-line"
    );
    // the post-jump layer equals the payoff there and the interpolated value elsewhere
    let (m, _) = mesh.div_steps[0];
    let pre = american.layer(m);
    let post = american.jump_post_layer(0);
    let mut interp = vec![0.0; mesh.n_nodes()];
    mesh.div_stencils[0].apply(pre, &mut interp);
    for j in 0..mesh.n_nodes() {
        let psi = spec.payoff(mesh.x[j].exp());
        let expect = if mask[j] == 1 { interp[j] } else { psi };
        assert!((post[j] - expect).abs() < 1e-12, "post-jump node {j}");
    }
    // European mode keeps the whole interpolated layer
    let euro = solve_backward(&mesh, &spec, &vol, &market, Mode::European, true, &mut ws).unwrap();
    assert!(euro.jump_mask(0).iter().all(|&m| m == 1));
    // frozen re-solve reproduces the price
    let frozen = price_frozen(&mesh, &spec, &vol, &market, &american, &mut ws).unwrap();
    assert!((frozen - american.price).abs() < 1e-12);
}

#[test]
fn time_dependent_volatility_prices_at_the_rms_vol() {
    let (k, r, q, t) = (100.0, 0.03, 0.01, 1.0);
    let mesh = mesh_for(0.3, t, t, 400, 250, &[]);
    let market = MarketSlice::flat(&mesh, S0, r, q, &[]);
    let sig_t = |t: f64| 0.15 + 0.15 * (-t / 0.25).exp();
    let field = CallbackVol::new(move |_x, t| sig_t(t));
    let vol = mesh.node_field(&field);
    // the discrete RMS over the step mid-times the solver actually uses
    let var: f64 = mesh
        .t_mid
        .iter()
        .zip(&mesh.dt)
        .map(|(&tm, &dt)| sig_t(tm).powi(2) * dt)
        .sum();
    let rms = (var / t).sqrt();
    let flat = flat_field(&mesh, rms);
    let mut ws = Workspace::new();
    for right in [PutOrCall::Put, PutOrCall::Call] {
        let spec = QuoteSpec::new(k, t, right);
        let p_t = price_only(&mesh, &spec, &vol, &market, Mode::European, &mut ws).unwrap();
        let p_flat = price_only(&mesh, &spec, &flat, &market, Mode::European, &mut ws).unwrap();
        // same-mesh difference: O(dt^2) from the time-varying coefficient
        // (measured 9.7e-6 relative at 250 steps, 2.5e-6 at 500, 6e-7 at 1000)
        assert!(
            (p_t - p_flat).abs() < 2e-5 * p_flat,
            "{right:?}: sigma(t) {p_t} vs flat rms {p_flat} on the same mesh (diff {:.2e})",
            p_t - p_flat
        );
        let bs = bs_price(S0, k, r, q, rms, t, right);
        assert!(
            (p_t - bs).abs() < tol_of(bs),
            "{right:?}: {p_t} vs BS({rms}) = {bs}"
        );
    }
}

#[test]
fn flat_vol_on_a_sloped_curve_inverts_to_the_flat_vol_with_effective_rates() {
    let sigma = 0.2;
    let r_inst = |t: f64| 0.02 + 0.03 * t; // instantaneous forward rate
    let df = |t: f64| (-(0.02 * t + 0.015 * t * t)).exp();
    let q_fn = |_t: f64| 0.01;
    let mut ws = Workspace::new();
    let _ = r_inst;
    for &t in &[1.0 / 12.0, 0.25, 1.0] {
        let mesh = mesh_for(sigma, 1.0, t, 400, grid::n_t_default(t), &[]);
        let market = MarketSlice::from_discount_factors(&mesh, S0, &df, &q_fn, &[]);
        let (r_eff, q_eff, fwd, d) = effective_rates(&mesh, &market);
        assert!((d - df(t)).abs() < 1e-14, "discount factor");
        assert!(
            (fwd - S0 * (0.02 * t + 0.015 * t * t - 0.01 * t).exp()).abs() < 1e-10,
            "forward"
        );
        let flat_market = MarketSlice::flat(&mesh, S0, r_eff, q_eff, &[]);
        let vol = flat_field(&mesh, sigma);
        for &m in &[0.8, 0.9, 1.0, 1.1, 1.2] {
            let k = m * S0;
            for right in [PutOrCall::Put, PutOrCall::Call] {
                let spec = QuoteSpec::new(k, t, right);
                let p_curve =
                    price_only(&mesh, &spec, &vol, &market, Mode::European, &mut ws).unwrap();
                let p_flat =
                    price_only(&mesh, &spec, &vol, &flat_market, Mode::European, &mut ws).unwrap();
                let iv_curve =
                    implied_vol_from_price(S0, k, r_eff, q_eff, t, p_curve, right).unwrap();
                let iv_flat =
                    implied_vol_from_price(S0, k, r_eff, q_eff, t, p_flat, right).unwrap();
                // the sloped-curve price inverted with (r_eff, q_eff) agrees with
                // the flat-rate price on the same mesh: the effective rates are
                // exact, only the mesh error remains
                assert!(
                    (iv_curve - iv_flat).abs() < 1e-6,
                    "T = {t:.4} K = {k} {right:?}: iv(curve) {iv_curve} vs iv(flat r_eff) {iv_flat}"
                );
                // the remaining mesh error in vol units on the working mesh, for
                // the OTM right (the ITM right's vega is too small for a
                // meaningful inversion, as in the pipeline): measured <= 9.2e-4
                // (1m, 20% OTM), <= 1.8e-4 within 10% of the money at 1m,
                // <= 8e-5 at 3m, <= 5e-6 at 1y
                let otm =
                    (k >= S0 && right == PutOrCall::Call) || (k <= S0 && right == PutOrCall::Put);
                let mesh_tol = if t < 0.2 {
                    1e-3
                } else if t < 0.9 {
                    2e-4
                } else {
                    2e-5
                };
                assert!(
                    !otm || (iv_curve - sigma).abs() < mesh_tol,
                    "T = {t:.4} K = {k} {right:?}: iv {iv_curve} vs {sigma} (mesh error {:.2e})",
                    iv_curve - sigma
                );
            }
        }
    }
}

#[test]
fn boundary_dollar_gamma_satisfies_the_smooth_fit_identity_on_the_fine_mesh() {
    let (k, r, q, sigma, t) = (100.0, 0.05, 0.0, 0.25, 0.5);
    let l = half_width(sigma, t, 0.05, S0, 0.7 * S0, 1.3 * S0);
    let mesh = Mesh::new(S0, l, 1600, t, 400, &[]).unwrap();
    let market = MarketSlice::flat(&mesh, S0, r, q, &[]);
    let vol = flat_field(&mesh, sigma);
    let spec = QuoteSpec::new(k, t, PutOrCall::Put);
    let mut ws = Workspace::new();
    let sol = solve_backward(&mesh, &spec, &vol, &market, Mode::american(), true, &mut ws).unwrap();
    let n = mesh.n_steps();
    for &level in &[n / 4, n / 2, 3 * n / 4] {
        let bg = boundary_dollar_gamma(&mesh, &spec, &sol, &market, level, sigma)
            .expect("free boundary inside the grid");
        assert!(
            bg.relative_deviation() < 0.02,
            "level {level} (t = {:.3}): b = {:.4}, discrete {:.4} vs analytic {:.4} ({:.2}%)",
            mesh.t[level],
            bg.boundary,
            bg.discrete,
            bg.analytic,
            100.0 * bg.relative_deviation()
        );
        assert!(
            bg.boundary < k && bg.boundary > 0.5 * k,
            "put boundary below the strike: {}",
            bg.boundary
        );
    }
    let check = sol.boundary_gamma_check.unwrap();
    assert!(check < 0.02, "stored check {check}");
    // a call with q > r has a boundary above the strike and satisfies the mirrored identity
    let market_c = MarketSlice::flat(&mesh, S0, 0.02, 0.06, &[]);
    let spec_c = QuoteSpec::new(k, t, PutOrCall::Call);
    let sol_c = solve_backward(
        &mesh,
        &spec_c,
        &vol,
        &market_c,
        Mode::american(),
        true,
        &mut ws,
    )
    .unwrap();
    let bg = boundary_dollar_gamma(&mesh, &spec_c, &sol_c, &market_c, n / 2, sigma)
        .expect("call boundary");
    assert!(
        bg.boundary > k,
        "call boundary above the strike: {}",
        bg.boundary
    );
    assert!(
        bg.relative_deviation() < 0.02,
        "call: discrete {} vs analytic {}",
        bg.discrete,
        bg.analytic
    );
    // dollar gamma at x0 of a European layer matches the closed form
    let euro = solve_backward(&mesh, &spec, &vol, &market, Mode::European, true, &mut ws).unwrap();
    let d1 = ((S0 / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let gamma_bs = (-q * t).exp() * norm_pdf(d1) / (S0 * sigma * t.sqrt());
    let dg = dollar_gamma_at_x0(&mesh, euro.final_layer());
    assert!(
        (dg - S0 * S0 * gamma_bs).abs() < 1e-3 * S0 * S0 * gamma_bs,
        "dollar gamma {dg} vs {}",
        S0 * S0 * gamma_bs
    );
    let _ = norm_cdf(0.0);
}

#[test]
fn invalid_inputs_are_rejected() {
    let mesh = mesh_for(0.25, 0.5, 0.5, 40, 10, &[]);
    let market = MarketSlice::flat(&mesh, S0, 0.04, 0.0, &[]);
    let vol = flat_field(&mesh, 0.25);
    let mut ws = Workspace::new();
    assert!(
        price_only(
            &mesh,
            &QuoteSpec::new(100.0, 0.25, PutOrCall::Put),
            &vol,
            &market,
            Mode::European,
            &mut ws
        )
        .is_err(),
        "expiry mismatch"
    );
    assert!(
        price_only(
            &mesh,
            &QuoteSpec::new(-1.0, 0.5, PutOrCall::Put),
            &vol,
            &market,
            Mode::European,
            &mut ws
        )
        .is_err(),
        "negative strike"
    );
    assert!(
        price_only(
            &mesh,
            &QuoteSpec::new(100.0, 0.5, PutOrCall::Put),
            &vol,
            &market,
            Mode::American { rho: 0.0 },
            &mut ws
        )
        .is_err(),
        "zero rho"
    );
    let other = mesh_for(0.25, 0.5, 0.5, 40, 12, &[]);
    let bad_vol = flat_field(&other, 0.25);
    assert!(
        price_only(
            &mesh,
            &QuoteSpec::new(100.0, 0.5, PutOrCall::Put),
            &bad_vol,
            &market,
            Mode::European,
            &mut ws
        )
        .is_err(),
        "node field mismatch"
    );
    let bad_market = MarketSlice::flat(&other, S0, 0.04, 0.0, &[]);
    assert!(
        price_only(
            &mesh,
            &QuoteSpec::new(100.0, 0.5, PutOrCall::Put),
            &vol,
            &bad_market,
            Mode::European,
            &mut ws
        )
        .is_err(),
        "market mismatch"
    );
    let unretained = solve_backward(
        &mesh,
        &QuoteSpec::new(100.0, 0.5, PutOrCall::Put),
        &vol,
        &market,
        Mode::american(),
        false,
        &mut ws,
    )
    .unwrap();
    assert!(!unretained.retained());
    assert!(
        price_frozen(
            &mesh,
            &QuoteSpec::new(100.0, 0.5, PutOrCall::Put),
            &vol,
            &market,
            &unretained,
            &mut ws
        )
        .is_err(),
        "frozen needs retention"
    );
}
