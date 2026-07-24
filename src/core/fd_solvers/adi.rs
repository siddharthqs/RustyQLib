//! ADI (alternating direction implicit) time steppers for 1-D, 2-D and
//! 3-D parabolic PDEs `u_t = sum_k A_k u + A_0 u`, where each `A_k` is a
//! per-axis tridiagonal [`AxisOperator`] and `A_0` is an optional
//! explicitly-treated part (typically the mixed derivatives of correlated
//! factors, e.g. the `rho S v u_Sv` term of Heston).
//!
//! Two schemes, the standard choices in finance:
//!
//! - [`douglas_step`] (Do): one explicit predictor plus one implicit
//!   correction per axis. First-order in time when a mixed term is
//!   present, second-order (Crank-Nicolson-like at `theta = 1/2`)
//!   without one. With a single axis and no mixed term it reduces
//!   exactly to the 1-D theta scheme.
//! - [`hundsdorfer_verwer_step`] (HV): Douglas plus a corrector sweep;
//!   second-order in time including the mixed term, at roughly twice the
//!   cost. The scheme of choice for Heston-type problems.
//!
//! Every stage's implicit solve is a line-by-line Thomas pass, so a step
//! is O(nodes) regardless of dimension. Boundary rows of the operators
//! encode the boundary conditions (all-zero row = value held fixed).

use super::axis_operator::{AxisOperator, TensorGrid};

/// `sum_k A_k u + A_0 u`: the full spatial operator applied explicitly.
fn apply_full(
    grid: &TensorGrid,
    ops: &[AxisOperator],
    mixed: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    u: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; u.len()];
    for op in ops {
        for (o, v) in out.iter_mut().zip(op.apply(grid, u)) {
            *o += v;
        }
    }
    if let Some(a0) = mixed {
        for (o, v) in out.iter_mut().zip(a0(u)) {
            *o += v;
        }
    }
    out
}

/// One Douglas ADI step of size `dt` from `u`; `theta` is the implicit
/// weight (1/2 is standard, 1 fully implicit stages).
///
/// ```text
/// Y_0 = u + dt (A u + A_0 u)
/// (I - theta dt A_k) Y_k = Y_{k-1} - theta dt A_k u      k = 1..d
/// u_next = Y_d
/// ```
pub fn douglas_step(
    grid: &TensorGrid,
    ops: &[AxisOperator],
    mixed: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    u: &[f64],
    dt: f64,
    theta: f64,
) -> Vec<f64> {
    assert!(!ops.is_empty(), "at least one axis operator is required");
    assert_eq!(u.len(), grid.len());
    let f_u = apply_full(grid, ops, mixed, u);
    let mut y: Vec<f64> = u.iter().zip(&f_u).map(|(ui, fi)| ui + dt * fi).collect();
    for op in ops {
        let a_u = op.apply(grid, u);
        for (yi, ai) in y.iter_mut().zip(&a_u) {
            *yi -= theta * dt * ai;
        }
        y = op.solve_shifted(grid, theta * dt, &y);
    }
    y
}

/// One Hundsdorfer-Verwer ADI step of size `dt` from `u`: a Douglas
/// predictor followed by a corrector sweep with weight `mu` (1/2 is the
/// standard choice giving second order with mixed terms).
///
/// ```text
/// Y_0 = u + dt F(u)
/// (I - theta dt A_k) Y_k     = Y_{k-1}     - theta dt A_k u      k = 1..d
/// Yt_0 = Y_0 + mu dt (F(Y_d) - F(u))
/// (I - theta dt A_k) Yt_k    = Yt_{k-1}    - theta dt A_k Y_d    k = 1..d
/// u_next = Yt_d
/// ```
pub fn hundsdorfer_verwer_step(
    grid: &TensorGrid,
    ops: &[AxisOperator],
    mixed: Option<&dyn Fn(&[f64]) -> Vec<f64>>,
    u: &[f64],
    dt: f64,
    theta: f64,
    mu: f64,
) -> Vec<f64> {
    assert!(!ops.is_empty(), "at least one axis operator is required");
    assert_eq!(u.len(), grid.len());
    let f_u = apply_full(grid, ops, mixed, u);
    let y0: Vec<f64> = u.iter().zip(&f_u).map(|(ui, fi)| ui + dt * fi).collect();

    // predictor (Douglas) sweep
    let mut y = y0.clone();
    for op in ops {
        let a_u = op.apply(grid, u);
        for (yi, ai) in y.iter_mut().zip(&a_u) {
            *yi -= theta * dt * ai;
        }
        y = op.solve_shifted(grid, theta * dt, &y);
    }

    // corrector sweep around the predictor solution
    let f_y = apply_full(grid, ops, mixed, &y);
    let mut yt: Vec<f64> = y0
        .iter()
        .zip(f_y.iter().zip(&f_u))
        .map(|(y0i, (fyi, fui))| y0i + mu * dt * (fyi - fui))
        .collect();
    for op in ops {
        let a_y = op.apply(grid, &y);
        for (yi, ai) in yt.iter_mut().zip(&a_y) {
            *yi -= theta * dt * ai;
        }
        yt = op.solve_shifted(grid, theta * dt, &yt);
    }
    yt
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Dirichlet-zero Laplacian along `axis` with spacing `h`.
    fn laplacian(grid: &TensorGrid, axis: usize, h: f64) -> AxisOperator {
        let mut op = AxisOperator::zero(grid, axis);
        let stride = grid.strides()[axis];
        let n = grid.dims()[axis];
        for i in 0..grid.len() {
            let j = (i / stride) % n;
            if j > 0 && j < n - 1 {
                op.sub[i] = 1.0 / (h * h);
                op.diag[i] = -2.0 / (h * h);
                op.sup[i] = 1.0 / (h * h);
            }
        }
        op
    }

    /// Product-of-sines initial data, the heat equation's eigenfunction.
    fn sine_product(grid: &TensorGrid, h: f64) -> Vec<f64> {
        let dims = grid.dims().to_vec();
        let strides = grid.strides();
        (0..grid.len())
            .map(|i| {
                dims.iter()
                    .zip(&strides)
                    .map(|(&n, &s)| (PI * ((i / s) % n) as f64 * h).sin())
                    .product()
            })
            .collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// Heat equation decay test in `ndim` dimensions: u0 = prod sin(pi x_k)
    /// decays by e^{-ndim pi^2 t}.
    fn heat_decay(ndim: usize, n: usize, steps: usize, dt: f64, hv: bool) -> f64 {
        let h = 1.0 / (n - 1) as f64;
        let grid = TensorGrid::new(&vec![n; ndim]);
        let ops: Vec<AxisOperator> = (0..ndim).map(|k| laplacian(&grid, k, h)).collect();
        let mut u = sine_product(&grid, h);
        for _ in 0..steps {
            u = if hv {
                hundsdorfer_verwer_step(&grid, &ops, None, &u, dt, 0.5, 0.5)
            } else {
                douglas_step(&grid, &ops, None, &u, dt, 0.5)
            };
        }
        let exact = (-(ndim as f64) * PI * PI * (steps as f64 * dt)).exp();
        let u0 = sine_product(&grid, h);
        // relative error at the grid maximum of the exact solution
        let (imax, _) = u0
            .iter()
            .enumerate()
            .fold((0, 0.0), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        (u[imax] / (exact * u0[imax]) - 1.0).abs()
    }

    #[test]
    fn one_dimension_reduces_to_crank_nicolson_heat_solution() {
        // 1 axis, no mixed term: Douglas = theta scheme
        let err = heat_decay(1, 41, 200, 5e-4, false);
        assert!(err < 5e-3, "1-D heat relative error {err}");
    }

    #[test]
    fn douglas_solves_the_2d_heat_equation() {
        let err = heat_decay(2, 21, 100, 1e-3, false);
        assert!(err < 1e-2, "2-D heat relative error {err}");
    }

    #[test]
    fn douglas_and_hv_solve_the_3d_heat_equation() {
        let do_err = heat_decay(3, 11, 50, 1e-3, false);
        let hv_err = heat_decay(3, 11, 50, 1e-3, true);
        assert!(do_err < 3e-2, "3-D Douglas relative error {do_err}");
        assert!(hv_err < 3e-2, "3-D HV relative error {hv_err}");
    }

    #[test]
    fn mixed_derivative_term_matches_an_explicit_reference() {
        // u_t = u_xx + u_yy + u_xy on a coarse grid: ADI with the mixed
        // term explicit must track a tiny-step forward-Euler reference of
        // the same semi-discrete system
        let n = 9;
        let h = 1.0 / (n - 1) as f64;
        let grid = TensorGrid::new(&[n, n]);
        let ops = [laplacian(&grid, 0, h), laplacian(&grid, 1, h)];
        let (sx, sy) = (grid.strides()[0], grid.strides()[1]);
        let dims = grid.dims().to_vec();
        let mixed = move |u: &[f64]| -> Vec<f64> {
            (0..u.len())
                .map(|i| {
                    let (jx, jy) = ((i / sx) % dims[0], (i / sy) % dims[1]);
                    if jx == 0 || jx == dims[0] - 1 || jy == 0 || jy == dims[1] - 1 {
                        0.0
                    } else {
                        (u[i + sx + sy] - u[i + sx - sy] - u[i - sx + sy] + u[i - sx - sy])
                            / (4.0 * h * h)
                    }
                })
                .collect()
        };
        let u0 = sine_product(&grid, h);
        let t_end: f64 = 0.01;

        // explicit reference with 1000x smaller steps
        let mut reference = u0.clone();
        let dt_ref = 1e-5;
        for _ in 0..(t_end / dt_ref).round() as usize {
            let f = apply_full(&grid, &ops, Some(&mixed), &reference);
            for (r, fi) in reference.iter_mut().zip(&f) {
                *r += dt_ref * fi;
            }
        }

        let dt = 1e-3;
        let steps = (t_end / dt).round() as usize;
        let mut douglas = u0.clone();
        let mut hv = u0;
        for _ in 0..steps {
            douglas = douglas_step(&grid, &ops, Some(&mixed), &douglas, dt, 0.5);
            hv = hundsdorfer_verwer_step(&grid, &ops, Some(&mixed), &hv, dt, 0.5, 0.5);
        }
        assert!(max_abs_diff(&douglas, &reference) < 1e-2, "douglas vs reference");
        assert!(max_abs_diff(&hv, &reference) < 1e-2, "hv vs reference");

        // and the mixed term genuinely matters: dropping it moves the answer
        let mut no_mixed = sine_product(&grid, h);
        for _ in 0..steps {
            no_mixed = douglas_step(&grid, &ops, None, &no_mixed, dt, 0.5);
        }
        assert!(max_abs_diff(&no_mixed, &reference) > 1e-3, "mixed term had no effect");
    }
}
