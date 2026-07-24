//! C2 cubic spline interpolation with the three classical boundary
//! conditions.
//!
//! The spline is built on the knots' second derivatives `M_i`, solved
//! from the tridiagonal continuity equations plus two boundary rows:
//!
//! - **Natural**: `M = 0` at both ends (zero curvature — the default
//!   when nothing is known about the ends, slightly flattens there);
//! - **Clamped**: first derivatives prescribed at both ends (use when
//!   end slopes are known, e.g. from a model);
//! - **Not-a-Knot**: third-derivative continuity across the first and
//!   last interior knots — the best all-round accuracy without extra
//!   information; reproduces a single cubic polynomial exactly.
//!
//! Evaluation outside the knot range extrapolates linearly with the end
//! slope (curvature is not continued — safer for financial data).

use crate::core::optimization::numerics::solve_dense;
use crate::core::errors::RustyQLibError;

/// Boundary condition for [`CubicSpline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    Natural,
    /// Prescribed end slopes `(start, end)`.
    Clamped { start_slope: f64, end_slope: f64 },
    NotAKnot,
}

/// A cubic spline through `(xs, ys)` knots.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Second derivatives at the knots.
    m: Vec<f64>,
}

impl CubicSpline {
    /// Build a spline through the knots (`xs` strictly increasing,
    /// at least two points; Not-a-Knot needs at least four and falls
    /// back to Natural below that).
    pub fn new(xs: &[f64], ys: &[f64], bc: BoundaryCondition) -> Result<Self, RustyQLibError> {
        let n = xs.len();
        if n < 2 || ys.len() != n {
            return Err(RustyQLibError::invalid_input("cubic_spline", "need at least two knots with matching y values"));
        }
        if xs.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RustyQLibError::invalid_input("cubic_spline", "knots must be strictly increasing"));
        }
        if n == 2 {
            // a segment: linear, zero curvature
            return Ok(CubicSpline { xs: xs.to_vec(), ys: ys.to_vec(), m: vec![0.0; 2] });
        }
        let bc = match bc {
            BoundaryCondition::NotAKnot if n < 4 => BoundaryCondition::Natural,
            other => other,
        };

        // dense (n x n) system in the second derivatives M; knot counts
        // are small (pillars), so O(n^3) is irrelevant next to clarity
        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let mut a = vec![vec![0.0; n]; n];
        let mut b = vec![0.0; n];
        for i in 1..n - 1 {
            a[i][i - 1] = h[i - 1] / 6.0;
            a[i][i] = (h[i - 1] + h[i]) / 3.0;
            a[i][i + 1] = h[i] / 6.0;
            b[i] = (ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1];
        }
        match bc {
            BoundaryCondition::Natural => {
                a[0][0] = 1.0;
                a[n - 1][n - 1] = 1.0;
            }
            BoundaryCondition::Clamped { start_slope, end_slope } => {
                a[0][0] = h[0] / 3.0;
                a[0][1] = h[0] / 6.0;
                b[0] = (ys[1] - ys[0]) / h[0] - start_slope;
                a[n - 1][n - 2] = h[n - 2] / 6.0;
                a[n - 1][n - 1] = h[n - 2] / 3.0;
                b[n - 1] = end_slope - (ys[n - 1] - ys[n - 2]) / h[n - 2];
            }
            BoundaryCondition::NotAKnot => {
                // third-derivative continuity at the second and
                // second-to-last knots
                a[0][0] = h[1];
                a[0][1] = -(h[0] + h[1]);
                a[0][2] = h[0];
                a[n - 1][n - 3] = h[n - 2];
                a[n - 1][n - 2] = -(h[n - 3] + h[n - 2]);
                a[n - 1][n - 1] = h[n - 3];
            }
        }
        let m = solve_dense(&mut a, &mut b).ok_or(RustyQLibError::invalid_input("cubic_spline", "singular spline system"))?;
        Ok(CubicSpline { xs: xs.to_vec(), ys: ys.to_vec(), m })
    }

    fn segment(&self, x: f64) -> usize {
        let n = self.xs.len();
        self.xs[1..n - 1].partition_point(|&xi| xi < x)
    }

    /// Spline value at `x` (linear extrapolation beyond the knots).
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return self.ys[0] + self.derivative(self.xs[0]) * (x - self.xs[0]);
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1] + self.derivative(self.xs[n - 1]) * (x - self.xs[n - 1]);
        }
        let i = self.segment(x);
        let h = self.xs[i + 1] - self.xs[i];
        let (dl, dr) = (self.xs[i + 1] - x, x - self.xs[i]);
        self.m[i] * dl * dl * dl / (6.0 * h)
            + self.m[i + 1] * dr * dr * dr / (6.0 * h)
            + (self.ys[i] / h - self.m[i] * h / 6.0) * dl
            + (self.ys[i + 1] / h - self.m[i + 1] * h / 6.0) * dr
    }

    /// First derivative at `x` (constant beyond the knots).
    pub fn derivative(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let x = x.clamp(self.xs[0], self.xs[n - 1]);
        let i = self.segment(x).min(n - 2);
        let h = self.xs[i + 1] - self.xs[i];
        let (dl, dr) = (self.xs[i + 1] - x, x - self.xs[i]);
        -self.m[i] * dl * dl / (2.0 * h)
            + self.m[i + 1] * dr * dr / (2.0 * h)
            + (self.ys[i + 1] - self.ys[i]) / h
            - (self.m[i + 1] - self.m[i]) * h / 6.0
    }

    /// Second derivative at `x` (zero beyond the knots).
    pub fn second_derivative(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x < self.xs[0] || x > self.xs[n - 1] {
            return 0.0;
        }
        let i = self.segment(x).min(n - 2);
        let h = self.xs[i + 1] - self.xs[i];
        self.m[i] * (self.xs[i + 1] - x) / h + self.m[i + 1] * (x - self.xs[i]) / h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn knots() -> (Vec<f64>, Vec<f64>) {
        let xs: Vec<f64> = (0..7).map(|i| i as f64 * 0.5).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        (xs, ys)
    }

    #[test]
    fn all_boundary_conditions_interpolate_the_knots() {
        let (xs, ys) = knots();
        for bc in [
            BoundaryCondition::Natural,
            BoundaryCondition::Clamped { start_slope: 1.0, end_slope: 3.0_f64.cos() },
            BoundaryCondition::NotAKnot,
        ] {
            let s = CubicSpline::new(&xs, &ys, bc).unwrap();
            for (x, y) in xs.iter().zip(&ys) {
                assert!((s.eval(*x) - y).abs() < 1e-12, "{bc:?} at {x}");
            }
        }
    }

    #[test]
    fn not_a_knot_reproduces_a_cubic_exactly() {
        // the defining property: for data from one cubic polynomial the
        // not-a-knot spline IS that polynomial
        let p = |x: f64| 2.0 - x + 3.0 * x * x - 0.5 * x * x * x;
        let xs: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| p(x)).collect();
        let s = CubicSpline::new(&xs, &ys, BoundaryCondition::NotAKnot).unwrap();
        for i in 0..=50 {
            let x = i as f64 * 0.1;
            assert!((s.eval(x) - p(x)).abs() < 1e-9, "x = {x}");
        }
    }

    #[test]
    fn natural_ends_have_zero_curvature() {
        let (xs, ys) = knots();
        let s = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();
        assert!(s.second_derivative(xs[0]).abs() < 1e-10);
        assert!(s.second_derivative(*xs.last().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn clamped_ends_match_the_prescribed_slopes() {
        let (xs, ys) = knots();
        let bc = BoundaryCondition::Clamped { start_slope: 1.0, end_slope: 3.0_f64.cos() };
        let s = CubicSpline::new(&xs, &ys, bc).unwrap();
        assert!((s.derivative(0.0) - 1.0).abs() < 1e-10);
        assert!((s.derivative(3.0) - 3.0_f64.cos()).abs() < 1e-10);
        // clamping with the true sin slopes beats natural near the ends
        let natural = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();
        let x = 0.1;
        assert!((s.eval(x) - x.sin()).abs() < (natural.eval(x) - x.sin()).abs());
    }

    #[test]
    fn rejects_bad_knots() {
        assert!(CubicSpline::new(&[0.0, 0.0, 1.0], &[1.0, 2.0, 3.0], BoundaryCondition::Natural)
            .is_err());
        assert!(CubicSpline::new(&[0.0], &[1.0], BoundaryCondition::Natural).is_err());
    }
}
