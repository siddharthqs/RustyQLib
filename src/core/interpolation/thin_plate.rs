//! Bivariate thin-plate spline: the minimum-bending-energy surface
//! through scattered `(x, y, z)` points — the natural interpolant for an
//! irregular volatility quote grid (no rectangular layout required).
//!
//! `f(p) = a0 + a1 x + a2 y + sum_i w_i phi(|p - p_i|)` with
//! `phi(r) = r^2 ln r`, coefficients from the standard augmented linear
//! system. A `smoothing` parameter `>= 0` relaxes exact interpolation
//! toward a smoother least-squares surface (useful for noisy quotes).

use crate::core::optimization::numerics::solve_dense;
use crate::core::errors::RustyQLibError;

#[derive(Debug, Clone)]
pub struct ThinPlateSpline {
    centers: Vec<(f64, f64)>,
    w: Vec<f64>,
    affine: [f64; 3],
}

/// `phi(r) = r^2 ln r`, written on `r^2` to avoid the square root.
fn phi_sq(r_sq: f64) -> f64 {
    if r_sq <= 0.0 { 0.0 } else { 0.5 * r_sq * r_sq.ln() }
}

impl ThinPlateSpline {
    /// Fit to scattered points; `smoothing = 0` interpolates exactly.
    /// Needs at least three non-collinear points.
    pub fn new(points: &[(f64, f64, f64)], smoothing: f64) -> Result<Self, RustyQLibError> {
        let n = points.len();
        if n < 3 {
            return Err(RustyQLibError::invalid_input("thin_plate", "need at least three points"));
        }
        // augmented system [K + lambda I, P; P^T, 0] [w; a] = [z; 0]
        let dim = n + 3;
        let mut a = vec![vec![0.0; dim]; dim];
        let mut b = vec![0.0; dim];
        for i in 0..n {
            let (xi, yi, zi) = points[i];
            for j in 0..n {
                let (xj, yj, _) = points[j];
                let r_sq = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj);
                a[i][j] = phi_sq(r_sq);
            }
            a[i][i] += smoothing;
            a[i][n] = 1.0;
            a[i][n + 1] = xi;
            a[i][n + 2] = yi;
            a[n][i] = 1.0;
            a[n + 1][i] = xi;
            a[n + 2][i] = yi;
            b[i] = zi;
        }
        let sol = solve_dense(&mut a, &mut b)
            .ok_or(RustyQLibError::invalid_input("thin_plate", "thin-plate system is singular (collinear or duplicate points?)"))?;
        Ok(ThinPlateSpline {
            centers: points.iter().map(|&(x, y, _)| (x, y)).collect(),
            w: sol[..n].to_vec(),
            affine: [sol[n], sol[n + 1], sol[n + 2]],
        })
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let mut v = self.affine[0] + self.affine[1] * x + self.affine[2] * y;
        for (&(cx, cy), &wi) in self.centers.iter().zip(&self.w) {
            let r_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy);
            v += wi * phi_sq(r_sq);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scattered() -> Vec<(f64, f64, f64)> {
        // an irregular quote layout with a smooth smile-like value
        let f = |x: f64, y: f64| 0.2 + 0.05 * (x - 1.0) * (x - 1.0) + 0.02 * y;
        [
            (0.5, 0.25), (1.0, 0.25), (1.5, 0.25), (0.7, 1.0), (1.3, 1.0),
            (0.9, 2.0), (1.1, 2.0), (0.6, 1.5), (1.4, 0.6),
        ]
        .iter()
        .map(|&(x, y)| (x, y, f(x, y)))
        .collect()
    }

    #[test]
    fn interpolates_scattered_points_exactly() {
        let pts = scattered();
        let tps = ThinPlateSpline::new(&pts, 0.0).unwrap();
        for &(x, y, z) in &pts {
            assert!((tps.eval(x, y) - z).abs() < 1e-9, "({x}, {y})");
        }
    }

    #[test]
    fn reproduces_affine_surfaces_everywhere() {
        // an affine function has zero bending energy: the TPS must be it
        let g = |x: f64, y: f64| 1.0 + 2.0 * x - 3.0 * y;
        let pts: Vec<(f64, f64, f64)> =
            scattered().iter().map(|&(x, y, _)| (x, y, g(x, y))).collect();
        let tps = ThinPlateSpline::new(&pts, 0.0).unwrap();
        for i in 0..=20 {
            for j in 0..=20 {
                let (x, y) = (i as f64 * 0.15, j as f64 * 0.15);
                assert!((tps.eval(x, y) - g(x, y)).abs() < 1e-8, "({x}, {y})");
            }
        }
    }

    #[test]
    fn smoothing_relaxes_exact_interpolation() {
        let mut pts = scattered();
        pts[4].2 += 0.05; // a noisy quote
        let exact = ThinPlateSpline::new(&pts, 0.0).unwrap();
        let smooth = ThinPlateSpline::new(&pts, 0.1).unwrap();
        let (x, y, z) = pts[4];
        assert!((exact.eval(x, y) - z).abs() < 1e-9);
        // the smoothed surface pulls away from the noisy point
        assert!((smooth.eval(x, y) - z).abs() > 1e-3);
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(ThinPlateSpline::new(&[(0.0, 0.0, 1.0), (1.0, 1.0, 2.0)], 0.0).is_err());
        // collinear points make the affine part singular
        let collinear: Vec<(f64, f64, f64)> =
            (0..5).map(|i| (i as f64, 2.0 * i as f64, 1.0)).collect();
        assert!(ThinPlateSpline::new(&collinear, 0.0).is_err());
    }
}
