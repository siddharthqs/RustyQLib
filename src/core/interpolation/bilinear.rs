//! Bilinear interpolation on a rectangular grid — the baseline 2-D
//! scheme for strike x expiry volatility grids.

use super::linear::bracket;
use crate::core::errors::RustyQLibError;

/// A rectangular grid `z[i][j]` sampled at `(xs[i], ys[j])`, both axes
/// strictly increasing. Queries outside the grid are clamped (flat
/// extrapolation), matching the wing behavior of the 1-D smiles.
#[derive(Debug, Clone)]
pub struct BilinearGrid {
    xs: Vec<f64>,
    ys: Vec<f64>,
    z: Vec<Vec<f64>>,
}

impl BilinearGrid {
    pub fn new(xs: &[f64], ys: &[f64], z: &[Vec<f64>]) -> Result<Self, RustyQLibError> {
        if xs.len() < 2 || ys.len() < 2 {
            return Err(RustyQLibError::invalid_input("bilinear", "need at least a 2 x 2 grid"));
        }
        if xs.windows(2).any(|w| w[1] <= w[0]) || ys.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RustyQLibError::invalid_input("bilinear", "grid axes must be strictly increasing"));
        }
        if z.len() != xs.len() || z.iter().any(|row| row.len() != ys.len()) {
            return Err(RustyQLibError::invalid_input("bilinear", "z must be an xs.len() x ys.len() matrix"));
        }
        Ok(BilinearGrid { xs: xs.to_vec(), ys: ys.to_vec(), z: z.to_vec() })
    }

    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let (i, wx) = bracket(&self.xs, x);
        let (j, wy) = bracket(&self.ys, y);
        let z00 = self.z[i - 1][j - 1];
        let z01 = self.z[i - 1][j];
        let z10 = self.z[i][j - 1];
        let z11 = self.z[i][j];
        z00 * (1.0 - wx) * (1.0 - wy)
            + z01 * (1.0 - wx) * wy
            + z10 * wx * (1.0 - wy)
            + z11 * wx * wy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_bilinear_function_exactly() {
        // f = 2 + 3x + 4y + 5xy is in the bilinear span
        let f = |x: f64, y: f64| 2.0 + 3.0 * x + 4.0 * y + 5.0 * x * y;
        let xs = [0.0, 0.7, 1.5, 3.0];
        let ys = [-1.0, 0.5, 2.0];
        let z: Vec<Vec<f64>> =
            xs.iter().map(|&x| ys.iter().map(|&y| f(x, y)).collect()).collect();
        let grid = BilinearGrid::new(&xs, &ys, &z).unwrap();
        for i in 0..=30 {
            for j in 0..=30 {
                let (x, y) = (i as f64 * 0.1, -1.0 + j as f64 * 0.1);
                assert!((grid.eval(x, y) - f(x, y)).abs() < 1e-12, "({x}, {y})");
            }
        }
    }

    #[test]
    fn clamps_outside_the_grid() {
        let grid = BilinearGrid::new(
            &[0.0, 1.0],
            &[0.0, 1.0],
            &[vec![1.0, 2.0], vec![3.0, 4.0]],
        )
        .unwrap();
        assert_eq!(grid.eval(-5.0, -5.0), 1.0);
        assert_eq!(grid.eval(9.0, 9.0), 4.0);
        assert_eq!(grid.eval(0.5, -3.0), 2.0); // clamped in y only
    }
}
