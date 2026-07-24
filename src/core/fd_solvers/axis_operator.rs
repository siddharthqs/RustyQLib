//! Per-axis tridiagonal operators on tensor-product grids — the building
//! block for 1-D, 2-D and 3-D finite-difference schemes.
//!
//! A discretized diffusion operator splits into one tridiagonal operator
//! per spatial axis (plus an explicit mixed-derivative part). Each
//! [`AxisOperator`] stores per-node coefficient triples, so coefficients
//! may vary over the whole grid — exactly what local vol (1-D) or Heston
//! (2-D in spot x variance) discretizations produce. The ADI time
//! steppers in [`adi`](super::adi) consume these directly.

use super::tridiagonal::thomas_algorithm;

/// A tensor-product grid: node counts per axis, flattened row-major (the
/// last axis is contiguous). Designed and tested for 1 to 3 dimensions.
#[derive(Debug, Clone)]
pub struct TensorGrid {
    dims: Vec<usize>,
}

impl TensorGrid {
    pub fn new(dims: &[usize]) -> Self {
        assert!(!dims.is_empty() && dims.len() <= 3, "1 to 3 axes supported");
        assert!(dims.iter().all(|&n| n >= 1), "every axis needs at least one node");
        Self { dims: dims.to_vec() }
    }

    /// Total number of nodes.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        false // dims are validated >= 1
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Flat-index stride of each axis (row-major).
    pub fn strides(&self) -> Vec<usize> {
        let mut s = vec![1; self.dims.len()];
        for k in (0..self.dims.len().saturating_sub(1)).rev() {
            s[k] = s[k + 1] * self.dims[k + 1];
        }
        s
    }

    /// Flat index of a multi-index.
    pub fn index(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.dims.len());
        idx.iter().zip(self.strides()).map(|(i, s)| i * s).sum()
    }

    /// Flat indices of the first node of every grid line along `axis`.
    fn line_starts(&self, axis: usize) -> Vec<usize> {
        let stride = self.strides()[axis];
        let n = self.dims[axis];
        (0..self.len()).filter(|i| (i / stride) % n == 0).collect()
    }
}

/// A tridiagonal operator along one axis of a [`TensorGrid`], with
/// per-node coefficients: row `i` of `A u` reads
/// `sub[i] * u[i - stride] + diag[i] * u[i] + sup[i] * u[i + stride]`.
///
/// `sub` must be zero on the first plane of the axis and `sup` on the
/// last (there is no neighbor there); boundary conditions are whatever
/// the boundary rows encode — an all-zero row holds the boundary value
/// fixed through both explicit application and implicit solves.
#[derive(Debug, Clone)]
pub struct AxisOperator {
    pub axis: usize,
    pub sub: Vec<f64>,
    pub diag: Vec<f64>,
    pub sup: Vec<f64>,
}

impl AxisOperator {
    /// An all-zero operator along `axis` (a starting point to fill in).
    pub fn zero(grid: &TensorGrid, axis: usize) -> Self {
        assert!(axis < grid.ndim());
        let n = grid.len();
        Self { axis, sub: vec![0.0; n], diag: vec![0.0; n], sup: vec![0.0; n] }
    }

    /// `A u`.
    pub fn apply(&self, grid: &TensorGrid, u: &[f64]) -> Vec<f64> {
        let stride = grid.strides()[self.axis];
        let n = grid.dims()[self.axis];
        assert_eq!(u.len(), grid.len());
        (0..grid.len())
            .map(|i| {
                let j = (i / stride) % n;
                let mut v = self.diag[i] * u[i];
                if j > 0 {
                    v += self.sub[i] * u[i - stride];
                }
                if j < n - 1 {
                    v += self.sup[i] * u[i + stride];
                }
                v
            })
            .collect()
    }

    /// Solve `(I - c A) x = rhs`, line by line with the Thomas algorithm —
    /// the implicit stage of theta and ADI schemes.
    pub fn solve_shifted(&self, grid: &TensorGrid, c: f64, rhs: &[f64]) -> Vec<f64> {
        let stride = grid.strides()[self.axis];
        let n = grid.dims()[self.axis];
        assert_eq!(rhs.len(), grid.len());
        let mut x = rhs.to_vec();
        if n == 1 {
            for (xi, &r) in x.iter_mut().zip(rhs) {
                *xi = r / (1.0 - c * self.diag[0]);
            }
            return x;
        }
        let mut a = vec![0.0; n - 1];
        let mut b = vec![0.0; n];
        let mut cc = vec![0.0; n - 1];
        let mut d = vec![0.0; n];
        for start in grid.line_starts(self.axis) {
            for j in 0..n {
                let i = start + j * stride;
                b[j] = 1.0 - c * self.diag[i];
                d[j] = rhs[i];
                if j > 0 {
                    a[j - 1] = -c * self.sub[i];
                }
                if j < n - 1 {
                    cc[j] = -c * self.sup[i];
                }
            }
            let line = thomas_algorithm(&a, &b, &cc, &d);
            for (j, v) in line.into_iter().enumerate() {
                x[start + j * stride] = v;
            }
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-D Laplacian with Dirichlet (zero) boundary rows on [0, 1].
    pub fn laplacian_1d(grid: &TensorGrid, axis: usize, h: f64) -> AxisOperator {
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

    #[test]
    fn strides_and_indexing_are_row_major() {
        let g = TensorGrid::new(&[3, 4, 5]);
        assert_eq!(g.strides(), vec![20, 5, 1]);
        assert_eq!(g.index(&[1, 2, 3]), 33);
        assert_eq!(g.len(), 60);
    }

    #[test]
    fn apply_matches_dense_stencil_2d() {
        // second difference along axis 1 of a 2-D grid, checked by hand
        let g = TensorGrid::new(&[2, 4]);
        let h = 1.0;
        let op = laplacian_1d(&g, 1, h);
        let u: Vec<f64> = (0..8).map(|i| (i * i) as f64).collect();
        let au = op.apply(&g, &u);
        // row 0: nodes 0..4 with u = [0,1,4,9]: interior j=1 -> 0-2+4=2, j=2 -> 1-8+9=2
        assert_eq!(&au[0..4], &[0.0, 2.0, 2.0, 0.0]);
        // row 1: u = [16,25,36,49]: j=1 -> 16-50+36=2, j=2 -> 25-72+49=2
        assert_eq!(&au[4..8], &[0.0, 2.0, 2.0, 0.0]);
    }

    #[test]
    fn solve_shifted_inverts_apply() {
        // x solves (I - cA) x = rhs  <=>  rhs = x - c A x
        let g = TensorGrid::new(&[3, 5, 4]);
        let op = laplacian_1d(&g, 1, 0.25);
        let rhs: Vec<f64> = (0..g.len()).map(|i| ((i % 7) as f64) - 3.0).collect();
        let c = 0.37;
        let x = op.solve_shifted(&g, c, &rhs);
        let ax = op.apply(&g, &x);
        for i in 0..g.len() {
            let back = x[i] - c * ax[i];
            assert!((back - rhs[i]).abs() < 1e-10, "node {i}: {back} vs {}", rhs[i]);
        }
    }
}
