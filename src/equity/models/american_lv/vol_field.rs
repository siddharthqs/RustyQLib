//! How a solver reads the local volatility.
//!
//! Every solver in this module evaluates `Sigma` at the mesh nodes
//! `x_j = ln S_j` and at the **mid-time of each backward step**
//! `t_{n+1/2}`; the same evaluation points are used by the forward march
//! and by the adjoint, so a gradient field `dF/dSigma_{j,n}` refers to the
//! volatility at (node `j`, step `n`). A [`NodeField`] stores exactly
//! those values (precomputed once per expiry mesh per parameter vector),
//! so the hot loops never evaluate a spline.

/// A local-volatility field `Sigma(x, t)` with `x = ln S`, in years.
///
/// Implementations must be `Sync`: the calibration evaluates the field
/// from several threads at once.
pub trait VolField: Sync {
    /// Volatility (annualised, e.g. `0.2`) at log-spot `x` and time `t`.
    fn vol(&self, x: f64, t: f64) -> f64;
}

/// A constant volatility.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlatVol(pub f64);

impl VolField for FlatVol {
    #[inline]
    fn vol(&self, _x: f64, _t: f64) -> f64 {
        self.0
    }
}

/// A closure-backed field, for synthetic ground-truth surfaces.
pub struct CallbackVol(pub Box<dyn Fn(f64, f64) -> f64 + Send + Sync>);

impl CallbackVol {
    /// Wrap a closure `sigma(x, t)`.
    pub fn new(f: impl Fn(f64, f64) -> f64 + Send + Sync + 'static) -> Self {
        CallbackVol(Box::new(f))
    }
}

impl VolField for CallbackVol {
    #[inline]
    fn vol(&self, x: f64, t: f64) -> f64 {
        (self.0)(x, t)
    }
}

impl std::fmt::Debug for CallbackVol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CallbackVol(<closure>)")
    }
}

/// Volatility sampled at the nodes of one expiry mesh: `values[n * stride
/// + j]` is `Sigma(x_j, t_{n+1/2})` for backward step `n` (counted from
/// `t = 0`, i.e. `n = 0` is the step adjacent to the valuation date) and
/// node `j`.
///
/// Built by the B-spline surface once per (expiry mesh, parameter
/// vector); read by the solver and the adjoint through
/// [`NodeField::at`]. The [`VolField`] impl is provided for generic
/// callers and looks up the nearest step by time, which is exact at the
/// step mid-times the mesh defines.
#[derive(Debug, Clone)]
pub struct NodeField {
    /// Flat storage, `n_steps * stride` entries.
    pub values: Vec<f64>,
    /// Nodes per level (`n_x + 1`).
    pub stride: usize,
    /// Mid-time of each backward step, increasing, one per level.
    pub t_mid: Vec<f64>,
    /// First node `x_0` and uniform spacing `dx` of the sampled mesh
    /// (for the nearest-node read of [`VolField::vol`]).
    pub x_min: f64,
    /// Node spacing (`0` when the mesh has a single node).
    pub dx: f64,
}

impl NodeField {
    /// Build from the (uniform, increasing) mesh nodes, the step
    /// mid-times and a field evaluated at every `(x_j, t_mid[n])`.
    pub fn sample(x_nodes: &[f64], t_mid: &[f64], field: &dyn VolField) -> Self {
        let stride = x_nodes.len();
        let mut values = Vec::with_capacity(stride * t_mid.len());
        for &t in t_mid {
            for &x in x_nodes {
                values.push(field.vol(x, t));
            }
        }
        let x_min = x_nodes.first().copied().unwrap_or(0.0);
        let dx = if stride > 1 {
            (x_nodes[stride - 1] - x_min) / (stride - 1) as f64
        } else {
            0.0
        };
        NodeField {
            values,
            stride,
            t_mid: t_mid.to_vec(),
            x_min,
            dx,
        }
    }

    /// The node nearest `x` (clamped to the mesh).
    #[inline]
    pub fn node_of(&self, x: f64) -> usize {
        if self.dx <= 0.0 || self.stride == 0 {
            return 0;
        }
        let j = ((x - self.x_min) / self.dx).round();
        j.clamp(0.0, (self.stride - 1) as f64) as usize
    }

    /// Volatility at node `j` of step `n`.
    #[inline]
    pub fn at(&self, step: usize, node: usize) -> f64 {
        self.values[step * self.stride + node]
    }

    /// Number of steps stored.
    #[inline]
    pub fn steps(&self) -> usize {
        self.t_mid.len()
    }

    /// The step whose mid-time is nearest `t` (binary search; the mesh
    /// is increasing).
    pub fn step_of(&self, t: f64) -> usize {
        match self.t_mid.binary_search_by(|m| m.partial_cmp(&t).unwrap()) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) if i >= self.t_mid.len() => self.t_mid.len() - 1,
            Err(i) => {
                if (self.t_mid[i] - t).abs() < (t - self.t_mid[i - 1]).abs() {
                    i
                } else {
                    i - 1
                }
            }
        }
    }
}

impl VolField for NodeField {
    /// Nearest step in time, nearest node in `x`: exact at the mesh
    /// points (where the solvers read it), a convenience elsewhere.
    fn vol(&self, x: f64, t: f64) -> f64 {
        self.at(self.step_of(t), self.node_of(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_field_is_constant_and_node_field_reads_back_its_samples() {
        let flat = FlatVol(0.25);
        assert_eq!(flat.vol(0.0, 0.5), 0.25);
        let x: Vec<f64> = (0..5).map(|i| 4.0 + 0.1 * i as f64).collect();
        let t_mid = vec![0.05, 0.15, 0.25];
        let cb = CallbackVol::new(|x, t| 0.1 * x + t);
        let nf = NodeField::sample(&x, &t_mid, &cb);
        assert_eq!(nf.steps(), 3);
        assert!(
            (nf.at(1, 2) - (0.1 * 4.2 + 0.15)).abs() < 1e-14,
            "node read"
        );
        assert_eq!(nf.step_of(0.16), 1);
        assert_eq!(nf.step_of(-1.0), 0);
        assert_eq!(nf.step_of(9.0), 2);
        assert_eq!(nf.node_of(4.21), 2);
        assert_eq!(nf.node_of(-3.0), 0);
        assert_eq!(nf.node_of(7.0), 4);
        assert!(
            (nf.vol(4.21, 0.16) - nf.at(1, 2)).abs() < 1e-15,
            "nearest read"
        );
    }
}
