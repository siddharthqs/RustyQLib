//! Piecewise linear interpolation and the pillar bracketing shared by
//! curves and smiles.

/// Linear blend `a + w (b - a)`.
pub fn lerp(a: f64, b: f64, w: f64) -> f64 {
    a + w * (b - a)
}

/// Bracket `x` in the sorted pillar grid `xs`: returns `(idx, w)` where
/// `xs[idx - 1] <= x <= xs[idx]` and `w` is the weight of the upper
/// pillar. `x` is clamped to the grid, so `idx` is always in
/// `[1, xs.len() - 1]` and `w` in `[0, 1]`.
pub fn bracket(xs: &[f64], x: f64) -> (usize, f64) {
    assert!(xs.len() >= 2, "need at least two pillars");
    let n = xs.len();
    if x <= xs[0] {
        return (1, 0.0);
    }
    if x >= xs[n - 1] {
        return (n - 1, 1.0);
    }
    let idx = xs.partition_point(|&xi| xi < x);
    let (x0, x1) = (xs[idx - 1], xs[idx]);
    (idx, (x - x0) / (x1 - x0))
}

/// Piecewise linear interpolation of `(xs, ys)` at `x`, flat beyond the
/// ends. `xs` must be sorted strictly increasing.
pub fn linear_interp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    assert_eq!(xs.len(), ys.len());
    let (idx, w) = bracket(xs, x);
    lerp(ys[idx - 1], ys[idx], w)
}

/// [`linear_interp`] over `(x, y)` pairs sorted by `x` — the smile
/// storage shape.
pub fn interp_pairs(points: &[(f64, f64)], x: f64) -> f64 {
    assert!(points.len() >= 2, "need at least two points");
    let n = points.len();
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points[n - 1].0 {
        return points[n - 1].1;
    }
    let idx = points.partition_point(|&(xi, _)| xi < x);
    let (x0, y0) = points[idx - 1];
    let (x1, y1) = points[idx];
    lerp(y0, y1, (x - x0) / (x1 - x0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolates_and_extrapolates_flat() {
        let xs = [1.0, 2.0, 4.0];
        let ys = [10.0, 20.0, 40.0];
        assert_eq!(linear_interp(&xs, &ys, 1.5), 15.0);
        assert_eq!(linear_interp(&xs, &ys, 3.0), 30.0);
        assert_eq!(linear_interp(&xs, &ys, 0.0), 10.0); // flat left
        assert_eq!(linear_interp(&xs, &ys, 9.0), 40.0); // flat right
        assert_eq!(interp_pairs(&[(1.0, 10.0), (2.0, 20.0)], 1.25), 12.5);
    }

    #[test]
    fn bracket_clamps_to_the_grid() {
        let xs = [0.0, 1.0, 3.0];
        assert_eq!(bracket(&xs, -5.0), (1, 0.0));
        assert_eq!(bracket(&xs, 5.0), (2, 1.0));
        let (idx, w) = bracket(&xs, 2.0);
        assert_eq!(idx, 2);
        assert!((w - 0.5).abs() < 1e-15);
    }
}
