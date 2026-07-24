//! PCHIP: Fritsch-Carlson monotone piecewise cubic Hermite
//! interpolation. Shape-preserving — the interpolant is monotone
//! wherever the data are, so it never overshoots or invents wiggles the
//! way an unconstrained cubic spline can. The safe choice for curves
//! that must stay monotone (discount factors, CDFs) or non-negative.
use crate::core::errors::RustyQLibError;

/// A monotonicity-preserving cubic Hermite interpolant.
#[derive(Debug, Clone)]
pub struct Pchip {
    xs: Vec<f64>,
    ys: Vec<f64>,
    /// Knot slopes after the Fritsch-Carlson limiter.
    d: Vec<f64>,
}

impl Pchip {
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, RustyQLibError> {
        let n = xs.len();
        if n < 2 || ys.len() != n {
            return Err(RustyQLibError::invalid_input("pchip", "need at least two knots with matching y values"));
        }
        if xs.windows(2).any(|w| w[1] <= w[0]) {
            return Err(RustyQLibError::invalid_input("pchip", "knots must be strictly increasing"));
        }
        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let delta: Vec<f64> =
            (0..n - 1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

        let mut d = vec![0.0; n];
        if n == 2 {
            d[0] = delta[0];
            d[1] = delta[0];
        } else {
            // interior: weighted harmonic mean when the secants agree in
            // sign, zero otherwise (this is what preserves monotonicity)
            for i in 1..n - 1 {
                if delta[i - 1] * delta[i] > 0.0 {
                    let w1 = 2.0 * h[i] + h[i - 1];
                    let w2 = h[i] + 2.0 * h[i - 1];
                    d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
                }
            }
            d[0] = end_slope(h[0], h[1], delta[0], delta[1]);
            d[n - 1] = end_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        }
        Ok(Pchip { xs: xs.to_vec(), ys: ys.to_vec(), d })
    }

    /// Interpolant value at `x` (linear extrapolation with the end slope).
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        if x <= self.xs[0] {
            return self.ys[0] + self.d[0] * (x - self.xs[0]);
        }
        if x >= self.xs[n - 1] {
            return self.ys[n - 1] + self.d[n - 1] * (x - self.xs[n - 1]);
        }
        let i = self.xs[1..n - 1].partition_point(|&xi| xi < x);
        hermite(
            x,
            self.xs[i],
            self.xs[i + 1],
            self.ys[i],
            self.ys[i + 1],
            self.d[i],
            self.d[i + 1],
        )
    }
}

/// Three-point end slope with the Fritsch-Carlson clips.
fn end_slope(h0: f64, h1: f64, delta0: f64, delta1: f64) -> f64 {
    let d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / (h0 + h1);
    if d * delta0 <= 0.0 {
        0.0
    } else if delta0 * delta1 < 0.0 && d.abs() > 3.0 * delta0.abs() {
        3.0 * delta0
    } else {
        d
    }
}

/// Cubic Hermite basis evaluation on `[x0, x1]`.
pub(crate) fn hermite(x: f64, x0: f64, x1: f64, y0: f64, y1: f64, d0: f64, d1: f64) -> f64 {
    let h = x1 - x0;
    let t = (x - x0) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    y0 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + y1 * (-2.0 * t3 + 3.0 * t2)
        + d0 * h * (t3 - 2.0 * t2 + t)
        + d1 * h * (t3 - t2)
}

#[cfg(test)]
mod tests {
    use super::super::cubic_spline::{BoundaryCondition, CubicSpline};
    use super::*;

    /// A monotone step-like data set that makes free splines overshoot.
    fn step_data() -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
    }

    #[test]
    fn interpolates_the_knots() {
        let (xs, ys) = step_data();
        let p = Pchip::new(&xs, &ys).unwrap();
        for (x, y) in xs.iter().zip(&ys) {
            assert!((p.eval(*x) - y).abs() < 1e-14);
        }
    }

    #[test]
    fn preserves_monotonicity_where_a_cubic_spline_overshoots() {
        let (xs, ys) = step_data();
        let pchip = Pchip::new(&xs, &ys).unwrap();
        let spline = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();

        let mut prev = f64::NEG_INFINITY;
        let mut spline_overshoots = false;
        for i in 0..=500 {
            let x = i as f64 * 0.01;
            let v = pchip.eval(x);
            // monotone and inside the data range
            assert!(v >= prev - 1e-12, "pchip not monotone at {x}");
            assert!((-1e-12..=1.0 + 1e-12).contains(&v), "pchip overshoots at {x}");
            prev = v;
            let s = spline.eval(x);
            if !(0.0..=1.0).contains(&s) {
                spline_overshoots = true;
            }
        }
        // and the comparison is meaningful: the free spline DOES overshoot
        assert!(spline_overshoots, "cubic spline unexpectedly shape-preserving");
    }

    #[test]
    fn flat_data_stays_exactly_flat() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [5.0, 5.0, 5.0, 5.0];
        let p = Pchip::new(&xs, &ys).unwrap();
        for i in 0..=30 {
            assert!((p.eval(i as f64 * 0.1) - 5.0).abs() < 1e-12);
        }
    }
}
