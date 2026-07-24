//! Akima's spline: cubic Hermite interpolation with slopes chosen from
//! local weighted secant differences. Purely local (a moved point only
//! affects its neighborhood) and far less prone to the wide oscillations
//! a global cubic spline develops around outliers and flat runs.

use super::pchip::hermite;

/// An Akima spline interpolant.
#[derive(Debug, Clone)]
pub struct Akima {
    xs: Vec<f64>,
    ys: Vec<f64>,
    d: Vec<f64>,
}

impl Akima {
    pub fn new(xs: &[f64], ys: &[f64]) -> Result<Self, String> {
        let n = xs.len();
        if n < 2 || ys.len() != n {
            return Err("need at least two knots with matching y values".into());
        }
        if xs.windows(2).any(|w| w[1] <= w[0]) {
            return Err("knots must be strictly increasing".into());
        }
        // secants with Akima's quadratic extension at both ends:
        // ext[k] corresponds to delta_{k-2} for knot arithmetic below
        let mut ext = Vec::with_capacity(n + 3);
        ext.resize(2, 0.0);
        for i in 0..n - 1 {
            ext.push((ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]));
        }
        let m = ext.len();
        ext.push(2.0 * ext[m - 1] - ext[m - 2]);
        ext.push(2.0 * ext[m] - ext[m - 1]);
        ext[1] = 2.0 * ext[2] - ext[3];
        ext[0] = 2.0 * ext[1] - ext[2];

        let d: Vec<f64> = (0..n)
            .map(|i| {
                // slopes around knot i: ext[i..i+4] = delta_{i-2..i+1}
                let w1 = (ext[i + 3] - ext[i + 2]).abs();
                let w2 = (ext[i + 1] - ext[i]).abs();
                if w1 + w2 > 1e-300 {
                    (w1 * ext[i + 1] + w2 * ext[i + 2]) / (w1 + w2)
                } else {
                    0.5 * (ext[i + 1] + ext[i + 2])
                }
            })
            .collect();
        Ok(Akima { xs: xs.to_vec(), ys: ys.to_vec(), d })
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

#[cfg(test)]
mod tests {
    use super::super::cubic_spline::{BoundaryCondition, CubicSpline};
    use super::*;

    #[test]
    fn interpolates_the_knots() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.0, 0.5, 2.0, 1.5, 1.0, 3.0];
        let a = Akima::new(&xs, &ys).unwrap();
        for (x, y) in xs.iter().zip(&ys) {
            assert!((a.eval(*x) - y).abs() < 1e-13);
        }
    }

    #[test]
    fn flat_runs_stay_flat_where_a_spline_rings() {
        // Akima's classic showcase: a flat run next to a jump — the
        // global spline rings along the flat section, Akima does not
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ys = [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0];
        let akima = Akima::new(&xs, &ys).unwrap();
        let spline = CubicSpline::new(&xs, &ys, BoundaryCondition::Natural).unwrap();

        // on the flat left section the Akima interpolant is exactly zero
        let mut spline_rings = false;
        for i in 0..=25 {
            let x = i as f64 * 0.1; // [0, 2.5]
            assert!(akima.eval(x).abs() < 1e-12, "akima rings at {x}");
            if spline.eval(x).abs() > 1e-3 {
                spline_rings = true;
            }
        }
        assert!(spline_rings, "cubic spline unexpectedly local");
    }

    #[test]
    fn straight_line_data_reproduces_the_line() {
        let xs = [0.0, 1.0, 3.0, 6.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x - 1.0).collect();
        let a = Akima::new(&xs, &ys).unwrap();
        for i in 0..=70 {
            let x = i as f64 * 0.1;
            assert!((a.eval(x) - (2.0 * x - 1.0)).abs() < 1e-12, "x = {x}");
        }
    }
}
