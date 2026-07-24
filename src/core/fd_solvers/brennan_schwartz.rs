//! Brennan-Schwartz solve of the American-exercise linear complementarity
//! problem on a tridiagonal system.

/// Brennan-Schwartz solve of the linear complementarity problem
/// `Ax = d, x >= exercise`: standard Thomas elimination with the constraint
/// applied during back-substitution. The substitution must sweep *toward*
/// the exercise region, so puts (exercise at low spot) use the natural
/// high-to-low sweep and calls are solved on the reversed system.
pub fn brennan_schwartz(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    exercise: &[f64],
    exercise_at_low_spot: bool,
) -> Vec<f64> {
    if exercise_at_low_spot {
        brennan_schwartz_sweep(a, b, c, d, exercise)
    } else {
        // reverse the system: row order flips, sub- and super-diagonals swap
        let n = d.len();
        let ar: Vec<f64> = c.iter().rev().copied().collect();
        let br: Vec<f64> = b.iter().rev().copied().collect();
        let cr: Vec<f64> = a.iter().rev().copied().collect();
        let dr: Vec<f64> = d.iter().rev().copied().collect();
        let er: Vec<f64> = exercise.iter().rev().copied().collect();
        let mut x = brennan_schwartz_sweep(&ar, &br, &cr, &dr, &er);
        x.reverse();
        debug_assert_eq!(x.len(), n);
        x
    }
}

fn brennan_schwartz_sweep(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    exercise: &[f64],
) -> Vec<f64> {
    let n = d.len();
    if n == 1 {
        return vec![(d[0] / b[0]).max(exercise[0])];
    }
    let mut c_ = c.to_vec();
    let mut d_ = d.to_vec();
    let mut x = vec![0.0; n];
    c_[0] = c_[0] / b[0];
    d_[0] = d_[0] / b[0];
    for i in 1..n - 1 {
        let id = 1.0 / (b[i] - a[i - 1] * c_[i - 1]);
        c_[i] = c_[i] * id;
        d_[i] = (d_[i] - a[i - 1] * d_[i - 1]) * id;
    }
    d_[n - 1] = (d_[n - 1] - a[n - 2] * d_[n - 2]) / (b[n - 1] - a[n - 2] * c_[n - 2]);

    x[n - 1] = d_[n - 1].max(exercise[n - 1]);
    for i in (0..n - 1).rev() {
        x[i] = (d_[i] - c_[i] * x[i + 1]).max(exercise[i]);
    }
    x
}

#[cfg(test)]
mod tests {
    use super::super::tridiagonal::thomas_algorithm;
    use super::*;

    #[test]
    fn reduces_to_thomas_when_unconstrained() {
        let a = [1.0, 1.0];
        let b = [3.0, 3.0, 3.0];
        let c = [1.0, 1.0];
        let d = [5.0, 10.0, 11.0];
        let free = thomas_algorithm(&a, &b, &c, &d);
        let low = [-1e9, -1e9, -1e9];
        for dir in [true, false] {
            let constrained = brennan_schwartz(&a, &b, &c, &d, &low, dir);
            for (x, y) in free.iter().zip(&constrained) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn enforces_floor() {
        let a = [1.0, 1.0];
        let b = [3.0, 3.0, 3.0];
        let c = [1.0, 1.0];
        let d = [5.0, 10.0, 11.0];
        let floor = [10.0, 10.0, 10.0];
        let x = brennan_schwartz(&a, &b, &c, &d, &floor, true);
        assert!(x.iter().all(|&v| v >= 10.0 - 1e-12), "{x:?}");
    }
}
