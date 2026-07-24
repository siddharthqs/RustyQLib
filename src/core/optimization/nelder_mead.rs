//! Nelder-Mead simplex: gradient-free local search by reflecting,
//! expanding and contracting a simplex of `n + 1` points. The tool for
//! noisy or non-smooth objectives where gradients mislead.

use super::{OptimConfig, OptimResult};

/// Minimize `f` from `x0` with the standard Nelder-Mead coefficients
/// (reflection 1, expansion 2, contraction 1/2, shrink 1/2). Converged
/// when the simplex's value spread falls below `tol` (relative to the
/// best value).
pub fn nelder_mead(cfg: &OptimConfig, f: &dyn Fn(&[f64]) -> f64, x0: &[f64]) -> OptimResult {
    let n = x0.len();
    // initial simplex: x0 plus a 5% step per coordinate (0.00025 at zero)
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += if x0[i] != 0.0 { 0.05 * x0[i] } else { 0.00025 };
        simplex.push(v);
    }
    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for it in 0..cfg.max_iter {
        // order best -> worst
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];
        if (values[worst] - values[best]).abs() <= cfg.tol * (1.0 + values[best].abs()) {
            return OptimResult {
                x: simplex[best].clone(),
                value: values[best],
                iterations: it,
                converged: true,
            };
        }
        // centroid of all but the worst
        let mut centroid = vec![0.0; n];
        for (idx, v) in simplex.iter().enumerate() {
            if idx != worst {
                for (c, vi) in centroid.iter_mut().zip(v) {
                    *c += vi / n as f64;
                }
            }
        }
        let point = |t: f64| -> Vec<f64> {
            centroid
                .iter()
                .zip(&simplex[worst])
                .map(|(c, w)| c + t * (c - w))
                .collect()
        };

        let reflected = point(1.0);
        let f_r = f(&reflected);
        if f_r < values[best] {
            let expanded = point(2.0);
            let f_e = f(&expanded);
            if f_e < f_r {
                simplex[worst] = expanded;
                values[worst] = f_e;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_r;
            }
        } else if f_r < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = f_r;
        } else {
            let contracted = if f_r < values[worst] { point(0.5) } else { point(-0.5) };
            let f_c = f(&contracted);
            if f_c < values[worst].min(f_r) {
                simplex[worst] = contracted;
                values[worst] = f_c;
            } else {
                // shrink toward the best vertex
                let anchor = simplex[best].clone();
                for idx in 0..=n {
                    if idx != best {
                        simplex[idx] = simplex[idx]
                            .iter()
                            .zip(&anchor)
                            .map(|(v, b)| b + 0.5 * (v - b))
                            .collect();
                        values[idx] = f(&simplex[idx]);
                    }
                }
            }
        }
    }
    let (best, &value) =
        values.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).expect("non-empty");
    OptimResult { x: simplex[best].clone(), value, iterations: cfg.max_iter, converged: false }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimizes_rosenbrock_without_gradients() {
        let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
        let r = nelder_mead(&OptimConfig::new(1e-12, 2000), &f, &[-1.2, 1.0]);
        assert!((r.x[0] - 1.0).abs() < 1e-4 && (r.x[1] - 1.0).abs() < 1e-4, "{r:?}");
    }

    #[test]
    fn handles_a_non_smooth_objective() {
        // |x| + |y - 2|: kinked at the optimum, gradients undefined there
        let f = |x: &[f64]| x[0].abs() + (x[1] - 2.0).abs();
        let r = nelder_mead(&OptimConfig::new(1e-12, 2000), &f, &[3.0, -3.0]);
        assert!(r.x[0].abs() < 1e-4 && (r.x[1] - 2.0).abs() < 1e-4, "{r:?}");
    }
}
