//! Variance-reduction building blocks: moment matching and the generic
//! regression-based control-variate estimator. (Antithetic pairing lives
//! where draws are generated — see
//! [`pseudo_normals`](super::rng::pseudo_normals) — and low-discrepancy
//! sampling in [`sobol`](super::sobol) / [`halton`](super::halton) is
//! itself the strongest variance reduction for smooth payoffs.)

/// Rescale draws in place to sample mean 0 and variance 1 exactly —
/// removes the O(1/sqrt(n)) noise in the first two sample moments.
pub fn moment_match(draws: &mut [f64]) {
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|z| (z - mean) * (z - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    if std > 0.0 {
        for z in draws.iter_mut() {
            *z = (*z - mean) / std;
        }
    }
}

/// Control-variate estimate of `E[payoff]` given per-path values of a
/// control with known expectation `control_mean`.
///
/// The optimal coefficient `beta = cov(payoff, control) / var(control)`
/// is estimated from the same sample, and the adjusted estimator is
/// `mean(payoff - beta (control - control_mean))`. Returns
/// `(estimate, standard_error, beta)`; the standard error is that of the
/// adjusted per-path values, so it shows the variance actually removed.
pub fn control_variate_estimate(
    payoffs: &[f64],
    controls: &[f64],
    control_mean: f64,
) -> (f64, f64, f64) {
    assert_eq!(payoffs.len(), controls.len());
    let n = payoffs.len() as f64;
    assert!(n > 1.0, "need at least two paths");
    let mean_y = payoffs.iter().sum::<f64>() / n;
    let mean_c = controls.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_c = 0.0;
    for (y, c) in payoffs.iter().zip(controls) {
        cov += (y - mean_y) * (c - mean_c);
        var_c += (c - mean_c) * (c - mean_c);
    }
    let beta = if var_c > 0.0 { cov / var_c } else { 0.0 };

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for (y, c) in payoffs.iter().zip(controls) {
        let adjusted = y - beta * (c - control_mean);
        sum += adjusted;
        sum_sq += adjusted * adjusted;
    }
    let mean = sum / n;
    let var = (sum_sq / n - mean * mean).max(0.0);
    (mean, (var / n).sqrt(), beta)
}

#[cfg(test)]
mod tests {
    use super::super::rng::path_normals;
    use super::*;

    #[test]
    fn moment_matching_standardizes_exactly() {
        let mut z = vec![0.0; 1001];
        path_normals(3, 0, &mut z);
        moment_match(&mut z);
        let n = z.len() as f64;
        let mean: f64 = z.iter().sum::<f64>() / n;
        let var: f64 = z.iter().map(|x| x * x).sum::<f64>() / n;
        assert!(mean.abs() < 1e-12 && (var - 1.0).abs() < 1e-12);
    }

    #[test]
    fn control_variate_removes_correlated_noise() {
        // payoff = 3 + 2 C + independent noise, control C with known mean 0:
        // the fit must find beta ~ 2 and cut the standard error sharply
        let n = 4000;
        let mut c = vec![0.0; n];
        let mut eps = vec![0.0; n];
        path_normals(11, 0, &mut c);
        path_normals(11, 1, &mut eps);
        let payoffs: Vec<f64> =
            c.iter().zip(&eps).map(|(ci, ei)| 3.0 + 2.0 * ci + 0.1 * ei).collect();

        let (est, se, beta) = control_variate_estimate(&payoffs, &c, 0.0);
        assert!((beta - 2.0).abs() < 0.05, "beta {beta}");
        assert!((est - 3.0).abs() < 0.01, "estimate {est}");
        // raw standard error of the payoffs for comparison
        let mean_y: f64 = payoffs.iter().sum::<f64>() / n as f64;
        let var_y: f64 =
            payoffs.iter().map(|y| (y - mean_y) * (y - mean_y)).sum::<f64>() / n as f64;
        let raw_se = (var_y / n as f64).sqrt();
        assert!(se < 0.1 * raw_se, "cv se {se} vs raw {raw_se}");
    }

    #[test]
    fn zero_variance_control_degrades_gracefully() {
        let payoffs = [1.0, 2.0, 3.0];
        let controls = [5.0, 5.0, 5.0];
        let (est, _, beta) = control_variate_estimate(&payoffs, &controls, 5.0);
        assert_eq!(beta, 0.0);
        assert!((est - 2.0).abs() < 1e-12);
    }
}
