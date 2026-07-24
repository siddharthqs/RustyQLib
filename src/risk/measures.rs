//! Value-at-Risk and Expected Shortfall in the three standard flavors:
//! historical (empirical), parametric normal (with a Cornish-Fisher
//! higher-moment correction), and delta-normal for multi-asset books.
//!
//! Conventions: `confidence` is the one-sided level (0.99 = 99%), and
//! both VaR and ES are reported as **positive loss amounts** in the P&L
//! currency. ES is always >= VaR at the same level (asserted in tests).

use crate::core::utils::{norm_pdf, inv_norm_cdf};

/// Linear-interpolation (type-7) empirical quantile of a sample.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    assert!(n > 0);
    let h = (n as f64 - 1.0) * p.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Historical (empirical) VaR from a P&L sample.
pub fn historical_var(pnl: &[f64], confidence: f64) -> f64 {
    assert!(!pnl.is_empty() && confidence > 0.5 && confidence < 1.0);
    let mut losses: Vec<f64> = pnl.iter().map(|x| -x).collect();
    losses.sort_by(f64::total_cmp);
    quantile(&losses, confidence).max(0.0)
}

/// Historical Expected Shortfall: the average loss at or beyond the VaR
/// quantile.
pub fn historical_expected_shortfall(pnl: &[f64], confidence: f64) -> f64 {
    assert!(!pnl.is_empty() && confidence > 0.5 && confidence < 1.0);
    let mut losses: Vec<f64> = pnl.iter().map(|x| -x).collect();
    losses.sort_by(f64::total_cmp);
    let var = quantile(&losses, confidence);
    let tail: Vec<f64> = losses.iter().copied().filter(|&l| l >= var).collect();
    if tail.is_empty() {
        return var.max(0.0);
    }
    (tail.iter().sum::<f64>() / tail.len() as f64).max(0.0)
}

/// Parametric VaR under normal P&L with the given `mean` and `std`.
/// `VaR = -mean + std * z_alpha`.
pub fn parametric_var(mean: f64, std: f64, confidence: f64) -> f64 {
    assert!(std >= 0.0 && confidence > 0.5 && confidence < 1.0);
    (-mean + std * inv_norm_cdf(confidence)).max(0.0)
}

/// Parametric Expected Shortfall under normal P&L:
/// `ES = -mean + std * phi(z_alpha) / (1 - alpha)`.
pub fn parametric_expected_shortfall(mean: f64, std: f64, confidence: f64) -> f64 {
    assert!(std >= 0.0 && confidence > 0.5 && confidence < 1.0);
    let z = inv_norm_cdf(confidence);
    (-mean + std * norm_pdf(z) / (1.0 - confidence)).max(0.0)
}

/// Cornish-Fisher VaR: the normal quantile adjusted for the sample's
/// skewness and excess kurtosis — a standard desk correction for fat,
/// asymmetric P&L. With zero skew and excess kurtosis it reduces to
/// [`parametric_var`].
pub fn cornish_fisher_var(
    mean: f64,
    std: f64,
    skewness: f64,
    excess_kurtosis: f64,
    confidence: f64,
) -> f64 {
    // VaR lives in the LOWER tail of P&L: expand the (1 - confidence)
    // quantile. The odd (linear-in-z) terms flip sign with the tail,
    // the skew term does not - negative P&L skew lengthens the loss
    // tail and raises VaR.
    let zl = -inv_norm_cdf(confidence);
    let z2 = zl * zl;
    let q = zl
        + (z2 - 1.0) * skewness / 6.0
        + zl * (z2 - 3.0) * excess_kurtosis / 24.0
        - zl * (2.0 * z2 - 5.0) * skewness * skewness / 36.0;
    (-(mean + std * q)).max(0.0)
}

/// Delta-normal (variance-covariance) VaR of a linear book with the
/// **Euler decomposition** into per-position components.
#[derive(Debug, Clone)]
pub struct DeltaNormalVar {
    /// Total portfolio VaR.
    pub var: f64,
    /// Portfolio P&L standard deviation.
    pub std: f64,
    /// Component VaR per position (sums exactly to `var`).
    pub component_var: Vec<f64>,
    /// Marginal VaR per position (`d var / d exposure_i`).
    pub marginal_var: Vec<f64>,
}

/// Delta-normal VaR: `exposures[i]` is the currency P&L per unit return
/// of asset `i` (delta x spot), `covariance` the per-horizon return
/// covariance matrix.
pub fn delta_normal_var(
    exposures: &[f64],
    covariance: &[Vec<f64>],
    confidence: f64,
) -> DeltaNormalVar {
    let n = exposures.len();
    assert!(n > 0 && covariance.len() == n);
    assert!(covariance.iter().all(|row| row.len() == n));
    // sigma_p^2 = w' C w  and the gradient C w
    let cw: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| covariance[i][j] * exposures[j]).sum())
        .collect();
    let variance: f64 = exposures.iter().zip(&cw).map(|(w, c)| w * c).sum();
    assert!(variance >= -1e-12, "covariance matrix is not PSD on this exposure");
    let std = variance.max(0.0).sqrt();
    let z = inv_norm_cdf(confidence);
    let var = std * z;
    // Euler: component_i = w_i (Cw)_i / sigma_p * z; sums to VaR exactly
    let (component_var, marginal_var) = if std > 0.0 {
        (
            exposures.iter().zip(&cw).map(|(w, c)| w * c / std * z).collect(),
            cw.iter().map(|c| c / std * z).collect(),
        )
    } else {
        (vec![0.0; n], vec![0.0; n])
    };
    DeltaNormalVar { var, std, component_var, marginal_var }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::montecarlo::path_rng;
    use rand::Rng;

    #[test]
    fn historical_measures_on_a_hand_checked_sample() {
        // losses: 10, 8, 6, 4, 2 and five gains
        let pnl = [-10.0, -8.0, -6.0, -4.0, -2.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let var90 = historical_var(&pnl, 0.9);
        // type-7 quantile at p=0.9 on n=10 sorted losses: index 8.1
        assert!((var90 - 8.2).abs() < 1e-12, "{var90}");
        let es90 = historical_expected_shortfall(&pnl, 0.9);
        assert!(es90 >= var90 && es90 <= 10.0, "{es90}");
        // monotone in confidence
        assert!(historical_var(&pnl, 0.95) >= var90);
    }

    #[test]
    fn historical_converges_to_parametric_on_normal_pnl() {
        let n = 60_000;
        let (mu, sd) = (0.5, 10.0);
        let mut rng = path_rng(7, 0);
        let pnl: Vec<f64> =
            (0..n).map(|_| mu + sd * rng.sample::<f64, _>(rand_distr::StandardNormal)).collect();
        for conf in [0.95, 0.99] {
            let hist = historical_var(&pnl, conf);
            let para = parametric_var(mu, sd, conf);
            assert!((hist - para).abs() < 0.4, "VaR {conf}: {hist} vs {para}");
            let hist_es = historical_expected_shortfall(&pnl, conf);
            let para_es = parametric_expected_shortfall(mu, sd, conf);
            assert!((hist_es - para_es).abs() < 0.5, "ES {conf}: {hist_es} vs {para_es}");
            assert!(hist_es > hist && para_es > para, "ES dominates VaR");
        }
    }

    #[test]
    fn parametric_es_matches_numerical_tail_integration() {
        // ES = E[loss | loss >= VaR] under the normal, by brute quadrature
        let (mu, sd, conf) = (0.0, 1.0, 0.975);
        let es = parametric_expected_shortfall(mu, sd, conf);
        let z = inv_norm_cdf(conf);
        let steps = 400_000;
        let (a, b) = (z, 10.0);
        let h = (b - a) / steps as f64;
        let mut num = 0.0;
        for i in 0..=steps {
            let x = a + i as f64 * h;
            let w = if i == 0 || i == steps { 0.5 } else { 1.0 };
            num += w * x * norm_pdf(x) * h;
        }
        let tail_mean = num / (1.0 - conf);
        assert!((es - tail_mean).abs() < 1e-4, "{es} vs {tail_mean}");
    }

    #[test]
    fn cornish_fisher_reduces_to_normal_and_penalizes_left_skew() {
        let base = parametric_var(0.0, 5.0, 0.99);
        assert!((cornish_fisher_var(0.0, 5.0, 0.0, 0.0, 0.99) - base).abs() < 1e-12);
        // negative skew (long left tail of P&L = big losses) raises VaR;
        // note the skew of LOSSES enters with the P&L sign convention
        let skewed = cornish_fisher_var(0.0, 5.0, -0.8, 0.0, 0.99);
        assert!(skewed > base, "{skewed} vs {base}");
        let fat = cornish_fisher_var(0.0, 5.0, 0.0, 3.0, 0.99);
        assert!(fat > base, "excess kurtosis must raise tail risk");
    }

    #[test]
    fn delta_normal_var_and_euler_decomposition() {
        // two assets, hand-computable: sigma1=2%, sigma2=3%, rho=0.5,
        // exposures 1m and -0.5m
        let cov = vec![
            vec![0.02 * 0.02, 0.5 * 0.02 * 0.03],
            vec![0.5 * 0.02 * 0.03, 0.03 * 0.03],
        ];
        let w = [1_000_000.0, -200_000.0];
        let out = delta_normal_var(&w, &cov, 0.99);
        let variance = w[0] * w[0] * cov[0][0]
            + 2.0 * w[0] * w[1] * cov[0][1]
            + w[1] * w[1] * cov[1][1];
        assert!((out.std - variance.sqrt()).abs() < 1e-6);
        assert!((out.var - variance.sqrt() * inv_norm_cdf(0.99)).abs() < 1e-6);
        // Euler: components sum exactly to the total
        let sum: f64 = out.component_var.iter().sum();
        assert!((sum - out.var).abs() < 1e-6, "{sum} vs {}", out.var);
        // Euler components are marginal-scaling contributions, not
        // leave-one-out: at this size the short position's hedge effect
        // dominates its own variance and its component is negative
        assert!(out.component_var[1] < 0.0, "{:?}", out.component_var);
        // scaled up, the short's own variance takes over and the
        // component turns positive - both signs are legitimate
        let big_short = delta_normal_var(&[1_000_000.0, -500_000.0], &cov, 0.99);
        assert!(big_short.component_var[1] > 0.0);
        let sum2: f64 = big_short.component_var.iter().sum();
        assert!((sum2 - big_short.var).abs() < 1e-6);
    }
}
