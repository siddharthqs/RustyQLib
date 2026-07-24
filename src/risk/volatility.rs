//! Volatility estimation from return series: realized (close-to-close)
//! and EWMA (RiskMetrics).

/// Annualized realized volatility of a per-period return series
/// (sample standard deviation, mean removed).
pub fn realized_volatility(returns: &[f64], periods_per_year: f64) -> f64 {
    let n = returns.len();
    assert!(n >= 2, "need at least two returns");
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n as f64 - 1.0);
    (var * periods_per_year).sqrt()
}

/// EWMA (RiskMetrics) volatility: `sigma_t^2 = lambda sigma_{t-1}^2 +
/// (1 - lambda) r_t^2`, seeded with the first squared return. Returns
/// the **annualized** latest estimate; `lambda = 0.94` is the classic
/// daily-decay choice.
pub fn ewma_volatility(returns: &[f64], lambda: f64, periods_per_year: f64) -> f64 {
    assert!(!returns.is_empty());
    assert!((0.0..1.0).contains(&lambda));
    let mut variance = returns[0] * returns[0];
    for &r in &returns[1..] {
        variance = lambda * variance + (1.0 - lambda) * r * r;
    }
    (variance * periods_per_year).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn realized_vol_recovers_the_generating_sigma() {
        use crate::core::montecarlo::path_rng;
        use rand::Rng;
        let daily = 0.2 / 252.0_f64.sqrt();
        let mut rng = path_rng(3, 0);
        let returns: Vec<f64> = (0..20_000)
            .map(|_| daily * rng.sample::<f64, _>(rand_distr::StandardNormal))
            .collect();
        let vol = realized_volatility(&returns, 252.0);
        assert!((vol - 0.2).abs() < 0.005, "{vol}");
        let ewma = ewma_volatility(&returns, 0.94, 252.0);
        assert!((ewma - 0.2).abs() < 0.05, "{ewma}");
    }

    #[test]
    fn ewma_recursion_matches_a_hand_computation() {
        let returns = [0.01, -0.02, 0.015];
        let lambda = 0.9;
        let v1 = 0.01f64 * 0.01;
        let v2 = lambda * v1 + 0.1 * 0.02 * 0.02;
        let v3 = lambda * v2 + 0.1 * 0.015 * 0.015;
        let expect = (v3 * 252.0).sqrt();
        assert!((ewma_volatility(&returns, lambda, 252.0) - expect).abs() < 1e-12);
        // constant series: realized vol is zero (mean removed), EWMA is not
        let flat = [0.01; 10];
        assert!(realized_volatility(&flat, 252.0).abs() < 1e-15);
        assert!(ewma_volatility(&flat, 0.94, 252.0) > 0.0);
    }
}
