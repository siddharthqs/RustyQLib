//! Performance and path-risk statistics: drawdowns and risk-adjusted
//! return ratios.

/// Maximum drawdown of a value series, as a positive fraction of the
/// running peak (0.25 = a 25% peak-to-trough fall), with the peak and
/// trough indices.
pub fn max_drawdown(values: &[f64]) -> (f64, usize, usize) {
    assert!(!values.is_empty());
    let mut peak = values[0];
    let mut peak_idx = 0;
    let mut best = 0.0;
    let mut best_pair = (0, 0);
    for (i, &v) in values.iter().enumerate() {
        if v > peak {
            peak = v;
            peak_idx = i;
        }
        let dd = (peak - v) / peak;
        if dd > best {
            best = dd;
            best_pair = (peak_idx, i);
        }
    }
    (best, best_pair.0, best_pair.1)
}

/// Annualized Sharpe ratio of per-period returns against a per-period
/// risk-free rate.
pub fn sharpe_ratio(returns: &[f64], risk_free_per_period: f64, periods_per_year: f64) -> f64 {
    let n = returns.len();
    assert!(n >= 2);
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free_per_period).collect();
    let mean = excess.iter().sum::<f64>() / n as f64;
    let var = excess.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (n as f64 - 1.0);
    mean / var.sqrt() * periods_per_year.sqrt()
}

/// Annualized Sortino ratio: excess return over the downside deviation
/// (root mean square of returns below the risk-free rate).
pub fn sortino_ratio(returns: &[f64], risk_free_per_period: f64, periods_per_year: f64) -> f64 {
    let n = returns.len();
    assert!(n >= 2);
    let mean_excess =
        returns.iter().map(|r| r - risk_free_per_period).sum::<f64>() / n as f64;
    let downside_sq = returns
        .iter()
        .map(|r| (r - risk_free_per_period).min(0.0).powi(2))
        .sum::<f64>()
        / n as f64;
    assert!(downside_sq > 0.0, "no downside observations");
    mean_excess / downside_sq.sqrt() * periods_per_year.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawdown_finds_the_peak_to_trough() {
        let nav = [100.0, 110.0, 105.0, 120.0, 90.0, 95.0, 130.0];
        let (dd, peak, trough) = max_drawdown(&nav);
        assert!((dd - 0.25).abs() < 1e-12, "{dd}"); // 120 -> 90
        assert_eq!((peak, trough), (3, 4));
        // monotone series has zero drawdown
        assert_eq!(max_drawdown(&[1.0, 2.0, 3.0]).0, 0.0);
    }

    #[test]
    fn ratios_are_hand_checkable_and_ordered() {
        // symmetric returns: sharpe and sortino positive, sortino larger
        // (only half the deviation is downside)
        let returns = [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.0, 252.0);
        let sortino = sortino_ratio(&returns, 0.0, 252.0);
        assert!(sharpe > 0.0 && sortino > sharpe, "{sharpe} vs {sortino}");
        // scaling returns leaves sharpe unchanged
        let scaled: Vec<f64> = returns.iter().map(|r| r * 3.0).collect();
        assert!((sharpe_ratio(&scaled, 0.0, 252.0) - sharpe).abs() < 1e-12);
    }
}
