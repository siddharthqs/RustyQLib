//! Performance and path-risk statistics: drawdowns and risk-adjusted
//! return ratios.

use crate::core::errors::RustyQLibError;

use super::measures::validate_sample;

/// Reject a `periods_per_year` scaling that is not finite and positive.
fn validate_periods(periods_per_year: f64) -> Result<(), RustyQLibError> {
    if !(periods_per_year.is_finite() && periods_per_year > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "periods_per_year",
            format!("must be finite and positive, got {periods_per_year}"),
        ));
    }
    Ok(())
}

/// Reject a return series too short for a ratio, or a non-finite
/// risk-free rate.
fn validate_returns(returns: &[f64], risk_free_per_period: f64) -> Result<(), RustyQLibError> {
    validate_sample("returns", returns)?;
    if returns.len() < 2 {
        return Err(RustyQLibError::invalid_input(
            "returns",
            format!("need at least two returns, got {}", returns.len()),
        ));
    }
    if !risk_free_per_period.is_finite() {
        return Err(RustyQLibError::invalid_input(
            "risk_free_per_period",
            format!("must be finite, got {risk_free_per_period}"),
        ));
    }
    Ok(())
}

/// Maximum drawdown of a **NAV (value) series** — every value finite
/// and strictly positive — as a positive fraction of the running peak
/// (0.25 = a 25% peak-to-trough fall), with the peak and trough indices.
/// Errors on an empty series or one with a value that is not finite and
/// positive (a zero or negative NAV has no percentage drawdown).
pub fn max_drawdown(values: &[f64]) -> Result<(f64, usize, usize), RustyQLibError> {
    validate_sample("values", values)?;
    if let Some((i, v)) = values.iter().enumerate().find(|(_, v)| **v <= 0.0) {
        return Err(RustyQLibError::invalid_input(
            "values",
            format!("NAV series must be strictly positive, values[{i}] = {v}"),
        ));
    }
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
    Ok((best, best_pair.0, best_pair.1))
}

/// Annualized Sharpe ratio of per-period returns against a per-period
/// risk-free rate. Errors on fewer than two returns, a non-finite
/// input, or a zero-variance series (the ratio is undefined, not
/// infinite).
pub fn sharpe_ratio(
    returns: &[f64],
    risk_free_per_period: f64,
    periods_per_year: f64,
) -> Result<f64, RustyQLibError> {
    validate_returns(returns, risk_free_per_period)?;
    validate_periods(periods_per_year)?;
    let n = returns.len();
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free_per_period).collect();
    let mean = excess.iter().sum::<f64>() / n as f64;
    let var = excess.iter().map(|e| (e - mean) * (e - mean)).sum::<f64>() / (n as f64 - 1.0);
    if var <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "returns",
            "the excess-return series has zero variance; the Sharpe ratio is undefined"
                .to_string(),
        ));
    }
    Ok(mean / var.sqrt() * periods_per_year.sqrt())
}

/// Annualized Sortino ratio: excess return over the downside deviation
/// (root mean square of returns below the risk-free rate). Errors on
/// fewer than two returns, a non-finite input, or a series with no
/// return below the risk-free rate (zero downside deviation).
pub fn sortino_ratio(
    returns: &[f64],
    risk_free_per_period: f64,
    periods_per_year: f64,
) -> Result<f64, RustyQLibError> {
    validate_returns(returns, risk_free_per_period)?;
    validate_periods(periods_per_year)?;
    let n = returns.len();
    let mean_excess = returns
        .iter()
        .map(|r| r - risk_free_per_period)
        .sum::<f64>()
        / n as f64;
    let downside_sq = returns
        .iter()
        .map(|r| (r - risk_free_per_period).min(0.0).powi(2))
        .sum::<f64>()
        / n as f64;
    if downside_sq <= 0.0 {
        return Err(RustyQLibError::invalid_input(
            "returns",
            "no return below the risk-free rate; the Sortino ratio is undefined".to_string(),
        ));
    }
    Ok(mean_excess / downside_sq.sqrt() * periods_per_year.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drawdown_finds_the_peak_to_trough() {
        let nav = [100.0, 110.0, 105.0, 120.0, 90.0, 95.0, 130.0];
        let (dd, peak, trough) = max_drawdown(&nav).unwrap();
        assert!((dd - 0.25).abs() < 1e-12, "{dd}"); // 120 -> 90
        assert_eq!((peak, trough), (3, 4));
        // monotone series has zero drawdown
        assert_eq!(max_drawdown(&[1.0, 2.0, 3.0]).unwrap().0, 0.0);
    }

    #[test]
    fn drawdown_rejects_non_nav_series() {
        assert!(max_drawdown(&[]).is_err(), "empty");
        for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            let err = max_drawdown(&[100.0, bad, 110.0]).unwrap_err();
            assert!(err.to_string().contains("values"), "{bad}: {err}");
        }
    }

    #[test]
    fn ratios_are_hand_checkable_and_ordered() {
        // symmetric returns: sharpe and sortino positive, sortino larger
        // (only half the deviation is downside)
        let returns = [0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025, 0.01];
        let sharpe = sharpe_ratio(&returns, 0.0, 252.0).unwrap();
        let sortino = sortino_ratio(&returns, 0.0, 252.0).unwrap();
        assert!(sharpe > 0.0 && sortino > sharpe, "{sharpe} vs {sortino}");
        // scaling returns leaves sharpe unchanged
        let scaled: Vec<f64> = returns.iter().map(|r| r * 3.0).collect();
        assert!((sharpe_ratio(&scaled, 0.0, 252.0).unwrap() - sharpe).abs() < 1e-12);
    }

    #[test]
    fn ratios_reject_degenerate_inputs() {
        // too short
        assert!(sharpe_ratio(&[0.01], 0.0, 252.0).is_err());
        assert!(sortino_ratio(&[0.01], 0.0, 252.0).is_err());
        // zero variance used to be +-inf / NaN; no downside used to panic
        let flat = [0.01; 5];
        let err = sharpe_ratio(&flat, 0.0, 252.0).unwrap_err();
        assert!(err.to_string().contains("zero variance"), "{err}");
        let err = sortino_ratio(&flat, 0.0, 252.0).unwrap_err();
        assert!(err.to_string().contains("below the risk-free"), "{err}");
        // non-finite anywhere
        let poisoned = [0.01, f64::NAN, -0.02];
        assert!(sharpe_ratio(&poisoned, 0.0, 252.0).is_err());
        assert!(sortino_ratio(&poisoned, 0.0, 252.0).is_err());
        let ok = [0.02, -0.01, 0.03];
        assert!(sharpe_ratio(&ok, f64::NAN, 252.0).is_err());
        assert!(sharpe_ratio(&ok, 0.0, 0.0).is_err());
        assert!(sortino_ratio(&ok, 0.0, f64::INFINITY).is_err());
    }
}
