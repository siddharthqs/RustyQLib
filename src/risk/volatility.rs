//! Volatility estimation from return series: realized (close-to-close)
//! and EWMA (RiskMetrics).

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

/// Annualized realized volatility of a per-period return series
/// (sample standard deviation, mean removed). Errors on fewer than two
/// returns or a non-finite input.
pub fn realized_volatility(returns: &[f64], periods_per_year: f64) -> Result<f64, RustyQLibError> {
    validate_sample("returns", returns)?;
    let n = returns.len();
    if n < 2 {
        return Err(RustyQLibError::invalid_input(
            "returns",
            format!("need at least two returns, got {n}"),
        ));
    }
    validate_periods(periods_per_year)?;
    let mean = returns.iter().sum::<f64>() / n as f64;
    let var = returns.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / (n as f64 - 1.0);
    Ok((var * periods_per_year).sqrt())
}

/// EWMA (RiskMetrics) volatility: `sigma_t^2 = lambda sigma_{t-1}^2 +
/// (1 - lambda) r_t^2`, seeded with the first squared return. Returns
/// the **annualized** latest estimate; `lambda = 0.94` is the classic
/// daily-decay choice. Errors on an empty or non-finite series, or a
/// `lambda` outside `[0, 1)`.
pub fn ewma_volatility(
    returns: &[f64],
    lambda: f64,
    periods_per_year: f64,
) -> Result<f64, RustyQLibError> {
    validate_sample("returns", returns)?;
    if !(0.0..1.0).contains(&lambda) {
        return Err(RustyQLibError::invalid_input(
            "lambda",
            format!("must be in [0, 1), got {lambda}"),
        ));
    }
    validate_periods(periods_per_year)?;
    let mut variance = returns[0] * returns[0];
    for &r in &returns[1..] {
        variance = lambda * variance + (1.0 - lambda) * r * r;
    }
    Ok((variance * periods_per_year).sqrt())
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
        let vol = realized_volatility(&returns, 252.0).unwrap();
        assert!((vol - 0.2).abs() < 0.005, "{vol}");
        let ewma = ewma_volatility(&returns, 0.94, 252.0).unwrap();
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
        assert!((ewma_volatility(&returns, lambda, 252.0).unwrap() - expect).abs() < 1e-12);
        // constant series: realized vol is zero (mean removed), EWMA is not
        let flat = [0.01; 10];
        assert!(realized_volatility(&flat, 252.0).unwrap().abs() < 1e-15);
        assert!(ewma_volatility(&flat, 0.94, 252.0).unwrap() > 0.0);
    }

    #[test]
    fn estimators_reject_bad_inputs() {
        assert!(realized_volatility(&[], 252.0).is_err());
        assert!(realized_volatility(&[0.01], 252.0).is_err(), "one return");
        assert!(realized_volatility(&[0.01, f64::NAN], 252.0).is_err());
        assert!(realized_volatility(&[0.01, 0.02], 0.0).is_err());
        assert!(ewma_volatility(&[], 0.94, 252.0).is_err());
        assert!(ewma_volatility(&[0.01, f64::INFINITY], 0.94, 252.0).is_err());
        for lambda in [-0.1, 1.0, 1.5, f64::NAN] {
            let err = ewma_volatility(&[0.01, 0.02], lambda, 252.0).unwrap_err();
            assert!(err.to_string().contains("lambda"), "{lambda}: {err}");
        }
        assert!(ewma_volatility(&[0.01], 0.94, f64::NAN).is_err());
    }
}
