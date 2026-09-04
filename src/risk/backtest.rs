//! VaR backtesting: the Kupiec proportion-of-failures (POF) test.

use crate::core::errors::RustyQLibError;

use super::measures::validate_confidence;

/// 95th percentile of the chi-squared distribution with one degree of
/// freedom: the rejection threshold of the Kupiec LR statistic at the
/// 5% test level.
pub const CHI2_1DOF_95: f64 = 3.841_458_820_694_124;

/// Result of a Kupiec POF likelihood-ratio test.
#[derive(Debug, Clone, Copy)]
pub struct KupiecTest {
    /// Observed VaR exceptions.
    pub exceptions: usize,
    pub observations: usize,
    /// Exceptions implied by the VaR level.
    pub expected: f64,
    /// Likelihood-ratio statistic (chi-squared with 1 dof under H0).
    pub lr_statistic: f64,
    /// True when the model is rejected at the 95% test level
    /// (LR > [`CHI2_1DOF_95`]).
    pub rejected: bool,
}

/// Kupiec (1995) unconditional-coverage test of a VaR model:
/// `exceptions` days out of `observations` breached a
/// `confidence`-level VaR. Under a correct model the breach frequency
/// is `1 - confidence`; the LR statistic is asymptotically
/// chi-squared(1). Errors on zero observations, more exceptions than
/// observations, or a confidence outside `(0.5, 1)`.
pub fn kupiec_pof(
    exceptions: usize,
    observations: usize,
    confidence: f64,
) -> Result<KupiecTest, RustyQLibError> {
    if observations == 0 {
        return Err(RustyQLibError::invalid_input(
            "observations",
            "need at least one observation".to_string(),
        ));
    }
    if exceptions > observations {
        return Err(RustyQLibError::invalid_input(
            "exceptions",
            format!("{exceptions} exceptions exceed {observations} observations"),
        ));
    }
    validate_confidence(confidence)?;
    let p = 1.0 - confidence;
    let n = observations as f64;
    let x = exceptions as f64;
    let expected = p * n;
    // log-likelihoods, with the empty-cell conventions 0 ln 0 = 0
    let ll = |prob: f64| -> f64 {
        let mut v = 0.0;
        if x > 0.0 {
            v += x * prob.ln();
        }
        if x < n {
            v += (n - x) * (1.0 - prob).ln();
        }
        v
    };
    let observed_rate = (x / n).clamp(1e-12, 1.0 - 1e-12);
    let lr = -2.0 * (ll(p) - ll(observed_rate));
    Ok(KupiecTest {
        exceptions,
        observations,
        expected,
        lr_statistic: lr,
        rejected: lr > CHI2_1DOF_95,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correct_coverage_passes_and_bad_coverage_fails() {
        // 99% VaR over 1000 days: ~10 exceptions expected
        let ok = kupiec_pof(10, 1000, 0.99).unwrap();
        assert!(!ok.rejected, "{ok:?}");
        assert!(ok.lr_statistic < 0.1);
        // a model that breaches 30 times is rejected
        let bad = kupiec_pof(30, 1000, 0.99).unwrap();
        assert!(bad.rejected, "{bad:?}");
        // suspiciously few exceptions is also evidence against the model
        let conservative = kupiec_pof(0, 1000, 0.99).unwrap();
        assert!(conservative.rejected, "{conservative:?}");
    }

    #[test]
    fn rejects_impossible_counts_and_levels() {
        assert!(kupiec_pof(0, 0, 0.99).is_err(), "no observations");
        assert!(kupiec_pof(11, 10, 0.99).is_err(), "exceptions > n");
        for bad in [0.5, 1.0, f64::NAN] {
            assert!(kupiec_pof(1, 10, bad).is_err(), "confidence {bad}");
        }
        // the threshold constant is the chi2(1) 95% point
        assert!((CHI2_1DOF_95 - 3.841_458_8).abs() < 1e-7);
    }
}
