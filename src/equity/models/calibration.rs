//! Shared calibration plumbing for the parametric models: the generic
//! fit outcome every calibrate returns, the unconstrained-transform
//! contract, and the COS-based Levenberg-Marquardt driver the
//! characteristic-function models (Heston, both Bates variants) run
//! through. One implementation instead of a hand-rolled copy per model.

use crate::core::errors::RustyQLibError;
use crate::equity::cos::{group_by_maturity, CosPricer, CALIBRATION_TERMS};
use crate::equity::heston::{Cpx, HestonQuote};

/// Calibration outcome: fitted parameters plus fit diagnostics.
///
/// The `rmse` unit is the residual unit of the model's calibrate —
/// price for the characteristic-function models, implied vol for the
/// smile parameterizations (see each `calibrate`'s doc).
#[derive(Debug, Clone)]
pub struct Fit<P> {
    pub params: P,
    /// Root-mean-square error over the quotes.
    pub rmse: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// The unconstrained transform space a parameter set calibrates in
/// (`ln` for positive parameters, `atanh` for correlations, logit for
/// probabilities): every point of the space maps to admissible
/// parameters, so the optimizer never needs constraints.
pub(crate) trait TransformSpace: Sized {
    fn to_unconstrained(&self) -> Vec<f64>;
    /// Inverse map; must accept any `u` of the right length.
    fn from_unconstrained(u: &[f64]) -> Self;
}

/// Market-input gate shared by the characteristic-function calibrations:
/// at least one quote, every strike and maturity finite and positive,
/// every price finite and non-negative. A bad quote would otherwise
/// poison the residual vector silently (a NaN price makes every
/// residual NaN; a zero maturity degenerates the COS truncation range).
pub(crate) fn check_quotes(quotes: &[HestonQuote]) -> Result<(), RustyQLibError> {
    if quotes.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "calibration quotes",
            "calibration needs at least one quote",
        ));
    }
    for (i, q) in quotes.iter().enumerate() {
        if !(q.strike.is_finite() && q.strike > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "calibration quotes",
                format!("quote {i}: strike must be finite and positive (got {})", q.strike),
            ));
        }
        if !(q.maturity.is_finite() && q.maturity > 0.0) {
            return Err(RustyQLibError::invalid_input(
                "calibration quotes",
                format!(
                    "quote {i}: maturity must be finite and positive (got {})",
                    q.maturity
                ),
            ));
        }
        if !(q.price.is_finite() && q.price >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "calibration quotes",
                format!(
                    "quote {i}: price must be finite and non-negative (got {})",
                    q.price
                ),
            ));
        }
    }
    Ok(())
}

/// Levenberg-Marquardt calibration of a characteristic-function model
/// to European vanilla quotes, run in the parameter set's
/// [`TransformSpace`]. One COS pricer (one CF sweep) per expiry per
/// residual evaluation: the whole smile prices for the cost of one
/// option.
///
/// The quotes are validated first ([`check_quotes`]); the caller
/// validates `start`. Invalid input is a returned
/// [`invalid_input`](RustyQLibError::invalid_input) error.
pub(crate) fn calibrate_generic<P: TransformSpace>(
    quotes: &[HestonQuote],
    start: &P,
    r: f64,
    tol: f64,
    cf: impl Fn(&P, Cpx, f64) -> Cpx,
) -> Result<Fit<P>, RustyQLibError> {
    use crate::core::optimization::{levenberg_marquardt, OptimConfig};
    check_quotes(quotes)?;
    let groups = group_by_maturity(quotes.iter().map(|q| q.maturity));
    let residuals = |u: &[f64]| -> Vec<f64> {
        let p = P::from_unconstrained(u);
        let mut out = vec![0.0; quotes.len()];
        for (t, idxs) in &groups {
            let pricer = CosPricer::new(&|uu| cf(&p, uu, *t), r, *t, CALIBRATION_TERMS);
            for &i in idxs {
                out[i] = pricer.price(quotes[i].strike, quotes[i].put_or_call) - quotes[i].price;
            }
        }
        out
    };
    let fit = levenberg_marquardt(
        &OptimConfig::new(tol, 100),
        &residuals,
        None,
        &start.to_unconstrained(),
    );
    Ok(Fit {
        params: P::from_unconstrained(&fit.x),
        rmse: (fit.value / quotes.len() as f64).sqrt(),
        iterations: fit.iterations,
        converged: fit.converged,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;

    fn quote(strike: f64, maturity: f64, price: f64) -> HestonQuote {
        HestonQuote {
            strike,
            maturity,
            price,
            put_or_call: PutOrCall::Call,
        }
    }

    #[test]
    fn quote_gate_rejects_bad_market_data() {
        assert!(check_quotes(&[]).is_err());
        let good = [quote(100.0, 1.0, 8.0), quote(110.0, 0.5, 2.0)];
        assert!(check_quotes(&good).is_ok());
        for bad in [
            quote(0.0, 1.0, 8.0),
            quote(-5.0, 1.0, 8.0),
            quote(f64::NAN, 1.0, 8.0),
            quote(f64::INFINITY, 1.0, 8.0),
            quote(100.0, 0.0, 8.0),
            quote(100.0, -0.5, 8.0),
            quote(100.0, f64::NAN, 8.0),
            quote(100.0, f64::INFINITY, 8.0),
            quote(100.0, 1.0, -0.01),
            quote(100.0, 1.0, f64::NAN),
            quote(100.0, 1.0, f64::INFINITY),
        ] {
            assert!(
                check_quotes(&[good[0], bad]).is_err(),
                "should reject {bad:?}"
            );
        }
        // a zero price is a legitimate (deep OTM) quote
        assert!(check_quotes(&[quote(500.0, 0.1, 0.0)]).is_ok());
    }
}
