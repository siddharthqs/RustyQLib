//! Shared calibration plumbing for the parametric models: the generic
//! fit outcome every calibrate returns, the unconstrained-transform
//! contract, and the COS-based Levenberg-Marquardt driver the
//! characteristic-function models (Heston, both Bates variants) run
//! through. One implementation instead of a hand-rolled copy per model.

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

/// Levenberg-Marquardt calibration of a characteristic-function model
/// to European vanilla quotes, run in the parameter set's
/// [`TransformSpace`]. One COS pricer (one CF sweep) per expiry per
/// residual evaluation: the whole smile prices for the cost of one
/// option.
pub(crate) fn calibrate_generic<P: TransformSpace>(
    quotes: &[HestonQuote],
    start: &P,
    r: f64,
    tol: f64,
    cf: impl Fn(&P, Cpx, f64) -> Cpx,
) -> Fit<P> {
    use crate::core::optimization::{levenberg_marquardt, OptimConfig};
    assert!(!quotes.is_empty(), "calibration needs at least one quote");
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
    Fit {
        params: P::from_unconstrained(&fit.x),
        rmse: (fit.value / quotes.len() as f64).sqrt(),
        iterations: fit.iterations,
        converged: fit.converged,
    }
}
