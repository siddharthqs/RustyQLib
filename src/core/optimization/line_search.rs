//! Backtracking Armijo line search, shared by the gradient-descent
//! family (steepest descent, conjugate gradient, BFGS).

/// Walk back from `alpha0` along `dir` until the Armijo sufficient-
/// decrease condition `f(x + a d) <= fx + c1 a g.d` holds. Returns
/// `(x_new, f_new)`, or `None` when no acceptable step exists down to
/// machine scale (a signal to stop or reset the direction).
pub(crate) fn backtracking(
    f: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    fx: f64,
    dir: &[f64],
    slope: f64,
    alpha0: f64,
) -> Option<(Vec<f64>, f64)> {
    const C1: f64 = 1e-4;
    let mut alpha = alpha0;
    while alpha > 1e-16 {
        let x_new: Vec<f64> = x.iter().zip(dir).map(|(xi, di)| xi + alpha * di).collect();
        let f_new = f(&x_new);
        if f_new <= fx + C1 * alpha * slope {
            return Some((x_new, f_new));
        }
        alpha *= 0.5;
    }
    None
}
