//! Shared one-dimensional quadrature rules.

/// Composite Simpson's rule for `int_a^b f(x) dx` on `n` uniform
/// intervals.
///
/// Simpson needs an even interval count; an odd `n` is rounded **up**
/// to `n + 1` (so the requested resolution is never lost). Every caller
/// in the library already passes an even `n`, for which the rule is
/// exactly the textbook `h/3 [f_0 + 4 f_1 + 2 f_2 + ... + 4 f_{n-1} + f_n]`.
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let w = if i % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(a + i as f64 * h);
    }
    sum * h / 3.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrates_polynomials_up_to_cubic_exactly() {
        // Simpson is exact for cubics on any even grid
        let f = |x: f64| 2.0 * x * x * x - x * x + 3.0 * x - 1.0;
        let exact = |x: f64| 0.5 * x.powi(4) - x.powi(3) / 3.0 + 1.5 * x * x - x;
        let got = simpson(f, -1.0, 2.0, 4);
        assert!((got - (exact(2.0) - exact(-1.0))).abs() < 1e-12);
    }

    #[test]
    fn odd_interval_counts_round_up() {
        // n = 5 must behave exactly like n = 6
        let f = |x: f64| (x * 1.3).sin() + x.exp();
        assert_eq!(simpson(f, 0.0, 1.5, 5), simpson(f, 0.0, 1.5, 6));
    }

    #[test]
    fn converges_on_a_smooth_integrand() {
        let got = simpson(|x: f64| x.exp(), 0.0, 1.0, 64);
        assert!((got - (std::f64::consts::E - 1.0)).abs() < 1e-9);
    }
}
