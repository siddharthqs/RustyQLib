//! Special functions the pricing formulas need beyond the normal
//! distribution: the log-gamma function, the regularized incomplete
//! gamma `P(a, x)` (the chi-square CDF in disguise), and the noncentral
//! chi-square CDF that prices bond options under square-root (CIR)
//! dynamics.

/// `ln Gamma(x)` for `x > 0` (Lanczos, ~1e-15 relative).
pub fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut sum = COEF[0];
    for (i, &c) in COEF.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + G + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// The regularized lower incomplete gamma `P(a, x) = gamma(a, x) / Gamma(a)`
/// for `a > 0`, `x >= 0`: series for `x < a + 1`, Lentz continued
/// fraction otherwise (Numerical Recipes `gammp`). The chi-square CDF
/// with `k` degrees of freedom at `x` is `P(k/2, x/2)`.
pub fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !(a > 0.0) || !x.is_finite() {
        return f64::NAN;
    }
    let ln_prefactor = a * x.ln() - x - ln_gamma(a);
    if x < a + 1.0 {
        // series: sum x^n / (a (a+1) ... (a+n))
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut ap = a;
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        (sum * ln_prefactor.exp()).clamp(0.0, 1.0)
    } else {
        // continued fraction for Q = 1 - P (modified Lentz)
        let tiny = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..10_000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (1.0 - ln_prefactor.exp() * h).clamp(0.0, 1.0)
    }
}

/// The chi-square CDF with `k` degrees of freedom.
pub fn chi_square_cdf(x: f64, k: f64) -> f64 {
    regularized_gamma_p(0.5 * k, 0.5 * x)
}

/// The noncentral chi-square CDF `F(x; k, lambda)` with `k` degrees of
/// freedom and noncentrality `lambda`: the Poisson mixture of central
/// chi-squares `sum_j Pois(j; lambda/2) chi2(x; k + 2j)`, summed
/// outward from the Poisson mode so large noncentralities (the usual
/// case in CIR bond options) converge in `O(sqrt(lambda))` terms.
pub fn noncentral_chi_square_cdf(x: f64, k: f64, lambda: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if !(k > 0.0) || !(lambda >= 0.0) || !x.is_finite() {
        return f64::NAN;
    }
    if lambda == 0.0 {
        return chi_square_cdf(x, k);
    }
    let half = 0.5 * lambda;
    let log_weight = |j: f64| -half + j * half.ln() - ln_gamma(j + 1.0);
    let mode = half.floor();
    let mut total = 0.0;
    // upward from the mode
    let mut j = mode;
    loop {
        let w = log_weight(j).exp();
        let term = w * chi_square_cdf(x, k + 2.0 * j);
        total += term;
        if w < 1e-18 && j > mode {
            break;
        }
        j += 1.0;
        if j > mode + 2_000_000.0 {
            break;
        }
    }
    // downward
    let mut j = mode - 1.0;
    while j >= 0.0 {
        let w = log_weight(j).exp();
        let term = w * chi_square_cdf(x, k + 2.0 * j);
        total += term;
        if w < 1e-18 {
            break;
        }
        j -= 1.0;
    }
    total.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::norm_cdf;

    #[test]
    fn ln_gamma_hits_factorials_and_the_half_integers() {
        for (n, fact) in [(1.0, 1.0), (2.0, 1.0), (5.0, 24.0), (11.0, 3_628_800.0)] {
            assert!(
                (ln_gamma(n) - (fact as f64).ln()).abs() < 1e-12,
                "Gamma({n})"
            );
        }
        // Gamma(1/2) = sqrt(pi)
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-13);
        // large argument against Stirling
        let x = 500.0_f64;
        let stirling =
            (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln() + 1.0 / (12.0 * x);
        assert!((ln_gamma(x) - stirling).abs() < 1e-9);
    }

    #[test]
    fn incomplete_gamma_is_the_chi_square_cdf() {
        // two degrees of freedom: exponential, F(x) = 1 - e^{-x/2}
        for x in [0.1_f64, 1.0, 3.7, 12.0, 40.0] {
            let expected = 1.0 - (-0.5 * x).exp();
            assert!(
                (chi_square_cdf(x, 2.0) - expected).abs() < 1e-14,
                "k=2, x={x}"
            );
        }
        // one degree of freedom: 2 Phi(sqrt x) - 1
        for x in [0.2_f64, 1.0, 4.0, 9.0] {
            let expected = 2.0 * norm_cdf(x.sqrt()) - 1.0;
            assert!(
                (chi_square_cdf(x, 1.0) - expected).abs() < 1e-12,
                "k=1, x={x}"
            );
        }
        // both branches (series / continued fraction) agree at the seam
        let a = 30.0;
        let left = regularized_gamma_p(a, a + 0.999_999);
        let right = regularized_gamma_p(a, a + 1.000_001);
        assert!((left - right).abs() < 1e-6);
        assert_eq!(regularized_gamma_p(2.0, 0.0), 0.0);
        assert!((regularized_gamma_p(2.0, 1e6) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn noncentral_chi_square_matches_the_one_degree_identity() {
        // with k = 1, (Z + sqrt(lambda))^2 <= x  <=>  |Z + sqrt(lambda)| <= sqrt(x)
        for (lambda, x) in [
            (0.5_f64, 1.0_f64),
            (4.0, 3.0),
            (25.0, 30.0),
            (400.0, 420.0),
            (2500.0, 2450.0),
        ] {
            let s = lambda.sqrt();
            let expected = norm_cdf(x.sqrt() - s) - norm_cdf(-x.sqrt() - s);
            let got = noncentral_chi_square_cdf(x, 1.0, lambda);
            assert!(
                (got - expected).abs() < 1e-10,
                "lambda={lambda}, x={x}: {got} vs {expected}"
            );
        }
        // zero noncentrality is the central distribution
        assert!(
            (noncentral_chi_square_cdf(3.0, 4.0, 0.0) - chi_square_cdf(3.0, 4.0)).abs() < 1e-15
        );
        // the mean is k + lambda: the CDF there sits near one half and
        // grows monotonically in x
        let (k, lambda) = (6.0_f64, 50.0_f64);
        let mid = noncentral_chi_square_cdf(k + lambda, k, lambda);
        assert!(mid > 0.45 && mid < 0.55, "{mid}");
        let mut previous = 0.0;
        for i in 1..40 {
            let f = noncentral_chi_square_cdf(5.0 * i as f64, k, lambda);
            assert!(f >= previous);
            previous = f;
        }
        assert!((previous - 1.0).abs() < 1e-9);
    }
}
