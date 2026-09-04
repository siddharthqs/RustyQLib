//! Minimal complex arithmetic in rectangular form — the one complex type
//! of the numerical core, shared by the FFT ([`fft`](crate::core::fft)),
//! the characteristic-function pricers (Heston, Bates, COS, Carr-Madan)
//! and the rough Bergomi hybrid scheme.
//!
//! Transcendental functions take their principal branches (`ln` and
//! `sqrt` cut along the negative real axis), which is what the
//! trap-free Heston characteristic function relies on. Both operator
//! syntax (`a + b`, `a * b`) and the method set (`a.add(b)`, `a.mul(b)`)
//! are provided; they evaluate the identical expressions.

/// Complex number in rectangular form.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// A real number as a complex one.
    pub fn real(re: f64) -> Complex {
        Complex { re, im: 0.0 }
    }

    /// `e^{i theta}` on the unit circle.
    pub fn cis(theta: f64) -> Complex {
        Complex {
            re: theta.cos(),
            im: theta.sin(),
        }
    }

    pub fn conj(self) -> Complex {
        Complex {
            re: self.re,
            im: -self.im,
        }
    }

    // the arithmetic methods mirror the `std::ops` impls below (same
    // expressions) for call chains that read better as methods
    pub(crate) fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }

    pub(crate) fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }

    pub(crate) fn mul(self, o: Complex) -> Complex {
        Complex::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    pub(crate) fn div(self, o: Complex) -> Complex {
        let denom = o.re * o.re + o.im * o.im;
        Complex::new(
            (self.re * o.re + self.im * o.im) / denom,
            (self.im * o.re - self.re * o.im) / denom,
        )
    }

    /// Multiply by a real scalar.
    pub fn scale(self, x: f64) -> Complex {
        Complex::new(self.re * x, self.im * x)
    }

    pub fn exp(self) -> Complex {
        let m = self.re.exp();
        Complex::new(m * self.im.cos(), m * self.im.sin())
    }

    /// Principal logarithm.
    pub fn ln(self) -> Complex {
        Complex::new(self.norm().ln(), self.im.atan2(self.re))
    }

    /// Principal square root.
    pub fn sqrt(self) -> Complex {
        let m = self.norm().sqrt();
        let half_arg = 0.5 * self.im.atan2(self.re);
        Complex::new(m * half_arg.cos(), m * half_arg.sin())
    }

    /// Modulus `|z|`.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for Complex {
    type Output = Complex;
    fn add(self, o: Complex) -> Complex {
        Complex::add(self, o)
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    fn sub(self, o: Complex) -> Complex {
        Complex::sub(self, o)
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    fn mul(self, o: Complex) -> Complex {
        Complex::mul(self, o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_arithmetic_sanity() {
        let z = Complex::new(3.0, 4.0);
        assert!((z.norm() - 5.0).abs() < 1e-14);
        let e = Complex::new(0.0, std::f64::consts::PI).exp();
        assert!(
            (e.re + 1.0).abs() < 1e-12 && e.im.abs() < 1e-12,
            "e^{{i pi}} = -1"
        );
        let s = Complex::new(-1.0, 0.0).sqrt();
        assert!(
            s.re.abs() < 1e-12 && (s.im - 1.0).abs() < 1e-12,
            "sqrt(-1) = i"
        );
        let l = z.ln().exp();
        assert!((l.re - z.re).abs() < 1e-12 && (l.im - z.im).abs() < 1e-12);
    }

    #[test]
    fn operators_and_methods_agree_exactly() {
        let a = Complex::new(1.25, -0.5);
        let b = Complex::new(-2.0, 3.75);
        assert_eq!(a + b, a.add(b));
        assert_eq!(a - b, a.sub(b));
        assert_eq!(a * b, a.mul(b));
        // division against the multiplicative inverse
        let q = a.div(b);
        let back = q.mul(b);
        assert!((back.re - a.re).abs() < 1e-14 && (back.im - a.im).abs() < 1e-14);
        assert_eq!(Complex::real(2.0), Complex::new(2.0, 0.0));
        assert_eq!(a.conj(), Complex::new(1.25, 0.5));
        let c = Complex::cis(0.3);
        assert!((c.norm() - 1.0).abs() < 1e-15);
        assert_eq!(Complex::default(), Complex::ZERO);
    }
}
