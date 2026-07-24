//! The differentiable value type: `Var` records every operation on the
//! tape via operator overloading, so pricing code written over `Var`
//! looks like ordinary arithmetic.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::core::utils::{norm_pdf, norm_cdf};

use super::tape::{Gradients, Tape};

/// A value recorded on an AAD [`Tape`]. Copyable and cheap; all
/// arithmetic allocates one tape node.
#[derive(Debug, Clone, Copy)]
pub struct Var<'a> {
    pub(crate) tape: &'a Tape,
    pub(crate) idx: usize,
    pub(crate) val: f64,
}

impl<'a> Var<'a> {
    pub fn value(self) -> f64 {
        self.val
    }

    /// Backward sweep: the gradient of `self` with respect to every
    /// variable on the tape (query via [`Gradients::wrt`]).
    pub fn grad(self) -> Gradients {
        Gradients { adjoints: self.tape.backward(self.idx) }
    }

    fn unary(self, val: f64, partial: f64) -> Var<'a> {
        Var { tape: self.tape, idx: self.tape.push1(self.idx, partial), val }
    }

    pub fn exp(self) -> Var<'a> {
        let e = self.val.exp();
        self.unary(e, e)
    }

    pub fn ln(self) -> Var<'a> {
        self.unary(self.val.ln(), 1.0 / self.val)
    }

    pub fn sqrt(self) -> Var<'a> {
        let s = self.val.sqrt();
        self.unary(s, 0.5 / s)
    }

    pub fn powf(self, n: f64) -> Var<'a> {
        self.unary(self.val.powf(n), n * self.val.powf(n - 1.0))
    }

    pub fn sin(self) -> Var<'a> {
        self.unary(self.val.sin(), self.val.cos())
    }

    pub fn cos(self) -> Var<'a> {
        self.unary(self.val.cos(), -self.val.sin())
    }

    /// Standard normal CDF (derivative: the density).
    pub fn norm_cdf(self) -> Var<'a> {
        self.unary(norm_cdf(self.val), norm_pdf(self.val))
    }

    /// `max(self, other)` with the one-sided subgradient at ties.
    pub fn max(self, other: Var<'a>) -> Var<'a> {
        if self.val >= other.val {
            Var {
                tape: self.tape,
                idx: self.tape.push2(self.idx, 1.0, other.idx, 0.0),
                val: self.val,
            }
        } else {
            Var {
                tape: self.tape,
                idx: self.tape.push2(self.idx, 0.0, other.idx, 1.0),
                val: other.val,
            }
        }
    }

    /// `max(self, constant)` — the positive-part operator for payoffs
    /// (`x.maxf(0.0)`), differentiable almost everywhere.
    pub fn maxf(self, c: f64) -> Var<'a> {
        if self.val >= c {
            self.unary(self.val, 1.0)
        } else {
            self.unary(c, 0.0)
        }
    }
}

impl<'a> Add for Var<'a> {
    type Output = Var<'a>;
    fn add(self, rhs: Var<'a>) -> Var<'a> {
        Var {
            tape: self.tape,
            idx: self.tape.push2(self.idx, 1.0, rhs.idx, 1.0),
            val: self.val + rhs.val,
        }
    }
}

impl<'a> Sub for Var<'a> {
    type Output = Var<'a>;
    fn sub(self, rhs: Var<'a>) -> Var<'a> {
        Var {
            tape: self.tape,
            idx: self.tape.push2(self.idx, 1.0, rhs.idx, -1.0),
            val: self.val - rhs.val,
        }
    }
}

impl<'a> Mul for Var<'a> {
    type Output = Var<'a>;
    fn mul(self, rhs: Var<'a>) -> Var<'a> {
        Var {
            tape: self.tape,
            idx: self.tape.push2(self.idx, rhs.val, rhs.idx, self.val),
            val: self.val * rhs.val,
        }
    }
}

impl<'a> Div for Var<'a> {
    type Output = Var<'a>;
    fn div(self, rhs: Var<'a>) -> Var<'a> {
        let v = self.val / rhs.val;
        Var {
            tape: self.tape,
            idx: self.tape.push2(self.idx, 1.0 / rhs.val, rhs.idx, -v / rhs.val),
            val: v,
        }
    }
}

impl<'a> Neg for Var<'a> {
    type Output = Var<'a>;
    fn neg(self) -> Var<'a> {
        self.unary(-self.val, -1.0)
    }
}

// mixed Var / f64 arithmetic
impl<'a> Add<f64> for Var<'a> {
    type Output = Var<'a>;
    fn add(self, rhs: f64) -> Var<'a> {
        self.unary(self.val + rhs, 1.0)
    }
}

impl<'a> Add<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn add(self, rhs: Var<'a>) -> Var<'a> {
        rhs + self
    }
}

impl<'a> Sub<f64> for Var<'a> {
    type Output = Var<'a>;
    fn sub(self, rhs: f64) -> Var<'a> {
        self.unary(self.val - rhs, 1.0)
    }
}

impl<'a> Sub<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn sub(self, rhs: Var<'a>) -> Var<'a> {
        rhs.unary(self - rhs.val, -1.0)
    }
}

impl<'a> Mul<f64> for Var<'a> {
    type Output = Var<'a>;
    fn mul(self, rhs: f64) -> Var<'a> {
        self.unary(self.val * rhs, rhs)
    }
}

impl<'a> Mul<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn mul(self, rhs: Var<'a>) -> Var<'a> {
        rhs * self
    }
}

impl<'a> Div<f64> for Var<'a> {
    type Output = Var<'a>;
    fn div(self, rhs: f64) -> Var<'a> {
        self.unary(self.val / rhs, 1.0 / rhs)
    }
}

impl<'a> Div<Var<'a>> for f64 {
    type Output = Var<'a>;
    fn div(self, rhs: Var<'a>) -> Var<'a> {
        rhs.unary(self / rhs.val, -self / (rhs.val * rhs.val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_rules_match_calculus() {
        let tape = Tape::new();
        let x = tape.var(1.3);
        let y = tape.var(0.7);
        // f = x*y + sin(x) + x/y
        let f = x * y + x.sin() + x / y;
        let g = f.grad();
        assert!((g.wrt(x) - (0.7 + 1.3f64.cos() + 1.0 / 0.7)).abs() < 1e-14);
        assert!((g.wrt(y) - (1.3 - 1.3 / (0.7 * 0.7))).abs() < 1e-14);
    }

    #[test]
    fn fan_out_accumulates_adjoints() {
        // x used twice: d(x*x)/dx = 2x
        let tape = Tape::new();
        let x = tape.var(3.0);
        let g = (x * x).grad();
        assert!((g.wrt(x) - 6.0).abs() < 1e-14);
        // deep chain: exp(ln(sqrt(x^4))) = x^2
        let h = x.powf(4.0).sqrt().ln().exp().grad();
        assert!((h.wrt(x) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn composite_matches_finite_differences() {
        let f_val = |x: f64, y: f64| ((x * y).exp() + (x / y).sqrt()).ln() * y.cos();
        let tape = Tape::new();
        let x = tape.var(0.8);
        let y = tape.var(1.9);
        let f = ((x * y).exp() + (x / y).sqrt()).ln() * y.cos();
        assert!((f.value() - f_val(0.8, 1.9)).abs() < 1e-14);
        let g = f.grad();
        let h = 1e-6;
        let fd_x = (f_val(0.8 + h, 1.9) - f_val(0.8 - h, 1.9)) / (2.0 * h);
        let fd_y = (f_val(0.8, 1.9 + h) - f_val(0.8, 1.9 - h)) / (2.0 * h);
        assert!((g.wrt(x) - fd_x).abs() < 1e-8, "{} vs {fd_x}", g.wrt(x));
        assert!((g.wrt(y) - fd_y).abs() < 1e-8, "{} vs {fd_y}", g.wrt(y));
    }

    #[test]
    fn positive_part_has_the_indicator_derivative() {
        let tape = Tape::new();
        let x = tape.var(2.0);
        let up = (x - 1.0).maxf(0.0).grad();
        assert!((up.wrt(x) - 1.0).abs() < 1e-14);
        let x2 = tape.var(0.5);
        let down = (x2 - 1.0).maxf(0.0).grad();
        assert_eq!(down.wrt(x2), 0.0);
        // two-variable max routes the adjoint to the winner
        let a = tape.var(3.0);
        let b = tape.var(4.0);
        let g = (a.max(b) * 2.0).grad();
        assert_eq!(g.wrt(a), 0.0);
        assert!((g.wrt(b) - 2.0).abs() < 1e-14);
    }

    #[test]
    fn norm_cdf_differentiates_to_the_density() {
        let tape = Tape::new();
        let x = tape.var(0.37);
        let g = x.norm_cdf().grad();
        assert!((g.wrt(x) - crate::core::utils::norm_pdf(0.37)).abs() < 1e-14);
    }
}
