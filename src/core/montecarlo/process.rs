//! Generic Itô-process abstraction: the SDE's coefficients live in the
//! process object and the discretization schemes are written **once**
//! against them — the QuantLib `StochasticProcess` / TF Quant Finance
//! `GenericItoProcess` pattern.
//!
//! ```text
//! dX = a(t, X) dt + b(t, X) dW
//! ```
//!
//! - [`StochasticProcess1D`] is the scalar contract; Euler and Milstein
//!   are provided methods over [`drift`](StochasticProcess1D::drift) /
//!   [`diffusion`](StochasticProcess1D::diffusion), so a new SDE
//!   (Vasicek / Hull-White, CEV, CIR, ...) only supplies coefficients.
//! - Processes with a closed-form transition density override
//!   [`exact_step`](StochasticProcess1D::exact_step) (lognormal for
//!   Black-Scholes, Gaussian for Ornstein-Uhlenbeck); models whose good
//!   schemes are genuinely model-specific (Heston full-truncation / QE)
//!   override [`evolve`](StochasticProcess::evolve) wholesale.
//! - State constraints belong to the process, not the stepper: a
//!   lognormal equity floors at zero via
//!   [`constrain`](StochasticProcess1D::constrain), while a normal-SDE
//!   short rate legitimately goes negative (the default is the
//!   identity).
//! - [`StochasticProcess`] is the N-state / M-factor generalization
//!   (Heston: 2 states driven by 2 correlated factors, the correlation
//!   folded into the diffusion matrix rows).

use std::str::FromStr;

/// Time-stepping scheme for path-wise simulation.
/// `Exact` samples the process's closed-form transition where one exists
/// (no discretization bias) and degrades to Euler where none does;
/// Euler and Milstein are the standard approximate schemes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscretizationScheme {
    Exact,
    Euler,
    Milstein,
}

impl FromStr for DiscretizationScheme {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "exact" => Ok(DiscretizationScheme::Exact),
            "euler" => Ok(DiscretizationScheme::Euler),
            "milstein" => Ok(DiscretizationScheme::Milstein),
            other => Err(format!("Invalid discretization scheme '{other}'")),
        }
    }
}

/// Central finite difference of the diffusion coefficient in the state —
/// the default for [`StochasticProcess1D::diffusion_dx`], callable from
/// implementations that override it for one branch of their dynamics.
pub fn numeric_diffusion_dx<P: StochasticProcess1D + ?Sized>(p: &P, t: f64, x: f64) -> f64 {
    let h = 1e-5 * x.abs().max(1.0);
    (p.diffusion(t, x + h) - p.diffusion(t, x - h)) / (2.0 * h)
}

/// A scalar Itô process `dX = a(t, X) dt + b(t, X) dW`.
///
/// Implementations are shared across rayon path chunks, hence `Sync`.
pub trait StochasticProcess1D: Sync {
    /// Drift coefficient `a(t, x)`.
    fn drift(&self, t: f64, x: f64) -> f64;

    /// Diffusion coefficient `b(t, x)`.
    fn diffusion(&self, t: f64, x: f64) -> f64;

    /// `∂b/∂x`, the extra coefficient Milstein needs.
    fn diffusion_dx(&self, t: f64, x: f64) -> f64 {
        numeric_diffusion_dx(self, t, x)
    }

    /// One draw from the closed-form transition `X_{t+dt} | X_t = x`,
    /// when the process has one. `None` (the default) makes the `Exact`
    /// scheme fall back to Euler.
    fn exact_step(&self, _t: f64, _x: f64, _dt: f64, _dw: f64) -> Option<f64> {
        None
    }

    /// State constraint applied after every step. Identity by default —
    /// only processes whose state space is genuinely bounded (lognormal
    /// equity at zero, truncated variance) should clamp.
    fn constrain(&self, x: f64) -> f64 {
        x
    }

    /// Advance the state by one step of `scheme`: the generic stepping,
    /// written against `drift`/`diffusion` alone. Override only to
    /// exploit model structure (e.g. one volatility lookup shared by the
    /// coefficients, or a scheme the enum cannot express).
    fn evolve(&self, scheme: DiscretizationScheme, t: f64, x: f64, dt: f64, dw: f64) -> f64 {
        let next = match scheme {
            DiscretizationScheme::Exact => self
                .exact_step(t, x, dt, dw)
                .unwrap_or_else(|| x + self.drift(t, x) * dt + self.diffusion(t, x) * dw),
            DiscretizationScheme::Euler => x + self.drift(t, x) * dt + self.diffusion(t, x) * dw,
            DiscretizationScheme::Milstein => {
                let b = self.diffusion(t, x);
                x + self.drift(t, x) * dt
                    + b * dw
                    + 0.5 * b * self.diffusion_dx(t, x) * (dw * dw - dt)
            }
        };
        self.constrain(next)
    }
}

/// An N-state Itô process driven by M independent Brownian factors:
/// `dX_i = a_i(t, X) dt + Σ_j b_ij(t, X) dW_j`. Factor correlation is
/// expressed through the rows of the diffusion matrix (its Cholesky
/// structure), so `dw` always carries **independent** increments.
pub trait StochasticProcess: Sync {
    /// Number of state variables.
    fn dim(&self) -> usize;

    /// Number of driving Brownian factors (`dw.len()`).
    fn factors(&self) -> usize;

    /// Drift vector `a(t, x)` into `out` (`dim` long).
    fn drift(&self, t: f64, x: &[f64], out: &mut [f64]);

    /// Diffusion matrix `b(t, x)` into `out`, row-major `dim × factors`.
    fn diffusion(&self, t: f64, x: &[f64], out: &mut [f64]);

    /// State constraint applied after every step (identity by default).
    fn constrain(&self, _x: &mut [f64]) {}

    /// One Euler-Maruyama step from `x` into `out`. The default
    /// allocates small scratch buffers; hot loops should override with
    /// model-specific stepping (which is also where non-Euler schemes —
    /// exact transitions, Heston full-truncation/QE — live, since
    /// generic multi-factor Milstein would need Lévy areas).
    fn evolve(&self, t: f64, x: &[f64], dt: f64, dw: &[f64], out: &mut [f64]) {
        let (dim, factors) = (self.dim(), self.factors());
        debug_assert_eq!(dw.len(), factors);
        let mut a = vec![0.0; dim];
        let mut b = vec![0.0; dim * factors];
        self.drift(t, x, &mut a);
        self.diffusion(t, x, &mut b);
        for i in 0..dim {
            let shock: f64 =
                (0..factors).map(|j| b[i * factors + j] * dw[j]).sum();
            out[i] = x[i] + a[i] * dt + shock;
        }
        self.constrain(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vasicek short rate `dr = kappa (theta - r) dt + sigma dW` — an
    /// additive, mean-reverting SDE the old GBM-shaped stepper could not
    /// express: the state may go negative and the exact transition is
    /// Gaussian, not lognormal.
    struct Vasicek {
        kappa: f64,
        theta: f64,
        sigma: f64,
    }

    impl StochasticProcess1D for Vasicek {
        fn drift(&self, _t: f64, x: f64) -> f64 {
            self.kappa * (self.theta - x)
        }
        fn diffusion(&self, _t: f64, _x: f64) -> f64 {
            self.sigma
        }
        fn exact_step(&self, _t: f64, x: f64, dt: f64, dw: f64) -> Option<f64> {
            // Ornstein-Uhlenbeck transition: mean-revert the state and
            // scale the shock to the transition's standard deviation
            let decay = (-self.kappa * dt).exp();
            let mean = self.theta + (x - self.theta) * decay;
            let var = self.sigma * self.sigma * (1.0 - decay * decay) / (2.0 * self.kappa);
            let z = dw / dt.sqrt();
            Some(mean + var.sqrt() * z)
        }
    }

    #[test]
    fn euler_matches_hand_computed_step() {
        let p = Vasicek { kappa: 2.0, theta: 0.03, sigma: 0.01 };
        let (t, x, dt, dw) = (0.5, 0.05, 0.01, -0.02);
        let expected = x + 2.0 * (0.03 - 0.05) * dt + 0.01 * dw;
        let got = p.evolve(DiscretizationScheme::Euler, t, x, dt, dw);
        assert!((got - expected).abs() < 1e-15);
    }

    #[test]
    fn milstein_reduces_to_euler_for_additive_noise() {
        // constant diffusion => db/dx = 0 => the Milstein correction
        // vanishes (the numeric default derivative must see that)
        let p = Vasicek { kappa: 2.0, theta: 0.03, sigma: 0.01 };
        let (t, x, dt, dw) = (0.5, 0.05, 0.01, 0.03);
        let euler = p.evolve(DiscretizationScheme::Euler, t, x, dt, dw);
        let milstein = p.evolve(DiscretizationScheme::Milstein, t, x, dt, dw);
        assert!((euler - milstein).abs() < 1e-12);
    }

    #[test]
    fn state_can_go_negative_without_a_gbm_floor() {
        // a large negative shock takes the rate below zero — legitimate
        // for a normal SDE, and the default constrain must not clamp it
        let p = Vasicek { kappa: 0.5, theta: 0.01, sigma: 0.02 };
        let next = p.evolve(DiscretizationScheme::Euler, 0.0, 0.001, 0.01, -0.5);
        assert!(next < 0.0);
    }

    #[test]
    fn exact_step_hits_the_ou_transition_mean() {
        // dw = 0 => the exact step lands exactly on the conditional mean
        let p = Vasicek { kappa: 2.0, theta: 0.03, sigma: 0.01 };
        let next = p.evolve(DiscretizationScheme::Exact, 0.0, 0.05, 0.25, 0.0);
        let mean = 0.03 + (0.05 - 0.03) * (-2.0_f64 * 0.25).exp();
        assert!((next - mean).abs() < 1e-15);
    }

    #[test]
    fn numeric_diffusion_dx_recovers_multiplicative_slope() {
        // b(x) = sigma * x => db/dx = sigma
        struct Gbmish;
        impl StochasticProcess1D for Gbmish {
            fn drift(&self, _t: f64, x: f64) -> f64 {
                0.05 * x
            }
            fn diffusion(&self, _t: f64, x: f64) -> f64 {
                0.2 * x
            }
        }
        let d = numeric_diffusion_dx(&Gbmish, 0.0, 100.0);
        assert!((d - 0.2).abs() < 1e-8, "{d}");
    }

    #[test]
    fn multi_dim_default_euler() {
        // 2-state, 2-factor linear process with a non-diagonal diffusion
        struct TwoDim;
        impl StochasticProcess for TwoDim {
            fn dim(&self) -> usize {
                2
            }
            fn factors(&self) -> usize {
                2
            }
            fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
                out[0] = 0.1 * x[0];
                out[1] = -0.2 * x[1];
            }
            fn diffusion(&self, _t: f64, _x: &[f64], out: &mut [f64]) {
                out.copy_from_slice(&[0.3, 0.0, 0.1, 0.2]);
            }
        }
        let (x, dt, dw) = ([1.0, 2.0], 0.01, [0.05, -0.03]);
        let mut out = [0.0; 2];
        TwoDim.evolve(0.0, &x, dt, &dw, &mut out);
        assert!((out[0] - (1.0 + 0.1 * 1.0 * dt + 0.3 * 0.05)).abs() < 1e-15);
        assert!((out[1] - (2.0 - 0.2 * 2.0 * dt + 0.1 * 0.05 - 0.2 * 0.03)).abs() < 1e-15);
    }
}
