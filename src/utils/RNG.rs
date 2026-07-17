//! Normal draw generation for Monte Carlo pricing.
//!
//! Two samplers:
//! - **Seeded pseudo-random** (PCG64): reproducible across runs, with
//!   antithetic pairing and moment matching as variance reduction.
//! - **Low-discrepancy** (1-D Sobol, i.e. the van der Corput base-2
//!   sequence) mapped through the inverse normal CDF — near-O(1/n)
//!   convergence for terminal-value (single-dimension) simulation.
//!
//! Multi-dimensional (path-wise) draws use the seeded pseudo-random
//! generator; proper multi-dimensional Sobol with a Brownian bridge is
//! future work.

use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

use crate::core::utils::inv_N;

/// Radical inverse in base 2 (van der Corput sequence) — the 1-D Sobol
/// sequence. `i >= 1`; returns a value in (0, 1).
fn van_der_corput_base2(mut i: u64) -> f64 {
    let mut f = 0.5;
    let mut x = 0.0;
    while i > 0 {
        if i & 1 == 1 {
            x += f;
        }
        i >>= 1;
        f *= 0.5;
    }
    x
}

/// `n` low-discrepancy standard normal draws (1-D Sobol through the
/// inverse normal CDF). Deterministic.
pub fn sobol_normals(n: usize) -> Vec<f64> {
    (1..=n as u64).map(|i| inv_N(van_der_corput_base2(i))).collect()
}

/// `n` seeded pseudo-random standard normals with antithetic pairing and
/// moment matching (mean 0, variance 1 exactly). Deterministic per seed.
pub fn pseudo_normals(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut draws = Vec::with_capacity(n + 1);
    while draws.len() < n {
        let z: f64 = rng.sample(StandardNormal);
        draws.push(z);
        draws.push(-z);
    }
    draws.truncate(n);
    moment_match(&mut draws);
    draws
}

/// `paths x steps` matrix of seeded pseudo-random standard normals for
/// path-wise simulation. Deterministic per seed.
pub fn pseudo_normal_matrix(paths: usize, steps: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Pcg64::seed_from_u64(seed);
    (0..paths)
        .map(|_| (0..steps).map(|_| rng.sample(StandardNormal)).collect())
        .collect()
}

fn moment_match(draws: &mut [f64]) {
    let n = draws.len() as f64;
    let mean = draws.iter().sum::<f64>() / n;
    let var = draws.iter().map(|z| (z - mean) * (z - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    if std > 0.0 {
        for z in draws.iter_mut() {
            *z = (*z - mean) / std;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn van_der_corput_first_values() {
        // 1/2, 1/4, 3/4, 1/8, 5/8, ...
        let expected = [0.5, 0.25, 0.75, 0.125, 0.625];
        for (i, want) in expected.iter().enumerate() {
            assert_eq!(van_der_corput_base2(i as u64 + 1), *want);
        }
    }

    #[test]
    fn pseudo_normals_are_reproducible_and_standardized() {
        let a = pseudo_normals(10_000, 42);
        let b = pseudo_normals(10_000, 42);
        assert_eq!(a, b);
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let var: f64 = a.iter().map(|z| z * z).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 1e-12);
        assert!((var - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sobol_normals_have_near_perfect_moments() {
        let draws = sobol_normals(65_536);
        let mean: f64 = draws.iter().sum::<f64>() / draws.len() as f64;
        let var: f64 = draws.iter().map(|z| z * z).sum::<f64>() / draws.len() as f64;
        assert!(mean.abs() < 1e-3, "mean {mean}");
        assert!((var - 1.0).abs() < 1e-2, "var {var}");
    }
}
