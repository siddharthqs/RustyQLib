//! Deterministic pseudo-random generation for simulation.
//!
//! The core idea is **per-path streams**: each path derives its own
//! statistically independent PCG64 generator from `(seed, path_index)`
//! via the SplitMix64 finalizer, so results are bit-identical however
//! paths are scheduled across threads — the foundation of reproducible
//! parallel Monte Carlo.

use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use rand_pcg::Pcg64;

use super::variance_reduction::moment_match;

/// SplitMix64 finalizer — a high-quality 64-bit mixer, used to derive
/// statistically independent stream seeds from `(seed, index)` pairs and
/// as a tiny counter-based RNG for shuffles and shifts.
pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic, independent RNG stream for one simulation path — the
/// basis of parallel path generation (each path seeds its own generator,
/// so results are identical regardless of thread scheduling).
pub fn path_rng(seed: u64, path_index: u64) -> Pcg64 {
    Pcg64::seed_from_u64(splitmix64(seed ^ splitmix64(path_index)))
}

/// Standard normal draws for one path from its own stream.
pub fn path_normals(seed: u64, path_index: u64, out: &mut [f64]) {
    let mut rng = path_rng(seed, path_index);
    for z in out.iter_mut() {
        *z = rng.sample(StandardNormal);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn path_streams_are_deterministic_and_distinct() {
        let mut a = [0.0; 8];
        let mut b = [0.0; 8];
        let mut a2 = [0.0; 8];
        path_normals(42, 0, &mut a);
        path_normals(42, 1, &mut b);
        path_normals(42, 0, &mut a2);
        assert_eq!(a, a2);
        assert_ne!(a, b);
    }
}
