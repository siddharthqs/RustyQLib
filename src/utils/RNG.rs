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

/// SplitMix64 finalizer — used to derive statistically independent
/// per-path RNG streams from (seed, path index).
fn splitmix64(mut x: u64) -> u64 {
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

fn first_primes(n: usize) -> Vec<u64> {
    let mut primes: Vec<u64> = Vec::with_capacity(n);
    let mut candidate = 2u64;
    while primes.len() < n {
        if primes.iter().take_while(|&&p| p * p <= candidate).all(|&p| candidate % p != 0) {
            primes.push(candidate);
        }
        candidate += 1;
    }
    primes
}

fn radical_inverse(mut i: u64, base: u64) -> f64 {
    let inv_base = 1.0 / base as f64;
    let mut f = inv_base;
    let mut x = 0.0;
    while i > 0 {
        x += f * (i % base) as f64;
        i /= base;
        f *= inv_base;
    }
    x
}

/// Multi-dimensional low-discrepancy sequence: Halton with prime bases and
/// a deterministic Cranley-Patterson rotation per dimension (derived from
/// the seed), mapped to standard normals through the inverse CDF.
///
/// Combined with [`BrownianBridge`] ordering, the well-distributed leading
/// dimensions carry the coarse structure of each path. Direction-number
/// Sobol (Joe-Kuo) is a drop-in upgrade behind this same interface.
pub struct QmcSequence {
    bases: Vec<u64>,
    shifts: Vec<f64>,
}

impl QmcSequence {
    pub fn new(dims: usize, seed: u64) -> Self {
        let bases = first_primes(dims);
        let shifts = (0..dims)
            .map(|d| (splitmix64(seed ^ splitmix64(0xC0FFEE ^ d as u64)) >> 11) as f64
                / (1u64 << 53) as f64)
            .collect();
        QmcSequence { bases, shifts }
    }

    /// Fill `out` with the standard normals of point `index` (1-based).
    pub fn normals(&self, index: u64, out: &mut [f64]) {
        for (d, z) in out.iter_mut().enumerate() {
            let mut u = radical_inverse(index, self.bases[d]) + self.shifts[d];
            if u >= 1.0 {
                u -= 1.0;
            }
            *z = crate::core::utils::inv_N(u.clamp(1e-15, 1.0 - 1e-15));
        }
    }
}

/// Brownian bridge path construction: the first draw fixes the terminal
/// value, subsequent draws fill midpoints by bisection, so low-discrepancy
/// coordinates are spent on the dimensions that matter most. Weights are
/// precomputed once per pricing call and shared across paths.
///
/// Produces per-step Brownian increments; a future multi-factor model
/// (e.g. Heston) uses one bridge per factor.
pub struct BrownianBridge {
    steps: usize,
    sqrt_t: f64,
    /// (mid, left, right, weight_left, weight_right, stddev); left == usize::MAX
    /// encodes the origin (t = 0, W = 0)
    plan: Vec<(usize, usize, usize, f64, f64, f64)>,
}

impl BrownianBridge {
    pub fn new(steps: usize, dt: f64) -> Self {
        assert!(steps >= 1);
        let t_at = |i: usize| {
            if i == usize::MAX { 0.0 } else { (i + 1) as f64 * dt }
        };
        let mut plan = Vec::with_capacity(steps.saturating_sub(1));
        let mut queue = std::collections::VecDeque::new();
        queue.push_back((usize::MAX, steps - 1));
        while let Some((l, r)) = queue.pop_front() {
            let lo = if l == usize::MAX { 0 } else { l + 1 };
            if r <= lo {
                continue;
            }
            let mid = (lo + r) / 2;
            let (tl, tm, tr) = (t_at(l), t_at(mid), t_at(r));
            let wl = (tr - tm) / (tr - tl);
            let wr = (tm - tl) / (tr - tl);
            let sd = ((tm - tl) * (tr - tm) / (tr - tl)).sqrt();
            plan.push((mid, l, r, wl, wr, sd));
            queue.push_back((l, mid));
            queue.push_back((mid, r));
        }
        BrownianBridge { steps, sqrt_t: (steps as f64 * dt).sqrt(), plan }
    }

    /// Consume `steps` standard normals, produce `steps` Brownian increments.
    pub fn increments(&self, z: &[f64], w_buf: &mut [f64], out: &mut [f64]) {
        assert!(z.len() == self.steps && w_buf.len() == self.steps && out.len() == self.steps);
        w_buf[self.steps - 1] = self.sqrt_t * z[0];
        for (k, &(mid, l, r, wl, wr, sd)) in self.plan.iter().enumerate() {
            let w_l = if l == usize::MAX { 0.0 } else { w_buf[l] };
            w_buf[mid] = wl * w_l + wr * w_buf[r] + sd * z[k + 1];
        }
        let mut prev = 0.0;
        for i in 0..self.steps {
            out[i] = w_buf[i] - prev;
            prev = w_buf[i];
        }
    }
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

    #[test]
    fn brownian_bridge_reproduces_marginal_variance() {
        // increments must sum to W_T = sqrt(T) z_0 and have the right
        // per-step variance under iid normals
        let steps = 13;
        let dt = 1.0 / steps as f64;
        let bridge = BrownianBridge::new(steps, dt);
        let mut w = vec![0.0; steps];
        let mut inc = vec![0.0; steps];
        let mut sum_sq = vec![0.0; steps];
        let n = 20_000;
        for path in 0..n {
            let mut z = vec![0.0; steps];
            path_normals(7, path, &mut z);
            bridge.increments(&z, &mut w, &mut inc);
            let total: f64 = inc.iter().sum();
            assert!((total - z[0] * (1.0_f64).sqrt()).abs() < 1e-12);
            for (i, d) in inc.iter().enumerate() {
                sum_sq[i] += d * d;
            }
        }
        for (i, s) in sum_sq.iter().enumerate() {
            let var = s / n as f64;
            assert!((var - dt).abs() < 0.02 * dt.max(0.001), "step {i}: var {var} vs {dt}");
        }
    }

    #[test]
    fn qmc_sequence_normals_are_standardized() {
        let steps = 32;
        let qmc = QmcSequence::new(steps, 42);
        let mut z = vec![0.0; steps];
        let n = 8192;
        let mut mean = vec![0.0; steps];
        let mut var = vec![0.0; steps];
        for i in 1..=n {
            qmc.normals(i as u64, &mut z);
            for d in 0..steps {
                mean[d] += z[d];
                var[d] += z[d] * z[d];
            }
        }
        for d in 0..steps {
            let m = mean[d] / n as f64;
            let v = var[d] / n as f64;
            assert!(m.abs() < 0.05, "dim {d}: mean {m}");
            assert!((v - 1.0).abs() < 0.1, "dim {d}: var {v}");
        }
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
