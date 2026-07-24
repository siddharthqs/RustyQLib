//! Halton low-discrepancy sequences: prime radical inverses with a
//! seeded Cranley-Patterson rotation. Any number of dimensions, so this
//! is the quasi-random workhorse when the problem's dimension exceeds
//! the embedded Sobol table ([`SobolSequence`](super::sobol::SobolSequence)).

use crate::core::utils::inv_N;

use super::rng::splitmix64;

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
/// Combined with [`BrownianBridge`](super::brownian_bridge::BrownianBridge)
/// ordering, the well-distributed leading dimensions carry the coarse
/// structure of each path.
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
            *z = inv_N(u.clamp(1e-15, 1.0 - 1e-15));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
