//! Multi-dimensional Sobol low-discrepancy sequences.
//!
//! Gray-code construction over per-dimension direction numbers
//! (primitive polynomials + initial values from the Joe-Kuo tables), with
//! optional seeded **digital-shift scrambling** — XOR with a fixed random
//! word per dimension, which preserves the dyadic equidistribution that
//! makes the sequence low-discrepancy while decorrelating runs.
//!
//! Every 1-D projection is a (0,1)-sequence in base 2: any first `2^k`
//! points land exactly once in each dyadic interval of length `2^-k`
//! (the tests assert this, scrambled or not). Dimension counts up to
//! [`SobolSequence::MAX_DIMS`] are supported; for higher-dimensional
//! problems use [`QmcSequence`](super::halton::QmcSequence) (Halton),
//! which extends to any dimension.

use crate::core::utils::inv_N;

use super::rng::splitmix64;

/// Bits of resolution per coordinate (fits the f64 mantissa).
const BITS: usize = 52;

/// Primitive-polynomial degree `s`, encoded coefficients `a`, and initial
/// direction values `m_1..m_s` for dimensions 2..=22 (dimension 1 is the
/// degenerate van der Corput case), from the Joe-Kuo "new-joe-kuo-6"
/// tables.
const JOE_KUO: [(usize, u64, [u64; 7]); 21] = [
    (1, 0, [1, 0, 0, 0, 0, 0, 0]),
    (2, 1, [1, 3, 0, 0, 0, 0, 0]),
    (3, 1, [1, 3, 1, 0, 0, 0, 0]),
    (3, 2, [1, 1, 1, 0, 0, 0, 0]),
    (4, 1, [1, 1, 3, 3, 0, 0, 0]),
    (4, 4, [1, 3, 5, 13, 0, 0, 0]),
    (5, 2, [1, 1, 5, 5, 17, 0, 0]),
    (5, 4, [1, 1, 5, 5, 5, 0, 0]),
    (5, 7, [1, 1, 7, 11, 19, 0, 0]),
    (5, 11, [1, 1, 5, 1, 1, 0, 0]),
    (5, 13, [1, 1, 1, 3, 11, 0, 0]),
    (5, 14, [1, 3, 5, 5, 31, 0, 0]),
    (6, 1, [1, 3, 3, 9, 7, 49, 0]),
    (6, 13, [1, 1, 1, 15, 21, 21, 0]),
    (6, 16, [1, 3, 1, 13, 27, 49, 0]),
    (6, 19, [1, 1, 1, 15, 7, 5, 0]),
    (6, 22, [1, 3, 1, 15, 13, 25, 0]),
    (6, 25, [1, 1, 5, 5, 19, 61, 0]),
    (7, 1, [1, 3, 7, 11, 23, 15, 103]),
    (7, 4, [1, 3, 7, 13, 13, 15, 69]),
    (7, 7, [1, 1, 7, 13, 25, 5, 37]),
];

/// A multi-dimensional Sobol sequence, optionally digitally scrambled.
pub struct SobolSequence {
    /// Direction integers per dimension, aligned to the top of `BITS`.
    v: Vec<[u64; BITS]>,
    /// Per-dimension digital shift (all zero when unscrambled).
    shift: Vec<u64>,
}

impl SobolSequence {
    /// Dimensions supported by the embedded direction-number table.
    pub const MAX_DIMS: usize = JOE_KUO.len() + 1;

    /// An unscrambled Sobol sequence in `dims` dimensions.
    pub fn new(dims: usize) -> Self {
        assert!(
            (1..=Self::MAX_DIMS).contains(&dims),
            "SobolSequence supports 1..={} dimensions (got {dims}); use QmcSequence beyond",
            Self::MAX_DIMS
        );
        let mut v = Vec::with_capacity(dims);
        // dimension 1: van der Corput, m_k = 1 for every k
        let mut first = [0u64; BITS];
        for (k, vk) in first.iter_mut().enumerate() {
            *vk = 1u64 << (BITS - 1 - k);
        }
        v.push(first);
        for &(s, a, m_init) in JOE_KUO.iter().take(dims - 1) {
            v.push(direction_integers(s, a, &m_init[..s]));
        }
        SobolSequence { shift: vec![0; dims], v }
    }

    /// A digitally-scrambled sequence: each dimension is XORed with a
    /// fixed seed-derived word. Deterministic per seed.
    pub fn scrambled(dims: usize, seed: u64) -> Self {
        let mut sobol = Self::new(dims);
        let mask = (1u64 << BITS) - 1;
        for (d, shift) in sobol.shift.iter_mut().enumerate() {
            *shift = splitmix64(seed ^ splitmix64(0x50B0 ^ d as u64)) & mask;
        }
        sobol
    }

    pub fn dims(&self) -> usize {
        self.v.len()
    }

    /// Fill `out` with the uniforms of point `index` (Gray-code order;
    /// index 0 is the origin — start at 1 for draws that feed `inv_N`).
    pub fn uniforms(&self, index: u64, out: &mut [f64]) {
        assert!(out.len() <= self.v.len());
        let gray = index ^ (index >> 1);
        let scale = 1.0 / (1u64 << BITS) as f64;
        for (d, u) in out.iter_mut().enumerate() {
            let mut x = self.shift[d];
            let mut g = gray;
            let mut k = 0;
            while g != 0 {
                if g & 1 == 1 {
                    x ^= self.v[d][k];
                }
                g >>= 1;
                k += 1;
            }
            *u = x as f64 * scale;
        }
    }

    /// Fill `out` with the standard normals of point `index` (1-based)
    /// through the inverse normal CDF.
    pub fn normals(&self, index: u64, out: &mut [f64]) {
        self.uniforms(index, out);
        for u in out.iter_mut() {
            *u = inv_N(u.clamp(1e-15, 1.0 - 1e-15));
        }
    }
}

/// Direction integers for one dimension from its primitive polynomial
/// (degree `s`, packed coefficients `a`) and initial values `m_init`.
fn direction_integers(s: usize, a: u64, m_init: &[u64]) -> [u64; BITS] {
    let mut m = [0u64; BITS];
    m[..s].copy_from_slice(m_init);
    for k in s..BITS {
        // m_k = 2 a_1 m_{k-1} ^ ... ^ 2^{s-1} a_{s-1} m_{k-s+1}
        //       ^ 2^s m_{k-s} ^ m_{k-s}
        let mut mk = m[k - s] ^ (m[k - s] << s);
        for j in 1..s {
            if (a >> (s - 1 - j)) & 1 == 1 {
                mk ^= m[k - j] << j;
            }
        }
        m[k] = mk;
    }
    let mut v = [0u64; BITS];
    for k in 0..BITS {
        v[k] = m[k] << (BITS - 1 - k);
    }
    v
}

/// Radical inverse in base 2 (van der Corput sequence) — the 1-D Sobol
/// sequence. `i >= 1`; returns a value in (0, 1).
pub(crate) fn van_der_corput_base2(mut i: u64) -> f64 {
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
    fn first_dimension_is_van_der_corput_as_a_set() {
        // Gray-code generation reorders each dyadic block, so compare the
        // point sets: the first 64 points of dimension 1 are exactly
        // {0, 1/64, ..., 63/64}
        let sobol = SobolSequence::new(3);
        let mut u = [0.0; 3];
        let mut values: Vec<f64> = (0..64u64)
            .map(|i| {
                sobol.uniforms(i, &mut u);
                u[0]
            })
            .collect();
        values.sort_by(f64::total_cmp);
        for (j, v) in values.iter().enumerate() {
            assert!((v - j as f64 / 64.0).abs() < 1e-15, "slot {j}: {v}");
        }
    }

    #[test]
    fn every_dimension_stratifies_dyadic_intervals_exactly() {
        // the (0,1)-sequence property: among the first 256 points each
        // dimension hits each interval [j/256, (j+1)/256) exactly once
        for sobol in [SobolSequence::new(SobolSequence::MAX_DIMS),
                      SobolSequence::scrambled(SobolSequence::MAX_DIMS, 9)] {
            let dims = sobol.dims();
            let n = 256usize;
            let mut hits = vec![vec![0u32; n]; dims];
            let mut u = vec![0.0; dims];
            for i in 0..n as u64 {
                sobol.uniforms(i, &mut u);
                for d in 0..dims {
                    hits[d][(u[d] * n as f64) as usize] += 1;
                }
            }
            for (d, h) in hits.iter().enumerate() {
                assert!(h.iter().all(|&c| c == 1), "dim {d} is not a (0,1)-sequence");
            }
        }
    }

    #[test]
    fn scrambling_is_seeded_and_decorrelates() {
        let a = SobolSequence::scrambled(4, 1);
        let b = SobolSequence::scrambled(4, 1);
        let c = SobolSequence::scrambled(4, 2);
        let (mut ua, mut ub, mut uc) = ([0.0; 4], [0.0; 4], [0.0; 4]);
        a.uniforms(17, &mut ua);
        b.uniforms(17, &mut ub);
        c.uniforms(17, &mut uc);
        assert_eq!(ua, ub, "same seed must reproduce");
        assert_ne!(ua, uc, "different seeds must differ");
    }

    #[test]
    fn qmc_integration_beats_the_crude_rate() {
        // integrate x*y*z over the unit cube (= 1/8) with 4096 points:
        // Sobol error should be far below the ~0.005 Monte Carlo sigma
        let sobol = SobolSequence::new(3);
        let mut u = [0.0; 3];
        let n = 4096u64;
        let sum: f64 = (1..=n)
            .map(|i| {
                sobol.uniforms(i, &mut u);
                u[0] * u[1] * u[2]
            })
            .sum();
        let estimate = sum / n as f64;
        assert!((estimate - 0.125).abs() < 1e-3, "estimate {estimate}");
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
