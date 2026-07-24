//! Stratified sampling and Latin hypercube designs — seeded, exact
//! stratification of the unit interval / hypercube.

use crate::core::utils::inv_norm_cdf;

use super::rng::splitmix64;

/// Tiny counter-based uniform generator on top of SplitMix64.
struct Counter {
    state: u64,
}

impl Counter {
    fn new(seed: u64) -> Self {
        Self { state: splitmix64(seed) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(1);
        splitmix64(self.state)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// `n` stratified uniforms: exactly one jittered draw per stratum
/// `[i/n, (i+1)/n)`. Cuts the variance of smooth 1-D integrands from
/// O(1/n) to O(1/n^3). Deterministic per seed.
pub fn stratified_uniforms(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Counter::new(seed);
    (0..n).map(|i| (i as f64 + rng.uniform()) / n as f64).collect()
}

/// `n` stratified standard normals (stratified uniforms through the
/// inverse normal CDF).
pub fn stratified_normals(n: usize, seed: u64) -> Vec<f64> {
    stratified_uniforms(n, seed)
        .into_iter()
        .map(|u| inv_norm_cdf(u.clamp(1e-15, 1.0 - 1e-15)))
        .collect()
}

/// An `n x dims` Latin hypercube: every dimension is exactly stratified
/// (one point per stratum), with independent random pairings across
/// dimensions. Deterministic per seed.
pub fn latin_hypercube(n: usize, dims: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Counter::new(seed);
    // one stratified, independently shuffled column per dimension
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(dims);
    for _ in 0..dims {
        let mut column: Vec<f64> =
            (0..n).map(|i| (i as f64 + rng.uniform()) / n as f64).collect();
        // Fisher-Yates
        for i in (1..n).rev() {
            let j = (rng.next_u64() % (i as u64 + 1)) as usize;
            column.swap(i, j);
        }
        columns.push(column);
    }
    (0..n).map(|i| columns.iter().map(|c| c[i]).collect()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stratified_uniforms_hit_every_stratum_once() {
        let n = 128;
        let u = stratified_uniforms(n, 5);
        let mut hits = vec![0u32; n];
        for &x in &u {
            hits[(x * n as f64) as usize] += 1;
        }
        assert!(hits.iter().all(|&c| c == 1));
        assert_eq!(u, stratified_uniforms(n, 5), "seeded determinism");
    }

    #[test]
    fn stratified_normals_beat_plain_sampling_on_the_mean() {
        // the stratified sample mean of N(0,1) is far tighter than the
        // ~1/sqrt(n) of iid draws
        let z = stratified_normals(4096, 11);
        let mean: f64 = z.iter().sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 1e-3, "mean {mean}");
    }

    #[test]
    fn latin_hypercube_stratifies_every_dimension() {
        let (n, dims) = (64, 5);
        let points = latin_hypercube(n, dims, 3);
        assert_eq!(points.len(), n);
        for d in 0..dims {
            let mut hits = vec![0u32; n];
            for p in &points {
                hits[(p[d] * n as f64) as usize] += 1;
            }
            assert!(hits.iter().all(|&c| c == 1), "dimension {d}");
        }
        // different dimensions are paired differently (not comonotone)
        let same_order = points.windows(2).all(|w| (w[0][0] < w[1][0]) == (w[0][1] < w[1][1]));
        assert!(!same_order, "columns appear perfectly rank-correlated");
    }
}
