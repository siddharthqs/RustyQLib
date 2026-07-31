//! Public path generation over the stochastic-process traits — the
//! library's `sample_paths` API (the TF-Quant-Finance idiom): give it a
//! process, an initial state and a sampling configuration, get back the
//! simulated paths as a dense matrix, generated in parallel with the
//! same deterministic draw discipline the pricing engines use.
//!
//! Two entry points:
//! - [`sample_paths_1d`] for scalar processes ([`StochasticProcess1D`]),
//!   with the full sampler menu: seeded pseudo-random antithetic pairs,
//!   or a low-discrepancy sequence routed through the Brownian bridge;
//! - [`sample_paths`] for N-state / M-factor processes
//!   ([`StochasticProcess`]); the QMC route runs one bridge per factor.
//!
//! Paths exclude the initial state (they start at the first step), the
//! convention every payoff in the library assumes. Draws are per-path
//! deterministic — path `i` under seed `s` is the same regardless of
//! thread scheduling or how many other paths are requested alongside it.

use std::str::FromStr;

use rayon::prelude::*;

use super::brownian_bridge::BrownianBridge;
use super::halton::QmcSequence;
use super::process::{DiscretizationScheme, StochasticProcess, StochasticProcess1D};
use super::rng::path_normals;

/// Draw sampler. `Sobol` selects the low-discrepancy family: true Sobol
/// (van der Corput) in one dimension, a scrambled multi-dimensional
/// sequence through a Brownian bridge for path-wise simulation.
/// `PseudoRandom` uses seeded per-path PCG64 streams with antithetic
/// pairing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampler {
    Sobol,
    PseudoRandom,
}

impl FromStr for Sampler {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_lowercase().as_str() {
            "sobol" | "quasi" => Ok(Sampler::Sobol),
            "pseudo" | "pseudorandom" | "pseudo_random" => Ok(Sampler::PseudoRandom),
            other => Err(format!("Invalid sampler '{other}'")),
        }
    }
}

/// Deterministic per-path Brownian increment source. Pseudo-random paths
/// come in antithetic pairs (2k, 2k+1) from independent per-pair streams;
/// low-discrepancy paths are sequence points routed through the Brownian
/// bridge.
pub enum PathDraws {
    Pseudo { seed: u64, sqrt_dt: f64 },
    Qmc { seq: QmcSequence, bridge: BrownianBridge },
}

impl PathDraws {
    pub fn new(sampler: Sampler, seed: u64, steps: usize, dt: f64) -> Self {
        match sampler {
            Sampler::Sobol => PathDraws::Qmc {
                seq: QmcSequence::new(steps, seed),
                bridge: BrownianBridge::new(steps, dt),
            },
            Sampler::PseudoRandom => PathDraws::Pseudo { seed, sqrt_dt: dt.sqrt() },
        }
    }

    pub fn pseudo(seed: u64, dt: f64) -> Self {
        PathDraws::Pseudo { seed, sqrt_dt: dt.sqrt() }
    }

    /// Fill `dw` with the Brownian increments of path `index`; `z` and
    /// `w` are caller-provided scratch of the same length.
    pub fn fill(&self, index: usize, z: &mut [f64], w: &mut [f64], dw: &mut [f64]) {
        match self {
            PathDraws::Pseudo { seed, sqrt_dt } => {
                path_normals(*seed, (index / 2) as u64, z);
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                for (d, zi) in dw.iter_mut().zip(z.iter()) {
                    *d = sign * sqrt_dt * zi;
                }
            }
            PathDraws::Qmc { seq, bridge } => {
                seq.normals(index as u64 + 1, z);
                bridge.increments(z, w, dw);
            }
        }
    }
}

/// Sampling configuration shared by both entry points.
#[derive(Debug, Clone, Copy)]
pub struct SampleConfig {
    /// Number of paths.
    pub paths: usize,
    /// Time steps per path (the grid is uniform over `horizon`).
    pub steps: usize,
    /// Path length in years.
    pub horizon: f64,
    pub sampler: Sampler,
    pub seed: u64,
}

/// Simulated scalar paths in dense row-major storage: path `i` occupies
/// `steps` consecutive values, excluding the initial state.
pub struct Paths {
    steps: usize,
    data: Vec<f64>,
}

impl Paths {
    pub fn n_paths(&self) -> usize {
        self.data.len() / self.steps
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    /// The levels of path `i`, one per step.
    pub fn path(&self, i: usize) -> &[f64] {
        &self.data[i * self.steps..(i + 1) * self.steps]
    }

    pub fn iter(&self) -> impl Iterator<Item = &[f64]> {
        self.data.chunks(self.steps)
    }
}

/// Simulated multi-state paths: path `i`, step `j` is a `dim`-long state
/// vector at `data[(i * steps + j) * dim ..][..dim]`.
pub struct MultiPaths {
    steps: usize,
    dim: usize,
    data: Vec<f64>,
}

impl MultiPaths {
    pub fn n_paths(&self) -> usize {
        self.data.len() / (self.steps * self.dim)
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Path `i` as `steps` consecutive `dim`-long state vectors.
    pub fn path(&self, i: usize) -> &[f64] {
        let stride = self.steps * self.dim;
        &self.data[i * stride..(i + 1) * stride]
    }

    /// The state vector of path `i` at step `j`.
    pub fn state(&self, i: usize, j: usize) -> &[f64] {
        let at = (i * self.steps + j) * self.dim;
        &self.data[at..at + self.dim]
    }
}

/// Simulate paths of a scalar process from `x0` under the given
/// discretization scheme.
pub fn sample_paths_1d<P: StochasticProcess1D>(
    process: &P,
    x0: f64,
    scheme: DiscretizationScheme,
    cfg: &SampleConfig,
) -> Paths {
    let steps = cfg.steps.max(1);
    let dt = cfg.horizon / steps as f64;
    let draws = PathDraws::new(cfg.sampler, cfg.seed, steps, dt);
    let mut data = vec![0.0; cfg.paths * steps];
    data.par_chunks_mut(steps).enumerate().for_each_init(
        || (vec![0.0; steps], vec![0.0; steps], vec![0.0; steps]),
        |(z, w, dw), (i, out)| {
            draws.fill(i, z, w, dw);
            let mut x = x0;
            for (j, d) in dw.iter().enumerate() {
                x = process.evolve(scheme, j as f64 * dt, x, dt, *d);
                out[j] = x;
            }
        },
    );
    Paths { steps, data }
}

/// Per-path factor draws for multi-state processes, step-major: the
/// increments of step `j` occupy `dw[j * factors ..][..factors]`.
/// Public so streaming multi-asset engines can consume draws without
/// materializing a [`MultiPaths`] matrix.
pub enum MultiDraws {
    Pseudo { seed: u64, sqrt_dt: f64 },
    /// One low-discrepancy point per path over `factors * steps`
    /// coordinates; each factor's block runs through its own pass of the
    /// Brownian bridge.
    Qmc { seq: QmcSequence, bridge: BrownianBridge },
}

impl MultiDraws {
    pub fn new(sampler: Sampler, seed: u64, factors: usize, steps: usize, dt: f64) -> Self {
        match sampler {
            Sampler::Sobol => MultiDraws::Qmc {
                seq: QmcSequence::new(factors * steps, seed),
                bridge: BrownianBridge::new(steps, dt),
            },
            Sampler::PseudoRandom => MultiDraws::Pseudo { seed, sqrt_dt: dt.sqrt() },
        }
    }

    pub fn fill(
        &self,
        index: usize,
        factors: usize,
        steps: usize,
        scratch: &mut FactorScratch,
        dw: &mut [f64],
    ) {
        match self {
            MultiDraws::Pseudo { seed, sqrt_dt } => {
                path_normals(*seed, (index / 2) as u64, &mut scratch.z);
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                for (d, zi) in dw.iter_mut().zip(scratch.z.iter()) {
                    *d = sign * sqrt_dt * zi;
                }
            }
            MultiDraws::Qmc { seq, bridge } => {
                seq.normals(index as u64 + 1, &mut scratch.z);
                for f in 0..factors {
                    bridge.increments(
                        &scratch.z[f * steps..(f + 1) * steps],
                        &mut scratch.w,
                        &mut scratch.dwf,
                    );
                    for (j, d) in scratch.dwf.iter().enumerate() {
                        dw[j * factors + f] = *d;
                    }
                }
            }
        }
    }
}

/// Per-thread scratch for multi-factor draw generation.
pub struct FactorScratch {
    z: Vec<f64>,
    w: Vec<f64>,
    dwf: Vec<f64>,
}

impl FactorScratch {
    pub fn new(factors: usize, steps: usize) -> Self {
        FactorScratch {
            z: vec![0.0; factors * steps],
            w: vec![0.0; steps],
            dwf: vec![0.0; steps],
        }
    }
}

/// Simulate paths of an N-state process from the state `x0`. The
/// stepping scheme is the process's own [`evolve`](StochasticProcess::evolve)
/// (multi-factor schemes are process-owned — e.g. Heston full-truncation
/// or QE); `dw` carries independent increments, any factor correlation
/// lives inside the process.
pub fn sample_paths<P: StochasticProcess>(process: &P, x0: &[f64], cfg: &SampleConfig) -> MultiPaths {
    let (dim, factors) = (process.dim(), process.factors());
    assert_eq!(x0.len(), dim, "initial state must have process.dim() entries");
    let steps = cfg.steps.max(1);
    let dt = cfg.horizon / steps as f64;
    let draws = MultiDraws::new(cfg.sampler, cfg.seed, factors, steps, dt);
    let mut data = vec![0.0; cfg.paths * steps * dim];
    data.par_chunks_mut(steps * dim).enumerate().for_each_init(
        || {
            (
                FactorScratch::new(factors, steps),
                vec![0.0; factors * steps], // step-major increments
                vec![0.0; dim],             // x
                vec![0.0; dim],             // x_next
            )
        },
        |(scratch, dw, x, x_next), (i, out)| {
            draws.fill(i, factors, steps, scratch, dw);
            x.copy_from_slice(x0);
            for j in 0..steps {
                process.evolve(j as f64 * dt, x, dt, &dw[j * factors..(j + 1) * factors], x_next);
                x.copy_from_slice(x_next);
                out[j * dim..(j + 1) * dim].copy_from_slice(x);
            }
        },
    );
    MultiPaths { steps, dim, data }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Gbm {
        mu: f64,
        sigma: f64,
    }

    impl StochasticProcess1D for Gbm {
        fn drift(&self, _t: f64, x: f64) -> f64 {
            self.mu * x
        }
        fn diffusion(&self, _t: f64, x: f64) -> f64 {
            self.sigma * x
        }
        fn exact_step(&self, _t: f64, x: f64, dt: f64, dw: f64) -> Option<f64> {
            Some(x * ((self.mu - 0.5 * self.sigma * self.sigma) * dt + self.sigma * dw).exp())
        }
    }

    fn cfg(sampler: Sampler) -> SampleConfig {
        SampleConfig { paths: 20_000, steps: 12, horizon: 1.0, sampler, seed: 7 }
    }

    #[test]
    fn terminal_moments_match_the_lognormal_law() {
        let p = Gbm { mu: 0.05, sigma: 0.2 };
        for sampler in [Sampler::Sobol, Sampler::PseudoRandom] {
            let paths = sample_paths_1d(&p, 100.0, DiscretizationScheme::Exact, &cfg(sampler));
            let n = paths.n_paths() as f64;
            let mean: f64 = paths.iter().map(|path| path[path.len() - 1]).sum::<f64>() / n;
            let log_var: f64 = paths
                .iter()
                .map(|path| {
                    let l = (path[path.len() - 1] / 100.0).ln();
                    (l - (0.05 - 0.02)) * (l - (0.05 - 0.02))
                })
                .sum::<f64>()
                / n;
            let target = 100.0 * (0.05_f64).exp();
            assert!((mean - target).abs() / target < 0.01, "{sampler:?}: mean {mean}");
            assert!((log_var - 0.04).abs() / 0.04 < 0.05, "{sampler:?}: log-var {log_var}");
        }
    }

    #[test]
    fn pseudo_paths_come_in_antithetic_pairs() {
        // under the exact GBM step, mirrored increments mirror the log
        // path around the deterministic drift
        let p = Gbm { mu: 0.05, sigma: 0.2 };
        let paths = sample_paths_1d(&p, 100.0, DiscretizationScheme::Exact, &cfg(Sampler::PseudoRandom));
        let dt = 1.0 / 12.0;
        for j in 0..paths.steps() {
            let drift = (0.05 - 0.02) * dt * (j + 1) as f64;
            let sum_logs = (paths.path(0)[j] / 100.0).ln() + (paths.path(1)[j] / 100.0).ln();
            assert!((sum_logs - 2.0 * drift).abs() < 1e-12, "step {j}");
        }
    }

    #[test]
    fn same_seed_reproduces_and_different_seed_differs() {
        let p = Gbm { mu: 0.02, sigma: 0.3 };
        let a = sample_paths_1d(&p, 50.0, DiscretizationScheme::Exact, &cfg(Sampler::Sobol));
        let b = sample_paths_1d(&p, 50.0, DiscretizationScheme::Exact, &cfg(Sampler::Sobol));
        assert_eq!(a.path(123), b.path(123));
        let other = SampleConfig { seed: 8, ..cfg(Sampler::Sobol) };
        let c = sample_paths_1d(&p, 50.0, DiscretizationScheme::Exact, &other);
        assert_ne!(a.path(123), c.path(123));
    }

    /// Two independent log-normal assets as one 2-state, 2-factor process.
    struct TwoGbm;

    impl StochasticProcess for TwoGbm {
        fn dim(&self) -> usize {
            2
        }
        fn factors(&self) -> usize {
            2
        }
        fn drift(&self, _t: f64, x: &[f64], out: &mut [f64]) {
            out[0] = 0.05 * x[0];
            out[1] = 0.01 * x[1];
        }
        fn diffusion(&self, _t: f64, x: &[f64], out: &mut [f64]) {
            out.copy_from_slice(&[0.2 * x[0], 0.0, 0.0, 0.3 * x[1]]);
        }
    }

    #[test]
    fn multi_state_terminal_means_track_their_drifts() {
        for sampler in [Sampler::Sobol, Sampler::PseudoRandom] {
            let cfg = SampleConfig { paths: 40_000, steps: 50, horizon: 1.0, sampler, seed: 3 };
            let paths = sample_paths(&TwoGbm, &[100.0, 200.0], &cfg);
            assert_eq!((paths.n_paths(), paths.steps(), paths.dim()), (40_000, 50, 2));
            let n = paths.n_paths() as f64;
            let (mut m0, mut m1) = (0.0, 0.0);
            for i in 0..paths.n_paths() {
                let last = paths.state(i, paths.steps() - 1);
                m0 += last[0];
                m1 += last[1];
            }
            let (t0, t1) = (100.0 * (0.05_f64).exp(), 200.0 * (0.01_f64).exp());
            // Euler at 50 steps: discretization bias well under the noise floor
            assert!((m0 / n - t0).abs() / t0 < 0.01, "{sampler:?}: {m0}");
            assert!((m1 / n - t1).abs() / t1 < 0.01, "{sampler:?}: {m1}");
        }
    }
}
