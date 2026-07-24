//! Differential evolution (DE/rand/1/bin): seeded, bounded global
//! search. The tool for multimodal calibration landscapes — find the
//! basin globally, then polish with BFGS or Levenberg-Marquardt.

use super::{OptimConfig, OptimResult};

const DIFFERENTIAL_WEIGHT: f64 = 0.8; // F
const CROSSOVER_RATE: f64 = 0.9; // CR

/// Minimize `f` inside the per-parameter `bounds` box with
/// DE/rand/1/bin. Deterministic for a given `seed`; the population size
/// is `max(15, 10 * dim)`. Converged when the population's value spread
/// falls below `tol`; `iterations` counts generations.
pub fn differential_evolution(
    cfg: &OptimConfig,
    f: &dyn Fn(&[f64]) -> f64,
    bounds: &[(f64, f64)],
    seed: u64,
) -> OptimResult {
    let dim = bounds.len();
    assert!(dim > 0, "bounds must give at least one parameter");
    for &(lo, hi) in bounds {
        assert!(lo < hi, "each bound needs lo < hi");
    }
    let np = (10 * dim).max(15);
    let mut rng = Xorshift64Star::new(seed);

    // random initial population in the box
    let mut pop: Vec<Vec<f64>> = (0..np)
        .map(|_| bounds.iter().map(|&(lo, hi)| lo + (hi - lo) * rng.uniform()).collect())
        .collect();
    let mut values: Vec<f64> = pop.iter().map(|x| f(x)).collect();

    for gen in 1..=cfg.max_iter {
        for i in 0..np {
            // three distinct partners, none equal to i
            let mut pick = || loop {
                let j = (rng.next() % np as u64) as usize;
                if j != i {
                    return j;
                }
            };
            let (a, b, c) = {
                let a = pick();
                let b = loop {
                    let b = pick();
                    if b != a {
                        break b;
                    }
                };
                let c = loop {
                    let c = pick();
                    if c != a && c != b {
                        break c;
                    }
                };
                (a, b, c)
            };
            // mutate + binomial crossover (j_rand guarantees one gene moves)
            let j_rand = (rng.next() % dim as u64) as usize;
            let mut trial = pop[i].clone();
            for j in 0..dim {
                if j == j_rand || rng.uniform() < CROSSOVER_RATE {
                    let v = pop[a][j] + DIFFERENTIAL_WEIGHT * (pop[b][j] - pop[c][j]);
                    trial[j] = v.clamp(bounds[j].0, bounds[j].1);
                }
            }
            let f_trial = f(&trial);
            if f_trial <= values[i] {
                pop[i] = trial;
                values[i] = f_trial;
            }
        }
        let best = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if worst - best <= cfg.tol * (1.0 + best.abs()) {
            let (i, &value) =
                values.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).expect("non-empty");
            return OptimResult { x: pop[i].clone(), value, iterations: gen, converged: true };
        }
    }
    let (i, &value) =
        values.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).expect("non-empty");
    OptimResult { x: pop[i].clone(), value, iterations: cfg.max_iter, converged: false }
}

/// Small deterministic RNG (xorshift64*), so runs are reproducible for a
/// given seed without pulling in an external crate.
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn finds_the_global_minimum_of_rastrigin() {
        // many local minima; the global one is 0 at the origin
        let f = |x: &[f64]| {
            20.0 + x.iter().map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos()).sum::<f64>()
        };
        let bounds = [(-5.12, 5.12), (-5.12, 5.12)];
        let r = differential_evolution(&OptimConfig::new(1e-10, 600), &f, &bounds, 7);
        assert!(r.value < 1e-6, "stuck at a local minimum: {r:?}");
        assert!(r.x.iter().all(|xi| xi.abs() < 1e-3), "{:?}", r.x);
    }

    #[test]
    fn is_deterministic_for_a_seed_and_respects_bounds() {
        let f = |x: &[f64]| (x[0] - 0.5).powi(2) + (x[1] - 0.25).powi(2);
        let bounds = [(0.0, 1.0), (0.0, 1.0)];
        let cfg = OptimConfig::new(1e-12, 300);
        let a = differential_evolution(&cfg, &f, &bounds, 123);
        let b = differential_evolution(&cfg, &f, &bounds, 123);
        assert_eq!(a.x, b.x, "same seed must reproduce the same run");
        assert!(a.x.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!((a.x[0] - 0.5).abs() < 1e-5 && (a.x[1] - 0.25).abs() < 1e-5, "{a:?}");
        // a different seed still finds the optimum
        let c = differential_evolution(&cfg, &f, &bounds, 999);
        assert!((c.x[0] - 0.5).abs() < 1e-5, "{c:?}");
    }
}
