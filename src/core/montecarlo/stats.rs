//! Simulation statistics: mean / standard error from accumulated sums
//! (the shape parallel path loops naturally produce) and a numerically
//! stable Welford accumulator for streaming use.

/// A simulation estimate: sample mean, its standard error, and the
/// sample size.
#[derive(Debug, Clone, Copy)]
pub struct SimStats {
    pub mean: f64,
    pub std_err: f64,
    pub n: usize,
}

/// Mean and standard error of the mean from `sum` and `sum_sq` of `n`
/// samples — the reduction shape of a parallel path loop.
pub fn mean_std_err(sum: f64, sum_sq: f64, n: usize) -> (f64, f64) {
    let nf = n as f64;
    let mean = sum / nf;
    let var = (sum_sq / nf - mean * mean).max(0.0);
    (mean, (var / nf).sqrt())
}

/// Welford's online mean/variance accumulator: numerically stable for
/// long streams and mergeable across partial results.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunningStats {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    /// Merge another accumulator (parallel reduction).
    pub fn merge(&mut self, other: &RunningStats) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let n1 = self.n as f64;
        let n2 = other.n as f64;
        let delta = other.mean - self.mean;
        let n = n1 + n2;
        self.mean += delta * n2 / n;
        self.m2 += other.m2 + delta * delta * n1 * n2 / n;
        self.n += other.n;
    }

    pub fn count(&self) -> usize {
        self.n
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Population variance (divide by n, matching [`mean_std_err`]).
    pub fn variance(&self) -> f64 {
        if self.n == 0 { 0.0 } else { (self.m2 / self.n as f64).max(0.0) }
    }

    pub fn std_err(&self) -> f64 {
        if self.n == 0 { 0.0 } else { (self.variance() / self.n as f64).sqrt() }
    }

    pub fn stats(&self) -> SimStats {
        SimStats { mean: self.mean(), std_err: self.std_err(), n: self.n }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_the_two_pass_computation() {
        let xs: Vec<f64> = (0..1000).map(|i| ((i * 37) % 101) as f64 * 0.13 - 5.0).collect();
        let mut running = RunningStats::new();
        for &x in &xs {
            running.push(x);
        }
        let sum: f64 = xs.iter().sum();
        let sum_sq: f64 = xs.iter().map(|x| x * x).sum();
        let (mean, se) = mean_std_err(sum, sum_sq, xs.len());
        assert!((running.mean() - mean).abs() < 1e-10);
        assert!((running.std_err() - se).abs() < 1e-10);
    }

    #[test]
    fn merged_accumulators_equal_a_single_pass() {
        let xs: Vec<f64> = (0..500).map(|i| (i as f64).sin() * 3.0).collect();
        let mut whole = RunningStats::new();
        let mut left = RunningStats::new();
        let mut right = RunningStats::new();
        for (i, &x) in xs.iter().enumerate() {
            whole.push(x);
            if i < 200 { left.push(x) } else { right.push(x) }
        }
        left.merge(&right);
        assert_eq!(left.count(), whole.count());
        assert!((left.mean() - whole.mean()).abs() < 1e-12);
        assert!((left.variance() - whole.variance()).abs() < 1e-12);
    }
}
