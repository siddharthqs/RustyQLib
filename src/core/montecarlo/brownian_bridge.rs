//! Brownian-bridge path construction: the first draw fixes the terminal
//! value, subsequent draws fill midpoints by bisection, so
//! low-discrepancy coordinates are spent on the dimensions that matter
//! most. Weights are precomputed once and shared across paths.
//!
//! Produces per-step Brownian increments; a multi-factor model uses one
//! bridge per factor.

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

#[cfg(test)]
mod tests {
    use super::super::rng::path_normals;
    use super::*;

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
}
