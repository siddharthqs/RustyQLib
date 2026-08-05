//! The public path-generation API: simulate model paths directly —
//! scenario generation, stress paths, exposure profiles — without going
//! through a pricer.
//!
//! - GBM paths from a `BlackScholesProcess` under the Sobol sampler,
//!   with a terminal-distribution summary against the lognormal law.
//! - Heston `(S, v)` paths through the two-factor `HestonProcess`
//!   (Andersen QE), with a peek at the joint terminal state.
//! - A scenario statistic a pricer would never expose: the distribution
//!   of each path's running maximum drawdown.
//!
//! Run with:  cargo run --release --example sample_paths

use rustyqlib::core::montecarlo::{
    sample_paths, sample_paths_1d, DiscretizationScheme, SampleConfig, Sampler,
};
use rustyqlib::equity::heston::HestonParams;
use rustyqlib::equity::processes::{BlackScholesProcess, HestonProcess, HestonScheme, VolDynamics};

const SPOT: f64 = 100.0;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;
const VOL: f64 = 0.30;
const T: f64 = 1.0;

fn percentile(sorted: &[f64], p: f64) -> f64 {
    sorted[((sorted.len() - 1) as f64 * p) as usize]
}

fn main() {
    let cfg = SampleConfig {
        paths: 100_000,
        steps: 252,
        horizon: T,
        sampler: Sampler::Sobol,
        seed: 42,
    };

    // ── 1. GBM paths from the scalar process
    let gbm = BlackScholesProcess::new(RATE - DIV, VolDynamics::Const(VOL));
    let paths = sample_paths_1d(&gbm, SPOT, DiscretizationScheme::Exact, &cfg);

    let mut terminals: Vec<f64> = paths.iter().map(|p| p[p.len() - 1]).collect();
    terminals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = terminals.iter().sum::<f64>() / terminals.len() as f64;
    println!(
        "GBM terminal spot over {} paths x {} steps",
        paths.n_paths(),
        paths.steps()
    );
    println!(
        "  mean    {:>8.3}   (forward {:.3})",
        mean,
        SPOT * ((RATE - DIV) * T).exp()
    );
    println!(
        "  p5/p50/p95  {:>7.2} / {:>7.2} / {:>7.2}",
        percentile(&terminals, 0.05),
        percentile(&terminals, 0.50),
        percentile(&terminals, 0.95)
    );

    // ── 2. a path statistic no pricer exposes: maximum drawdown
    let mut drawdowns: Vec<f64> = paths
        .iter()
        .map(|path| {
            let (mut peak, mut worst) = (SPOT, 0.0_f64);
            for &s in path {
                peak = peak.max(s);
                worst = worst.max(1.0 - s / peak);
            }
            worst
        })
        .collect();
    drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("\nmax drawdown distribution");
    println!(
        "  p50 {:.1}%   p95 {:.1}%   p99 {:.1}%",
        100.0 * percentile(&drawdowns, 0.50),
        100.0 * percentile(&drawdowns, 0.95),
        100.0 * percentile(&drawdowns, 0.99)
    );

    // ── 3. Heston (S, v) paths through the two-factor process
    let heston = HestonProcess {
        drift_rate: RATE - DIV,
        params: HestonParams {
            v0: 0.09,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        },
        scheme: HestonScheme::QuadraticExponential,
    };
    let sv = sample_paths(&heston, &[SPOT, 0.09], &cfg);

    let n = sv.n_paths();
    let last = sv.steps() - 1;
    let mut s_mean = 0.0;
    let mut vol_mean = 0.0;
    let mut crash_vol = Vec::new(); // instantaneous vol on paths that fell 20%+
    for i in 0..n {
        let state = sv.state(i, last);
        s_mean += state[0];
        vol_mean += state[1].max(0.0).sqrt();
        if state[0] < 0.8 * SPOT {
            crash_vol.push(state[1].max(0.0).sqrt());
        }
    }
    println!("\nHeston terminal state over {n} paths (QE)");
    println!(
        "  E[S_T]      {:>8.3}   (forward {:.3})",
        s_mean / n as f64,
        SPOT * ((RATE - DIV) * T).exp()
    );
    println!(
        "  E[sqrt(v_T)]  {:>6.1}%  (long-run {:.1}%)",
        100.0 * vol_mean / n as f64,
        100.0 * 0.09_f64.sqrt()
    );
    let crash_mean = crash_vol.iter().sum::<f64>() / crash_vol.len().max(1) as f64;
    println!(
        "  E[sqrt(v_T) | S_T < 80]  {:>5.1}%   <- rho = -0.7: crashes end in high vol",
        100.0 * crash_mean
    );
}
