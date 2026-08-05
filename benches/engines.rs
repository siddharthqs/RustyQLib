//! Engine benchmarks: one representative pricing per engine, plus implied
//! vol and a small batch, all through the public builder + `price()` API.
//!
//! Run with `cargo bench`; criterion compares against the previous run and
//! flags statistically significant regressions. Seeds and grids are fixed
//! so results are comparable run to run.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

use rustyqlib::core::trade::PutOrCall;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::heston::HestonParams;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::EquityOption;
use rustyqlib::Instrument;

fn vanilla(engine: Engine) -> EquityOption {
    EquityOptionBuilder::new()
        .spot(100.0)
        .strike(100.0)
        .flat_vol(0.30)
        .flat_rate(0.05)
        .dividend_yield(0.01)
        .years_to_maturity(1.0)
        .vanilla(PutOrCall::Call)
        .engine(engine)
        .seed(42)
        .build()
        .expect("bench option must build")
}

fn american(engine: Engine) -> EquityOption {
    EquityOptionBuilder::new()
        .spot(100.0)
        .strike(100.0)
        .flat_vol(0.30)
        .flat_rate(0.05)
        .years_to_maturity(1.0)
        .american()
        .vanilla(PutOrCall::Put)
        .engine(engine)
        .seed(42)
        .build()
        .expect("bench option must build")
}

fn closed_forms(c: &mut Criterion) {
    let bs = vanilla(Engine::BlackScholes);
    c.bench_function("bs_closed_form_npv", |b| b.iter(|| black_box(&bs).npv()));
    c.bench_function("bs_closed_form_price_all_greeks", |b| {
        b.iter(|| black_box(&bs).price().unwrap())
    });

    let target = bs.npv();
    c.bench_function("implied_vol_solve", |b| {
        b.iter(|| black_box(&bs).try_imp_vol(black_box(target)).unwrap())
    });

    let baw = american(Engine::BaroneAdesiWhaley);
    c.bench_function("american_baw_npv", |b| b.iter(|| black_box(&baw).npv()));
    let bs2002 = american(Engine::BjerksundStensland);
    c.bench_function("american_bs2002_npv", |b| {
        b.iter(|| black_box(&bs2002).npv())
    });
}

fn heston_smile(c: &mut Criterion) {
    use rustyqlib::core::trade::PutOrCall;
    use rustyqlib::equity::heston::{cos_smile, heston_price};
    let hp = HestonParams {
        v0: 0.09,
        kappa: 2.0,
        theta: 0.09,
        vol_of_vol: 0.4,
        rho: -0.7,
    };
    let strikes: Vec<f64> = (0..20).map(|i| 70.0 + 3.0 * i as f64).collect();
    let (s, r, q, t) = (100.0, 0.03, 0.01, 1.0);

    // the calibration workload: a whole smile per objective evaluation
    c.bench_function("heston_smile_20_strikes_integration", |b| {
        b.iter(|| {
            strikes
                .iter()
                .map(|&k| heston_price(s, k, r, q, t, black_box(&hp), PutOrCall::Call))
                .sum::<f64>()
        })
    });
    c.bench_function("heston_smile_20_strikes_cos", |b| {
        b.iter(|| {
            cos_smile(s, r, q, t, black_box(&hp), &strikes, PutOrCall::Call)
                .iter()
                .sum::<f64>()
        })
    });
}

fn heston_cf(c: &mut Criterion) {
    let heston = EquityOptionBuilder::new()
        .spot(100.0)
        .strike(100.0)
        .flat_rate(0.05)
        .years_to_maturity(1.0)
        .vanilla(PutOrCall::Call)
        .heston(HestonParams {
            v0: 0.09,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        })
        .engine(Engine::BlackScholes)
        .build()
        .expect("bench option must build");
    c.bench_function("heston_cf_vanilla_npv", |b| {
        b.iter(|| black_box(&heston).npv())
    });
}

fn lattice_and_grid(c: &mut Criterion) {
    let mut group = c.benchmark_group("grids");
    group.sample_size(20);

    let bino = american(Engine::Binomial);
    group.bench_function("binomial_american_npv", |b| {
        b.iter(|| black_box(&bino).npv())
    });

    // Leisen-Reimer reaches CRR-1000 accuracy at ~100 steps: ~100x faster
    let mut lr = american(Engine::Binomial);
    lr.engine = rustyqlib::equity::utils::PricingEngine::Binomial(
        rustyqlib::core::lattice::LatticeConfig {
            tree_type: rustyqlib::core::lattice::BinomialTreeType::LeisenReimer,
            steps: 101,
            ..Default::default()
        },
    );
    group.bench_function("binomial_lr101_american_npv", |b| {
        b.iter(|| black_box(&lr).npv())
    });

    let fd = american(Engine::FiniteDifference); // default 400 x 400 grid
    group.bench_function("fd_american_400x400_npv", |b| {
        b.iter(|| black_box(&fd).npv())
    });

    group.finish();
}

fn monte_carlo(c: &mut Criterion) {
    let mut group = c.benchmark_group("monte_carlo");
    // full 100k-path simulations per iteration: keep sampling cheap
    group.sample_size(10);

    let mc = vanilla(Engine::MonteCarlo); // default 100k paths, Sobol
    group.bench_function("mc_vanilla_100k_paths_npv", |b| {
        b.iter(|| black_box(&mc).npv())
    });

    group.finish();
}

fn batch(c: &mut Criterion) {
    // a small book of vanillas across strikes: per-price overhead of the
    // public API, the shape a service or the Python batch layer would use
    let book: Vec<EquityOption> = (0..1000)
        .map(|i| {
            EquityOptionBuilder::new()
                .spot(100.0)
                .strike(60.0 + 0.08 * i as f64)
                .flat_vol(0.30)
                .flat_rate(0.05)
                .years_to_maturity(1.0)
                .vanilla(if i % 2 == 0 {
                    PutOrCall::Call
                } else {
                    PutOrCall::Put
                })
                .engine(Engine::BlackScholes)
                .build()
                .expect("bench option must build")
        })
        .collect();
    c.bench_function("batch_1000_closed_form_npv", |b| {
        b.iter(|| black_box(&book).iter().map(|o| o.npv()).sum::<f64>())
    });
}

criterion_group!(
    benches,
    closed_forms,
    heston_smile,
    heston_cf,
    lattice_and_grid,
    monte_carlo,
    batch
);
criterion_main!(benches);
