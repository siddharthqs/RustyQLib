//! Binomial trees: every parameterization priced, compared, timed.
//!
//! - Prices a spread of equity options (European ATM / deep OTM, cash
//!   binary, American put, Bermudan put) on all six tree types.
//! - Runs convergence ladders per scheme against analytic references and
//!   writes log-log convergence diagrams (slope = convergence order):
//!   `runs/binomial_tree/convergence_{european,american}.html`.
//! - Reports per-scheme stats: time per price, estimated convergence
//!   order, and the steps / time needed to reach 1e-2 and 1e-3 accuracy.
//! - Shows the diagnostic engine: exercise boundary, tree Greeks, exact
//!   agreement with the optimized rolling-array engine.
//!
//! Run with:  cargo run --release --example binomial_tree

mod common;

use std::time::Instant;

use chrono::NaiveDate;
use common::plot3d::{save_lines_html, LineSeries};
use rustyqlib::core::lattice::{convergence_study, BinomialTreeType};
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::{BinaryType, EquityOption};
use rustyqlib::Instrument;

const SPOT: f64 = 100.0;
const STRIKE: f64 = 100.0;
const RATE: f64 = 0.05;
const DIV: f64 = 0.02;
const VOL: f64 = 0.30;
const T: f64 = 1.0;

fn schemes() -> [(&'static str, BinomialTreeType); 6] {
    use BinomialTreeType::*;
    [
        ("CRR", CoxRossRubinstein),
        ("JarrowRudd", JarrowRudd),
        ("Tian", Tian),
        ("Trigeorgis", Trigeorgis),
        ("LeisenReimer", LeisenReimer),
        ("EQP", AdditiveEqp),
    ]
}

fn base() -> EquityOptionBuilder {
    EquityOptionBuilder::new()
        .spot(SPOT)
        .strike(STRIKE)
        .flat_vol(VOL)
        .flat_rate(RATE)
        .dividend_yield(DIV)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 5).unwrap())
        .engine(Engine::Binomial)
}

fn tree(builder: EquityOptionBuilder, tree_type: BinomialTreeType, steps: usize) -> EquityOption {
    builder.tree_type(tree_type).tree_steps(steps).build().expect("option must build")
}

/// The product set: (label, builder configurator, reference price).
fn products() -> Vec<(&'static str, fn(EquityOptionBuilder) -> EquityOptionBuilder, f64)> {
    let quarterly = || {
        vec![
            NaiveDate::from_ymd_opt(2026, 4, 6).unwrap(),
            NaiveDate::from_ymd_opt(2026, 7, 6).unwrap(),
            NaiveDate::from_ymd_opt(2026, 10, 5).unwrap(),
        ]
    };
    let bs_ref = |k: f64, pc: PutOrCall| bs_price(SPOT, k, RATE, DIV, VOL, T, pc);
    // American / Bermudan references: converged Leisen-Reimer at 5001 steps
    let amer_ref = tree(base().american().vanilla(PutOrCall::Put), BinomialTreeType::LeisenReimer, 5001).npv();
    let berm_dates = quarterly();
    let berm_ref = tree(
        base().bermudan(berm_dates).vanilla(PutOrCall::Put),
        BinomialTreeType::LeisenReimer,
        5001,
    )
    .npv();
    // binary reference: closed form cash-or-nothing = e^{-rT} N(d2)
    let d1 = ((SPOT / STRIKE).ln() + (RATE - DIV + 0.5 * VOL * VOL) * T) / (VOL * T.sqrt());
    let d2 = d1 - VOL * T.sqrt();
    let binary_ref = (-RATE * T).exp() * 0.5 * (1.0 + libm::erf(d2 / std::f64::consts::SQRT_2));

    println!("  references: american put {amer_ref:.6} (LR-5001), bermudan put {berm_ref:.6} (LR-5001)");
    vec![
        ("European ATM call", |b: EquityOptionBuilder| b.vanilla(PutOrCall::Call), bs_ref(STRIKE, PutOrCall::Call)),
        ("European 130-call (OTM)", |b| b.strike(130.0).vanilla(PutOrCall::Call), bs_ref(130.0, PutOrCall::Call)),
        ("Cash binary call", |b| b.binary(PutOrCall::Call, BinaryType::CashOrNothing, 1.0), binary_ref),
        ("American put", |b| b.american().vanilla(PutOrCall::Put), amer_ref),
        (
            "Bermudan put (quarterly)",
            |b| {
                b.bermudan(vec![
                    NaiveDate::from_ymd_opt(2026, 4, 6).unwrap(),
                    NaiveDate::from_ymd_opt(2026, 7, 6).unwrap(),
                    NaiveDate::from_ymd_opt(2026, 10, 5).unwrap(),
                ])
                .vanilla(PutOrCall::Put)
            },
            berm_ref,
        ),
    ]
}

fn section_pricing_table() {
    common::section("Prices across schemes at 501 steps (abs error vs reference)");
    print!("  {:<26} {:>12}", "product", "reference");
    for (name, _) in schemes() {
        print!(" {:>14}", name);
    }
    println!();
    for (label, configure, reference) in products() {
        print!("  {:<26} {:>12.6}", label, reference);
        for (_, tree_type) in schemes() {
            let price = tree(configure(base()), tree_type, 501).npv();
            print!(" {:>8.4}({:>4.0e})", price, (price - reference).abs());
        }
        println!();
    }
    common::note("Leisen-Reimer centers the tree on the strike: watch its error column");
    common::note("the binary's discontinuous payoff makes every non-LR scheme oscillate hard");
}

/// Convergence ladders on the core lattice directly; returns per-scheme
/// (steps, |error|) curves.
fn ladders(
    american: bool,
    reference: f64,
    steps_ladder: &[usize],
) -> Vec<(String, Vec<(usize, f64, std::time::Duration)>)> {
    let b = RATE - DIV;
    let terminal_call = |s: f64| (s - STRIKE).max(0.0);
    let terminal_put = |s: f64| (STRIKE - s).max(0.0);
    let exercise = |_: usize, s: f64, cont: f64| (STRIKE - s).max(0.0).max(cont);
    schemes()
        .into_iter()
        .map(|(name, tree_type)| {
            let points = if american {
                convergence_study(
                    tree_type, SPOT, STRIKE, b, VOL, RATE, T, steps_ladder,
                    &terminal_put, Some(&exercise),
                )
            } else {
                convergence_study(
                    tree_type, SPOT, STRIKE, b, VOL, RATE, T, steps_ladder,
                    &terminal_call, None,
                )
            }
            .expect("ladder must price");
            let curve = points
                .iter()
                .map(|p| (p.steps, (p.price - reference).abs().max(1e-12), p.elapsed))
                .collect();
            (name.to_string(), curve)
        })
        .collect()
}

/// Least-squares slope of ln(err) on ln(n): the empirical convergence order.
fn convergence_order(curve: &[(usize, f64, std::time::Duration)]) -> f64 {
    let pts: Vec<(f64, f64)> = curve
        .iter()
        .filter(|(_, e, _)| *e > 1e-11)
        .map(|(n, e, _)| ((*n as f64).ln(), e.ln()))
        .collect();
    let n = pts.len() as f64;
    let (sx, sy): (f64, f64) = pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0, a.1 + p.1));
    let (sxx, sxy): (f64, f64) =
        pts.iter().fold((0.0, 0.0), |a, p| (a.0 + p.0 * p.0, a.1 + p.0 * p.1));
    -(n * sxy - sx * sy) / (n * sxx - sx * sx)
}

fn section_convergence() {
    common::section("Convergence: |error| vs steps (log-log HTML diagrams)");
    let ladder: Vec<usize> = vec![25, 51, 101, 201, 401, 801, 1601, 3201];

    let euro_ref = bs_price(SPOT, STRIKE, RATE, DIV, VOL, T, PutOrCall::Call);
    let amer_ref =
        tree(base().american().vanilla(PutOrCall::Put), BinomialTreeType::LeisenReimer, 5001).npv();

    for (title, american, reference, file) in [
        ("European ATM call vs Black-Scholes", false, euro_ref, "convergence_european"),
        ("American ATM put vs LR-5001", true, amer_ref, "convergence_american"),
    ] {
        let curves = ladders(american, reference, &ladder);
        println!("  {title}");
        print!("  {:<14}", "steps");
        for (name, _) in schemes() {
            print!(" {:>12}", name);
        }
        println!();
        for (row, &steps) in ladder.iter().enumerate() {
            print!("  {:<14}", steps);
            for (_, curve) in &curves {
                print!(" {:>12.2e}", curve[row].1);
            }
            println!();
        }
        print!("  {:<14}", "order (slope)");
        for (_, curve) in &curves {
            print!(" {:>12.2}", convergence_order(curve));
        }
        println!("\n");

        let series: Vec<LineSeries> = curves
            .iter()
            .map(|(name, curve)| LineSeries {
                name: name.clone(),
                xs: curve.iter().map(|(n, _, _)| *n as f64).collect(),
                ys: curve.iter().map(|(_, e, _)| *e).collect(),
            })
            .collect();
        let path = format!("runs/binomial_tree/{file}.html");
        save_lines_html(&series, &path, title, "steps", "|error|", true, true);
        common::note(&format!("saved {path}"));
    }
}

fn section_performance() {
    common::section("Performance: cost of accuracy (American ATM put)");
    let ladder: Vec<usize> = vec![25, 51, 101, 201, 401, 801, 1601, 3201];
    let amer_ref =
        tree(base().american().vanilla(PutOrCall::Put), BinomialTreeType::LeisenReimer, 5001).npv();
    let curves = ladders(true, amer_ref, &ladder);

    println!(
        "  {:<14} {:>12} {:>10} {:>16} {:>16}",
        "scheme", "us @ n=1001", "order", "steps to 1e-2", "steps to 1e-3"
    );
    for ((name, tree_type), (_, curve)) in schemes().into_iter().zip(&curves) {
        // wall time per price at n = 1001 through the public engine
        let option = tree(base().american().vanilla(PutOrCall::Put), tree_type, 1001);
        let reps = 5;
        let start = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(option.npv());
        }
        let micros = start.elapsed().as_micros() as f64 / reps as f64;

        let first_below = |tol: f64| {
            curve
                .iter()
                .find(|(_, e, _)| *e < tol)
                .map(|(n, _, d)| format!("{n} ({:.0}us)", d.as_micros()))
                .unwrap_or_else(|| "> 3201".to_string())
        };
        println!(
            "  {:<14} {:>12.0} {:>10.2} {:>16} {:>16}",
            name,
            micros,
            convergence_order(curve),
            first_below(1e-2),
            first_below(1e-3),
        );
        let _ = tree_type;
    }
    common::note("same O(n^2) work per scheme: the accuracy per step is what differs");
    common::note("LR reaches tolerance with ~10x fewer steps => ~100x less work than CRR");
}

fn section_diagnostics() {
    common::section("Diagnostic engine: boundary, tree Greeks, timing");
    let option = tree(
        base().american().vanilla(PutOrCall::Put),
        BinomialTreeType::LeisenReimer,
        501,
    );
    let diag = rustyqlib::equity::binomial::npv_with_diagnostics(&option);
    common::check("diagnostics price == fast engine", diag.price, option.npv(), 1e-12);
    println!(
        "  tree: {:?} n={} elapsed {:?} (full trees kept; fast engine is O(n) memory)",
        diag.tree_type, diag.steps, diag.elapsed
    );
    println!(
        "  tree Greeks: delta {:.4}  gamma {:.4}  theta {:.4}",
        diag.delta, diag.gamma, diag.theta
    );
    // exercise boundary: highest spot where exercise is optimal, sampled
    let sample: Vec<String> = [50usize, 150, 250, 350, 450]
        .iter()
        .filter_map(|&i| diag.exercise_boundary[i].map(|(_, hi)| format!("t{}: {:.2}", i, hi)))
        .collect();
    println!("  exercise boundary (upper edge by layer): {}", sample.join("  "));
    common::note("the boundary rises toward the strike as expiry approaches — as theory demands");
}

fn main() {
    common::title(&format!(
        "BINOMIAL TREES — S={SPOT} K={STRIKE} r={RATE} q={DIV} vol={VOL} T={T}y"
    ));
    section_pricing_table();
    section_convergence();
    section_performance();
    section_diagnostics();
}
