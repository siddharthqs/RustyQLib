//! Rainbow (multi-asset) options: best-of, worst-of, spread, basket and
//! exchange payoffs on correlated assets.
//!
//! Run with:  cargo run --release --example rainbow_option

mod common;

use chrono::{Duration, Local};
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::equity::rainbow::{RainbowAssetData, RainbowOption, RainbowOptionData};
use rustyqlib::equity::utils::Engine;

const SPOT_A: f64 = 100.0;
const SPOT_B: f64 = 95.0;
const VOL_A: f64 = 0.30;
const VOL_B: f64 = 0.25;
const DIV_A: f64 = 0.02;
const DIV_B: f64 = 0.01;
const RATE: f64 = 0.05;

fn maturity_1y() -> String {
    (Local::now().date_naive() + Duration::days(365)).format("%Y-%m-%d").to_string()
}

fn two_assets() -> Vec<RainbowAssetData> {
    vec![
        RainbowAssetData { symbol: "AAA".into(), spot: SPOT_A, volatility: VOL_A, dividend: Some(DIV_A) },
        RainbowAssetData { symbol: "BBB".into(), spot: SPOT_B, volatility: VOL_B, dividend: Some(DIV_B) },
    ]
}

fn build(
    rainbow_type: &str,
    pc: &str,
    strike: Option<f64>,
    rho: f64,
    pricer: &str,
    assets: Vec<RainbowAssetData>,
    correlations: Vec<Vec<f64>>,
    weights: Option<Vec<f64>>,
) -> Box<RainbowOption> {
    let _ = rho;
    RainbowOption::from_json(&RainbowOptionData {
        symbol: rainbow_type.to_uppercase(),
        rainbow_type: rainbow_type.to_string(),
        put_or_call: Some(pc.to_string()),
        assets,
        correlations,
        strike_price: strike,
        weights,
        maturity: maturity_1y(),
        risk_free_rate: Some(RATE),
        discount_curve: None,
        pricer: Some(pricer.to_string()),
        simulation: Some(100_000),
        mc_sampler: None,
        mc_seed: None,
    })
}

fn two_asset(rainbow_type: &str, pc: &str, strike: Option<f64>, rho: f64, pricer: &str) -> Box<RainbowOption> {
    build(
        rainbow_type,
        pc,
        strike,
        rho,
        pricer,
        two_assets(),
        vec![vec![1.0, rho], vec![rho, 1.0]],
        None,
    )
}

fn print_rainbow(label: &str, option: &RainbowOption) {
    let pv = option.npv();
    let stats = option.npv_with_stats();
    let deltas: Vec<String> = option.deltas().iter().map(|d| format!("{d:.4}")).collect();
    let vegas: Vec<String> = option.vegas().iter().map(|v| format!("{v:.2}")).collect();
    let se = match stats {
        Some(s) => format!("{:.5}", s.std_err),
        None => "-".to_string(),
    };
    println!(
        "{label:<38} {pv:>12.6}  stderr={se:>9}  deltas=[{}]  vegas=[{}]",
        deltas.join(", "),
        vegas.join(", ")
    );
}

fn main() {
    common::title(&format!(
        "RAINBOW OPTIONS — A: S={SPOT_A} sigma={VOL_A} q={DIV_A} | B: S={SPOT_B} sigma={VOL_B} q={DIV_B} | r={RATE} T=1y"
    ));

    common::section("Exchange option (Margrabe, exact) — pays (S_A - S_B)+");
    print_rainbow("Analytical (Margrabe)", &two_asset("exchange", "C", None, 0.6, "Analytical"));
    print_rainbow("Monte Carlo", &two_asset("exchange", "C", None, 0.6, "MC"));

    common::section("Spread option (Kirk approximation) — pays (S_A - S_B - K)+");
    for k in [0.0, 5.0, 10.0] {
        print_rainbow(
            &format!("Analytical (Kirk), K={k}"),
            &two_asset("spread", "C", Some(k), 0.6, "Analytical"),
        );
        print_rainbow(
            &format!("Monte Carlo,     K={k}"),
            &two_asset("spread", "C", Some(k), 0.6, "MC"),
        );
    }
    common::note("at K=0 the spread option must equal the Margrabe exchange option");

    common::section("Best-of and worst-of (Monte Carlo only)");
    for k in [90.0, 100.0, 110.0] {
        print_rainbow(&format!("best-of call,  K={k}"), &two_asset("best_of", "C", Some(k), 0.6, "MC"));
        print_rainbow(&format!("worst-of call, K={k}"), &two_asset("worst_of", "C", Some(k), 0.6, "MC"));
    }
    print_rainbow(
        "best-of, analytic (unsupported)",
        &two_asset("best_of", "C", Some(100.0), 0.6, "MC"),
    );

    common::section("Correlation sweep (worst-of call, K=100)");
    for rho in [-0.5, 0.0, 0.5, 0.9, 0.99] {
        print_rainbow(&format!("rho = {rho:>5}"), &two_asset("worst_of", "C", Some(100.0), rho, "MC"));
    }
    common::note("higher correlation lifts the minimum, so the worst-of call gains value");

    common::section("Basket option (3 assets, moment matching)");
    let assets3 = vec![
        RainbowAssetData { symbol: "AAA".into(), spot: 100.0, volatility: 0.30, dividend: None },
        RainbowAssetData { symbol: "BBB".into(), spot: 90.0, volatility: 0.25, dividend: None },
        RainbowAssetData { symbol: "CCC".into(), spot: 110.0, volatility: 0.35, dividend: None },
    ];
    let corr3 = vec![
        vec![1.0, 0.5, 0.3],
        vec![0.5, 1.0, 0.4],
        vec![0.3, 0.4, 1.0],
    ];
    print_rainbow(
        "Analytical (moment matching)",
        &build("basket", "C", Some(100.0), 0.0, "Analytical", assets3.clone(), corr3.clone(), None),
    );
    print_rainbow(
        "Monte Carlo",
        &build("basket", "C", Some(100.0), 0.0, "MC", assets3.clone(), corr3.clone(), None),
    );
    print_rainbow(
        "Weighted 40/30/30, analytic",
        &build(
            "basket",
            "C",
            Some(100.0),
            0.0,
            "Analytical",
            assets3,
            corr3,
            Some(vec![0.4, 0.3, 0.3]),
        ),
    );

    common::section("Identities");
    let spread_k0 = two_asset("spread", "C", Some(0.0), 0.6, "Analytical").npv();
    let exchange = two_asset("exchange", "C", None, 0.6, "Analytical").npv();
    common::check("spread(K=0) = Margrabe", spread_k0, exchange, 1e-10);

    // max + min = S_A + S_B pathwise, so the two options sum to the vanillas
    let k = 100.0;
    let best = two_asset("best_of", "C", Some(k), 0.6, "MC").npv();
    let worst = two_asset("worst_of", "C", Some(k), 0.6, "MC").npv();
    let vanillas = bs_price(SPOT_A, k, RATE, DIV_A, VOL_A, 1.0, PutOrCall::Call)
        + bs_price(SPOT_B, k, RATE, DIV_B, VOL_B, 1.0, PutOrCall::Call);
    common::check("best-of + worst-of = sum of vanillas", best + worst, vanillas, 0.1);
    println!();
}
