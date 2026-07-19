//! Shared reporting helpers for the runnable product examples.
//!
//! Each example builds one product, prices it on every applicable engine
//! and model, and prints NPV plus Greeks in a single table. Engines that
//! refuse a combination (by design) are caught and reported rather than
//! aborting the run, so these files double as a support matrix.

use std::panic::{catch_unwind, AssertUnwindSafe};

use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::montecarlo;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanila_option::EquityOption;

pub fn title(text: &str) {
    println!("\n{}", "=".repeat(96));
    println!("  {text}");
    println!("{}", "=".repeat(96));
}

pub fn section(text: &str) {
    println!("\n-- {text} {}", "-".repeat(90usize.saturating_sub(text.len())));
}

pub fn table_header() {
    println!(
        "{:<34} {:>12} {:>10} {:>10} {:>9} {:>9} {:>9} {:>9}",
        "method", "npv", "delta", "gamma", "vega", "theta", "rho", "std err"
    );
    println!("{}", "-".repeat(96));
}

/// Price `option` and print one row. Panics from unsupported combinations
/// are caught and shown as `unsupported`.
pub fn row(label: &str, option: &EquityOption) {
    // keep the table readable: the caught panic is reported in the row
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (pv, std_err) = if option.engine == Engine::MonteCarlo {
            let s = montecarlo::npv_with_stats(option);
            (s.pv, Some(s.std_err))
        } else {
            (option.npv(), None)
        };
        (pv, option.delta(), option.gamma(), option.vega(), option.theta(), option.rho(), std_err)
    }));
    std::panic::set_hook(hook);
    match result {
        Ok((pv, delta, gamma, vega, theta, rho, std_err)) => {
            let se = match std_err {
                Some(v) => format!("{v:.5}"),
                None => "-".to_string(),
            };
            println!(
                "{label:<34} {pv:>12.6} {delta:>10.5} {gamma:>10.5} {vega:>9.3} {theta:>9.3} {rho:>9.3} {se:>9}"
            );
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panicked".to_string());
            let short: String = msg.split(';').next().unwrap_or(&msg).chars().take(52).collect();
            println!("{label:<34} {:>12}  ({short})", "unsupported");
        }
    }
}

/// Print a labelled scalar, for identities and cross-checks.
pub fn check(label: &str, value: f64, expected: f64, tol: f64) {
    let diff = (value - expected).abs();
    let mark = if diff < tol { "OK " } else { "BAD" };
    println!("  [{mark}] {label:<52} {value:>13.6}  expected {expected:>13.6}  diff {diff:.2e}");
}

pub fn note(text: &str) {
    println!("  . {text}");
}
