//! Shared reporting helpers for the runnable product examples.
//!
//! Each example builds one product, prices it on every applicable engine
//! and model, and prints NPV plus Greeks in a single table. Engines that
//! refuse a combination (by design) are caught and reported rather than
//! aborting the run, so these files double as a support matrix.

// each example compiles this module separately and uses a different
// subset of the helpers, so per-example dead-code warnings are noise
#![allow(dead_code)]

use std::panic::{catch_unwind, AssertUnwindSafe};

use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::montecarlo;
use rustyqlib::equity::utils::Engine;
use rustyqlib::equity::vanilla_option::EquityOption;

// Not every example uses the plotter; silence dead-code warnings there.
#[allow(dead_code)]
pub mod plot3d;

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
/// Print a priced row, or the typed build-time refusal for combinations
/// the library rejects (`build()` now enforces engine support upfront).
#[allow(dead_code)]
pub fn row_or_refusal(
    label: &str,
    built: Result<EquityOption, rustyqlib::RustyQLibError>,
) {
    match built {
        Ok(option) => row(label, &option),
        Err(e) => note(&format!("{label}: refused at build() — {e}")),
    }
}

pub fn row(label: &str, option: &EquityOption) {
    // keep the table readable: the caught panic is reported in the row
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = catch_unwind(AssertUnwindSafe(|| {
        // let std_err = if option.engine.kind() == Engine::MonteCarlo {
        //     Some(montecarlo::npv_with_stats(option).std_err)
        // } else {
        //     None
        // };
        let result = option.price().unwrap();
        (result.pv, result.greeks.delta, result.greeks.gamma, result.greeks.vega, result.greeks.theta,
         option.rho(), result.std_err)
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
