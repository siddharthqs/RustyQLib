//! Integration tests for the `rustyqlib` binary: exit codes, stream
//! discipline (results on stdout, errors on stderr), stdin/stdout piping,
//! and the per-contract error contract in batch pricing.
//!
//! Fixtures pin `valuation_date`, so prices are deterministic across run
//! dates; `GOLDEN_PV` is the Black-Scholes value of the vanilla fixture.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str::contains;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// ATM call, spot 100, strike 100, vol 0.25, rate 0.03, one year (Act/365).
const GOLDEN_PV: f64 = 11.348476825143514;

fn cli() -> Command {
    Command::cargo_bin("rustyqlib").expect("binary builds with --features cli")
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn stdout_json(cmd: &mut Command) -> Value {
    let assert = cmd.assert().success();
    serde_json::from_slice(&assert.get_output().stdout).expect("stdout must be valid JSON")
}

// ---------------------------------------------------------------- exit codes

#[test]
fn no_args_prints_help_and_exits_2() {
    cli()
        .assert()
        .code(2)
        .stderr(contains("Usage:"))
        .stderr(contains("price"));
}

#[test]
fn unknown_subcommand_exits_2() {
    cli()
        .arg("frobnicate")
        .assert()
        .code(2)
        .stderr(contains("unrecognized subcommand"));
}

#[test]
fn help_lists_new_commands_and_hides_deprecated_ones() {
    cli()
        .arg("--help")
        .assert()
        .success()
        .stdout(contains("price"))
        .stdout(contains("stress"))
        .stdout(contains("risk"))
        .stdout(contains("implied-vol"))
        .stdout(contains("deprecated").not());
}

// ---------------------------------------------------------------------- price

#[test]
fn price_file_to_stdout_matches_golden() {
    let results = stdout_json(cli().args(["price", "-i"]).arg(fixture("vanilla.json")));
    let output = &results[0]["output"];
    let pv = output["pv"].as_f64().expect("pv must be a number");
    assert!(
        (pv - GOLDEN_PV).abs() < 1e-9,
        "pv {pv} vs golden {GOLDEN_PV}"
    );
    assert!(output["delta"].as_f64().unwrap() > 0.5, "ATM call delta");
}

#[test]
fn price_reads_stdin_and_writes_stdout() {
    let contents = std::fs::read_to_string(fixture("vanilla.json")).unwrap();
    let results = stdout_json(cli().args(["price", "-i", "-"]).write_stdin(contents));
    let pv = results[0]["output"]["pv"].as_f64().unwrap();
    assert!((pv - GOLDEN_PV).abs() < 1e-9);
}

#[test]
fn price_writes_output_file() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("results.json");
    cli()
        .args(["price", "-i"])
        .arg(fixture("vanilla.json"))
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    let results: Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).expect("valid JSON file");
    assert!((results[0]["output"]["pv"].as_f64().unwrap() - GOLDEN_PV).abs() < 1e-9);
}

#[test]
fn price_format_flag_forces_xml_on_stdout() {
    cli()
        .args(["price", "--format", "xml", "-i"])
        .arg(fixture("vanilla.json"))
        .assert()
        .success()
        .stdout(contains("<?xml"))
        .stdout(contains("<results>"));
}

#[test]
fn price_missing_input_fails_cleanly() {
    cli()
        .args(["price", "-i", "does_not_exist.json"])
        .assert()
        .code(1)
        .stdout(predicates::str::is_empty())
        .stderr(contains("error: failed to read does_not_exist.json"));
}

#[test]
fn price_malformed_document_fails_cleanly() {
    cli()
        .args(["price", "-i", "-"])
        .write_stdin("{ this is not json")
        .assert()
        .code(1)
        .stderr(contains("failed to parse"));
}

#[test]
fn price_empty_contract_list_writes_nothing() {
    cli()
        .args(["price", "-i", "-"])
        .write_stdin(r#"{ "asset": "EQ", "contracts": [] }"#)
        .assert()
        .success()
        .stdout(predicates::str::is_empty());
}

#[test]
fn bad_contract_in_a_batch_does_not_abort_the_rest() {
    let results = stdout_json(cli().args(["price", "-i"]).arg(fixture("bad_batch.json")));
    // the good contract still prices
    assert!((results[0]["output"]["pv"].as_f64().unwrap() - GOLDEN_PV).abs() < 1e-9);
    // the bad one reports its error in place instead of killing the batch
    let error = results[1]["output"]["error"].as_str().expect("error field");
    assert!(
        error.contains("unsupported action/asset"),
        "unexpected error: {error}"
    );
}

#[test]
fn price_directory_requires_output_dir() {
    let dir = tempfile::tempdir().unwrap();
    cli()
        .args(["price", "-i"])
        .arg(dir.path())
        .assert()
        .code(1)
        .stderr(contains("--output <DIR> is required"));
}

#[test]
fn price_directory_creates_missing_output_dir() {
    let dir = tempfile::tempdir().unwrap();
    let in_dir = dir.path().join("in");
    let out_dir = dir.path().join("out").join("nested");
    std::fs::create_dir(&in_dir).unwrap();
    std::fs::copy(fixture("vanilla.json"), in_dir.join("vanilla.json")).unwrap();

    cli()
        .args(["price", "-i"])
        .arg(&in_dir)
        .arg("-o")
        .arg(&out_dir)
        .assert()
        .success();
    let results: Value =
        serde_json::from_str(&std::fs::read_to_string(out_dir.join("vanilla.json")).unwrap())
            .unwrap();
    assert!((results[0]["output"]["pv"].as_f64().unwrap() - GOLDEN_PV).abs() < 1e-9);
}

// --------------------------------------------------------------------- stress

#[test]
fn stress_reports_per_scenario_and_per_trade_results() {
    let results = stdout_json(
        cli()
            .args(["stress", "-i"])
            .arg(fixture("portfolio.json"))
            .arg("-c")
            .arg(fixture("scenarios.toml")),
    );
    let scenarios = results.as_array().expect("array of scenarios");
    assert_eq!(scenarios.len(), 2);
    assert_eq!(scenarios[0]["scenario"], "equity_crash");
    assert_eq!(scenarios[0]["trades"].as_array().unwrap().len(), 2);
    // long call + short put: a crash loses money on both legs
    let crash_pnl = scenarios[0]["stress_pnl"].as_f64().unwrap();
    assert!(crash_pnl < 0.0, "crash pnl {crash_pnl}");
    // aggregation identity: portfolio = sum of trades
    let sum: f64 = scenarios[0]["trades"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["stress_pnl"].as_f64().unwrap())
        .sum();
    assert!((crash_pnl - sum).abs() < 1e-9);
}

#[test]
fn stress_rejects_stdin_for_both_inputs() {
    cli()
        .args(["stress", "-i", "-", "-c", "-"])
        .assert()
        .code(1)
        .stderr(contains("only one of --input and --config"));
}

// ----------------------------------------------------------------------- risk

#[test]
fn risk_reports_both_methods_by_default() {
    let report = stdout_json(
        cli()
            .args(["risk", "--scenarios", "500", "-i"])
            .arg(fixture("portfolio.json")),
    );
    for method in ["delta_gamma", "full_revaluation"] {
        let var = report[method]["var"].as_f64().expect("var must be present");
        let es = report[method]["expected_shortfall"].as_f64().unwrap();
        assert!(var > 0.0, "{method} var {var}");
        assert!(es >= var, "{method} es {es} < var {var}");
    }
    assert_eq!(report["config"]["scenarios"], 500);
}

#[test]
fn risk_method_flag_selects_one_estimator() {
    let report = stdout_json(
        cli()
            .args([
                "risk",
                "--scenarios",
                "500",
                "--method",
                "delta-gamma",
                "-i",
            ])
            .arg(fixture("portfolio.json")),
    );
    assert!(report.get("delta_gamma").is_some());
    assert!(report.get("full_revaluation").is_none());
}

#[test]
fn risk_rejects_invalid_confidence() {
    cli()
        .args(["risk", "--confidence", "1.5", "-i"])
        .arg(fixture("portfolio.json"))
        .assert()
        .code(1)
        .stderr(contains("--confidence must be strictly between 0 and 1"));
}

#[test]
fn risk_rejects_a_mixed_underlying_portfolio() {
    let mixed = std::fs::read_to_string(fixture("portfolio.json"))
        .unwrap()
        .replacen("\"ACME\"", "\"OTHER\"", 1);
    cli()
        .args(["risk", "-i", "-"])
        .write_stdin(mixed)
        .assert()
        .code(1)
        .stderr(contains("must share one underlying"));
}

// ---------------------------------------------------------------- implied-vol

#[test]
fn implied_vol_round_trips_the_golden_price() {
    // invert the vanilla fixture's own price: same terms, so the solve
    // must recover the fixture's 25% vol
    let report = stdout_json(cli().args([
        "implied-vol",
        "--spot",
        "100",
        "--strike",
        "100",
        "--rate",
        "0.03",
        "--maturity",
        "1.0",
        "-p",
        "C",
        "--price",
        &GOLDEN_PV.to_string(),
    ]));
    let vol = report["implied_vol"].as_f64().unwrap();
    assert!((vol - 0.25).abs() < 1e-6, "implied vol {vol}");
}

#[test]
fn implied_vol_rejects_arbitrage_violating_prices() {
    cli()
        .args([
            "implied-vol",
            "--spot",
            "100",
            "--strike",
            "100",
            "--maturity",
            "1.0",
            "-p",
            "C",
            "--price",
            "200",
        ])
        .assert()
        .code(1)
        .stderr(contains("arbitrage"));
}

// ------------------------------------------------------ completions, verbosity

#[test]
fn completions_generate_for_bash() {
    cli()
        .args(["completions", "bash"])
        .assert()
        .success()
        .stdout(contains("_rustyqlib"));
}

#[test]
fn warnings_appear_by_default() {
    cli()
        .env_remove("RUST_LOG")
        .args(["price", "-i", "-"])
        .write_stdin(r#"{ "asset": "EQ", "contracts": [] }"#)
        .assert()
        .success()
        .stderr(contains("no contracts found"));
}

#[test]
fn quiet_flag_suppresses_warnings() {
    cli()
        .env_remove("RUST_LOG")
        .args(["price", "-i", "-", "-q"])
        .write_stdin(r#"{ "asset": "EQ", "contracts": [] }"#)
        .assert()
        .success()
        .stderr(predicates::str::is_empty());
}

#[test]
fn verbose_flag_enables_debug_timings() {
    let contents = std::fs::read_to_string(fixture("vanilla.json")).unwrap();
    cli()
        .env_remove("RUST_LOG")
        .args(["price", "-i", "-", "-vv"])
        .write_stdin(contents)
        .assert()
        .success()
        .stderr(contains("time taken for price"));
}

#[test]
fn verbose_and_quiet_conflict() {
    cli()
        .args(["price", "-i", "x", "-v", "-q"])
        .assert()
        .code(2)
        .stderr(contains("cannot be used with"));
}

// ---------------------------------------------------------------------- color

#[test]
fn error_output_is_plain_when_piped() {
    // auto mode: no ANSI escapes when stderr is not a terminal
    cli()
        .args(["price", "-i", "nope.json"])
        .assert()
        .code(1)
        .stderr(contains("error:"))
        .stderr(contains("\u{1b}[").not());
}

#[test]
fn color_always_forces_ansi_styling() {
    cli()
        .args(["price", "-i", "nope.json", "--color", "always"])
        .assert()
        .code(1)
        .stderr(contains("\u{1b}[31m")) // red `error:` prefix
        .stderr(contains("failed to read nope.json"));
}

#[test]
fn color_never_strips_ansi_styling() {
    cli()
        .args(["price", "-i", "nope.json", "--color", "never"])
        .assert()
        .code(1)
        .stderr(contains("\u{1b}[").not());
}

// ---------------------------------------------------------------- interactive

#[test]
fn interactive_without_a_terminal_fails_instead_of_hanging() {
    cli()
        .arg("interactive")
        .write_stdin("")
        .assert()
        .code(1)
        .stderr(contains("interactive mode needs a terminal"));
}

// -------------------------------------------------------------- legacy aliases

#[test]
fn deprecated_file_and_dir_subcommands_still_work() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("results.json");
    cli()
        .args(["file", "-i"])
        .arg(fixture("vanilla.json"))
        .arg("-o")
        .arg(&out)
        .assert()
        .success();
    let results: Value = serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    assert!((results[0]["output"]["pv"].as_f64().unwrap() - GOLDEN_PV).abs() < 1e-9);

    let in_dir = dir.path().join("in");
    let out_dir = dir.path().join("out");
    std::fs::create_dir(&in_dir).unwrap();
    std::fs::copy(fixture("vanilla.json"), in_dir.join("v.json")).unwrap();
    cli()
        .args(["dir", "-i"])
        .arg(&in_dir)
        .arg("-o")
        .arg(&out_dir)
        .assert()
        .success();
    assert!(out_dir.join("v.json").exists());
}

// ------------------------------------------------------------- fixed income

#[test]
fn price_ust_bonds_and_bill_reports_full_analytics() {
    let results = stdout_json(cli().args(["price", "-i"]).arg(fixture("ust_bond.json")));
    assert_eq!(results.as_array().unwrap().len(), 4);

    // contract 0: 4.5% 2028 note quoted at a clean 99.50
    let note = &results[0]["output"];
    assert_eq!(note["instrument"], "Bond");
    assert_eq!(note["settlement_date"], "2026-08-06"); // T+1 from Wed Aug 5
    assert_eq!(note["clean_price"].as_f64().unwrap(), 99.50);
    let y = note["yield"].as_f64().unwrap();
    assert!(
        y > 0.045 && y < 0.055,
        "discount to par: yield {y} above coupon"
    );
    let accrued = note["accrued_interest"].as_f64().unwrap();
    // May 15 -> Aug 6 is 83 days of a 184-day period at 2.25 per half
    assert!(
        (accrued - 2.25 * 83.0 / 184.0).abs() < 1e-9,
        "accrued {accrued}"
    );
    assert!(note["modified_duration"].as_f64().unwrap() > 1.5);
    assert!(note["dv01"].as_f64().unwrap() > 0.0);

    // contract 1: 10y from a 4.1% yield -> premium over its 4.25% coupon
    let ten_year = &results[1]["output"];
    let clean = ten_year["clean_price"].as_f64().unwrap();
    assert!(clean > 100.0 && clean < 105.0, "clean {clean}");

    // contract 2: priced off a flat 4% curve
    let on_curve = &results[2]["output"];
    assert!(on_curve["clean_price"].as_f64().unwrap() > 100.0);

    // contract 3: 13-week bill at a 4.8% discount rate
    let bill = &results[3]["output"];
    assert_eq!(bill["instrument"], "Bill");
    assert_eq!(bill["days_to_maturity"].as_i64().unwrap(), 91);
    let expected_price = 100.0 * (1.0 - 0.048 * 91.0 / 360.0);
    assert!((bill["price"].as_f64().unwrap() - expected_price).abs() < 1e-9);
    assert!(bill["bond_equivalent_yield"].as_f64().unwrap() > 0.048);
}

// ---------------------------------------------------------------------- fetch

#[test]
fn fetch_from_file_emits_the_curve_as_published() {
    let doc = stdout_json(
        cli()
            .args(["fetch", "ust", "--date", "2026-08-05", "--from-file"])
            .arg(fixture("ust_par_yields_2026.csv")),
    );
    // provenance and identity live in the metadata block
    let meta = &doc["metadata"];
    assert!(meta["source"].as_str().unwrap().contains("Treasury"));
    assert_eq!(meta["curve_date"], "2026-08-05");
    assert_eq!(meta["unit"], "percent");
    assert!(meta["file"].as_str().unwrap().contains("ust_par_yields"));

    // the points are the feed verbatim: tenor label + percent yield only
    let points = doc["points"].as_array().unwrap();
    assert_eq!(points.len(), 14);
    assert_eq!(points[0]["tenor"], "1 Mo");
    assert_eq!(points[0]["yield"], 3.77);
    assert_eq!(points[13]["tenor"], "30 Yr");
    assert_eq!(points[13]["yield"], 5.17);
    assert_eq!(points[0].as_object().unwrap().len(), 2, "nothing invented");
}

#[test]
fn fetch_without_date_picks_the_latest_published_row() {
    let doc = stdout_json(
        cli()
            .args(["fetch", "ust", "--from-file"])
            .arg(fixture("ust_par_yields_2026.csv")),
    );
    assert_eq!(doc["metadata"]["curve_date"], "2026-08-05");
}

#[test]
fn fetch_missing_date_names_the_nearest_published_day() {
    // Sunday Aug 2: nearest earlier row in the fixture is Friday Jul 31
    cli()
        .args(["fetch", "ust", "--date", "2026-08-02", "--from-file"])
        .arg(fixture("ust_par_yields_2026.csv"))
        .assert()
        .code(1)
        .stdout(predicates::str::is_empty())
        .stderr(contains("2026-07-31"));
}

#[test]
fn fetch_rejects_a_malformed_date() {
    cli()
        .args(["fetch", "ust", "--date", "08/05/2026", "--from-file"])
        .arg(fixture("ust_par_yields_2026.csv"))
        .assert()
        .code(1)
        .stderr(contains("--date must be YYYY-MM-DD"));
}

#[test]
fn fetch_xml_output_follows_the_file_extension() {
    let dir = tempfile::tempdir().unwrap();
    let doc_path = dir.path().join("ust.xml");
    cli()
        .args(["fetch", "ust", "--date", "2026-08-05", "--from-file"])
        .arg(fixture("ust_par_yields_2026.csv"))
        .arg("-o")
        .arg(&doc_path)
        .assert()
        .success();
    let xml = std::fs::read_to_string(&doc_path).unwrap();
    assert!(xml.starts_with("<?xml"), "not XML: {}", &xml[..40]);
    assert!(xml.contains("<curve>"));
    assert!(xml.contains("<tenor>1 Mo</tenor>"));
    assert!(xml.contains("<yield>3.77</yield>"));
    assert!(xml.contains("<curve_date>2026-08-05</curve_date>"));
}

#[test]
fn fetch_sofr_emits_the_observation_as_published() {
    let doc = stdout_json(
        cli()
            .args(["fetch", "sofr", "--from-file"])
            .arg(fixture("nyfed_sofr.json")),
    );
    let meta = &doc["metadata"];
    assert_eq!(meta["rate_type"], "SOFR");
    assert_eq!(meta["effective_date"], "2026-08-05", "latest of the two");
    assert_eq!(meta["unit"], "percent");
    assert!(meta["source"].as_str().unwrap().contains("New York"));
    assert!(meta["file"].as_str().unwrap().contains("nyfed_sofr"));
    // the record is the feed verbatim
    assert_eq!(doc["rate"]["percentRate"], 3.64);
    assert_eq!(doc["rate"]["volumeInBillions"], 2989);
    assert_eq!(doc["rate"]["type"], "SOFR");
}

#[test]
fn fetch_effr_keeps_the_target_range_and_honors_date() {
    let doc = stdout_json(
        cli()
            .args(["fetch", "effr", "--date", "2026-08-04", "--from-file"])
            .arg(fixture("nyfed_effr.json")),
    );
    assert_eq!(doc["metadata"]["rate_type"], "EFFR");
    assert_eq!(doc["metadata"]["effective_date"], "2026-08-04");
    assert_eq!(doc["rate"]["percentRate"], 3.63);
    assert_eq!(doc["rate"]["targetRateFrom"], 3.50);
    assert_eq!(doc["rate"]["targetRateTo"], 3.75);
}

#[test]
fn fetch_rate_missing_date_names_the_nearest_published_day() {
    // Sunday Aug 9: nearest earlier observation is Wednesday Aug 5
    cli()
        .args(["fetch", "sofr", "--date", "2026-08-09", "--from-file"])
        .arg(fixture("nyfed_sofr.json"))
        .assert()
        .code(1)
        .stdout(predicates::str::is_empty())
        .stderr(contains("2026-08-05"));
}

#[test]
fn fetch_rate_xml_output_works() {
    cli()
        .args(["fetch", "effr", "--format", "xml", "--from-file"])
        .arg(fixture("nyfed_effr.json"))
        .assert()
        .success()
        .stdout(contains("<?xml"))
        .stdout(contains("<reference_rate>"))
        .stdout(contains("<percentRate>3.63</percentRate>"));
}

#[test]
fn fetch_format_flag_forces_xml_on_stdout() {
    cli()
        .args(["fetch", "ust", "--format", "xml", "--from-file"])
        .arg(fixture("ust_par_yields_2026.csv"))
        .assert()
        .success()
        .stdout(contains("<?xml"))
        .stdout(contains("<curve>"));
}

#[test]
fn build_bootstraps_a_treasury_curve_from_quotes() {
    let dir = tempfile::tempdir().unwrap();
    cli()
        .args(["build", "-i"])
        .arg(fixture("build_ust_curve.json"))
        .arg("-o")
        .arg(dir.path())
        .assert()
        .success();
    let csv = std::fs::read_to_string(dir.path().join("term_structure").join("term_structure.csv"))
        .expect("curve csv is written");
    let mut lines = csv.lines();
    assert_eq!(
        lines.next().unwrap(),
        "date,discount_factor,zero_rate_continuous"
    );
    let rows: Vec<(String, f64, f64)> = lines
        .map(|line| {
            let mut parts = line.split(',');
            (
                parts.next().unwrap().to_string(),
                parts.next().unwrap().parse().unwrap(),
                parts.next().unwrap().parse().unwrap(),
            )
        })
        .collect();
    // five pillars: two bills, three notes, in maturity order
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0].0, "2026-11-05");
    assert_eq!(rows[4].0, "2036-05-15");
    // discount factors strictly decreasing, zeros in a sane Treasury range
    assert!(rows.windows(2).all(|w| w[1].1 < w[0].1), "{rows:?}");
    assert!(
        rows.iter().all(|r| r.2 > 0.03 && r.2 < 0.06),
        "zeros out of range: {rows:?}"
    );
}
