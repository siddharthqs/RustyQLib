use std::process::ExitCode;

use clap::Parser;
use rustyqlib::utils::build_cli::{self, Cli, ColorWhen, Commands};
use rustyqlib::utils::style;

/// Map `-v`/`-q` counts to a log filter. When either flag is given it
/// wins; otherwise `RUST_LOG` applies with a default of `warn`.
/// Diagnostics go to stderr; stdout stays reserved for pricing output.
fn init_logger(verbose: u8, quiet: u8, color: ColorWhen) {
    let level = match (quiet, verbose) {
        (q, _) if q >= 2 => "off",
        (1, _) => "error",
        (_, 0) => "warn",
        (_, 1) => "info",
        (_, 2) => "debug",
        _ => "trace",
    };
    let write_style = match color {
        ColorWhen::Auto => env_logger::WriteStyle::Auto,
        ColorWhen::Always => env_logger::WriteStyle::Always,
        ColorWhen::Never => env_logger::WriteStyle::Never,
    };
    if verbose > 0 || quiet > 0 {
        env_logger::Builder::new()
            .parse_filters(level)
            .write_style(write_style)
            .init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(level))
            .write_style(write_style)
            .init();
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    // one switch for every anstream-printed message (styles are stripped
    // automatically for non-terminals and NO_COLOR in auto mode)
    anstream::ColorChoice::from(cli.color).write_global();
    init_logger(cli.verbose, cli.quiet, cli.color);

    let result = match &cli.command {
        Commands::Price(args) => build_cli::handle_price(args),
        Commands::Build(args) => build_cli::handle_build(args),
        Commands::Stress(args) => build_cli::handle_stress(args),
        Commands::Risk(args) => build_cli::handle_risk(args),
        Commands::ImpliedVol(args) => build_cli::handle_implied_vol(args),
        Commands::Interactive => build_cli::handle_interactive(),
        Commands::Completions { shell } => build_cli::handle_completions(*shell),
        Commands::File(args) => build_cli::handle_file(args),
        Commands::Dir(args) => build_cli::handle_dir(args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            let error = style::ERROR;
            anstream::eprintln!("{error}error:{error:#} {e:#}");
            ExitCode::FAILURE
        }
    }
}
