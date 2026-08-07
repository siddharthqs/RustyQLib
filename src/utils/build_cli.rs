use crate::core::serialization::{self, Format};
use crate::core::trade::PutOrCall;
use crate::data::nyfed;
use crate::data::treasury;
use crate::equity::blackscholes::implied_vol_from_price;
use crate::risk::{delta_gamma_var, full_revaluation_var, stress_mtm, RiskConfig, StressConfig};
use crate::utils::interactive;
use crate::utils::parse_contracts;
use anyhow::{bail, Context, Result};
use chrono::{Datelike, Local, NaiveDate};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum, ValueHint};
use std::fs;
use std::io;
use std::io::Read;
use std::path::Path;
use std::time::Instant;

/// Palette for `--help` output: headers and usage in bold yellow,
/// command/flag literals in green, value placeholders in cyan. Clap
/// strips the colors automatically when stdout is not a terminal or
/// `NO_COLOR` is set.
const HELP_STYLES: clap::builder::styling::Styles = {
    use clap::builder::styling::{AnsiColor, Effects, Styles};
    Styles::styled()
        .header(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
        .usage(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
        .literal(AnsiColor::Green.on_default())
        .placeholder(AnsiColor::Cyan.on_default())
        .error(AnsiColor::Red.on_default().effects(Effects::BOLD))
        .valid(AnsiColor::Green.on_default())
        .invalid(AnsiColor::Red.on_default())
};

/// Pricing and risk management of financial derivatives.
#[derive(Parser)]
#[command(
    name = "rustyqlib",
    version,
    author = "Siddharth Singh <siddharth_qs@outlook.com>",
    about = "Pricing and risk management of financial derivatives",
    subcommand_required = true,
    arg_required_else_help = true,
    styles = HELP_STYLES
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,

    /// More diagnostics on stderr (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Fewer diagnostics on stderr (-q errors only, -qq silent)
    #[arg(short, long, action = clap::ArgAction::Count, global = true, conflicts_with = "verbose")]
    pub quiet: u8,

    /// When to color output ('auto' detects a terminal and honors NO_COLOR)
    #[arg(long, value_enum, default_value_t = ColorWhen::Auto, global = true)]
    pub color: ColorWhen,
}

/// Color policy for the CLI's own messages and log diagnostics.
#[derive(Clone, Copy, ValueEnum)]
pub enum ColorWhen {
    Auto,
    Always,
    Never,
}

impl From<ColorWhen> for anstream::ColorChoice {
    fn from(when: ColorWhen) -> Self {
        match when {
            ColorWhen::Auto => anstream::ColorChoice::Auto,
            ColorWhen::Always => anstream::ColorChoice::Always,
            ColorWhen::Never => anstream::ColorChoice::Never,
        }
    }
}

#[derive(Subcommand)]
pub enum Commands {
    /// Price contracts from a file, a directory, or stdin
    Price(PriceArgs),
    /// Building the curve / Vol surface
    Build(BuildArgs),
    /// Fetch a free official end-of-day curve (as published, with metadata)
    Fetch(FetchArgs),
    /// Stress MtM: revalue an options book under TOML shock scenarios
    Stress(StressArgs),
    /// VaR / Expected Shortfall for an options book by scenario simulation
    Risk(RiskArgs),
    /// Implied Black-Scholes volatility from a European vanilla price
    ImpliedVol(ImpliedVolArgs),
    /// Interactive mode
    Interactive,
    /// Generate shell completions to stdout
    Completions {
        /// The shell to generate completions for
        shell: clap_complete::Shell,
    },
    /// Pricing a single contract (deprecated: use `price`)
    #[command(hide = true)]
    File(FileArgs),
    /// Pricing all contracts in a directory (deprecated: use `price`)
    #[command(hide = true)]
    Dir(FileArgs),
}

/// Output document format.
#[derive(Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Json,
    Xml,
}

impl From<OutputFormat> for Format {
    fn from(format: OutputFormat) -> Format {
        match format {
            OutputFormat::Json => Format::Json,
            OutputFormat::Xml => Format::Xml,
        }
    }
}

/// Option side for `implied-vol`.
#[derive(Clone, Copy, ValueEnum)]
pub enum SideArg {
    #[value(name = "C", alias = "call")]
    Call,
    #[value(name = "P", alias = "put")]
    Put,
}

impl From<SideArg> for PutOrCall {
    fn from(side: SideArg) -> PutOrCall {
        match side {
            SideArg::Call => PutOrCall::Call,
            SideArg::Put => PutOrCall::Put,
        }
    }
}

/// VaR estimator selection for `risk`.
#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RiskMethod {
    DeltaGamma,
    Full,
    Both,
}

#[derive(Args)]
pub struct PriceArgs {
    /// Contracts file or directory ('-' reads from stdin)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::AnyPath)]
    pub input: String,
    /// Output file or directory (default: stdout for file/stdin input)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::AnyPath)]
    pub output: Option<String>,
    /// Output format (default: the output file extension, else the input format)
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,
}

#[derive(Args)]
pub struct BuildArgs {
    /// Input financial contracts to use in construction ('-' reads from stdin)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub input: String,
    /// Output directory
    #[arg(short, long, value_name = "DIR", value_hint = ValueHint::DirPath)]
    pub output: String,
}

/// Free official end-of-day data sources for `fetch`.
#[derive(Clone, Copy, ValueEnum)]
pub enum FetchSource {
    /// US Treasury daily par yield curve (home.treasury.gov)
    #[value(name = "ust", alias = "ust-par-yields")]
    UstParYields,
    /// Secured Overnight Financing Rate (markets.newyorkfed.org)
    #[value(name = "sofr")]
    Sofr,
    /// Effective Federal Funds Rate (markets.newyorkfed.org)
    #[value(name = "effr")]
    Effr,
}

#[derive(Args)]
pub struct FetchArgs {
    /// Data source
    #[arg(value_enum)]
    pub source: FetchSource,
    /// Curve date, YYYY-MM-DD (default: the latest published business day)
    #[arg(long, value_name = "DATE")]
    pub date: Option<String>,
    /// Parse a previously downloaded CSV instead of hitting the network
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub from_file: Option<String>,
    /// Output file (default: stdout)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub output: Option<String>,
    /// Output format (default: the output file extension, else JSON)
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,
}

#[derive(Args)]
pub struct StressArgs {
    /// Portfolio of option contracts, one underlying ('-' reads from stdin);
    /// signed quantity comes from each contract's `long_short` (default 1)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub input: String,
    /// TOML stress-scenario configuration
    #[arg(short, long, value_name = "TOML", value_hint = ValueHint::FilePath)]
    pub config: String,
    /// Output JSON file (default: stdout)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub output: Option<String>,
}

#[derive(Args)]
pub struct RiskArgs {
    /// Portfolio of option contracts, one underlying ('-' reads from stdin);
    /// signed quantity comes from each contract's `long_short` (default 1)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub input: String,
    /// Output JSON file (default: stdout)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub output: Option<String>,
    /// Delta-gamma Taylor VaR, full revaluation, or both
    #[arg(long, value_enum, default_value_t = RiskMethod::Both)]
    pub method: RiskMethod,
    /// One-sided confidence level in (0, 1)
    #[arg(long, default_value_t = 0.99)]
    pub confidence: f64,
    /// Risk horizon in trading days (252 per year)
    #[arg(long, default_value_t = 1.0)]
    pub horizon_days: f64,
    /// Number of simulated scenarios
    #[arg(long, default_value_t = 20_000)]
    pub scenarios: usize,
    /// Annualized volatility of the underlying's return
    #[arg(long, default_value_t = 0.2)]
    pub spot_vol: f64,
    /// Annualized absolute volatility of the implied-vol move
    #[arg(long, default_value_t = 0.0)]
    pub vol_of_vol: f64,
    /// Spot-vol move correlation in [-1, 1]
    #[arg(long, default_value_t = -0.5, allow_negative_numbers = true)]
    pub corr: f64,
    /// Random seed (runs are deterministic per seed)
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

#[derive(Args)]
pub struct ImpliedVolArgs {
    /// Current price of the underlying
    #[arg(long)]
    pub spot: f64,
    /// Strike price
    #[arg(long)]
    pub strike: f64,
    /// Observed option price to invert
    #[arg(long)]
    pub price: f64,
    /// Time to expiry in years (e.g. 0.5) or a YYYY-MM-DD date
    #[arg(long, value_name = "T|DATE")]
    pub maturity: String,
    /// Option side
    #[arg(short = 'p', long, value_enum)]
    pub put_or_call: SideArg,
    /// Continuously compounded risk-free rate
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate: f64,
    /// Continuous dividend yield
    #[arg(long, default_value_t = 0.0)]
    pub dividend: f64,
    /// Output JSON file (default: stdout)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub output: Option<String>,
}

/// The deprecated `file` / `dir` argument shape (input and output both
/// required).
#[derive(Args)]
pub struct FileArgs {
    /// Input contracts file or directory
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::AnyPath)]
    pub input: String,
    /// Output file or directory
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::AnyPath)]
    pub output: String,
}

/// The full clap command, for completion generation and compatibility.
pub fn build_cli() -> clap::Command {
    Cli::command()
}

/// Read the whole input: stdin when `path` is `-`, the file otherwise.
fn read_input(path: &str) -> Result<String> {
    if path == "-" {
        let mut contents = String::new();
        io::stdin()
            .read_to_string(&mut contents)
            .context("failed to read stdin")?;
        Ok(contents)
    } else {
        fs::read_to_string(path).with_context(|| format!("failed to read {path}"))
    }
}

/// How the input is named in messages: "stdin" for `-`, the path otherwise.
fn input_label(path: &str) -> &str {
    if path == "-" {
        "stdin"
    } else {
        path
    }
}

/// Write to the file when given (and not `-`), stdout otherwise.
fn write_output(path: Option<&String>, content: &str) -> Result<()> {
    match path.map(String::as_str) {
        Some("-") | None => {
            println!("{content}");
            Ok(())
        }
        Some(p) => fs::write(p, content).with_context(|| format!("failed to write {p}")),
    }
}

/// Handle the "price" subcommand: a single file, stdin, or a directory.
pub fn handle_price(args: &PriceArgs) -> Result<()> {
    let input = &args.input;
    let output = args.output.as_ref();
    let format = args.format.map(Format::from);

    let input_path = Path::new(input);
    if input != "-" && input_path.is_dir() {
        let output_dir =
            output.context("--output <DIR> is required when the input is a directory")?;
        return measure_time("price (directory)", || {
            price_directory(input_path, Path::new(output_dir), format)
        });
    }

    measure_time("price", || {
        let contents = read_input(input)?;
        // output format precedence: --format, output extension, input format
        let out_format = format.or_else(|| output.and_then(Format::from_path));
        let rendered = parse_contracts::price_document(&contents, out_format)
            .with_context(|| format!("failed to price {}", input_label(input)))?;
        match rendered {
            Some(output_str) => write_output(output, &output_str),
            None => {
                log::warn!(
                    "no contracts found in {}; nothing written",
                    input_label(input)
                );
                Ok(())
            }
        }
    })
}

/// Price every contract document in `input_path` into `output_path`.
fn price_directory(input_path: &Path, output_path: &Path, format: Option<Format>) -> Result<()> {
    fs::create_dir_all(output_path).with_context(|| {
        format!(
            "failed to create output directory {}",
            output_path.display()
        )
    })?;
    let files = fs::read_dir(input_path)
        .with_context(|| format!("failed to read input directory {}", input_path.display()))?;

    for file_result in files {
        let dir_entry = file_result
            .with_context(|| format!("failed to read entry in {}", input_path.display()))?;
        let path = dir_entry.path();

        let is_contract_file = path.is_file()
            && matches!(
                path.extension()
                    .and_then(|s| s.to_str())
                    .map(|e| e.to_lowercase())
                    .as_deref(),
                Some("json") | Some("xml")
            );
        if is_contract_file {
            // Construct the corresponding output file path
            let file_name = path
                .file_name()
                .with_context(|| format!("no file name in {}", path.display()))?;
            let output_file_path = output_path.join(file_name);

            parse_contracts::parse_contract(&path, &output_file_path, format)?;
            log::debug!("priced contracts {:?} -> {:?}", path, output_file_path);
        }
    }
    Ok(())
}

/// Handle the "build" subcommand.
pub fn handle_build(args: &BuildArgs) -> Result<()> {
    // We measure the time of the operation
    measure_time("build_curve", || {
        let contents = read_input(&args.input)?;
        parse_contracts::build_curve(&contents, input_label(&args.input), Path::new(&args.output))
    })
}

/// Handle the "fetch" subcommand: download (or read) a free official
/// end-of-day curve and emit it exactly as published — tenor labels and
/// yields untouched — under a `metadata` block recording what the curve
/// is and where it came from. The network is confined to this command,
/// so anything downstream is reproducible from the emitted file alone.
pub fn handle_fetch(args: &FetchArgs) -> Result<()> {
    let date = args
        .date
        .as_deref()
        .map(|s| {
            NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .with_context(|| format!("--date must be YYYY-MM-DD, got '{s}'"))
        })
        .transpose()?;
    measure_time("fetch", || match args.source {
        FetchSource::UstParYields => fetch_ust_par_yields(args, date),
        FetchSource::Sofr => fetch_nyfed_rate(args, date, nyfed::ReferenceRate::Sofr),
        FetchSource::Effr => fetch_nyfed_rate(args, date, nyfed::ReferenceRate::Effr),
    })
}

fn fetch_ust_par_yields(args: &FetchArgs, date: Option<NaiveDate>) -> Result<()> {
    // provenance of the raw CSV, merged into the document metadata
    let (rows, origin) = match &args.from_file {
        Some(path) => {
            let text = read_input(path)?;
            let rows = treasury::parse_csv(&text)
                .with_context(|| format!("failed to parse {}", input_label(path)))?;
            (rows, serde_json::json!({ "file": input_label(path) }))
        }
        None => {
            let mut year = date.map_or_else(|| Local::now().date_naive().year(), |d| d.year());
            let mut rows = treasury::parse_csv(&treasury::fetch_year_csv(year)?)?;
            if rows.is_empty() && date.is_none() {
                // early January: nothing published for the year yet
                year -= 1;
                rows = treasury::parse_csv(&treasury::fetch_year_csv(year)?)?;
            }
            let origin = serde_json::json!({
                "url": treasury::csv_url(year),
                "fetched_at": Local::now().to_rfc3339(),
            });
            (rows, origin)
        }
    };

    let row = treasury::select_row(&rows, date)?;
    log::info!(
        "US Treasury par yields for {}: {} curve points",
        row.date,
        row.points.len()
    );
    emit_document(args, treasury::to_document(row), origin, "curve")
}

fn fetch_nyfed_rate(
    args: &FetchArgs,
    date: Option<NaiveDate>,
    rate: nyfed::ReferenceRate,
) -> Result<()> {
    let (observations, origin) = match &args.from_file {
        Some(path) => {
            let text = read_input(path)?;
            let observations = nyfed::parse_response(&text)
                .with_context(|| format!("failed to parse {}", input_label(path)))?;
            (
                observations,
                serde_json::json!({ "file": input_label(path) }),
            )
        }
        None => {
            let (text, url) = match date {
                // fetch a window back from the requested date, so a
                // holiday can name the nearest earlier published day
                Some(d) => {
                    let start = d - chrono::Days::new(10);
                    let url = nyfed::search_url(rate, start, d);
                    (nyfed::fetch_range(rate, start, d)?, url)
                }
                None => (nyfed::fetch_last(rate, 5)?, nyfed::last_url(rate, 5)),
            };
            let origin = serde_json::json!({
                "url": url,
                "fetched_at": Local::now().to_rfc3339(),
            });
            (nyfed::parse_response(&text)?, origin)
        }
    };
    let observation = nyfed::select_observation(&observations, date)?;
    log::info!(
        "{} for {}: {}%",
        rate.name(),
        observation.date,
        observation.record["percentRate"]
    );
    emit_document(
        args,
        nyfed::to_document(observation, rate),
        origin,
        "reference_rate",
    )
}

/// Merge fetch provenance into the document's `metadata` block and write
/// it in the requested format (`--format`, then the output file
/// extension, then JSON).
fn emit_document(
    args: &FetchArgs,
    mut document: serde_json::Value,
    origin: serde_json::Value,
    root: &str,
) -> Result<()> {
    if let (Some(serde_json::Value::Object(meta)), serde_json::Value::Object(origin)) =
        (document.get_mut("metadata"), origin)
    {
        meta.extend(origin);
    }
    let format = args
        .format
        .map(Format::from)
        .or_else(|| args.output.as_ref().and_then(Format::from_path))
        .unwrap_or(Format::Json);
    write_output(
        args.output.as_ref(),
        &serialization::render_value(&document, format, root),
    )
}

/// Handle the "stress" subcommand.
pub fn handle_stress(args: &StressArgs) -> Result<()> {
    if args.input == "-" && args.config == "-" {
        bail!("only one of --input and --config can read from stdin");
    }

    measure_time("stress_mtm", || {
        let book =
            parse_contracts::build_portfolio(&read_input(&args.input)?).with_context(|| {
                format!("failed to load portfolio from {}", input_label(&args.input))
            })?;
        let config =
            StressConfig::from_toml_str(&read_input(&args.config)?).with_context(|| {
                format!(
                    "failed to load stress config from {}",
                    input_label(&args.config)
                )
            })?;
        let results = stress_mtm(&book, &config).context("stress run failed")?;
        let rendered =
            serde_json::to_string_pretty(&results).context("failed to serialize stress results")?;
        write_output(args.output.as_ref(), &rendered)
    })
}

/// Handle the "risk" subcommand.
pub fn handle_risk(args: &RiskArgs) -> Result<()> {
    if !(0.0..1.0).contains(&args.confidence) || args.confidence == 0.0 {
        bail!(
            "--confidence must be strictly between 0 and 1, got {}",
            args.confidence
        );
    }
    if args.horizon_days <= 0.0 {
        bail!("--horizon-days must be positive, got {}", args.horizon_days);
    }
    if !(-1.0..=1.0).contains(&args.corr) {
        bail!("--corr must be in [-1, 1], got {}", args.corr);
    }
    let cfg = RiskConfig {
        horizon: args.horizon_days / 252.0,
        spot_vol: args.spot_vol,
        vol_of_vol: args.vol_of_vol,
        spot_vol_corr: args.corr,
        scenarios: args.scenarios,
        confidence: args.confidence,
        seed: args.seed,
    };

    measure_time("portfolio_risk", || {
        let book =
            parse_contracts::build_portfolio(&read_input(&args.input)?).with_context(|| {
                format!("failed to load portfolio from {}", input_label(&args.input))
            })?;
        let spot = book.positions[0].option.market.spot.value();

        let mut report = serde_json::Map::new();
        report.insert("spot".into(), serde_json::json!(spot));
        report.insert("config".into(), serde_json::to_value(cfg)?);
        if matches!(args.method, RiskMethod::DeltaGamma | RiskMethod::Both) {
            let dg = delta_gamma_var(&book, spot, &cfg);
            report.insert("delta_gamma".into(), serde_json::to_value(&dg)?);
        }
        if matches!(args.method, RiskMethod::Full | RiskMethod::Both) {
            let full = full_revaluation_var(&book, spot, &cfg);
            report.insert("full_revaluation".into(), serde_json::to_value(&full)?);
        }
        let rendered = serde_json::to_string_pretty(&serde_json::Value::Object(report))
            .context("failed to serialize risk report")?;
        write_output(args.output.as_ref(), &rendered)
    })
}

/// Handle the "implied-vol" subcommand.
pub fn handle_implied_vol(args: &ImpliedVolArgs) -> Result<()> {
    let put_or_call = PutOrCall::from(args.put_or_call);
    // years as a number, or a date measured Act/365 from today
    let t = match args.maturity.parse::<f64>() {
        Ok(years) => years,
        Err(_) => {
            let date =
                NaiveDate::parse_from_str(&args.maturity, "%Y-%m-%d").with_context(|| {
                    format!(
                        "--maturity must be a year fraction or YYYY-MM-DD date, got '{}'",
                        args.maturity
                    )
                })?;
            (date - Local::now().date_naive()).num_days() as f64 / 365.0
        }
    };
    if t <= 0.0 {
        bail!("maturity must be in the future (t = {t:.4} years)");
    }

    let vol = implied_vol_from_price(
        args.spot,
        args.strike,
        args.rate,
        args.dividend,
        t,
        args.price,
        put_or_call,
    )
    .context("implied vol solve failed")?;
    let rendered = serde_json::to_string_pretty(&serde_json::json!({
        "spot": args.spot,
        "strike": args.strike,
        "rate": args.rate,
        "dividend": args.dividend,
        "t": t,
        "price": args.price,
        "put_or_call": format!("{put_or_call:?}"),
        "implied_vol": vol,
    }))?;
    write_output(args.output.as_ref(), &rendered)
}

/// Handle the "completions" subcommand.
pub fn handle_completions(shell: clap_complete::Shell) -> Result<()> {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, "rustyqlib", &mut io::stdout());
    Ok(())
}

/// Handle the deprecated "file" subcommand.
pub fn handle_file(args: &FileArgs) -> Result<()> {
    measure_time("parse_contract (single file)", || {
        parse_contracts::parse_contract(Path::new(&args.input), Path::new(&args.output), None)
    })
}

/// Handle the deprecated "dir" subcommand.
pub fn handle_dir(args: &FileArgs) -> Result<()> {
    measure_time("parse_contract (directory)", || {
        price_directory(Path::new(&args.input), Path::new(&args.output), None)
    })
}

/// Handle the "interactive" subcommand: a menu-driven session. Esc backs
/// out of the current wizard, Ctrl-C (or "Exit") ends the session; a bad
/// pricing input reports its error and returns to the menu.
pub fn handle_interactive() -> Result<()> {
    use std::io::IsTerminal;
    if !io::stdin().is_terminal() {
        bail!("interactive mode needs a terminal; pipe contracts into `price -i -` instead");
    }
    println!("Welcome to the RustyQLib pricing CLI (Esc cancels a wizard, Ctrl-C exits)");
    loop {
        let choice = match inquire::Select::new(
            "What would you like to do?",
            vec![
                "Price an option",
                "Price a bond",
                "Implied volatility",
                "Exit",
            ],
        )
        .prompt()
        {
            Ok(choice) => choice,
            Err(
                inquire::InquireError::OperationCanceled
                | inquire::InquireError::OperationInterrupted,
            ) => break,
            Err(e) => return Err(e.into()),
        };

        let result = match choice {
            "Price an option" => interactive::price_option_wizard(),
            "Price a bond" => interactive::price_bond_wizard(),
            "Implied volatility" => interactive::implied_vol_wizard(),
            _ => break,
        };
        match result {
            Ok(()) => {}
            Err(e) if interactive::is_cancelled(&e) => println!("(cancelled)"),
            Err(e) => {
                use crate::utils::style::ERROR;
                anstream::eprintln!("{ERROR}error:{ERROR:#} {e:#}");
            }
        }
    }
    println!("Goodbye!");
    Ok(())
}

/// Helper function to measure the time taken by a closure.
fn measure_time<T, F: FnOnce() -> T>(label: &str, f: F) -> T {
    let start_time = Instant::now();
    let result = f();
    let elapsed_time = start_time.elapsed();
    log::debug!("time taken for {}: {:?}", label, elapsed_time);
    result
}
