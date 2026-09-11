use crate::core::serialization::{self, Format};
use crate::core::trade::PutOrCall;
use crate::data::cboe;
use crate::data::dtcc;
use crate::data::nyfed;
use crate::data::ppd;
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
    /// Input financial contracts or an option chain document ('-' reads
    /// from stdin)
    #[arg(short, long, value_name = "FILE", value_hint = ValueHint::FilePath)]
    pub input: String,
    /// Output directory
    #[arg(short, long, value_name = "DIR", value_hint = ValueHint::DirPath)]
    pub output: String,
    /// Flat continuously compounded rate used to discount an option
    /// chain (parity forwards); ignored for other inputs
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pub rate: f64,
    /// Discount an option chain off a fetched US Treasury par-yield
    /// curve document (`fetch ust`) instead of --rate
    #[arg(long, value_name = "FILE", value_hint = ValueHint::FilePath, conflicts_with = "rate")]
    pub curve: Option<String>,
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
    /// DTCC GCF Repo Index: overnight GC repo rate and par, Treasury and MBS (dtcc.com)
    #[value(name = "gcf", alias = "gcf-repo")]
    Gcf,
    /// Listed option chain, 15-minute delayed (cdn.cboe.com); needs --symbol
    #[value(name = "chain")]
    Chain,
    /// Credit derivative prints: index and single-name CDS from DTCC's public
    /// price dissemination (pddata.dtcc.com); --symbol filters by index or name
    #[value(name = "cds")]
    Cds,
}

#[derive(Args)]
pub struct FetchArgs {
    /// Data source
    #[arg(value_enum)]
    pub source: FetchSource,
    /// Date, YYYY-MM-DD (default: the latest published business day; for
    /// `cds`, the report day)
    #[arg(long, value_name = "DATE")]
    pub date: Option<String>,
    /// Underlying ticker for `chain` (e.g. AAPL, _SPX); for `cds`, an index,
    /// entity name or reference-id filter (e.g. CDX.NA.IG)
    #[arg(long, value_name = "SYMBOL")]
    pub symbol: Option<String>,
    /// For `chain`: emit the normalized OptionChain document instead of
    /// the verbatim feed response
    #[arg(long)]
    pub normalize: bool,
    /// Parse a previously downloaded file instead of hitting the network
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
    /// One-sided confidence level in (0.5, 1)
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

    let mut priced = 0usize;
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
            priced += 1;
            log::debug!("priced contracts {:?} -> {:?}", path, output_file_path);
        }
    }
    if priced == 0 {
        // exiting 0 in silence looks like success; say the directory held
        // nothing this command knows how to price
        log::warn!(
            "no .json or .xml contract files in {}; nothing written",
            input_path.display()
        );
    }
    Ok(())
}

/// Handle the "build" subcommand.
pub fn handle_build(args: &BuildArgs) -> Result<()> {
    if args.input == "-" && args.curve.as_deref() == Some("-") {
        bail!("only one of --input and --curve can read from stdin");
    }
    // We measure the time of the operation
    measure_time("build_curve", || {
        let contents = read_input(&args.input)?;
        let discount = match &args.curve {
            Some(path) => {
                let text = read_input(path)?;
                let value = serialization::parse_value(&text, Format::detect(&text)).with_context(
                    || format!("failed to parse the curve document {}", input_label(path)),
                )?;
                let (curve, description) =
                    treasury::bootstrap_from_document(&value).with_context(|| {
                        format!(
                            "failed to bootstrap a Treasury curve from {}",
                            input_label(path)
                        )
                    })?;
                log::info!(
                    "discounting off the bootstrapped Treasury curve of {}",
                    curve.reference_date()
                );
                parse_contracts::ChainDiscount::Curve { curve, description }
            }
            None => parse_contracts::ChainDiscount::Flat(args.rate),
        };
        parse_contracts::build_curve(
            &contents,
            input_label(&args.input),
            Path::new(&args.output),
            discount,
        )
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
    // --symbol and --normalize only mean anything for the sources that
    // read them; silently ignoring them would hand back a document the
    // user did not ask for (same contract as the chain source's own
    // --date check)
    if args.symbol.is_some() && !matches!(args.source, FetchSource::Chain | FetchSource::Cds) {
        bail!("--symbol only applies to the chain and cds sources; omit it");
    }
    if args.normalize && !matches!(args.source, FetchSource::Chain) {
        bail!("--normalize only applies to the chain source; omit it");
    }
    measure_time("fetch", || match args.source {
        FetchSource::UstParYields => fetch_ust_par_yields(args, date),
        FetchSource::Sofr => fetch_nyfed_rate(args, date, nyfed::ReferenceRate::Sofr),
        FetchSource::Effr => fetch_nyfed_rate(args, date, nyfed::ReferenceRate::Effr),
        FetchSource::Gcf => fetch_gcf_repo_index(args, date),
        FetchSource::Chain => fetch_cboe_chain(args, date),
        FetchSource::Cds => fetch_cds_prints(args, date),
    })
}

fn fetch_cboe_chain(args: &FetchArgs, date: Option<NaiveDate>) -> Result<()> {
    if date.is_some() {
        bail!("the chain source is a live snapshot with no history; omit --date");
    }
    let symbol = args
        .symbol
        .as_deref()
        .context("the chain source needs --symbol (e.g. --symbol AAPL)")?;
    let (text, origin) = match &args.from_file {
        Some(path) => (
            read_input(path)?,
            serde_json::json!({ "file": input_label(path) }),
        ),
        None => (
            cboe::fetch(symbol)?,
            serde_json::json!({
                "url": cboe::url(symbol),
                "fetched_at": Local::now().to_rfc3339(),
            }),
        ),
    };
    if args.normalize {
        let chain = cboe::to_chain(&text)
            .with_context(|| format!("failed to normalize the chain for {symbol}"))?;
        log::info!(
            "option chain for {}: {} quotes as of {}",
            chain.symbol,
            chain.quotes.len(),
            chain.as_of
        );
        // the chain's own `metadata` block is the same shape every other
        // fetched document carries, so provenance merging and format
        // resolution go through the one emitter
        let value = serde_json::to_value(&chain).context("failed to serialize the chain")?;
        emit_document(args, value, origin, "option_chain")
    } else {
        let document = cboe::to_document(&text, symbol)
            .with_context(|| format!("failed to read the chain response for {symbol}"))?;
        emit_document(args, document, origin, "option_chain")
    }
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

fn fetch_gcf_repo_index(args: &FetchArgs, date: Option<NaiveDate>) -> Result<()> {
    let (rows, origin) = match &args.from_file {
        Some(path) => {
            let text = read_input(path)?;
            let rows = dtcc::parse_csv(&text)
                .with_context(|| format!("failed to parse {}", input_label(path)))?;
            (rows, serde_json::json!({ "file": input_label(path) }))
        }
        None => {
            // the feed is one rolling one-year file; any --date inside
            // that window is served by the same download
            let (text, url) = dtcc::fetch()?;
            let origin = serde_json::json!({
                "url": url,
                "fetched_at": Local::now().to_rfc3339(),
            });
            (dtcc::parse_csv(&text)?, origin)
        }
    };
    let row = dtcc::select_row(&rows, date)?;
    let rate = |c: Option<dtcc::GcfComponent>| {
        c.map_or("n/a".to_string(), |c| {
            format!("{}%", c.weighted_average_rate)
        })
    };
    log::info!(
        "DTCC GCF Repo Index for {}: Treasury {}, MBS {}",
        row.date,
        rate(row.treasury),
        rate(row.mbs)
    );
    emit_document(args, dtcc::to_document(row), origin, "gcf_repo_index")
}

/// How many days back from the requested (or current) day `fetch cds`
/// looks for a complete daily report: the current day's file appears
/// only after the close, and weekends and holidays have none.
const PPD_LOOKBACK_DAYS: u64 = 7;

fn fetch_cds_prints(args: &FetchArgs, date: Option<NaiveDate>) -> Result<()> {
    let (mut prints, report_date, origin) = match &args.from_file {
        Some(path) => {
            // a downloaded zip or the CSV inside it; the file name, when
            // it is DTCC's, says which repository and day it is
            let is_zip = path != "-"
                && Path::new(path)
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("zip"));
            let text = if is_zip {
                let bytes =
                    std::fs::read(path).with_context(|| format!("failed to read {path}"))?;
                ppd::unzip_csv(&bytes).with_context(|| format!("failed to open {path}"))?
            } else {
                read_input(path)?
            };
            let prints = ppd::parse_csv(&text, ppd::Jurisdiction::from_file_name(path))
                .with_context(|| format!("failed to parse {}", input_label(path)))?;
            let report_date = date
                .or_else(|| ppd::date_from_file_name(path))
                .or_else(|| {
                    prints
                        .iter()
                        .filter_map(|p| p.event_timestamp.get(..10)?.parse::<NaiveDate>().ok())
                        .max()
                })
                .context("cannot tell which day this report covers; pass --date")?;
            (
                prints,
                report_date,
                serde_json::json!({ "file": input_label(path) }),
            )
        }
        None => {
            let start = date.unwrap_or_else(|| Local::now().date_naive());
            let mut candidate = start;
            let (report_date, files) = loop {
                let mut files = Vec::with_capacity(ppd::Jurisdiction::ALL.len());
                let mut failure = None;
                for jurisdiction in ppd::Jurisdiction::ALL {
                    match ppd::fetch_cumulative(jurisdiction, candidate) {
                        Ok(bytes) => files.push((jurisdiction, bytes)),
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    }
                }
                match failure {
                    None => break (candidate, files),
                    Some(e) if date.is_some() => {
                        return Err(e).with_context(|| {
                            format!("no complete PPD CREDITS report for {candidate}")
                        });
                    }
                    Some(e) if candidate <= start - chrono::Days::new(PPD_LOOKBACK_DAYS) => {
                        return Err(e).with_context(|| {
                            format!(
                                "no complete PPD CREDITS report in the {PPD_LOOKBACK_DAYS} days \
                                 up to {start}"
                            )
                        });
                    }
                    Some(e) => {
                        log::debug!("{candidate}: {e}; trying the day before");
                        candidate = candidate - chrono::Days::new(1);
                    }
                }
            };
            let mut prints = Vec::new();
            for (jurisdiction, bytes) in &files {
                let text = ppd::unzip_csv(bytes).with_context(|| {
                    format!("failed to open the {} report", jurisdiction.name())
                })?;
                prints.extend(ppd::parse_csv(&text, Some(*jurisdiction))?);
            }
            let urls: Vec<String> = files
                .iter()
                .map(|(j, _)| ppd::cumulative_url(*j, report_date))
                .collect();
            let origin = serde_json::json!({
                "urls": urls,
                "fetched_at": Local::now().to_rfc3339(),
            });
            (prints, report_date, origin)
        }
    };
    if let Some(needle) = &args.symbol {
        prints = ppd::filter_underlier(prints, needle);
    }
    log::info!(
        "PPD credit prints for {report_date}: {} rows{}",
        prints.len(),
        args.symbol
            .as_deref()
            .map_or(String::new(), |s| format!(" matching {s}"))
    );
    emit_document(
        args,
        ppd::to_document(&prints, report_date, args.symbol.as_deref()),
        origin,
        "cds_prints",
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
    // a one-sided VaR level below 0.5 is not a lower confidence, it is
    // the wrong tail: the loss quantile would sit in the profit region
    if !(args.confidence > 0.5 && args.confidence < 1.0) {
        bail!(
            "--confidence must be strictly between 0.5 and 1, got {}",
            args.confidence
        );
    }
    if !(args.horizon_days.is_finite() && args.horizon_days > 0.0) {
        bail!(
            "--horizon-days must be finite and positive, got {}",
            args.horizon_days
        );
    }
    if args.scenarios == 0 {
        bail!("--scenarios must be at least 1");
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
            let dg = delta_gamma_var(&book, spot, &cfg).context("delta-gamma VaR failed")?;
            report.insert("delta_gamma".into(), serde_json::to_value(&dg)?);
        }
        if matches!(args.method, RiskMethod::Full | RiskMethod::Both) {
            let full =
                full_revaluation_var(&book, spot, &cfg).context("full-revaluation VaR failed")?;
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
            years_from_today(date)
        }
    };
    // the solver's bracketing degenerates on these instead of failing,
    // so it would report the vol cap (or the initial guess) as an answer
    if !t.is_finite() || t <= 0.0 {
        bail!("--maturity must be a finite time in the future (t = {t} years)");
    }
    for (flag, value) in [
        ("--spot", args.spot),
        ("--strike", args.strike),
        ("--price", args.price),
    ] {
        if !value.is_finite() || value <= 0.0 {
            bail!("{flag} must be finite and positive, got {value}");
        }
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

/// Time to `date` in years, Act/365 from today — the convention the
/// non-contract entry points (the `implied-vol` command and the
/// interactive wizard) measure a bare maturity date with. Negative for a
/// date in the past; callers reject that themselves.
pub(crate) fn years_from_today(date: NaiveDate) -> f64 {
    use crate::core::daycount::DayCountConvention;
    DayCountConvention::Act365.year_fraction(Local::now().date_naive(), date)
}

/// Helper function to measure the time taken by a closure.
fn measure_time<T, F: FnOnce() -> T>(label: &str, f: F) -> T {
    let start_time = Instant::now();
    let result = f();
    let elapsed_time = start_time.elapsed();
    log::debug!("time taken for {}: {:?}", label, elapsed_time);
    result
}
