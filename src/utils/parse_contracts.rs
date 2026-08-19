use crate::bonds::build_contracts::bootstrap_from_contracts;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::data_models::ProductData;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::utils::{CombinedContract, Contract, ContractOutput, Contracts};
use crate::core::vols::{VolInput, VolSurface};
use crate::data::cboe;
use crate::equity::build_contracts::build_eq_contracts_from_json;
use crate::equity::handle_equity_contracts::handle_equity_contract;
use crate::equity::local_vol::{LocalVol, LocalVolGrid};
use crate::equity::option_chain::{implied_vol_surface_from_chain, FilterConfig, OptionChain};
use crate::equity::portfolio::EquityPortfolio;
use crate::equity::sabr::SabrSurfaceFit;
use crate::equity::svi::SviSurfaceFit;
use crate::equity::usability::{usability_report, UsabilityConfig};
use crate::equity::vanilla_option::EquityOption;
use crate::utils::plot3d::{self, linspace, GreekSurface, Labels};
use anyhow::{bail, Context, Result};
use chrono::Local;
use std::fs;
use std::path::{Path, PathBuf};

use crate::core::serialization::{self, Format};
use rayon::prelude::*;
use serde_json::Value;

/// Confirm a build artifact on stdout, in green when it's a terminal.
fn saved_note(what: &str, path: &Path) {
    let success = crate::utils::style::SUCCESS;
    anstream::println!("{success}{what} saved to {}{success:#}", path.display());
}

/// Write `output` to `output_folder[/subfolder]/filename`, creating the
/// directories as needed, and return the path written to.
pub fn save_to_file(
    output_folder: &Path,
    subfolder: &str,
    filename: &str,
    output: &str,
) -> Result<PathBuf> {
    let mut dir = output_folder.to_path_buf();
    if !subfolder.is_empty() {
        dir.push(subfolder);
    }
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create output directory {}", dir.display()))?;
    dir.push(filename);
    fs::write(&dir, output).with_context(|| format!("failed to write {}", dir.display()))?;
    Ok(dir)
}

/// How an option chain is discounted during `build` (parity forwards
/// and de-discounting of mids).
pub enum ChainDiscount {
    /// Flat continuously compounded rate anchored at the chain's date.
    Flat(f64),
    /// A pre-built discount curve — e.g. bootstrapped from a fetched
    /// Treasury par-yield document — with a metadata description of its
    /// origin for the surface document.
    Curve {
        curve: YieldCurve,
        description: serde_json::Value,
    },
}

/// Build a curve (term structure, volatility surface, ...) from a
/// contracts document — or an implied vol surface from an option chain
/// document — and save it under `output_folder`. `source` names the
/// input (a path or "stdin") in error messages; `chain_discount` says
/// how chains are discounted (ignored for other inputs).
pub fn build_curve(
    contents: &str,
    source: &str,
    output_folder: &Path,
    chain_discount: ChainDiscount,
) -> Result<()> {
    let format = Format::detect(contents);
    let value = serialization::parse_value(contents, format)
        .with_context(|| format!("failed to parse {format:?} document {source}"))?;
    if let Some(chain) = chain_from_document(&value)
        .transpose()
        .with_context(|| format!("failed to read the option chain in {source}"))?
    {
        return build_chain_surface(&chain, chain_discount, output_folder);
    }
    let list_contracts: Contracts = serde_json::from_value(value)
        .map_err(|e| RustyQLibError::ParseError(format!("document does not match the schema: {e}")))
        .with_context(|| format!("failed to parse {format:?} curve definition {source}"))?;
    if list_contracts.contracts.is_empty() {
        bail!("no contracts found in {source}");
    }
    match list_contracts.asset.as_str() {
        "EQ" => {
            log::info!("building implied volatility surface");
            let contracts: Vec<Box<EquityOption>> =
                build_eq_contracts_from_json(list_contracts.contracts);
            let vol_surface = crate::equity::vol_surface::build_implied_vol_surface(&contracts)
                .context("failed to build implied vol surface")?;
            log::debug!("implied vol surface:\n{}", vol_surface);
            let vol_value =
                serde_json::to_value(&vol_surface).context("failed to serialize vol surface")?;
            let serialized_vol_surface =
                serialization::render_value(&vol_value, format, "vol_surface");
            let filename = format!("vol_surface.{}", format.extension());
            let out_path = save_to_file(
                output_folder,
                "vol_surface",
                &filename,
                &serialized_vol_surface,
            )?;
            saved_note("Volatility surface", &out_path);
        }
        "IR" => {
            log::info!("bootstrapping discount curve");
            let today = Local::now().date_naive();
            let curve = bootstrap_from_contracts(&list_contracts.contracts, today)
                .context("failed to bootstrap discount curve")?;
            log::debug!("bootstrapped curve:\n{curve}");
            let mut output = String::from("date,discount_factor,zero_rate_continuous\n");
            for pillar in curve.pillars() {
                let date = pillar
                    .date
                    .map_or_else(|| pillar.time.to_string(), |d| d.to_string());
                output.push_str(&format!("{},{},{}\n", date, pillar.df, pillar.zero_rate));
            }

            let out_path = save_to_file(
                output_folder,
                "term_structure",
                "term_structure.csv",
                &output,
            )?;
            saved_note("Term structure", &out_path);
        }
        "CO" => bail!("commodity curve building is not supported yet"),
        other => bail!("unsupported asset class `{other}` (expected EQ or IR)"),
    }
    Ok(())
}

/// Recognize an option chain document in any of its three shapes: the
/// normalized [`OptionChain`] JSON (`quotes` + `as_of`), the wrapped
/// `fetch chain` document (`response.data.options`), or a raw Cboe
/// response (`data.options`). `None` means "not a chain — try the
/// contracts schema".
fn chain_from_document(value: &serde_json::Value) -> Option<Result<OptionChain, RustyQLibError>> {
    if value.get("quotes").is_some_and(Value::is_array) && value.get("as_of").is_some() {
        return Some(serde_json::from_value(value.clone()).map_err(|e| {
            RustyQLibError::ParseError(format!("invalid option chain document: {e}"))
        }));
    }
    let response = value
        .get("response")
        .filter(|r| r["data"]["options"].is_array())
        .or_else(|| Some(value).filter(|v| v["data"]["options"].is_array()))?;
    Some(cboe::chain_from_value(response))
}

/// Build an implied vol surface from an option chain and save it as a
/// reloadable surface document plus an interactive 3-D plot. The
/// document's metadata records the forwards used, every dropped quote by
/// reason, and the discounting assumption.
fn build_chain_surface(
    chain: &OptionChain,
    discount: ChainDiscount,
    output_folder: &Path,
) -> Result<()> {
    log::info!(
        "building an implied vol surface from the {} chain ({} quotes as of {})",
        chain.symbol,
        chain.quotes.len(),
        chain.as_of
    );
    let (curve, discount_meta) = match discount {
        ChainDiscount::Flat(rate) => {
            if rate == 0.0 {
                log::warn!(
                    "discounting the chain at a 0% flat rate; pass --rate <r> or \
                     --curve <ust.json> for realistic parity forwards"
                );
            }
            let curve = YieldCurve::flat(
                rate,
                chain.as_of,
                DayCountConvention::Act365,
                Compounding::Continuous,
            )
            .map_err(RustyQLibError::from)?;
            let meta = serde_json::json!({
                "type": "flat",
                "rate": rate,
                "compounding": "continuous",
                "day_count": "Act365",
            });
            (curve, meta)
        }
        ChainDiscount::Curve { curve, description } => {
            let gap = (chain.as_of - curve.reference_date()).num_days();
            if gap.abs() > 7 {
                log::warn!(
                    "the discount curve is dated {} but the chain is as of {} \
                     ({gap} days apart)",
                    curve.reference_date(),
                    chain.as_of
                );
            }
            (curve, description)
        }
    };
    let (surface, report) = implied_vol_surface_from_chain(chain, &curve, &FilterConfig::default())
        .context("failed to build the implied vol surface")?;
    log::debug!("implied vol surface:\n{surface}");

    let mut metadata = report.to_metadata(chain);
    metadata["discount"] = discount_meta.clone();
    write_surface_artifacts(
        &surface,
        metadata.clone(),
        output_folder,
        "vol_surface",
        &format!("{} implied vol \u{2014} {}", chain.symbol, chain.as_of),
        "Volatility surface",
    )?;

    // Dupire local vol calibrated from that implied surface (a
    // non-parametric transformation, sampled on a level x time grid)
    let spot = match chain.spot {
        Some(spot) => spot,
        None => {
            // discount the front parity forward back to a spot proxy
            let (front, forward) = report.forwards[0];
            let df = curve.df_date(front) / curve.df_date(chain.as_of);
            let implied = forward * df;
            log::info!("chain has no spot; using {implied:.4} from the front parity forward");
            implied
        }
    };
    let numeric_note = "clamped to [1%, 300%]; wings and short times lean on numerical \
                        derivatives of the interpolated implied surface — trust the interior";
    let raw_local_vol = LocalVol::new(&surface, &curve, spot, 0.0, 0.0);
    write_local_vol_artifacts(
        &surface,
        &|level, t| raw_local_vol.vol_checked(level, t),
        spot,
        &curve,
        &discount_meta,
        chain,
        output_folder,
        &LocalVolSpec {
            stem: "local_vol",
            title: &format!("{} Dupire local vol \u{2014} {}", chain.symbol, chain.as_of),
            model: "Dupire local volatility",
            derived_from: "the as-quoted implied surface",
            note: numeric_note,
        },
    )?;

    // the same transformation as the pricing engines consume it: sampled
    // on the artifact axes and neighbour-repaired (guarded nodes bridged
    // from Dupire-valid neighbours instead of implied-vol fallbacks)
    let (levels, times) = local_vol_axes(&surface);
    let raw_grid = LocalVolGrid::sample(&raw_local_vol, levels, times);
    let raw_grid_note = format!(
        "sampled + neighbour-repaired grid (the pricing engines' form); \
         {} guarded nodes bridged from valid neighbours; clamped to [1%, 300%]",
        raw_grid.repaired_nodes()
    );
    write_local_vol_artifacts(
        &surface,
        &|level, t| raw_grid.vol_checked(level, t),
        spot,
        &curve,
        &discount_meta,
        chain,
        output_folder,
        &LocalVolSpec {
            stem: "local_vol_grid",
            title: &format!(
                "{} Dupire local vol (pricing grid) \u{2014} {}",
                chain.symbol, chain.as_of
            ),
            model: "Dupire local volatility (sampled + neighbour-repaired grid)",
            derived_from: "the as-quoted implied surface",
            note: &raw_grid_note,
        },
    )?;

    // cleaned versions: minimal-change static-arbitrage repair (convex
    // hull of call prices per expiry + forward total-variance sweep),
    // then the same artifacts again from the repaired surface
    let day_count = DayCountConvention::Act365;
    let forward_points: Vec<(f64, f64)> = report
        .forwards
        .iter()
        .map(|(expiry, forward)| (day_count.year_fraction(chain.as_of, *expiry), *forward))
        .collect();
    let forward_of = |t: f64| match forward_points.len() {
        1 => forward_points[0].1,
        _ => crate::core::interpolation::interp_pairs(&forward_points, t),
    };
    let (cleaned, repair) = crate::equity::surface_repair::repair_arbitrage(&surface, forward_of)
        .context("arbitrage repair failed")?;
    let adjusted = repair.butterfly_adjustments + repair.calendar_adjustments;
    if !repair.clean {
        log::warn!(
            "arbitrage repair did not fully converge after {} passes; \
             the cleaned surface still carries violations",
            repair.iterations
        );
    } else if adjusted > 0 {
        log::info!(
            "arbitrage repair adjusted {adjusted} pillar vols \
             (max change {:.4}) and dropped {}",
            repair.max_vol_change,
            repair.dropped_points
        );
    }
    let mut cleaned_meta = metadata;
    let mut repair_meta =
        serde_json::to_value(&repair).context("failed to serialize the repair report")?;
    repair_meta["method"] = serde_json::json!(
        "minimal-change: per-expiry convex-hull projection of call prices \
         + forward total-variance monotonicity sweep"
    );
    cleaned_meta["repair"] = repair_meta;
    cleaned_meta["diagnostics"] = cleaned.diagnostics(forward_of).to_metadata();
    write_surface_artifacts(
        &cleaned,
        cleaned_meta,
        output_folder,
        "vol_surface_cleaned",
        &format!(
            "{} implied vol (arbitrage-repaired) \u{2014} {}",
            chain.symbol, chain.as_of
        ),
        "Cleaned volatility surface",
    )?;
    let cleaned_local_vol = LocalVol::new(&cleaned, &curve, spot, 0.0, 0.0);
    write_local_vol_artifacts(
        &cleaned,
        &|level, t| cleaned_local_vol.vol_checked(level, t),
        spot,
        &curve,
        &discount_meta,
        chain,
        output_folder,
        &LocalVolSpec {
            stem: "local_vol_cleaned",
            title: &format!(
                "{} Dupire local vol (arbitrage-repaired) \u{2014} {}",
                chain.symbol, chain.as_of
            ),
            model: "Dupire local volatility",
            derived_from: "the arbitrage-repaired implied surface",
            note: numeric_note,
        },
    )?;

    let (levels, times) = local_vol_axes(&cleaned);
    let cleaned_grid = LocalVolGrid::sample(&cleaned_local_vol, levels, times);
    let cleaned_grid_note = format!(
        "sampled + neighbour-repaired grid (the pricing engines' form); \
         {} guarded nodes bridged from valid neighbours; clamped to [1%, 300%]",
        cleaned_grid.repaired_nodes()
    );
    write_local_vol_artifacts(
        &cleaned,
        &|level, t| cleaned_grid.vol_checked(level, t),
        spot,
        &curve,
        &discount_meta,
        chain,
        output_folder,
        &LocalVolSpec {
            stem: "local_vol_grid_cleaned",
            title: &format!(
                "{} Dupire local vol (arbitrage-repaired, pricing grid) \u{2014} {}",
                chain.symbol, chain.as_of
            ),
            model: "Dupire local volatility (sampled + neighbour-repaired grid)",
            derived_from: "the arbitrage-repaired implied surface",
            note: &cleaned_grid_note,
        },
    )?;

    // third flavor: the SVI smoother, fitted to the cleaned smiles —
    // C^2 in strike, so its Dupire local vol is analytic and smooth
    match SviSurfaceFit::fit(&cleaned, forward_of) {
        Ok(fit) => {
            let sampled = fit.to_vol_surface(61).map_err(RustyQLibError::from)?;
            let poorly_fit = fit.slices.iter().filter(|s| s.rmse > 0.005).count();
            let arbitrage_slices = fit.slices.iter().filter(|s| s.min_g < 0.0).count();
            if poorly_fit + arbitrage_slices > 0 {
                log::warn!(
                    "SVI fit: {poorly_fit} slices with vol RMSE above 50 bps, \
                     {arbitrage_slices} with negative butterfly g in the quoted range \
                     — see the fit metadata"
                );
            }
            let mut svi_meta = report.to_metadata(chain);
            svi_meta["discount"] = discount_meta.clone();
            svi_meta["svi_fit"] = fit.metadata();
            svi_meta["diagnostics"] = sampled.diagnostics(forward_of).to_metadata();
            write_surface_artifacts(
                &sampled,
                svi_meta,
                output_folder,
                "vol_surface_svi",
                &format!(
                    "{} implied vol (SVI fit) \u{2014} {}",
                    chain.symbol, chain.as_of
                ),
                "SVI volatility surface",
            )?;
            write_local_vol_artifacts(
                &sampled,
                &|level, t| fit.local_vol_checked(level, t),
                spot,
                &curve,
                &discount_meta,
                chain,
                output_folder,
                &LocalVolSpec {
                    stem: "local_vol_svi",
                    title: &format!(
                        "{} Dupire local vol (SVI fit) \u{2014} {}",
                        chain.symbol, chain.as_of
                    ),
                    model: "Dupire local volatility (analytic on the per-expiry SVI fit)",
                    derived_from: "the SVI fit of the arbitrage-repaired surface",
                    note: "clamped to [1%, 300%]; Gatheral's formula with closed-form SVI \
                           derivatives — smooth by construction inside the quoted region",
                },
            )?;
        }
        Err(e) => log::warn!("SVI fit skipped: {e}"),
    }

    // fourth flavor: the SABR smoother (Hagan lognormal, beta = 1 — the
    // equity backbone convention), also fitted to the cleaned smiles:
    // three parameters per expiry, wings extrapolated by the model's
    // dynamics rather than a spline
    match SabrSurfaceFit::fit(&cleaned, forward_of, 1.0) {
        Ok(fit) => {
            let sampled = fit.to_vol_surface(61).map_err(RustyQLibError::from)?;
            let poorly_fit = fit.slices.iter().filter(|s| s.rmse > 0.005).count();
            let arbitrage_slices = fit.slices.iter().filter(|s| s.min_g < 0.0).count();
            if poorly_fit + arbitrage_slices > 0 {
                log::warn!(
                    "SABR fit: {poorly_fit} slices with vol RMSE above 50 bps, \
                     {arbitrage_slices} with negative butterfly g in the quoted range \
                     — see the fit metadata"
                );
            }
            let mut sabr_meta = report.to_metadata(chain);
            sabr_meta["discount"] = discount_meta.clone();
            sabr_meta["sabr_fit"] = fit.metadata();
            sabr_meta["diagnostics"] = sampled.diagnostics(forward_of).to_metadata();
            write_surface_artifacts(
                &sampled,
                sabr_meta,
                output_folder,
                "vol_surface_sabr",
                &format!(
                    "{} implied vol (SABR fit, beta = 1) \u{2014} {}",
                    chain.symbol, chain.as_of
                ),
                "SABR volatility surface",
            )?;
        }
        Err(e) => log::warn!("SABR fit skipped: {e}"),
    }
    Ok(())
}

/// Write one surface as its reloadable document plus 3-D plot under
/// `output_folder/vol_surface/<stem>.{json,html}`.
fn write_surface_artifacts(
    surface: &VolSurface,
    metadata: serde_json::Value,
    output_folder: &Path,
    stem: &str,
    title: &str,
    label: &str,
) -> Result<()> {
    let document = surface.to_document(Some(metadata));
    let rendered =
        serde_json::to_string_pretty(&document).context("failed to serialize the surface")?;
    let json_path = save_to_file(
        output_folder,
        "vol_surface",
        &format!("{stem}.json"),
        &rendered,
    )?;
    saved_note(label, &json_path);
    let html = plot3d::vol_surface_html(surface, title);
    let html_path = save_to_file(output_folder, "vol_surface", &format!("{stem}.html"), &html)?;
    saved_note(&format!("{label} plot"), &html_path);
    Ok(())
}

/// Naming and provenance for one local-vol artifact pair.
struct LocalVolSpec<'a> {
    stem: &'a str,
    title: &'a str,
    model: &'a str,
    derived_from: &'a str,
    note: &'a str,
}

/// Sample the instrumented `local_vol(level, t) -> (vol, guard_fired)`
/// over the axes implied by `axes_surface`, attach the usability report
/// (round-trip repricing, clamp/fallback fractions, trusted region),
/// and write the grid document plus 3-D plot under
/// `output_folder/local_vol/<stem>.{json,html}`.
#[allow(clippy::too_many_arguments)]
fn write_local_vol_artifacts(
    axes_surface: &VolSurface,
    local_vol: &dyn Fn(f64, f64) -> (f64, bool),
    spot: f64,
    curve: &YieldCurve,
    discount_meta: &serde_json::Value,
    chain: &OptionChain,
    output_folder: &Path,
    spec: &LocalVolSpec,
) -> Result<()> {
    let (levels, times) = local_vol_axes(axes_surface);
    let vols: Vec<Vec<f64>> = levels
        .iter()
        .map(|&level| times.iter().map(|&t| local_vol(level, t).0).collect())
        .collect();
    let usability = usability_report(
        axes_surface,
        local_vol,
        &levels,
        &times,
        curve,
        spot,
        &UsabilityConfig::default(),
    );
    if !usability.within_desk_tolerance {
        log::warn!(
            "{}: outside desk tolerance (round trip mean {:.1} / max {:.1} vol bps, \
             {:.1}% clamped, {:.1}% guard fallbacks)",
            spec.stem,
            usability.roundtrip.mean_vol_bps,
            usability.roundtrip.max_vol_bps,
            usability.clamped_fraction * 100.0,
            usability.fallback_fraction * 100.0
        );
    }
    let document = serde_json::json!({
        "metadata": {
            "model": spec.model,
            "derived_from": {
                "symbol": chain.symbol,
                "as_of": chain.as_of.to_string(),
                "surface": spec.derived_from,
            },
            "spot": spot,
            "dividend_yield": 0.0,
            "discount": discount_meta,
            "grid": "vols[i][j] = local vol at levels[i], times[j] (years, Act/365)",
            "note": spec.note,
            "usability": serde_json::to_value(&usability)
                .context("failed to serialize the usability report")?,
        },
        "levels": &levels,
        "times": &times,
        "vols": &vols,
    });
    let rendered = serde_json::to_string_pretty(&document)
        .context("failed to serialize the local vol grid")?;
    let json_path = save_to_file(
        output_folder,
        "local_vol",
        &format!("{}.json", spec.stem),
        &rendered,
    )?;
    saved_note("Local vol grid", &json_path);
    let sampled = GreekSurface {
        xs: levels,
        ys: times,
        z: vols,
    };
    let labels = Labels {
        title: spec.title,
        x: "underlying level",
        y: "time (years)",
        z: "local vol",
    };
    let html_path = save_to_file(
        output_folder,
        "local_vol",
        &format!("{}.html", spec.stem),
        &plot3d::surface_html(&sampled, &labels),
    )?;
    saved_note("Local vol plot", &html_path);
    Ok(())
}

/// Grid axes for sampling a local vol surface: the implied surface's own
/// quoted strike span, and times from two weeks out to the last pillar
/// (short times and wings are where Dupire's numerical derivatives get
/// noisy, so the grid stays inside the quoted region).
fn local_vol_axes(surface: &VolSurface) -> (Vec<f64>, Vec<f64>) {
    let (mut lo, mut hi) = (50.0, 150.0);
    let mut t_max: f64 = 2.0;
    if let VolInput::StrikeSmiles {
        expiries, smiles, ..
    } = surface.to_input()
    {
        let strikes: Vec<f64> = smiles.iter().flatten().map(|&(k, _)| k).collect();
        lo = strikes.iter().copied().fold(f64::MAX, f64::min);
        hi = strikes.iter().copied().fold(f64::MIN, f64::max);
        t_max = expiries
            .iter()
            .map(|tenor| match tenor {
                crate::core::curves::Tenor::YearFraction(t) => *t,
                crate::core::curves::Tenor::Date(_) => 0.0,
            })
            .fold(0.0, f64::max);
    }
    let t_lo = (2.0 / 52.0_f64).min(0.5 * t_max);
    (linspace(lo, hi, 50), linspace(t_lo, t_max, 30))
}

/// Price every contract in a document and return the rendered results.
/// The input format is detected from the content (JSON or XML); the
/// output format defaults to the input format unless overridden. Returns
/// `Ok(None)` when the document holds no contracts.
pub fn price_document(contents: &str, out_format: Option<Format>) -> Result<Option<String>> {
    let in_format = Format::detect(contents);
    let out_format = out_format.unwrap_or(in_format);

    let list_contracts: Contracts = serialization::parse(contents, in_format)
        .with_context(|| format!("failed to parse {in_format:?} contracts"))?;

    if list_contracts.contracts.is_empty() {
        return Ok(None);
    }
    // parallel processing of each contract using rayon
    let mut output_vec: Vec<_> = list_contracts
        .contracts
        .par_iter()
        .enumerate()
        .map(|(index, data)| (index, process_contract(data)))
        .collect();
    output_vec.sort_by_key(|k| k.0);

    let results: Vec<Value> = output_vec.into_iter().map(|(_, v)| v).collect();
    Ok(Some(serialization::render_results(&results, out_format)))
}

/// Price every contract in `input_file` into `output_file`. The output
/// format comes from `format_override`, then the output file extension,
/// then the input format.
pub fn parse_contract(
    input_file: &Path,
    output_file: &Path,
    format_override: Option<Format>,
) -> Result<()> {
    let contents = fs::read_to_string(input_file)
        .with_context(|| format!("failed to read contract file {}", input_file.display()))?;

    let out_format = format_override.or_else(|| Format::from_path(output_file));
    let rendered = price_document(&contents, out_format)
        .with_context(|| format!("failed to price {}", input_file.display()))?;
    match rendered {
        Some(output_str) => fs::write(output_file, output_str)
            .with_context(|| format!("failed to write {}", output_file.display()))?,
        None => log::warn!(
            "no contracts found in {}; nothing written",
            input_file.display()
        ),
    }
    Ok(())
}

/// Load an equity options book from a contracts document for risk and
/// stress runs. Every contract must be an option on the same underlying;
/// the signed position quantity is taken from each contract's
/// `long_short` field (default 1).
pub fn build_portfolio(contents: &str) -> Result<EquityPortfolio> {
    let format = Format::detect(contents);
    let list_contracts: Contracts = serialization::parse(contents, format)
        .with_context(|| format!("failed to parse {format:?} portfolio document"))?;
    if list_contracts.contracts.is_empty() {
        bail!("no contracts found in portfolio document");
    }
    let mut book = EquityPortfolio::new();
    for (index, contract) in list_contracts.contracts.iter().enumerate() {
        let Some(ProductData::Option(data)) = &contract.product_type else {
            bail!("contract {index}: only option contracts can go into a risk/stress portfolio");
        };
        let option = *EquityOption::try_from_json(data)
            .with_context(|| format!("contract {index} failed to build"))?;
        if let Some(first) = book.positions.first() {
            if first.option.base.symbol != option.base.symbol {
                bail!(
                    "contract {index}: the portfolio must share one underlying \
                     (book is '{}', contract is '{}')",
                    first.option.base.symbol,
                    option.base.symbol
                );
            }
        }
        let quantity = data.base.long_short.unwrap_or(1) as f64;
        book.add(option, quantity);
    }
    Ok(book)
}

/// Price one contract, always producing one result `Value` per contract.
/// Failures are reported in the result's `error` field rather than by
/// panicking, so one bad contract cannot abort the batch (this runs on a
/// rayon worker thread).
pub fn process_contract(data: &Contract) -> Value {
    match (data.action.as_str(), data.asset.as_str()) {
        ("PV", "EQ") => handle_equity_contract(data),
        ("PV", "IR") => {
            let today = Local::now().date_naive();
            crate::bonds::service::price_ir_contract(data, today)
                .unwrap_or_else(|e| error_result(data, e.to_string()))
        }
        (action, asset) => error_result(
            data,
            format!("unsupported action/asset combination `{action}`/`{asset}`"),
        ),
    }
}

/// Render a failed contract in the same `{contract, output}` shape as a
/// priced one, with the message in `output.error`.
fn error_result(data: &Contract, msg: String) -> Value {
    log::warn!("contract error: {msg}");
    let combined = CombinedContract {
        contract: data.clone(),
        output: ContractOutput::from_error(msg),
    };
    serde_json::to_value(&combined)
        .unwrap_or_else(|e| serde_json::json!({ "error": e.to_string() }))
}
