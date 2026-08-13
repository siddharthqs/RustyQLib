//! Market data to model, end to end: a real Cboe option chain ->
//! cleaned quotes -> parity-implied forwards -> Black-76 implied vols ->
//! implied surface (saved and reloaded as JSON) -> Dupire local vol ->
//! reprice the calibrating vanillas to confirm the calibration.
//!
//! Run with:   cargo run --release --features fetch --example chain_to_local_vol
//! Live data:  cargo run --release --features fetch --example chain_to_local_vol -- --live AAPL
//!
//! The default run uses the checked-in Cboe AAPL snapshot (real delayed
//! quotes, 2026-08-08), so it is reproducible offline; `--live` pulls
//! the current chain from cdn.cboe.com instead.

mod common;

use rustyqlib::core::curves::{Compounding, YieldCurve};
use rustyqlib::core::daycount::DayCountConvention;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::core::vols::VolSurface;
use rustyqlib::data::cboe;
use rustyqlib::equity::blackscholes::bs_price;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::local_vol::LocalVol;
use rustyqlib::equity::option_chain::{implied_vol_surface_from_chain, FilterConfig, OptionChain};
use rustyqlib::equity::utils::{Engine, Model};

/// Overnight-ish USD rate for discounting and forwards. The CLI
/// equivalent is `build --curve ust.json`, which bootstraps the fetched
/// Treasury par-yield document into a full discount curve; a flat rate
/// keeps this example self-contained.
const RATE: f64 = 0.037;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    common::title("OPTION CHAIN -> IMPLIED VOL SURFACE -> LOCAL VOL -> REPRICE");

    common::section("Step 1: the option chain");
    let args: Vec<String> = std::env::args().collect();
    let raw = if args.iter().any(|a| a == "--live") {
        let symbol = args
            .iter()
            .skip_while(|a| *a != "--live")
            .nth(1)
            .map(String::as_str)
            .unwrap_or("AAPL");
        println!("  fetching the live delayed chain for {symbol} from cdn.cboe.com ...");
        cboe::fetch(symbol)?
    } else {
        println!("  using the checked-in Cboe snapshot (tests/fixtures/cboe_chain_sample.json)");
        std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/cboe_chain_sample.json"
        ))?
    };
    let chain: OptionChain = cboe::to_chain(&raw)?;
    let spot = chain.spot.ok_or("chain has no underlying price")?;
    println!(
        "  {}: {} quotes as of {}, spot {spot}",
        chain.symbol,
        chain.quotes.len(),
        chain.as_of
    );

    common::section("Step 2: clean, imply forwards, solve implied vols");
    let curve = YieldCurve::flat(
        RATE,
        chain.as_of,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )?;
    let (surface, report) =
        implied_vol_surface_from_chain(&chain, &curve, &FilterConfig::default())?;
    for (expiry, forward) in &report.forwards {
        println!("  parity forward {expiry}: {forward:>10.4}   (spot {spot})");
    }
    println!(
        "  quotes used: {}   dropped: {:?}",
        report.quotes_used, report.dropped
    );
    println!("{surface}");

    common::section("Step 3: the surface is a file (save, reload, same surface)");
    let document = surface.to_document(Some(report.to_metadata(&chain)));
    let text = serde_json::to_string_pretty(&document)?;
    let reloaded = VolSurface::from_json(&text)?;
    let day_count = DayCountConvention::Act365;
    let (expiry, forward) = report.forwards[report.forwards.len() - 1];
    let t = day_count.year_fraction(chain.as_of, expiry);
    common::check(
        &format!("reloaded vol at K={spot:.0} T={t:.3}"),
        reloaded.vol(spot, forward, t),
        surface.vol(spot, forward, t),
        1e-14,
    );
    println!(
        "  ({} bytes of JSON, metadata records forwards and drop counts)",
        text.len()
    );

    common::section("Step 4: Dupire local vol from the implied surface");
    let lv = LocalVol::new(&surface, &curve, spot, 0.0, 0.0);
    println!(
        "  {:>10} {:>12} {:>12}",
        "level",
        "t=0.15",
        &format!("t={t:.2}")
    );
    for pct in [0.85, 0.95, 1.0, 1.05, 1.15] {
        let level = spot * pct;
        println!(
            "  {level:>10.2} {:>12.4} {:>12.4}",
            lv.vol(level, 0.15),
            lv.vol(level, t)
        );
    }
    common::note("steeper in strike than the implied smile (the 'twice the slope' rule);");
    common::note("wings are numerical-derivative territory — trust the interior.");

    common::section("Step 5: reprice the calibrating vanillas under local vol");
    println!(
        "  local vol FD price vs Black-Scholes at the surface's own implied vol\n  (agreement = the Dupire transformation is consistent with the surface it came from)\n"
    );
    println!(
        "  {:>8} {:>7} {:>12} {:>12} {:>12} {:>10}",
        "strike", "side", "market mid", "BS @ impl", "local vol", "diff"
    );
    let mut worst: f64 = 0.0;
    for pct in [0.95, 1.0, 1.05, 1.1] {
        // nearest listed strike to the moneyness point
        let target = forward * pct;
        let Some(strike) = nearest_strike(&chain, expiry, target) else {
            continue;
        };
        let side = if strike >= forward {
            PutOrCall::Call
        } else {
            PutOrCall::Put
        };
        let implied = surface.vol(strike, forward, t);
        let bs = bs_price(spot, strike, RATE, 0.0, implied, t, side);
        let option = EquityOptionBuilder::new()
            .spot(spot)
            .strike(strike)
            .vol_surface(surface.clone())
            .flat_rate(RATE)
            .valuation_date(chain.as_of)
            .maturity_date(expiry)
            .vanilla(side)
            .engine(Engine::FiniteDifference)
            .model(Model::LocalVol)
            .build()?;
        let local = option.price()?.pv;
        let market = market_mid(&chain, expiry, strike, side)
            .map_or_else(|| "-".to_string(), |m| format!("{m:>12.4}"));
        let diff = local - bs;
        worst = worst.max(diff.abs() / bs.max(0.05));
        println!(
            "  {strike:>8.1} {:>7} {market:>12} {bs:>12.4} {local:>12.4} {diff:>+10.4}",
            match side {
                PutOrCall::Call => "call",
                PutOrCall::Put => "put",
            }
        );
    }
    println!();
    common::check("worst relative reprice error", worst, 0.0, 0.02);
    common::note("market mid differs slightly from both models: it embeds the dividend-");
    common::note("adjusted forward, while this example reprices with q = 0 for clarity.");
    println!();
    Ok(())
}

/// The listed strike closest to `target` for `expiry`.
fn nearest_strike(chain: &OptionChain, expiry: chrono::NaiveDate, target: f64) -> Option<f64> {
    chain
        .quotes
        .iter()
        .filter(|q| q.expiry == expiry)
        .map(|q| q.strike)
        .min_by(|a, b| (a - target).abs().partial_cmp(&(b - target).abs()).unwrap())
}

fn market_mid(
    chain: &OptionChain,
    expiry: chrono::NaiveDate,
    strike: f64,
    side: PutOrCall,
) -> Option<f64> {
    chain
        .quotes
        .iter()
        .find(|q| q.expiry == expiry && q.right == side && (q.strike - strike).abs() < 1e-9)
        .and_then(|q| q.mid())
}
