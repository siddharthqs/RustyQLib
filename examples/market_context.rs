//! The typed Market store: market data separated from instruments.
//!
//! What this demonstrates, step by step:
//!   1. an instrument is a contract (`base`) plus the market it is
//!      currently **bound to** (`market: EquityMarketData` — the copy its
//!      engines read); the `Market` store is the **external, shared**
//!      source of the same objects, addressed by typed keys;
//!   2. `snapshot_market()` lifts the embedded objects into a store, and
//!      `npv_in(&market)` rebinds + reprices — reproducing `npv()` exactly;
//!   3. `Market::bumped(shocks)` builds a stressed market **without
//!      touching any instrument**: each risk factor object bumps itself
//!      (Quote, VolSurface, YieldCurve), then the whole book reprices
//!      under the new snapshot;
//!   4. the store is open-ended: a key type defined *in this example file*
//!      (a correlation) stores and retrieves its own value type — no
//!      library change needed;
//!   5. a Heston position's model parameters follow a vol scenario
//!      through the market path (the library's vega convention).
//!
//! Run with:  cargo run --release --example market_context

mod common;

use chrono::NaiveDate;
use rustyqlib::core::trade::PutOrCall;
use rustyqlib::core::traits::Instrument;
use rustyqlib::equity::builder::EquityOptionBuilder;
use rustyqlib::equity::bump::{Bump, BumpedMarket};
use rustyqlib::equity::heston::HestonParams;
use rustyqlib::equity::utils::{Engine, Model};
use rustyqlib::equity::vanilla_option::EquityOption;
// everything market-related is re-exported at the crate root
use rustyqlib::{BumpMode, Compounding, Discount, MarketKey, RiskFactor, Shock, Spot, Vol};

fn option(symbol: &str, strike: f64, engine: Engine) -> EquityOption {
    EquityOptionBuilder::new()
        .symbol(symbol)
        .spot(100.0)
        .strike(strike)
        .flat_vol(0.25)
        .flat_rate(0.03)
        .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
        .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 4).unwrap())
        .vanilla(PutOrCall::Call)
        .engine(engine)
        .build()
        .expect("option must build")
}

fn shock(factor: RiskFactor, mode: BumpMode, size: f64) -> Shock {
    Shock {
        factor,
        mode,
        size,
        underlying: None,
        tenors: None,
        shifts: None,
    }
}

fn main() {
    common::title("MARKET CONTEXT — one typed store, shared, bumped, repriced");

    // ── 1. instruments embed their market; the store shares it ─────────
    common::section("1. Snapshot: instrument-embedded market -> typed store");
    let call = option("ACME", 100.0, Engine::BlackScholes);
    let tree = option("ACME", 110.0, Engine::Binomial);

    let market = call.snapshot_market();
    println!(
        "  Market {{ valuation_date: {}, entries: {} }}",
        market.valuation_date(),
        market.len()
    );
    // typed lookups: the KEY pins the return type at compile time
    let spot = market.get(&Spot("ACME".into())).unwrap(); // -> &Quote
    let surf = market.get(&Vol("ACME".into())).unwrap(); // -> &VolSurface
    let curve = market.get(&Discount("USD".into())).unwrap(); // -> &YieldCurve
    println!(
        "  Spot(\"ACME\")     -> Quote {{ value: {} }}",
        spot.value()
    );
    println!(
        "  Vol(\"ACME\")      -> VolSurface, atm 1y vol = {:.4}",
        surf.vol(100.0, 100.0, 1.0)
    );
    println!(
        "  Discount(\"USD\")  -> YieldCurve, 1y zero = {:.4}",
        curve.zero_rate_with(1.0, Compounding::Continuous)
    );
    // a key with no entry is a typed error naming the key, not a panic
    println!(
        "  Vol(\"ZENO\")      -> {}",
        market.get(&Vol("ZENO".into())).unwrap_err()
    );

    // ── 2. rebinding reproduces npv() exactly ──────────────────────────
    common::section("2. Reprice under the snapshot: npv_in == npv, on any engine");
    for (label, opt) in [("BlackScholes K=100", &call), ("Binomial     K=110", &tree)] {
        let direct = opt.npv();
        let rebound = opt.npv_in(&market).unwrap();
        println!(
            "  {label}:  npv() = {direct:.6}   npv_in(&market) = {rebound:.6}   diff = {:.1e}",
            (rebound - direct).abs()
        );
    }

    // ── 3. bump the MARKET, not the instruments ────────────────────────
    common::section("3. Scenario: bump the market once, reprice everything under it");
    let crash = market
        .bumped(&[
            shock(RiskFactor::Spot, BumpMode::Relative, -0.20), // spot -20%
            shock(RiskFactor::Vol, BumpMode::Absolute, 0.10),   // vols +10 pts
        ])
        .unwrap();
    println!(
        "  crash market: spot {} -> {}, atm vol {:.2} -> {:.2}",
        spot.value(),
        crash.get(&Spot("ACME".into())).unwrap().value(),
        surf.vol(100.0, 100.0, 1.0),
        crash.get(&Vol("ACME".into())).unwrap().vol(80.0, 80.0, 1.0),
    );
    for (label, opt) in [("BlackScholes K=100", &call), ("Binomial     K=110", &tree)] {
        let base = opt.npv_in(&market).unwrap();
        let stressed = opt.npv_in(&crash).unwrap();
        println!(
            "  {label}:  base = {base:>9.4}   crash = {stressed:>9.4}   P&L = {:>+9.4}",
            stressed - base
        );
    }
    // the instruments were never touched: their embedded copy still says 100
    println!(
        "  (instrument untouched: call.market.spot = {})",
        call.market.spot.value()
    );

    // rate and time shocks go through the same door
    let later = market
        .bumped(&[
            shock(RiskFactor::Rate, BumpMode::Absolute, 0.01), // +100bp, curve bumps itself
            shock(RiskFactor::Time, BumpMode::Absolute, 30.0), // a month passes
        ])
        .unwrap();
    println!(
        "  rates+100bp & 30d decay: valuation {} -> {}, 1y zero {:.4} -> {:.4}, call {:.4} -> {:.4}",
        market.valuation_date(),
        later.valuation_date(),
        curve.zero_rate_with(1.0, Compounding::Continuous),
        later.get(&Discount("USD".into())).unwrap().zero_rate_with(1.0, Compounding::Continuous),
        call.npv_in(&market).unwrap(),
        call.npv_in(&later).unwrap(),
    );

    // ── 4. the store is open: define your own key type, here, today ────
    common::section("4. Open extension: a key type defined in THIS file");
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct Correlation(String, String);
    impl MarketKey for Correlation {
        type Value = f64;
    }
    let market = market.with(Correlation("ACME".into(), "ZENO".into()), 0.65);
    println!(
        "  Correlation(\"ACME\", \"ZENO\") -> {}   (no change to the library: the key type is local)",
        market.get(&Correlation("ACME".into(), "ZENO".into())).unwrap()
    );
    println!(
        "  Correlation(\"ACME\", \"OTHER\") -> {}",
        market
            .get(&Correlation("ACME".into(), "OTHER".into()))
            .unwrap_err()
    );

    // ── 5. models follow the market: Heston vega convention ────────────
    common::section("5. A Heston position under a vol scenario (no recalibration)");
    let mut heston = option("ACME", 100.0, Engine::BlackScholes);
    heston.model = Model::Heston(HestonParams {
        v0: 0.0625,
        kappa: 1.5,
        theta: 0.0625,
        vol_of_vol: 0.4,
        rho: -0.6,
    });
    let vols_up = market
        .bumped(&[shock(RiskFactor::Vol, BumpMode::Absolute, 0.02)])
        .unwrap();
    let base = heston.npv_in(&market).unwrap();
    let bumped = heston.npv_in(&vols_up).unwrap();
    println!(
        "  heston npv: base = {base:.6}, vols +2pts = {bumped:.6}  (params moved with the surface)"
    );
    let scalar = heston.price_bumped(&BumpedMarket::new(&heston.market, Bump::vol(0.02)));
    println!("  reference scalar path (vol bump +2pts) = {scalar:.6}");

    println!();
    common::note("EquityOption = base (contract terms, immutable) + market (EquityMarketData, the");
    common::note("bound copy engines read) + payoff/engine/model. The Market store is the shared");
    common::note("source of truth; with_market() swaps the bound copy in one move.");
}
