//! Cliquet (ratchet) options: locally capped/floored period returns,
//! priced closed-form under Black-Scholes and by Monte Carlo under GBM
//! and Heston — the product where the forward smile earns its keep.
//!
//! Run with:  cargo run --release --example cliquet_option

mod common;

use rustyqlib::equity::cliquet::{Cliquet, CliquetPricer, CliquetStyle};
use rustyqlib::equity::heston::HestonParams;

fn base() -> Cliquet {
    Cliquet {
        resets: 12,
        t: 1.0,
        r: 0.03,
        q: 0.01,
        sigma: 0.20,
        local_floor: 0.0,
        local_cap: Some(0.03),
        global_floor: None,
        global_cap: None,
        notional: 1.0,
        heston: None,
        style: CliquetStyle::Standard,
        pricer: CliquetPricer::Analytical,
        paths: 200_000,
        seed: 42,
    }
}

fn row(label: &str, pv: f64, se: Option<f64>) {
    match se {
        Some(se) => println!("  {label:<44} {pv:>10.6}   (+/- {se:.6})"),
        None => println!("  {label:<44} {pv:>10.6}"),
    }
}

fn main() {
    common::title("CLIQUET / RATCHET — 12 monthly resets, floor 0, cap 3%, r=3% q=1% sigma=20%");

    common::section("Engines agree under Black-Scholes dynamics");
    let c = base();
    let analytic = c.analytic_npv().unwrap();
    let (mc, se) = c.mc_npv();
    row("Analytical (forward-start call spreads)", analytic, None);
    row("Monte Carlo (GBM, 200k paths)", mc, Some(se));
    common::check("MC vs closed form", mc, analytic, 4.0 * se);

    common::section("Local cap sweep (tighter cap, cheaper coupon strip)");
    for cap in [0.01, 0.02, 0.03, 0.05, 0.10] {
        let mut swept = base();
        swept.local_cap = Some(cap);
        row(&format!("local cap {:.0}%", cap * 100.0), swept.analytic_npv().unwrap(), None);
    }
    let mut ratchet = base();
    ratchet.local_cap = None;
    row("uncapped ratchet (floor 0 only)", ratchet.analytic_npv().unwrap(), None);

    common::section("Global constraints couple the periods (Monte Carlo territory)");
    let mut floored = base();
    floored.global_floor = Some(0.06);
    let (fl, fl_se) = floored.mc_npv();
    row("global floor 6%", fl, Some(fl_se));
    let mut capped = base();
    capped.global_cap = Some(0.15);
    let (cp, cp_se) = capped.mc_npv();
    row("global cap 15%", cp, Some(cp_se));
    common::note("the analytic engine refuses these and falls back to MC by design");

    common::section("Forward smile: Heston reprices the same cliquet away from flat vol");
    let mut heston = base();
    heston.heston = Some(HestonParams {
        v0: 0.04,
        kappa: 1.5,
        theta: 0.04,
        vol_of_vol: 0.7,
        rho: -0.7,
    });
    heston.paths = 100_000;
    let (hp, hp_se) = heston.mc_npv();
    row("Heston (same 20% total vol, xi=0.7, rho=-0.7)", hp, Some(hp_se));
    row("Black-Scholes flat 20%", analytic, None);
    common::note(&format!(
        "difference {:+.5} >> MC noise: the capped strip is short forward-smile",
        hp - analytic
    ));
    common::note("convexity — exactly why cliquets are priced on stochastic vol models");

    common::section("Reverse cliquet and Napoleon (coupon eroded by down months / worst month)");
    let mut reverse = base();
    reverse.style = CliquetStyle::Reverse { coupon: 0.20 };
    reverse.local_cap = None;
    let unfloored = reverse.analytic_npv().unwrap();
    reverse.global_floor = Some(0.0);
    let (rev, rev_se) = reverse.mc_npv();
    row("reverse: 20% coupon + sum of down months", rev, Some(rev_se));
    row("  (analytic without the 0% floor)", unfloored, None);
    let mut napoleon = base();
    napoleon.style = CliquetStyle::Napoleon { coupon: 0.10 };
    napoleon.local_cap = None;
    napoleon.global_floor = Some(0.0);
    let (nap, nap_se) = napoleon.mc_npv();
    row("napoleon: 10% coupon + worst month", nap, Some(nap_se));
    let mut nap_heston = napoleon.clone();
    nap_heston.heston = Some(HestonParams {
        v0: 0.04, kappa: 1.5, theta: 0.04, vol_of_vol: 0.7, rho: -0.7,
    });
    let (nap_h, nap_h_se) = nap_heston.mc_npv();
    row("napoleon under Heston (same total vol)", nap_h, Some(nap_h_se));
    common::note("note the direction: the 0% floor makes the buyer CONVEX in the worst");
    common::note("month, so vol-of-vol raises the floored Napoleon's value - the famous");
    common::note("Napoleon losses were the sellers' short position in that convexity");
    println!();
}
