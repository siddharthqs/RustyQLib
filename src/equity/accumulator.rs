//! Accumulators and decumulators — the daily-accrual structured
//! products ("I-kill-you-later"): the holder is committed to trade a
//! fixed quantity at a fixed price on every observation date while the
//! structure is alive, with a knock-out barrier on the favorable side
//! and **geared** (typically doubled) quantity on the adverse side.
//!
//! - **Accumulator**: buy `shares_per_day` at strike `K < S_0` each day;
//!   knocked out when the spot rises to the barrier `H > S_0`; when the
//!   spot closes below `K` the holder must buy `gearing x` the quantity.
//!   Day value while alive: `q [ (S_i - K)+ - gearing (K - S_i)+ ]`.
//! - **Decumulator**: the mirror — sell at `K > S_0`, knocked out at
//!   `H < S_0`, geared when the spot closes above `K`.
//!
//! Priced two ways:
//! - **Analytical**: each observation day is a pair of Reiner-Rubinstein
//!   knock-out barrier options maturing on that day (up-and-out call
//!   minus geared up-and-out put for the accumulator; down-and-out put
//!   minus geared down-and-out call for the decumulator), so the value
//!   is a strip of closed forms. The barrier is **continuously**
//!   monitored in this representation.
//! - **Monte Carlo**: simulates the observation grid directly, with the
//!   knock-out checked **discretely** at each observation — the usual
//!   contractual convention. The discrete knockout survives slightly
//!   longer than the continuous one, so the two conventions bracket the
//!   product; the tests assert exact agreement in the barrier-free
//!   degenerate cases and closeness with dense observations.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::core::montecarlo::{mean_std_err, path_rng};
use crate::core::traits::Instrument;
use crate::equity::barrier::{barrier_price, BarrierDirection, KnockType};
use rand::Rng;
use rand_distr::StandardNormal;
use crate::core::errors::RustyQLibError;

/// Which side of the trade the holder accrues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorSide {
    /// Daily buyer at a discount, knocked out above.
    Accumulator,
    /// Daily seller at a premium, knocked out below.
    Decumulator,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulatorPricer {
    Analytical,
    MonteCarlo,
}

/// JSON contract data (`"product_type": "accumulator"`).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AccumulatorData {
    pub symbol: String,
    /// "accumulator" | "decumulator".
    pub side: String,
    pub underlying_price: f64,
    /// Contractual trade price (below spot for accumulators).
    pub strike: f64,
    /// Knock-out level (above spot for accumulators, below for decumulators).
    pub barrier: f64,
    /// Number of equally spaced observation days (last = maturity).
    pub observations: usize,
    /// Maturity date, `YYYY-MM-DD`.
    pub maturity: String,
    /// Shares traded per observation (default 1).
    pub shares_per_day: Option<f64>,
    /// Quantity multiplier on the adverse side (default 2 = double-up).
    pub gearing: Option<f64>,
    pub risk_free_rate: f64,
    pub dividend: Option<f64>,
    pub volatility: f64,
    pub pricer: Option<String>,
    pub simulation: Option<u64>,
    pub mc_seed: Option<u64>,
    /// Pricing as-of date (`YYYY-MM-DD`); defaults to today.
    pub valuation_date: Option<String>,
}

/// An accumulator/decumulator on equally spaced daily observations.
#[derive(Debug, Clone)]
pub struct Accumulator {
    pub side: AccumulatorSide,
    pub s0: f64,
    pub strike: f64,
    pub barrier: f64,
    pub observations: usize,
    /// Year fraction to maturity.
    pub t: f64,
    pub r: f64,
    pub q: f64,
    pub sigma: f64,
    pub shares_per_day: f64,
    pub gearing: f64,
    pub pricer: AccumulatorPricer,
    pub paths: usize,
    pub seed: u64,
}

impl Accumulator {
    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.observations < 1 || self.t <= 0.0 || self.sigma <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "accumulator",
                "observations must be >= 1, maturity and volatility must be positive",
            ));
        }
        if self.gearing < 0.0 || self.shares_per_day <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "accumulator",
                "gearing must be non-negative and shares_per_day positive",
            ));
        }
        match self.side {
            AccumulatorSide::Accumulator if self.barrier <= self.s0 => {
                Err(RustyQLibError::invalid_input(
                    "barrier",
                    "accumulator knock-out must be above the spot",
                ))
            }
            AccumulatorSide::Decumulator if self.barrier >= self.s0 => {
                Err(RustyQLibError::invalid_input(
                    "barrier",
                    "decumulator knock-out must be below the spot",
                ))
            }
            _ => Ok(()),
        }
    }

    /// Strip-of-barrier-options closed form (continuous knock-out
    /// monitoring). Each observation contributes a knock-out pair
    /// maturing on its date.
    pub fn analytic_npv(&self) -> f64 {
        let dt = self.t / self.observations as f64;
        let mut value = 0.0;
        for i in 1..=self.observations {
            let ti = i as f64 * dt;
            value += match self.side {
                AccumulatorSide::Accumulator => {
                    let uoc = barrier_price(
                        self.s0, self.strike, self.barrier, self.r, self.q, self.sigma,
                        ti, BarrierDirection::Up, KnockType::Out, crate::core::trade::PutOrCall::Call,
                    );
                    let uop = barrier_price(
                        self.s0, self.strike, self.barrier, self.r, self.q, self.sigma,
                        ti, BarrierDirection::Up, KnockType::Out, crate::core::trade::PutOrCall::Put,
                    );
                    uoc - self.gearing * uop
                }
                AccumulatorSide::Decumulator => {
                    let dop = barrier_price(
                        self.s0, self.strike, self.barrier, self.r, self.q, self.sigma,
                        ti, BarrierDirection::Down, KnockType::Out, crate::core::trade::PutOrCall::Put,
                    );
                    let doc = barrier_price(
                        self.s0, self.strike, self.barrier, self.r, self.q, self.sigma,
                        ti, BarrierDirection::Down, KnockType::Out, crate::core::trade::PutOrCall::Call,
                    );
                    dop - self.gearing * doc
                }
            };
        }
        self.shares_per_day * value
    }

    /// Monte Carlo on the observation grid: discrete knock-out at each
    /// observation (the contractual daily-close convention), accrual up
    /// to but excluding the knock-out day. Deterministic per seed.
    pub fn mc_npv(&self) -> (f64, f64) {
        let dt = self.t / self.observations as f64;
        let drift = (self.r - self.q - 0.5 * self.sigma * self.sigma) * dt;
        let vol = self.sigma * dt.sqrt();
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for i in 0..self.paths {
            let mut rng = path_rng(self.seed, i as u64);
            let mut s = self.s0;
            let mut value = 0.0;
            for obs in 1..=self.observations {
                let z: f64 = rng.sample(StandardNormal);
                s *= (drift + vol * z).exp();
                let knocked = match self.side {
                    AccumulatorSide::Accumulator => s >= self.barrier,
                    AccumulatorSide::Decumulator => s <= self.barrier,
                };
                if knocked {
                    break;
                }
                let ti = obs as f64 * dt;
                let df = (-self.r * ti).exp();
                let day = match self.side {
                    AccumulatorSide::Accumulator => {
                        (s - self.strike).max(0.0) - self.gearing * (self.strike - s).max(0.0)
                    }
                    AccumulatorSide::Decumulator => {
                        (self.strike - s).max(0.0) - self.gearing * (s - self.strike).max(0.0)
                    }
                };
                value += self.shares_per_day * day * df;
            }
            sum += value;
            sum_sq += value * value;
        }
        mean_std_err(sum, sum_sq, self.paths)
    }

    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`Accumulator::try_from_json`].
    pub fn from_json(data: &AccumulatorData) -> Box<Accumulator> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &AccumulatorData) -> Result<Box<Accumulator>, RustyQLibError> {
        let today =
            crate::core::data_models::parse_valuation_date(data.valuation_date.as_deref())?;
        let maturity = NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d")
            .map_err(|_| RustyQLibError::invalid_input(
                "maturity",
                format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
            ))?;
        let t = (maturity - today).num_days() as f64 / 365.0;
        if t <= 0.0 {
            return Err(RustyQLibError::invalid_input("maturity", "accumulator is expired"));
        }
        let side = match data.side.trim().to_lowercase().as_str() {
            "accumulator" | "accu" => AccumulatorSide::Accumulator,
            "decumulator" | "decu" => AccumulatorSide::Decumulator,
            other => return Err(RustyQLibError::invalid_input(
                "side",
                format!("invalid accumulator side '{other}' (use accumulator or decumulator)"),
            )),
        };
        let pricer = match data.pricer.as_deref().map(str::trim) {
            None | Some("Analytical") | Some("analytical") => AccumulatorPricer::Analytical,
            Some("MonteCarlo") | Some("montecarlo") | Some("MC") | Some("mc") => {
                AccumulatorPricer::MonteCarlo
            }
            Some(other) => return Err(RustyQLibError::invalid_input(
                "pricer",
                format!("invalid accumulator pricer '{other}' (use Analytical or MonteCarlo)"),
            )),
        };
        let out = Accumulator {
            side,
            s0: data.underlying_price,
            strike: data.strike,
            barrier: data.barrier,
            observations: data.observations,
            t,
            r: data.risk_free_rate,
            q: data.dividend.unwrap_or(0.0),
            sigma: data.volatility,
            shares_per_day: data.shares_per_day.unwrap_or(1.0),
            gearing: data.gearing.unwrap_or(2.0),
            pricer,
            paths: data.simulation.unwrap_or(100_000) as usize,
            seed: data.mc_seed.unwrap_or(42),
        };
        out.validate()?;
        Ok(Box::new(out))
    }
}

/// The accumulator as a mainline [`Payoff`], pricing inside
/// [`EquityOption`](crate::equity::vanilla_option::EquityOption) on the
/// shared Monte Carlo engine (GBM, local vol and Heston via QE-M) — which
/// gives it the market context for free: `snapshot_market`/`npv_in`
/// rebinding, portfolio membership and the stress runner. The strike is
/// the contract's `strike_price`; spot, curve and surface come from the
/// bound market. The standalone [`Accumulator`] remains the closed-form
/// (continuously monitored) validation reference.
///
/// Cash flows land on their own observation dates, so like
/// [`AutocallablePayoff`](crate::equity::autocallable::AutocallablePayoff)
/// it is valued per path through [`path_value`](Self::path_value) with
/// per-date discount factors, not through `path_payoff`.
#[derive(Debug, Clone)]
pub struct AccumulatorPayoff {
    pub exercise_style: crate::core::utils::ContractStyle,
    pub side: AccumulatorSide,
    /// Knock-out level (above spot for accumulators, below for
    /// decumulators; enforced at build).
    pub barrier: f64,
    /// Equally spaced observation days over the life (last = maturity).
    pub observations: usize,
    pub shares_per_day: f64,
    /// Quantity multiplier on the adverse side (2 = classic double-up).
    pub gearing: f64,
}

impl AccumulatorPayoff {
    /// Value of one simulated path: daily accrual `q [ (S_i - K)+ -
    /// gearing (K - S_i)+ ]` (mirrored for decumulators), each day
    /// discounted on its own date, stopping — without accruing — on the
    /// first observation at or through the knock-out. `obs_idx` maps
    /// observation m to its path step; `dfs[m]` discounts its date.
    pub fn path_value(&self, path: &[f64], obs_idx: &[usize], dfs: &[f64], strike: f64) -> f64 {
        let mut value = 0.0;
        for (m, &idx) in obs_idx.iter().enumerate() {
            let s = path[idx];
            let (knocked, day) = match self.side {
                AccumulatorSide::Accumulator => (
                    s >= self.barrier,
                    (s - strike).max(0.0) - self.gearing * (strike - s).max(0.0),
                ),
                AccumulatorSide::Decumulator => (
                    s <= self.barrier,
                    (strike - s).max(0.0) - self.gearing * (s - strike).max(0.0),
                ),
            };
            if knocked {
                break;
            }
            value += self.shares_per_day * day * dfs[m];
        }
        value
    }
}

impl crate::equity::utils::Payoff for AccumulatorPayoff {
    /// Degenerate single-point value: zero (all value is schedule- and
    /// path-dependent).
    fn payoff(&self, _spot: f64, _strike: f64) -> f64 {
        0.0
    }
    fn path_payoff(&self, _path: &[f64], _strike: f64) -> f64 {
        panic!(
            "Accumulators pay at multiple dates and cannot be valued through \
             path_payoff; the Monte Carlo engine prices them via path_value"
        );
    }
    fn is_path_dependent(&self) -> bool {
        true
    }
    fn payoff_kind(&self) -> crate::equity::utils::PayoffType {
        crate::equity::utils::PayoffType::Accumulator
    }
    fn put_or_call(&self) -> &crate::core::trade::PutOrCall {
        // the holder's daily optionality is call-shaped for accumulators
        // (buy below), put-shaped for decumulators; not used by pricing
        match self.side {
            AccumulatorSide::Accumulator => &crate::core::trade::PutOrCall::Call,
            AccumulatorSide::Decumulator => &crate::core::trade::PutOrCall::Put,
        }
    }
    fn exercise_style(&self) -> &crate::core::utils::ContractStyle {
        &self.exercise_style
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn crate::equity::utils::Payoff> {
        Box::new(self.clone())
    }
}

impl Instrument for Accumulator {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.price()?.pv)
    }

    fn price(&self) -> Result<crate::core::results::PricingResult, RustyQLibError> {
        // typed rejection for directly constructed accumulators;
        // try_from_json validates at construction
        self.validate()?;
        let (pv, std_err) = match self.pricer {
            AccumulatorPricer::Analytical => (self.analytic_npv(), None),
            AccumulatorPricer::MonteCarlo => {
                let (pv, se) = self.mc_npv();
                (pv, Some(se))
            }
        };
        Ok(crate::core::results::PricingResult { pv, greeks: Default::default(), std_err })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::blackscholes::bs_price;

    fn base() -> Accumulator {
        Accumulator {
            side: AccumulatorSide::Accumulator,
            s0: 100.0,
            strike: 95.0,
            barrier: 110.0,
            observations: 252,
            t: 1.0,
            r: 0.03,
            q: 0.01,
            sigma: 0.25,
            shares_per_day: 1.0,
            gearing: 2.0,
            pricer: AccumulatorPricer::Analytical,
            paths: 40_000,
            seed: 42,
        }
    }

    #[test]
    fn barrier_free_accumulator_is_an_exact_vanilla_strip() {
        // barrier far away: each day is exactly call(K) - gearing put(K)
        let mut a = base();
        a.barrier = 1e6;
        a.observations = 12;
        let dt = a.t / 12.0;
        let strip: f64 = (1..=12)
            .map(|i| {
                let ti = i as f64 * dt;
                bs_price(a.s0, a.strike, a.r, a.q, a.sigma, ti, PutOrCall::Call)
                    - a.gearing * bs_price(a.s0, a.strike, a.r, a.q, a.sigma, ti, PutOrCall::Put)
            })
            .sum();
        assert!((a.analytic_npv() - strip).abs() < 1e-8, "{} vs {strip}", a.analytic_npv());
        // and MC agrees with the exact strip within noise
        let (mc, se) = a.mc_npv();
        assert!((mc - strip).abs() < 3.0 * se + 0.05, "mc {mc} +/- {se} vs {strip}");
    }

    #[test]
    fn no_gearing_no_barrier_is_a_forward_strip() {
        // gearing 1, no barrier: day value is S_i - K, a pure forward
        let mut a = base();
        a.barrier = 1e6;
        a.gearing = 1.0;
        a.observations = 4;
        let dt = a.t / 4.0;
        let forwards: f64 = (1..=4)
            .map(|i| {
                let ti = i as f64 * dt;
                a.s0 * (-a.q * ti).exp() - a.strike * (-a.r * ti).exp()
            })
            .sum();
        assert!((a.analytic_npv() - forwards).abs() < 1e-8);
        let (mc, se) = a.mc_npv();
        assert!((mc - forwards).abs() < 3.0 * se + 0.05, "mc {mc} vs {forwards}");
    }

    #[test]
    fn analytic_strip_tracks_dense_monte_carlo_with_the_barrier() {
        // dense observations shrink the discrete-vs-continuous knockout
        // gap; the discretely monitored MC survives longer, so it sits
        // above the continuous strip for this (positive-value) structure
        let a = base(); // 252 daily observations
        let analytic = a.analytic_npv();
        let (mc, se) = a.mc_npv();
        assert!(mc > analytic - 3.0 * se, "discrete KO should not lose value");
        assert!(
            (mc - analytic).abs() < 0.05 * analytic.abs().max(5.0) + 3.0 * se,
            "mc {mc} +/- {se} vs analytic {analytic}"
        );
    }

    #[test]
    fn decumulator_mirrors_and_orders_sensibly() {
        let mut d = base();
        d.side = AccumulatorSide::Decumulator;
        d.strike = 105.0;
        d.barrier = 90.0;
        let analytic = d.analytic_npv();
        let (mc, se) = d.mc_npv();
        assert!((mc - analytic).abs() < 0.05 * analytic.abs().max(5.0) + 3.0 * se,
            "mc {mc} vs analytic {analytic}");
        // the discount/premium is what the knockout takes away: without
        // gearing and barrier the holder would simply be long value
        let mut favorable = d.clone();
        favorable.barrier = 1e-6;
        favorable.gearing = 1.0;
        assert!(favorable.analytic_npv() > analytic);
    }

    #[test]
    fn risk_features_move_the_price_the_right_way() {
        let a = base();
        let baseline = a.analytic_npv();
        // more gearing hurts the holder
        let mut geared = base();
        geared.gearing = 3.0;
        assert!(geared.analytic_npv() < baseline);
        // barrier direction is regime-dependent for the geared holder:
        // surviving paths are skewed below the strike (the toxic tail the
        // nickname warns about), so here a TIGHTER knockout helps by
        // killing the structure faster
        let mut tight = base();
        tight.barrier = 103.0;
        assert!(tight.analytic_npv() > baseline, "tight KO should truncate the toxic tail");
        // ... whereas for the ungeared long-call strip the theorem holds:
        // a tighter up-and-out barrier can only remove value
        let mut long_only = base();
        long_only.gearing = 0.0;
        let mut long_only_tight = long_only.clone();
        long_only_tight.barrier = 103.0;
        assert!(long_only_tight.analytic_npv() < long_only.analytic_npv());
        // a deeper strike discount helps
        let mut cheap = base();
        cheap.strike = 90.0;
        assert!(cheap.analytic_npv() > baseline);
        // higher vol hurts the geared holder (short the wings)
        let mut vol = base();
        vol.sigma = 0.40;
        assert!(vol.analytic_npv() < baseline, "accumulator holder is short vol");
    }

    // ── the mainline payoff: EquityOption integration ───────────────────

    use crate::core::market::{BumpMode, RiskFactor, Shock};
    use crate::core::utils::ContractStyle;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::portfolio::EquityPortfolio;
    use crate::equity::utils::Engine;
    use crate::equity::vanilla_option::EquityOption;
    use crate::risk::stress::{stress_mtm, ArbitrageCheck, StressConfig, StressScenario};

    fn payoff() -> AccumulatorPayoff {
        AccumulatorPayoff {
            exercise_style: ContractStyle::European,
            side: AccumulatorSide::Accumulator,
            barrier: 110.0,
            observations: 4,
            shares_per_day: 1.0,
            gearing: 2.0,
        }
    }

    fn option_accumulator(observations: usize, paths: usize) -> EquityOption {
        // the builder twin of `base()`: same market and contract terms
        EquityOptionBuilder::new()
            .symbol("ACCU")
            .spot(100.0)
            .strike(95.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .dividend_yield(0.01)
            .years_to_maturity(1.0)
            .accumulator(110.0, observations, 1.0, 2.0)
            .engine(Engine::MonteCarlo)
            .paths(paths)
            .seed(42)
            .build()
            .expect("accumulator option must build")
    }

    #[test]
    fn payoff_accrues_daily_and_stops_without_accruing_at_knockout() {
        let accu = payoff();
        let obs_idx = [0, 1, 2, 3];
        let dfs = [0.99, 0.98, 0.97, 0.96];
        // day 1: +5; day 2: 0 - 2*(95-94) = -2 (geared); day 3: knocked
        // at 112 >= 110 with no accrual; day 4 never reached
        let path = [100.0, 94.0, 112.0, 120.0];
        let value = accu.path_value(&path, &obs_idx, &dfs, 95.0);
        assert!((value - (5.0 * 0.99 - 2.0 * 0.98)).abs() < 1e-12, "{value}");
        // the decumulator mirrors: sell at 105, geared above, KO below 90
        let mut decu = payoff();
        decu.side = AccumulatorSide::Decumulator;
        decu.barrier = 90.0;
        // day 1: (105-100)=+5; day 2: 0 - 2*(112-105) = -14; day 3: 89 knocks
        let path = [100.0, 112.0, 89.0, 80.0];
        let value = decu.path_value(&path, &obs_idx, &dfs, 105.0);
        assert!((value - (5.0 * 0.99 - 14.0 * 0.98)).abs() < 1e-12, "{value}");
        // shares_per_day scales linearly
        let mut sized = payoff();
        sized.shares_per_day = 100.0;
        let path = [100.0, 94.0, 112.0, 120.0];
        let value = sized.path_value(&path, &obs_idx, &dfs, 95.0);
        assert!((value - 100.0 * (5.0 * 0.99 - 2.0 * 0.98)).abs() < 1e-10);
    }

    #[test]
    fn equity_option_route_tracks_the_standalone_reference() {
        // same contract on both routes: the standalone continuous-KO strip
        // vs the engine's discretely monitored Monte Carlo (Sobol, 252
        // observations) — same tolerance shape as the standalone MC test
        let reference = base().analytic_npv();
        let mc = option_accumulator(252, 20_000).npv();
        assert!(
            (mc - reference).abs() < 0.05 * reference.abs().max(5.0) + 0.5,
            "engine mc {mc} vs standalone analytic {reference}"
        );
        // direction of the monitoring gap: discrete KO survives longer,
        // and for the GEARED holder living longer is worse (the same
        // toxic-tail economics as `risk_features_move_the_price_the_right
        // _way`: a tighter/earlier knock-out helps) — so the discretely
        // monitored value sits at or below the continuous strip
        assert!(mc < reference + 1.0, "discrete KO {mc} vs continuous {reference}");
    }

    #[test]
    fn accumulator_reprices_in_the_market_context_and_stresses_sensibly() {
        let option = option_accumulator(12, 8_000);
        // snapshot / rebind parity is exact: the engine is seeded
        let market = option.snapshot_market();
        let direct = option.npv();
        let rebound = option.npv_in(&market).expect("must reprice");
        assert!((rebound - direct).abs() < 1e-12, "rebound {rebound} direct {direct}");

        // and the whole point of the migration: the stress runner sees it
        let mut book = EquityPortfolio::new();
        book.add(option, 1.0);
        let config = StressConfig {
            scenarios: vec![
                StressScenario {
                    name: "crash".into(),
                    shocks: vec![Shock {
                        factor: RiskFactor::Spot,
                        mode: BumpMode::Relative,
                        size: -0.20,
                        underlying: None,
                        tenors: None,
                        shifts: None,
                    }],
                },
                StressScenario {
                    name: "vols_up".into(),
                    shocks: vec![Shock {
                        factor: RiskFactor::Vol,
                        mode: BumpMode::Absolute,
                        size: 0.10,
                        underlying: None,
                        tenors: None,
                        shifts: None,
                    }],
                },
            ],
            arbitrage: ArbitrageCheck::default(),
        };
        let results = stress_mtm(&book, &config).expect("stress must run");
        // spot -20% through the geared strike is the toxic scenario
        assert!(results[0].stress_pnl < 0.0, "crash pnl {:?}", results[0].stress_pnl);
        // the geared holder is short vol (short the wings)
        assert!(results[1].stress_pnl < 0.0, "vol pnl {:?}", results[1].stress_pnl);
        // labels identify the product in reports
        assert!(results[0].trades[0].label.contains("Accumulator"), "{}", results[0].trades[0].label);
    }

    #[test]
    fn heston_route_degenerates_to_gbm_when_vol_of_vol_vanishes() {
        // v0 = theta = 0.25^2 and vanishing vol-of-vol: the QE-M paths
        // are (near) constant-variance, so the Heston route must land on
        // the GBM route's value up to sampler differences
        let gbm = option_accumulator(12, 8_000).npv();
        let heston = EquityOptionBuilder::new()
            .symbol("ACCU")
            .spot(100.0)
            .strike(95.0)
            .flat_rate(0.03)
            .dividend_yield(0.01)
            .years_to_maturity(1.0)
            .accumulator(110.0, 12, 1.0, 2.0)
            .heston(crate::equity::heston::HestonParams {
                v0: 0.0625,
                kappa: 2.0,
                theta: 0.0625,
                vol_of_vol: 1e-4,
                rho: 0.0,
            })
            .engine(Engine::MonteCarlo)
            .paths(8_000)
            .seed(42)
            .build()
            .expect("heston accumulator must build")
            .npv();
        assert!(
            (heston - gbm).abs() < 0.05 * gbm.abs().max(5.0),
            "heston {heston} vs gbm {gbm}"
        );
    }

    #[test]
    fn builder_validates_sides_and_engine_support() {
        let build = |barrier: f64| {
            EquityOptionBuilder::new()
                .symbol("ACCU")
                .spot(100.0)
                .strike(95.0)
                .flat_vol(0.25)
                .flat_rate(0.03)
                .years_to_maturity(1.0)
                .accumulator(barrier, 12, 1.0, 2.0)
                .engine(Engine::MonteCarlo)
                .build()
        };
        // accumulator knock-out below the spot is rejected
        assert!(build(90.0).is_err());
        assert!(build(110.0).is_ok());
        // decumulator mirrored
        let decu = EquityOptionBuilder::new()
            .symbol("ACCU")
            .spot(100.0)
            .strike(105.0)
            .flat_vol(0.25)
            .flat_rate(0.03)
            .years_to_maturity(1.0)
            .decumulator(110.0, 12, 1.0, 2.0)
            .engine(Engine::MonteCarlo)
            .build();
        assert!(decu.is_err(), "decumulator KO above spot must be rejected");
        // non-positive quantity and negative gearing are rejected
        let bad_shares = EquityOptionBuilder::new()
            .spot(100.0).strike(95.0).flat_vol(0.25).flat_rate(0.03)
            .years_to_maturity(1.0)
            .accumulator(110.0, 12, 0.0, 2.0)
            .engine(Engine::MonteCarlo)
            .build();
        assert!(bad_shares.is_err());
        // the analytic engine refuses accumulators at build time (build
        // runs check_engine_support), naming the engine that can price
        let err = EquityOptionBuilder::new()
            .spot(100.0).strike(95.0).flat_vol(0.25).flat_rate(0.03)
            .years_to_maturity(1.0)
            .accumulator(110.0, 12, 1.0, 2.0)
            .engine(Engine::BlackScholes)
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("MonteCarlo"), "{err}");
    }

    #[test]
    fn json_contract_round_trip() {
        let json = r#"{
            "symbol": "ACCU", "side": "accumulator", "underlying_price": 100.0,
            "strike": 95.0, "barrier": 110.0, "observations": 126,
            "maturity": "2030-01-01", "shares_per_day": 100.0,
            "risk_free_rate": 0.03, "dividend": 0.01, "volatility": 0.25,
            "pricer": "MC", "simulation": 20000
        }"#;
        let data: AccumulatorData = serde_json::from_str(json).unwrap();
        let accu = Accumulator::from_json(&data);
        assert_eq!(accu.side, AccumulatorSide::Accumulator);
        assert_eq!(accu.gearing, 2.0); // double-up default
        let pv = accu.npv();
        assert!(pv.is_finite() && pv.abs() < 100.0 * 126.0 * 20.0, "{pv}");
    }
}
