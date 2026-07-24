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

use chrono::{Local, NaiveDate};
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
        self.validate();
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
        self.validate();
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

    pub fn price(&self) -> f64 {
        match self.pricer {
            AccumulatorPricer::Analytical => self.analytic_npv(),
            AccumulatorPricer::MonteCarlo => self.mc_npv().0,
        }
    }

    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`Accumulator::try_from_json`].
    pub fn from_json(data: &AccumulatorData) -> Box<Accumulator> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    pub fn try_from_json(data: &AccumulatorData) -> Result<Box<Accumulator>, RustyQLibError> {
        let today = Local::now().date_naive();
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

impl Instrument for Accumulator {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.price())
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
