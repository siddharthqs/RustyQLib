//! Black-Karasinski: the lognormal short rate
//! `d ln r = (theta(t) - a ln r) dt + sigma dW`, fitted to the curve on a
//! trinomial tree.
//!
//! Rates stay positive and the volatility is proportional to the level
//! — the model desks used before negative rates — at the price of no
//! closed-form bonds at all: even `P(t,T)` is a tree calculation. The
//! tree is Hull-White's two-stage construction on `x = ln r - alpha(t)`:
//!
//! 1. `x` is an Ornstein-Uhlenbeck process from zero, `dx = -a x dt +
//!    sigma dW`, put on the recombining trinomial lattice with spacing
//!    `sigma sqrt(3 dt)` and Hull's edge-switching branches
//!    ([`hull_white_branching`]).
//! 2. Each layer's displacement `alpha_i` is solved so that the tree's
//!    discount factor to the next layer, `sum_j Q_ij e^{-r_ij dt}` with
//!    `r_ij = e^{alpha_i + j dx}` and `Q` the Arrow-Debreu prices,
//!    equals the curve's — a 1-D root per layer, forward in time.
//!
//! Everything else is backward induction on that tree: zero-bond
//! options, European and Bermudan swaptions ([`TailSwap`]), and the
//! [`ShortRateModel`] reconstitution `P(t, T | r)` by rolling ones back
//! from the maturity layer and interpolating in the state. Times off
//! the layer grid round to the nearest layer (within `dt / 2`).
//!
//! [`hull_white_branching`]: crate::core::lattice::hull_white_branching

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::lattice::{hull_white_branching, TrinomialLattice};
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::rates::models::ShortRateModel;
use crate::rates::PayerReceiver;

const FIELD: &str = "black_karasinski";

/// The swap a Bermudan exercises into at one date, in year fractions:
/// exercise at `expiry`, the notional exchanged at `start` and returned
/// at `last`, fixed `coupons` as `(payment time, amount)` in currency.
#[derive(Debug, Clone)]
pub struct TailSwap {
    pub expiry: f64,
    pub start: f64,
    pub coupons: Vec<(f64, f64)>,
    pub last: f64,
}

#[derive(Debug, Clone)]
pub struct BlackKarasinski {
    /// Mean reversion of `ln r`.
    pub a: f64,
    /// Volatility of `ln r`.
    pub sigma: f64,
    curve: YieldCurve,
    dt: f64,
    dx: f64,
    lattice: TrinomialLattice,
    /// `alpha_i` per layer, `0..=steps`.
    alphas: Vec<f64>,
}

impl BlackKarasinski {
    /// Build the fitted tree over `[0, horizon]` with `steps_per_year`
    /// layers per year. Products maturing beyond the horizon cannot be
    /// priced.
    pub fn new(
        a: f64,
        sigma: f64,
        curve: YieldCurve,
        horizon: f64,
        steps_per_year: usize,
    ) -> Result<Self, RustyQLibError> {
        if !(a > 0.0 && a.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("mean reversion must be positive, got {a}"),
            ));
        }
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("sigma must be positive, got {sigma}"),
            ));
        }
        if !(horizon > 0.0 && horizon.is_finite()) || steps_per_year == 0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need a positive horizon and at least one step per year",
            ));
        }
        let steps = (horizon * steps_per_year as f64).ceil() as usize;
        let dt = horizon / steps as f64;
        let dx = sigma * (3.0 * dt).sqrt();
        let j_cap = (0.184 / (a * dt)).ceil() as i32;
        let lattice =
            TrinomialLattice::build(steps, dt, dx, &|_, j| hull_white_branching(a, dt, j, j_cap))?;

        // forward induction: fit alpha_i layer by layer so the tree
        // reproduces P(0, t_{i+1}) with Arrow-Debreu prices Q_i
        let mut alphas = Vec::with_capacity(steps + 1);
        let mut q: Vec<f64> = vec![1.0];
        let short_end = {
            let t = 1e-5;
            -(curve.df(t).ln()) / t
        };
        let mut alpha_guess = short_end.max(1e-6).ln();
        for i in 0..=steps {
            let (lo, hi) = lattice.layer_range(i);
            let target = curve.df((i + 1) as f64 * dt);
            let tree_df = |alpha: f64| -> f64 {
                (lo..=hi)
                    .map(|j| q[(j - lo) as usize] * (-(alpha + j as f64 * dx).exp() * dt).exp())
                    .sum::<f64>()
            };
            // the tree df falls monotonically in alpha: bracket and bisect
            let (mut a_lo, mut a_hi) = (alpha_guess - 5.0, alpha_guess + 5.0);
            while tree_df(a_lo) < target {
                a_lo -= 5.0;
            }
            while tree_df(a_hi) > target {
                a_hi += 5.0;
            }
            let root =
                Solver1d::new(1e-14, 300).bisection(|alpha| tree_df(alpha) - target, a_lo, a_hi)?;
            alphas.push(root.x);
            alpha_guess = root.x;
            if i < steps {
                let (next_lo, next_hi) = lattice.layer_range(i + 1);
                let mut next = vec![0.0; (next_hi - next_lo + 1) as usize];
                for j in lo..=hi {
                    let b = lattice.branch(i, j);
                    let flow = q[(j - lo) as usize] * (-(root.x + j as f64 * dx).exp() * dt).exp();
                    let k = (b.target - next_lo) as usize;
                    next[k + 1] += b.p_up * flow;
                    next[k] += b.p_mid * flow;
                    next[k - 1] += b.p_down * flow;
                }
                q = next;
            }
        }
        Ok(BlackKarasinski {
            a,
            sigma,
            curve,
            dt,
            dx,
            lattice,
            alphas,
        })
    }

    pub fn curve(&self) -> &YieldCurve {
        &self.curve
    }

    /// Layer spacing in years.
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Layers in the tree.
    pub fn steps(&self) -> usize {
        self.lattice.steps()
    }

    /// The short rate at node `(layer, j)`.
    pub fn short_rate_at(&self, layer: usize, j: i32) -> f64 {
        (self.alphas[layer] + j as f64 * self.dx).exp()
    }

    /// The layer nearest to `t`, refusing times beyond the horizon.
    fn layer_of(&self, t: f64) -> Result<usize, RustyQLibError> {
        let layer = (t / self.dt).round();
        if !(t >= 0.0) || layer > self.lattice.steps() as f64 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "time {t} lies outside the tree horizon {}",
                    self.lattice.steps() as f64 * self.dt
                ),
            ));
        }
        Ok(layer as usize)
    }

    fn node_df(&self, layer: usize, j: i32) -> f64 {
        (-self.short_rate_at(layer, j) * self.dt).exp()
    }

    /// Roll a value vector on layer `from` back to layer `to`, adding
    /// `cash(layer)` at each layer passed (a coupon at that layer) — the
    /// vector at layer `k` is indexed `j - j_min(k)`.
    fn roll_back(
        &self,
        mut values: Vec<f64>,
        from: usize,
        to: usize,
        cash: &dyn Fn(usize) -> Option<Vec<f64>>,
    ) -> Vec<f64> {
        for i in (to..from).rev() {
            let (lo, hi) = self.lattice.layer_range(i);
            let next_lo = self.lattice.layer_range(i + 1).0;
            let mut layer = Vec::with_capacity((hi - lo + 1) as usize);
            for j in lo..=hi {
                let b = self.lattice.branch(i, j);
                let k = (b.target - next_lo) as usize;
                let expected =
                    b.p_up * values[k + 1] + b.p_mid * values[k] + b.p_down * values[k - 1];
                layer.push(self.node_df(i, j) * expected);
            }
            if let Some(add) = cash(i) {
                for (v, c) in layer.iter_mut().zip(add) {
                    *v += c;
                }
            }
            values = layer;
        }
        values
    }

    /// `P(layer, maturity)` per node of `layer`.
    fn bonds_at(&self, layer: usize, maturity: f64) -> Result<Vec<f64>, RustyQLibError> {
        let m = self.layer_of(maturity)?;
        if m < layer {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("maturity {maturity} precedes the valuation layer"),
            ));
        }
        let (lo, hi) = self.lattice.layer_range(m);
        Ok(self.roll_back(vec![1.0; (hi - lo + 1) as usize], m, layer, &|_| None))
    }

    /// The tree's own discount factor to `t` (equals the curve's at
    /// every layer by construction).
    pub fn tree_df(&self, t: f64) -> Result<f64, RustyQLibError> {
        Ok(self.bonds_at(0, t)?[0])
    }

    /// Value today of a vector of payoffs on `layer`.
    fn present_value(&self, layer: usize, payoff: Vec<f64>) -> f64 {
        self.roll_back(payoff, layer, 0, &|_| None)[0]
    }

    /// European option on a zero-coupon bond.
    pub fn zero_bond_option(
        &self,
        expiry: f64,
        bond_maturity: f64,
        strike: f64,
        put_or_call: PutOrCall,
    ) -> Result<f64, RustyQLibError> {
        if !(expiry > 0.0 && bond_maturity > expiry && strike > 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need 0 < expiry < bond maturity and a positive strike",
            ));
        }
        let e = self.layer_of(expiry)?;
        let bonds = self.bonds_at(e, bond_maturity)?;
        let payoff: Vec<f64> = bonds
            .iter()
            .map(|&p| match put_or_call {
                PutOrCall::Call => (p - strike).max(0.0),
                PutOrCall::Put => (strike - p).max(0.0),
            })
            .collect();
        Ok(self.present_value(e, payoff))
    }

    /// The signed swap value per node at the expiry layer of `tail`:
    /// `N [P(e, start) - P(e, last)] - sum coupons P(e, t_i)` for a payer.
    fn tail_values(
        &self,
        tail: &TailSwap,
        notional: f64,
        side: PayerReceiver,
    ) -> Result<Vec<f64>, RustyQLibError> {
        let e = self.layer_of(tail.expiry)?;
        let start = self.bonds_at(e, tail.start)?;
        let last = self.bonds_at(e, tail.last)?;
        let mut fixed = vec![0.0; start.len()];
        for &(pay, amount) in &tail.coupons {
            for (f, p) in fixed.iter_mut().zip(self.bonds_at(e, pay)?) {
                *f += amount * p;
            }
        }
        let sign = match side {
            PayerReceiver::Payer => 1.0,
            PayerReceiver::Receiver => -1.0,
        };
        Ok(start
            .iter()
            .zip(&last)
            .zip(&fixed)
            .map(|((s, l), f)| sign * (notional * (s - l) - f))
            .collect())
    }

    /// European swaption on a swap starting at `swap_start >= expiry`:
    /// the fixed leg pays `notional * strike_rate * tau` at each
    /// `(payment_time, tau)` and the notional at the last payment.
    pub fn european_swaption(
        &self,
        expiry: f64,
        swap_start: f64,
        fixed_leg: &[(f64, f64)],
        strike_rate: f64,
        notional: f64,
        payer_receiver: PayerReceiver,
    ) -> Result<f64, RustyQLibError> {
        if !(expiry > 0.0 && swap_start >= expiry && notional > 0.0 && strike_rate > 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "need expiry > 0, swap start >= expiry, positive notional and strike",
            ));
        }
        let Some(&(last, _)) = fixed_leg.last() else {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "the fixed leg has no payments",
            ));
        };
        let tail = TailSwap {
            expiry,
            start: swap_start,
            coupons: fixed_leg
                .iter()
                .map(|&(t, tau)| (t, notional * strike_rate * tau))
                .collect(),
            last,
        };
        let values = self.tail_values(&tail, notional, payer_receiver)?;
        let e = self.layer_of(expiry)?;
        Ok(self.present_value(e, values.into_iter().map(|v| v.max(0.0)).collect()))
    }

    /// Bermudan swaption: on each `tails[k].expiry` (ascending) the
    /// holder may enter that tail; `max(continuation, exercise)` rolled
    /// back on the tree.
    pub fn bermudan_swaption(
        &self,
        tails: &[TailSwap],
        notional: f64,
        payer_receiver: PayerReceiver,
    ) -> Result<f64, RustyQLibError> {
        if tails.is_empty() {
            return Err(RustyQLibError::invalid_input(FIELD, "no exercise dates"));
        }
        for pair in tails.windows(2) {
            if pair[1].expiry <= pair[0].expiry {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    "exercise dates must be ascending",
                ));
            }
        }
        let mut values: Option<Vec<f64>> = None;
        let mut current_layer = 0;
        for tail in tails.iter().rev() {
            let e = self.layer_of(tail.expiry)?;
            let exercise = self.tail_values(tail, notional, payer_receiver)?;
            let continuation = match values {
                Some(v) => self.roll_back(v, current_layer, e, &|_| None),
                None => vec![0.0; exercise.len()],
            };
            values = Some(
                continuation
                    .iter()
                    .zip(&exercise)
                    .map(|(c, x)| c.max(*x))
                    .collect(),
            );
            current_layer = e;
        }
        Ok(self.present_value(current_layer, values.expect("at least one tail")))
    }

    /// `alpha(t)` between layers, linearly.
    fn alpha_at(&self, t: f64) -> f64 {
        let position = (t / self.dt).clamp(0.0, self.lattice.steps() as f64);
        let i = (position.floor() as usize).min(self.lattice.steps() - 1);
        let w = position - i as f64;
        self.alphas[i] * (1.0 - w) + self.alphas[i + 1] * w
    }
}

impl ShortRateModel for BlackKarasinski {
    fn initial_short_rate(&self) -> f64 {
        self.alphas[0].exp()
    }

    /// `P(t, T | r)`: the bond vector at the layer nearest `t`, read at
    /// the state `x = ln r - alpha_i` by linear interpolation in `j`.
    fn zero_bond(&self, t: f64, maturity: f64, short_rate: f64) -> Result<f64, RustyQLibError> {
        if !(short_rate > 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("a lognormal short rate must be positive, got {short_rate}"),
            ));
        }
        if maturity < t {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("maturity {maturity} must be at or after t {t}"),
            ));
        }
        let i = self.layer_of(t)?;
        let bonds = self.bonds_at(i, maturity)?;
        let (lo, hi) = self.lattice.layer_range(i);
        let position = (short_rate.ln() - self.alphas[i]) / self.dx - lo as f64;
        let n = (hi - lo) as f64;
        if position <= 0.0 {
            return Ok(bonds[0]);
        }
        if position >= n {
            return Ok(bonds[bonds.len() - 1]);
        }
        let k = position.floor() as usize;
        let w = position - k as f64;
        Ok(bonds[k] * (1.0 - w) + bonds[k + 1] * w)
    }

    /// Exact Ornstein-Uhlenbeck step of `ln r - alpha(t)`.
    fn evolve(&self, t: f64, short_rate: f64, dt: f64, z: f64) -> Result<f64, RustyQLibError> {
        if !(dt > 0.0 && dt.is_finite() && t >= 0.0 && short_rate > 0.0) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "need t >= 0, dt > 0 and a positive rate, got t={t}, dt={dt}, r={short_rate}"
                ),
            ));
        }
        let x = short_rate.ln() - self.alpha_at(t);
        let decay = (-self.a * dt).exp();
        let std = (self.sigma * self.sigma * (1.0 - decay * decay) / (2.0 * self.a)).sqrt();
        Ok((x * decay + std * z + self.alpha_at(t + dt)).exp())
    }

    fn short_rate_floor(&self, _t: f64) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::engines::jamshidian::european_swaption_settled;
    use crate::rates::models::HullWhite;
    use chrono::NaiveDate;

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            NaiveDate::from_ymd_opt(2026, 8, 13).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn model(sigma: f64) -> BlackKarasinski {
        BlackKarasinski::new(0.1, sigma, market_curve(), 8.0, 50).unwrap()
    }

    fn fixed_leg() -> Vec<(f64, f64)> {
        (1..=5).map(|i| (1.0 + i as f64, 1.0)).collect()
    }

    #[test]
    fn the_fitted_tree_reproduces_the_curve_at_every_layer() {
        let m = model(0.2);
        for t in [0.02, 0.5, 1.0, 3.0, 6.0, 8.0] {
            let tree = m.tree_df(t).unwrap();
            let market = m.curve().df(t);
            assert!(
                (tree / market - 1.0).abs() < 1e-10,
                "t={t}: {tree} vs {market}"
            );
        }
        assert!((m.initial_short_rate() - 0.04).abs() < 2e-3);
        // rates are positive on every node
        for i in [0, 100, 400] {
            let (lo, hi) = m.lattice.layer_range(i);
            assert!(m.short_rate_at(i, lo) > 0.0 && m.short_rate_at(i, hi) > 0.0);
        }
    }

    #[test]
    fn bond_options_and_swaptions_keep_parity_on_the_tree() {
        let m = model(0.2);
        let curve = m.curve();
        let (expiry, maturity, strike) = (1.0, 5.0, 0.83);
        let call = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Call)
            .unwrap();
        let put = m
            .zero_bond_option(expiry, maturity, strike, PutOrCall::Put)
            .unwrap();
        let parity = curve.df(maturity) - strike * curve.df(expiry);
        assert!(
            (call - put - parity).abs() < 1e-9,
            "{call} - {put} vs {parity}"
        );
        assert!(call > 0.0 && put > 0.0);
        let leg = fixed_leg();
        let payer = m
            .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Payer)
            .unwrap();
        let receiver = m
            .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Receiver)
            .unwrap();
        let fixed: f64 = leg
            .iter()
            .map(|&(t, tau)| 0.045 * tau * curve.df(t))
            .sum::<f64>()
            + curve.df(6.0);
        let forward_swap = 1_000_000.0 * (curve.df(1.0) - fixed);
        assert!(
            (payer - receiver - forward_swap).abs() < 1e-3,
            "{payer} - {receiver} vs {forward_swap}"
        );
        // more vol, more value
        assert!(
            model(0.3)
                .european_swaption(1.0, 1.0, &leg, 0.045, 1_000_000.0, PayerReceiver::Payer)
                .unwrap()
                > payer
        );
    }

    #[test]
    fn lognormal_vol_near_the_rate_level_prices_like_hull_white() {
        // sigma_BK * r is the equivalent normal vol; with a small a and
        // short expiry the two models agree to a few percent
        let bk = model(0.25);
        let hw = HullWhite::new(0.1, 0.25 * 0.0405, market_curve()).unwrap();
        let leg = fixed_leg();
        let bk_price = bk
            .european_swaption(1.0, 1.0, &leg, 0.0455, 1.0, PayerReceiver::Payer)
            .unwrap();
        let hw_price =
            european_swaption_settled(&hw, 1.0, 1.0, &leg, 0.0455, 1.0, PayerReceiver::Payer)
                .unwrap();
        assert!(
            (bk_price - hw_price).abs() < 0.08 * hw_price,
            "BK {bk_price} vs HW {hw_price}"
        );
    }

    #[test]
    fn bermudan_dominates_the_europeans_and_reduces_to_one_with_one_date() {
        let m = model(0.2);
        let notional = 1_000_000.0;
        let tails: Vec<TailSwap> = (1..=3)
            .map(|k| {
                let expiry = k as f64;
                let coupons: Vec<(f64, f64)> = ((k + 1)..=6)
                    .map(|i| (i as f64, notional * 0.045))
                    .collect();
                TailSwap {
                    expiry,
                    start: expiry,
                    coupons,
                    last: 6.0,
                }
            })
            .collect();
        let single = m
            .bermudan_swaption(&tails[..1], notional, PayerReceiver::Payer)
            .unwrap();
        let european = m
            .european_swaption(
                1.0,
                1.0,
                &fixed_leg(),
                0.045,
                notional,
                PayerReceiver::Payer,
            )
            .unwrap();
        assert!((single - european).abs() < 1e-6, "{single} vs {european}");
        let bermudan = m
            .bermudan_swaption(&tails, notional, PayerReceiver::Payer)
            .unwrap();
        let best = tails
            .iter()
            .map(|t| {
                let leg: Vec<(f64, f64)> = t.coupons.iter().map(|&(time, _)| (time, 1.0)).collect();
                m.european_swaption(
                    t.expiry,
                    t.start,
                    &leg,
                    0.045,
                    notional,
                    PayerReceiver::Payer,
                )
                .unwrap()
            })
            .fold(f64::MIN, f64::max);
        assert!(bermudan > best, "{bermudan} vs {best}");
    }

    #[test]
    fn reconstituted_bonds_and_simulation_are_consistent_with_the_tree() {
        use rand::{Rng, SeedableRng};
        let m = model(0.2);
        // P(t, T | r) falls in r and is one at maturity
        let r = m.initial_short_rate();
        assert!(m.zero_bond(1.0, 4.0, r * 1.5).unwrap() < m.zero_bond(1.0, 4.0, r).unwrap());
        assert!((m.zero_bond(1.0, 1.0, r).unwrap() - 1.0).abs() < 1e-12);
        // simulated discounting lands near the curve (the lattice alpha is
        // a tree quantity, so a percent-level agreement is what to expect)
        let (horizon, steps, paths) = (3.0_f64, 150usize, 20_000usize);
        let dt = horizon / steps as f64;
        let mut rng = rand_pcg::Pcg64::seed_from_u64(9);
        let mut sum = 0.0;
        for _ in 0..paths {
            let mut rate = m.initial_short_rate();
            let mut integral = 0.0;
            for step in 0..steps {
                let z: f64 = rng.sample(rand_distr::StandardNormal);
                let next = m.evolve(step as f64 * dt, rate, dt, z).unwrap();
                integral += 0.5 * (rate + next) * dt;
                rate = next;
            }
            sum += (-integral).exp();
        }
        let mc = sum / paths as f64;
        let market = m.curve().df(horizon);
        assert!(
            (mc / market - 1.0).abs() < 0.01,
            "MC {mc} vs market {market}"
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(BlackKarasinski::new(0.0, 0.2, market_curve(), 5.0, 20).is_err());
        assert!(BlackKarasinski::new(0.1, 0.0, market_curve(), 5.0, 20).is_err());
        assert!(BlackKarasinski::new(0.1, 0.2, market_curve(), 5.0, 0).is_err());
        let m = BlackKarasinski::new(0.1, 0.2, market_curve(), 3.0, 20).unwrap();
        // beyond the horizon
        assert!(m.zero_bond_option(1.0, 5.0, 0.8, PutOrCall::Call).is_err());
        assert!(m.zero_bond(0.0, 4.0, 0.04).is_err());
        assert!(m.zero_bond(0.0, 2.0, -0.01).is_err());
        assert!(m.bermudan_swaption(&[], 1.0, PayerReceiver::Payer).is_err());
    }
}
