//! Convertible bonds under the Tsiveriotis-Fernandes model.
//!
//! **Why this lives in `bonds`**: a convertible is a corporate bond
//! with an embedded equity option, not an equity derivative with
//! coupons. It is quoted per 100 face with accrued interest, carries
//! the issuer's credit, and shares its call/put mechanics with the
//! straight corporates in this module — so it reuses [`FixedRateBond`]
//! wholesale (schedule, accrued, conventions) plus [`CallOption`] /
//! [`PutOption`]. The equity leg enters only as market inputs.
//!
//! Pricing is the Tsiveriotis-Fernandes (1998) split on a binomial
//! equity tree, the industry-standard single-factor treatment
//! (QuantLib's convertible engine): the node value is decomposed as
//! `V = E + B`, where `E` is the part that ends as shares (discounted
//! **risk-free** — delivering your own stock carries no default risk)
//! and `B` the part that ends as cash (discounted at **risk-free +
//! credit spread**). The tree uses CRR spacing with the risk-neutral
//! drift taken from the discount curve's own forward factors per step,
//! so a never-converted bond reprices the analytic risky bond exactly.
//!
//! Exercise logic per node, inside the conversion window:
//!
//! - issuer call (optionally gated by a **soft-call trigger** on the
//!   share price): the holder responds by converting when parity beats
//!   the call price — `V = max(call_dirty, ratio * S)` if that improves
//!   the issuer's position;
//! - holder put: `V = max(V, put_dirty)`, all cash;
//! - voluntary conversion: `V = max(V, ratio * S)`, all equity.

use chrono::NaiveDate;

use crate::bonds::{CallOption, FixedRateBond, PutOption};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;

/// Default number of tree steps for the pricing methods.
pub const DEFAULT_TREE_STEPS: usize = 800;

/// Equity and credit inputs for convertible pricing.
#[derive(Debug, Clone, Copy)]
pub struct ConvertibleMarket {
    /// Share price.
    pub spot: f64,
    /// Flat lognormal equity volatility.
    pub volatility: f64,
    /// Continuous dividend yield.
    pub dividend_yield: f64,
    /// Issuer credit spread applied to the cash-only part
    /// (continuously compounded, e.g. from the issuer's z-spread).
    pub credit_spread: f64,
}

impl ConvertibleMarket {
    fn validate(&self) -> Result<(), RustyQLibError> {
        if !(self.spot > 0.0 && self.spot.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("spot must be positive, got {}", self.spot),
            ));
        }
        if !(self.volatility > 0.0 && self.volatility.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("volatility must be positive, got {}", self.volatility),
            ));
        }
        if !self.dividend_yield.is_finite() || !self.credit_spread.is_finite() {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                "dividend yield and credit spread must be finite",
            ));
        }
        Ok(())
    }
}

/// A convertible bond: a straight bond chassis plus the conversion
/// right, an optional (soft-)call schedule and an optional put
/// schedule.
#[derive(Debug, Clone)]
pub struct ConvertibleBond {
    pub bond: FixedRateBond,
    /// Shares received on converting one bond of `bond.face_value`.
    pub conversion_ratio: f64,
    /// First date conversion is allowed (default: the dated date).
    pub convert_from: Option<NaiveDate>,
    /// Last date conversion is allowed (default: maturity).
    pub convert_until: Option<NaiveDate>,
    /// Issuer calls (dirty strike = price + accrued, as for straights).
    pub calls: Vec<CallOption>,
    /// Soft-call trigger: calls are exercisable only when the share
    /// price is at or above this level (`None` = hard calls).
    pub soft_call_trigger: Option<f64>,
    /// Holder puts.
    pub puts: Vec<PutOption>,
}

impl ConvertibleBond {
    pub fn new(bond: FixedRateBond, conversion_ratio: f64) -> Result<Self, RustyQLibError> {
        if !(conversion_ratio > 0.0 && conversion_ratio.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("conversion ratio must be positive, got {conversion_ratio}"),
            ));
        }
        Ok(ConvertibleBond {
            bond,
            conversion_ratio,
            convert_from: None,
            convert_until: None,
            calls: Vec::new(),
            soft_call_trigger: None,
            puts: Vec::new(),
        })
    }

    /// The share price at which conversion breaks even against face.
    pub fn conversion_price(&self) -> f64 {
        self.bond.face_value / self.conversion_ratio
    }

    /// Conversion (parity) value per 100 face at a share price.
    pub fn parity(&self, spot: f64) -> f64 {
        self.conversion_ratio * spot * 100.0 / self.bond.face_value
    }

    /// Conversion premium of a clean price over parity, as a fraction
    /// (`0.15` = 15% premium).
    pub fn conversion_premium(&self, clean_price: f64, spot: f64) -> Result<f64, RustyQLibError> {
        let parity = self.parity(spot);
        if parity <= 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "non-positive parity {parity}"
            )));
        }
        Ok(clean_price / parity - 1.0)
    }

    /// The straight-bond floor: the chassis priced at the credit
    /// spread, ignoring the conversion right.
    pub fn bond_floor(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.bond
            .clean_price_from_curve_with_spread(curve, market.credit_spread, settlement)
    }

    /// Dirty price per 100 face on a Tsiveriotis-Fernandes tree with
    /// `steps` time steps.
    pub fn dirty_price_with_steps(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
    ) -> Result<f64, RustyQLibError> {
        market.validate()?;
        if steps < 10 {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("the tree needs at least 10 steps, got {steps}"),
            ));
        }
        tf_tree_value(self, market, curve, settlement, steps)
    }

    /// Dirty price per 100 face with [`DEFAULT_TREE_STEPS`].
    pub fn dirty_price(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.dirty_price_with_steps(market, curve, settlement, DEFAULT_TREE_STEPS)
    }

    /// Clean price per 100 face.
    pub fn clean_price(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(
            self.dirty_price(market, curve, settlement)?
                - self.bond.accrued_interest(settlement)?,
        )
    }

    /// Equity delta: price change per 100 face per unit share move,
    /// from a symmetric 1% spot bump.
    pub fn delta(
        &self,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let bump = 0.01 * market.spot;
        let up = ConvertibleMarket {
            spot: market.spot + bump,
            ..*market
        };
        let down = ConvertibleMarket {
            spot: market.spot - bump,
            ..*market
        };
        let price_up = self.dirty_price(&up, curve, settlement)?;
        let price_down = self.dirty_price(&down, curve, settlement)?;
        Ok((price_up - price_down) / (2.0 * bump))
    }

    /// The credit spread implied by a quoted clean price, holding the
    /// equity inputs fixed.
    pub fn implied_credit_spread(
        &self,
        clean_price: f64,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !(clean_price > 0.0 && clean_price.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                "convertible",
                format!("clean price must be positive, got {clean_price}"),
            ));
        }
        // price is decreasing in the spread (only the cash part reacts):
        // target - price(s) is increasing
        let objective = |spread: f64| {
            let with_spread = ConvertibleMarket {
                credit_spread: spread,
                ..*market
            };
            clean_price
                - self
                    .clean_price(&with_spread, curve, settlement)
                    .expect("the spread bracket keeps the tree valid")
        };
        let probe = ConvertibleMarket {
            credit_spread: -0.2,
            ..*market
        };
        self.clean_price(&probe, curve, settlement)?;
        let root = Solver1d::new(1e-8, 100).bisection(objective, -0.2, 3.0)?;
        if !root.converged {
            return Err(RustyQLibError::CalibrationFailed {
                iterations: root.iterations,
                residual: objective(root.x).abs(),
                reason: "implied credit spread solve did not converge".to_string(),
            });
        }
        Ok(root.x)
    }
}

/// The Tsiveriotis-Fernandes backward induction. Returns the dirty
/// value per 100 of outstanding face at `settlement`.
fn tf_tree_value(
    convertible: &ConvertibleBond,
    market: &ConvertibleMarket,
    curve: &YieldCurve,
    settlement: NaiveDate,
    steps: usize,
) -> Result<f64, RustyQLibError> {
    let bond = &convertible.bond;
    // settlement validity as for every other bond pricer
    bond.accrued_interest(settlement)?;

    let year_fraction = |date: NaiveDate| {
        curve
            .day_count()
            .year_fraction(curve.reference_date(), date)
    };
    let cashflows: Vec<_> = bond
        .cashflows()
        .iter()
        .filter(|cf| cf.accrual_end > settlement)
        .cloned()
        .collect();
    let last = cashflows.last().ok_or_else(|| {
        RustyQLibError::invalid_input("convertible", "no cash flows after settlement")
    })?;

    let t0 = year_fraction(settlement);
    let horizon = year_fraction(last.payment_date);
    if horizon <= t0 {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            "the bond matures at settlement",
        ));
    }
    let dt = (horizon - t0) / steps as f64;
    let sqrt_dt = dt.sqrt();
    let up = (market.volatility * sqrt_dt).exp();
    let down = 1.0 / up;

    // per-step curve forwards, risky discounts and risk-neutral
    // probabilities
    let times: Vec<f64> = (0..=steps).map(|i| t0 + i as f64 * dt).collect();
    let spread_df = (-market.credit_spread * dt).exp();
    let mut riskfree_df = Vec::with_capacity(steps);
    let mut risky_df = Vec::with_capacity(steps);
    let mut probability = Vec::with_capacity(steps);
    for i in 0..steps {
        let df_step = curve.df(times[i + 1]) / curve.df(times[i]);
        let growth = (-market.dividend_yield * dt).exp() / df_step;
        let p = (growth - down) / (up - down);
        if !(0.0..=1.0).contains(&p) {
            return Err(RustyQLibError::NumericalError(format!(
                "risk-neutral probability {p} outside [0, 1] at step {i}; \
                 increase the tree steps or check the inputs"
            )));
        }
        riskfree_df.push(df_step);
        risky_df.push(df_step * spread_df);
        probability.push(p);
    }

    // coupons (ex the final flow) assigned to the step interval that
    // contains their payment time; forward rates discount them back to
    // the step's left edge
    let mut coupon_at_step: Vec<f64> = vec![0.0; steps];
    for cf in cashflows.iter().take(cashflows.len() - 1) {
        let time = year_fraction(cf.payment_date).clamp(t0, horizon);
        let index = (((time - t0) / dt).ceil() as usize).clamp(1, steps) - 1;
        let forward = -(riskfree_df[index].ln()) / dt;
        let discount_to_edge = (-(forward + market.credit_spread) * (time - times[index])).exp();
        coupon_at_step[index] += cf.amount * discount_to_edge;
    }

    // decision windows in step time
    let convert_from = year_fraction(convertible.convert_from.unwrap_or(bond.dated_date));
    let convert_until = year_fraction(convertible.convert_until.unwrap_or(bond.maturity_date));
    let mut call_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    for call in &convertible.calls {
        if call.call_date <= settlement || call.call_date >= bond.maturity_date {
            continue;
        }
        let outstanding = bond.outstanding_face(call.call_date);
        let strike = outstanding * call.call_price / 100.0
            + outstanding * bond.accrued_interest(call.call_date)? / 100.0;
        let index = (((year_fraction(call.call_date) - t0) / dt).round() as usize).min(steps);
        call_at_step[index] = Some(strike);
    }
    let mut put_at_step: Vec<Option<f64>> = vec![None; steps + 1];
    for put in &convertible.puts {
        if put.put_date <= settlement || put.put_date >= bond.maturity_date {
            continue;
        }
        let outstanding = bond.outstanding_face(put.put_date);
        let strike = outstanding * put.put_price / 100.0
            + outstanding * bond.accrued_interest(put.put_date)? / 100.0;
        let index = (((year_fraction(put.put_date) - t0) / dt).round() as usize).min(steps);
        put_at_step[index] = Some(strike);
    }

    let ratio = convertible.conversion_ratio;
    let spot_at =
        |step: usize, j: usize| market.spot * up.powi(j as i32) * down.powi((step - j) as i32);

    // terminal: convert against the full redemption package
    let redemption = last.amount;
    let mut equity: Vec<f64> = Vec::with_capacity(steps + 1);
    let mut cash: Vec<f64> = Vec::with_capacity(steps + 1);
    let can_convert_at_maturity = convert_until >= horizon - 1e-9;
    for j in 0..=steps {
        let shares = ratio * spot_at(steps, j);
        if can_convert_at_maturity && shares > redemption {
            equity.push(shares);
            cash.push(0.0);
        } else {
            equity.push(0.0);
            cash.push(redemption);
        }
    }

    // backward induction
    for step in (0..steps).rev() {
        let p = probability[step];
        let time = times[step];
        let in_window = time >= convert_from - 1e-9 && time <= convert_until + 1e-9;
        for j in 0..=step {
            let mut e = riskfree_df[step] * (p * equity[j + 1] + (1.0 - p) * equity[j]);
            let mut b = risky_df[step] * (p * cash[j + 1] + (1.0 - p) * cash[j]);
            b += coupon_at_step[step];
            let spot = spot_at(step, j);
            let shares = ratio * spot;

            // issuer call (soft trigger permitting): the holder answers
            // with conversion when parity beats the strike
            if let Some(strike) = call_at_step[step] {
                let triggered = convertible
                    .soft_call_trigger
                    .is_none_or(|trigger| spot >= trigger);
                if triggered {
                    let forced = if in_window {
                        strike.max(shares)
                    } else {
                        strike
                    };
                    if e + b > forced {
                        if in_window && shares > strike {
                            e = shares;
                            b = 0.0;
                        } else {
                            e = 0.0;
                            b = strike;
                        }
                    }
                }
            }
            // holder put: all cash
            if let Some(strike) = put_at_step[step] {
                if strike > e + b {
                    e = 0.0;
                    b = strike;
                }
            }
            // voluntary conversion: all equity
            if in_window && shares > e + b {
                e = shares;
                b = 0.0;
            }
            equity[j] = e;
            cash[j] = b;
        }
    }

    Ok((equity[0] + cash[0]) * 100.0 / bond.outstanding_face(settlement))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;
    use crate::core::utils::norm_cdf;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn flat(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(2026, 8, 13),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// A 2% five-year convertible on a 1000 face, 20 shares per bond
    /// (conversion price 50).
    fn convertible() -> ConvertibleBond {
        let bond =
            FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        ConvertibleBond::new(bond, 20.0).unwrap()
    }

    fn market(spot: f64) -> ConvertibleMarket {
        ConvertibleMarket {
            spot,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.02,
        }
    }

    #[test]
    fn deep_out_of_the_money_collapses_to_the_risky_straight_bond() {
        // with the shares nearly worthless the cash part evolves
        // deterministically, so the tree must reproduce the analytic
        // risky bond almost exactly
        let cv = convertible();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let market = market(0.01);
        let tree = cv.clean_price(&market, &curve, settlement).unwrap();
        let straight = cv.bond_floor(&market, &curve, settlement).unwrap();
        assert!((tree - straight).abs() < 1e-6, "{tree} vs {straight}");
    }

    #[test]
    fn deep_in_the_money_trades_at_parity() {
        let cv = convertible();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let spot = 250.0; // parity 500 vs redemption 100
        let dirty = cv.dirty_price(&market(spot), &curve, settlement).unwrap();
        let parity = cv.parity(spot);
        assert!(
            (dirty - parity) / parity < 0.01,
            "dirty {dirty} vs parity {parity}"
        );
        assert!(dirty >= parity - 1e-9, "conversion floor violated");
        // delta approaches the full ratio per 100 face
        let delta = cv.delta(&market(spot), &curve, settlement).unwrap();
        let full = cv.conversion_ratio * 100.0 / cv.bond.face_value;
        assert!(
            (delta - full).abs() < 0.05 * full,
            "delta {delta} vs {full}"
        );
    }

    #[test]
    fn maturity_only_conversion_is_bond_plus_european_call() {
        // restrict conversion to maturity with zero credit spread: the
        // convertible = risk-free straight bond + ratio European calls
        // struck at redemption/ratio (Black-Scholes anchor)
        let bond =
            FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        let mut cv = ConvertibleBond::new(bond, 20.0).unwrap();
        cv.convert_from = Some(d(2031, 5, 14));
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = ConvertibleMarket {
            spot: 48.0,
            volatility: 0.30,
            dividend_yield: 0.01,
            credit_spread: 0.0,
        };
        let tree = cv
            .dirty_price_with_steps(&m, &curve, settlement, 1600)
            .unwrap();

        // Black-Scholes call on the terminal package: strike is the full
        // redemption (face + final coupon) per share
        let dc = curve.day_count();
        let last = cv.bond.cashflows().last().unwrap().clone();
        let t = dc.year_fraction(settlement, last.payment_date);
        let strike = last.amount / cv.conversion_ratio;
        let df = curve.df_date(last.payment_date) / curve.df_date(settlement);
        let forward = m.spot * (-m.dividend_yield * t).exp() / df;
        let sd = m.volatility * t.sqrt();
        let d1 = ((forward / strike).ln() + 0.5 * sd * sd) / sd;
        let call = df * (forward * norm_cdf(d1) - strike * norm_cdf(d1 - sd));
        let straight = cv.bond.dirty_price_from_curve(&curve, settlement).unwrap();
        let expected = straight + cv.conversion_ratio * call * 100.0 / cv.bond.face_value;
        assert!((tree - expected).abs() < 0.15, "{tree} vs {expected}");
    }

    #[test]
    fn price_sits_above_both_floors_and_orders_in_vol_and_spread() {
        let cv = convertible();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        for spot in [30.0, 45.0, 55.0, 70.0] {
            let m = market(spot);
            let clean = cv.clean_price(&m, &curve, settlement).unwrap();
            let floor = cv.bond_floor(&m, &curve, settlement).unwrap();
            let parity = cv.parity(spot);
            assert!(
                clean >= floor - 0.05,
                "spot {spot}: {clean} vs floor {floor}"
            );
            assert!(
                clean >= parity - cv.bond.accrued_interest(settlement).unwrap() - 0.05,
                "spot {spot}: {clean} vs parity {parity}"
            );
        }
        // vega and credit ordering at the money
        let base = cv.clean_price(&market(48.0), &curve, settlement).unwrap();
        let hot = ConvertibleMarket {
            volatility: 0.45,
            ..market(48.0)
        };
        assert!(cv.clean_price(&hot, &curve, settlement).unwrap() > base);
        let wide = ConvertibleMarket {
            credit_spread: 0.05,
            ..market(48.0)
        };
        assert!(cv.clean_price(&wide, &curve, settlement).unwrap() < base);
    }

    #[test]
    fn calls_cap_puts_floor_and_the_soft_trigger_softens() {
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(48.0);
        let base = convertible();
        let free = base.clean_price(&m, &curve, settlement).unwrap();

        let mut hard_called = base.clone();
        hard_called.calls = vec![CallOption {
            call_date: d(2028, 5, 15),
            call_price: 102.0,
        }];
        let called = hard_called.clean_price(&m, &curve, settlement).unwrap();
        assert!(called < free, "{called} vs {free}");

        // a high soft-call trigger makes the call harder to exercise:
        // price between hard-called and call-free
        let mut soft_called = hard_called.clone();
        soft_called.soft_call_trigger = Some(65.0); // 130% of conversion price
        let softened = soft_called.clean_price(&m, &curve, settlement).unwrap();
        assert!(
            called < softened && softened <= free + 1e-9,
            "{called} < {softened} <= {free}"
        );

        let mut puttable = base.clone();
        puttable.puts = vec![PutOption {
            put_date: d(2029, 5, 15),
            put_price: 100.0,
        }];
        let put_price = puttable.clean_price(&m, &curve, settlement).unwrap();
        assert!(put_price > free, "{put_price} vs {free}");
    }

    #[test]
    fn implied_credit_spread_round_trips() {
        let cv = convertible();
        let curve = flat(0.04);
        let settlement = d(2026, 8, 14);
        let m = market(48.0);
        let clean = cv.clean_price(&m, &curve, settlement).unwrap();
        let implied = cv
            .implied_credit_spread(clean, &m, &curve, settlement)
            .unwrap();
        assert!(
            (implied - m.credit_spread).abs() < 1e-5,
            "implied {implied}"
        );
    }

    #[test]
    fn parity_premium_and_validation() {
        let cv = convertible();
        assert!((cv.conversion_price() - 50.0).abs() < 1e-12);
        assert!((cv.parity(48.0) - 96.0).abs() < 1e-12);
        let premium = cv.conversion_premium(105.6, 48.0).unwrap();
        assert!((premium - 0.10).abs() < 1e-12);
        // validation
        let bond =
            FixedRateBond::us_corporate(1000.0, 0.02, d(2026, 5, 15), d(2031, 5, 15)).unwrap();
        assert!(ConvertibleBond::new(bond.clone(), 0.0).is_err());
        let cv = ConvertibleBond::new(bond, 20.0).unwrap();
        let curve = flat(0.04);
        let bad = ConvertibleMarket {
            spot: -1.0,
            volatility: 0.3,
            dividend_yield: 0.0,
            credit_spread: 0.0,
        };
        assert!(cv.dirty_price(&bad, &curve, d(2026, 8, 14)).is_err());
        let m = market(48.0);
        assert!(cv
            .dirty_price_with_steps(&m, &curve, d(2026, 8, 14), 5)
            .is_err());
        assert!(cv
            .implied_credit_spread(-10.0, &m, &curve, d(2026, 8, 14))
            .is_err());
    }
}
