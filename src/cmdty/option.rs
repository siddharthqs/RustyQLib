//! European option on a commodity future, priced with Black-76.
//!
//! Commodity options overwhelmingly trade as options on the future (CME
//! LO on WTI, OG on gold, ON on natural gas, ...), so the natural model
//! input is the futures price itself — read off the
//! [`CommodityForwardCurve`] at the underlying's delivery date — and the
//! natural pricer is Black-76, which the equity module already provides
//! ([`crate::equity::black76`]). This contract is the date-based wrapper:
//! it resolves `F` from the forward curve, the discount rate and time to
//! expiry from a [`YieldCurve`], and quotes premium and Greeks per
//! contract (`quantity` units).
//!
//! Both settlement styles are supported through
//! [`FuturesSettlement`]: `Discounted` (up-front premium, the standard
//! Black-76) and `Margined` (futures-style, undiscounted).
//!
//! The distribution model travels with the vol quote
//! ([`CommodityVol`]): a bare `f64` vol prices under Black-76, a
//! [`CommodityVol::ShiftedLognormal`] quote under displaced Black-76
//! (the same kernel on `F + shift`, `K + shift`), and a
//! [`CommodityVol::Normal`] quote under [`bachelier`] —
//! the route for underlyings that can print negative (Waha or AECO
//! basis, WTI in an April-2020 dislocation).

use chrono::NaiveDate;

use crate::cmdty::bachelier;
use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::cmdty::vol::CommodityVol;
use crate::cmdty::{expiry_inputs, ExpiryInputs};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::results::Greeks;
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::equity::black76;

pub use crate::equity::black76::FuturesSettlement;

const FIELD: &str = "commodity option";

/// A European option on a commodity future. Premium and Greeks are
/// quoted for the whole contract: `quantity` units of the index
/// (a CME WTI option is 1,000 bbl, a gold option 100 oz).
#[derive(Debug, Clone)]
pub struct CommodityOption {
    /// Contract size in the index's units.
    pub quantity: f64,
    /// Strike price per unit.
    pub strike: f64,
    pub put_or_call: PutOrCall,
    /// Option expiry (last exercise date).
    pub expiry_date: NaiveDate,
    /// Delivery/pricing date of the underlying future — where the
    /// forward curve is read. Commodity options expire shortly before
    /// their future, so this is usually a few days after `expiry_date`.
    pub underlying_date: NaiveDate,
    pub settlement: FuturesSettlement,
}

impl CommodityOption {
    pub fn new(
        quantity: f64,
        strike: f64,
        put_or_call: PutOrCall,
        expiry_date: NaiveDate,
        underlying_date: NaiveDate,
        settlement: FuturesSettlement,
    ) -> Result<Self, RustyQLibError> {
        if !quantity.is_finite() || quantity <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("quantity must be positive, got {quantity}"),
            ));
        }
        if !strike.is_finite() {
            // sign is a model question: Black-76 needs K > 0 (checked at
            // pricing), Bachelier takes any strike (basis can be negative)
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("strike must be finite, got {strike}"),
            ));
        }
        if underlying_date < expiry_date {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("underlying date {underlying_date} must not precede expiry {expiry_date}"),
            ));
        }
        Ok(CommodityOption {
            quantity,
            strike,
            put_or_call,
            expiry_date,
            underlying_date,
            settlement,
        })
    }

    /// The underlying futures price: the forward curve read at the
    /// underlying's delivery date (any sign — whether a model can price
    /// it is checked when pricing).
    pub fn forward_price(&self, forward: &CommodityForwardCurve) -> f64 {
        forward.price(self.underlying_date)
    }

    /// Premium for the whole contract. A bare `f64` vol prices under
    /// Black-76; pass a [`CommodityVol`] to select the shifted or
    /// normal model.
    pub fn price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: impl Into<CommodityVol>,
    ) -> Result<f64, RustyQLibError> {
        let quote = vol.into().validated("commodity option")?;
        let (f, m) = self.market_inputs(discount, forward)?;
        let (r, t) = (m.r, m.t);
        let (pc, s) = (self.put_or_call, self.settlement);
        Ok(self.quantity
            * match quote {
                CommodityVol::Lognormal(v) => {
                    let (f, k) = self.displaced_inputs(f, 0.0)?;
                    black76::price(f, k, r, v, t, pc, s)
                }
                CommodityVol::ShiftedLognormal { vol, shift } => {
                    let (f, k) = self.displaced_inputs(f, shift)?;
                    black76::price(f, k, r, vol, t, pc, s)
                }
                CommodityVol::Normal(v) => bachelier::price(f, self.strike, r, v, t, pc, s),
            })
    }

    /// All sensitivities under the quote's model, scaled to the
    /// contract. Delta and gamma are with respect to the futures price,
    /// vega per unit of the quote's vol, theta per year of calendar
    /// time, rho per unit of rate. (Under the shifted model, delta and
    /// gamma with respect to `F` equal the Black Greeks on `F + shift`;
    /// `gamma_p` is the elasticity of the displaced underlying.)
    pub fn greeks(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: impl Into<CommodityVol>,
    ) -> Result<Greeks, RustyQLibError> {
        let quote = vol.into().validated("commodity option")?;
        let (f, m) = self.market_inputs(discount, forward)?;
        let (r, t) = (m.r, m.t);
        let (pc, s, q) = (self.put_or_call, self.settlement, self.quantity);
        // with no time or no diffusion left the kernels' `d` is 0/0, so
        // the whole dispatch collapses to one model-free branch — but
        // the lognormal quotes still refuse a non-positive displaced
        // market first, exactly as `price` does
        if t <= 0.0 || quote.vol() <= 0.0 {
            match quote {
                CommodityVol::Lognormal(_) => {
                    self.displaced_inputs(f, 0.0)?;
                }
                CommodityVol::ShiftedLognormal { shift, .. } => {
                    self.displaced_inputs(f, shift)?;
                }
                CommodityVol::Normal(_) => {}
            }
            return self.degenerate_greeks(f, m);
        }
        Ok(match quote {
            CommodityVol::Lognormal(vol) | CommodityVol::ShiftedLognormal { vol, .. } => {
                let shift = match quote {
                    CommodityVol::ShiftedLognormal { shift, .. } => shift,
                    _ => 0.0,
                };
                let (f, k) = self.displaced_inputs(f, shift)?;
                Greeks {
                    delta: q * black76::delta(f, k, r, vol, t, pc, s),
                    gamma: q * black76::gamma(f, k, r, vol, t, s),
                    vega: q * black76::vega(f, k, r, vol, t, s),
                    theta: q * black76::theta(f, k, r, vol, t, pc, s),
                    rho: q * black76::rho(f, k, r, vol, t, pc, s),
                    vanna: q * black76::vanna(f, k, r, vol, t, s),
                    charm: q * black76::charm(f, k, r, vol, t, pc, s),
                    // elasticity is scale-free: quantity cancels
                    gamma_p: black76::gamma_p(f, k, r, vol, t, pc, s),
                    zomma: q * black76::zomma(f, k, r, vol, t, s),
                }
            }
            CommodityVol::Normal(vol) => {
                let k = self.strike;
                Greeks {
                    delta: q * bachelier::delta(f, k, r, vol, t, pc, s),
                    gamma: q * bachelier::gamma(f, k, r, vol, t, s),
                    vega: q * bachelier::vega(f, k, r, vol, t, s),
                    theta: q * bachelier::theta(f, k, r, vol, t, pc, s),
                    rho: q * bachelier::rho(f, k, r, vol, t, pc, s),
                    vanna: q * bachelier::vanna(f, k, r, vol, t, s),
                    charm: q * bachelier::charm(f, k, r, vol, t, pc, s),
                    gamma_p: bachelier::gamma_p(f, k, r, vol, t, pc, s),
                    zomma: q * bachelier::zomma(f, k, r, vol, t, s),
                }
            }
        })
    }

    /// The Greeks of an option with no diffusion left — expiring today
    /// (`t = 0`) or quoted at zero vol — where the kernels' `d` is
    /// `0/0`. The contract is then its (discounted) intrinsic, so the
    /// second-order Greeks vanish and delta is the payoff's own slope;
    /// this is the same convention [`bachelier`] applies, and it is
    /// model-free, so the branch is shared by all three quotes.
    ///
    /// Exactly at the money the payoff has a kink: the one-sided deltas
    /// are 1 and 0, and no single value is defensible, so that case is
    /// an error rather than a silent half.
    fn degenerate_greeks(&self, f: f64, m: ExpiryInputs) -> Result<Greeks, RustyQLibError> {
        let ExpiryInputs { t, r, .. } = m;
        let df = match self.settlement {
            // the curve's own discount factor to expiry, which the
            // kernels reproduce as e^{-rt}
            FuturesSettlement::Discounted => m.df,
            FuturesSettlement::Margined => 1.0,
        };
        let (k, q) = (self.strike, self.quantity);
        let intrinsic = match self.put_or_call {
            PutOrCall::Call => (f - k).max(0.0),
            PutOrCall::Put => (k - f).max(0.0),
        };
        if f == k {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "delta is undefined for a struck-at-the-money option with no \
                     diffusion left (forward {f} = strike {k}, t = {t}): the payoff \
                     kinks there, so the one-sided deltas are 1 and 0"
                ),
            ));
        }
        let delta = match self.put_or_call {
            PutOrCall::Call => {
                if f > k {
                    1.0
                } else {
                    0.0
                }
            }
            PutOrCall::Put => {
                if f < k {
                    -1.0
                } else {
                    0.0
                }
            }
        };
        Ok(Greeks {
            delta: q * df * delta,
            gamma: 0.0,
            vega: 0.0,
            // only the discount factor on the intrinsic still decays
            theta: match self.settlement {
                FuturesSettlement::Discounted => q * r * df * intrinsic,
                FuturesSettlement::Margined => 0.0,
            },
            // and the premium's own rate sensitivity is that discounting
            rho: match self.settlement {
                FuturesSettlement::Discounted => -t * q * df * intrinsic,
                FuturesSettlement::Margined => 0.0,
            },
            vanna: 0.0,
            charm: 0.0,
            gamma_p: 0.0,
            zomma: 0.0,
        })
    }

    /// The flat Black-76 volatility that reproduces `premium` (for the
    /// whole contract), by bisection. Errors when the premium sits
    /// outside the no-arbitrage bounds.
    pub fn implied_vol(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        premium: f64,
    ) -> Result<f64, RustyQLibError> {
        self.invert_premium(premium, 10.0, |v| {
            self.price(discount, forward, CommodityVol::Lognormal(v))
        })
    }

    /// The shifted Black-76 volatility (for the given `shift`) that
    /// reproduces `premium`, by bisection.
    pub fn implied_vol_shifted(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        premium: f64,
        shift: f64,
    ) -> Result<f64, RustyQLibError> {
        self.invert_premium(premium, 10.0, |v| {
            self.price(
                discount,
                forward,
                CommodityVol::ShiftedLognormal { vol: v, shift },
            )
        })
    }

    /// The Bachelier (normal) volatility that reproduces `premium`, by
    /// bisection. Quoted in price units per √year.
    pub fn implied_vol_normal(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        premium: f64,
    ) -> Result<f64, RustyQLibError> {
        // a normal vol has price units: scale the bracket to the market
        let f = self.forward_price(forward);
        let hi = 100.0 * (f.abs() + self.strike.abs() + 1.0);
        self.invert_premium(premium, hi, |v| {
            self.price(discount, forward, CommodityVol::Normal(v))
        })
    }

    /// Shared bisection on a monotone premium-in-vol function over
    /// `[~0, hi]`.
    fn invert_premium(
        &self,
        premium: f64,
        hi: f64,
        price_at: impl Fn(f64) -> Result<f64, RustyQLibError>,
    ) -> Result<f64, RustyQLibError> {
        if !premium.is_finite() || premium < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium must be non-negative, got {premium}"),
            ));
        }
        let lo = 1e-9;
        if price_at(lo)? > premium + 1e-12 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium {premium} is below intrinsic value"),
            ));
        }
        if price_at(hi)? < premium {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium {premium} exceeds the vol={hi} price"),
            ));
        }
        // premium is monotone in vol and the bracket straddles the root:
        // the shared bisection converges on it. A quote the solver only
        // bracketed (rather than hit to tolerance) is still the best
        // estimate of the vol, so the root is returned either way.
        let root = Solver1d::new(1e-12, 200).bisection(
            |v| price_at(v).unwrap_or(f64::NAN) - premium,
            lo,
            hi,
        )?;
        Ok(root.x)
    }

    /// Shift and validate `(F, K)` for the (displaced) Black-76 kernel,
    /// which is lognormal in `F + shift`.
    fn displaced_inputs(&self, f: f64, shift: f64) -> Result<(f64, f64), RustyQLibError> {
        let (fs, ks) = (f + shift, self.strike + shift);
        if fs <= 0.0 || ks <= 0.0 {
            let model = if shift == 0.0 {
                "Black-76".to_string()
            } else {
                format!("shift {shift}")
            };
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!(
                    "forward {f} / strike {} not priceable under {model} \
                     (shifted values must be positive; use a larger shift \
                     or a normal vol)",
                    self.strike
                ),
            ));
        }
        Ok((fs, ks))
    }

    /// Resolve `(F, r, t)`: the futures price off the forward curve, the
    /// Act/365 vol time to expiry, and the continuous rate reproducing
    /// the curve's discount factor over that time (so `e^{-rt}` is
    /// exactly `discount.df_date(expiry)` — see the [`crate::cmdty`]
    /// conventions).
    fn market_inputs(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
    ) -> Result<(f64, ExpiryInputs), RustyQLibError> {
        let inputs = expiry_inputs(FIELD, self.expiry_date, discount)?;
        Ok((self.forward_price(forward), inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::Compounding;
    use crate::core::daycount::DayCountConvention;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    // one Act/365 year from the curve reference, so t = 1.0 exactly
    const REF: (i32, u32, u32) = (2026, 9, 1);
    const EXPIRY: (i32, u32, u32) = (2027, 9, 1);

    fn flat_discount(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            d(REF.0, REF.1, REF.2),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    fn call(strike: f64, settlement: FuturesSettlement) -> CommodityOption {
        CommodityOption::new(
            1_000.0,
            strike,
            PutOrCall::Call,
            d(EXPIRY.0, EXPIRY.1, EXPIRY.2),
            d(2027, 9, 20), // the future delivers after option expiry
            settlement,
        )
        .unwrap()
    }

    #[test]
    fn reproduces_the_black76_golden_value() {
        // same inputs as the engine's golden test: F=K=100, r=5%, vol=30%, t=1
        let discount = flat_discount(0.05);
        let forward = CommodityForwardCurve::flat(100.0, d(REF.0, REF.1, REF.2)).unwrap();
        let option = call(100.0, FuturesSettlement::Discounted);
        let price = option.price(&discount, &forward, 0.30).unwrap();
        assert!((price - 1_000.0 * 11.34202064).abs() < 1e-3, "{price}");
        // margined premium is the same number undiscounted
        let margined = call(100.0, FuturesSettlement::Margined)
            .price(&discount, &forward, 0.30)
            .unwrap();
        assert!((margined - price * (0.05f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn put_call_parity_off_the_curves() {
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, d(REF.0, REF.1, REF.2)).unwrap();
        let c = call(70.0, FuturesSettlement::Discounted);
        let mut p = c.clone();
        p.put_or_call = PutOrCall::Put;
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let parity = 1_000.0 * df * (72.0 - 70.0);
        let diff = c.price(&discount, &forward, 0.35).unwrap()
            - p.price(&discount, &forward, 0.35).unwrap();
        assert!((diff - parity).abs() < 1e-8, "{diff} vs {parity}");
    }

    #[test]
    fn underlying_date_reads_the_right_forward_pillar() {
        let reference = d(REF.0, REF.1, REF.2);
        let discount = flat_discount(0.04);
        // steep contango: the underlying's delivery date matters
        let forward = CommodityForwardCurve::from_prices(
            reference,
            vec![
                (d(EXPIRY.0, EXPIRY.1, EXPIRY.2), 70.0),
                (d(2027, 9, 20), 80.0),
            ],
        )
        .unwrap();
        let option = call(70.0, FuturesSettlement::Discounted);
        assert_eq!(option.forward_price(&forward), 80.0);
        // an option reading F=80 must be worth more than one reading F=70
        let mut at_expiry = option.clone();
        at_expiry.underlying_date = d(EXPIRY.0, EXPIRY.1, EXPIRY.2);
        let far = option.price(&discount, &forward, 0.30).unwrap();
        let near = at_expiry.price(&discount, &forward, 0.30).unwrap();
        assert!(far > near, "{far} vs {near}");
    }

    #[test]
    fn greeks_match_curve_bumps() {
        let reference = d(REF.0, REF.1, REF.2);
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, reference).unwrap();
        let option = call(70.0, FuturesSettlement::Discounted);
        let greeks = option.greeks(&discount, &forward, 0.35).unwrap();
        // delta against a central forward-curve bump
        let h = 1e-4;
        let up = option
            .price(&discount, &forward.bumped(h).unwrap(), 0.35)
            .unwrap();
        let down = option
            .price(&discount, &forward.bumped(-h).unwrap(), 0.35)
            .unwrap();
        let bumped_delta = (up - down) / (2.0 * h);
        assert!((greeks.delta - bumped_delta).abs() < 1e-3);
        // vega against a vol bump
        let v_up = option.price(&discount, &forward, 0.35 + h).unwrap();
        let v_down = option.price(&discount, &forward, 0.35 - h).unwrap();
        assert!((greeks.vega - (v_up - v_down) / (2.0 * h)).abs() < 1e-2);
        // an in-the-money call: positive delta, gamma and vega
        assert!(greeks.delta > 0.0 && greeks.gamma > 0.0 && greeks.vega > 0.0);
        // discounted premium decays with rates
        assert!(greeks.rho < 0.0);
    }

    #[test]
    fn implied_vol_round_trips() {
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, d(REF.0, REF.1, REF.2)).unwrap();
        for settlement in [FuturesSettlement::Discounted, FuturesSettlement::Margined] {
            let option = call(75.0, settlement);
            let premium = option.price(&discount, &forward, 0.42).unwrap();
            let vol = option.implied_vol(&discount, &forward, premium).unwrap();
            assert!((vol - 0.42).abs() < 1e-8, "{settlement:?}: {vol}");
        }
        // below intrinsic and absurdly high premia are rejected
        let option = call(60.0, FuturesSettlement::Discounted);
        assert!(option.implied_vol(&discount, &forward, 0.0).is_err());
        assert!(option
            .implied_vol(&discount, &forward, 1_000.0 * 73.0)
            .is_err());
    }

    #[test]
    fn expired_and_degenerate_inputs_error() {
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, d(REF.0, REF.1, REF.2)).unwrap();
        // expiry before the valuation date
        let expired = CommodityOption::new(
            1_000.0,
            70.0,
            PutOrCall::Call,
            d(2026, 8, 1),
            d(2026, 8, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        assert!(expired.price(&discount, &forward, 0.3).is_err());
        // negative forward prices cannot go through Black-76
        let negative = CommodityForwardCurve::flat(-37.63, d(REF.0, REF.1, REF.2)).unwrap();
        let option = call(70.0, FuturesSettlement::Discounted);
        assert!(option.price(&discount, &negative, 0.3).is_err());
        // negative vol
        assert!(option.price(&discount, &forward, -0.1).is_err());
    }

    #[test]
    fn expiring_today_is_worth_intrinsic() {
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, d(REF.0, REF.1, REF.2)).unwrap();
        let option = CommodityOption::new(
            1_000.0,
            70.0,
            PutOrCall::Call,
            d(REF.0, REF.1, REF.2),
            d(REF.0, REF.1, REF.2),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        let price = option.price(&discount, &forward, 0.3).unwrap();
        assert!((price - 1_000.0 * 2.0).abs() < 1e-10, "{price}");
    }

    #[test]
    fn degenerate_greeks_are_finite_for_both_quote_kinds() {
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(72.0, reference).unwrap();
        // (a) expiring today, (b) quoted at zero vol
        let expiring = CommodityOption::new(
            1_000.0,
            70.0,
            PutOrCall::Call,
            reference,
            reference,
            FuturesSettlement::Discounted,
        )
        .unwrap();
        let year_out = call(70.0, FuturesSettlement::Discounted);
        for (option, quote, t) in [
            (&expiring, CommodityVol::Lognormal(0.35), 0.0),
            (&expiring, CommodityVol::Normal(20.0), 0.0),
            (&year_out, CommodityVol::Lognormal(0.0), 1.0),
            (&year_out, CommodityVol::Normal(0.0), 1.0),
            (
                &year_out,
                CommodityVol::ShiftedLognormal {
                    vol: 0.0,
                    shift: 10.0,
                },
                1.0,
            ),
        ] {
            let g = option.greeks(&discount, &forward, quote).unwrap();
            for value in [
                g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.charm, g.gamma_p, g.zomma,
            ] {
                assert!(value.is_finite(), "{quote:?} at t={t}: {value}");
            }
            // an in-the-money call: full discounted delta, no convexity
            let df = discount.df_date(option.expiry_date);
            assert!((g.delta - 1_000.0 * df).abs() < 1e-12, "{}", g.delta);
            assert_eq!(
                [g.gamma, g.vega, g.vanna, g.charm, g.zomma, g.gamma_p],
                [0.0; 6]
            );
            // rho and theta are the decay of the discounting on the
            // intrinsic — and at t = 0 the effective rate is zero, so
            // both vanish
            let intrinsic = 1_000.0 * df * 2.0;
            let r = if t > 0.0 { -df.ln() / t } else { 0.0 };
            assert!((g.rho + t * intrinsic).abs() < 1e-10, "{}", g.rho);
            assert!((g.theta - r * intrinsic).abs() < 1e-10, "{}", g.theta);
        }
    }

    #[test]
    fn degenerate_greeks_follow_the_settlement_style() {
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(72.0, reference).unwrap();
        // margined: no discounting, so no rate sensitivity and no decay
        let margined = call(70.0, FuturesSettlement::Margined);
        let g = margined.greeks(&discount, &forward, 0.0).unwrap();
        assert_eq!(g.delta, 1_000.0);
        assert_eq!(g.rho, 0.0);
        assert_eq!(g.theta, 0.0);
        // out of the money: no delta either way
        let otm = call(80.0, FuturesSettlement::Discounted);
        let g_otm = otm.greeks(&discount, &forward, 0.0).unwrap();
        assert_eq!(g_otm.delta, 0.0);
        assert_eq!(g_otm.rho, 0.0);
        assert_eq!(g_otm.theta, 0.0);
        // a put is short the underlying
        let mut put = call(80.0, FuturesSettlement::Discounted);
        put.put_or_call = PutOrCall::Put;
        let df = discount.df_date(put.expiry_date);
        let g_put = put.greeks(&discount, &forward, 0.0).unwrap();
        assert!(
            (g_put.delta + 1_000.0 * df).abs() < 1e-12,
            "{}",
            g_put.delta
        );
        // struck exactly at the money the payoff kinks: an error, not a
        // silent half-delta
        assert!(call(72.0, FuturesSettlement::Discounted)
            .greeks(&discount, &forward, 0.0)
            .is_err());
        // and the lognormal branch still refuses a negative forward
        let negative = CommodityForwardCurve::flat(-37.63, reference).unwrap();
        assert!(call(70.0, FuturesSettlement::Discounted)
            .greeks(&discount, &negative, 0.0)
            .is_err());
    }

    #[test]
    fn degenerate_greeks_are_the_limit_of_the_live_ones() {
        // shrinking the vol walks the live Greeks onto the degenerate
        // branch, so the guard is continuous rather than a special case
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(72.0, d(REF.0, REF.1, REF.2)).unwrap();
        let option = call(70.0, FuturesSettlement::Discounted);
        let limit = option.greeks(&discount, &forward, 0.0).unwrap();
        let near = option.greeks(&discount, &forward, 1e-6).unwrap();
        assert!((limit.delta - near.delta).abs() < 1e-6, "{}", near.delta);
        assert!((limit.rho - near.rho).abs() < 1e-6, "{}", near.rho);
        assert!(near.vega.abs() < 1e-6, "{}", near.vega);
    }

    #[test]
    fn validation_rejects_bad_contracts() {
        let expiry = d(EXPIRY.0, EXPIRY.1, EXPIRY.2);
        let s = FuturesSettlement::Discounted;
        assert!(CommodityOption::new(0.0, 70.0, PutOrCall::Call, expiry, expiry, s).is_err());
        assert!(
            CommodityOption::new(1_000.0, f64::NAN, PutOrCall::Call, expiry, expiry, s).is_err()
        );
        // a negative strike is a legal contract (basis options); only the
        // lognormal model refuses to price it
        let negative_strike =
            CommodityOption::new(1_000.0, -1.0, PutOrCall::Call, expiry, expiry, s).unwrap();
        let discount = flat_discount(0.04);
        let forward = CommodityForwardCurve::flat(2.0, d(REF.0, REF.1, REF.2)).unwrap();
        assert!(negative_strike.price(&discount, &forward, 0.3).is_err());
        assert!(negative_strike
            .price(&discount, &forward, CommodityVol::Normal(1.5))
            .is_ok());
        // underlying before expiry makes no sense
        assert!(
            CommodityOption::new(1_000.0, 70.0, PutOrCall::Call, expiry, d(2027, 8, 1), s).is_err()
        );
    }

    #[test]
    fn shifted_model_is_black76_on_displaced_market() {
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(72.0, reference).unwrap();
        let option = call(70.0, FuturesSettlement::Discounted);
        // shift 0 collapses to the plain lognormal quote
        let plain = option.price(&discount, &forward, 0.35).unwrap();
        let shifted0 = option
            .price(
                &discount,
                &forward,
                CommodityVol::ShiftedLognormal {
                    vol: 0.35,
                    shift: 0.0,
                },
            )
            .unwrap();
        assert!((plain - shifted0).abs() < 1e-12);
        // shift s equals a plain option on the market displaced by s
        let shift = 10.0;
        let displaced_curve = CommodityForwardCurve::flat(82.0, reference).unwrap();
        let displaced_option = call(80.0, FuturesSettlement::Discounted);
        let via_shift = option
            .price(
                &discount,
                &forward,
                CommodityVol::ShiftedLognormal { vol: 0.35, shift },
            )
            .unwrap();
        let via_displacement = displaced_option
            .price(&discount, &displaced_curve, 0.35)
            .unwrap();
        assert!((via_shift - via_displacement).abs() < 1e-10);
    }

    #[test]
    fn shifted_model_prices_a_negative_forward() {
        // Waha-style: forward at -1.50, strike 0.50, shift 10 keeps the
        // displaced market lognormal
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(-1.50, reference).unwrap();
        let option = CommodityOption::new(
            10_000.0,
            0.50,
            PutOrCall::Call,
            d(EXPIRY.0, EXPIRY.1, EXPIRY.2),
            d(2027, 9, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        // plain lognormal refuses
        assert!(option.price(&discount, &forward, 0.35).is_err());
        let quote = CommodityVol::ShiftedLognormal {
            vol: 0.35,
            shift: 10.0,
        };
        let price = option.price(&discount, &forward, quote).unwrap();
        assert!(price > 0.0);
        // parity still holds: c - p = df * (F - K) * q
        let mut put = option.clone();
        put.put_or_call = PutOrCall::Put;
        let df = discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2));
        let parity = 10_000.0 * df * (-1.50 - 0.50);
        let diff = price - put.price(&discount, &forward, quote).unwrap();
        assert!((diff - parity).abs() < 1e-6, "{diff} vs {parity}");
        // a shift too small to displace the forward positive still errors
        assert!(option
            .price(
                &discount,
                &forward,
                CommodityVol::ShiftedLognormal {
                    vol: 0.35,
                    shift: 1.0
                }
            )
            .is_err());
    }

    #[test]
    fn normal_model_prices_any_sign_and_matches_the_kernel() {
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(-5.25, reference).unwrap();
        let option = CommodityOption::new(
            10_000.0,
            -2.0,
            PutOrCall::Call,
            d(EXPIRY.0, EXPIRY.1, EXPIRY.2),
            d(2027, 9, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        let price = option
            .price(&discount, &forward, CommodityVol::Normal(4.5))
            .unwrap();
        // t = 1 exactly, r from the curve: compare against the kernel
        let r = -discount.df_date(d(EXPIRY.0, EXPIRY.1, EXPIRY.2)).ln();
        let kernel = crate::cmdty::bachelier::price(
            -5.25,
            -2.0,
            r,
            4.5,
            1.0,
            PutOrCall::Call,
            FuturesSettlement::Discounted,
        );
        assert!((price - 10_000.0 * kernel).abs() < 1e-6, "{price}");
        // normal greeks: an OTM call on a negative underlying still has
        // positive delta and vega
        let greeks = option
            .greeks(&discount, &forward, CommodityVol::Normal(4.5))
            .unwrap();
        assert!(greeks.delta > 0.0 && greeks.vega > 0.0 && greeks.gamma > 0.0);
    }

    #[test]
    fn shifted_and_normal_implied_vols_round_trip() {
        let discount = flat_discount(0.04);
        let reference = d(REF.0, REF.1, REF.2);
        let forward = CommodityForwardCurve::flat(-1.50, reference).unwrap();
        let option = CommodityOption::new(
            10_000.0,
            0.50,
            PutOrCall::Call,
            d(EXPIRY.0, EXPIRY.1, EXPIRY.2),
            d(2027, 9, 20),
            FuturesSettlement::Discounted,
        )
        .unwrap();
        let shifted_quote = CommodityVol::ShiftedLognormal {
            vol: 0.42,
            shift: 10.0,
        };
        let premium = option.price(&discount, &forward, shifted_quote).unwrap();
        let vol = option
            .implied_vol_shifted(&discount, &forward, premium, 10.0)
            .unwrap();
        assert!((vol - 0.42).abs() < 1e-8, "{vol}");
        let premium_n = option
            .price(&discount, &forward, CommodityVol::Normal(4.5))
            .unwrap();
        let vol_n = option
            .implied_vol_normal(&discount, &forward, premium_n)
            .unwrap();
        assert!((vol_n - 4.5).abs() < 1e-6, "{vol_n}");
    }
}
