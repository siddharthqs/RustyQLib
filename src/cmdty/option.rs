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

use chrono::NaiveDate;

use crate::cmdty::forward_curve::CommodityForwardCurve;
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::results::Greeks;
use crate::core::trade::PutOrCall;
use crate::equity::black76;

pub use crate::equity::black76::FuturesSettlement;

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
        if !strike.is_finite() || strike <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("strike must be positive for Black-76, got {strike}"),
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
    /// underlying's delivery date. Errors on a non-positive price —
    /// Black-76 is lognormal and cannot price a negative underlying.
    pub fn forward_price(&self, forward: &CommodityForwardCurve) -> Result<f64, RustyQLibError> {
        let f = forward.price(self.underlying_date);
        if f <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("Black-76 needs a positive futures price, got {f}"),
            ));
        }
        Ok(f)
    }

    /// Premium for the whole contract.
    pub fn price(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<f64, RustyQLibError> {
        let (f, r, t) = self.market_inputs(discount, forward, vol)?;
        Ok(self.quantity
            * black76::price(f, self.strike, r, vol, t, self.put_or_call, self.settlement))
    }

    /// All Black-76 sensitivities, scaled to the contract. Delta and
    /// gamma are with respect to the futures price, vega per unit of
    /// vol, theta per year of calendar time, rho per unit of rate.
    pub fn greeks(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<Greeks, RustyQLibError> {
        let (f, r, t) = self.market_inputs(discount, forward, vol)?;
        let (k, pc, s, q) = (
            self.strike,
            self.put_or_call,
            self.settlement,
            self.quantity,
        );
        Ok(Greeks {
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
        if !premium.is_finite() || premium < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium must be non-negative, got {premium}"),
            ));
        }
        let (mut lo, mut hi) = (1e-9, 10.0);
        let price_at = |v: f64| self.price(discount, forward, v);
        if price_at(lo)? > premium + 1e-12 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium {premium} is below intrinsic value"),
            ));
        }
        if price_at(hi)? < premium {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("premium {premium} exceeds the vol=1000% price"),
            ));
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if price_at(mid)? < premium {
                lo = mid;
            } else {
                hi = mid;
            }
            if hi - lo < 1e-12 {
                break;
            }
        }
        Ok(0.5 * (lo + hi))
    }

    /// Resolve `(F, r, t)` for the Black-76 formulas: the futures price
    /// off the forward curve, and the continuously compounded zero rate
    /// and year fraction to expiry off the discount curve (so
    /// `e^{-rt}` is exactly the curve's discount factor to expiry).
    fn market_inputs(
        &self,
        discount: &YieldCurve,
        forward: &CommodityForwardCurve,
        vol: f64,
    ) -> Result<(f64, f64, f64), RustyQLibError> {
        if !vol.is_finite() || vol < 0.0 {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("volatility must be non-negative, got {vol}"),
            ));
        }
        let valuation = discount.reference_date();
        if self.expiry_date < valuation {
            return Err(RustyQLibError::invalid_input(
                "commodity option",
                format!("option expired {} (valuing {valuation})", self.expiry_date),
            ));
        }
        let f = self.forward_price(forward)?;
        let t = discount
            .day_count()
            .year_fraction(valuation, self.expiry_date);
        let r = if t > 0.0 {
            discount.zero_rate_with(t, Compounding::Continuous)
        } else {
            0.0
        };
        Ok((f, r, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert_eq!(option.forward_price(&forward).unwrap(), 80.0);
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
    fn validation_rejects_bad_contracts() {
        let expiry = d(EXPIRY.0, EXPIRY.1, EXPIRY.2);
        let s = FuturesSettlement::Discounted;
        assert!(CommodityOption::new(0.0, 70.0, PutOrCall::Call, expiry, expiry, s).is_err());
        assert!(CommodityOption::new(1_000.0, -1.0, PutOrCall::Call, expiry, expiry, s).is_err());
        // underlying before expiry makes no sense
        assert!(
            CommodityOption::new(1_000.0, 70.0, PutOrCall::Call, expiry, d(2027, 8, 1), s).is_err()
        );
    }
}
