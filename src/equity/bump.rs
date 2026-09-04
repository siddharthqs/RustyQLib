//! [`BumpedMarket`]: an equity market snapshot read *through* a scalar
//! [`Bump`] — the single source of shifted market values for every engine.
//!
//! The view borrows the option's bound [`EquityMarketData`] and applies
//! the bump at read time: `spot() = base + d_spot`, `risk_free_rate() =
//! base + d_rate`, `volatility() = base + d_vol`, `time_to_maturity() =
//! base − d_time`. No allocation, no mutation, and a zero bump reads the
//! base market bit for bit — which is why engines take
//! `Option<&BumpedMarket>` and treat `None` as a zero-bump view.
//!
//! Two deliberate semantics, both matching the historical scalar path:
//! - **Sticky-strike vol**: `volatility()` looks the surface up at the
//!   *base* forward, then adds `d_vol` — a spot bump does not slide the
//!   smile read-point, so delta stencils measure pure spot sensitivity.
//! - **Base-dated rates**: `risk_free_rate()` reads the zero rate at the
//!   *base* maturity tenor; `d_time` shortens the maturity engines price,
//!   not the tenor the rate is read at.
//!
//! Callers construct the view themselves (`BumpedMarket::new(&option.market,
//! bump)`) and hand it to [`EquityOption::price_bumped`]
//! (crate::equity::vanilla_option::EquityOption::price_bumped) — the
//! option never bumps its own market. For object-level scenarios (curve
//! reshapes, per-name shocks, whole-day aging) use the typed
//! [`Market`](crate::core::market::Market) store and `npv_in` instead.

use chrono::NaiveDate;

pub use crate::core::bump::Bump;
use crate::core::curves::Compounding;
use crate::equity::vanilla_option::EquityMarketData;

/// A borrowed market snapshot read through a [`Bump`]; see module docs.
#[derive(Debug, Clone, Copy)]
pub struct BumpedMarket<'a> {
    market: &'a EquityMarketData,
    bump: Bump,
}

impl<'a> BumpedMarket<'a> {
    /// Zero-bump view: every accessor returns the base market value.
    pub fn base(market: &'a EquityMarketData) -> Self {
        BumpedMarket {
            market,
            bump: Bump::NONE,
        }
    }

    pub fn new(market: &'a EquityMarketData, bump: Bump) -> Self {
        BumpedMarket { market, bump }
    }

    /// The shift this view applies — for engines that must *interpret* a
    /// bump rather than read shifted scalars (Heston's vol-parameter
    /// shift, the finite-difference theta roll, term-structure lattices).
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// The underlying unbumped market data.
    pub fn market(&self) -> &EquityMarketData {
        self.market
    }

    /// Bumped raw spot (no dividend escrow), floored just above zero so
    /// a deep down-bump cannot hand the engines a negative spot (see
    /// [`conventions::MIN_BUMPED_SPOT_FRAC`](crate::equity::conventions::MIN_BUMPED_SPOT_FRAC)).
    pub fn spot(&self) -> f64 {
        let base = self.market.spot.value();
        (base + self.bump.d_spot).max(base * crate::equity::conventions::MIN_BUMPED_SPOT_FRAC)
    }

    /// Total continuous carry (dividend yield + borrow); no bump dimension.
    pub fn carry_yield(&self) -> f64 {
        self.market.dividend_yield + self.market.borrow_cost
    }

    /// Remaining maturity after `d_time` of elapsed calendar time,
    /// **unfloored** — engines apply their own numerical floors.
    pub fn time_to_maturity(&self, maturity: NaiveDate) -> f64 {
        self.base_time_to_maturity(maturity) - self.bump.d_time
    }

    fn base_time_to_maturity(&self, maturity: NaiveDate) -> f64 {
        crate::equity::conventions::year_fraction(self.market.valuation_date, maturity)
    }

    fn base_rate(&self, maturity: NaiveDate) -> f64 {
        self.market.discount_curve.zero_rate_with(
            self.base_time_to_maturity(maturity),
            Compounding::Continuous,
        )
    }

    /// Bumped continuously compounded zero rate, read at the **base**
    /// maturity tenor (see module docs).
    pub fn risk_free_rate(&self, maturity: NaiveDate) -> f64 {
        self.base_rate(maturity) + self.bump.d_rate
    }

    /// Escrow value of cash dividends inside the option's life, at base
    /// market levels (the net-carry discounting of
    /// `EquityOption::pv_cash_dividends`).
    pub fn pv_cash_dividends(&self, maturity: NaiveDate) -> f64 {
        let carry = self.carry_yield();
        self.market
            .cash_dividends
            .iter()
            .filter(|(date, _)| *date > self.market.valuation_date && *date <= maturity)
            .map(|(date, amount)| {
                let t =
                    crate::equity::conventions::year_fraction(self.market.valuation_date, *date);
                amount * self.market.discount_curve.df(t) * (carry * t).exp()
            })
            .sum()
    }

    /// Bumped escrowed spot: base spot minus the PV of cash dividends,
    /// plus `d_spot` — floored just above zero, so a deep spot-down
    /// stress prices at a near-zero escrowed spot (intrinsic for puts,
    /// worthless calls) instead of feeding the engines a negative one.
    /// Dividends exceeding the *base* spot remain a data error.
    pub fn effective_spot(&self, maturity: NaiveDate) -> f64 {
        let base = self.market.spot.value();
        let s = base - self.pv_cash_dividends(maturity);
        // Unreachable backstop: `EquityOptionBuilder::build` refuses cash
        // dividends whose escrow value reaches the spot, so a
        // builder-constructed option can never reach this. It stays a hard
        // assert for markets assembled directly (or mutated after build).
        assert!(s > 0.0, "cash dividends exceed the spot price");
        (s + self.bump.d_spot).max(base * crate::equity::conventions::MIN_BUMPED_SPOT_FRAC)
    }

    pub(crate) fn base_forward(&self, maturity: NaiveDate) -> f64 {
        let t = self.base_time_to_maturity(maturity);
        let s = self.market.spot.value() - self.pv_cash_dividends(maturity);
        s * ((self.base_rate(maturity) - self.carry_yield()) * t).exp()
    }

    /// Bumped Black vol for `strike`, looked up at the **base** forward
    /// and maturity (sticky-strike; see module docs). Floored at
    /// [`conventions::MIN_BUMPED_VOL`](crate::equity::conventions::MIN_BUMPED_VOL)
    /// so a vega down-bump on a tiny-vol option cannot hand the engines
    /// a non-positive vol.
    pub fn volatility(&self, strike: f64, maturity: NaiveDate) -> f64 {
        let t = self.base_time_to_maturity(maturity);
        (self
            .market
            .vol_surface
            .vol(strike, self.base_forward(maturity), t)
            + self.bump.d_vol)
            .max(crate::equity::conventions::MIN_BUMPED_VOL)
    }

    /// Bumped vol at an explicit surface point — for engines that manage
    /// their own forward/tenor (futures options, term-structure lattices).
    /// Floored like [`volatility`](Self::volatility).
    pub fn vol_at(&self, strike: f64, forward: f64, t: f64) -> f64 {
        (self.market.vol_surface.vol(strike, forward, t) + self.bump.d_vol)
            .max(crate::equity::conventions::MIN_BUMPED_VOL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::core::traits::Instrument;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn degenerate_bumps_price_at_the_floors_instead_of_panicking() {
        // a tiny-vol option on the Monte Carlo (bump-route) engine: the
        // 1% vega down-bump used to push sigma negative
        let option = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.005)
            .flat_rate(0.03)
            .valuation_date(date(2026, 1, 1))
            .maturity_date(date(2027, 1, 1))
            .vanilla(PutOrCall::Call)
            .engine(Engine::MonteCarlo)
            .paths(4_000)
            .build()
            .expect("tiny-vol option must build");
        let result = option.price().expect("bumped Greeks must not panic");
        assert!(result.pv.is_finite() && result.greeks.vega.is_finite());

        // a deep spot-down stress on a cash-dividend name: the escrowed
        // spot floors just above zero instead of going negative
        let divd = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.03)
            .valuation_date(date(2026, 1, 1))
            .maturity_date(date(2027, 1, 1))
            .cash_dividend(date(2026, 7, 1), 5.0)
            .vanilla(PutOrCall::Put)
            .build()
            .expect("dividend option must build");
        let crushed = BumpedMarket::new(
            &divd.market,
            Bump {
                d_spot: -150.0,
                d_vol: 0.0,
                d_rate: 0.0,
                d_time: 0.0,
            },
        );
        let pv = divd.price_bumped(&crushed);
        assert!(pv.is_finite(), "stress must value, got {pv}");
        // an ATM put on a near-zero spot is worth ~ the discounted strike
        assert!(pv > 90.0, "deep-crash put must be near max value: {pv}");
    }

    #[test]
    #[should_panic(expected = "cash dividends exceed the spot price")]
    fn effective_spot_backstop_still_guards_directly_assembled_markets() {
        // EquityOptionBuilder::build refuses cash dividends whose escrow
        // value reaches the spot, so the only way to this assert is a
        // market assembled directly or mutated after build — which is
        // exactly what the backstop must keep catching
        let mut divd = EquityOptionBuilder::new()
            .spot(100.0)
            .strike(100.0)
            .flat_vol(0.3)
            .flat_rate(0.03)
            .valuation_date(date(2026, 1, 1))
            .maturity_date(date(2027, 1, 1))
            .cash_dividend(date(2026, 7, 1), 5.0)
            .vanilla(PutOrCall::Put)
            .build()
            .expect("dividend option must build");
        divd.market.cash_dividends[0].1 = 500.0;
        BumpedMarket::base(&divd.market).effective_spot(divd.base.maturity_date);
    }
}
