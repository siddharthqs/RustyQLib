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

    /// Bumped raw spot (no dividend escrow).
    pub fn spot(&self) -> f64 {
        self.market.spot.value() + self.bump.d_spot
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
        (maturity - self.market.valuation_date).num_days() as f64 / 365.0
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
                let t = (*date - self.market.valuation_date).num_days() as f64 / 365.0;
                amount * self.market.discount_curve.df(t) * (carry * t).exp()
            })
            .sum()
    }

    /// Bumped escrowed spot: base spot minus the PV of cash dividends,
    /// plus `d_spot`.
    pub fn effective_spot(&self, maturity: NaiveDate) -> f64 {
        let s = self.market.spot.value() - self.pv_cash_dividends(maturity);
        assert!(s > 0.0, "cash dividends exceed the spot price");
        s + self.bump.d_spot
    }

    fn base_forward(&self, maturity: NaiveDate) -> f64 {
        let t = self.base_time_to_maturity(maturity);
        let s = self.market.spot.value() - self.pv_cash_dividends(maturity);
        s * ((self.base_rate(maturity) - self.carry_yield()) * t).exp()
    }

    /// Bumped Black vol for `strike`, looked up at the **base** forward
    /// and maturity (sticky-strike; see module docs).
    pub fn volatility(&self, strike: f64, maturity: NaiveDate) -> f64 {
        let t = self.base_time_to_maturity(maturity);
        self.market
            .vol_surface
            .vol(strike, self.base_forward(maturity), t)
            + self.bump.d_vol
    }

    /// Bumped vol at an explicit surface point — for engines that manage
    /// their own forward/tenor (futures options, term-structure lattices).
    pub fn vol_at(&self, strike: f64, forward: f64, t: f64) -> f64 {
        self.market.vol_surface.vol(strike, forward, t) + self.bump.d_vol
    }
}
