//! Portfolio of options on a single underlying: aggregated Greeks and
//! risk-based PnL attribution.
//!
//! Positions are quantity-weighted (negative quantity = short). All Greeks
//! are additive, so the book's risk is the weighted sum of per-position
//! Greeks — each computed by that position's own pricing engine, so a book
//! can mix analytic vanillas, FD Americans and MC barriers.
//!
//! PnL attribution explains the book's change in value over a market move
//! `(d_spot, d_vol, d_rate, d_time)` with a second-order Taylor expansion:
//!
//! ```text
//! dV =  delta dS  +  1/2 gamma dS^2          (spot)
//!    +  vega  dv  +  1/2 volga dv^2          (implied vol)
//!    +  vanna dS dv                          (cross)
//!    +  theta dt  +  rho dr                  (time, rate)
//!    +  unexplained
//! ```
//!
//! The `actual` PnL is a full reprice of every position under the shifted
//! market ([`EquityOption::price_with`]), so `unexplained` is a true
//! residual — third-order terms and any cross terms not in the expansion.

use crate::core::traits::Instrument;
use crate::equity::vanilla_option::EquityOption;

/// A signed position in one option: `quantity` contracts (negative = short).
pub struct Position {
    pub option: EquityOption,
    pub quantity: f64,
}

/// A book of option positions on the same underlying.
#[derive(Default)]
pub struct EquityPortfolio {
    pub positions: Vec<Position>,
}

/// Quantity-weighted sums of the per-position Greeks.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    pub npv: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub vanna: f64,
    pub charm: f64,
    pub zomma: f64,
    pub volga: f64,
}

/// A market move to attribute PnL over. All fields default to zero, so a
/// scenario can set only what moves, e.g.
/// `MarketMove { d_spot: 2.0, d_time: 1.0 / 365.0, ..Default::default() }`.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarketMove {
    /// Absolute change in the underlying price.
    pub d_spot: f64,
    /// Parallel shift of the implied volatility (absolute, e.g. 0.01 = 1 pt).
    pub d_vol: f64,
    /// Parallel shift of the risk-free rate.
    pub d_rate: f64,
    /// Elapsed calendar time in years (1.0 / 365.0 = one day).
    pub d_time: f64,
}

/// Risk-based PnL explain for one market move.
#[derive(Debug, Clone, Copy)]
pub struct PnlAttribution {
    pub delta_pnl: f64,
    pub gamma_pnl: f64,
    pub vega_pnl: f64,
    pub volga_pnl: f64,
    pub vanna_pnl: f64,
    pub theta_pnl: f64,
    pub rho_pnl: f64,
    /// Sum of the Taylor terms above.
    pub explained: f64,
    /// Full-reprice PnL of the book under the shifted market.
    pub actual: f64,
    /// `actual - explained`: third-order and unmodeled cross terms.
    pub unexplained: f64,
}

impl EquityPortfolio {
    pub fn new() -> Self {
        Self { positions: Vec::new() }
    }

    /// Add `quantity` contracts of `option` (negative = short). All
    /// positions must share one underlying; the first position pins the
    /// symbol and a mismatch panics — this book aggregates risk against a
    /// single spot.
    pub fn add(&mut self, option: EquityOption, quantity: f64) -> &mut Self {
        if let Some(first) = self.positions.first() {
            assert_eq!(
                first.option.base.symbol, option.base.symbol,
                "EquityPortfolio aggregates one underlying: book is '{}', position is '{}'",
                first.option.base.symbol, option.base.symbol
            );
        }
        self.positions.push(Position { option, quantity });
        self
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Book value: quantity-weighted sum of position NPVs.
    pub fn npv(&self) -> f64 {
        self.positions.iter().map(|p| p.quantity * p.option.npv()).sum()
    }

    /// Aggregated Greeks, each position computed by its own engine.
    pub fn greeks(&self) -> PortfolioGreeks {
        let mut g = PortfolioGreeks::default();
        for p in &self.positions {
            let q = p.quantity;
            g.npv += q * p.option.npv();
            g.delta += q * p.option.delta();
            g.gamma += q * p.option.gamma();
            g.vega += q * p.option.vega();
            g.theta += q * p.option.theta();
            g.rho += q * p.option.rho();
            g.vanna += q * p.option.vanna();
            g.charm += q * p.option.charm();
            g.zomma += q * p.option.zomma();
            g.volga += q * p.option.volga();
        }
        g
    }

    /// Explain the book's PnL over `m` with second-order Greeks; `actual` is
    /// a full reprice of every position under the shifted market.
    pub fn pnl_attribution(&self, m: &MarketMove) -> PnlAttribution {
        let g = self.greeks();

        let delta_pnl = g.delta * m.d_spot;
        let gamma_pnl = 0.5 * g.gamma * m.d_spot * m.d_spot;
        let vega_pnl = g.vega * m.d_vol;
        let volga_pnl = 0.5 * g.volga * m.d_vol * m.d_vol;
        let vanna_pnl = g.vanna * m.d_spot * m.d_vol;
        let theta_pnl = g.theta * m.d_time;
        let rho_pnl = g.rho * m.d_rate;
        let explained =
            delta_pnl + gamma_pnl + vega_pnl + volga_pnl + vanna_pnl + theta_pnl + rho_pnl;

        // base from price_with(0,0,0,0), not npv(): under Monte Carlo both
        // legs then share the same draws and the difference is noise-free
        let actual: f64 = self
            .positions
            .iter()
            .map(|p| {
                p.quantity
                    * (p.option.price_with(m.d_spot, m.d_vol, m.d_rate, m.d_time)
                        - p.option.price_with(0.0, 0.0, 0.0, 0.0))
            })
            .sum();

        PnlAttribution {
            delta_pnl,
            gamma_pnl,
            vega_pnl,
            volga_pnl,
            vanna_pnl,
            theta_pnl,
            rho_pnl,
            explained,
            actual,
            unexplained: actual - explained,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::trade::PutOrCall;
    use crate::equity::builder::EquityOptionBuilder;
    use crate::equity::utils::Engine;
    use chrono::NaiveDate;

    fn option(put_or_call: PutOrCall, strike: f64) -> EquityOption {
        EquityOptionBuilder::new()
            .symbol("ACME")
            .spot(100.0)
            .strike(strike)
            .flat_vol(0.30)
            .flat_rate(0.05)
            .dividend_yield(0.02)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(put_or_call)
            .engine(Engine::BlackScholes)
            .build()
    }

    #[test]
    fn aggregation_is_quantity_weighted() {
        // 1 + 1 of the same option equals 2 of it
        let mut two_singles = EquityPortfolio::new();
        two_singles.add(option(PutOrCall::Call, 100.0), 1.0);
        two_singles.add(option(PutOrCall::Call, 100.0), 1.0);
        let mut one_double = EquityPortfolio::new();
        one_double.add(option(PutOrCall::Call, 100.0), 2.0);
        let (a, b) = (two_singles.greeks(), one_double.greeks());
        assert!((a.npv - b.npv).abs() < 1e-12);
        assert!((a.delta - b.delta).abs() < 1e-12);
        assert!((a.volga - b.volga).abs() < 1e-12);

        // long + short cancels exactly
        let mut flat = EquityPortfolio::new();
        flat.add(option(PutOrCall::Call, 100.0), 5.0);
        flat.add(option(PutOrCall::Call, 100.0), -5.0);
        let g = flat.greeks();
        for v in [g.npv, g.delta, g.gamma, g.vega, g.theta, g.rho, g.vanna, g.volga] {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn straddle_greeks_have_the_expected_shape() {
        let mut straddle = EquityPortfolio::new();
        straddle.add(option(PutOrCall::Call, 100.0), 1.0);
        straddle.add(option(PutOrCall::Put, 100.0), 1.0);
        let g = straddle.greeks();
        // near-ATM straddle: small residual delta, long gamma and vega
        assert!(g.delta.abs() < 0.25);
        assert!(g.gamma > 0.0);
        assert!(g.vega > 0.0);
        assert!(g.theta < 0.0);
    }

    #[test]
    #[should_panic(expected = "one underlying")]
    fn mixed_underlyings_are_rejected() {
        let other = EquityOptionBuilder::new()
            .symbol("OTHER")
            .spot(50.0)
            .strike(50.0)
            .flat_vol(0.2)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(PutOrCall::Call)
            .engine(Engine::BlackScholes)
            .build();
        let mut book = EquityPortfolio::new();
        book.add(option(PutOrCall::Call, 100.0), 1.0);
        book.add(other, 1.0);
    }

    #[test]
    fn attribution_explains_small_moves() {
        let mut book = EquityPortfolio::new();
        book.add(option(PutOrCall::Call, 100.0), 10.0);
        book.add(option(PutOrCall::Call, 110.0), -15.0);
        book.add(option(PutOrCall::Put, 95.0), 5.0);

        let m = MarketMove { d_spot: 1.0, d_vol: 0.01, d_rate: 1e-4, d_time: 1.0 / 365.0 };
        let a = book.pnl_attribution(&m);

        // the Taylor terms must reproduce the reprice up to third order
        assert!((a.explained - (a.delta_pnl + a.gamma_pnl + a.vega_pnl + a.volga_pnl
            + a.vanna_pnl + a.theta_pnl + a.rho_pnl)).abs() < 1e-12);
        assert!(
            a.unexplained.abs() < 0.01 * a.actual.abs().max(1.0),
            "unexplained {} vs actual {}",
            a.unexplained,
            a.actual
        );
        assert!((a.actual - a.explained - a.unexplained).abs() < 1e-12);
    }

    #[test]
    fn pure_time_move_is_theta() {
        let mut book = EquityPortfolio::new();
        book.add(option(PutOrCall::Call, 100.0), 10.0);
        let m = MarketMove { d_time: 1.0 / 365.0, ..Default::default() };
        let a = book.pnl_attribution(&m);
        assert_eq!(a.delta_pnl, 0.0);
        assert_eq!(a.vega_pnl, 0.0);
        // theta term explains an overnight move to within second-order time decay
        assert!((a.actual - a.theta_pnl).abs() < 5e-4 * a.theta_pnl.abs().max(1.0));
    }

    #[test]
    fn attribution_holds_across_engines() {
        // same book priced analytically and on the FD grid: attribution
        // buckets must broadly agree (grid discretization is the tolerance)
        let m = MarketMove { d_spot: 2.0, d_vol: 0.02, d_rate: 0.0, d_time: 1.0 / 365.0 };

        let mut analytic = EquityPortfolio::new();
        analytic.add(option(PutOrCall::Call, 100.0), 10.0);
        let a = analytic.pnl_attribution(&m);

        let mut fd_book = EquityPortfolio::new();
        let mut fd = option(PutOrCall::Call, 100.0);
        fd.engine = Engine::FiniteDifference;
        fd_book.add(fd, 10.0);
        let f = fd_book.pnl_attribution(&m);

        assert!((a.actual - f.actual).abs() < 0.05 * a.actual.abs().max(1.0),
            "analytic actual {} vs fd actual {}", a.actual, f.actual);
        assert!((a.delta_pnl - f.delta_pnl).abs() < 0.05 * a.delta_pnl.abs().max(1.0));
    }
}
