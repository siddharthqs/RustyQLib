//! The pricing API, provided once for every [`ConvertibleInstrument`]
//! through a blanket implementation: tree and finite-difference
//! prices, greeks and implied solves, generic over the
//! [`CreditModel`]. Bring the trait into scope to price a bond or a
//! preferred.

use chrono::NaiveDate;

use super::models::{ConvertibleMarket, CreditModel, JumpToDefaultMarket};
use super::fd::{self, ConvertibleFdGreeks, ConvertibleFdGrid, ConvertibleFdValuation, FdVolModel};
use super::implied::{
    central_spot_delta, solve_implied_credit_spread, solve_implied_hazard_rate,
    solve_implied_volatility,
};
use super::instrument::ConvertibleInstrument;
use super::tree::tree_value;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::vols::VolSurface;
use crate::equity::local_vol::LocalVolGrid;

/// Default number of tree steps for the pricing methods.
pub const DEFAULT_TREE_STEPS: usize = 800;

pub(crate) fn check_steps(steps: usize) -> Result<(), RustyQLibError> {
    if steps < 10 {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            format!("the tree needs at least 10 steps, got {steps}"),
        ));
    }
    Ok(())
}

pub(crate) fn check_quoted_price(clean_price: f64) -> Result<(), RustyQLibError> {
    if !(clean_price > 0.0 && clean_price.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            "convertible",
            format!("clean price must be positive, got {clean_price}"),
        ));
    }
    Ok(())
}

/// Pricing, greeks and implied solves for any convertible instrument.
pub trait ConvertiblePricing: ConvertibleInstrument {
    /// The straight floor under the market's credit model: the schedule
    /// ignoring the conversion right (and, for a mandatory, the
    /// principal). The engines collapse onto this when the shares are
    /// worthless.
    fn straight_floor<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.floor(market, curve, settlement)
    }

    /// Dirty price on the tree with `steps` time steps.
    fn dirty_price_with_steps<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
        steps: usize,
    ) -> Result<f64, RustyQLibError> {
        market.validate()?;
        check_steps(steps)?;
        tree_value(self, market, curve, settlement, steps)
    }

    /// Dirty price with [`DEFAULT_TREE_STEPS`].
    fn dirty_price<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        self.dirty_price_with_steps(market, curve, settlement, DEFAULT_TREE_STEPS)
    }

    /// Clean price: the dirty price less the accrued.
    fn clean_price<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.dirty_price(market, curve, settlement)? - self.accrued(settlement)?)
    }

    /// Equity delta: price change per unit share move, from a symmetric
    /// 1% spot bump on the tree. (The finite-difference engine reads a
    /// smoother delta off its grid.)
    fn delta<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        central_spot_delta(market.equity().spot, |spot| {
            self.dirty_price(&market.with_spot(spot), curve, settlement)
        })
    }

    /// The credit spread implied by a quoted clean price under
    /// Tsiveriotis-Fernandes, holding the equity inputs fixed.
    fn implied_credit_spread(
        &self,
        clean_price: f64,
        market: &ConvertibleMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        check_quoted_price(clean_price)?;
        solve_implied_credit_spread(clean_price, market, |m| {
            self.clean_price(m, curve, settlement)
        })
    }

    /// The hazard rate implied by a quoted clean price under jump to
    /// default, holding the equity inputs and the recovery fixed.
    ///
    /// The price is not monotone in the hazard (see the module docs of
    /// [`convertible`](crate::hybrid::convertible)): it falls from the
    /// zero-hazard value while the survival claims dominate, bottoms
    /// out, then rises as the recovery leg takes over. Only the falling
    /// branch is a credit reading, so the solve walks out from zero
    /// along it and fails with a
    /// [`CalibrationFailed`](RustyQLibError::CalibrationFailed) when
    /// the quote is above the zero-hazard value or below the model's
    /// minimum.
    fn implied_hazard_rate(
        &self,
        clean_price: f64,
        market: &JumpToDefaultMarket,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        check_quoted_price(clean_price)?;
        solve_implied_hazard_rate(clean_price, |hazard_rate| {
            let with_hazard = JumpToDefaultMarket {
                hazard_rate,
                ..*market
            };
            self.clean_price(&with_hazard, curve, settlement)
        })
    }

    /// The flat volatility implied by a quoted clean price, holding the
    /// credit inputs fixed — the usual desk read once the spread or
    /// hazard is pinned from straight debt or CDS.
    ///
    /// The price must be increasing in the volatility over the bracket
    /// (0.02 to 3.0), which holds for the plain conversion right; an
    /// issuer call that caps the upside can leave a flat or falling
    /// stretch, and a busted instrument is nearly vol-insensitive, so
    /// the solve then fails or returns an extreme value rather than a
    /// meaningful one.
    fn implied_volatility<M: CreditModel>(
        &self,
        clean_price: f64,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        check_quoted_price(clean_price)?;
        solve_implied_volatility(clean_price, |volatility| {
            self.clean_price(&market.with_volatility(volatility), curve, settlement)
        })
    }

    // --- finite differences --------------------------------------------

    /// Dupire local volatility from an implied surface, sampled on the
    /// equity module's default grid over this instrument's life from
    /// `settlement`, for [`FdVolModel::Local`]. `spot` and
    /// `dividend_yield` are the surface's reference inputs (the market
    /// struct's, normally).
    fn local_vol_grid(
        &self,
        surface: &VolSurface,
        curve: &YieldCurve,
        settlement: NaiveDate,
        spot: f64,
        dividend_yield: f64,
    ) -> Result<LocalVolGrid, RustyQLibError> {
        fd::local_vol_grid(self, surface, curve, settlement, spot, dividend_yield)
    }

    /// Value and spot greeks by finite differences at the market's flat
    /// volatility, under the market's credit model.
    fn fd_valuation<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
        grid: ConvertibleFdGrid,
    ) -> Result<ConvertibleFdValuation, RustyQLibError> {
        self.fd_valuation_with_vol(market, curve, settlement, grid, &FdVolModel::Flat)
    }

    /// Value and spot greeks by finite differences under a chosen
    /// volatility model (under jump to default, the diffusion
    /// conditional on survival).
    fn fd_valuation_with_vol<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
        grid: ConvertibleFdGrid,
        vol: &FdVolModel,
    ) -> Result<ConvertibleFdValuation, RustyQLibError> {
        fd::valuation(self, market, curve, settlement, grid, vol)
    }

    /// Greeks by finite differences at the flat volatility: the base
    /// valuation plus vega, theta, the parallel and key-rate DV01s and
    /// the credit DV01 (spread or hazard). `key_rate_tenors` may be
    /// empty.
    fn fd_greeks<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
        grid: ConvertibleFdGrid,
        key_rate_tenors: &[f64],
    ) -> Result<ConvertibleFdGreeks, RustyQLibError> {
        self.fd_greeks_with_vol(
            market,
            curve,
            settlement,
            grid,
            key_rate_tenors,
            &FdVolModel::Flat,
        )
    }

    /// [`fd_greeks`](Self::fd_greeks) under a chosen volatility model;
    /// vega shifts the whole field in parallel.
    fn fd_greeks_with_vol<M: CreditModel>(
        &self,
        market: &M,
        curve: &YieldCurve,
        settlement: NaiveDate,
        grid: ConvertibleFdGrid,
        key_rate_tenors: &[f64],
        vol: &FdVolModel,
    ) -> Result<ConvertibleFdGreeks, RustyQLibError> {
        fd::greeks(self, market, curve, settlement, grid, key_rate_tenors, vol)
    }
}

impl<T: ConvertibleInstrument> ConvertiblePricing for T {}
