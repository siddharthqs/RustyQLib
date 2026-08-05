//! The central instrument: [`EquityOption`] pairs an immutable contract
//! ([`EquityOptionBase`]) with the market state it is currently bound to
//! ([`EquityMarketData`]), plus the payoff, engine and model it prices
//! with. JSON construction lives in the `from_json` sibling module;
//! payoff structs live with their product files.

use crate::core::curves::{Compounding, YieldCurve};
use crate::core::errors::RustyQLibError;
use crate::core::quotes::Quote;
use crate::core::results::PricingResult;
use crate::core::traits::Instrument;
use crate::core::utils::ContractStyle;
use crate::core::vols::VolSurface;
use crate::equity::barrier::BarrierPayoff;
use crate::equity::blackscholes;
use crate::equity::bump::BumpedMarket;
use crate::equity::heston;
use crate::equity::utils::{LongShort, Model, Payoff, PayoffType, PricingEngine};
use crate::equity::{baw, binomial, bjerksund_stensland, finite_difference, greeks, montecarlo};
use blackscholes::BlackScholesPricer;
use chrono::NaiveDate;
use std::sync::Arc;

/// Contract terms and trade identity — **no market state**. Immutable
/// for the life of the trade; everything that moves with the market
/// lives in [`EquityMarketData`].
#[derive(Debug, Clone)]
pub struct EquityOptionBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    pub strike_price: f64,
    pub maturity_date: NaiveDate,
    /// When set, the underlying is a future priced with Black-76
    /// (the market spot is the futures price `F`), settled either with an
    /// up-front discounted premium or futures-style margined. European
    /// vanilla only, on the Analytical engine.
    pub futures_settlement: Option<crate::equity::black76::FuturesSettlement>,
    pub multiplier: f64,

    // trade info (candidate for a future Trade struct)
    pub current_price: Quote,
    pub entry_price: f64,
    pub long_short: LongShort,
}

/// The market state one equity instrument is currently **bound to** — the
/// pricing-view companion of the contract (QuantLib's process, Strata's
/// provider). Resolved from / snapshotted to the typed
/// [`Market`](crate::core::market::Market) store; swapped wholesale by
/// [`EquityOption::with_market`].
#[derive(Debug, Clone)]
pub struct EquityMarketData {
    /// The as-of date of this market snapshot; anchors every year fraction.
    pub valuation_date: NaiveDate,
    pub spot: Quote,
    pub dividend_yield: f64,
    /// Continuous stock borrow (repo) cost; part of the carry alongside
    /// the dividend yield.
    pub borrow_cost: f64,
    /// Discrete cash dividend forecasts (ex-date, amount per share).
    /// Analytic, tree and terminal Monte Carlo engines use the escrowed
    /// model (spot minus PV of dividends); path-wise Monte Carlo and
    /// finite difference apply the jumps at the ex-dates.
    pub cash_dividends: Vec<(NaiveDate, f64)>,
    /// Volatility surface; a flat surface represents a single constant vol.
    /// `Arc`-shared with the [`Market`](crate::core::market::Market) store:
    /// rebinding an instrument is a refcount bump, and replacing the
    /// surface means installing a **new** `Arc` (copy-on-write), never
    /// mutating through it.
    pub vol_surface: Arc<VolSurface>,
    /// Discounting curve anchored at `valuation_date`; discount factors are
    /// the source of truth, rates are derived views. `Arc`-shared and
    /// copy-on-write, like `vol_surface`.
    pub discount_curve: Arc<YieldCurve>,
}

#[derive(Debug)]
pub struct EquityOption {
    /// The contract (and trade identity): pure data, never market state.
    pub base: EquityOptionBase,
    /// The market this instrument is currently bound to.
    pub market: EquityMarketData,
    pub payoff: Box<dyn Payoff>,
    /// The numerical method, carrying its own settings.
    pub engine: PricingEngine,
    /// The dynamics of the underlying (GBM, local vol, or Heston with
    /// its parameters); consulted by the MC, FD and analytic engines.
    pub model: Model,
}

// manual impl: `payoff` clones through the trait object, engine and model
// are Copy
impl Clone for EquityOption {
    fn clone(&self) -> Self {
        EquityOption {
            base: self.base.clone(),
            market: self.market.clone(),
            payoff: self.payoff.clone_box(),
            engine: self.engine,
            model: self.model,
        }
    }
}

impl EquityOption {
    /// The Monte Carlo settings. Invariant: only called on the Monte
    /// Carlo engine's code paths (the dispatch guarantees it).
    pub(crate) fn mc_cfg(&self) -> &montecarlo::MonteCarloConfig {
        match &self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    pub(crate) fn fd_cfg(&self) -> &finite_difference::FdConfig {
        match &self.engine {
            PricingEngine::FiniteDifference(cfg) => cfg,
            _ => unreachable!("finite-difference code path reached on a non-FD engine"),
        }
    }

    /// Heston parameters. Invariant: only called on Heston-model code
    /// paths (the model dispatch guarantees it).
    pub(crate) fn heston_params(&self) -> &crate::equity::heston::HestonParams {
        match &self.model {
            Model::Heston(hp) => hp,
            _ => unreachable!("Heston code path reached on a non-Heston model"),
        }
    }

    pub(crate) fn lattice_cfg(&self) -> &crate::core::lattice::LatticeConfig {
        match &self.engine {
            PricingEngine::Binomial(cfg) => cfg,
            _ => unreachable!("lattice code path reached on a non-Binomial engine"),
        }
    }

    /// Test-only shortcuts for tweaking engine configuration in place;
    /// production code configures engines through the builder.
    #[cfg(test)]
    pub(crate) fn mc_cfg_mut(&mut self) -> &mut montecarlo::MonteCarloConfig {
        match &mut self.engine {
            PricingEngine::MonteCarlo(cfg) => cfg,
            _ => unreachable!("Monte Carlo code path reached on a non-MC engine"),
        }
    }

    #[cfg(test)]
    pub(crate) fn fd_cfg_mut(&mut self) -> &mut finite_difference::FdConfig {
        match &mut self.engine {
            PricingEngine::FiniteDifference(cfg) => cfg,
            _ => unreachable!("finite-difference code path reached on a non-FD engine"),
        }
    }
}

impl EquityOptionBase {
    /// True when the underlying is a future priced with Black-76.
    pub fn is_futures_option(&self) -> bool {
        self.futures_settlement.is_some()
    }
    /// The contract's currency code, falling back to
    /// [`DEFAULT_CURRENCY`](crate::core::market::DEFAULT_CURRENCY) when the
    /// contract does not state one — used as the [`Discount`]
    /// (crate::core::market::Discount) key into a [`Market`]
    /// (crate::core::market::Market).
    pub fn currency_code(&self) -> &str {
        self.currency
            .as_deref()
            .unwrap_or(crate::core::market::DEFAULT_CURRENCY)
    }
}

// Contract-and-market quantities: these straddle the boundary (a year
// fraction needs the contract's maturity AND the market's valuation
// date), so they live on the pairing — the option — not on either half.
impl EquityOption {
    pub fn time_to_maturity(&self) -> f64 {
        (self.base.maturity_date - self.market.valuation_date).num_days() as f64 / 365.0
    }
    /// Discount factor from the valuation date to maturity, off the curve.
    pub fn maturity_discount_factor(&self) -> f64 {
        self.market.discount_curve.df(self.time_to_maturity())
    }
    /// Continuously compounded zero rate to maturity implied by the curve.
    /// This is the `r` that enters d1/d2; it is consistent with
    /// [`maturity_discount_factor`](Self::maturity_discount_factor) by construction.
    pub fn risk_free_rate(&self) -> f64 {
        self.market
            .discount_curve
            .zero_rate_with(self.time_to_maturity(), Compounding::Continuous)
    }
    /// Total continuous carry on the underlying: dividend yield plus
    /// borrow cost. This is the "q" every pricing formula uses.
    pub fn carry_yield(&self) -> f64 {
        self.market.dividend_yield + self.market.borrow_cost
    }
    /// Escrow value of the cash dividends with ex-dates inside the option's
    /// life: the amount to carve out of spot so the risky stub reproduces
    /// the jump-model forward.
    ///
    /// Each dividend is discounted at the **net carry rate** `r - carry`,
    /// not the risk-free rate, so that the escrow accretes at the same rate
    /// the risky stub grows (`effective_spot` is grown at `r - carry` in
    /// [`forward_price`](Self::forward_price)). This makes the analytic
    /// forward match the well-defined jump model
    /// `F = (S - D e^{-(r-carry)t}) e^{(r-carry)T}` used by the FD and
    /// path-wise Monte Carlo engines. With no continuous carry this
    /// reduces to plain risk-free discounting.
    pub fn pv_cash_dividends(&self) -> f64 {
        let carry = self.carry_yield();
        self.market
            .cash_dividends
            .iter()
            .filter(|(date, _)| {
                *date > self.market.valuation_date && *date <= self.base.maturity_date
            })
            .map(|(date, amount)| {
                let t = (*date - self.market.valuation_date).num_days() as f64 / 365.0;
                // df(t) = e^{-r t}; multiplying by e^{carry t} discounts at
                // the net carry (r - carry), generalizing to any curve shape.
                amount * self.market.discount_curve.df(t) * (carry * t).exp()
            })
            .sum()
    }
    /// Escrowed-model spot: the quoted spot minus the PV of cash dividends
    /// paid over the option's life. This is the lognormal driver for the
    /// analytic and terminal-simulation engines.
    pub fn effective_spot(&self) -> f64 {
        let s = self.market.spot.value() - self.pv_cash_dividends();
        assert!(s > 0.0, "cash dividends exceed the spot price");
        s
    }
    /// Forward price of the underlying at maturity: escrowed spot grown at
    /// the carry-adjusted rate, `(S - PV(divs)) * exp((r - q - b) * T)`.
    pub fn forward_price(&self) -> f64 {
        let t = self.time_to_maturity();
        self.effective_spot() * ((self.risk_free_rate() - self.carry_yield()) * t).exp()
    }
    /// Black volatility for this option's strike and expiry, read off the
    /// surface (a flat surface returns its single vol).
    pub fn volatility(&self) -> f64 {
        self.market.vol_surface.vol(
            self.base.strike_price,
            self.forward_price(),
            self.time_to_maturity(),
        )
    }
    pub fn d1(&self) -> f64 {
        // Black-Scholes-Merton d1 on the escrowed spot and total carry
        let volatility = self.volatility();
        let d1_numerator = (self.effective_spot() / self.base.strike_price).ln()
            + (self.risk_free_rate() - self.carry_yield() + 0.5 * volatility.powi(2))
                * self.time_to_maturity();
        let d1_denominator = volatility * (self.time_to_maturity().sqrt());
        d1_numerator / d1_denominator
    }
    pub fn d2(&self) -> f64 {
        self.d1() - self.volatility() * self.time_to_maturity().sqrt()
    }
}
impl EquityOption {
    pub fn get_premium_at_risk(&self) -> f64 {
        let value = self.npv();
        let pay_off = self
            .payoff
            .payoff_amount(self.market.spot.value(), self.base.strike_price);
        if pay_off > 0.0 {
            value - pay_off
        } else {
            value
        }
    }

    /// Implied Black-Scholes volatility for `option_price` (safeguarded
    /// Newton with arbitrage-bound checks); does not modify the option.
    pub fn try_imp_vol(&self, option_price: f64) -> Result<f64, RustyQLibError> {
        blackscholes::implied_vol_from_price(
            self.effective_spot(),
            self.base.strike_price,
            self.risk_free_rate(),
            self.carry_yield(),
            self.time_to_maturity(),
            option_price,
            *self.payoff.put_or_call(),
        )
    }
    /// Implied vol for `option_price`; leaves the option holding a flat
    /// surface at the solved vol. Panics on arbitrage-violating prices —
    /// use [`try_imp_vol`](Self::try_imp_vol) to handle those gracefully.
    pub fn imp_vol(&mut self, option_price: f64) -> f64 {
        let vol = self
            .try_imp_vol(option_price)
            .expect("implied vol solve failed");
        self.set_flat_vol(vol.max(1e-8));
        vol
    }
    pub fn get_imp_vol(&mut self) -> f64 {
        let target = self.base.current_price.mid();
        self.imp_vol(target)
    }
    fn set_flat_vol(&mut self, vol: f64) {
        self.market.vol_surface = Arc::new(
            VolSurface::flat(
                vol,
                self.market.vol_surface.reference_date(),
                self.market.vol_surface.day_count(),
            )
            .expect("vol must be positive"),
        );
    }
}

impl EquityOption {
    /// Reject engine/model/payoff combinations the library cannot price,
    /// with an error naming the engine that can.
    pub(crate) fn check_engine_support(&self) -> Result<(), RustyQLibError> {
        let unsupported = |msg: &str| Err(RustyQLibError::UnsupportedEngine(msg.to_string()));
        let bermudan = matches!(self.payoff.exercise_style(), ContractStyle::Bermudan(_));
        // American and Bermudan share the early-exercise engine rules
        let american = matches!(self.payoff.exercise_style(), ContractStyle::American) || bermudan;
        if self.base.is_futures_option() {
            if !matches!(self.engine, PricingEngine::BlackScholes) {
                return unsupported(
                    "Options on futures (Black-76) price on the Analytical engine only",
                );
            }
            if american {
                return unsupported("Black-76 supports European exercise only");
            }
        }
        if matches!(self.payoff.payoff_kind(), PayoffType::Chooser) {
            if !matches!(self.engine, PricingEngine::BlackScholes) {
                return unsupported(
                    "Chooser options price on the Analytical engine (closed form) only",
                );
            }
            if self.model.is_heston() {
                return unsupported(
                    "The chooser closed form assumes Black-Scholes dynamics; the analytic \
                     Heston pricer covers vanilla and binary payoffs only",
                );
            }
        }
        if self.payoff.is_path_dependent() {
            if american {
                return unsupported(
                    "early-exercise (American/Bermudan) path-dependent options are not supported yet",
                );
            }
            if matches!(self.engine, PricingEngine::Binomial(_)) {
                return unsupported(
                    "Path-dependent payoffs are not supported on the Binomial engine",
                );
            }
            if matches!(self.engine, PricingEngine::FiniteDifference(_))
                && !matches!(self.payoff.payoff_kind(), PayoffType::Barrier)
            {
                return unsupported(
                    "Of the path-dependent payoffs only barriers price on the FD \
                     engine; use MonteCarlo",
                );
            }
            if matches!(self.engine, PricingEngine::BlackScholes)
                && matches!(
                    self.payoff.payoff_kind(),
                    PayoffType::Autocallable | PayoffType::Accumulator
                )
            {
                return unsupported(
                    "Autocallables and accumulators price on the MonteCarlo engine only",
                );
            }
            if matches!(self.engine, PricingEngine::MonteCarlo(_)) {
                if let Some(barrier) = self.payoff.as_any().downcast_ref::<BarrierPayoff>() {
                    if barrier.rebate != 0.0 && barrier.rebate_at_hit {
                        return unsupported(
                            "at-hit rebates need the touch time: price on the Analytical \
                             engine (Monte Carlo supports the at-expiry rebate convention)",
                        );
                    }
                }
            }
        }
        let heston = self.model.is_heston();
        if heston && matches!(self.engine, PricingEngine::Binomial(_)) {
            return unsupported(
                "The Heston model is supported on the Analytical, MonteCarlo and \
                 FiniteDifference (2-D ADI) engines, not Binomial",
            );
        }
        if heston
            && matches!(self.engine, PricingEngine::FiniteDifference(_))
            && !matches!(
                self.payoff.payoff_kind(),
                PayoffType::Vanilla | PayoffType::Binary
            )
        {
            return unsupported(
                "The Heston ADI engine prices vanilla and binary payoffs; \
                 use MonteCarlo for path-dependent payoffs",
            );
        }
        match self.engine {
            PricingEngine::BlackScholes if american => unsupported(
                "Analytical engine cannot price early exercise; \
                 use Binomial, FiniteDifference or MonteCarlo",
            ),
            PricingEngine::BaroneAdesiWhaley | PricingEngine::BjerksundStensland => {
                let name = match self.engine {
                    PricingEngine::BaroneAdesiWhaley => "Barone-Adesi-Whaley",
                    _ => "Bjerksund-Stensland",
                };
                if !matches!(self.payoff.payoff_kind(), PayoffType::Vanilla) {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} approximates vanilla options only"
                    )));
                }
                if heston {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} assumes constant-vol Black-Scholes dynamics, not Heston"
                    )));
                }
                if bermudan {
                    return Err(RustyQLibError::UnsupportedEngine(format!(
                        "{name} approximates American exercise only; Bermudan prices on \
                         Binomial, FiniteDifference or MonteCarlo"
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl Instrument for EquityOption {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        self.check_engine_support()?;
        let heston = self.model.is_heston();
        Ok(match self.engine {
            PricingEngine::BlackScholes if heston => heston::analytic_npv(self, None),
            PricingEngine::BlackScholes => BlackScholesPricer::new().npv(self),
            PricingEngine::MonteCarlo(_) => montecarlo::npv(self, None),
            PricingEngine::Binomial(_) => binomial::npv(self, None),
            PricingEngine::FiniteDifference(_) => finite_difference::npv(self, None),
            PricingEngine::BaroneAdesiWhaley => baw::npv(self, None),
            PricingEngine::BjerksundStensland => bjerksund_stensland::npv(self, None),
        })
    }

    /// Value, all nine Greeks, and (on the Monte Carlo engine) the
    /// standard error, from one call — batched through the central
    /// sensitivity engine ([`crate::equity::greeks`]), which shares
    /// solves and reprices across the Greeks.
    fn price(&self) -> Result<PricingResult, RustyQLibError> {
        self.check_engine_support()?;
        Ok(crate::equity::greeks::pricing_result(self))
    }
}

/// Greeks route through the central sensitivity engine
/// ([`crate::equity::greeks`]): the FD and Binomial engines read
/// delta/gamma/theta off their own grid/tree with higher orders from
/// bumped solutions; the analytic engine uses the payoff-aware
/// Black-Scholes closed forms (including Black-76 futures); the bump
/// engines (Monte Carlo with common random numbers, BAW,
/// Bjerksund-Stensland, analytic Heston) share one set of
/// central-difference stencils with per-engine bump sizes.
impl EquityOption {
    pub(crate) fn analytic_heston(&self) -> bool {
        matches!(
            self.engine,
            PricingEngine::BlackScholes | PricingEngine::Binomial(_)
        ) && self.model.is_heston()
    }
    pub fn delta(&self) -> f64 {
        greeks::delta(self)
    }
    pub fn gamma(&self) -> f64 {
        greeks::gamma(self)
    }
    pub fn vega(&self) -> f64 {
        greeks::vega(self)
    }
    pub fn theta(&self) -> f64 {
        greeks::theta(self)
    }
    pub fn rho(&self) -> f64 {
        greeks::rho(self)
    }
    /// Vanna: change in delta per unit change in implied volatility.
    pub fn vanna(&self) -> f64 {
        greeks::vanna(self)
    }
    /// Charm: change in delta per year of calendar time.
    pub fn charm(&self) -> f64 {
        greeks::charm(self)
    }
    /// Delta elasticity (`S * gamma / delta`), also called percentage gamma.
    pub fn gamma_p(&self) -> f64 {
        greeks::gamma_p(self)
    }
    /// Zomma: change in gamma per unit change in implied volatility.
    pub fn zomma(&self) -> f64 {
        greeks::zomma(self)
    }
    /// Volga (vomma): change in vega per unit change in implied volatility.
    pub fn volga(&self) -> f64 {
        greeks::volga(self)
    }
    /// Reprice under a caller-built market view. The caller bumps the
    /// market ([`BumpedMarket::new`]) and this method only routes the view
    /// to the option's engine; a zero-bump view is the base price, bit for
    /// bit. The Greeks stencils and the PnL attribution are the main
    /// callers.
    ///
    /// Monte Carlo repricing uses common random numbers, so bumped-minus-
    /// base differences are free of sampling noise.
    pub fn price_bumped(&self, m: &BumpedMarket) -> f64 {
        if self.base.is_futures_option() {
            // the market spot IS the futures price F; the surface is read
            // at the base (F, T) point, sticky-strike like the view
            let f_base = self.market.spot.value();
            let k = self.base.strike_price;
            let t_base = self.time_to_maturity();
            return crate::equity::black76::price(
                m.spot(),
                k,
                m.risk_free_rate(self.base.maturity_date),
                m.vol_at(k, f_base, t_base),
                m.time_to_maturity(self.base.maturity_date).max(1e-6),
                *self.payoff.put_or_call(),
                self.base
                    .futures_settlement
                    .expect("futures option must carry a settlement"),
            );
        }
        match self.engine {
            PricingEngine::MonteCarlo(_) => montecarlo::npv(self, Some(m)),
            PricingEngine::FiniteDifference(_) => finite_difference::npv(self, Some(m)),
            PricingEngine::BaroneAdesiWhaley => baw::npv(self, Some(m)),
            PricingEngine::BjerksundStensland => bjerksund_stensland::npv(self, Some(m)),
            _ if self.analytic_heston() => heston::analytic_npv(self, Some(m)),
            PricingEngine::Binomial(_) => binomial::npv(self, Some(m)),
            _ => BlackScholesPricer::price_bumped(self, m),
        }
    }
}
