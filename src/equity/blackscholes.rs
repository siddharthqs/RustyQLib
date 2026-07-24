use libm::{exp, log};
use std::f64::consts::{PI, SQRT_2};
use std::{io, thread};
use crate::core::quotes::Quote;
use chrono::{Datelike, Local, NaiveDate};
//use utils::{N,dN};
//use vanilla_option::{EquityOption,OptionType};
use crate::core::utils::{ContractStyle, norm_pdf, norm_cdf};
use crate::core::trade::{PutOrCall, Transection};
use super::asian::{self, AsianStrikeType, AveragingType};
use super::barrier;
use super::vanilla_option::{AsianPayoff, BarrierPayoff, BinaryPayoff, BinaryType, EquityOption, EquityOptionBase, VanillaPayoff};
use super::utils::{Engine, PayoffType, Payoff, LongShort};
use crate::core::curves::{Compounding, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::vols::VolSurface;
use super::super::core::traits::{Instrument,Greeks};

pub struct BlackScholesPricer;
impl BlackScholesPricer {
    pub fn new() -> Self {
        BlackScholesPricer
    }
    pub fn npv(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(bsd_option.time_to_maturity() >= 0.0, "Option is expired or negative time");
        assert!(bsd_option.base.underlying_price.value >= 0.0, "Negative underlying price not allowed");
        if bsd_option.base.is_futures_option() {
            return self.npv_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.npv_vanilla(bsd_option),
            PayoffType::Binary => self.npv_binary(bsd_option),
            PayoffType::Barrier => self.npv_barrier(bsd_option),
            PayoffType::Asian => self.npv_asian(bsd_option),
            PayoffType::Lookback => Self::lookback_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0),
            PayoffType::ForwardStart => self.npv_forward_start(bsd_option),
            _ => {0.0}
        }
    }
    pub fn delta(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(bsd_option.time_to_maturity() >= 0.0, "Option is expired or negative time");
        assert!(bsd_option.base.underlying_price.value >= 0.0, "Negative underlying price not allowed");
        if bsd_option.base.is_futures_option() {
            return self.delta_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.delta_vanilla(bsd_option),
            PayoffType::Binary => self.delta_binary(bsd_option),
            PayoffType::Barrier => self.delta_barrier(bsd_option),
            PayoffType::Asian => self.delta_asian(bsd_option),
            PayoffType::ForwardStart => self.delta_forward_start(bsd_option),
            PayoffType::Lookback => self.delta_lookback(bsd_option),
            _ => {0.0}
        }
    }
    pub fn gamma(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.gamma_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.gamma_vanilla(bsd_option),
            PayoffType::Binary => self.gamma_binary(bsd_option),
            PayoffType::Barrier => self.gamma_barrier(bsd_option),
            PayoffType::Asian => self.gamma_asian(bsd_option),
            PayoffType::ForwardStart => self.gamma_forward_start(bsd_option),
            PayoffType::Lookback => self.gamma_lookback(bsd_option),
            _ => {0.0}
        }
    }
    pub fn vega(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.vega_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.vega_vanilla(bsd_option),
            PayoffType::Binary => self.vega_binary(bsd_option),
            PayoffType::Barrier => self.vega_barrier(bsd_option),
            PayoffType::Asian => self.vega_asian(bsd_option),
            PayoffType::ForwardStart => self.vega_forward_start(bsd_option),
            PayoffType::Lookback => self.vega_lookback(bsd_option),
            _ => {0.0}
        }
    }
    pub fn theta(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.theta_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.theta_vanilla(bsd_option),
            PayoffType::Binary => self.theta_binary(bsd_option),
            PayoffType::Barrier => self.theta_barrier(bsd_option),
            PayoffType::Asian => self.theta_asian(bsd_option),
            PayoffType::ForwardStart => self.theta_forward_start(bsd_option),
            PayoffType::Lookback => self.theta_lookback(bsd_option),
            _ => {0.0}
        }
    }
    pub fn rho(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.rho_black76(bsd_option);
        }
        match &bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => self.rho_vanilla(bsd_option),
            PayoffType::Binary => self.rho_binary(bsd_option),
            PayoffType::Barrier => self.rho_barrier(bsd_option),
            PayoffType::Asian => self.rho_asian(bsd_option),
            PayoffType::ForwardStart => self.rho_forward_start(bsd_option),
            PayoffType::Lookback => self.rho_lookback(bsd_option),
            _ => {0.0}
        }
    }
    /// Vanna (`d delta / d volatility`).  Vanilla and Black-76 options use
    /// their closed forms; other analytic payoffs use a stable mixed bump.
    pub fn vanna(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.vanna_black76(bsd_option);
        }
        if matches!(bsd_option.payoff.payoff_kind(), PayoffType::Vanilla) {
            return bs_vanna(
                bsd_option.base.effective_spot(), bsd_option.base.strike_price,
                bsd_option.base.risk_free_rate(), bsd_option.base.carry_yield(),
                bsd_option.base.volatility(), bsd_option.time_to_maturity(),
            );
        }
        self.vanna_bumped(bsd_option)
    }
    /// Charm (`d delta / d calendar time`).  A positive value means delta
    /// rises as one year of calendar time elapses.
    pub fn charm(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.charm_black76(bsd_option);
        }
        if matches!(bsd_option.payoff.payoff_kind(), PayoffType::Vanilla) {
            return bs_charm(
                bsd_option.base.effective_spot(), bsd_option.base.strike_price,
                bsd_option.base.risk_free_rate(), bsd_option.base.carry_yield(),
                bsd_option.base.volatility(), bsd_option.time_to_maturity(),
                *bsd_option.payoff.put_or_call(),
            );
        }
        self.charm_bumped(bsd_option)
    }
    /// Delta elasticity (`S * gamma / delta`), sometimes called percentage
    /// gamma. It is undefined when delta is zero.
    pub fn gamma_p(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.gamma_p_black76(bsd_option);
        }
        let delta = self.delta(bsd_option);
        if delta == 0.0 {
            f64::NAN
        } else {
            bsd_option.base.underlying_price.value() * self.gamma(bsd_option) / delta
        }
    }
    /// Zomma (`d gamma / d volatility`). Vanilla and Black-76 use closed
    /// forms; other analytic payoffs use a central volatility bump.
    pub fn zomma(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.zomma_black76(bsd_option);
        }
        if matches!(bsd_option.payoff.payoff_kind(), PayoffType::Vanilla) {
            return bs_zomma(
                bsd_option.base.effective_spot(), bsd_option.base.strike_price,
                bsd_option.base.risk_free_rate(), bsd_option.base.carry_yield(),
                bsd_option.base.volatility(), bsd_option.time_to_maturity(),
            );
        }
        self.zomma_bumped(bsd_option)
    }
    /// Volga / vomma (`d vega / d volatility`). Vanilla and Black-76 use
    /// closed forms; other analytic payoffs use a central volatility bump.
    pub fn volga(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.volga_black76(bsd_option);
        }
        if matches!(bsd_option.payoff.payoff_kind(), PayoffType::Vanilla) {
            return bs_volga(
                bsd_option.base.effective_spot(), bsd_option.base.strike_price,
                bsd_option.base.risk_free_rate(), bsd_option.base.carry_yield(),
                bsd_option.base.volatility(), bsd_option.time_to_maturity(),
            );
        }
        self.volga_bumped(bsd_option)
    }
    // ── Black-76: European options on a future ─────────────────────────
    // The underlying_price is the futures price F; there is no spot,
    // dividend or carry. Vol is read from the surface at (K, F, T).

    fn black76_inputs(bsd_option: &EquityOption)
        -> (f64, f64, f64, f64, f64, PutOrCall, crate::equity::black76::FuturesSettlement)
    {
        let f = bsd_option.base.underlying_price.value();
        let k = bsd_option.base.strike_price;
        let r = bsd_option.base.risk_free_rate();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.vol_surface.vol(k, f, t);
        let settlement = bsd_option
            .base
            .futures_settlement
            .expect("black76 pricer called on a non-futures option");
        (f, k, r, sigma, t, *bsd_option.payoff.put_or_call(), settlement)
    }
    fn npv_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::price(f, k, r, sig, t, pc, s)
    }
    fn delta_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::delta(f, k, r, sig, t, pc, s)
    }
    fn gamma_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, _pc, s) = Self::black76_inputs(o);
        crate::equity::black76::gamma(f, k, r, sig, t, s)
    }
    fn vega_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, _pc, s) = Self::black76_inputs(o);
        crate::equity::black76::vega(f, k, r, sig, t, s)
    }
    fn theta_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::theta(f, k, r, sig, t, pc, s)
    }
    fn rho_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::rho(f, k, r, sig, t, pc, s)
    }
    fn vanna_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, _pc, s) = Self::black76_inputs(o);
        crate::equity::black76::vanna(f, k, r, sig, t, s)
    }
    fn charm_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::charm(f, k, r, sig, t, pc, s)
    }
    fn gamma_p_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, pc, s) = Self::black76_inputs(o);
        crate::equity::black76::gamma_p(f, k, r, sig, t, pc, s)
    }
    fn zomma_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, _pc, s) = Self::black76_inputs(o);
        crate::equity::black76::zomma(f, k, r, sig, t, s)
    }
    fn volga_black76(&self, o: &EquityOption) -> f64 {
        let (f, k, r, sig, t, _pc, s) = Self::black76_inputs(o);
        crate::equity::black76::volga(f, k, r, sig, t, s)
    }
    fn npv_vanilla(&self, bsd_option: &EquityOption) -> f64 {

        let n_d1 = norm_cdf(bsd_option.base.d1());
        let n_d2 = norm_cdf(bsd_option.base.d2());
        let df_d = exp(-bsd_option.base.carry_yield() * bsd_option.time_to_maturity());
        let df_r = bsd_option.base.maturity_discount_factor();
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {bsd_option.base.effective_spot()*n_d1 *df_d
                -bsd_option.base.strike_price*n_d2*df_r
            }
            PutOrCall::Put => {bsd_option.base.strike_price*norm_cdf(-bsd_option.base.d2())*df_r-
                bsd_option.base.effective_spot()*norm_cdf(-bsd_option.base.d1()) *df_d
                }

        }
    }
    fn delta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // spot delta: e^{-qT} N(d1) for a call, e^{-qT}(N(d1)-1) for a put
        let n_d1 = norm_cdf(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.carry_yield() * bsd_option.time_to_maturity());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {n_d1 * df_d }
            PutOrCall::Put => {(n_d1-1.0) * df_d }
        }
    }
    fn gamma_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // e^{-qT} dN(d1) / (S sigma sqrt(T))
        let dn_d1 = norm_pdf(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.carry_yield() * bsd_option.time_to_maturity());
        let var_sqrt = bsd_option.base.volatility() * (bsd_option.time_to_maturity().sqrt());
        dn_d1 * df_d / (bsd_option.base.effective_spot() * var_sqrt)
    }
    fn vega_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // S e^{-qT} dN(d1) sqrt(T)
        let dn_d1 = norm_pdf(bsd_option.base.d1());
        let df_d = exp(-bsd_option.base.carry_yield() * bsd_option.time_to_maturity());
        let df_S = bsd_option.base.effective_spot() * df_d;
        let vega = df_S * dn_d1 * bsd_option.time_to_maturity().sqrt();
        vega
    }
    fn theta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
        // put:  -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
        let q = bsd_option.base.carry_yield();
        let r = bsd_option.base.risk_free_rate();
        let k = bsd_option.base.strike_price;
        let dn_d1 = norm_pdf(bsd_option.base.d1());
        let n_d1 = norm_cdf(bsd_option.base.d1());
        let n_d2 = norm_cdf(bsd_option.base.d2());
        let df_d = exp(-q * bsd_option.time_to_maturity());
        let df_r = bsd_option.base.maturity_discount_factor();
        let df_S = bsd_option.base.effective_spot() * df_d;
        let t1 = -df_S * dn_d1 * bsd_option.base.volatility()
            / (2.0 * bsd_option.time_to_maturity().sqrt());

        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                t1 + q * df_S * n_d1 - r * k * df_r * n_d2
            }
            PutOrCall::Put => {
                t1 - q * df_S * norm_cdf(-bsd_option.base.d1()) + r * k * df_r * norm_cdf(-bsd_option.base.d2())
            }
        }
    }
    fn rho_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: K T e^{-rT} N(d2); put: -K T e^{-rT} N(-d2)
        let n_d2 = norm_cdf(bsd_option.base.d2());
        let df_r = bsd_option.base.maturity_discount_factor();
        let r1 = bsd_option.time_to_maturity()*bsd_option.base.strike_price;
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                r1*n_d2*df_r
            }
            PutOrCall::Put => {-r1*norm_cdf(-bsd_option.base.d2())*df_r
            }

        }
    }

    /// Analytic price with bumpable spot, volatility, rate and expiry.  This
    /// supports the cross-Greeks of payoffs whose first-order formulas are
    /// deliberately implemented by bump-and-reprice, and the portfolio
    /// PnL-attribution reprice.
    pub(crate) fn price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        match bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => bs_price(
                bsd_option.base.effective_spot() + ds,
                bsd_option.base.strike_price,
                bsd_option.base.risk_free_rate() + dr,
                bsd_option.base.carry_yield(),
                bsd_option.base.volatility() + dsigma,
                bsd_option.time_to_maturity() + dt_shift,
                *bsd_option.payoff.put_or_call(),
            ),
            PayoffType::Binary => Self::binary_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Barrier => Self::barrier_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Asian => Self::asian_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::ForwardStart => Self::forward_start_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Lookback => Self::lookback_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            _ => panic!("cross-Greeks are not available for this analytic payoff"),
        }
    }
    fn vanna_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hs = bsd_option.base.underlying_price.value() * 1e-4;
        let hv = 1e-4;
        (Self::price_with(bsd_option, hs, hv, 0.0, 0.0)
            - Self::price_with(bsd_option, -hs, hv, 0.0, 0.0)
            - Self::price_with(bsd_option, hs, -hv, 0.0, 0.0)
            + Self::price_with(bsd_option, -hs, -hv, 0.0, 0.0))
            / (4.0 * hs * hv)
    }
    fn charm_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hs = bsd_option.base.underlying_price.value() * 1e-4;
        let ht = (1.0 / 365.0_f64).min(0.5 * bsd_option.time_to_maturity());
        -(Self::price_with(bsd_option, hs, 0.0, 0.0, ht)
            - Self::price_with(bsd_option, -hs, 0.0, 0.0, ht)
            - Self::price_with(bsd_option, hs, 0.0, 0.0, -ht)
            + Self::price_with(bsd_option, -hs, 0.0, 0.0, -ht))
            / (4.0 * hs * ht)
    }
    fn zomma_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hv = 1e-4;
        let hs = bsd_option.base.underlying_price.value() * 1e-3;
        let gamma_at_vol = |dsigma: f64| {
            (Self::price_with(bsd_option, hs, dsigma, 0.0, 0.0)
                - 2.0 * Self::price_with(bsd_option, 0.0, dsigma, 0.0, 0.0)
                + Self::price_with(bsd_option, -hs, dsigma, 0.0, 0.0))
                / (hs * hs)
        };
        (gamma_at_vol(hv) - gamma_at_vol(-hv)) / (2.0 * hv)
    }
    /// Volga as the second price derivative in volatility (`d2 V / d sigma2`).
    fn volga_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hv = 1e-3;
        (Self::price_with(bsd_option, 0.0, hv, 0.0, 0.0)
            - 2.0 * Self::price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
            + Self::price_with(bsd_option, 0.0, -hv, 0.0, 0.0))
            / (hv * hv)
    }

    // ── Binary (digital) options ───────────────────────────────────────
    // cash-or-nothing:  cash * e^{-rT} N(+-d2)
    // asset-or-nothing: S e^{-qT} N(+-d1)
    // All asset-or-nothing Greeks are implemented directly (not via the
    // vanilla replication identity), so the replication tests are a real
    // cross-check.

    fn binary_details(bsd_option: &EquityOption) -> (BinaryType, f64) {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<BinaryPayoff>()
            .expect("payoff of kind Binary must be a BinaryPayoff");
        (payoff.binary_type, payoff.cash)
    }

    fn binary_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let s = bsd_option.base.effective_spot() + ds;
        let k = bsd_option.base.strike_price;
        let r = bsd_option.base.risk_free_rate() + dr;
        let q = bsd_option.base.carry_yield();
        let sigma = bsd_option.base.volatility() + dsigma;
        let t = bsd_option.time_to_maturity() + dt_shift;
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        match (binary_type, bsd_option.payoff.put_or_call()) {
            (BinaryType::CashOrNothing, PutOrCall::Call) => cash * (-r * t).exp() * norm_cdf(d2),
            (BinaryType::CashOrNothing, PutOrCall::Put) => cash * (-r * t).exp() * norm_cdf(-d2),
            (BinaryType::AssetOrNothing, PutOrCall::Call) => s * (-q * t).exp() * norm_cdf(d1),
            (BinaryType::AssetOrNothing, PutOrCall::Put) => s * (-q * t).exp() * norm_cdf(-d1),
        }
    }

    fn npv_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let df_r = bsd_option.base.maturity_discount_factor();
        let df_q = exp(-bsd_option.base.carry_yield() * bsd_option.time_to_maturity());
        let s = bsd_option.base.effective_spot();
        match (binary_type, bsd_option.payoff.put_or_call()) {
            (BinaryType::CashOrNothing, PutOrCall::Call) => cash * df_r * norm_cdf(bsd_option.base.d2()),
            (BinaryType::CashOrNothing, PutOrCall::Put) => cash * df_r * norm_cdf(-bsd_option.base.d2()),
            (BinaryType::AssetOrNothing, PutOrCall::Call) => s * df_q * norm_cdf(bsd_option.base.d1()),
            (BinaryType::AssetOrNothing, PutOrCall::Put) => s * df_q * norm_cdf(-bsd_option.base.d1()),
        }
    }
    fn delta_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let s = bsd_option.base.effective_spot();
        let vol_sqrt_t = sigma * t.sqrt();
        match binary_type {
            BinaryType::CashOrNothing => {
                // +- cash e^{-rT} dN(d2) / (S sigma sqrt(T))
                let df_r = bsd_option.base.maturity_discount_factor();
                let delta_call = cash * df_r * norm_pdf(bsd_option.base.d2()) / (s * vol_sqrt_t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => delta_call,
                    PutOrCall::Put => -delta_call,
                }
            }
            BinaryType::AssetOrNothing => {
                // e^{-qT} (N(+-d1) +- dN(d1)/(sigma sqrt(T)))
                let df_q = exp(-bsd_option.base.dividend_yield * t);
                let d1 = bsd_option.base.d1();
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => df_q * (norm_cdf(d1) + norm_pdf(d1) / vol_sqrt_t),
                    PutOrCall::Put => df_q * (norm_cdf(-d1) - norm_pdf(d1) / vol_sqrt_t),
                }
            }
        }
    }
    fn gamma_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let s = bsd_option.base.effective_spot();
        let vol_sqrt_t = sigma * t.sqrt();
        let gamma_call = match binary_type {
            BinaryType::CashOrNothing => {
                // - cash e^{-rT} dN(d2) d1 / (S^2 sigma^2 T)
                let df_r = bsd_option.base.maturity_discount_factor();
                -cash * df_r * norm_pdf(bsd_option.base.d2()) * bsd_option.base.d1()
                    / (s * s * sigma * sigma * t)
            }
            BinaryType::AssetOrNothing => {
                // e^{-qT} dN(d1) (1 - d1/(sigma sqrt(T))) / (S sigma sqrt(T))
                let df_q = exp(-bsd_option.base.dividend_yield * t);
                let d1 = bsd_option.base.d1();
                df_q * norm_pdf(d1) * (1.0 - d1 / vol_sqrt_t) / (s * vol_sqrt_t)
            }
        };
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => gamma_call,
            PutOrCall::Put => -gamma_call,
        }
    }
    fn vega_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let s = bsd_option.base.effective_spot();
        let vega_call = match binary_type {
            BinaryType::CashOrNothing => {
                // - cash e^{-rT} dN(d2) d1 / sigma
                let df_r = bsd_option.base.maturity_discount_factor();
                -cash * df_r * norm_pdf(bsd_option.base.d2()) * bsd_option.base.d1() / sigma
            }
            BinaryType::AssetOrNothing => {
                // - S e^{-qT} dN(d1) d2 / sigma
                let df_q = exp(-bsd_option.base.dividend_yield * t);
                -s * df_q * norm_pdf(bsd_option.base.d1()) * bsd_option.base.d2() / sigma
            }
        };
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => vega_call,
            PutOrCall::Put => -vega_call,
        }
    }
    fn theta_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let r = bsd_option.base.risk_free_rate();
        let q = bsd_option.base.carry_yield();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let s = bsd_option.base.effective_spot();
        match binary_type {
            BinaryType::CashOrNothing => {
                // dd2/dT = (r - q - sigma^2/2)/(sigma sqrt(T)) - d2/(2T)
                let df_r = bsd_option.base.maturity_discount_factor();
                let d2 = bsd_option.base.d2();
                let dd2_dt = (r - q - 0.5 * sigma * sigma) / (sigma * t.sqrt()) - d2 / (2.0 * t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => cash * (r * df_r * norm_cdf(d2) - df_r * norm_pdf(d2) * dd2_dt),
                    PutOrCall::Put => cash * (r * df_r * norm_cdf(-d2) + df_r * norm_pdf(d2) * dd2_dt),
                }
            }
            BinaryType::AssetOrNothing => {
                // dd1/dT = (r - q + sigma^2/2)/(sigma sqrt(T)) - d1/(2T)
                let df_q = exp(-q * t);
                let d1 = bsd_option.base.d1();
                let dd1_dt = (r - q + 0.5 * sigma * sigma) / (sigma * t.sqrt()) - d1 / (2.0 * t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => q * s * df_q * norm_cdf(d1) - s * df_q * norm_pdf(d1) * dd1_dt,
                    PutOrCall::Put => q * s * df_q * norm_cdf(-d1) + s * df_q * norm_pdf(d1) * dd1_dt,
                }
            }
        }
    }
    fn rho_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.base.volatility();
        let s = bsd_option.base.effective_spot();
        match binary_type {
            BinaryType::CashOrNothing => {
                let df_r = bsd_option.base.maturity_discount_factor();
                let d2 = bsd_option.base.d2();
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => cash * (-t * df_r * norm_cdf(d2) + df_r * norm_pdf(d2) * t.sqrt() / sigma),
                    PutOrCall::Put => cash * (-t * df_r * norm_cdf(-d2) - df_r * norm_pdf(d2) * t.sqrt() / sigma),
                }
            }
            BinaryType::AssetOrNothing => {
                // +- S e^{-qT} dN(d1) sqrt(T)/sigma
                let df_q = exp(-bsd_option.base.dividend_yield * t);
                let rho_call = s * df_q * norm_pdf(bsd_option.base.d1()) * t.sqrt() / sigma;
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => rho_call,
                    PutOrCall::Put => -rho_call,
                }
            }
        }
    }

    // ── Barrier options (Reiner-Rubinstein) ────────────────────────────
    // NPV is the closed form; Greeks are central-difference bumps of it
    // (the standard approach — the analytic derivatives are long and easy
    // to get wrong, and near the barrier bumped Greeks are what desks use).

    /// Reprice the barrier with additive bumps to (spot, vol, rate, expiry).
    fn barrier_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<BarrierPayoff>()
            .expect("payoff of kind Barrier must be a BarrierPayoff");
        let s = bsd_option.base.effective_spot() + ds;
        let k = bsd_option.base.strike_price;
        let r = bsd_option.base.risk_free_rate() + dr;
        let q = bsd_option.base.carry_yield();
        let sigma = bsd_option.base.volatility() + dsigma;
        let t = bsd_option.time_to_maturity() + dt_shift;
        let pc = *bsd_option.payoff.put_or_call();
        match payoff.barrier2 {
            Some(b2) => {
                assert!(
                    payoff.rebate == 0.0,
                    "double-barrier rebates are not supported analytically; use MonteCarlo                      (rebate at expiry)"
                );
                let (lo, hi) = (payoff.barrier.min(b2), payoff.barrier.max(b2));
                barrier::double_barrier_price(s, k, lo, hi, r, q, sigma, t, payoff.knock, pc)
            }
            None => {
                let timing = if payoff.rebate_at_hit {
                    barrier::RebateTiming::AtHit
                } else {
                    barrier::RebateTiming::AtExpiry
                };
                barrier::barrier_price_with_rebate(
                    s, k, payoff.barrier, payoff.rebate, r, q, sigma, t,
                    payoff.direction, payoff.knock, timing, pc,
                )
            }
        }
    }
    fn npv_barrier(&self, bsd_option: &EquityOption) -> f64 {
        Self::barrier_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-4;
        (Self::barrier_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::barrier_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-3;
        (Self::barrier_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::barrier_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
            + Self::barrier_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::barrier_price_with(bsd_option, 0.0, h, 0.0, 0.0)
            - Self::barrier_price_with(bsd_option, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * bsd_option.time_to_maturity());
        -(Self::barrier_price_with(bsd_option, 0.0, 0.0, 0.0, h)
            - Self::barrier_price_with(bsd_option, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::barrier_price_with(bsd_option, 0.0, 0.0, h, 0.0)
            - Self::barrier_price_with(bsd_option, 0.0, 0.0, -h, 0.0))
            / (2.0 * h)
    }

    // ── Asian options ──────────────────────────────────────────────────
    // geometric average price: exact closed form (continuous averaging)
    // arithmetic average price: Turnbull-Wakeman approximation
    // floating strike: Monte Carlo only
    // Greeks by central-difference bumps, like barriers.

    fn asian_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<AsianPayoff>()
            .expect("payoff of kind Asian must be an AsianPayoff");
        let s = bsd_option.base.effective_spot() + ds;
        let k = bsd_option.base.strike_price;
        let r = bsd_option.base.risk_free_rate() + dr;
        let q = bsd_option.base.carry_yield();
        let sigma = bsd_option.base.volatility() + dsigma;
        let t = bsd_option.time_to_maturity() + dt_shift;
        let pc = *bsd_option.payoff.put_or_call();
        match (payoff.strike_type, payoff.averaging) {
            (AsianStrikeType::FixedStrike, AveragingType::Geometric) => {
                asian::geometric_asian_price(s, k, r, q, sigma, t, None, pc)
            }
            (AsianStrikeType::FixedStrike, AveragingType::Arithmetic) => {
                asian::turnbull_wakeman_price(s, k, r, q, sigma, t, pc)
            }
            (AsianStrikeType::FloatingStrike, AveragingType::Geometric) => {
                // exact exchange-option closed form (continuous averaging)
                asian::geometric_average_strike_price(s, r, q, sigma, t, None, pc)
            }
            (AsianStrikeType::FloatingStrike, AveragingType::Arithmetic) => {
                // Henderson-Wojakowski symmetry + Turnbull-Wakeman
                asian::turnbull_wakeman_average_strike_price(s, r, q, sigma, t, pc)
            }
        }
    }
    /// Continuous-monitoring lookback closed forms (fresh options: the
    /// running extremum is the current spot).
    fn lookback_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<crate::equity::vanilla_option::LookbackPayoff>()
            .expect("payoff of kind Lookback must be a LookbackPayoff");
        let anchor = bsd_option.base.effective_spot();
        let s = anchor + ds;
        let k = bsd_option.base.strike_price;
        let r = bsd_option.base.risk_free_rate() + dr;
        let q = bsd_option.base.carry_yield();
        let sigma = bsd_option.base.volatility() + dsigma;
        let t = bsd_option.time_to_maturity() + dt_shift;
        let pc = *bsd_option.payoff.put_or_call();
        // the running extremum is a historical observable: spot bumps hold
        // it at the unbumped level (clamped so min <= s / max >= s stays
        // true), giving the market-standard hedge delta rather than the
        // fresh-reissue homogeneity delta
        use crate::equity::vanilla_option::LookbackType;
        match (payoff.lookback_type, pc) {
            (LookbackType::FloatingStrike, PutOrCall::Call) => {
                crate::equity::lookback::floating_strike_lookback_price(
                    s, anchor.min(s), r, q, sigma, t, pc,
                )
            }
            (LookbackType::FloatingStrike, PutOrCall::Put) => {
                crate::equity::lookback::floating_strike_lookback_price(
                    s, anchor.max(s), r, q, sigma, t, pc,
                )
            }
            (LookbackType::FixedStrike, PutOrCall::Call) => {
                crate::equity::lookback::fixed_strike_lookback_price(
                    s, k, anchor.max(s), r, q, sigma, t, pc,
                )
            }
            (LookbackType::FixedStrike, PutOrCall::Put) => {
                crate::equity::lookback::fixed_strike_lookback_price(
                    s, k, anchor.min(s), r, q, sigma, t, pc,
                )
            }
        }
    }

    fn delta_lookback(&self, o: &EquityOption) -> f64 {
        let h = o.base.underlying_price.value() * 1e-4;
        (Self::lookback_price_with(o, h, 0.0, 0.0, 0.0)
            - Self::lookback_price_with(o, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_lookback(&self, o: &EquityOption) -> f64 {
        let h = o.base.underlying_price.value() * 1e-3;
        (Self::lookback_price_with(o, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::lookback_price_with(o, 0.0, 0.0, 0.0, 0.0)
            + Self::lookback_price_with(o, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_lookback(&self, o: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::lookback_price_with(o, 0.0, h, 0.0, 0.0)
            - Self::lookback_price_with(o, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_lookback(&self, o: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * o.time_to_maturity());
        -(Self::lookback_price_with(o, 0.0, 0.0, 0.0, h)
            - Self::lookback_price_with(o, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_lookback(&self, o: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::lookback_price_with(o, 0.0, 0.0, h, 0.0)
            - Self::lookback_price_with(o, 0.0, 0.0, -h, 0.0))
            / (2.0 * h)
    }

    fn npv_asian(&self, bsd_option: &EquityOption) -> f64 {
        Self::asian_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-4;
        (Self::asian_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::asian_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-3;
        (Self::asian_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::asian_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
            + Self::asian_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::asian_price_with(bsd_option, 0.0, h, 0.0, 0.0)
            - Self::asian_price_with(bsd_option, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * bsd_option.time_to_maturity());
        -(Self::asian_price_with(bsd_option, 0.0, 0.0, 0.0, h)
            - Self::asian_price_with(bsd_option, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::asian_price_with(bsd_option, 0.0, 0.0, h, 0.0)
            - Self::asian_price_with(bsd_option, 0.0, 0.0, -h, 0.0))
            / (2.0 * h)
    }

    // -- Forward-start options (Rubinstein closed form, GBM) ------------

    fn forward_start_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<crate::equity::forward_start_option::ForwardStartPayoff>()
            .expect("payoff of kind ForwardStart must be a ForwardStartPayoff");
        let t = bsd_option.time_to_maturity() + dt_shift;
        crate::equity::forward_start_option::forward_start_price(
            bsd_option.base.effective_spot() + ds,
            payoff.strike_fraction,
            bsd_option.base.risk_free_rate() + dr,
            bsd_option.base.carry_yield(),
            bsd_option.base.volatility() + dsigma,
            payoff.start_fraction * t,
            t,
            *bsd_option.payoff.put_or_call(),
        )
    }
    fn npv_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        Self::forward_start_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-4;
        (Self::forward_start_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::forward_start_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.base.underlying_price.value() * 1e-3;
        (Self::forward_start_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::forward_start_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
            + Self::forward_start_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::forward_start_price_with(bsd_option, 0.0, h, 0.0, 0.0)
            - Self::forward_start_price_with(bsd_option, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.25 * bsd_option.time_to_maturity());
        -(Self::forward_start_price_with(bsd_option, 0.0, 0.0, 0.0, h)
            - Self::forward_start_price_with(bsd_option, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::forward_start_price_with(bsd_option, 0.0, 0.0, h, 0.0)
            - Self::forward_start_price_with(bsd_option, 0.0, 0.0, -h, 0.0))
            / (2.0 * h)
    }

}


/// Black-Scholes price of a European vanilla as a pure function of its
/// inputs (no option object needed).
pub fn bs_price(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, put_or_call: PutOrCall) -> f64 {
    if t <= 0.0 || sigma <= 0.0 {
        return match put_or_call {
            PutOrCall::Call => (s * exp(-q * t) - k * exp(-r * t)).max(0.0),
            PutOrCall::Put => (k * exp(-r * t) - s * exp(-q * t)).max(0.0),
        };
    }
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    match put_or_call {
        PutOrCall::Call => s * exp(-q * t) * norm_cdf(d1) - k * exp(-r * t) * norm_cdf(d2),
        PutOrCall::Put => k * exp(-r * t) * norm_cdf(-d2) - s * exp(-q * t) * norm_cdf(-d1),
    }
}

/// Black-Scholes vega as a pure function (per unit of vol).
pub fn bs_vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    s * exp(-q * t) * norm_pdf(d1) * sqrt_t
}

/// Black-Scholes vanna, the change in spot delta per unit of volatility.
pub fn bs_vanna(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    exp(-q * t) * norm_pdf(d1) * (sqrt_t - d1 / sigma)
}

/// Black-Scholes charm, the change in spot delta per year of calendar time.
pub fn bs_charm(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    put_or_call: PutOrCall,
) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let df_q = exp(-q * t);
    let d1_dt = (r - q + 0.5 * sigma * sigma) / (sigma * sqrt_t) - d1 / (2.0 * t);
    let delta_component = match put_or_call {
        PutOrCall::Call => norm_cdf(d1),
        PutOrCall::Put => norm_cdf(d1) - 1.0,
    };
    q * df_q * delta_component - df_q * norm_pdf(d1) * d1_dt
}

/// Black-Scholes zomma, the change in spot gamma per unit of volatility.
pub fn bs_zomma(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let gamma = exp(-q * t) * norm_pdf(d1) / (s * sigma * sqrt_t);
    gamma * (d1 * d2 - 1.0) / sigma
}

/// Black-Scholes volga (vomma), the change in vega per unit of volatility,
/// `vega * d1 * d2 / sigma`. Same for calls and puts (parity is
/// volatility-independent); negative at the money, positive in the wings.
pub fn bs_volga(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    let vega = s * exp(-q * t) * norm_pdf(d1) * sqrt_t;
    vega * d1 * d2 / sigma
}

const IMPLIED_VOL_MIN: f64 = 1e-4;
const IMPLIED_VOL_MAX: f64 = 5.0;

/// Implied Black-Scholes volatility for a European vanilla price.
///
/// Safeguarded Newton: full Newton steps while they stay inside the current
/// bisection bracket `[1e-4, 5.0]`, bisection otherwise, so it converges for
/// deep in/out-of-the-money quotes where raw Newton diverges. Prices outside
/// the arbitrage bounds return an error.
pub fn implied_vol_from_price(
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    t: f64,
    target: f64,
    put_or_call: PutOrCall,
) -> Result<f64, String> {
    if t <= 0.0 {
        return Err("option is expired".to_string());
    }
    let lower_bound = bs_price(s, k, r, q, 0.0, t, put_or_call);
    let upper_bound = match put_or_call {
        PutOrCall::Call => s * exp(-q * t),
        PutOrCall::Put => k * exp(-r * t),
    };
    if target < lower_bound - 1e-12 || target > upper_bound + 1e-12 {
        return Err(format!(
            "price {target} violates arbitrage bounds [{lower_bound}, {upper_bound}]"
        ));
    }

    let (lo, hi) = (IMPLIED_VOL_MIN, IMPLIED_VOL_MAX);
    if bs_price(s, k, r, q, lo, t, put_or_call) > target {
        return Ok(lo); // at or below the vol floor
    }
    if bs_price(s, k, r, q, hi, t, put_or_call) < target {
        return Err(format!("implied vol above {IMPLIED_VOL_MAX}"));
    }

    // price is increasing in vol (vega > 0), the shape newton_safeguarded
    // wants; the answer is best-effort even if the tolerance isn't met
    let tol = 1e-12 * target.max(1.0);
    let root = crate::core::solvers::Solver1d::new(tol, 100).newton_safeguarded(
        |sigma| bs_price(s, k, r, q, sigma, t, put_or_call) - target,
        |sigma| bs_vega(s, k, r, q, sigma, t),
        lo,
        hi,
        0.5,
    );
    Ok(root.x)
}

pub fn option_pricing() {
    println!("Welcome to the Black-Scholes Option pricer.");
    print!(">>");
    println!(" What is the current price of the underlying asset?");
    print!(">>");
    let mut curr_price = String::new();
    io::stdin()
        .read_line(&mut curr_price)
        .expect("Failed to read line");
    println!(" Do you want a call option ('C') or a put option ('P') ?");
    print!(">>");
    let mut side_input = String::new();
    io::stdin()
        .read_line(&mut side_input)
        .expect("Failed to read line");
    let side: PutOrCall;
    match side_input.trim() {
        "C" | "c" | "Call" | "call" => side = PutOrCall::Call,
        "P" | "p" | "Put" | "put" => side = PutOrCall::Put,
        _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
    }
    println!("Stike price:");
    print!(">>");
    let mut strike = String::new();
    io::stdin()
        .read_line(&mut strike)
        .expect("Failed to read line");
    println!("Expected annualized volatility in %:");
    println!("E.g.: Enter 50% chance as 0.50 ");
    print!(">>");
    let mut vol = String::new();
    io::stdin()
        .read_line(&mut vol)
        .expect("Failed to read line");

    println!("Risk-free rate in %:");
    print!(">>");
    let mut rf = String::new();
    io::stdin().read_line(&mut rf).expect("Failed to read line");
    println!(" Maturity date in YYYY-MM-DD format:");

    let mut expiry = String::new();
    println!("E.g.: Enter 2020-12-31 for 31st December 2020");
    print!(">>");
    io::stdin()
        .read_line(&mut expiry)
        .expect("Failed to read line");
    println!("{:?}", expiry.trim());
    let _d = expiry.trim();
    let future_date = NaiveDate::parse_from_str(&_d, "%Y-%m-%d").expect("Invalid date format");
    //println!("{:?}", future_date);
    println!("Dividend yield on this stock:");
    print!(">>");
    let mut div = String::new();
    io::stdin()
        .read_line(&mut div)
        .expect("Failed to read line");

    let valuation_date = Local::now().date_naive();
    let discount_curve = YieldCurve::flat(
        rf.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
        Compounding::Continuous,
    )
    .expect("Invalid risk free rate");
    let vol_surface = VolSurface::flat(
        vol.trim().parse::<f64>().unwrap(),
        valuation_date,
        DayCountConvention::Act365,
    )
    .expect("Invalid volatility");
    let curr_quote = Quote::new( curr_price.trim().parse::<f64>().unwrap());
    let option = EquityOptionBase {

        symbol:"ABC".to_string(),
        currency: None,
        exchange:None,
        name: None,
        cusip: None,
        isin: None,
        settlement_type: Some("ABC".to_string()),
        entry_price: 0.0,
        long_short: LongShort::LONG,
        underlying_price: curr_quote,
        current_price: Quote::new(0.0),
        strike_price: strike.trim().parse::<f64>().unwrap(),
        vol_surface,
        maturity_date: future_date,
        discount_curve,
        dividend_yield: div.trim().parse::<f64>().unwrap(),
        borrow_cost: 0.0,
        cash_dividends: vec![],
        futures_settlement: None,
        valuation_date,
        multiplier: 1.0,
    };
    //println!("{:?}", option.time_to_maturity());
    let payoff = Box::new(VanillaPayoff{put_or_call:side,
                                    exercise_style:ContractStyle::European});
    let option = EquityOption {
        base: option,
        payoff:payoff,
        engine:Engine::BlackScholes,
        mc: crate::equity::montecarlo::MonteCarloConfig::default(),
        fd: crate::equity::finite_difference::FdConfig::default(),
        heston: None
    };
    println!("Theoretical Price ${}", option.npv());
    println!("Premium at risk ${}", option.get_premium_at_risk());
    println!("Delta {}", option.delta());
    println!("Gamma {}", option.gamma());
    println!("Vega {}", option.vega() * 0.01);
    println!("Theta {}", option.theta() * (1.0 / 365.0));
    println!("Rho {}", option.rho() * 0.01);
    let mut wait = String::new();
    io::stdin()
        .read_line(&mut wait)
        .expect("Failed to read line");
}
pub fn implied_volatility(){}
// pub fn implied_volatility() {
//     println!("Welcome to the Black-Scholes Option pricer.");
//     println!("(Step 1/7) What is the current price of the underlying asset?");
//     let mut curr_price = String::new();
//     io::stdin()
//         .read_line(&mut curr_price)
//         .expect("Failed to read line");
//
//     println!("(Step 2/7) Do you want a call option ('C') or a put option ('P') ?");
//     let mut side_input = String::new();
//     io::stdin()
//         .read_line(&mut side_input)
//         .expect("Failed to read line");
//
//     let side: OptionType;
//     match side_input.trim() {
//         "C" | "c" | "Call" | "call" => side = OptionType::Call,
//         "P" | "p" | "Put" | "put" => side = OptionType::Put,
//         _ => panic!("Invalide side argument! Side has to be either 'C' or 'P'."),
//     }
//
//     println!("Stike price:");
//     let mut strike = String::new();
//     io::stdin()
//         .read_line(&mut strike)
//         .expect("Failed to read line");
//
//     println!("What is option price:");
//     let mut option_price = String::new();
//     io::stdin()
//         .read_line(&mut option_price)
//         .expect("Failed to read line");
//
//     println!("Risk-free rate in %:");
//     let mut rf = String::new();
//     io::stdin().read_line(&mut rf).expect("Failed to read line");
//
//     println!(" Maturity date in YYYY-MM-DD format:");
//     let mut expiry = String::new();
//     io::stdin()
//         .read_line(&mut expiry)
//         .expect("Failed to read line");
//     let future_date = NaiveDate::parse_from_str(&expiry.trim(), "%Y-%m-%d").expect("Invalid date format");
//     println!("Dividend yield on this stock:");
//     let mut div = String::new();
//     io::stdin()
//         .read_line(&mut div)
//         .expect("Failed to read line");
//
//     //let ts = YieldTermStructure{
//     //    date: vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0],
//     //    rates: vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12]
//     //};
//     let date =  vec![0.01,0.02,0.05,0.1,0.5,1.0,2.0,3.0];
//     let rates = vec![0.01,0.02,0.05,0.07,0.08,0.1,0.11,0.12];
//     let ts = YieldTermStructure::new(date,rates);
//     let curr_quote = Quote::new( curr_price.trim().parse::<f64>().unwrap());
//     let sim = Some(10000);
//     let mut option = EquityOption {
//         option_type: side,
//         transection: Transection::Buy,
//         underlying_price: curr_quote,
//         current_price: Quote::new(0.0),
//         strike_price: strike.trim().parse::<f64>().unwrap(),
//         volatility: 0.20,
//         maturity_date: future_date,
//         risk_free_rate: rf.trim().parse::<f64>().unwrap(),
//         dividend_yield: div.trim().parse::<f64>().unwrap(),
//         transection_price: 0.0,
//         term_structure: ts,
//         engine: Engine::BlackScholes,
//         simulation:sim,
//         //style:Option::from("European".to_string()),
//         style: ContractStyle::European,
//         valuation_date: Local::today().naive_utc(),
//     };
//     option.set_risk_free_rate();
//     println!("Implied Volatility  {}%", 100.0*option.imp_vol(option_price.trim().parse::<f64>().unwrap()));
//
//     let mut div1 = String::new();
//     io::stdin()
//         .read_line(&mut div)
//         .expect("Failed to read line");
// }


#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
    use crate::core::daycount::DayCountConvention;
    use crate::core::utils::ContractStyle;

    /// S=100, K=100, sigma=30%, q=0, T=1y (2026-01-01 -> 2027-01-01, Act/365).
    fn test_option_with(payoff: Box<dyn Payoff>, curve: YieldCurve) -> EquityOption {
        let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let base = EquityOptionBase {
            symbol: "TEST".to_string(),
            currency: None,
            exchange: None,
            name: None,
            cusip: None,
            isin: None,
            settlement_type: None,
            underlying_price: Quote::new(100.0),
            current_price: Quote::new(0.0),
            strike_price: 100.0,
            dividend_yield: 0.0,
            borrow_cost: 0.0,
            cash_dividends: vec![],
        futures_settlement: None,
            vol_surface: VolSurface::flat(0.3, valuation_date, DayCountConvention::Act365)
                .unwrap(),
            maturity_date: NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(),
            valuation_date,
            discount_curve: curve,
            entry_price: 0.0,
            long_short: LongShort::LONG,
            multiplier: 1.0,
        };
        EquityOption {
            base,
            payoff,
            engine: Engine::BlackScholes,
            mc: crate::equity::montecarlo::MonteCarloConfig::default(),
            fd: crate::equity::finite_difference::FdConfig::default(),
            heston: None,
        }
    }

    fn test_option(put_or_call: PutOrCall, curve: YieldCurve) -> EquityOption {
        test_option_with(
            Box::new(VanillaPayoff { put_or_call, exercise_style: ContractStyle::European }),
            curve,
        )
    }

    fn binary_option_of(
        put_or_call: PutOrCall,
        binary_type: BinaryType,
        cash: f64,
    ) -> EquityOption {
        test_option_with(
            Box::new(BinaryPayoff {
                put_or_call,
                exercise_style: ContractStyle::European,
                binary_type,
                cash,
            }),
            flat_5pct(),
        )
    }

    fn binary_option(put_or_call: PutOrCall) -> EquityOption {
        binary_option_of(put_or_call, BinaryType::CashOrNothing, 1.0)
    }

    fn flat_5pct() -> YieldCurve {
        YieldCurve::flat(
            0.05,
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    // Golden values computed independently (erf-based reference implementation)
    #[test]
    fn golden_call_npv_and_greeks() {
        let option = test_option(PutOrCall::Call, flat_5pct());
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.delta(), 0.6242517279, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        assert_approx_eq!(option.theta(), -8.1011898970, 1e-8);
        assert_approx_eq!(option.rho(), 48.1939180046, 1e-8);
    }

    #[test]
    fn golden_put_npv_and_greeks() {
        let option = test_option(PutOrCall::Put, flat_5pct());
        assert_approx_eq!(option.npv(), 9.3541972361, 1e-8);
        assert_approx_eq!(option.delta(), -0.3757482721, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        assert_approx_eq!(option.theta(), -3.3450427745, 1e-8);
        assert_approx_eq!(option.rho(), -46.9290244455, 1e-8);
    }

    #[test]
    fn vanilla_vanna_and_charm_match_mixed_price_bumps() {
        let option = test_option(PutOrCall::Call, flat_5pct());
        let s = option.base.effective_spot();
        let k = option.base.strike_price;
        let r = option.base.risk_free_rate();
        let q = option.base.carry_yield();
        let sigma = option.base.volatility();
        let t = option.time_to_maturity();
        let hs = 1e-3;
        let hv = 1e-5;
        let ht = 1e-5;
        let pc = PutOrCall::Call;
        let vanna_bump = (bs_price(s + hs, k, r, q, sigma + hv, t, pc)
            - bs_price(s - hs, k, r, q, sigma + hv, t, pc)
            - bs_price(s + hs, k, r, q, sigma - hv, t, pc)
            + bs_price(s - hs, k, r, q, sigma - hv, t, pc))
            / (4.0 * hs * hv);
        let charm_bump = -(bs_price(s + hs, k, r, q, sigma, t + ht, pc)
            - bs_price(s - hs, k, r, q, sigma, t + ht, pc)
            - bs_price(s + hs, k, r, q, sigma, t - ht, pc)
            + bs_price(s - hs, k, r, q, sigma, t - ht, pc))
            / (4.0 * hs * ht);
        assert_approx_eq!(option.vanna(), vanna_bump, 1e-6);
        assert_approx_eq!(option.charm(), charm_bump, 1e-6);
    }

    #[test]
    fn vanilla_gamma_p_and_zomma_match_definitions() {
        let option = test_option(PutOrCall::Call, flat_5pct());
        let s = option.base.underlying_price.value();
        assert_approx_eq!(option.gamma_p(), s * option.gamma() / option.delta(), 1e-12);

        let h = 1e-5;
        let k = option.base.strike_price;
        let r = option.base.risk_free_rate();
        let q = option.base.carry_yield();
        let sigma = option.base.volatility();
        let t = option.time_to_maturity();
        // zomma = d(gamma)/d(sigma); bump the analytic gamma in vol rather
        // than a price-based gamma (double price-differencing is far too
        // noisy to check against a 1e-6 tolerance)
        let gamma_at_vol = |vol: f64| {
            let d1 = ((s / k).ln() + (r - q + 0.5 * vol * vol) * t) / (vol * t.sqrt());
            (-q * t).exp() * norm_pdf(d1) / (s * vol * t.sqrt())
        };
        let zomma_bump = (gamma_at_vol(sigma + h) - gamma_at_vol(sigma - h)) / (2.0 * h);
        assert_approx_eq!(option.zomma(), zomma_bump, 1e-6);
    }

    #[test]
    fn vanilla_volga_matches_vega_bump_and_closed_form() {
        let option = test_option(PutOrCall::Call, flat_5pct());
        let s = option.base.effective_spot();
        let k = option.base.strike_price;
        let r = option.base.risk_free_rate();
        let q = option.base.carry_yield();
        let sigma = option.base.volatility();
        let t = option.time_to_maturity();

        // the analytic engine returns the closed form
        assert_approx_eq!(option.volga(), bs_volga(s, k, r, q, sigma, t), 1e-12);

        // volga = d(vega)/d(sigma); bump the analytic vega in vol
        let h = 1e-5;
        let vega_at_vol = |vol: f64| bs_vega(s, k, r, q, vol, t);
        let volga_bump = (vega_at_vol(sigma + h) - vega_at_vol(sigma - h)) / (2.0 * h);
        assert_approx_eq!(option.volga(), volga_bump, 1e-6);

        // put and call share volga (parity is volatility-independent)
        let put = test_option(PutOrCall::Put, flat_5pct());
        assert_approx_eq!(option.volga(), put.volga(), 1e-12);
    }

    #[test]
    fn volga_agrees_across_engines() {
        let analytic = test_option(PutOrCall::Call, flat_5pct()).volga();

        let mut fd = test_option(PutOrCall::Call, flat_5pct());
        fd.engine = Engine::FiniteDifference;
        assert!((fd.volga() - analytic).abs() < 0.5, "fd {} vs analytic {analytic}", fd.volga());

        let mut mc = test_option(PutOrCall::Call, flat_5pct());
        mc.engine = Engine::MonteCarlo;
        mc.mc.paths = 200_000;
        assert!((mc.volga() - analytic).abs() < 1.5, "mc {} vs analytic {analytic}", mc.volga());
    }

    #[test]
    fn put_call_parity() {
        let call = test_option(PutOrCall::Call, flat_5pct());
        let put = test_option(PutOrCall::Put, flat_5pct());
        let s = call.base.underlying_price.value();
        let k_df = call.base.strike_price * call.base.maturity_discount_factor();
        assert_approx_eq!(call.npv() - put.npv(), s - k_df, 1e-10);
    }

    // Binary golden values computed independently and bump-verified
    #[test]
    fn golden_binary_call_npv_and_greeks() {
        let option = binary_option(PutOrCall::Call);
        assert_approx_eq!(option.npv(), 0.4819391800, 1e-8);
        assert_approx_eq!(option.delta(), 0.0126477644, 1e-8);
        assert_approx_eq!(option.gamma(), -0.0001335042, 1e-8);
        assert_approx_eq!(option.vega(), -0.4005125405, 1e-8);
        assert_approx_eq!(option.theta(), 0.0209350179, 1e-8);
        assert_approx_eq!(option.rho(), 0.7828372637, 1e-8);
    }

    #[test]
    fn golden_binary_put_npv_and_greeks() {
        let option = binary_option(PutOrCall::Put);
        assert_approx_eq!(option.npv(), 0.4692902445, 1e-8);
        assert_approx_eq!(option.delta(), -0.0126477644, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0001335042, 1e-8);
        assert_approx_eq!(option.vega(), 0.4005125405, 1e-8);
        assert_approx_eq!(option.theta(), 0.0266264533, 1e-8);
        assert_approx_eq!(option.rho(), -1.7340666882, 1e-8);
    }

    #[test]
    fn binary_call_plus_put_equals_discount_factor() {
        let call = binary_option(PutOrCall::Call);
        let put = binary_option(PutOrCall::Put);
        assert_approx_eq!(call.npv() + put.npv(), call.base.maturity_discount_factor(), 1e-12);
    }

    #[test]
    fn cash_amount_scales_cash_or_nothing_linearly() {
        let unit = binary_option(PutOrCall::Call);
        let sized = binary_option_of(PutOrCall::Call, BinaryType::CashOrNothing, 1000.0);
        assert_approx_eq!(sized.npv(), 1000.0 * unit.npv(), 1e-9);
        assert_approx_eq!(sized.delta(), 1000.0 * unit.delta(), 1e-9);
        assert_approx_eq!(sized.vega(), 1000.0 * unit.vega(), 1e-9);
    }

    // Asset-or-nothing goldens computed independently and bump-verified,
    // with q = 2% so the dividend terms are exercised
    #[test]
    fn golden_asset_or_nothing_call_npv_and_greeks() {
        let mut option = binary_option_of(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0);
        option.base.dividend_yield = 0.02;
        assert_approx_eq!(option.npv(), 58.6851146135, 1e-8);
        assert_approx_eq!(option.delta(), 1.8502230631, 1e-8);
        assert_approx_eq!(option.gamma(), 0.0021056199, 1e-8);
        assert_approx_eq!(option.vega(), 6.3168595850, 1e-8);
        assert_approx_eq!(option.theta(), -3.5639423965, 1e-8);
        assert_approx_eq!(option.rho(), 126.3371917001, 1e-8);
    }

    #[test]
    fn golden_asset_or_nothing_put_npv_and_greeks() {
        let mut option = binary_option_of(PutOrCall::Put, BinaryType::AssetOrNothing, 0.0);
        option.base.dividend_yield = 0.02;
        assert_approx_eq!(option.npv(), 39.3347527172, 1e-8);
        assert_approx_eq!(option.delta(), -0.8700243898, 1e-8);
        assert_approx_eq!(option.gamma(), -0.0021056199, 1e-8);
        assert_approx_eq!(option.vega(), -6.3168595850, 1e-8);
        assert_approx_eq!(option.theta(), 5.5243397431, 1e-8);
        assert_approx_eq!(option.rho(), -126.3371917001, 1e-8);
    }

    #[test]
    fn asset_call_plus_put_equals_forward_leg() {
        let call = binary_option_of(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0);
        let put = binary_option_of(PutOrCall::Put, BinaryType::AssetOrNothing, 0.0);
        // A_c + A_p = S e^{-qT}; q = 0 in the test setup
        assert_approx_eq!(call.npv() + put.npv(), 100.0, 1e-10);
    }

    /// Replication: an asset-or-nothing call is a long vanilla call plus
    /// K cash-or-nothing calls — payoff-wise S·1{S>K} = (S-K)^+ + K·1{S>K}.
    /// Both sides are implemented independently, so this checks the closed
    /// forms (price and every Greek) against each other.
    #[test]
    fn asset_digital_replicated_by_call_plus_cash_digitals() {
        let k = 100.0;
        let asset = binary_option_of(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0);
        let vanilla = test_option(PutOrCall::Call, flat_5pct());
        let cash = binary_option_of(PutOrCall::Call, BinaryType::CashOrNothing, k);

        assert_approx_eq!(asset.npv(), vanilla.npv() + cash.npv(), 1e-10);
        assert_approx_eq!(asset.delta(), vanilla.delta() + cash.delta(), 1e-10);
        assert_approx_eq!(asset.gamma(), vanilla.gamma() + cash.gamma(), 1e-10);
        assert_approx_eq!(asset.vega(), vanilla.vega() + cash.vega(), 1e-10);
        assert_approx_eq!(asset.theta(), vanilla.theta() + cash.theta(), 1e-10);
        assert_approx_eq!(asset.rho(), vanilla.rho() + cash.rho(), 1e-10);
    }

    /// The same replication must hold on the numerical engines, which see
    /// only the payoff function.
    #[test]
    fn asset_digital_replication_holds_across_engines() {
        let k = 100.0;
        let priced = |engine: Engine, payoff: Box<dyn Payoff>| {
            let mut option = test_option_with(payoff, flat_5pct());
            option.engine = engine.clone();
            option.npv()
        };
        let asset_payoff = || -> Box<dyn Payoff> {
            Box::new(BinaryPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: ContractStyle::European,
                binary_type: BinaryType::AssetOrNothing,
                cash: 0.0,
            })
        };
        let cash_payoff = || -> Box<dyn Payoff> {
            Box::new(BinaryPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: ContractStyle::European,
                binary_type: BinaryType::CashOrNothing,
                cash: k,
            })
        };
        let vanilla_payoff = || -> Box<dyn Payoff> {
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: ContractStyle::European,
            })
        };
        for (engine, tol) in [
            (Engine::FiniteDifference, 0.01),
            (Engine::Binomial, 0.05),
            (Engine::MonteCarlo, 0.05),
        ] {
            let asset = priced(engine.clone(), asset_payoff());
            let replicated =
                priced(engine.clone(), vanilla_payoff()) + priced(engine.clone(), cash_payoff());
            assert!(
                (asset - replicated).abs() < tol,
                "{engine:?}: asset={asset} replicated={replicated}"
            );
        }
    }

    #[test]
    fn asset_digital_matches_analytic_across_engines() {
        let analytic = binary_option_of(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0).npv();
        for (engine, tol) in [
            (Engine::FiniteDifference, 0.05),
            (Engine::Binomial, 2.0), // digitals on a CRR tree oscillate; jump size is ~K
            (Engine::MonteCarlo, 0.05),
        ] {
            let mut option = binary_option_of(PutOrCall::Call, BinaryType::AssetOrNothing, 0.0);
            option.engine = engine.clone();
            let value = option.npv();
            assert!(
                (value - analytic).abs() < tol,
                "{engine:?}: {value} vs analytic {analytic}"
            );
        }
    }

    // ── Cross-engine agreement ──────────────────────────────────────────

    #[test]
    fn finite_difference_matches_analytic_vanilla() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut option = test_option(pc, flat_5pct());
            let analytic = option.npv();
            option.engine = Engine::FiniteDifference;
            let fd = option.npv();
            assert!(
                (fd - analytic).abs() < 0.01,
                "{pc:?}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn finite_difference_matches_analytic_binary() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut option = binary_option(pc);
            let analytic = option.npv();
            option.engine = Engine::FiniteDifference;
            let fd = option.npv();
            assert!(
                (fd - analytic).abs() < 0.002,
                "{pc:?}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn binomial_matches_analytic_vanilla_and_binary() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut vanilla = test_option(pc, flat_5pct());
            let analytic = vanilla.npv();
            vanilla.engine = Engine::Binomial;
            let tree = vanilla.npv();
            assert!((tree - analytic).abs() < 0.02, "vanilla {pc:?}: tree={tree} bs={analytic}");

            let mut binary = binary_option(pc);
            let analytic = binary.npv();
            binary.engine = Engine::Binomial;
            let tree = binary.npv();
            assert!((tree - analytic).abs() < 0.02, "binary {pc:?}: tree={tree} bs={analytic}");
        }
    }

    #[test]
    fn monte_carlo_matches_analytic_vanilla_and_binary() {
        // default config: Sobol low-discrepancy terminal simulation
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let mut vanilla = test_option(pc, flat_5pct());
            let analytic = vanilla.npv();
            vanilla.engine = Engine::MonteCarlo;
            let mc = vanilla.npv();
            assert!((mc - analytic).abs() < 0.02, "vanilla {pc:?}: mc={mc} bs={analytic}");

            let mut binary = binary_option(pc);
            let analytic = binary.npv();
            binary.engine = Engine::MonteCarlo;
            let mc = binary.npv();
            assert!((mc - analytic).abs() < 0.005, "binary {pc:?}: mc={mc} bs={analytic}");
        }
    }

    #[test]
    fn monte_carlo_sobol_beats_default_tolerance_and_is_reproducible() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        let first = option.npv();
        let second = option.npv();
        assert_eq!(first, second, "deterministic sampler must reproduce exactly");
        assert!((first - 14.2312547860).abs() < 0.02, "sobol mc = {first}");
    }

    #[test]
    fn monte_carlo_path_wise_starts_at_spot_and_schemes_converge() {
        // regression for the path-wise bug (paths used to start at the
        // option premium instead of the underlying spot)
        let analytic = test_option(PutOrCall::Call, flat_5pct()).npv();
        for scheme in ["exact", "euler", "milstein"] {
            let mut option = test_option(PutOrCall::Call, flat_5pct());
            option.engine = Engine::MonteCarlo;
            option.mc.scheme = scheme.parse().unwrap();
            option.mc.time_steps = 252;
            option.mc.paths = 50_000;
            let mc = option.npv();
            assert!(
                (mc - analytic).abs() < 0.35,
                "{scheme}: mc={mc} analytic={analytic}"
            );
        }
    }

    #[test]
    fn monte_carlo_greeks_match_analytic() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        // common-random-number bumps against the analytic golden values
        assert!((option.delta() - 0.6242517279).abs() < 0.01, "delta {}", option.delta());
        assert!((option.gamma() - 0.0126477644).abs() < 0.003, "gamma {}", option.gamma());
        assert!((option.vega() - 37.9432933117).abs() < 1.0, "vega {}", option.vega());
        assert!((option.theta() - -8.1011898970).abs() < 0.5, "theta {}", option.theta());
        assert!((option.rho() - 48.1939180046).abs() < 0.5, "rho {}", option.rho());
    }

    #[test]
    fn lsmc_american_put_close_to_tree_and_dominates_european() {
        let european = test_option(PutOrCall::Put, flat_5pct()).npv();
        let mut tree_option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        tree_option.engine = Engine::Binomial;
        let tree = tree_option.npv();

        let mut lsmc_option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        lsmc_option.engine = Engine::MonteCarlo;
        lsmc_option.mc.paths = 20_000;
        let lsmc = lsmc_option.npv();

        // LSMC is biased slightly low (suboptimal exercise policy) but must
        // sit between the European price and just above the tree price
        assert!(lsmc > european, "lsmc {lsmc} must exceed european {european}");
        assert!((lsmc - tree).abs() < 0.25, "lsmc={lsmc} tree={tree}");
    }

    // ── Implied vol solver ──────────────────────────────────────────────

    #[test]
    fn implied_vol_round_trips_across_strikes_and_vols() {
        let (s, r, q) = (100.0, 0.05, 0.02);
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            for k in [50.0, 80.0, 100.0, 120.0, 200.0] {
                for vol in [0.05, 0.2, 0.6, 1.5] {
                    for t in [0.05, 0.5, 2.0] {
                        let price = bs_price(s, k, r, q, vol, t, pc);
                        // skip quotes indistinguishable from intrinsic
                        if price - bs_price(s, k, r, q, 0.0, t, pc) < 1e-10 {
                            continue;
                        }
                        let iv = implied_vol_from_price(s, k, r, q, t, price, pc).unwrap();
                        // deep in-the-money short-dated quotes have vega ~1e-7,
                        // so a double-precision price only pins the vol to
                        // ~1e-6 — 1e-5 is the attainable accuracy everywhere
                        assert!(
                            (iv - vol).abs() < 1e-5,
                            "{pc:?} K={k} vol={vol} t={t}: recovered {iv}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn implied_vol_rejects_arbitrage_violating_prices() {
        // below intrinsic
        assert!(implied_vol_from_price(100.0, 80.0, 0.05, 0.0, 1.0, 10.0, PutOrCall::Call)
            .is_err());
        // above the underlying
        assert!(implied_vol_from_price(100.0, 100.0, 0.05, 0.0, 1.0, 101.0, PutOrCall::Call)
            .is_err());
    }

    // ── Implied surface construction + Dupire local vol round trip ──────

    /// Quotes generated from a known smile: sigma(K, T) = base(T) - 0.001*(K-100)
    fn smile_vol(k: f64, base: f64) -> f64 {
        base - 0.001 * (k - 100.0)
    }

    fn quoted_option(
        k: f64,
        maturity: NaiveDate,
        market_price: f64,
    ) -> Box<EquityOption> {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.strike_price = k;
        option.base.maturity_date = maturity;
        option.base.current_price = Quote::new(market_price);
        Box::new(option)
    }

    fn build_surface_from_quotes() -> crate::core::vols::VolSurface {
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let maturities = [
            (NaiveDate::from_ymd_opt(2026, 7, 2).unwrap(), 0.23),
            (NaiveDate::from_ymd_opt(2027, 1, 1).unwrap(), 0.25),
        ];
        let mut quotes = Vec::new();
        for (maturity, base) in maturities {
            let t = (maturity - valuation).num_days() as f64 / 365.0;
            for i in 0..13 {
                let k = 70.0 + 5.0 * i as f64;
                let vol = smile_vol(k, base);
                let price = bs_price(100.0, k, 0.05, 0.0, vol, t, PutOrCall::Call);
                quotes.push(quoted_option(k, maturity, price));
            }
        }
        crate::equity::vol_surface::build_implied_vol_surface(&quotes).unwrap()
    }

    #[test]
    fn implied_surface_recovers_input_vols() {
        let surface = build_surface_from_quotes();
        // exact at the quoted pillars (forward is irrelevant on a strike axis)
        for (t, base) in [(182.0 / 365.0, 0.23), (1.0, 0.25)] {
            for k in [70.0, 85.0, 100.0, 115.0, 130.0] {
                let vol = surface.vol(k, 100.0, t);
                assert!(
                    (vol - smile_vol(k, base)).abs() < 1e-7,
                    "K={k} t={t}: {vol} vs {}",
                    smile_vol(k, base)
                );
            }
        }
    }

    fn local_vol_option(surface: crate::core::vols::VolSurface, k: f64) -> EquityOption {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.strike_price = k;
        option.base.vol_surface = surface;
        option.engine = Engine::MonteCarlo;
        option.mc.model = crate::equity::montecarlo::McModel::LocalVol;
        option.mc.paths = 20_000;
        option
    }

    #[test]
    fn local_vol_prices_back_vanilla_from_calibrated_surface() {
        // implied quotes -> implied surface -> Dupire local vol -> MC price
        // must reproduce the original Black-Scholes prices
        let surface = build_surface_from_quotes();
        for k in [90.0, 100.0, 110.0] {
            let expected = bs_price(100.0, k, 0.05, 0.0, smile_vol(k, 0.25), 1.0, PutOrCall::Call);
            let lv_price = local_vol_option(surface.clone(), k).npv();
            assert!(
                (lv_price - expected).abs() < 0.3,
                "K={k}: local vol {lv_price} vs BS {expected}"
            );
        }
    }

    #[test]
    fn local_vol_flat_surface_reproduces_black_scholes() {
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface =
            crate::core::vols::VolSurface::flat(0.3, valuation, DayCountConvention::Act365)
                .unwrap();
        let expected = 14.2312547860; // flat-30% golden
        let lv_price = local_vol_option(surface, 100.0).npv();
        assert!((lv_price - expected).abs() < 0.3, "{lv_price} vs {expected}");
    }

    #[test]
    fn local_vol_term_structure_reproduces_terminal_implied() {
        // 20% to 6M, 25% to 1Y: pricing a 1Y option through the local vol
        // (which steps at ~20% then at the ~29.2% forward vol) must recover
        // the 25% terminal implied price
        let valuation = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface = crate::core::vols::VolSurface::from_strike_smiles(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0)],
            &[vec![(100.0, 0.20)], vec![(100.0, 0.25)]],
            valuation,
            DayCountConvention::Act365,
        )
        .unwrap();
        let expected = bs_price(100.0, 100.0, 0.05, 0.0, 0.25, 1.0, PutOrCall::Call);
        let lv_price = local_vol_option(surface, 100.0).npv();
        assert!((lv_price - expected).abs() < 0.3, "{lv_price} vs {expected}");
    }

    // ── Barrier options ─────────────────────────────────────────────────

    fn barrier_option(
        put_or_call: PutOrCall,
        direction: crate::equity::barrier::BarrierDirection,
        knock: crate::equity::barrier::KnockType,
        barrier: f64,
    ) -> EquityOption {
        let mut option = test_option_with(
            Box::new(BarrierPayoff {
                put_or_call,
                exercise_style: ContractStyle::European,
                direction,
                knock,
                barrier,
                barrier2: None,
                rebate: 0.0,
                rebate_at_hit: false,
            }),
            flat_5pct(),
        );
        option.base.dividend_yield = 0.02; // match the oracle setup
        option
    }

    #[test]
    fn golden_barrier_prices_all_eight_types() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        // independently generated Reiner-Rubinstein oracle values
        // (S=100, K=100, r=5%, q=2%, sigma=30%, T=1)
        let cases = [
            (Down, In, PutOrCall::Call, 90.0, 4.5095197744),
            (Down, Out, PutOrCall::Call, 90.0, 8.5107614943),
            (Down, In, PutOrCall::Put, 90.0, 10.0710164338),
            (Down, Out, PutOrCall::Put, 90.0, 0.0523399543),
            (Up, In, PutOrCall::Call, 120.0, 12.5974705742),
            (Up, Out, PutOrCall::Call, 120.0, 0.4228106946),
            (Up, In, PutOrCall::Put, 120.0, 1.4297711810),
            (Up, Out, PutOrCall::Put, 120.0, 8.6935852071),
        ];
        for (direction, knock, pc, h, expected) in cases {
            let option = barrier_option(pc, direction, knock, h);
            assert_approx_eq!(option.npv(), expected, 1e-8);
        }
    }

    #[test]
    fn barrier_greeks_satisfy_in_out_parity() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        // KI + KO = vanilla holds for the Greeks too (no rebate)
        let ki = barrier_option(PutOrCall::Call, Down, In, 90.0);
        let ko = barrier_option(PutOrCall::Call, Down, Out, 90.0);
        let mut vanilla = test_option(PutOrCall::Call, flat_5pct());
        vanilla.base.dividend_yield = 0.02;
        assert_approx_eq!(ki.npv() + ko.npv(), vanilla.npv(), 1e-10);
        assert_approx_eq!(ki.delta() + ko.delta(), vanilla.delta(), 1e-5);
        assert_approx_eq!(ki.gamma() + ko.gamma(), vanilla.gamma(), 1e-4);
        assert_approx_eq!(ki.vega() + ko.vega(), vanilla.vega(), 1e-4);
        assert_approx_eq!(ki.theta() + ko.theta(), vanilla.theta(), 1e-4);
        assert_approx_eq!(ki.rho() + ko.rho(), vanilla.rho(), 1e-4);
    }

    #[test]
    fn monte_carlo_barrier_matches_analytic() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        let cases = [
            (Down, Out, PutOrCall::Call, 90.0),
            (Down, In, PutOrCall::Put, 90.0),
            (Up, Out, PutOrCall::Put, 120.0),
            (Up, In, PutOrCall::Call, 110.0),
        ];
        for (direction, knock, pc, h) in cases {
            let analytic = barrier_option(pc, direction, knock, h).npv();
            let mut option = barrier_option(pc, direction, knock, h);
            option.engine = Engine::MonteCarlo;
            option.mc.paths = 50_000;
            let mc = option.npv();
            assert!(
                (mc - analytic).abs() < 0.3,
                "{direction:?} {knock:?} {pc:?} H={h}: mc={mc} analytic={analytic}"
            );
        }
    }

    // ── Asian options ───────────────────────────────────────────────────

    fn asian_option(
        put_or_call: PutOrCall,
        averaging: crate::equity::asian::AveragingType,
        strike_type: crate::equity::asian::AsianStrikeType,
    ) -> EquityOption {
        let mut option = test_option_with(
            Box::new(AsianPayoff {
                put_or_call,
                exercise_style: ContractStyle::European,
                averaging,
                strike_type,
            }),
            flat_5pct(),
        );
        option.base.dividend_yield = 0.02; // match the oracle setup
        option
    }

    #[test]
    fn golden_asian_analytic_prices() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        // independently generated oracle values (S=100 K=100 r=5% q=2% sigma=30% T=1)
        let geo = asian_option(PutOrCall::Call, Geometric, FixedStrike);
        assert_approx_eq!(geo.npv(), 6.953600, 1e-5);
        let arith = asian_option(PutOrCall::Call, Arithmetic, FixedStrike);
        assert_approx_eq!(arith.npv(), 7.409272, 1e-5);
    }

    #[test]
    fn geometric_asian_mc_matches_discrete_closed_form() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        let mut option = asian_option(PutOrCall::Call, Geometric, FixedStrike);
        option.engine = Engine::MonteCarlo;
        option.mc.paths = 50_000;
        let mc = option.npv(); // generic path route, 100 monitoring steps
        let closed = crate::equity::asian::geometric_asian_price(
            100.0, 100.0, 0.05, 0.02, 0.3, 1.0, Some(100), PutOrCall::Call,
        );
        assert!((mc - closed).abs() < 0.15, "mc={mc} closed={closed}");
    }

    #[test]
    fn geometric_average_strike_analytic_matches_monte_carlo() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        // the analytic engine now prices floating-strike geometric Asians
        // via the exact exchange-option closed form (continuous averaging)
        let analytic_option = asian_option(PutOrCall::Call, Geometric, FloatingStrike);
        let analytic = analytic_option.npv();
        let closed = crate::equity::asian::geometric_average_strike_price(
            100.0, 0.05, 0.02, 0.3, 1.0, None, PutOrCall::Call,
        );
        assert_approx_eq!(analytic, closed, 1e-12);
        // engine-level Greeks flow through the same closed form
        assert!(analytic_option.delta() > 0.0 && analytic_option.vega() > 0.0);

        // Monte Carlo (100 discrete fixings) against the discrete form
        let mut mc_option = asian_option(PutOrCall::Call, Geometric, FloatingStrike);
        mc_option.engine = Engine::MonteCarlo;
        mc_option.mc.paths = 50_000;
        let mc = mc_option.npv();
        let discrete = crate::equity::asian::geometric_average_strike_price(
            100.0, 0.05, 0.02, 0.3, 1.0, Some(100), PutOrCall::Call,
        );
        assert!((mc - discrete).abs() < 0.10, "mc={mc} closed={discrete}");
    }

    #[test]
    fn arithmetic_average_strike_analytic_matches_monte_carlo() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        // the analytic engine prices arithmetic floating strikes via the
        // Henderson-Wojakowski symmetry + Turnbull-Wakeman
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let analytic_option = asian_option(pc, Arithmetic, FloatingStrike);
            let analytic = analytic_option.npv();
            let direct = crate::equity::asian::turnbull_wakeman_average_strike_price(
                100.0, 0.05, 0.02, 0.3, 1.0, pc,
            );
            // 1e-9: the engine's curve-implied rate differs from the flat
            // input at ln/exp roundoff level, amplified by the TW moments
            assert_approx_eq!(analytic, direct, 1e-9);
            assert!(analytic_option.vega() > 0.0, "{pc:?}");

            let mut mc_option = asian_option(pc, Arithmetic, FloatingStrike);
            mc_option.engine = Engine::MonteCarlo;
            mc_option.mc.paths = 50_000;
            let mc = mc_option.npv();
            // TW approximation + discrete-vs-continuous monitoring gap
            assert!((mc - analytic).abs() < 0.2, "{pc:?}: mc={mc} analytic={analytic}");
        }
    }

    #[test]
    fn double_barriers_and_rebates_price_across_engines() {
        use crate::equity::barrier::KnockType;
        use crate::equity::builder::EquityOptionBuilder;
        use chrono::NaiveDate;
        let build = || {
            EquityOptionBuilder::new()
                .symbol("DKO")
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.3)
                .flat_rate(0.05)
                .dividend_yield(0.02)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
        };
        // double knock-out call: analytic (Ikeda-Kunitomo) vs the direct fn
        let dko = build().double_barrier(PutOrCall::Call, KnockType::Out, 85.0, 120.0)
            .engine(Engine::BlackScholes).build();
        let direct = crate::equity::barrier::double_barrier_price(
            100.0, 100.0, 85.0, 120.0, 0.05, 0.02, 0.3, 1.0, KnockType::Out, PutOrCall::Call,
        );
        assert_approx_eq!(dko.npv(), direct, 1e-9);
        // MC (discrete monitoring, generic path route) sits above and near
        let mut mc = build().double_barrier(PutOrCall::Call, KnockType::Out, 85.0, 120.0)
            .engine(Engine::MonteCarlo).build();
        mc.mc.paths = 50_000;
        mc.mc.time_steps = 500;
        let mc_px = mc.npv();
        assert!(mc_px > dko.npv() - 0.02 && (mc_px - dko.npv()).abs() < 0.30,
            "mc {mc_px} vs analytic {}", dko.npv());

        // knock-out with an at-expiry rebate: analytic vs MC agree in
        // convention; the rebate lifts the price by less than its PV
        let rebate = 5.0;
        let plain = build()
            .barrier(PutOrCall::Call, crate::equity::barrier::BarrierDirection::Up,
                KnockType::Out, 120.0)
            .engine(Engine::BlackScholes).build().npv();
        let rebated = build()
            .barrier(PutOrCall::Call, crate::equity::barrier::BarrierDirection::Up,
                KnockType::Out, 120.0)
            .barrier_rebate(rebate, false)
            .engine(Engine::BlackScholes).build();
        assert!(rebated.npv() > plain && rebated.npv() < plain + rebate);
        let mut mc_rebate = build()
            .barrier(PutOrCall::Call, crate::equity::barrier::BarrierDirection::Up,
                KnockType::Out, 120.0)
            .barrier_rebate(rebate, false)
            .engine(Engine::MonteCarlo).build();
        mc_rebate.mc.paths = 50_000;
        mc_rebate.mc.time_steps = 500;
        assert!((mc_rebate.npv() - rebated.npv()).abs() < 0.30,
            "mc {} vs analytic {}", mc_rebate.npv(), rebated.npv());
        // at-hit rebates are analytic-engine territory
        let at_hit = build()
            .barrier(PutOrCall::Call, crate::equity::barrier::BarrierDirection::Up,
                KnockType::Out, 120.0)
            .barrier_rebate(rebate, true)
            .engine(Engine::BlackScholes).build();
        assert!(at_hit.npv() > rebated.npv(), "earlier payment is worth more");
    }

    #[test]
    fn lookback_analytic_engine_matches_the_closed_forms_and_mc_converges() {
        use crate::equity::vanilla_option::{LookbackPayoff, LookbackType};
        let lookback = |lookback_type, pc| {
            let mut option = test_option_with(
                Box::new(LookbackPayoff {
                    put_or_call: pc,
                    exercise_style: ContractStyle::European,
                    lookback_type,
                }),
                flat_5pct(),
            );
            option.base.dividend_yield = 0.02;
            option
        };
        // analytic engine = the continuous closed forms
        let float_call = lookback(LookbackType::FloatingStrike, PutOrCall::Call);
        let direct = crate::equity::lookback::floating_strike_lookback_price(
            100.0, 100.0, 0.05, 0.02, 0.3, 1.0, PutOrCall::Call,
        );
        assert_approx_eq!(float_call.npv(), direct, 1e-9);
        // Greeks are populated (not silently zero). The fresh floating
        // lookback is a volatility machine, not a directional bet: its
        // delta is small (~0.225 -- V is homogeneous in S) while its vega
        // is enormous
        assert!(float_call.delta() > 0.1 && float_call.delta() < 0.4, "{}", float_call.delta());
        assert!(float_call.vega() > 30.0, "{}", float_call.vega());

        // MC monitors discretely: below the continuous form for the
        // max/min-based payoffs, and converging as steps grow
        let mut coarse = lookback(LookbackType::FloatingStrike, PutOrCall::Call);
        coarse.engine = Engine::MonteCarlo;
        coarse.mc.paths = 40_000;
        coarse.mc.time_steps = 50;
        let coarse_px = coarse.npv();
        let mut fine = lookback(LookbackType::FloatingStrike, PutOrCall::Call);
        fine.engine = Engine::MonteCarlo;
        fine.mc.paths = 40_000;
        fine.mc.time_steps = 400;
        let fine_px = fine.npv();
        assert!(coarse_px < direct && fine_px < direct, "{coarse_px} {fine_px} vs {direct}");
        assert!(direct - fine_px < direct - coarse_px, "finer grid must close the gap");
        assert!(direct - fine_px < 1.5, "fine-grid gap {}", direct - fine_px);

        // fixed strike side too
        let fixed_put = lookback(LookbackType::FixedStrike, PutOrCall::Put);
        let direct_put = crate::equity::lookback::fixed_strike_lookback_price(
            100.0, 100.0, 100.0, 0.05, 0.02, 0.3, 1.0, PutOrCall::Put,
        );
        assert_approx_eq!(fixed_put.npv(), direct_put, 1e-9);
        let mut mc_put = lookback(LookbackType::FixedStrike, PutOrCall::Put);
        mc_put.engine = Engine::MonteCarlo;
        mc_put.mc.paths = 40_000;
        mc_put.mc.time_steps = 400;
        let mc_px = mc_put.npv();
        assert!(mc_px < direct_put && direct_put - mc_px < 1.5, "{mc_px} vs {direct_put}");
    }

    #[test]
    fn phoenix_certificate_prices_and_orders_sensibly() {
        use crate::equity::builder::EquityOptionBuilder;
        use chrono::NaiveDate;
        let build = |coupon_barrier: f64, memory: bool| {
            EquityOptionBuilder::new()
                .symbol("PHX")
                .spot(100.0)
                .strike(100.0)
                .flat_vol(0.25)
                .flat_rate(0.03)
                .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
                .maturity_date(NaiveDate::from_ymd_opt(2028, 1, 1).unwrap())
                .phoenix(110.0, coupon_barrier, 60.0, 2.0, 8, 100.0, memory)
                .engine(Engine::MonteCarlo)
                .paths(30_000)
                .build()
        };
        let plain = build(80.0, false).npv();
        let with_memory = build(80.0, true).npv();
        let lower_barrier = build(60.0, false).npv();
        // memory can only add coupons; a lower coupon barrier pays more often
        assert!(with_memory > plain, "memory {with_memory} vs {plain}");
        assert!(lower_barrier > plain, "cb60 {lower_barrier} vs cb80 {plain}");
        // sane range: below max possible (notional + all coupons), above
        // a deeply protected floor
        assert!(plain > 80.0 && with_memory < 100.0 + 16.0 + 1.0, "{plain} {with_memory}");
    }

    #[test]
    fn arithmetic_asian_cv_mc_close_to_turnbull_wakeman() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        let analytic = asian_option(PutOrCall::Call, Arithmetic, FixedStrike).npv();
        let mut option = asian_option(PutOrCall::Call, Arithmetic, FixedStrike);
        option.engine = Engine::MonteCarlo;
        option.mc.paths = 50_000;
        let mc = option.npv(); // control-variate route
        // TW is a moment-matching approximation and the MC monitors
        // discretely, so agreement is at the approximation level, not
        // sampler noise level
        assert!((mc - analytic).abs() < 0.08, "cv-mc={mc} tw={analytic}");
    }

    #[test]
    fn arithmetic_average_dominates_geometric_on_same_paths() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        let price_mc = |averaging| {
            let mut option = asian_option(PutOrCall::Call, averaging, FixedStrike);
            option.engine = Engine::MonteCarlo;
            option.mc.paths = 20_000;
            // force the generic path route for both by disabling the CV's
            // exact-scheme precondition
            option.mc.scheme = crate::equity::montecarlo::DiscretizationScheme::Euler;
            option.mc.time_steps = 100;
            option.npv()
        };
        assert!(price_mc(Arithmetic) > price_mc(Geometric), "AM-GM inequality");
    }

    #[test]
    fn arithmetic_floating_strike_asian_prices_on_mc() {
        use crate::equity::asian::{AsianStrikeType::*, AveragingType::*};
        let mut option = asian_option(PutOrCall::Call, Arithmetic, FloatingStrike);
        option.engine = Engine::MonteCarlo;
        option.mc.paths = 20_000;
        let price = option.npv();
        // floating-strike call: pays (S_T - average)^+; positive, below vanilla
        let vanilla = test_option(PutOrCall::Call, flat_5pct()).npv();
        assert!(price > 0.0 && price < vanilla, "{price}");
    }


    // ── FD upgrades: grid Greeks, barriers, local vol, config ───────────

    #[test]
    fn fd_grid_greeks_match_analytic_for_european() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::FiniteDifference;
        assert!((option.delta() - 0.6242517279).abs() < 1e-3, "delta {}", option.delta());
        assert!((option.gamma() - 0.0126477644).abs() < 1e-4, "gamma {}", option.gamma());
        assert!((option.theta() - -8.1011898970).abs() < 0.03, "theta {}", option.theta());
        assert!((option.vega() - 37.9432933117).abs() < 0.05, "vega {}", option.vega());
        assert!((option.rho() - 48.1939180046).abs() < 0.05, "rho {}", option.rho());
    }

    #[test]
    fn fd_american_put_greeks_differ_from_european_correctly() {
        let mut american = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        american.engine = Engine::FiniteDifference;
        let european_delta = -0.3757482721; // analytic European put delta
        // early exercise makes the American put delta more negative and
        // theta less negative than the European
        assert!(
            american.delta() < european_delta,
            "american delta {} vs european {european_delta}",
            american.delta()
        );
        assert!(american.npv() > test_option(PutOrCall::Put, flat_5pct()).npv());
    }

    #[test]
    fn fd_brennan_schwartz_american_matches_tree() {
        let mut fd = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        fd.engine = Engine::FiniteDifference;
        let mut tree = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        tree.engine = Engine::Binomial;
        assert!((fd.npv() - tree.npv()).abs() < 0.02, "fd={} tree={}", fd.npv(), tree.npv());
    }

    #[test]
    fn fd_barrier_matches_reiner_rubinstein() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        for (direction, knock, pc, h) in [
            (Down, Out, PutOrCall::Call, 90.0),
            (Down, In, PutOrCall::Call, 90.0),
            (Up, Out, PutOrCall::Put, 120.0),
            (Up, In, PutOrCall::Put, 120.0),
        ] {
            let analytic = barrier_option(pc, direction, knock, h).npv();
            let mut option = barrier_option(pc, direction, knock, h);
            option.engine = Engine::FiniteDifference;
            let fd = option.npv();
            assert!(
                (fd - analytic).abs() < 0.02,
                "{direction:?} {knock:?} {pc:?} H={h}: fd={fd} analytic={analytic}"
            );
        }
    }

    #[test]
    fn fd_barrier_in_out_parity_on_grid() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        let mut ki = barrier_option(PutOrCall::Call, Down, In, 90.0);
        let mut ko = barrier_option(PutOrCall::Call, Down, Out, 90.0);
        ki.engine = Engine::FiniteDifference;
        ko.engine = Engine::FiniteDifference;
        let mut vanilla = test_option(PutOrCall::Call, flat_5pct());
        vanilla.base.dividend_yield = 0.02;
        vanilla.engine = Engine::FiniteDifference;
        assert!((ki.npv() + ko.npv() - vanilla.npv()).abs() < 1e-9);
        assert!((ki.delta() + ko.delta() - vanilla.delta()).abs() < 1e-9);
    }

    #[test]
    fn fd_local_vol_flat_surface_matches_black_scholes() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::FiniteDifference;
        option.mc.model = crate::equity::montecarlo::McModel::LocalVol;
        // flat surface: local vol == implied vol, FD-LV must equal FD-GBM
        assert_approx_eq!(option.npv(), 14.2312547860, 5e-3);
    }

    #[test]
    fn fd_grid_is_configurable() {
        let mut coarse = test_option(PutOrCall::Call, flat_5pct());
        coarse.engine = Engine::FiniteDifference;
        coarse.fd.spot_steps = 100;
        coarse.fd.time_steps = 50;
        // still accurate at a quarter of the resolution
        assert!((coarse.npv() - 14.2312547860).abs() < 0.02, "{}", coarse.npv());
    }

    // ── MC upgrades: QMC paths, stats, determinism ──────────────────────

    #[test]
    fn qmc_path_wise_prices_accurately() {
        // multi-step path simulation through the Brownian bridge + QMC
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        option.mc.time_steps = 64;
        option.mc.paths = 20_000;
        let qmc = option.npv();
        assert!((qmc - 14.2312547860).abs() < 0.05, "qmc path-wise {qmc}");
    }

    #[test]
    fn mc_stats_reports_consistent_standard_error() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.engine = Engine::MonteCarlo;
        option.mc.sampler = crate::equity::montecarlo::Sampler::PseudoRandom;
        let stats = crate::equity::montecarlo::npv_with_stats(&option);
        assert!(stats.std_err > 0.0 && stats.std_err < 1.0);
        assert!(stats.paths == 100_000 && stats.steps == 1);
        // iid pseudo draws: the analytic value must sit within a few
        // standard errors of the estimate
        assert!(
            (stats.pv - 14.2312547860).abs() < 5.0 * stats.std_err,
            "pv={} stderr={}",
            stats.pv,
            stats.std_err
        );
    }

    #[test]
    fn parallel_paths_are_bit_reproducible() {
        for sampler in
            [crate::equity::montecarlo::Sampler::Sobol, crate::equity::montecarlo::Sampler::PseudoRandom]
        {
            let mut option = test_option(PutOrCall::Call, flat_5pct());
            option.engine = Engine::MonteCarlo;
            option.mc.sampler = sampler;
            option.mc.time_steps = 32;
            option.mc.paths = 30_000;
            assert_eq!(option.npv(), option.npv());
        }
    }

    // ── Heston stochastic vol ───────────────────────────────────────────

    fn heston_option(payoff: Box<dyn Payoff>) -> EquityOption {
        let mut option = test_option_with(payoff, flat_5pct());
        option.base.dividend_yield = 0.02;
        option.mc.model = crate::equity::montecarlo::McModel::Heston;
        option.heston = Some(crate::equity::heston::HestonParams {
            v0: 0.09,
            kappa: 2.0,
            theta: 0.09,
            vol_of_vol: 0.4,
            rho: -0.7,
        });
        option
    }

    fn heston_vanilla(pc: PutOrCall) -> EquityOption {
        heston_option(Box::new(VanillaPayoff {
            put_or_call: pc,
            exercise_style: ContractStyle::European,
        }))
    }

    #[test]
    fn heston_mc_matches_semi_analytic() {
        for pc in [PutOrCall::Call, PutOrCall::Put] {
            let analytic = heston_vanilla(pc).npv();
            let mut mc = heston_vanilla(pc);
            mc.engine = Engine::MonteCarlo;
            mc.mc.paths = 50_000;
            let mc_price = mc.npv();
            // full-truncation Euler bias + sampler noise at 50k x 250
            assert!(
                (mc_price - analytic).abs() < 0.15,
                "{pc:?}: mc={mc_price} analytic={analytic}"
            );
        }
    }

    #[test]
    fn heston_binary_mc_matches_semi_analytic() {
        let payoff = || -> Box<dyn Payoff> {
            Box::new(BinaryPayoff {
                put_or_call: PutOrCall::Call,
                exercise_style: ContractStyle::European,
                binary_type: BinaryType::CashOrNothing,
                cash: 1.0,
            })
        };
        let analytic = heston_option(payoff()).npv();
        let mut mc = heston_option(payoff());
        mc.engine = Engine::MonteCarlo;
        mc.mc.paths = 50_000;
        assert!((mc.npv() - analytic).abs() < 0.01, "mc={} analytic={analytic}", mc.npv());
    }

    #[test]
    fn heston_greeks_are_consistent() {
        let call = heston_vanilla(PutOrCall::Call);
        let put = heston_vanilla(PutOrCall::Put);
        // parity: delta_call - delta_put = e^{-qT}
        let dfq = (-0.02_f64).exp();
        assert!((call.delta() - put.delta() - dfq).abs() < 1e-4);
        // same gamma and vega for call and put by parity
        assert!((call.gamma() - put.gamma()).abs() < 1e-6);
        assert!((call.vega() - put.vega()).abs() < 1e-4);
        assert!(call.vega() > 0.0);
    }

    #[test]
    fn heston_barrier_and_asian_price_on_mc() {
        // knock-out <= vanilla under the same dynamics; asian < vanilla
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        let vanilla = {
            let mut o = heston_vanilla(PutOrCall::Call);
            o.engine = Engine::MonteCarlo;
            o.mc.paths = 20_000;
            o.npv()
        };
        let mut ko = heston_option(Box::new(BarrierPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: ContractStyle::European,
            direction: Down,
            knock: Out,
            barrier: 90.0,
                barrier2: None,
                rebate: 0.0,
                rebate_at_hit: false,
        }));
        ko.engine = Engine::MonteCarlo;
        ko.mc.paths = 20_000;
        let ko_price = ko.npv();
        assert!(ko_price > 0.0 && ko_price < vanilla, "ko={ko_price} vanilla={vanilla}");

        let mut asian = heston_option(Box::new(AsianPayoff {
            put_or_call: PutOrCall::Call,
            exercise_style: ContractStyle::European,
            averaging: crate::equity::asian::AveragingType::Arithmetic,
            strike_type: crate::equity::asian::AsianStrikeType::FixedStrike,
        }));
        asian.engine = Engine::MonteCarlo;
        asian.mc.paths = 20_000;
        let asian_price = asian.npv();
        assert!(asian_price > 0.0 && asian_price < vanilla);
    }

    // ── Borrow cost and dividends ───────────────────────────────────────

    #[test]
    fn borrow_cost_is_equivalent_to_extra_dividend_yield() {
        for engine in [Engine::BlackScholes, Engine::FiniteDifference, Engine::MonteCarlo] {
            let mut with_borrow = test_option(PutOrCall::Call, flat_5pct());
            with_borrow.base.dividend_yield = 0.01;
            with_borrow.base.borrow_cost = 0.03;
            with_borrow.engine = engine.clone();
            let mut with_yield = test_option(PutOrCall::Call, flat_5pct());
            with_yield.base.dividend_yield = 0.04;
            with_yield.engine = engine.clone();
            assert!(
                (with_borrow.npv() - with_yield.npv()).abs() < 1e-12,
                "{engine:?}: borrow {} vs yield {}",
                with_borrow.npv(),
                with_yield.npv()
            );
        }
    }

    fn dividend_paying_option(pc: PutOrCall) -> EquityOption {
        let mut option = test_option(pc, flat_5pct());
        option.base.cash_dividends =
            vec![(NaiveDate::from_ymd_opt(2026, 7, 1).unwrap(), 3.0)];
        option
    }

    #[test]
    fn cash_dividend_prices_as_escrowed_spot_analytically() {
        let option = dividend_paying_option(PutOrCall::Call);
        let t_div = (NaiveDate::from_ymd_opt(2026, 7, 1).unwrap()
            - NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .num_days() as f64
            / 365.0;
        let s_eff = 100.0 - 3.0 * (-0.05 * t_div).exp();
        assert!((option.base.effective_spot() - s_eff).abs() < 1e-10);
        let expected = bs_price(s_eff, 100.0, 0.05, 0.0, 0.3, 1.0, PutOrCall::Call);
        assert_approx_eq!(option.npv(), expected, 1e-10);
    }

    #[test]
    fn put_call_parity_with_dividends_and_borrow() {
        let mut call = dividend_paying_option(PutOrCall::Call);
        let mut put = dividend_paying_option(PutOrCall::Put);
        call.base.borrow_cost = 0.02;
        put.base.borrow_cost = 0.02;
        let parity = call.base.effective_spot() * (-call.base.carry_yield()).exp()
            - 100.0 * (-0.05_f64).exp();
        assert_approx_eq!(call.npv() - put.npv(), parity, 1e-10);
    }

    #[test]
    fn mc_dividend_jumps_close_to_escrowed_analytic() {
        // the jump model (dividends subtracted on the path) and the
        // escrowed model differ slightly by construction; they must agree
        // at the tens-of-basis-points level for moderate dividends
        let analytic = dividend_paying_option(PutOrCall::Call).npv();
        let mut mc = dividend_paying_option(PutOrCall::Call);
        mc.engine = Engine::MonteCarlo;
        mc.mc.time_steps = 100;
        mc.mc.paths = 50_000;
        assert!((mc.npv() - analytic).abs() < 0.3, "mc={} analytic={analytic}", mc.npv());
    }

    #[test]
    fn fd_dividend_jump_condition_consistent_with_mc_jump_model() {
        // FD and path-MC both implement the jump (piecewise lognormal)
        // dividend model and must agree tightly; both sit a known
        // ~0.1-0.2 above the escrowed analytic for a call (the classic
        // escrowed-vs-jump model difference)
        let mut fd = dividend_paying_option(PutOrCall::Call);
        fd.engine = Engine::FiniteDifference;
        let mut mc = dividend_paying_option(PutOrCall::Call);
        mc.engine = Engine::MonteCarlo;
        mc.mc.time_steps = 100;
        mc.mc.paths = 50_000;
        assert!((fd.npv() - mc.npv()).abs() < 0.1, "fd={} mc={}", fd.npv(), mc.npv());
        let escrowed = dividend_paying_option(PutOrCall::Call).npv();
        assert!((fd.npv() - escrowed).abs() < 0.3, "fd={} escrowed={escrowed}", fd.npv());
    }

    #[test]
    fn forward_price_reflects_borrow_and_cash_dividends() {
        let mut option = dividend_paying_option(PutOrCall::Call);
        option.base.borrow_cost = 0.02;
        let expected = option.base.effective_spot() * ((0.05 - 0.02) * 1.0_f64).exp();
        assert_approx_eq!(option.base.forward_price(), expected, 1e-10);
    }

    #[test]
    fn cash_dividend_with_carry_discounts_at_net_carry() {
        // with a continuous carry present, the cash dividend must be
        // discounted at (r - carry), not r, so the escrowed spot and the
        // analytic forward match the jump-model ground truth
        // F = (S - D e^{-(r-carry)t}) e^{(r-carry)T}
        let carry = 0.03;
        let mut option = dividend_paying_option(PutOrCall::Call);
        option.base.borrow_cost = carry;
        let (r, s, d, t) = (0.05, 100.0, 3.0, 1.0);
        let t_div = (NaiveDate::from_ymd_opt(2026, 7, 1).unwrap()
            - NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
        .num_days() as f64
            / 365.0;

        let s_eff = s - d * (-(r - carry) * t_div).exp();
        assert_approx_eq!(option.base.effective_spot(), s_eff, 1e-10);

        let jump_forward = s_eff * ((r - carry) * t).exp();
        assert_approx_eq!(option.base.forward_price(), jump_forward, 1e-10);
    }

    #[test]
    fn net_carry_discounting_flows_through_to_price_and_stays_near_jump_engines() {
        // The fix guarantees forward consistency (checked above); this
        // confirms it flows through to the analytic price, which is exactly
        // the escrowed lognormal on the net-carry spot, and that the price
        // stays within the escrowed-vs-jump tolerance of the FD engine.
        //
        // Note: matching the forward does NOT make the escrowed *price*
        // equal the jump price — the escrowed model applies vol to S - PV
        // rather than to S with a jump, an intrinsic approximation. So we
        // check the band, not exact agreement.
        let carry = 0.03;
        let mut analytic = dividend_paying_option(PutOrCall::Call);
        analytic.base.borrow_cost = carry;
        let a = analytic.npv();

        let expected =
            bs_price(analytic.base.effective_spot(), 100.0, 0.05, carry, 0.3, 1.0, PutOrCall::Call);
        assert_approx_eq!(a, expected, 1e-10);

        let mut fd = dividend_paying_option(PutOrCall::Call);
        fd.base.borrow_cost = carry;
        fd.engine = Engine::FiniteDifference;
        assert!((a - fd.npv()).abs() < 0.2, "analytic {a} vs fd {}", fd.npv());
    }

    // ── Options on futures (Black-76) ───────────────────────────────────

    fn futures_option(
        pc: PutOrCall,
        settlement: crate::equity::black76::FuturesSettlement,
    ) -> EquityOption {
        crate::equity::builder::EquityOptionBuilder::new()
            .symbol("FUT")
            .spot(100.0) // interpreted as the futures price F
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(0.05)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(pc)
            .on_future(settlement)
            .engine(Engine::BlackScholes)
            .build()
    }

    #[test]
    fn black76_option_api_matches_closed_form() {
        use crate::equity::black76::FuturesSettlement::*;
        for (settlement, gold) in [(Discounted, 11.34202064), (Margined, 11.92353847)] {
            let option = futures_option(PutOrCall::Call, settlement);
            assert_approx_eq!(option.npv(), gold, 1e-7);
        }
        // spot-check a Greek reaches the option API too
        let call = futures_option(PutOrCall::Call, Discounted);
        assert_approx_eq!(call.delta(), 0.53232482, 1e-7);
        assert_approx_eq!(call.rho(), -11.34202064, 1e-6);
    }

    #[test]
    fn margined_futures_option_has_zero_rho_and_exceeds_discounted() {
        use crate::equity::black76::FuturesSettlement::*;
        let disc = futures_option(PutOrCall::Call, Discounted).npv();
        let marg = futures_option(PutOrCall::Call, Margined);
        assert_eq!(marg.rho(), 0.0);
        assert!(marg.npv() > disc);
        assert_approx_eq!(marg.npv(), disc * (0.05_f64).exp(), 1e-9);
    }

    #[test]
    fn black76_on_the_forward_equals_spot_black_scholes() {
        // a discounted Black-76 option on F = S e^{(r-q)T} must equal the
        // equivalent spot option priced by the equity Black-Scholes engine
        let (s, q, r, t): (f64, f64, f64, f64) = (100.0, 0.02, 0.05, 1.0);
        let fwd = s * ((r - q) * t).exp();
        let futures_opt = crate::equity::builder::EquityOptionBuilder::new()
            .spot(fwd)
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(r)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(PutOrCall::Call)
            .on_future(crate::equity::black76::FuturesSettlement::Discounted)
            .build();
        let spot_opt = crate::equity::builder::EquityOptionBuilder::new()
            .spot(s)
            .strike(100.0)
            .flat_vol(0.30)
            .flat_rate(r)
            .dividend_yield(q)
            .valuation_date(NaiveDate::from_ymd_opt(2026, 1, 1).unwrap())
            .maturity_date(NaiveDate::from_ymd_opt(2027, 1, 1).unwrap())
            .vanilla(PutOrCall::Call)
            .build();
        assert_approx_eq!(futures_opt.npv(), spot_opt.npv(), 1e-10);
    }

    #[test]
    fn put_call_parity_on_futures_both_styles() {
        use crate::equity::black76::FuturesSettlement::*;
        for (settlement, df) in [(Discounted, (-0.05_f64).exp()), (Margined, 1.0)] {
            let c = futures_option(PutOrCall::Call, settlement).npv();
            let p = futures_option(PutOrCall::Put, settlement).npv();
            // F = K = 100 -> parity value is 0
            assert_approx_eq!(c - p, df * (100.0 - 100.0), 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Options on futures (Black-76) price on the Analytical engine only")]
    fn futures_option_rejects_non_analytic_engine() {
        let mut option =
            futures_option(PutOrCall::Call, crate::equity::black76::FuturesSettlement::Discounted);
        option.engine = Engine::MonteCarlo;
        option.npv();
    }

    // ── Forward-start options ───────────────────────────────────────────

    fn forward_start_option(pc: PutOrCall) -> EquityOption {
        test_option_with(
            Box::new(crate::equity::forward_start_option::ForwardStartPayoff {
                put_or_call: pc,
                exercise_style: ContractStyle::European,
                strike_fraction: 1.0,
                start_fraction: 0.5,
            }),
            flat_5pct(),
        )
    }

    #[test]
    fn forward_start_analytic_matches_monte_carlo() {
        let analytic = forward_start_option(PutOrCall::Call).npv();
        let mut mc = forward_start_option(PutOrCall::Call);
        mc.engine = Engine::MonteCarlo;
        mc.mc.paths = 50_000;
        assert!((mc.npv() - analytic).abs() < 0.15, "mc={} analytic={analytic}", mc.npv());
    }

    #[test]
    fn forward_start_heston_degenerates_to_black_scholes() {
        let bs = forward_start_option(PutOrCall::Call).npv();
        let mut heston = forward_start_option(PutOrCall::Call);
        heston.engine = Engine::MonteCarlo;
        heston.mc.model = crate::equity::montecarlo::McModel::Heston;
        heston.mc.paths = 50_000;
        heston.heston = Some(crate::equity::heston::HestonParams {
            v0: 0.09,
            kappa: 1.0,
            theta: 0.09,
            vol_of_vol: 1e-3,
            rho: 0.0,
        });
        assert!((heston.npv() - bs).abs() < 0.2, "heston={} bs={bs}", heston.npv());
    }

    // ── Autocallables ───────────────────────────────────────────────────

    fn autocall_note(autocall_barrier: f64, protection_barrier: f64, coupon: f64) -> EquityOption {
        let mut option = test_option_with(
            Box::new(crate::equity::autocallable::AutocallablePayoff {
                exercise_style: ContractStyle::European,
                autocall_barrier,
                protection_barrier,
                coupon,
                observations: 4,
                notional: 100.0,
                initial_fixing: 100.0,
                coupon_barrier: None,
                memory: false,
            }),
            flat_5pct(),
        );
        option.engine = Engine::MonteCarlo;
        option.mc.paths = 20_000;
        option
    }

    #[test]
    fn autocall_that_always_calls_pays_coupon_at_first_observation() {
        // barrier below any reachable spot: every path calls at t1 = T/4
        let note = autocall_note(1e-9, 50.0, 5.0);
        let stats = crate::equity::montecarlo::npv_with_stats(&note);
        let expected = 105.0 * (-0.05 * 0.25_f64).exp();
        assert_approx_eq!(stats.pv, expected, 1e-9);
        // identical path values: stderr is pure floating-point cancellation
        assert!(stats.std_err < 1e-6, "deterministic payoff: stderr {}", stats.std_err);
    }

    #[test]
    fn autocall_never_called_with_full_protection_is_a_zero_coupon_bond() {
        let note = autocall_note(1e12, 1e-9, 5.0);
        let expected = 100.0 * (-0.05_f64).exp();
        assert_approx_eq!(note.npv(), expected, 1e-9);
    }

    #[test]
    fn autocall_full_downside_is_the_discounted_forward() {
        // protection always breached, never called: pays N * S_T / S_0,
        // whose discounted expectation is N (q = 0)
        let note = autocall_note(1e12, 1e12, 0.0);
        assert!((note.npv() - 100.0).abs() < 0.3, "{}", note.npv());
    }

    #[test]
    fn autocall_value_increases_with_coupon_and_lower_protection() {
        let base = autocall_note(105.0, 70.0, 5.0).npv();
        assert!(autocall_note(105.0, 70.0, 8.0).npv() > base, "higher coupon");
        assert!(autocall_note(105.0, 50.0, 5.0).npv() > base, "lower knock-in barrier");
    }

    #[test]
    fn autocall_prices_under_local_vol() {
        // flat surface: local vol must reproduce the GBM value
        let gbm = autocall_note(105.0, 70.0, 5.0).npv();
        let mut lv = autocall_note(105.0, 70.0, 5.0);
        lv.mc.model = crate::equity::montecarlo::McModel::LocalVol;
        assert!((lv.npv() - gbm).abs() < 0.5, "lv={} gbm={gbm}", lv.npv());
    }

    #[test]
    #[should_panic(expected = "Autocallables price on the MonteCarlo engine only")]
    fn analytic_engine_rejects_autocallables() {
        let mut note = autocall_note(105.0, 70.0, 5.0);
        note.engine = Engine::BlackScholes;
        note.npv();
    }

    #[test]
    #[should_panic(expected = "only barriers price on the FD")]
    fn fd_engine_rejects_forward_start() {
        let mut option = forward_start_option(PutOrCall::Call);
        option.engine = Engine::FiniteDifference;
        option.npv();
    }

    #[test]
    #[should_panic(expected = "Heston model is supported on the Analytical and MonteCarlo")]
    fn fd_engine_rejects_heston() {
        let mut option = heston_vanilla(PutOrCall::Call);
        option.engine = Engine::FiniteDifference;
        option.npv();
    }

    #[test]
    #[should_panic(expected = "not supported on the Binomial engine")]
    fn tree_engine_rejects_barrier_options() {
        use crate::equity::barrier::{BarrierDirection::*, KnockType::*};
        let mut option = barrier_option(PutOrCall::Call, Down, Out, 90.0);
        option.engine = Engine::Binomial;
        option.npv();
    }

    #[test]
    #[should_panic(expected = "Analytical engine cannot price American")]
    fn analytic_engine_rejects_american_exercise() {
        let option = test_option_with(
            Box::new(VanillaPayoff {
                put_or_call: PutOrCall::Put,
                exercise_style: ContractStyle::American,
            }),
            flat_5pct(),
        );
        option.npv();
    }

    #[test]
    fn american_put_fd_and_tree_agree_and_dominate_european() {
        let european_put = test_option(PutOrCall::Put, flat_5pct()).npv();
        let american = |engine: Engine| {
            let mut option = test_option_with(
                Box::new(VanillaPayoff {
                    put_or_call: PutOrCall::Put,
                    exercise_style: ContractStyle::American,
                }),
                flat_5pct(),
            );
            option.engine = engine;
            option.npv()
        };
        let fd = american(Engine::FiniteDifference);
        let tree = american(Engine::Binomial);
        assert!(fd > european_put, "american {fd} must exceed european {european_put}");
        assert!(tree > european_put);
        assert!((fd - tree).abs() < 0.02, "fd={fd} tree={tree}");
    }

    #[test]
    fn smile_surface_prices_with_interpolated_vol() {
        // K=100 sits midway between the 90 and 110 pillars at the 1y expiry,
        // so the option must price at the interpolated 30% vol — i.e. match
        // the flat-30% golden values exactly.
        let valuation_date = NaiveDate::from_ymd_opt(2026, 1, 1).unwrap();
        let surface = crate::core::vols::VolSurface::from_strike_grid(
            &[Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
            &[90.0, 100.0, 110.0],
            &[vec![0.32, 0.30, 0.28], vec![0.36, 0.34, 0.32]],
            valuation_date,
            DayCountConvention::Act365,
        )
        .unwrap();
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        option.base.vol_surface = surface;
        assert_approx_eq!(option.base.volatility(), 0.30, 1e-14);
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.vega(), 37.9432933117, 1e-8);
        // a lower strike picks up the skew: vol(95) = 0.31
        option.base.strike_price = 95.0;
        assert_approx_eq!(option.base.volatility(), 0.31, 1e-14);
    }

    #[test]
    fn implied_vol_recovers_input_vol() {
        let mut option = test_option(PutOrCall::Call, flat_5pct());
        let target_price = option.npv(); // priced at 30% flat
        // start the solve from a different vol level
        option.base.vol_surface = crate::core::vols::VolSurface::flat(
            0.6,
            option.base.valuation_date,
            DayCountConvention::Act365,
        )
        .unwrap();
        let iv = option.imp_vol(target_price);
        assert_approx_eq!(iv, 0.30, 1e-10);
    }

    #[test]
    fn zero_curve_prices_off_maturity_pillar() {
        // A non-flat zero curve whose 1y pillar is 5% must reproduce the
        // flat-5% price: discounting reads df at maturity, not any other node.
        let curve = YieldCurve::from_zero_rates(
            &[Tenor::YearFraction(0.5), Tenor::YearFraction(1.0), Tenor::YearFraction(2.0)],
            &[0.02, 0.05, 0.07],
            NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap();
        let option = test_option(PutOrCall::Call, curve);
        assert_approx_eq!(option.npv(), 14.2312547860, 1e-8);
        assert_approx_eq!(option.base.risk_free_rate(), 0.05, 1e-12);
    }
}
