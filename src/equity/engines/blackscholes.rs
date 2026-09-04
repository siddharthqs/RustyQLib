use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::core::utils::{norm_cdf, norm_pdf};
use crate::equity::asian::{self, AsianStrikeType, AveragingType};
use crate::equity::barrier;
use crate::equity::utils::PayoffType;
use crate::equity::vanilla_option::{
    AsianPayoff, BarrierPayoff, BinaryPayoff, BinaryType, EquityOption,
};
use libm::exp;

pub struct BlackScholesPricer;

/// One evaluation's cached market reads for the vanilla closed forms;
/// see [`BlackScholesPricer::vanilla_inputs`].
struct VanillaInputs {
    s: f64,
    k: f64,
    r: f64,
    q: f64,
    sigma: f64,
    t: f64,
    sqrt_t: f64,
    d1: f64,
    d2: f64,
    /// Curve discount factor to maturity.
    df_r: f64,
    /// Carry discount `e^{-qT}`.
    df_q: f64,
}
impl Default for BlackScholesPricer {
    fn default() -> Self {
        Self::new()
    }
}

impl BlackScholesPricer {
    pub fn new() -> Self {
        BlackScholesPricer
    }
    pub fn npv(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(
            bsd_option.time_to_maturity() >= 0.0,
            "Option is expired or negative time"
        );
        assert!(
            bsd_option.market.spot.mid() >= 0.0,
            "Negative underlying price not allowed"
        );
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
            PayoffType::Chooser => Self::chooser_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0),
            PayoffType::VarianceSwap => self.npv_variance_swap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
        }
    }
    pub fn delta(&self, bsd_option: &EquityOption) -> f64 {
        //assert!(bsd_option.volatility >= 0.0);
        assert!(
            bsd_option.time_to_maturity() >= 0.0,
            "Option is expired or negative time"
        );
        assert!(
            bsd_option.market.spot.mid() >= 0.0,
            "Negative underlying price not allowed"
        );
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
            PayoffType::Chooser => self.delta_chooser(bsd_option),
            PayoffType::VarianceSwap => self.delta_vswap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
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
            PayoffType::Chooser => self.gamma_chooser(bsd_option),
            PayoffType::VarianceSwap => self.gamma_vswap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
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
            PayoffType::Chooser => self.vega_chooser(bsd_option),
            PayoffType::VarianceSwap => self.vega_vswap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
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
            PayoffType::Chooser => self.theta_chooser(bsd_option),
            PayoffType::VarianceSwap => self.theta_vswap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
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
            PayoffType::Chooser => self.rho_chooser(bsd_option),
            PayoffType::VarianceSwap => self.rho_vswap(bsd_option),
            other => unreachable!(
                "check_engine_support admits only analytic payoffs here; got {other:?}"
            ),
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
                bsd_option.effective_spot(),
                bsd_option.base.strike_price,
                bsd_option.risk_free_rate(),
                bsd_option.carry_yield(),
                bsd_option.volatility(),
                bsd_option.time_to_maturity(),
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
                bsd_option.effective_spot(),
                bsd_option.base.strike_price,
                bsd_option.risk_free_rate(),
                bsd_option.carry_yield(),
                bsd_option.volatility(),
                bsd_option.time_to_maturity(),
                *bsd_option.payoff.put_or_call(),
            );
        }
        self.charm_bumped(bsd_option)
    }
    /// Percentage gamma (Haug's GammaP), `S * gamma / 100`: the change
    /// in delta per 1% move in the underlying.
    pub fn gamma_p(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.gamma_p_black76(bsd_option);
        }
        bsd_option.market.spot.value() * self.gamma(bsd_option) / 100.0
    }
    /// Zomma (`d gamma / d volatility`). Vanilla and Black-76 use closed
    /// forms; other analytic payoffs use a central volatility bump.
    pub fn zomma(&self, bsd_option: &EquityOption) -> f64 {
        if bsd_option.base.is_futures_option() {
            return self.zomma_black76(bsd_option);
        }
        if matches!(bsd_option.payoff.payoff_kind(), PayoffType::Vanilla) {
            return bs_zomma(
                bsd_option.effective_spot(),
                bsd_option.base.strike_price,
                bsd_option.risk_free_rate(),
                bsd_option.carry_yield(),
                bsd_option.volatility(),
                bsd_option.time_to_maturity(),
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
                bsd_option.effective_spot(),
                bsd_option.base.strike_price,
                bsd_option.risk_free_rate(),
                bsd_option.carry_yield(),
                bsd_option.volatility(),
                bsd_option.time_to_maturity(),
            );
        }
        self.volga_bumped(bsd_option)
    }
    // ── Black-76: European options on a future ─────────────────────────
    // The underlying_price is the futures price F; there is no spot,
    // dividend or carry. Vol is read from the surface at (K, F, T).

    fn black76_inputs(
        bsd_option: &EquityOption,
    ) -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        PutOrCall,
        crate::equity::black76::FuturesSettlement,
    ) {
        let f = bsd_option.market.spot.value();
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.market.vol_surface.vol(k, f, t);
        let settlement = bsd_option
            .base
            .futures_settlement
            .expect("black76 pricer called on a non-futures option");
        (
            f,
            k,
            r,
            sigma,
            t,
            *bsd_option.payoff.put_or_call(),
            settlement,
        )
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
    /// The market reads one vanilla evaluation needs, gathered **once**:
    /// each accessor cascade — the surface lookup (with its
    /// dividend-escrowed forward), the curve solve, the dividend PV
    /// loop, the day count — previously re-ran for every `d1()`/`d2()`
    /// and discount-factor reference, ~10 cascades per price where one
    /// suffices. Formulas match the `EquityOption` accessors expression
    /// for expression, so values are bit-identical.
    fn vanilla_inputs(o: &EquityOption) -> VanillaInputs {
        let t = o.time_to_maturity();
        let sqrt_t = t.sqrt();
        let s = o.effective_spot();
        let k = o.base.strike_price;
        let r = o.risk_free_rate();
        let q = o.carry_yield();
        let sigma = o.volatility();
        let d1 = ((s / k).ln() + (r - q + 0.5 * sigma.powi(2)) * t) / (sigma * sqrt_t);
        let d2 = d1 - sigma * sqrt_t;
        VanillaInputs {
            s,
            k,
            r,
            q,
            sigma,
            t,
            sqrt_t,
            d1,
            d2,
            df_r: o.maturity_discount_factor(),
            df_q: exp(-q * t),
        }
    }
    fn npv_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        let i = Self::vanilla_inputs(bsd_option);
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => i.s * norm_cdf(i.d1) * i.df_q - i.k * norm_cdf(i.d2) * i.df_r,
            PutOrCall::Put => i.k * norm_cdf(-i.d2) * i.df_r - i.s * norm_cdf(-i.d1) * i.df_q,
        }
    }
    fn delta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // spot delta: e^{-qT} N(d1) for a call, e^{-qT}(N(d1)-1) for a put
        let i = Self::vanilla_inputs(bsd_option);
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => norm_cdf(i.d1) * i.df_q,
            PutOrCall::Put => (norm_cdf(i.d1) - 1.0) * i.df_q,
        }
    }
    fn gamma_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // e^{-qT} dN(d1) / (S sigma sqrt(T))
        let i = Self::vanilla_inputs(bsd_option);
        norm_pdf(i.d1) * i.df_q / (i.s * (i.sigma * i.sqrt_t))
    }
    fn vega_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // S e^{-qT} dN(d1) sqrt(T)
        let i = Self::vanilla_inputs(bsd_option);
        (i.s * i.df_q) * norm_pdf(i.d1) * i.sqrt_t
    }
    fn theta_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) + q S e^{-qT} N(d1) - r K e^{-rT} N(d2)
        // put:  -S e^{-qT} dN(d1) sigma/(2 sqrt(T)) - q S e^{-qT} N(-d1) + r K e^{-rT} N(-d2)
        let i = Self::vanilla_inputs(bsd_option);
        let df_s = i.s * i.df_q;
        let t1 = -df_s * norm_pdf(i.d1) * i.sigma / (2.0 * i.sqrt_t);
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => {
                t1 + i.q * df_s * norm_cdf(i.d1) - i.r * i.k * i.df_r * norm_cdf(i.d2)
            }
            PutOrCall::Put => {
                t1 - i.q * df_s * norm_cdf(-i.d1) + i.r * i.k * i.df_r * norm_cdf(-i.d2)
            }
        }
    }
    fn rho_vanilla(&self, bsd_option: &EquityOption) -> f64 {
        // call: K T e^{-rT} N(d2); put: -K T e^{-rT} N(-d2)
        let i = Self::vanilla_inputs(bsd_option);
        let r1 = i.t * i.k;
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => r1 * norm_cdf(i.d2) * i.df_r,
            PutOrCall::Put => -r1 * norm_cdf(-i.d2) * i.df_r,
        }
    }

    /// Price through a market view — the analytic engine's entry in the
    /// central bumped-market reprice path. The payoff arms below take the
    /// shifts as scalars (they are pure closed forms); the view owns the
    /// conventions, and its elapsed time is the arms' maturity shift with
    /// the sign flipped.
    /// Escrowed spot under an additive bump, floored just above zero —
    /// a deep down-bump must not hand the closed forms a negative spot
    /// (`ln` of which is NaN).
    fn bumped_spot(o: &EquityOption, ds: f64) -> f64 {
        (o.effective_spot() + ds)
            .max(o.market.spot.value() * crate::equity::conventions::MIN_BUMPED_SPOT_FRAC)
    }

    /// Vol under an additive bump, floored at
    /// [`conventions::MIN_BUMPED_VOL`](crate::equity::conventions::MIN_BUMPED_VOL).
    fn bumped_vol(o: &EquityOption, dsigma: f64) -> f64 {
        (o.volatility() + dsigma).max(crate::equity::conventions::MIN_BUMPED_VOL)
    }

    pub(crate) fn price_bumped(
        option: &EquityOption,
        m: &crate::equity::bump::BumpedMarket,
    ) -> f64 {
        let b = m.bump();
        Self::price_with(option, b.d_spot, b.d_vol, b.d_rate, -b.d_time)
    }

    /// Analytic price with bumpable spot, volatility, rate and expiry.  This
    /// supports the cross-Greeks of payoffs whose first-order formulas are
    /// deliberately implemented by bump-and-reprice, and the bumped-market
    /// reprice through [`price_bumped`](Self::price_bumped).
    fn price_with(bsd_option: &EquityOption, ds: f64, dsigma: f64, dr: f64, dt_shift: f64) -> f64 {
        match bsd_option.payoff.payoff_kind() {
            PayoffType::Vanilla => bs_price(
                Self::bumped_spot(bsd_option, ds),
                bsd_option.base.strike_price,
                bsd_option.risk_free_rate() + dr,
                bsd_option.carry_yield(),
                Self::bumped_vol(bsd_option, dsigma),
                bsd_option.time_to_maturity() + dt_shift,
                *bsd_option.payoff.put_or_call(),
            ),
            PayoffType::Binary => Self::binary_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Barrier => Self::barrier_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Asian => Self::asian_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::ForwardStart => {
                Self::forward_start_price_with(bsd_option, ds, dsigma, dr, dt_shift)
            }
            PayoffType::Lookback => Self::lookback_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::Chooser => Self::chooser_price_with(bsd_option, ds, dsigma, dr, dt_shift),
            PayoffType::VarianceSwap => {
                Self::variance_swap_price_with(bsd_option, ds, dsigma, dr, dt_shift)
            }
            _ => panic!("cross-Greeks are not available for this analytic payoff"),
        }
    }
    fn vanna_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hs = bsd_option.market.spot.value() * 1e-4;
        let hv = 1e-4;
        (Self::price_with(bsd_option, hs, hv, 0.0, 0.0)
            - Self::price_with(bsd_option, -hs, hv, 0.0, 0.0)
            - Self::price_with(bsd_option, hs, -hv, 0.0, 0.0)
            + Self::price_with(bsd_option, -hs, -hv, 0.0, 0.0))
            / (4.0 * hs * hv)
    }
    fn charm_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hs = bsd_option.market.spot.value() * 1e-4;
        let ht = (1.0 / 365.0_f64).min(0.5 * bsd_option.time_to_maturity());
        -(Self::price_with(bsd_option, hs, 0.0, 0.0, ht)
            - Self::price_with(bsd_option, -hs, 0.0, 0.0, ht)
            - Self::price_with(bsd_option, hs, 0.0, 0.0, -ht)
            + Self::price_with(bsd_option, -hs, 0.0, 0.0, -ht))
            / (4.0 * hs * ht)
    }
    fn zomma_bumped(&self, bsd_option: &EquityOption) -> f64 {
        let hv = 1e-4;
        let hs = bsd_option.market.spot.value() * 1e-3;
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
        let s = Self::bumped_spot(bsd_option, ds);
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let sigma = Self::bumped_vol(bsd_option, dsigma);
        let t = bsd_option.time_to_maturity() + dt_shift;
        let (d1, d2) = bs_d1_d2(s, k, r, q, sigma, t);
        match (binary_type, bsd_option.payoff.put_or_call()) {
            (BinaryType::CashOrNothing, PutOrCall::Call) => cash * (-r * t).exp() * norm_cdf(d2),
            (BinaryType::CashOrNothing, PutOrCall::Put) => cash * (-r * t).exp() * norm_cdf(-d2),
            (BinaryType::AssetOrNothing, PutOrCall::Call) => s * (-q * t).exp() * norm_cdf(d1),
            (BinaryType::AssetOrNothing, PutOrCall::Put) => s * (-q * t).exp() * norm_cdf(-d1),
        }
    }

    fn npv_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let df_r = bsd_option.maturity_discount_factor();
        let df_q = exp(-bsd_option.carry_yield() * bsd_option.time_to_maturity());
        let s = bsd_option.effective_spot();
        match (binary_type, bsd_option.payoff.put_or_call()) {
            (BinaryType::CashOrNothing, PutOrCall::Call) => cash * df_r * norm_cdf(bsd_option.d2()),
            (BinaryType::CashOrNothing, PutOrCall::Put) => cash * df_r * norm_cdf(-bsd_option.d2()),
            (BinaryType::AssetOrNothing, PutOrCall::Call) => s * df_q * norm_cdf(bsd_option.d1()),
            (BinaryType::AssetOrNothing, PutOrCall::Put) => s * df_q * norm_cdf(-bsd_option.d1()),
        }
    }
    fn delta_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.volatility();
        let s = bsd_option.effective_spot();
        let vol_sqrt_t = sigma * t.sqrt();
        match binary_type {
            BinaryType::CashOrNothing => {
                // +- cash e^{-rT} dN(d2) / (S sigma sqrt(T))
                let df_r = bsd_option.maturity_discount_factor();
                let delta_call = cash * df_r * norm_pdf(bsd_option.d2()) / (s * vol_sqrt_t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => delta_call,
                    PutOrCall::Put => -delta_call,
                }
            }
            BinaryType::AssetOrNothing => {
                // e^{-qT} (N(+-d1) +- dN(d1)/(sigma sqrt(T)))
                let df_q = exp(-bsd_option.carry_yield() * t);
                let d1 = bsd_option.d1();
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
        let sigma = bsd_option.volatility();
        let s = bsd_option.effective_spot();
        let vol_sqrt_t = sigma * t.sqrt();
        let gamma_call = match binary_type {
            BinaryType::CashOrNothing => {
                // - cash e^{-rT} dN(d2) d1 / (S^2 sigma^2 T)
                let df_r = bsd_option.maturity_discount_factor();
                -cash * df_r * norm_pdf(bsd_option.d2()) * bsd_option.d1()
                    / (s * s * sigma * sigma * t)
            }
            BinaryType::AssetOrNothing => {
                // e^{-qT} dN(d1) (1 - d1/(sigma sqrt(T))) / (S sigma sqrt(T))
                let df_q = exp(-bsd_option.carry_yield() * t);
                let d1 = bsd_option.d1();
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
        let sigma = bsd_option.volatility();
        let s = bsd_option.effective_spot();
        let vega_call = match binary_type {
            BinaryType::CashOrNothing => {
                // - cash e^{-rT} dN(d2) d1 / sigma
                let df_r = bsd_option.maturity_discount_factor();
                -cash * df_r * norm_pdf(bsd_option.d2()) * bsd_option.d1() / sigma
            }
            BinaryType::AssetOrNothing => {
                // - S e^{-qT} dN(d1) d2 / sigma
                let df_q = exp(-bsd_option.carry_yield() * t);
                -s * df_q * norm_pdf(bsd_option.d1()) * bsd_option.d2() / sigma
            }
        };
        match bsd_option.payoff.put_or_call() {
            PutOrCall::Call => vega_call,
            PutOrCall::Put => -vega_call,
        }
    }
    fn theta_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let r = bsd_option.risk_free_rate();
        let q = bsd_option.carry_yield();
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.volatility();
        let s = bsd_option.effective_spot();
        match binary_type {
            BinaryType::CashOrNothing => {
                // dd2/dT = (r - q - sigma^2/2)/(sigma sqrt(T)) - d2/(2T)
                let df_r = bsd_option.maturity_discount_factor();
                let d2 = bsd_option.d2();
                let dd2_dt = (r - q - 0.5 * sigma * sigma) / (sigma * t.sqrt()) - d2 / (2.0 * t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => {
                        cash * (r * df_r * norm_cdf(d2) - df_r * norm_pdf(d2) * dd2_dt)
                    }
                    PutOrCall::Put => {
                        cash * (r * df_r * norm_cdf(-d2) + df_r * norm_pdf(d2) * dd2_dt)
                    }
                }
            }
            BinaryType::AssetOrNothing => {
                // dd1/dT = (r - q + sigma^2/2)/(sigma sqrt(T)) - d1/(2T)
                let df_q = exp(-q * t);
                let d1 = bsd_option.d1();
                let dd1_dt = (r - q + 0.5 * sigma * sigma) / (sigma * t.sqrt()) - d1 / (2.0 * t);
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => {
                        q * s * df_q * norm_cdf(d1) - s * df_q * norm_pdf(d1) * dd1_dt
                    }
                    PutOrCall::Put => {
                        q * s * df_q * norm_cdf(-d1) + s * df_q * norm_pdf(d1) * dd1_dt
                    }
                }
            }
        }
    }
    fn rho_binary(&self, bsd_option: &EquityOption) -> f64 {
        let (binary_type, cash) = Self::binary_details(bsd_option);
        let t = bsd_option.time_to_maturity();
        let sigma = bsd_option.volatility();
        let s = bsd_option.effective_spot();
        match binary_type {
            BinaryType::CashOrNothing => {
                let df_r = bsd_option.maturity_discount_factor();
                let d2 = bsd_option.d2();
                match bsd_option.payoff.put_or_call() {
                    PutOrCall::Call => {
                        cash * (-t * df_r * norm_cdf(d2) + df_r * norm_pdf(d2) * t.sqrt() / sigma)
                    }
                    PutOrCall::Put => {
                        cash * (-t * df_r * norm_cdf(-d2) - df_r * norm_pdf(d2) * t.sqrt() / sigma)
                    }
                }
            }
            BinaryType::AssetOrNothing => {
                // +- S e^{-qT} dN(d1) sqrt(T)/sigma
                let df_q = exp(-bsd_option.carry_yield() * t);
                let rho_call = s * df_q * norm_pdf(bsd_option.d1()) * t.sqrt() / sigma;
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
        let s = Self::bumped_spot(bsd_option, ds);
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let sigma = Self::bumped_vol(bsd_option, dsigma);
        let t = bsd_option.time_to_maturity() + dt_shift;
        let pc = *bsd_option.payoff.put_or_call();
        match payoff.barrier2 {
            Some(b2) => {
                // Unreachable backstop: `check_engine_support` refuses a
                // rebated double barrier on the Analytical engine before
                // pricing (the closed form below has no rebate term), so
                // neither the builder nor the JSON path can reach this. It
                // stays a hard assert for options assembled around those
                // boundaries — silently dropping the rebate would misprice.
                assert!(
                    payoff.rebate == 0.0,
                    "double-barrier rebates are not supported analytically; use MonteCarlo (rebate at expiry)"
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
                    s,
                    k,
                    payoff.barrier,
                    payoff.rebate,
                    r,
                    q,
                    sigma,
                    t,
                    payoff.direction,
                    payoff.knock,
                    timing,
                    pc,
                )
            }
        }
    }
    fn npv_barrier(&self, bsd_option: &EquityOption) -> f64 {
        Self::barrier_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-4;
        (Self::barrier_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::barrier_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_barrier(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-3;
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

    // ── Variance swaps (log-contract replication) ──────────────────────
    // NPV integrates the bound smile through the model-free replication
    // (continuous monitoring); Greeks are central-difference bumps of it,
    // the barrier pattern. The forward is the escrowed one (cash
    // dividends carved out), the standard price-return convention —
    // dividend jumps do not accrue variance.

    /// Replication value with additive bumps to (spot, vol, rate,
    /// expiry). The vol bump shifts the whole smile in parallel.
    fn variance_swap_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        use crate::equity::variance_swap::{
            fair_corridor_variance_strike, fair_gamma_swap_strike, fair_variance_strike,
            VarianceSwapKind, VarianceSwapPayoff,
        };
        let vs = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<VarianceSwapPayoff>()
            .expect("variance-swap route reached with a non-variance-swap payoff");
        let t = (bsd_option.time_to_maturity() + dt_shift).max(1e-6);
        let s = Self::bumped_spot(bsd_option, ds);
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let forward = s * ((r - q) * t).exp();
        let surface = &bsd_option.market.vol_surface;
        let smile = |k: f64| {
            (surface.vol(k, forward, t) + dsigma).max(crate::equity::conventions::MIN_BUMPED_VOL)
        };
        let fair = match vs.kind {
            VarianceSwapKind::Variance => fair_variance_strike(forward, t, smile),
            VarianceSwapKind::Gamma => fair_gamma_swap_strike(s, forward, t, smile),
            VarianceSwapKind::Corridor { low, high } => {
                fair_corridor_variance_strike(forward, t, low, high, smile)
            }
        };
        let total = vs.blend(fair, t);
        vs.notional * (total - vs.strike_variance) * (-r * t).exp()
    }
    fn npv_variance_swap(&self, bsd_option: &EquityOption) -> f64 {
        Self::variance_swap_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_vswap(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-4;
        (Self::variance_swap_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::variance_swap_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_vswap(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-3;
        (Self::variance_swap_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::variance_swap_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
            + Self::variance_swap_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_vswap(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::variance_swap_price_with(bsd_option, 0.0, h, 0.0, 0.0)
            - Self::variance_swap_price_with(bsd_option, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_vswap(&self, bsd_option: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * bsd_option.time_to_maturity());
        -(Self::variance_swap_price_with(bsd_option, 0.0, 0.0, 0.0, h)
            - Self::variance_swap_price_with(bsd_option, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_vswap(&self, bsd_option: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::variance_swap_price_with(bsd_option, 0.0, 0.0, h, 0.0)
            - Self::variance_swap_price_with(bsd_option, 0.0, 0.0, -h, 0.0))
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
        let s = Self::bumped_spot(bsd_option, ds);
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let sigma = Self::bumped_vol(bsd_option, dsigma);
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
        let anchor = bsd_option.effective_spot();
        let s = anchor + ds;
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let sigma = Self::bumped_vol(bsd_option, dsigma);
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
                    s,
                    anchor.min(s),
                    r,
                    q,
                    sigma,
                    t,
                    pc,
                )
            }
            (LookbackType::FloatingStrike, PutOrCall::Put) => {
                crate::equity::lookback::floating_strike_lookback_price(
                    s,
                    anchor.max(s),
                    r,
                    q,
                    sigma,
                    t,
                    pc,
                )
            }
            (LookbackType::FixedStrike, PutOrCall::Call) => {
                crate::equity::lookback::fixed_strike_lookback_price(
                    s,
                    k,
                    anchor.max(s),
                    r,
                    q,
                    sigma,
                    t,
                    pc,
                )
            }
            (LookbackType::FixedStrike, PutOrCall::Put) => {
                crate::equity::lookback::fixed_strike_lookback_price(
                    s,
                    k,
                    anchor.min(s),
                    r,
                    q,
                    sigma,
                    t,
                    pc,
                )
            }
        }
    }

    /// Chooser (Rubinstein 1991): two vanilla evaluations for the simple
    /// contract, critical-spot solve plus bivariate normals for the
    /// complex one. The choice and leg dates are calendar anchors, so a
    /// maturity shift moves them together with the expiry.
    fn chooser_price_with(
        bsd_option: &EquityOption,
        ds: f64,
        dsigma: f64,
        dr: f64,
        dt_shift: f64,
    ) -> f64 {
        let payoff = bsd_option
            .payoff
            .as_any()
            .downcast_ref::<crate::equity::chooser::ChooserPayoff>()
            .expect("payoff of kind Chooser must be a ChooserPayoff");
        let s = Self::bumped_spot(bsd_option, ds);
        let k = bsd_option.base.strike_price;
        let r = bsd_option.risk_free_rate() + dr;
        let q = bsd_option.carry_yield();
        let sigma = Self::bumped_vol(bsd_option, dsigma);
        let t_base = bsd_option.time_to_maturity();
        let t = (t_base + dt_shift).max(1e-8);
        let t1 = (payoff.choice_fraction * t_base + dt_shift).clamp(1e-8, t);
        match payoff.legs {
            None => crate::equity::chooser::simple_chooser_price(s, k, r, q, sigma, t1, t),
            Some(legs) => {
                let t_call = (legs.call_expiry_fraction * t_base + dt_shift).clamp(t1, t);
                let t_put = (legs.put_expiry_fraction * t_base + dt_shift).clamp(t1, t);
                crate::equity::chooser::complex_chooser_price(
                    s,
                    legs.call_strike,
                    legs.put_strike,
                    r,
                    q,
                    sigma,
                    t1,
                    t_call,
                    t_put,
                )
            }
        }
    }

    fn delta_chooser(&self, o: &EquityOption) -> f64 {
        let h = o.market.spot.value() * 1e-4;
        (Self::chooser_price_with(o, h, 0.0, 0.0, 0.0)
            - Self::chooser_price_with(o, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_chooser(&self, o: &EquityOption) -> f64 {
        let h = o.market.spot.value() * 1e-3;
        (Self::chooser_price_with(o, h, 0.0, 0.0, 0.0)
            - 2.0 * Self::chooser_price_with(o, 0.0, 0.0, 0.0, 0.0)
            + Self::chooser_price_with(o, -h, 0.0, 0.0, 0.0))
            / (h * h)
    }
    fn vega_chooser(&self, o: &EquityOption) -> f64 {
        let h = 1e-4;
        (Self::chooser_price_with(o, 0.0, h, 0.0, 0.0)
            - Self::chooser_price_with(o, 0.0, -h, 0.0, 0.0))
            / (2.0 * h)
    }
    fn theta_chooser(&self, o: &EquityOption) -> f64 {
        let h = (1.0 / 365.0_f64).min(0.5 * o.time_to_maturity());
        -(Self::chooser_price_with(o, 0.0, 0.0, 0.0, h)
            - Self::chooser_price_with(o, 0.0, 0.0, 0.0, -h))
            / (2.0 * h)
    }
    fn rho_chooser(&self, o: &EquityOption) -> f64 {
        let h = 1e-5;
        (Self::chooser_price_with(o, 0.0, 0.0, h, 0.0)
            - Self::chooser_price_with(o, 0.0, 0.0, -h, 0.0))
            / (2.0 * h)
    }

    fn delta_lookback(&self, o: &EquityOption) -> f64 {
        let h = o.market.spot.value() * 1e-4;
        (Self::lookback_price_with(o, h, 0.0, 0.0, 0.0)
            - Self::lookback_price_with(o, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_lookback(&self, o: &EquityOption) -> f64 {
        let h = o.market.spot.value() * 1e-3;
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
        let h = bsd_option.market.spot.value() * 1e-4;
        (Self::asian_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::asian_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_asian(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-3;
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
            Self::bumped_spot(bsd_option, ds),
            payoff.strike_fraction,
            bsd_option.risk_free_rate() + dr,
            bsd_option.carry_yield(),
            Self::bumped_vol(bsd_option, dsigma),
            payoff.start_fraction * t,
            t,
            *bsd_option.payoff.put_or_call(),
        )
    }
    fn npv_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        Self::forward_start_price_with(bsd_option, 0.0, 0.0, 0.0, 0.0)
    }
    fn delta_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-4;
        (Self::forward_start_price_with(bsd_option, h, 0.0, 0.0, 0.0)
            - Self::forward_start_price_with(bsd_option, -h, 0.0, 0.0, 0.0))
            / (2.0 * h)
    }
    fn gamma_forward_start(&self, bsd_option: &EquityOption) -> f64 {
        let h = bsd_option.market.spot.value() * 1e-3;
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

/// The Black-Scholes `(d1, d2)` pair as a pure function of its inputs,
/// shared by the closed forms in this file.
fn bs_d1_d2(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> (f64, f64) {
    let sqrt_t = t.sqrt();
    let d1 = ((s / k).ln() + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    (d1, d2)
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
    let (d1, d2) = bs_d1_d2(s, k, r, q, sigma, t);
    match put_or_call {
        PutOrCall::Call => s * exp(-q * t) * norm_cdf(d1) - k * exp(-r * t) * norm_cdf(d2),
        PutOrCall::Put => k * exp(-r * t) * norm_cdf(-d2) - s * exp(-q * t) * norm_cdf(-d1),
    }
}

/// Black-Scholes vega as a pure function (per unit of vol).
pub fn bs_vega(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let (d1, _) = bs_d1_d2(s, k, r, q, sigma, t);
    s * exp(-q * t) * norm_pdf(d1) * sqrt_t
}

/// Black-Scholes vanna, the change in spot delta per unit of volatility.
pub fn bs_vanna(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let (d1, _) = bs_d1_d2(s, k, r, q, sigma, t);
    exp(-q * t) * norm_pdf(d1) * (sqrt_t - d1 / sigma)
}

/// Black-Scholes charm, the change in spot delta per year of calendar time.
pub fn bs_charm(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64, put_or_call: PutOrCall) -> f64 {
    let sqrt_t = t.sqrt();
    let (d1, _) = bs_d1_d2(s, k, r, q, sigma, t);
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
    let (d1, d2) = bs_d1_d2(s, k, r, q, sigma, t);
    let gamma = exp(-q * t) * norm_pdf(d1) / (s * sigma * sqrt_t);
    gamma * (d1 * d2 - 1.0) / sigma
}

/// Black-Scholes volga (vomma), the change in vega per unit of volatility,
/// `vega * d1 * d2 / sigma`. Same for calls and puts (parity is
/// volatility-independent); negative at the money, positive in the wings.
pub fn bs_volga(s: f64, k: f64, r: f64, q: f64, sigma: f64, t: f64) -> f64 {
    let sqrt_t = t.sqrt();
    let (d1, d2) = bs_d1_d2(s, k, r, q, sigma, t);
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
) -> Result<f64, RustyQLibError> {
    if t <= 0.0 {
        return Err(RustyQLibError::NumericalError(
            "option is expired".to_string(),
        ));
    }
    if !t.is_finite() {
        return Err(RustyQLibError::invalid_input(
            "time_to_maturity",
            format!("time to maturity must be finite, got {t}"),
        ));
    }
    for (name, x) in [("spot", s), ("strike", k)] {
        if !(x.is_finite() && x > 0.0) {
            return Err(RustyQLibError::invalid_input(
                name,
                format!("{name} must be positive and finite, got {x}"),
            ));
        }
    }
    // a live option's price is strictly positive; zero, negative or NaN
    // quotes would otherwise ride through the bound checks as NaN
    if !(target.is_finite() && target > 0.0) {
        return Err(RustyQLibError::invalid_input(
            "option_price",
            format!("option price must be positive and finite, got {target}"),
        ));
    }
    let lower_bound = bs_price(s, k, r, q, 0.0, t, put_or_call);
    let upper_bound = match put_or_call {
        PutOrCall::Call => s * exp(-q * t),
        PutOrCall::Put => k * exp(-r * t),
    };
    if target < lower_bound - 1e-12 || target > upper_bound + 1e-12 {
        return Err(RustyQLibError::NumericalError(format!(
            "price {target} violates arbitrage bounds [{lower_bound}, {upper_bound}]"
        )));
    }

    let (lo, hi) = (IMPLIED_VOL_MIN, IMPLIED_VOL_MAX);
    if bs_price(s, k, r, q, lo, t, put_or_call) > target {
        return Ok(lo); // at or below the vol floor
    }
    if bs_price(s, k, r, q, hi, t, put_or_call) < target {
        return Err(RustyQLibError::NumericalError(format!(
            "implied vol above {IMPLIED_VOL_MAX}"
        )));
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

#[cfg(test)]
#[path = "blackscholes_tests.rs"]
mod tests;
