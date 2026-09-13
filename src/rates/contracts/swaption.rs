//! European swaption: the option to enter a [`VanillaSwap`] at expiry,
//! priced under a one-factor affine short-rate model.
//!
//! The product owns the dates and conventions — the underlying swap's
//! schedule, day counts and calendar — and maps them to the year
//! fractions the [`jamshidian`] engine works in. The map is taken
//! against an **anchor** curve's reference date and day count (for
//! Hull-White, the model's own fitted curve, see
//! [`npv_hull_white`](Swaption::npv_hull_white)), which is what keeps
//! the models date-agnostic.
//!
//! Under a one-factor model the floating leg is worth par at the swap
//! start whatever its frequency, so only the fixed leg's schedule
//! enters the price: a payer swaption is a put on the fixed leg
//! (coupons plus redemption) struck at the notional, a receiver the
//! call — the bond-option equivalence, decomposed by Jamshidian into
//! zero-bond options. The settlement lag between exercise and the
//! swap's (adjusted) effective date is carried exactly: the notional
//! is exchanged at the swap start, so the strike is that many
//! start-date bonds, and each Jamshidian piece is an exchange option.
//!
//! The market quotes swaptions in vol, not in a short-rate model:
//! [`npv_black`](Swaption::npv_black) prices from a normal, Black or
//! shifted-Black vol ([`RateVol`]) as `annuity * kernel`, and
//! [`implied_normal_vol`](Swaption::implied_normal_vol) /
//! [`implied_black_vol`](Swaption::implied_black_vol) read a premium —
//! a market price or a model price — back as a vol. That is the bridge
//! both ways: quote a vol, get the calibration price
//! ([`to_quote_from_vol`](Swaption::to_quote_from_vol)); price under
//! Hull-White, see the normal vol it implies
//! ([`implied_normal_vol_hull_white`](Swaption::implied_normal_vol_hull_white)).
//!
//! [`jamshidian`]: crate::rates::engines::jamshidian

use chrono::NaiveDate;

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::rates::contracts::vanilla_swap::VanillaSwap;
use crate::rates::engines::black::{
    implied_black_vol, implied_normal_vol, swaption_from_vol, swaption_side, RateVol,
};
use crate::rates::engines::jamshidian::european_swaption_settled;
use crate::rates::leg::annuity;
use crate::rates::models::calibration::SwaptionQuote;
use crate::rates::models::{HullWhite, OneFactorAffine};
use crate::rates::PayerReceiver;

const FIELD: &str = "swaption";

/// A European option, exercisable on `expiry_date`, to enter `swap` as
/// written. The strike is the swap's `fixed_rate` and the payer /
/// receiver side is the swap's.
#[derive(Debug, Clone)]
pub struct Swaption {
    pub swap: VanillaSwap,
    /// Exercise date; must not be after the swap's effective date.
    pub expiry_date: NaiveDate,
}

impl Swaption {
    pub fn new(swap: VanillaSwap, expiry_date: NaiveDate) -> Result<Self, RustyQLibError> {
        if expiry_date > swap.effective_date {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "expiry {expiry_date} must not be after the swap's effective date {}",
                    swap.effective_date
                ),
            ));
        }
        if swap.fixed_rate <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the bond-option equivalence needs a positive fixed rate, got {}",
                    swap.fixed_rate
                ),
            ));
        }
        Ok(Swaption { swap, expiry_date })
    }

    /// The swaption's side: the underlying swap's.
    pub fn payer_receiver(&self) -> PayerReceiver {
        self.swap.payer_receiver
    }

    /// Option expiry as a year fraction from `anchor`'s reference date.
    pub fn expiry(&self, anchor: &YieldCurve) -> Result<f64, RustyQLibError> {
        let t = year_fraction_from(anchor, self.expiry_date);
        if t <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "expiry {} is not after the anchor date {}",
                    self.expiry_date,
                    anchor.reference_date()
                ),
            ));
        }
        Ok(t)
    }

    /// The swap start — the first fixed period's adjusted start — as a
    /// year fraction from `anchor`'s reference date.
    pub fn swap_start(&self, anchor: &YieldCurve) -> Result<f64, RustyQLibError> {
        let first = self.swap.fixed_periods()?[0];
        Ok(year_fraction_from(anchor, first.start))
    }

    /// The fixed leg as the engine sees it: `(payment_time, accrual)`
    /// pairs, payment times on `anchor`'s day count from its reference
    /// date and accruals on the swap's fixed day count.
    pub fn fixed_leg_times(&self, anchor: &YieldCurve) -> Result<Vec<(f64, f64)>, RustyQLibError> {
        Ok(self
            .swap
            .fixed_periods()?
            .iter()
            .map(|p| {
                (
                    year_fraction_from(anchor, p.payment),
                    self.swap.fixed_day_count.year_fraction(p.start, p.end),
                )
            })
            .collect())
    }

    /// The forward par rate of the underlying swap — the ATM strike.
    pub fn forward_swap_rate(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        self.swap.par_rate(discount, forecast)
    }

    /// The fixed-leg annuity on the swap's notional: the PV of one unit
    /// of fixed rate, and the swaption's vega scale.
    pub fn annuity(&self, discount: &YieldCurve) -> Result<f64, RustyQLibError> {
        Ok(self.swap.notional
            * annuity(
                &self.swap.fixed_periods()?,
                self.swap.fixed_day_count,
                discount,
            ))
    }

    /// Value under `model`, with dates mapped to year fractions against
    /// `anchor`. For a curve-fitted model pass the curve it was fitted
    /// to (or use [`npv_hull_white`](Self::npv_hull_white)).
    pub fn npv(
        &self,
        model: &impl OneFactorAffine,
        anchor: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        european_swaption_settled(
            model,
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
        )
    }

    /// [`npv`](Self::npv) under Hull-White, anchored on the model's own
    /// fitted curve.
    pub fn npv_hull_white(&self, model: &HullWhite) -> Result<f64, RustyQLibError> {
        self.npv(model, model.curve())
    }

    /// Value under any [`Gaussian1dModel`] (Hull-White in Gaussian1d
    /// form, the Markov functional model) through the deflated
    /// quadrature engine, with dates mapped against `anchor`.
    ///
    /// [`Gaussian1dModel`]: crate::rates::models::gaussian1d::Gaussian1dModel
    pub fn npv_gaussian1d(
        &self,
        model: &impl crate::rates::models::gaussian1d::Gaussian1dModel,
        anchor: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        crate::rates::engines::gaussian1d::european_swaption(
            model,
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
        )
    }

    /// Value under Hull-White by finite differences on the state PDE.
    pub fn npv_fd_hull_white(
        &self,
        model: &HullWhite,
        config: &crate::rates::engines::fd_hull_white::FdConfig,
    ) -> Result<f64, RustyQLibError> {
        let anchor = model.curve();
        crate::rates::engines::fd_hull_white::european_swaption(
            model,
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
            config,
        )
    }

    /// Value under G2++ by ADI finite differences on the two-factor PDE.
    pub fn npv_fd_g2pp(
        &self,
        model: &crate::rates::models::g2pp::G2pp,
        config: &crate::rates::engines::fd_g2pp::FdG2Config,
    ) -> Result<f64, RustyQLibError> {
        let anchor = model.curve();
        crate::rates::engines::fd_g2pp::european_swaption(
            model,
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
            config,
        )
    }

    /// Value under Black-Karasinski on its fitted tree, anchored on the
    /// model's curve.
    pub fn npv_black_karasinski(
        &self,
        model: &crate::rates::models::black_karasinski::BlackKarasinski,
    ) -> Result<f64, RustyQLibError> {
        let anchor = model.curve();
        model.european_swaption(
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
        )
    }

    /// Value under the two-factor G2++ model, anchored on its curve.
    pub fn npv_g2pp(
        &self,
        model: &crate::rates::models::g2pp::G2pp,
    ) -> Result<f64, RustyQLibError> {
        let anchor = model.curve();
        model.european_swaption(
            self.expiry(anchor)?,
            self.swap_start(anchor)?,
            &self.fixed_leg_times(anchor)?,
            self.swap.fixed_rate,
            self.swap.notional,
            self.swap.payer_receiver,
        )
    }

    /// The calibration quote for this swaption at `market_price` (on
    /// the swap's notional; the quote is stored per unit notional).
    pub fn to_quote(
        &self,
        anchor: &YieldCurve,
        market_price: f64,
    ) -> Result<SwaptionQuote, RustyQLibError> {
        Ok(SwaptionQuote {
            expiry: self.expiry(anchor)?,
            swap_start: self.swap_start(anchor)?,
            fixed_leg: self.fixed_leg_times(anchor)?,
            strike_rate: self.swap.fixed_rate,
            market_price: market_price / self.swap.notional,
            payer_receiver: self.swap.payer_receiver,
        })
    }

    // ── market vol: Black / Bachelier on the forward swap rate ─────────

    /// The market formula: `annuity * kernel(forward, strike, vol, T)`
    /// with the forward swap rate off `forecast`, the annuity and the
    /// expiry off `discount`. Pass the same curve twice for a
    /// single-curve setup.
    pub fn npv_black(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        vol: RateVol,
    ) -> Result<f64, RustyQLibError> {
        swaption_from_vol(
            self.annuity(discount)?,
            self.forward_swap_rate(discount, forecast)?,
            self.swap.fixed_rate,
            self.expiry(discount)?,
            vol,
            self.swap.payer_receiver,
        )
    }

    /// Value off a SABR cube: the smile at this swaption's expiry and
    /// tenor (the swap's length in years) read at its strike and
    /// forward, priced through the market formula.
    pub fn npv_sabr(
        &self,
        cube: &crate::rates::models::sabr::SabrSwaptionCube,
        discount: &YieldCurve,
        forecast: &YieldCurve,
    ) -> Result<f64, RustyQLibError> {
        let expiry = self.expiry(discount)?;
        let periods = self.swap.fixed_periods()?;
        let tenor = year_fraction_from(discount, periods[periods.len() - 1].end)
            - year_fraction_from(discount, periods[0].start);
        let forward = self.forward_swap_rate(discount, forecast)?;
        let vol = cube.vol(expiry, tenor, forward, self.swap.fixed_rate)?;
        self.npv_black(discount, forecast, vol)
    }

    /// The Bachelier (normal) vol, in absolute rate units per √year,
    /// that reproduces `premium` (on the swap's notional).
    pub fn implied_normal_vol(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        premium: f64,
    ) -> Result<f64, RustyQLibError> {
        implied_normal_vol(
            self.annuity(discount)?,
            self.forward_swap_rate(discount, forecast)?,
            self.swap.fixed_rate,
            self.expiry(discount)?,
            swaption_side(self.swap.payer_receiver),
            premium,
        )
    }

    /// The (shifted) Black-76 vol that reproduces `premium`; `shift`
    /// zero for the plain lognormal quote.
    pub fn implied_black_vol(
        &self,
        discount: &YieldCurve,
        forecast: &YieldCurve,
        premium: f64,
        shift: f64,
    ) -> Result<f64, RustyQLibError> {
        implied_black_vol(
            self.annuity(discount)?,
            self.forward_swap_rate(discount, forecast)?,
            self.swap.fixed_rate,
            self.expiry(discount)?,
            swaption_side(self.swap.payer_receiver),
            premium,
            shift,
        )
    }

    /// The normal vol the Hull-White model implies for this swaption:
    /// its model price read through the market formula on the model's
    /// own curve. The number to put next to the screen quote.
    pub fn implied_normal_vol_hull_white(&self, model: &HullWhite) -> Result<f64, RustyQLibError> {
        let curve = model.curve();
        self.implied_normal_vol(curve, curve, self.npv_hull_white(model)?)
    }

    /// The calibration quote for this swaption from a market vol: the
    /// vol is turned into a price on `curve` and stored per unit
    /// notional — the way a screen's normal-vol grid feeds
    /// [`calibrate_hull_white`](crate::rates::models::calibration::calibrate_hull_white).
    pub fn to_quote_from_vol(
        &self,
        curve: &YieldCurve,
        vol: RateVol,
    ) -> Result<SwaptionQuote, RustyQLibError> {
        let price = self.npv_black(curve, curve, vol)?;
        self.to_quote(curve, price)
    }
}

/// Year fraction from `anchor`'s reference date to `date` on the
/// curve's day count — the same map the curve uses for `df_date`.
pub(crate) fn year_fraction_from(anchor: &YieldCurve, date: NaiveDate) -> f64 {
    anchor
        .day_count()
        .year_fraction(anchor.reference_date(), date)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::curves::{Compounding, InterpolationMethod, Tenor};
    use crate::core::daycount::DayCountConvention;
    use crate::rates::models::calibration::calibrate_hull_white_sigma;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn asof() -> NaiveDate {
        d(2026, 8, 13)
    }

    fn market_curve() -> YieldCurve {
        YieldCurve::from_zero_rates(
            &[
                Tenor::YearFraction(0.5),
                Tenor::YearFraction(1.0),
                Tenor::YearFraction(2.0),
                Tenor::YearFraction(5.0),
                Tenor::YearFraction(10.0),
                Tenor::YearFraction(30.0),
            ],
            &[0.040, 0.041, 0.042, 0.044, 0.045, 0.046],
            asof(),
            DayCountConvention::Act365,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )
        .unwrap()
    }

    fn model(sigma: f64) -> HullWhite {
        HullWhite::new(0.05, sigma, market_curve()).unwrap()
    }

    /// 1y-into-5y USD-standard swap on 10mm.
    fn swap(fixed_rate: f64, side: PayerReceiver) -> VanillaSwap {
        VanillaSwap::usd_standard(
            10_000_000.0,
            fixed_rate,
            side,
            d(2027, 8, 16),
            d(2032, 8, 16),
        )
        .unwrap()
    }

    #[test]
    fn payer_minus_receiver_is_the_forward_swap() {
        // parity: the difference of the two swaptions is the underlying
        // forward-starting swap, whose value the curve gives directly —
        // and the float leg's frequency is irrelevant (it telescopes)
        let m = model(0.011);
        let curve = m.curve();
        let strike = 0.045;
        let payer = Swaption::new(swap(strike, PayerReceiver::Payer), d(2027, 8, 12)).unwrap();
        let receiver =
            Swaption::new(swap(strike, PayerReceiver::Receiver), d(2027, 8, 12)).unwrap();
        let diff = payer.npv_hull_white(&m).unwrap() - receiver.npv_hull_white(&m).unwrap();
        let forward_swap = payer.swap.pv(curve).unwrap();
        assert!(
            (diff - forward_swap).abs() < 1e-6 * payer.swap.notional,
            "{diff} vs {forward_swap}"
        );
    }

    #[test]
    fn atm_swaptions_are_symmetric_and_grow_with_volatility() {
        let calm = model(0.005);
        let wild = model(0.012);
        let curve = calm.curve();
        let probe = swap(0.04, PayerReceiver::Payer);
        let atm = Swaption::new(probe, d(2027, 8, 12))
            .unwrap()
            .forward_swap_rate(curve, curve)
            .unwrap();
        let payer = Swaption::new(swap(atm, PayerReceiver::Payer), d(2027, 8, 12)).unwrap();
        let receiver = Swaption::new(swap(atm, PayerReceiver::Receiver), d(2027, 8, 12)).unwrap();
        let p = payer.npv_hull_white(&calm).unwrap();
        let r = receiver.npv_hull_white(&calm).unwrap();
        assert!((p - r).abs() < 1e-6 * payer.swap.notional, "{p} vs {r}");
        assert!(p > 0.0);
        assert!(payer.npv_hull_white(&wild).unwrap() > p);
        assert!(payer.annuity(curve).unwrap() > 0.0);
    }

    #[test]
    fn product_matches_the_engine_on_its_own_schedule() {
        let m = model(0.011);
        let curve = m.curve();
        let s = Swaption::new(swap(0.045, PayerReceiver::Payer), d(2027, 8, 12)).unwrap();
        let direct = european_swaption_settled(
            &m,
            s.expiry(curve).unwrap(),
            s.swap_start(curve).unwrap(),
            &s.fixed_leg_times(curve).unwrap(),
            0.045,
            10_000_000.0,
            PayerReceiver::Payer,
        )
        .unwrap();
        assert_eq!(s.npv(&m, curve).unwrap(), direct);
        // the schedule is semiannual: 10 fixed periods over 5 years
        assert_eq!(s.fixed_leg_times(curve).unwrap().len(), 10);
    }

    #[test]
    fn quotes_from_products_calibrate_the_generating_model() {
        let market = model(0.009);
        let curve = market.curve();
        let grid = [
            (d(2027, 8, 12), d(2027, 8, 16), d(2031, 8, 16)),
            (d(2028, 8, 14), d(2028, 8, 16), d(2031, 8, 16)),
        ];
        let quotes: Vec<SwaptionQuote> = grid
            .iter()
            .map(|&(expiry, effective, maturity)| {
                let probe =
                    VanillaSwap::usd_standard(1.0, 0.04, PayerReceiver::Payer, effective, maturity)
                        .unwrap();
                let atm = probe.par_rate(curve, curve).unwrap();
                let underlying =
                    VanillaSwap::usd_standard(1.0, atm, PayerReceiver::Payer, effective, maturity)
                        .unwrap();
                let s = Swaption::new(underlying, expiry).unwrap();
                let price = s.npv_hull_white(&market).unwrap();
                s.to_quote(curve, price).unwrap()
            })
            .collect();
        let fit = calibrate_hull_white_sigma(curve, &quotes, 0.05, 0.02).unwrap();
        assert!(
            (fit.model.sigma() - 0.009).abs() < 1e-5,
            "sigma {}",
            fit.model.sigma()
        );
    }

    #[test]
    fn market_vol_prices_and_implied_vols_round_trip() {
        let m = model(0.011);
        let curve = m.curve();
        let probe = swap(0.04, PayerReceiver::Payer);
        let atm = Swaption::new(probe, d(2027, 8, 12))
            .unwrap()
            .forward_swap_rate(curve, curve)
            .unwrap();
        let payer = Swaption::new(swap(atm, PayerReceiver::Payer), d(2027, 8, 12)).unwrap();
        let receiver = Swaption::new(swap(atm, PayerReceiver::Receiver), d(2027, 8, 12)).unwrap();
        // ATM normal: payer = receiver, and the price is annuity * sigma sqrt(T) / sqrt(2 pi)
        let vol = RateVol::Normal(0.0085);
        let p = payer.npv_black(curve, curve, vol).unwrap();
        let r = receiver.npv_black(curve, curve, vol).unwrap();
        assert!((p - r).abs() < 1e-6, "{p} vs {r}");
        let textbook = payer.annuity(curve).unwrap() * 0.0085 * payer.expiry(curve).unwrap().sqrt()
            / (2.0 * std::f64::consts::PI).sqrt();
        assert!((p - textbook).abs() < 1e-6, "{p} vs {textbook}");
        // round trips through both quotes
        let v = payer.implied_normal_vol(curve, curve, p).unwrap();
        assert!((v - 0.0085).abs() < 1e-10, "normal {v}");
        let black = payer
            .npv_black(curve, curve, RateVol::Lognormal(0.21))
            .unwrap();
        let v = payer.implied_black_vol(curve, curve, black, 0.0).unwrap();
        assert!((v - 0.21).abs() < 1e-10, "black {v}");
        // the Hull-White price reads as a normal vol near the model's
        // own sigma (a = 5% pulls it a little below 110bp over 1y-5y)
        let hw_vol = payer.implied_normal_vol_hull_white(&m).unwrap();
        assert!(
            hw_vol > 0.008 && hw_vol < 0.011,
            "HW-implied normal vol {hw_vol}"
        );
        // and a more volatile model implies a higher vol
        assert!(payer.implied_normal_vol_hull_white(&model(0.013)).unwrap() > hw_vol);
    }

    #[test]
    fn vol_quotes_calibrate_hull_white() {
        // a screen of ATM normal vols -> prices -> a fitted sigma that
        // reproduces the screen through the model
        let curve = market_curve();
        let grid = [
            (d(2027, 8, 12), d(2027, 8, 16), d(2032, 8, 16), 0.0090),
            (d(2028, 8, 14), d(2028, 8, 16), d(2033, 8, 16), 0.0088),
            (d(2029, 8, 13), d(2029, 8, 15), d(2034, 8, 15), 0.0085),
        ];
        let swaptions: Vec<Swaption> = grid
            .iter()
            .map(|&(expiry, effective, maturity, _)| {
                let probe =
                    VanillaSwap::usd_standard(1.0, 0.04, PayerReceiver::Payer, effective, maturity)
                        .unwrap();
                let atm = probe.par_rate(&curve, &curve).unwrap();
                let underlying =
                    VanillaSwap::usd_standard(1.0, atm, PayerReceiver::Payer, effective, maturity)
                        .unwrap();
                Swaption::new(underlying, expiry).unwrap()
            })
            .collect();
        let quotes: Vec<SwaptionQuote> = swaptions
            .iter()
            .zip(grid.iter())
            .map(|(s, &(_, _, _, vol))| s.to_quote_from_vol(&curve, RateVol::Normal(vol)).unwrap())
            .collect();
        let fit = calibrate_hull_white_sigma(&curve, &quotes, 0.05, 0.02).unwrap();
        assert!(fit.price_rmse < 0.02, "rmse {}", fit.price_rmse);
        assert!(
            fit.model.sigma() > 0.008 && fit.model.sigma() < 0.012,
            "sigma {}",
            fit.model.sigma()
        );
        // the fitted model's implied vols sit near the screen
        for (s, &(_, _, _, vol)) in swaptions.iter().zip(grid.iter()) {
            let implied = s.implied_normal_vol_hull_white(&fit.model).unwrap();
            assert!((implied - vol).abs() < 0.0005, "{implied} vs screen {vol}");
        }
    }

    #[test]
    fn a_sabr_cube_prices_by_strike() {
        use crate::rates::engines::black::RateVolKind;
        use crate::rates::models::sabr::{RateSabr, SabrSwaptionCube};
        let curve = market_curve();
        let mk = |alpha: f64| RateSabr::new(alpha, 0.0, -0.3, 0.4, 0.0).unwrap();
        let cube = SabrSwaptionCube::new(
            vec![1.0, 2.0],
            vec![5.0],
            vec![vec![mk(0.0090)], vec![mk(0.0085)]],
            RateVolKind::Normal,
        )
        .unwrap();
        let probe = swap(0.04, PayerReceiver::Payer);
        let atm = Swaption::new(probe, d(2027, 8, 12))
            .unwrap()
            .forward_swap_rate(&curve, &curve)
            .unwrap();
        // at the money the cube quote is the node smile's ATM vol
        let at = Swaption::new(swap(atm, PayerReceiver::Payer), d(2027, 8, 12)).unwrap();
        let expiry = at.expiry(&curve).unwrap();
        let smile = cube.smile(expiry, 5.0);
        let expected = at
            .npv_black(
                &curve,
                &curve,
                RateVol::Normal(smile.normal_vol(atm, atm, expiry).unwrap()),
            )
            .unwrap();
        let priced = at.npv_sabr(&cube, &curve, &curve).unwrap();
        assert!((priced - expected).abs() < 1e-6, "{priced} vs {expected}");
        // a low-strike receiver picks up the skew: dearer than at the ATM vol
        let low = Swaption::new(swap(atm - 0.01, PayerReceiver::Receiver), d(2027, 8, 12)).unwrap();
        let skewed = low.npv_sabr(&cube, &curve, &curve).unwrap();
        let flat = low
            .npv_black(
                &curve,
                &curve,
                RateVol::Normal(smile.normal_vol(atm, atm, expiry).unwrap()),
            )
            .unwrap();
        assert!(skewed > flat, "{skewed} vs {flat}");
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        // expiry after the swap start
        assert!(Swaption::new(swap(0.045, PayerReceiver::Payer), d(2027, 9, 1)).is_err());
        // non-positive strike
        assert!(Swaption::new(swap(0.0, PayerReceiver::Payer), d(2027, 8, 12)).is_err());
        // already expired against the anchor
        let spot = VanillaSwap::usd_standard(
            1.0,
            0.04,
            PayerReceiver::Payer,
            d(2026, 8, 13),
            d(2031, 8, 13),
        )
        .unwrap();
        let s = Swaption::new(spot, d(2026, 8, 13)).unwrap();
        assert!(s.npv_hull_white(&model(0.01)).is_err());
    }
}
