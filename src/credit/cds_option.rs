//! European options on a single-name credit default swap.
//!
//! A payer option is the right, at `expiry`, to **buy** protection on
//! the reference entity from `expiry` to the underlying's maturity at
//! the strike coupon; a receiver is the right to sell it. The strike is
//! the underlying [`CreditDefaultSwap`]'s coupon and the side is its
//! [`ProtectionSide`] — a protection buyer's option is the payer, a
//! seller's the receiver — so the contract has one source of truth for
//! both, the way [`Swaption`](crate::rates::contracts::swaption::Swaption)
//! takes its strike and side from the underlying swap.
//!
//! Everything is priced in the **forward risky annuity measure**. With
//! valuation date `t0`, expiry `te`, protection running to `T`, coupon
//! `c` and recovery `R`, all three quantities discounted to `t0` and
//! carrying survival from `t0`:
//!
//! ```text
//! A   = sum_i dcf_i df(t_i) S(t_i) + accrual on default   (te..T)
//! P   = (1 - R) integral_te^T  df(t) (-dS(t))
//! FEP = (1 - R) integral_t0^te df(t) (-dS(t))
//! F*  = (P + FEP) / A
//! ```
//!
//! `A` and `P` are the forward CDS's two legs — the same ISDA segment
//! integrals [`cds`](super::cds) uses, read at a valuation date before
//! the contract starts. `FEP` is the **front-end protection**: single-
//! name CDS options do not knock out, so a holder whose name defaults
//! before expiry still exercises and collects `1 - R`. Folding that
//! value into the forward gives the loss-adjusted spread `F*` (Pedersen
//! 2003), and the option is then an ordinary call or put on `F*`:
//!
//! ```text
//! payer    = A * Black_call(F*, c, sigma, te - t0)
//! receiver = A * Black_put (F*, c, sigma, te - t0)
//! ```
//!
//! Because `A` is built from survival *past* expiry, the annuity
//! measure already knocks the pre-expiry default states out of both
//! legs; `FEP` is exactly what puts them back into the payer. Put-call
//! parity is then the no-knockout forward contract,
//! [`forward_value`](CdsOption::forward_value):
//!
//! ```text
//! payer - receiver = A (F* - c) = P + FEP - c A
//! ```
//!
//! Setting [`knockout`](CdsOption#structfield.knockout) drops `FEP` and
//! prices `F* = P / A` — the knockout convention some index and
//! bespoke contracts trade on.
//!
//! The vol is quoted through [`RateVol`], so a lognormal quote on the
//! spread (the single-name standard), a normal quote (the usual choice
//! for tight index spreads) and a shifted-lognormal all price through
//! the same kernel, and [`implied_vol`](CdsOption::implied_vol) reads a
//! premium back in any of them.

use chrono::NaiveDate;

use super::cds::{CreditDefaultSwap, ProtectionSide};
use super::curve::CreditCurve;
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::trade::PutOrCall;
use crate::rates::engines::black::{implied_black_vol, implied_normal_vol, rate_option_kernel};

pub use crate::rates::engines::black::{RateVol, RateVolKind};

const FIELD: &str = "cds option";

/// A European option to enter `underlying` at `expiry`, struck at that
/// contract's coupon and taking its protection side.
#[derive(Debug, Clone)]
pub struct CdsOption {
    /// The forward-starting CDS delivered on exercise. Its
    /// `effective_date` is the protection start, its `coupon` the
    /// strike and its `side` the option's payer / receiver sense.
    pub underlying: CreditDefaultSwap,
    /// Exercise date; must not be after the underlying's effective
    /// date.
    pub expiry: NaiveDate,
    /// Whether the option is cancelled by a default before expiry. The
    /// single-name market standard is `false` — the holder keeps the
    /// front-end protection.
    pub knockout: bool,
}

impl CdsOption {
    /// A standard contract: no knockout, so a default before expiry is
    /// still collected by the payer.
    pub fn new(underlying: CreditDefaultSwap, expiry: NaiveDate) -> Result<Self, RustyQLibError> {
        let option = CdsOption {
            underlying,
            expiry,
            knockout: false,
        };
        option.validate()?;
        Ok(option)
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.expiry > self.underlying.effective_date {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "expiry {} must not follow the underlying's effective date {}",
                    self.expiry, self.underlying.effective_date
                ),
            ));
        }
        if !(self.underlying.coupon > 0.0 && self.underlying.coupon.is_finite()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "the strike is the underlying's coupon and must be positive, got {}",
                    self.underlying.coupon
                ),
            ));
        }
        Ok(())
    }

    /// The strike spread: the coupon of the CDS entered on exercise.
    pub fn strike(&self) -> f64 {
        self.underlying.coupon
    }

    /// The option side on the spread: a payer (the protection buyer's
    /// option) profits when the spread widens, so it is the call.
    pub fn put_or_call(&self) -> PutOrCall {
        match self.underlying.side {
            ProtectionSide::Buyer => PutOrCall::Call,
            ProtectionSide::Seller => PutOrCall::Put,
        }
    }

    /// Whether this is a payer — the option to buy protection.
    pub fn is_payer(&self) -> bool {
        self.underlying.side == ProtectionSide::Buyer
    }

    /// Time to expiry on the discount curve's day count, measured from
    /// the valuation date.
    pub fn time_to_expiry(&self, curve: &YieldCurve, valuation: NaiveDate) -> f64 {
        curve.day_count().year_fraction(valuation, self.expiry)
    }

    /// The forward legs per unit notional as `(A, P)`: the risky
    /// annuity and the protection value of the CDS starting at expiry,
    /// both discounted to `valuation` and carrying survival from it.
    fn forward_legs(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<(f64, f64), RustyQLibError> {
        if valuation > self.expiry {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!(
                    "valuation {valuation} must not follow the expiry {}",
                    self.expiry
                ),
            ));
        }
        let annuity = self.underlying.risky_annuity(curve, credit, valuation)?;
        if !(annuity > 0.0 && annuity.is_finite()) {
            return Err(RustyQLibError::NumericalError(
                "the forward risky annuity is not positive".to_string(),
            ));
        }
        let protection = self
            .underlying
            .protection_leg_pv(curve, credit, valuation)?
            / self.underlying.notional;
        Ok((annuity, protection))
    }

    /// The forward risky annuity per unit notional: the value today of
    /// one unit of running coupon over the option's underlying, zero in
    /// every state where the name defaults before expiry.
    pub fn forward_annuity(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.forward_legs(curve, credit, valuation)?.0)
    }

    /// The plain forward par spread `P / A` — the coupon that makes the
    /// forward-starting CDS worth nothing, ignoring any default before
    /// expiry.
    pub fn forward_spread(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, protection) = self.forward_legs(curve, credit, valuation)?;
        Ok(protection / annuity)
    }

    /// The front-end protection per unit notional: the value of the
    /// loss paid on a default between `valuation` and expiry, which a
    /// no-knockout holder still collects. Zero for a knockout contract.
    pub fn front_end_protection(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if self.knockout || valuation >= self.expiry {
            return Ok(0.0);
        }
        // the protection leg of a zero-coupon stub over the option's life
        let stub = CreditDefaultSwap::new(
            1.0,
            0.0,
            self.underlying.recovery_rate,
            valuation,
            self.expiry,
        )?;
        stub.protection_leg_pv(curve, credit, valuation)
    }

    /// The loss-adjusted forward spread `F* = (P + FEP) / A` — the
    /// lognormal (or normal) variate the option is written on.
    pub fn loss_adjusted_forward_spread(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        Ok(self.forward_measures(curve, credit, valuation)?.1)
    }

    /// `(A, F*)` in one pass over the curves.
    fn forward_measures(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<(f64, f64), RustyQLibError> {
        let (annuity, protection) = self.forward_legs(curve, credit, valuation)?;
        let fep = self.front_end_protection(curve, credit, valuation)?;
        Ok((annuity, (protection + fep) / annuity))
    }

    /// Value to this side of the forward contract the option settles
    /// into, front-end protection included: `notional (P + FEP - c A)`
    /// for a payer, its negative for a receiver. Put-call parity is
    /// exactly this number.
    pub fn forward_value(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, forward) = self.forward_measures(curve, credit, valuation)?;
        let sign = if self.is_payer() { 1.0 } else { -1.0 };
        Ok(sign * self.underlying.notional * annuity * (forward - self.strike()))
    }

    /// The market formula: `notional * A * kernel(F*, c, vol, te)`.
    /// At or after expiry the option is worth its intrinsic value.
    pub fn npv(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
        vol: RateVol,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, forward) = self.forward_measures(curve, credit, valuation)?;
        let scale = self.underlying.notional * annuity;
        let expiry = self.time_to_expiry(curve, valuation);
        if expiry <= 0.0 {
            let sign = if self.is_payer() { 1.0 } else { -1.0 };
            return Ok(scale * (sign * (forward - self.strike())).max(0.0));
        }
        Ok(scale * rate_option_kernel(forward, self.strike(), expiry, vol, self.put_or_call())?)
    }

    /// The volatility of `kind` that reproduces `premium` on the
    /// option's notional — the bridge from a broker price back to the
    /// quote the desk carries.
    pub fn implied_vol(
        &self,
        curve: &YieldCurve,
        credit: &CreditCurve,
        valuation: NaiveDate,
        premium: f64,
        kind: RateVolKind,
    ) -> Result<f64, RustyQLibError> {
        let (annuity, forward) = self.forward_measures(curve, credit, valuation)?;
        let scale = self.underlying.notional * annuity;
        let expiry = self.time_to_expiry(curve, valuation);
        let (strike, side) = (self.strike(), self.put_or_call());
        match kind {
            RateVolKind::Normal => {
                implied_normal_vol(scale, forward, strike, expiry, side, premium)
            }
            RateVolKind::Lognormal => {
                implied_black_vol(scale, forward, strike, expiry, side, premium, 0.0)
            }
            RateVolKind::ShiftedLognormal { shift } => {
                implied_black_vol(scale, forward, strike, expiry, side, premium, shift)
            }
        }
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

    fn today() -> NaiveDate {
        d(2026, 8, 14)
    }

    /// The option expiry and the forward CDS's protection start.
    fn expiry() -> NaiveDate {
        d(2027, 6, 20)
    }

    fn flat_curve(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            today(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    /// 10mm payer struck at 120bp on a 5y CDS starting at the June 2027
    /// IMM date, 40% recovery.
    fn payer(strike: f64) -> CdsOption {
        let forward_cds =
            CreditDefaultSwap::new(10_000_000.0, strike, 0.40, expiry(), d(2032, 6, 20)).unwrap();
        CdsOption::new(forward_cds, expiry()).unwrap()
    }

    fn receiver(strike: f64) -> CdsOption {
        let mut option = payer(strike);
        option.underlying.side = ProtectionSide::Seller;
        option
    }

    /// On flat curves the spread term structure is flat, so the forward
    /// par spread sits on top of the spot par spread of the same tenor.
    #[test]
    fn forward_spread_matches_a_flat_spread_curve() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        let forward = payer(0.012)
            .forward_spread(&curve, &credit, today())
            .unwrap();
        let spot = CreditDefaultSwap::new(1.0, 0.01, 0.40, today(), d(2031, 6, 20))
            .unwrap()
            .par_spread(&curve, &credit, today())
            .unwrap();
        assert!((forward - spot).abs() < 1e-4, "{forward} vs {spot}");
        // and it is the credit triangle's spread, a shade inside
        assert!(
            (forward - 0.012).abs() < 0.0003 && forward < 0.012,
            "{forward}"
        );
    }

    /// FEP over `[t0, te]` on flat curves is the closed-form protection
    /// integral `(1 - R) lambda / mu (1 - exp(-mu te))`.
    #[test]
    fn front_end_protection_matches_the_closed_form() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        let option = payer(0.012);
        let fep = option
            .front_end_protection(&curve, &credit, today())
            .unwrap();
        let te = DayCountConvention::Act365.year_fraction(today(), expiry());
        let mu: f64 = 0.06;
        let expected = 0.6 * 0.02 / mu * (1.0 - (-mu * te).exp());
        assert!((fep - expected).abs() < 1e-9, "{fep} vs {expected}");
        // the loss-adjusted forward is the plain forward lifted by FEP/A
        let (annuity, plain) = (
            option.forward_annuity(&curve, &credit, today()).unwrap(),
            option.forward_spread(&curve, &credit, today()).unwrap(),
        );
        let adjusted = option
            .loss_adjusted_forward_spread(&curve, &credit, today())
            .unwrap();
        assert!((adjusted - plain - fep / annuity).abs() < 1e-12);
        assert!(adjusted > plain);
        // a knockout contract carries none of it
        let mut knockout = payer(0.012);
        knockout.knockout = true;
        assert_eq!(
            knockout
                .front_end_protection(&curve, &credit, today())
                .unwrap(),
            0.0
        );
        let ko = knockout
            .loss_adjusted_forward_spread(&curve, &credit, today())
            .unwrap();
        assert!((ko - plain).abs() < 1e-12);
    }

    /// `payer - receiver = notional A (F* - c)`, whatever the vol.
    #[test]
    fn put_call_parity_holds_at_every_vol() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        for strike in [0.005, 0.012, 0.03] {
            let (p, r) = (payer(strike), receiver(strike));
            let forward = p.forward_value(&curve, &credit, today()).unwrap();
            assert!((forward + r.forward_value(&curve, &credit, today()).unwrap()).abs() < 1e-9);
            for vol in [RateVol::Lognormal(0.6), RateVol::Normal(0.004)] {
                let payer_pv = p.npv(&curve, &credit, today(), vol).unwrap();
                let receiver_pv = r.npv(&curve, &credit, today(), vol).unwrap();
                assert!(
                    (payer_pv - receiver_pv - forward).abs() < 1e-6,
                    "{strike} {vol:?}: {payer_pv} - {receiver_pv} vs {forward}"
                );
            }
        }
    }

    /// Front-end protection is worth something to the payer and costs
    /// the receiver: it can only widen the forward.
    #[test]
    fn no_knockout_is_dearer_for_the_payer_and_cheaper_for_the_receiver() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        let vol = RateVol::Lognormal(0.6);
        for build in [payer as fn(f64) -> CdsOption, receiver] {
            let live = build(0.012);
            let mut knockout = build(0.012);
            knockout.knockout = true;
            let (with, without) = (
                live.npv(&curve, &credit, today(), vol).unwrap(),
                knockout.npv(&curve, &credit, today(), vol).unwrap(),
            );
            if live.is_payer() {
                assert!(with > without, "payer {with} vs {without}");
            } else {
                assert!(with < without, "receiver {with} vs {without}");
            }
        }
    }

    #[test]
    fn zero_vol_is_the_intrinsic_and_price_rises_with_vol() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        let option = payer(0.012);
        let (annuity, forward) = (
            option.forward_annuity(&curve, &credit, today()).unwrap(),
            option
                .loss_adjusted_forward_spread(&curve, &credit, today())
                .unwrap(),
        );
        let intrinsic = 10_000_000.0 * annuity * (forward - 0.012).max(0.0);
        let at_zero = option
            .npv(&curve, &credit, today(), RateVol::Lognormal(0.0))
            .unwrap();
        assert!(
            (at_zero - intrinsic).abs() < 1e-8,
            "{at_zero} vs {intrinsic}"
        );
        // and at expiry the option is worth exactly its intrinsic
        let at_expiry = option
            .npv(&curve, &credit, expiry(), RateVol::Lognormal(0.6))
            .unwrap();
        let spot_annuity = option.forward_annuity(&curve, &credit, expiry()).unwrap();
        let spot_forward = option.forward_spread(&curve, &credit, expiry()).unwrap();
        let expected = 10_000_000.0 * spot_annuity * (spot_forward - 0.012).max(0.0);
        assert!((at_expiry - expected).abs() < 1e-8, "{at_expiry}");

        let mut last = at_zero;
        for vol in [0.2, 0.4, 0.8, 1.2] {
            let pv = option
                .npv(&curve, &credit, today(), RateVol::Lognormal(vol))
                .unwrap();
            assert!(pv > last, "{vol}: {pv} vs {last}");
            last = pv;
        }
    }

    #[test]
    fn implied_vol_round_trips_in_every_quote() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        for option in [payer(0.012), receiver(0.012), payer(0.02)] {
            for quoted in [
                RateVol::Lognormal(0.55),
                RateVol::Normal(0.0045),
                RateVol::ShiftedLognormal {
                    vol: 0.5,
                    shift: 0.002,
                },
            ] {
                let premium = option.npv(&curve, &credit, today(), quoted).unwrap();
                let implied = option
                    .implied_vol(&curve, &credit, today(), premium, quoted.kind())
                    .unwrap();
                assert!(
                    (implied - quoted.vol()).abs() < 1e-6,
                    "{quoted:?}: {implied}"
                );
            }
        }
    }

    #[test]
    fn validation() {
        let (curve, credit) = (flat_curve(0.04), CreditCurve::flat(0.02).unwrap());
        // expiry after the underlying's protection start
        let forward_cds =
            CreditDefaultSwap::new(1.0, 0.012, 0.40, expiry(), d(2032, 6, 20)).unwrap();
        assert!(CdsOption::new(forward_cds.clone(), d(2027, 9, 20)).is_err());
        // an expiry before the start is fine — the settlement lag
        assert!(CdsOption::new(forward_cds, d(2027, 6, 18)).is_ok());
        // a zero strike has no lognormal quote
        let zero_coupon = CreditDefaultSwap::new(1.0, 0.0, 0.40, expiry(), d(2032, 6, 20)).unwrap();
        assert!(CdsOption::new(zero_coupon, expiry()).is_err());
        // valuation after expiry
        let option = payer(0.012);
        assert!(option
            .npv(&curve, &credit, d(2027, 8, 1), RateVol::Lognormal(0.6))
            .is_err());
        // a negative premium has no implied vol
        assert!(option
            .implied_vol(&curve, &credit, today(), -1.0, RateVolKind::Lognormal)
            .is_err());
    }
}
