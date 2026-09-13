//! Market-quote engine for rate options: Black-76 (lognormal, plain or
//! shifted) and Bachelier (normal) on the forward rate, times the
//! annuity — the formulas swaption and cap/floor vols are **quoted**
//! in, and the bridge between those quotes and the short-rate models.
//!
//! - A European swaption is `annuity * kernel(F, K, sigma, T)` with `F`
//!   the forward swap rate and the annuity the fixed leg's PV01: a
//!   payer is a call on the rate, a receiver a put.
//! - A caplet is `notional * tau * df(pay) * kernel(L, K, sigma, T)`
//!   with `L` the period's forward rate; a cap sums its caplets, all at
//!   one **flat** vol.
//!
//! Since 2015 or so swaptions and caps trade in normal (Bachelier) vol,
//! quoted in basis points per √year — the lognormal quote broke down
//! when rates went negative — with shifted-lognormal a common
//! alternative for EUR/CHF. All three are here as [`RateVol`].
//!
//! [`implied_normal_vol`] and [`implied_black_vol`] invert a premium
//! by bisection on the monotone premium-in-vol map; the products wrap
//! them so a model price can be read as a market vol and a market vol
//! can be turned into a calibration price.

use crate::cmdty::bachelier;
use crate::core::errors::RustyQLibError;
use crate::core::solvers::Solver1d;
use crate::core::trade::PutOrCall;
use crate::equity::black76::{self, FuturesSettlement};
use crate::rates::PayerReceiver;

const FIELD: &str = "rate option";

/// The quoting convention of a [`RateVol`] without its number — what a
/// vol surface or a caplet term structure is expressed in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateVolKind {
    Normal,
    Lognormal,
    ShiftedLognormal { shift: f64 },
}

impl RateVolKind {
    /// A quote of this kind at `vol`.
    pub fn with_vol(self, vol: f64) -> RateVol {
        match self {
            RateVolKind::Normal => RateVol::Normal(vol),
            RateVolKind::Lognormal => RateVol::Lognormal(vol),
            RateVolKind::ShiftedLognormal { shift } => RateVol::ShiftedLognormal { vol, shift },
        }
    }
}

/// How a rate option's volatility is quoted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateVol {
    /// Bachelier (normal) vol in absolute rate units per √year:
    /// `0.0080` is 80 basis points — the current market standard.
    Normal(f64),
    /// Black-76 lognormal vol on the forward rate; needs positive
    /// forward and strike.
    Lognormal(f64),
    /// Displaced lognormal: `F + shift` is lognormal, supporting rates
    /// down to `-shift`. The shift is a market convention, not a fit.
    ShiftedLognormal { vol: f64, shift: f64 },
}

impl RateVol {
    /// The volatility number of the quote.
    pub fn vol(&self) -> f64 {
        match self {
            RateVol::Normal(v) | RateVol::Lognormal(v) => *v,
            RateVol::ShiftedLognormal { vol, .. } => *vol,
        }
    }

    /// The quoting convention without the number.
    pub fn kind(&self) -> RateVolKind {
        match self {
            RateVol::Normal(_) => RateVolKind::Normal,
            RateVol::Lognormal(_) => RateVolKind::Lognormal,
            RateVol::ShiftedLognormal { shift, .. } => {
                RateVolKind::ShiftedLognormal { shift: *shift }
            }
        }
    }

    fn validated(self) -> Result<Self, RustyQLibError> {
        let vol = self.vol();
        if !vol.is_finite() || vol < 0.0 {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("volatility must be non-negative, got {vol}"),
            ));
        }
        if let RateVol::ShiftedLognormal { shift, .. } = self {
            if !shift.is_finite() {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("shift must be finite, got {shift}"),
                ));
            }
        }
        Ok(self)
    }
}

/// The undiscounted option kernel on a forward rate: the expected
/// payoff `E[(F_T - K)^+]` (or the put) at `expiry` under the quote's
/// dynamics. Multiply by the annuity (swaption) or `tau * df` (caplet).
pub fn rate_option_kernel(
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: RateVol,
    put_or_call: PutOrCall,
) -> Result<f64, RustyQLibError> {
    let vol = vol.validated()?;
    if !(expiry > 0.0 && expiry.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("expiry must be positive, got {expiry}"),
        ));
    }
    if !forward.is_finite() || !strike.is_finite() {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("forward and strike must be finite, got {forward} and {strike}"),
        ));
    }
    let margined = FuturesSettlement::Margined; // the caller discounts
    Ok(match vol {
        RateVol::Normal(v) => {
            bachelier::price(forward, strike, 0.0, v, expiry, put_or_call, margined)
        }
        RateVol::Lognormal(v) => {
            displaced_inputs(forward, strike, 0.0)?;
            black76::price(forward, strike, 0.0, v, expiry, put_or_call, margined)
        }
        RateVol::ShiftedLognormal { vol, shift } => {
            let (f, k) = displaced_inputs(forward, strike, shift)?;
            black76::price(f, k, 0.0, vol, expiry, put_or_call, margined)
        }
    })
}

/// Shift and validate `(F, K)` for the (displaced) Black-76 kernel,
/// which is lognormal in `F + shift`.
fn displaced_inputs(forward: f64, strike: f64, shift: f64) -> Result<(f64, f64), RustyQLibError> {
    let (f, k) = (forward + shift, strike + shift);
    if f <= 0.0 || k <= 0.0 {
        let model = if shift == 0.0 {
            "a lognormal vol".to_string()
        } else {
            format!("shift {shift}")
        };
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!(
                "forward {forward} and strike {strike} must both be positive after \
                 the displacement under {model}; quote a normal vol or a larger shift"
            ),
        ));
    }
    Ok((f, k))
}

/// The option side of a swaption on the swap rate: a payer profits when
/// the rate is high (a call), a receiver when it is low (a put).
pub fn swaption_side(payer_receiver: PayerReceiver) -> PutOrCall {
    match payer_receiver {
        PayerReceiver::Payer => PutOrCall::Call,
        PayerReceiver::Receiver => PutOrCall::Put,
    }
}

/// European swaption premium from a quoted vol:
/// `annuity * kernel(forward, strike, vol, expiry)`.
pub fn swaption_from_vol(
    annuity: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    vol: RateVol,
    payer_receiver: PayerReceiver,
) -> Result<f64, RustyQLibError> {
    if !(annuity > 0.0 && annuity.is_finite()) {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("annuity must be positive, got {annuity}"),
        ));
    }
    Ok(annuity * rate_option_kernel(forward, strike, expiry, vol, swaption_side(payer_receiver))?)
}

/// The Bachelier vol that reproduces `premium` for an option whose
/// price is `scale * kernel(forward, strike, vol, expiry)` — `scale`
/// the annuity for a swaption, `tau * df` for a caplet. Bisection on
/// the monotone premium-in-vol map; errors when the premium sits
/// outside the no-arbitrage bounds.
pub fn implied_normal_vol(
    scale: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    put_or_call: PutOrCall,
    premium: f64,
) -> Result<f64, RustyQLibError> {
    // a normal vol has rate units: bracket to the market's scale
    let hi = 10.0 * (forward.abs() + strike.abs() + 0.01);
    invert_premium(premium, hi, |v| {
        Ok(scale * rate_option_kernel(forward, strike, expiry, RateVol::Normal(v), put_or_call)?)
    })
}

/// The (shifted) Black-76 vol that reproduces `premium`, with `shift`
/// zero for the plain lognormal quote. See [`implied_normal_vol`] for
/// the `scale`.
pub fn implied_black_vol(
    scale: f64,
    forward: f64,
    strike: f64,
    expiry: f64,
    put_or_call: PutOrCall,
    premium: f64,
    shift: f64,
) -> Result<f64, RustyQLibError> {
    displaced_inputs(forward, strike, shift)?;
    invert_premium(premium, 10.0, |vol| {
        Ok(scale
            * rate_option_kernel(
                forward,
                strike,
                expiry,
                RateVol::ShiftedLognormal { vol, shift },
                put_or_call,
            )?)
    })
}

/// Shared bisection on a monotone premium-in-vol function over
/// `[~0, hi]`. Written generically so a cap's flat vol (a sum of
/// caplets) inverts through the same path as a single option.
pub(crate) fn invert_premium(
    premium: f64,
    hi: f64,
    price_at: impl Fn(f64) -> Result<f64, RustyQLibError>,
) -> Result<f64, RustyQLibError> {
    if !premium.is_finite() || premium < 0.0 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("premium must be non-negative, got {premium}"),
        ));
    }
    let lo = 1e-10;
    let floor = price_at(lo)?;
    if floor > premium * (1.0 + 1e-9) + 1e-12 {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("premium {premium} is below the intrinsic value {floor}"),
        ));
    }
    if price_at(hi)? < premium {
        return Err(RustyQLibError::invalid_input(
            FIELD,
            format!("premium {premium} exceeds the price at vol {hi}"),
        ));
    }
    // the bracket straddles the root of a monotone function; a root the
    // solver only bracketed (rather than hit to tolerance) is still the
    // best estimate of the vol, so it is returned either way
    let root = Solver1d::new(1e-14, 200).bisection(
        |v| price_at(v).unwrap_or(f64::NAN) - premium,
        lo,
        hi,
    )?;
    Ok(root.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atm_payer_and_receiver_agree_and_parity_is_the_forward_swap() {
        let (annuity, forward, expiry) = (4.3, 0.045, 1.0);
        for vol in [
            RateVol::Normal(0.008),
            RateVol::Lognormal(0.20),
            RateVol::ShiftedLognormal {
                vol: 0.15,
                shift: 0.02,
            },
        ] {
            let payer =
                swaption_from_vol(annuity, forward, forward, expiry, vol, PayerReceiver::Payer)
                    .unwrap();
            let receiver = swaption_from_vol(
                annuity,
                forward,
                forward,
                expiry,
                vol,
                PayerReceiver::Receiver,
            )
            .unwrap();
            assert!(
                (payer - receiver).abs() < 1e-12,
                "{vol:?}: {payer} vs {receiver}"
            );
            assert!(payer > 0.0);
            // away from the money: payer - receiver = annuity (F - K)
            let strike = 0.05;
            let p = swaption_from_vol(annuity, forward, strike, expiry, vol, PayerReceiver::Payer)
                .unwrap();
            let r = swaption_from_vol(
                annuity,
                forward,
                strike,
                expiry,
                vol,
                PayerReceiver::Receiver,
            )
            .unwrap();
            assert!(
                (p - r - annuity * (forward - strike)).abs() < 1e-12,
                "{vol:?}"
            );
        }
    }

    #[test]
    fn atm_normal_price_is_the_textbook_closed_form() {
        // ATM Bachelier: annuity * sigma * sqrt(T) / sqrt(2 pi)
        let (annuity, forward, sigma, expiry) = (4.3, 0.045, 0.008, 2.25);
        let price = swaption_from_vol(
            annuity,
            forward,
            forward,
            expiry,
            RateVol::Normal(sigma),
            PayerReceiver::Payer,
        )
        .unwrap();
        let textbook = annuity * sigma * expiry.sqrt() / (2.0 * std::f64::consts::PI).sqrt();
        assert!((price - textbook).abs() < 1e-12, "{price} vs {textbook}");
    }

    #[test]
    fn implied_vols_round_trip() {
        let (annuity, forward, strike, expiry) = (4.3, 0.045, 0.05, 1.5);
        let side = PutOrCall::Call;
        let normal = swaption_from_vol(
            annuity,
            forward,
            strike,
            expiry,
            RateVol::Normal(0.0075),
            PayerReceiver::Payer,
        )
        .unwrap();
        let v = implied_normal_vol(annuity, forward, strike, expiry, side, normal).unwrap();
        assert!((v - 0.0075).abs() < 1e-10, "normal {v}");
        let black = swaption_from_vol(
            annuity,
            forward,
            strike,
            expiry,
            RateVol::Lognormal(0.22),
            PayerReceiver::Payer,
        )
        .unwrap();
        let v = implied_black_vol(annuity, forward, strike, expiry, side, black, 0.0).unwrap();
        assert!((v - 0.22).abs() < 1e-10, "black {v}");
        let shifted = swaption_from_vol(
            annuity,
            forward,
            strike,
            expiry,
            RateVol::ShiftedLognormal {
                vol: 0.18,
                shift: 0.02,
            },
            PayerReceiver::Payer,
        )
        .unwrap();
        let v = implied_black_vol(annuity, forward, strike, expiry, side, shifted, 0.02).unwrap();
        assert!((v - 0.18).abs() < 1e-10, "shifted {v}");
        // the same premium reads as a different number in each quote
        let as_normal = implied_normal_vol(annuity, forward, strike, expiry, side, black).unwrap();
        assert!(as_normal > 0.005 && as_normal < 0.015, "{as_normal}");
    }

    #[test]
    fn negative_rates_need_a_normal_or_shifted_quote() {
        let (annuity, forward, strike, expiry) = (4.3, -0.002, 0.001, 1.0);
        assert!(swaption_from_vol(
            annuity,
            forward,
            strike,
            expiry,
            RateVol::Lognormal(0.2),
            PayerReceiver::Payer
        )
        .is_err());
        assert!(
            swaption_from_vol(
                annuity,
                forward,
                strike,
                expiry,
                RateVol::Normal(0.006),
                PayerReceiver::Payer
            )
            .unwrap()
                > 0.0
        );
        assert!(
            swaption_from_vol(
                annuity,
                forward,
                strike,
                expiry,
                RateVol::ShiftedLognormal {
                    vol: 0.3,
                    shift: 0.01
                },
                PayerReceiver::Payer
            )
            .unwrap()
                > 0.0
        );
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        let (annuity, forward, strike, expiry) = (4.3, 0.045, 0.05, 1.0);
        assert!(swaption_from_vol(
            annuity,
            forward,
            strike,
            expiry,
            RateVol::Normal(-0.01),
            PayerReceiver::Payer
        )
        .is_err());
        assert!(swaption_from_vol(
            annuity,
            forward,
            strike,
            0.0,
            RateVol::Normal(0.01),
            PayerReceiver::Payer
        )
        .is_err());
        assert!(swaption_from_vol(
            0.0,
            forward,
            strike,
            expiry,
            RateVol::Normal(0.01),
            PayerReceiver::Payer
        )
        .is_err());
        // below intrinsic (in-the-money receiver worth less than annuity (K - F))
        let intrinsic = annuity * (strike - forward);
        assert!(implied_normal_vol(
            annuity,
            forward,
            strike,
            expiry,
            PutOrCall::Put,
            0.5 * intrinsic
        )
        .is_err());
        assert!(
            implied_normal_vol(annuity, forward, strike, expiry, PutOrCall::Call, -1.0).is_err()
        );
    }
}
