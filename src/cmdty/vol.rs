//! The volatility quote a commodity option is priced with.
//!
//! The distribution model travels with the vol quote, not the contract
//! (the QuantLib/ORE convention): the same option prices under
//! Black-76, shifted Black-76 or Bachelier depending on how the market
//! quotes the vol. A bare `f64` converts to [`CommodityVol::Lognormal`],
//! so the common Black-76 case stays `option.price(&d, &f, 0.35)`.

use crate::core::errors::RustyQLibError;

/// A flat volatility quote and the model it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommodityVol {
    /// Black-76 lognormal vol (the quoting standard for positive-price
    /// hubs — Henry Hub, WTI in normal times, gold).
    Lognormal(f64),
    /// Displaced lognormal: `F + shift` follows geometric Brownian
    /// motion, supporting prices down to `-shift`. The shift is a
    /// market convention chosen per hub, not calibrated from a quote.
    ShiftedLognormal { vol: f64, shift: f64 },
    /// Bachelier (normal) vol, quoted in price units per √year. Any
    /// price sign — the model CME switched energy options to when WTI
    /// went negative in April 2020, and the natural quote for basis.
    Normal(f64),
}

impl From<f64> for CommodityVol {
    fn from(vol: f64) -> Self {
        CommodityVol::Lognormal(vol)
    }
}

impl CommodityVol {
    /// The vol number, whatever the model.
    pub fn vol(self) -> f64 {
        match self {
            CommodityVol::Lognormal(v) | CommodityVol::Normal(v) => v,
            CommodityVol::ShiftedLognormal { vol, .. } => vol,
        }
    }

    /// The same quote with its vol moved by `h`, floored at zero (for
    /// central bumps).
    pub(crate) fn bumped_vol(self, h: f64) -> Self {
        match self {
            CommodityVol::Lognormal(v) => CommodityVol::Lognormal((v + h).max(0.0)),
            CommodityVol::ShiftedLognormal { vol, shift } => CommodityVol::ShiftedLognormal {
                vol: (vol + h).max(0.0),
                shift,
            },
            CommodityVol::Normal(v) => CommodityVol::Normal((v + h).max(0.0)),
        }
    }

    /// Reject non-finite or negative vols and non-finite shifts.
    pub(crate) fn validated(self, field: &str) -> Result<Self, RustyQLibError> {
        let vol = self.vol();
        if !vol.is_finite() || vol < 0.0 {
            return Err(RustyQLibError::invalid_input(
                field,
                format!("volatility must be non-negative, got {vol}"),
            ));
        }
        if let CommodityVol::ShiftedLognormal { shift, .. } = self {
            if !shift.is_finite() {
                return Err(RustyQLibError::invalid_input(
                    field,
                    format!("shift must be finite, got {shift}"),
                ));
            }
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_f64_is_a_lognormal_quote() {
        assert_eq!(CommodityVol::from(0.35), CommodityVol::Lognormal(0.35));
        assert_eq!(CommodityVol::Lognormal(0.35).vol(), 0.35);
    }

    #[test]
    fn validation_rejects_bad_quotes() {
        assert!(CommodityVol::Lognormal(-0.1).validated("x").is_err());
        assert!(CommodityVol::Normal(f64::NAN).validated("x").is_err());
        assert!(CommodityVol::ShiftedLognormal {
            vol: 0.3,
            shift: f64::INFINITY
        }
        .validated("x")
        .is_err());
        assert!(CommodityVol::ShiftedLognormal {
            vol: 0.3,
            shift: 10.0
        }
        .validated("x")
        .is_ok());
    }
}
