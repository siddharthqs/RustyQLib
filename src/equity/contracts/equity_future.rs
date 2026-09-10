//! An equity futures position: `multiplier` units marked against the
//! entry price, so the value is the mark-to-market P&L
//! `sign * (F - entry) * multiplier`. The mark `F` is the contract's
//! `current_price` when supplied, otherwise the theoretical futures
//! price `S e^{(r - q - b) T}` on the flat carry.
use crate::core::data_models::EquityFutureData;
use crate::core::errors::RustyQLibError;
use crate::core::quotes::Quote;
use crate::core::traits::Instrument;
use crate::equity::utils::LongShort;
use chrono::NaiveDate;

pub struct EquityFuture {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    pub underlying_price: Quote,
    /// The futures mark: the contract's `current_price`, or the
    /// theoretical `S e^{(r - q - b) T}` when none was supplied.
    pub current_price: Quote,
    pub entry_price: f64,
    pub multiplier: f64,
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    /// Continuous stock borrow (repo) cost; part of the carry.
    pub borrow_cost: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub long_short: LongShort,
}

impl EquityFuture {
    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`EquityFuture::try_from_json`].
    pub fn from_json(data: &EquityFutureData) -> Box<Self> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build from contract data. `risk_free_rate` is required;
    /// `current_price` defaults to the theoretical futures price,
    /// `entry_price` to 0 (the position is then valued rather than
    /// marked against a trade), `multiplier` to 1.
    pub fn try_from_json(data: &EquityFutureData) -> Result<Box<Self>, RustyQLibError> {
        let today =
            crate::core::data_models::parse_valuation_date(data.base.valuation_date.as_deref())?;
        let maturity_date =
            NaiveDate::parse_from_str(&data.maturity, "%Y-%m-%d").map_err(|_| {
                RustyQLibError::invalid_input(
                    "maturity",
                    format!("invalid date '{}' (expected YYYY-MM-DD)", data.maturity),
                )
            })?;
        if maturity_date <= today {
            return Err(RustyQLibError::invalid_input(
                "maturity",
                format!("maturity {maturity_date} must be after the valuation date {today}"),
            ));
        }

        let spot = data.base.underlying_price;
        let multiplier = data.multiplier.unwrap_or(1.0);
        for (name, x) in [("underlying_price", spot), ("multiplier", multiplier)] {
            if !(x.is_finite() && x > 0.0) {
                return Err(RustyQLibError::invalid_input(
                    name,
                    format!("{name} must be finite and positive, got {x}"),
                ));
            }
        }
        let entry_price = data.entry_price.unwrap_or(0.0);
        if !(entry_price.is_finite() && entry_price >= 0.0) {
            return Err(RustyQLibError::invalid_input(
                "entry_price",
                format!("entry_price must be finite and non-negative, got {entry_price}"),
            ));
        }

        let risk_free_rate = data.base.risk_free_rate.ok_or_else(|| {
            RustyQLibError::invalid_input("risk_free_rate", "risk_free_rate is required")
        })?;
        let dividend = data.dividend.unwrap_or(0.0);
        let borrow_cost = data.base.borrow_cost.unwrap_or(0.0);
        for (name, x) in [
            ("risk_free_rate", risk_free_rate),
            ("dividend", dividend),
            ("borrow_cost", borrow_cost),
        ] {
            if !x.is_finite() {
                return Err(RustyQLibError::invalid_input(
                    name,
                    format!("{name} must be finite, got {x}"),
                ));
            }
            crate::equity::conventions::check_rate_band(name, x)?;
        }
        let position = LongShort::try_from(data.base.long_short.unwrap_or(1))?;

        let current_price = match data.current_price {
            Some(f) if f.is_finite() && f > 0.0 => f,
            Some(f) => {
                return Err(RustyQLibError::invalid_input(
                    "current_price",
                    format!("current_price must be finite and positive, got {f}"),
                ))
            }
            None => {
                let t = crate::equity::conventions::year_fraction(today, maturity_date);
                spot * ((risk_free_rate - dividend - borrow_cost) * t).exp()
            }
        };

        Ok(Box::new(Self {
            symbol: data.base.symbol.clone(),
            currency: data.base.currency.clone(),
            exchange: data.base.exchange.clone(),
            name: data.base.name.clone(),
            cusip: data.base.cusip.clone(),
            isin: data.base.isin.clone(),
            settlement_type: data.base.settlement_type.clone(),

            underlying_price: Quote::new(spot),
            current_price: Quote::new(current_price),
            entry_price,
            multiplier,
            risk_free_rate,
            dividend_yield: dividend,
            borrow_cost,
            maturity_date,
            valuation_date: today,
            long_short: position,
        }))
    }
    fn pnl(&self) -> f64 {
        let pnl = (self.current_price.value() - self.entry_price) * self.multiplier;
        match self.long_short {
            LongShort::LONG => pnl,
            LongShort::SHORT => -pnl,
        }
    }
}
impl Instrument for EquityFuture {
    fn try_npv(&self) -> Result<f64, RustyQLibError> {
        Ok(self.pnl())
    }

    fn price(&self) -> Result<crate::core::results::PricingResult, RustyQLibError> {
        Ok(crate::core::results::PricingResult {
            pv: self.try_npv()?,
            greeks: crate::core::results::Greeks {
                delta: self.delta(),
                ..Default::default()
            },
            std_err: None,
            asset_greeks: None,
        })
    }
}
impl EquityFuture {
    /// Value change per unit move in the futures mark: `sign *
    /// multiplier` (the P&L is linear in the mark).
    pub fn delta(&self) -> f64 {
        self.long_short.sign() * self.multiplier
    }
    pub fn gamma(&self) -> f64 {
        0.0
    }
    pub fn vega(&self) -> f64 {
        0.0
    }
    pub fn theta(&self) -> f64 {
        0.0
    }
    pub fn rho(&self) -> f64 {
        0.0
    }
    pub fn vanna(&self) -> f64 {
        0.0
    }
    pub fn charm(&self) -> f64 {
        0.0
    }
    pub fn gamma_p(&self) -> f64 {
        0.0
    }
    pub fn zomma(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(json: &str) -> EquityFutureData {
        serde_json::from_str(json).expect("future contract must parse")
    }

    const BASE: &str = r#"{
        "symbol": "ABC", "underlying_price": 96.0, "current_price": 99.0,
        "entry_price": 103.0, "risk_free_rate": 0.06, "dividend": 0.01,
        "multiplier": 1000.0, "long_short": -1,
        "valuation_date": "2026-01-05", "maturity": "2026-09-30"
    }"#;

    #[test]
    fn pnl_is_signed_and_scaled_by_the_multiplier() {
        let short = EquityFuture::try_from_json(&data(BASE)).unwrap();
        // short 1000 at 103, marked at 99: +4000
        assert!((short.npv() - 4000.0).abs() < 1e-9, "{}", short.npv());
        assert_eq!(short.delta(), -1000.0);
        assert_eq!(short.price().unwrap().greeks.delta, -1000.0);
        let long = EquityFuture::try_from_json(&data(
            &BASE.replace("\"long_short\": -1", "\"long_short\": 1"),
        ))
        .unwrap();
        assert!((long.npv() + 4000.0).abs() < 1e-9);
        assert_eq!(long.delta(), 1000.0);
    }

    #[test]
    fn missing_mark_defaults_to_the_theoretical_futures_price() {
        let json = BASE.replace("\"current_price\": 99.0,", "");
        assert!(!json.contains("current_price"), "the mark must be removed");
        let fut = EquityFuture::try_from_json(&data(&json)).unwrap();
        let t = (NaiveDate::from_ymd_opt(2026, 9, 30).unwrap()
            - NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
        .num_days() as f64
            / 365.0;
        let theoretical = 96.0 * ((0.06 - 0.01) * t).exp();
        assert!(
            (fut.current_price.value() - theoretical).abs() < 1e-12,
            "{} vs {theoretical}",
            fut.current_price.value()
        );
        // short at 103 against a mark below it: a gain, not -entry * mult
        assert!((fut.npv() - (103.0 - theoretical) * 1000.0).abs() < 1e-9);
    }

    #[test]
    fn required_fields_and_ranges_are_enforced() {
        let expect_field = |json: &str, field: &str| {
            let err = EquityFuture::try_from_json(&data(json))
                .err()
                .expect("must be rejected")
                .to_string();
            assert!(err.contains(field), "expected '{field}' in: {err}");
        };
        expect_field(
            &BASE.replace("\"risk_free_rate\": 0.06, ", ""),
            "risk_free_rate",
        );
        expect_field(
            &BASE.replace("\"risk_free_rate\": 0.06", "\"risk_free_rate\": 6.0"),
            "risk_free_rate",
        );
        expect_field(
            &BASE.replace("\"dividend\": 0.01", "\"dividend\": 1.0"),
            "dividend",
        );
        expect_field(
            &BASE.replace(
                "\"maturity\": \"2026-09-30\"",
                "\"maturity\": \"2026-01-05\"",
            ),
            "maturity",
        );
        expect_field(
            &BASE.replace("\"multiplier\": 1000.0", "\"multiplier\": 0.0"),
            "multiplier",
        );
        expect_field(
            &BASE.replace("\"current_price\": 99.0", "\"current_price\": -1.0"),
            "current_price",
        );
        expect_field(
            &BASE.replace("\"underlying_price\": 96.0", "\"underlying_price\": 0.0"),
            "underlying_price",
        );
        expect_field(
            &BASE.replace("\"long_short\": -1", "\"long_short\": 0"),
            "long_short",
        );
        expect_field(
            &BASE.replace("\"long_short\": -1", "\"long_short\": 2"),
            "long_short",
        );
    }
}
