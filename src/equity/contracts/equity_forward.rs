use crate::core::data_models::EquityForwardData;
use crate::core::errors::RustyQLibError;
use crate::core::quotes::Quote;
use crate::core::traits::Instrument;
use crate::equity::utils::LongShort;
use chrono::NaiveDate;
///A forward contract is an agreement between two parties to buy or sell, as the case may be,
/// a commodity (or financial instrument or currency or any other underlying)
/// on a pre-determined future date at a price agreed when the contract is entered into.
///
/// Sized by a **currency notional**: the position is `notional / K`
/// shares, where `K` is the locked-in forward price (`forward_price`).
/// The value and every Greek are quoted for that share count, signed by
/// `long_short`.
pub struct EquityForward {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub settlement_type: Option<String>,

    pub underlying_price: Quote,
    pub forward_price: Quote, //Forward price you actually locked in.
    pub risk_free_rate: f64,
    pub dividend_yield: f64,
    /// Continuous stock borrow (repo) cost; part of the carry.
    pub borrow_cost: f64,
    pub maturity_date: NaiveDate,
    pub valuation_date: NaiveDate,
    pub long_short: LongShort,
    /// Currency amount of the position; the share count is
    /// `notional / forward_price`.
    pub notional: f64,
}
impl EquityForward {
    /// Build from contract data, panicking on any invalid field. Fallible
    /// callers should use [`EquityForward::try_from_json`].
    pub fn from_json(data: &EquityForwardData) -> Box<Self> {
        Self::try_from_json(data).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build from contract data. `entry_price` (the locked-in forward
    /// price) and `risk_free_rate` are required; `notional` is a currency
    /// amount defaulting to 1.0; `dividend`, `borrow_cost` default to 0.
    pub fn try_from_json(data: &EquityForwardData) -> Result<Box<Self>, RustyQLibError> {
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

        let entry_price = data.entry_price.ok_or_else(|| {
            RustyQLibError::invalid_input(
                "entry_price",
                "entry_price (the locked-in forward price) is required for forwards",
            )
        })?;
        let notional = data.notional.unwrap_or(1.0);
        for (name, x) in [
            ("underlying_price", data.base.underlying_price),
            ("entry_price", entry_price),
            ("notional", notional),
        ] {
            if !(x.is_finite() && x > 0.0) {
                return Err(RustyQLibError::invalid_input(
                    name,
                    format!("{name} must be finite and positive, got {x}"),
                ));
            }
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

        Ok(Box::new(Self {
            symbol: data.base.symbol.clone(),
            currency: data.base.currency.clone(),
            exchange: data.base.exchange.clone(),
            name: data.base.name.clone(),
            cusip: data.base.cusip.clone(),
            isin: data.base.isin.clone(),
            settlement_type: data.base.settlement_type.clone(),

            underlying_price: Quote::new(data.base.underlying_price),
            forward_price: Quote::new(entry_price),
            risk_free_rate,
            dividend_yield: dividend,
            borrow_cost,
            maturity_date,
            valuation_date: today,
            notional,
            long_short: position,
        }))
    }

    fn time_to_maturity(&self) -> f64 {
        crate::equity::conventions::year_fraction(self.valuation_date, self.maturity_date)
    }
    fn forward(&self) -> f64 {
        let discount_df = 1.0 / (self.risk_free_rate * self.time_to_maturity()).exp();
        let dividend_df =
            1.0 / ((self.dividend_yield + self.borrow_cost) * self.time_to_maturity()).exp();

        self.underlying_price.value() * dividend_df / discount_df
    }
    /// Number of shares the notional buys at the locked-in price.
    fn shares(&self) -> f64 {
        self.notional / self.forward_price.value()
    }
}
impl Instrument for EquityForward {
    fn try_npv(&self) -> Result<f64, crate::core::errors::RustyQLibError> {
        // e −r(T−t) (Ft −K),
        let df_r = 1.0 / (self.risk_free_rate * self.time_to_maturity()).exp();
        let share = self.shares();
        Ok(match self.long_short {
            LongShort::LONG => (self.forward() - self.forward_price.value()) * share * df_r,
            LongShort::SHORT => -(self.forward() - self.forward_price.value()) * share * df_r,
        })
    }

    fn price(
        &self,
    ) -> Result<crate::core::results::PricingResult, crate::core::errors::RustyQLibError> {
        Ok(crate::core::results::PricingResult {
            pv: self.try_npv()?,
            greeks: crate::core::results::Greeks {
                delta: self.delta(),
                theta: self.theta(),
                rho: self.rho(),
                ..Default::default()
            },
            std_err: None,
            asset_greeks: None,
        })
    }
}

/// Closed-form sensitivities of `V = sign * shares * (S e^{-(q+b)T} -
/// K e^{-rT})`. The value is linear in `S` and has no volatility input,
/// so gamma, vega, vanna, charm, gamma_p and zomma are identically zero.
impl EquityForward {
    /// `dV/dS = sign * shares * e^{-(q+b)T}`.
    pub fn delta(&self) -> f64 {
        let t = self.time_to_maturity();
        self.long_short.sign()
            * self.shares()
            * (-(self.dividend_yield + self.borrow_cost) * t).exp()
    }
    pub fn gamma(&self) -> f64 {
        0.0
    }
    pub fn vega(&self) -> f64 {
        0.0
    }
    /// Calendar theta `-dV/dT = sign * shares * ((q+b) S e^{-(q+b)T} -
    /// r K e^{-rT})`: the carry the position earns (or pays) per year.
    pub fn theta(&self) -> f64 {
        let t = self.time_to_maturity();
        let carry = self.dividend_yield + self.borrow_cost;
        let s = self.underlying_price.value();
        let k = self.forward_price.value();
        self.long_short.sign()
            * self.shares()
            * (carry * s * (-carry * t).exp()
                - self.risk_free_rate * k * (-self.risk_free_rate * t).exp())
    }
    /// `dV/dr = sign * shares * T K e^{-rT}`.
    pub fn rho(&self) -> f64 {
        let t = self.time_to_maturity();
        let k = self.forward_price.value();
        self.long_short.sign() * self.shares() * t * k * (-self.risk_free_rate * t).exp()
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

    fn data(json: &str) -> EquityForwardData {
        serde_json::from_str(json).expect("forward contract must parse")
    }

    const BASE: &str = r#"{
        "symbol": "ABC", "underlying_price": 96.0, "entry_price": 103.0,
        "risk_free_rate": 0.06, "dividend": 0.01, "borrow_cost": 0.005,
        "notional": 10000.0, "long_short": 1,
        "valuation_date": "2026-01-05", "maturity": "2026-09-30"
    }"#;

    fn forward() -> Box<EquityForward> {
        EquityForward::try_from_json(&data(BASE)).expect("valid forward must build")
    }

    #[test]
    fn required_fields_and_ranges_are_enforced() {
        let expect_field = |json: &str, field: &str| {
            let err = EquityForward::try_from_json(&data(json))
                .err()
                .expect("must be rejected")
                .to_string();
            assert!(err.contains(field), "expected '{field}' in: {err}");
        };
        // entry_price is required: without it the share count would be
        // notional / 0
        expect_field(
            r#"{"symbol": "ABC", "underlying_price": 96.0, "risk_free_rate": 0.06,
                "valuation_date": "2026-01-05", "maturity": "2026-09-30"}"#,
            "entry_price",
        );
        // and the rate is required rather than silently zero
        expect_field(
            r#"{"symbol": "ABC", "underlying_price": 96.0, "entry_price": 103.0,
                "valuation_date": "2026-01-05", "maturity": "2026-09-30"}"#,
            "risk_free_rate",
        );
        expect_field(
            &BASE.replace("\"entry_price\": 103.0", "\"entry_price\": 0.0"),
            "entry_price",
        );
        expect_field(
            &BASE.replace("\"notional\": 10000.0", "\"notional\": -5.0"),
            "notional",
        );
        expect_field(
            &BASE.replace("\"underlying_price\": 96.0", "\"underlying_price\": 0.0"),
            "underlying_price",
        );
        // a percent-vs-decimal slip on any rate-like input is caught
        expect_field(
            &BASE.replace("\"risk_free_rate\": 0.06", "\"risk_free_rate\": 6.0"),
            "risk_free_rate",
        );
        expect_field(
            &BASE.replace("\"dividend\": 0.01", "\"dividend\": 1.0"),
            "dividend",
        );
        expect_field(
            &BASE.replace("\"borrow_cost\": 0.005", "\"borrow_cost\": 0.75"),
            "borrow_cost",
        );
        // expired or same-day maturity is refused
        expect_field(
            &BASE.replace(
                "\"maturity\": \"2026-09-30\"",
                "\"maturity\": \"2026-01-05\"",
            ),
            "maturity",
        );
        expect_field(
            &BASE.replace(
                "\"maturity\": \"2026-09-30\"",
                "\"maturity\": \"2025-01-05\"",
            ),
            "maturity",
        );
        // long_short outside {1, -1} is rejected, not read as long
        expect_field(
            &BASE.replace("\"long_short\": 1", "\"long_short\": 0"),
            "long_short",
        );
        expect_field(
            &BASE.replace("\"long_short\": 1", "\"long_short\": 2"),
            "long_short",
        );
        let short = EquityForward::try_from_json(&data(
            &BASE.replace("\"long_short\": 1", "\"long_short\": -1"),
        ))
        .unwrap();
        assert_eq!(short.long_short, LongShort::SHORT);
    }

    #[test]
    fn value_is_the_discounted_forward_minus_strike_on_the_share_count() {
        let fwd = forward();
        let t = (NaiveDate::from_ymd_opt(2026, 9, 30).unwrap()
            - NaiveDate::from_ymd_opt(2026, 1, 5).unwrap())
        .num_days() as f64
            / 365.0;
        let shares = 10000.0 / 103.0;
        let expect = shares * (96.0 * (-0.015 * t).exp() - 103.0 * (-0.06 * t).exp());
        assert!(
            (fwd.npv() - expect).abs() < 1e-9,
            "{} vs {expect}",
            fwd.npv()
        );
        // the short is the mirror
        let short = EquityForward::try_from_json(&data(
            &BASE.replace("\"long_short\": 1", "\"long_short\": -1"),
        ))
        .unwrap();
        assert!((short.npv() + fwd.npv()).abs() < 1e-12);
    }

    #[test]
    fn greeks_match_central_differences_of_the_value() {
        for long_short in [1, -1] {
            let json = BASE.replace(
                "\"long_short\": 1",
                &format!("\"long_short\": {long_short}"),
            );
            let fwd = EquityForward::try_from_json(&data(&json)).unwrap();
            let result = fwd.price().unwrap();
            assert_eq!(result.pv, fwd.npv());

            // delta: bump the spot
            let ds = 1e-3;
            let mut up = data(&json);
            up.base.underlying_price += ds;
            let mut dn = data(&json);
            dn.base.underlying_price -= ds;
            let fd_delta = (EquityForward::try_from_json(&up).unwrap().npv()
                - EquityForward::try_from_json(&dn).unwrap().npv())
                / (2.0 * ds);
            assert!(
                (fwd.delta() - fd_delta).abs() < 1e-6,
                "delta {} vs fd {fd_delta}",
                fwd.delta()
            );
            assert_eq!(result.greeks.delta, fwd.delta());
            assert!(fwd.delta() * long_short as f64 > 0.0);

            // rho: bump the rate
            let dr = 1e-5;
            let mut up = data(&json);
            up.base.risk_free_rate = Some(0.06 + dr);
            let mut dn = data(&json);
            dn.base.risk_free_rate = Some(0.06 - dr);
            let fd_rho = (EquityForward::try_from_json(&up).unwrap().npv()
                - EquityForward::try_from_json(&dn).unwrap().npv())
                / (2.0 * dr);
            assert!(
                (fwd.rho() - fd_rho).abs() < 1e-4,
                "rho {} vs fd {fd_rho}",
                fwd.rho()
            );
            assert_eq!(result.greeks.rho, fwd.rho());

            // theta = -dV/dT: move the maturity one day either side
            let h = 1.0 / 365.0;
            let mut later = data(&json);
            later.maturity = "2026-10-01".to_string();
            let mut earlier = data(&json);
            earlier.maturity = "2026-09-29".to_string();
            let fd_theta = -(EquityForward::try_from_json(&later).unwrap().npv()
                - EquityForward::try_from_json(&earlier).unwrap().npv())
                / (2.0 * h);
            assert!(
                (fwd.theta() - fd_theta).abs() < 1e-3,
                "theta {} vs fd {fd_theta}",
                fwd.theta()
            );
            assert_eq!(result.greeks.theta, fwd.theta());

            // no convexity or vol exposure
            assert_eq!(fwd.gamma(), 0.0);
            assert_eq!(fwd.vega(), 0.0);
            assert_eq!(result.greeks.gamma, 0.0);
            assert_eq!(result.greeks.vega, 0.0);
        }
    }
}
