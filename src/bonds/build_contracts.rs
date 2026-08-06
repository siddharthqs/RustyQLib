//! Build curve instruments from JSON contract data.
//!
//! Translates the `rate_data` (deposits, FRAs) and `bond_data` (bills,
//! bonds) blocks of an `"IR"` contract document into curve instruments
//! and bootstraps them into a [`YieldCurve`]. Dates in the document may
//! be ISO dates (`"2026-07-15"`) or tenors relative to the valuation
//! date (`"3M"`, `"2W"`, `"1Y"`); tenor-derived dates roll forward off
//! weekends.

use chrono::NaiveDate;

use crate::bonds::schedule::{coupon_dates, is_end_of_month};
use crate::bonds::{
    bootstrap_curve, BillQuote, BondQuote, CurveInstrument, Deposit, FixedRateBond, Fra, Frequency,
    TreasuryBill,
};
use crate::core::calendar::{BusinessDayConvention, Calendar, Period};
use crate::core::curves::YieldCurve;
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::core::utils::{BondData, Contract};

/// Parse a day-count string from contract data (`"A360"`, `"Act/365"`,
/// `"30/360"`, `"Act/Act ICMA"`, ...).
pub fn parse_day_count(s: &str) -> Result<DayCountConvention, RustyQLibError> {
    match s {
        "Act360" | "A360" | "Act/360" | "ACT/360" | "act360" => Ok(DayCountConvention::Act360),
        "Act365" | "A365" | "Act/365" | "ACT/365" | "act365" => Ok(DayCountConvention::Act365),
        "Thirty360" | "30/360" | "thirty360" => Ok(DayCountConvention::Thirty360),
        "ActActIcma" | "ActActICMA" | "Act/Act ICMA" | "act_act_icma" => {
            Ok(DayCountConvention::ActActIcma)
        }
        "ActActIsda" | "ActActISDA" | "Act/Act ISDA" | "Act/Act" | "ACT/ACT" | "act_act" => {
            Ok(DayCountConvention::ActActIsda)
        }
        other => Err(RustyQLibError::invalid_input(
            "day_count",
            format!("unrecognized day count `{other}`"),
        )),
    }
}

/// Parse a coupon-frequency string (`"semiannual"`, `"annual"`, ...).
pub fn parse_frequency(s: &str) -> Result<Frequency, RustyQLibError> {
    match s.to_ascii_lowercase().as_str() {
        "semiannual" | "semi-annual" | "semi_annual" => Ok(Frequency::Semiannual),
        "annual" => Ok(Frequency::Annual),
        "quarterly" => Ok(Frequency::Quarterly),
        "monthly" => Ok(Frequency::Monthly),
        other => Err(RustyQLibError::invalid_input(
            "frequency",
            format!("unrecognized frequency `{other}`"),
        )),
    }
}

/// Resolve a date field: an ISO date (`YYYY-MM-DD`) is taken as-is; a
/// tenor (`<n>D`, `<n>W`, `<n>M`, `<n>Y`, case-insensitive) is advanced
/// from `base` and adjusted to the next business day (weekends-only
/// calendar). `"0M"` therefore means `base` itself, weekend-rolled.
pub fn resolve_date(s: &str, base: NaiveDate) -> Result<NaiveDate, RustyQLibError> {
    if let Ok(date) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(date);
    }
    let s = s.trim();
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
    let n: i64 = num.parse().map_err(|_| {
        RustyQLibError::invalid_input(
            "date",
            format!("`{s}` is neither an ISO date (YYYY-MM-DD) nor a tenor like `3M`"),
        )
    })?;
    let period = match unit.to_ascii_uppercase().as_str() {
        "D" => Period::Days(n),
        "W" => Period::Weeks(n),
        "M" => Period::Months(n as i32),
        "Y" => Period::Years(n as i32),
        other => {
            return Err(RustyQLibError::invalid_input(
                "date",
                format!("unknown tenor unit `{other}` in `{s}` (expected D, W, M or Y)"),
            ))
        }
    };
    let calendar = Calendar::WeekendsOnly;
    Ok(calendar.advance(base, period, BusinessDayConvention::Following))
}

/// A [`FixedRateBond`] built from a `bond_data` block together with the
/// trade context resolved from the document.
pub struct BuiltBond {
    pub bond: FixedRateBond,
    pub valuation_date: NaiveDate,
    pub settlement: NaiveDate,
}

/// Trade context (valuation date, T+n settlement on the SIFMA calendar)
/// shared by bonds and bills.
fn trade_context(
    bond_data: &BondData,
    today: NaiveDate,
) -> Result<(NaiveDate, NaiveDate), RustyQLibError> {
    let valuation_date = match &bond_data.valuation_date {
        Some(s) => resolve_date(s, today)?,
        None => today,
    };
    let settlement_days = bond_data.settlement_days.unwrap_or(1);
    if settlement_days < 0 {
        return Err(RustyQLibError::invalid_input(
            "settlement_days",
            format!("must be non-negative, got {settlement_days}"),
        ));
    }
    let settlement = Calendar::UsGovernmentBond.add_business_days(valuation_date, settlement_days);
    Ok((valuation_date, settlement))
}

/// Build a [`FixedRateBond`] from a `bond_data` block, applying US
/// Treasury defaults for everything not specified. A missing dated date
/// defaults to the coupon anchor at or before settlement, so a seasoned
/// bond can be described by coupon and maturity alone.
pub fn build_bond(bond_data: &BondData, today: NaiveDate) -> Result<BuiltBond, RustyQLibError> {
    let (valuation_date, settlement) = trade_context(bond_data, today)?;
    let maturity_date = resolve_date(&bond_data.maturity_date, valuation_date)?;
    let coupon_rate = bond_data.coupon_rate.ok_or_else(|| {
        RustyQLibError::invalid_input("coupon_rate", "a Bond needs `coupon_rate`")
    })?;
    let frequency = match &bond_data.frequency {
        Some(s) => parse_frequency(s)?,
        None => Frequency::Semiannual,
    };
    let day_count = match &bond_data.day_count {
        Some(s) => parse_day_count(s)?,
        None => DayCountConvention::ActActIcma,
    };
    let end_of_month = is_end_of_month(maturity_date);
    let dated_date = match &bond_data.dated_date {
        Some(s) => resolve_date(s, valuation_date)?,
        None => {
            // the coupon anchor at or before settlement
            coupon_dates(settlement, maturity_date, frequency.months(), end_of_month)?.prev_anchor
        }
    };
    let bond = FixedRateBond::new(
        bond_data.face_value.unwrap_or(100.0),
        coupon_rate,
        frequency,
        dated_date,
        maturity_date,
        day_count,
        Calendar::UsGovernmentBond,
        BusinessDayConvention::Following,
        bond_data.settlement_days.unwrap_or(1),
        end_of_month,
    )?;
    Ok(BuiltBond {
        bond,
        valuation_date,
        settlement,
    })
}

/// A [`TreasuryBill`] built from a `bond_data` block with its trade
/// context.
pub struct BuiltBill {
    pub bill: TreasuryBill,
    pub valuation_date: NaiveDate,
    pub settlement: NaiveDate,
}

/// Build a [`TreasuryBill`] from a `bond_data` block.
pub fn build_bill(bond_data: &BondData, today: NaiveDate) -> Result<BuiltBill, RustyQLibError> {
    let (valuation_date, settlement) = trade_context(bond_data, today)?;
    let maturity_date = resolve_date(&bond_data.maturity_date, valuation_date)?;
    let bill = TreasuryBill::new(bond_data.face_value.unwrap_or(100.0), maturity_date)?;
    Ok(BuiltBill {
        bill,
        valuation_date,
        settlement,
    })
}

/// Build a curve pillar from a `bond_data` block: a [`BillQuote`] from a
/// discount rate (or price), or a [`BondQuote`] from a clean price (or
/// yield).
fn build_bond_curve_instrument(
    bond_data: &BondData,
    today: NaiveDate,
) -> Result<(NaiveDate, Box<dyn CurveInstrument>), RustyQLibError> {
    match bond_data.instrument.as_str() {
        "Bill" => {
            let built = build_bill(bond_data, today)?;
            let discount_rate = match (bond_data.discount_rate, bond_data.clean_price) {
                (Some(rate), _) => rate,
                (None, Some(price)) => built
                    .bill
                    .discount_rate_from_price(price, built.settlement)?,
                (None, None) => {
                    return Err(RustyQLibError::invalid_input(
                        "bond_data",
                        "a Bill curve quote needs `discount_rate` or `clean_price`",
                    ))
                }
            };
            let quote = BillQuote::new(built.bill, discount_rate, built.settlement)?;
            Ok((built.valuation_date, Box::new(quote)))
        }
        "Bond" => {
            let built = build_bond(bond_data, today)?;
            let clean_price = match (bond_data.clean_price, bond_data.yield_rate) {
                (Some(price), _) => price,
                (None, Some(y)) => built.bond.clean_price_from_yield(y, built.settlement)?,
                (None, None) => {
                    return Err(RustyQLibError::invalid_input(
                        "bond_data",
                        "a Bond curve quote needs `clean_price` or `yield_rate`",
                    ))
                }
            };
            let quote = BondQuote::new(built.bond, clean_price, built.settlement)?;
            Ok((built.valuation_date, Box::new(quote)))
        }
        other => Err(RustyQLibError::invalid_input(
            "instrument",
            format!("unsupported bond instrument `{other}` (expected Bond or Bill)"),
        )),
    }
}

/// Build one curve instrument from an `"IR"` contract. `today` anchors
/// tenor-style dates: the valuation date is resolved relative to it, and
/// start/maturity relative to the valuation date.
pub fn build_curve_instrument(
    contract: &Contract,
    today: NaiveDate,
) -> Result<(NaiveDate, Box<dyn CurveInstrument>), RustyQLibError> {
    if let Some(bond_data) = &contract.bond_data {
        return build_bond_curve_instrument(bond_data, today);
    }
    let rate_data = contract.rate_data.as_ref().ok_or_else(|| {
        RustyQLibError::invalid_input(
            "rate_data",
            "IR contract is missing `rate_data` (or `bond_data`)",
        )
    })?;
    let valuation_date = resolve_date(&rate_data.valuation_date, today)?;
    let start_date = resolve_date(&rate_data.start_date, valuation_date)?;
    let maturity_date = resolve_date(&rate_data.maturity_date, valuation_date)?;
    let day_count = parse_day_count(&rate_data.day_count)?;

    let instrument: Box<dyn CurveInstrument> = match rate_data.instrument.as_str() {
        "Deposit" => Box::new(Deposit::new(
            start_date,
            maturity_date,
            rate_data.notional,
            rate_data.fix_rate,
            day_count,
        )?),
        "FRA" => Box::new(Fra::new(
            start_date,
            maturity_date,
            rate_data.notional,
            rate_data.fix_rate,
            day_count,
        )?),
        other => {
            return Err(RustyQLibError::invalid_input(
                "instrument",
                format!("unsupported IR instrument `{other}` (expected Deposit or FRA)"),
            ))
        }
    };
    Ok((valuation_date, instrument))
}

/// Build every instrument in an `"IR"` document and bootstrap them into a
/// discount curve. All contracts must share one valuation date; the curve
/// converts pillar dates to times with Act/365.
pub fn bootstrap_from_contracts(
    contracts: &[Contract],
    today: NaiveDate,
) -> Result<YieldCurve, RustyQLibError> {
    let mut instruments: Vec<Box<dyn CurveInstrument>> = Vec::with_capacity(contracts.len());
    let mut reference: Option<NaiveDate> = None;
    for contract in contracts {
        let (valuation_date, instrument) = build_curve_instrument(contract, today)?;
        match reference {
            None => reference = Some(valuation_date),
            Some(existing) if existing != valuation_date => {
                return Err(RustyQLibError::invalid_input(
                    "valuation_date",
                    format!(
                        "contracts disagree on the valuation date ({existing} vs {valuation_date})"
                    ),
                ));
            }
            Some(_) => {}
        }
        instruments.push(instrument);
    }
    let reference = reference.ok_or_else(|| {
        RustyQLibError::invalid_input("contracts", "no contracts to build a curve from")
    })?;
    bootstrap_curve(&instruments, reference, DayCountConvention::Act365)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::utils::RateData;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn resolve_date_handles_iso_and_tenors() {
        // Thursday base date
        let base = d(2026, 1, 15);
        assert_eq!(resolve_date("2026-07-15", base).unwrap(), d(2026, 7, 15));
        // 0M = the base date itself (a business day)
        assert_eq!(resolve_date("0M", base).unwrap(), base);
        // 1M lands on Sunday Feb 15 -> rolls to Monday Feb 16
        assert_eq!(resolve_date("1M", base).unwrap(), d(2026, 2, 16));
        assert_eq!(resolve_date("3M", base).unwrap(), d(2026, 4, 15));
        assert_eq!(resolve_date("1Y", base).unwrap(), d(2027, 1, 15));
        assert_eq!(resolve_date("2W", base).unwrap(), d(2026, 1, 29));
        assert!(resolve_date("garbage", base).is_err());
        assert!(resolve_date("3Q", base).is_err());
    }

    #[test]
    fn parse_day_count_aliases() {
        assert_eq!(parse_day_count("A360").unwrap(), DayCountConvention::Act360);
        assert_eq!(
            parse_day_count("Act365").unwrap(),
            DayCountConvention::Act365
        );
        assert_eq!(
            parse_day_count("30/360").unwrap(),
            DayCountConvention::Thirty360
        );
        assert!(parse_day_count("bogus").is_err());
    }

    fn ir_contract(instrument: &str, start: &str, maturity: &str, rate: f64) -> Contract {
        // build through JSON so the test also covers the wire schema
        let json = format!(
            r#"{{
                "action": "PV",
                "asset": "IR",
                "rate_data": {{
                    "instrument": "{instrument}",
                    "currency": "USD",
                    "start_date": "{start}",
                    "maturity_date": "{maturity}",
                    "valuation_date": "0M",
                    "notional": 1000000,
                    "fix_rate": {rate},
                    "day_count": "A360",
                    "business_day_adjustment": 0
                }}
            }}"#
        );
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn bootstraps_the_sample_document_shape() {
        let today = d(2026, 1, 15); // Thursday
        let contracts = vec![
            ir_contract("Deposit", "0M", "1M", 0.055),
            ir_contract("Deposit", "0M", "3M", 0.05),
            ir_contract("FRA", "3M", "6M", 0.06),
            ir_contract("FRA", "6M", "9M", 0.065),
        ];
        let curve = bootstrap_from_contracts(&contracts, today).unwrap();
        assert_eq!(curve.reference_date(), today);
        assert_eq!(curve.pillars().len(), 4);
        // 3M deposit pillar (Jan 15 -> Apr 15 = 90 days at 5% Act/360)
        let df_3m = 1.0 / (1.0 + 0.05 * 90.0 / 360.0);
        assert!((curve.df_date(d(2026, 4, 15)) - df_3m).abs() < 1e-12);
        // discount factors decrease along the curve
        let dfs: Vec<f64> = curve.pillars().iter().map(|p| p.df).collect();
        assert!(dfs.windows(2).all(|w| w[1] < w[0]));
    }

    #[test]
    fn missing_rate_data_is_an_error() {
        let mut contract = ir_contract("Deposit", "0M", "3M", 0.05);
        contract.rate_data = None;
        assert!(build_curve_instrument(&contract, d(2026, 1, 15)).is_err());
    }

    #[test]
    fn unknown_instrument_is_an_error() {
        let contract = ir_contract("Swaption", "0M", "3M", 0.05);
        assert!(build_curve_instrument(&contract, d(2026, 1, 15)).is_err());
    }

    #[test]
    fn rate_data_deserializes_standalone() {
        let rd: RateData = serde_json::from_str(
            r#"{
                "instrument": "Deposit", "currency": "USD",
                "start_date": "0M", "maturity_date": "6M",
                "valuation_date": "0M", "notional": 1000000,
                "fix_rate": 0.05, "day_count": "A360",
                "business_day_adjustment": 0
            }"#,
        )
        .unwrap();
        assert_eq!(rd.instrument, "Deposit");
    }

    fn bond_contract(bond_data: serde_json::Value) -> Contract {
        serde_json::from_value(serde_json::json!({
            "action": "PV",
            "asset": "IR",
            "bond_data": bond_data,
        }))
        .unwrap()
    }

    #[test]
    fn bootstraps_a_treasury_curve_from_bond_data_quotes() {
        let today = d(2026, 8, 5); // Wednesday; T+1 settles Thursday Aug 6
        let contracts = vec![
            bond_contract(serde_json::json!({
                "instrument": "Bill",
                "maturity_date": "2026-11-05",
                "valuation_date": "2026-08-05",
                "discount_rate": 0.048,
            })),
            bond_contract(serde_json::json!({
                "instrument": "Bill",
                "maturity_date": "2027-02-04",
                "valuation_date": "2026-08-05",
                "discount_rate": 0.047,
            })),
            bond_contract(serde_json::json!({
                "instrument": "Bond",
                "coupon_rate": 0.045,
                "dated_date": "2026-05-15",
                "maturity_date": "2028-05-15",
                "valuation_date": "2026-08-05",
                "clean_price": 99.50,
            })),
            // quoted by yield: the builder converts it to a clean price
            bond_contract(serde_json::json!({
                "instrument": "Bond",
                "coupon_rate": 0.0425,
                "dated_date": "2026-05-15",
                "maturity_date": "2031-05-15",
                "valuation_date": "2026-08-05",
                "yield_rate": 0.042,
            })),
        ];
        let curve = bootstrap_from_contracts(&contracts, today).unwrap();
        assert_eq!(curve.reference_date(), today);
        assert_eq!(curve.pillars().len(), 4);
        let dfs: Vec<f64> = curve.pillars().iter().map(|p| p.df).collect();
        assert!(dfs.windows(2).all(|w| w[1] < w[0]), "{dfs:?}");

        // the curve must reprice the 2y note at its quoted clean price
        let bond =
            FixedRateBond::us_treasury(100.0, 0.045, d(2026, 5, 15), d(2028, 5, 15)).unwrap();
        let clean = bond.clean_price_from_curve(&curve, d(2026, 8, 6)).unwrap();
        assert!((clean - 99.50).abs() < 1e-7, "repriced clean {clean}");
    }
}
