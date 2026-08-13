//! JSON contract pricing for fixed income (`"PV"` / `"IR"`).
//!
//! Prices the `bond_data` block of an IR contract and renders the full
//! analytics set as a `{contract, output}` JSON value, mirroring the
//! equity contract service. Bonds price from exactly one of
//! `clean_price`, `yield_rate` or `curve`; bills from `discount_rate` or
//! `clean_price`.

use chrono::NaiveDate;
use serde_json::{json, Value};

use crate::bonds::build_contracts::{build_bill, build_bond};
use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;
use crate::core::utils::{BondData, Contract};

/// Price one IR contract. Errors bubble up for the caller to render in
/// its batch error shape.
pub fn price_ir_contract(data: &Contract, today: NaiveDate) -> Result<Value, RustyQLibError> {
    let Some(bond_data) = &data.bond_data else {
        return Err(RustyQLibError::UnsupportedEngine(
            "standalone PV for IR contracts needs a `bond_data` block (Bond or Bill); \
             deposits and FRAs feed curve construction via the `build` command"
                .to_string(),
        ));
    };
    let output = match bond_data.instrument.as_str() {
        "Bond" | "Corporate" => price_bond(bond_data, today)?,
        "Bill" => price_bill(bond_data, today)?,
        other => {
            return Err(RustyQLibError::invalid_input(
                "instrument",
                format!(
                    "unsupported IR instrument `{other}` for pricing \
                     (expected Bond, Corporate or Bill)"
                ),
            ))
        }
    };
    Ok(json!({ "contract": data, "output": output }))
}

fn price_bond(bond_data: &BondData, today: NaiveDate) -> Result<Value, RustyQLibError> {
    let built = build_bond(bond_data, today)?;
    let bond = &built.bond;
    let settlement = built.settlement;

    // exactly one market quote — a curve alone prices the bond, a curve
    // alongside a quote serves as the benchmark for spread analytics
    if bond_data.clean_price.is_some() && bond_data.yield_rate.is_some() {
        return Err(RustyQLibError::invalid_input(
            "bond_data",
            "provide `clean_price` or `yield_rate`, not both",
        ));
    }
    if bond_data.clean_price.is_none()
        && bond_data.yield_rate.is_none()
        && bond_data.curve.is_none()
    {
        return Err(RustyQLibError::invalid_input(
            "bond_data",
            "provide one of `clean_price`, `yield_rate` or `curve`",
        ));
    }
    let benchmark = bond_data
        .curve
        .as_ref()
        .map(|input| YieldCurve::from_input(input, built.valuation_date))
        .transpose()?;

    let (clean_price, yield_rate) = if let Some(clean) = bond_data.clean_price {
        (clean, bond.yield_from_clean_price(clean, settlement)?)
    } else if let Some(y) = bond_data.yield_rate {
        (bond.clean_price_from_yield(y, settlement)?, y)
    } else {
        let curve = benchmark.as_ref().expect("curve is the only quote");
        let clean = bond.clean_price_from_curve(curve, settlement)?;
        (clean, bond.yield_from_clean_price(clean, settlement)?)
    };

    let accrued = bond.accrued_interest(settlement)?;
    let dirty = clean_price + accrued;
    let mut output = json!({
        "instrument": bond_data.instrument,
        "valuation_date": built.valuation_date.to_string(),
        "settlement_date": settlement.to_string(),
        "clean_price": clean_price,
        "dirty_price": dirty,
        "accrued_interest": accrued,
        "yield": yield_rate,
        "macaulay_duration": bond.macaulay_duration(yield_rate, settlement)?,
        "modified_duration": bond.modified_duration(yield_rate, settlement)?,
        "convexity": bond.convexity(yield_rate, settlement)?,
        "dv01": bond.dv01(yield_rate, settlement)?,
        "pv": dirty * bond.face_value / 100.0,
    });

    // a benchmark curve next to a market quote adds spread analytics
    if let Some(curve) = &benchmark {
        if bond_data.clean_price.is_some() || bond_data.yield_rate.is_some() {
            let z = bond.z_spread(clean_price, curve, settlement)?;
            output["z_spread"] = json!(z);
            output["spread_dv01"] = json!(bond.spread_dv01(curve, z, settlement)?);
        }
    }
    Ok(output)
}

fn price_bill(bond_data: &BondData, today: NaiveDate) -> Result<Value, RustyQLibError> {
    let built = build_bill(bond_data, today)?;
    let bill = &built.bill;
    let settlement = built.settlement;

    let discount_rate = match (bond_data.discount_rate, bond_data.clean_price) {
        (Some(rate), None) => rate,
        (None, Some(price)) => bill.discount_rate_from_price(price, settlement)?,
        _ => {
            return Err(RustyQLibError::invalid_input(
                "bond_data",
                "provide exactly one of `discount_rate` or `clean_price` for a Bill",
            ))
        }
    };
    let price = bill.price_from_discount_rate(discount_rate, settlement)?;
    Ok(json!({
        "instrument": "Bill",
        "valuation_date": built.valuation_date.to_string(),
        "settlement_date": settlement.to_string(),
        "price": price,
        "discount_rate": discount_rate,
        "bond_equivalent_yield": bill.bond_equivalent_yield(discount_rate, settlement)?,
        "days_to_maturity": (bill.maturity_date - settlement).num_days(),
        "pv": price * bill.face_value / 100.0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn contract(bond_data: Value) -> Contract {
        serde_json::from_value(json!({
            "action": "PV",
            "asset": "IR",
            "bond_data": bond_data,
        }))
        .unwrap()
    }

    #[test]
    fn bond_prices_from_a_clean_price() {
        let data = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "dated_date": "2026-05-15",
            "maturity_date": "2028-05-15",
            "valuation_date": "2026-08-05",
            "clean_price": 99.50,
        }));
        let result = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        let output = &result["output"];
        // T+1 from Wednesday Aug 5
        assert_eq!(output["settlement_date"], "2026-08-06");
        assert_eq!(output["clean_price"].as_f64().unwrap(), 99.50);
        // slightly below par -> yield a touch above the 4.5% coupon
        let y = output["yield"].as_f64().unwrap();
        assert!(y > 0.045 && y < 0.055, "yield {y}");
        // dirty = clean + accrued, both positive mid-period
        let accrued = output["accrued_interest"].as_f64().unwrap();
        assert!(accrued > 0.0);
        assert!((output["dirty_price"].as_f64().unwrap() - 99.50 - accrued).abs() < 1e-12);
        assert!(output["modified_duration"].as_f64().unwrap() > 1.0);
        assert!(output["dv01"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn bond_prices_from_yield_and_round_trips() {
        let quote = |field: &str, value: f64| {
            contract(json!({
                "instrument": "Bond",
                "coupon_rate": 0.045,
                "dated_date": "2026-05-15",
                "maturity_date": "2028-05-15",
                "valuation_date": "2026-08-05",
                field: value,
            }))
        };
        let today = d(2026, 8, 5);
        let from_price = price_ir_contract(&quote("clean_price", 99.50), today).unwrap();
        let y = from_price["output"]["yield"].as_f64().unwrap();
        let from_yield = price_ir_contract(&quote("yield_rate", y), today).unwrap();
        let clean = from_yield["output"]["clean_price"].as_f64().unwrap();
        assert!((clean - 99.50).abs() < 1e-8, "round trip clean {clean}");
    }

    #[test]
    fn bond_prices_from_a_flat_curve() {
        let data = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "dated_date": "2026-05-15",
            "maturity_date": "2028-05-15",
            "valuation_date": "2026-08-05",
            "curve": { "type": "flat", "rate": 0.04 },
        }));
        let result = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        let output = &result["output"];
        // 4.5% coupon on a 4% curve -> premium to par
        let clean = output["clean_price"].as_f64().unwrap();
        assert!(clean > 100.0 && clean < 103.0, "clean {clean}");
        let y = output["yield"].as_f64().unwrap();
        assert!(y > 0.035 && y < 0.045, "yield {y}");
    }

    #[test]
    fn dated_date_defaults_to_the_previous_coupon_anchor() {
        // no dated_date: the May 15 anchor is inferred from maturity
        let data = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "maturity_date": "2028-05-15",
            "valuation_date": "2026-08-05",
            "clean_price": 99.50,
        }));
        let with_default = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        // accrued matches the explicit dated-date contract
        let explicit = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "dated_date": "2026-05-15",
            "maturity_date": "2028-05-15",
            "valuation_date": "2026-08-05",
            "clean_price": 99.50,
        }));
        let with_dated = price_ir_contract(&explicit, d(2026, 8, 5)).unwrap();
        assert_eq!(
            with_default["output"]["accrued_interest"],
            with_dated["output"]["accrued_interest"]
        );
    }

    #[test]
    fn bill_prices_from_a_discount_rate() {
        let data = contract(json!({
            "instrument": "Bill",
            "maturity_date": "2026-11-05",
            "valuation_date": "2026-08-05",
            "discount_rate": 0.048,
        }));
        let result = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        let output = &result["output"];
        // settle Aug 6 -> 91 days
        assert_eq!(output["days_to_maturity"].as_i64().unwrap(), 91);
        let expected = 100.0 * (1.0 - 0.048 * 91.0 / 360.0);
        assert!((output["price"].as_f64().unwrap() - expected).abs() < 1e-12);
        assert!(output["bond_equivalent_yield"].as_f64().unwrap() > 0.048);
    }

    #[test]
    fn quote_ambiguity_and_gaps_are_rejected() {
        let today = d(2026, 8, 5);
        // no quote at all
        let none = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "maturity_date": "2028-05-15",
        }));
        assert!(price_ir_contract(&none, today).is_err());
        // two quotes
        let both = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "maturity_date": "2028-05-15",
            "clean_price": 99.5,
            "yield_rate": 0.045,
        }));
        assert!(price_ir_contract(&both, today).is_err());
        // bond without a coupon
        let no_coupon = contract(json!({
            "instrument": "Bond",
            "maturity_date": "2028-05-15",
            "clean_price": 99.5,
        }));
        assert!(price_ir_contract(&no_coupon, today).is_err());
        // unknown instrument
        let swap = contract(json!({
            "instrument": "Swap",
            "maturity_date": "2028-05-15",
        }));
        assert!(price_ir_contract(&swap, today).is_err());
        // rate_data only
        let deposit: Contract = serde_json::from_value(json!({
            "action": "PV", "asset": "IR",
            "rate_data": {
                "instrument": "Deposit", "currency": "USD",
                "start_date": "0M", "maturity_date": "3M",
                "valuation_date": "0M", "notional": 1e6,
                "fix_rate": 0.05, "day_count": "A360",
                "business_day_adjustment": 0
            }
        }))
        .unwrap();
        assert!(price_ir_contract(&deposit, today).is_err());
    }

    #[test]
    fn corporate_bond_with_benchmark_curve_reports_spread_analytics() {
        let data = contract(json!({
            "instrument": "Corporate",
            "coupon_rate": 0.055,
            "dated_date": "2026-05-15",
            "maturity_date": "2031-05-15",
            "valuation_date": "2026-08-05",
            "clean_price": 98.75,
            "curve": { "type": "flat", "rate": 0.04 },
        }));
        let result = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        let output = &result["output"];
        assert_eq!(output["instrument"], "Corporate");
        // corporate defaults: T+2 from Wednesday Aug 5 -> Friday Aug 7,
        // 30/360 accrued May 15 -> Aug 7 = 82/360 at 5.5%
        assert_eq!(output["settlement_date"], "2026-08-07");
        let accrued = output["accrued_interest"].as_f64().unwrap();
        assert!(
            (accrued - 100.0 * 0.055 * 82.0 / 360.0).abs() < 1e-9,
            "accrued {accrued}"
        );
        // below par at a 5.5% coupon over a 4% curve: a healthy z-spread
        let z = output["z_spread"].as_f64().unwrap();
        assert!(z > 0.01 && z < 0.03, "z-spread {z}");
        assert!(output["spread_dv01"].as_f64().unwrap() > 0.0);
    }

    #[test]
    fn treasury_quote_without_curve_omits_spread_fields() {
        let data = contract(json!({
            "instrument": "Bond",
            "coupon_rate": 0.045,
            "dated_date": "2026-05-15",
            "maturity_date": "2028-05-15",
            "valuation_date": "2026-08-05",
            "clean_price": 99.50,
        }));
        let result = price_ir_contract(&data, d(2026, 8, 5)).unwrap();
        assert!(result["output"].get("z_spread").is_none());
    }
}
