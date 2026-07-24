//! Implied volatility surface construction from quoted options.
//!
//! Takes a list of options carrying market prices (`current_price`), solves
//! each for its Black-Scholes implied vol (robust safeguarded Newton), and
//! assembles the per-maturity smiles into a canonical
//! [`crate::core::vols::VolSurface`] on absolute strikes — the same type
//! the pricers consume, so a built surface can immediately price other
//! options (including through the Dupire local vol model).

use std::collections::BTreeMap;
use chrono::NaiveDate;

use crate::core::curves::Tenor;
use crate::core::daycount::DayCountConvention;
use crate::core::vols::VolSurface;
use super::vanilla_option::EquityOption;
use crate::core::errors::RustyQLibError;

/// Build an implied vol surface from quoted options. Quotes without a
/// positive market price or violating arbitrage bounds are skipped (with a
/// warning); at least one valid quote is required.
pub fn build_implied_vol_surface(contracts: &[Box<EquityOption>]) -> Result<VolSurface, RustyQLibError> {
    if contracts.is_empty() {
        return Err(RustyQLibError::invalid_input("vol surface", "no contracts provided".to_string()));
    }
    let reference_date = contracts[0].base.valuation_date;
    let mut smiles: BTreeMap<NaiveDate, Vec<(f64, f64)>> = BTreeMap::new();
    let mut skipped = 0usize;

    for option in contracts {
        let target = option.base.current_price.value();
        if target <= 0.0 {
            skipped += 1;
            continue;
        }
        match option.try_imp_vol(target) {
            Ok(vol) => smiles
                .entry(option.base.maturity_date)
                .or_default()
                .push((option.base.strike_price, vol)),
            Err(err) => {
                eprintln!(
                    "skipping quote {} K={} T={}: {err}",
                    option.base.symbol, option.base.strike_price, option.base.maturity_date
                );
                skipped += 1;
            }
        }
    }
    if smiles.is_empty() {
        return Err(RustyQLibError::invalid_input("vol surface", format!("no valid quotes ({skipped} skipped)")));
    }

    let mut tenors = Vec::new();
    let mut smile_points = Vec::new();
    for (maturity, mut points) in smiles {
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
        tenors.push(Tenor::Date(maturity));
        smile_points.push(points);
    }
    VolSurface::from_strike_smiles(&tenors, &smile_points, reference_date, DayCountConvention::Act365)
        .map_err(RustyQLibError::from)
}
