//! US Treasury bill: a zero-coupon discount instrument.
//!
//! Bills quote a **discount rate** `d` on an Act/360 basis:
//! `price = 100 * (1 - d * n/360)` for `n` days from settlement to
//! maturity. The bond-equivalent yield (BEY, "coupon equivalent")
//! restates that price on the Act/365 basis used to compare bills with
//! notes and bonds.

use chrono::NaiveDate;

use crate::core::curves::YieldCurve;
use crate::core::errors::RustyQLibError;

#[derive(Debug, Clone)]
pub struct TreasuryBill {
    pub face_value: f64,
    pub maturity_date: NaiveDate,
}

impl TreasuryBill {
    pub fn new(face_value: f64, maturity_date: NaiveDate) -> Result<Self, RustyQLibError> {
        if !face_value.is_finite() || face_value <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "treasury bill",
                format!("face value must be positive, got {face_value}"),
            ));
        }
        Ok(TreasuryBill {
            face_value,
            maturity_date,
        })
    }

    /// Days from `settlement` to maturity, rejecting settlement at or
    /// after maturity.
    fn days_to_maturity(&self, settlement: NaiveDate) -> Result<i64, RustyQLibError> {
        let n = (self.maturity_date - settlement).num_days();
        if n <= 0 {
            return Err(RustyQLibError::invalid_input(
                "treasury bill",
                format!(
                    "settlement {settlement} is on or after maturity {}",
                    self.maturity_date
                ),
            ));
        }
        Ok(n)
    }

    /// Price per 100 face from the quoted discount rate:
    /// `100 * (1 - d * n/360)`.
    pub fn price_from_discount_rate(
        &self,
        discount_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let n = self.days_to_maturity(settlement)? as f64;
        let price = 100.0 * (1.0 - discount_rate * n / 360.0);
        if !price.is_finite() || price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "treasury bill",
                format!("discount rate {discount_rate} implies a non-positive price"),
            ));
        }
        Ok(price)
    }

    /// Quoted discount rate from a price per 100 face:
    /// `(100 - P)/100 * 360/n`.
    pub fn discount_rate_from_price(
        &self,
        price: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        if !price.is_finite() || price <= 0.0 {
            return Err(RustyQLibError::invalid_input(
                "treasury bill",
                format!("price must be positive, got {price}"),
            ));
        }
        let n = self.days_to_maturity(settlement)? as f64;
        Ok((100.0 - price) / 100.0 * 360.0 / n)
    }

    /// Bond-equivalent yield (coupon-equivalent yield) for the quoted
    /// discount rate.
    ///
    /// Up to 182 days this is the simple Act/365 yield
    /// `365 d / (360 - d n)`. Beyond 182 days the Treasury convention
    /// compounds one semiannual period: BEY solves
    /// `(1 + y/2) * (1 + (n/365 - 1/2) y) = 100/P`, the positive root of
    /// a quadratic.
    pub fn bond_equivalent_yield(
        &self,
        discount_rate: f64,
        settlement: NaiveDate,
    ) -> Result<f64, RustyQLibError> {
        let n = self.days_to_maturity(settlement)? as f64;
        let price = self.price_from_discount_rate(discount_rate, settlement)?;
        if n <= 182.0 {
            return Ok(365.0 * discount_rate / (360.0 - discount_rate * n));
        }
        // (x - 1/2)/2 * y^2 + x * y + (1 - R) = 0 with x = n/365, R = 100/P
        let x = n / 365.0;
        let ratio = 100.0 / price;
        let a = (x - 0.5) / 2.0;
        let b = x;
        let c = 1.0 - ratio;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 {
            return Err(RustyQLibError::NumericalError(format!(
                "no real bond-equivalent yield for discount rate {discount_rate}"
            )));
        }
        Ok((-b + disc.sqrt()) / (2.0 * a))
    }

    /// Present value on `curve`: the face discounted from maturity.
    pub fn pv(&self, curve: &YieldCurve) -> f64 {
        self.face_value * curve.df_date(self.maturity_date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    #[test]
    fn thirteen_week_bill_price_and_bey() {
        // 91-day bill at a 5% discount rate
        let bill = TreasuryBill::new(100.0, d(2026, 11, 4)).unwrap();
        let settle = d(2026, 8, 5);
        assert_eq!(bill.days_to_maturity(settle).unwrap(), 91);
        let price = bill.price_from_discount_rate(0.05, settle).unwrap();
        assert!((price - 100.0 * (1.0 - 0.05 * 91.0 / 360.0)).abs() < 1e-12);
        // simple-branch BEY: 365 d / (360 - d n)
        let bey = bill.bond_equivalent_yield(0.05, settle).unwrap();
        assert!((bey - 365.0 * 0.05 / (360.0 - 0.05 * 91.0)).abs() < 1e-14);
        // BEY exceeds the discount rate (paid on price, not face)
        assert!(bey > 0.05);
    }

    #[test]
    fn discount_rate_round_trips_through_price() {
        let bill = TreasuryBill::new(100.0, d(2027, 2, 3)).unwrap();
        let settle = d(2026, 8, 5);
        for dr in [0.001, 0.02, 0.055] {
            let price = bill.price_from_discount_rate(dr, settle).unwrap();
            let back = bill.discount_rate_from_price(price, settle).unwrap();
            assert!((back - dr).abs() < 1e-14, "dr={dr}");
        }
    }

    #[test]
    fn long_bill_bey_satisfies_the_compounding_equation() {
        // 52-week bill: 364 days > 182 -> quadratic branch
        let bill = TreasuryBill::new(100.0, d(2027, 8, 4)).unwrap();
        let settle = d(2026, 8, 5);
        let n = bill.days_to_maturity(settle).unwrap() as f64;
        assert!(n > 182.0);
        let dr = 0.048;
        let price = bill.price_from_discount_rate(dr, settle).unwrap();
        let y = bill.bond_equivalent_yield(dr, settle).unwrap();
        // definition: (1 + y/2)(1 + (n/365 - 1/2) y) = 100/P
        let residual = (1.0 + y / 2.0) * (1.0 + (n / 365.0 - 0.5) * y) - 100.0 / price;
        assert!(residual.abs() < 1e-12, "residual {residual}");
        assert!(y > dr);
    }

    #[test]
    fn pv_discounts_the_face() {
        use crate::core::curves::Compounding;
        use crate::core::daycount::DayCountConvention;
        let bill = TreasuryBill::new(1_000_000.0, d(2027, 8, 4)).unwrap();
        let curve = YieldCurve::flat(
            0.05,
            d(2026, 8, 5),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap();
        let t: f64 = 364.0 / 365.0;
        assert!((bill.pv(&curve) - 1_000_000.0 * (-0.05 * t).exp()).abs() < 1e-6);
    }

    #[test]
    fn validation_rejects_bad_inputs() {
        assert!(TreasuryBill::new(0.0, d(2027, 1, 1)).is_err());
        let bill = TreasuryBill::new(100.0, d(2027, 1, 1)).unwrap();
        // settlement after maturity
        assert!(bill.price_from_discount_rate(0.05, d(2027, 6, 1)).is_err());
        // absurd discount rate implying negative price
        assert!(bill.price_from_discount_rate(2.0, d(2026, 1, 1)).is_err());
        assert!(bill.discount_rate_from_price(-1.0, d(2026, 1, 1)).is_err());
    }
}
