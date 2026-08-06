//! Sequential discount-curve bootstrap.
//!
//! Instruments are processed in maturity order; each pins one pillar via
//! [`CurveInstrument::implied_df`], seeing only the curve built from the
//! instruments before it. The result is an ordinary
//! [`YieldCurve`], so bumping, key-rate shifts and serialization all work
//! on bootstrapped curves for free.

use chrono::NaiveDate;

use crate::bonds::CurveInstrument;
use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;

/// Bootstrap a discount curve from curve instruments — money-market
/// instruments ([`Deposit`](crate::bonds::Deposit),
/// [`Fra`](crate::bonds::Fra)) and/or quoted Treasuries
/// ([`BillQuote`](crate::bonds::BillQuote),
/// [`BondQuote`](crate::bonds::BondQuote)).
///
/// `day_count` is the *curve's* convention for converting pillar dates to
/// times (instruments keep their own accrual conventions). Instruments may
/// be passed in any order; maturities must be distinct and after
/// `reference_date`. The curve quotes zero rates continuously compounded
/// and interpolates log-linearly in discount factors.
pub fn bootstrap_curve(
    instruments: &[Box<dyn CurveInstrument>],
    reference_date: NaiveDate,
    day_count: DayCountConvention,
) -> Result<YieldCurve, RustyQLibError> {
    if instruments.is_empty() {
        return Err(RustyQLibError::invalid_input(
            "bootstrap",
            "no instruments to bootstrap from",
        ));
    }
    let mut order: Vec<usize> = (0..instruments.len()).collect();
    order.sort_by_key(|&i| instruments[i].maturity_date());
    for pair in order.windows(2) {
        let (a, b) = (
            instruments[pair[0]].maturity_date(),
            instruments[pair[1]].maturity_date(),
        );
        if a == b {
            return Err(RustyQLibError::invalid_input(
                "bootstrap",
                format!("two instruments share the maturity date {a}"),
            ));
        }
    }
    if instruments[order[0]].maturity_date() <= reference_date {
        return Err(RustyQLibError::invalid_input(
            "bootstrap",
            format!(
                "instrument maturity {} is not after the reference date {reference_date}",
                instruments[order[0]].maturity_date()
            ),
        ));
    }

    let mut tenors: Vec<Tenor> = Vec::with_capacity(instruments.len());
    let mut dfs: Vec<f64> = Vec::with_capacity(instruments.len());
    let mut curve: Option<YieldCurve> = None;

    for &i in &order {
        let instrument = &instruments[i];
        let df = instrument.implied_df(reference_date, curve.as_ref())?;
        if !(df > 0.0 && df.is_finite()) {
            return Err(RustyQLibError::NumericalError(format!(
                "instrument maturing {} implies a non-positive discount factor {df}",
                instrument.maturity_date()
            )));
        }
        tenors.push(Tenor::Date(instrument.maturity_date()));
        dfs.push(df);
        curve = Some(YieldCurve::from_discount_factors(
            &tenors,
            &dfs,
            reference_date,
            day_count,
            Compounding::Continuous,
            InterpolationMethod::LogLinearDf,
        )?);
    }

    Ok(curve.expect("at least one instrument was bootstrapped"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::{Deposit, Fra};

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    /// Deposits and FRAs mirroring the sample IR contract set: spot 1M and
    /// 3M deposits, then 3M-6M and 6M-9M FRAs chained on top.
    fn sample_instruments(reference: NaiveDate) -> Vec<Box<dyn CurveInstrument>> {
        let dc = DayCountConvention::Act360;
        vec![
            Box::new(Deposit::new(reference, d(2026, 2, 16), 1e6, 0.055, dc).unwrap()),
            Box::new(Deposit::new(reference, d(2026, 4, 15), 1e6, 0.05, dc).unwrap()),
            Box::new(Fra::new(d(2026, 4, 15), d(2026, 7, 15), 1e6, 0.06, dc).unwrap()),
            Box::new(Fra::new(d(2026, 7, 15), d(2026, 10, 15), 1e6, 0.065, dc).unwrap()),
        ]
    }

    #[test]
    fn deposits_and_fras_chain_multiplicatively() {
        let reference = d(2026, 1, 15);
        let curve = bootstrap_curve(
            &sample_instruments(reference),
            reference,
            DayCountConvention::Act365,
        )
        .unwrap();

        // 3M deposit pillar: df = 1/(1 + 5% * 90/360)
        let df_3m = 1.0 / (1.0 + 0.05 * 90.0 / 360.0);
        assert!((curve.df_date(d(2026, 4, 15)) - df_3m).abs() < 1e-12);

        // 6M pillar: 3M df rolled through the 3M-6M FRA (91 days)
        let df_6m = df_3m / (1.0 + 0.06 * 91.0 / 360.0);
        assert!((curve.df_date(d(2026, 7, 15)) - df_6m).abs() < 1e-12);

        // 9M pillar continues the chain (92 days)
        let df_9m = df_6m / (1.0 + 0.065 * 92.0 / 360.0);
        assert!((curve.df_date(d(2026, 10, 15)) - df_9m).abs() < 1e-12);

        // four instruments -> four pillars
        assert_eq!(curve.pillars().len(), 4);
    }

    #[test]
    fn input_order_does_not_matter() {
        let reference = d(2026, 1, 15);
        let mut shuffled = sample_instruments(reference);
        shuffled.reverse();
        let a = bootstrap_curve(
            &sample_instruments(reference),
            reference,
            DayCountConvention::Act365,
        )
        .unwrap();
        let b = bootstrap_curve(&shuffled, reference, DayCountConvention::Act365).unwrap();
        for t in [0.1, 0.25, 0.5, 0.7] {
            assert!((a.df(t) - b.df(t)).abs() < 1e-15);
        }
    }

    #[test]
    fn duplicate_maturities_are_rejected() {
        let reference = d(2026, 1, 15);
        let dc = DayCountConvention::Act360;
        let instruments: Vec<Box<dyn CurveInstrument>> = vec![
            Box::new(Deposit::new(reference, d(2026, 4, 15), 1e6, 0.05, dc).unwrap()),
            Box::new(Deposit::new(reference, d(2026, 4, 15), 1e6, 0.051, dc).unwrap()),
        ];
        assert!(bootstrap_curve(&instruments, reference, DayCountConvention::Act365).is_err());
    }

    #[test]
    fn empty_input_is_rejected() {
        assert!(bootstrap_curve(&[], d(2026, 1, 15), DayCountConvention::Act365).is_err());
    }

    #[test]
    fn fra_first_without_short_end_is_rejected() {
        let reference = d(2026, 1, 15);
        let dc = DayCountConvention::Act360;
        let instruments: Vec<Box<dyn CurveInstrument>> = vec![Box::new(
            Fra::new(d(2026, 4, 15), d(2026, 7, 15), 1e6, 0.06, dc).unwrap(),
        )];
        assert!(bootstrap_curve(&instruments, reference, DayCountConvention::Act365).is_err());
    }
}
