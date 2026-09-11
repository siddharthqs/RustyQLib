//! Multi-curve calibration: discount and forecast curves solved
//! **together** from market quotes, with the Jacobian that turns curve
//! sensitivities into sensitivities to the quotes.
//!
//! The classic post-2008 setup: an OIS curve for discounting and one
//! forecast curve per index tenor, each pinned by its own instruments —
//! deposits, FRAs, SOFR futures, vanilla swaps, OIS and basis swaps
//! ([`RateInstrument`]). Every instrument pins one pillar (its maturity)
//! on one curve, and prices off the full set of curves, so the pillars
//! cannot be solved one at a time: a 5y swap on the 3M curve needs the
//! OIS curve out to 5y for discounting. [`MultiCurveBuilder::calibrate`]
//! solves all pillars at once — the continuously compounded zero rate
//! at every pillar of every curve — by Newton's method on the vector of
//! residuals `model rate - quote`, with a finite-difference Jacobian
//! and a backtracking line search. A single curve (discount = forecast)
//! is the one-curve special case.
//!
//! The Jacobian `J = d residual / d zero rate` at the solution is kept
//! on the [`MultiCurve`]: since each residual moves one-for-one with its
//! own quote, `d zero / d quote = J^-1`, and any trade's sensitivity to
//! the quotes is `(J^-1)^T (d PV / d zero)` — the **bucketed PV01 to
//! market instruments** of [`MultiCurve::quote_sensitivities`], the
//! number a desk hedges with. A calibrated instrument itself has
//! sensitivity only to its own quote, exactly.

use std::collections::BTreeMap;

use chrono::NaiveDate;

use crate::bonds::{Deposit, Fra};
use crate::core::curves::{Compounding, InterpolationMethod, Tenor, YieldCurve};
use crate::core::daycount::DayCountConvention;
use crate::core::errors::RustyQLibError;
use crate::rates::contracts::{BasisSwap, OvernightIndexSwap, SofrFuture, VanillaSwap};

const FIELD: &str = "multi-curve";

/// A quoted instrument and the curves it prices off. Discounting is
/// always on the builder's discount curve; each variant names the
/// forecast curve(s) it forecasts on and the curve whose pillar it pins.
#[derive(Debug, Clone)]
pub enum RateInstrument {
    /// A cash deposit pinning `curve` at its maturity with its simple
    /// rate.
    Deposit { deposit: Deposit, curve: String },
    /// An FRA pinning `forecast` at its maturity with its fixed rate.
    Fra { fra: Fra, forecast: String },
    /// A SOFR future quoted at `price`, pinning `curve` at the end of
    /// its reference period; `convexity` (a rate) is subtracted from the
    /// futures-implied rate to get the forward.
    SofrFuture {
        future: SofrFuture,
        price: f64,
        convexity: f64,
        curve: String,
    },
    /// A par swap pinning `forecast` at its maturity with its fixed rate.
    Swap { swap: VanillaSwap, forecast: String },
    /// A par OIS pinning `curve` at its maturity with its fixed rate;
    /// forecasts on `curve`, discounts on the discount curve (the same
    /// curve when `curve` is the discount curve).
    Ois {
        ois: OvernightIndexSwap,
        curve: String,
    },
    /// A basis swap at its quoted spread, forecasting leg A on
    /// `forecast_a` and leg B on `forecast_b`, pinning `pins` at its
    /// maturity.
    BasisSwap {
        swap: BasisSwap,
        forecast_a: String,
        forecast_b: String,
        pins: String,
    },
}

impl RateInstrument {
    /// The curve whose pillar this instrument determines.
    pub fn pins(&self) -> &str {
        match self {
            RateInstrument::Deposit { curve, .. } => curve,
            RateInstrument::Fra { forecast, .. } => forecast,
            RateInstrument::SofrFuture { curve, .. } => curve,
            RateInstrument::Swap { forecast, .. } => forecast,
            RateInstrument::Ois { curve, .. } => curve,
            RateInstrument::BasisSwap { pins, .. } => pins,
        }
    }

    /// The pillar date: the instrument's maturity.
    pub fn pillar_date(&self) -> NaiveDate {
        match self {
            RateInstrument::Deposit { deposit, .. } => deposit.maturity_date,
            RateInstrument::Fra { fra, .. } => fra.maturity_date,
            RateInstrument::SofrFuture { future, .. } => future.end,
            RateInstrument::Swap { swap, .. } => swap.maturity_date,
            RateInstrument::Ois { ois, .. } => ois.maturity_date,
            RateInstrument::BasisSwap { swap, .. } => swap.maturity_date,
        }
    }

    /// The market quote as a rate: the deposit, FRA, swap or OIS rate,
    /// the futures-implied forward net of convexity, or the basis spread.
    pub fn quote(&self) -> f64 {
        match self {
            RateInstrument::Deposit { deposit, .. } => deposit.fix_rate,
            RateInstrument::Fra { fra, .. } => fra.fix_rate,
            RateInstrument::SofrFuture {
                price, convexity, ..
            } => SofrFuture::rate_from_price(*price) - convexity,
            RateInstrument::Swap { swap, .. } => swap.fixed_rate,
            RateInstrument::Ois { ois, .. } => ois.fixed_rate,
            RateInstrument::BasisSwap { swap, .. } => swap.spread,
        }
    }

    /// A short description for reports.
    pub fn label(&self) -> String {
        let kind = match self {
            RateInstrument::Deposit { .. } => "deposit",
            RateInstrument::Fra { .. } => "fra",
            RateInstrument::SofrFuture { .. } => "sofr future",
            RateInstrument::Swap { .. } => "swap",
            RateInstrument::Ois { .. } => "ois",
            RateInstrument::BasisSwap { .. } => "basis swap",
        };
        format!("{kind} {} @ {}", self.pillar_date(), self.pins())
    }

    /// Every curve this instrument reads.
    fn curves_used(&self) -> Vec<&str> {
        match self {
            RateInstrument::Deposit { curve, .. } => vec![curve],
            RateInstrument::Fra { forecast, .. } => vec![forecast],
            RateInstrument::SofrFuture { curve, .. } => vec![curve],
            RateInstrument::Swap { forecast, .. } => vec![forecast],
            RateInstrument::Ois { curve, .. } => vec![curve],
            RateInstrument::BasisSwap {
                forecast_a,
                forecast_b,
                pins,
                ..
            } => vec![forecast_a, forecast_b, pins],
        }
    }

    /// `model rate - quote` on the given curves.
    fn residual(&self, curves: &MultiCurve) -> Result<f64, RustyQLibError> {
        let model = match self {
            RateInstrument::Deposit { deposit, curve } => {
                let c = curves.curve(curve)?;
                (c.df_date(deposit.start_date) / c.df_date(deposit.maturity_date) - 1.0)
                    / deposit.accrual()
            }
            RateInstrument::Fra { fra, forecast } => fra.forward_rate(curves.curve(forecast)?)?,
            RateInstrument::SofrFuture { future, curve, .. } => {
                future.fair_rate(curves.curve(curve)?)?
            }
            RateInstrument::Swap { swap, forecast } => {
                swap.par_rate(curves.discount(), curves.curve(forecast)?)?
            }
            RateInstrument::Ois { ois, curve } => {
                ois.par_rate(curves.discount(), curves.curve(curve)?)?
            }
            RateInstrument::BasisSwap {
                swap,
                forecast_a,
                forecast_b,
                ..
            } => swap.fair_spread(
                curves.discount(),
                curves.curve(forecast_a)?,
                curves.curve(forecast_b)?,
            )?,
        };
        Ok(model - self.quote())
    }
}

/// One solved pillar: which curve and date, and the instrument behind it.
#[derive(Debug, Clone)]
pub struct Pillar {
    pub curve: String,
    pub date: NaiveDate,
    pub label: String,
    pub quote: f64,
}

/// A trade's sensitivity to one calibration quote.
#[derive(Debug, Clone)]
pub struct QuoteSensitivity {
    pub label: String,
    pub curve: String,
    pub pillar_date: NaiveDate,
    pub quote: f64,
    /// PV change for a one-basis-point rise of this quote, the other
    /// quotes held (every curve re-solved).
    pub pv_per_bp: f64,
}

/// The calibrated curve set.
#[derive(Debug, Clone)]
pub struct MultiCurve {
    pub reference_date: NaiveDate,
    pub day_count: DayCountConvention,
    discount_name: String,
    curves: BTreeMap<String, YieldCurve>,
    /// Pillars in unknown order; `pillars[k]` is zero rate `k`.
    pillars: Vec<Pillar>,
    /// `zero[k]` for `pillars[k]`.
    zeros: Vec<f64>,
    /// `jacobian[i][k] = d residual_i / d zero_k` at the solution;
    /// empty until calibrated.
    jacobian: Vec<Vec<f64>>,
    pub iterations: usize,
    /// Largest `|model rate - quote|` after the solve.
    pub max_residual: f64,
}

impl MultiCurve {
    /// The discount curve.
    pub fn discount(&self) -> &YieldCurve {
        &self.curves[&self.discount_name]
    }

    /// The discount curve's name.
    pub fn discount_name(&self) -> &str {
        &self.discount_name
    }

    /// A curve by name.
    pub fn curve(&self, name: &str) -> Result<&YieldCurve, RustyQLibError> {
        self.curves
            .get(name)
            .ok_or_else(|| RustyQLibError::invalid_input(FIELD, format!("no curve named {name:?}")))
    }

    /// All curves by name.
    pub fn curves(&self) -> &BTreeMap<String, YieldCurve> {
        &self.curves
    }

    /// The solved pillars, in the order the Jacobian and sensitivities use.
    pub fn pillars(&self) -> &[Pillar] {
        &self.pillars
    }

    /// Rebuild the curves from a zero-rate vector.
    fn with_zeros(&self, zeros: &[f64]) -> Result<MultiCurve, RustyQLibError> {
        let mut by_curve: BTreeMap<&str, (Vec<Tenor>, Vec<f64>)> = BTreeMap::new();
        for (pillar, &z) in self.pillars.iter().zip(zeros) {
            let entry = by_curve.entry(pillar.curve.as_str()).or_default();
            entry.0.push(Tenor::Date(pillar.date));
            entry.1.push(z);
        }
        let mut curves = BTreeMap::new();
        for (name, (tenors, rates)) in by_curve {
            curves.insert(
                name.to_string(),
                YieldCurve::from_zero_rates(
                    &tenors,
                    &rates,
                    self.reference_date,
                    self.day_count,
                    Compounding::Continuous,
                    InterpolationMethod::LogLinearDf,
                )?,
            );
        }
        Ok(MultiCurve {
            curves,
            zeros: zeros.to_vec(),
            ..self.clone()
        })
    }

    /// Sensitivity of `pv` to every calibration quote, one basis point
    /// each, through the calibration Jacobian: `(J^-1)^T (d PV / d zero)`.
    /// `pv` prices the trade on a curve set (bumped ones are passed in).
    pub fn quote_sensitivities(
        &self,
        pv: impl Fn(&MultiCurve) -> Result<f64, RustyQLibError>,
    ) -> Result<Vec<QuoteSensitivity>, RustyQLibError> {
        if self.jacobian.is_empty() {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                "the curve set has no calibration Jacobian",
            ));
        }
        let n = self.zeros.len();
        let h = 1e-6;
        let mut gradient = vec![0.0; n];
        for (k, g) in gradient.iter_mut().enumerate() {
            let mut up = self.zeros.clone();
            up[k] += h;
            let mut down = self.zeros.clone();
            down[k] -= h;
            *g = (pv(&self.with_zeros(&up)?)? - pv(&self.with_zeros(&down)?)?) / (2.0 * h);
        }
        // solve J^T s = gradient
        let transposed: Vec<Vec<f64>> = (0..n)
            .map(|k| (0..n).map(|i| self.jacobian[i][k]).collect())
            .collect();
        let s = solve_linear(transposed, gradient)?;
        Ok(self
            .pillars
            .iter()
            .zip(s)
            .map(|(p, s)| QuoteSensitivity {
                label: p.label.clone(),
                curve: p.curve.clone(),
                pillar_date: p.date,
                quote: p.quote,
                pv_per_bp: s * 1e-4,
            })
            .collect())
    }
}

/// Instruments and settings for a multi-curve calibration.
#[derive(Debug, Clone)]
pub struct MultiCurveBuilder {
    pub reference_date: NaiveDate,
    /// The curves' day count for pillar times.
    pub day_count: DayCountConvention,
    /// The curve every instrument discounts on.
    pub discount: String,
    pub instruments: Vec<RateInstrument>,
    /// Convergence: largest `|model rate - quote|`.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl MultiCurveBuilder {
    pub fn new(reference_date: NaiveDate, day_count: DayCountConvention, discount: &str) -> Self {
        MultiCurveBuilder {
            reference_date,
            day_count,
            discount: discount.to_string(),
            instruments: Vec::new(),
            tolerance: 1e-12,
            max_iterations: 50,
        }
    }

    pub fn add(mut self, instrument: RateInstrument) -> Self {
        self.instruments.push(instrument);
        self
    }

    fn validate(&self) -> Result<(), RustyQLibError> {
        if self.instruments.is_empty() {
            return Err(RustyQLibError::invalid_input(FIELD, "no instruments"));
        }
        let mut pinned: BTreeMap<&str, Vec<NaiveDate>> = BTreeMap::new();
        for inst in &self.instruments {
            let date = inst.pillar_date();
            if date <= self.reference_date {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!(
                        "{} matures on or before the reference date {}",
                        inst.label(),
                        self.reference_date
                    ),
                ));
            }
            let dates = pinned.entry(inst.pins()).or_default();
            if dates.contains(&date) {
                return Err(RustyQLibError::invalid_input(
                    FIELD,
                    format!("two instruments pin {:?} at {date}", inst.pins()),
                ));
            }
            dates.push(date);
        }
        if !pinned.contains_key(self.discount.as_str()) {
            return Err(RustyQLibError::invalid_input(
                FIELD,
                format!("no instrument pins the discount curve {:?}", self.discount),
            ));
        }
        for inst in &self.instruments {
            for name in inst.curves_used() {
                if !pinned.contains_key(name) {
                    return Err(RustyQLibError::invalid_input(
                        FIELD,
                        format!(
                            "{} prices off {name:?}, which no instrument pins",
                            inst.label()
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Solve every curve's pillars together.
    pub fn calibrate(&self) -> Result<MultiCurve, RustyQLibError> {
        self.validate()?;
        // unknowns: pillars grouped by curve, dated in order; the
        // instrument order is fixed once here and reused for residuals
        let mut order: Vec<usize> = (0..self.instruments.len()).collect();
        order.sort_by_key(|&i| {
            (
                self.instruments[i].pins().to_string(),
                self.instruments[i].pillar_date(),
            )
        });
        let instruments: Vec<&RateInstrument> =
            order.iter().map(|&i| &self.instruments[i]).collect();
        let pillars: Vec<Pillar> = instruments
            .iter()
            .map(|inst| Pillar {
                curve: inst.pins().to_string(),
                date: inst.pillar_date(),
                label: inst.label(),
                quote: inst.quote(),
            })
            .collect();
        let n = pillars.len();

        // initial guess: each curve flat at the mean of its rate quotes
        let mut zeros = vec![0.0; n];
        {
            let mut sums: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
            for inst in &instruments {
                if !matches!(inst, RateInstrument::BasisSwap { .. }) {
                    let e = sums.entry(inst.pins()).or_default();
                    e.0 += inst.quote();
                    e.1 += 1;
                }
            }
            for (k, p) in pillars.iter().enumerate() {
                zeros[k] = match sums.get(p.curve.as_str()) {
                    Some(&(sum, count)) if count > 0 => sum / count as f64,
                    _ => 0.03,
                };
            }
        }

        let template = MultiCurve {
            reference_date: self.reference_date,
            day_count: self.day_count,
            discount_name: self.discount.clone(),
            curves: BTreeMap::new(),
            pillars,
            zeros: Vec::new(),
            jacobian: Vec::new(),
            iterations: 0,
            max_residual: f64::INFINITY,
        };
        let residuals = |z: &[f64]| -> Result<Vec<f64>, RustyQLibError> {
            let curves = template.with_zeros(z)?;
            instruments.iter().map(|i| i.residual(&curves)).collect()
        };
        let norm = |r: &[f64]| r.iter().fold(0.0_f64, |m, x| m.max(x.abs()));

        let mut r = residuals(&zeros)?;
        let mut jacobian = Vec::new();
        let mut iterations = 0;
        while norm(&r) > self.tolerance {
            if iterations >= self.max_iterations {
                return Err(RustyQLibError::CalibrationFailed {
                    iterations,
                    residual: norm(&r),
                    reason: "multi-curve Newton solve did not converge".to_string(),
                });
            }
            jacobian = numeric_jacobian(&residuals, &zeros, &r)?;
            let step = solve_linear(jacobian.clone(), r.iter().map(|x| -x).collect())?;
            // backtracking line search on the residual norm
            let mut lambda = 1.0;
            let current = norm(&r);
            loop {
                let trial: Vec<f64> = zeros
                    .iter()
                    .zip(&step)
                    .map(|(z, s)| z + lambda * s)
                    .collect();
                match residuals(&trial) {
                    Ok(tr) if norm(&tr) < current || lambda < 1e-3 => {
                        zeros = trial;
                        r = tr;
                        break;
                    }
                    _ => lambda *= 0.5,
                }
            }
            iterations += 1;
        }
        // the Jacobian at the solution, for sensitivities
        if jacobian.is_empty() || iterations > 0 {
            jacobian = numeric_jacobian(&residuals, &zeros, &r)?;
        }
        let mut solved = template.with_zeros(&zeros)?;
        solved.jacobian = jacobian;
        solved.iterations = iterations;
        solved.max_residual = norm(&r);
        Ok(solved)
    }
}

/// Forward-difference Jacobian of `residuals` at `z` (`r0 = residuals(z)`).
fn numeric_jacobian(
    residuals: &impl Fn(&[f64]) -> Result<Vec<f64>, RustyQLibError>,
    z: &[f64],
    r0: &[f64],
) -> Result<Vec<Vec<f64>>, RustyQLibError> {
    let n = z.len();
    let h = 1e-7;
    let mut columns = Vec::with_capacity(n);
    for k in 0..n {
        let mut bumped = z.to_vec();
        bumped[k] += h;
        let rk = residuals(&bumped)?;
        columns.push(
            rk.iter()
                .zip(r0)
                .map(|(a, b)| (a - b) / h)
                .collect::<Vec<f64>>(),
        );
    }
    // columns[k][i] -> jacobian[i][k]
    Ok((0..n)
        .map(|i| (0..n).map(|k| columns[k][i]).collect())
        .collect())
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, RustyQLibError> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .expect("non-empty system");
        if a[pivot][col].abs() < 1e-300 {
            return Err(RustyQLibError::NumericalError(
                "singular calibration Jacobian: an instrument does not move its pillar".to_string(),
            ));
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let factor = a[row][col] / a[col][col];
            if factor != 0.0 {
                for k in col..n {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let sum: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - sum) / a[row][row];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bonds::bootstrap_curve;
    use crate::bonds::CurveInstrument;
    use crate::rates::PayerReceiver;

    fn d(y: i32, m: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, day).unwrap()
    }

    fn reference() -> NaiveDate {
        d(2026, 8, 13)
    }

    fn flat(rate: f64) -> YieldCurve {
        YieldCurve::flat(
            rate,
            reference(),
            DayCountConvention::Act365,
            Compounding::Continuous,
        )
        .unwrap()
    }

    fn ois(maturity: NaiveDate, rate: f64) -> OvernightIndexSwap {
        OvernightIndexSwap::sofr_standard(1.0, rate, PayerReceiver::Payer, reference(), maturity)
            .unwrap()
    }

    fn swap(maturity: NaiveDate, rate: f64) -> VanillaSwap {
        VanillaSwap::usd_standard(1.0, rate, PayerReceiver::Payer, reference(), maturity).unwrap()
    }

    /// A dual-curve market generated from known curves: OIS flat at 4%
    /// and the 3M forecast curve flat at 4.5%.
    fn dual_curve_market() -> (MultiCurveBuilder, YieldCurve, YieldCurve) {
        let ois_curve = flat(0.04);
        let fwd_curve = flat(0.045);
        let dc = DayCountConvention::Act360;
        let deposit_end = d(2026, 11, 13);
        let deposit_rate = (fwd_curve.df_date(reference()) / fwd_curve.df_date(deposit_end) - 1.0)
            / dc.year_fraction(reference(), deposit_end);
        let fra = Fra::new(deposit_end, d(2027, 2, 15), 1.0, 0.0, dc).unwrap();
        let fra_rate = fra.forward_rate(&fwd_curve).unwrap();
        let mut builder = MultiCurveBuilder::new(reference(), DayCountConvention::Act365, "OIS")
            .add(RateInstrument::Deposit {
                deposit: Deposit::new(reference(), deposit_end, 1.0, deposit_rate, dc).unwrap(),
                curve: "3M".into(),
            })
            .add(RateInstrument::Fra {
                fra: Fra::new(deposit_end, d(2027, 2, 15), 1.0, fra_rate, dc).unwrap(),
                forecast: "3M".into(),
            });
        for maturity in [
            d(2027, 8, 13),
            d(2028, 8, 14),
            d(2031, 8, 13),
            d(2036, 8, 13),
        ] {
            let par = ois(maturity, 0.04)
                .par_rate(&ois_curve, &ois_curve)
                .unwrap();
            builder = builder.add(RateInstrument::Ois {
                ois: ois(maturity, par),
                curve: "OIS".into(),
            });
        }
        for maturity in [d(2028, 8, 14), d(2031, 8, 13), d(2036, 8, 13)] {
            let par = swap(maturity, 0.04)
                .par_rate(&ois_curve, &fwd_curve)
                .unwrap();
            builder = builder.add(RateInstrument::Swap {
                swap: swap(maturity, par),
                forecast: "3M".into(),
            });
        }
        (builder, ois_curve, fwd_curve)
    }

    #[test]
    fn dual_curve_solve_recovers_the_generating_curves_and_reprices_every_quote() {
        let (builder, ois_curve, fwd_curve) = dual_curve_market();
        let curves = builder.calibrate().unwrap();
        assert!(curves.max_residual < 1e-12, "{}", curves.max_residual);
        assert!(curves.iterations <= 10, "{} iterations", curves.iterations);
        assert_eq!(curves.pillars().len(), 9);
        // flat generating curves are exactly representable, so every
        // pillar df comes back (interpolation is log-linear in both)
        for p in curves.pillars() {
            let (truth, built) = match p.curve.as_str() {
                "OIS" => (&ois_curve, curves.discount()),
                _ => (&fwd_curve, curves.curve("3M").unwrap()),
            };
            assert!(
                (built.df_date(p.date) / truth.df_date(p.date) - 1.0).abs() < 1e-9,
                "{}: {} vs {}",
                p.label,
                built.df_date(p.date),
                truth.df_date(p.date)
            );
        }
        // the two curves are genuinely different
        assert!(
            curves.discount().df_date(d(2031, 8, 13))
                > curves.curve("3M").unwrap().df_date(d(2031, 8, 13))
        );
    }

    #[test]
    fn single_curve_matches_the_sequential_bootstrap() {
        // deposits and FRAs only, one curve: the global solve and the
        // sequential bootstrap pin the same pillars to the same dfs
        let dc = DayCountConvention::Act360;
        let reference = d(2026, 1, 15);
        let deposits = [
            Deposit::new(reference, d(2026, 2, 16), 1e6, 0.055, dc).unwrap(),
            Deposit::new(reference, d(2026, 4, 15), 1e6, 0.05, dc).unwrap(),
        ];
        let fras = [
            Fra::new(d(2026, 4, 15), d(2026, 7, 15), 1e6, 0.06, dc).unwrap(),
            Fra::new(d(2026, 7, 15), d(2026, 10, 15), 1e6, 0.065, dc).unwrap(),
        ];
        let sequential: Vec<Box<dyn CurveInstrument>> = vec![
            Box::new(deposits[0].clone()),
            Box::new(deposits[1].clone()),
            Box::new(fras[0].clone()),
            Box::new(fras[1].clone()),
        ];
        let expected = bootstrap_curve(&sequential, reference, DayCountConvention::Act365).unwrap();
        let mut builder = MultiCurveBuilder::new(reference, DayCountConvention::Act365, "USD");
        for deposit in deposits {
            builder = builder.add(RateInstrument::Deposit {
                deposit,
                curve: "USD".into(),
            });
        }
        for fra in fras {
            builder = builder.add(RateInstrument::Fra {
                fra,
                forecast: "USD".into(),
            });
        }
        let curves = builder.calibrate().unwrap();
        for p in curves.pillars() {
            assert!(
                (curves.discount().df_date(p.date) - expected.df_date(p.date)).abs() < 1e-10,
                "{}",
                p.label
            );
        }
    }

    #[test]
    fn a_calibration_instrument_is_sensitive_only_to_its_own_quote() {
        let (builder, _, _) = dual_curve_market();
        let curves = builder.calibrate().unwrap();
        // the 5y swap from the set, on 10mm: its PV01 to the 5y swap
        // quote is the annuity per bp (the quote rising lifts the par
        // rate above the payer's fixed rate), and zero to every other
        // quote; fixed_rate_pv01 is the same number with the opposite
        // sign, being the sensitivity to the trade's own fixed rate
        let five_year = {
            let q = builder
                .instruments
                .iter()
                .find_map(|i| match i {
                    RateInstrument::Swap { swap, .. } if swap.maturity_date == d(2031, 8, 13) => {
                        Some(swap.clone())
                    }
                    _ => None,
                })
                .unwrap();
            VanillaSwap {
                notional: 10_000_000.0,
                ..q
            }
        };
        let sens = curves
            .quote_sensitivities(|c| five_year.pv_with(c.discount(), c.curve("3M")?))
            .unwrap();
        let annuity_per_bp = -five_year.fixed_rate_pv01(curves.discount()).unwrap();
        assert!(annuity_per_bp > 0.0);
        for s in &sens {
            if s.label.starts_with("swap 2031-08-13") {
                assert!(
                    (s.pv_per_bp - annuity_per_bp).abs() < 1e-3 * annuity_per_bp.abs(),
                    "{}: {} vs {annuity_per_bp}",
                    s.label,
                    s.pv_per_bp
                );
            } else {
                assert!(s.pv_per_bp.abs() < 1e-3, "{}: {}", s.label, s.pv_per_bp);
            }
        }
        // an off-pillar 3y swap spreads its risk over the 2y and 5y
        // swap quotes, and the total is its fixed-rate PV01
        let three_year = VanillaSwap::usd_standard(
            10_000_000.0,
            0.045,
            PayerReceiver::Payer,
            reference(),
            d(2029, 8, 13),
        )
        .unwrap();
        let sens = curves
            .quote_sensitivities(|c| three_year.pv_with(c.discount(), c.curve("3M")?))
            .unwrap();
        let swap_buckets: f64 = sens
            .iter()
            .filter(|s| s.label.starts_with("swap"))
            .map(|s| s.pv_per_bp)
            .sum();
        let total: f64 = sens.iter().map(|s| s.pv_per_bp).sum();
        let own = -three_year.fixed_rate_pv01(curves.discount()).unwrap();
        assert!(swap_buckets > 0.0, "{swap_buckets}");
        assert!((total - own).abs() < 0.1 * own.abs(), "{total} vs {own}");
        let two_year = sens
            .iter()
            .find(|s| s.label.starts_with("swap 2028"))
            .unwrap();
        let five = sens
            .iter()
            .find(|s| s.label.starts_with("swap 2031"))
            .unwrap();
        assert!(two_year.pv_per_bp > 0.0 && five.pv_per_bp > 0.0);
    }

    #[test]
    fn validation_rejects_bad_setups() {
        let dc = DayCountConvention::Act360;
        // a forecast curve nobody pins: the basis swap reads "6M"
        let basis = BasisSwap::new(
            1.0,
            0.001,
            reference(),
            d(2029, 8, 13),
            crate::rates::BasisSwapLeg::new(crate::core::calendar::Frequency::Quarterly, dc),
            crate::rates::BasisSwapLeg::new(crate::core::calendar::Frequency::Semiannual, dc),
            crate::core::calendar::Calendar::UsGovernmentBond,
            crate::core::calendar::BusinessDayConvention::ModifiedFollowing,
        )
        .unwrap();
        let bad = MultiCurveBuilder::new(reference(), DayCountConvention::Act365, "OIS")
            .add(RateInstrument::Ois {
                ois: ois(d(2027, 8, 13), 0.04),
                curve: "OIS".into(),
            })
            .add(RateInstrument::Swap {
                swap: swap(d(2028, 8, 14), 0.045),
                forecast: "3M".into(),
            })
            .add(RateInstrument::BasisSwap {
                swap: basis,
                forecast_a: "3M".into(),
                forecast_b: "6M".into(),
                pins: "3M".into(),
            });
        assert!(bad.calibrate().is_err());
        // no instrument on the discount curve
        let bad = MultiCurveBuilder::new(reference(), DayCountConvention::Act365, "OIS").add(
            RateInstrument::Deposit {
                deposit: Deposit::new(reference(), d(2026, 11, 13), 1.0, 0.04, dc).unwrap(),
                curve: "3M".into(),
            },
        );
        assert!(bad.calibrate().is_err());
        // two instruments on one pillar
        let bad = MultiCurveBuilder::new(reference(), DayCountConvention::Act365, "OIS")
            .add(RateInstrument::Ois {
                ois: ois(d(2027, 8, 13), 0.04),
                curve: "OIS".into(),
            })
            .add(RateInstrument::Ois {
                ois: ois(d(2027, 8, 13), 0.041),
                curve: "OIS".into(),
            });
        assert!(bad.calibrate().is_err());
        assert!(
            MultiCurveBuilder::new(reference(), DayCountConvention::Act365, "OIS")
                .calibrate()
                .is_err()
        );
    }
}
