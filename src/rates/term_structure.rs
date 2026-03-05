/// Term Structure Module
///
/// Provides types for managing interest rate term structures with various
/// day count conventions and interpolation methods.
use std::fmt;

// ─── Day Count Convention ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayCountConvention {
    Actual360,
    Actual365,
    ActualActual,
    Thirty360,
    Thirty360European,
}

impl fmt::Display for DayCountConvention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Actual360 => "Actual/360",
            Self::Actual365 => "Actual/365",
            Self::ActualActual => "Actual/Actual",
            Self::Thirty360 => "30/360",
            Self::Thirty360European => "30E/360",
        };
        write!(f, "{s}")
    }
}

fn is_leap_year(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_year(year: i32) -> f64 {
    if is_leap_year(year) { 366.0 } else { 365.0 }
}

/// A simple date type (calendar date only, no time component).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

impl Date {
    pub fn new(year: i32, month: u32, day: u32) -> Self {
        Self { year, month, day }
    }

    /// Days since a fixed epoch (0000-03-01 proleptic Gregorian) – used for
    /// computing day differences without pulling in a date library.
    fn days_since_epoch(self) -> i64 {
        let y = self.year as i64;
        let m = self.month as i64;
        let d = self.day as i64;
        // Shift months so March = 1, to simplify leap-year handling
        let (y, m) = if m <= 2 { (y - 1, m + 9) } else { (y, m - 3) };
        let era = y.div_euclid(400);
        let yoe = y.rem_euclid(400); // year of era [0, 399]
        let doy = (153 * m + 2) / 5 + d - 1; // day of year [0, 365]
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // day of era
        era * 146097 + doe
    }

    /// Compute the number of calendar days between two dates (self is start).
    pub fn days_until(self, other: Date) -> i64 {
        other.days_since_epoch() - self.days_since_epoch()
    }

    /// Add a given number of days to this date.
    pub fn add_days(self, days: i64) -> Date {
        // Leverage epoch arithmetic
        let epoch = self.days_since_epoch() + days;
        Date::from_epoch(epoch)
    }

    fn from_epoch(z: i64) -> Date {
        let z = z + 719468; // shift to civil epoch
        let era = z.div_euclid(146097);
        let doe = z.rem_euclid(146097);
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let (y, m) = if mp < 10 { (y, mp + 3) } else { (y + 1, mp - 9) };
        Date::new(y as i32, m as u32, d as u32)
    }

    /// Last day of the month for this date.
    pub fn days_in_month(self) -> u32 {
        match self.month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => if is_leap_year(self.year) { 29 } else { 28 },
            _ => panic!("invalid month {}", self.month),
        }
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl DayCountConvention {
    fn actual_actual(self, start: Date, end: Date) -> f64 {
        let days = start.days_until(end);
        let y1 = start.year;
        let y2 = end.year;

        if y1 == y2 {
            return days as f64 / days_in_year(y1);
        }

        // Multi-year: accumulate fractional years
        let mut total = 0.0_f64;
        let mut current = start;

        while current.year < y2 {
            let year_end = Date::new(current.year, 12, 31);
            let days_to_end = current.days_until(year_end) + 1;
            total += days_to_end as f64 / days_in_year(current.year);
            current = Date::new(current.year + 1, 1, 1);
        }

        let days_in_final = current.days_until(end);
        total += days_in_final as f64 / days_in_year(y2);
        total
    }

    /// Compute the year fraction between two dates under this convention.
    pub fn year_fraction(self, start: Date, end: Date) -> f64 {
        let days = start.days_until(end) as f64;

        match self {
            Self::Actual360 => days / 360.0,
            Self::Actual365 => days / 365.0,
            Self::ActualActual => self.actual_actual(start, end),

            Self::Thirty360 => {
                let (mut d1, m1, y1) = (start.day as i32, start.month as i32, start.year);
                let (mut d2, m2, y2) = (end.day as i32, end.month as i32, end.year);
                // 30/360 US (Bond Basis)
                if d1 == 31 { d1 = 30; }
                if d2 == 31 && d1 >= 30 { d2 = 30; }
                // TODO: Add Feb end-of-month adjustment for 30/360 US
                let days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1);
                days as f64 / 360.0
            }

            Self::Thirty360European => {
                let (mut d1, m1, y1) = (start.day as i32, start.month as i32, start.year);
                let (mut d2, m2, y2) = (end.day as i32, end.month as i32, end.year);
                if d1 == 31 { d1 = 30; }
                if d2 == 31 { d2 = 30; }
                let days = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1);
                days as f64 / 360.0
            }
        }
    }
}

// ─── Interpolation Method ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    Linear,
    LogLinear,
    CubicSpline,
    FlatForward,
}

impl fmt::Display for InterpolationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Linear => "linear",
            Self::LogLinear => "log_linear",
            Self::CubicSpline => "cubic_spline",
            Self::FlatForward => "flat_forward",
        };
        write!(f, "{s}")
    }
}

// ─── Term Structure ──────────────────────────────────────────────────────────

/// An interest rate term structure.
///
/// Stores a grid of (date, discount-factor) pairs and derives zero rates and
/// an interpolator for off-grid queries.
#[derive(Debug, Clone)]
pub struct TermStructure {
    pub dates: Vec<Date>,
    pub discount_factors: Vec<f64>,
    pub day_count_convention: DayCountConvention,
    pub interpolation_method: InterpolationMethod,
    pub asof_date: Date,

    // Derived / cached fields
    pub year_fractions: Vec<f64>,
    pub zero_rates: Vec<f64>,

    // For cubic-spline: precomputed second derivatives
    spline_m: Vec<f64>,
}

impl TermStructure {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a new term structure from dates and discount factors.
    pub fn new(
        dates: Vec<Date>,
        discount_factors: Vec<f64>,
        day_count_convention: DayCountConvention,
        interpolation_method: InterpolationMethod,
        asof_date: Date,
    ) -> Result<Self, String> {
        // Validation
        if dates.len() != discount_factors.len() {
            return Err("dates and discount_factors must have the same length".into());
        }
        if dates.len() < 2 {
            return Err("Need at least 2 points to define a term structure".into());
        }
        if !dates.windows(2).all(|w| w[0] < w[1]) {
            return Err("dates must be strictly increasing".into());
        }
        if !discount_factors.iter().all(|&df| df > 0.0) {
            return Err("Discount factors must be positive".into());
        }
        if !discount_factors.iter().all(|&df| df <= 1.0) {
            return Err("Discount factors must be <= 1.0".into());
        }

        let year_fractions: Vec<f64> = dates
            .iter()
            .map(|&d| day_count_convention.year_fraction(asof_date, d))
            .collect();

        let zero_rates: Vec<f64> = discount_factors
            .iter()
            .zip(year_fractions.iter())
            .map(|(&df, &t)| {
                if t > 0.0 && df > 0.0 { -df.ln() / t } else { 0.0 }
            })
            .collect();

        let spline_m = if interpolation_method == InterpolationMethod::CubicSpline {
            compute_natural_spline(&year_fractions, &discount_factors)
        } else {
            vec![]
        };

        Ok(Self {
            dates,
            discount_factors,
            day_count_convention,
            interpolation_method,
            asof_date,
            year_fractions,
            zero_rates,
            spline_m,
        })
    }

    /// Build a flat (constant-rate) term structure.
    pub fn flat_curve(
        rate: f64,
        asof_date: Date,
        day_count_convention: DayCountConvention,
        max_tenor_years: f64,
    ) -> Result<Self, String> {
        let tenors = [0.001, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, max_tenor_years];
        let dates: Vec<Date> = tenors
            .iter()
            .map(|&t| asof_date.add_days((t * 365.25) as i64))
            .collect();
        let discount_factors: Vec<f64> = tenors.iter().map(|&t| (-rate * t).exp()).collect();

        Self::new(
            dates,
            discount_factors,
            day_count_convention,
            InterpolationMethod::LogLinear,
            asof_date,
        )
    }

    /// Build a term structure from continuously-compounded zero rates.
    pub fn from_zero_rates(
        dates: Vec<Date>,
        zero_rates: Vec<f64>,
        day_count_convention: DayCountConvention,
        asof_date: Date,
    ) -> Result<Self, String> {
        if dates.len() != zero_rates.len() {
            return Err("dates and zero_rates must have the same length".into());
        }
        let discount_factors: Vec<f64> = dates
            .iter()
            .zip(zero_rates.iter())
            .map(|(&d, &r)| {
                let t = day_count_convention.year_fraction(asof_date, d);
                (-r * t).exp()
            })
            .collect();

        Self::new(dates, discount_factors, day_count_convention, InterpolationMethod::LogLinear, asof_date)
    }

    // ── Core query methods ────────────────────────────────────────────────────

    /// Discount factor for a given maturity date.
    pub fn discount_factor(&self, maturity_date: Date) -> f64 {
        let t = self.day_count_convention.year_fraction(self.asof_date, maturity_date);
        self.discount_factor_at_time(t)
    }

    /// Discount factor at time `t` (in years).
    pub fn discount_factor_at_time(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 1.0;
        }
        match self.interpolation_method {
            InterpolationMethod::Linear => {
                interp_linear(&self.year_fractions, &self.discount_factors, t)
            }
            InterpolationMethod::LogLinear => {
                let log_dfs: Vec<f64> = self.discount_factors.iter().map(|df| df.ln()).collect();
                interp_linear(&self.year_fractions, &log_dfs, t).exp()
            }
            InterpolationMethod::CubicSpline => {
                interp_cubic(&self.year_fractions, &self.discount_factors, &self.spline_m, t)
            }
            InterpolationMethod::FlatForward => {
                flat_forward_df(&self.year_fractions, &self.discount_factors, t)
            }
        }
    }

    /// Zero-coupon rate (continuously compounded) for a given maturity date.
    pub fn zero_rate(&self, maturity_date: Date) -> f64 {
        let t = self.day_count_convention.year_fraction(self.asof_date, maturity_date);
        self.zero_rate_at_time(t)
    }

    /// Zero-coupon rate at time `t` (in years).
    pub fn zero_rate_at_time(&self, t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        let df = self.discount_factor_at_time(t);
        if df > 0.0 { -df.ln() / t } else { 0.0 }
    }

    /// Continuously-compounded forward rate between two dates.
    pub fn forward_rate(&self, start_date: Date, end_date: Date) -> Result<f64, String> {
        let t1 = self.day_count_convention.year_fraction(self.asof_date, start_date);
        let t2 = self.day_count_convention.year_fraction(self.asof_date, end_date);
        self.forward_rate_at_time(t1, t2)
    }

    /// Continuously-compounded forward rate between two times.
    pub fn forward_rate_at_time(&self, t1: f64, t2: f64) -> Result<f64, String> {
        if t2 <= t1 {
            return Err("t2 must be greater than t1".into());
        }
        let df1 = self.discount_factor_at_time(t1);
        let df2 = self.discount_factor_at_time(t2);
        if df1 > 0.0 && df2 > 0.0 {
            Ok(-(df2 / df1).ln() / (t2 - t1))
        } else {
            Ok(0.0)
        }
    }

    // ── Display helpers ───────────────────────────────────────────────────────

    pub fn summary(&self) -> String {
        let mut lines = vec![
            "Term Structure Summary".to_string(),
            "=".repeat(70),
            format!("Reference Date:       {}", self.asof_date),
            format!("Day Count Convention: {}", self.day_count_convention),
            format!("Interpolation Method: {}", self.interpolation_method),
            format!("Number of Points:     {}", self.dates.len()),
            String::new(),
            format!(
                "{:<12} {:<12} {:<18} {:<12}",
                "Date", "Year Frac", "Discount Factor", "Zero Rate"
            ),
            "-".repeat(70),
        ];

        for (((date, &t), &df), &rate) in self
            .dates.iter()
            .zip(self.year_fractions.iter())
            .zip(self.discount_factors.iter())
            .zip(self.zero_rates.iter())
        {
            lines.push(format!(
                "{:<12} {:<12.6} {:<18.6} {:.4}%",
                date.to_string(),
                t,
                df,
                rate * 100.0
            ));
        }

        lines.join("\n")
    }
}

impl fmt::Display for TermStructure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TermStructure(asof_date={}, points={}, day_count={}, interpolation={})",
            self.asof_date,
            self.dates.len(),
            self.day_count_convention,
            self.interpolation_method,
        )
    }
}

// ─── Interpolation helpers ────────────────────────────────────────────────────

/// Find the index `i` such that xs[i] <= x < xs[i+1].
/// Clamps to valid range; extrapolates flat beyond endpoints.
fn search_sorted(xs: &[f64], x: f64) -> usize {
    match xs.binary_search_by(|probe| probe.partial_cmp(&x).unwrap()) {
        Ok(i) => i.min(xs.len() - 2),
        Err(i) => {
            if i == 0 { 0 }
            else if i >= xs.len() { xs.len() - 2 }
            else { i - 1 }
        }
    }
}

/// Piecewise-linear interpolation (with linear extrapolation).
fn interp_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] {
        // Linear extrapolation on the left
        let slope = (ys[1] - ys[0]) / (xs[1] - xs[0]);
        return ys[0] + slope * (x - xs[0]);
    }
    if x >= xs[n - 1] {
        // Linear extrapolation on the right
        let slope = (ys[n - 1] - ys[n - 2]) / (xs[n - 1] - xs[n - 2]);
        return ys[n - 1] + slope * (x - xs[n - 1]);
    }
    let i = search_sorted(xs, x);
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] * (1.0 - t) + ys[i + 1] * t
}

/// Compute second derivatives for a natural cubic spline (tridiagonal solve).
fn compute_natural_spline(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut m = vec![0.0_f64; n]; // second derivatives (moments)
    if n < 3 {
        return m;
    }

    // Thomas algorithm for tridiagonal system
    let mut h: Vec<f64> = (0..n - 1).map(|i| xs[i + 1] - xs[i]).collect();
    let mut alpha: Vec<f64> = vec![0.0; n];
    for i in 1..n - 1 {
        alpha[i] = 3.0 / h[i] * (ys[i + 1] - ys[i]) - 3.0 / h[i - 1] * (ys[i] - ys[i - 1]);
    }

    let mut c = vec![0.0_f64; n];
    let mut d = vec![0.0_f64; n];
    let mut l = vec![1.0_f64; n];
    let mut mu = vec![0.0_f64; n];
    let mut z = vec![0.0_f64; n];

    for i in 1..n - 1 {
        l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    for j in (0..n - 1).rev() {
        c[j] = z[j] - mu[j] * c[j + 1];
    }
    // c contains the second derivatives / 2; return 2*c = full second derivatives
    c.iter().map(|&v| 2.0 * v).collect()
}

/// Evaluate a natural cubic spline at `x` given precomputed second derivatives `m`.
fn interp_cubic(xs: &[f64], ys: &[f64], m: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if x <= xs[0] { return ys[0]; }
    if x >= xs[n - 1] { return ys[n - 1]; }

    let i = search_sorted(xs, x);
    let h = xs[i + 1] - xs[i];
    let t = (x - xs[i]) / h;
    let a = ys[i];
    let b = (ys[i + 1] - ys[i]) / h - h * (2.0 * m[i] + m[i + 1]) / 6.0;
    let c = m[i] / 2.0;
    let d = (m[i + 1] - m[i]) / (6.0 * h);
    let dt = x - xs[i];
    a + b * dt + c * dt * dt + d * dt * dt * dt
}

/// Flat-forward discount-factor interpolation.
fn flat_forward_df(xs: &[f64], dfs: &[f64], t: f64) -> f64 {
    let n = xs.len();
    if t <= xs[0] { return dfs[0]; }

    for i in 0..n - 1 {
        if xs[i] <= t && t <= xs[i + 1] {
            let (t1, t2) = (xs[i], xs[i + 1]);
            let (df1, df2) = (dfs[i], dfs[i + 1]);
            if t2 > t1 {
                let fwd_rate = -(df2 / df1).ln() / (t2 - t1);
                return df1 * (-fwd_rate * (t - t1)).exp();
            } else {
                return df1;
            }
        }
    }

    // Beyond the last point: extrapolate with the last forward rate
    if n >= 2 {
        let (t1, t2) = (xs[n - 2], xs[n - 1]);
        let (df1, df2) = (dfs[n - 2], dfs[n - 1]);
        if t2 > t1 {
            let fwd_rate = -(df2 / df1).ln() / (t2 - t1);
            return dfs[n - 1] * (-fwd_rate * (t - t2)).exp();
        }
    }
    dfs[n - 1]
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ts() -> TermStructure {
        let asof = Date::new(2024, 1, 1);
        TermStructure::flat_curve(0.05, asof, DayCountConvention::Actual365, 30.0).unwrap()
    }

    #[test]
    fn test_date_arithmetic() {
        let d = Date::new(2024, 1, 1);
        let d2 = d.add_days(365);
        assert_eq!(d2, Date::new(2025, 1, 1));
    }

    #[test]
    fn test_flat_curve_discount_factor() {
        let ts = sample_ts();
        let d = ts.asof_date.add_days(365);
        let df = ts.discount_factor(d);
        // For a flat 5% curve over 1 year: df ≈ exp(-0.05)
        let expected = (-0.05_f64).exp();
        assert!((df - expected).abs() < 1e-3, "df={df}, expected={expected}");
    }

    #[test]
    fn test_zero_rate_roundtrip() {
        let ts = sample_ts();
        let d = ts.asof_date.add_days(730); // 2 years
        let r = ts.zero_rate(d);
        assert!((r - 0.05).abs() < 1e-3, "zero rate={r}");
    }

    #[test]
    fn test_forward_rate() {
        let ts = sample_ts();
        let t1 = ts.asof_date.add_days(365);
        let t2 = ts.asof_date.add_days(730);
        let fwd = ts.forward_rate(t1, t2).unwrap();
        // On a flat curve the forward rate equals the spot rate
        assert!((fwd - 0.05).abs() < 1e-3, "fwd={fwd}");
    }

    #[test]
    fn test_from_zero_rates() {
        let asof = Date::new(2024, 1, 1);
        let dates = vec![
            asof.add_days(365),
            asof.add_days(730),
            asof.add_days(1825),
        ];
        let rates = vec![0.04, 0.045, 0.05];
        let ts = TermStructure::from_zero_rates(
            dates, rates, DayCountConvention::Actual365, asof,
        ).unwrap();
        assert!((ts.zero_rates[0] - 0.04).abs() < 1e-6);
    }

    #[test]
    fn test_thirty_360() {
        let d1 = Date::new(2020, 1, 31);
        let d2 = Date::new(2020, 7, 31);
        let yf = DayCountConvention::Thirty360.year_fraction(d1, d2);
        assert!((yf - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_discount_factor_at_zero() {
        let ts = sample_ts();
        assert_eq!(ts.discount_factor_at_time(0.0), 1.0);
    }

    #[test]
    fn test_validation_errors() {
        let asof = Date::new(2024, 1, 1);
        let d1 = asof.add_days(365);
        let d2 = asof.add_days(730);
        // Mismatched lengths
        assert!(TermStructure::new(
            vec![d1, d2], vec![0.95], DayCountConvention::Actual365,
            InterpolationMethod::LogLinear, asof,
        ).is_err());
        // Discount factor > 1
        assert!(TermStructure::new(
            vec![d1, d2], vec![1.1, 0.9], DayCountConvention::Actual365,
            InterpolationMethod::LogLinear, asof,
        ).is_err());
    }
}
