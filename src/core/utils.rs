//mod dis{
use libm::erf;
use std::f64::consts::{PI, SQRT_2};
use serde::{Deserialize, Serialize};
use crate::core::data_models::ProductData;

#[derive(PartialEq,Clone,Debug)]
pub enum ContractStyle {
    European,
    American,
}

#[derive(strum_macros::Display)]
pub enum EngineType {
    Analytical,
    MonteCarlo,
    Binomial,
    FiniteDifference,
    FFT,
}
pub trait Engine<I> {
    fn npv(&self, instrument: &I) -> f64;
}

impl EngineType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineType::Analytical => "Analytical",
            EngineType::MonteCarlo => "MonteCarlo",
            EngineType::Binomial => "Binomial",
            EngineType::FiniteDifference => "FiniteDifference",
            EngineType::FFT => "FFT",
        }
    }
}



// #[derive(Clone,Debug,Deserialize,Serialize)]
// pub struct MarketData {
//     pub underlying_price:f64,
//     pub option_type:Option<String>,
//     pub strike_price:Option<f64>,
//     pub volatility:Option<f64>,
//     pub option_price:Option<f64>,
//     pub risk_free_rate:Option<f64>,
//     pub maturity:String,
//     pub dividend: Option<f64>,
//     pub simulation:Option<u64>,
//     pub current_price:Option<f64>,
//     pub notional: Option<f64>,
//     pub long_short:Option<i32>,
//     pub multiplier:Option<f64>,
//     pub entry_price:Option<f64>,
// }


#[derive(Clone,Debug,Deserialize,Serialize)]
pub struct RateData {
    pub instrument: String,
    pub currency: String,
    pub start_date: String,
    pub maturity_date: String,
    pub valuation_date: String,
    pub notional: f64,
    pub fix_rate: f64,
    pub day_count: String,
    pub business_day_adjustment: i8,
}

#[derive(Clone,Debug,Deserialize,Serialize)]
pub struct Contract {
    pub action: String,
    pub asset: String,
    pub product_type: ProductData,
    pub rate_data: Option<RateData>,
}
#[derive(Deserialize,Serialize)]
pub struct CombinedContract{
    pub contract: Contract,
    pub output: ContractOutput
}

#[derive(Debug, Deserialize,Serialize)]
pub struct Contracts {
    pub asset: String,
    pub contracts: Vec<Contract>,
}
#[derive(Debug, Deserialize,Serialize)]
pub struct OutputJson {
    pub contracts: Vec<String>,
}
#[derive(Deserialize,Serialize)]
pub struct ContractOutput {
    pub pv: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    /// Change in delta per unit change in implied volatility.
    pub vanna: f64,
    /// Change in delta per year of calendar time.
    pub charm: f64,
    /// Delta elasticity, `S * gamma / delta`.
    pub gamma_p: f64,
    /// Change in gamma per unit change in implied volatility.
    pub zomma: f64,
    /// Monte Carlo standard error of `pv` (None for deterministic engines).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_err: Option<f64>,
    /// Per-asset deltas for multi-asset (rainbow) products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub deltas: Option<Vec<f64>>,
    /// Per-asset vegas for multi-asset (rainbow) products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vegas: Option<Vec<f64>>,
    pub error: Option<String>
}

/// Probability density function of a standard normal random variable x.
pub fn norm_pdf(x: f64) -> f64 {
    let t = -0.5 * x * x;
    t.exp() / (SQRT_2 * PI.sqrt())
}

/// Cumulative distribution function of a standard normal random variable x.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / SQRT_2))
}

/// Cumulative bivariate normal distribution `P(X <= a, Y <= b)` for
/// standard normals with correlation `rho`.
///
/// Genz (2004) Gauss-Legendre quadrature on the arcsine form for
/// `|rho| <= 0.925` (accurate to ~1e-14); an adaptive-free Simpson
/// integration of `phi(x) N((b - rho x)/sqrt(1-rho^2))` for the highly
/// correlated tail. Used by the Bjerksund-Stensland (2002) two-boundary
/// American approximation.
pub fn bivariate_norm_cdf(a: f64, b: f64, rho: f64) -> f64 {
    assert!((-1.0..=1.0).contains(&rho), "correlation must be in [-1, 1]");
    if rho == 1.0 {
        return norm_cdf(a.min(b));
    }
    if rho == -1.0 {
        return (norm_cdf(a) + norm_cdf(b) - 1.0).max(0.0);
    }
    if rho.abs() <= 0.925 {
        // 10-point Gauss-Legendre on each half interval
        const WEIGHTS: [f64; 10] = [
            0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
            0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
            0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
            0.1527533871307259,
        ];
        const ABSCISSAE: [f64; 10] = [
            0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
            0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
            0.5108670019508271, 0.3737060887154196, 0.2277858511416451,
            0.07652652113349733,
        ];
        let (h, k) = (-a, -b);
        let hs = 0.5 * (h * h + k * k);
        let asr = rho.asin();
        let mut sum = 0.0;
        for (w, x) in WEIGHTS.iter().zip(&ABSCISSAE) {
            for sign in [-1.0, 1.0] {
                let sn = (asr * (sign * x + 1.0) / 2.0).sin();
                sum += w * ((sn * h * k - hs) / (1.0 - sn * sn)).exp();
            }
        }
        sum * asr / (4.0 * PI) + norm_cdf(-h) * norm_cdf(-k)
    } else {
        // high correlation: integrate the conditional CDF directly
        let denom = (1.0 - rho * rho).sqrt();
        let lo = -8.5_f64;
        if a <= lo {
            return 0.0;
        }
        let n_steps = 2000;
        let dx = (a - lo) / n_steps as f64;
        let f = |x: f64| norm_pdf(x) * norm_cdf((b - rho * x) / denom);
        let mut sum = f(lo) + f(a);
        for i in 1..n_steps {
            let x = lo + i as f64 * dx;
            sum += if i % 2 == 1 { 4.0 } else { 2.0 } * f(x);
        }
        sum * dx / 3.0
    }
}

/// Inverse of the standard normal CDF (quantile function).
///
/// Acklam's rational approximation refined with one Halley step against the
/// erf-based [`norm_cdf`], giving close to machine precision. `p` must be in (0, 1);
/// values outside return NaN.
pub fn inv_norm_cdf(p: f64) -> f64 {
    if !(p > 0.0 && p < 1.0) {
        return f64::NAN;
    }
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    const P_LOW: f64 = 0.02425;

    let tail = |q: f64| -> f64 {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    let x = if p < P_LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p <= 1.0 - P_LOW {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    };
    // one Halley refinement step on f(x) = N(x) - p (f' = phi, f'' = -x phi)
    crate::core::solvers::Solver1d::new(0.0, 1)
        .halley(|x| norm_cdf(x) - p, norm_pdf, |x| -x * norm_pdf(x), x)
        .x
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bivariate_normal_identities() {
        // exact identity at the origin: 1/4 + asin(rho)/(2 pi)
        for rho in [-0.9_f64, -0.5, 0.0, 0.3, 0.786, 0.9] {
            let exact = 0.25 + rho.asin() / (2.0 * PI);
            assert!(
                (bivariate_norm_cdf(0.0, 0.0, rho) - exact).abs() < 1e-12,
                "rho {rho}"
            );
        }
        // independence factorizes
        assert!((bivariate_norm_cdf(1.0, -0.5, 0.0) - norm_cdf(1.0) * norm_cdf(-0.5)).abs() < 1e-12);
        // symmetry in the arguments
        assert!(
            (bivariate_norm_cdf(0.7, -0.2, 0.4) - bivariate_norm_cdf(-0.2, 0.7, 0.4)).abs() < 1e-12
        );
        // golden value from the cross-checked reference implementation
        assert!((bivariate_norm_cdf(0.5, -0.3, 0.786) - 0.367657814886).abs() < 1e-9);
    }

    #[test]
    fn bivariate_normal_high_correlation_branch() {
        // rho -> 1: P(X <= a, Y <= b) -> N(min(a, b))
        assert!((bivariate_norm_cdf(0.5, 1.2, 1.0) - norm_cdf(0.5)).abs() < 1e-14);
        // the Simpson branch (|rho| > 0.925) must join the Genz branch
        // smoothly: compare both at rho just inside each side
        let genz = bivariate_norm_cdf(0.4, -0.1, 0.92);
        let simpson = bivariate_norm_cdf(0.4, -0.1, 0.93);
        assert!((genz - simpson).abs() < 5e-3, "{genz} vs {simpson}");
        // and rho = 0.99 stays close to the perfect-correlation limit
        let near = bivariate_norm_cdf(0.5, 1.2, 0.99);
        assert!(near < norm_cdf(0.5) + 1e-9 && near > norm_cdf(0.5) - 0.02, "{near}");
    }
}
