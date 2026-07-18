use serde::{Deserialize, Serialize};
use crate::core::curves::CurveInput;
use crate::core::vols::VolInput;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "product_type", rename_all = "snake_case")]
pub enum ProductData {
    Option(EquityOptionData),
    Future(EquityFutureData),
    Forward(EquityForwardData),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityInstrumentBase {
    pub symbol: String,
    pub currency: Option<String>,
    pub exchange: Option<String>,
    pub name: Option<String>,
    pub cusip: Option<String>,
    pub isin: Option<String>,
    pub underlying_price: f64,
    pub long_short: Option<i32>,
    pub risk_free_rate: Option<f64>,
    pub settlement_type: Option<String>,
}


#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityFutureData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub current_price: Option<f64>,
    pub multiplier:Option<f64>,
    pub entry_price:Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,

}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityForwardData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub current_price: Option<f64>,
    pub notional: Option<f64>,
    pub entry_price:Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,

}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EquityOptionData {
    #[serde(flatten)]
    pub base: EquityInstrumentBase,
    pub put_or_call: String, // "Call"/"Put"
    pub payoff_type: String, // Vanilla/Barrier/Binary
    /// Binary settlement: "cash" (default) or "asset".
    pub binary_type: Option<String>,
    /// Amount paid by a cash-or-nothing binary (default 1.0).
    pub cash_amount: Option<f64>,
    /// Barrier variant: "up_in" | "up_out" | "down_in" | "down_out".
    pub barrier_type: Option<String>,
    pub barrier_level: Option<f64>,
    /// Asian averaging: "arithmetic" (default) | "geometric".
    pub averaging_type: Option<String>,
    /// Asian strike: "fixed" (default, average price) | "floating" (average strike).
    pub asian_strike_type: Option<String>,
    pub strike_price: f64,
    /// Constant volatility; the simple alternative to `vol_surface`.
    pub volatility: Option<f64>,
    pub maturity: String,
    pub dividend: Option<f64>,
    pub current_price: Option<f64>,
    pub multiplier:Option<f64>,
    pub entry_price:Option<f64>,
    /// Monte Carlo path count (engine "MC" only).
    pub simulation: Option<u64>,
    /// MC time steps: 1 = terminal simulation; > 1 = path-wise stepping.
    pub mc_time_steps: Option<usize>,
    /// "exact" (default) | "euler" | "milstein"
    pub mc_scheme: Option<String>,
    /// "sobol" (default, low-discrepancy) | "pseudo" (seeded PCG64)
    pub mc_sampler: Option<String>,
    pub mc_seed: Option<u64>,
    /// "gbm" (default, constant vol) | "local_vol" (Dupire from the
    /// option's vol surface). Applies to the MonteCarlo and
    /// FiniteDifference engines.
    pub mc_model: Option<String>,
    /// Finite difference grid nodes in spot (default 400).
    pub fd_spot_steps: Option<usize>,
    /// Finite difference time steps (default 400).
    pub fd_time_steps: Option<usize>,
    /// Heston parameters; required when `mc_model` is "heston".
    pub heston: Option<crate::equity::heston::HestonParams>,
    pub exercise_style: Option<String>, //European, American,
    pub pricer:Option<String>,
    /// Optional discount curve; when absent a flat curve is built from
    /// `risk_free_rate` (which stays the simple way to specify a rate).
    pub discount_curve: Option<CurveInput>,
    /// Optional volatility surface; when absent a flat surface is built
    /// from `volatility`. One of the two must be provided.
    pub vol_surface: Option<VolInput>,
}

