use serde::Deserialize;
use crate::equity::vanilla_option::EquityOptionBase;
use std::str::FromStr;
use std::error::Error;
use crate::core::trade::{PutOrCall};
use std::fmt::Debug;
use crate::core::utils::ContractStyle;

///Enum for different engines to price options
#[derive(PartialEq,Clone,Debug)]
pub enum Engine{
    BlackScholes,
    MonteCarlo,
    Binomial,
    FiniteDifference,
    /// Barone-Adesi-Whaley quadratic approximation for American vanillas.
    BaroneAdesiWhaley,
    /// Bjerksund-Stensland (2002) two-boundary approximation for
    /// American vanillas — a lower bound, generally tighter than BAW.
    BjerksundStensland,
}
#[derive(Debug)]
pub enum LongShort{
    LONG,
    SHORT
}
#[derive(Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PayoffType {
    Vanilla,
    Binary,
    Barrier,
    Asian,
    ForwardStart,
    Autocallable,
    Lookback,
}
impl FromStr for PayoffType {
    type Err = Box<dyn Error>;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "vanilla" => Ok(PayoffType::Vanilla),
            "binary" => Ok(PayoffType::Binary),
            "barrier" => Ok(PayoffType::Barrier),
            "asian" => Ok(PayoffType::Asian),
            "forward_start" | "forwardstart" => Ok(PayoffType::ForwardStart),
            "autocallable" | "autocall" => Ok(PayoffType::Autocallable),
            "lookback" => Ok(PayoffType::Lookback),
            _ => Err("Invalid payoff type".into()),
        }
    }
}


/// Common interface linking all payoffs (Vanilla, Binary, Barrier, Asian).
///
/// Terminal payoffs implement [`payoff`](Payoff::payoff); path-dependent
/// payoffs (Asian, Barrier) additionally override
/// [`path_payoff`](Payoff::path_payoff), which defaults to evaluating the
/// terminal payoff on the last point of the path. Engines only ever call
/// these two methods, so a new payoff plugs into every engine at once.
pub trait Payoff: Debug + Send + Sync {
    /// Payoff for a given level of the underlying: the terminal spot for
    /// European exercise, or the exercise spot for American.
    fn payoff(&self, spot: f64, strike: f64) -> f64;

    /// Payoff for a full simulated path (used by Monte Carlo). Terminal
    /// payoffs default to the last point; Asian/Barrier override this.
    /// The path excludes the initial spot (it starts at the first step).
    fn path_payoff(&self, path: &[f64], strike: f64) -> f64 {
        self.payoff(*path.last().expect("empty path"), strike)
    }

    /// True when the payoff depends on the whole path (Asian, Barrier), so
    /// engines must simulate paths rather than terminal values.
    fn is_path_dependent(&self) -> bool {
        false
    }

    /// Intrinsic value at the option's current underlying price.
    fn payoff_amount(&self, base: &EquityOptionBase) -> f64 {
        self.payoff(base.underlying_price.value(), base.strike_price)
    }

    fn payoff_kind(&self) -> PayoffType;
    fn put_or_call(&self) -> &PutOrCall;
    fn exercise_style(&self)->&ContractStyle;

    /// Downcast hook so pricers that need payoff-specific details (e.g. the
    /// analytic pricer distinguishing cash- from asset-or-nothing binaries)
    /// can recover the concrete payoff type.
    fn as_any(&self) -> &dyn std::any::Any;
}