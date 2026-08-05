use crate::core::trade::PutOrCall;
use crate::core::utils::ContractStyle;
use serde::Deserialize;
use std::error::Error;
use std::fmt::Debug;
use std::str::FromStr;

///Enum for different engines to price options
#[derive(PartialEq, Clone, Debug)]
pub enum Engine {
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

/// The numerical method **with its own settings** — each variant carries
/// exactly the configuration that engine consults, so an option never
/// stores dead config for engines it does not use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PricingEngine {
    /// Closed forms (Black-Scholes / Black-76 / Heston CF).
    BlackScholes,
    MonteCarlo(crate::equity::montecarlo::MonteCarloConfig),
    Binomial(crate::core::lattice::LatticeConfig),
    FiniteDifference(crate::equity::finite_difference::FdConfig),
    BaroneAdesiWhaley,
    BjerksundStensland,
}

impl PricingEngine {
    /// The engine selector without its configuration.
    pub fn kind(&self) -> Engine {
        match self {
            PricingEngine::BlackScholes => Engine::BlackScholes,
            PricingEngine::MonteCarlo(_) => Engine::MonteCarlo,
            PricingEngine::Binomial(_) => Engine::Binomial,
            PricingEngine::FiniteDifference(_) => Engine::FiniteDifference,
            PricingEngine::BaroneAdesiWhaley => Engine::BaroneAdesiWhaley,
            PricingEngine::BjerksundStensland => Engine::BjerksundStensland,
        }
    }

    /// Build from a selector with default per-engine configuration.
    pub fn from_kind(kind: Engine) -> PricingEngine {
        match kind {
            Engine::BlackScholes => PricingEngine::BlackScholes,
            Engine::MonteCarlo => PricingEngine::MonteCarlo(Default::default()),
            Engine::Binomial => PricingEngine::Binomial(Default::default()),
            Engine::FiniteDifference => PricingEngine::FiniteDifference(Default::default()),
            Engine::BaroneAdesiWhaley => PricingEngine::BaroneAdesiWhaley,
            Engine::BjerksundStensland => PricingEngine::BjerksundStensland,
        }
    }
}

/// The dynamics of the underlying — orthogonal to the numerical engine
/// (Monte Carlo and finite difference both consult it). Heston carries
/// its parameters, so "Heston selected but parameters missing" cannot be
/// represented.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum Model {
    /// Black-Scholes dynamics on the option's vol surface.
    #[default]
    Gbm,
    /// Dupire local volatility calibrated from the vol surface.
    LocalVol,
    /// Heston stochastic volatility.
    Heston(crate::equity::heston::HestonParams),
}

impl Model {
    pub fn is_heston(&self) -> bool {
        matches!(self, Model::Heston(_))
    }

    /// The model under a parallel implied-vol shift — the model is a risk
    /// factor owner like a surface or a curve. GBM and local vol read the
    /// (already bumped) surface at pricing time, so they pass through
    /// unchanged; Heston applies the library's vega convention: shift
    /// `sqrt(v0)` and `sqrt(theta)` in parallel
    /// ([`HestonParams::with_vol_shift`](crate::equity::heston::HestonParams::with_vol_shift)),
    /// rather than recalibrating to the bumped surface.
    pub fn with_vol_shift(&self, shift: f64) -> Model {
        match self {
            Model::Heston(params) => Model::Heston(params.with_vol_shift(shift)),
            other => *other,
        }
    }

    /// Parse from contract fields: the `mc_model` string plus the
    /// `heston` parameter block (required when the model is Heston).
    pub fn from_contract(
        mc_model: Option<&str>,
        heston: Option<crate::equity::heston::HestonParams>,
    ) -> Result<Model, crate::core::errors::RustyQLibError> {
        use crate::core::errors::RustyQLibError;
        match mc_model.map(str::trim) {
            None | Some("gbm") | Some("GBM") | Some("Gbm") => Ok(Model::Gbm),
            Some("local_vol") | Some("localvol") | Some("LocalVol") | Some("local") => {
                Ok(Model::LocalVol)
            }
            Some("heston") | Some("Heston") => {
                let params = heston.ok_or_else(|| {
                    RustyQLibError::invalid_input(
                        "heston",
                        "heston parameters are required when mc_model = heston",
                    )
                })?;
                params.validate()?;
                Ok(Model::Heston(params))
            }
            Some(other) => Err(RustyQLibError::invalid_input(
                "mc_model",
                format!("unknown model '{other}' (use gbm, local_vol or heston)"),
            )),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongShort {
    LONG,
    SHORT,
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
    Accumulator,
    Chooser,
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
            "chooser" => Ok(PayoffType::Chooser),
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

    /// Intrinsic value at the given spot (the option's current market
    /// spot at its contract strike).
    fn payoff_amount(&self, spot: f64, strike: f64) -> f64 {
        self.payoff(spot, strike)
    }

    /// Path payoff in AAD arithmetic — the mirror of
    /// [`path_payoff`](Payoff::path_payoff) over tape variables, used by
    /// the adjoint Monte Carlo Greeks
    /// ([`montecarlo::aad_greeks`](crate::equity::montecarlo)). `None`
    /// (the default) opts a payoff out: **discontinuous payoffs (barrier,
    /// binary, autocallable) must stay out**, because the
    /// almost-everywhere derivative of an indicator is zero — their
    /// Greeks come from the bump stencils instead.
    fn path_payoff_var<'t>(
        &self,
        _path: &[crate::core::aad::Var<'t>],
        _strike: f64,
    ) -> Option<crate::core::aad::Var<'t>> {
        None
    }

    fn payoff_kind(&self) -> PayoffType;
    fn put_or_call(&self) -> &PutOrCall;
    fn exercise_style(&self) -> &ContractStyle;

    /// Downcast hook so pricers that need payoff-specific details (e.g. the
    /// analytic pricer distinguishing cash- from asset-or-nothing binaries)
    /// can recover the concrete payoff type.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Clone through the trait object, so instruments holding a
    /// `Box<dyn Payoff>` are cloneable (repricing a contract under another
    /// market clones the instrument). Implementors write
    /// `Box::new(self.clone())`.
    fn clone_box(&self) -> Box<dyn Payoff>;
}

impl Clone for Box<dyn Payoff> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
