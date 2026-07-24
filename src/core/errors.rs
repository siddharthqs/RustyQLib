//! The library-wide error type.
//!
//! Every fallible public operation returns [`RustyQLibError`], usually
//! through the [`Result`] alias. Variants are grouped by failure domain so
//! callers can match on the class of failure without parsing message text:
//! bad user input, an engine/product combination the library refuses to
//! price, a calibration that did not converge, a numerical breakdown inside
//! an otherwise valid computation, or malformed contract/market data.

use thiserror::Error;

/// Library-wide result alias.
pub type Result<T> = std::result::Result<T, RustyQLibError>;

/// The error type returned by all fallible RustyQLib operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RustyQLibError {
    /// A user-supplied value is outside its valid domain (negative
    /// volatility, maturity in the past, correlation outside [-1, 1], ...).
    #[error("invalid input `{field}`: {reason}")]
    InvalidInput {
        /// The contract/market-data field or function argument at fault.
        field: String,
        /// Why the value was rejected.
        reason: String,
    },

    /// The requested engine cannot price the requested product/model
    /// combination (e.g. path-dependent payoffs on the binomial engine).
    #[error("unsupported: {0}")]
    UnsupportedEngine(String),

    /// An iterative calibration or root search terminated without meeting
    /// its convergence criterion.
    #[error("calibration failed after {iterations} iterations (residual {residual:.6e}): {reason}")]
    CalibrationFailed {
        /// Iterations performed before giving up.
        iterations: usize,
        /// Final objective / residual value at termination.
        residual: f64,
        /// What was being calibrated and why it stopped.
        reason: String,
    },

    /// A numerical method broke down on otherwise valid input (singular
    /// matrix, bracketing failure, NaN in an intermediate, ...).
    #[error("numerical error: {0}")]
    NumericalError(String),

    /// Contract or market data could not be parsed / deserialized.
    #[error("parse error: {0}")]
    ParseError(String),
}

impl From<crate::core::vols::VolError> for RustyQLibError {
    fn from(e: crate::core::vols::VolError) -> Self {
        RustyQLibError::invalid_input("vol_surface", e.to_string())
    }
}

impl From<crate::core::curves::CurveError> for RustyQLibError {
    fn from(e: crate::core::curves::CurveError) -> Self {
        RustyQLibError::invalid_input("discount_curve", e.to_string())
    }
}

impl RustyQLibError {
    /// Convenience constructor for [`RustyQLibError::InvalidInput`].
    pub fn invalid_input(field: impl Into<String>, reason: impl Into<String>) -> Self {
        RustyQLibError::InvalidInput { field: field.into(), reason: reason.into() }
    }
}
