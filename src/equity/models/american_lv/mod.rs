//! Local volatility from American options: the penalized variational
//! inequality as a forward map, its discrete adjoint, and the Tikhonov
//! calibration that turns a chain of American quotes into an
//! arbitrage-free *synthetic European surface*.
//!
//! The module is the numerical companion of the paper "American implied
//! volatility is not European implied volatility". It is self-contained
//! numerics: no `EquityOption`, `Market` or `VolSurface` types enter, and
//! nothing here changes the behaviour of any other module.
//!
//! ```text
//! Model:    dS/S = (r(t) - q(t)) dt + Sigma(S, t) dW,  cash dividends S -> S - delta_j at t_j
//! Backward: min{ -u_t - L u, u - psi } = 0,  L u = 1/2 Sigma^2 S^2 u_SS + (r - q) S u_S - r u
//! Penalty:  -u_t - L u = rho (psi - u)^+                       (Forsyth & Vetzal 2002)
//! Adjoint:  lambda_t - L^* lambda + rho 1{u < psi} lambda = 0,  lambda(., 0) = delta_{S0}
//! Gradient: dF/dSigma(S, t) = Sigma S^2 u_SS lambda                (the American kernel)
//! ```
//!
//! Layout (one concern per file; every file states the discretization
//! conventions it relies on, because the adjoint must be the exact
//! transpose of the forward march):
//!
//! - [`vol_field`]: the `VolField` trait — how a solver reads
//!   `Sigma(x = ln S, t)` — with flat, callback and node-field
//!   implementations.
//! - [`grid`]: the log-spot mesh shared by every quote of a capture and
//!   the per-expiry time grid (ex-dividend nodes, Rannacher pattern).
//! - [`solver`]: the penalized American / European backward solver, with
//!   optional retention of the whole grid for the adjoint.
//! - [`adjoint`]: the discrete adjoint recursion, the gradient field
//!   `dF/dSigma` on the mesh, the drift sensitivity, and the two kernels
//!   at a flat volatility.
//! - [`bspline`]: the tensor-product B-spline local-volatility surface in
//!   forward log-moneyness and time, its basis cache, projection of a
//!   gradient field onto the basis, and the exact anisotropic `H^1`
//!   regularizer matrix.
//! - [`european_surface`]: the Dupire forward equation for the dense
//!   European surface, implied-vol inversion with effective flat rates,
//!   and static-arbitrage scans.
//! - [`american_iv`]: the American implied volatility, the vega ratio of
//!   the engine-independence proposition, and the industry fixed-point
//!   iteration.
//! - [`calibration`]: the Tikhonov least-squares problem (two-sided and
//!   one-sided quotes, joint carry), the Levenberg–Marquardt driver with
//!   alpha continuation and the discrepancy principle, the linearized
//!   first Gauss–Newton step, and the European variant used by the
//!   pointwise-identification benchmark.
//!
//! The existing [`crate::equity::local_vol`] (Dupire local volatility
//! *from an implied surface*) is untouched; the surface type here,
//! [`bspline::BSplineLocalVol`], is a calibration unknown, not a
//! transformation of an implied surface.

pub mod adjoint;
pub mod american_iv;
pub mod bspline;
pub mod calibration;
pub mod european_surface;
pub mod grid;
pub mod solver;
pub mod vol_field;

pub use adjoint::{
    adjoint, adjoint_into, adjoint_stream, kernels_flat, vega_from_adjoint, Adjoint,
    AdjointWorkspace, Kernels,
};
pub use american_iv::{
    american_implied_vol, dea_fixed_point, vega_ratio, vega_ratio_bump, AmericanIv, DeaFixedPoint,
    IvOutcome, IvWorkspace, VegaMethod, VegaRatio,
};
pub use bspline::{BSplineLocalVol, BasisCache};
pub use calibration::{
    american_prices, calibrate_american, calibrate_european, european_implied_vols,
    european_prices, gauss_newton_step, identification_quotes, jacobian_rows, linearized_step,
    model_prices, scaled_residuals, CalibrationConfig, CalibrationFlags, CalibrationResult,
    CaptureMeshes, JacobianSnapshot, Linearized, PathKind, PathPoint, Quote, RateInput,
    SupportGrid, SupportMap,
};
pub use european_surface::{
    arbitrage_scan, arbitrage_scan_quotes, dupire_forward, european_prices_backward, implied_vols,
    ArbReport, DenseSurface, DupireConfig, ForwardMarket, IvError, TermStructure,
};
pub use grid::{effective_rates, half_width, n_t_default, MarketSlice, Mesh};
pub use solver::{
    price_frozen, price_only, solve_backward, solve_into, Mode, QuoteSpec, Solution, Workspace,
};
pub use vol_field::{CallbackVol, FlatVol, NodeField, VolField};
