//! What you can trade: payoff definitions and product logic. The central
//! instrument type [`EquityOption`](equity_option::EquityOption) lives in
//! [`equity_option`]; each product file owns its payoff struct, and
//! [`vanilla_option`] re-exports the historical flat paths.

pub mod accumulator;
pub mod asian;
pub mod autocallable;
pub mod barrier;
pub mod binary_option;
pub mod chooser;
pub mod cliquet;
pub mod equity_forward;
pub mod equity_future;
pub mod equity_option;
pub mod forward_start_option;
mod from_json;
pub mod lookback;
pub mod perpetual;
pub mod rainbow;
pub mod vanilla_option;
pub mod variance_swap;
pub mod worst_of;
