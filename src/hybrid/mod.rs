//! Hybrid securities: instruments that are a bond, an equity claim and
//! a credit exposure at once.
//!
//! The convertible bond ([`ConvertibleBond`]) and the convertible
//! preferred ([`ConvertiblePreferred`]) are the [`ConvertibleInstrument`]s
//! priced through [`ConvertiblePricing`]: two credit models
//! ([`ConvertibleMarket`] for Tsiveriotis-Fernandes, [`JumpToDefaultMarket`]
//! for jump to default) on a CRR tree or a Crank-Nicolson grid, with
//! pluggable volatility, the contractual extras (calls and puts,
//! contingent conversion, make-wholes, mandatory conversion, cash
//! dividends and their protection), greeks and implied solves. The
//! module sits above [`bonds`](crate::bonds) (the fixed-rate chassis),
//! [`equity`](crate::equity) (implied and local volatility) and
//! [`credit`](crate::credit) (hazard curves); nothing below depends on
//! it.

pub mod convertible;
pub mod preferred;

pub use convertible::{
    dejump_implied_vol, dejump_surface, CashDividend, ContingentConversion, ConvertibleBond,
    ConvertibleFdGreeks, ConvertibleFdGrid, ConvertibleFdValuation, ConvertibleInstrument,
    ConvertibleMarket, ConvertiblePricing, CouponMakeWhole, CreditModel, DividendProtection,
    EquityInputs, EventGrid, FdVolModel, FundamentalChangeMakeWhole, JumpToDefaultMarket,
    MandatoryConversion, NodeValue, Split, DEFAULT_TREE_STEPS,
};
pub use preferred::{ConvertiblePreferred, PERPETUAL_HORIZON_YEARS};
