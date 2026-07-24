//! Convenience re-exports of the Monte Carlo machinery, which lives in
//! [`core::montecarlo`](crate::core::montecarlo) as an asset-agnostic
//! toolkit (RNG streams, Sobol/Halton sequences, Brownian bridge, variance
//! reduction, sampling designs, statistics). New code should import from
//! `core::montecarlo` directly.

pub use crate::core::montecarlo::{
    path_normals, path_rng, pseudo_normal_matrix, pseudo_normals, sobol_normals, BrownianBridge,
    QmcSequence,
};
