//! Compatibility shim: the Monte Carlo machinery moved to
//! [`core::montecarlo`](crate::core::montecarlo), where it lives as an
//! asset-agnostic toolkit (RNG streams, Sobol/Halton sequences, Brownian
//! bridge, variance reduction, sampling designs, statistics). These
//! re-exports keep the old import paths working; new code should import
//! from `core::montecarlo` directly.

pub use crate::core::montecarlo::{
    path_normals, path_rng, pseudo_normal_matrix, pseudo_normals, sobol_normals, BrownianBridge,
    QmcSequence,
};
