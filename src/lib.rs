//! Selphi: SELfDecode PHasing & Imputation

// Suppress architectural/stylistic clippy warnings that are intentional
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::missing_safety_doc)]

// Logging (must be first for macro exports)
pub mod log;

// Central config registry + typed env accessors (single point of env-var access)
pub mod config;

// Common data structures
pub mod common;
pub mod contig;
pub mod srp;
pub mod genmap;

// Haploid phasing engine
pub mod haploid;

// Diploid phasing engine
pub mod diploid;

// Imputation engine
pub mod imputation;

// Low-coverage WGS imputation engine (GLIMPSE2-style GL-aware Li-Stephens)
pub mod lcwgs;

// Reference-faithful lcWGS engine (validation/ablation, --ls-exact); reimplements
// the GLIMPSE2 model — see the repo NOTICE for attribution.
pub mod sparse_ls;

// I/O and output
pub mod io;

// Evaluation
pub mod eval;
