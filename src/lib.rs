//! Selphi: SELfDecode PHasing & Imputation

// Suppress architectural/stylistic clippy warnings that are intentional
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::type_complexity)]
#![allow(clippy::missing_safety_doc)]

// Logging (must be first for macro exports)
pub mod log;

// Common data structures
pub mod common;
pub mod srp;
pub mod genmap;

// Haploid phasing engine
pub mod haploid;

// Diploid phasing engine
pub mod diploid;

// Imputation engine
pub mod imputation;

// I/O and output
pub mod io;

// Evaluation
pub mod eval;
