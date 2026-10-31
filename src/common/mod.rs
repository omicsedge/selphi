//! Common data structures shared across phasing and imputation engines.

pub mod bitmatrix;
pub mod bit_gather;
pub mod utils;

pub use bitmatrix::HaplotypeBitmatrix;
pub mod mem;
