//! I/O: VCF/BCF reading, streaming output pipeline, native BCF encoder.

pub mod pipeline;
pub mod bcf_encode;
pub mod parquet_output;
pub mod pgen_output;
pub mod selfdecode_output;
pub mod target_io;

pub mod indexing;
