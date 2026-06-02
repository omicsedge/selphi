//! I/O: VCF/BCF reading, streaming output pipeline, native BCF encoder.

pub mod dosage_stats;
pub mod pipeline;
pub mod bcf_encode;
pub mod bcf_batch;
pub mod bcf_merge;
pub mod vcf_batch;
pub mod vcf_merge;
pub mod sd_batch;
pub mod sd_merge;
pub mod pgen_batch;
pub mod pgen_merge;
pub mod parquet_batch;
pub mod parquet_merge;
pub mod parquet_output;
pub mod pgen_output;
pub mod selfdecode_output;
pub mod target_io;

pub mod indexing;
