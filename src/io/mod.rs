//! I/O: VCF/BCF reading, streaming output pipeline, BCF writer.

pub mod pipeline;
pub mod bcf_writer;
pub mod bcf_encode;
pub mod parquet_output;
pub mod pgen_output;
pub mod selfdecode_output;
pub mod vcf_io;
