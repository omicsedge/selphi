//! Imputation accuracy evaluation: stream-merge imputed + truth, compute R²/concordance.
//!
//! Native Rust replacement for bench/evaluate_accuracy.py.
//! Reads VCF.gz/BCF directly, parallel by genomic region, O(1) memory per variant.

pub mod accuracy;
