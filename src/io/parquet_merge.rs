//! Native Parquet sample-merger.
//!
//! Reads N per-batch parquets (each containing AP1/AP2 per sample, see
//! `parquet_batch.rs`), reads matching row groups in parallel, recomputes
//! AC/AF/DR2 from the *full* per-sample AP1/AP2 set, collapses AP1+AP2 → DS
//! for the final parquet schema, and emits a single merged parquet.
//!
//! Final schema (matches `parquet_output::build_schema`):
//!   CHROM, POS, ID, REF, ALT, AF, DR2, IMP, sample_1, sample_2, …
//! where sample_i is f32 DS = ap1+ap2.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::*;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::io::parquet_output::build_schema as build_final_schema;

/// Merge N per-batch Parquet files into a single merged parquet at
/// `output_path` (`.parquet` extension added if missing).
pub fn merge_batch_parquets(
    batch_paths: &[PathBuf],
    output_path: &Path,
    all_sample_names: &[String],
) -> std::io::Result<()> {
    if batch_paths.is_empty() {
        return Err(std::io::Error::other("no batch files to merge"));
    }

    let final_path = if output_path.extension().is_none_or(|e| e != "parquet") {
        output_path.with_extension("parquet")
    } else {
        output_path.to_path_buf()
    };

    let schema = Arc::new(build_final_schema(all_sample_names));
    let file = std::fs::File::create(&final_path)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_size(1024)
        .build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    // Open per-batch readers (RecordBatch iterators).
    let mut readers: Vec<_> = Vec::with_capacity(batch_paths.len());
    for path in batch_paths {
        let f = std::fs::File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)
            .map_err(|e| std::io::Error::other(format!("open {path:?}: {e}")))?;
        let reader = builder.build()
            .map_err(|e| std::io::Error::other(format!("build {path:?}: {e}")))?;
        readers.push(reader);
    }

    let n_haps_total = (all_sample_names.len() * 2) as u32;
    let n_samples_total = all_sample_names.len();

    loop {
        // Pull one record batch from each per-batch reader; all must align.
        let mut batches: Vec<arrow::record_batch::RecordBatch> = Vec::with_capacity(readers.len());
        let mut any_eof = false;
        for r in &mut readers {
            match r.next() {
                Some(Ok(rb)) => batches.push(rb),
                Some(Err(e)) => return Err(std::io::Error::other(format!("read batch: {e}"))),
                None => { any_eof = true; break; }
            }
        }
        if any_eof {
            // Either all readers ended (clean EOF) or some ended early (mismatch).
            if !batches.is_empty() {
                return Err(std::io::Error::other(format!(
                    "row-group count mismatch across batches (read {} of {} before EOF)",
                    batches.len(), readers.len(),
                )));
            }
            break;
        }
        // Verify row counts match across batches.
        let n_rows = batches[0].num_rows();
        for (i, b) in batches.iter().enumerate().skip(1) {
            if b.num_rows() != n_rows {
                return Err(std::io::Error::other(format!(
                    "row-group row count mismatch: batch 0 = {n_rows}, batch {i} = {}",
                    b.num_rows()
                )));
            }
        }
        if n_rows == 0 { continue; }

        // Build merged columns.
        // Cols 0..5 (CHROM, POS, ID, REF, ALT) — copy directly from batch 0.
        let mut cols: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(8 + n_samples_total);
        for col_idx in 0..5 {
            cols.push(batches[0].column(col_idx).clone());
        }
        // AF (Float32, nullable), DR2 (Float32, nullable), IMP (Bool)
        let mut af_b = Float32Builder::with_capacity(n_rows);
        let mut dr2_b = Float32Builder::with_capacity(n_rows);
        let mut imp_b = BooleanBuilder::with_capacity(n_rows);

        // For each row, gather AP1/AP2 across all batches → recompute AC/AF/DR2.
        // Also need DS per sample for final columns. Pre-allocate per-sample
        // f32 builders (one per total sample).
        let mut ds_builders: Vec<Float32Builder> = (0..n_samples_total)
            .map(|_| Float32Builder::with_capacity(n_rows)).collect();

        // Pre-extract IMP from batch 0 column 5.
        let imp_col = batches[0].column(5).as_any().downcast_ref::<BooleanArray>()
            .ok_or_else(|| std::io::Error::other("IMP column type mismatch"))?;

        // Pre-extract per-batch AP1/AP2 arrays (cols 6.. are alternating ap1/ap2).
        let mut batch_ap_arrays: Vec<Vec<&Float32Array>> = Vec::with_capacity(batches.len());
        for b in &batches {
            let n_cols = b.num_columns();
            let mut arrs = Vec::with_capacity(n_cols - 6);
            for c in 6..n_cols {
                let arr = b.column(c).as_any().downcast_ref::<Float32Array>()
                    .ok_or_else(|| std::io::Error::other(format!(
                        "expected Float32 AP column at index {c}"
                    )))?;
                arrs.push(arr);
            }
            batch_ap_arrays.push(arrs);
        }

        // Reusable per-row scratch: flattened (ap1, ap2) in global sample order,
        // plus the helper's per-sample dosage output (also feeds the DS columns).
        let mut flat: Vec<(f32, f32)> = Vec::with_capacity(n_samples_total);
        let mut ds_scratch = vec![0f32; n_samples_total];

        for row in 0..n_rows {
            let is_imp = imp_col.value(row);
            imp_b.append_value(is_imp);

            // Gather AP1/AP2 across all batches for this row in global sample order.
            flat.clear();
            for bi in 0..batches.len() {
                let arrs = &batch_ap_arrays[bi];
                let k_in_batch = arrs.len() / 2;
                for s in 0..k_in_batch {
                    flat.push((arrs[s * 2].value(row), arrs[s * 2 + 1].value(row)));
                }
            }
            // AC + dosage-R² via the shared helper; ds_scratch[s] = ap1+ap2 (the DS column).
            let (ac, dr2_f64) = crate::io::dosage_stats::imputed_ac_dr2(
                n_samples_total, n_haps_total as usize, |s| flat[s], &mut ds_scratch,
            );
            for s in 0..n_samples_total {
                ds_builders[s].append_value(ds_scratch[s]);
            }
            let af = ac as f32 / n_haps_total as f32;
            af_b.append_value(af);
            if is_imp {
                dr2_b.append_value(dr2_f64 as f32);
            } else {
                dr2_b.append_null();
            }
        }

        cols.push(Arc::new(af_b.finish()));
        cols.push(Arc::new(dr2_b.finish()));
        cols.push(Arc::new(imp_b.finish()));
        for b in ds_builders.iter_mut() {
            cols.push(Arc::new(b.finish()));
        }

        // Verify column count matches the final schema.
        if cols.len() != schema.fields().len() {
            return Err(std::io::Error::other(format!(
                "merged column count mismatch: built {} expected {}",
                cols.len(), schema.fields().len()
            )));
        }

        let rb = arrow::record_batch::RecordBatch::try_new(schema.clone(), cols)
            .map_err(|e| std::io::Error::other(format!("RecordBatch::try_new: {e}")))?;
        writer.write(&rb).map_err(|e| std::io::Error::other(format!("write: {e}")))?;
    }

    // Completeness: every batch must be fully consumed. The loop breaks on the
    // first reader to hit EOF while a longer batch may still have trailing
    // row-groups (the in-loop `!batches.is_empty()` check misses the
    // batch-0-shortest case) — silently truncating the merged output.
    for (i, r) in readers.iter_mut().enumerate() {
        if r.next().is_some() {
            return Err(std::io::Error::other(format!(
                "batch {i} has trailing row-groups beyond the merged set — \
                 mismatched/truncated intermediate parquet")));
        }
    }

    writer.close().map_err(|e| std::io::Error::other(format!("close: {e}")))?;
    Ok(())
}
