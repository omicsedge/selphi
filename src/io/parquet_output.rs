//! Native Parquet output for imputation dosages.
//!
//! Schema: variant-major (rows = variants, columns = fixed fields + per-sample DS).
//!   CHROM (Utf8), POS (Int64), ID (Utf8), REF (Utf8), ALT (Utf8),
//!   AF (Float32), DR2 (Float32), IMP (Boolean),
//!   sample_1 (Float32), sample_2 (Float32), ...
//!
//! Writes row groups in tiles (~4000 variants each) for streaming.
//! Uses zstd compression.

use std::path::Path;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

/// Build the Arrow schema for imputation output.
pub fn build_schema(sample_names: &[String]) -> Schema {
    let mut fields = vec![
        Field::new("CHROM", DataType::Utf8, false),
        Field::new("POS", DataType::Int64, false),
        Field::new("ID", DataType::Utf8, true),
        Field::new("REF", DataType::Utf8, false),
        Field::new("ALT", DataType::Utf8, false),
        Field::new("AF", DataType::Float32, true),
        Field::new("DR2", DataType::Float32, true),
        Field::new("IMP", DataType::Boolean, false),
    ];
    for name in sample_names {
        fields.push(Field::new(name, DataType::Float32, true));
    }
    Schema::new(fields)
}

/// Setup Parquet writer with zstd compression.
pub fn setup_parquet_writer(
    output_path: &Path,
    sample_names: &[String],
) -> std::io::Result<(ArrowWriter<std::fs::File>, Arc<Schema>)> {
    let parquet_path = if output_path.extension().is_none_or(|e| e != "parquet") {
        output_path.with_extension("parquet")
    } else {
        output_path.to_path_buf()
    };

    let schema = Arc::new(build_schema(sample_names));
    let file = std::fs::File::create(&parquet_path)?;

    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .set_max_row_group_size(8192)
        .build();

    let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    Ok((writer, schema))
}

/// Write a tile of imputed + chip variants as a Parquet row group.
///
/// Uses a flat variant-major DS buffer (n_variants × n_samples) to minimize
/// allocation overhead. Arrow arrays are created from contiguous column slices.
pub fn write_tile_to_parquet(
    writer: &mut ArrowWriter<std::fs::File>,
    schema: &Arc<Schema>,
    alt_probs: &[f32],
    tile_n: usize,
    n_samples: usize,
    n_haps: usize,
    global_start: usize,
    vid_prefixes: &[Vec<u8>],
    vp_offset: usize,
    is_chip: &[bool],
    chip_local_idx: &[usize],
    chip_genotypes: &[u8],
    n_ref_variants: usize,
) -> std::io::Result<()> {
    if tile_n == 0 { return Ok(()); }

    let mut chroms = StringBuilder::with_capacity(tile_n, tile_n * 4);
    let mut positions = Int64Builder::with_capacity(tile_n);
    let mut ids = StringBuilder::with_capacity(tile_n, tile_n * 16);
    let mut refs_b = StringBuilder::with_capacity(tile_n, tile_n * 2);
    let mut alts_b = StringBuilder::with_capacity(tile_n, tile_n * 2);
    let mut afs = Float32Builder::with_capacity(tile_n);
    let mut dr2s = Float32Builder::with_capacity(tile_n);
    let mut imps = BooleanBuilder::with_capacity(tile_n);

    // Flat DS buffer: variant-major (row = variant, col = sample)
    // Layout: ds_flat[v * n_samples + s]
    let mut ds_flat = vec![0.0f32; tile_n * n_samples];
    let mut n_rows = 0usize;

    for v in 0..tile_n {
        let wgs_i = global_start + v;
        if wgs_i >= n_ref_variants { break; }
        let local_i = vp_offset + v;

        let prefix = &vid_prefixes[local_i];
        let prefix_str = std::str::from_utf8(prefix).unwrap_or("");
        let parts: Vec<&str> = prefix_str.splitn(5, '\t').collect();
        if parts.len() < 5 { continue; }

        chroms.append_value(parts[0]);
        positions.append_value(parts[1].parse::<i64>().unwrap_or(0));
        ids.append_value(parts[2]);
        refs_b.append_value(parts[3]);
        alts_b.append_value(parts[4]);

        let ds_row = &mut ds_flat[n_rows * n_samples..(n_rows + 1) * n_samples];

        if is_chip[local_i] {
            let ci = chip_local_idx[local_i];
            let mut ac = 0u32;
            for s in 0..n_samples {
                let a0 = chip_genotypes[ci * n_haps + s * 2] as f32;
                let a1 = chip_genotypes[ci * n_haps + s * 2 + 1] as f32;
                ds_row[s] = a0 + a1;
                ac += (a0 + a1) as u32;
            }
            afs.append_value(ac as f32 / n_haps as f32);
            dr2s.append_null();
            imps.append_value(false);
        } else {
            // Two-pass DR2: var(dosage) / var_expected, numerically stable
            // Pass 1: dosage sum + hardcall AC
            let mut ac = 0u32;
            let mut p_sum = 0.0f64;
            for s in 0..n_samples {
                let ap1 = alt_probs[(s * 2) * tile_n + v];
                let ap2 = alt_probs[(s * 2 + 1) * tile_n + v];
                let ds = ap1 + ap2;
                ds_row[s] = ds;
                if ap1 > 0.5 { ac += 1; }
                if ap2 > 0.5 { ac += 1; }
                p_sum += (ap1 + ap2) as f64;
            }
            let af = ac as f32 / n_haps as f32;
            let p_hat = p_sum / n_haps as f64;
            // Pass 2: variance of dosage
            let mut var_sum = 0.0f64;
            for s in 0..n_samples {
                let d = ds_row[s] as f64 - 2.0 * p_hat;
                var_sum += d * d;
            }
            let var_dosage = var_sum / n_haps as f64;
            let var_expected = 2.0 * p_hat * (1.0 - p_hat);
            let dr2 = if var_expected > 1e-10 { (var_dosage / var_expected).clamp(0.0, 1.0) as f32 } else { 0.0 };
            afs.append_value(af);
            dr2s.append_value(dr2);
            imps.append_value(true);
        }
        n_rows += 1;
    }

    if n_rows == 0 { return Ok(()); }

    // Build per-sample Float32Arrays from flat buffer columns
    let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(8 + n_samples);
    columns.push(Arc::new(chroms.finish()));
    columns.push(Arc::new(positions.finish()));
    columns.push(Arc::new(ids.finish()));
    columns.push(Arc::new(refs_b.finish()));
    columns.push(Arc::new(alts_b.finish()));
    columns.push(Arc::new(afs.finish()));
    columns.push(Arc::new(dr2s.finish()));
    columns.push(Arc::new(imps.finish()));

    // Extract per-sample columns from flat buffer (column-stride copy)
    for s in 0..n_samples {
        let mut col = Float32Builder::with_capacity(n_rows);
        for v in 0..n_rows {
            col.append_value(ds_flat[v * n_samples + s]);
        }
        columns.push(Arc::new(col.finish()));
    }

    let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    writer.write(&batch).map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

/// Finalize the Parquet writer (flush + write footer).
pub fn finish_parquet_writer(writer: ArrowWriter<std::fs::File>) -> std::io::Result<()> {
    writer.close().map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}
