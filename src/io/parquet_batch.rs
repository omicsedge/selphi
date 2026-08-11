//! Per-batch Parquet writer.
//!
//! Intermediate schema differs from the final output:
//!   CHROM (Utf8), POS (Int64), ID (Utf8), REF (Utf8), ALT (Utf8), IMP (Bool),
//!   sample_1_ap1 (Float32), sample_1_ap2 (Float32),
//!   sample_2_ap1 (Float32), sample_2_ap2 (Float32),
//!   …
//!
//! Storing per-hap probabilities (AP1, AP2) instead of pre-summed dosage
//! lets the merger reproduce the non-batched encoder's AC computation
//! `(ap1>0.5) + (ap2>0.5)` bit-identically — DS alone is insufficient.
//! The merger then collapses AP1+AP2 → DS for the final parquet column.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use crate::io::batch_driver::{BatchSink, WindowBatchInput, WindowCtx};

/// Build the per-batch Parquet schema (AP1/AP2 columns per sample).
pub fn build_batch_schema(sample_names: &[String]) -> Schema {
    let mut fields = vec![
        Field::new("CHROM", DataType::Utf8, false),
        Field::new("POS", DataType::Int64, false),
        Field::new("ID", DataType::Utf8, true),
        Field::new("REF", DataType::Utf8, false),
        Field::new("ALT", DataType::Utf8, false),
        Field::new("IMP", DataType::Boolean, false),
    ];
    for name in sample_names {
        fields.push(Field::new(format!("{name}_ap1"), DataType::Float32, true));
        fields.push(Field::new(format!("{name}_ap2"), DataType::Float32, true));
    }
    Schema::new(fields)
}

pub struct ParquetBatchWriter {
    pub writer: ArrowWriter<std::fs::File>,
    pub schema: Arc<Schema>,
    pub path: PathBuf,
    pub hap_start: usize,
    pub hap_end: usize,
}

pub fn setup_parquet_batch_writers(
    n_haps: usize,
    batch_size: usize,
    tmp_dir: &Path,
    all_sample_names: &[String],
) -> std::io::Result<Vec<ParquetBatchWriter>> {
    if batch_size == 0 || n_haps == 0 {
        return Ok(Vec::new());
    }
    std::fs::create_dir_all(tmp_dir)?;

    let mut writers = Vec::new();
    crate::io::batch_driver::for_each_batch(n_haps, batch_size, |r| {
        let path = tmp_dir.join(format!("selphi_batch_{:04}.parquet", r.batch_idx));
        let batch_names = &all_sample_names[r.sample_start..r.sample_end];
        let schema = Arc::new(build_batch_schema(batch_names));
        let file = std::fs::File::create(&path)?;
        let props = WriterProperties::builder()
            .set_compression(Compression::ZSTD(Default::default()))
            .set_max_row_group_size(1024)
            .build();
        let writer = ArrowWriter::try_new(file, schema.clone(), Some(props))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writers.push(ParquetBatchWriter { writer, schema, path, hap_start: r.hap_start, hap_end: r.hap_end });
        Ok::<(), std::io::Error>(())
    })?;
    Ok(writers)
}

pub fn finalize_parquet_batch_writers(writers: Vec<ParquetBatchWriter>) -> std::io::Result<Vec<PathBuf>> {
    crate::io::batch_driver::finalize_writers(writers, |w| -> std::io::Result<PathBuf> {
        let ParquetBatchWriter { writer, path, .. } = w;
        writer.close().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(path)
    })
}

/// [`BatchSink`] for the per-batch Parquet writer. Accumulates tile-sized row
/// groups of CHROM/POS/ID/REF/ALT/IMP + per-sample AP1/AP2 columns; flushes a
/// `RecordBatch` whenever the buffered rows reach `cap` (tile size) and once
/// more at window end for the partial tile.
struct ParquetSink<'w> {
    bw: &'w mut ParquetBatchWriter,
    cap: usize,
    chroms: StringBuilder,
    positions: Int64Builder,
    ids: StringBuilder,
    refs_b: StringBuilder,
    alts_b: StringBuilder,
    imps: BooleanBuilder,
    // Per-sample AP1/AP2 builders, sample-major: ap1_s0, ap2_s0, ap1_s1, …
    ap_builders: Vec<Float32Builder>,
}

impl ParquetSink<'_> {
    /// Flush the buffered rows as one Parquet RecordBatch (no-op if empty).
    fn flush(&mut self) -> std::io::Result<()> {
        if self.positions.len() == 0 { return Ok(()); }
        let mut cols: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(6 + self.ap_builders.len());
        cols.push(Arc::new(self.chroms.finish()));
        cols.push(Arc::new(self.positions.finish()));
        cols.push(Arc::new(self.ids.finish()));
        cols.push(Arc::new(self.refs_b.finish()));
        cols.push(Arc::new(self.alts_b.finish()));
        cols.push(Arc::new(self.imps.finish()));
        for b in self.ap_builders.iter_mut() {
            cols.push(Arc::new(b.finish()));
        }
        let batch = arrow::record_batch::RecordBatch::try_new(self.bw.schema.clone(), cols)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.bw.writer.write(&batch).map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }
}

impl BatchSink for ParquetSink<'_> {
    fn begin_window(&mut self, ctx: &WindowCtx) -> std::io::Result<()> {
        // Tile-level row buffers (flush as row groups). cap == the kernel
        // tile_size (4000); the historical `.max(64)` floor is a no-op here.
        let cap = 4000usize;
        self.cap = cap;
        self.chroms = StringBuilder::with_capacity(cap, cap * 4);
        self.positions = Int64Builder::with_capacity(cap);
        self.ids = StringBuilder::with_capacity(cap, cap * 16);
        self.refs_b = StringBuilder::with_capacity(cap, cap * 2);
        self.alts_b = StringBuilder::with_capacity(cap, cap * 2);
        self.imps = BooleanBuilder::with_capacity(cap);
        self.ap_builders = (0..ctx.n_samples_in_batch * 2)
            .map(|_| Float32Builder::with_capacity(cap)).collect();
        Ok(())
    }

    fn emit_chip(&mut self, wgs_i: usize, local_i: usize, ctx: &WindowCtx) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        let pos: i64 = pos_s.parse().unwrap_or(0);
        self.chroms.append_value(chrom);
        self.positions.append_value(pos);
        self.ids.append_value(rsid);
        self.refs_b.append_value(ref_a);
        self.alts_b.append_value(alt_a);
        self.imps.append_value(false);
        let ci = ctx.chip_local_idx[local_i];
        for s in 0..ctx.n_samples_in_batch {
            let gs = ctx.sample_start + s;
            let a0 = ctx.chip_genotypes.get(ci, gs * 2) as u8 as f32;
            let a1 = ctx.chip_genotypes.get(ci, gs * 2 + 1) as u8 as f32;
            self.ap_builders[s * 2].append_value(a0);
            self.ap_builders[s * 2 + 1].append_value(a1);
        }
        if self.positions.len() >= self.cap { self.flush()?; }
        Ok(())
    }

    fn emit_imputed(
        &mut self, wgs_i: usize, local_i: usize,
        alt: &[f32], tile_n: usize, v: usize, ctx: &WindowCtx,
    ) -> std::io::Result<()> {
        let (chrom, pos_s, rsid, ref_a, alt_a) =
            crate::io::pipeline::parse_variant_parts(ctx.srp, wgs_i)
                .ok_or_else(|| std::io::Error::other(format!("bad variant id at {wgs_i}")))?;
        let pos: i64 = pos_s.parse().unwrap_or(0);
        self.chroms.append_value(chrom);
        self.positions.append_value(pos);
        self.ids.append_value(rsid);
        self.refs_b.append_value(ref_a);
        self.alts_b.append_value(alt_a);
        self.imps.append_value(true);
        // R4b: a confident sample at a re-routed input chip site emits its
        // verbatim hard call; soft / pure-imputed → alt_probs.
        let ci = ctx.chip_local_idx[local_i];
        for s in 0..ctx.n_samples_in_batch {
            let (p1, p2) = if ctx.use_hardcall(local_i, s) {
                let gs = ctx.sample_start + s;
                (ctx.chip_genotypes.get(ci, gs * 2) as u8 as f32,
                 ctx.chip_genotypes.get(ci, gs * 2 + 1) as u8 as f32)
            } else {
                (alt[(s * 2) * tile_n + v], alt[(s * 2 + 1) * tile_n + v])
            };
            self.ap_builders[s * 2].append_value(p1);
            self.ap_builders[s * 2 + 1].append_value(p2);
        }
        if self.positions.len() >= self.cap { self.flush()?; }
        Ok(())
    }

    fn end_window(&mut self) -> std::io::Result<()> {
        // Flush any partial tile.
        self.flush()
    }
}

pub fn write_window_parquet_batched(
    input: WindowBatchInput<'_>,
    bw: &mut ParquetBatchWriter,
) -> std::io::Result<()> {
    let WindowBatchInput {
        srp, weights, hap_start, hap_end, win_chip_start, own_chip_start, own_chip_end,
        wgs_idx, n_samples_total, chip_genotypes, no_ap: _, site_conf, site_conf_per_sample, refine_thr,
        interp_cum_cm,
    } = input;
    let mut sink = ParquetSink {
        bw,
        cap: 0,
        chroms: StringBuilder::new(),
        positions: Int64Builder::new(),
        ids: StringBuilder::new(),
        refs_b: StringBuilder::new(),
        alts_b: StringBuilder::new(),
        imps: BooleanBuilder::new(),
        ap_builders: Vec::new(),
    };
    crate::io::batch_driver::run_window(
        &mut sink, srp.as_ref(), weights, hap_start, hap_end,
        win_chip_start, own_chip_start, own_chip_end, wgs_idx, n_samples_total, chip_genotypes,
        site_conf, site_conf_per_sample, refine_thr, interp_cum_cm,
    )
}
