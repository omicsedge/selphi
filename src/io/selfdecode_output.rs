//! SelfDecode output format: per-sample chunked Parquet files in a ZIP archive.
//!
//! Schema (per-sample, one row per variant):
//!   pos (Int32), rsid (Utf8), ref (Utf8), alt (Utf8),
//!   gt (Utf8), gt1 (Int32), gt2 (Int32), phased (Bool),
//!   ap1 (Float32), ap2 (Float32)
//!
//! Output structure inside ZIP:
//!   Single sample:  chrom={chr}/{chunk}.parquet
//!   Multi-sample:   {sample}/chrom={chr}/{chunk}.parquet
//!
//! Chunk size: 100,000 rows per file.
//! Compression: Snappy, dictionary encoding enabled.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

const CHUNK_SIZE: usize = 100_000;

/// Build the per-sample Arrow schema for SelfDecode output.
fn build_schema() -> Schema {
    Schema::new(vec![
        Field::new("pos", DataType::Int32, false),
        Field::new("rsid", DataType::Utf8, true),
        Field::new("ref", DataType::Utf8, false),
        Field::new("alt", DataType::Utf8, false),
        Field::new("gt", DataType::Utf8, true),
        Field::new("gt1", DataType::Int32, true),
        Field::new("gt2", DataType::Int32, true),
        Field::new("phased", DataType::Boolean, false),
        Field::new("ap1", DataType::Float32, true),
        Field::new("ap2", DataType::Float32, true),
    ])
}

/// Static GT strings for zero-alloc genotype formatting (alleles always 0 or 1).
static GT_STRS: [&str; 5] = ["0|0", "0|1", "1|0", "1|1", ".|."];

#[inline]
fn gt_str(g1: i32, g2: i32) -> &'static str {
    match (g1, g2) { (0,0) => GT_STRS[0], (0,1) => GT_STRS[1], (1,0) => GT_STRS[2], (1,1) => GT_STRS[3], _ => GT_STRS[4] }
}

/// Per-sample row buffer.
struct SampleBuffer {
    pos: Vec<i32>,
    rsid: Vec<Option<String>>,
    ref_allele: Vec<String>,
    alt_allele: Vec<String>,
    /// Static text for the 0/1 calls Selphi writes; owned text for a
    /// passthrough record's input genotype.
    gt: Vec<std::borrow::Cow<'static, str>>,
    gt1: Vec<Option<i32>>,
    gt2: Vec<Option<i32>>,
    phased: Vec<bool>,
    ap1: Vec<Option<f32>>,
    ap2: Vec<Option<f32>>,
}

impl SampleBuffer {
    fn new() -> Self {
        Self {
            pos: Vec::with_capacity(CHUNK_SIZE),
            rsid: Vec::with_capacity(CHUNK_SIZE),
            ref_allele: Vec::with_capacity(CHUNK_SIZE),
            alt_allele: Vec::with_capacity(CHUNK_SIZE),
            gt: Vec::with_capacity(CHUNK_SIZE),
            gt1: Vec::with_capacity(CHUNK_SIZE),
            gt2: Vec::with_capacity(CHUNK_SIZE),
            phased: Vec::with_capacity(CHUNK_SIZE),
            ap1: Vec::with_capacity(CHUNK_SIZE),
            ap2: Vec::with_capacity(CHUNK_SIZE),
        }
    }

    fn len(&self) -> usize {
        self.pos.len()
    }

    fn clear(&mut self) {
        self.pos.clear();
        self.rsid.clear();
        self.ref_allele.clear();
        self.alt_allele.clear();
        self.gt.clear();
        self.gt1.clear();
        self.gt2.clear();
        self.phased.clear();
        self.ap1.clear();
        self.ap2.clear();
    }

    /// Convert buffer to an Arrow RecordBatch and write as a Parquet file.
    fn to_parquet_bytes(&self, schema: &Arc<Schema>) -> std::io::Result<Vec<u8>> {
        if self.pos.is_empty() { return Ok(Vec::new()); }

        let n = self.pos.len();
        let pos_arr = Int32Array::from_iter_values(self.pos.iter().copied());
        let rsid_arr: StringArray = self.rsid.iter().map(|s| s.as_deref()).collect();
        let ref_arr: StringArray = self.ref_allele.iter().map(|s| Some(s.as_str())).collect();
        let alt_arr: StringArray = self.alt_allele.iter().map(|s| Some(s.as_str())).collect();
        let gt_arr: StringArray = self.gt.iter().map(|s| Some(s.as_ref())).collect();
        let gt1_arr: Int32Array = self.gt1.iter().copied().collect();
        let gt2_arr: Int32Array = self.gt2.iter().copied().collect();
        let phased_arr = BooleanArray::from_iter(self.phased.iter().map(|&b| Some(b)));
        let ap1_arr: Float32Array = self.ap1.iter().copied().collect();
        let ap2_arr: Float32Array = self.ap2.iter().copied().collect();

        let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(pos_arr),
            Arc::new(rsid_arr),
            Arc::new(ref_arr),
            Arc::new(alt_arr),
            Arc::new(gt_arr),
            Arc::new(gt1_arr),
            Arc::new(gt2_arr),
            Arc::new(phased_arr),
            Arc::new(ap1_arr),
            Arc::new(ap2_arr),
        ];

        let batch = arrow::record_batch::RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        let mut buf = Vec::with_capacity(n * 50);
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_dictionary_enabled(true)
            .set_max_row_group_size(CHUNK_SIZE)
            .build();

        let mut writer = ArrowWriter::try_new(&mut buf, schema.clone(), Some(props))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        writer.write(&batch).map_err(|e| std::io::Error::other(e.to_string()))?;
        writer.close().map_err(|e| std::io::Error::other(e.to_string()))?;

        Ok(buf)
    }
}

/// Writer for the SelfDecode output format.
/// Buffers per-sample rows and flushes to Parquet chunks in a ZIP archive.
pub struct SelfdecodeWriter {
    zip: zip::ZipWriter<std::fs::File>,
    schema: Arc<Schema>,
    sample_names: Vec<String>,
    multi_sample: bool,
    /// Per-sample buffers, indexed by sample index.
    buffers: Vec<SampleBuffer>,
    /// Current chromosome for each sample (for directory partitioning).
    current_chrom: Vec<String>,
    /// Chunk counter per (sample_idx, chrom) for file naming.
    chunk_counts: HashMap<(usize, String), usize>,
    /// Whether to filter out homozygous-ref rows.
    filter_hom_ref: bool,
}

impl SelfdecodeWriter {
    /// Create a new SelfDecode writer.
    pub fn new(
        output_path: &Path,
        sample_names: &[String],
        filter_hom_ref: bool,
    ) -> std::io::Result<Self> {
        Self::new_inner(output_path, sample_names, filter_hom_ref, sample_names.len() > 1)
    }

    /// Create a SelfDecode writer for one shard of a batched output.
    /// Forces `multi_sample=true` so per-batch ZIP entries always carry the
    /// sample-name prefix — preventing collisions when the merger
    /// (`sd_merge::merge_batch_sds`) concatenates entries from N batches.
    pub fn new_batched(
        output_path: &Path,
        sample_names: &[String],
        filter_hom_ref: bool,
    ) -> std::io::Result<Self> {
        Self::new_inner(output_path, sample_names, filter_hom_ref, true)
    }

    fn new_inner(
        output_path: &Path,
        sample_names: &[String],
        filter_hom_ref: bool,
        multi_sample: bool,
    ) -> std::io::Result<Self> {
        let zip_path = output_path.with_extension("selfdecode.zip");
        let file = std::fs::File::create(&zip_path)?;
        let zip = zip::ZipWriter::new(file);
        let buffers: Vec<SampleBuffer> = (0..sample_names.len())
            .map(|_| SampleBuffer::new())
            .collect();
        let current_chrom: Vec<String> = (0..sample_names.len())
            .map(|_| String::new())
            .collect();

        Ok(Self {
            zip,
            schema: Arc::new(build_schema()),
            sample_names: sample_names.to_vec(),
            multi_sample,
            buffers,
            current_chrom,
            chunk_counts: HashMap::new(),
            filter_hom_ref,
        })
    }

    /// Write one variant for all samples.
    ///
    /// `alt_probs` layout: alt_probs[(sample * 2 + hap) * tile_n + v]
    /// For chip sites, pass ap1/ap2 derived from chip_genotypes.
    #[allow(clippy::too_many_arguments)]
    pub fn write_variant(
        &mut self,
        chrom: &str,
        pos: i32,
        rsid: &str,
        ref_allele: &str,
        alt_allele: &str,
        // Per-sample genotype data:
        gt1_values: &[i32],    // n_samples: first allele (0 or 1)
        gt2_values: &[i32],    // n_samples: second allele (0 or 1)
        ap1_values: &[f32],    // n_samples: allele prob hap 1
        ap2_values: &[f32],    // n_samples: allele prob hap 2
        is_chip: bool,
        // R4b: at an imputed (`is_chip=false`) re-routed chip site, samples whose
        // bit is true emit a verbatim hard call → AP1/AP2 are null (chip-like)
        // for them, even though the record as a whole is imputed. `None` →
        // pre-R4b behavior (AP nullness is the single `is_chip` flag for all).
        hardcall_mask: Option<&[bool]>,
    ) -> std::io::Result<()> {
        let n_samples = self.sample_names.len();
        // Full-token contig classification (chr-prefix-tolerant, case-insensitive)
        // rather than a `contains('M')` substring test, so a non-standard scaffold
        // whose label merely contains 'M' is not mistaken for chrMT. Identical to
        // the old test on every standard contig (1-22/X/Y/M/MT/chrM/chrMT).
        let is_chrm = crate::contig::classify_contig(chrom) == crate::contig::ContigClass::ChrMt;

        for s in 0..n_samples {
            let g1 = gt1_values[s];
            let g2 = gt2_values[s];

            // Filter homozygous-ref if enabled (except chrM)
            if self.filter_hom_ref && !is_chrm && g1 == 0 && g2 == 0 {
                continue;
            }

            // R4b: a per-sample hard-call (mask[s]) is null-AP like a chip site.
            let sample_is_hardcall = is_chip
                || hardcall_mask.is_some_and(|m| m[s]);
            let (ap1, ap2) = if sample_is_hardcall { (None, None) }
                else { (Some(ap1_values[s]), Some(ap2_values[s])) };
            // Selphi's own calls are always phased.
            self.push_row(s, chrom, pos, rsid, ref_allele, alt_allele,
                std::borrow::Cow::Borrowed(gt_str(g1, g2)), Some(g1), Some(g2), true, ap1, ap2)?;
        }

        Ok(())
    }

    /// Write a passthrough record (a target site absent from the panel; see
    /// `io::passthrough`) for samples `samples`, as given: GT text with its own
    /// separator, `gt1`/`gt2` = the alleles as 0/1 (null when missing, or for
    /// the absent second allele of a haploid call), `phased` from the input, AP
    /// null. Hom-ref rows follow the same filter as every other row.
    pub fn write_passthrough(
        &mut self, chrom: &str, rec: &crate::io::passthrough::PassRecord,
        samples: std::ops::Range<usize>,
    ) -> std::io::Result<()> {
        use crate::io::passthrough::{ABSENT, MISSING};
        let is_chrm = crate::contig::classify_contig(chrom) == crate::contig::ContigClass::ChrMt;
        let alt = rec.alt_text();
        for (s, gs) in samples.enumerate() {
            let (a0, a1, phased) = rec.alleles(gs);
            let bit = |a: u8| -> Option<i32> { match a { MISSING | ABSENT => None, a => Some((a > 0) as i32) } };
            if self.filter_hom_ref && !is_chrm && a0 == 0 && (a1 == 0 || a1 == ABSENT) {
                continue;
            }
            self.push_row(s, chrom, rec.pos as i32, &rec.id, &rec.ref_allele, &alt,
                std::borrow::Cow::Owned(crate::io::passthrough::gt_text(a0, a1, phased)),
                bit(a0), bit(a1), phased && a1 != ABSENT, None, None)?;
        }
        Ok(())
    }

    /// Append one row to sample `s`'s buffer (flushing on a contig change and
    /// when the chunk is full).
    #[allow(clippy::too_many_arguments)]
    fn push_row(
        &mut self, s: usize, chrom: &str, pos: i32, rsid: &str, ref_allele: &str, alt_allele: &str,
        gt: std::borrow::Cow<'static, str>, gt1: Option<i32>, gt2: Option<i32>, phased: bool,
        ap1: Option<f32>, ap2: Option<f32>,
    ) -> std::io::Result<()> {
        if self.current_chrom[s] != chrom {
            self.flush_sample(s)?;
            self.current_chrom[s] = chrom.to_string();
        }
        let buf = &mut self.buffers[s];
        buf.pos.push(pos);
        buf.rsid.push(if rsid.starts_with("rs") { Some(rsid.to_string()) } else { None });
        buf.ref_allele.push(ref_allele.to_string());
        buf.alt_allele.push(alt_allele.to_string());
        buf.gt.push(gt);
        buf.gt1.push(gt1);
        buf.gt2.push(gt2);
        buf.phased.push(phased);
        buf.ap1.push(ap1);
        buf.ap2.push(ap2);
        if buf.len() >= CHUNK_SIZE {
            self.flush_sample(s)?;
        }
        Ok(())
    }

    /// Flush one sample's buffer to a parquet file in the ZIP.
    fn flush_sample(&mut self, sample_idx: usize) -> std::io::Result<()> {
        let buf = &self.buffers[sample_idx];
        if buf.pos.is_empty() { return Ok(()); }

        let chrom = &self.current_chrom[sample_idx];
        // Ensure chrom has 'chr' prefix for Hive-style partitioning
        let chrom_display = if chrom.starts_with("chr") { chrom.clone() } else { format!("chr{}", chrom) };
        let key = (sample_idx, chrom.clone());
        let chunk_num = self.chunk_counts.get(&key).copied().unwrap_or(0) + 1;
        self.chunk_counts.insert(key, chunk_num);

        // Build path inside ZIP
        let path = if self.multi_sample {
            format!("{}/chrom={}/{}.parquet", self.sample_names[sample_idx], chrom_display, chunk_num)
        } else {
            format!("chrom={}/{}.parquet", chrom_display, chunk_num)
        };

        let parquet_bytes = buf.to_parquet_bytes(&self.schema)?;
        if parquet_bytes.is_empty() { return Ok(()); }

        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored) // parquet already compressed
            // Fixed entry timestamp (ZIP epoch) so the archive is bit-identical
            // across runs — otherwise the per-entry mod-time defaults to now(),
            // breaking Selphi's deterministic-output guarantee for this format.
            .last_modified_time(zip::DateTime::default());
        self.zip.start_file(&path, options)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        self.zip.write_all(&parquet_bytes)?;

        self.buffers[sample_idx].clear();
        Ok(())
    }

    /// Flush all remaining buffers and finalize the ZIP.
    pub fn finish(mut self) -> std::io::Result<()> {
        let n = self.sample_names.len();
        for s in 0..n {
            self.flush_sample(s)?;
        }
        let total_chunks: usize = self.chunk_counts.values().sum();
        crate::selphi_info!("  SelfDecode: {} samples, {} chunks written",
            self.sample_names.len(), total_chunks);
        self.zip.finish().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }
}
