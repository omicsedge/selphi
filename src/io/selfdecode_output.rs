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

/// Per-sample row buffer.
struct SampleBuffer {
    pos: Vec<i32>,
    rsid: Vec<Option<String>>,
    ref_allele: Vec<String>,
    alt_allele: Vec<String>,
    gt: Vec<Option<String>>,
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
        let pos_arr = Int32Array::from(self.pos.clone());
        let rsid_arr: StringArray = self.rsid.iter().map(|s| s.as_deref()).collect();
        let ref_arr: StringArray = self.ref_allele.iter().map(|s| Some(s.as_str())).collect();
        let alt_arr: StringArray = self.alt_allele.iter().map(|s| Some(s.as_str())).collect();
        let gt_arr: StringArray = self.gt.iter().map(|s| s.as_deref()).collect();
        let gt1_arr: Int32Array = self.gt1.iter().copied().collect();
        let gt2_arr: Int32Array = self.gt2.iter().copied().collect();
        let phased_arr = BooleanArray::from(self.phased.clone());
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
        let zip_path = output_path.with_extension("selfdecode.zip");
        let file = std::fs::File::create(&zip_path)?;
        let zip = zip::ZipWriter::new(file);
        let multi_sample = sample_names.len() > 1;
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
    ) -> std::io::Result<()> {
        let n_samples = self.sample_names.len();
        let is_chrm = chrom.contains('M');

        for s in 0..n_samples {
            let g1 = gt1_values[s];
            let g2 = gt2_values[s];

            // Filter homozygous-ref if enabled (except chrM)
            if self.filter_hom_ref && !is_chrm && g1 == 0 && g2 == 0 {
                continue;
            }

            // Flush if chrom changed
            if self.current_chrom[s] != chrom {
                self.flush_sample(s)?;
                self.current_chrom[s] = chrom.to_string();
            }

            let buf = &mut self.buffers[s];
            buf.pos.push(pos);
            buf.rsid.push(if rsid.starts_with("rs") { Some(rsid.to_string()) } else { None });
            buf.ref_allele.push(ref_allele.to_string());
            buf.alt_allele.push(alt_allele.to_string());
            buf.gt.push(Some(format!("{}|{}", g1, g2)));
            buf.gt1.push(Some(g1));
            buf.gt2.push(Some(g2));
            buf.phased.push(true); // selphi always produces phased output
            if is_chip {
                buf.ap1.push(None);
                buf.ap2.push(None);
            } else {
                buf.ap1.push(Some(ap1_values[s]));
                buf.ap2.push(Some(ap2_values[s]));
            }

            // Flush chunk if buffer full
            if buf.len() >= CHUNK_SIZE {
                self.flush_sample(s)?;
            }
        }

        Ok(())
    }

    /// Flush one sample's buffer to a parquet file in the ZIP.
    fn flush_sample(&mut self, sample_idx: usize) -> std::io::Result<()> {
        let buf = &self.buffers[sample_idx];
        if buf.len() == 0 { return Ok(()); }

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
            .compression_method(zip::CompressionMethod::Stored); // parquet already compressed
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
        self.zip.finish().map_err(|e| std::io::Error::other(e.to_string()))?;
        Ok(())
    }
}
