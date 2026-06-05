//! Stream-merge imputed + truth VCF/BCF, compute per-site and per-sample accuracy.
//!
//! Metrics:
//!   - Per-site R² (Pearson correlation² between dosage and truth genotype)
//!   - Per-site concordance (fraction of matching hardcalls)
//!   - Per-sample R² (across all variants for each sample)
//!   - MAF-binned R² (paper-style MAF bins)
//!
//! Designed for 50K+ samples: O(n_samples) memory per variant, streaming.

use std::io::{self, BufRead, BufReader, Read};
use std::path::Path;

/// MAF bins matching the paper standard.
pub const MAF_BINS: &[(f64, f64, &str)] = &[
    (0.0005, 0.001,  "0.05-0.1%"),
    (0.001,  0.002,  "0.1-0.2%"),
    (0.002,  0.005,  "0.2-0.5%"),
    (0.005,  0.01,   "0.5-1%"),
    (0.01,   0.02,   "1-2%"),
    (0.02,   0.05,   "2-5%"),
    (0.05,   0.10,   "5-10%"),
    (0.10,   0.20,   "10-20%"),
    (0.20,   0.50,   "20-50%"),
];

/// Accumulator for per-sample statistics (online, across all variants).
pub struct SampleAccumulator {
    pub n_samples: usize,
    pub n_variants: u64,
    // Welford online correlation components
    pub sum_ds: Vec<f64>,
    pub sum_gt: Vec<f64>,
    pub sum_ds2: Vec<f64>,
    pub sum_gt2: Vec<f64>,
    pub sum_ds_gt: Vec<f64>,
    pub n_correct: Vec<u64>,
    pub n_total: Vec<u64>,
    // Per-MAF-bin errors
    pub bin_errors: Vec<Vec<u64>>,  // [bin][sample]
    pub bin_n: Vec<u64>,            // [bin] variant count
}

impl SampleAccumulator {
    pub fn new(n_samples: usize) -> Self {
        let n_bins = MAF_BINS.len();
        SampleAccumulator {
            n_samples, n_variants: 0,
            sum_ds: vec![0.0; n_samples],
            sum_gt: vec![0.0; n_samples],
            sum_ds2: vec![0.0; n_samples],
            sum_gt2: vec![0.0; n_samples],
            sum_ds_gt: vec![0.0; n_samples],
            n_correct: vec![0; n_samples],
            n_total: vec![0; n_samples],
            bin_errors: (0..n_bins).map(|_| vec![0u64; n_samples]).collect(),
            bin_n: vec![0; n_bins],
        }
    }

    /// Add one variant's dosage/truth for all samples.
    pub fn add_variant(&mut self, ds: &[f32], truth_gt: &[f32], maf: f64) {
        self.n_variants += 1;
        // The MAF bin depends only on `maf` (per-variant), so resolve it ONCE
        // instead of rescanning MAF_BINS for every sample. `position` returns the
        // first matching bin — identical to the original first-match-then-break
        // scan, so the accumulated bin_errors/bin_n are bit-identical.
        let bin = MAF_BINS.iter().position(|&(lo, hi, _)| maf >= lo && maf < hi);
        for s in 0..self.n_samples {
            let d = ds[s] as f64;
            let g = truth_gt[s] as f64;
            if d < 0.0 || g < 0.0 { continue; } // missing
            self.sum_ds[s] += d;
            self.sum_gt[s] += g;
            self.sum_ds2[s] += d * d;
            self.sum_gt2[s] += g * g;
            self.sum_ds_gt[s] += d * g;

            let d_call = if d > 1.5 { 2 } else if d > 0.5 { 1 } else { 0 };
            let g_call = g.round() as i32;
            if d_call == g_call { self.n_correct[s] += 1; }
            self.n_total[s] += 1;

            // MAF bin errors (bin resolved once above)
            if d_call != g_call {
                if let Some(bi) = bin { self.bin_errors[bi][s] += 1; }
            }
        }

        // Track bin variant counts
        if let Some(bi) = bin { self.bin_n[bi] += 1; }
    }

    /// Compute per-sample R² from accumulated statistics.
    pub fn compute_r2(&self) -> Vec<f64> {
        (0..self.n_samples).map(|s| {
            let n = self.n_total[s] as f64;
            if n < 2.0 { return 0.0; }
            let num = n * self.sum_ds_gt[s] - self.sum_ds[s] * self.sum_gt[s];
            let den_x = n * self.sum_ds2[s] - self.sum_ds[s] * self.sum_ds[s];
            let den_y = n * self.sum_gt2[s] - self.sum_gt[s] * self.sum_gt[s];
            let den = (den_x.max(0.0) * den_y.max(0.0)).sqrt();
            if den > 0.0 { (num / den).powi(2).clamp(0.0, 1.0) } else { 0.0 }
        }).collect()
    }

    /// Compute per-sample concordance.
    pub fn compute_concordance(&self) -> Vec<f64> {
        (0..self.n_samples).map(|s| {
            if self.n_total[s] > 0 { self.n_correct[s] as f64 / self.n_total[s] as f64 } else { 0.0 }
        }).collect()
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: &SampleAccumulator) {
        self.n_variants += other.n_variants;
        for s in 0..self.n_samples {
            self.sum_ds[s] += other.sum_ds[s];
            self.sum_gt[s] += other.sum_gt[s];
            self.sum_ds2[s] += other.sum_ds2[s];
            self.sum_gt2[s] += other.sum_gt2[s];
            self.sum_ds_gt[s] += other.sum_ds_gt[s];
            self.n_correct[s] += other.n_correct[s];
            self.n_total[s] += other.n_total[s];
        }
        for bi in 0..MAF_BINS.len() {
            for s in 0..self.n_samples {
                self.bin_errors[bi][s] += other.bin_errors[bi][s];
            }
            self.bin_n[bi] += other.bin_n[bi];
        }
    }
}

/// Per-site accumulator for MAF-binned R².
pub struct SiteAccumulator {
    // Per-MAF-bin: sum of R², count, sum of concordance (only R²-valid variants)
    pub bin_r2_sum: Vec<f64>,
    pub bin_conc_sum: Vec<f64>,
    pub bin_n: Vec<u64>,
    // Overall R² (only R²-valid variants)
    pub total_r2_sum: f64,
    pub total_r2_n: u64,
    // Overall concordance (ALL variants including monomorphic)
    pub total_conc_sum: f64,
    pub total_n: u64,
}

impl Default for SiteAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl SiteAccumulator {
    pub fn new() -> Self {
        let n_bins = MAF_BINS.len();
        SiteAccumulator {
            bin_r2_sum: vec![0.0; n_bins],
            bin_conc_sum: vec![0.0; n_bins],
            bin_n: vec![0; n_bins],
            total_r2_sum: 0.0,
            total_r2_n: 0,
            total_conc_sum: 0.0,
            total_n: 0,
        }
    }

    /// Add one variant's per-site metrics.
    /// Concordance always counted. R² only counted when not NaN AND when the
    /// variant falls into one of the declared MAF bins (MAF ≥ 0.0005). This
    /// matches Beagle/IMPUTE5/Minimac4 convention and avoids ultra-rare
    /// singleton noise (MAF < 0.05%) dragging the mean to meaningless values.
    pub fn add(&mut self, maf: f64, r2: f64, concordance: f64) {
        // Concordance counts for ALL variants (including monomorphic)
        self.total_conc_sum += concordance;
        self.total_n += 1;

        if !r2.is_nan() {
            for (bi, &(lo, hi, _)) in MAF_BINS.iter().enumerate() {
                if maf >= lo && maf < hi {
                    self.bin_r2_sum[bi] += r2;
                    self.bin_conc_sum[bi] += concordance;
                    self.bin_n[bi] += 1;
                    self.total_r2_sum += r2;
                    self.total_r2_n += 1;
                    break;
                }
            }
        }
    }

    pub fn merge(&mut self, other: &SiteAccumulator) {
        for bi in 0..MAF_BINS.len() {
            self.bin_r2_sum[bi] += other.bin_r2_sum[bi];
            self.bin_conc_sum[bi] += other.bin_conc_sum[bi];
            self.bin_n[bi] += other.bin_n[bi];
        }
        self.total_r2_sum += other.total_r2_sum;
        self.total_r2_n += other.total_r2_n;
        self.total_conc_sum += other.total_conc_sum;
        self.total_n += other.total_n;
    }
}

/// Compute per-site R² for one variant (Pearson correlation²).
/// ds: imputed dosage per sample. gt: truth genotype dosage per sample.
#[inline]
pub fn site_r2(ds: &[f32], gt: &[f32], n: usize) -> (f64, f64) {
    let mut sum_d = 0.0f64;
    let mut sum_g = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_g2 = 0.0f64;
    let mut sum_dg = 0.0f64;
    let mut count = 0u32;

    for i in 0..n {
        let d = ds[i] as f64;
        let g = gt[i] as f64;
        if d < 0.0 || g < 0.0 { continue; }
        sum_d += d; sum_g += g;
        sum_d2 += d * d; sum_g2 += g * g;
        sum_dg += d * g;
        count += 1;
    }

    if count < 2 { return (f64::NAN, 0.0); }
    let nf = count as f64;
    let num = nf * sum_dg - sum_d * sum_g;
    let den_x = nf * sum_d2 - sum_d * sum_d;
    let den_y = nf * sum_g2 - sum_g * sum_g;
    let den = (den_x * den_y).max(0.0).sqrt();
    let r2 = if den > 0.0 { (num / den).powi(2) } else { f64::NAN };

    // Concordance
    let mut correct = 0u32;
    for i in 0..n {
        if ds[i] < 0.0 || gt[i] < 0.0 { continue; }
        let d_call = if ds[i] > 1.5 { 2 } else if ds[i] > 0.5 { 1 } else { 0 };
        let g_call = gt[i].round() as i32;
        if d_call == g_call { correct += 1; }
    }
    let conc = correct as f64 / count as f64;

    (r2, conc)
}

/// Normalize indel alleles: trim common suffix then prefix.
fn norm_alleles(r: &[u8], a: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut r = r.to_vec();
    let mut a = a.to_vec();
    while r.len() > 1 && a.len() > 1 && r.last() == a.last() { r.pop(); a.pop(); }
    while r.len() > 1 && a.len() > 1 && r[0] == a[0] { r.remove(0); a.remove(0); }
    (r, a)
}

/// Match alleles accounting for different indel normalization and REF/ALT swaps.
/// Returns (matched, swapped).
fn match_alleles(imp_ref: &[u8], imp_alt: &[u8], truth_ref: &[u8], truth_alt: &[u8]) -> (bool, bool) {
    if imp_ref == truth_ref && imp_alt == truth_alt { return (true, false); }
    if imp_ref == truth_alt && imp_alt == truth_ref { return (true, true); }
    let ni = norm_alleles(imp_ref, imp_alt);
    let nt = norm_alleles(truth_ref, truth_alt);
    if ni.0 == nt.0 && ni.1 == nt.1 { return (true, false); }
    if ni.0 == nt.1 && ni.1 == nt.0 { return (true, true); }
    (false, false)
}

/// Strip a trailing `\n` or `\r\n` from a line read via `read_until(b'\n', ...)`.
#[inline]
fn strip_line_endings(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    if end > 0 && line[end - 1] == b'\n' { end -= 1; }
    if end > 0 && line[end - 1] == b'\r' { end -= 1; }
    &line[..end]
}

/// Parse a VCF/BCF text line to extract dosage for all samples.
/// Returns (chrom, pos, ref, alt, dosages) where dosages[i] is DS for sample i.
/// If DS not available, uses GT (0/0→0, 0/1→1, 1/1→2).
/// Returns None for header lines or multi-allelic.
pub fn parse_vcf_line(line: &[u8], n_samples: usize, ds_buf: &mut Vec<f32>) -> Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> {
    // Strip trailing newline (\n, \r\n). Without this, `is_empty()` below is
    // false for blank records whose line_buf contains just "\n", and the
    // pos/allele slicing would include the trailing byte.
    let line = strip_line_endings(line);
    if line.is_empty() || line[0] == b'#' { return None; }

    // Find first 9 tabs (fixed VCF columns)
    let mut tabs = [0usize; 9];
    let mut nt = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' { if nt < 9 { tabs[nt] = i; } nt += 1; if nt >= 9 { break; } }
    }
    if nt < 9 { return None; }

    let chrom = line[..tabs[0]].to_vec();
    let pos: i64 = std::str::from_utf8(&line[tabs[0]+1..tabs[1]]).ok()?.parse().ok()?;
    let ref_a = line[tabs[2]+1..tabs[3]].to_vec();
    let alt_field = &line[tabs[3]+1..tabs[4]];

    // Skip multi-allelic (contains comma in ALT)
    if alt_field.contains(&b',') { return None; }
    let alt_a = alt_field.to_vec();

    // Find DS index in FORMAT field
    let format = &line[tabs[7]+1..tabs[8]];
    let mut ds_idx: Option<usize> = None;
    let mut gt_idx: Option<usize> = None;
    for (fi, field) in format.split(|&b| b == b':').enumerate() {
        if field == b"DS" { ds_idx = Some(fi); }
        if field == b"GT" { gt_idx = Some(fi); }
    }

    // Parse sample fields
    ds_buf.clear();
    let sample_region = &line[tabs[8]+1..];
    for sample_field in sample_region.split(|&b| b == b'\t') {
        if ds_buf.len() >= n_samples { break; }

        if let Some(di) = ds_idx {
            // Extract DS subfield
            if let Some(val) = sample_field.split(|&b| b == b':').nth(di) {
                let s = std::str::from_utf8(val).unwrap_or(".");
                ds_buf.push(s.parse().unwrap_or(-1.0));
            } else {
                ds_buf.push(-1.0);
            }
        } else if let Some(gi) = gt_idx {
            // Fall back to GT. Missing alleles fold to 0 (hom-ref) to match
            // SRP/BCF readers and keep MAF denominators aligned.
            if let Some(gt) = sample_field.split(|&b| b == b':').nth(gi) {
                if gt.len() >= 3 {
                    let a0 = if gt[0] == b'.' { 0i32 } else { (gt[0] - b'0') as i32 };
                    let a1 = if gt[2] == b'.' { 0i32 } else { (gt[2] - b'0') as i32 };
                    ds_buf.push((a0 + a1) as f32);
                } else if gt.len() == 1 {
                    // Haploid GT (e.g. chrX males): one allele → its biallelic ALT
                    // count (0/1). Matches the BCF reader's vector_end handling
                    // (gt_allele_to_dose), so VCF-truth vs BCF-imputed agree on
                    // haploid sites. Missing "." → 0. No-op on autosomal diploid
                    // (always len>=3).
                    let a = if gt[0] == b'.' { 0i32 } else { (gt[0] - b'0') as i32 };
                    ds_buf.push(a.min(1) as f32);
                } else {
                    ds_buf.push(0.0);
                }
            } else {
                ds_buf.push(0.0);
            }
        } else {
            ds_buf.push(0.0);
        }
    }

    // Pad if needed
    while ds_buf.len() < n_samples { ds_buf.push(0.0); }

    Some((chrom, pos, ref_a, alt_a))
}

/// Parse VCF/BCF header to get sample names.
pub fn parse_header_samples(path: &Path) -> io::Result<Vec<String>> {
    let p = path.to_string_lossy();
    let is_bcf = p.ends_with(".bcf") || p.ends_with(".bcf.gz");

    if is_bcf {
        let hdr = crate::srp::bcf_reader::read_header_only(path)?;
        return Ok(hdr.sample_names);
    }

    // VCF.gz: read through BGZF
    let f = std::fs::File::open(path)?;
    let bgzf = noodles_bgzf::io::Reader::new(BufReader::new(f));
    let reader = BufReader::new(bgzf);

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("#CHROM") {
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() > 9 {
                return Ok(fields[9..].iter().map(|s| s.to_string()).collect());
            }
        }
    }
    Err(io::Error::new(io::ErrorKind::InvalidData, "no #CHROM header found"))
}

/// Evaluation result counts.
/// Find samples shared between imputed and truth files.
/// Returns (imputed_samples, truth_samples, shared_samples).
pub fn find_shared_samples(imp_path: &Path, truth_path: &Path) -> io::Result<(Vec<String>, Vec<String>, Vec<String>)> {
    let imp_samples = parse_header_samples(imp_path)?;
    let truth_samples = parse_header_samples(truth_path)?;
    let imp_set: std::collections::HashSet<&str> = imp_samples.iter().map(|s| s.as_str()).collect();
    let shared: Vec<String> = truth_samples.iter()
        .filter(|s| imp_set.contains(s.as_str()))
        .cloned().collect();
    Ok((imp_samples, truth_samples, shared))
}

pub struct EvalCounts {
    pub n_matched: u64,
    pub n_imp_variants: u64,
    pub n_truth_variants: u64,
    pub chromosomes: Vec<String>,
}

/// Variant record reader — abstracts over VCF.gz and BCF.
/// Returns (chrom, pos, ref_allele, alt_allele) and fills dosage buffer.
enum VariantReader {
    Vcf {
        reader: BufReader<noodles_bgzf::io::Reader<BufReader<std::fs::File>>>,
        line_buf: Vec<u8>,
        n_file_samples: usize,
        /// Path for TBI-based seek
        path: std::path::PathBuf,
        /// Virtual position of the first data line (post-header)
        header_end_vpos: u64,
    },
    Bcf {
        reader: noodles_bgzf::io::Reader<BufReader<std::fs::File>>,
        contig_names: Vec<String>,
        gt_key: u16,
        ds_key: Option<u16>,
        n_file_samples: usize,
        sample_indices: Option<Vec<usize>>,
        /// Path for re-opening with seek
        path: std::path::PathBuf,
        /// Header size for seeking past header
        header_end_vpos: u64,
        sb: Vec<u8>,
        ib: Vec<u8>,
    },
}

impl VariantReader {
    fn open(path: &Path) -> io::Result<(Self, Vec<String>)> {
        let s = path.to_string_lossy();
        let is_bcf = s.ends_with(".bcf") || s.ends_with(".bcf.gz");

        if is_bcf {
            let hdr = crate::srp::bcf_reader::read_header_only(path)?;
            let f = std::fs::File::open(path)?;
            let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));
            // Skip BCF header
            let mut magic = [0u8; 5]; bgzf.read_exact(&mut magic)?;
            let mut hlen_buf = [0u8; 4]; bgzf.read_exact(&mut hlen_buf)?;
            let hlen = u32::from_le_bytes(hlen_buf) as usize;
            let mut hdr_bytes = vec![0u8; hlen]; bgzf.read_exact(&mut hdr_bytes)?;

            // Find DS key IDX from header
            let hdr_text = String::from_utf8_lossy(&hdr_bytes);
            let mut ds_key: Option<u16> = None;
            for line in hdr_text.lines() {
                if line.starts_with("##FORMAT=<ID=DS,") && let Some(p) = line.find("IDX=") {
                    let s = p + 4;
                    let e = line[s..].find([',', '>']).map(|p| s + p).unwrap_or(line.len());
                    ds_key = line[s..e].parse().ok();
                }
            }

            let samples = hdr.sample_names.clone();
            let ns = hdr.n_samples;
            let header_end_vpos = u64::from(bgzf.virtual_position());
            Ok((VariantReader::Bcf {
                reader: bgzf, contig_names: hdr.contig_names, gt_key: hdr.gt_key_id,
                ds_key, n_file_samples: ns, sample_indices: None,
                path: path.to_path_buf(), header_end_vpos,
                sb: Vec::with_capacity(512), ib: Vec::with_capacity(ns * 4),
            }, samples))
        } else {
            // Read header directly through bgzf's own block buffer so
            // `virtual_position()` stays exactly at the first byte after
            // `#CHROM\n`. Wrapping bgzf in an outer BufReader would prefetch
            // data-section bytes before we capture `header_end_vpos`, and
            // `reader.get_ref().virtual_position()` then points past the first
            // data record — causing parallel-region seeks to start mid-first-line.
            let f = std::fs::File::open(path)?;
            let mut bgzf_inner = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));
            let mut samples: Vec<String> = Vec::new();
            let mut line_bytes: Vec<u8> = Vec::with_capacity(4096);
            'hdr: loop {
                line_bytes.clear();
                loop {
                    let buf = bgzf_inner.fill_buf()?;
                    if buf.is_empty() { break 'hdr; }
                    if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                        line_bytes.extend_from_slice(&buf[..nl]);
                        bgzf_inner.consume(nl + 1);
                        break;
                    } else {
                        let len = buf.len();
                        line_bytes.extend_from_slice(buf);
                        bgzf_inner.consume(len);
                    }
                }
                if line_bytes.starts_with(b"#CHROM") {
                    let line = String::from_utf8_lossy(&line_bytes);
                    let fields: Vec<&str> = line.trim_end().split('\t').collect();
                    if fields.len() > 9 {
                        samples = fields[9..].iter().map(|s| s.to_string()).collect();
                    }
                    break;
                }
            }
            let ns = samples.len();
            let header_end_vpos = u64::from(bgzf_inner.virtual_position());
            let reader = BufReader::new(bgzf_inner);
            Ok((VariantReader::Vcf {
                reader, line_buf: Vec::with_capacity(ns * 8), n_file_samples: ns,
                path: path.to_path_buf(), header_end_vpos,
            }, samples))
        }
    }

    /// Set sample indices to extract (selective extraction for BCF only).
    fn set_sample_filter(&mut self, indices: Vec<usize>) {
        match self {
            VariantReader::Bcf { sample_indices, .. } => { *sample_indices = Some(indices); }
            VariantReader::Vcf { .. } => { let _ = indices; }
        }
    }

    /// Seek to a genomic position using CSI index. BCF only.
    /// pos is 1-based VCF position.
    fn seek_to_position(&mut self, pos: i64) -> io::Result<()> {
        match self {
            VariantReader::Bcf { reader, path, header_end_vpos, .. } => {
                // Always re-open on seek (matches the VCF path). Seeking an
                // already-driven noodles_bgzf::Reader that's wrapped in a
                // BufReader leaves stale block/line state in ways that dropped
                // ~5% of records at region boundaries on parallel eval.
                let open_fresh = |vp: noodles_bgzf::VirtualPosition, p: &std::path::Path|
                    -> io::Result<noodles_bgzf::io::Reader<BufReader<std::fs::File>>> {
                    let f = std::fs::File::open(p)?;
                    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));
                    bgzf.seek(vp)?;
                    Ok(bgzf)
                };
                if pos <= 1 {
                    let vp = noodles_bgzf::VirtualPosition::from(*header_end_vpos);
                    *reader = open_fresh(vp, path)?;
                    return Ok(());
                }
                let csi_path = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
                if csi_path.exists() {
                    let csi = crate::srp::csi::parse_csi(&csi_path)?;
                    let vp = crate::srp::csi::seek_for_position(&csi, pos - 1); // CSI uses 0-based
                    *reader = open_fresh(vp, path)?;
                    Ok(())
                } else {
                    // No index — fall back to header start and let the caller
                    // scan sequentially. Don't leave the reader half-consumed.
                    let vp = noodles_bgzf::VirtualPosition::from(*header_end_vpos);
                    *reader = open_fresh(vp, path)?;
                    Ok(())
                }
            }
            VariantReader::Vcf { reader, path, header_end_vpos, .. } => {
                if pos <= 1 {
                    // Re-open and reset to first data line
                    let f = std::fs::File::open(path)?;
                    let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));
                    let vp = noodles_bgzf::VirtualPosition::from(*header_end_vpos);
                    bgzf.seek(vp)?;
                    *reader = BufReader::new(bgzf);
                    return Ok(());
                }
                let tbi_path = { let mut p = path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
                if !tbi_path.exists() { return Ok(()); }
                let tbi = match crate::srp::csi::parse_tbi(&tbi_path) { Ok(t) => t, Err(_) => return Ok(()) };
                // Pick the first contig with non-empty linear index (typical single-chr eval).
                let contig_idx = tbi.linear.iter().position(|lin| !lin.is_empty()).unwrap_or(0);
                let vp = crate::srp::csi::tbi_seek(&tbi, contig_idx, pos - 1);
                let f = std::fs::File::open(path)?;
                let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, f));
                bgzf.seek(vp)?;
                *reader = BufReader::new(bgzf);
                Ok(())
            }
        }
    }

    /// Read next biallelic variant. Returns (chrom, pos, ref, alt) and fills ds_buf with dosages.
    /// For BCF: if skip_genotypes=true, skips the individual data (fast position scan).
    fn next_record(&mut self, ds_buf: &mut Vec<f32>) -> Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> {
        self.next_record_inner(ds_buf, false)
    }

    fn next_record_inner(&mut self, ds_buf: &mut Vec<f32>, skip_gt: bool) -> Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> {
        match self {
            VariantReader::Vcf { reader, line_buf, n_file_samples, .. } => {
                loop {
                    line_buf.clear();
                    if reader.read_until(b'\n', line_buf).ok()? == 0 { return None; }
                    if let Some(rec) = parse_vcf_line(line_buf, *n_file_samples, ds_buf) {
                        return Some(rec);
                    }
                }
            }
            VariantReader::Bcf { reader, contig_names, gt_key, ds_key, n_file_samples, sample_indices, sb, ib, .. } => {
                let ns = *n_file_samples;
                let gtk = *gt_key;
                let dsk = *ds_key;
                let si_filter = sample_indices.as_deref();
                loop {
                    // Read record header
                    let mut lbuf = [0u8; 4];
                    let mut total = 0;
                    loop {
                        match reader.read(&mut lbuf[total..]) {
                            Ok(0) => { if total == 0 { return None; } return None; }
                            Ok(n) => { total += n; if total == 4 { break; } }
                            Err(_) => return None,
                        }
                    }
                    let ls = u32::from_le_bytes(lbuf) as usize;
                    if ls == 0 { return None; }
                    // BCF SHARED block has a 24-byte fixed header (chrom + pos + rlen +
                    // qual + n_info + n_allele + n_fmt); a truncated record (ls < 24)
                    // would panic on the sb[..] slicing below. Mirror the guard in
                    // srp/bcf_reader.rs and bail cleanly on malformed input.
                    if ls < 24 { return None; }
                    let mut libuf = [0u8; 4];
                    reader.read_exact(&mut libuf).ok()?;
                    let li = u32::from_le_bytes(libuf) as usize;

                    sb.resize(ls, 0); reader.read_exact(sb).ok()?;
                    let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap()) as usize;
                    let pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64 + 1;
                    let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;

                    if na != 2 || skip_gt {
                        // Skip individual data without reading
                        let mut rem = li;
                        let mut skip_buf = [0u8; 65536];
                        while rem > 0 { let c = rem.min(skip_buf.len()); reader.read_exact(&mut skip_buf[..c]).ok()?; rem -= c; }
                        if na != 2 { continue; }
                        // skip_gt: return position info without dosages
                        let chrom = if ci < contig_names.len() { contig_names[ci].as_bytes().to_vec() } else { format!("{}", ci).into_bytes() };
                        let _nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
                        let mut o = 24usize;
                        let _id = rtstr(sb, &mut o);
                        let mut alleles = Vec::with_capacity(na);
                        for _ in 0..na { alleles.push(rtstr_bytes(sb, &mut o)); }
                        ds_buf.clear();
                        return Some((chrom, pos, alleles.first().cloned().unwrap_or_default(), alleles.get(1).cloned().unwrap_or_default()));
                    }

                    ib.resize(li, 0); reader.read_exact(ib).ok()?;

                    let chrom = if ci < contig_names.len() { contig_names[ci].as_bytes().to_vec() } else { format!("{}", ci).into_bytes() };

                    // Parse alleles from shared data
                    let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
                    let mut o = 24usize;
                    let _id = rtstr(sb, &mut o);
                    let mut alleles = Vec::with_capacity(na);
                    for _ in 0..na { alleles.push(rtstr_bytes(sb, &mut o)); }
                    let ref_a = alleles.first().cloned().unwrap_or_default();
                    let alt_a = alleles.get(1).cloned().unwrap_or_default();

                    // Parse FORMAT fields — look for DS first, fallback to GT
                    ds_buf.clear();
                    let mut io2 = 0usize;
                    let mut found_ds = false;
                    let mut found_gt = false;

                    for _ in 0..nf {
                        if io2 >= ib.len() { break; }
                        let k = rtint(ib, &mut io2) as u16;
                        if io2 >= ib.len() { break; }
                        let tb = ib[io2]; io2 += 1;
                        let tid = tb & 0x0F;
                        let vl = { let r = (tb >> 4) as usize; if r == 15 { rtint(ib, &mut io2) as usize } else { r } };
                        let es = match tid { 1=>1, 2=>2, 3=>4, 5=>4, 7=>1, _=>1 };
                        let fs = vl * es * ns;

                        if dsk == Some(k) && tid == 5 && vl == 1 {
                            // DS field: float32 per sample (selective if filter set).
                            // Clear any hardcalls previously pushed by the GT branch —
                            // DS is authoritative when present.
                            ds_buf.clear();
                            for_each_sample(si_filter, ns, |si| {
                                let off = io2 + si * 4;
                                if off + 4 <= ib.len() {
                                    ds_buf.push(f32::from_le_bytes(ib[off..off+4].try_into().unwrap()));
                                } else { ds_buf.push(-1.0); }
                            });
                            found_ds = true;
                            // io2 not advanced past DS field — break exits the field loop
                            break;
                        } else if k == gtk && !found_ds {
                            // GT field: int8 per sample × ploidy (selective if filter set)
                            let ge = (io2 + fs).min(ib.len());
                            // Missing alleles are folded to 0 (hom-ref) to keep
                            // MAF denominators aligned with the SRP truth reader
                            // (1-bit-per-allele has no missing encoding).
                            for_each_sample(si_filter, ns, |si| {
                                let b = io2 + si * vl * es;
                                if b + 1 < ge {
                                    let a0c = gt_allele_to_dose(ib[b]);
                                    let a1c = gt_allele_to_dose(ib[b+1]);
                                    ds_buf.push(a0c as f32 + a1c as f32);
                                } else { ds_buf.push(0.0); }
                            });
                            found_gt = true;
                            io2 += fs;
                        } else {
                            io2 += fs;
                        }
                    }

                    let expected_n = if let Some(f) = si_filter { f.len() } else { ns };
                    if !found_ds && !found_gt {
                        ds_buf.clear();
                        ds_buf.resize(expected_n, -1.0);
                    }
                    while ds_buf.len() < expected_n { ds_buf.push(-1.0); }

                    return Some((chrom, pos, ref_a, alt_a));
                }
            }
        }
    }
}

/// Decode one BCF int8 GT allele byte into a biallelic ALT indicator (0 or 1).
///
/// BCF GT encoding (int8): each allele byte = `(allele + 1) << 1 | phased`, so
/// `allele = (raw >> 1) - 1` and the low bit is the phase. A missing allele
/// (`allele = -1`) is byte `0x00`/`0x01` → `(raw>>1)-1` wraps to 255.
/// The int8 **missing value** `0x80` (-128) and **end-of-vector** pad `0x81`
/// (-127, used to pad haploid samples in a diploid-ploidy record, e.g. chrX)
/// are SENTINELS, not alleles: `(0x80>>1)-1 = 63` would otherwise mis-decode to
/// ALT. All of {missing allele, 0x80, 0x81} fold to 0 (no ALT contribution);
/// any present ALT (`allele >= 1`, incl. multiallelic) folds to 1.
#[inline]
fn gt_allele_to_dose(raw: u8) -> u8 {
    if raw == 0x80 || raw == 0x81 { return 0; }
    let a = (raw >> 1).wrapping_sub(1);
    if a > 127 { 0 } else { a.min(1) }
}

/// Invoke `f(sample_index)` for each kept sample, in order: the filtered
/// positions in `si_filter` when present (selective BCF sample extraction),
/// otherwise every sample `0..ns`. Folds the byte-identical filter/no-filter
/// branch pairs in the BCF DS and GT extraction loops into one body.
#[inline]
fn for_each_sample(si_filter: Option<&[usize]>, ns: usize, mut f: impl FnMut(usize)) {
    if let Some(indices) = si_filter {
        for &si in indices { f(si); }
    } else {
        for si in 0..ns { f(si); }
    }
}

// BCF typed-atom parsers shared with srp::bcf_reader + srp::bref3_writer.
use crate::srp::bcf_types::{read_typed_i32 as rtint, read_typed_str_bytes as rtstr_bytes};

/// Parse a BCF typed string, with a lossy UTF-8 conversion (eval-side behavior).
fn rtstr(buf: &[u8], o: &mut usize) -> String {
    String::from_utf8_lossy(&rtstr_bytes(buf, o)).to_string()
}

/// Load 16 kb checkpoint positions (0-based) from the file's seek index.
/// Returns None if no index is present next to the file.
fn load_checkpoint_positions(path: &Path) -> Option<Vec<i64>> {
    let s = path.to_string_lossy();
    if s.ends_with(".bcf") || s.ends_with(".bcf.gz") {
        let csi_path = {
            let mut p = path.as_os_str().to_owned();
            p.push(".csi");
            std::path::PathBuf::from(p)
        };
        if !csi_path.exists() { return None; }
        let csi = crate::srp::csi::parse_csi(&csi_path).ok()?;
        let mut positions: Vec<i64> = csi.checkpoints.iter().map(|&(pos, _)| pos).collect();
        positions.sort();
        positions.dedup();
        if positions.is_empty() { return None; }
        Some(positions)
    } else if s.ends_with(".vcf.gz") {
        let tbi_path = {
            let mut p = path.as_os_str().to_owned();
            p.push(".tbi");
            std::path::PathBuf::from(p)
        };
        if !tbi_path.exists() { return None; }
        let tbi = crate::srp::csi::parse_tbi(&tbi_path).ok()?;
        // TBI linear index: one vpos per 16 kbp interval. Non-zero entries
        // mark intervals with records. Use the contig that actually has data.
        let linear = tbi.linear.iter().find(|l| !l.is_empty())?;
        let positions: Vec<i64> = linear.iter().enumerate()
            .filter(|&(_, &v)| v != 0)
            .map(|(i, _)| (i as i64) * 16384)
            .collect();
        if positions.is_empty() { return None; }
        Some(positions)
    } else {
        None
    }
}

/// Build parallel evaluation regions. Prefers variant-balanced split
/// (equal number of 16 kb checkpoints per region) using the imputed file's
/// seek index; falls back to equal-bp split when no index is available.
fn build_eval_regions(imputed_path: &Path, n_regions: usize) -> Vec<(i64, i64)> {
    let n_regions = n_regions.max(1);

    if let Some(cp) = load_checkpoint_positions(imputed_path) {
        // Spread checkpoints across regions. Each region covers `stride`
        // consecutive checkpoints, giving approximately equal record count
        // since each 16 kb checkpoint represents a similar number of records
        // in BCF-sorted imputation output.
        let n_cp = cp.len();
        let regions_count = n_regions.min(n_cp);
        let mut regions = Vec::with_capacity(regions_count);
        for i in 0..regions_count {
            let start_idx = (i * n_cp) / regions_count;
            let start = if i == 0 { 1 } else { cp[start_idx] + 1 };
            let end = if i + 1 == regions_count {
                i64::MAX
            } else {
                let end_idx = ((i + 1) * n_cp) / regions_count;
                cp[end_idx] + 1
            };
            regions.push((start, end));
        }
        return regions;
    }

    // Fallback: equal-bp split across a human-genome-sized interval.
    let chunk_bp = 300_000_000i64 / n_regions as i64;
    (0..n_regions)
        .map(|i| {
            let start = i as i64 * chunk_bp + 1;
            let end = if i + 1 == n_regions {
                i64::MAX
            } else {
                (i as i64 + 1) * chunk_bp + 1
            };
            (start, end)
        })
        .collect()
}

/// Parallel evaluation: split by genomic region, one thread per region.
/// Each thread opens its own file handles and evaluates its portion independently.
/// Make sure a seek-enabling index exists for a VCF.gz or BCF file, so the
/// parallel evaluator can skip to its assigned region via pread instead of
/// decompressing the whole file in every thread. No-op when an index already
/// exists next to the file.
fn ensure_seek_index(path: &Path) -> io::Result<()> {
    let s = path.to_string_lossy();
    let is_bcf = s.ends_with(".bcf") || s.ends_with(".bcf.gz");
    if is_bcf {
        let csi = { let mut p = path.as_os_str().to_owned(); p.push(".csi"); std::path::PathBuf::from(p) };
        if !csi.exists() {
            crate::selphi_info!("  Building CSI index for parallel eval: {}", path.display());
            crate::srp::csi::build_csi_index(path).map_err(|e|
                io::Error::other(format!("CSI build failed: {}", e)))?;
        }
    } else if s.ends_with(".vcf.gz") {
        let tbi = { let mut p = path.as_os_str().to_owned(); p.push(".tbi"); std::path::PathBuf::from(p) };
        if !tbi.exists() {
            crate::selphi_info!("  Building TBI index for parallel eval: {}", path.display());
            crate::srp::csi::build_tbi_index(path)?;
        }
    }
    Ok(())
}

pub fn evaluate_parallel(
    imputed_path: &Path,
    truth_path: &Path,
    shared_samples: &[String],
    n_threads: usize,
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    use rayon::prelude::*;

    let n_samples = shared_samples.len();

    // Ensure parallel-seek indexes exist so each thread can pread-seek its
    // region instead of re-scanning from BOF. Without this, 16 threads each
    // decompress the whole 10+ GB file → pathologically slow.
    ensure_seek_index(imputed_path)?;
    ensure_seek_index(truth_path)?;

    // Get sample info
    let (_, imp_samples) = VariantReader::open(imputed_path)?;
    let (_, truth_samples) = VariantReader::open(truth_path)?;
    let imp_map: std::collections::HashMap<&str, usize> = imp_samples.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let truth_map: std::collections::HashMap<&str, usize> = truth_samples.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let imp_reindex: Vec<usize> = shared_samples.iter().map(|s| *imp_map.get(s.as_str()).expect("shared sample missing from imputed file")).collect();
    let truth_reindex: Vec<usize> = shared_samples.iter().map(|s| *truth_map.get(s.as_str()).expect("shared sample missing from truth file")).collect();

    // Build variant-balanced regions from the imputed file's index.
    //
    // Previous code sliced the genome into equal-bp chunks (300 Mb / n_regions).
    // On chr20 (61 Mb of records) that left 50/64 regions empty and concentrated
    // all work in ~14 regions with very uneven record counts — the slowest
    // region was 1.48× heavier than the fastest, bounding wall at ~269 s on
    // MESA 5k despite 3072 s of aggregate CPU work.
    //
    // Now we use the index's 16 kb checkpoints (CSI for BCF, TBI linear for
    // VCF.gz) as approximate record-density markers. Each region covers an
    // equal number of checkpoints → roughly equal record count. Falls back to
    // equal-bp splitting if no index is available.
    let n_regions = n_threads * 4;
    let regions = build_eval_regions(imputed_path, n_regions);

    let imp_path = imputed_path.to_path_buf();
    let truth_path_buf = truth_path.to_path_buf();

    let results: Vec<(SiteAccumulator, SampleAccumulator, EvalCounts, f64)> = regions
        .par_iter()
        .map(|&(region_start, region_end)| {
            let t0 = std::time::Instant::now();
            // Each thread: open files, scan to region, evaluate with full genotypes
            let result = (|| -> io::Result<_> {
                let (mut imp_reader, _) = VariantReader::open(&imp_path)?;
                let (mut truth_reader, _) = VariantReader::open(&truth_path_buf)?;
                imp_reader.set_sample_filter(imp_reindex.clone());
                truth_reader.set_sample_filter(truth_reindex.clone());

                // Seek to region start via CSI/TBI index (O(1) instead of scanning)
                imp_reader.seek_to_position(region_start)?;
                truth_reader.seek_to_position(region_start)?;

                let mut site_acc = SiteAccumulator::new();
                let mut sample_acc = SampleAccumulator::new(n_samples);
                let mut n_matched = 0u64;
                let mut n_imp = 0u64;
                let mut n_truth = 0u64;
                let mut chr_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

                let mut imp_ds_raw = Vec::with_capacity(n_samples);
                let mut truth_ds_raw = Vec::with_capacity(n_samples);
                let mut imp_ds = vec![0.0f32; n_samples];
                let mut truth_ds = vec![0.0f32; n_samples];

                // Read first record in region (skip any before region_start from seek imprecision)
                let mut imp_rec = loop {
                    let r = imp_reader.next_record(&mut imp_ds_raw);
                    match &r {
                        Some(rec) if rec.1 < region_start => continue,
                        _ => break r,
                    }
                };
                let mut truth_rec = loop {
                    let r = truth_reader.next_record(&mut truth_ds_raw);
                    match &r {
                        Some(rec) if rec.1 < region_start => continue,
                        _ => break r,
                    }
                };

                // Merge within region. A record is "owned" by this region iff
                // its pos is in [region_start, region_end). The skip loop above
                // already discarded pos < region_start. The merge advances the
                // smaller side; counters increment only while pos < region_end.
                while let (Some(imp), Some(truth)) = (&imp_rec, &truth_rec) {
                    if imp.1 >= region_end && truth.1 >= region_end { break; }

                    if imp.1 < truth.1 {
                        if imp.1 < region_end { n_imp += 1; }
                        imp_rec = imp_reader.next_record(&mut imp_ds_raw);
                    } else if imp.1 > truth.1 {
                        if truth.1 < region_end { n_truth += 1; }
                        truth_rec = truth_reader.next_record(&mut truth_ds_raw);
                    } else if imp.1 < region_end {
                        // Same position, in-region.
                        n_imp += 1;
                        n_truth += 1;

                        let (matched, swapped) = match_alleles(&imp.2, &imp.3, &truth.2, &truth.3);
                        if matched {
                            chr_set.insert(String::from_utf8_lossy(&imp.0).to_string());
                            imp_ds[..n_samples].copy_from_slice(&imp_ds_raw[..n_samples]);
                            if swapped { for si in 0..n_samples { if imp_ds[si] >= 0.0 { imp_ds[si] = 2.0 - imp_ds[si]; } } }
                            truth_ds[..n_samples].copy_from_slice(&truth_ds_raw[..n_samples]);

                            let mut gt_sum = 0.0f64;
                            let mut gt_n = 0u32;
                            for s in 0..n_samples {
                                if truth_ds[s] >= 0.0 { gt_sum += truth_ds[s] as f64; gt_n += 1; }
                            }
                            let maf = if gt_n > 0 { let af = gt_sum / (gt_n as f64 * 2.0); af.min(1.0 - af) } else { 0.0 };
                            let (r2, conc) = site_r2(&imp_ds, &truth_ds, n_samples);
                            site_acc.add(maf, r2, conc);
                            sample_acc.add_variant(&imp_ds, &truth_ds, maf);
                            n_matched += 1;
                        }

                        imp_rec = imp_reader.next_record(&mut imp_ds_raw);
                        truth_rec = truth_reader.next_record(&mut truth_ds_raw);
                    } else {
                        // Same position, both past region_end.
                        break;
                    }
                }

                // Drain remaining in-region records when the other side hit
                // EOF. Without this, imp/truth records left over after a
                // premature None on one side were silently dropped from the
                // displayed counts (R² unaffected, but totals under-reported).
                while let Some(ref rec) = imp_rec {
                    if rec.1 >= region_end { break; }
                    n_imp += 1;
                    imp_rec = imp_reader.next_record(&mut imp_ds_raw);
                }
                while let Some(ref rec) = truth_rec {
                    if rec.1 >= region_end { break; }
                    n_truth += 1;
                    truth_rec = truth_reader.next_record(&mut truth_ds_raw);
                }

                Ok((site_acc, sample_acc, EvalCounts { n_matched, n_imp_variants: n_imp, n_truth_variants: n_truth, chromosomes: chr_set.into_iter().collect() }))
            })();
            let wall = t0.elapsed().as_secs_f64();
            match result {
                Ok((sa, sm, ec)) => (sa, sm, ec, wall),
                Err(_) => (SiteAccumulator::new(), SampleAccumulator::new(n_samples), EvalCounts { n_matched: 0, n_imp_variants: 0, n_truth_variants: 0, chromosomes: Vec::new() }, wall),
            }
        })
        .collect();

    // Log the slowest / fastest / median region wall time under --debug so
    // the imbalance between regions is visible without spamming on every run.
    {
        let mut walls: Vec<(f64, u64)> = results.iter()
            .map(|(_, _, ec, w)| (*w, ec.n_matched))
            .collect();
        walls.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let total_wall: f64 = walls.iter().map(|(w, _)| w).sum();
        let slowest = walls.first().copied().unwrap_or((0.0, 0));
        let fastest = walls.iter().rev().find(|(_, n)| *n > 0).copied().unwrap_or((0.0, 0));
        let median = walls.get(walls.len() / 2).copied().unwrap_or((0.0, 0));
        let n_nonempty = walls.iter().filter(|(_, n)| *n > 0).count();
        crate::selphi_debug!(
            "  eval regions: {} nonempty of {}, slowest={:.2}s ({} vars), median={:.2}s ({} vars), fastest={:.2}s ({} vars), sum={:.1}s",
            n_nonempty, walls.len(),
            slowest.0, slowest.1, median.0, median.1, fastest.0, fastest.1, total_wall,
        );
    }

    // Merge all region results
    let mut site_acc = SiteAccumulator::new();
    let mut sample_acc = SampleAccumulator::new(n_samples);
    let mut total = EvalCounts { n_matched: 0, n_imp_variants: 0, n_truth_variants: 0, chromosomes: Vec::new() };
    let mut all_chrs: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for (sa, sam, c, _wall) in results {
        site_acc.merge(&sa);
        sample_acc.merge(&sam);
        total.n_matched += c.n_matched;
        total.n_imp_variants += c.n_imp_variants;
        total.n_truth_variants += c.n_truth_variants;
        for chr in c.chromosomes { all_chrs.insert(chr); }
    }
    total.chromosomes = all_chrs.into_iter().collect();

    Ok((site_acc, sample_acc, total))
}

/// Stream-merge and evaluate imputed vs truth. Supports VCF.gz and BCF.
/// Uses parallel evaluation (multiple threads, each processes a genomic region).
pub fn evaluate(
    imputed_path: &Path,
    truth_path: &Path,
    shared_samples: &[String],
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    let n_threads = rayon::current_num_threads().max(1);
    evaluate_parallel(imputed_path, truth_path, shared_samples, n_threads)
}

/// Print MAF-binned summary table.
pub fn print_summary(site_acc: &SiteAccumulator, sample_acc: &SampleAccumulator, counts: &EvalCounts) {
    let n_matched = counts.n_matched;
    crate::selphi_info!("\n{:<20} {:>12} {:>10} {:>12}", "MAF bin", "N variants", "Mean R²", "Concordance");
    crate::selphi_info!("{}", "-".repeat(60));

    for (bi, &(_, _, label)) in MAF_BINS.iter().enumerate() {
        let n = site_acc.bin_n[bi];
        if n == 0 {
            crate::selphi_info!("{:<20} {:>12} {:>10} {:>12}", label, "0", "N/A", "N/A");
        } else {
            let mean_r2 = site_acc.bin_r2_sum[bi] / n as f64;
            let mean_conc = site_acc.bin_conc_sum[bi] / n as f64;
            crate::selphi_info!("{:<20} {:>12} {:>10.4} {:>12.4}", label, n, mean_r2, mean_conc);
        }
    }

    crate::selphi_info!("{}", "-".repeat(60));
    if site_acc.total_n > 0 {
        let overall_r2 = if site_acc.total_r2_n > 0 { site_acc.total_r2_sum / site_acc.total_r2_n as f64 } else { 0.0 };
        let overall_conc = site_acc.total_conc_sum / site_acc.total_n as f64;
        crate::selphi_info!("{:<20} {:>12} {:>10.4} {:>12.4}", "OVERALL", n_matched, overall_r2, overall_conc);
    }

    // Per-sample summary
    let sample_r2 = sample_acc.compute_r2();
    let sample_conc = sample_acc.compute_concordance();
    if !sample_r2.is_empty() {
        let mean_r2: f64 = sample_r2.iter().sum::<f64>() / sample_r2.len() as f64;
        let min_r2 = sample_r2.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r2 = sample_r2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_conc: f64 = sample_conc.iter().sum::<f64>() / sample_conc.len() as f64;
        crate::selphi_info!("\nPer-sample (n={}):", sample_acc.n_samples);
        crate::selphi_info!("  R²:          mean={:.6}, min={:.6}, max={:.6}", mean_r2, min_r2, max_r2);
        crate::selphi_info!("  Concordance: mean={:.6}", mean_conc);
    }

    let match_pct = if counts.n_truth_variants > 0 { 100.0 * n_matched as f64 / counts.n_truth_variants as f64 } else { 0.0 };
    crate::selphi_info!("\n  Imputed:  {} variants", counts.n_imp_variants);
    crate::selphi_info!("  Truth:    {} variants", counts.n_truth_variants);
    crate::selphi_info!("  Matched:  {} variants ({:.1}% of truth)", n_matched, match_pct);
    if !counts.chromosomes.is_empty() {
        crate::selphi_info!("  Chromosomes: {} ({})", counts.chromosomes.len(),
            counts.chromosomes.join(", "));
    }
}

/// Write JSON summary to file.
pub fn write_json_summary(
    path: &Path, site_acc: &SiteAccumulator, sample_acc: &SampleAccumulator, counts: &EvalCounts,
    sample_names: Option<&[String]>,
) -> io::Result<()> {
    use std::io::Write;
    let mut map = serde_json::Map::new();

    for (bi, &(_, _, label)) in MAF_BINS.iter().enumerate() {
        let n = site_acc.bin_n[bi];
        let mut bin = serde_json::Map::new();
        bin.insert("n".into(), serde_json::json!(n));
        if n > 0 {
            bin.insert("mean_r2".into(), serde_json::json!((site_acc.bin_r2_sum[bi] / n as f64 * 1e6).round() / 1e6));
            bin.insert("concordance".into(), serde_json::json!((site_acc.bin_conc_sum[bi] / n as f64 * 1e6).round() / 1e6));
        }
        map.insert(label.to_string(), serde_json::Value::Object(bin));
    }

    if site_acc.total_n > 0 {
        let mut overall = serde_json::Map::new();
        overall.insert("n".into(), serde_json::json!(counts.n_matched));
        let r2 = if site_acc.total_r2_n > 0 { site_acc.total_r2_sum / site_acc.total_r2_n as f64 } else { 0.0 };
        overall.insert("mean_r2".into(), serde_json::json!((r2 * 1e6).round() / 1e6));
        overall.insert("concordance".into(), serde_json::json!((site_acc.total_conc_sum / site_acc.total_n as f64 * 1e6).round() / 1e6));
        map.insert("overall".into(), serde_json::Value::Object(overall));
    }

    map.insert("n_imp_variants".into(), serde_json::json!(counts.n_imp_variants));
    map.insert("n_truth_variants".into(), serde_json::json!(counts.n_truth_variants));
    map.insert("n_matched".into(), serde_json::json!(counts.n_matched));
    map.insert("n_samples".into(), serde_json::json!(sample_acc.n_samples));

    let sample_r2 = sample_acc.compute_r2();
    let sample_conc = sample_acc.compute_concordance();
    if !sample_r2.is_empty() {
        let mean: f64 = sample_r2.iter().sum::<f64>() / sample_r2.len() as f64;
        map.insert("per_sample_mean_r2".into(), serde_json::json!((mean * 1e6).round() / 1e6));

        // Per-sample detail array
        let per_sample: Vec<serde_json::Value> = (0..sample_acc.n_samples).map(|s| {
            let mut obj = serde_json::Map::new();
            if let Some(names) = sample_names {
                obj.insert("sample".into(), serde_json::json!(names[s]));
            }
            obj.insert("r2".into(), serde_json::json!((sample_r2[s] * 1e6).round() / 1e6));
            obj.insert("concordance".into(), serde_json::json!((sample_conc[s] * 1e6).round() / 1e6));
            obj.insert("n_correct".into(), serde_json::json!(sample_acc.n_correct[s]));
            obj.insert("n_total".into(), serde_json::json!(sample_acc.n_total[s]));
            serde_json::Value::Object(obj)
        }).collect();
        map.insert("per_sample".into(), serde_json::json!(per_sample));
    }

    let json = serde_json::Value::Object(map);
    let mut f = std::fs::File::create(path)?;
    f.write_all(serde_json::to_string_pretty(&json)?.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::gt_allele_to_dose;

    #[test]
    fn bcf_gt_allele_decode_covers_sentinels() {
        // Encoding: allele byte = (allele+1)<<1 | phased. Low bit = phase.
        // REF (allele 0): 0x02 unphased, 0x03 phased -> dose 0.
        assert_eq!(gt_allele_to_dose(0x02), 0);
        assert_eq!(gt_allele_to_dose(0x03), 0);
        // ALT (allele 1): 0x04 unphased, 0x05 phased -> dose 1.
        assert_eq!(gt_allele_to_dose(0x04), 1);
        assert_eq!(gt_allele_to_dose(0x05), 1);
        // Multiallelic ALT (allele 2): 0x06 -> folds to 1 (biallelic indicator).
        assert_eq!(gt_allele_to_dose(0x06), 1);
        // Missing allele (allele -1): byte 0x00/0x01 -> dose 0.
        assert_eq!(gt_allele_to_dose(0x00), 0);
        assert_eq!(gt_allele_to_dose(0x01), 0);
        // THE BUG: int8 missing sentinel 0x80 and end-of-vector pad 0x81 must
        // fold to 0, NOT decode to allele (0x80>>1)-1 = 63 -> 1.
        assert_eq!(gt_allele_to_dose(0x80), 0);
        assert_eq!(gt_allele_to_dose(0x81), 0);
    }
}
