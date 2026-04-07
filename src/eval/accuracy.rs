//! Stream-merge imputed + truth VCF/BCF, compute per-site and per-sample accuracy.
//!
//! Metrics:
//!   - Per-site R² (Pearson correlation² between dosage and truth genotype)
//!   - Per-site concordance (fraction of matching hardcalls)
//!   - Per-sample R² (across all variants for each sample)
//!   - MAF-binned R² (paper-style MAF bins)
//!
//! Designed for 50K+ samples: O(n_samples) memory per variant, streaming.

use std::io::{self, BufRead, BufReader};
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

/// Per-site result for one variant.
pub struct SiteResult {
    pub maf: f64,
    pub r2: f64,
    pub concordance: f64,
    pub n_samples: usize,
}

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

            // MAF bin errors
            for (bi, &(lo, hi, _)) in MAF_BINS.iter().enumerate() {
                if maf >= lo && maf < hi {
                    if d_call != g_call { self.bin_errors[bi][s] += 1; }
                    break;
                }
            }
        }

        // Track bin variant counts
        for (bi, &(lo, hi, _)) in MAF_BINS.iter().enumerate() {
            if maf >= lo && maf < hi { self.bin_n[bi] += 1; break; }
        }
    }

    /// Compute per-sample R² from accumulated statistics.
    pub fn compute_r2(&self) -> Vec<f64> {
        (0..self.n_samples).map(|s| {
            let n = self.n_total[s] as f64;
            if n < 2.0 { return 0.0; }
            let num = n * self.sum_ds_gt[s] - self.sum_ds[s] * self.sum_gt[s];
            let den_x = n * self.sum_ds2[s] - self.sum_ds[s] * self.sum_ds[s];
            let den_y = n * self.sum_gt2[s] - self.sum_gt[s] * self.sum_gt[s];
            let den = (den_x * den_y).max(0.0).sqrt();
            if den > 0.0 { (num / den).powi(2) } else { 0.0 }
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
    /// Concordance always counted. R² only counted when not NaN (non-zero variance).
    pub fn add(&mut self, maf: f64, r2: f64, concordance: f64) {
        // Concordance counts for ALL variants (including monomorphic)
        self.total_conc_sum += concordance;
        self.total_n += 1;

        // R² and MAF-binned metrics only for variants with valid R²
        if !r2.is_nan() {
            self.total_r2_sum += r2;
            self.total_r2_n += 1;
            for (bi, &(lo, hi, _)) in MAF_BINS.iter().enumerate() {
                if maf >= lo && maf < hi {
                    self.bin_r2_sum[bi] += r2;
                    self.bin_conc_sum[bi] += concordance;
                    self.bin_n[bi] += 1;
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

/// Parse a VCF/BCF text line to extract dosage for all samples.
/// Returns (chrom, pos, ref, alt, dosages) where dosages[i] is DS for sample i.
/// If DS not available, uses GT (0/0→0, 0/1→1, 1/1→2).
/// Returns None for header lines or multi-allelic.
pub fn parse_vcf_line(line: &[u8], n_samples: usize, ds_buf: &mut Vec<f32>) -> Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> {
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
            // Fall back to GT
            if let Some(gt) = sample_field.split(|&b| b == b':').nth(gi) {
                if gt.len() >= 3 {
                    let a0 = if gt[0] == b'.' { -1i32 } else { (gt[0] - b'0') as i32 };
                    let a1 = if gt[2] == b'.' { -1i32 } else { (gt[2] - b'0') as i32 };
                    if a0 < 0 || a1 < 0 { ds_buf.push(-1.0); }
                    else { ds_buf.push((a0 + a1) as f32); }
                } else {
                    ds_buf.push(-1.0);
                }
            } else {
                ds_buf.push(-1.0);
            }
        } else {
            ds_buf.push(-1.0);
        }
    }

    // Pad if needed
    while ds_buf.len() < n_samples { ds_buf.push(-1.0); }

    Some((chrom, pos, ref_a, alt_a))
}

/// Parse VCF/BCF header to get sample names.
pub fn parse_header_samples(path: &Path) -> io::Result<Vec<String>> {
    let is_bcf = path.to_string_lossy().ends_with(".bcf");

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
pub struct EvalCounts {
    pub n_matched: u64,
    pub n_imp_variants: u64,
    pub n_truth_variants: u64,
}

/// Stream-merge and evaluate imputed vs truth.
#[allow(unused_assignments)]
pub fn evaluate_stream(
    imputed_path: &Path,
    truth_path: &Path,
    shared_samples: &[String],
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    let n_samples = shared_samples.len();

    // Open both files as BGZF text streams
    let imp_file = std::fs::File::open(imputed_path)?;
    let truth_file = std::fs::File::open(truth_path)?;

    let is_imp_bcf = imputed_path.to_string_lossy().ends_with(".bcf");
    let is_truth_bcf = truth_path.to_string_lossy().ends_with(".bcf");

    // For BCF: decompress to get text lines won't work. Use VCF text path.
    // For BCF input, convert via bcftools or read raw.
    // For now: support VCF.gz only in the evaluator. BCF support via noodles later.
    if is_imp_bcf || is_truth_bcf {
        return Err(io::Error::new(io::ErrorKind::Unsupported,
            "BCF input not yet supported in native evaluator. Use VCF.gz or convert with: bcftools view -Oz"));
    }

    let imp_bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, imp_file));
    let truth_bgzf = noodles_bgzf::io::Reader::new(BufReader::with_capacity(4 << 20, truth_file));
    let mut imp_reader = BufReader::new(imp_bgzf);
    let mut truth_reader = BufReader::new(truth_bgzf);

    // Build sample reindex maps
    let imp_samples = {
        let mut samples = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            imp_reader.read_line(&mut line)?;
            if line.starts_with("#CHROM") {
                let fields: Vec<&str> = line.trim().split('\t').collect();
                if fields.len() > 9 { samples = fields[9..].iter().map(|s| s.to_string()).collect(); }
                break;
            }
            if line.is_empty() { break; }
        }
        samples
    };
    let truth_samples = {
        let mut samples = Vec::new();
        let mut line = String::new();
        loop {
            line.clear();
            truth_reader.read_line(&mut line)?;
            if line.starts_with("#CHROM") {
                let fields: Vec<&str> = line.trim().split('\t').collect();
                if fields.len() > 9 { samples = fields[9..].iter().map(|s| s.to_string()).collect(); }
                break;
            }
            if line.is_empty() { break; }
        }
        samples
    };

    // Build reindex: shared_samples order → file column index
    let imp_map: std::collections::HashMap<&str, usize> = imp_samples.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let truth_map: std::collections::HashMap<&str, usize> = truth_samples.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    let imp_reindex: Vec<usize> = shared_samples.iter().map(|s| imp_map[s.as_str()]).collect();
    let truth_reindex: Vec<usize> = shared_samples.iter().map(|s| truth_map[s.as_str()]).collect();

    let n_imp_samples = imp_samples.len();
    let n_truth_samples = truth_samples.len();

    let mut site_acc = SiteAccumulator::new();
    let mut sample_acc = SampleAccumulator::new(n_samples);
    let mut n_matched = 0u64;
    let mut n_imp_variants = 0u64;
    let mut n_truth_variants = 0u64;

    let mut imp_line = Vec::with_capacity(n_imp_samples * 8);
    let mut truth_line = Vec::with_capacity(n_truth_samples * 8);
    let mut imp_ds_raw = Vec::with_capacity(n_imp_samples);
    let mut truth_ds_raw = Vec::with_capacity(n_truth_samples);
    let mut imp_ds = vec![0.0f32; n_samples];
    let mut truth_ds = vec![0.0f32; n_samples];

    // Current positions
    let mut imp_rec: Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> = None;
    let mut truth_rec: Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> = None;

    // Read first records
    let read_next_imp = |reader: &mut BufReader<_>, line: &mut Vec<u8>, ds: &mut Vec<f32>, ns: usize| -> Option<(Vec<u8>, i64, Vec<u8>, Vec<u8>)> {
        loop {
            line.clear();
            if reader.read_until(b'\n', line).ok()? == 0 { return None; }
            if let Some(rec) = parse_vcf_line(line, ns, ds) { return Some(rec); }
        }
    };

    imp_rec = read_next_imp(&mut imp_reader, &mut imp_line, &mut imp_ds_raw, n_imp_samples);
    if imp_rec.is_some() { n_imp_variants += 1; }
    truth_rec = read_next_imp(&mut truth_reader, &mut truth_line, &mut truth_ds_raw, n_truth_samples);
    if truth_rec.is_some() { n_truth_variants += 1; }

    while let (Some(imp), Some(truth)) = (&imp_rec, &truth_rec) {
        let imp_pos = imp.1;
        let truth_pos = truth.1;

        if imp_pos < truth_pos {
            imp_rec = read_next_imp(&mut imp_reader, &mut imp_line, &mut imp_ds_raw, n_imp_samples);
            if imp_rec.is_some() { n_imp_variants += 1; }
            continue;
        }
        if imp_pos > truth_pos {
            truth_rec = read_next_imp(&mut truth_reader, &mut truth_line, &mut truth_ds_raw, n_truth_samples);
            if truth_rec.is_some() { n_truth_variants += 1; }
            continue;
        }

        // Same position — match by alleles (handles indel normalization + REF/ALT swaps)
        let (matched, swapped) = match_alleles(&imp.2, &imp.3, &truth.2, &truth.3);
        if matched {
            for (si, &ii) in imp_reindex.iter().enumerate() {
                let v = imp_ds_raw[ii];
                imp_ds[si] = if swapped && v >= 0.0 { 2.0 - v } else { v };
            }
            for (si, &ti) in truth_reindex.iter().enumerate() { truth_ds[si] = truth_ds_raw[ti]; }

            // Compute MAF from truth
            let mut gt_sum = 0.0f64;
            let mut gt_n = 0u32;
            for s in 0..n_samples {
                if truth_ds[s] >= 0.0 { gt_sum += truth_ds[s] as f64; gt_n += 1; }
            }
            let maf = if gt_n > 0 {
                let af = gt_sum / (gt_n as f64 * 2.0);
                af.min(1.0 - af)
            } else { 0.0 };

            // Per-site R² and concordance
            let (r2, conc) = site_r2(&imp_ds, &truth_ds, n_samples);
            site_acc.add(maf, r2, conc);

            // Per-sample accumulation
            sample_acc.add_variant(&imp_ds, &truth_ds, maf);

            n_matched += 1;
        }

        // Advance both
        imp_rec = read_next_imp(&mut imp_reader, &mut imp_line, &mut imp_ds_raw, n_imp_samples);
        if imp_rec.is_some() { n_imp_variants += 1; }
        truth_rec = read_next_imp(&mut truth_reader, &mut truth_line, &mut truth_ds_raw, n_truth_samples);
        if truth_rec.is_some() { n_truth_variants += 1; }
    }

    // Count remaining variants after merge loop ends
    while { imp_rec = read_next_imp(&mut imp_reader, &mut imp_line, &mut imp_ds_raw, n_imp_samples); imp_rec.is_some() } { n_imp_variants += 1; }
    while { truth_rec = read_next_imp(&mut truth_reader, &mut truth_line, &mut truth_ds_raw, n_truth_samples); truth_rec.is_some() } { n_truth_variants += 1; }

    Ok((site_acc, sample_acc, EvalCounts { n_matched, n_imp_variants, n_truth_variants }))
}

/// Print summary table (matches Python output format).
pub fn print_summary(site_acc: &SiteAccumulator, sample_acc: &SampleAccumulator, counts: &EvalCounts) {
    let n_matched = counts.n_matched;
    eprintln!("\n{:<20} {:>12} {:>10} {:>12}", "MAF bin", "N variants", "Mean R²", "Concordance");
    eprintln!("{}", "-".repeat(60));

    for (bi, &(_, _, label)) in MAF_BINS.iter().enumerate() {
        let n = site_acc.bin_n[bi];
        if n == 0 {
            eprintln!("{:<20} {:>12} {:>10} {:>12}", label, "0", "N/A", "N/A");
        } else {
            let mean_r2 = site_acc.bin_r2_sum[bi] / n as f64;
            let mean_conc = site_acc.bin_conc_sum[bi] / n as f64;
            eprintln!("{:<20} {:>12} {:>10.4} {:>12.4}", label, n, mean_r2, mean_conc);
        }
    }

    eprintln!("{}", "-".repeat(60));
    if site_acc.total_n > 0 {
        let overall_r2 = if site_acc.total_r2_n > 0 { site_acc.total_r2_sum / site_acc.total_r2_n as f64 } else { 0.0 };
        let overall_conc = site_acc.total_conc_sum / site_acc.total_n as f64;
        eprintln!("{:<20} {:>12} {:>10.4} {:>12.4}", "OVERALL", n_matched, overall_r2, overall_conc);
    }

    // Per-sample summary
    let sample_r2 = sample_acc.compute_r2();
    let sample_conc = sample_acc.compute_concordance();
    if !sample_r2.is_empty() {
        let mean_r2: f64 = sample_r2.iter().sum::<f64>() / sample_r2.len() as f64;
        let min_r2 = sample_r2.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_r2 = sample_r2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_conc: f64 = sample_conc.iter().sum::<f64>() / sample_conc.len() as f64;
        eprintln!("\nPer-sample (n={}):", sample_acc.n_samples);
        eprintln!("  R²:          mean={:.6}, min={:.6}, max={:.6}", mean_r2, min_r2, max_r2);
        eprintln!("  Concordance: mean={:.6}", mean_conc);
    }

    let match_pct = if counts.n_truth_variants > 0 { 100.0 * n_matched as f64 / counts.n_truth_variants as f64 } else { 0.0 };
    eprintln!("\n  Imputed:  {} variants", counts.n_imp_variants);
    eprintln!("  Truth:    {} variants", counts.n_truth_variants);
    eprintln!("  Matched:  {} variants ({:.1}% of truth)", n_matched, match_pct);
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
