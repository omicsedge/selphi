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

/// Units of a FORMAT/DS value. GT is always decoded in its native ploidy.
#[derive(Clone, Copy, Debug, Default)]
pub enum DsScale {
    /// DS is the expected ALT count in the sample's original ploidy.
    #[default]
    Native,
    /// DS is already on the diploid-equivalent [0, 2] scale.
    Diploid,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InputDosageOptions {
    pub ds_scale: DsScale,
    /// Used only when GT does not identify ploidy; never overrides a known GT ploidy.
    pub fallback_ploidy: Option<u32>,
}

/// Scoring uses diploid-equivalent dosages on both sides. Truth AF uses original
/// allele counts and ploidies. Raw truth, when supplied, uses `truth` options too.
#[derive(Clone, Copy, Debug, Default)]
pub struct EvaluationOptions {
    pub imputed: InputDosageOptions,
    pub truth: InputDosageOptions,
}

impl InputDosageOptions {
    fn validate(self) -> io::Result<()> {
        if self.fallback_ploidy == Some(0) {
            return Err(invalid_dosage("fallback ploidy must be greater than zero"));
        }
        Ok(())
    }
}

impl EvaluationOptions {
    fn validate(&self) -> io::Result<()> {
        self.imputed.validate()?;
        self.truth.validate()
    }
}

fn invalid_dosage(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Copy, Debug, Default)]
struct GtCall {
    alt_count: Option<u32>,
    ploidy: Option<u32>,
}

#[derive(Clone, Copy, Default)]
struct SampleCall {
    gt: GtCall,
    // Some(-1) means DS is present but missing: GT must not replace it.
    ds: Option<f32>,
}

#[derive(Clone, Default)]
struct DosageMetadata {
    native_alt: Vec<f64>,
    // Zero means unknown, not zero-ploidy.
    ploidy: Vec<u32>,
}

impl DosageMetadata {
    fn clear(&mut self) {
        self.native_alt.clear();
        self.ploidy.clear();
    }
}

fn parse_gt_text(gt: &[u8]) -> GtCall {
    // The common case, one-character alleles of a diploid call ("0|1", "1/1", "./."),
    // without the general split below.
    if let [a, sep, b] = *gt && (sep == b'|' || sep == b'/') {
        let allele = |c: u8| match c { b'0' => Some(Some(0u32)), b'1' => Some(Some(1)), b'.' => Some(None), _ => None };
        if let (Some(x), Some(y)) = (allele(a), allele(b)) {
            let alt = match (x, y) { (Some(x), Some(y)) => Some(x + y), _ => None };
            return GtCall { alt_count: alt, ploidy: Some(2) };
        }
    }
    if gt.is_empty() || gt == b"." { return GtCall::default(); }
    let mut ploidy = 0;
    let mut alt_count = 0;
    let mut called = true;
    for allele in gt.split(|&b| b == b'/' || b == b'|') {
        ploidy += 1;
        match allele {
            b"0" => (),
            b"1" => alt_count += 1,
            b"." => called = false,
            _ => return GtCall::default(), // invalid allele does not identify ploidy
        }
    }
    GtCall { alt_count: called.then_some(alt_count), ploidy: Some(ploidy) }
}

/// Decode the declared integer width and each sample's own vector end. GT missing
/// alleles (0/1) and typed missing values are distinct from vector-end padding.
fn parse_gt_bcf(data: &[u8], tid: u8) -> GtCall {
    let (width, missing, vector_end) = match tid {
        1 => (1, i8::MIN as i32, i8::MIN as i32 + 1),
        2 => (2, i16::MIN as i32, i16::MIN as i32 + 1),
        3 => (4, i32::MIN, i32::MIN + 1),
        _ => return GtCall::default(),
    };
    let mut ploidy = 0;
    let mut alt_count = 0;
    let mut called = true;
    for bytes in data.chunks_exact(width) {
        let raw = match width {
            1 => bytes[0] as i8 as i32,
            2 => i16::from_le_bytes(bytes.try_into().unwrap()) as i32,
            _ => i32::from_le_bytes(bytes.try_into().unwrap()),
        };
        if raw == vector_end { break; }
        ploidy += 1;
        if raw == missing || raw == 0 || raw == 1 {
            called = false;
        } else if raw < 2 {
            return GtCall::default();
        } else {
            match (raw >> 1) - 1 {
                0 => (),
                1 => alt_count += 1,
                _ => return GtCall::default(), // invalid allele does not identify ploidy
            }
        }
    }
    // A lone missing allele represents GT '.', which does not identify ploidy.
    let known_ploidy = ploidy > 0 && (called || ploidy > 1);
    GtCall { alt_count: (called && ploidy > 0).then_some(alt_count),
        ploidy: known_ploidy.then_some(ploidy) }
}

/// How far a DS may exceed its ploidy and still be read as that ploidy: rounding in
/// the writer (two-decimal DS) or float noise, never a different scale.
const DS_ROUNDING_SLACK: f64 = 1e-3;

#[inline]
fn normalize_call(call: SampleCall, options: InputDosageOptions) -> io::Result<(f32, f64, u32)> {
    let ploidy = call.gt.ploidy.or(options.fallback_ploidy).unwrap_or(0);
    if let Some(ds) = call.ds {
        // DS is authoritative even when missing. Non-finite DS is a no-call.
        if !ds.is_finite() || ds < 0.0 { return Ok((-1.0, 0.0, ploidy)); }
        match options.ds_scale {
            DsScale::Native => {
                if ploidy == 0 {
                    return Err(invalid_dosage("native DS has no identifiable GT ploidy; declare a positive fallback ploidy or use diploid DS scale if the file is already normalized"));
                }
                if ds as f64 > ploidy as f64 + DS_ROUNDING_SLACK {
                    return Err(invalid_dosage(format!("native DS {ds} exceeds GT/fallback ploidy {ploidy}; declare the file's DS scale explicitly")));
                }
                // A DS rounded to a few decimals (or float noise) may sit a hair above the
                // ploidy; it is that ploidy, not an error.
                let ds = ds.min(ploidy as f32);
                let dosage2 = if ploidy == 2 { ds } else { (2.0 * ds as f64 / ploidy as f64) as f32 };
                Ok((dosage2, ds as f64, ploidy))
            }
            DsScale::Diploid => {
                if ds as f64 > 2.0 + DS_ROUNDING_SLACK { return Err(invalid_dosage(format!("diploid DS {ds} exceeds 2"))); }
                let ds = ds.min(2.0);
                let native_alt = if ploidy == 2 { ds as f64 } else { ds as f64 * ploidy as f64 / 2.0 };
                Ok((ds, native_alt, ploidy))
            }
        }
    } else if let (Some(alt), Some(p)) = (call.gt.alt_count, call.gt.ploidy) {
        let dosage2 = if p == 2 { alt as f32 } else { (2.0 * alt as f64 / p as f64) as f32 };
        Ok((dosage2, alt as f64, p))
    } else {
        Ok((-1.0, 0.0, ploidy))
    }
}

/// Normalise one sample's call and append it (`i` = its column, for the error text).
/// `keep_meta` false leaves `metadata` alone: a reader whose metadata is never
/// read (the imputed side of an evaluation) skips two pushes per sample.
#[inline]
fn push_normalized(call: SampleCall, i: usize, options: InputDosageOptions,
    ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata, keep_meta: bool) -> io::Result<()> {
    let (dose, native_alt, ploidy) = normalize_call(call, options)
        .map_err(|e| invalid_dosage(format!("sample column {}: {e}", i + 1)))?;
    ds_buf.push(dose);
    if keep_meta {
        metadata.native_alt.push(native_alt);
        metadata.ploidy.push(ploidy);
    }
    Ok(())
}

#[cfg(test)]
fn normalize_row(calls: &[SampleCall], indices: Option<&[usize]>, options: InputDosageOptions,
    ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata) -> io::Result<()> {
    normalize_row_m(calls, indices, options, ds_buf, metadata, true)
}

fn normalize_row_m(calls: &[SampleCall], indices: Option<&[usize]>, options: InputDosageOptions,
    ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata, keep_meta: bool) -> io::Result<()> {
    ds_buf.clear();
    metadata.clear();
    if let Some(indices) = indices {
        for &i in indices { push_normalized(calls.get(i).copied().unwrap_or_default(), i, options, ds_buf, metadata, keep_meta)?; }
    } else {
        for (i, &c) in calls.iter().enumerate() { push_normalized(c, i, options, ds_buf, metadata, keep_meta)?; }
    }
    Ok(())
}

/// A reader's shared-sample selection: file column of each shared sample, in
/// shared order, plus which columns that leaves to parse.
struct ColFilter {
    indices: Vec<usize>,
    /// `needed[c]`: column `c` (< n_file_samples) is selected.
    needed: Vec<bool>,
    /// One past the last selected column below n_file_samples.
    scan_end: usize,
}

impl ColFilter {
    fn new(indices: Vec<usize>, n_file_samples: usize) -> Self {
        let mut needed = vec![false; n_file_samples];
        let mut scan_end = 0;
        for &i in &indices {
            if i < n_file_samples { needed[i] = true; scan_end = scan_end.max(i + 1); }
        }
        Self { indices, needed, scan_end }
    }
}

/// `str::parse::<f32>` of a FORMAT/DS field, exactly, with a fast path for plain
/// decimals of at most seven digits (`123`, `0.5`, `1.234`): their digits are an
/// integer below 2^24 and their scale 10^k (k <= 7) is exact in f32, so one
/// correctly rounded division gives the correctly rounded value, as the parser
/// does. Anything else (sign, exponent, `.5`, `nan`, longer digit strings) goes
/// to the parser.
#[inline]
fn parse_ds_f32(field: &[u8]) -> Option<f32> {
    const POW10: [f32; 8] = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7];
    let n = field.len();
    if (1..=8).contains(&n) && field[0].is_ascii_digit() {
        let mut v: u32 = 0;
        let mut digits = 0usize;
        let mut frac: Option<usize> = None;
        let mut ok = true;
        for &b in field {
            if b.is_ascii_digit() {
                v = v * 10 + (b - b'0') as u32;
                digits += 1;
                if let Some(k) = frac.as_mut() { *k += 1; }
            } else if b == b'.' && frac.is_none() {
                frac = Some(0);
            } else { ok = false; break; }
        }
        if ok && digits <= 7 && frac != Some(0) {
            return Some(v as f32 / POW10[frac.unwrap_or(0)]);
        }
    }
    std::str::from_utf8(field).ok().and_then(|s| s.parse::<f32>().ok())
}

/// AF among called truth samples, in original allele-copy units. A wholly REF
/// site has AF=0 even if its ploidies are unknown (e.g. absent-from-strong truth).
/// At a variable site every called sample needs ploidy, including REF samples.
fn truth_af(dosage2: &[f32], metadata: &DosageMetadata) -> io::Result<f64> {
    let mut n_called = 0;
    let mut all_ref = true;
    let mut alt_sum = 0.0;
    let mut ploidy_sum = 0u64;
    let mut unknown_ploidy = false;
    for (s, &dose) in dosage2.iter().enumerate() {
        if dose < 0.0 { continue; }
        n_called += 1;
        all_ref &= dose == 0.0;
        alt_sum += metadata.native_alt[s];
        ploidy_sum += metadata.ploidy[s] as u64;
        unknown_ploidy |= metadata.ploidy[s] == 0;
    }
    if n_called == 0 { return Ok(f64::NAN); }
    if all_ref { return Ok(0.0); }
    if unknown_ploidy {
        return Err(invalid_dosage("truth AF requires GT ploidy or a positive truth fallback ploidy for every called sample at a non-REF site; diploid DS scale alone does not identify original ploidy"));
    }
    Ok(alt_sum / ploidy_sum as f64)
}


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
}

impl SampleAccumulator {
    pub fn new(n_samples: usize) -> Self {
        SampleAccumulator {
            n_samples, n_variants: 0,
            sum_ds: vec![0.0; n_samples],
            sum_gt: vec![0.0; n_samples],
            sum_ds2: vec![0.0; n_samples],
            sum_gt2: vec![0.0; n_samples],
            sum_ds_gt: vec![0.0; n_samples],
            n_correct: vec![0; n_samples],
            n_total: vec![0; n_samples],
        }
    }

    /// Add one variant's dosage/truth for all samples.
    pub fn add_variant(&mut self, ds: &[f32], truth_gt: &[f32]) {
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
        }
    }

    /// Compute per-sample R² from accumulated statistics.
    /// Per-sample R2. **NaN means UNDEFINED, not zero**: a sample with fewer than
    /// two comparable genotypes, or with no variance on either side, has no
    /// correlation to report. Returning 0.0 for those made the mean punish a
    /// sample for the truth's silence — with 20 of 40 truth genotypes no-called
    /// and the imputation exact, the per-sample mean read 0.5000 because the
    /// twenty samples with nothing to score contributed a zero each. Callers must
    /// average over the finite entries only.
    pub fn compute_r2(&self) -> Vec<f64> {
        (0..self.n_samples).map(|s| {
            let n = self.n_total[s] as f64;
            if n < 2.0 { return f64::NAN; }
            let num = n * self.sum_ds_gt[s] - self.sum_ds[s] * self.sum_gt[s];
            let den_x = n * self.sum_ds2[s] - self.sum_ds[s] * self.sum_ds[s];
            let den_y = n * self.sum_gt2[s] - self.sum_gt[s] * self.sum_gt[s];
            let den = (den_x.max(0.0) * den_y.max(0.0)).sqrt();
            if den > 0.0 { (num / den).powi(2).clamp(0.0, 1.0) } else { f64::NAN }
        }).collect()
    }

    /// Compute per-sample concordance.
    /// Per-sample concordance; NaN when the sample has nothing to compare, for the
    /// same reason as `compute_r2`.
    pub fn compute_concordance(&self) -> Vec<f64> {
        (0..self.n_samples).map(|s| {
            if self.n_total[s] > 0 { self.n_correct[s] as f64 / self.n_total[s] as f64 } else { f64::NAN }
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
    }
}

/// Why `--truth-raw` did or did not change the score.
///
/// A raw truth only ever acts through one narrow path: a (site, sample) where
/// the filtered truth has no alt call but the raw truth does, so the sample is
/// SKIPPED at that site instead of being scored as hom-ref. Zero rescues is a
/// legitimate outcome (the filtered truth already holds every carrier at the
/// scored sites), but it is indistinguishable from a key-matching failure
/// unless the intermediate counts are reported — so report them.
#[derive(Default, Clone, Copy)]
pub struct RawTruthDiag {
    /// Records loaded from the raw truth file.
    pub raw_sites_loaded: u64,
    /// Scored sites whose (contig, pos, REF, ALT) key was found in the raw truth.
    pub scored_sites_matched: u64,
    /// (site, sample) pairs the raw truth rescued from hom-ref scoring.
    pub genotypes_rescued: u64,
    /// No-calls in raw truth excluded because the strong record is absent.
    /// These are not carrier rescues and do not increment `genotypes_rescued`.
    pub genotypes_missing_raw: u64,
    /// Scored sites absent from the filtered truth entirely.
    pub scored_sites_absent_from_strong: u64,
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
    /// Concordance counted for every site with a scored sample. R² only counted when not NaN AND when the
    /// variant falls into one of the declared MAF bins (MAF ≥ 0.0005). This
    /// matches Beagle/IMPUTE5/Minimac4 convention and avoids ultra-rare
    /// singleton noise (MAF < 0.05%) dragging the mean to meaningless values.
    pub fn add(&mut self, maf: f64, r2: f64, concordance: f64) {
        // Concordance counts for every variant with a scored sample (monomorphic too).
        if concordance.is_nan() { return; }
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
/// `SELPHI_EVAL_CONST_DS_NAN=1` → alternative scoring (constant dosage at a variable
/// truth site is dropped from its MAF bin instead of scoring r² = 0). Cached once.
fn const_ds_nan() -> bool {
    use std::sync::OnceLock;
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| crate::config::present("SELPHI_EVAL_CONST_DS_NAN"))
}

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

    // No scored sample: neither r² nor concordance exists (a 0 here used to pull a
    // one-sample run's concordance down by its share of truth no-calls).
    if count == 0 { return (f64::NAN, f64::NAN); }
    // One scored sample (a single-sample run): r² is undefined, concordance is not.
    // It used to be forced to 0, so a one-sample evaluation printed 0.0000 for it.
    if count == 1 {
        let i = (0..n).find(|&i| ds[i] >= 0.0 && gt[i] >= 0.0).unwrap();
        let d_call = if ds[i] > 1.5 { 2 } else if ds[i] > 0.5 { 1 } else { 0 };
        return (f64::NAN, (d_call == gt[i].round() as i32) as u32 as f64);
    }
    let nf = count as f64;
    let num = nf * sum_dg - sum_d * sum_g;
    let den_x = nf * sum_d2 - sum_d * sum_d;
    let den_y = nf * sum_g2 - sum_g * sum_g;
    // Clamp each variance term at zero BEFORE multiplying, and clamp r² to [0, 1]
    // — the same guards `SampleAccumulator::compute_r2` already applies. Clamping
    // only the product lets a site where BOTH terms round to a tiny negative
    // (dosages and truth both effectively constant) produce a tiny positive
    // denominator and hence an unbounded r² > 1 that would then be summed into the
    // MAF-bin means.
    // R² is undefined only when the TRUTH is constant at the site (nothing to predict).
    // When the truth varies but the imputed dosage is constant, the tool explained none
    // of the variance: that is r² = 0, not "undefined". Returning NaN there
    // drops the site from the tool's MAF bin — a site set that differs by
    // tool and rewards a tool for emitting a flat dosage at hard sites (on 1000 Genomes
    // chr22 the rarest bin held 132,804 sites for Beagle against 136,724 for Selphi 2).
    // `SELPHI_EVAL_CONST_DS_NAN=1` restores the old behaviour for an exact A/B.
    let r2 = if den_y.max(0.0) <= 0.0 {
        f64::NAN
    } else if den_x.max(0.0) <= 0.0 {
        if const_ds_nan() { f64::NAN } else { 0.0 }
    } else {
        let den = (den_x * den_y).sqrt();
        (num / den).powi(2).clamp(0.0, 1.0)
    };

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

/// Normalize a contig name for cross-file matching: strip a leading `chr`
/// (case-insensitive) so `chr22` and `22` compare equal. The merge below keys
/// on (normalized-contig, pos) — without this it matched on POSITION ALONE,
/// silently producing wrong cross-chromosome pairings when the truth was a
/// multi-contig (e.g. whole-genome) file whose contig names didn't align with
/// the (single-chr) imputed file. For aligned single-contig inputs this is a
/// no-op (behaviour byte-for-byte unchanged).
#[inline]
fn norm_contig(c: &[u8]) -> &[u8] {
    if c.len() > 3 && c[..3].eq_ignore_ascii_case(b"chr") { &c[3..] } else { c }
}

/// FNV-1a hash of a (chr-prefix-normalized) contig name, for alloc-free site keys.
#[inline]
fn contig_hash(c: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in c { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

/// Strip a trailing `\n` or `\r\n` from a line read via `read_until(b'\n', ...)`.
#[inline]
fn strip_line_endings(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    if end > 0 && line[end - 1] == b'\n' { end -= 1; }
    if end > 0 && line[end - 1] == b'\r' { end -= 1; }
    &line[..end]
}

type VariantRecord = (Vec<u8>, i64, Vec<u8>, Vec<u8>);

/// Parse a VCF record on the default native scale. The Option API is retained
/// for callers of this low-level helper; evaluators use the fallible reader so
/// ambiguous DS units are reported as errors, never mistaken for end-of-file.
pub fn parse_vcf_line(line: &[u8], n_samples: usize, ds_buf: &mut Vec<f32>) -> Option<VariantRecord> {
    parse_vcf_record(line, n_samples, None, InputDosageOptions::default(), ds_buf,
        &mut DosageMetadata::default(), &mut Vec::new(), false).ok().flatten()
}

fn parse_vcf_record(line: &[u8], n_samples: usize, indices: Option<&[usize]>,
    options: InputDosageOptions, ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata,
    calls: &mut Vec<SampleCall>, skip_gt: bool) -> io::Result<Option<VariantRecord>> {
    let filter = indices.map(|ix| ColFilter::new(ix.to_vec(), n_samples));
    parse_vcf_record_with(line, n_samples, filter.as_ref(), options, ds_buf, metadata, true, calls, skip_gt)
}

/// `parse_vcf_record` with the reader's cached sample selection: only the
/// selected columns are split out and parsed (a call is parsed infallibly; only
/// the selected ones are normalised, so skipping the rest changes nothing).
#[allow(clippy::too_many_arguments)]
fn parse_vcf_record_with(line: &[u8], n_samples: usize, filter: Option<&ColFilter>,
    options: InputDosageOptions, ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata,
    keep_meta: bool, calls: &mut Vec<SampleCall>, skip_gt: bool) -> io::Result<Option<VariantRecord>> {
    let line = strip_line_endings(line);
    if line.is_empty() || line[0] == b'#' { return Ok(None); }
    let mut tabs = [0usize; 9];
    let mut nt = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' { tabs[nt] = i; nt += 1; if nt == 9 { break; } }
    }
    if nt < 9 { return Ok(None); }
    let chrom = line[..tabs[0]].to_vec();
    let pos = std::str::from_utf8(&line[tabs[0]+1..tabs[1]]).ok()
        .and_then(|s| s.parse::<i64>().ok()).ok_or_else(|| invalid_dosage("invalid VCF position"))?;
    let ref_a = line[tabs[2]+1..tabs[3]].to_vec();
    let alt_a = line[tabs[3]+1..tabs[4]].to_vec();
    if alt_a.contains(&b',') { return Ok(None); }
    let record = (chrom, pos, ref_a, alt_a);
    if skip_gt { ds_buf.clear(); metadata.clear(); return Ok(Some(record)); }

    let format = &line[tabs[7]+1..tabs[8]];
    let mut ds_idx = None;
    let mut gt_idx = None;
    for (i, field) in format.split(|&b| b == b':').enumerate() {
        if field == b"DS" { ds_idx = Some(i); }
        if field == b"GT" { gt_idx = Some(i); }
    }
    // Only the fields up to the last of GT/DS are looked at.
    let n_fields = gt_idx.max(ds_idx).map_or(0, |i| i + 1);
    let parse_call = |sample: &[u8]| {
        let mut call = SampleCall { ds: ds_idx.map(|_| -1.0), ..Default::default() };
        for (i, field) in sample.split(|&b| b == b':').take(n_fields).enumerate() {
            if gt_idx == Some(i) { call.gt = parse_gt_text(field); }
            if ds_idx == Some(i) {
                call.ds = Some(parse_ds_f32(field).filter(|v| v.is_finite()).unwrap_or(-1.0));
            }
        }
        call
    };
    let Some(filter) = filter else {
        // Every column in file order: normalise as it is read, no call buffer.
        ds_buf.clear();
        metadata.clear();
        let mut n = 0usize;
        for sample in line[tabs[8]+1..].split(|&b| b == b'\t').take(n_samples) {
            push_normalized(parse_call(sample), n, options, ds_buf, metadata, keep_meta)?;
            n += 1;
        }
        // Absent/truncated sample columns are missing, never synthetic hom-ref.
        for i in n..n_samples { push_normalized(SampleCall::default(), i, options, ds_buf, metadata, keep_meta)?; }
        return Ok(Some(record));
    };
    // Columns as `split('\t').take(n_samples)` yields them, up to the last
    // selected one; unselected and absent columns stay missing.
    calls.clear();
    calls.resize(n_samples, SampleCall::default());
    let samples = &line[tabs[8]+1..];
    let mut start = 0usize;
    for c in 0..filter.scan_end.min(n_samples) {
        let end = memchr::memchr(b'\t', &samples[start..]).map_or(samples.len(), |p| start + p);
        if filter.needed[c] { calls[c] = parse_call(&samples[start..end]); }
        if end >= samples.len() { break; }
        start = end + 1;
    }
    normalize_row_m(calls, Some(&filter.indices), options, ds_buf, metadata, keep_meta)?;
    Ok(Some(record))
}

/// Content-sniff whether `path` is a binary BCF (BGZF magic "BCF"), so a BCF
/// misnamed `.vcf.gz` is routed by content rather than extension alone
/// (mirrors the sniff in io::target_io / lcwgs::pl_reader). False on any read
/// error → treated as text/VCF.
fn sniff_bcf(path: &Path) -> bool {
    std::fs::File::open(path).ok().and_then(|f| {
        let mut bgzf = noodles_bgzf::io::Reader::new(BufReader::new(f));
        let mut magic = [0u8; 5];
        bgzf.read_exact(&mut magic).ok().map(|_| &magic[..3] == b"BCF")
    }).unwrap_or(false)
}

/// Parse VCF/BCF header to get sample names.
pub fn parse_header_samples(path: &Path) -> io::Result<Vec<String>> {
    let p = path.to_string_lossy();
    let is_bcf = sniff_bcf(path) || p.ends_with(".bcf") || p.ends_with(".bcf.gz");

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
        /// File-column index of each shared sample, in shared order. `None` =
        /// take the leading columns as they come, which is only right when the
        /// file's samples ARE the shared set in that order.
        sample_indices: Option<ColFilter>,
        /// Scratch: GT/DS calls before shared-sample reindexing.
        calls: Vec<SampleCall>,
        metadata: DosageMetadata,
        keep_metadata: bool,
        options: InputDosageOptions,
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
        calls: Vec<SampleCall>,
        metadata: DosageMetadata,
        keep_metadata: bool,
        options: InputDosageOptions,
    },
}

impl VariantReader {
    fn open(path: &Path) -> io::Result<(Self, Vec<String>)> {
        let s = path.to_string_lossy();
        let is_bcf = sniff_bcf(path) || s.ends_with(".bcf") || s.ends_with(".bcf.gz");

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
                calls: Vec::new(), metadata: DosageMetadata::default(), keep_metadata: true,
                options: InputDosageOptions::default(),
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
                sample_indices: None, calls: Vec::new(),
                metadata: DosageMetadata::default(), keep_metadata: true,
                options: InputDosageOptions::default(),
            }, samples))
        }
    }

    /// Set the file-column index of each shared sample (shared order). BCF
    /// extracts selectively; VCF parses the full row and reindexes. The VCF arm
    /// used to drop the request silently, so whenever a VCF.gz side's samples
    /// were not the leading columns in shared order, file column i was scored
    /// against file column i of the other side.
    ///
    /// The identity selection (every column, in file order) is stored as no
    /// selection: both paths then produce the same values and errors.
    fn set_sample_filter(&mut self, indices: Vec<usize>) {
        let identity = |n: usize| indices.len() == n && indices.iter().enumerate().all(|(i, &x)| i == x);
        match self {
            VariantReader::Bcf { sample_indices, n_file_samples, .. } => {
                *sample_indices = (!identity(*n_file_samples)).then_some(indices);
            }
            VariantReader::Vcf { sample_indices, n_file_samples, .. } => {
                *sample_indices = (!identity(*n_file_samples)).then(|| ColFilter::new(indices, *n_file_samples));
            }
        }
    }

    /// `false`: `metadata()` is never read for this reader, so do not fill it.
    fn set_keep_metadata(&mut self, keep: bool) {
        match self {
            Self::Vcf { keep_metadata, .. } | Self::Bcf { keep_metadata, .. } => *keep_metadata = keep,
        }
    }

    fn set_dosage_options(&mut self, dosage_options: InputDosageOptions) {
        match self {
            Self::Vcf { options, .. } | Self::Bcf { options, .. } => *options = dosage_options,
        }
    }

    fn metadata(&self) -> &DosageMetadata {
        match self {
            Self::Vcf { metadata, .. } | Self::Bcf { metadata, .. } => metadata,
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

    /// Read one biallelic variant and normalize dosages. Errors include the input
    /// path; malformed/ambiguous dosage data must not silently terminate a merge.
    fn next_record(&mut self, ds_buf: &mut Vec<f32>) -> io::Result<Option<VariantRecord>> {
        self.next_record_inner(ds_buf, false).map_err(|e| {
            let path = match self { Self::Vcf { path, .. } | Self::Bcf { path, .. } => path };
            io::Error::new(e.kind(), format!("{}: {e}", path.display()))
        })
    }

    fn next_record_inner(&mut self, ds_buf: &mut Vec<f32>, skip_gt: bool) -> io::Result<Option<VariantRecord>> {
        match self {
            VariantReader::Vcf { reader, line_buf, n_file_samples, sample_indices, calls, metadata, keep_metadata, options, .. } => {
                loop {
                    line_buf.clear();
                    if reader.read_until(b'\n', line_buf)? == 0 { return Ok(None); }
                    if let Some(rec) = parse_vcf_record_with(line_buf, *n_file_samples,
                        sample_indices.as_ref(), *options, ds_buf, metadata, *keep_metadata, calls, skip_gt)? {
                        return Ok(Some(rec));
                    }
                }
            }
            VariantReader::Bcf { reader, contig_names, gt_key, ds_key, n_file_samples,
                sample_indices, sb, ib, calls, metadata, keep_metadata, options, .. } => {
                let ns = *n_file_samples;
                loop {
                    let mut lbuf = [0u8; 4];
                    // Distinguish a clean EOF from a truncated record header.
                    if reader.read(&mut lbuf[..1])? == 0 { return Ok(None); }
                    reader.read_exact(&mut lbuf[1..])?;
                    let ls = u32::from_le_bytes(lbuf) as usize;
                    if ls < 24 { return Err(invalid_dosage("truncated BCF shared header")); }
                    let mut libuf = [0u8; 4];
                    reader.read_exact(&mut libuf)?;
                    let li = u32::from_le_bytes(libuf) as usize;
                    sb.resize(ls, 0);
                    reader.read_exact(sb)?;
                    let ci = i32::from_le_bytes(sb[0..4].try_into().unwrap()) as usize;
                    let pos = i32::from_le_bytes(sb[4..8].try_into().unwrap()) as i64 + 1;
                    let na = u16::from_le_bytes(sb[18..20].try_into().unwrap()) as usize;
                    if na != 2 || skip_gt {
                        let mut rem = li;
                        let mut skip_buf = [0u8; 65536];
                        while rem > 0 {
                            let c = rem.min(skip_buf.len());
                            reader.read_exact(&mut skip_buf[..c])?;
                            rem -= c;
                        }
                        if na != 2 { continue; }
                    } else {
                        ib.resize(li, 0);
                        reader.read_exact(ib)?;
                    }
                    let chrom = if ci < contig_names.len() { contig_names[ci].as_bytes().to_vec() }
                        else { format!("{ci}").into_bytes() };
                    let nf = (u32::from_le_bytes(sb[20..24].try_into().unwrap()) >> 24) as usize;
                    let mut o = 24usize;
                    let _id = rtstr(sb, &mut o);
                    let ref_a = rtstr_bytes(sb, &mut o);
                    let alt_a = rtstr_bytes(sb, &mut o);
                    if skip_gt {
                        ds_buf.clear(); metadata.clear();
                        return Ok(Some((chrom, pos, ref_a, alt_a)));
                    }
                    decode_bcf_calls(ib, nf, ns, *gt_key, *ds_key, sample_indices.as_deref(), calls)?;
                    normalize_row_m(calls, None, *options, ds_buf, metadata, *keep_metadata)?;
                    return Ok(Some((chrom, pos, ref_a, alt_a)));
                }
            }
        }
    }
}

/// Decode both GT and DS regardless of FORMAT order. GT supplies native ploidy
/// even when DS is authoritative, so a DS-before-GT record must not stop early.
fn decode_bcf_calls(ib: &[u8], nf: usize, ns: usize, gt_key: u16, ds_key: Option<u16>,
    indices: Option<&[usize]>, calls: &mut Vec<SampleCall>) -> io::Result<()> {
    calls.clear();
    calls.resize(indices.map_or(ns, |v| v.len()), SampleCall::default());
    let mut offset = 0;
    for _ in 0..nf {
        if offset >= ib.len() { return Err(invalid_dosage("truncated BCF FORMAT key")); }
        let key = rtint(ib, &mut offset) as u16;
        let tb = *ib.get(offset).ok_or_else(|| invalid_dosage("truncated BCF FORMAT descriptor"))?;
        offset += 1;
        let tid = tb & 15;
        let vl = if tb >> 4 == 15 {
            let length = rtint(ib, &mut offset);
            usize::try_from(length).map_err(|_| invalid_dosage("negative BCF FORMAT length"))?
        } else { (tb >> 4) as usize };
        let width = match tid { 0 => 0, 1 | 7 => 1, 2 => 2, 3 | 5 => 4, _ => return Err(invalid_dosage("unsupported BCF FORMAT type")) };
        let stride = vl.checked_mul(width).ok_or_else(|| invalid_dosage("BCF FORMAT length overflow"))?;
        let len = stride.checked_mul(ns).ok_or_else(|| invalid_dosage("BCF FORMAT length overflow"))?;
        let end = offset.checked_add(len).filter(|&e| e <= ib.len())
            .ok_or_else(|| invalid_dosage("truncated BCF FORMAT payload"))?;
        if ds_key == Some(key) {
            if tid != 5 || vl != 1 { return Err(invalid_dosage("biallelic FORMAT/DS must be one float per sample")); }
            for (i, call) in calls.iter_mut().enumerate() {
                let si = indices.map_or(i, |v| v[i]);
                if si >= ns { continue; }
                let start = offset + si * stride;
                let ds = f32::from_le_bytes(ib[start..start+4].try_into().unwrap());
                call.ds = Some(if ds.is_finite() { ds } else { -1.0 });
            }
        } else if key == gt_key {
            if !matches!(tid, 1..=3) { return Err(invalid_dosage("FORMAT/GT must use an integer type")); }
            for (i, call) in calls.iter_mut().enumerate() {
                let si = indices.map_or(i, |v| v[i]);
                if si >= ns { continue; }
                let start = offset + si * stride;
                call.gt = parse_gt_bcf(&ib[start..start+stride], tid);
            }
        }
        offset = end;
    }
    Ok(())
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
    if sniff_bcf(path) || s.ends_with(".bcf") || s.ends_with(".bcf.gz") {
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
    let is_bcf = sniff_bcf(path) || s.ends_with(".bcf") || s.ends_with(".bcf.gz");
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
    imputed_path: &Path, truth_path: &Path, shared_samples: &[String], n_threads: usize,
    exclude: &super::exclude::SiteExclusion,
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    evaluate_parallel_with_options(imputed_path, truth_path, shared_samples, n_threads,
        exclude, &EvaluationOptions::default())
}

fn evaluate_parallel_with_options(
    imputed_path: &Path,
    truth_path: &Path,
    shared_samples: &[String],
    n_threads: usize,
    // Sites to drop entirely (`--exclude-sites`, normally the genotyped/chip
    // sites, which are observed rather than imputed and would otherwise inflate
    // every bin). This path used to ignore the flag: it was only ever plumbed
    // into `evaluate_imputation` (the absent->hom-ref path), so a run with a
    // COMPLETE truth callset — which is what `--homref-absent auto` resolves to,
    // and the default — scored the typed sites in silence.
    exclude: &super::exclude::SiteExclusion,
    options: &EvaluationOptions,
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    use rayon::prelude::*;

    options.validate()?;

    let n_samples = shared_samples.len();
    let exclude_ref = exclude;

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
                // Only the truth side's metadata (AF) is read below.
                imp_reader.set_keep_metadata(false);
                imp_reader.set_dosage_options(options.imputed);
                truth_reader.set_dosage_options(options.truth);

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
                    let r = imp_reader.next_record(&mut imp_ds_raw)?;
                    match &r {
                        Some(rec) if rec.1 < region_start => continue,
                        _ => break r,
                    }
                };
                let mut truth_rec = loop {
                    let r = truth_reader.next_record(&mut truth_ds_raw)?;
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

                    // Key on (normalized-contig, pos) so chr22≡22 AND records on
                    // DIFFERENT chromosomes are never matched (previously the merge
                    // compared pos ALONE → a multi-contig truth silently produced
                    // wrong cross-chromosome pairings). For aligned single-contig
                    // inputs (the per-chromosome imputation path) the contig keys are
                    // always equal, so this reduces to the original pos comparison.
                    let ik = (norm_contig(&imp.0), imp.1);
                    let tk = (norm_contig(&truth.0), truth.1);
                    if ik < tk {
                        if imp.1 < region_end { n_imp += 1; }
                        imp_rec = imp_reader.next_record(&mut imp_ds_raw)?;
                    } else if ik > tk {
                        if truth.1 < region_end { n_truth += 1; }
                        truth_rec = truth_reader.next_record(&mut truth_ds_raw)?;
                    } else if imp.1 < region_end {
                        // Same (contig, position), in-region.
                        n_imp += 1;
                        n_truth += 1;

                        let (matched, swapped) = match_alleles(&imp.2, &imp.3, &truth.2, &truth.3);
                        // An excluded site is dropped before scoring; it still counted
                        // into n_imp/n_truth above, so the summary's
                        // n_imp_variants - n_matched still reports it as excluded.
                        let excluded = !exclude_ref.is_empty()
                            && exclude_ref.contains(&site_key(&imp.0, imp.1, &imp.2, &imp.3));
                        if matched && !excluded {
                            let chrom = String::from_utf8_lossy(&imp.0);
                            if !chr_set.contains(chrom.as_ref()) { chr_set.insert(chrom.into_owned()); }
                            imp_ds[..n_samples].copy_from_slice(&imp_ds_raw[..n_samples]);
                            if swapped { for si in 0..n_samples { if imp_ds[si] >= 0.0 { imp_ds[si] = 2.0 - imp_ds[si]; } } }
                            truth_ds[..n_samples].copy_from_slice(&truth_ds_raw[..n_samples]);

                            let af = truth_af(&truth_ds, truth_reader.metadata())?;
                            let maf = af.min(1.0 - af);
                            let (r2, conc) = site_r2(&imp_ds, &truth_ds, n_samples);
                            site_acc.add(maf, r2, conc);
                            sample_acc.add_variant(&imp_ds, &truth_ds);
                            n_matched += 1;
                        }

                        imp_rec = imp_reader.next_record(&mut imp_ds_raw)?;
                        truth_rec = truth_reader.next_record(&mut truth_ds_raw)?;
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
                    imp_rec = imp_reader.next_record(&mut imp_ds_raw)?;
                }
                while let Some(ref rec) = truth_rec {
                    if rec.1 >= region_end { break; }
                    n_truth += 1;
                    truth_rec = truth_reader.next_record(&mut truth_ds_raw)?;
                }

                Ok((site_acc, sample_acc, EvalCounts { n_matched, n_imp_variants: n_imp, n_truth_variants: n_truth, chromosomes: chr_set.into_iter().collect() }))
            })();
            let wall = t0.elapsed().as_secs_f64();
            // A region that fails must fail the evaluation: it used to come back
            // as an empty accumulator, silently scoring ~1/n_regions fewer sites.
            result.map(|(sa, sm, ec)| (sa, sm, ec, wall)).map_err(|e| io::Error::new(e.kind(),
                format!("evaluation region [{}, {}) failed: {}", region_start, region_end, e)))
        })
        .collect::<io::Result<Vec<_>>>()?;

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
    exclude: &super::exclude::SiteExclusion,
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    evaluate_with_options(imputed_path, truth_path, shared_samples, exclude, &EvaluationOptions::default())
}

/// Evaluate with declared DS units and optional native-ploidy fallbacks.
/// Concordance uses the existing diploid-equivalent dosage thresholds.
pub fn evaluate_with_options(
    imputed_path: &Path, truth_path: &Path, shared_samples: &[String],
    exclude: &super::exclude::SiteExclusion, options: &EvaluationOptions,
) -> io::Result<(SiteAccumulator, SampleAccumulator, EvalCounts)> {
    let n_threads = rayon::current_num_threads().max(1);
    evaluate_parallel_with_options(imputed_path, truth_path, shared_samples, n_threads, exclude, options)
}

/// FNV-1a hash of REF+ALT bytes (for an alloc-free site key).
#[inline]
fn refalt_hash(r: &[u8], a: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in r { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h ^= b'/' as u64; h = h.wrapping_mul(0x100000001b3);
    for &b in a { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}
#[inline]
pub(crate) fn site_key(c: &[u8], pos: i64, r: &[u8], a: &[u8]) -> (u64, i64, u64) {
    (contig_hash(norm_contig(c)), pos, refalt_hash(r, a))
}

/// Add the site key of every biallelic record in `path` to `out`; returns how
/// many were read. Genotypes are not decoded. A file that yields no sites is an
/// error: an empty exclusion list used to score every site in silence.
pub(crate) fn read_site_keys(path: &Path, out: &mut std::collections::HashSet<(u64, i64, u64)>) -> io::Result<usize> {
    let (mut r, _) = VariantReader::open(path)?;
    let mut b: Vec<f32> = Vec::new();
    let mut n = 0usize;
    while let Some(rec) = r.next_record_inner(&mut b, true)? { out.insert(site_key(&rec.0, rec.1, &rec.2, &rec.3)); n += 1; }
    if n == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
            "{}: no sites read for exclusion (a sites-only VCF needs at least one sample column)", path.display())));
    }
    Ok(n)
}

/// Reindex `shared` sample names to their column positions in `path`'s header.
fn shared_reindex(path: &Path, shared: &[String]) -> io::Result<Vec<usize>> {
    let (_, names) = VariantReader::open(path)?;
    let m: std::collections::HashMap<&str, usize> =
        names.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();
    Ok(shared.iter().map(|s| *m.get(s.as_str()).expect("shared sample missing from file")).collect())
}

#[derive(Clone)]
struct DosageRow {
    dosage2: Vec<f32>,
    metadata: DosageMetadata,
}

/// Keep original allele-copy metadata alongside normalized dosages, in shared
/// sample order. Raw truth uses the same declared options as strong truth.
fn load_dosage_map(path: &Path, shared: &[String], options: InputDosageOptions)
    -> io::Result<std::collections::HashMap<(u64, i64, u64), DosageRow>> {
    let ri = shared_reindex(path, shared)?;
    let (mut r, _) = VariantReader::open(path)?;
    r.set_sample_filter(ri);
    r.set_dosage_options(options);
    let mut buf = Vec::new();
    let mut map = std::collections::HashMap::new();
    while let Some(rec) = r.next_record(&mut buf)? {
        map.insert(site_key(&rec.0, rec.1, &rec.2, &rec.3), DosageRow {
            dosage2: buf.clone(), metadata: r.metadata().clone(),
        });
    }
    Ok(map)
}

/// Returns true if the truth VCF/BCF contains an EXPLICIT homozygous-reference call
/// (GT 0/0 or 0|0) in its first `sample_limit` data records — i.e. it is a COMPLETE
/// callset rather than variant-only. Used to auto-decide `--homref-absent`: a complete
/// truth needs no absent→hom-ref (evaluate matched sites only, the legacy behaviour);
/// a variant-only truth does. Reads the raw GT (not the folded dosage, which can't tell
/// 0/0 from ./.). VCF.gz: text scan. BCF: parse GT via the BCF reader.
pub fn truth_has_ref_calls(path: &Path) -> io::Result<bool> {
    let s = path.to_string_lossy();
    let is_bcf = sniff_bcf(path) || s.ends_with(".bcf") || s.ends_with(".bcf.gz");
    let mut seen = 0usize;
    if !is_bcf {
        let f = std::fs::File::open(path)?;
        let bgzf = noodles_bgzf::io::Reader::new(BufReader::new(f));
        let reader = BufReader::new(bgzf);
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') { continue; }
            // scan sample columns for an explicit ref genotype
            if line.contains("\t0/0") || line.contains("\t0|0")
                || line.contains(":0/0") || line.contains(":0|0")
                || line.contains("0/0:") || line.contains("0|0:")
                || line.ends_with("0/0") || line.ends_with("0|0") {
                return Ok(true);
            }
            seen += 1;
            if seen >= 4000 { break; }
        }
        return Ok(false);
    }
    // This auto-detection path does not scan BCF genotype metadata. A BCF
    // truth is therefore assumed COMPLETE (the safe legacy matched-only path); pass
    // `--homref-absent on` for the rare variant-only BCF truth. (Variant-only truths
    // are almost always VCF.gz, handled exactly above.)
    Ok(true)
}

/// Standard imputation-R² evaluation (the field convention; reproduces the benchmark
/// `06_eval.py`). For every imputed site:
///   * if it is in `exclude` (e.g. the chip/typed sites) → drop the site entirely;
///   * per shared sample `s`:
///       - if `s` carries the variant in the STRONG truth (dosage ≥ 0.5) → truth = strong dosage;
///       - explicit strong no-call, or raw no-call with no strong record → skip `s`;
///       - else if `s` carries it in the RAW (called-but-quality-filtered) truth → skip `s` here;
///       - else → truth = 0 (absent-from-truth ⇒ hom-ref).
/// Splits SNP vs indel and also accumulates the combined total. Truth/raw/exclude are
/// loaded as (norm-contig,pos,ref/alt-hash) maps (cheap per-chromosome; streams the imputed file).
pub fn evaluate_imputation(
    imputed_path: &Path, strong_path: &Path, shared: &[String], raw_path: Option<&Path>,
    exclude: &super::exclude::SiteExclusion,
) -> io::Result<(SampleAccumulator, SampleAccumulator, SampleAccumulator, EvalCounts, SiteAccumulator, RawTruthDiag)> {
    evaluate_imputation_with_options(imputed_path, strong_path, shared, raw_path,
        exclude, &EvaluationOptions::default())
}

/// Homref-absent evaluation with declared DS units. Explicit missing truth calls
/// are excluded; a wholly absent strong record supplies REF calls with unknown
/// ploidy, for which AF=0 remains exact. Raw truth shares `options.truth`.
pub fn evaluate_imputation_with_options(
    imputed_path: &Path,
    strong_path: &Path,
    shared: &[String],
    raw_path: Option<&Path>,
    exclude: &super::exclude::SiteExclusion,
    options: &EvaluationOptions,
) -> io::Result<(SampleAccumulator, SampleAccumulator, SampleAccumulator, EvalCounts, SiteAccumulator, RawTruthDiag)> {
    options.validate()?;
    let n = shared.len();
    let strong = load_dosage_map(strong_path, shared, options.truth)?;
    let raw = match raw_path { Some(p) => Some(load_dosage_map(p, shared, options.truth)?), None => None };

    let iri = shared_reindex(imputed_path, shared)?;
    let (mut imp, _) = VariantReader::open(imputed_path)?;
    imp.set_sample_filter(iri);
    imp.set_dosage_options(options.imputed);
    // Only the strong truth's metadata (AF) is read below.
    imp.set_keep_metadata(false);

    let mut comb = SampleAccumulator::new(n);
    let mut snp = SampleAccumulator::new(n);
    let mut indel = SampleAccumulator::new(n);
    // Per-site MAF-binned R², alongside the per-sample accumulators. MAF is
    // derived from the original truth ALT counts and ploidies at each scored
    // site. Diploid samples retain the existing 1/(2*n_truth) resolution floor;
    // mixed-ploidy sites use the actual total allele-copy count.
    let mut site = SiteAccumulator::new();
    let mut diag = RawTruthDiag {
        raw_sites_loaded: raw.as_ref().map(|m| m.len() as u64).unwrap_or(0),
        ..Default::default()
    };
    let mut counts = EvalCounts {
        n_matched: 0,
        n_imp_variants: 0,
        n_truth_variants: strong.len() as u64,
        chromosomes: vec![],
    };
    let mut ibuf: Vec<f32> = Vec::new();
    let mut truth_ds = vec![0f32; n];
    let mut truth_metadata = DosageMetadata { native_alt: vec![0.0; n], ploidy: vec![0; n] };
    while let Some(rec) = imp.next_record(&mut ibuf)? {
        counts.n_imp_variants += 1;
        let key = site_key(&rec.0, rec.1, &rec.2, &rec.3);
        if exclude.contains(&key) { continue; }
        let is_snp = rec.2.len() == 1 && rec.3.len() == 1;
        let sv = strong.get(&key);
        let rv = raw.as_ref().and_then(|m| m.get(&key));
        if sv.is_none() { diag.scored_sites_absent_from_strong += 1; }
        if rv.is_some() { diag.scored_sites_matched += 1; }
        for s in 0..n {
            truth_metadata.native_alt[s] = 0.0;
            truth_metadata.ploidy[s] = sv.map(|v| v.metadata.ploidy[s]).unwrap_or(0);
            if let Some(v) = sv {
                if v.dosage2[s] < 0.0 {
                    // An explicit no-call remains unknown, including in homref-absent mode.
                    truth_ds[s] = -1.0;
                    continue;
                }
                if v.dosage2[s] >= 0.5 {
                    truth_ds[s] = v.dosage2[s];
                    truth_metadata.native_alt[s] = v.metadata.native_alt[s];
                    continue;
                }
            }
            if sv.is_none() && rv.map(|v| v.dosage2[s] < 0.0).unwrap_or(false) {
                // Raw explicitly does not know; an absent strong record cannot
                // turn that no-call into evidence for REF. A valid strong REF
                // call, conversely, is not invalidated by a raw no-call.
                truth_ds[s] = -1.0;
                diag.genotypes_missing_raw += 1;
            } else if rv.map(|v| v.dosage2[s] >= 0.5).unwrap_or(false) {
                truth_ds[s] = -1.0;
                diag.genotypes_rescued += 1;
            } else {
                truth_ds[s] = 0.0;
            }
        }
        comb.add_variant(&ibuf, &truth_ds);
        if is_snp { snp.add_variant(&ibuf, &truth_ds); } else { indel.add_variant(&ibuf, &truth_ds); }
        let af = truth_af(&truth_ds, &truth_metadata)?;
        let maf = af.min(1.0 - af);
        let (r2, conc) = site_r2(&ibuf, &truth_ds, n);
        site.add(maf, r2, conc);
        counts.n_matched += 1;
    }
    Ok((comb, snp, indel, counts, site, diag))
}

/// Report what `--truth-raw` actually did. Printed only when a raw truth was
/// supplied, so the default output is unchanged.
pub fn print_raw_truth_diag(d: &RawTruthDiag) {
    if d.raw_sites_loaded == 0 { return; }
    crate::selphi_info!("\n  --truth-raw: {} raw site(s) loaded; {} scored site(s) matched by key; \
{} site(s) absent from the filtered truth; {} genotype(s) rescued from hom-ref scoring",
        d.raw_sites_loaded, d.scored_sites_matched, d.scored_sites_absent_from_strong, d.genotypes_rescued);
    if d.genotypes_missing_raw > 0 {
        crate::selphi_info!("  --truth-raw: {} raw no-call(s) excluded at sites absent from the filtered truth",
            d.genotypes_missing_raw);
    }
    if d.scored_sites_matched == 0 {
        crate::selphi_info!("  --truth-raw: NO scored site matched the raw truth by (contig, pos, REF, ALT) — \
the raw file does not overlap the scored sites (check it covers this region and shares the panel's allele representation)");
    } else if d.genotypes_rescued == 0 {
        crate::selphi_info!("  --truth-raw: keys matched but nothing was rescued — the filtered truth already \
holds every alt call the raw truth has at the scored sites, so the score is correctly unchanged");
    }
}

/// Print the MAF-bin table that accompanies the imputation-R² summary.
/// Bins below one allele copy in the truth cohort are structurally empty.
pub fn print_maf_bins(site_acc: &SiteAccumulator) {
    if site_acc.total_r2_n == 0 { return; }
    crate::selphi_info!("\n{}", crate::log::bold(&format!("{:<20} {:>12} {:>10} {:>12}", "MAF bin", "N variants", "Mean R²", "Concordance")));
    crate::selphi_info!("{}", "-".repeat(60));
    for (bi, &(_, _, label)) in MAF_BINS.iter().enumerate() {
        let n = site_acc.bin_n[bi];
        if n == 0 {
            crate::selphi_info!("{:<20} {:>12} {:>10} {:>12}", label, "0", "N/A", "N/A");
        } else {
            crate::selphi_info!("{:<20} {:>12} {:>10.4} {:>12.4}", label, n,
                site_acc.bin_r2_sum[bi] / n as f64, site_acc.bin_conc_sum[bi] / n as f64);
        }
    }
    crate::selphi_info!("{}", "-".repeat(60));
    crate::selphi_info!("{} {:>12} {}", crate::log::bold(&format!("{:<20}", "OVERALL (site)")),
        site_acc.total_r2_n, crate::log::cyan(&format!("{:>10.4}", site_acc.total_r2_sum / site_acc.total_r2_n as f64)));
}

/// Print MAF-binned summary table.
pub fn print_summary(site_acc: &SiteAccumulator, sample_acc: &SampleAccumulator, counts: &EvalCounts) {
    let n_matched = counts.n_matched;
    crate::selphi_info!("\n{}", crate::log::bold(&format!("{:<20} {:>12} {:>10} {:>12}", "MAF bin", "N variants", "Mean R²", "Concordance")));
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
        let overall_conc = site_acc.total_conc_sum / site_acc.total_n as f64;
        // No site r² exists with fewer than two samples: say so rather than print 0.
        let overall_r2 = if site_acc.total_r2_n > 0 {
            format!("{:>10.4}", site_acc.total_r2_sum / site_acc.total_r2_n as f64)
        } else { format!("{:>10}", "N/A") };
        crate::selphi_info!("{} {:>12} {} {:>12.4}", crate::log::bold(&format!("{:<20}", "OVERALL")), n_matched, crate::log::cyan(&overall_r2), overall_conc);
    }

    // Per-sample summary
    let sample_r2 = sample_acc.compute_r2();
    let sample_conc = sample_acc.compute_concordance();
    if !sample_r2.is_empty() {
        let finite: Vec<f64> = sample_r2.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_r2 = mean_sample_r2(sample_acc).0;
        let min_r2 = finite.iter().copied().reduce(f64::min).unwrap_or(f64::NAN);
        let max_r2 = finite.iter().copied().reduce(f64::max).unwrap_or(f64::NAN);
        let called: Vec<f64> = sample_conc.iter().copied().filter(|v| v.is_finite()).collect();
        let mean_conc = if called.is_empty() { f64::NAN }
            else { called.iter().sum::<f64>() / called.len() as f64 };
        crate::selphi_info!("\nPer-sample (n={}):", sample_acc.n_samples);
        crate::selphi_info!("  R²:          mean={}, min={:.6}, max={:.6}", crate::log::cyan(&format!("{:.6}", mean_r2)), min_r2, max_r2);
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
    // Both files have variants but nothing matched → almost always a scope/contig
    // mismatch (e.g. a whole-genome truth vs a single-chromosome imputed file, or
    // disjoint chromosomes). The merge keys on (normalized-contig, pos), so a 0
    // here means the contig SETS don't overlap — not a real accuracy of zero.
    if n_matched == 0 && counts.n_imp_variants > 0 && counts.n_truth_variants > 0 {
        crate::selphi_info!("  [WARN] 0 variants matched despite non-empty inputs — imputed and truth \
likely cover different chromosomes (check that both span the same contig set; chr-prefix is handled).");
    }
}

/// Mean per-sample R² over an accumulator (the imputation-R² headline).
fn mean_sample_r2(acc: &SampleAccumulator) -> (f64, usize) {
    let r2 = acc.compute_r2();
    let used: Vec<f64> = r2.into_iter().filter(|r| r.is_finite()).collect();
    if used.is_empty() { (f64::NAN, 0) } else { (used.iter().sum::<f64>() / used.len() as f64, used.len()) }
}

/// Print the standard imputation-R² summary (combined always; SNP/indel when `by_type`).
pub fn print_imputation_summary(
    comb: &SampleAccumulator, snp: &SampleAccumulator, indel: &SampleAccumulator,
    by_type: bool, counts: &EvalCounts, n_excluded: u64,
) {
    crate::selphi_info!("\n{}", crate::log::bold(&format!("{:<14} {:>14} {:>12}", "Type", "N sites", "Mean R²")));
    crate::selphi_info!("{}", "-".repeat(42));
    if by_type {
        let (rs, _) = mean_sample_r2(snp);
        let (ri, _) = mean_sample_r2(indel);
        crate::selphi_info!("{:<14} {:>14} {:>12.4}", "SNP", snp.n_variants, rs);
        crate::selphi_info!("{:<14} {:>14} {:>12.4}", "indel", indel.n_variants, ri);
        crate::selphi_info!("{}", "-".repeat(42));
    }
    let (rc, nc) = mean_sample_r2(comb);
    crate::selphi_info!("{} {:>14} {}", crate::log::bold(&format!("{:<14}", "COMBINED")),
        comb.n_variants, crate::log::cyan(&format!("{:>12.4}", rc)));
    crate::selphi_info!("\nPer-sample (n={}): combined mean R² = {}",
        nc, crate::log::cyan(&format!("{:.6}", rc)));
    crate::selphi_info!("  Imputed:  {} sites", counts.n_imp_variants);
    crate::selphi_info!("  Truth:    {} sites (strong)", counts.n_truth_variants);
    crate::selphi_info!("  Scored:   {} sites  ({} excluded)", counts.n_matched, n_excluded);
}

/// Write the imputation-R² summary to JSON.
pub fn write_imputation_json(
    path: &Path, comb: &SampleAccumulator, snp: &SampleAccumulator, indel: &SampleAccumulator,
    by_type: bool, counts: &EvalCounts, sample_names: Option<&[String]>,
    site: Option<&SiteAccumulator>, raw_diag: Option<&RawTruthDiag>,
) -> io::Result<()> {
    use std::io::Write;
    let round6 = |x: f64| (x * 1e6).round() / 1e6;
    let mut map = serde_json::Map::new();
    let (rc, nc) = mean_sample_r2(comb);
    map.insert("combined_mean_r2".into(), serde_json::json!(round6(rc)));
    map.insert("n_samples".into(), serde_json::json!(nc));
    map.insert("n_sites_scored".into(), serde_json::json!(counts.n_matched));
    map.insert("n_imputed_sites".into(), serde_json::json!(counts.n_imp_variants));
    map.insert("n_truth_strong_sites".into(), serde_json::json!(counts.n_truth_variants));
    if by_type {
        let (rs, _) = mean_sample_r2(snp);
        let (ri, _) = mean_sample_r2(indel);
        map.insert("snp_mean_r2".into(), serde_json::json!(round6(rs)));
        map.insert("snp_n_sites".into(), serde_json::json!(snp.n_variants));
        map.insert("indel_mean_r2".into(), serde_json::json!(round6(ri)));
        map.insert("indel_n_sites".into(), serde_json::json!(indel.n_variants));
    }
    if let Some(names) = sample_names {
        let r2 = comb.compute_r2();
        let per: serde_json::Map<String, serde_json::Value> = names.iter().enumerate()
            .map(|(i, nm)| (nm.clone(), serde_json::json!(round6(r2[i])))).collect();
        map.insert("per_sample_combined_r2".into(), serde_json::Value::Object(per));
        let scored: serde_json::Map<String, serde_json::Value> = names.iter().enumerate()
            .map(|(i, nm)| (nm.clone(), serde_json::json!(comb.n_total[i]))).collect();
        map.insert("per_sample_n_scored".into(), serde_json::Value::Object(scored));
    }
    if let Some(d) = raw_diag.filter(|d| d.raw_sites_loaded > 0) {
        let mut m = serde_json::Map::new();
        m.insert("raw_sites_loaded".into(), serde_json::json!(d.raw_sites_loaded));
        m.insert("scored_sites_matched".into(), serde_json::json!(d.scored_sites_matched));
        m.insert("scored_sites_absent_from_strong".into(), serde_json::json!(d.scored_sites_absent_from_strong));
        m.insert("genotypes_rescued".into(), serde_json::json!(d.genotypes_rescued));
        m.insert("genotypes_missing_raw".into(), serde_json::json!(d.genotypes_missing_raw));
        map.insert("truth_raw_diag".into(), serde_json::Value::Object(m));
    }
    // Per-MAF-bin site R², under `maf_bins` so the top-level keys above stay
    // exactly as they were for anything already parsing this file.
    if let Some(sa) = site {
        let mut bins = serde_json::Map::new();
        for (bi, &(_, _, label)) in MAF_BINS.iter().enumerate() {
            let n = sa.bin_n[bi];
            let mut bin = serde_json::Map::new();
            bin.insert("n".into(), serde_json::json!(n));
            if n > 0 {
                bin.insert("mean_r2".into(), serde_json::json!(round6(sa.bin_r2_sum[bi] / n as f64)));
                bin.insert("concordance".into(), serde_json::json!(round6(sa.bin_conc_sum[bi] / n as f64)));
            }
            bins.insert(label.to_string(), serde_json::Value::Object(bin));
        }
        map.insert("maf_bins".into(), serde_json::Value::Object(bins));
        if sa.total_r2_n > 0 {
            map.insert("overall_site_r2".into(),
                serde_json::json!(round6(sa.total_r2_sum / sa.total_r2_n as f64)));
            map.insert("overall_site_r2_n".into(), serde_json::json!(sa.total_r2_n));
        }
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(serde_json::to_string_pretty(&serde_json::Value::Object(map))?.as_bytes())?;
    Ok(())
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
        // null when no site has two scored samples (a one-sample run): there is no site r².
        let r2 = (site_acc.total_r2_n > 0).then(|| site_acc.total_r2_sum / site_acc.total_r2_n as f64);
        overall.insert("mean_r2".into(), serde_json::json!(r2.map(|r| (r * 1e6).round() / 1e6)));
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
        // Average over the samples that HAVE a value; a NaN means undefined.
        let n_def = sample_r2.iter().filter(|v| v.is_finite()).count();
        let mean: f64 = if n_def > 0 {
            sample_r2.iter().filter(|v| v.is_finite()).sum::<f64>() / n_def as f64
        } else { f64::NAN };
        map.insert("per_sample_mean_r2".into(), serde_json::json!((mean * 1e6).round() / 1e6));

        // Per-sample detail array
        let per_sample: Vec<serde_json::Value> = (0..sample_acc.n_samples).map(|s| {
            let mut obj = serde_json::Map::new();
            if let Some(names) = sample_names {
                obj.insert("sample".into(), serde_json::json!(names[s]));
            }
            // NaN is not representable in JSON: emit null so a consumer sees
            // "undefined" rather than a zero it would average in.
            let r2j = if sample_r2[s].is_finite() {
                serde_json::json!((sample_r2[s] * 1e6).round() / 1e6)
            } else { serde_json::Value::Null };
            let cj = if sample_conc[s].is_finite() {
                serde_json::json!((sample_conc[s] * 1e6).round() / 1e6)
            } else { serde_json::Value::Null };
            obj.insert("r2".into(), r2j);
            obj.insert("concordance".into(), cj);
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
    use super::*;
    use std::io::Write;

    fn text_row(format: &str, samples: &[&str], n: usize, indices: Option<&[usize]>,
        options: InputDosageOptions) -> io::Result<(Vec<f32>, DosageMetadata)> {
        let line = format!("X\t100\t.\tA\tG\t.\tPASS\t.\t{format}\t{}\n", samples.join("\t"));
        let mut doses = Vec::new();
        let mut metadata = DosageMetadata::default();
        assert!(parse_vcf_record(line.as_bytes(), n, indices, options, &mut doses,
            &mut metadata, &mut Vec::new(), false)?.is_some());
        Ok((doses, metadata))
    }

    fn native() -> InputDosageOptions { InputDosageOptions::default() }
    fn diploid() -> InputDosageOptions {
        InputDosageOptions { ds_scale: DsScale::Diploid, fallback_ploidy: None }
    }

    #[test]
    fn native_gt_normalizes_mixed_ploidy_and_preserves_missingness() {
        let (d, m) = text_row("GT", &["0", "1", "0/0", "0|1", "1/1", "./1", "./.", ".", "0/1/1"], 9, None, native()).unwrap();
        assert_eq!(&d[..8], &[0.0, 2.0, 0.0, 1.0, 2.0, -1.0, -1.0, -1.0]);
        assert_eq!(d[8], 4.0 / 3.0);
        assert_eq!(m.ploidy, vec![1, 1, 2, 2, 2, 2, 2, 0, 3]);
        assert_eq!(&m.native_alt[..5], &[0.0, 1.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn absent_vcf_fields_are_missing_and_metadata_reindexes_with_dosage() {
        let (d, m) = text_row("GT", &["1"], 3, None, native()).unwrap();
        assert_eq!(d, vec![2.0, -1.0, -1.0]);
        assert_eq!(m.ploidy, vec![1, 0, 0]);
        let (d, m) = text_row("GT", &["1", ".", "0/1"], 3, Some(&[2, 0]), native()).unwrap();
        assert_eq!(d, vec![1.0, 2.0]);
        assert_eq!(m.ploidy, vec![2, 1]);
        let (d, _) = text_row("DP:GT", &["9", "9:"], 2, None, native()).unwrap();
        assert_eq!(d, vec![-1.0, -1.0]);
        let (d, _) = text_row("DP", &["9"], 1, None, native()).unwrap();
        assert_eq!(d, vec![-1.0]);
    }

    #[test]
    fn ds_is_authoritative_in_both_vcf_format_orders() {
        let (a, am) = text_row("DS:GT", &["1:1", "0.5:0/1", ".:1/1", "0.25:./."], 4, None, native()).unwrap();
        let (b, bm) = text_row("GT:DS", &["1:1", "0/1:0.5", "1/1:.", "./.:0.25"], 4, None, native()).unwrap();
        assert_eq!(a, vec![2.0, 0.5, -1.0, 0.25]);
        assert_eq!(a, b);
        assert_eq!(am.native_alt, bm.native_alt);
        assert_eq!(am.ploidy, vec![1, 2, 2, 2]);
        assert_eq!(truth_af(&a, &am).unwrap(), 1.75 / 5.0);
    }

    #[test]
    fn native_ds_requires_known_ploidy_or_an_explicit_fallback() {
        assert!(text_row("DS", &["1"], 1, None, native()).is_err());
        assert!(text_row("GT:DS", &[".:1"], 1, None, native()).is_err());
        let opts = InputDosageOptions { fallback_ploidy: Some(1), ..native() };
        let (d, m) = text_row("DS", &["1"], 1, None, opts).unwrap();
        assert_eq!(d, vec![2.0]);
        assert_eq!(m.ploidy, vec![1]);
        let (d, m) = text_row("GT:DS", &["./.:1"], 1, None, opts).unwrap();
        assert_eq!(d, vec![1.0]);
        assert_eq!(m.ploidy, vec![2]); // fallback never overrides known GT ploidy
        assert!(text_row("GT:DS", &["1:2"], 1, None, native()).is_err());
        assert!(InputDosageOptions { fallback_ploidy: Some(0), ..native() }.validate().is_err());
    }

    #[test]
    fn ds_missing_and_nonfinite_do_not_require_ploidy_or_fall_back_to_gt() {
        let (d, _) = text_row("GT:DS", &["1:.", "1/1:nan", "1/1:inf", ".:."], 4, None, native()).unwrap();
        assert_eq!(d, vec![-1.0; 4]);
        // Unknown ploidy in a non-shared sample cannot reject a selected valid sample.
        let (d, _) = text_row("GT:DS", &[".:1", "0/1:0.75"], 2, Some(&[1]), native()).unwrap();
        assert_eq!(d, vec![0.75]);
    }

    #[test]
    fn diploid_ds_units_do_not_invent_native_ploidy_for_truth_af() {
        let (d, m) = text_row("DS", &["0", "1"], 2, None, diploid()).unwrap();
        assert_eq!(d, vec![0.0, 1.0]);
        assert_eq!(m.ploidy, vec![0, 0]);
        assert!(truth_af(&d, &m).is_err());
        let (d, m) = text_row("GT:DS", &[".:0", "0/1:1"], 2, None, diploid()).unwrap();
        assert!(truth_af(&d, &m).is_err()); // unknown REF P also affects denominator
        let (d, m) = text_row("DS", &["0", "0"], 2, None, diploid()).unwrap();
        assert_eq!(truth_af(&d, &m).unwrap(), 0.0);
        let opts = InputDosageOptions { fallback_ploidy: Some(2), ..diploid() };
        let (d, m) = text_row("DS", &["0", "1"], 2, None, opts).unwrap();
        assert_eq!(truth_af(&d, &m).unwrap(), 0.25);
    }

    #[test]
    fn truth_ds_disagreement_with_gt_uses_expected_alt_count_consistently() {
        let (d, m) = text_row("GT:DS", &["0/0:1.25", "1/1:0.25"], 2, None, native()).unwrap();
        assert_eq!(truth_af(&d, &m).unwrap(), 1.5 / 4.0);
        let (d, m) = text_row("GT:DS", &["0:1", "0/0:0"], 2, None, diploid()).unwrap();
        assert_eq!(d, vec![1.0, 0.0]);
        assert_eq!(truth_af(&d, &m).unwrap(), 0.5 / 3.0);
    }

    fn integers(tid: u8, values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|&v| match tid {
            1 => vec![v as i8 as u8],
            2 => (v as i16).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        }).collect()
    }

    #[test]
    fn bcf_gt_int8_int16_int32_vector_ends_and_missing_are_per_sample() {
        for tid in 1..=3 {
            let missing = match tid { 1 => i8::MIN as i32, 2 => i16::MIN as i32, _ => i32::MIN };
            let eov = missing + 1;
            for (values, alt, p) in [
                (vec![4, eov], Some(1), Some(1)),
                (vec![2, 5], Some(1), Some(2)),
                (vec![5, 5], Some(2), Some(2)),
                (vec![0, 4], None, Some(2)),
                (vec![1, 4], None, Some(2)),
                (vec![missing, 4], None, Some(2)),
                (vec![0, eov], None, None),
                (vec![0, 0], None, Some(2)),
                (vec![eov, eov], None, None),
            ] {
                let call = parse_gt_bcf(&integers(tid, &values), tid);
                assert_eq!((call.alt_count, call.ploidy), (alt, p), "tid={tid} raw={values:?}");
            }
        }
    }

    fn bcf_fields(gt_first: bool, tid: u8) -> Vec<u8> {
        let end = match tid { 1 => i8::MIN as i32 + 1, 2 => i16::MIN as i32 + 1, _ => i32::MIN + 1 };
        // key GT=1, stride=2; sample0 haploid, sample1 missing diploid, sample2 diploid REF.
        let mut gt = vec![0x11, 1, 0x20 | tid];
        gt.extend(integers(tid, &[4, end, 0, 0, 2, 2]));
        let mut ds = vec![0x11, 2, 0x15];
        for d in [1.0f32, f32::NAN, 0.5] { ds.extend(d.to_le_bytes()); }
        if gt_first { gt.extend(ds); gt } else { ds.extend(gt); ds }
    }

    #[test]
    fn bcf_ds_before_gt_and_selective_reindex_preserve_units_and_missing() {
        for tid in 1..=3 {
            for gt_first in [false, true] {
                let mut calls = Vec::new();
                decode_bcf_calls(&bcf_fields(gt_first, tid), 2, 3, 1, Some(2), Some(&[2, 0, 1]), &mut calls).unwrap();
                let (mut d, mut m) = (Vec::new(), DosageMetadata::default());
                normalize_row(&calls, None, native(), &mut d, &mut m).unwrap();
                assert_eq!(d, vec![0.5, 2.0, -1.0]);
                assert_eq!(m.ploidy, vec![2, 1, 2]);
                assert_eq!(m.native_alt, vec![0.5, 1.0, 0.0]);
            }
        }
        let mut truncated = bcf_fields(false, 2);
        truncated.pop();
        assert!(decode_bcf_calls(&truncated, 2, 3, 1, Some(2), None, &mut Vec::new()).is_err());
    }

    #[test]
    fn mixed_ploidy_af_and_metrics_use_distinct_units() {
        let (truth, metadata) = text_row("GT", &["1", "0/0"], 2, None, native()).unwrap();
        assert_eq!(truth_af(&truth, &metadata).unwrap(), 1.0 / 3.0);
        let (r2, concordance) = site_r2(&[2.0, 0.0], &truth, 2);
        assert_eq!((r2, concordance), (1.0, 1.0));
        let mut bins = SiteAccumulator::new();
        bins.add(truth_af(&truth, &metadata).unwrap(), r2, concordance);
        assert_eq!(bins.bin_n[8], 1);
        let mut sample = SampleAccumulator::new(1);
        for (gt, pred) in [("1/1", 2.0), ("0/0", 0.0), ("1", 2.0)] {
            let (d, _) = text_row("GT", &[gt], 1, None, native()).unwrap();
            sample.add_variant(&[pred], &d);
        }
        assert_eq!(sample.compute_r2(), vec![1.0]);
    }

    #[test]
    fn ref_alt_swap_applies_after_normalizing_haploid_dosage() {
        let (mut predicted, _) = text_row("GT", &["1", "0/0"], 2, None, native()).unwrap();
        let (matched, swapped) = match_alleles(b"G", b"A", b"A", b"G");
        assert!(matched && swapped);
        for d in &mut predicted { if *d >= 0.0 { *d = 2.0 - *d; } }
        assert_eq!(predicted, vec![0.0, 2.0]);
        assert_eq!(site_r2(&predicted, &[0.0, 2.0], 2), (1.0, 1.0));
    }

    #[test]
    fn valid_diploid_dosages_and_accumulation_remain_bit_identical() {
        let cases = [("0/0", 0.0f32), ("0/1", 0.1234567), ("1/1", 2.0), ("0/1", 0.5), ("0/0", 1.234567)];
        let mut legacy = SampleAccumulator::new(1);
        let mut normalized = SampleAccumulator::new(1);
        for (gt, ds) in cases {
            let text = format!("{gt}:{ds}");
            let (d, m) = text_row("GT:DS", &[&text], 1, None, native()).unwrap();
            assert_eq!(d[0].to_bits(), ds.to_bits());
            assert_eq!(m.native_alt[0].to_bits(), (ds as f64).to_bits());
            let gt_count = match gt { "0/0" => 0.0, "0/1" => 1.0, _ => 2.0 };
            legacy.add_variant(&[ds], &[gt_count]);
            let (truth, _) = text_row("GT", &[gt], 1, None, native()).unwrap();
            normalized.add_variant(&d, &truth);
        }
        assert_eq!(legacy.sum_ds, normalized.sum_ds);
        assert_eq!(legacy.sum_gt, normalized.sum_gt);
        assert_eq!(legacy.sum_ds_gt, normalized.sum_ds_gt);
        assert_eq!(legacy.n_correct, normalized.n_correct);
        assert_eq!(legacy.compute_r2()[0].to_bits(), normalized.compute_r2()[0].to_bits());
    }

    fn fixture(dir: &Path, name: &str, samples: &[&str], rows: &[(&str, &[&str])]) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut writer = noodles_bgzf::io::Writer::new(std::fs::File::create(&path).unwrap());
        writeln!(writer, "##fileformat=VCFv4.2\n##contig=<ID=X,length=10000>\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n##FORMAT=<ID=DS,Number=1,Type=Float,Description=\"Dosage\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}", samples.join("\t")).unwrap();
        for (i, (format, values)) in rows.iter().enumerate() {
            writeln!(writer, "X\t{}\t.\tA\tG\t.\tPASS\t.\t{}\t{}", (i + 1) * 100, format, values.join("\t")).unwrap();
        }
        writer.try_finish().unwrap();
        path
    }

    #[test]
    fn public_homref_evaluation_preserves_explicit_missing_and_native_af() {
        let dir = tempfile::tempdir().unwrap();
        let samples = vec!["A".to_string(), "B".to_string()];
        let imp = fixture(dir.path(), "imp.vcf.gz", &["B", "A"], &[
            ("GT:DS", &["0/0:0", "1/1:2"]),
            ("GT:DS", &["1/1:2", "0/0:0"]),
            ("GT:DS", &["0/1:1", "1/1:2"]),
            ("GT:DS", &["0/0:0", "1/1:2"]),
        ]);
        let strong = fixture(dir.path(), "strong.vcf.gz", &["A", "B"], &[
            ("GT", &["1", "0/0"]), ("GT", &["0", "1/1"]), ("GT", &[".", "0/1"]),
        ]);
        let raw = fixture(dir.path(), "raw.vcf.gz", &["A", "B"], &[
            ("GT", &["1", "0/0"]), ("GT", &["0", "1/1"]), ("GT", &["1", "0/1"]), ("GT", &["1", "0/0"]),
        ]);
        let (acc, _, _, counts, site, diag) = evaluate_imputation(&imp, &strong, &samples, Some(&raw), &super::super::exclude::SiteExclusion::default()).unwrap();
        assert_eq!(counts.n_matched, 4);
        assert_eq!(acc.n_total, vec![2, 4]);
        assert_eq!(acc.compute_r2(), vec![1.0, 1.0]);
        assert_eq!(site.bin_n[8], 2);
        assert_eq!(diag.genotypes_rescued, 1); // explicit strong no-call is not a raw rescue
        assert_eq!(diag.scored_sites_absent_from_strong, 1);
    }

    #[test]
    fn homref_subthreshold_truth_ds_has_zero_alt_in_af() {
        let dir = tempfile::tempdir().unwrap();
        let names = vec!["A".to_string(), "B".to_string()];
        let strong = fixture(dir.path(), "strong.vcf.gz", &["A", "B"], &[("GT:DS", &["0/0:0.2", "0/1:0.65"])]);
        let imp = fixture(dir.path(), "imp.vcf.gz", &["A", "B"], &[("GT:DS", &["0/0:0", "0/1:0.65"])]);
        let (_, _, _, _, site, _) = evaluate_imputation(&imp, &strong, &names, None, &super::super::exclude::SiteExclusion::default()).unwrap();
        assert_eq!(site.bin_n[7], 1); // AF = 0.65 / 4, not (0.2 + 0.65) / 4
        assert_eq!(site.bin_n[8], 0);
    }

    #[test]
    fn reader_reports_unknown_native_ds_instead_of_silently_ending_stream() {
        let dir = tempfile::tempdir().unwrap();
        let path = fixture(dir.path(), "unknown.vcf.gz", &["A"], &[("DS", &["1"])]);
        let (mut reader, _) = VariantReader::open(&path).unwrap();
        let error = reader.next_record(&mut Vec::new()).unwrap_err().to_string();
        assert!(error.contains("unknown.vcf.gz") && error.contains("ploidy"));
        // Exclusion scans do not need DS units or GT metadata.
        let mut keys = std::collections::HashSet::new();
        assert_eq!(read_site_keys(&path, &mut keys).unwrap(), 1);
    }

    #[test]
    fn all_undefined_and_empty_sample_means_remain_null_in_json() {
        let dir = tempfile::tempdir().unwrap();
        let names = vec!["A".to_string()];
        let truth = fixture(dir.path(), "empty.vcf.gz", &["A"], &[]);
        let imp = fixture(dir.path(), "imp.vcf.gz", &["A"], &[("GT", &["0/0"]), ("GT", &["1/1"])]);
        let (acc, snp, indel, counts, site, diag) = evaluate_imputation(&imp, &truth, &names, None, &super::super::exclude::SiteExclusion::default()).unwrap();
        assert_eq!(acc.n_total, vec![2]);
        assert!(mean_sample_r2(&acc).0.is_nan());
        assert_eq!(mean_sample_r2(&acc).1, 0);
        assert_eq!(site.total_r2_n, 0);
        let json_path = dir.path().join("summary.json");
        write_imputation_json(&json_path, &acc, &snp, &indel, true, &counts, Some(&names), Some(&site), Some(&diag)).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&std::fs::read(json_path).unwrap()).unwrap();
        assert!(json["combined_mean_r2"].is_null());
        assert!(json["per_sample_combined_r2"]["A"].is_null());
        assert_eq!(json["per_sample_n_scored"]["A"], 2);
        assert_eq!(json["n_samples"], 0);
        let empty = SampleAccumulator::new(0);
        assert!(mean_sample_r2(&empty).0.is_nan());
        assert_eq!(mean_sample_r2(&empty).1, 0);
    }

    #[test]
    fn mean_sample_r2_excludes_undefined_samples_with_many_scored_sites() {
        let mut acc = SampleAccumulator::new(2);
        acc.add_variant(&[0.0, 0.0], &[0.0, 0.0]);
        acc.add_variant(&[2.0, 2.0], &[2.0, 0.0]);
        assert_eq!(mean_sample_r2(&acc), (1.0, 1));
    }

    #[test]
    fn invalid_biallelic_gt_has_unknown_ploidy_in_vcf_and_bcf() {
        for gt in [b"2".as_slice(), b"0/2", b"0//1", b"x"] {
            let call = parse_gt_text(gt);
            assert!(call.alt_count.is_none() && call.ploidy.is_none());
        }
        for tid in 1..=3 {
            for values in [&[6][..], &[2, 6][..]] {
                let call = parse_gt_bcf(&integers(tid, values), tid);
                assert!(call.alt_count.is_none() && call.ploidy.is_none());
            }
        }
    }

    #[test]
    fn raw_no_call_is_not_homref_unless_strong_has_valid_ref_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let names = vec!["A".to_string(), "B".to_string()];
        let strong = fixture(dir.path(), "strong.vcf.gz", &["A", "B"], &[
            ("GT", &["0/0", "0/0"]), ("GT", &[".", "./."]),
        ]);
        let raw = fixture(dir.path(), "raw.vcf.gz", &["A", "B"], &[
            ("GT:DS", &[".:.", "0/0:."]), ("GT:DS", &["0/0:0", "0/0:0"]),
            ("GT:DS", &[".:.", "0/0:."]), ("GT:DS", &["0/0:0", "0/0:0"]),
        ]);
        let imp = fixture(dir.path(), "imp.vcf.gz", &["A", "B"], &[
            ("GT", &["0/0", "0/0"]), ("GT", &["0/0", "0/0"]),
            ("GT", &["0/0", "0/0"]), ("GT", &["0/0", "0/0"]),
        ]);
        let (acc, _, _, _, _, diag) = evaluate_imputation(&imp, &strong, &names,
            Some(&raw), &super::super::exclude::SiteExclusion::default()).unwrap();
        assert_eq!(acc.n_total, vec![2, 2]); // strong REF at 100 and absent/known raw REF at 400
        assert_eq!(diag.genotypes_missing_raw, 2); // 300 only; strong missing at 200 excluded separately
        assert_eq!(diag.genotypes_rescued, 0);
    }


    // ---- Fast evaluator paths vs. verbatim copies of the previous code ----

    /// Normalise one sample's call and append it (`i` = its column, for the error text).
    #[inline]
    fn old_push_normalized(call: SampleCall, i: usize, options: InputDosageOptions,
        ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata) -> io::Result<()> {
        let (dose, native_alt, ploidy) = normalize_call(call, options)
            .map_err(|e| invalid_dosage(format!("sample column {}: {e}", i + 1)))?;
        ds_buf.push(dose);
        metadata.native_alt.push(native_alt);
        metadata.ploidy.push(ploidy);
        Ok(())
    }

    fn old_normalize_row(calls: &[SampleCall], indices: Option<&[usize]>, options: InputDosageOptions,
        ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata) -> io::Result<()> {
        ds_buf.clear();
        metadata.clear();
        if let Some(indices) = indices {
            for &i in indices { old_push_normalized(calls.get(i).copied().unwrap_or_default(), i, options, ds_buf, metadata)?; }
        } else {
            for (i, &c) in calls.iter().enumerate() { old_push_normalized(c, i, options, ds_buf, metadata)?; }
        }
        Ok(())
    }

    fn old_parse_vcf_record(line: &[u8], n_samples: usize, indices: Option<&[usize]>,
        options: InputDosageOptions, ds_buf: &mut Vec<f32>, metadata: &mut DosageMetadata,
        calls: &mut Vec<SampleCall>, skip_gt: bool) -> io::Result<Option<VariantRecord>> {
        let line = strip_line_endings(line);
        if line.is_empty() || line[0] == b'#' { return Ok(None); }
        let mut tabs = [0usize; 9];
        let mut nt = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' { tabs[nt] = i; nt += 1; if nt == 9 { break; } }
        }
        if nt < 9 { return Ok(None); }
        let chrom = line[..tabs[0]].to_vec();
        let pos = std::str::from_utf8(&line[tabs[0]+1..tabs[1]]).ok()
            .and_then(|s| s.parse::<i64>().ok()).ok_or_else(|| invalid_dosage("invalid VCF position"))?;
        let ref_a = line[tabs[2]+1..tabs[3]].to_vec();
        let alt_a = line[tabs[3]+1..tabs[4]].to_vec();
        if alt_a.contains(&b',') { return Ok(None); }
        let record = (chrom, pos, ref_a, alt_a);
        if skip_gt { ds_buf.clear(); metadata.clear(); return Ok(Some(record)); }

        let format = &line[tabs[7]+1..tabs[8]];
        let mut ds_idx = None;
        let mut gt_idx = None;
        for (i, field) in format.split(|&b| b == b':').enumerate() {
            if field == b"DS" { ds_idx = Some(i); }
            if field == b"GT" { gt_idx = Some(i); }
        }
        // Only the fields up to the last of GT/DS are looked at.
        let n_fields = gt_idx.max(ds_idx).map_or(0, |i| i + 1);
        let parse_call = |sample: &[u8]| {
            let mut call = SampleCall { ds: ds_idx.map(|_| -1.0), ..Default::default() };
            for (i, field) in sample.split(|&b| b == b':').take(n_fields).enumerate() {
                if gt_idx == Some(i) { call.gt = parse_gt_text(field); }
                if ds_idx == Some(i) {
                    call.ds = Some(std::str::from_utf8(field).ok().and_then(|s| s.parse::<f32>().ok())
                        .filter(|v| v.is_finite()).unwrap_or(-1.0));
                }
            }
            call
        };
        if indices.is_none() {
            // Every column in file order: normalise as it is read, no call buffer.
            ds_buf.clear();
            metadata.clear();
            let mut n = 0usize;
            for sample in line[tabs[8]+1..].split(|&b| b == b'\t').take(n_samples) {
                old_push_normalized(parse_call(sample), n, options, ds_buf, metadata)?;
                n += 1;
            }
            // Absent/truncated sample columns are missing, never synthetic hom-ref.
            for i in n..n_samples { old_push_normalized(SampleCall::default(), i, options, ds_buf, metadata)?; }
            return Ok(Some(record));
        }
        calls.clear();
        calls.extend(line[tabs[8]+1..].split(|&b| b == b'\t').take(n_samples).map(parse_call));
        calls.resize(n_samples, SampleCall::default());
        old_normalize_row(calls, indices, options, ds_buf, metadata)?;
        Ok(Some(record))
    }

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    #[test]
    fn ds_fast_parse_is_str_parse() {
        let same = |t: &str| {
            let new = parse_ds_f32(t.as_bytes());
            let old = t.parse::<f32>().ok();
            assert_eq!(new.map(f32::to_bits), old.map(f32::to_bits), "{t:?}");
        };
        // Every writer dosage and AP: up to three decimals in [0, 2], as written
        // (trailing zeros trimmed) and zero-padded.
        for k in 0..=2000u32 {
            same(&format!("{}", k as f64 / 1000.0));
            same(&format!("{:.3}", k as f64 / 1000.0));
            same(&format!("{}.{:03}", k / 1000, k % 1000));
        }
        // Random plain decimals of 1-9 digits, every decimal-point position.
        let mut r = Xs(0x9E3779B97F4A7C15);
        for _ in 0..1_000_000 {
            let nd = 1 + r.below(9) as usize;
            let digits: String = (0..nd).map(|_| char::from(b'0' + r.below(10) as u8)).collect();
            same(&digits);
            let dot = r.below(nd as u64 + 1) as usize;
            same(&format!("{}.{}", &digits[..dot], &digits[dot..]));
        }
        for t in ["", ".", "5.", ".5", "-0", "-1.5", "+1", "1e3", "1E-3", "nan", "NaN", "inf", "-inf",
                  "1.2.3", "0x1", " 1", "1 ", "00000000", "9999999", "99999999", "16777217", "0.0000001",
                  "1677721.7", "8388609", "0.1234567", "12345678", "1.", "0..1", "\u{e9}"] {
            same(t);
        }
    }

    fn random_sample(r: &mut Xs, fields: &[&str]) -> String {
        let gts = ["0/0", "0|1", "1/1", "1|0", "./.", ".", "0", "1", "./1", "0/1/1", "2/1", ""];
        let dss = ["0", "0.5", "1", "1.999", "2", "2.0005", "2.5", ".", "nan", "-1", "0.1234567", "1e-3", "", "x", "1.0001"];
        fields.iter().map(|f| match *f {
            "GT" => gts[r.below(gts.len() as u64) as usize].to_string(),
            "DS" => dss[r.below(dss.len() as u64) as usize].to_string(),
            _ => "7".to_string(),
        }).collect::<Vec<_>>().join(":")
    }

    #[test]
    fn filtered_vcf_parse_matches_previous_reader() {
        let formats: [&[&str]; 5] = [&["GT"], &["GT", "DS"], &["DS", "GT"], &["AP", "DS"], &["GT", "AP", "DS", "XX"]];
        let mut r = Xs(12345);
        let mut n_err = 0;
        for case in 0..20_000 {
            let fmt = formats[r.below(5) as usize];
            let n = 1 + r.below(12) as usize;
            // Sometimes fewer or more columns than the header declares.
            let n_cols = match r.below(6) { 0 => r.below(n as u64) as usize, 1 => n + 1 + r.below(3) as usize, _ => n };
            let samples: Vec<String> = (0..n_cols).map(|_| random_sample(&mut r, fmt)).collect();
            let line = format!("22\t{}\trs1\tA\tG\t.\tPASS\t.\t{}\t{}\n", 100 + case, fmt.join(":"), samples.join("\t"));
            let options = match r.below(3) {
                0 => InputDosageOptions::default(),
                1 => InputDosageOptions { ds_scale: DsScale::Diploid, fallback_ploidy: None },
                _ => InputDosageOptions { ds_scale: DsScale::Native, fallback_ploidy: Some(2) },
            };
            let indices: Option<Vec<usize>> = match r.below(4) {
                0 => None,
                1 => Some((0..n).collect()),
                2 => Some((0..1 + r.below(n as u64) as usize).map(|_| r.below(n as u64 + 2) as usize).collect()),
                _ => { let mut v: Vec<usize> = (0..n).collect(); v.reverse(); v.truncate(1 + r.below(n as u64) as usize); Some(v) }
            };
            let (mut d0, mut m0) = (vec![9.0f32], DosageMetadata { native_alt: vec![7.0], ploidy: vec![7] });
            let old = old_parse_vcf_record(line.as_bytes(), n, indices.as_deref(), options, &mut d0, &mut m0, &mut Vec::new(), false);
            for keep in [true, false] {
                let filter = indices.as_ref().map(|ix| ColFilter::new(ix.clone(), n));
                let (mut d1, mut m1) = (vec![9.0f32], DosageMetadata { native_alt: vec![7.0], ploidy: vec![7] });
                let new = parse_vcf_record_with(line.as_bytes(), n, filter.as_ref(), options, &mut d1, &mut m1, keep, &mut Vec::new(), false);
                match (&old, &new) {
                    (Ok(a), Ok(b)) => {
                        assert_eq!(a, b);
                        assert_eq!(d0.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), d1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "{line}");
                        if keep {
                            assert_eq!(m0.ploidy, m1.ploidy);
                            assert_eq!(m0.native_alt.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), m1.native_alt.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                        } else {
                            assert!(m1.ploidy.is_empty() && m1.native_alt.is_empty());
                        }
                    }
                    (Err(a), Err(b)) => { assert_eq!(a.to_string(), b.to_string()); n_err += 1; }
                    _ => panic!("old {old:?} new {new:?} on {line}"),
                }
            }
            // The identity selection may be stored as no selection.
            if indices.as_ref().is_some_and(|ix| ix.len() == n && ix.iter().enumerate().all(|(i, &x)| i == x)) {
                let (mut d2, mut m2) = (Vec::new(), DosageMetadata::default());
                let none = parse_vcf_record_with(line.as_bytes(), n, None, options, &mut d2, &mut m2, true, &mut Vec::new(), false);
                match (&old, &none) {
                    (Ok(a), Ok(b)) => { assert_eq!(a, b); assert_eq!(d0, d2); assert_eq!(m0.ploidy, m2.ploidy); }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    _ => panic!("identity: old {old:?} new {none:?} on {line}"),
                }
            }
        }
        assert!(n_err > 100, "error paths barely exercised ({n_err})");
    }
}
