//! Target VCF/BCF reading, phased VCF writing, and variant intersection.

use std::path::Path;

use crate::srp::SrpReader;
use crate::{selphi_error, selphi_info};

// ---------------------------------------------------------------------------
// TargetMarker
// ---------------------------------------------------------------------------

/// Target marker: (chrom, pos, ref_hash, alt_hash)
#[derive(Debug, Clone)]
pub struct TargetMarker {
    pub chrom: String,
    pub pos: i64,
    pub ref_allele: String,
    pub alt_allele: String,
    pub ref_hash: String,
    pub alt_hash: String,
    /// Original variant ID (rsID) from the VCF/SRP/BREF3, "." or "" if none.
    /// Only populated by the panel-phasing cohort readers; the imputation
    /// target readers leave it empty (imputation output IDs come from the panel).
    pub id: String,
}

// ---------------------------------------------------------------------------
// Fast i64 parsing
// ---------------------------------------------------------------------------

/// Fast i64 parsing from ASCII bytes (no String allocation). Returns `-1`
/// as a sentinel if the input contains no digits or any non-digit/non-sign/
/// non-whitespace byte (so callers can `if pos < 1 { continue; }` to skip
/// malformed VCF POS columns instead of silently accepting POS=0).
///
/// `pub(crate)` so the lcWGS PL reader shares this one canonical definition
/// rather than carrying a byte-equivalent copy. Both callers pass the VCF POS
/// column (the bytes strictly between the 1st and 2nd tab of a `\n`-delimited
/// line), which in practice is pure ASCII digits.
#[inline]
pub(crate) fn fast_parse_i64(bytes: &[u8]) -> i64 {
    let mut n: i64 = 0;
    let mut seen_digit = false;
    for &b in bytes {
        if b.is_ascii_digit() {
            n = n * 10 + (b - b'0') as i64;
            seen_digit = true;
        } else if !(b.is_ascii_whitespace() || b == b'+') {
            return -1; // any unexpected byte → invalid POS
        }
    }
    if seen_digit { n } else { -1 }
}

// ---------------------------------------------------------------------------
// Shared VCF-text parsing primitives (used by all three text readers below:
// read_cohort_vcf / read_target_vcf / read_target_vcf_multi_chr). Binary BCF
// is dispatched to read_target_bcf by each reader before these are called.
// ---------------------------------------------------------------------------

/// Read a VCF path fully into a decompressed byte buffer (BGZF `.gz` or plain
/// text). Exits the process on I/O / decompression error. Does NOT handle
/// binary BCF — the caller dispatches that first (and re-checks the `BCF\2\2`
/// magic on the returned buffer for a mis-named `.vcf.gz`).
fn read_vcf_raw(path: &str) -> Vec<u8> {
    use std::io::Read;
    let is_gz = path.ends_with(".gz");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });
    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw)
            .unwrap_or_else(|e| { selphi_error!("BGZF decompress failed for {}: {}", path, e); std::process::exit(1) });
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw)
            .unwrap_or_else(|e| { selphi_error!("Failed to read VCF {}: {}", path, e); std::process::exit(1) });
    }
    raw
}

/// Core fields of one VCF data line, borrowed from the line buffer. `gt_region`
/// is the raw bytes of fields 9+ (per-sample columns), to be handed to
/// [`parse_gt_region`].
struct VcfFields<'a> {
    chrom: &'a str,
    pos: i64,
    id: &'a str,
    ref_allele: &'a str,
    alt_allele: &'a str,
    /// True if the ALT field carried a comma (more than one ALT allele).
    multiallelic: bool,
    gt_region: &'a [u8],
}

/// Split a VCF data line into [`VcfFields`]. Returns `None` for lines that all
/// three readers skip identically: fewer than 9 tabs, `POS < 1` (malformed),
/// or a missing/`.` ALT. The first ALT allele (before any comma) is kept.
fn split_vcf_fields(line: &[u8]) -> Option<VcfFields<'_>> {
    let mut tabs = [0usize; 9];
    let mut n_tabs = 0;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            if n_tabs < 9 { tabs[n_tabs] = i; }
            n_tabs += 1;
            if n_tabs >= 9 { break; }
        }
    }
    if n_tabs < 9 { return None; }

    let pos = fast_parse_i64(&line[tabs[0] + 1..tabs[1]]);
    if pos < 1 { return None; }

    let ref_bytes = &line[tabs[2] + 1..tabs[3]];
    let alt_field = &line[tabs[3] + 1..tabs[4]];
    let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
    let alt_bytes = &alt_field[..alt_end];
    if alt_bytes == b"." || alt_bytes.is_empty() { return None; }

    Some(VcfFields {
        chrom: std::str::from_utf8(&line[..tabs[0]]).unwrap_or(""),
        pos,
        id: std::str::from_utf8(&line[tabs[1] + 1..tabs[2]]).unwrap_or("."),
        ref_allele: std::str::from_utf8(ref_bytes).unwrap_or(""),
        alt_allele: std::str::from_utf8(alt_bytes).unwrap_or(""),
        multiallelic: alt_end < alt_field.len(),
        gt_region: &line[tabs[8] + 1..],
    })
}

/// Parse the per-sample diploid GT region (VCF fields 9+) into `[a0, a1]`
/// pairs, biallelic-projected to {0,1}: any ALT allele index (≥1, incl.
/// multiallelic 2+) folds to 1; REF / missing → 0. A HAPLOID GT (`len < 3`,
/// e.g. chrX males `"1"`) is read as `[allele, 0]` — matching the binary-BCF
/// reader (`read_target_bcf`), which reads the real allele — so the call
/// reaches the downstream chrX-haploid handling (detect_haploid_chrx /
/// reset_haploid_hets) instead of being silently dropped to `0|0`. Decrements
/// `*phase_checks` over the leading samples and clears `*is_phased` when a
/// checked sample uses the unphased `/` separator. (Replaces three byte-
/// identical copies; the former cohort `bin` and `.min(1)` clamps are equal.)
fn parse_gt_region(
    gt_region: &[u8],
    n_samples: usize,
    is_phased: &mut bool,
    phase_checks: &mut i32,
) -> Vec<[u8; 2]> {
    let mut var_gts = Vec::with_capacity(n_samples);
    let mut field_start = 0;
    for _ in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter()
            .position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
        let gt = &field[..gt_end];

        if *phase_checks > 0 {
            if gt.contains(&b'/') { *is_phased = false; }
            *phase_checks -= 1;
        }

        let (a0, a1) = if gt.len() >= 3 {
            // Diploid "a/b" or "a|b" (separator at index 1).
            (if gt[0].is_ascii_digit() { (gt[0] - b'0').min(1) } else { 0 },
             if gt[2].is_ascii_digit() { (gt[2] - b'0').min(1) } else { 0 })
        } else if !gt.is_empty() {
            // Haploid "a" — keep the allele in slot 0 (matches read_target_bcf).
            (if gt[0].is_ascii_digit() { (gt[0] - b'0').min(1) } else { 0 }, 0)
        } else {
            (0, 0)
        };
        var_gts.push([a0, a1]);

        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
    var_gts
}

// ---------------------------------------------------------------------------
// read_cohort_vcf  (panel self-phasing — no SRP intersection)
// ---------------------------------------------------------------------------

/// Read a full cohort VCF.gz for de-novo panel phasing: ALL biallelic
/// variants × ALL samples, no reference-panel intersection. Returns
/// (sample_names, markers, genotypes, is_phased). `genotypes[v][s] =
/// [allele0, allele1]` with missing alleles coerced to 0. Allele hashes in
/// the returned markers are left empty (not needed for panel output).
pub fn read_cohort_vcf(
    path: &str,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    // Real binary BCF → noodles decoder (captures the variant ID for panel output).
    if path.ends_with(".bcf") {
        return read_target_bcf(path, false, true);
    }
    let raw = read_vcf_raw(path);
    if raw.starts_with(b"BCF\x02\x02") {
        return read_target_bcf(path, false, true);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut is_phased = true;
    let mut phase_checks = 10i32;
    let mut sample_names: Vec<String> = Vec::new();
    let mut n_multiallelic = 0usize;

    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }
        let Some(f) = split_vcf_fields(line) else { continue };
        if f.multiallelic { n_multiallelic += 1; }
        // Panel phasing treats the cohort as biallelic (first ALT kept); the GT
        // binarisation in parse_gt_region collapses higher ALT indices to 1.
        markers.push(TargetMarker {
            chrom: f.chrom.to_string(), pos: f.pos,
            ref_allele: f.ref_allele.to_string(), alt_allele: f.alt_allele.to_string(),
            ref_hash: String::new(), alt_hash: String::new(), id: f.id.to_string(),
        });
        genotypes.push(parse_gt_region(f.gt_region, sample_names.len(), &mut is_phased, &mut phase_checks));
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }
    if n_multiallelic > 0 {
        selphi_info!("  WARNING: {} multi-allelic sites — kept first ALT, genotypes binarised (ref vs any-alt). Split multiallelics beforehand for exact handling.", n_multiallelic);
    }
    (sample_names, markers, genotypes, is_phased)
}

// ---------------------------------------------------------------------------
// read_target_vcf
// ---------------------------------------------------------------------------

/// First-ALT + per-sample diploid `[a0,a1]` GT extraction from a decoded BCF
/// record set. Shared binary-BCF target reader: a real BCF is BGZF-wrapped
/// BINARY (magic `BCF\2\2`), which the VCF byte-scanner below cannot read —
/// it would find zero markers. Uses the same noodles-bcf decoder + dispatch
/// pattern as the lcWGS PL reader. `hash_alleles` mirrors [`read_target_vcf`].
fn read_target_bcf(
    path: &str, hash_alleles: bool, capture_id: bool,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;
    use noodles_vcf::variant::record::samples::series::value::genotype::Phasing;

    let mut reader = bcf::io::reader::Builder::default().build_from_path(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open BCF {}: {}", path, e); std::process::exit(1) });
    let header = reader.read_header()
        .unwrap_or_else(|e| { selphi_error!("Cannot read BCF header {}: {}", path, e); std::process::exit(1) });
    let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut is_phased = true;
    let mut phase_checks = 10i32;

    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let alt_allele = match rec.alternate_bases().as_ref().first() {
            Some(a) if a != "." && !a.is_empty() => a.clone(),
            _ => continue,
        };
        let chrom = rec.reference_sequence_name().to_string();
        let ref_allele = rec.reference_bases().to_string();
        // Hash mode matches each text reader: cohort (capture_id) leaves hashes
        // empty; the imputation readers hash (when the panel uses hashed allele
        // ids) or store the plain allele otherwise.
        let (ref_hash, alt_hash) = if capture_id {
            (String::new(), String::new())
        } else if hash_alleles {
            (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
        } else {
            (ref_allele.clone(), alt_allele.clone())
        };
        let id = if capture_id {
            rec.ids().as_ref().iter().next().cloned().unwrap_or_else(|| ".".to_string())
        } else {
            String::new()
        };
        markers.push(TargetMarker { chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash, id });

        let mut var_gts = Vec::with_capacity(sample_names.len());
        for sample in rec.samples().values() {
            let (a0, a1, phased) = match sample.get("GT").flatten() {
                Some(Value::Genotype(gt)) => {
                    let al = gt.as_ref();
                    // Biallelic projection: any ALT allele index (>=1, incl. multiallelic
                    // 2+) folds to 1; missing/REF -> 0. Keeps the whole pipeline on the
                    // 0/1 bitmatrix domain so chip passthrough never emits a GT allele
                    // index beyond the single output ALT. No-op on biallelic input.
                    let a0 = al.first().and_then(|a| a.position()).unwrap_or(0).min(1) as u8;
                    let a1 = al.get(1).and_then(|a| a.position()).unwrap_or(0).min(1) as u8;
                    // VCF phasing is carried on the allele separators after the
                    // first; a diploid is phased iff its 2nd allele is Phased.
                    let phased = al.get(1).map(|a| a.phasing() == Phasing::Phased).unwrap_or(true);
                    (a0, a1, phased)
                }
                _ => (0, 0, true),
            };
            if phase_checks > 0 { if !phased { is_phased = false; } phase_checks -= 1; }
            var_gts.push([a0, a1]);
        }
        genotypes.push(var_gts);
    }
    (sample_names, markers, genotypes, is_phased)
}

/// Read target VCF/BCF using noodles bgzf + manual text parsing (real binary
/// BCF is dispatched to [`read_target_bcf`]).
/// Pure Rust — no bcftools dependency.
pub fn read_target_vcf(
    path: &str, srp: &SrpReader,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };

    // Real (binary) BCF → noodles decoder; the byte-scan below only handles VCF
    // text. Dispatch on extension first (avoids a double decompress), then sniff.
    if path.ends_with(".bcf") {
        return read_target_bcf(path, hash_alleles, false);
    }

    let raw = read_vcf_raw(path);
    // Content-sniff: a BGZF-wrapped binary BCF (magic "BCF\2\2") misnamed .vcf.gz.
    if raw.starts_with(b"BCF\x02\x02") {
        return read_target_bcf(path, hash_alleles, false);
    }

    let mut markers = Vec::new();
    let mut genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut is_phased = true;
    let mut phase_checks = 10i32;
    let mut sample_names: Vec<String> = Vec::new();

    // Parse from byte buffer — zero per-line allocations
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }

        let Some(f) = split_vcf_fields(line) else { continue };
        let (ref_hash, alt_hash) = if hash_alleles {
            (crate::srp::blake2b_hex(f.ref_allele), crate::srp::blake2b_hex(f.alt_allele))
        } else {
            (f.ref_allele.to_string(), f.alt_allele.to_string())
        };
        markers.push(TargetMarker {
            chrom: f.chrom.to_string(), pos: f.pos,
            ref_allele: f.ref_allele.to_string(), alt_allele: f.alt_allele.to_string(),
            ref_hash, alt_hash, id: String::new(),
        });
        genotypes.push(parse_gt_region(f.gt_region, sample_names.len(), &mut is_phased, &mut phase_checks));
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }

    (sample_names, markers, genotypes, is_phased)
}

// ---------------------------------------------------------------------------
// Shared GT-only VCF.gz writer building blocks (header + per-record GT row),
// used by both write_phased_vcf (chip sites vs a reference SRP) and
// write_panel_vcf (de-novo panel phasing). The record loops differ in indexing
// and the ID field, so only the header and the phased-GT row are shared.
// ---------------------------------------------------------------------------

/// Write the GT-only VCF header: fileformat, source (with `source_suffix`
/// appended after the SelfDecode tag), FILTER=PASS, AF/AN/AC INFO, GT FORMAT,
/// an optional contig line, and the `#CHROM ... FORMAT <samples...>` column row.
fn write_gt_vcf_header<W: std::io::Write>(
    w: &mut W,
    source_suffix: &str,
    contig: Option<&str>,
    sample_names: &[String],
) -> std::io::Result<()> {
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{} SelfDecode\u{2122}{}", env!("CARGO_PKG_VERSION"), source_suffix)?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    if let Some(c) = contig { writeln!(w, "{}", c)?; }
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;
    Ok(())
}

/// Build the phased diploid GT string for one variant row (`row` = the variant's
/// `n_haps` alleles) into `line_buf` (`a|b` per sample, tab-separated), returning
/// the raw ALT allele count. When `clamp`, alleles are projected to biallelic
/// 0/1 for the emitted GT (the AC still sums the raw alleles).
fn build_phased_gt_row(line_buf: &mut String, row: &[u8], n_samples: usize, clamp: bool) -> u32 {
    line_buf.clear();
    let mut ac = 0u32;
    for s in 0..n_samples {
        let a0 = row[s * 2];
        let a1 = row[s * 2 + 1];
        ac += a0 as u32 + a1 as u32;
        if s > 0 { line_buf.push('\t'); }
        let (g0, g1) = if clamp { (a0.min(1), a1.min(1)) } else { (a0, a1) };
        line_buf.push((b'0' + g0) as char);
        line_buf.push('|');
        line_buf.push((b'0' + g1) as char);
    }
    ac
}

// ---------------------------------------------------------------------------
// write_phased_vcf
// ---------------------------------------------------------------------------

/// Write phased-only VCF (chip sites only, GT format).
pub fn write_phased_vcf(
    phased: &[u8],               // (n_chip, n_haps) row-major
    target_markers: &[TargetMarker],
    target_idx: &[usize],        // chip → target marker index
    _wgs_idx: &[usize],          // chip → WGS variant index (for pos ordering)
    sample_names: &[String],
    srp: &SrpReader,
    n_chip: usize,
    n_haps: usize,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::io::{Write, BufWriter};

    let n_samples = n_haps / 2;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(4 << 20, bgzf);

    write_gt_vcf_header(&mut w, "", Some(srp.metadata.contig_field.as_str()), sample_names)?;

    let mut line_buf = String::with_capacity(n_samples * 6);
    for ci in 0..n_chip {
        let ti = target_idx[ci];
        let tm = &target_markers[ti];
        let row = &phased[ci * n_haps..ci * n_haps + n_haps];
        let ac = build_phased_gt_row(&mut line_buf, row, n_samples, false);
        let af = ac as f64 / n_haps as f64;
        writeln!(w, "{}\t{}\t.\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}",
            tm.chrom, tm.pos, tm.ref_allele, tm.alt_allele, af, ac, n_haps, line_buf)?;
    }

    w.flush()?;
    let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;

    // Build a TBI index natively (no bcftools subprocess).
    if let Err(e) = crate::srp::csi::build_tbi_index(output_path) {
        selphi_info!("  WARN: TBI index build failed for {}: {} — VCF is still valid, just unindexed.", output_path.display(), e);
    }

    Ok(())
}

/// Write a phased PANEL VCF.gz: every cohort marker with phased GT.
/// Independent of any reference/SRP — used by the de-novo panel-phasing path.
pub fn write_panel_vcf(
    phased: &[u8],               // (n_var × n_haps) row-major
    markers: &[TargetMarker],
    sample_names: &[String],
    n_var: usize,
    n_haps: usize,
    output_path: &Path,
) -> std::io::Result<()> {
    use std::io::{Write, BufWriter};
    let n_samples = n_haps / 2;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(4 << 20, bgzf);

    let contig = markers.first().map(|m0| format!("##contig=<ID={}>", m0.chrom));
    write_gt_vcf_header(&mut w, " (panel-phasing)", contig.as_deref(), sample_names)?;

    let mut line_buf = String::with_capacity(n_samples * 4);
    for v in 0..n_var {
        let m = &markers[v];
        let row = &phased[v * n_haps..v * n_haps + n_haps];
        let ac = build_phased_gt_row(&mut line_buf, row, n_samples, true);
        let af = ac as f64 / n_haps as f64;
        let id = if m.id.is_empty() { "." } else { m.id.as_str() };
        writeln!(w, "{}\t{}\t{}\t{}\t{}\t.\tPASS\tAF={:.4};AC={};AN={}\tGT\t{}",
            m.chrom, m.pos, id, m.ref_allele, m.alt_allele, af, ac, n_haps, line_buf)?;
    }

    w.flush()?;
    let mut bgzf = w.into_inner().map_err(|e| std::io::Error::other(e.to_string()))?;
    bgzf.finish()?;
    if let Err(e) = crate::srp::csi::build_tbi_index(output_path) {
        selphi_info!("  WARN: TBI index build failed for {}: {} — VCF is still valid, just unindexed.", output_path.display(), e);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// R2 --refine: per-chip-site input confidence c[v] ∈ [0,1]
// ---------------------------------------------------------------------------

/// Default GQ→confidence ramp endpoints (env overridable SELPHI_REFINE_GQ_LO/_HI).
const REFINE_GQ_LO: f64 = 10.0;
const REFINE_GQ_HI: f64 = 30.0;
/// DP→confidence ramp endpoints (fixed; the MVP only exposes GQ tuning).
const REFINE_DP_LO: f64 = 4.0;
const REFINE_DP_HI: f64 = 12.0;

fn refine_gq_endpoints() -> (f64, f64) {
    let lo = std::env::var("SELPHI_REFINE_GQ_LO").ok()
        .and_then(|s| s.parse::<f64>().ok()).unwrap_or(REFINE_GQ_LO);
    let hi = std::env::var("SELPHI_REFINE_GQ_HI").ok()
        .and_then(|s| s.parse::<f64>().ok()).unwrap_or(REFINE_GQ_HI);
    (lo, hi)
}

#[inline]
fn ramp(x: f64, lo: f64, hi: f64) -> f64 {
    if hi <= lo { return 1.0; }
    ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Map one sample's FORMAT subfields to a confidence in [0,1].
/// Priority: GQ (and/or DP → min), else PL-derived GQ-equiv, else DP, else 1.0.
/// `gq`/`dp` are the integer values (or None if absent/missing); `pl` is the
/// parsed diploid PL triple (or None). Returns 1.0 when nothing usable is
/// present (the site is treated as a fully trusted hard call → untouched HMM).
#[inline]
fn sample_confidence(
    gq: Option<i64>, pl: Option<[i32; 3]>, dp: Option<i64>,
    gq_lo: f64, gq_hi: f64,
) -> f64 {
    // GQ-equivalent: real GQ if present, else (2nd-smallest PL − smallest PL).
    let gq_eq: Option<f64> = match gq {
        Some(g) => Some(g as f64),
        None => pl.map(|p| {
            let mut s = p;
            s.sort_unstable();
            (s[1] - s[0]) as f64
        }),
    };
    let dp_conf: Option<f64> = dp.map(|d| ramp(d as f64, REFINE_DP_LO, REFINE_DP_HI));
    match (gq_eq, dp_conf) {
        (Some(g), Some(d)) => ramp(g, gq_lo, gq_hi).min(d),
        (Some(g), None) => ramp(g, gq_lo, gq_hi),
        (None, Some(d)) => d,
        (None, None) => 1.0,
    }
}

/// Index of `name` in a colon-separated FORMAT spec (`b"GT:GQ:DP"` → 1 for GQ).
#[inline]
fn format_field_index(format_bytes: &[u8], name: &[u8]) -> Option<usize> {
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in format_bytes.iter().enumerate() {
        if b == b':' {
            if &format_bytes[start..i] == name { return Some(idx); }
            idx += 1;
            start = i + 1;
        }
    }
    if &format_bytes[start..] == name { Some(idx) } else { None }
}

/// Extract the n-th colon-separated subfield from a per-sample byte slice.
#[inline]
fn nth_subfield(field: &[u8], n: usize) -> Option<&[u8]> {
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in field.iter().enumerate() {
        if b == b':' {
            if idx == n { return Some(&field[start..i]); }
            idx += 1;
            start = i + 1;
        }
    }
    if idx == n { Some(&field[start..]) } else { None }
}

/// Parse a `.`-tolerant integer subfield (GQ/DP). Returns None for "." / empty
/// / malformed (matches the "subfield absent" case → no contribution).
#[inline]
fn parse_int_subfield(bytes: &[u8]) -> Option<i64> {
    if bytes.is_empty() || bytes == b"." { return None; }
    let v = fast_parse_i64(bytes);
    if v < 0 { None } else { Some(v) }
}

/// Parse a `pl00,pl01,pl11` triple from a FORMAT PL subfield. None if missing /
/// not exactly three comma-separated non-negative ints.
#[inline]
fn parse_pl_triple(bytes: &[u8]) -> Option<[i32; 3]> {
    if bytes.is_empty() || bytes == b"." { return None; }
    let mut out = [0i32; 3];
    let mut idx = 0usize;
    let mut start = 0usize;
    for (i, &b) in bytes.iter().enumerate() {
        if b == b',' {
            if idx >= 3 { return None; }
            let s = &bytes[start..i];
            let v = fast_parse_i64(s);
            if v < 0 { return None; }
            out[idx] = v as i32;
            idx += 1;
            start = i + 1;
        }
    }
    if idx != 2 { return None; }
    let v = fast_parse_i64(&bytes[start..]);
    if v < 0 { return None; }
    out[2] = v as i32;
    Some(out)
}

/// Compute the per-marker confidence for one VCF data line's per-sample region.
/// MVP = the MINIMUM confidence across samples (a site is "soft" if ANY sample
/// is soft). When the FORMAT carries none of GQ/PL/DP, every sample yields 1.0
/// → the marker is fully trusted (untouched HMM).
fn line_min_confidence(
    format_bytes: &[u8], gt_region: &[u8], n_samples: usize, gq_lo: f64, gq_hi: f64,
) -> f64 {
    let gq_i = format_field_index(format_bytes, b"GQ");
    let pl_i = format_field_index(format_bytes, b"PL");
    let dp_i = format_field_index(format_bytes, b"DP");
    if gq_i.is_none() && pl_i.is_none() && dp_i.is_none() {
        return 1.0;
    }
    let mut min_c = f64::INFINITY;
    let mut field_start = 0usize;
    for _ in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter().position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gq = gq_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let pl = pl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_pl_triple);
        let dp = dp_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let c = sample_confidence(gq, pl, dp, gq_lo, gq_hi);
        if c < min_c { min_c = c; }
        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
    if min_c.is_finite() { min_c } else { 1.0 }
}

/// R4 per-hap confidence: like [`line_min_confidence`] but emits one confidence
/// per sample (NOT the min across samples), appended into `out` in sample order.
/// Exactly `n_samples` values are pushed per call. When the FORMAT carries none
/// of GQ/PL/DP every sample yields 1.0 (fully trusted). Missing/short per-sample
/// fields also fall back to 1.0 via `sample_confidence`'s (None,None) arm.
fn line_per_sample_confidence(
    format_bytes: &[u8], gt_region: &[u8], n_samples: usize, gq_lo: f64, gq_hi: f64,
    out: &mut Vec<f64>,
) {
    let gq_i = format_field_index(format_bytes, b"GQ");
    let pl_i = format_field_index(format_bytes, b"PL");
    let dp_i = format_field_index(format_bytes, b"DP");
    if gq_i.is_none() && pl_i.is_none() && dp_i.is_none() {
        out.extend(std::iter::repeat(1.0f64).take(n_samples));
        return;
    }
    let mut field_start = 0usize;
    for _ in 0..n_samples {
        let field_end = gt_region[field_start..]
            .iter().position(|&b| b == b'\t')
            .map(|p| field_start + p)
            .unwrap_or(gt_region.len());
        let field = &gt_region[field_start..field_end];
        let gq = gq_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        let pl = pl_i.and_then(|i| nth_subfield(field, i)).and_then(parse_pl_triple);
        let dp = dp_i.and_then(|i| nth_subfield(field, i)).and_then(parse_int_subfield);
        out.push(sample_confidence(gq, pl, dp, gq_lo, gq_hi));
        field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
    }
}

/// FORMAT spec bytes from a VCF data line (field 9, between the 9th tab and the
/// next tab/EOL). Mirrors `split_vcf_fields` (which only captures up to field 9).
#[inline]
fn vcf_format_bytes(line: &[u8]) -> Option<&[u8]> {
    let mut tab = 0usize;
    let mut last = 0usize;
    for (i, &b) in line.iter().enumerate() {
        if b == b'\t' {
            tab += 1;
            // FORMAT is VCF column 9 = the field AFTER the 8th tab (column N starts
            // after tab N-1). `tab == 9` would return column 10 (SAMPLE1), so GQ/PL/DP
            // were never found and every site defaulted to confidence 1.0.
            if tab == 8 {
                let rest = &line[i + 1..];
                let end = rest.iter().position(|&b| b == b'\t').unwrap_or(rest.len());
                return Some(&rest[..end]);
            }
            last = i;
        }
    }
    let _ = last;
    None
}

/// `--refine`: read the target VCF/BCF a second time and compute one confidence
/// value c ∈ [0,1] per RETAINED marker, in the SAME order `read_target_vcf`
/// produced its `markers` (it uses the identical `split_vcf_fields` skip logic /
/// BCF dispatch). The result must be re-indexed into chip-site order by the
/// caller via `target_idx` (see [`align_confidence_to_chip`]). Markers with no
/// usable GQ/PL/DP yield 1.0 (untouched).
///
/// Returns one f64 per marker (file order, parallel to `read_target_vcf`'s
/// `markers`). The hard-call path (`read_target_vcf` / `parse_gt_region`) is
/// NOT touched — this is an additive, opt-in pass.
pub fn extract_target_site_confidence(path: &str) -> Vec<f64> {
    let (gq_lo, gq_hi) = refine_gq_endpoints();

    // Real (binary) BCF → noodles decoder; mirror read_target_vcf's dispatch.
    if path.ends_with(".bcf") {
        return extract_site_confidence_bcf(path, gq_lo, gq_hi);
    }
    let raw = read_vcf_raw(path);
    if raw.starts_with(b"BCF\x02\x02") {
        return extract_site_confidence_bcf(path, gq_lo, gq_hi);
    }

    let mut conf: Vec<f64> = Vec::new();
    let mut n_samples = 0usize;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let n_tabs = line.iter().filter(|&&b| b == b'\t').count();
            n_samples = n_tabs.saturating_sub(8); // 9 fixed cols → tabs 1..8 before samples
            continue;
        }
        // Identical retention rule to read_target_vcf.
        let Some(f) = split_vcf_fields(line) else { continue };
        let c = match vcf_format_bytes(line) {
            Some(fmt) => line_min_confidence(fmt, f.gt_region, n_samples, gq_lo, gq_hi),
            None => 1.0,
        };
        conf.push(c);
    }
    conf
}

/// BCF path for [`extract_target_site_confidence`]. Walks records in the same
/// order + with the same retention rule as `read_target_bcf`.
/// Decoded BCF `PL` sample value (integer array) → `[pl00, pl01, pl11]`.
/// `None` if absent / non-integer / fewer than three present elements.
fn bcf_value_to_pl3(
    v: &noodles_vcf::variant::record_buf::samples::sample::value::Value,
) -> Option<[i32; 3]> {
    use noodles_vcf::variant::record_buf::samples::sample::value::{Array, Value};
    match v {
        Value::Array(Array::Integer(vals)) if vals.len() >= 3 => Some([vals[0]?, vals[1]?, vals[2]?]),
        _ => None,
    }
}

fn extract_site_confidence_bcf(path: &str, gq_lo: f64, gq_hi: f64) -> Vec<f64> {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let mut reader = match bcf::io::reader::Builder::default().build_from_path(path) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let header = match reader.read_header() { Ok(h) => h, Err(_) => return Vec::new() };

    let int_val = |v: &Value| -> Option<i64> {
        match v {
            Value::Integer(i) => Some(*i as i64),
            _ => None,
        }
    };

    let mut conf = Vec::new();
    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        // Same retention rule as read_target_bcf.
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let keep = matches!(rec.alternate_bases().as_ref().first(), Some(a) if a != "." && !a.is_empty());
        if !keep { continue; }

        let samples = rec.samples();
        let mut min_c = f64::INFINITY;
        let mut any_field = false;
        for sample in samples.values() {
            let gq = sample.get("GQ").flatten().and_then(int_val);
            let dp = sample.get("DP").flatten().and_then(int_val);
            // PL triple (first three values) if present.
            let pl: Option<[i32; 3]> = match sample.get("PL").flatten() {
                Some(v) => bcf_value_to_pl3(v),
                _ => None,
            };
            if gq.is_some() || pl.is_some() || dp.is_some() { any_field = true; }
            let c = sample_confidence(gq, pl, dp, gq_lo, gq_hi);
            if c < min_c { min_c = c; }
        }
        conf.push(if any_field && min_c.is_finite() { min_c } else { 1.0 });
    }
    conf
}

/// R4 per-(marker, sample) confidence. Returns a flat row-major matrix
/// `[marker * n_samples + sample]` (one row per RETAINED marker, in the SAME
/// file order as [`extract_target_site_confidence`] / `read_target_vcf`) plus
/// `n_samples`. Uses the SAME GQ→PL→DP `sample_confidence` map as the min path,
/// but keeps every sample's value instead of collapsing to the min. The caller
/// re-indexes into chip-site order via [`align_confidence_to_chip_per_sample`].
/// The hard-call path is untouched (additive, opt-in under `--refine`).
pub fn extract_target_site_confidence_per_sample(path: &str) -> (Vec<f64>, usize) {
    let (gq_lo, gq_hi) = refine_gq_endpoints();

    if path.ends_with(".bcf") {
        return extract_site_confidence_per_sample_bcf(path, gq_lo, gq_hi);
    }
    let raw = read_vcf_raw(path);
    if raw.starts_with(b"BCF\x02\x02") {
        return extract_site_confidence_per_sample_bcf(path, gq_lo, gq_hi);
    }

    let mut conf: Vec<f64> = Vec::new();
    let mut n_samples = 0usize;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let n_tabs = line.iter().filter(|&&b| b == b'\t').count();
            n_samples = n_tabs.saturating_sub(8);
            continue;
        }
        let Some(f) = split_vcf_fields(line) else { continue };
        match vcf_format_bytes(line) {
            Some(fmt) => line_per_sample_confidence(fmt, f.gt_region, n_samples, gq_lo, gq_hi, &mut conf),
            None => conf.extend(std::iter::repeat(1.0f64).take(n_samples)),
        }
    }
    (conf, n_samples)
}

/// BCF path for [`extract_target_site_confidence_per_sample`]. Same record
/// order + retention rule as `extract_site_confidence_bcf`, one value per sample.
fn extract_site_confidence_per_sample_bcf(path: &str, gq_lo: f64, gq_hi: f64) -> (Vec<f64>, usize) {
    use noodles_bcf as bcf;
    use noodles_vcf::variant::record_buf::samples::sample::Value;

    let mut reader = match bcf::io::reader::Builder::default().build_from_path(path) {
        Ok(r) => r,
        Err(_) => return (Vec::new(), 0),
    };
    let header = match reader.read_header() { Ok(h) => h, Err(_) => return (Vec::new(), 0) };

    let int_val = |v: &Value| -> Option<i64> {
        match v {
            Value::Integer(i) => Some(*i as i64),
            _ => None,
        }
    };

    let mut conf = Vec::new();
    let mut n_samples = 0usize;
    for result in reader.record_bufs(&header) {
        let rec = match result { Ok(r) => r, Err(_) => continue };
        let pos = match rec.variant_start() { Some(p) => usize::from(p) as i64, None => continue };
        if pos < 1 { continue; }
        let keep = matches!(rec.alternate_bases().as_ref().first(), Some(a) if a != "." && !a.is_empty());
        if !keep { continue; }

        let samples = rec.samples();
        let mut row_n = 0usize;
        for sample in samples.values() {
            let gq = sample.get("GQ").flatten().and_then(int_val);
            let dp = sample.get("DP").flatten().and_then(int_val);
            let pl: Option<[i32; 3]> = match sample.get("PL").flatten() {
                Some(v) => bcf_value_to_pl3(v),
                _ => None,
            };
            conf.push(sample_confidence(gq, pl, dp, gq_lo, gq_hi));
            row_n += 1;
        }
        if row_n > n_samples { n_samples = row_n; }
    }
    (conf, n_samples)
}

/// Re-index a per-marker (file-order) confidence vector into post-intersection
/// chip-site order via `target_idx` (chip site i → target marker index). The
/// result is `n_chip` long, aligned to `chip_cm` / the HMM. Out-of-range or
/// missing entries default to 1.0 (untouched). Returns `None` (→ no softening
/// anywhere) when every chip site is fully confident, so the whole pipeline is
/// byte-identical to the shipped path for a confident input.
pub fn align_confidence_to_chip(
    marker_conf: &[f64], target_idx: &[usize], n_chip: usize,
) -> Option<Vec<f64>> {
    let mut out = vec![1.0f64; n_chip];
    let mut any_soft = false;
    for (ci, &ti) in target_idx.iter().enumerate().take(n_chip) {
        if ti < marker_conf.len() {
            let c = marker_conf[ti].clamp(0.0, 1.0);
            out[ci] = c;
            if c < 1.0 { any_soft = true; }
        }
    }
    if any_soft { Some(out) } else { None }
}

/// R4: re-index a per-(marker, sample) confidence matrix into post-intersection
/// chip-site order. `marker_conf` is row-major `[marker * n_samples + sample]`
/// (file order, from [`extract_target_site_confidence_per_sample`]); the result
/// is row-major `[chip_site * n_samples + sample]`, `n_chip` rows long, aligned
/// to `chip_cm` / the HMM. Out-of-range markers default to a fully-confident row
/// (all 1.0). Returns `None` (→ no per-hap softening anywhere → byte-identical
/// scalar emission) when EVERY entry is 1.0.
pub fn align_confidence_to_chip_per_sample(
    marker_conf: &[f64], n_samples: usize, target_idx: &[usize], n_chip: usize,
) -> Option<Vec<f64>> {
    if n_samples == 0 { return None; }
    let n_markers = marker_conf.len() / n_samples;
    let mut out = vec![1.0f64; n_chip * n_samples];
    let mut any_soft = false;
    for (ci, &ti) in target_idx.iter().enumerate().take(n_chip) {
        if ti < n_markers {
            let src = &marker_conf[ti * n_samples..(ti + 1) * n_samples];
            let dst = &mut out[ci * n_samples..(ci + 1) * n_samples];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                let c = s.clamp(0.0, 1.0);
                *d = c;
                if c < 1.0 { any_soft = true; }
            }
        }
    }
    if any_soft { Some(out) } else { None }
}

// ---------------------------------------------------------------------------
// extract_target_alleles
// ---------------------------------------------------------------------------

/// Extract target alleles at chip sites into flat (n_chip, n_haps) row-major array.
pub fn extract_target_alleles(
    genotypes: &[Vec<[u8; 2]>],
    target_idx: &[usize],
    n_chip: usize,
    n_haps: usize,
) -> Vec<u8> {
    let n_samples = n_haps / 2;
    let mut out = vec![0u8; n_chip * n_haps];
    for (ci, &ti) in target_idx.iter().enumerate() {
        if ti >= genotypes.len() { continue; }
        let gt = &genotypes[ti];
        for s in 0..n_samples.min(gt.len()) {
            out[ci * n_haps + s * 2] = gt[s][0];
            out[ci * n_haps + s * 2 + 1] = gt[s][1];
        }
    }
    out
}

// ---------------------------------------------------------------------------
// intersect_variants
// ---------------------------------------------------------------------------

/// Intersect target markers with reference panel variants.
pub fn intersect_variants(srp: &SrpReader, targets: &[TargetMarker]) -> (Vec<usize>, Vec<usize>) {
    fn strip_chr(c: &str) -> &str {
        if let Some(stripped) = c.strip_prefix("chr") { stripped } else { c }
    }
    let ref_chrom = strip_chr(&srp.metadata.chromosome);

    // Sort target indices by position for merge-join
    let mut tgt_order: Vec<usize> = (0..targets.len())
        .filter(|&i| strip_chr(&targets[i].chrom) == ref_chrom)
        .collect();
    tgt_order.sort_by_key(|&i| targets[i].pos);

    // Merge-join: both ref variants and sorted targets are in position order
    let mut wgs_idx = Vec::with_capacity(targets.len());
    let mut target_idx = Vec::with_capacity(targets.len());
    let mut ri = 0usize;
    let mut n_hash_matches = 0usize;
    let mut n_plain_matches = 0usize;

    for &ti in &tgt_order {
        let tpos = targets[ti].pos;
        // Advance ref pointer to first variant at or beyond target pos
        while ri < srp.variants.len() && srp.variants[ri].pos < tpos { ri += 1; }
        // Check all ref variants at this position.
        // Match ref+alt as a coherent pair: either both via hash (new SRP format)
        // or both via plain alleles (old/compat format). Mixing (e.g. ref via hash,
        // alt via plain) is rejected to avoid ambiguous cross-format matches.
        let mut rj = ri;
        while rj < srp.variants.len() && srp.variants[rj].pos == tpos {
            let hash_match = srp.variants[rj].ref_allele == targets[ti].ref_hash
                && srp.variants[rj].alt_allele == targets[ti].alt_hash;
            let plain_match = !hash_match
                && srp.variants[rj].ref_allele == targets[ti].ref_allele
                && srp.variants[rj].alt_allele == targets[ti].alt_allele;
            if hash_match || plain_match {
                wgs_idx.push(rj);
                target_idx.push(ti);
                if hash_match { n_hash_matches += 1; } else { n_plain_matches += 1; }
                break;
            }
            rj += 1;
        }
    }
    if n_hash_matches > 0 || n_plain_matches > 0 {
        selphi_info!("  Variant intersection: {} hash matches, {} plain matches",
            n_hash_matches, n_plain_matches);
    }

    // Already sorted by wgs_idx (ref is in genomic order, merge preserves it)
    (wgs_idx, target_idx)
}

// ---------------------------------------------------------------------------
// Multi-chromosome VCF reading
// ---------------------------------------------------------------------------

/// Read a multi-chromosome target VCF once and partition markers+genotypes by chromosome.
/// Returns (sample_names, per_chr_data, is_phased).
pub fn read_target_vcf_multi_chr(
    path: &str,
) -> (Vec<String>, std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)>, bool) {
    type ByChr = std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)>;
    // Real binary BCF → noodles decoder, then partition by chromosome.
    let partition = |samples: Vec<String>, markers: Vec<TargetMarker>, gts: Vec<Vec<[u8; 2]>>, phased: bool| {
        let mut by_chr: ByChr = std::collections::BTreeMap::new();
        for (m, g) in markers.into_iter().zip(gts) {
            let e = by_chr.entry(m.chrom.clone()).or_default();
            e.0.push(m);
            e.1.push(g);
        }
        (samples, by_chr, phased)
    };
    if path.ends_with(".bcf") {
        let (s, m, g, p) = read_target_bcf(path, false, false);
        return partition(s, m, g, p);
    }
    let raw = read_vcf_raw(path);
    if raw.starts_with(b"BCF\x02\x02") {
        let (s, m, g, p) = read_target_bcf(path, false, false);
        return partition(s, m, g, p);
    }

    let mut all_markers: Vec<TargetMarker> = Vec::new();
    let mut all_genotypes: Vec<Vec<[u8; 2]>> = Vec::new();
    let mut is_phased = true;
    let mut phase_checks = 10i32;
    let mut sample_names: Vec<String> = Vec::new();

    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() || line.starts_with(b"##") { continue; }
        if line.starts_with(b"#CHROM") {
            let fields: Vec<&[u8]> = line.split(|&b| b == b'\t').collect();
            if fields.len() > 9 {
                sample_names = fields[9..].iter()
                    .map(|f| std::str::from_utf8(f).unwrap_or("").to_string())
                    .collect();
            }
            continue;
        }

        let Some(f) = split_vcf_fields(line) else { continue };
        all_markers.push(TargetMarker {
            chrom: f.chrom.to_string(), pos: f.pos,
            ref_allele: f.ref_allele.to_string(), alt_allele: f.alt_allele.to_string(),
            ref_hash: f.ref_allele.to_string(), alt_hash: f.alt_allele.to_string(),
            id: String::new(),
        });
        all_genotypes.push(parse_gt_region(f.gt_region, sample_names.len(), &mut is_phased, &mut phase_checks));
    }

    // Partition by chromosome
    let mut by_chr: std::collections::BTreeMap<String, (Vec<TargetMarker>, Vec<Vec<[u8; 2]>>)> =
        std::collections::BTreeMap::new();
    for (marker, gts) in all_markers.into_iter().zip(all_genotypes.into_iter()) {
        let chr = strip_chr_prefix(&marker.chrom).to_string();
        let entry = by_chr.entry(chr).or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(marker);
        entry.1.push(gts);
    }

    (sample_names, by_chr, is_phased)
}

fn strip_chr_prefix(s: &str) -> &str {
    s.strip_prefix("chr").unwrap_or(s)
}

// ---------------------------------------------------------------------------
// Generic variant intersection (works with any variant list + chromosome)
// ---------------------------------------------------------------------------

/// Intersect target markers with a reference variant list for a given chromosome.
/// Same logic as `intersect_variants` but works with raw variant/ID slices.
pub fn intersect_variants_for_chr(
    ref_chromosome: &str,
    ref_variants: &[crate::srp::Variant],
    ref_ids: &[String],
    targets: &[TargetMarker],
) -> (Vec<usize>, Vec<usize>) {
    fn strip_chr(c: &str) -> &str {
        if let Some(stripped) = c.strip_prefix("chr") { stripped } else { c }
    }
    let ref_chrom = strip_chr(ref_chromosome);

    let hash_alleles = !ref_ids.is_empty() && {
        let first_ref = &ref_variants[0].ref_allele;
        !ref_ids[0].contains(first_ref)
    };

    // Prepare target markers with correct hash if needed
    let targets_with_hash: Vec<TargetMarker> = if hash_alleles {
        targets.iter().map(|t| TargetMarker {
            chrom: t.chrom.clone(), pos: t.pos,
            ref_allele: t.ref_allele.clone(), alt_allele: t.alt_allele.clone(),
            ref_hash: crate::srp::blake2b_hex(&t.ref_allele),
            alt_hash: crate::srp::blake2b_hex(&t.alt_allele), id: t.id.clone(),
        }).collect()
    } else {
        targets.to_vec()
    };
    let targets_ref = &targets_with_hash;

    let mut tgt_order: Vec<usize> = (0..targets_ref.len())
        .filter(|&i| strip_chr(&targets_ref[i].chrom) == ref_chrom)
        .collect();
    tgt_order.sort_by_key(|&i| targets_ref[i].pos);

    let mut wgs_idx = Vec::with_capacity(targets_ref.len());
    let mut target_idx = Vec::with_capacity(targets_ref.len());
    let mut ri = 0usize;

    for &ti in &tgt_order {
        let tpos = targets_ref[ti].pos;
        while ri < ref_variants.len() && ref_variants[ri].pos < tpos { ri += 1; }
        let mut rj = ri;
        while rj < ref_variants.len() && ref_variants[rj].pos == tpos {
            if ref_variants[rj].ref_allele == targets_ref[ti].ref_hash
                && ref_variants[rj].alt_allele == targets_ref[ti].alt_hash {
                wgs_idx.push(rj);
                target_idx.push(ti);
                break;
            }
            rj += 1;
        }
    }

    (wgs_idx, target_idx)
}
