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
#[inline]
fn fast_parse_i64(bytes: &[u8]) -> i64 {
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
    use std::io::Read;
    let is_gz = path.ends_with(".gz") || path.ends_with(".bcf");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });
    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("BGZF decompress failed for {}: {}", path, e); std::process::exit(1) });
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("Failed to read VCF {}: {}", path, e); std::process::exit(1) });
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
        let mut tabs = [0usize; 9];
        let mut n_tabs = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' {
                if n_tabs < 9 { tabs[n_tabs] = i; }
                n_tabs += 1;
                if n_tabs >= 9 { break; }
            }
        }
        if n_tabs < 9 { continue; }

        let pos: i64 = fast_parse_i64(&line[tabs[0]+1..tabs[1]]);
        if pos < 1 { continue; } // skip malformed VCF POS (sentinel)
        let ref_bytes = &line[tabs[2]+1..tabs[3]];
        let alt_field = &line[tabs[3]+1..tabs[4]];
        let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
        let alt_bytes = &alt_field[..alt_end];
        if alt_bytes == b"." || alt_bytes.is_empty() { continue; }
        if alt_end < alt_field.len() { n_multiallelic += 1; } // ALT had a comma
        let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
        let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();
        let id = std::str::from_utf8(&line[tabs[1]+1..tabs[2]]).unwrap_or(".").to_string();
        markers.push(TargetMarker {
            chrom, pos, ref_allele, alt_allele,
            ref_hash: String::new(), alt_hash: String::new(), id,
        });

        let n_samples = sample_names.len();
        let mut var_gts = Vec::with_capacity(n_samples);
        let gt_region = &line[tabs[8]+1..];
        let mut field_start = 0;
        for _s in 0..n_samples {
            let field_end = gt_region[field_start..].iter()
                .position(|&b| b == b'\t')
                .map(|p| field_start + p)
                .unwrap_or(gt_region.len());
            let field = &gt_region[field_start..field_end];
            let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
            let gt = &field[..gt_end];
            if phase_checks > 0 {
                if gt.contains(&b'/') { is_phased = false; }
                phase_checks -= 1;
            }
            // Binarise to {0,1}: REF/missing → 0, ANY alt allele (1..9) → 1.
            // Panel phasing treats the cohort as biallelic (the ALT kept is
            // the first); collapsing higher alt indices here keeps phasing
            // and the written output consistent (no silent ref/missing/alt
            // mismatch downstream). Multiallelic sites are counted + warned.
            let bin = |b: u8| -> u8 { if b.is_ascii_digit() && b != b'0' { 1 } else { 0 } };
            let (a0, a1) = if gt.len() >= 3 {
                (bin(gt[0]), bin(gt[2]))
            } else { (0, 0) };
            var_gts.push([a0, a1]);
            field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
        }
        genotypes.push(var_gts);
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

/// Read target VCF/BCF using noodles bgzf + manual text parsing.
/// Pure Rust — no bcftools dependency.
pub fn read_target_vcf(
    path: &str, srp: &SrpReader,
) -> (Vec<String>, Vec<TargetMarker>, Vec<Vec<[u8; 2]>>, bool) {
    use std::io::Read;

    let hash_alleles = !srp.ids.is_empty() && {
        let first_ref = &srp.variants[0].ref_allele;
        !srp.ids[0].contains(first_ref)
    };

    // Read entire decompressed VCF into memory (avoid per-line String alloc)
    let is_gz = path.ends_with(".gz") || path.ends_with(".bcf");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });

    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("BGZF decompress failed for {}: {}", path, e); std::process::exit(1) });
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("Failed to read VCF {}: {}", path, e); std::process::exit(1) });
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

        // Fast field splitting: find first 5 tab-separated fields + genotype region
        let mut tabs = [0usize; 9]; // positions of first 9 tabs
        let mut n_tabs = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' {
                if n_tabs < 9 { tabs[n_tabs] = i; }
                n_tabs += 1;
                if n_tabs >= 9 { break; }
            }
        }
        if n_tabs < 9 { continue; }

        // Parse POS (field 1: between tab[0] and tab[1])
        let pos_bytes = &line[tabs[0]+1..tabs[1]];
        let pos: i64 = fast_parse_i64(pos_bytes);
        if pos < 1 { continue; } // skip malformed VCF POS (sentinel)

        // REF (field 3: between tab[2] and tab[3])
        let ref_bytes = &line[tabs[2]+1..tabs[3]];
        // ALT (field 4: between tab[3] and tab[4]), take first allele before comma
        let alt_field = &line[tabs[3]+1..tabs[4]];
        let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
        let alt_bytes = &alt_field[..alt_end];
        if alt_bytes == b"." || alt_bytes.is_empty() { continue; }

        let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
        let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();

        let (ref_hash, alt_hash) = if hash_alleles {
            (crate::srp::blake2b_hex(&ref_allele), crate::srp::blake2b_hex(&alt_allele))
        } else {
            (ref_allele.clone(), alt_allele.clone())
        };

        markers.push(TargetMarker { chrom, pos, ref_allele, alt_allele, ref_hash, alt_hash, id: String::new() });

        // Parse genotypes from byte slice (fields 9+)
        let n_samples = sample_names.len();
        let mut var_gts = Vec::with_capacity(n_samples);
        let gt_region = &line[tabs[8]+1..];
        let mut field_start = 0;
        for _s in 0..n_samples {
            // Find end of this sample's field (next tab or end of line)
            let field_end = gt_region[field_start..].iter()
                .position(|&b| b == b'\t')
                .map(|p| field_start + p)
                .unwrap_or(gt_region.len());
            let field = &gt_region[field_start..field_end];

            // GT is before first ':'
            let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
            let gt = &field[..gt_end];

            if phase_checks > 0 {
                if gt.contains(&b'/') { is_phased = false; }
                phase_checks -= 1;
            }

            // Fast GT parsing: "0|1" or "0/1" — allele is single digit at positions 0 and 2
            let (a0, a1) = if gt.len() >= 3 {
                let b0 = gt[0]; let b1 = gt[2];
                (if b0.is_ascii_digit() { b0 - b'0' } else { 0 },
                 if b1.is_ascii_digit() { b1 - b'0' } else { 0 })
            } else {
                (0, 0)
            };
            var_gts.push([a0, a1]);

            field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
        }
        genotypes.push(var_gts);
    }

    if sample_names.is_empty() {
        selphi_error!("No samples found in {}", path);
        std::process::exit(1);
    }

    (sample_names, markers, genotypes, is_phased)
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

    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{} SelfDecode\u{2122}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(w, "{}", srp.metadata.contig_field)?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;

    let mut line_buf = String::with_capacity(n_samples * 6);
    for ci in 0..n_chip {
        let ti = target_idx[ci];
        let tm = &target_markers[ti];

        let mut ac = 0u32;
        line_buf.clear();
        for s in 0..n_samples {
            let a0 = phased[ci * n_haps + s * 2];
            let a1 = phased[ci * n_haps + s * 2 + 1];
            ac += a0 as u32 + a1 as u32;
            if s > 0 { line_buf.push('\t'); }
            line_buf.push((b'0' + a0) as char);
            line_buf.push('|');
            line_buf.push((b'0' + a1) as char);
        }
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

    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{} SelfDecode\u{2122} (panel-phasing)", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">")?;
    writeln!(w, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">")?;
    writeln!(w, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    if let Some(m0) = markers.first() {
        writeln!(w, "##contig=<ID={}>", m0.chrom)?;
    }
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(w, "\t{}", name)?; }
    writeln!(w)?;

    let mut line_buf = String::with_capacity(n_samples * 4);
    for v in 0..n_var {
        let m = &markers[v];
        let mut ac = 0u32;
        line_buf.clear();
        for s in 0..n_samples {
            let a0 = phased[v * n_haps + s * 2];
            let a1 = phased[v * n_haps + s * 2 + 1];
            ac += a0 as u32 + a1 as u32;
            if s > 0 { line_buf.push('\t'); }
            line_buf.push((b'0' + a0.min(1)) as char);
            line_buf.push('|');
            line_buf.push((b'0' + a1.min(1)) as char);
        }
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
    use std::io::Read;

    let is_gz = path.ends_with(".gz") || path.ends_with(".bcf");
    let file = std::fs::File::open(path)
        .unwrap_or_else(|e| { selphi_error!("Cannot open {}: {}", path, e); std::process::exit(1) });

    let mut raw = Vec::new();
    if is_gz {
        let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
        bgzf.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("BGZF decompress failed for {}: {}", path, e); std::process::exit(1) });
    } else {
        let mut reader = std::io::BufReader::new(file);
        reader.read_to_end(&mut raw).unwrap_or_else(|e| { selphi_error!("Failed to read VCF {}: {}", path, e); std::process::exit(1) });
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

        let mut tabs = [0usize; 9];
        let mut n_tabs = 0;
        for (i, &b) in line.iter().enumerate() {
            if b == b'\t' {
                if n_tabs < 9 { tabs[n_tabs] = i; }
                n_tabs += 1;
                if n_tabs >= 9 { break; }
            }
        }
        if n_tabs < 9 { continue; }

        let pos: i64 = fast_parse_i64(&line[tabs[0]+1..tabs[1]]);
        if pos < 1 { continue; } // skip malformed VCF POS (sentinel)
        let ref_bytes = &line[tabs[2]+1..tabs[3]];
        let alt_field = &line[tabs[3]+1..tabs[4]];
        let alt_end = alt_field.iter().position(|&b| b == b',').unwrap_or(alt_field.len());
        let alt_bytes = &alt_field[..alt_end];
        if alt_bytes == b"." || alt_bytes.is_empty() { continue; }

        let ref_allele = std::str::from_utf8(ref_bytes).unwrap_or("").to_string();
        let alt_allele = std::str::from_utf8(alt_bytes).unwrap_or("").to_string();
        let chrom = std::str::from_utf8(&line[..tabs[0]]).unwrap_or("").to_string();

        all_markers.push(TargetMarker {
            chrom, pos,
            ref_allele: ref_allele.clone(), alt_allele: alt_allele.clone(),
            ref_hash: ref_allele, alt_hash: alt_allele, id: String::new(),
        });

        let n_samples = sample_names.len();
        let mut var_gts = Vec::with_capacity(n_samples);
        let gt_region = &line[tabs[8]+1..];
        let mut field_start = 0;
        for _s in 0..n_samples {
            let field_end = gt_region[field_start..].iter()
                .position(|&b| b == b'\t')
                .map(|p| field_start + p)
                .unwrap_or(gt_region.len());
            let field = &gt_region[field_start..field_end];
            let gt_end = field.iter().position(|&b| b == b':').unwrap_or(field.len());
            let gt = &field[..gt_end];

            if phase_checks > 0 {
                if gt.contains(&b'/') { is_phased = false; }
                phase_checks -= 1;
            }

            let (a0, a1) = if gt.len() >= 3 {
                let b0 = gt[0]; let b1 = gt[2];
                (if b0.is_ascii_digit() { b0 - b'0' } else { 0 },
                 if b1.is_ascii_digit() { b1 - b'0' } else { 0 })
            } else {
                (0, 0)
            };
            var_gts.push([a0, a1]);
            field_start = if field_end < gt_region.len() { field_end + 1 } else { gt_region.len() };
        }
        all_genotypes.push(var_gts);
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
