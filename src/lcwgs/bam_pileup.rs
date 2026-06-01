//! Native genotype-likelihood pileup from BAM, feeding the lcWGS engine.
//!
//! `noodles-bam` decodes the BGZF blocks + binary records (battle-tested I/O,
//! same crate family as the SRP/BCF/VCF paths); the CIGAR-aware pileup and the
//! per-site genotype-likelihood (GL) computation — the hot loop and the only
//! part with algorithmic value — are native and optimised here, matching the
//! standard `bcftools mpileup`/GLIMPSE2 base-quality model.
//!
//! For each biallelic SNP site of the reference panel, for each sample (one BAM
//! per sample), it walks the overlapping reads (handling the CIGAR M/=/X/I/D/N/S
//! ops to map each panel position to a read base + base quality), accumulates
//! the per-genotype log-likelihood, and emits the normalised 3-way GL in the
//! exact layout the Gibbs loop consumes (`gl3[v*n_samples*3 + 3*s + g]`). Sites
//! with no covering reads collapse to the flat `[1/3,1/3,1/3]` (the dominant
//! case at 1×), so the panel/LD carries them. Indel panel sites are scored by
//! local read-vs-haplotype realignment (a pair-HMM; see [`super::indel_realign`])
//! when a reference FASTA is supplied; otherwise they too stay flat.
//!
//! Model (per read base `b`, phred base-quality `q`, error `ε = 10^(-q/10)`):
//! `P(b|REF)=1-ε if b==ref else ε/3`, `P(b|ALT)` likewise, `P(b|het)=½P(b|REF)+½P(b|ALT)`.
//! `gl3` is `P(reads|genotype)` normalised to sum 1, identical to the PL path.

use std::io;
use rayon::prelude::*;
use noodles_bam as bam;
use noodles_cram as cram;
use noodles_fasta as fasta;
use noodles_core::{Position, Region};
use noodles_sam::alignment::record::cigar::op::Kind;
use noodles_sam::alignment::RecordBuf;
use noodles_sam::Header;
use super::indel_realign::{IndelModel, IndelScratch};

/// Per-sample GL pileup result, ready for the Gibbs engine.
pub struct BamGl {
    /// `gl3[v * n_samples * 3 + 3*s + g]` = normalised `P(reads | genotype g)`.
    pub gl3: Vec<f32>,
    /// Sample IDs (from each BAM's `@RG SM`, else the file stem).
    pub sample_ids: Vec<String>,
}

/// Read-filtering thresholds (defaults match GLIMPSE2 / bcftools mpileup).
#[derive(Clone, Copy)]
pub struct PileupParams {
    pub min_mapq: u8,
    pub min_bq: u8,
    pub max_depth: u32,
}
impl Default for PileupParams {
    fn default() -> Self {
        fn envu(k: &str, d: u32) -> u32 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }
        Self {
            min_mapq: envu("LCWGS_MIN_MAPQ", 20) as u8,
            min_bq: envu("LCWGS_MIN_BQ", 20) as u8,
            max_depth: envu("LCWGS_MAX_DEPTH", 250),
        }
    }
}

// SAM flag bits to exclude (unmapped, secondary, qc-fail, duplicate, supplementary).
const FLAG_EXCLUDE: u16 = 0x4 | 0x100 | 0x200 | 0x400 | 0x800;

/// Resolve a panel contig name against a BAM/CRAM header, tolerant to the `chr`
/// prefix (reference panels often use `1`/`22` while alignments use `chr1`, or
/// vice-versa). Returns the reference id AND the contig name exactly as it
/// appears in the file (needed for the indexed-query `Region`).
fn resolve_contig(header: &Header, chrom: &str) -> Option<(usize, Vec<u8>)> {
    let refseqs = header.reference_sequences();
    let lookup = |n: &[u8]| refseqs.get_index_of(n).map(|r| (r, n.to_vec()));
    if let Some(x) = lookup(chrom.as_bytes()) { return Some(x); }
    match chrom.strip_prefix("chr") {
        Some(stripped) => lookup(stripped.as_bytes()),
        None => lookup(format!("chr{chrom}").as_bytes()),
    }
}

/// Phred→error lookup tables (q ∈ 0..=93). Precomputed once: log10(1-ε),
/// log10(ε/3), and the linear (1-ε), (ε/3) for the het term — avoids powf/log10
/// in the per-base hot loop. Shared with the indel pair-HMM emission model.
pub(crate) struct PhredLut {
    pub(crate) lmatch: [f64; 94],
    pub(crate) lmis: [f64; 94],
    pub(crate) pmatch: [f64; 94],
    pub(crate) pmis: [f64; 94],
}
impl PhredLut {
    pub(crate) fn new() -> Self {
        let mut t = PhredLut { lmatch: [0.0; 94], lmis: [0.0; 94], pmatch: [0.0; 94], pmis: [0.0; 94] };
        for q in 0..94 {
            let eps = 10f64.powf(-(q as f64) / 10.0);
            let pm = 1.0 - eps;
            let pmis = eps / 3.0;
            t.pmatch[q] = pm; t.pmis[q] = pmis;
            t.lmatch[q] = pm.log10();
            t.lmis[q] = pmis.max(1e-300).log10();
        }
        t
    }
}

/// Compute per-sample 3-way GLs at the given panel SNP sites by piling up the
/// BAM(s). `chrom` is the contig the sites live on; `sites` are
/// `(pos_1based, ref_base, alt_base)` ascending by pos, with `is_snp` marking
/// biallelic SNPs (non-SNP sites are left flat). Returns `gl3` over ALL sites
/// (flat where no reads / not a SNP) in the given order, plus sample IDs.
#[allow(clippy::too_many_arguments)]
pub fn pileup_bams(
    bam_paths: &[String],
    chrom: &str,
    pos: &[i64],
    ref_base: &[u8],
    alt_base: &[u8],
    is_snp: &[bool],
    region: Option<(i64, i64)>,
    reference: Option<&str>,
    indel_model: Option<&IndelModel>,
    params: PileupParams,
) -> io::Result<BamGl> {
    let n_var = pos.len();
    let n_samples = bam_paths.len();
    assert_eq!(ref_base.len(), n_var);
    let lut = PhredLut::new();

    // Per-sample pileup in parallel (BAM/CRAM are independent). Each produces its
    // own [n_var*3] normalised GL block; assembled into the interleaved gl3 after.
    let per_sample: Vec<io::Result<(String, Vec<f32>)>> = bam_paths
        .par_iter()
        .map(|path| pileup_one(path, chrom, pos, ref_base, alt_base, is_snp, region, reference, indel_model, &params, &lut))
        .collect();

    let mut sample_ids = Vec::with_capacity(n_samples);
    let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
    for r in per_sample {
        let (id, blk) = r?;
        sample_ids.push(id);
        blocks.push(blk);
    }

    // Assemble interleaved gl3[v*ns*3 + 3s + g].
    let mut gl3 = vec![0.0f32; n_var * n_samples * 3];
    for (s, blk) in blocks.iter().enumerate() {
        for v in 0..n_var {
            let dst = v * n_samples * 3 + 3 * s;
            gl3[dst] = blk[v * 3];
            gl3[dst + 1] = blk[v * 3 + 1];
            gl3[dst + 2] = blk[v * 3 + 2];
        }
    }
    Ok(BamGl { gl3, sample_ids })
}

/// Pileup a single BAM/CRAM → per-site normalised 3-way GL (`[n_var*3]`).
/// `.cram` is dispatched to the CRAM path (which needs `reference`).
#[allow(clippy::too_many_arguments)]
fn pileup_one(
    path: &str,
    chrom: &str,
    pos: &[i64],
    ref_base: &[u8],
    alt_base: &[u8],
    is_snp: &[bool],
    region: Option<(i64, i64)>,
    reference: Option<&str>,
    indel_model: Option<&IndelModel>,
    params: &PileupParams,
    lut: &PhredLut,
) -> io::Result<(String, Vec<f32>)> {
    if path.to_ascii_lowercase().ends_with(".cram") {
        return pileup_one_cram(path, chrom, pos, ref_base, alt_base, is_snp, region, reference, indel_model, params, lut);
    }
    let n_var = pos.len();
    let mut reader = bam::io::reader::Builder::default().build_from_path(path)?;
    let header = reader.read_header()?;

    // Sample id: first @RG SM, else file stem.
    let sample_id = header
        .read_groups()
        .values()
        .find_map(|rg| rg.other_fields().get(b"SM").map(|v| String::from_utf8_lossy(v).into_owned()))
        .unwrap_or_else(|| {
            std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string())
        });

    // Map chrom name → reference id in this BAM (chr-prefix tolerant).
    let resolved = resolve_contig(&header, chrom);

    // Per-genotype log10-likelihood accumulators + read-depth (for cap).
    let mut ll = vec![[0.0f64; 3]; n_var];
    let mut depth = vec![0u32; n_var];
    let last_pos = pos[n_var - 1];

    if let Some((target_rid, contig_name)) = resolved {
        // Shared per-record pileup: CIGAR-walk + GL accumulation. Used by both the
        // whole-file streaming path and the indexed region-query path.
        let mut cigar_buf: Vec<(Kind, usize)> = Vec::with_capacity(16);
        let mut iscratch = IndelScratch::default();
        let mut process = |record: &bam::Record| {
            match record.reference_sequence_id() {
                Some(Ok(rid)) if rid == target_rid => {}
                _ => return,
            }
            if record.flags().bits() & FLAG_EXCLUDE != 0 { return; }
            let mapq = record.mapping_quality().map(|m| m.get()).unwrap_or(0);
            if mapq < params.min_mapq { return; }
            let start = match record.alignment_start() {
                Some(Ok(p)) => usize::from(p) as i64, // 1-based
                _ => return,
            };
            let seq = record.sequence();
            let qbytes = record.quality_scores().as_bytes();
            if qbytes.is_empty() { return; } // no base qualities → can't score
            cigar_buf.clear();
            for op in record.cigar().iter() {
                match op { Ok(o) => cigar_buf.push((o.kind(), o.len())), Err(_) => return }
            }
            walk_record(
                start, last_pos, &cigar_buf,
                |qi| seq.get(qi).unwrap_or(b'N'),
                |qi| qbytes[qi],
                pos, ref_base, alt_base, is_snp, params, lut, &mut ll, &mut depth,
            );
            if let Some(model) = indel_model {
                super::indel_realign::score_read(
                    start, &cigar_buf,
                    |qi| seq.get(qi).unwrap_or(b'N'),
                    |qi| qbytes[qi],
                    model, lut, &mut iscratch, &mut ll, &mut depth,
                    params.max_depth, params.min_bq,
                );
            }
        };

        // Region mode with a .bai index → fetch only the region's reads (avoids
        // reading the whole file). Otherwise stream all records (zero-alloc reuse).
        let bai_path = format!("{path}.bai");
        let region_query = region.filter(|_| std::path::Path::new(&bai_path).exists());
        if let Some((rs, re)) = region_query {
            let index = bam::bai::fs::read(&bai_path)?;
            let start = Position::try_from(rs.max(1) as usize)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            let end = Position::try_from(re.max(1) as usize)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            let reg = Region::new(contig_name.clone(), start..=end);
            let mut q = reader.query(&header, &index, &reg)?;
            let mut record = bam::Record::default();
            while q.read_record(&mut record)? != 0 {
                process(&record);
            }
        } else {
            let mut record = bam::Record::default();
            while reader.read_record(&mut record)? != 0 {
                process(&record);
            }
        }
    }

    Ok((sample_id, finalize_gl(&ll, &depth)))
}

/// Convert per-site log10-likelihoods → normalised 3-way GL. Sites with no
/// covering reads (`depth==0`) collapse to the flat `[1/3,1/3,1/3]`.
fn finalize_gl(ll: &[[f64; 3]], depth: &[u32]) -> Vec<f32> {
    let n_var = ll.len();
    let mut out = vec![1.0f32 / 3.0; n_var * 3];
    for v in 0..n_var {
        if depth[v] == 0 { continue; }
        let m = ll[v][0].max(ll[v][1]).max(ll[v][2]);
        let l0 = 10f64.powf(ll[v][0] - m);
        let l1 = 10f64.powf(ll[v][1] - m);
        let l2 = 10f64.powf(ll[v][2] - m);
        let s = l0 + l1 + l2;
        if s > 0.0 {
            out[v * 3] = (l0 / s) as f32;
            out[v * 3 + 1] = (l1 / s) as f32;
            out[v * 3 + 2] = (l2 / s) as f32;
        }
    }
    out
}

/// Pileup a single CRAM → per-site normalised 3-way GL (`[n_var*3]`).
///
/// CRAM stores read bases as differences from a reference, so a FASTA reference
/// (with a `.fai`) is required to decode them — passed via `--reference`. Each
/// sample opens its own reference repository (avoids contending on a shared
/// FASTA reader across the rayon sample threads). The per-read CIGAR-walk + GL
/// model is the SAME as the BAM path (`walk_record`); the only difference is the
/// record source. With a `.crai` index and a region, only the region is decoded.
#[allow(clippy::too_many_arguments)]
fn pileup_one_cram(
    path: &str,
    chrom: &str,
    pos: &[i64],
    ref_base: &[u8],
    alt_base: &[u8],
    is_snp: &[bool],
    region: Option<(i64, i64)>,
    reference: Option<&str>,
    indel_model: Option<&IndelModel>,
    params: &PileupParams,
    lut: &PhredLut,
) -> io::Result<(String, Vec<f32>)> {
    let n_var = pos.len();
    let ref_path = reference.ok_or_else(|| io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("CRAM input ({path}) requires --reference <fasta> (with .fai) to decode read bases"),
    ))?;
    let repo = {
        let ir = fasta::io::indexed_reader::Builder::default().build_from_path(ref_path)?;
        fasta::Repository::new(fasta::repository::adapters::IndexedReader::new(ir))
    };
    let mut reader = cram::io::reader::Builder::default()
        .set_reference_sequence_repository(repo)
        .build_from_path(path)?;
    let header = reader.read_header()?;

    // Sample id: first @RG SM, else file stem.
    let sample_id = header
        .read_groups()
        .values()
        .find_map(|rg| rg.other_fields().get(b"SM").map(|v| String::from_utf8_lossy(v).into_owned()))
        .unwrap_or_else(|| {
            std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string())
        });

    let mut ll = vec![[0.0f64; 3]; n_var];
    let mut depth = vec![0u32; n_var];
    let (target_rid, contig_name) = match resolve_contig(&header, chrom) {
        Some(x) => x,
        None => return Ok((sample_id, finalize_gl(&ll, &depth))), // contig absent → all flat
    };
    let last_pos = pos[n_var - 1];
    let mut cigar_buf: Vec<(Kind, usize)> = Vec::with_capacity(16);
    let mut iscratch = IndelScratch::default();

    // Same guards + walk as the BAM path, over decoded RecordBufs.
    let mut process = |record: &RecordBuf| {
        if record.reference_sequence_id() != Some(target_rid) { return; }
        if record.flags().bits() & FLAG_EXCLUDE != 0 { return; }
        let mapq = record.mapping_quality().map(|m| m.get()).unwrap_or(0);
        if mapq < params.min_mapq { return; }
        let start = match record.alignment_start() { Some(p) => usize::from(p) as i64, None => return };
        let seq = record.sequence();
        let qbytes = record.quality_scores().as_ref();
        if qbytes.is_empty() { return; }
        cigar_buf.clear();
        for op in record.cigar().as_ref() { cigar_buf.push((op.kind(), op.len())); }
        walk_record(
            start, last_pos, &cigar_buf,
            |qi| seq.get(qi).unwrap_or(b'N'),
            |qi| qbytes[qi],
            pos, ref_base, alt_base, is_snp, params, lut, &mut ll, &mut depth,
        );
        if let Some(model) = indel_model {
            super::indel_realign::score_read(
                start, &cigar_buf,
                |qi| seq.get(qi).unwrap_or(b'N'),
                |qi| qbytes[qi],
                model, lut, &mut iscratch, &mut ll, &mut depth,
                params.max_depth, params.min_bq,
            );
        }
    };

    // Region mode with a .crai index → decode only the region. Else stream all.
    let crai_path = format!("{path}.crai");
    let region_query = region.filter(|_| std::path::Path::new(&crai_path).exists());
    if let Some((rs, re)) = region_query {
        // noodles' CRAM Query decodes EVERY container of the reference sequence
        // (it filters reads by interval only after decoding, no container skip /
        // early-stop). Pre-filter the crai to the containers whose ref span
        // overlaps [rs,re] so only those are seeked + decoded — O(region), not
        // O(chromosome). The per-read interval filter inside Query still applies.
        let index: cram::crai::Index = cram::crai::fs::read(&crai_path)?
            .into_iter()
            .filter(|r| {
                r.reference_sequence_id() == Some(target_rid) && {
                    let astart = r.alignment_start().map(|p| usize::from(p) as i64).unwrap_or(0);
                    let aend = astart + r.alignment_span() as i64; // exclusive max read end
                    astart <= re && aend >= rs
                }
            })
            .collect();
        let start = Position::try_from(rs.max(1) as usize)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let end = Position::try_from(re.max(1) as usize)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let reg = Region::new(contig_name.clone(), start..=end);
        for result in reader.query(&header, &index, &reg)? {
            process(&result?);
        }
    } else {
        for result in reader.records(&header) {
            process(&result?);
        }
    }

    Ok((sample_id, finalize_gl(&ll, &depth)))
}

/// Shared CIGAR-walk + GL accumulation for one read, used by BOTH the BAM
/// (concrete) and CRAM (trait) paths. `cigar` is the read's ops as (kind, len);
/// `base_at(qi)`/`qual_at(qi)` fetch the read base (ASCII) / phred quality at a
/// query index. Maps each covered panel site to its read base+qual and folds it
/// into the per-genotype log-likelihoods. Format-agnostic (the only BAM/CRAM
/// difference — how a record exposes cigar/seq/qual — is supplied by the caller).
#[allow(clippy::too_many_arguments)]
fn walk_record<B: Fn(usize) -> u8, Q: Fn(usize) -> u8>(
    start: i64,
    last_pos: i64,
    cigar: &[(Kind, usize)],
    base_at: B,
    qual_at: Q,
    pos: &[i64],
    ref_base: &[u8],
    alt_base: &[u8],
    is_snp: &[bool],
    params: &PileupParams,
    lut: &PhredLut,
    ll: &mut [[f64; 3]],
    depth: &mut [u32],
) {
    let n_var = pos.len();
    // Reference span (for the early-out): sum of ref-consuming ops.
    let ref_span: i64 = cigar.iter().map(|&(k, l)| match k {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch | Kind::Deletion | Kind::Skip => l as i64,
        _ => 0,
    }).sum();
    let mut si = pos.partition_point(|&p| p < start);
    if si >= n_var || pos[si] > start + ref_span { return; }

    let mut refcur = start;     // 1-based ref cursor
    let mut qcur: usize = 0;    // 0-based query cursor
    for &(kind, len) in cigar {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                let ref_end = refcur + len as i64; // exclusive
                while si < n_var && pos[si] < refcur { si += 1; }
                while si < n_var && pos[si] < ref_end {
                    let v = si;
                    if is_snp[v] && depth[v] < params.max_depth {
                        let qi = qcur + (pos[v] - refcur) as usize;
                        let b = base_at(qi);
                        let q = qual_at(qi);
                        if b != b'N' && q >= params.min_bq {
                            let q = (q as usize).min(93);
                            accumulate(&mut ll[v], b, ref_base[v], alt_base[v], q, lut);
                            depth[v] += 1;
                        }
                    }
                    si += 1;
                }
                refcur = ref_end;
                qcur += len;
            }
            Kind::Deletion | Kind::Skip => { refcur += len as i64; }
            Kind::Insertion | Kind::SoftClip => { qcur += len; }
            Kind::HardClip | Kind::Pad => {}
        }
        if refcur > last_pos { break; }
    }
}

/// Accumulate one read base into the 3 genotype log10-likelihoods.
#[inline(always)]
fn accumulate(ll: &mut [f64; 3], b: u8, rb: u8, ab: u8, q: usize, lut: &PhredLut) {
    // hom-REF: P(b|ref); hom-ALT: P(b|alt); het: ½P(b|ref)+½P(b|alt).
    let (lref, pref) = if b == rb { (lut.lmatch[q], lut.pmatch[q]) } else { (lut.lmis[q], lut.pmis[q]) };
    let (lalt, palt) = if b == ab { (lut.lmatch[q], lut.pmatch[q]) } else { (lut.lmis[q], lut.pmis[q]) };
    ll[0] += lref;
    ll[2] += lalt;
    ll[1] += (0.5 * pref + 0.5 * palt).log10();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phred_lut_basic() {
        let lut = PhredLut::new();
        // q=20 → ε=0.01 → 1-ε=0.99, ε/3≈0.00333
        assert!((lut.pmatch[20] - 0.99).abs() < 1e-9);
        assert!((lut.pmis[20] - 0.01 / 3.0).abs() < 1e-9);
        assert!((lut.lmatch[20] - 0.99f64.log10()).abs() < 1e-9);
    }

    #[test]
    fn accumulate_homref_vs_het() {
        let lut = PhredLut::new();
        // ref=A, alt=G. One high-quality A read → favors hom-ref (g0) over het/hom-alt.
        let mut ll = [0.0f64; 3];
        accumulate(&mut ll, b'A', b'A', b'G', 30, &lut);
        assert!(ll[0] > ll[1] && ll[1] > ll[2], "A read: g0>g1>g2, got {:?}", ll);
        // One G read (alt) → favors hom-alt.
        let mut ll2 = [0.0f64; 3];
        accumulate(&mut ll2, b'G', b'A', b'G', 30, &lut);
        assert!(ll2[2] > ll2[1] && ll2[1] > ll2[0], "G read: g2>g1>g0, got {:?}", ll2);
        // One A + one G (het evidence) → het most likely.
        let mut llh = [0.0f64; 3];
        accumulate(&mut llh, b'A', b'A', b'G', 30, &lut);
        accumulate(&mut llh, b'G', b'A', b'G', 30, &lut);
        assert!(llh[1] > llh[0] && llh[1] > llh[2], "A+G: het wins, got {:?}", llh);
    }
}
