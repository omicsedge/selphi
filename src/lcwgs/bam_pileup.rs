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
//! Base Alignment Quality (BAQ, [`super::baq`]) is applied when a reference
//! FASTA is given, exactly as `bcftools mpileup` does by default: a first pass
//! over the reads gathers per-column indel/soft-clip statistics, a second pass
//! realigns the reads bcftools' partial heuristic selects and deposits their
//! BAQ-capped base qualities. Without `--reference` (or under `LCWGS_NO_BAQ`)
//! the single-pass raw-quality pileup runs unchanged.
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
use super::baq::{self, BaqScratch, ColumnStats};

/// Per-sample GL pileup result, ready for the Gibbs engine.
pub struct BamGl {
    /// `gl3[v * n_samples * 3 + 3*s + g]` = normalised `P(reads | genotype g)`.
    pub gl3: Vec<f32>,
    /// Sample IDs (from each BAM's `@RG SM`, else the file stem).
    pub sample_ids: Vec<String>,
}

/// Read-filtering thresholds. These are OURS and are deliberately stricter than
/// either reference — the earlier claim that they match GLIMPSE2 or bcftools
/// mpileup was wrong on every threshold. GLIMPSE2 defaults to mapq 10 and baseq 10
/// (`GLIMPSE2/phase/src/caller/caller_parameters.cpp:81-82`); bcftools mpileup's
/// documented defaults are looser still. We use 20/20 and a 250-read depth cap.
/// The paper's low-coverage results were all measured at these values, so treat a
/// change as a re-measurement, not a tweak (`LCWGS_MIN_MAPQ`, `LCWGS_MIN_BQ`,
/// `LCWGS_MAX_DEPTH`).
#[derive(Clone, Copy)]
pub struct PileupParams {
    pub min_mapq: u8,
    pub min_bq: u8,
    pub max_depth: u32,
    /// Keep ANOMALOUS read pairs (paired but not flagged proper-pair, 0x2).
    /// Default `false` = discard them, matching bcftools/samtools mpileup's default
    /// (`--count-orphans` off): such reads — typically soft-clipped near structural
    /// breakpoints — carry spurious alt alleles that manufacture false hets and
    /// degrade the ultra-rare bin. `LCWGS_COUNT_ORPHANS` flips it on (samtools `-A`).
    pub count_orphans: bool,
    /// SNP genotype-likelihood model. `false` (default) = faithful samtools/bcftools
    /// revised-MAQ errmod (correlated-read dependency cap + mapQ/neighbour baseQ caps).
    /// `true` (`LCWGS_NAIVE_GL`) = prior naive independent-product model (A/B only).
    pub naive_gl: bool,
    /// Apply BAQ (needs a reference FASTA). Default on, matching bcftools mpileup;
    /// `LCWGS_NO_BAQ` turns it off (bcftools `-B`).
    pub baq: bool,
    /// Realign EVERY read rather than only those bcftools' partial heuristic
    /// selects (bcftools `-D`, `--full-BAQ`). `LCWGS_FULL_BAQ`. Measured slightly
    /// worse than partial (−0.05 pp non-ref concordance), kept for A/B.
    pub full_baq: bool,
    /// Reproduce bcftools' streaming artefact exactly: a realigned read keeps its
    /// RAW qualities at the pileup columns before the one that triggered its
    /// realignment (bcftools has already consumed those columns when it
    /// realigns). Default off = BAQ qualities at every column of a realigned
    /// read. `LCWGS_BAQ_STREAMING` (A/B).
    pub baq_streaming: bool,
    /// Keep supplementary alignments (flag 0x800). Default off = drop them, which
    /// is a deviation from BOTH references, not a match to either: bcftools
    /// mpileup's `--ff` default excludes only UNMAP,SECONDARY,QCFAIL,DUP, and
    /// GLIMPSE2 excludes only UNMAP and SECONDARY — its supplementary exclusion is
    /// commented out in the source (`caller_initialise.cpp:196-199`), so it keeps
    /// supplementary, QC-fail and duplicate reads. Dropping duplicates and QC-fail
    /// for genotype-likelihood calling is standard and we keep doing it; the
    /// supplementary half is UNMEASURED. `LCWGS_KEEP_SUPPLEMENTARY` (A/B).
    pub keep_supplementary: bool,
}
impl Default for PileupParams {
    fn default() -> Self {
        use crate::config::u32_or as envu;
        Self {
            min_mapq: envu("LCWGS_MIN_MAPQ", 20) as u8,
            min_bq: envu("LCWGS_MIN_BQ", 20) as u8,
            max_depth: envu("LCWGS_MAX_DEPTH", 250),
            count_orphans: crate::config::present("LCWGS_COUNT_ORPHANS"),
            naive_gl: crate::config::present("LCWGS_NAIVE_GL"),
            baq: !crate::config::present("LCWGS_NO_BAQ"),
            full_baq: crate::config::present("LCWGS_FULL_BAQ"),
            baq_streaming: crate::config::present("LCWGS_BAQ_STREAMING"),
            keep_supplementary: crate::config::present("LCWGS_KEEP_SUPPLEMENTARY"),
        }
    }
}

/// ASCII nucleotide → 2-bit base (A=0,C=1,G=2,T=3; anything else=4), matching
/// htslib `seq_nt16_int`. Used to pack pileup bases for the errmod model.
#[inline]
fn ascii_to_base4(b: u8) -> usize {
    baq::nt4(b) as usize
}

/// Discard a read if it is part of an ANOMALOUS pair: paired (flag 0x1) but NOT
/// flagged proper-pair (0x2). Matches bcftools/samtools mpileup's default read
/// filtering (opt out via `LCWGS_COUNT_ORPHANS` → `count_orphans`). Single-end
/// reads (no 0x1) are always kept. `flags` is the raw SAM flag word.
#[inline]
fn is_anomalous_pair(flags: u16, count_orphans: bool) -> bool {
    !count_orphans && (flags & 0x1) != 0 && (flags & 0x2) == 0
}

// SAM flag bits to exclude (unmapped, secondary, qc-fail, duplicate, supplementary).
// Stricter than GLIMPSE2, which excludes only UNMAP|SECONDARY
// (`GLIMPSE2/phase/src/caller/caller_initialise.cpp:196`).
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

/// Sample id for a pileup output column: first `@RG SM` in the header, else the
/// alignment file's stem. Shared by the BAM and CRAM paths.
fn read_group_sample_id(header: &Header, path: &str) -> String {
    header
        .read_groups()
        .values()
        .find_map(|rg| rg.other_fields().get(b"SM").map(|v| String::from_utf8_lossy(v).into_owned()))
        .unwrap_or_else(|| {
            std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string())
        })
}

/// 1-based inclusive `[rs, re]` → a noodles `Region` on `name`, for an indexed
/// BAM/CRAM query. Shared by the BAM and CRAM region paths.
fn build_region(name: Vec<u8>, rs: i64, re: i64) -> io::Result<Region> {
    let start = Position::try_from(rs.max(1) as usize)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let end = Position::try_from(re.max(1) as usize)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    Ok(Region::new(name, start..=end))
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

/// The reference bases BAQ realigns against: `seq[0]` sits at 0-based genome
/// position `off`. Loaded once per contig (or region window) and shared by all
/// samples.
pub struct RefWindow {
    pub seq: Vec<u8>,
    pub off: i64,
}

/// Load the reference bases for `chrom` from an indexed FASTA — the whole contig,
/// or `[rs - margin, re + margin]` when a region is given (reads overlapping the
/// region, ≤ 500 bp for BAQ, never reach past the margin). Contig name resolution
/// is `chr`-prefix tolerant, as everywhere else in the pipeline.
pub fn load_reference_window(reference: &str, chrom: &str, region: Option<(i64, i64)>) -> io::Result<RefWindow> {
    let mut ir = fasta::io::indexed_reader::Builder::default().build_from_path(reference)?;
    let recs: Vec<(Vec<u8>, usize)> = ir.index().as_ref().iter()
        .map(|r| (r.name().to_vec(), r.length() as usize)).collect();
    let find = |n: &[u8]| recs.iter().find(|(x, _)| x.as_slice() == n).cloned();
    let hit = find(chrom.as_bytes())
        .or_else(|| chrom.strip_prefix("chr").and_then(|s| find(s.as_bytes())))
        .or_else(|| if chrom.starts_with("chr") { None } else { find(format!("chr{chrom}").as_bytes()) });
    let (name, len) = hit.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput,
        format!("reference FASTA has no contig matching '{chrom}' (tried with/without 'chr')")))?;
    const MARGIN: i64 = 2000;
    let (lo, hi) = match region {
        Some((rs, re)) => ((rs - MARGIN).max(1), (re + MARGIN).min(len as i64).max(1)),
        None => (1, len as i64),
    };
    let inval = |e: std::num::TryFromIntError| io::Error::new(io::ErrorKind::InvalidInput, e);
    let reg = Region::new(name, Position::try_from(lo as usize).map_err(inval)?..=Position::try_from(hi as usize).map_err(inval)?);
    let rec = ir.query(&reg)?;
    Ok(RefWindow { seq: rec.sequence().as_ref().to_vec(), off: lo - 1 })
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
    // Faithful samtools/bcftools SNP GL model (errmod). Built ONCE (shares its
    // ~33 MB tables across all samples). Skipped under LCWGS_NAIVE_GL or on the
    // indel-realign path (which uses the naive product). Default path.
    let em = if params.naive_gl || indel_model.is_some() {
        None
    } else {
        Some(crate::lcwgs::errmod::ErrMod::new())
    };

    // BAQ needs the reference. Loaded once, shared read-only across samples.
    let ref_win: Option<RefWindow> = if params.baq {
        match reference {
            Some(p) => {
                let w = load_reference_window(p, chrom, region)?;
                crate::selphi_info!("  lcWGS-BAM: BAQ on (extended, {} heuristic — bcftools mpileup default); reference window {} bp",
                    if params.full_baq { "full-realignment" } else { "partial-realignment" }, w.seq.len());
                Some(w)
            }
            None => {
                crate::selphi_info!("  lcWGS-BAM: BAQ off — no --reference given (bcftools mpileup applies BAQ by default; pass --reference <fasta> to enable)");
                None
            }
        }
    } else {
        crate::selphi_info!("  lcWGS-BAM: BAQ off (LCWGS_NO_BAQ)");
        None
    };
    let sites = SiteCtx { pos, ref_base, alt_base, is_snp };
    let baq_ctx = ref_win.as_ref().map(|w| BaqCtx { ref_seq: &w.seq, ref_off: w.off, partial: !params.full_baq });

    // Per-sample pileup in parallel (BAM/CRAM are independent). Each produces its
    // own [n_var*3] normalised GL block; assembled into the interleaved gl3 after.
    let per_sample: Vec<io::Result<(String, Vec<f32>, PileupReport)>> = bam_paths
        .par_iter()
        .map(|path| pileup_one(path, chrom, &sites, region, reference, indel_model, &params, &lut, em.as_ref(), baq_ctx.as_ref()))
        .collect();

    let mut sample_ids = Vec::with_capacity(n_samples);
    let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(n_samples);
    for r in per_sample {
        let (id, blk, rep) = r?;
        if baq_ctx.is_some() {
            crate::selphi_info!("  lcWGS-BAM: {}: {} reads, {} triggering sites, {} realigned ({:.1}%), {} BAQ no-ops",
                id, rep.n_reads, rep.n_trigger_sites, rep.n_realigned,
                100.0 * rep.n_realigned as f64 / rep.n_reads.max(1) as f64, rep.n_baq_noop);
        }
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

/// The panel sites being piled up.
struct SiteCtx<'a> {
    pos: &'a [i64],
    ref_base: &'a [u8],
    alt_base: &'a [u8],
    is_snp: &'a [bool],
}

/// BAQ inputs: the reference window and the realignment-selection mode.
struct BaqCtx<'a> {
    ref_seq: &'a [u8],
    ref_off: i64,
    partial: bool,
}

/// What the pileup did with a sample's reads (for the log).
#[derive(Default)]
pub struct PileupReport {
    pub n_reads: u64,
    pub n_realigned: u64,
    pub n_baq_noop: u64,
    pub n_trigger_sites: usize,
}

/// One aligned read as the pileup core sees it — the BAM and CRAM adapters fill
/// this from their own record types, so the two-pass logic exists once.
struct ReadView<'a> {
    /// 1-based alignment start.
    start: i64,
    flags: u16,
    mapq: u8,
    cigar: &'a [(Kind, usize)],
    /// Read bases, ASCII.
    seq: &'a [u8],
    /// Raw phred base qualities.
    qual: &'a [u8],
    /// Fragment hash (paired mates share a QNAME), for overlap collapsing.
    qhash: u64,
}

/// A record source replays every record of the sample on `chrom` (the contig
/// filter is the adapter's) to the sink. It is invoked once (raw pileup) or
/// twice (BAQ: statistics pass, then pileup pass).
type RecordSource<'s> = dyn FnMut(&mut dyn FnMut(&ReadView)) -> io::Result<()> + 's;

/// Pileup a single BAM/CRAM → per-site normalised 3-way GL (`[n_var*3]`).
/// `.cram` is dispatched to the CRAM path (which needs `reference`).
#[allow(clippy::too_many_arguments)]
fn pileup_one(
    path: &str,
    chrom: &str,
    sites: &SiteCtx,
    region: Option<(i64, i64)>,
    reference: Option<&str>,
    indel_model: Option<&IndelModel>,
    params: &PileupParams,
    lut: &PhredLut,
    em: Option<&crate::lcwgs::errmod::ErrMod>,
    baq: Option<&BaqCtx>,
) -> io::Result<(String, Vec<f32>, PileupReport)> {
    if path.to_ascii_lowercase().ends_with(".cram") {
        return pileup_one_cram(path, chrom, sites, region, reference, indel_model, params, lut, em, baq);
    }
    let mut reader = bam::io::reader::Builder.build_from_path(path)?;
    let header = reader.read_header()?;
    let sample_id = read_group_sample_id(&header, path);

    // Map chrom name → reference id in this BAM (chr-prefix tolerant).
    let Some((target_rid, contig_name)) = resolve_contig(&header, chrom) else {
        // Contig absent → every site flat.
        return Ok((sample_id, vec![1.0f32 / 3.0; sites.pos.len() * 3], PileupReport::default()));
    };
    // Region mode with a .bai index → fetch only the region's reads (avoids
    // reading the whole file). Otherwise stream all records (zero-alloc reuse).
    let bai_path = format!("{path}.bai");
    let region_query = region.filter(|_| std::path::Path::new(&bai_path).exists());
    let index = match region_query { Some(_) => Some(bam::bai::fs::read(&bai_path)?), None => None };

    let mut cigar_buf: Vec<(Kind, usize)> = Vec::with_capacity(16);
    let mut seq_buf: Vec<u8> = Vec::with_capacity(256);
    let mut source = |sink: &mut dyn FnMut(&ReadView)| -> io::Result<()> {
        // A fresh reader per replay: the BGZF stream is not rewindable.
        let mut reader = bam::io::reader::Builder.build_from_path(path)?;
        let header = reader.read_header()?;
        let mut record = bam::Record::default();
        let mut emit = |record: &bam::Record, sink: &mut dyn FnMut(&ReadView)| {
            match record.reference_sequence_id() {
                Some(Ok(rid)) if rid == target_rid => {}
                _ => return,
            }
            let start = match record.alignment_start() {
                Some(Ok(p)) => usize::from(p) as i64, // 1-based
                _ => return,
            };
            // SAM reserves 255 for "mapping quality unavailable", and noodles models
            // that as None (MappingQuality::new(255) is None). Folding it to 0 made
            // every such read fail the min-MAPQ filter below AND, via mq_cap =
            // min(mapq, 60) in the GL model, cap its base qualities to zero. STAR
            // emits 255 for uniquely-mapped reads by default, so an entire library
            // was being discarded. samtools and bcftools mpileup compare the raw
            // value, i.e. 255 passes -q and caps nothing; match them.
            let mapq = record.mapping_quality().map_or(255, |m| m.get());
            cigar_buf.clear();
            for op in record.cigar().iter() {
                match op { Ok(o) => cigar_buf.push((o.kind(), o.len())), Err(_) => return }
            }
            seq_buf.clear();
            seq_buf.extend(record.sequence().iter());
            let qual = record.quality_scores().as_bytes();
            let qhash = record.name().map(|n| fnv1a(n.as_ref())).unwrap_or((start as u64) << 1 | 1);
            sink(&ReadView { start, flags: record.flags().bits(), mapq, cigar: &cigar_buf, seq: &seq_buf, qual, qhash });
        };
        if let (Some((rs, re)), Some(index)) = (region_query, index.as_ref()) {
            let reg = build_region(contig_name.clone(), rs, re)?;
            let mut q = reader.query(&header, index, &reg)?;
            while q.read_record(&mut record)? != 0 {
                emit(&record, sink);
            }
        } else {
            while reader.read_record(&mut record)? != 0 {
                emit(&record, sink);
            }
        }
        Ok(())
    };

    let (gl, rep) = pileup_core(&mut source, sites, params, lut, em, indel_model, baq)?;
    Ok((sample_id, gl, rep))
}

/// Pileup a single CRAM → per-site normalised 3-way GL (`[n_var*3]`).
///
/// CRAM stores read bases as differences from a reference, so a FASTA reference
/// (with a `.fai`) is required to decode them — passed via `--reference`. Each
/// sample opens its own reference repository (avoids contending on a shared
/// FASTA reader across the rayon sample threads). The per-read CIGAR-walk + GL
/// model is the SAME as the BAM path (`pileup_core`); the only difference is the
/// record source. With a `.crai` index and a region, only the region is decoded.
#[allow(clippy::too_many_arguments)]
fn pileup_one_cram(
    path: &str,
    chrom: &str,
    sites: &SiteCtx,
    region: Option<(i64, i64)>,
    reference: Option<&str>,
    indel_model: Option<&IndelModel>,
    params: &PileupParams,
    lut: &PhredLut,
    em: Option<&crate::lcwgs::errmod::ErrMod>,
    baq: Option<&BaqCtx>,
) -> io::Result<(String, Vec<f32>, PileupReport)> {
    let ref_path = reference.ok_or_else(|| io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("CRAM input ({path}) requires --reference <fasta> (with .fai) to decode read bases"),
    ))?;
    let open = || -> io::Result<(cram::io::Reader<std::fs::File>, Header)> {
        let ir = fasta::io::indexed_reader::Builder::default().build_from_path(ref_path)?;
        let repo = fasta::Repository::new(fasta::repository::adapters::IndexedReader::new(ir));
        let mut reader = cram::io::reader::Builder::default()
            .set_reference_sequence_repository(repo)
            .build_from_path(path)?;
        let header = reader.read_header()?;
        Ok((reader, header))
    };
    let (_, header) = open()?;
    let sample_id = read_group_sample_id(&header, path);
    let Some((target_rid, contig_name)) = resolve_contig(&header, chrom) else {
        return Ok((sample_id, vec![1.0f32 / 3.0; sites.pos.len() * 3], PileupReport::default()));
    };
    // Region mode with a .crai index → decode only the region. Else stream all.
    let crai_path = format!("{path}.crai");
    let region_query = region.filter(|_| std::path::Path::new(&crai_path).exists());
    let index: Option<cram::crai::Index> = match region_query {
        Some((rs, re)) => {
            // noodles' CRAM Query decodes EVERY container of the reference sequence
            // (it filters reads by interval only after decoding, no container skip /
            // early-stop). Pre-filter the crai to the containers whose ref span
            // overlaps [rs,re] so only those are seeked + decoded — O(region), not
            // O(chromosome). The per-read interval filter inside Query still applies.
            Some(cram::crai::fs::read(&crai_path)?
                .into_iter()
                .filter(|r| {
                    r.reference_sequence_id() == Some(target_rid) && {
                        let astart = r.alignment_start().map(|p| usize::from(p) as i64).unwrap_or(0);
                        let aend = astart + r.alignment_span() as i64; // exclusive max read end
                        astart <= re && aend >= rs
                    }
                })
                .collect())
        }
        None => None,
    };

    let mut cigar_buf: Vec<(Kind, usize)> = Vec::with_capacity(16);
    let mut seq_buf: Vec<u8> = Vec::with_capacity(256);
    let mut source = |sink: &mut dyn FnMut(&ReadView)| -> io::Result<()> {
        let (mut reader, header) = open()?;
        let mut emit = |record: &RecordBuf, sink: &mut dyn FnMut(&ReadView)| {
            if record.reference_sequence_id() != Some(target_rid) { return; }
            let start = match record.alignment_start() { Some(p) => usize::from(p) as i64, None => return };
            // SAM reserves 255 for "mapping quality unavailable", and noodles models
            // that as None (MappingQuality::new(255) is None). Folding it to 0 made
            // every such read fail the min-MAPQ filter below AND, via mq_cap =
            // min(mapq, 60) in the GL model, cap its base qualities to zero. STAR
            // emits 255 for uniquely-mapped reads by default, so an entire library
            // was being discarded. samtools and bcftools mpileup compare the raw
            // value, i.e. 255 passes -q and caps nothing; match them.
            let mapq = record.mapping_quality().map_or(255, |m| m.get());
            cigar_buf.clear();
            for op in record.cigar().as_ref() { cigar_buf.push((op.kind(), op.len())); }
            seq_buf.clear();
            seq_buf.extend_from_slice(record.sequence().as_ref());
            let qual: &[u8] = record.quality_scores().as_ref();
            let qhash = record.name().map(|n| fnv1a(n.as_ref())).unwrap_or((start as u64) << 1 | 1);
            sink(&ReadView { start, flags: record.flags().bits(), mapq, cigar: &cigar_buf, seq: &seq_buf, qual, qhash });
        };
        if let (Some((rs, re)), Some(index)) = (region_query, index.as_ref()) {
            let reg = build_region(contig_name.clone(), rs, re)?;
            for result in reader.query(&header, index, &reg)? {
                emit(&result?, sink);
            }
        } else {
            for result in reader.records(&header) {
                emit(&result?, sink);
            }
        }
        Ok(())
    };

    let (gl, rep) = pileup_core(&mut source, sites, params, lut, em, indel_model, baq)?;
    Ok((sample_id, gl, rep))
}

/// Format-agnostic pileup of one sample: read filtering, the optional BAQ
/// statistics pass, then the GL-accumulating pass, then GL finalisation.
fn pileup_core(
    source: &mut RecordSource,
    sites: &SiteCtx,
    params: &PileupParams,
    lut: &PhredLut,
    em: Option<&crate::lcwgs::errmod::ErrMod>,
    indel_model: Option<&IndelModel>,
    baq: Option<&BaqCtx>,
) -> io::Result<(Vec<f32>, PileupReport)> {
    let n_var = sites.pos.len();
    let mut report = PileupReport::default();
    let excl = if params.keep_supplementary { FLAG_EXCLUDE & !0x800 } else { FLAG_EXCLUDE };
    let accept = |r: &ReadView| -> bool {
        r.flags & excl == 0
            && !is_anomalous_pair(r.flags, params.count_orphans)
            && r.mapq >= params.min_mapq
            && !r.qual.is_empty()
    };

    // Pass 1 (BAQ only): per-column read statistics for bcftools' realignment
    // heuristic, over exactly the reads the pileup will use.
    let baq_sel: Option<(ColumnStats, Vec<bool>)> = match baq {
        Some(b) => {
            let mut stats = ColumnStats::new(n_var);
            let last_pos = sites.pos[n_var - 1];
            source(&mut |r: &ReadView| {
                if !accept(r) { return; }
                let has_indel = r.cigar.iter().any(|&(k, _)| matches!(k, Kind::Insertion | Kind::Deletion | Kind::Skip));
                let has_clip = r.cigar.iter().any(|&(k, _)| k == Kind::SoftClip);
                for_each_covered_site(r.start, last_pos, r.cigar, sites.pos, |v, indel_after| {
                    stats.add(v, has_indel, has_clip, indel_after);
                });
            })?;
            let trig: Vec<bool> = (0..n_var)
                .map(|v| if b.partial { stats.triggers_realign(v) } else { stats.nt[v] > 0 })
                .collect();
            report.n_trigger_sites = trig.iter().filter(|&&t| t).count();
            Some((stats, trig))
        }
        None => None,
    };

    // Pass 2: pile up, with BAQ-capped qualities on the reads selected above.
    let use_em = em.is_some();
    let mut st = PassState {
        sites, params, lut, em, indel_model, baq, baq_sel: baq_sel.as_ref(),
        ll: vec![[0.0f64; 3]; n_var],
        bases: if use_em { vec![Vec::new(); n_var] } else { Vec::new() },
        depth: vec![0u32; n_var],
        ov: OverlapState::new(n_var),
        iscratch: IndelScratch::default(),
        bsc: BaqScratch::new(),
        eq: Vec::with_capacity(256),
        last_pos: sites.pos[n_var - 1],
        report: &mut report,
    };
    source(&mut |r: &ReadView| { if accept(r) { st.on_read(r); } })?;
    let PassState { mut bases, ll, depth, .. } = st;

    let gl = match em {
        Some(e) => finalize_gl_errmod(&mut bases, sites.ref_base, sites.alt_base, sites.is_snp, e),
        None => finalize_gl(&ll, &depth),
    };
    Ok((gl, report))
}

/// Mutable state of the GL-accumulating pass.
struct PassState<'a> {
    sites: &'a SiteCtx<'a>,
    params: &'a PileupParams,
    lut: &'a PhredLut,
    em: Option<&'a crate::lcwgs::errmod::ErrMod>,
    indel_model: Option<&'a IndelModel>,
    baq: Option<&'a BaqCtx<'a>>,
    baq_sel: Option<&'a (ColumnStats, Vec<bool>)>,
    ll: Vec<[f64; 3]>,
    bases: Vec<Vec<u16>>,
    depth: Vec<u32>,
    ov: OverlapState,
    iscratch: IndelScratch,
    bsc: BaqScratch,
    /// BAQ-capped qualities of the current read (when realigned).
    eq: Vec<u8>,
    last_pos: i64,
    report: &'a mut PileupReport,
}
impl PassState<'_> {
    fn on_read(&mut self, r: &ReadView) {
        self.report.n_reads += 1;
        // BAQ: judge the read once, at the first triggering column it covers
        // (bcftools marks a read realigned at that column whatever the outcome).
        let mut realigned = false;
        if let (Some(b), Some((stats, trig))) = (self.baq, self.baq_sel) {
            let mut decision: Option<bool> = None;
            let mut trigger_v: usize = 0;
            for_each_covered_site(r.start, self.last_pos, r.cigar, self.sites.pos, |v, _| {
                if decision.is_none() && trig[v] {
                    decision = Some(baq::read_passes_realign_rule(r.qual.len(), r.cigar, stats.nt[v], stats.has_clip[v], b.partial));
                    trigger_v = v;
                }
            });
            if decision == Some(true) {
                if baq::baq_effective_quals(r.start - 1, r.cigar, r.seq, r.qual, b.ref_seq, b.ref_off, &mut self.bsc, &mut self.eq) {
                    realigned = true;
                    self.report.n_realigned += 1;
                    if self.params.baq_streaming {
                        // bcftools realigns the read when its pileup reaches the
                        // trigger column; columns before it were scored with raw
                        // qualities. Restore them for the query bases up to that column.
                        if let Some(qi) = query_index_at(r.start, r.cigar, self.sites.pos[trigger_v]) {
                            let qi = qi.min(self.eq.len());
                            self.eq[..qi].copy_from_slice(&r.qual[..qi]);
                        }
                    }
                } else {
                    self.report.n_baq_noop += 1;
                }
            }
        }
        let quals: &[u8] = if realigned { &self.eq } else { r.qual };
        let base_at = |qi: usize| r.seq.get(qi).copied().unwrap_or(b'N');
        let qual_at = |qi: usize| quals.get(qi).copied().unwrap_or(0);
        if self.em.is_some() {
            walk_record_em(
                r.start, self.last_pos, r.cigar, base_at, qual_at,
                quals.len(), r.mapq, r.flags & 0x10 != 0,
                self.sites.pos, self.sites.is_snp, self.params, &mut self.bases, &mut self.depth, r.qhash, &mut self.ov,
            );
        } else {
            walk_record(
                r.start, self.last_pos, r.cigar, base_at, qual_at,
                self.sites.pos, self.sites.ref_base, self.sites.alt_base, self.sites.is_snp, self.params, self.lut,
                &mut self.ll, &mut self.depth, r.qhash, &mut self.ov,
            );
            if let Some(model) = self.indel_model {
                super::indel_realign::score_read(
                    r.start, r.cigar, base_at, qual_at,
                    model, self.lut, &mut self.iscratch, &mut self.ll, &mut self.depth,
                    self.params.max_depth, self.params.min_bq, r.qhash, &mut self.ov.last_frag,
                );
            }
        }
    }
}

/// Query (read) index of the base aligned at 1-based reference position
/// `target`; for a position spanned by a deletion/skip, the index of the next
/// aligned base. `None` if the read's aligned span does not reach `target`.
fn query_index_at(start: i64, cigar: &[(Kind, usize)], target: i64) -> Option<usize> {
    let mut refcur = start;
    let mut qcur: usize = 0;
    for &(kind, len) in cigar {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                if target >= refcur && target < refcur + len as i64 {
                    return Some(qcur + (target - refcur) as usize);
                }
                refcur += len as i64;
                qcur += len;
            }
            Kind::Deletion | Kind::Skip => {
                if target >= refcur && target < refcur + len as i64 { return Some(qcur); }
                refcur += len as i64;
            }
            Kind::Insertion | Kind::SoftClip => { qcur += len; }
            Kind::HardClip | Kind::Pad => {}
        }
    }
    None
}

/// Visit every panel site a read's alignment covers, in ascending order — bases
/// under M/=/X ops and positions spanned by D/N ops (a deleted base is still a
/// member of the pileup column). `indel_after` is htslib's `p->indel`: the
/// signed length of an insertion (+) or deletion (−) immediately following this
/// base, 0 otherwise.
fn for_each_covered_site(start: i64, last_pos: i64, cigar: &[(Kind, usize)], pos: &[i64], mut f: impl FnMut(usize, i32)) {
    let n_var = pos.len();
    let ref_span: i64 = cigar.iter().map(|&(k, l)| match k {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch | Kind::Deletion | Kind::Skip => l as i64,
        _ => 0,
    }).sum();
    let mut si = pos.partition_point(|&p| p < start);
    if si >= n_var || pos[si] > start + ref_span { return; }
    let mut refcur = start;
    for (idx, &(kind, len)) in cigar.iter().enumerate() {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                let ref_end = refcur + len as i64;
                let indel_after: i32 = match cigar.get(idx + 1) {
                    Some(&(Kind::Insertion, l)) => l as i32,
                    Some(&(Kind::Deletion, l)) => -(l as i32),
                    _ => 0,
                };
                while si < n_var && pos[si] < refcur { si += 1; }
                while si < n_var && pos[si] < ref_end {
                    f(si, if pos[si] == ref_end - 1 { indel_after } else { 0 });
                    si += 1;
                }
                refcur = ref_end;
            }
            Kind::Deletion | Kind::Skip => {
                let ref_end = refcur + len as i64;
                while si < n_var && pos[si] < refcur { si += 1; }
                while si < n_var && pos[si] < ref_end {
                    f(si, 0);
                    si += 1;
                }
                refcur = ref_end;
            }
            Kind::Insertion | Kind::SoftClip | Kind::HardClip | Kind::Pad => {}
        }
        if refcur > last_pos { break; }
    }
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

/// FNV-1a 64-bit hash of a read name, used to detect overlapping paired-end
/// mates (the two mates share a QNAME). `u64::MAX` is reserved as "unset".
#[inline]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
    if h == u64::MAX { 0 } else { h }
}

/// Per-site state for collapsing overlapping paired-end mates. A fragment's two
/// mates cover the same DNA molecule, so an overlapped site is ONE observation:
/// `last_frag[v]` is the last fragment to score site `v`; `first_base`/
/// `first_qual` hold that observation so the partner mate can be merged in
/// (agreeing bases → best quality kept; disagreeing → higher-quality base,
/// quality reduced by the conflict) rather than double-counted.
/// `first_idx[v]` is the index in the errmod `bases[v]` buffer of the current
/// fragment's first-mate observation, so the overlapping mate can be merged into
/// it in place (the errmod path; the naive path uses `first_base`/`first_qual`).
struct OverlapState { last_frag: Vec<u64>, first_base: Vec<u8>, first_qual: Vec<u8>, first_idx: Vec<u32> }
impl OverlapState {
    fn new(n: usize) -> Self {
        OverlapState { last_frag: vec![u64::MAX; n], first_base: vec![0; n], first_qual: vec![0; n], first_idx: vec![0; n] }
    }
}

/// Shared CIGAR-walk + GL accumulation for one read, used by BOTH the BAM
/// (concrete) and CRAM (trait) paths. `cigar` is the read's ops as (kind, len);
/// `base_at(qi)`/`qual_at(qi)` fetch the read base (ASCII) / phred quality at a
/// query index. Maps each covered panel site to its read base+qual and folds it
/// into the per-genotype log-likelihoods. Format-agnostic (the only BAM/CRAM
/// difference — how a record exposes cigar/seq/qual — is supplied by the caller).
///
/// Overlapping paired-end mates (same fragment) are counted ONCE per site:
/// `qhash` is the read's fragment hash and `last_frag[v]` records the last
/// fragment that contributed to site `v`; a second mate of the same fragment is
/// skipped. Without this, both mates double-count the fragment's evidence,
/// inflating GL confidence at ~30-50% of sites (overlap region) and degrading
/// imputation — matching `samtools`/`bcftools`/GLIMPSE2 overlap handling.
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
    qhash: u64,
    ov: &mut OverlapState,
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
                    if is_snp[v] {
                        let qi = qcur + (pos[v] - refcur) as usize;
                        let b = base_at(qi);
                        let qraw = qual_at(qi);
                        if b != b'N' && qraw >= params.min_bq {
                            let q = (qraw as usize).min(93);
                            if ov.last_frag[v] == qhash {
                                // Overlapping mate of the same fragment: merge into the
                                // existing observation (one molecule) instead of adding.
                                let fb = ov.first_base[v];
                                let fq = ov.first_qual[v] as usize;
                                let (mb, mq) = if b == fb {
                                    (fb, q.max(fq))                       // agree → best quality
                                } else if q >= fq {
                                    (b, (q - fq).max(1))                  // disagree → higher-q base, reduced
                                } else {
                                    (fb, (fq - q).max(1))
                                };
                                accumulate(&mut ll[v], fb, ref_base[v], alt_base[v], fq, lut, -1.0);
                                accumulate(&mut ll[v], mb, ref_base[v], alt_base[v], mq, lut, 1.0);
                                ov.first_base[v] = mb;
                                ov.first_qual[v] = mq as u8;
                            } else if depth[v] < params.max_depth {
                                accumulate(&mut ll[v], b, ref_base[v], alt_base[v], q, lut, 1.0);
                                depth[v] += 1;
                                ov.last_frag[v] = qhash;
                                ov.first_base[v] = b;
                                ov.first_qual[v] = q as u8;
                            }
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

/// errmod variant of [`walk_record`]: instead of accumulating the naive
/// independent-product log-likelihood, it COLLECTS each covered SNP site's read
/// base as a packed `u16` (`qual<<5 | strand<<4 | base`, base 0..=3) into
/// `bases[v]`, applying bcftools `mpileup`'s exact per-base quality processing:
/// neighbour-quality cap (`q ≤ neighbour+30`), `min_baseQ` skip, `max_baseQ=60`
/// cap, mapping-quality cap (`q ≤ min(mapQ,60)`), and the `[4,63]` clamp. The
/// per-site base lists are later fed to [`crate::lcwgs::errmod::ErrMod::cal`].
/// Overlapping mates of one fragment are counted once (keep first).
#[allow(clippy::too_many_arguments)]
fn walk_record_em<B: Fn(usize) -> u8, Q: Fn(usize) -> u8>(
    start: i64,
    last_pos: i64,
    cigar: &[(Kind, usize)],
    base_at: B,
    qual_at: Q,
    read_len: usize,
    mapq: u8,
    is_rev: bool,
    pos: &[i64],
    is_snp: &[bool],
    params: &PileupParams,
    bases: &mut [Vec<u16>],
    depth: &mut [u32],
    qhash: u64,
    ov: &mut OverlapState,
) {
    let n_var = pos.len();
    let ref_span: i64 = cigar.iter().map(|&(k, l)| match k {
        Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch | Kind::Deletion | Kind::Skip => l as i64,
        _ => 0,
    }).sum();
    let mut si = pos.partition_point(|&p| p < start);
    if si >= n_var || pos[si] > start + ref_span { return; }
    let mq_cap = (mapq as i32).min(60);
    let strand_bit = (is_rev as u16) << 4;
    let mut refcur = start;
    let mut qcur: usize = 0;
    for &(kind, len) in cigar {
        match kind {
            Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                let ref_end = refcur + len as i64;
                while si < n_var && pos[si] < refcur { si += 1; }
                while si < n_var && pos[si] < ref_end {
                    let v = si;
                    if is_snp[v] {
                        let qi = qcur + (pos[v] - refcur) as usize;
                        let base4 = ascii_to_base4(base_at(qi));
                        if base4 <= 3 && (depth[v] < params.max_depth || ov.last_frag[v] == qhash) {
                            let mut q = qual_at(qi) as i32;
                            // neighbour-quality cap (bcftools delta_baseQ = 30)
                            if qi > 0 { let nq = qual_at(qi - 1) as i32; if q > nq + 30 { q = nq + 30; } }
                            if qi + 1 < read_len { let nq = qual_at(qi + 1) as i32; if q > nq + 30 { q = nq + 30; } }
                            if q >= params.min_bq as i32 {
                                if q > 60 { q = 60; }          // max_baseQ
                                if q > mq_cap { q = mq_cap; }   // mapping-quality cap
                                if q > 63 { q = 63; }
                                if q < 4 { q = 4; }
                                if ov.last_frag[v] == qhash {
                                    // Overlapping mate of one fragment: merge into the first
                                    // mate's buffered base (samtools tweak_overlap_quality):
                                    // agreeing bases → SUM the qualities (cap 60); disagreeing
                                    // → keep the higher-quality base at 0.8×, drop the other.
                                    let idx = ov.first_idx[v] as usize;
                                    let p = bases[v][idx];
                                    let q1 = (p >> 5) as i32;
                                    let bs1 = p & 0x1f; // first mate's strand|base
                                    let base1 = (p & 0xf) as usize;
                                    let (keep_bs, nq) = if base4 == base1 {
                                        (bs1, (q1 + q).min(60))
                                    } else if q > q1 {
                                        (strand_bit | base4 as u16, ((q as f64 * 0.8) as i32).max(4))
                                    } else {
                                        (bs1, ((q1 as f64 * 0.8) as i32).max(4))
                                    };
                                    let nq = nq.clamp(4, 63) as u16;
                                    bases[v][idx] = (nq << 5) | keep_bs;
                                } else {
                                    ov.first_idx[v] = bases[v].len() as u32;
                                    bases[v].push(((q as u16) << 5) | strand_bit | base4 as u16);
                                    depth[v] += 1;
                                    ov.last_frag[v] = qhash;
                                }
                            }
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

/// Per-site 3-way GL from the collected errmod bases. For each SNP site runs the
/// samtools/bcftools `errmod_cal` over `bases[v]`, extracts the (REF,REF),
/// (REF,ALT), (ALT,ALT) phred likelihoods for the panel alleles and normalises
/// to a 3-way GL. Sites with no reads (or non-ACGT alleles) stay flat `[⅓,⅓,⅓]`.
fn finalize_gl_errmod(
    bases: &mut [Vec<u16>],
    ref_base: &[u8],
    alt_base: &[u8],
    is_snp: &[bool],
    em: &crate::lcwgs::errmod::ErrMod,
) -> Vec<f32> {
    let n_var = bases.len();
    let mut out = vec![1.0f32 / 3.0; n_var * 3];
    let mut q = [0.0f32; 25];
    for v in 0..n_var {
        if !is_snp[v] || bases[v].is_empty() { continue; }
        let r = ascii_to_base4(ref_base[v]);
        let a = ascii_to_base4(alt_base[v]);
        if r > 3 || a > 3 { continue; }
        em.cal(&mut bases[v], 5, &mut q);
        // phred → likelihood (lower phred = more likely); normalise.
        let l0 = 10f64.powf(-(q[r * 5 + r] as f64) / 10.0);
        let l1 = 10f64.powf(-(q[r * 5 + a] as f64) / 10.0);
        let l2 = 10f64.powf(-(q[a * 5 + a] as f64) / 10.0);
        let s = l0 + l1 + l2;
        if s > 0.0 {
            out[v * 3] = (l0 / s) as f32;
            out[v * 3 + 1] = (l1 / s) as f32;
            out[v * 3 + 2] = (l2 / s) as f32;
        }
    }
    out
}

/// Accumulate one read base into the 3 genotype log10-likelihoods, scaled by
/// `sign` (+1 to add an observation, −1 to remove a previously-added one — used
/// when an overlapping mate replaces the first mate's base).
#[inline(always)]
fn accumulate(ll: &mut [f64; 3], b: u8, rb: u8, ab: u8, q: usize, lut: &PhredLut, sign: f64) {
    // hom-REF: P(b|ref); hom-ALT: P(b|alt); het: ½P(b|ref)+½P(b|alt).
    let (lref, pref) = if b == rb { (lut.lmatch[q], lut.pmatch[q]) } else { (lut.lmis[q], lut.pmis[q]) };
    let (lalt, palt) = if b == ab { (lut.lmatch[q], lut.pmatch[q]) } else { (lut.lmis[q], lut.pmis[q]) };
    ll[0] += sign * lref;
    ll[2] += sign * lalt;
    ll[1] += sign * (0.5 * pref + 0.5 * palt).log10();
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
        accumulate(&mut ll, b'A', b'A', b'G', 30, &lut, 1.0);
        assert!(ll[0] > ll[1] && ll[1] > ll[2], "A read: g0>g1>g2, got {:?}", ll);
        // One G read (alt) → favors hom-alt.
        let mut ll2 = [0.0f64; 3];
        accumulate(&mut ll2, b'G', b'A', b'G', 30, &lut, 1.0);
        assert!(ll2[2] > ll2[1] && ll2[1] > ll2[0], "G read: g2>g1>g0, got {:?}", ll2);
        // One A + one G (het evidence) → het most likely.
        let mut llh = [0.0f64; 3];
        accumulate(&mut llh, b'A', b'A', b'G', 30, &lut, 1.0);
        accumulate(&mut llh, b'G', b'A', b'G', 30, &lut, 1.0);
        assert!(llh[1] > llh[0] && llh[1] > llh[2], "A+G: het wins, got {:?}", llh);
    }

    #[test]
    fn covered_sites_report_indel_after_last_base_and_deleted_columns() {
        // read: 10M 2I 5M 3D 4M starting at 100 → ref 100..109 (M), 110..114 (M), 115..117 (D), 118..121 (M)
        let cigar = vec![(Kind::Match, 10), (Kind::Insertion, 2), (Kind::Match, 5), (Kind::Deletion, 3), (Kind::Match, 4)];
        let pos: Vec<i64> = vec![99, 105, 109, 114, 116, 121, 122];
        let mut seen = Vec::new();
        for_each_covered_site(100, 200, &cigar, &pos, |v, ia| seen.push((pos[v], ia)));
        assert_eq!(seen, vec![(105, 0), (109, 2), (114, -3), (116, 0), (121, 0)]);
    }
}
