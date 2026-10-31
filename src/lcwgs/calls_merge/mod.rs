//! `--calls`: one sample's variant caller VCF (or gVCF), its reads and the reference panel in one
//! VCF. The lcWGS engine runs on the reads exactly as `--lcwgs --bam` does (same pileup, same
//! chain, same seed: the same genotype posteriors), counting the q20 read depth at every panel
//! site inside the pileup's own pass over the reads; then the caller's file is streamed once, in
//! position order, and written back with Selphi's genotypes where the caller is not trusted:
//!
//! - where the caller has **no variant call** at a panel allele (nothing, a 0/0 record or block, a
//!   no-call, a filtered call), the allele goes to Selphi when the q20 depth is below
//!   `--calls-depth` T (default 20);
//! - a caller **PASS variant call** goes to Selphi when its GQ is below `--calls-min-gq` G (default
//!   0 = never), at any depth, SNP calls only unless `--calls-gq-scope all`.
//!
//! Caller and panel alleles are compared after the same normalisation (`io::allele_norm`, as
//! `bcftools norm -m -any -f FASTA -c w`). A Selphi row is written only where Selphi's genotype
//! differs from what the caller's file says about that allele. Caller records are replaced whole
//! or not at all (records that share an allele as one group); overlapping records and rows are
//! decided together, and a cluster whose result would contradict itself (a non-reference row over
//! a kept call, more than two alleles at a base, anything inside a homozygous Selphi deletion) is
//! written as the caller wrote it. Every replaced or compared record is named in FORMAT
//! provenance fields. The caller's header is kept verbatim; blocks are cut into fragments around
//! Selphi rows.

mod decide;
mod header;
mod merge;
mod reader;
mod record;
mod writer;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};

use crate::io::allele_norm::RefSlice;
use crate::io::contig_length::same_contig;
use crate::io::target_io::VcfLines;
use crate::srp::SrpReader;

use super::multi::PanelSet;
use super::LcwgsParams;

pub use writer::temporary_names;

/// The merge policy (section 5 of the design).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallsPolicy {
    /// `--calls-depth`: a panel allele where the caller has no variant call goes to Selphi when the
    /// q20 depth there is below this.
    pub depth: u32,
    /// `--calls-min-gq`: a PASS variant call with FORMAT/GQ below this goes to Selphi (0 = never).
    pub min_gq: i64,
    /// `--calls-gq-scope all`: the GQ test applies to every variant record, not SNP records only.
    pub gq_all: bool,
    /// `--calls-gvcf`: a position nothing covers is a no-call (else reference).
    pub gvcf: bool,
    /// `SELPHI_CALLS_WINDOW`: how far normalisation may move a record left, and the merge's lookahead.
    pub window: i64,
}

impl CallsPolicy {
    pub const DEFAULT_DEPTH: u32 = 20;
    pub const DEFAULT_WINDOW: i64 = 10_000;

    /// The longest cluster decided; a longer chain of overlapping intervals is written as read.
    #[inline]
    pub fn cluster_max(&self) -> i64 { self.window.saturating_mul(10) }

    /// `SELPHI_CALLS_WINDOW` (bp), default 10,000, within [100, 1,000,000].
    pub fn window_from_env() -> Result<i64, String> {
        match crate::config::raw("SELPHI_CALLS_WINDOW") {
            None => Ok(Self::DEFAULT_WINDOW),
            Some(v) => match v.trim().parse::<i64>() {
                Ok(w) if (100..=1_000_000).contains(&w) => Ok(w),
                _ => Err(format!("SELPHI_CALLS_WINDOW={v}: give a number of bases from 100 to 1000000")),
            },
        }
    }
}

/// A slab of reusable items addressed by index (their buffers keep their capacity).
pub(super) struct Slab<T> { items: Vec<T>, free: Vec<u32> }

impl<T> Default for Slab<T> { fn default() -> Self { Slab { items: Vec::new(), free: Vec::new() } } }

impl<T: Default> Slab<T> {
    pub(super) fn alloc(&mut self) -> u32 {
        match self.free.pop() {
            Some(i) => i,
            None => { self.items.push(T::default()); (self.items.len() - 1) as u32 }
        }
    }
    pub(super) fn release(&mut self, i: u32) { self.free.push(i); }
    pub(super) fn live(&self) -> usize { self.items.len() - self.free.len() }
}

impl<T> std::ops::Index<u32> for Slab<T> {
    type Output = T;
    #[inline] fn index(&self, i: u32) -> &T { &self.items[i as usize] }
}

impl<T> std::ops::IndexMut<u32> for Slab<T> {
    #[inline] fn index_mut(&mut self, i: u32) -> &mut T { &mut self.items[i as usize] }
}

/// Everything a `--calls` run needs besides the panel.
pub struct CallsArgs<'a> {
    pub calls: &'a str,
    pub policy: CallsPolicy,
    /// The one BAM/CRAM.
    pub bams: &'a [String],
    pub reference: &'a str,
    pub threads: usize,
    /// `<out>.vcf.gz`.
    pub out: &'a Path,
    pub extra_header: &'a [String],
    /// `--refpanel` (or `--refpanel-dir`) and `--map` (or `--map-dir`), for the provenance line.
    pub panel: &'a str,
    pub map: &'a str,
}

/// A checked calls file: its header, and its records' stream positioned after it.
pub struct Prepared {
    header: header::CallsHeader,
    lines: VcfLines,
    /// The reads' sample name (their `@RG SM`, else the file stem).
    pub reads_sample: String,
}

impl Prepared {
    /// The caller's sample name: the run's identity.
    pub fn caller_sample(&self) -> &str { &self.header.sample }
}

/// A refusal before the run (written to stderr: no log is open yet).
fn refuse(msg: impl Into<String>) -> crate::Fatal { crate::Fatal::error(2, msg.into()) }

/// Check the calls file, the reads' sample and the `--haploids` list before anything is written
/// (section 3): one sample column, one sample in the reads, a header the merge can extend, and a
/// `--haploids` entry naming the caller's sample (the run's identity), not only the reads'.
/// Returns the checked file and the `--haploids` list the engine matches against the reads.
pub fn prepare(args: &CallsArgs, haploids: Option<&[String]>) -> Result<(Prepared, Option<Vec<String>>), crate::Fatal> {
    let mut lines = VcfLines::open(args.calls).map_err(|e| refuse(format!("--calls {}: {e}", args.calls)))?;
    let header = header::CallsHeader::read(args.calls, &mut lines)?;
    header.check_extra(args.extra_header)?;
    let [bam] = args.bams else {
        return Err(refuse(format!("--calls merges one sample: give one BAM/CRAM (got {})", args.bams.len())));
    };
    let (names, reads_sample) = crate::lcwgs::bam_pileup::read_group_samples(bam)
        .map_err(|e| refuse(format!("--calls: cannot read the header of {bam}: {e}")))?;
    if names.len() > 1 {
        return Err(refuse(format!("--calls: the read groups of {bam} name {} samples ({}); --calls merges one sample",
            names.len(), names.join(", "))));
    }
    let caller = header.sample.as_str();
    let haploids = match haploids {
        Some(list) if caller != reads_sample && list.contains(&reads_sample) && !list.iter().any(|n| n == caller) => {
            return Err(refuse(format!("--haploids names {reads_sample}, the reads' sample, but not {caller}, the calls' sample: \
                with --calls the sample is named by the calls file")));
        }
        Some(list) if caller != reads_sample && list.iter().any(|n| n == caller) => {
            let mut l = list.to_vec();
            l.push(reads_sample.clone());
            Some(l)
        }
        Some(list) => Some(list.to_vec()),
        None => None,
    };
    Ok((Prepared { header, lines, reads_sample }, haploids))
}

/// The inputs `--out` (and its log) must not overwrite: the calls file and its index, the reads and
/// theirs, the FASTA and its `.fai`, and every genetic map file the run can read (`map`: a file,
/// `a,b` files, or a directory, with the PAR maps a chrX map is joined with).
pub fn inputs(calls: &str, bam: &str, reference: &str, map: &str) -> std::io::Result<Vec<String>> {
    let mut v = vec![calls.to_string(), format!("{calls}.tbi"), format!("{calls}.csi"), bam.to_string(),
        reference.to_string(), format!("{reference}.fai")];
    for ix in [".bai", ".crai", ".csi"] { v.push(format!("{bam}{ix}")); }
    if let Some(stem) = bam.strip_suffix(".bam") { v.push(format!("{stem}.bai")); }
    if let Some(stem) = bam.strip_suffix(".cram") { v.push(format!("{stem}.crai")); }
    let path = Path::new(map);
    let mut add = |spec: &Path| v.extend(crate::genmap::map_files(spec).0.iter().map(|p| p.to_string_lossy().into_owned()));
    if path.is_dir() {
        // A map found in the directory is read as a specification of its own (a name with a comma lists files).
        for e in std::fs::read_dir(path)? { add(&e?.path()); }
    } else {
        add(path);
    }
    Ok(v)
}

/// The contigs of a FASTA's `.fai`.
fn fasta_contigs(reference: &str) -> std::io::Result<Vec<String>> {
    let text = std::fs::read_to_string(format!("{reference}.fai"))
        .map_err(|e| std::io::Error::new(e.kind(), format!("--reference {reference}: its .fai: {e}")))?;
    Ok(text.lines().filter_map(|l| l.split('\t').next()).filter(|n| !n.is_empty()).map(str::to_string).collect())
}

/// The one name among `names` that means `contig` (the same name, or one differing by a `chr`
/// prefix). `Err` when two do, even if one of them is the exact name: the stages of the run that
/// resolve names on their own (the pileup, the reference loader) must not land on another one.
fn resolve<'n>(names: &'n [String], contig: &str, what: &str) -> Result<Option<&'n str>, crate::Fatal> {
    let mut hits = names.iter().filter(|n| same_contig(n, contig));
    match (hits.next(), hits.next()) {
        (Some(n), None) => Ok(Some(n)),
        (None, _) => Ok(None),
        (Some(a), Some(b)) => Err(crate::Fatal::error(2, format!("--calls: contig {contig} matches two {what} contigs, {a} and {b}"))),
    }
}

/// The provenance header line.
fn provenance(args: &CallsArgs, header: &header::CallsHeader, reads_sample: &str, params: &LcwgsParams) -> String {
    use header::quoted;
    let base = |p: &str| Path::new(p).file_name().map_or_else(|| p.to_string(), |f| f.to_string_lossy().into_owned());
    let pol = &args.policy;
    format!("##SelphiCalls=<Version={},Mode={},Depth={},MinGQ={},GqScope={},Window={},Panel={},Map={},Reads={},Reference={},Seed={},CallerSample={},ReadsSample={}>",
        quoted(env!("CARGO_PKG_VERSION")), quoted(if pol.gvcf { "gvcf" } else { "variant" }), pol.depth, pol.min_gq,
        quoted(if pol.gq_all { "all" } else { "snp" }), pol.window, quoted(&base(args.panel)), quoted(&base(args.map)),
        quoted(&base(&args.bams[0])), quoted(&base(args.reference)), params.seed_or_default(), quoted(&header.sample),
        quoted(reads_sample))
}

fn log_start(args: &CallsArgs, header: &header::CallsHeader, reads_sample: &str) {
    let p = &args.policy;
    crate::selphi_step!("--calls: {} (sample {}; the reads' sample is {}), {} VCF; Selphi where the caller has no variant call and the q20 depth is < {}{}; window {} bp",
        args.calls, header.sample, reads_sample, if p.gvcf { "gVCF" } else { "variant" }, p.depth,
        if p.min_gq > 0 { format!(", and for PASS {} calls with GQ < {}", if p.gq_all { "variant" } else { "SNP" }, p.min_gq) } else { String::new() },
        p.window);
}

/// Merge on a single-chromosome panel (or one chromosome's view, with `--region`).
pub fn run_single(args: &CallsArgs, prep: Prepared, params: &LcwgsParams, srp: &SrpReader, map: &str,
    region: Option<(&str, i64, i64)>) -> std::io::Result<()> {
    let Prepared { header, lines, reads_sample } = prep;
    let header = &header;
    log_start(args, header, &reads_sample);
    let chrom = srp.chromosome();
    let Some(cname) = resolve(&header.contigs, chrom, "calls-header")? else {
        return Err(crate::Fatal::error(2, format!("--calls: the calls header declares no contig for the panel's chromosome {chrom}")).into());
    };
    let contig = header.contigs.iter().position(|c| c == cname).expect("resolved") as u32;
    let fasta = fasta_contigs(args.reference)?;
    let Some(fa_name) = resolve(&fasta, cname, "FASTA")? else {
        return Err(crate::Fatal::error(2, format!("--calls: the FASTA has no contig {cname}")).into());
    };
    // The contig's first record by the calls file's own .tbi (no index: stream and skip).
    let tbi = format!("{}.tbi", args.calls);
    let start = Path::new(&tbi).exists().then(|| crate::srp::csi::parse_tbi(Path::new(&tbi)).ok()).flatten()
        .and_then(|ix| crate::srp::csi::tbi_contig_start(&ix, cname));
    let lines = match start {
        Some(at) => VcfLines::open_at(args.calls, at)?,
        None => lines,
    };
    let mut reader = reader::CallsReader::new(lines, header, Some(contig))?;

    let t = std::time::Instant::now();
    // The engine is given the panel's own name for the chromosome, which the region must mean.
    if let Some((c, _, _)) = region && !same_contig(c, chrom) {
        return Err(crate::Fatal::error(2, format!("--calls: --region names {c}, but the panel holds {chrom}")).into());
    }
    let region = region.map(|(_, s, e)| (chrom, s, e));
    let (out, depth) = super::pipeline::run_lcwgs_bam_with_depth(args.bams, srp, map, params, region, Some(args.reference), args.threads)?;
    crate::selphi_step!("lcwgs done: {} variants (Selphi + read depth)", out.n_variants);
    crate::common::mem::release_freed_memory();
    let t_merge = std::time::Instant::now();
    let refseq = crate::lcwgs::bam_pileup::load_reference_window(args.reference, fa_name, None)?;
    let mut w = writer::OutWriter::create(args.out, args.threads, header.contigs.len())?;
    let mut hb = Vec::with_capacity(header.lines.len() + 8192);
    header.write(&mut hb, args.extra_header, &provenance(args, header, &reads_sample, params))?;
    w.header(&hb)?;
    let mut sh = merge::Shared::default();
    let cx = merge::Contig { pol: &args.policy, srp, out: &out, depth: &depth, fa: RefSlice::new(&refseq.seq, refseq.off),
        contig, name: cname, region: region.map(|(_, s, e)| (s, e)) };
    merge::merge_contig(&cx, &mut sh, &mut reader, &mut w)?;
    let lines_written = w.lines;
    w.finish()?;
    log_end(&sh, lines_written, t.elapsed().as_secs_f64(), t_merge.elapsed().as_secs_f64(), args.out);
    Ok(())
}

/// Merge on a multi-chromosome panel: every calls contig in header order; the run contigs (in the
/// panel, with a map, not chrY/chrM, covered by the reads, in the FASTA) are merged, the others'
/// records written as read.
pub fn run_multi(args: &CallsArgs, prep: Prepared, params: &LcwgsParams, set: &PanelSet) -> std::io::Result<()> {
    let Prepared { header, lines, reads_sample } = prep;
    let header = &header;
    log_start(args, header, &reads_sample);
    let fasta = fasta_contigs(args.reference)?;
    let covered = crate::lcwgs::bam_pileup::covered_contigs(args.bams)?;
    // The run contigs: (panel chromosome, FASTA contig) per header contig.
    let mut run: Vec<Option<(String, String)>> = vec![None; header.contigs.len()];
    let mut claimed: Vec<(String, usize)> = Vec::new();
    for (ci, c) in header.contigs.iter().enumerate() {
        let chr = match set.find(c) {
            Ok(Some(chr)) => chr.to_string(),
            Ok(None) => continue,
            Err(e) => return Err(crate::Fatal::error(2, format!("--calls: {e}")).into()),
        };
        if crate::contig::nonrecomb_refusal(&chr).is_some() || !covered.iter().any(|c| same_contig(c, &chr)) || set.map(&chr).is_err() { continue; }
        let Some(fa) = resolve(&fasta, c, "FASTA")? else {
            return Err(crate::Fatal::error(2, format!("--calls: contig {c} is in the panel, the maps and the reads but not in the FASTA")).into());
        };
        if let Some((_, other)) = claimed.iter().find(|(p, _)| *p == chr) {
            return Err(crate::Fatal::error(2, format!("--calls: the calls contigs {} and {c} both mean the panel's {chr}",
                header.contigs[*other])).into());
        }
        claimed.push((chr.clone(), ci));
        run[ci] = Some((chr, fa.to_string()));
    }
    crate::selphi_step!("--calls on a multi-chromosome panel: {} of the calls file's {} contigs are run ({}); the others' records are written as read",
        claimed.len(), header.contigs.len(), claimed.iter().map(|(_, ci)| header.contigs[*ci].as_str()).collect::<Vec<_>>().join(", "));

    let mut reader = reader::CallsReader::new(lines, header, None)?;
    let mut w = writer::OutWriter::create(args.out, args.threads, header.contigs.len())?;
    let mut hb = Vec::with_capacity(header.lines.len() + 8192);
    header.write(&mut hb, args.extra_header, &provenance(args, header, &reads_sample, params))?;
    w.header(&hb)?;
    let mut sh = merge::Shared::default();
    let t = std::time::Instant::now();
    let mut t_merge = 0.0f64;
    let mut line: Vec<u8> = Vec::new();
    for (ci, c) in header.contigs.iter().enumerate() {
        let ci32 = ci as u32;
        let has_records = reader.peek().is_some_and(|(rc, _)| rc == ci32);
        let merged = match &run[ci] {
            Some((chr, fa)) => {
                crate::selphi_step!("--calls: contig {c} (panel {chr})");
                let srp = set.load(chr)?;
                let map = set.map(chr)?;
                match super::pipeline::run_lcwgs_bam_with_depth(args.bams, &srp, &map, params, None, Some(args.reference), args.threads) {
                    Ok((out, depth)) => {
                        crate::common::mem::release_freed_memory();
                        let tm = std::time::Instant::now();
                        let refseq = crate::lcwgs::bam_pileup::load_reference_window(args.reference, fa, None)?;
                        let cx = merge::Contig { pol: &args.policy, srp: &srp, out: &out, depth: &depth,
                            fa: RefSlice::new(&refseq.seq, refseq.off), contig: ci32, name: c, region: None };
                        merge::merge_contig(&cx, &mut sh, &mut reader, &mut w)?;
                        t_merge += tm.elapsed().as_secs_f64();
                        true
                    }
                    // The reads cover it (checked above): the engine finding none is a naming mismatch.
                    Err(e) if super::pipeline::is_no_data(&e) => {
                        return Err(crate::Fatal::error(2, format!("--calls: contig {c} (panel {chr}): {e}, though the reads' index lists it")).into());
                    }
                    Err(e) => return Err(e),
                }
            }
            None => false,
        };
        if merged || !has_records { continue; }
        // Not run: the contig's records as read.
        while reader.peek().is_some_and(|(rc, _)| rc == ci32) {
            let m = reader.take_into(&mut line)?.expect("peeked");
            w.record(ci32, c, m.pos, record::tbi_rlen_of(&line, m.pos, m.line_no)?, &line)?;
        }
    }
    let lines_written = w.lines;
    w.finish()?;
    log_end(&sh, lines_written, t.elapsed().as_secs_f64(), t_merge, args.out);
    Ok(())
}

fn log_end(sh: &merge::Shared, lines: u64, total_s: f64, merge_s: f64, out: &Path) {
    let c = &sh.counters;
    crate::selphi_step!("--calls: {} panel rows; handed to Selphi {} (no variant call, depth < T) + {} (low-GQ variant calls); \
        vetoed {} repeat / {} unnormalised panel / {} duplicate panel rows",
        c.rows, c.handed_h1, c.handed_h2, c.veto_repeat, c.veto_unnormalised, c.veto_duplicate);
    crate::selphi_step!("--calls: wrote {} Selphi rows ({} non-reference); {} caller records replaced in {} groups; \
        compared and kept: {} both / {} partial / {} overlap",
        c.written, c.written_nonref, c.replaced_records, c.groups_replaced, c.both, c.partial, c.overlap);
    crate::selphi_step!("--calls: incoherent clusters: {} resolved without their indel rows ({} handed indel rows taken back, {} of \
        them written by the first resolution), {} reverted ({} rows; C1 {} / C2 {} / C3 {}); {} frozen clusters ({} bp at first, {} rows)",
        c.snp_only, c.snp_only_indel_rows, c.snp_only_written, c.reverts, c.revert_rows, c.revert_c1, c.revert_c2, c.revert_c3,
        c.frozen, c.frozen_bases, c.frozen_rows);
    crate::selphi_step!("--calls: {} unresolved caller records ({} past the window), {} context records, {} blocks cut into {} fragments \
        ({} blocks frozen); queue high-water {} items, {} records alive at most",
        c.unresolved, c.unresolved_shift, c.context, c.blocks_cut, c.fragments, c.blocks_frozen, sh.queue_high, sh.live_high);
    crate::selphi_step!("--calls: wrote {} ({} lines; {:.1} s merge of {:.1} s)", out.display(), lines, merge_s, total_s);
}

/// The output path of a `--calls` run (`<out>.vcf.gz`) and every file it writes: itself, its index
/// and their temporary names.
pub fn outputs(vcf: &Path) -> Vec<PathBuf> {
    let mut tbi = vcf.as_os_str().to_owned();
    tbi.push(".tbi");
    let mut v = vec![vcf.to_path_buf(), PathBuf::from(tbi)];
    v.extend(temporary_names(vcf));
    v
}
