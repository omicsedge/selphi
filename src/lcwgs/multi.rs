//! lcWGS on a multi-chromosome panel: each panel chromosome the input covers runs as its own
//! single-chromosome lcWGS run (`pipeline::run_lcwgs` / `run_lcwgs_bam` on that chromosome's
//! panel), written in panel order into one VCF.gz and one dosage TSV. A chromosome's records are
//! those of the same run made with that chromosome's panel alone.

use std::path::{Path, PathBuf};

use super::output::{DoseTsvWriter, LcwgsVcfWriter};
use super::LcwgsParams;
use crate::srp::multi_chr_reader::MultiChrSrpReader;
use crate::srp::SrpReader;

/// The panel: one multi-chromosome SRP, or a directory's single-chromosome SRPs.
pub enum Panels {
    Multi(String),
    Files(Vec<PathBuf>),
}

/// Where each chromosome's genetic map is.
pub enum Maps<'a> {
    /// One file: a map holding every chromosome (each run reads its own rows).
    File(&'a str),
    /// A directory of per-chromosome maps (`genmap::find_map_in_dir`).
    Dir(&'a Path),
}

/// The likelihood input.
pub enum Input<'a> {
    Pl(&'a str),
    Bam { paths: &'a [String], reference: Option<&'a str> },
}

fn bare(c: &str) -> &str { c.strip_prefix("chr").unwrap_or(c) }

/// The `.srp` files of a directory, in name order.
pub fn srp_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?.flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "srp")).collect();
    v.sort();
    if v.is_empty() {
        return Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("no .srp file in {}", dir.display())));
    }
    Ok(v)
}

/// The metadata of each single-chromosome SRP in `files`, refusing two that hold one chromosome.
pub fn file_chromosomes(files: &[PathBuf]) -> std::io::Result<Vec<(crate::srp::SrpMetadata, PathBuf)>> {
    let mut out: Vec<(crate::srp::SrpMetadata, PathBuf)> = Vec::new();
    for f in files {
        let meta = SrpReader::read_metadata(f)?;
        if out.iter().any(|(m, _)| bare(&m.chromosome) == bare(&meta.chromosome)) {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!(
                "two panels in the directory hold chromosome {} ({} among them)", meta.chromosome, f.display())));
        }
        out.push((meta, f.clone()));
    }
    Ok(out)
}

/// One chromosome of the panel: its name, and how to load it.
enum Source<'a> { View(&'a MultiChrSrpReader), File(PathBuf) }

#[allow(clippy::too_many_arguments)]
pub fn run(
    panels: Panels, input: Input, maps: Maps, params: &LcwgsParams, threads: usize,
    vcf_path: &Path, tsv_path: &Path,
    // `--header-line` lines for the VCF header.
    extra_header: &[String],
    // `--contig-length-from-input`: the input header's contig lengths (none when off).
    contig_lengths: &crate::io::contig_length::InputContigLengths,
) -> std::io::Result<()> {
    // The panel's chromosomes in its order, and the header's ##contig lines (each ID once).
    let multi = match &panels { Panels::Multi(p) => Some(MultiChrSrpReader::open(p)?), Panels::Files(_) => None };
    let mut chromosomes: Vec<(String, Source)> = Vec::new();
    let mut contig_lines: Vec<String> = Vec::new();
    match (&panels, &multi) {
        (Panels::Multi(_), Some(reader)) => {
            for c in reader.chromosomes() { chromosomes.push((c.to_string(), Source::View(reader))); }
            let lines = crate::io::bcf_encode::dedup_contig_lines(&reader.global_meta.contig_fields);
            contig_lines.extend(lines.lines().filter(|l| !l.trim().is_empty()).map(str::to_string));
        }
        (Panels::Files(files), _) => {
            for (meta, f) in file_chromosomes(files)? {
                let own = meta.contig_field.lines().find(|l| crate::io::bcf_encode::contig_names_from_header_lines(l)
                    .first().is_some_and(|id| bare(id) == bare(&meta.chromosome))).map(str::to_string);
                // Without the IDX= of the file it came from, as the panel merge writes it.
                contig_lines.push(own.map(|l| crate::srp::multi_chr_writer::strip_idx_from_contig(&l).to_string())
                    .unwrap_or_else(|| format!("##contig=<ID={}>", meta.chromosome)));
                chromosomes.push((meta.chromosome.clone(), Source::File(f)));
            }
        }
        _ => unreachable!("a multi-chromosome panel is opened above"),
    }
    let declared = crate::io::bcf_encode::contig_names_from_header_lines(&contig_lines.join("\n"));
    for (c, _) in &chromosomes {
        if !declared.iter().any(|d| bare(d) == bare(c)) { contig_lines.push(format!("##contig=<ID={c}>")); }
    }
    let contig_lines = crate::io::bcf_encode::dedup_contig_lines(&contig_lines.join("\n"));
    let contig_lines = contig_lengths.apply_logged(&contig_lines, |_| None).into_owned();

    // The chromosomes the input covers; the others are left out without a run.
    let covered = match &input {
        Input::Pl(p) => super::pl_reader::covered_contigs(p)?,
        Input::Bam { paths, .. } => super::bam_pileup::covered_contigs(paths)?,
    };
    crate::selphi_step!("lcWGS on a multi-chromosome panel: {} chromosome(s), {} covered by the input, one run each",
        chromosomes.len(), chromosomes.iter().filter(|(c, _)| covered.contains(bare(c))).count());
    let mut writers: Option<(LcwgsVcfWriter, DoseTsvWriter, Vec<String>)> = None;
    let (mut done, mut skipped): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for (chr, source) in &chromosomes {
        if !covered.contains(bare(chr)) { continue; }
        if crate::contig::nonrecomb_refusal(chr).is_some() {
            skipped.push(format!("{chr} (non-recombining)"));
            continue;
        }
        // A covered chromosome needs its map, as the genotype path requires.
        let map = match &maps {
            Maps::File(f) => f.to_string(),
            Maps::Dir(d) => crate::genmap::find_map_in_dir(d, bare(chr))
                .map(|p| p.to_string_lossy().into_owned())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("no genetic map for chromosome {chr} in {} (the input covers it)", d.display())))?,
        };
        crate::selphi_step!("lcWGS chromosome {}", chr);
        let srp = match source {
            Source::View(reader) => reader.load_chr_view(chr)?.into_srp_reader(),
            Source::File(f) => SrpReader::open(f, 0)?,
        };
        let res = match &input {
            Input::Pl(p) => super::pipeline::run_lcwgs(p, &srp, &map, params, threads),
            Input::Bam { paths, reference } => super::pipeline::run_lcwgs_bam(paths, &srp, &map, params, None, *reference, threads),
        };
        let out = match res {
            Ok(o) => o,
            Err(e) if super::pipeline::is_no_data(&e) => {
                crate::selphi_step!("  {}: nothing to impute from ({}): left out", chr, e);
                skipped.push(format!("{chr} (no data)"));
                continue;
            }
            // Already reported: passed on as it is, for the caller's exit status.
            Err(e) if crate::Fatal::of(&e).is_some() => return Err(e),
            Err(e) => return Err(std::io::Error::new(e.kind(), format!("chromosome {chr}: {e}"))),
        };
        if writers.is_none() {
            writers = Some((LcwgsVcfWriter::create(vcf_path, &out.sample_ids, &contig_lines, extra_header)?,
                DoseTsvWriter::create(tsv_path, &out.sample_ids)?, out.sample_ids.clone()));
        }
        let (v, d, ids) = writers.as_mut().expect("writers created above");
        if *ids != out.sample_ids {
            return Err(std::io::Error::other(format!("chromosome {chr}: the samples differ from the first chromosome's")));
        }
        v.append(&out, &srp)?;
        d.append(&out, &srp)?;
        done.push(chr.clone());
        drop(out);
        drop(srp);
        crate::common::mem::release_freed_memory();
    }
    let Some((v, d, _)) = writers else {
        return Err(std::io::Error::other("no chromosome of the panel carries the input's likelihoods"));
    };
    v.finish()?;
    d.finish()?;
    crate::selphi_step!("lcWGS: {} chromosome(s) imputed ({}){}", done.len(), done.join(", "),
        if skipped.is_empty() { String::new() } else { format!("; left out: {}", skipped.join(", ")) });
    Ok(())
}
