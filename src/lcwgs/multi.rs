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

/// The panel's chromosomes in its order, each with how to load it and where its map is: opened
/// once per run (the multi-chromosome lcWGS runs and `--calls` share it).
pub struct PanelSet<'a> {
    multi: Option<MultiChrSrpReader>,
    /// Each chromosome's name, with its file when the panel is a directory's SRPs (else a view of
    /// `multi`).
    chromosomes: Vec<(String, Option<PathBuf>)>,
    /// One `##contig` line per chromosome (each ID once), as the panel declares them.
    contig_lines: String,
    maps: Maps<'a>,
}

impl<'a> PanelSet<'a> {
    pub fn open(panels: Panels, maps: Maps<'a>) -> std::io::Result<Self> {
        // The panel's chromosomes in its order, and the header's ##contig lines (each ID once).
        let mut chromosomes: Vec<(String, Option<PathBuf>)> = Vec::new();
        let mut contig_lines: Vec<String> = Vec::new();
        let multi = match &panels {
            Panels::Multi(p) => {
                let reader = MultiChrSrpReader::open(p)?;
                for c in reader.chromosomes() { chromosomes.push((c.to_string(), None)); }
                let lines = crate::io::bcf_encode::dedup_contig_lines(&reader.global_meta.contig_fields);
                contig_lines.extend(lines.lines().filter(|l| !l.trim().is_empty()).map(str::to_string));
                Some(reader)
            }
            Panels::Files(files) => {
                for (meta, f) in file_chromosomes(files)? {
                    let own = meta.contig_field.lines().find(|l| crate::io::bcf_encode::contig_names_from_header_lines(l)
                        .first().is_some_and(|id| bare(id) == bare(&meta.chromosome))).map(str::to_string);
                    // Without the IDX= of the file it came from, as the panel merge writes it.
                    contig_lines.push(own.map(|l| crate::srp::multi_chr_writer::strip_idx_from_contig(&l).to_string())
                        .unwrap_or_else(|| format!("##contig=<ID={}>", meta.chromosome)));
                    chromosomes.push((meta.chromosome.clone(), Some(f)));
                }
                None
            }
        };
        let declared = crate::io::bcf_encode::contig_names_from_header_lines(&contig_lines.join("\n"));
        for (c, _) in &chromosomes {
            if !declared.iter().any(|d| bare(d) == bare(c)) { contig_lines.push(format!("##contig=<ID={c}>")); }
        }
        let contig_lines = crate::io::bcf_encode::dedup_contig_lines(&contig_lines.join("\n"));
        Ok(PanelSet { multi, chromosomes, contig_lines, maps })
    }

    /// The chromosomes, in the panel's order.
    pub fn chromosomes(&self) -> impl Iterator<Item = &str> { self.chromosomes.iter().map(|(c, _)| c.as_str()) }

    /// The `##contig` lines of every chromosome.
    pub fn contig_lines(&self) -> &str { &self.contig_lines }

    /// The panel chromosome a contig name means (the same name, or one that differs from it by a
    /// `chr` prefix). `Err` when two do.
    pub fn find(&self, contig: &str) -> Result<Option<&str>, String> {
        let mut hits = self.chromosomes.iter().filter(|(c, _)| crate::io::contig_length::same_contig(c, contig));
        match (hits.next(), hits.next()) {
            (Some((c, _)), None) => Ok(Some(c)),
            (None, _) => Ok(None),
            (Some((a, _)), Some((b, _))) => Err(format!("contig {contig} matches two panel chromosomes, {a} and {b}")),
        }
    }

    /// The genetic map of chromosome `chr` (one map file holds them all, or a directory's file).
    pub fn map(&self, chr: &str) -> std::io::Result<String> {
        match &self.maps {
            Maps::File(f) => Ok(f.to_string()),
            Maps::Dir(d) => crate::genmap::find_map_in_dir(d, crate::io::contig_length::bare(chr))
                .map(|p| p.to_string_lossy().into_owned())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound,
                    format!("no genetic map for chromosome {chr} in {} (the input covers it)", d.display()))),
        }
    }

    /// Load chromosome `chr` (a name [`Self::chromosomes`] gives).
    pub fn load(&self, chr: &str) -> std::io::Result<SrpReader> {
        match (self.chromosomes.iter().find(|(c, _)| c == chr), &self.multi) {
            (Some((_, Some(f))), _) => SrpReader::open(f, 0),
            (Some((_, None)), Some(reader)) => Ok(reader.load_chr_view(chr)?.into_srp_reader()),
            _ => Err(std::io::Error::new(std::io::ErrorKind::NotFound, format!("the panel has no chromosome {chr}"))),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    panels: Panels, input: Input, maps: Maps, params: &LcwgsParams, threads: usize,
    vcf_path: &Path, tsv_path: &Path,
    // `--header-line` lines for the VCF header.
    extra_header: &[String],
    // `--contig-length-from-input`: the input header's contig lengths (none when off).
    contig_lengths: &crate::io::contig_length::InputContigLengths,
) -> std::io::Result<()> {
    let set = PanelSet::open(panels, maps)?;
    let contig_lines = contig_lengths.apply_logged(set.contig_lines(), |_| None).into_owned();
    let chromosomes: Vec<&str> = set.chromosomes().collect();

    // The chromosomes the input covers; the others are left out without a run.
    let covered = match &input {
        Input::Pl(p) => super::pl_reader::covered_contigs(p)?,
        Input::Bam { paths, .. } => super::bam_pileup::covered_contigs(paths)?,
    };
    crate::selphi_step!("lcWGS on a multi-chromosome panel: {} chromosome(s), {} covered by the input, one run each",
        chromosomes.len(), chromosomes.iter().filter(|c| covered.contains(bare(c))).count());
    let mut writers: Option<(LcwgsVcfWriter, DoseTsvWriter, Vec<String>)> = None;
    let (mut done, mut skipped): (Vec<String>, Vec<String>) = (Vec::new(), Vec::new());
    for &chr in &chromosomes {
        if !covered.contains(bare(chr)) { continue; }
        if crate::contig::nonrecomb_refusal(chr).is_some() {
            skipped.push(format!("{chr} (non-recombining)"));
            continue;
        }
        // A covered chromosome needs its map, as the genotype path requires.
        let map = set.map(chr)?;
        crate::selphi_step!("lcWGS chromosome {}", chr);
        let srp = set.load(chr)?;
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
        done.push(chr.to_string());
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
