//! BCF/VCF.gz output writer using noodles-bgzf (no subprocess).
//!
//! Writes VCF text directly to a multi-threaded BGZF compressor,
//! eliminating the bgzip subprocess. Supports BCF conversion via
//! post-processing with bcftools.

use std::io::{self, Write, BufWriter};
use std::path::{Path, PathBuf};
use crate::selphi_info;

type BgzfInner = noodles_bgzf::io::multithreaded_writer::MultithreadedWriter<std::fs::File>;

/// Direct BGZF writer — VCF text records go straight to compressed output.
pub struct BcfWriter {
    writer: BufWriter<BgzfInner>,
    path: PathBuf,
    convert_to_bcf: bool,
}

pub fn setup(
    n_samples: usize,
    sample_names: &[String],
    contig_field: &str,
    version: &str,
    output_path: &Path,
) -> io::Result<BcfWriter> {
    // Always write VCF.gz first, convert to BCF at the end
    let vcf_gz_path = output_path.with_extension("vcf.gz");

    let file = std::fs::File::create(&vcf_gz_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut writer = BufWriter::with_capacity(4 << 20, bgzf);

    // Write VCF header
    write!(writer, "##fileformat=VCFv4.2\n")?;
    write!(writer, "##source=Selphi_v{version} SelfDecode™\n")?;
    write!(writer, "##FILTER=<ID=PASS,Description=\"All filters passed\">\n")?;
    write!(writer, "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">\n")?;
    write!(writer, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">\n")?;
    write!(writer, "##INFO=<ID=AN,Number=1,Type=Integer,Description=\"Allele Number\">\n")?;
    write!(writer, "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Estimated Allele Count\">\n")?;
    write!(writer, "##INFO=<ID=DR2,Number=1,Type=Float,Description=\"Dosage R-squared: estimated imputation accuracy\">\n")?;
    write!(writer, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")?;
    write!(writer, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"estimated ALT dose\">\n")?;
    write!(writer, "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"estimated ALT dose on first haplotype\">\n")?;
    write!(writer, "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"estimated ALT dose on second haplotype\">\n")?;
    write!(writer, "{}\n", contig_field)?;
    write!(writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for name in sample_names { write!(writer, "\t{}", name)?; }
    write!(writer, "\n")?;

    selphi_info!("  BCF output: {} ({} samples)", vcf_gz_path.display(), n_samples);
    Ok(BcfWriter { writer, path: vcf_gz_path, convert_to_bcf: true })
}

impl BcfWriter {
    pub fn path(&self) -> &Path { &self.path }

    /// Write a block of VCF text lines directly to BGZF.
    #[inline]
    pub fn write_vcf_lines(&mut self, data: &[u8]) -> io::Result<()> {
        self.writer.write_all(data)
    }

    /// Finish: flush, convert VCF.gz → BCF with bcftools, index.
    pub fn finish(self) -> io::Result<PathBuf> {
        let vcf_path = self.path.clone();
        let convert = self.convert_to_bcf;

        // Flush and close
        let bgzf = self.writer.into_inner().map_err(|e| io::Error::other(e.to_string()))?;
        drop(bgzf); // MultithreadedWriter flushes on drop

        if convert {
            let bcf_path = vcf_path.with_extension("").with_extension("bcf");
            selphi_info!("  Converting VCF.gz → BCF...");
            let status = std::process::Command::new("bcftools")
                .args(["view", "-Ob", "--threads", "8",
                       "-o", bcf_path.to_str().unwrap(),
                       vcf_path.to_str().unwrap()])
                .status()?;
            if status.success() {
                let _ = std::fs::remove_file(&vcf_path);
                let _ = std::process::Command::new("bcftools")
                    .args(["index", "--threads", "4", bcf_path.to_str().unwrap()])
                    .status();
                return Ok(bcf_path);
            }
            selphi_info!("  bcftools conversion failed, keeping VCF.gz");
        }

        let _ = std::process::Command::new("bcftools")
            .args(["index", vcf_path.to_str().unwrap()])
            .status();
        Ok(vcf_path)
    }
}
