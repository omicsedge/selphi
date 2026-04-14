//! Native PLINK2 PGEN/PVAR/PSAM output.
//!
//! Mode 0x03: fixed-width records, unphased dosage (vrtype 0x40).
//! Each variant: ceil(N/4) bytes hardcall + N*2 bytes dosage.
//! Dosage: uint16 LE, 0=0.0, 16384=1.0, 32768=2.0, 65535=missing.
//!
//! Spec: github.com/chrchang/plink-ng (pgen_spec.pdf, pgenlib_misc.h)

use std::io::{self, Write, BufWriter, Seek, SeekFrom};
use std::path::Path;

/// Write .psam file (sample metadata).
pub fn write_psam(path: &Path, sample_names: &[String]) -> io::Result<()> {
    let psam_path = path.with_extension("psam");
    let mut f = BufWriter::new(std::fs::File::create(&psam_path)?);
    writeln!(f, "#IID")?;
    for name in sample_names { writeln!(f, "{}", name)?; }
    f.flush()?;
    Ok(())
}

/// Write .pvar file header. Returns the writer for appending variant lines.
pub fn write_pvar(path: &Path) -> io::Result<BufWriter<std::fs::File>> {
    let pvar_path = path.with_extension("pvar");
    let mut f = BufWriter::new(std::fs::File::create(&pvar_path)?);
    writeln!(f, "#CHROM\tPOS\tID\tREF\tALT")?;
    Ok(f)
}

/// Write a variant line to the .pvar file.
#[inline]
pub fn write_pvar_variant(
    pvar: &mut BufWriter<std::fs::File>,
    chrom: &str, pos: &str, id: &str, ref_a: &str, alt_a: &str,
) -> io::Result<()> {
    writeln!(pvar, "{}\t{}\t{}\t{}\t{}", chrom, pos, id, ref_a, alt_a)
}

/// PGEN writer — mode 0x03 (fixed-width, unphased dosage for all samples).
///
/// Binary layout per variant:
///   [ceil(N/4) bytes] 2-bit packed hardcalls (00=hom_ref, 01=het, 10=hom_alt, 11=missing)
///   [N*2 bytes]       uint16 LE dosage per sample (0..32768 = 0.0..2.0, 65535=missing)
///
/// Header (12 bytes):
///   [2B] magic 0x6c 0x1b
///   [1B] mode 0x03 (fixed-width, unphased dosage)
///   [4B] variant_ct (LE, placeholder patched at finish)
///   [4B] sample_ct (LE)
///   [1B] control byte: 0x80 (all-ref-trusted, no explicit nonref flags)
pub struct PgenWriter {
    file: BufWriter<std::fs::File>,
    n_samples: usize,
    n_variants: u32,
    bytes_per_hardcall: usize,
}

impl PgenWriter {
    pub fn new(path: &Path, n_samples: usize) -> io::Result<Self> {
        let pgen_path = path.with_extension("pgen");
        let mut file = BufWriter::new(std::fs::File::create(&pgen_path)?);

        let bytes_per_hardcall = n_samples.div_ceil(4);

        // Header: 12 bytes
        file.write_all(&[0x6c, 0x1b])?;                          // magic
        file.write_all(&[0x03])?;                                  // mode: fixed-width unphased dosage
        file.write_all(&(0u32).to_le_bytes())?;                   // variant_ct placeholder
        file.write_all(&(n_samples as u32).to_le_bytes())?;       // sample_ct
        file.write_all(&[0x80])?;                                  // control: all-ref-trusted

        Ok(PgenWriter { file, n_samples, n_variants: 0, bytes_per_hardcall })
    }

    /// Write one variant: 2-bit hardcalls + 16-bit dosage per sample.
    ///
    /// `hardcalls`: 0 (hom_ref), 1 (het), 2 (hom_alt) per sample.
    /// `dosages`: f32 dosage per sample (0.0–2.0).
    pub fn write_variant(&mut self, hardcalls: &[u8], dosages: &[f32]) -> io::Result<()> {
        debug_assert_eq!(hardcalls.len(), self.n_samples);
        debug_assert_eq!(dosages.len(), self.n_samples);

        // 1. Pack 2-bit hardcalls: 4 samples per byte, LSB first
        let mut packed = vec![0u8; self.bytes_per_hardcall];
        for (i, &g) in hardcalls.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            // Encoding: 00=hom_ref, 01=het, 10=hom_alt, 11=missing
            let val = g.min(3);
            packed[byte_idx] |= val << bit_offset;
        }
        self.file.write_all(&packed)?;

        // 2. 16-bit dosage per sample
        // Scale: 0 = 0.0, 16384 = 1.0, 32768 = 2.0
        // 65535 = missing (not used for imputed data)
        for i in 0..self.n_samples {
            let d = dosages[i].clamp(0.0, 2.0);
            let d16 = (d * 16384.0).round() as u32;
            let d16 = d16.min(32768) as u16;
            self.file.write_all(&d16.to_le_bytes())?;
        }

        self.n_variants += 1;
        Ok(())
    }

    /// Finalize: patch variant count in header and flush.
    pub fn finish(mut self) -> io::Result<()> {
        self.file.flush()?;

        // Patch variant_ct at offset 3
        let inner = self.file.into_inner().map_err(|e| io::Error::other(e.to_string()))?;
        let mut file = inner;
        file.seek(SeekFrom::Start(3))?;
        file.write_all(&self.n_variants.to_le_bytes())?;
        file.flush()?;
        Ok(())
    }
}
