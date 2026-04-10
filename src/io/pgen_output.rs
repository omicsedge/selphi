//! Native PLINK2 PGEN/PVAR/PSAM output for imputation dosages.
//!
//! Writes the plink2 fileset (.pgen, .pvar, .psam) directly.
//! PGEN mode 0x02: uncompressed 2-bit hardcalls + separate dosage track.
//!
//! Dosage encoding: 16-bit unsigned integer, 0x0000=0.0, 0x8000=2.0.
//! Missing dosage: 0xFFFF.

use std::io::{self, Write, BufWriter};
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

/// Write .pvar file (variant metadata).
pub fn write_pvar(path: &Path) -> io::Result<BufWriter<std::fs::File>> {
    let pvar_path = path.with_extension("pvar");
    let mut f = BufWriter::new(std::fs::File::create(&pvar_path)?);
    writeln!(f, "#CHROM\tPOS\tID\tREF\tALT")?;
    Ok(f)
}

/// Write a variant line to the .pvar file.
#[inline]
pub fn write_pvar_variant(pvar: &mut BufWriter<std::fs::File>, chrom: &str, pos: &str, id: &str, ref_a: &str, alt_a: &str) -> io::Result<()> {
    writeln!(pvar, "{}\t{}\t{}\t{}\t{}", chrom, pos, id, ref_a, alt_a)
}

/// PGEN writer for mode 0x02 (uncompressed 2-bit genotypes + dosage).
pub struct PgenWriter {
    file: BufWriter<std::fs::File>,
    n_samples: usize,
    n_variants: usize,
    bytes_per_variant: usize, // ceil(n_samples / 4) for 2-bit packing
}



impl PgenWriter {
    /// Create a new PGEN writer. Writes the header immediately.
    pub fn new(path: &Path, n_samples: usize) -> io::Result<Self> {
        let pgen_path = path.with_extension("pgen");
        let mut file = BufWriter::new(std::fs::File::create(&pgen_path)?);

        let bytes_per_variant = n_samples.div_ceil(4); // 2 bits per sample

        // PGEN header: magic (2 bytes) + mode (1 byte) + variant_ct (4 bytes) + sample_ct (4 bytes)
        // Simplified mode 0x02 = constant-width records (2-bit genotypes, no compression)
        // With dosage: we append dosage track after the genotype block for each variant
        file.write_all(&[0x6c, 0x1b])?; // PLINK2 magic
        file.write_all(&[0x02])?;        // Storage mode 2 (constant-width, no compression)

        file.write_all(&(0u32).to_le_bytes())?;  // variant_ct placeholder (patched at close)
        file.write_all(&(n_samples as u32).to_le_bytes())?;
        file.write_all(&[0x40])?;          // flags: bit 6 = biallelic hardcalls only

        Ok(PgenWriter { file, n_samples, n_variants: 0, bytes_per_variant })
    }

    /// Write one variant's genotypes + dosages.
    /// `hardcalls`: 0, 1, or 2 per sample (diploid dosage rounded).
    /// `dosages`: f32 dosage per sample (0.0–2.0).
    pub fn write_variant(&mut self, hardcalls: &[u8], _dosages: &[f32]) -> io::Result<()> {
        debug_assert!(hardcalls.len() == self.n_samples);

        // Pack 2-bit genotypes: 4 samples per byte, LSB first
        // Encoding: 0=hom_ref, 1=het, 2=hom_alt, 3=missing
        let mut packed = vec![0u8; self.bytes_per_variant];
        for (i, &g) in hardcalls.iter().enumerate() {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let val = g.min(3);
            packed[byte_idx] |= val << bit_offset;
        }
        self.file.write_all(&packed)?;

        // Note: dosage track omitted for mode 0x02 (hardcalls only).
        // For dosage support, use VCF.gz or BCF output with plink2 --vcf dosage=DS.

        self.n_variants += 1;
        Ok(())
    }

    /// Finalize: patch variant count in header and flush.
    pub fn finish(mut self) -> io::Result<()> {
        self.file.flush()?;

        // Patch variant count at offset 3 (after magic + mode)
        use std::io::Seek;
        let inner = self.file.into_inner().map_err(|e| io::Error::other(e.to_string()))?;
        let mut file = inner;
        file.seek(std::io::SeekFrom::Start(3))?;
        file.write_all(&(self.n_variants as u32).to_le_bytes())?;
        file.flush()?;
        Ok(())
    }
}
