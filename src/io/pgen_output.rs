//! Native PLINK2 PGEN/PVAR/PSAM output.
//!
//! Mode 0x10 (variable-type records, as plink2 writes them), every record vrtype
//! 0x40: ceil(N/4) bytes 2-bit hardcall + N*2 bytes unconditional dosage.
//! Dosage: uint16 LE, 0=0.0, 16384=1.0, 32768=2.0, 65535=missing.
//!
//! Spec: github.com/chrchang/plink-ng (pgen_spec.pdf, pgenlib_misc.h)
//!
//! HARDCALL CONVENTION (intentional — differs from VCF/BCF/SelfDecode):
//! PGEN derives its 2-bit hardcall by ROUNDING THE SUMMED DOSAGE
//! (`ds = ap1 + ap2`; `2` if `ds > 1.5`, `1` if `ds > 0.5`, else `0`) — the
//! standard PLINK dosage→hardcall convention, internally consistent with the
//! 16-bit dosage it stores alongside. VCF/BCF/SelfDecode instead take a
//! PER-HAPLOTYPE argmax (`gt1 = ap1 > 0.5`, `gt2 = ap2 > 0.5`). The DOSAGE
//! fields are identical across all formats; only the hardcall/GT can differ
//! on borderline per-hap probabilities (e.g. `ap1 = ap2 = 0.4` → PGEN het
//! `1`, VCF `0|0`). Chip (hard-call) sites never diverge. This is a format
//! convention, not a bug: PGEN stays self-consistent for PLINK tooling.

use std::io::{self, Write, BufWriter, Seek, SeekFrom};
use std::path::Path;

/// Write .psam file (sample metadata).
pub fn write_psam(path: &Path, sample_names: &[String]) -> io::Result<()> {
    let psam_path = crate::common::utils::out_path(path, "psam");
    let mut f = BufWriter::new(std::fs::File::create(&psam_path)?);
    writeln!(f, "#IID")?;
    for name in sample_names { writeln!(f, "{}", name)?; }
    f.flush()?;
    Ok(())
}

/// Write .pvar file header. Returns the writer for appending variant lines.
pub fn write_pvar(path: &Path) -> io::Result<BufWriter<std::fs::File>> {
    let pvar_path = crate::common::utils::out_path(path, "pvar");
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

/// Variants per PGEN vblock (pgenlib `kPglVblockSize`).
const VBLOCK: usize = 65536;
/// Variant record type: 2-bit hardcalls stored plainly + an unconditional 16-bit
/// dosage track for every sample (pgenlib vrtype 0x40).
const VRTYPE_DOSAGE: u8 = 0x40;

/// Bytes of the mode-0x10 header for `n_var` variants whose record lengths take
/// `len_bytes` bytes each: magic+mode (3), M (4), N (4), control (1), one 8-byte
/// file offset per vblock, then per variant a 1-byte vrtype and its length.
fn header_len(n_var: usize, len_bytes: usize) -> u64 {
    (12 + 8 * n_var.div_ceil(VBLOCK) + n_var * (1 + len_bytes)) as u64
}

/// PGEN writer — mode 0x10 (variable-type records, the layout plink2 itself
/// writes), every record of vrtype 0x40: 2-bit hardcalls then an unconditional
/// 16-bit dosage per sample, so every record has the same length.
///
/// Mode 0x03 (fixed-width dosage), written until 2026-10-14, is a format plink2
/// reads single-threaded but does not validate and failed to read with threads
/// (`Failed to unpack variant #N`, at a varying N, on files whose every record
/// decodes correctly by hand).
///
/// Records are streamed after a header region reserved for `max_variants`; the
/// real header is written at `finish`, and the gap a smaller final count leaves
/// between header and first record is allowed by the format (pgenlib reads
/// records from the stored vblock offsets).
pub struct PgenWriter {
    file: BufWriter<std::fs::File>,
    n_samples: usize,
    n_variants: u32,
    max_variants: usize,
    bytes_per_hardcall: usize,
    /// Record length in bytes and the bytes that store it in the header.
    rec_len: usize,
    len_bytes: usize,
    /// File offset of the first record.
    data_start: u64,
    rec: Vec<u8>,
}

impl PgenWriter {
    pub fn new(path: &Path, n_samples: usize, max_variants: usize) -> io::Result<Self> {
        let pgen_path = crate::common::utils::out_path(path, "pgen");
        let mut file = BufWriter::with_capacity(4 << 20, std::fs::File::create(&pgen_path)?);
        let bytes_per_hardcall = n_samples.div_ceil(4);
        let rec_len = bytes_per_hardcall + 2 * n_samples;
        let len_bytes = match rec_len { 0..=0xff => 1, 0x100..=0xffff => 2, 0x1_0000..=0xff_ffff => 3, _ => 4 };
        let data_start = header_len(max_variants, len_bytes);
        file.seek(SeekFrom::Start(data_start))?;
        Ok(PgenWriter {
            file, n_samples, n_variants: 0, max_variants, bytes_per_hardcall, rec_len, len_bytes,
            data_start, rec: vec![0u8; rec_len],
        })
    }

    /// Write one variant: 2-bit hardcalls + 16-bit dosage per sample.
    ///
    /// `hardcalls`: 0 (hom_ref), 1 (het), 2 (hom_alt), 3 (missing) per sample.
    /// `dosages`: f32 dosage per sample (0.0–2.0; NaN = missing).
    pub fn write_variant(&mut self, hardcalls: &[u8], dosages: &[f32]) -> io::Result<()> {
        debug_assert_eq!(hardcalls.len(), self.n_samples);
        debug_assert_eq!(dosages.len(), self.n_samples);
        if self.n_variants as usize >= self.max_variants {
            return Err(io::Error::other(format!(
                "PGEN: more variants than the {} the header was reserved for", self.max_variants)));
        }
        let (hc, ds) = self.rec.split_at_mut(self.bytes_per_hardcall);
        // 1. 2-bit hardcalls, 4 samples per byte, LSB first: 00 hom ref, 01 het,
        //    10 hom alt, 11 missing.
        hc.fill(0);
        for (i, &g) in hardcalls.iter().enumerate() {
            hc[i / 4] |= g.min(3) << ((i % 4) * 2);
        }
        // 2. Dosage, u16 LE: 0 = 0.0, 16384 = 1.0, 32768 = 2.0, 65535 = missing
        //    (a no-call at a passthrough site; imputed data has none).
        for (i, &d) in dosages.iter().enumerate() {
            let v = if d.is_nan() { u16::MAX } else { ((d.clamp(0.0, 2.0) * 16384.0).round() as u32).min(32768) as u16 };
            ds[2 * i..2 * i + 2].copy_from_slice(&v.to_le_bytes());
        }
        self.file.write_all(&self.rec)?;
        self.n_variants += 1;
        Ok(())
    }

    /// Write one record from its already-encoded parts (the sample merger, which
    /// concatenates per-batch records): packed 2-bit hardcalls, then the u16 LE
    /// dosages.
    pub fn write_raw(&mut self, hardcalls_packed: &[u8], dosage_le: &[u8]) -> io::Result<()> {
        if hardcalls_packed.len() + dosage_le.len() != self.rec_len {
            return Err(io::Error::other("PGEN: record parts do not match the sample count"));
        }
        if self.n_variants as usize >= self.max_variants {
            return Err(io::Error::other(format!(
                "PGEN: more variants than the {} the header was reserved for", self.max_variants)));
        }
        self.file.write_all(hardcalls_packed)?;
        self.file.write_all(dosage_le)?;
        self.n_variants += 1;
        Ok(())
    }

    /// Finalize: write the header for the variants actually written and flush.
    pub fn finish(mut self) -> io::Result<()> {
        let n_var = self.n_variants as usize;
        let mut h = Vec::with_capacity(header_len(n_var, self.len_bytes) as usize);
        h.extend_from_slice(&[0x6c, 0x1b, 0x10]);
        h.extend_from_slice(&self.n_variants.to_le_bytes());
        h.extend_from_slice(&(self.n_samples as u32).to_le_bytes());
        // control: (record-length bytes - 1) | 4 (8-bit vrtypes: dosage present) |
        // 01 << 6 (every REF allele is the reference; no provisional REF flags).
        h.push((self.len_bytes as u8 - 1) | 4 | (1 << 6));
        for b in 0..n_var.div_ceil(VBLOCK) {
            let fpos = self.data_start + (b * VBLOCK * self.rec_len) as u64;
            h.extend_from_slice(&fpos.to_le_bytes());
        }
        let len_le = (self.rec_len as u32).to_le_bytes();
        for b in 0..n_var.div_ceil(VBLOCK) {
            let k = (n_var - b * VBLOCK).min(VBLOCK);
            h.extend(std::iter::repeat_n(VRTYPE_DOSAGE, k));
            for _ in 0..k { h.extend_from_slice(&len_le[..self.len_bytes]); }
        }
        debug_assert!(h.len() as u64 <= self.data_start);
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.flush()?;
        Ok(())
    }
}

/// Layout of a PGEN this module wrote (mode 0x10 with constant-length records, or
/// the earlier mode 0x03): `(n_variants, n_samples, first record offset, record length)`.
pub fn read_layout(path: &Path) -> io::Result<(usize, usize, u64, usize)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 12];
    f.read_exact(&mut hdr)?;
    if hdr[0..2] != [0x6c, 0x1b] {
        return Err(io::Error::other(format!("bad PGEN magic in {path:?}")));
    }
    let n_var = u32::from_le_bytes(hdr[3..7].try_into().unwrap()) as usize;
    let n_sam = u32::from_le_bytes(hdr[7..11].try_into().unwrap()) as usize;
    let rec_len = n_sam.div_ceil(4) + 2 * n_sam;
    match hdr[2] {
        0x03 => Ok((n_var, n_sam, 12, rec_len)),
        0x10 => {
            let mut fpos = [0u8; 8];
            if n_var == 0 { return Ok((0, n_sam, 12, rec_len)); }
            f.read_exact(&mut fpos)?;
            Ok((n_var, n_sam, u64::from_le_bytes(fpos), rec_len))
        }
        m => Err(io::Error::other(format!("PGEN mode {m:#x} in {path:?} is not one Selphi writes"))),
    }
}
#[cfg(test)]
mod layout_tests {
    use super::*;

    /// A file reserved for more variants than written keeps a valid mode-0x10 header:
    /// count, control byte, vblock offset pointing past the gap, and the records.
    #[test]
    fn header_reserved_for_more_variants_than_written() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("target")
            .join(format!("test_pgen_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t");
        let mut w = PgenWriter::new(&p, 3, 10).unwrap();
        w.write_variant(&[0, 1, 2], &[0.0, 1.0, 2.0]).unwrap();
        w.write_variant(&[3, 0, 1], &[f32::NAN, 0.25, 0.75]).unwrap();
        w.finish().unwrap();
        let bytes = std::fs::read(p.with_extension("pgen")).unwrap();
        assert_eq!(&bytes[0..3], &[0x6c, 0x1b, 0x10]);
        assert_eq!(u32::from_le_bytes(bytes[3..7].try_into().unwrap()), 2);
        assert_eq!(bytes[11], 0x44, "1-byte record lengths, dosage present, all REF");
        let (n, s, start, len) = read_layout(&p.with_extension("pgen")).unwrap();
        assert_eq!((n, s, len), (2, 3, 1 + 6));
        assert_eq!(start, header_len(10, 1));
        assert_eq!(&bytes[20..22], &[0x40, 0x40], "vrtypes");
        assert_eq!(&bytes[22..24], &[7, 7], "record lengths");
        let r1 = &bytes[start as usize + 7..start as usize + 14];
        assert_eq!(r1[0], 0b01_00_11, "hardcalls missing, hom ref, het");
        assert_eq!(&r1[1..3], &[0xff, 0xff], "a no-call's dosage is missing");
        assert_eq!(u16::from_le_bytes([r1[3], r1[4]]), 4096);
        std::fs::remove_dir_all(&dir).ok();
    }
}
