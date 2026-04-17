//! Target-Augmented Dynamic Panel (TADP) scaffold: append-only haplotype
//! store at chip positions only.
//!
//! File layout (little-endian throughout):
//!   magic: b"SCFF\x00\x01\x00\x00"          — 8 bytes
//!   header_len: u32                           — 4 bytes (length of zstd blob)
//!   header_zstd: [u8; header_len]             — zstd-compressed JSON with
//!                                                chromosome, n_chip_vars, n_words
//!   data_offset = 8 + 4 + header_len
//!   body: [hap 0 bits][hap 1 bits]...         — each hap = n_words × u64,
//!                                                bit j of variant v packed at
//!                                                word_index = v / 64, bit = v % 64
//!
//! The number of appended haps is inferred from file size, not stored in the
//! header — this makes append atomic (single write + fsync) without header
//! rewrites.

use std::fs::{File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde_json::json;

use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"SCFF\x00\x01\x00\x00";

#[derive(Debug, Clone)]
struct Header {
    chromosome: String,
    n_chip_vars: usize,
    n_words: usize,
    data_offset: u64,
}

impl Header {
    fn read<R: Read + Seek>(r: &mut R) -> io::Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not a selphi scaffold file (bad magic)",
            ));
        }
        let mut len_buf = [0u8; 4];
        r.read_exact(&mut len_buf)?;
        let hdr_len = u32::from_le_bytes(len_buf) as usize;
        let mut comp = vec![0u8; hdr_len];
        r.read_exact(&mut comp)?;
        let raw = zstd::decode_all(Cursor::new(&comp))
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let j: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let chromosome = j["chromosome"].as_str().unwrap_or("").to_string();
        let n_chip_vars = j["n_chip_vars"].as_u64().unwrap_or(0) as usize;
        let n_words = j["n_words"].as_u64().unwrap_or(0) as usize;
        let data_offset = r.stream_position()?;
        Ok(Self { chromosome, n_chip_vars, n_words, data_offset })
    }

    fn write<W: Write>(w: &mut W, chromosome: &str, n_chip_vars: usize) -> io::Result<u64> {
        let n_words = n_chip_vars.div_ceil(64);
        let body = json!({
            "chromosome": chromosome,
            "n_chip_vars": n_chip_vars,
            "n_words": n_words,
            "version": 1,
        })
        .to_string();
        let comp = zstd::encode_all(Cursor::new(body.as_bytes()), 3)
            .map_err(io::Error::other)?;
        w.write_all(MAGIC)?;
        w.write_all(&(comp.len() as u32).to_le_bytes())?;
        w.write_all(&comp)?;
        Ok((8 + 4 + comp.len()) as u64)
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Append-only writer. New haps are packed row-major into the body.
pub struct ScaffoldWriter {
    path: PathBuf,
    n_chip_vars: usize,
    n_words: usize,
    n_haps: usize,
    f: File,
}

impl ScaffoldWriter {
    /// Create a new scaffold file. Errors if the file already exists.
    pub fn create(path: &Path, chromosome: &str, n_chip_vars: usize) -> io::Result<Self> {
        let mut f = OpenOptions::new()
            .read(true).write(true).create_new(true)
            .open(path)?;
        let data_offset = Header::write(&mut f, chromosome, n_chip_vars)?;
        let _ = data_offset;
        Ok(Self {
            path: path.to_path_buf(),
            n_chip_vars,
            n_words: n_chip_vars.div_ceil(64),
            n_haps: 0,
            f,
        })
    }

    /// Open an existing scaffold for appending. Infers n_haps from file size.
    pub fn open_append(path: &Path) -> io::Result<Self> {
        let mut f = OpenOptions::new().read(true).write(true).open(path)?;
        let hdr = Header::read(&mut f)?;
        let file_len = f.metadata()?.len();
        let body_bytes = file_len.saturating_sub(hdr.data_offset);
        let bytes_per_hap = (hdr.n_words * 8) as u64;
        if bytes_per_hap == 0 || body_bytes % bytes_per_hap != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("scaffold body not aligned: {} bytes, {} per hap",
                        body_bytes, bytes_per_hap),
            ));
        }
        let n_haps = (body_bytes / bytes_per_hap) as usize;
        f.seek(SeekFrom::End(0))?;
        Ok(Self {
            path: path.to_path_buf(),
            n_chip_vars: hdr.n_chip_vars,
            n_words: hdr.n_words,
            n_haps,
            f,
        })
    }

    /// Append one hap, given its packed bits (`n_words` u64s).
    pub fn append_hap_bits(&mut self, bits: &[u64]) -> io::Result<()> {
        if bits.len() != self.n_words {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} words, got {}", self.n_words, bits.len()),
            ));
        }
        // Safe reinterpretation: u64 LE bytes on disk.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 8)
        };
        self.f.write_all(bytes)?;
        self.n_haps += 1;
        Ok(())
    }

    /// Flush buffered writes to disk. Call between batches.
    pub fn flush(&mut self) -> io::Result<()> {
        self.f.flush()
    }

    pub fn n_haps(&self) -> usize { self.n_haps }
    pub fn n_chip_vars(&self) -> usize { self.n_chip_vars }
    pub fn path(&self) -> &Path { &self.path }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// mmap-backed reader. Zero-copy access to hap bits.
pub struct ScaffoldReader {
    _mmap: Mmap,
    chromosome: String,
    n_chip_vars: usize,
    n_words: usize,
    n_haps: usize,
    data_offset: usize,
    // raw pointer into mmap's body region (u64-aligned)
    body_ptr: *const u64,
}

// Safety: ScaffoldReader holds an Mmap and a pointer derived from it. Mmap is
// Send+Sync; the pointer is read-only and tied to `_mmap`'s lifetime.
unsafe impl Send for ScaffoldReader {}
unsafe impl Sync for ScaffoldReader {}

impl ScaffoldReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = File::open(path)?;
        let hdr = Header::read(&mut f)?;
        let mmap = unsafe { Mmap::map(&f)? };
        let data_offset = hdr.data_offset as usize;
        let body_bytes = mmap.len().saturating_sub(data_offset);
        let bytes_per_hap = hdr.n_words * 8;
        if bytes_per_hap == 0 || body_bytes % bytes_per_hap != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("scaffold body not aligned: {} bytes, {} per hap",
                        body_bytes, bytes_per_hap),
            ));
        }
        let n_haps = body_bytes / bytes_per_hap;
        // SAFETY: we checked alignment — data_offset points inside the mmap
        // and body_bytes is a multiple of 8 (n_words × 8).
        let body_ptr = unsafe {
            mmap.as_ptr().add(data_offset) as *const u64
        };
        Ok(Self {
            _mmap: mmap,
            chromosome: hdr.chromosome,
            n_chip_vars: hdr.n_chip_vars,
            n_words: hdr.n_words,
            n_haps,
            data_offset,
            body_ptr,
        })
    }

    pub fn chromosome(&self) -> &str { &self.chromosome }
    pub fn n_chip_vars(&self) -> usize { self.n_chip_vars }
    pub fn n_words(&self) -> usize { self.n_words }
    pub fn n_haps(&self) -> usize { self.n_haps }

    /// Read-only slice of packed bits for hap `h`. O(1).
    #[inline]
    pub fn hap_bits(&self, h: usize) -> &[u64] {
        assert!(h < self.n_haps, "hap {} out of range ({})", h, self.n_haps);
        unsafe {
            let base = self.body_ptr.add(h * self.n_words);
            std::slice::from_raw_parts(base, self.n_words)
        }
    }

    /// Allele (0 or 1) for hap `h` at variant `v`.
    #[inline]
    pub fn get(&self, h: usize, v: usize) -> u8 {
        debug_assert!(v < self.n_chip_vars);
        let bits = self.hap_bits(h);
        ((bits[v / 64] >> (v % 64)) & 1) as u8
    }

    /// File offset where the body starts (useful for debugging).
    pub fn data_offset(&self) -> usize { self.data_offset }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrip() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        drop(tmp); // remove the file so create() doesn't see it

        let n_vars = 150; // across 3 words (last partially filled)
        {
            let mut w = ScaffoldWriter::create(&path, "20", n_vars).unwrap();
            let n_words = n_vars.div_ceil(64);

            // hap 0: alternating bits at positions 0, 2, 4, ...
            let mut h0 = vec![0u64; n_words];
            for v in (0..n_vars).step_by(2) { h0[v / 64] |= 1u64 << (v % 64); }
            w.append_hap_bits(&h0).unwrap();

            // hap 1: all 1s (first n_vars bits)
            let mut h1 = vec![0u64; n_words];
            for v in 0..n_vars { h1[v / 64] |= 1u64 << (v % 64); }
            w.append_hap_bits(&h1).unwrap();

            w.flush().unwrap();
            assert_eq!(w.n_haps(), 2);
        }

        // reopen + append a third hap
        {
            let mut w = ScaffoldWriter::open_append(&path).unwrap();
            assert_eq!(w.n_haps(), 2);
            assert_eq!(w.n_chip_vars(), n_vars);

            let mut h2 = vec![0u64; n_vars.div_ceil(64)];
            h2[0] = 0x01; // only variant 0
            w.append_hap_bits(&h2).unwrap();
            w.flush().unwrap();
        }

        // read back
        let r = ScaffoldReader::open(&path).unwrap();
        assert_eq!(r.n_haps(), 3);
        assert_eq!(r.n_chip_vars(), n_vars);
        assert_eq!(r.chromosome(), "20");
        // hap 0: even positions → 1
        for v in 0..n_vars {
            assert_eq!(r.get(0, v), if v % 2 == 0 { 1 } else { 0 });
        }
        // hap 1: all 1
        for v in 0..n_vars { assert_eq!(r.get(1, v), 1); }
        // hap 2: only v=0 is 1
        for v in 0..n_vars {
            assert_eq!(r.get(2, v), if v == 0 { 1 } else { 0 });
        }

        let _ = std::fs::remove_file(&path);
    }
}
