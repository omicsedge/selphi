//! BREF3 (Binary Reference Format v3) reader.
//!
//! Reads phased reference panel haplotypes from .bref3 files.

use std::io::{Read, BufReader};
use std::path::Path;

pub(crate) const MAGIC_NUMBER_V3: i32 = 2055763188;
pub(crate) const SEQ_CODED: u8 = 0;
pub(crate) const ALLELE_CODED: u8 = 1;

/// The orderings of the four bases, in lexicographic order. A BREF3 record for
/// an SNV stores no allele strings: its allele code holds `n_alleles - 1` in the
/// low two bits and, in the high six, the index into this table of an ordering
/// whose first `n_alleles` bases are its alleles, REF first. Shared by the
/// reader and the writer.
pub(crate) const SNV_ORDERS: [[u8; 4]; 24] = [
    *b"ACGT", *b"ACTG", *b"AGCT", *b"AGTC", *b"ATCG", *b"ATGC",
    *b"CAGT", *b"CATG", *b"CGAT", *b"CGTA", *b"CTAG", *b"CTGA",
    *b"GACT", *b"GATC", *b"GCAT", *b"GCTA", *b"GTAC", *b"GTCA",
    *b"TACG", *b"TAGC", *b"TCAG", *b"TCGA", *b"TGAC", *b"TGCA",
];

/// Alleles of an SNV record from its allele code: the first `n_alleles` bases of
/// its ordering in [`SNV_ORDERS`].
fn snv_alleles(allele_code: i8) -> Result<(&'static [u8], usize), String> {
    let n_alleles = 1 + (allele_code as u8 & 0b11) as usize;
    let order = (allele_code as u8 >> 2) as usize;
    match SNV_ORDERS.get(order) {
        Some(bases) => Ok((&bases[..n_alleles], n_alleles)),
        None => Err(format!("BREF3 SNV allele code {allele_code} names ordering {order} (of {})",
            SNV_ORDERS.len())),
    }
}

fn base_string(b: u8) -> String { (b as char).to_string() }

/// A variant record from bref3.
pub struct Bref3Variant {
    pub chrom: String,
    pub pos: i32,
    pub ref_allele: String,
    pub alt_alleles: Vec<String>,
    pub id: String,
    /// Allele for each haplotype (0 = ref, 1+ = alt).
    pub alleles: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Big-endian I/O helpers (BREF3 uses big-endian encoding)
// ---------------------------------------------------------------------------

fn read_i32<R: Read>(r: &mut R) -> Result<i32, String> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(|e| format!("read_i32: {}", e))?;
    Ok(i32::from_be_bytes(buf))
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(|e| format!("read_u16: {}", e))?;
    Ok(u16::from_be_bytes(buf))
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(|e| format!("read_u8: {}", e))?;
    Ok(buf[0])
}

/// Read a modified UTF-8 string: 2-byte big-endian length prefix, then UTF-8 bytes.
fn read_utf<R: Read>(r: &mut R) -> Result<String, String> {
    let len = read_u16(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(|e| format!("read_utf: {}", e))?;
    // Modified UTF-8 is mostly compatible with standard UTF-8 for ASCII
    String::from_utf8(buf).map_err(|e| format!("invalid UTF-8: {}", e))
}

/// Read a string array: int32 length, then length × UTF-8 strings.
fn read_string_array<R: Read>(r: &mut R) -> Result<Vec<String>, String> {
    let len = read_i32(r)?;
    if len < 0 { return Ok(Vec::new()); }
    // Cap the EAGER reservation against an untrusted on-disk length (a malformed
    // header could otherwise pre-allocate ~48 GB of String before reading a byte);
    // the bounded loop below grows it as read_utf actually yields elements / EOFs.
    let mut arr = Vec::with_capacity((len as usize).min(4096));
    for _ in 0..len {
        arr.push(read_utf(r)?);
    }
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Streaming BREF3 reader — yields one variant at a time
// ---------------------------------------------------------------------------

/// Streaming BREF3 reader that yields one variant at a time without loading
/// all variants into memory. Keeps only the current block's hap_to_seq mapping.
pub struct Bref3StreamReader<R: Read> {
    reader: R,
    pub sample_ids: Vec<String>,
    pub n_haps: usize,
    // Current block state
    block_chrom: String,
    block_n_seq: usize,
    block_hap_to_seq: Vec<u16>,
    block_remaining: i32,
    finished: bool,
}

impl<R: Read> Bref3StreamReader<R> {
    /// Open a BREF3 stream and read the header. Does NOT read any variant data.
    pub fn new(mut reader: R) -> Result<Self, String> {
        let magic = read_i32(&mut reader)?;
        if magic != MAGIC_NUMBER_V3 {
            return Err(format!("Invalid bref3 magic number: {} (expected {})", magic, MAGIC_NUMBER_V3));
        }

        let _program = read_utf(&mut reader)?;
        let sample_ids = read_string_array(&mut reader)?;
        let n_haps = sample_ids.len() * 2;

        Ok(Self {
            reader, sample_ids, n_haps,
            block_chrom: String::new(),
            block_n_seq: 0,
            block_hap_to_seq: Vec::new(),
            block_remaining: 0,
            finished: false,
        })
    }

    /// Read the next variant. Returns None when all blocks are exhausted.
    pub fn next_variant(&mut self) -> Result<Option<Bref3Variant>, String> {
        if self.finished { return Ok(None); }

        // If current block is exhausted, read next block header
        while self.block_remaining == 0 {
            let n_recs = read_i32(&mut self.reader)?;
            if n_recs == 0 {
                self.finished = true;
                return Ok(None);
            }
            self.block_chrom = read_utf(&mut self.reader)?;
            self.block_n_seq = read_u16(&mut self.reader)? as usize;
            self.block_hap_to_seq.resize(self.n_haps, 0);
            for h in 0..self.n_haps {
                self.block_hap_to_seq[h] = read_u16(&mut self.reader)?;
            }
            self.block_remaining = n_recs;
        }

        self.block_remaining -= 1;

        // Read one record (identical logic to read_bref3)
        let pos = read_i32(&mut self.reader)?;
        let n_ids = read_u8(&mut self.reader)? as usize;
        let mut id_parts = Vec::new();
        for _ in 0..n_ids {
            id_parts.push(read_utf(&mut self.reader)?);
        }
        let id = if id_parts.is_empty() { ".".to_string() } else { id_parts.join(";") };

        let allele_code = read_u8(&mut self.reader)? as i8;
        let alleles: Vec<String> = if allele_code == -1 {
            let str_alleles = read_string_array(&mut self.reader)?;
            let _end = read_i32(&mut self.reader)?;
            str_alleles
        } else {
            snv_alleles(allele_code)?.0.iter().map(|&b| base_string(b)).collect()
        };

        let ref_allele = alleles.first().cloned().unwrap_or_default();
        let alt_alleles: Vec<String> = alleles[1..].to_vec();

        let flag = read_u8(&mut self.reader)?;
        let n_haps = self.n_haps;

        let hap_alleles = if flag == SEQ_CODED {
            let mut seq_to_allele = vec![0u8; self.block_n_seq];
            self.reader.read_exact(&mut seq_to_allele)
                .map_err(|e| format!("read error: {}", e))?;
            let mut out = vec![0u8; n_haps];
            for h in 0..n_haps {
                let seq = self.block_hap_to_seq[h] as usize;
                if seq >= self.block_n_seq {
                    // Hard-error instead of silently mapping out-of-range
                    // hap_to_seq entries to REF (would corrupt genotypes).
                    return Err(format!("BREF3 block: hap {} maps to seq {} >= n_seq {} (corrupt block)", h, seq, self.block_n_seq));
                }
                out[h] = seq_to_allele[seq];
            }
            out
        } else if flag == ALLELE_CODED {
            let n_alleles = alleles.len();
            let mut assigned = vec![false; n_haps];
            let mut out = vec![0u8; n_haps];
            let mut major_allele = 0u8;
            for a in 0..n_alleles {
                let count = read_i32(&mut self.reader)?;
                if count == -1 {
                    major_allele = a as u8;
                } else if (0..=n_haps as i32).contains(&count) {
                    for _ in 0..count {
                        let hap_idx = read_i32(&mut self.reader)? as usize;
                        if hap_idx < n_haps {
                            out[hap_idx] = a as u8;
                            assigned[hap_idx] = true;
                        }
                    }
                } else {
                    // Malformed: a negative-but-not-(-1) count makes `0..count` a no-op
                    // and DESYNCS the stream (silently wrong decode); >n_haps is impossible.
                    return Err(format!("BREF3 ALLELE_CODED count {} out of range [-1, {}]", count, n_haps));
                }
            }
            for h in 0..n_haps {
                if !assigned[h] { out[h] = major_allele; }
            }
            out
        } else {
            return Err(format!("Unknown bref3 record flag: {}", flag));
        };

        Ok(Some(Bref3Variant {
            chrom: self.block_chrom.clone(),
            pos,
            ref_allele,
            alt_alleles,
            id,
            alleles: hap_alleles,
        }))
    }
    /// Read next variant's metadata only — skip allele decoding.
    /// Returns (chrom, pos, ref_allele, alt_allele, id). Much faster for counting/metadata passes.
    pub fn next_variant_meta_only(&mut self) -> Result<Option<(String, i32, String, String, String)>, String> {
        if self.finished { return Ok(None); }

        while self.block_remaining == 0 {
            let n_recs = read_i32(&mut self.reader)?;
            if n_recs == 0 {
                self.finished = true;
                return Ok(None);
            }
            self.block_chrom = read_utf(&mut self.reader)?;
            self.block_n_seq = read_u16(&mut self.reader)? as usize;
            // Skip hap_to_seq (n_haps × 2 bytes)
            let skip = self.n_haps * 2;
            let mut buf = vec![0u8; skip];
            self.reader.read_exact(&mut buf).map_err(|e| format!("skip hap_to_seq: {}", e))?;
            self.block_remaining = n_recs;
        }

        self.block_remaining -= 1;

        let pos = read_i32(&mut self.reader)?;
        let n_ids = read_u8(&mut self.reader)? as usize;
        let mut id_parts = Vec::new();
        for _ in 0..n_ids {
            id_parts.push(read_utf(&mut self.reader)?);
        }
        let id = if id_parts.is_empty() { ".".to_string() } else { id_parts.join(";") };

        let allele_code = read_u8(&mut self.reader)? as i8;
        let (ref_allele, alt_allele, n_alleles) = if allele_code == -1 {
            let str_alleles = read_string_array(&mut self.reader)?;
            let _end = read_i32(&mut self.reader)?;
            let r = str_alleles.first().cloned().unwrap_or_default();
            let a = str_alleles.get(1).cloned().unwrap_or_default();
            let n = str_alleles.len();
            (r, a, n)
        } else {
            let (bases, n_alleles) = snv_alleles(allele_code)?;
            let r = base_string(bases[0]);
            let a = bases.get(1).map_or_else(String::new, |&b| base_string(b));
            (r, a, n_alleles)
        };

        // Skip allele data without decoding
        let flag = read_u8(&mut self.reader)?;
        if flag == SEQ_CODED {
            let mut buf = vec![0u8; self.block_n_seq];
            self.reader.read_exact(&mut buf).map_err(|e| format!("skip seq: {}", e))?;
        } else if flag == ALLELE_CODED {
            for _ in 0..n_alleles {
                let count = read_i32(&mut self.reader)?;
                if count == -1 { continue; }
                if count < 0 {
                    return Err(format!("BREF3 ALLELE_CODED negative count: {}", count));
                }
                // Skip count × i32 via a bounded loop — NEVER pre-allocate a
                // count-sized buffer: a malformed huge count (or the old `count as
                // usize * 4` sign-wrap) would abort the process before read_exact
                // could fail. Byte-identical for valid input (same bytes consumed).
                let mut remaining = count as u64 * 4;
                let mut tmp = [0u8; 8192];
                while remaining > 0 {
                    let n = remaining.min(tmp.len() as u64) as usize;
                    self.reader.read_exact(&mut tmp[..n]).map_err(|e| format!("skip allele: {}", e))?;
                    remaining -= n as u64;
                }
            }
        } else {
            return Err(format!("Unknown bref3 record flag: {}", flag));
        }

        Ok(Some((self.block_chrom.clone(), pos, ref_allele, alt_allele, id)))
    }
}

/// Open a BREF3 file for streaming (convenience wrapper).
pub fn open_bref3_stream(path: &Path) -> Result<Bref3StreamReader<BufReader<std::fs::File>>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open bref3 file: {}", e))?;
    let reader = BufReader::with_capacity(1 << 20, file);
    Bref3StreamReader::new(reader)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snv_orders_are_the_lexicographic_permutations_of_acgt() {
        for o in &SNV_ORDERS {
            let mut sorted = *o;
            sorted.sort_unstable();
            assert_eq!(&sorted, b"ACGT", "{:?} is not an ordering of the four bases", o);
        }
        for w in SNV_ORDERS.windows(2) {
            assert!(w[0] < w[1], "{:?} does not precede {:?}", w[0], w[1]);
        }
    }

    #[test]
    fn snv_allele_code_reads_count_and_ordering() {
        // Ordering 9 is CGTA; two alleles C,G (code 9 << 2 | 1), four C,G,T,A.
        assert_eq!(snv_alleles((9 << 2 | 1) as i8).unwrap(), (&b"CG"[..], 2));
        assert_eq!(snv_alleles((9 << 2 | 3) as i8).unwrap(), (&b"CGTA"[..], 4));
        assert_eq!(snv_alleles((23 << 2) as i8).unwrap(), (&b"T"[..], 1));
        assert!(snv_alleles((24 << 2 | 1) as i8).is_err());
    }
}
