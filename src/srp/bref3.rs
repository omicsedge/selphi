//! BREF3 (Binary Reference Format v3) reader.
//!
//! Reads phased reference panel haplotypes from .bref3 files.

use std::io::{Read, BufReader};
use std::path::Path;

const MAGIC_NUMBER_V3: i32 = 2055763188;
const SEQ_CODED: u8 = 0;
const ALLELE_CODED: u8 = 1;

/// SNV permutation table: all 24 permutations of [A, C, G, T] in lexicographic order.
fn snv_perms() -> Vec<Vec<String>> {
    let bases: Vec<String> = vec!["A".into(), "C".into(), "G".into(), "T".into()];
    let mut perms = Vec::with_capacity(24);
    permute(&[], &bases, &mut perms);
    perms
}

fn permute(start: &[String], end: &[String], out: &mut Vec<Vec<String>>) {
    if end.is_empty() {
        out.push(start.to_vec());
    } else {
        for j in 0..end.len() {
            let mut new_start = start.to_vec();
            new_start.push(end[j].clone());
            let mut new_end = Vec::with_capacity(end.len() - 1);
            new_end.extend_from_slice(&end[..j]);
            new_end.extend_from_slice(&end[j+1..]);
            permute(&new_start, &new_end, out);
        }
    }
}

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

/// Result of reading a bref3 file: sample IDs + variants with genotypes.
pub struct Bref3Data {
    pub sample_ids: Vec<String>,
    pub variants: Vec<Bref3Variant>,
}

/// Read a complete bref3 file and return all variants with decoded genotypes.
pub fn read_bref3(path: &Path) -> Result<Bref3Data, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Cannot open bref3 file: {}", e))?;
    let mut r = BufReader::with_capacity(1 << 20, file);

    let snv_perms = snv_perms();

    // -- Header --
    let magic = read_i32(&mut r)?;
    if magic != MAGIC_NUMBER_V3 {
        return Err(format!("Invalid bref3 magic number: {} (expected {})", magic, MAGIC_NUMBER_V3));
    }

    let _program = read_utf(&mut r)?;
    let sample_ids = read_string_array(&mut r)?;
    let n_haps = sample_ids.len() * 2;

    let mut variants = Vec::new();

    // -- Blocks --
    loop {
        let n_recs = read_i32(&mut r)?;
        if n_recs == 0 { break; } // END_OF_DATA sentinel

        let chrom = read_utf(&mut r)?;

        // nSeq (unsigned short, big-endian)
        let n_seq = read_u16(&mut r)? as usize;

        // hapToSeq: char[nHaps] (2 bytes each, big-endian)
        let mut hap_to_seq = vec![0u16; n_haps];
        for h in 0..n_haps {
            hap_to_seq[h] = read_u16(&mut r)?;
        }

        // Read records
        for _ in 0..n_recs {
            // Marker
            let pos = read_i32(&mut r)?;
            let n_ids = read_u8(&mut r)? as usize;
            let mut id_parts = Vec::new();
            for _ in 0..n_ids {
                id_parts.push(read_utf(&mut r)?);
            }
            let id = if id_parts.is_empty() { ".".to_string() } else { id_parts.join(";") };

            let allele_code = read_u8(&mut r)? as i8;
            let alleles: Vec<String> = if allele_code == -1 {
                // Non-SNV: read explicit allele strings
                let str_alleles = read_string_array(&mut r)?;
                let _end = read_i32(&mut r)?;
                str_alleles
            } else {
                // SNV: decode from permutation
                let n_alleles = 1 + (allele_code & 0b11) as usize;
                let perm_index = (allele_code as u8 >> 2) as usize;
                snv_perms[perm_index][..n_alleles].to_vec()
            };

            let ref_allele = alleles.first().cloned().unwrap_or_default();
            let alt_alleles: Vec<String> = alleles[1..].to_vec();

            // Flag: SEQ_CODED or ALLELE_CODED
            let flag = read_u8(&mut r)?;

            let hap_alleles = if flag == SEQ_CODED {
                // Read seqToAllele[nSeq]
                let mut seq_to_allele = vec![0u8; n_seq];
                r.read_exact(&mut seq_to_allele).map_err(|e| format!("read error: {}", e))?;
                // Decode: allele[h] = seqToAllele[hapToSeq[h]]
                let mut out = vec![0u8; n_haps];
                for h in 0..n_haps {
                    let seq = hap_to_seq[h] as usize;
                    out[h] = if seq < n_seq { seq_to_allele[seq] } else { 0 };
                }
                out
            } else if flag == ALLELE_CODED {
                // Per-allele haplotype lists
                let n_alleles = alleles.len();
                let mut assigned = vec![false; n_haps];
                let mut out = vec![0u8; n_haps];
                let mut major_allele = 0u8;
                for a in 0..n_alleles {
                    let count = read_i32(&mut r)?;
                    if count == -1 {
                        major_allele = a as u8;
                    } else {
                        for _ in 0..count {
                            let hap_idx = read_i32(&mut r)? as usize;
                            if hap_idx < n_haps {
                                out[hap_idx] = a as u8;
                                assigned[hap_idx] = true;
                            }
                        }
                    }
                }
                // Fill unassigned haps with major allele
                for h in 0..n_haps {
                    if !assigned[h] {
                        out[h] = major_allele;
                    }
                }
                out
            } else {
                return Err(format!("Unknown bref3 record flag: {}", flag));
            };

            variants.push(Bref3Variant {
                chrom: chrom.clone(),
                pos,
                ref_allele,
                alt_alleles,
                id,
                alleles: hap_alleles,
            });
        }
    }

    Ok(Bref3Data { sample_ids, variants })
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
    let mut arr = Vec::with_capacity(len as usize);
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
    snv_perms: Vec<Vec<String>>,
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
        let snv_perms = snv_perms();

        let magic = read_i32(&mut reader)?;
        if magic != MAGIC_NUMBER_V3 {
            return Err(format!("Invalid bref3 magic number: {} (expected {})", magic, MAGIC_NUMBER_V3));
        }

        let _program = read_utf(&mut reader)?;
        let sample_ids = read_string_array(&mut reader)?;
        let n_haps = sample_ids.len() * 2;

        Ok(Self {
            reader, snv_perms, sample_ids, n_haps,
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
            let n_alleles = 1 + (allele_code & 0b11) as usize;
            let perm_index = (allele_code as u8 >> 2) as usize;
            self.snv_perms[perm_index][..n_alleles].to_vec()
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
                out[h] = if seq < self.block_n_seq { seq_to_allele[seq] } else { 0 };
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
                } else {
                    for _ in 0..count {
                        let hap_idx = read_i32(&mut self.reader)? as usize;
                        if hap_idx < n_haps {
                            out[hap_idx] = a as u8;
                            assigned[hap_idx] = true;
                        }
                    }
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
            let n_alleles = 1 + (allele_code & 0b11) as usize;
            let perm_index = (allele_code as u8 >> 2) as usize;
            let p = &self.snv_perms[perm_index];
            let r = p[0].clone();
            let a = if n_alleles > 1 { p[1].clone() } else { String::new() };
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
                if count != -1 {
                    // Skip count × i32
                    let skip = count as usize * 4;
                    let mut buf = vec![0u8; skip];
                    self.reader.read_exact(&mut buf).map_err(|e| format!("skip allele: {}", e))?;
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
