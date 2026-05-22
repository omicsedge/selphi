//! Native PGEN sample-merger.
//!
//! Reads N per-batch PGEN files (each with same variants, different sample
//! subsets), concatenates the 2-bit hardcall and 16-bit dosage data per
//! variant, and emits a single merged PGEN + corresponding PVAR + PSAM.
//!
//! PGEN format (mode 0x03): header (12 bytes) + per-variant fixed records:
//!   [ceil(K/4) bytes] 2-bit packed hardcalls (4 samples per byte, LSB first)
//!   [K*2 bytes]       u16 LE dosage per sample (16384 = 1.0, 32768 = 2.0)

use std::io::{Read, Seek, SeekFrom, Write, BufReader, BufWriter};
use std::path::{Path, PathBuf};

/// Merge N per-batch PGEN files into a single merged PGEN at `output_path`.
/// Also writes the merged PSAM (concatenated sample names) and copies the
/// PVAR from the first batch (all batches share the same variants).
pub fn merge_batch_pgens(
    batch_paths: &[(PathBuf, PathBuf)],   // (pgen, pvar) per batch
    output_path: &Path,
    all_sample_names: &[String],
) -> std::io::Result<()> {
    if batch_paths.is_empty() {
        return Err(std::io::Error::other("no batch files to merge"));
    }

    // 1. Write merged PSAM
    crate::io::pgen_output::write_psam(output_path, all_sample_names)?;

    // 2. Copy PVAR from first batch (all batches have identical variant lines)
    {
        let src = std::fs::File::open(&batch_paths[0].1)?;
        let dst_path = output_path.with_extension("pvar");
        let mut br = BufReader::new(src);
        let mut bw = BufWriter::new(std::fs::File::create(&dst_path)?);
        std::io::copy(&mut br, &mut bw)?;
        bw.flush()?;
    }

    // 3. Open all N PGEN readers, validate headers, get K_b per batch.
    let mut readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(batch_paths.len());
    let mut k_per_batch: Vec<usize> = Vec::with_capacity(batch_paths.len());
    let mut variant_count: Option<u32> = None;
    for (pgen, _pvar) in batch_paths {
        let f = std::fs::File::open(pgen)?;
        let mut r = BufReader::with_capacity(4 << 20, f);
        let mut hdr = [0u8; 12];
        r.read_exact(&mut hdr)?;
        if hdr[0..2] != [0x6c, 0x1b] {
            return Err(std::io::Error::other(format!("bad PGEN magic in {pgen:?}")));
        }
        if hdr[2] != 0x03 {
            return Err(std::io::Error::other(format!("PGEN mode {} not supported by merger (only 0x03)", hdr[2])));
        }
        let n_var = u32::from_le_bytes(hdr[3..7].try_into().unwrap());
        let n_sam = u32::from_le_bytes(hdr[7..11].try_into().unwrap()) as usize;
        if let Some(prev) = variant_count {
            if prev != n_var {
                return Err(std::io::Error::other(format!(
                    "variant count mismatch: prev={prev} this={n_var}"
                )));
            }
        } else {
            variant_count = Some(n_var);
        }
        k_per_batch.push(n_sam);
        readers.push(r);
    }

    let total_samples: usize = k_per_batch.iter().sum();
    if total_samples != all_sample_names.len() {
        return Err(std::io::Error::other(format!(
            "sample count mismatch: PGEN batches total {total_samples}, sample_names {}",
            all_sample_names.len(),
        )));
    }

    // 4. Open output PGEN and write header (placeholder variant_ct, patched at end).
    let out_path = output_path.with_extension("pgen");
    let mut out = BufWriter::with_capacity(4 << 20, std::fs::File::create(&out_path)?);
    out.write_all(&[0x6c, 0x1b])?;
    out.write_all(&[0x03])?;
    out.write_all(&(0u32).to_le_bytes())?;
    out.write_all(&(total_samples as u32).to_le_bytes())?;
    out.write_all(&[0x80])?;

    let n_var = variant_count.unwrap_or(0) as usize;
    let total_bytes_hc = total_samples.div_ceil(4);
    let mut packed_out = vec![0u8; total_bytes_hc];

    // Per-batch scratch buffers
    let mut batch_hc_bytes: Vec<Vec<u8>> = k_per_batch.iter()
        .map(|k| vec![0u8; k.div_ceil(4)]).collect();
    let mut batch_unpacked: Vec<Vec<u8>> = k_per_batch.iter()
        .map(|k| vec![0u8; *k]).collect();
    let mut batch_dosage: Vec<Vec<u8>> = k_per_batch.iter()
        .map(|k| vec![0u8; k * 2]).collect();

    for _ in 0..n_var {
        // 4a. Read each batch's record.
        for b in 0..readers.len() {
            readers[b].read_exact(&mut batch_hc_bytes[b])?;
            readers[b].read_exact(&mut batch_dosage[b])?;
            unpack_2bit(&batch_hc_bytes[b], k_per_batch[b], &mut batch_unpacked[b]);
        }

        // 4b. Concatenate hardcalls into 1-byte-per-sample, then repack.
        for b in &mut packed_out { *b = 0; }
        let mut sample_idx = 0usize;
        for b in 0..readers.len() {
            let unp = &batch_unpacked[b];
            for &g in unp {
                let byte_idx = sample_idx / 4;
                let bit_off = (sample_idx % 4) * 2;
                packed_out[byte_idx] |= (g & 0x03) << bit_off;
                sample_idx += 1;
            }
        }
        out.write_all(&packed_out)?;

        // 4c. Dosages: simple byte concat (already 2-bytes-per-sample).
        for b in 0..readers.len() {
            out.write_all(&batch_dosage[b])?;
        }
    }

    out.flush()?;
    drop(out);

    // 5. Patch variant_ct at offset 3.
    let mut f = std::fs::OpenOptions::new().write(true).open(&out_path)?;
    f.seek(SeekFrom::Start(3))?;
    f.write_all(&(n_var as u32).to_le_bytes())?;
    f.flush()?;
    Ok(())
}

/// Unpack 2-bit-packed hardcalls (4 samples per byte, LSB first) into one
/// byte per sample.
fn unpack_2bit(packed: &[u8], n_samples: usize, out: &mut [u8]) {
    debug_assert!(out.len() >= n_samples);
    for i in 0..n_samples {
        let byte_idx = i / 4;
        let bit_off = (i % 4) * 2;
        out[i] = (packed[byte_idx] >> bit_off) & 0x03;
    }
}
