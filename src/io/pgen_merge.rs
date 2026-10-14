//! Native PGEN sample-merger.
//!
//! Reads N per-batch PGEN files (each with same variants, different sample
//! subsets), concatenates the 2-bit hardcall and 16-bit dosage data per
//! variant, and emits a single merged PGEN + corresponding PVAR + PSAM.
//!
//! Every record, per batch and merged: [ceil(K/4) bytes] 2-bit packed hardcalls
//! (4 samples per byte, LSB first), then [K*2 bytes] u16 LE dosage per sample
//! (16384 = 1.0, 32768 = 2.0); see pgen_output for the file layout (mode 0x10).

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

    // 3. Open all N PGEN readers at their first record, get K_b per batch.
    let mut readers: Vec<BufReader<std::fs::File>> = Vec::with_capacity(batch_paths.len());
    let mut k_per_batch: Vec<usize> = Vec::with_capacity(batch_paths.len());
    let mut variant_count: Option<usize> = None;
    for (pgen, _pvar) in batch_paths {
        let (n_var, n_sam, start, _) = crate::io::pgen_output::read_layout(pgen)?;
        if let Some(prev) = variant_count {
            if prev != n_var {
                return Err(std::io::Error::other(format!(
                    "variant count mismatch: prev={prev} this={n_var}"
                )));
            }
        } else {
            variant_count = Some(n_var);
        }
        let mut f = std::fs::File::open(pgen)?;
        f.seek(SeekFrom::Start(start))?;
        k_per_batch.push(n_sam);
        readers.push(BufReader::with_capacity(4 << 20, f));
    }

    let total_samples: usize = k_per_batch.iter().sum();
    if total_samples != all_sample_names.len() {
        return Err(std::io::Error::other(format!(
            "sample count mismatch: PGEN batches total {total_samples}, sample_names {}",
            all_sample_names.len(),
        )));
    }

    // 4. Output PGEN (mode 0x10, see pgen_output::PgenWriter) of exactly n_var records.
    let n_var = variant_count.unwrap_or(0);
    let mut out = crate::io::pgen_output::PgenWriter::new(output_path, total_samples, n_var)?;
    let total_bytes_hc = total_samples.div_ceil(4);
    let mut packed_out = vec![0u8; total_bytes_hc];
    let mut dosage_out = vec![0u8; 2 * total_samples];

    // Per-batch scratch buffers
    let mut batch_hc_bytes: Vec<Vec<u8>> = k_per_batch.iter()
        .map(|k| vec![0u8; k.div_ceil(4)]).collect();
    let mut batch_unpacked: Vec<Vec<u8>> = k_per_batch.iter()
        .map(|k| vec![0u8; *k]).collect();

    for _ in 0..n_var {
        // 4a. Read each batch's record: hardcalls, then dosages straight into place.
        let mut d_off = 0usize;
        for b in 0..readers.len() {
            readers[b].read_exact(&mut batch_hc_bytes[b])?;
            let k2 = 2 * k_per_batch[b];
            readers[b].read_exact(&mut dosage_out[d_off..d_off + k2])?;
            d_off += k2;
            unpack_2bit(&batch_hc_bytes[b], k_per_batch[b], &mut batch_unpacked[b]);
        }

        // 4b. Concatenate hardcalls into 1-byte-per-sample, then repack.
        packed_out.fill(0);
        let mut sample_idx = 0usize;
        for unp in &batch_unpacked {
            for &g in unp {
                packed_out[sample_idx / 4] |= (g & 0x03) << ((sample_idx % 4) * 2);
                sample_idx += 1;
            }
        }
        out.write_raw(&packed_out, &dosage_out)?;
    }
    out.finish()
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
