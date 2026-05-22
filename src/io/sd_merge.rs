//! Native SelfDecode ZIP merger.
//!
//! Each per-batch SelfDecode writer (`sd_batch::SdBatchWriter`) produces a
//! `*.selfdecode.zip` containing `{sample}/chrom={chr}/{chunk}.parquet`
//! entries. Since sample names are unique across batches, the merger simply
//! copies all entries from every per-batch ZIP into a single output ZIP —
//! no re-encoding, no parquet-level work.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions, CompressionMethod};

/// Merge N per-batch SelfDecode ZIPs into one merged ZIP at `output_path`.
///
/// Output path will be given the `.selfdecode.zip` extension (mirrors
/// `SelfdecodeWriter::new`'s behaviour).
pub fn merge_batch_sds(batch_paths: &[PathBuf], output_path: &Path) -> std::io::Result<()> {
    if batch_paths.is_empty() {
        return Err(std::io::Error::other("no batch files to merge"));
    }
    let final_path = output_path.with_extension("selfdecode.zip");
    let out_file = std::fs::File::create(&final_path)?;
    let mut zw = ZipWriter::new(out_file);
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored); // parquet entries are already compressed

    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024 * 1024);
    for path in batch_paths {
        let f = std::fs::File::open(path)?;
        let mut za = ZipArchive::new(f)
            .map_err(|e| std::io::Error::other(format!("open zip {path:?}: {e}")))?;
        let n = za.len();
        for i in 0..n {
            let mut entry = za.by_index(i)
                .map_err(|e| std::io::Error::other(format!("zip entry {i}: {e}")))?;
            let name = entry.name().to_string();
            buf.clear();
            entry.read_to_end(&mut buf)?;
            zw.start_file(&name, options)
                .map_err(|e| std::io::Error::other(format!("start_file {name}: {e}")))?;
            zw.write_all(&buf)?;
        }
    }
    zw.finish().map_err(|e| std::io::Error::other(format!("zip finalize: {e}")))?;
    Ok(())
}
