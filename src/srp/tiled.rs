//! Tiled SRP format: 2D tiled sparse reference panel for maximum interpolation speed.
//!
//! Layout:
//!   [8 bytes]  Magic: "SRPt\0\0\0\0"
//!   [4 bytes]  header_size (u32 LE)
//!   [header_size bytes] zstd-compressed JSON header
//!   [4 bytes]  n_tiles (u32 LE) = n_tile_rows × n_tile_cols
//!   [n_tiles × 12 bytes] tile index: {offset: u64 LE, comp_size: u32 LE}
//!   [tile 0 data] LZ4-compressed SparseTile
//!   [tile 1 data] ...
//!
//! Tile = CSC sub-matrix of 1024 variants × 4096 haplotypes.
//! u16 row indices (max 1024), u32 indptr, LZ4 compression.
//! Each tile ~500 KB decompressed — fits in L2 cache per core.
//!
//! I/O strategy: sequential bulk reads per batch of stripes, NOT mmap.
//! Tiles for a stripe are contiguous in the file (row-major layout),
//! so one pread gets all bands for a batch of stripes.

use std::io;
use std::path::{Path, PathBuf};
use rayon::prelude::*;

use super::{SparseTile, TILE_ROWS, TILE_COLS};

const MAGIC: &[u8; 8] = b"SRPt\0\0\0\0";

// ============================================================================
// Tile index entry
// ============================================================================

#[derive(Clone, Copy)]
struct TileEntry {
    offset: u64,
    comp_size: u32,
}

/// Public tile entry for constructing TiledSrpReader from external index data.
#[derive(Clone, Copy)]
pub struct TileEntryPub {
    pub offset: u64,
    pub comp_size: u32,
}

// ============================================================================
// PreloadedStripes: compressed tile data pre-read into RAM
// ============================================================================

/// Pre-loaded compressed tile data for a contiguous range of stripes.
/// One sequential read from disk fills the buffer; decompress_tile reads from RAM.
/// No page faults, no mmap — pure sequential I/O.
pub struct PreloadedStripes {
    pub buf: Vec<u8>,
    /// (offset_in_buf, comp_size) for each tile, indexed as [(stripe - first_stripe) * n_tile_cols + band]
    offsets: Vec<(usize, usize)>,
    pub first_stripe: usize,
    pub n_stripes: usize,
    pub n_tile_cols: usize,
    pub n_variants: usize,
}

/// Per-thread zstd decompression context and output buffer, reused across
/// tiles while a `TileDecoderScope` is held. `zstd::decode_all` builds a fresh
/// context and grows a fresh `Vec` by doubling for every tile: 2.9x the time of
/// this path on the 75,552-haplotype panel's tiles (1,900 tiles: 0.515 s vs
/// 0.176 s, one thread), and tiles are decoded twice per chromosome (bitmatrix
/// extraction, then output).
struct ZstdTileDecoder {
    dec: zstd::bulk::Decompressor<'static>,
    buf: Vec<u8>,
}

thread_local! {
    static ZSTD_TILE_DECODER: std::cell::RefCell<Option<ZstdTileDecoder>> = const { std::cell::RefCell::new(None) };
    /// `TileDecoderScope::this_thread` scopes held by this thread.
    static THREAD_DECODER_SCOPES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// `TileDecoderScope::pool` scopes held anywhere.
static POOL_DECODER_SCOPES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Owner of the per-thread decoder cache for one decoding stage. While a stage
/// holds one, tiles are decoded with a reused context and buffer: by every
/// rayon worker for a `pool` scope, by the holding thread for a `this_thread`
/// scope. Anywhere else `decompress_tile` decodes as it always did — a fresh
/// context and buffer per tile, nothing kept. Dropping the scope, including on
/// an early `?` return, drops the decoders it enabled, so none is carried into
/// the stages that follow (the HMM's memory peak among them).
pub struct TileDecoderScope {
    pool: bool,
    // Thread-bound: a `this_thread` scope counts on the creating thread.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl TileDecoderScope {
    /// For a stage that decodes on the rayon pool (bitmatrix extraction, the
    /// output writer). Its drop broadcasts to the pool: hold it on a thread
    /// that is not waited on by pool work.
    pub fn pool() -> Self {
        POOL_DECODER_SCOPES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Self { pool: true, _not_send: std::marker::PhantomData }
    }

    /// For a stage that decodes on the calling thread only (the sample-batched
    /// driver). Its drop touches no other thread.
    pub fn this_thread() -> Self {
        THREAD_DECODER_SCOPES.with(|c| c.set(c.get() + 1));
        Self { pool: false, _not_send: std::marker::PhantomData }
    }
}

impl Drop for TileDecoderScope {
    fn drop(&mut self) {
        if self.pool {
            POOL_DECODER_SCOPES.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            // Do not broadcast while unwinding: the run is failing anyway. (A
            // stage overlapping this one would only rebuild its decoders.)
            if std::thread::panicking() { drop_this_thread_decoder(); } else { release_tile_decoders(); }
        } else {
            let left = THREAD_DECODER_SCOPES.with(|c| { let n = c.get() - 1; c.set(n); n });
            if left == 0 { drop_this_thread_decoder(); }
        }
    }
}

/// The reused decoder is used on this thread: a `this_thread` scope is held
/// here, or a `pool` scope is held and this is a rayon worker (the threads its
/// drop reaches).
#[inline]
fn decoder_cache_enabled() -> bool {
    THREAD_DECODER_SCOPES.with(|c| c.get() > 0)
        || (POOL_DECODER_SCOPES.load(std::sync::atomic::Ordering::Relaxed) > 0
            && rayon::current_thread_index().is_some())
}

fn drop_this_thread_decoder() {
    ZSTD_TILE_DECODER.with(|c| { if let Ok(mut d) = c.try_borrow_mut() { d.take(); } });
}

/// This thread holds a cached decoder (tests).
#[cfg(test)]
fn decoder_cached_here() -> bool {
    ZSTD_TILE_DECODER.with(|c| c.borrow().is_some())
}

/// Tiles decoded by the slow path (tests check it is exercised).
#[cfg(test)]
static ZSTD_SLOW_PATH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Output buffers above this are not kept for the next tile.
const ZSTD_KEEP_BUF: usize = 32 << 20;

/// Decompress a zstd tile with this thread's reused context. `None` when the
/// frame does not fit the buffer's capacity (or does not decode): the caller
/// then decodes it the old way, which also reports a corrupt tile.
fn zstd_tile_fast(compressed: &[u8]) -> Option<SparseTile> {
    ZSTD_TILE_DECODER.with(|cell| {
        // No rayon call happens under this borrow, so it cannot be re-entered.
        let mut cell = cell.borrow_mut();
        if cell.is_none() {
            *cell = Some(ZstdTileDecoder { dec: zstd::bulk::Decompressor::new().ok()?, buf: Vec::new() });
        }
        let d = cell.as_mut().unwrap();
        d.buf.clear();
        let want = (compressed.len() * 8).max(1 << 20);
        if d.buf.capacity() < want { d.buf.reserve(want); }
        d.dec.decompress_to_buffer(compressed, &mut d.buf).ok()?;
        Some(SparseTile::from_bytes(&d.buf))
    })
}

/// Drop the tile decoder of every rayon worker and of this thread (a decoder
/// in use at that moment is left alone).
fn release_tile_decoders() {
    rayon::broadcast(|_| drop_this_thread_decoder());
    drop_this_thread_decoder();
}

/// After a slow-path decode of `raw_len` bytes, make the next one fit.
fn zstd_tile_grow(raw_len: usize) {
    if raw_len > ZSTD_KEEP_BUF { return; }
    ZSTD_TILE_DECODER.with(|cell| {
        if let Some(d) = cell.borrow_mut().as_mut() {
            d.buf.clear();
            d.buf.reserve(raw_len + raw_len / 4);
        }
    });
}

impl PreloadedStripes {
    /// Decompress a single tile from the pre-loaded buffer. Lock-free, no I/O.
    #[inline]
    pub fn decompress_tile(&self, stripe: usize, band: usize) -> SparseTile {
        let local = (stripe - self.first_stripe) * self.n_tile_cols + band;
        let (off, len) = self.offsets[local];
        let compressed = &self.buf[off..off + len];
        // Auto-detect: zstd magic = 0xFD2FB528, LZ4 prepend-size starts with u32 size.
        // Panics on corrupt tile data are unrecoverable here (hot path, fn
        // returns SparseTile not Result); include stripe/band/len so the
        // operator can locate the bad tile.
        let raw = if compressed.len() >= 4 && compressed[0] == 0x28 && compressed[1] == 0xB5 && compressed[2] == 0x2F && compressed[3] == 0xFD {
            let cached = decoder_cache_enabled();
            if cached && let Some(tile) = zstd_tile_fast(compressed) { return tile; }
            let raw = zstd::decode_all(std::io::Cursor::new(compressed)).unwrap_or_else(|e|
                panic!("zstd decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e));
            if cached {
                zstd_tile_grow(raw.len());
                #[cfg(test)]
                ZSTD_SLOW_PATH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            raw
        } else {
            lz4_flex::decompress_size_prepended(compressed).unwrap_or_else(|e|
                panic!("LZ4 decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        };
        SparseTile::from_bytes(&raw)
    }

    /// Check if a stripe is within this preloaded range.
    #[inline]
    pub fn contains_stripe(&self, stripe: usize) -> bool {
        stripe >= self.first_stripe && stripe < self.first_stripe + self.n_stripes
    }

}

// ============================================================================
// Reader: file-based tiled SRP reader (sequential I/O, no mmap)
// ============================================================================

/// `extract_bitmatrix`: stripes with at most this many distinct wanted rows are
/// read by binary search per column instead of a pass over every nonzero.
const EXTRACT_SEARCH_MAX_ROWS: usize = 2;

/// `extract_bitmatrix`: at most this many compressed bytes per read. Tiny under
/// test, so the tests' small panels are read in many chunks.
const EXTRACT_READ_BYTES: usize = if cfg!(test) { 64 << 10 } else { 64 << 20 };

/// `extract_bitmatrix` read its stripes this many stripes-with-sites at a time,
/// one read in memory at once; the largest such read bounds the compressed
/// bytes it may now hold. (Small under test, to make that bound bind.)
const EXTRACT_PREV_READ_STRIPES: usize = if cfg!(test) { 3 } else { 400 };

/// How `extract_bitmatrix` reads the stripes with sites (`stripes`, ascending).
struct ExtractReads {
    /// Each read: a range of `stripes`, read as one contiguous span.
    reads: Vec<std::ops::Range<usize>>,
    /// Compressed bytes of each read.
    bytes: Vec<usize>,
    /// Read `k + 1` while read `k` is decoded.
    prefetch: Vec<bool>,
    /// Compressed bytes the previous scheme held at its largest read.
    #[cfg_attr(not(test), allow(dead_code))]
    bound: usize,
}

/// Plan `extract_bitmatrix`'s reads so that the compressed bytes in memory —
/// the read being decoded plus the one being read behind it — never exceed what
/// the previous scheme held (`EXTRACT_PREV_READ_STRIPES` stripes-with-sites per
/// read, one at a time). A read spans at most that many stripes-with-sites and,
/// unless it is a single one, at most half that bound (and EXTRACT_READ_BYTES);
/// the next read is started early only when the two fit the bound together.
/// `read_len(first, n)` is the compressed size of stripes `first..first + n`;
/// `prev_read_stripes` and `read_cap` are EXTRACT_PREV_READ_STRIPES and
/// EXTRACT_READ_BYTES outside tests.
fn plan_extract_reads(stripes: &[usize], read_len: impl Fn(usize, usize) -> usize,
                      prev_read_stripes: usize, read_cap: usize) -> ExtractReads {
    let span = |a: usize, b: usize| read_len(stripes[a], stripes[b] - stripes[a] + 1);
    let bound = (0..stripes.len()).step_by(prev_read_stripes)
        .map(|a| span(a, (a + prev_read_stripes).min(stripes.len()) - 1))
        .max().unwrap_or(0);
    let cap = read_cap.min(bound / 2);
    let mut reads = Vec::new();
    let mut i = 0;
    while i < stripes.len() {
        let mut j = i + 1;
        while j < stripes.len() && j - i < prev_read_stripes && span(i, j) <= cap { j += 1; }
        reads.push(i..j);
        i = j;
    }
    let bytes: Vec<usize> = reads.iter().map(|r| span(r.start, r.end - 1)).collect();
    let prefetch = (0..reads.len()).map(|k| k + 1 < reads.len() && bytes[k] + bytes[k + 1] <= bound).collect();
    ExtractReads { reads, bytes, prefetch, bound }
}

pub struct TiledSrpReader {
    file_path: PathBuf,
    n_variants: usize,
    n_haps: usize,
    tile_index: Vec<TileEntry>,
    pub n_tile_rows: usize,
    pub n_tile_cols: usize,
}

impl TiledSrpReader {
    /// Open tiled SRP: reads only magic + tile_index (~9 MB for TOPMed).
    /// Skips header entirely — n_variants/n_haps come from the SRP reader.
    pub fn open(path: &Path, n_variants: usize, n_haps: usize) -> io::Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path)?;

        // Read magic
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a tiled SRP file"));
        }

        // Skip header (don't decompress — metadata comes from SRP reader)
        let mut hdr_size_buf = [0u8; 4];
        file.read_exact(&mut hdr_size_buf)?;
        let header_size = u32::from_le_bytes(hdr_size_buf) as usize;
        file.seek(SeekFrom::Current(header_size as i64))?;

        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);

        // Read tile index
        let mut n_tiles_buf = [0u8; 4];
        file.read_exact(&mut n_tiles_buf)?;
        let n_tiles = u32::from_le_bytes(n_tiles_buf) as usize;

        // Cap the eager zero-fill against the file size so a corrupt u32 count
        // cannot request a multi-GB allocation before the read fails (matches
        // the single-chr reader / BREF3 / CSI capped-reservation idiom).
        let file_len = file.metadata().map(|m| m.len()).unwrap_or(u64::MAX);
        if n_tiles as u64 * 12 > file_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                "corrupt tiled SRP: tile index count {} exceeds file size; regenerate the panel", n_tiles)));
        }
        let mut idx_data = vec![0u8; n_tiles * 12];
        file.read_exact(&mut idx_data)?;
        let tile_index: Vec<TileEntry> = (0..n_tiles).map(|i| {
            let base = i * 12;
            TileEntry {
                offset: u64::from_le_bytes(idx_data[base..base+8].try_into().unwrap()),
                comp_size: u32::from_le_bytes(idx_data[base+8..base+12].try_into().unwrap()),
            }
        }).collect();

        Ok(Self {
            file_path: path.to_path_buf(),
            n_variants, n_haps,
            tile_index, n_tile_rows, n_tile_cols,
        })
    }

    /// Construct from pre-parsed tile index (used by SRP reader).
    pub fn from_entries(
        file_path: PathBuf, n_variants: usize, n_haps: usize,
        entries: Vec<TileEntryPub>, n_tile_rows: usize, n_tile_cols: usize,
    ) -> Self {
        let tile_index = entries.into_iter()
            .map(|e| TileEntry { offset: e.offset, comp_size: e.comp_size })
            .collect();
        Self { file_path, n_variants, n_haps, tile_index, n_tile_rows, n_tile_cols }
    }

    pub fn n_variants(&self) -> usize { self.n_variants }
    pub fn n_haps(&self) -> usize { self.n_haps }
    pub fn file_path(&self) -> &Path { &self.file_path }

    /// Estimate compressed bytes for one stripe (sum of comp_size for all bands).
    pub fn stripe_compressed_bytes(&self, stripe: usize) -> usize {
        let base = stripe * self.n_tile_cols;
        (0..self.n_tile_cols)
            .map(|b| self.tile_index[base + b].comp_size as usize)
            .sum()
    }

    /// Bytes `preload_stripes(first_stripe, n_stripes)` reads.
    pub fn stripes_read_len(&self, first_stripe: usize, n_stripes: usize) -> usize {
        let first_tile = first_stripe * self.n_tile_cols;
        let last_tile = (first_stripe + n_stripes) * self.n_tile_cols - 1;
        let last_entry = &self.tile_index[last_tile];
        (last_entry.offset + last_entry.comp_size as u64 - self.tile_index[first_tile].offset) as usize
    }

    /// Pre-read compressed tile data for a contiguous range of stripes.
    /// One sequential pread — converts random mmap access into a single bulk read.
    pub fn preload_stripes(&self, first_stripe: usize, n_stripes: usize) -> io::Result<PreloadedStripes> {
        self.preload_stripes_into(first_stripe, n_stripes, Vec::new())
    }

    /// `preload_stripes` into `buf`, reusing its allocation when it is large
    /// enough. A buffer allocated by the thread that will drop it is returned to
    /// the OS promptly; one allocated on a short-lived reader thread and freed on
    /// another lingers in the allocator (measured: +250-300 MB RSS during the
    /// 75,552-haplotype panel's bitmatrix extraction).
    pub fn preload_stripes_into(&self, first_stripe: usize, n_stripes: usize, mut buf: Vec<u8>) -> io::Result<PreloadedStripes> {
        let first_tile = first_stripe * self.n_tile_cols;
        let read_start = self.tile_index[first_tile].offset;
        let read_len = self.stripes_read_len(first_stripe, n_stripes);

        // One sequential read from disk; a large one in up to four contiguous
        // parts read concurrently (from the page cache a single reader copies
        // ~2 GB/s, four ~7 GB/s on the 75,552-haplotype panel). Same bytes.
        let file = std::fs::File::open(&self.file_path)?;
        if buf.capacity() < read_len {
            buf = vec![0u8; read_len];
        } else {
            buf.clear();
            buf.resize(read_len, 0);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::FileExt;
            const PART: usize = 32 << 20;
            let k = (read_len / PART).min(4);
            if k >= 2 {
                let chunk = read_len.div_ceil(k);
                std::thread::scope(|sc| -> io::Result<()> {
                    let parts: Vec<_> = buf.chunks_mut(chunk).enumerate().map(|(i, part)| {
                        let file = &file;
                        sc.spawn(move || file.read_exact_at(part, read_start + (i * chunk) as u64))
                    }).collect();
                    for h in parts { h.join().expect("stripe read thread panicked")?; }
                    Ok(())
                })?;
            } else {
                file.read_exact_at(&mut buf, read_start)?;
            }
        }
        #[cfg(not(unix))]
        {
            use std::io::{Seek, SeekFrom, Read};
            let mut file = file;
            file.seek(SeekFrom::Start(read_start))?;
            file.read_exact(&mut buf)?;
        }

        // Build offset table: map (local_stripe, band) → (offset_in_buf, comp_size)
        let n_entries = n_stripes * self.n_tile_cols;
        let mut offsets = Vec::with_capacity(n_entries);
        for s in 0..n_stripes {
            for b in 0..self.n_tile_cols {
                let tid = (first_stripe + s) * self.n_tile_cols + b;
                let entry = &self.tile_index[tid];
                let local_off = (entry.offset - read_start) as usize;
                offsets.push((local_off, entry.comp_size as usize));
            }
        }

        Ok(PreloadedStripes {
            buf,
            offsets,
            first_stripe,
            n_stripes,
            n_tile_cols: self.n_tile_cols,
            n_variants: self.n_variants,
        })
    }

    /// Extract reference alleles for `wgs_idx` (chip variant indices) as a
    /// haplotype bitmatrix, reading directly from the compressed tiles. Shared
    /// by `SrpReader::extract_ref_alleles_bitmatrix` and
    /// `ChrSrpView::extract_ref_alleles_bitmatrix` (each resolves its own tiled
    /// backend and supplies `n_haps` from its metadata).
    pub fn extract_bitmatrix(&self, n_haps: usize, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use std::collections::HashMap;

        let n_chip = wgs_idx.len();
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_chip * n_words];

        let mut stripe_groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (ci, &wi) in wgs_idx.iter().enumerate() {
            stripe_groups.entry(wi / TILE_ROWS).or_default().push((ci, wi % TILE_ROWS));
        }
        let mut sorted: Vec<_> = stripe_groups.into_iter().collect();
        sorted.sort_by_key(|(s, _)| *s);

        let bits_ptr = bits.as_mut_ptr() as usize;
        let n_tc = self.n_tile_cols;
        let _decoders = TileDecoderScope::pool();

        // Reads, planned within the compressed bytes the previous code held: see
        // `plan_extract_reads`. Buffers are allocated on this thread (one freed
        // here after a reader thread allocated it lingered in the allocator).
        let stripes: Vec<usize> = sorted.iter().map(|(s, _)| *s).collect();
        let plan = plan_extract_reads(&stripes, |fs, n| self.stripes_read_len(fs, n),
            EXTRACT_PREV_READ_STRIPES, EXTRACT_READ_BYTES);
        let read_args = |k: usize| {
            let r = &plan.reads[k];
            let fs = stripes[r.start];
            (fs, stripes[r.end - 1] - fs + 1, Vec::with_capacity(plan.bytes[k]))
        };

        std::thread::scope(|sc| {
        let mut next: Option<std::thread::ScopedJoinHandle<'_, io::Result<PreloadedStripes>>> = None;
        for (k, r) in plan.reads.iter().enumerate() {
            let batch = &sorted[r.clone()];
            let loaded = match next.take() {
                Some(h) => h.join().expect("stripe read thread panicked"),
                None => { let (fs, n, buf) = read_args(k); self.preload_stripes_into(fs, n, buf) }
            }.expect("preload failed");
            // Read the next while this one is decoded, when both fit together.
            if plan.prefetch[k] {
                let (fs, n, buf) = read_args(k + 1);
                next = Some(sc.spawn(move || self.preload_stripes_into(fs, n, buf)));
            }

            batch.par_iter().for_each(|(stripe, sites)| {
                // Stripes are row-disjoint across workers, so no two workers touch the
                // same word; write through the raw base pointer instead of materializing
                // a full-length &mut per worker (overlapping &mut = aliasing UB).
                let base = bits_ptr as *mut u64;
                // Per stripe: which of its TILE_ROWS rows are wanted (a bitmap tested per
                // nonzero: one shift and AND, where a HashMap lookup per nonzero cost a
                // SipHash for every ALT allele in the panel, nearly all misses at array
                // density), the chip index of each wanted row, and any further chip indices
                // that map to the same row (duplicate sites; rare). `|=` is
                // order-independent and idempotent, so the output is bit-identical.
                let mut want = [0u64; TILE_ROWS / 64];
                let mut first = [u32::MAX; TILE_ROWS];
                let mut extra: Vec<(usize, usize)> = Vec::new();
                for &(ci, cr) in sites.iter() {
                    want[cr / 64] |= 1u64 << (cr % 64);
                    if first[cr] == u32::MAX { first[cr] = ci as u32; } else { extra.push((cr, ci)); }
                }
                // With few wanted rows, binary-search each in every column's
                // sorted rows; with more, one pass over the column's nonzeros
                // (measured on the 75,552-haplotype panel's tiles: search wins at
                // 1-2 rows a stripe, ties at 3, loses from 10). Same bits set.
                let mut rows: Vec<u16> = sites.iter().map(|&(_, cr)| cr as u16).collect();
                rows.sort_unstable();
                rows.dedup();
                let search = rows.len() <= EXTRACT_SEARCH_MAX_ROWS;
                for band in 0..n_tc {
                    let tile = loaded.decompress_tile(*stripe, band);
                    let cb = band * TILE_COLS;
                    for col in 0..tile.n_cols as usize {
                        let gh = cb + col;
                        if gh >= n_haps { break; }
                        let wi = gh / 64;
                        let bit = 1u64 << (gh % 64);
                        let (lo, hi) = tile.col_range(col);
                        let set = |lr: usize| {
                            unsafe { let p = base.add(first[lr] as usize * n_words + wi); *p |= bit; }
                            for &(r, ci) in &extra {
                                if r == lr { unsafe { let p = base.add(ci * n_words + wi); *p |= bit; } }
                            }
                        };
                        if search {
                            let col_rows = &tile.indices[lo..hi];
                            for &r in &rows {
                                if col_rows.binary_search(&r).is_ok() { set(r as usize); }
                            }
                        } else {
                            for k in lo..hi {
                                let lr = tile.indices[k] as usize;
                                if (want[lr / 64] >> (lr % 64)) & 1 == 0 { continue; }
                                set(lr);
                            }
                        }
                    }
                }
            });
            drop(loaded);
        }
        });
        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }
}

/// Compress one batch of pending stripes into `(stripe_id, band, zstd_bytes)`,
/// sorted by `(stripe_id, band)` for deterministic on-disk tile order.
///
/// `pending` is a list of `(stripe_id, stripe_columns)` where `stripe_columns[gc]`
/// is the (already row-local) sorted indices for global column `gc`. Each stripe
/// is split into `n_tile_cols` bands of `TILE_COLS` columns; the band's
/// `SparseTile` is built and zstd-3 compressed in parallel. Shared by the
/// single-chr `StreamingTileWriter` flush and the multi-chr `flush_tiles` so the
/// tile-compression path has exactly one definition.
pub fn compress_pending_tiles(
    pending: &[(usize, Vec<Vec<u16>>)],
    n_haps: usize,
    n_variants: usize,
    n_tile_cols: usize,
) -> Vec<(usize, usize, Vec<u8>)> {
    use std::io::Cursor;
    // Build the (batch_idx, stripe_id, band) task list.
    let mut tasks: Vec<(usize, usize, usize)> = Vec::new();
    for (i, (stripe_id, _)) in pending.iter().enumerate() {
        for band in 0..n_tile_cols {
            tasks.push((i, *stripe_id, band));
        }
    }
    // Compress all tiles in parallel; collect preserves task order.
    let mut results: Vec<(usize, usize, Vec<u8>)> = tasks.into_par_iter().map(|(batch_idx, stripe_id, band)| {
        let stripe_cols = &pending[batch_idx].1;
        let svs = stripe_id * TILE_ROWS;
        // saturating_sub: a corrupt panel chunk with an out-of-range CSC row index
        // (scatter_chunk_into_active, writer.rs) can derive a stripe past the declared
        // variant count, making svs >= n_variants. Clamp instead of underflowing
        // (debug builds assert loudly; release builds fail soft rather than wrapping
        // to a huge u16 and corrupting tile row counts). No effect on valid panels.
        debug_assert!(svs < n_variants, "tile stripe start {svs} >= n_variants {n_variants} (corrupt chunk indices)");
        let n_rows = (TILE_ROWS.min(n_variants.saturating_sub(svs))) as u16;
        let col_start = band * TILE_COLS;
        let col_end = (col_start + TILE_COLS).min(n_haps);
        let n_cols = col_end - col_start;

        let mut indptr = Vec::with_capacity(n_cols + 1);
        let mut indices = Vec::new();
        indptr.push(0u32);
        for lc in 0..n_cols {
            let gc = col_start + lc;
            if gc < stripe_cols.len() {
                indices.extend_from_slice(&stripe_cols[gc]);
            }
            indptr.push(indices.len() as u32);
        }
        let tile = SparseTile { indptr, indices, n_rows, n_cols: n_cols as u16 };
        let compressed = zstd::encode_all(Cursor::new(&tile.to_bytes()), 3).unwrap();
        (stripe_id, band, compressed)
    }).collect();
    // File order = (stripe_id, band).
    results.sort_by_key(|&(s, b, _)| (s, b));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- verbatim copies of the previous code (renamed; `self` -> parameter) ----

    /// Decompress a single tile from the pre-loaded buffer. Lock-free, no I/O.
    #[inline]
    fn old_decompress_tile(this: &PreloadedStripes, stripe: usize, band: usize) -> SparseTile {
        let local = (stripe - this.first_stripe) * this.n_tile_cols + band;
        let (off, len) = this.offsets[local];
        let compressed = &this.buf[off..off + len];
        // Auto-detect: zstd magic = 0xFD2FB528, LZ4 prepend-size starts with u32 size.
        // Panics on corrupt tile data are unrecoverable here (hot path, fn
        // returns SparseTile not Result); include stripe/band/len so the
        // operator can locate the bad tile.
        let raw = if compressed.len() >= 4 && compressed[0] == 0x28 && compressed[1] == 0xB5 && compressed[2] == 0x2F && compressed[3] == 0xFD {
            zstd::decode_all(std::io::Cursor::new(compressed)).unwrap_or_else(|e|
                panic!("zstd decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        } else {
            lz4_flex::decompress_size_prepended(compressed).unwrap_or_else(|e|
                panic!("LZ4 decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        };
        SparseTile::from_bytes(&raw)
    }

    /// Extract reference alleles for `wgs_idx` (chip variant indices) as a
    /// haplotype bitmatrix, reading directly from the compressed tiles. Shared
    /// by `SrpReader::extract_ref_alleles_bitmatrix` and
    /// `ChrSrpView::extract_ref_alleles_bitmatrix` (each resolves its own tiled
    /// backend and supplies `n_haps` from its metadata).
    fn old_extract_bitmatrix(this: &TiledSrpReader, n_haps: usize, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        use super::{TILE_ROWS, TILE_COLS};
        use std::collections::HashMap;

        let n_chip = wgs_idx.len();
        let n_words = n_haps.div_ceil(64);
        let mut bits = vec![0u64; n_chip * n_words];

        let mut stripe_groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for (ci, &wi) in wgs_idx.iter().enumerate() {
            stripe_groups.entry(wi / TILE_ROWS).or_default().push((ci, wi % TILE_ROWS));
        }
        let mut sorted: Vec<_> = stripe_groups.into_iter().collect();
        sorted.sort_by_key(|(s, _)| *s);

        let bits_ptr = bits.as_mut_ptr() as usize;
        let n_tc = this.n_tile_cols;

        for batch in sorted.chunks(400) {
            let fs = batch[0].0;
            let ls = batch.last().unwrap().0;
            let loaded = this.preload_stripes(fs, ls - fs + 1).expect("preload failed");

            batch.par_iter().for_each(|(stripe, sites)| {
                // Stripes are row-disjoint across workers, so no two workers touch the
                // same word; write through the raw base pointer instead of materializing
                // a full-length &mut per worker (overlapping &mut = aliasing UB).
                let base = bits_ptr as *mut u64;
                // Per stripe: which of its TILE_ROWS rows are wanted (a bitmap tested per
                // nonzero: one shift and AND, where a HashMap lookup per nonzero cost a
                // SipHash for every ALT allele in the panel, nearly all misses at array
                // density), the chip index of each wanted row, and any further chip indices
                // that map to the same row (duplicate sites; rare). `|=` is
                // order-independent and idempotent, so the output is bit-identical.
                let mut want = [0u64; TILE_ROWS / 64];
                let mut first = [u32::MAX; TILE_ROWS];
                let mut extra: Vec<(usize, usize)> = Vec::new();
                for &(ci, cr) in sites.iter() {
                    want[cr / 64] |= 1u64 << (cr % 64);
                    if first[cr] == u32::MAX { first[cr] = ci as u32; } else { extra.push((cr, ci)); }
                }
                for band in 0..n_tc {
                    let tile = old_decompress_tile(&loaded, *stripe, band);
                    let cb = band * TILE_COLS;
                    for col in 0..tile.n_cols as usize {
                        let gh = cb + col;
                        if gh >= n_haps { break; }
                        let wi = gh / 64;
                        let bit = 1u64 << (gh % 64);
                        let (lo, hi) = tile.col_range(col);
                        for k in lo..hi {
                            let lr = tile.indices[k] as usize;
                            if (want[lr / 64] >> (lr % 64)) & 1 == 0 { continue; }
                            unsafe { let p = base.add(first[lr] as usize * n_words + wi); *p |= bit; }
                            for &(r, ci) in &extra {
                                if r == lr { unsafe { let p = base.add(ci * n_words + wi); *p |= bit; } }
                            }
                        }
                    }
                }
            });
        }
        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }

    struct Xs(u64);
    impl Xs {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn below(&mut self, n: u64) -> u64 { self.next() % n }
    }

    fn random_tile(r: &mut Xs, n_rows: usize, n_cols: usize, per_mille: u64) -> SparseTile {
        let mut indptr = vec![0u32];
        let mut indices = Vec::new();
        for _ in 0..n_cols {
            for row in 0..n_rows { if r.below(1000) < per_mille { indices.push(row as u16); } }
            indptr.push(indices.len() as u32);
        }
        SparseTile { indptr, indices, n_rows: n_rows as u16, n_cols: n_cols as u16 }
    }

    /// A tiled panel file: every tile zstd-3 (as the writer) or LZ4-compressed.
    fn panel(dir: &Path, r: &mut Xs, n_variants: usize, n_haps: usize, per_mille: u64, lz4_every: u64)
        -> (TiledSrpReader, Vec<SparseTile>) {
        let n_tile_rows = n_variants.div_ceil(TILE_ROWS);
        let n_tile_cols = n_haps.div_ceil(TILE_COLS);
        let path = dir.join(format!("p{}.tiles", r.next()));
        let mut file = Vec::new();
        let mut entries = Vec::new();
        let mut tiles = Vec::new();
        for s in 0..n_tile_rows {
            for b in 0..n_tile_cols {
                let rows = TILE_ROWS.min(n_variants - s * TILE_ROWS);
                let cols = TILE_COLS.min(n_haps - b * TILE_COLS);
                let t = random_tile(r, rows, cols, per_mille);
                let raw = t.to_bytes();
                let comp = if lz4_every > 0 && r.below(lz4_every) == 0 {
                    lz4_flex::compress_prepend_size(&raw)
                } else {
                    zstd::encode_all(std::io::Cursor::new(&raw), 3).unwrap()
                };
                entries.push(TileEntryPub { offset: file.len() as u64, comp_size: comp.len() as u32 });
                file.extend_from_slice(&comp);
                tiles.push(t);
            }
        }
        std::fs::write(&path, &file).unwrap();
        (TiledSrpReader::from_entries(path, n_variants, n_haps, entries, n_tile_rows, n_tile_cols), tiles)
    }

    fn tile_bytes(n_rows: u16, indptr: Vec<u32>, indices: Vec<u16>) -> Vec<u8> {
        let n_cols = (indptr.len() - 1) as u16;
        SparseTile { indptr, indices, n_rows, n_cols }.to_bytes()
    }

    /// Valid tiles: rows ascend inside each column and may drop at a column
    /// boundary, empty columns included; repeated rows were accepted before
    /// and still are.
    #[test]
    fn from_bytes_accepts_valid_tiles() {
        for (indptr, indices) in [
            (vec![0u32, 0, 3, 3, 5, 6], vec![2u16, 5, 9, 1, 7, 0]),
            (vec![0, 3, 3], vec![0, 1, 1023]),
            (vec![0, 2, 4], vec![3, 3, 0, 1]),
            (vec![0, 0, 0], vec![]),
        ] {
            let t = SparseTile::from_bytes(&tile_bytes(1024, indptr.clone(), indices.clone()));
            assert_eq!((t.indptr, t.indices), (indptr, indices));
        }
    }

    /// A row past the tile: the old bitmatrix scan panicked on its index, the
    /// binary search would read REF; now decoding stops.
    #[test]
    #[should_panic(expected = "row index 1024 >= n_rows 1024 (corrupt tile)")]
    fn from_bytes_rejects_a_row_past_the_tile() {
        SparseTile::from_bytes(&tile_bytes(1024, vec![0, 1, 1], vec![1024]));
    }

    /// Rows out of order inside a column: the old scan found row 1, the binary
    /// search would not; now decoding stops.
    #[test]
    #[should_panic(expected = "rows not ascending within a column (corrupt tile)")]
    fn from_bytes_rejects_an_unsorted_column() {
        SparseTile::from_bytes(&tile_bytes(1024, vec![0, 2, 2], vec![1, 0]));
    }

    /// The same, after a boundary descent and an empty column.
    #[test]
    #[should_panic(expected = "rows not ascending within a column (corrupt tile)")]
    fn from_bytes_rejects_an_unsorted_column_after_a_boundary() {
        SparseTile::from_bytes(&tile_bytes(1024, vec![0, 2, 2, 5], vec![7, 9, 3, 8, 4]));
    }

    /// Through extraction too (Codex's case): extract row 0 of a tile holding
    /// row 1024 of a 1024-row stripe.
    #[test]
    #[should_panic(expected = "corrupt tile")]
    fn extraction_of_a_corrupt_tile_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.tiles");
        let comp = zstd::encode_all(std::io::Cursor::new(&tile_bytes(1024, vec![0, 1, 1], vec![1024])), 3).unwrap();
        std::fs::write(&path, &comp).unwrap();
        let entries = vec![TileEntryPub { offset: 0, comp_size: comp.len() as u32 }];
        let t = TiledSrpReader::from_entries(path, 1024, 2, entries, 1, 1);
        let _ = t.extract_bitmatrix(2, &[0]);
    }

    /// A decoder is kept only inside a scope and dropped with it: on this thread
    /// for `this_thread`, on every worker of the pool for `pool` (checked on a
    /// private pool, so tests running alongside cannot interfere).
    #[test]
    fn decoder_cache_lives_only_inside_a_scope() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0x5C0BE);
        let (t, tiles) = panel(dir.path(), &mut r, 3000, 300, 40, 0);
        let loaded = t.preload_stripes(0, t.n_tile_rows).unwrap();
        // No scope: decoded as before, nothing kept.
        assert!(same_tile(&loaded.decompress_tile(1, 0), &tiles[1]));
        assert!(!decoder_cached_here());
        // This thread's scope.
        {
            let _s = TileDecoderScope::this_thread();
            assert!(same_tile(&loaded.decompress_tile(2, 0), &tiles[2]));
            assert!(decoder_cached_here());
            {
                let _nested = TileDecoderScope::this_thread();
                loaded.decompress_tile(0, 0);
            }
            assert!(decoder_cached_here(), "an inner scope must not drop the outer one's decoder");
        }
        assert!(!decoder_cached_here());
        // A pool scope, on a private pool.
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        pool.install(|| {
            let s = TileDecoderScope::pool();
            let got: Vec<SparseTile> = (0..t.n_tile_rows).into_par_iter().map(|i| loaded.decompress_tile(i, 0)).collect();
            assert!(got.iter().zip(&tiles).all(|(a, b)| same_tile(a, b)));
            assert!(rayon::broadcast(|_| decoder_cached_here()).iter().any(|&c| c));
            drop(s);
            assert!(rayon::broadcast(|_| decoder_cached_here()).iter().all(|&c| !c));
        });
    }

    /// Compressed bytes in memory during extraction (the read being decoded,
    /// plus the next one when it is read early) never exceed the previous
    /// scheme's single largest read.
    fn check_plan(stripes: &[usize], sizes: &[usize], prev: usize, cap: usize) -> ExtractReads {
        let prefix: Vec<usize> = std::iter::once(0).chain(sizes.iter().scan(0, |a, &x| { *a += x; Some(*a) })).collect();
        let read_len = |fs: usize, n: usize| prefix[fs + n] - prefix[fs];
        let p = plan_extract_reads(stripes, read_len, prev, cap);
        // The previous scheme, verbatim: chunks of `prev` stripes-with-sites.
        let old_peak = stripes.chunks(prev).map(|c| read_len(c[0], c[c.len() - 1] - c[0] + 1)).max().unwrap_or(0);
        assert_eq!(p.bound, old_peak);
        let mut next = 0;
        for (k, r) in p.reads.iter().enumerate() {
            assert_eq!(r.start, next);
            assert!(!r.is_empty() && r.len() <= prev);
            next = r.end;
            assert_eq!(p.bytes[k], read_len(stripes[r.start], stripes[r.end - 1] - stripes[r.start] + 1));
            if r.len() > 1 { assert!(p.bytes[k] <= cap.min(old_peak / 2)); }
            let live = p.bytes[k] + if p.prefetch[k] { p.bytes[k + 1] } else { 0 };
            assert!(live <= old_peak, "read {k}: {live} bytes live > {old_peak}");
        }
        assert_eq!(next, stripes.len());
        p
    }

    #[test]
    fn extraction_reads_stay_within_the_previous_memory() {
        // Codex's case: 800 stripes of 128 KiB, one wanted row each. The previous
        // scheme read 400 at a time: 50 MiB.
        let stripes: Vec<usize> = (0..800).collect();
        let p = check_plan(&stripes, &vec![128 << 10; 800], 400, 64 << 20);
        assert_eq!(p.bound, 50 << 20);
        assert!(p.prefetch.iter().any(|&x| x), "should still read ahead where it fits");
        // Random panels: gaps between wanted stripes, stripe sizes over three
        // orders of magnitude, a few huge ones.
        let mut r = Xs(0xB0D6E7);
        for _ in 0..2000 {
            let n_stripes = 1 + r.below(3000) as usize;
            let sizes: Vec<usize> = (0..n_stripes).map(|_| match r.below(20) {
                0 => 1 + r.below(64 << 20) as usize, _ => 1 + r.below(2 << 20) as usize }).collect();
            let density = 1 + r.below(100);
            let stripes: Vec<usize> = (0..n_stripes).filter(|_| r.below(100) < density).collect();
            if stripes.is_empty() { continue; }
            let prev = [1usize, 3, 7, 400][r.below(4) as usize];
            let cap = [64 << 10, 4 << 20, 64 << 20][r.below(3) as usize];
            check_plan(&stripes, &sizes, prev, cap);
        }
    }

    fn same_tile(a: &SparseTile, b: &SparseTile) -> bool {
        a.indptr == b.indptr && a.indices == b.indices && a.n_rows == b.n_rows && a.n_cols == b.n_cols
    }

    #[test]
    fn decompress_tile_matches_previous_decoder() {
        // Reused decoders on this thread and on the pool, as the stages use them.
        let _here = TileDecoderScope::this_thread();
        let _pool = TileDecoderScope::pool();
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0xDEC0DE);
        // Dense tiles exceed the first-guess buffer (8x the compressed size,
        // 1 MB minimum): the slow path runs, then the grown buffer serves the rest.
        let slow0 = ZSTD_SLOW_PATH.load(std::sync::atomic::Ordering::Relaxed);
        // (1024, 4096, 1000), first on this thread: an all-ones tile, 8.4 MB raw
        // from a few KB.
        for (n_variants, n_haps, per_mille) in [(1024, 4096, 1000), (3000, 300, 40), (2100, 4096 + 900, 120), (777, 9000, 2)] {
            let lz4_every = if per_mille == 1000 { 0 } else { 4 };
            let (t, tiles) = panel(dir.path(), &mut r, n_variants, n_haps, per_mille, lz4_every);
            let loaded = t.preload_stripes(0, t.n_tile_rows).unwrap();
            for s in 0..t.n_tile_rows {
                for b in 0..t.n_tile_cols {
                    let new = loaded.decompress_tile(s, b);
                    let old = old_decompress_tile(&loaded, s, b);
                    assert!(same_tile(&new, &old) && same_tile(&new, &tiles[s * t.n_tile_cols + b]));
                }
            }
            // From fresh threads too (empty per-thread state), concurrently.
            let fresh: Vec<SparseTile> = (0..t.n_tile_rows * t.n_tile_cols).into_par_iter()
                .map(|i| loaded.decompress_tile(i / t.n_tile_cols, i % t.n_tile_cols)).collect();
            assert!(fresh.iter().zip(&tiles).all(|(a, b)| same_tile(a, b)));
        }
        assert!(ZSTD_SLOW_PATH.load(std::sync::atomic::Ordering::Relaxed) > slow0, "slow path not exercised");
    }

    #[test]
    fn extract_bitmatrix_matches_previous_scan() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0xB17);
        for (n_variants, n_haps, per_mille) in [(5000, 300, 50), (2500, 4096 + 333, 20), (9000, 64, 300)] {
            let (t, _) = panel(dir.path(), &mut r, n_variants, n_haps, per_mille, 0);
            // Wanted rows per stripe from one (binary search) to many (scan),
            // with duplicate sites; sorted as the callers pass them, or not.
            for per_stripe in [1usize, 2, 3, 7, 40] {
                let mut idx = Vec::new();
                for s in 0..t.n_tile_rows {
                    let rows = TILE_ROWS.min(n_variants - s * TILE_ROWS);
                    for _ in 0..per_stripe.min(rows) { idx.push(s * TILE_ROWS + r.below(rows as u64) as usize); }
                }
                if r.below(2) == 0 { idx.sort_unstable(); }
                for _ in 0..idx.len() / 20 { let k = r.below(idx.len() as u64) as usize; idx.push(idx[k]); }
                let new = t.extract_bitmatrix(n_haps, &idx);
                let old = old_extract_bitmatrix(&t, n_haps, &idx);
                for ci in 0..idx.len() {
                    for h in 0..n_haps { assert_eq!(new.get(ci, h), old.get(ci, h), "site {ci} hap {h}"); }
                }
            }
        }
    }
}
