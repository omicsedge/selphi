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
    /// Identity of the file read, taken before reading (`retain::put`).
    pub(crate) key: Option<super::retain::FileKey>,
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

    /// For budgeting the decode of stripes `first..=last` before it runs: upper bounds
    /// (`decoded_bound`) on what their tiles decode to, in total and for the largest
    /// one, and the largest compressed tile.
    pub fn decoded_bound(&self, first: usize, last: usize) -> (usize, usize, usize) {
        (first..=last).flat_map(|s| (0..self.n_tile_cols).map(move |b| (s, b)))
            .map(|(s, b)| {
                let (off, len) = self.offsets[(s - self.first_stripe) * self.n_tile_cols + b];
                (decoded_bound(&self.buf[off..off + len]), len)
            })
            .fold((0, 0, 0), |(t, m, c), (x, l)| (t + x, m.max(x), c.max(l)))
    }

    /// The capacity a worker's reused decode buffer may reach decoding a tile of
    /// `comp` compressed bytes that decodes to at most `bound` (`zstd_tile_fast`'s
    /// first guess, else the slow path's regrowth, kept up to `ZSTD_KEEP_BUF`).
    pub fn decoder_buffer_bound(comp: usize, bound: usize) -> usize {
        (comp * 8).max(1 << 20).max((bound + bound / 4).min(ZSTD_KEEP_BUF))
    }

    /// Check if a stripe is within this preloaded range.
    #[inline]
    pub fn contains_stripe(&self, stripe: usize) -> bool {
        stripe >= self.first_stripe && stripe < self.first_stripe + self.n_stripes
    }

    /// Decode one tile and call `f` with it read in place (no `SparseTile`): the
    /// decoded bytes, checked as `SparseTile::from_bytes` checks them (same
    /// panics), viewed as words, with `TILE_PAD` zero bytes after the row indices.
    /// Within a decoder scope the thread's reused buffer holds it; otherwise, or
    /// when that buffer does not fit or the tile is LZ4 or not aligned, an owned
    /// buffer does.
    pub fn with_tile<R>(&self, stripe: usize, band: usize, f: impl FnOnce(&TileView<'_>) -> R) -> R {
        let local = (stripe - self.first_stripe) * self.n_tile_cols + band;
        let (off, len) = self.offsets[local];
        let compressed = &self.buf[off..off + len];
        let zstd = compressed.len() >= 4 && compressed[0] == 0x28 && compressed[1] == 0xB5 && compressed[2] == 0x2F && compressed[3] == 0xFD;
        let mut f = Some(f);
        let cached = zstd && decoder_cache_enabled();
        if cached {
            let done = ZSTD_TILE_DECODER.with(|cell| {
                // `f` does no decoding of its own, so this borrow is not re-entered.
                let mut cell = cell.borrow_mut();
                if cell.is_none() {
                    *cell = Some(ZstdTileDecoder { dec: zstd::bulk::Decompressor::new().ok()?, buf: Vec::new() });
                }
                let d = cell.as_mut().unwrap();
                d.buf.clear();
                let want = (compressed.len() * 8).max(1 << 20);
                if d.buf.capacity() < want { d.buf.reserve(want); }
                d.dec.decompress_to_buffer(compressed, &mut d.buf).ok()?;
                let raw_len = d.buf.len();
                d.buf.resize(raw_len + TILE_PAD, 0);
                let view = TileView::parse(&d.buf, raw_len)?;
                Some((f.take().unwrap())(&view))
            });
            if let Some(r) = done { return r; }
        }
        let raw = if zstd {
            let raw = zstd::decode_all(std::io::Cursor::new(compressed)).unwrap_or_else(|e|
                panic!("zstd decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e));
            if cached { zstd_tile_grow(raw.len()); }
            raw
        } else {
            lz4_flex::decompress_size_prepended(compressed).unwrap_or_else(|e|
                panic!("LZ4 decompress failed at stripe={} band={} (comp_len={}): {}", stripe, band, len, e))
        };
        // An owned, word-aligned copy (a Vec<u64>'s bytes) with the padding.
        let raw_len = raw.len();
        let mut words = vec![0u64; (raw_len + TILE_PAD).div_ceil(8)];
        // SAFETY: `words` holds at least raw_len + TILE_PAD bytes.
        let bytes = unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut u8, words.len() * 8) };
        bytes[..raw_len].copy_from_slice(&raw);
        match TileView::parse(bytes, raw_len) {
            Some(view) => (f.take().unwrap())(&view),
            None => {
                // Big-endian host: the tile as decoded today, re-serialised
                // little-endian is what TileView reads, so read SparseTile instead.
                let t = super::SparseTile::from_bytes(&raw);
                let mut ix: Vec<u16> = t.indices.clone();
                ix.resize(t.indices.len() + TILE_PAD / 2, 0);
                let view = TileView { n_rows: t.n_rows, n_cols: t.n_cols as usize, nnz: t.indices.len(),
                    indptr: &t.indptr, indices: &ix };
                (f.take().unwrap())(&view)
            }
        }
    }
}

/// An upper bound on the bytes the compressed tile `c` decodes to, from its headers
/// alone. LZ4 (`lz4_flex` size-prepended): the stored size. zstd: over every frame,
/// the sum over its blocks of a raw or RLE block's size and, for a compressed block,
/// the largest a block may decode to (the window size, at most 128 KiB). Anything
/// unreadable: the largest tile a writer makes (`TILE_ROWS` x `TILE_COLS` two-byte row
/// indices plus the column pointers).
pub fn decoded_bound(c: &[u8]) -> usize {
    const MAX_TILE: usize = 8 + 4 * (TILE_COLS + 1) + 2 * TILE_ROWS * TILE_COLS;
    const ZSTD: [u8; 4] = [0x28, 0xB5, 0x2F, 0xFD];
    if c.len() < 4 || c[..4] != ZSTD {
        return if c.len() >= 4 { u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize } else { MAX_TILE };
    }
    // Every frame in turn (a decoder reads concatenated frames): zstd frames, and
    // skippable frames (magic 0x184D2A5?, a 4-byte size), which decode to nothing.
    let zstd = || -> Option<usize> {
        let (mut at, mut total) = (0usize, 0usize);
        while at < c.len() {
            let magic = u32::from_le_bytes(c.get(at..at + 4)?.try_into().ok()?);
            if magic & 0xFFFF_FFF0 == 0x184D_2A50 {
                let n = u32::from_le_bytes(c.get(at + 4..at + 8)?.try_into().ok()?) as usize;
                at = at.checked_add(8 + n)?;
                continue;
            }
            if magic != 0xFD2F_B528 { return None; }
            let fhd = *c.get(at + 4)?;
            let (fcs_flag, single, checksum, did_flag) = (fhd >> 6, (fhd >> 5) & 1 == 1, (fhd >> 2) & 1 == 1, fhd & 3);
            at += 5;
            let window = if single { None } else {
                let wd = *c.get(at)?;
                at += 1;
                let base = 1u64 << (10 + (wd >> 3));
                Some(base + (base / 8) * (wd & 7) as u64)
            };
            at += [0, 1, 2, 4][did_flag as usize];
            at += match (fcs_flag, single) { (0, false) => 0, (0, true) => 1, (1, _) => 2, (2, _) => 4, _ => 8 };
            // Blocks: a raw or RLE block's own size; a compressed one at most the block
            // maximum (the window size, at most 128 KiB). (A content-size field, when
            // present, is not trusted over the blocks.)
            let block_max = window.map_or(128 << 10, |w| w.min(128 << 10)) as usize;
            loop {
                let h = c.get(at..at + 3)?;
                let h = h[0] as u32 | (h[1] as u32) << 8 | (h[2] as u32) << 16;
                let (last, kind, size) = (h & 1 == 1, (h >> 1) & 3, (h >> 3) as usize);
                at += 3;
                match kind {
                    0 => { total = total.checked_add(size)?; at = at.checked_add(size)?; }
                    1 => { total = total.checked_add(size)?; at += 1; }
                    2 => { total = total.checked_add(block_max)?; at = at.checked_add(size)?; }
                    _ => return None,
                }
                if last { break; }
            }
            if checksum { at += 4; }
        }
        Some(total)
    };
    zstd().unwrap_or(MAX_TILE)
}

/// Zero bytes `PreloadedStripes::with_tile` puts after a tile's row indices, so
/// a 32-row window loaded at any column start stays inside the buffer.
pub const TILE_PAD: usize = 64;

/// A decoded tile read in place (see `PreloadedStripes::with_tile`).
pub struct TileView<'a> {
    pub n_rows: u16,
    pub n_cols: usize,
    pub nnz: usize,
    /// Column pointers (`n_cols + 1`).
    pub indptr: &'a [u32],
    /// Row indices (`nnz`), then at least `TILE_PAD / 2` zeros.
    pub indices: &'a [u16],
}

impl<'a> TileView<'a> {
    /// The tile serialised in `buf[..raw_len]` (`TILE_PAD` more bytes readable),
    /// checked as `SparseTile::from_bytes` checks it. `None` if the host or the
    /// buffer cannot be read as little-endian words in place.
    fn parse(buf: &'a [u8], raw_len: usize) -> Option<Self> {
        let data = &buf[..raw_len];
        let (n_rows, n_cols, nnz, indptr_end) = super::tile_layout(data);
        let indptr = super::cast_le::<u32>(&data[8..indptr_end])?;
        let padded_end = (indptr_end + nnz * 2 + TILE_PAD).min(buf.len()) & !1;
        let indices = super::cast_le::<u16>(&buf[indptr_end..padded_end])?;
        if indices.len() < nnz + TILE_PAD / 2 { return None; }
        super::check_tile(n_rows, nnz, indptr, &indices[..nnz]);
        Some(TileView { n_rows, n_cols, nnz, indptr, indices })
    }
}

// ============================================================================
// Reader: file-based tiled SRP reader (sequential I/O, no mmap)
// ============================================================================

/// `extract_bitmatrix`: stripes with at most this many distinct wanted rows
/// compare each column's rows with every wanted row (one 32-row window per
/// column, `scan_compare`); stripes with more read every nonzero once against a
/// table of the wanted rows (`scan_table`). Measured on the 75,552-haplotype
/// panel's tiles: ~8 us per wanted row per tile against ~80 us for the pass.
const EXTRACT_COMPARE_MAX_ROWS: usize = 8;

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
    /// Reads `k` and `k + 1` fit `bound` together (either may be read while the
    /// other is decoded; `extract_bitmatrix` checks the pair it visits in turn).
    #[cfg_attr(not(test), allow(dead_code))]
    prefetch: Vec<bool>,
    /// Compressed bytes the previous scheme held at its largest read.
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

/// Refuse a tile index read from a file whose entries run past the file or run
/// backwards (`offset`, `comp_size` pairs in tile order). The stripe reads size
/// their buffer by `offset + comp_size` differences between entries, so a corrupt
/// index used to ask for a wrapped length; every writer lays the tiles out in
/// index order, so a valid index passes.
pub fn check_tile_index(entries: impl Iterator<Item = (u64, u32)>, file_len: u64, path: &Path) -> io::Result<()> {
    let mut prev = 0u64;
    for (i, (offset, comp_size)) in entries.enumerate() {
        let end = offset.checked_add(comp_size as u64).filter(|&e| e <= file_len);
        if end.is_none() || offset < prev {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!(
                "corrupt SRP {}: tile {i} at offset {offset} (+{comp_size} bytes) lies outside the {file_len}-byte \
                 file or before the tile before it; regenerate the panel", path.display())));
        }
        prev = offset;
    }
    Ok(())
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
        check_tile_index(tile_index.iter().map(|e| (e.offset, e.comp_size)), file_len, path)?;

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

    /// Compressed bytes of every tile.
    pub fn tile_bytes(&self) -> u64 {
        self.tile_index.iter().map(|e| e.comp_size as u64).sum()
    }

    /// File offset of stripe `stripe`'s first tile (the file length past the last stripe).
    pub fn stripe_offset(&self, stripe: usize) -> u64 {
        match self.tile_index.get(stripe * self.n_tile_cols) {
            Some(e) => e.offset,
            None => self.tile_index.last().map_or(0, |e| e.offset + e.comp_size as u64),
        }
    }

    /// The panel file's identity, for `retain`.
    pub fn file_key(&self) -> Option<super::retain::FileKey> {
        super::retain::FileKey::of_path(&self.file_path)
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

        let file = std::fs::File::open(&self.file_path)?;
        if buf.capacity() < read_len {
            buf = vec![0u8; read_len];
        } else {
            buf.clear();
            buf.resize(read_len, 0);
        }
        // Spans held in RAM for these reads (`retain`) are copied from there; only the
        // rest is read from the file. Same bytes.
        let key = super::retain::FileKey::of(&file);
        let holes = if super::retain::held_bytes() > 0 { super::retain::fill(key, read_start, &mut buf) } else { vec![0..read_len] };
        for h in holes {
            read_file_at(&file, &mut buf[h.clone()], read_start + h.start as u64)?;
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
            key,
        })
    }

    /// Extract reference alleles for `wgs_idx` (chip variant indices) as a
    /// haplotype bitmatrix, reading directly from the compressed tiles. Shared
    /// by `SrpReader::extract_ref_alleles_bitmatrix` and
    /// `ChrSrpView::extract_ref_alleles_bitmatrix` (each resolves its own tiled
    /// backend and supplies `n_haps` from its metadata).
    pub fn extract_bitmatrix(&self, n_haps: usize, wgs_idx: &[usize]) -> crate::common::HaplotypeBitmatrix {
        self.extract_bitmatrix_retaining(n_haps, wgs_idx, 0, u64::MAX)
    }

    /// `extract_bitmatrix`, leaving the head-most spans it read — up to
    /// `retain_bytes` of them — held in `retain` for the interpolation's reads. On a
    /// cache-limited run (`retain::limited`) the held spans and every read past the
    /// next `keep_cached` bytes (head first) are dropped from the page cache once
    /// decoded: the cache then keeps the reads the interpolation makes first.
    pub fn extract_bitmatrix_retaining(&self, n_haps: usize, wgs_idx: &[usize], retain_bytes: usize,
                                       keep_cached: u64) -> crate::common::HaplotypeBitmatrix {
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

        // Visit order (`extract_tail_first`), and the reads left held for the
        // interpolation: the head-most ones within `retain_bytes`.
        let n_reads = plan.reads.len();
        let order: Vec<usize> = if extract_tail_first() { (0..n_reads).rev().collect() } else { (0..n_reads).collect() };
        let n_keep = if retain_bytes == 0 { 0 } else {
            plan.bytes.iter().scan(0usize, |acc, &b| { *acc += b; Some(*acc) })
                .take_while(|&acc| acc <= retain_bytes).count()
        };
        // The file held spans and page-cache drops refer to (open while used).
        let pin = if n_keep > 0 || super::retain::limited() { std::fs::File::open(&self.file_path).ok() } else { None };
        // Compressed bytes of the reads before read k (head first), for `keep_cached`.
        let before: Vec<u64> = std::iter::once(0u64)
            .chain(plan.bytes.iter().scan(0u64, |a, &b| { *a += b as u64; Some(*a) })).collect();
        let held_end = before[n_keep];

        std::thread::scope(|sc| {
        let mut next: Option<std::thread::ScopedJoinHandle<'_, io::Result<PreloadedStripes>>> = None;
        for (i, &k) in order.iter().enumerate() {
            let batch = &sorted[plan.reads[k].clone()];
            let loaded = match next.take() {
                Some(h) => h.join().expect("stripe read thread panicked"),
                None => { let (fs, n, buf) = read_args(k); self.preload_stripes_into(fs, n, buf) }
            }.expect("preload failed");
            // Read the next while this one is decoded, when both fit together.
            if let Some(&k2) = order.get(i + 1) && plan.bytes[k] + plan.bytes[k2] <= plan.bound {
                let (fs, n, buf) = read_args(k2);
                next = Some(sc.spawn(move || self.preload_stripes_into(fs, n, buf)));
            }

            batch.par_iter().for_each(|(stripe, sites)| {
                // Stripes are row-disjoint across workers, so no two workers touch the
                // same word; write through the raw base pointer instead of materializing
                // a full-length &mut per worker (overlapping &mut = aliasing UB).
                let base = bits_ptr as *mut u64;
                // Per stripe: its distinct wanted rows (ascending) and, per row, every
                // chip index that maps to it (duplicate sites; rare). The scan yields,
                // per wanted row and 64-haplotype word of a tile, the haplotypes carrying
                // it; `|=` into each of the row's chip rows sets exactly the bits the
                // per-nonzero scan set (order-independent, idempotent).
                let mut rows: Vec<u16> = sites.iter().map(|&(_, cr)| cr as u16).collect();
                rows.sort_unstable();
                rows.dedup();
                let mut chips: Vec<Vec<usize>> = vec![Vec::new(); rows.len()];
                for &(ci, cr) in sites.iter() {
                    let k = rows.binary_search(&(cr as u16)).unwrap();
                    chips[k].push(ci);
                }
                for band in 0..n_tc {
                    let cb = band * TILE_COLS;
                    loaded.with_tile(*stripe, band, |tile| {
                        // Columns past the panel's last haplotype are not read.
                        let n_cols = tile.n_cols.min(n_haps.saturating_sub(cb));
                        let mut emit = |k: usize, w: usize, bits: u64| {
                            let wi = cb / 64 + w;
                            for &ci in &chips[k] {
                                unsafe { let p = base.add(ci * n_words + wi); *p |= bits; }
                            }
                        };
                        if tile.n_rows as usize > TILE_ROWS {
                            // Not a tile any writer makes (rows past the stripe pass the
                            // checks only here): read exactly as before, panics included.
                            scan_oversized(tile, n_cols, &rows, &mut emit);
                        } else if rows.len() <= EXTRACT_COMPARE_MAX_ROWS {
                            scan_compare(tile, n_cols, &rows, &mut emit);
                        } else {
                            scan_table(tile, n_cols, &rows, &mut emit);
                        }
                    });
                }
            });
            // Held: in RAM now. Past the cache window: the cache cannot keep it until
            // the interpolation reads it. Either way its page-cache copy only crowds
            // out the reads after the held ones (a no-op unless the run is limited).
            let offset = self.tile_index[loaded.first_stripe * n_tc].offset;
            let held = k < n_keep;
            if let Some(f) = &pin {
                if held || before[k + 1] > held_end.saturating_add(keep_cached) {
                    super::retain::drop_cached(f, offset, offset + plan.bytes[k] as u64);
                }
                // (Copied into an anonymous mapping: any thread may release it.)
                if held && let Some(key) = loaded.key && let Some(d) = super::retain::SpanData::copy_of(&loaded.buf) {
                    super::retain::put(f, key, offset, d);
                }
            }
        }
        });
        crate::common::HaplotypeBitmatrix::from_raw(bits, n_chip, n_haps)
    }
}

/// Read `buf.len()` bytes of `file` from `offset`: one sequential read, a large one
/// in up to four contiguous parts read concurrently (from the page cache a single
/// reader copies ~2 GB/s, four ~7 GB/s on the 75,552-haplotype panel). Same bytes.
pub(crate) fn read_file_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        const PART: usize = 32 << 20;
        let read_len = buf.len();
        let k = (read_len / PART).min(4);
        if k >= 2 {
            let chunk = read_len.div_ceil(k);
            std::thread::scope(|sc| -> io::Result<()> {
                let parts: Vec<_> = buf.chunks_mut(chunk).enumerate().map(|(i, part)| {
                    sc.spawn(move || file.read_exact_at(part, offset + (i * chunk) as u64))
                }).collect();
                for h in parts { h.join().expect("stripe read thread panicked")?; }
                Ok(())
            })
        } else {
            file.read_exact_at(buf, offset)
        }
    }
    #[cfg(not(unix))]
    {
        use std::io::{Seek, SeekFrom, Read};
        let mut file = file;
        file.seek(SeekFrom::Start(offset))?;
        file.read_exact(buf)
    }
}

/// `extract_bitmatrix` visits its reads from the file's tail to its head. A panel
/// just downloaded into a container too small to cache it leaves its TAIL in the
/// page cache (the download evicted the head), and reading head first evicted that
/// tail before reaching it: one user x the 75,552-haplotype chr2 panel in a 6 GiB
/// cgroup read all 7.07 GB of its spans from disk. Ending at the head also leaves
/// the head, where the interpolation starts, as the most recently read part.
/// `SELPHI_EXTRACT_ORDER=head` reads head first, as before. The bitmatrix is the same
/// either way (each stripe sets its own rows).
fn extract_tail_first() -> bool {
    crate::config::raw("SELPHI_EXTRACT_ORDER").as_deref() != Some("head")
}

/// For every wanted row `rows[k]` (at most 64, ascending) and 64-column word `w`
/// of the first `n_cols` columns of `tile`: `emit(k, w, bits)` with bit `c` set
/// when column `64 w + c` holds the row (only non-zero `bits` are emitted). Each
/// column's rows are read as one 32-entry window (the view's padding keeps it in
/// bounds; lanes past the column are masked) compared with every wanted row.
fn scan_compare(tile: &TileView<'_>, n_cols: usize, rows: &[u16], emit: &mut impl FnMut(usize, usize, u64)) {
    debug_assert!(!rows.is_empty() && rows.len() <= 64 && n_cols <= tile.n_cols);
    // `SELPHI_FORCE_SCALAR=1` keeps every SIMD kernel scalar (parity checks).
    static SCALAR: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *SCALAR.get_or_init(|| crate::config::is_one("SELPHI_FORCE_SCALAR")) {
        return scan_compare_scalar(tile, n_cols, rows, emit);
    }
    let k = rows.len();
    let mut acc = [0u64; 64];
    let acc = &mut acc[..k];
    let ip = tile.indptr;
    let idx = tile.indices;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    // SAFETY: every window starts at lo <= nnz and idx holds nnz + TILE_PAD/2 >= nnz + 32 entries.
    unsafe {
        use std::arch::x86_64::*;
        let mut bc = [_mm256_setzero_si256(); 64];
        for (j, &r) in rows.iter().enumerate() { bc[j] = _mm256_set1_epi16(r as i16); }
        let p = idx.as_ptr();
        for col in 0..n_cols {
            let lo = *ip.get_unchecked(col) as usize;
            let n = *ip.get_unchecked(col + 1) as usize - lo;
            let sh = col & 63;
            if n <= 32 {
                let v0 = _mm256_loadu_si256(p.add(lo) as *const __m256i);
                let v1 = _mm256_loadu_si256(p.add(lo + 16) as *const __m256i);
                let lm: u64 = if n == 32 { !0 } else { (1u64 << (2 * n)) - 1 };
                for j in 0..k {
                    let b = *bc.get_unchecked(j);
                    let m0 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v0, b)) as u32 as u64;
                    let m1 = _mm256_movemask_epi8(_mm256_cmpeq_epi16(v1, b)) as u32 as u64;
                    *acc.get_unchecked_mut(j) |= ((((m1 << 32) | m0) & lm != 0) as u64) << sh;
                }
            } else {
                let s = &idx[lo..lo + n];
                for j in 0..k { acc[j] |= (s.contains(&rows[j]) as u64) << sh; }
            }
            if sh == 63 || col + 1 == n_cols {
                for j in 0..k { if acc[j] != 0 { emit(j, col >> 6, acc[j]); acc[j] = 0; } }
            }
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above (NEON is part of the aarch64 baseline).
    unsafe {
        use std::arch::aarch64::*;
        let lane: [u16; 32] = std::array::from_fn(|i| i as u16);
        let l0 = vld1q_u16(lane.as_ptr());
        let l1 = vld1q_u16(lane.as_ptr().add(8));
        let l2 = vld1q_u16(lane.as_ptr().add(16));
        let l3 = vld1q_u16(lane.as_ptr().add(24));
        let p = idx.as_ptr();
        for col in 0..n_cols {
            let lo = *ip.get_unchecked(col) as usize;
            let n = *ip.get_unchecked(col + 1) as usize - lo;
            let sh = col & 63;
            if n <= 32 {
                let nv = vdupq_n_u16(n as u16);
                let m0 = vcltq_u16(l0, nv);
                let m1 = vcltq_u16(l1, nv);
                let m2 = vcltq_u16(l2, nv);
                let m3 = vcltq_u16(l3, nv);
                let v0 = vld1q_u16(p.add(lo));
                let v1 = vld1q_u16(p.add(lo + 8));
                let v2 = vld1q_u16(p.add(lo + 16));
                let v3 = vld1q_u16(p.add(lo + 24));
                for j in 0..k {
                    let b = vdupq_n_u16(*rows.get_unchecked(j));
                    let e = vorrq_u16(
                        vorrq_u16(vandq_u16(vceqq_u16(v0, b), m0), vandq_u16(vceqq_u16(v1, b), m1)),
                        vorrq_u16(vandq_u16(vceqq_u16(v2, b), m2), vandq_u16(vceqq_u16(v3, b), m3)));
                    *acc.get_unchecked_mut(j) |= ((vmaxvq_u16(e) != 0) as u64) << sh;
                }
            } else {
                let s = &idx[lo..lo + n];
                for j in 0..k { acc[j] |= (s.contains(&rows[j]) as u64) << sh; }
            }
            if sh == 63 || col + 1 == n_cols {
                for j in 0..k { if acc[j] != 0 { emit(j, col >> 6, acc[j]); acc[j] = 0; } }
            }
        }
        return;
    }
    #[allow(unreachable_code)]
    scan_compare_scalar(tile, n_cols, rows, emit)
}

/// Portable `scan_compare` (also its test oracle and the `SELPHI_FORCE_SCALAR` path).
fn scan_compare_scalar(tile: &TileView<'_>, n_cols: usize, rows: &[u16], emit: &mut impl FnMut(usize, usize, u64)) {
    let k = rows.len();
    let mut acc = [0u64; 64];
    for col in 0..n_cols {
        let (lo, hi) = (tile.indptr[col] as usize, tile.indptr[col + 1] as usize);
        let s = &tile.indices[lo..hi];
        let sh = col & 63;
        for j in 0..k { acc[j] |= (s.contains(&rows[j]) as u64) << sh; }
        if sh == 63 || col + 1 == n_cols {
            for j in 0..k { if acc[j] != 0 { emit(j, col >> 6, acc[j]); acc[j] = 0; } }
        }
    }
}

/// The scan `extract_bitmatrix` ran before `scan_compare` / `scan_table`, for a
/// tile declaring more than TILE_ROWS rows: binary search per column for at most
/// two wanted rows, else a pass over every nonzero tested against a TILE_ROWS-bit
/// table (which panics on a row past it, as it always did).
fn scan_oversized(tile: &TileView<'_>, n_cols: usize, rows: &[u16], emit: &mut impl FnMut(usize, usize, u64)) {
    let mut want = [0u64; TILE_ROWS / 64];
    for &r in rows { want[r as usize / 64] |= 1u64 << (r % 64); }
    let search = rows.len() <= 2;
    for col in 0..n_cols {
        let (lo, hi) = (tile.indptr[col] as usize, tile.indptr[col + 1] as usize);
        let bit = 1u64 << (col % 64);
        if search {
            let col_rows = &tile.indices[lo..hi];
            for (k, r) in rows.iter().enumerate() {
                if col_rows.binary_search(r).is_ok() { emit(k, col / 64, bit); }
            }
        } else {
            for kk in lo..hi {
                let lr = tile.indices[kk] as usize;
                if (want[lr / 64] >> (lr % 64)) & 1 == 0 { continue; }
                emit(rows.binary_search(&(lr as u16)).unwrap(), col / 64, bit);
            }
        }
    }
}

/// `scan_compare` for any number of wanted rows: one pass over every nonzero,
/// looking each row up in a table of the wanted rows' slots.
fn scan_table(tile: &TileView<'_>, n_cols: usize, rows: &[u16], emit: &mut impl FnMut(usize, usize, u64)) {
    let k = rows.len();
    let mut slot = [u16::MAX; TILE_ROWS];
    for (j, &r) in rows.iter().enumerate() { slot[r as usize] = j as u16; }
    let mut acc = vec![0u64; k];
    let mut touched: Vec<u16> = Vec::with_capacity(k);
    for col in 0..n_cols {
        let (lo, hi) = (tile.indptr[col] as usize, tile.indptr[col + 1] as usize);
        let bit = 1u64 << (col & 63);
        // Rows are < n_rows <= TILE_ROWS (checked when the tile was read).
        for &r in &tile.indices[lo..hi] {
            let j = slot[r as usize];
            if j == u16::MAX { continue; }
            if acc[j as usize] == 0 { touched.push(j); }
            acc[j as usize] |= bit;
        }
        if col & 63 == 63 || col + 1 == n_cols {
            for &j in &touched { emit(j as usize, col >> 6, acc[j as usize]); acc[j as usize] = 0; }
            touched.clear();
        }
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

    /// Extraction boundary case: extract row 0 of a tile holding
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
        // 800 stripes of 128 KiB, one wanted row each. Reading 400 at a time
        // requires 50 MiB.
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
            for per_stripe in [1usize, 2, 3, 7, 8, 9, 12, 40] {
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

    /// An all-ones 1024 x 4096 tile, serialised (8 MB).
    fn tiles_bytes_dense() -> Vec<u8> {
        let mut r = Xs(1);
        random_tile(&mut r, TILE_ROWS, TILE_COLS, 1000).to_bytes()
    }

    /// `decoded_bound` is at least what every tile decodes to (zstd at several
    /// densities, all-ones and empty tiles, LZ4), and close to it for typical tiles.
    #[test]
    fn decoded_bound_covers_every_tile() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0xB0B0);
        for (n_variants, n_haps, per_mille) in [(1024, 4096, 1000), (1024, 4096, 500), (3000, 300, 40),
                                                (2100, 4096 + 900, 120), (777, 9000, 2), (1024, 64, 0)] {
            let (t, tiles) = panel(dir.path(), &mut r, n_variants, n_haps, per_mille, 3);
            let loaded = t.preload_stripes(0, t.n_tile_rows).unwrap();
            for st in 0..t.n_tile_rows {
                for b in 0..t.n_tile_cols {
                    let raw = tiles[st * t.n_tile_cols + b].to_bytes().len();
                    let bound = loaded.decoded_bound(st, st).0 - (0..b).chain(b + 1..t.n_tile_cols)
                        .map(|bb| { let (o, l) = loaded.offsets[st * t.n_tile_cols + bb]; decoded_bound(&loaded.buf[o..o + l]) }).sum::<usize>();
                    assert!(bound >= raw, "stripe {st} band {b}: bound {bound} < {raw}");
                    assert!(bound <= raw + (128 << 10) * (raw.div_ceil(128 << 10) + 1), "stripe {st} band {b}: bound {bound} for {raw}");
                }
            }
        }
        // Concatenated frames (an empty one first, a skippable one, two halves): the
        // bound covers them all.
        let big = tiles_bytes_dense();
        let (a, b) = big.split_at(big.len() / 2);
        let mut cat = zstd::encode_all(std::io::Cursor::new(&[][..]), 3).unwrap();
        cat.extend_from_slice(&[0x50, 0x2A, 0x4D, 0x18, 3, 0, 0, 0, 1, 2, 3]);
        cat.extend(zstd::encode_all(std::io::Cursor::new(a), 3).unwrap());
        cat.extend(zstd::encode_all(std::io::Cursor::new(b), 3).unwrap());
        assert_eq!(zstd::decode_all(std::io::Cursor::new(&cat)).unwrap(), big);
        assert!(decoded_bound(&cat) >= big.len(), "{} < {}", decoded_bound(&cat), big.len());
        assert_eq!(decoded_bound(&[]), 8 + 4 * (TILE_COLS + 1) + 2 * TILE_ROWS * TILE_COLS);
        // A worker's reused buffer: the fast path's first guess (8x the compressed size,
        // 1 MiB at least) or the slow path's regrowth, whichever is larger.
        assert_eq!(PreloadedStripes::decoder_buffer_bound(100, 50), 1 << 20);
        assert_eq!(PreloadedStripes::decoder_buffer_bound(8 << 20, 8 << 20), 64 << 20);
        assert_eq!(PreloadedStripes::decoder_buffer_bound(100_000, 4 << 20), 5 << 20);
        assert_eq!(PreloadedStripes::decoder_buffer_bound(100_000, 64 << 20), ZSTD_KEEP_BUF);
        assert_eq!(decoded_bound(&[0x28, 0xB5, 0x2F, 0xFD, 0x00]), 8 + 4 * (TILE_COLS + 1) + 2 * TILE_ROWS * TILE_COLS);
    }

    /// The extraction with retention gives the same bitmatrix, leaves exactly its
    /// head-most reads within the budget held (each the file's bytes), and every
    /// later read — wholly, partly or not covered by them — returns the file's bytes;
    /// the interpolation's release drops only the spans behind its read.
    #[test]
    fn retained_extraction_reads_serve_later_reads_with_the_files_bytes() {
        let _g = super::super::retain::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use super::super::retain;
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0x4E7A1);
        for (n_variants, n_haps, per_mille) in [(40 * TILE_ROWS + 77, 300, 30), (12 * TILE_ROWS, 4096 + 100, 8)] {
            let (t, _) = panel(dir.path(), &mut r, n_variants, n_haps, per_mille, 3);
            let file = std::fs::read(t.file_path()).unwrap();
            // Sites in about half the stripes, so the reads leave gaps between them.
            let mut idx = Vec::new();
            for s in 0..t.n_tile_rows {
                if r.below(2) == 0 { continue; }
                let rows = TILE_ROWS.min(n_variants - s * TILE_ROWS);
                for _ in 0..1 + r.below(4) { idx.push(s * TILE_ROWS + r.below(rows as u64) as usize); }
            }
            retain::release_all();
            let plain = t.extract_bitmatrix(n_haps, &idx);
            assert_eq!(retain::held_bytes(), 0);
            for (n, budget) in [0usize, 1, 100 << 10, usize::MAX].into_iter().enumerate() {
                retain::release_all();
                // A cache-limited run also drops pages from the page cache (advice only).
                retain::set_limited((n % 2 == 1).then(|| t.file_path()), true);
                let got = t.extract_bitmatrix_retaining(n_haps, &idx, budget, if n < 2 { u64::MAX } else { 0 });
                for ci in 0..idx.len() {
                    for h in 0..n_haps { assert_eq!(got.get(ci, h), plain.get(ci, h)); }
                }
                let held = retain::held_spans();
                assert!(retain::held_bytes() <= budget);
                if budget == usize::MAX { assert!(held.len() > 1, "every read held"); }
                // The held spans are the reads nearest the head, in order.
                for (i, sp) in held.iter().enumerate() {
                    if i > 0 { assert!(sp.start >= held[i - 1].end); }
                }
                if let Some(first) = held.first() { assert_eq!(first.start, t.stripe_offset(idx.iter().min().unwrap() / TILE_ROWS)); }
                for (n, (fs, k)) in [(0usize, t.n_tile_rows), (3, 5), (1, 1), (t.n_tile_rows - 2, 2), (5, 9)].into_iter().enumerate() {
                    let k = k.min(t.n_tile_rows - fs);
                    let loaded = t.preload_stripes_into(fs, k, if n % 2 == 0 { Vec::new() } else { vec![7u8; 3] }).unwrap();
                    let lo = t.stripe_offset(fs) as usize;
                    assert_eq!(&loaded.buf[..], &file[lo..lo + t.stripes_read_len(fs, k)]);
                }
                // Releasing before stripe 4's offset keeps every span that reaches past it.
                let cut = t.stripe_offset(4);
                retain::consumed_before(t.file_path(), cut);
                assert_eq!(retain::held_spans(), held.iter().filter(|sp| sp.end > cut).cloned().collect::<Vec<_>>());
            }
            retain::release_all();
            retain::set_limited(None, false);
        }
    }

    /// A tile read in place equals the decoded SparseTile, through the reused    /// A tile read in place equals the decoded SparseTile, through the reused
    /// decoder, the slow path (buffer too small), LZ4 tiles and no scope at all;
    /// and the three scans set the same bits as a scan of the SparseTile, on
    /// columns shorter and longer than the 32-row window.
    #[test]
    fn tile_view_and_scans_match_the_sparse_tile() {
        let dir = tempfile::tempdir().unwrap();
        let mut r = Xs(0x7E57);
        for (scoped, (n_variants, n_haps, per_mille)) in [(true, (1024, 4096, 1000)), (true, (3000, 300, 40)),
                (false, (2100, 4096 + 900, 120)), (true, (777, 9000, 2)), (false, (1024, 700, 45))] {
            let _here = scoped.then(TileDecoderScope::this_thread);
            let (t, tiles) = panel(dir.path(), &mut r, n_variants, n_haps, per_mille, 3);
            let loaded = t.preload_stripes(0, t.n_tile_rows).unwrap();
            for st in 0..t.n_tile_rows {
                for b in 0..t.n_tile_cols {
                    let want = &tiles[st * t.n_tile_cols + b];
                    loaded.with_tile(st, b, |v| {
                        assert_eq!((v.n_rows, v.n_cols, v.nnz), (want.n_rows, want.n_cols as usize, want.indices.len()));
                        assert_eq!(v.indptr, &want.indptr[..]);
                        assert_eq!(&v.indices[..v.nnz], &want.indices[..]);
                        assert!(v.indices[v.nnz..].len() >= TILE_PAD / 2 && v.indices[v.nnz..].iter().all(|&x| x == 0));
                        for n_rows_wanted in [1usize, 2, 5, 8, 33, 64] {
                            let mut rows: Vec<u16> = (0..n_rows_wanted).map(|_| r.below(want.n_rows as u64) as u16).collect();
                            rows.sort_unstable();
                            rows.dedup();
                            let n_cols = v.n_cols.saturating_sub(r.below(3) as usize * 37);
                            let mut expect = vec![0u64; rows.len() * 64];
                            for col in 0..n_cols {
                                let (lo, hi) = want.col_range(col);
                                for (k, &row) in rows.iter().enumerate() {
                                    if want.indices[lo..hi].binary_search(&row).is_ok() { expect[k * 64 + col / 64] |= 1 << (col % 64); }
                                }
                            }
                            let run = |f: &dyn Fn(&mut dyn FnMut(usize, usize, u64))| {
                                let mut got = vec![0u64; rows.len() * 64];
                                f(&mut |k, w, bits| { assert!(bits != 0); got[k * 64 + w] |= bits; });
                                got
                            };
                            if rows.len() <= 64 {
                                assert_eq!(run(&|e| scan_compare(v, n_cols, &rows, &mut |k, w, b| e(k, w, b))), expect);
                                assert_eq!(run(&|e| scan_compare_scalar(v, n_cols, &rows, &mut |k, w, b| e(k, w, b))), expect);
                            }
                            assert_eq!(run(&|e| scan_table(v, n_cols, &rows, &mut |k, w, b| e(k, w, b))), expect);
                        }
                    });
                }
            }
        }
    }

    /// A corrupt tile read in place stops with from_bytes' message.
    #[test]
    #[should_panic(expected = "rows not ascending within a column (corrupt tile)")]
    fn tile_view_rejects_what_from_bytes_rejects() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt2.tiles");
        let comp = zstd::encode_all(std::io::Cursor::new(&tile_bytes(1024, vec![0, 2, 2, 5], vec![7, 9, 3, 8, 4])), 3).unwrap();
        std::fs::write(&path, &comp).unwrap();
        let entries = vec![TileEntryPub { offset: 0, comp_size: comp.len() as u32 }];
        let t = TiledSrpReader::from_entries(path, 1024, 3, entries, 1, 1);
        let loaded = t.preload_stripes(0, 1).unwrap();
        let _s = TileDecoderScope::this_thread();
        loaded.with_tile(0, 0, |_| ());
    }

    #[test]
    fn tile_index_outside_the_file_or_running_backwards_is_refused() {
        let p = std::path::Path::new("x.srp");
        assert!(check_tile_index([(0u64, 10u32), (10, 5), (15, 0), (15, 85)].into_iter(), 100, p).is_ok());
        assert!(check_tile_index(std::iter::empty(), 0, p).is_ok());
        let past_end = check_tile_index([(0u64, 10u32), (10, 91)].into_iter(), 100, p).unwrap_err();
        assert_eq!(past_end.kind(), std::io::ErrorKind::InvalidData);
        assert!(past_end.to_string().contains("x.srp") && past_end.to_string().contains("tile 1"), "{past_end}");
        let backwards = check_tile_index([(50u64, 10u32), (40, 10)].into_iter(), 100, p).unwrap_err();
        assert!(backwards.to_string().contains("tile 1"), "{backwards}");
        let wrapped = check_tile_index([(u64::MAX, 1u32)].into_iter(), 100, p).unwrap_err();
        assert!(wrapped.to_string().contains("tile 0"), "{wrapped}");
    }
}
