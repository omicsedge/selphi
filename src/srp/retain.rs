//! Panel tile bytes held in RAM, and page-cache hygiene, for a chip run that reads a
//! tiled SRP twice.
//!
//! The chip-site bitmatrix extraction reads the tiles of every stripe holding a chip
//! site (~95% of the tile bytes on the 75,552-haplotype panel, in contiguous spans),
//! and the interpolation reads every tile again, window by window from the head of
//! the file. When the page cache can keep the panel the second read costs nothing.
//! In a container whose memory limit is below the panel's size (prod: a 7.5 GB chr2
//! panel downloaded into a 6 GiB task) both reads went to disk, ~60 s each at the
//! ~120 MB/s of the task's ephemeral disk. What helps, all byte-identical:
//!
//! * The extraction reads the file TAIL first (`tiled::extract_tail_first`), where
//!   the download leaves the cache warm, and ends at the head, where the
//!   interpolation starts.
//! * On such a run (`plan`: the tiles exceed half the room under the limit) each
//!   window, during its HMM (the disk is idle there), prefetches the stripes its
//!   interpolation reads into spans held here (`window_io`, `window_prefetch_limit`),
//!   keeping a floor of room and releasing spans when the HMM needs it.
//! * The interpolation's reads (`tiled::TiledSrpReader::preload_stripes_into`) copy
//!   what is held and read only the rest; behind its reads it releases the spans
//!   (`consumed_before`).
//! * Opt-in: the extraction can leave its head-most spans held (`Plan::hold`), and a
//!   run whose panel is its own can drop the pages it holds or is done with from the
//!   page cache (`Plan::drop_cache`, `drop_cached`).
//!
//! A host with the memory to spare holds and drops nothing.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// Identity of an open panel file: spans are only served to reads of the same file.
/// Device, inode, length and modification/change times: a file replaced (new inode)
/// or rewritten (new times) is another file. The store keeps the file open while it
/// holds spans, so its inode cannot be reused meanwhile.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FileKey {
    dev: u64,
    ino: u64,
    len: u64,
    mtime: (i64, i64),
    ctime: (i64, i64),
}

impl FileKey {
    /// The key of an open file (`None` where the platform gives no file identity).
    pub fn of(file: &std::fs::File) -> Option<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let m = file.metadata().ok()?;
            Some(FileKey { dev: m.dev(), ino: m.ino(), len: m.len(),
                mtime: (m.mtime(), m.mtime_nsec()), ctime: (m.ctime(), m.ctime_nsec()) })
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            None
        }
    }

    /// The key of the file at `path`.
    pub fn of_path(path: &std::path::Path) -> Option<Self> {
        Self::of(&std::fs::File::open(path).ok()?)
    }
}

/// A held span's bytes: an anonymous mapping, which whichever thread drops it returns
/// to the kernel at once. (A heap block freed by another thread than the one that
/// allocated it can wait in the allocator for that thread — mimalloc's delayed frees
/// — and spans are released by the interpolation and by `window_io`'s monitor, not by
/// the thread that read them.) Heap spans exist only in tests.
pub enum SpanData {
    #[cfg(test)]
    Heap(Vec<u8>),
    Anon(memmap2::MmapMut),
}

impl SpanData {
    /// `len` zero bytes in an anonymous mapping; `None` if mapping fails (the caller
    /// then holds nothing).
    pub fn anon(len: usize) -> Option<Self> {
        memmap2::MmapMut::map_anon(len).ok().map(SpanData::Anon)
    }

    /// `bytes` copied into an anonymous mapping.
    pub fn copy_of(bytes: &[u8]) -> Option<Self> {
        let mut d = Self::anon(bytes.len())?;
        d.as_mut_slice().copy_from_slice(bytes);
        Some(d)
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        match self {
            #[cfg(test)]
            SpanData::Heap(v) => v,
            SpanData::Anon(m) => &mut m[..],
        }
    }
}

impl std::ops::Deref for SpanData {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            #[cfg(test)]
            SpanData::Heap(v) => v,
            SpanData::Anon(m) => &m[..],
        }
    }
}

struct Store {
    key: FileKey,
    /// The file, kept open while spans of it are held.
    _pin: Option<std::fs::File>,
    /// Held spans by file offset; they do not overlap.
    spans: BTreeMap<u64, Arc<SpanData>>,
    bytes: usize,
}

static STORE: Mutex<Option<Store>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<Store>> {
    // A panic while holding the lock leaves the map consistent (every mutation is
    // a single insert/remove), so a poisoned lock is still usable.
    STORE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Hold `data`: the bytes at `offset..offset + data.len()` of the file `pin` is open
/// on, read while that file's identity was `key` (taken before the read). Nothing is
/// held, and false returned, when the file no longer has that identity (it was
/// rewritten or replaced meanwhile) or a held span overlaps it. Spans of another file
/// are released first.
pub fn put(pin: &std::fs::File, key: FileKey, offset: u64, data: SpanData) -> bool {
    if data.is_empty() || FileKey::of(pin) != Some(key) { return false; }
    let mut g = lock();
    if g.as_ref().is_some_and(|s| s.key != key) { *g = None; }
    let s = g.get_or_insert_with(|| Store { key, _pin: pin.try_clone().ok(), spans: BTreeMap::new(), bytes: 0 });
    let end = offset + data.len() as u64;
    let overlaps = s.spans.range(..end).next_back()
        .is_some_and(|(&o, d)| o + d.len() as u64 > offset);
    if overlaps { return false; }
    s.bytes += data.len();
    s.spans.insert(offset, Arc::new(data));
    true
}

/// Copy into `buf` (the file's bytes from `offset`) every part a held span covers;
/// returns the ranges of `buf` left to read, ascending.
pub fn fill(key: Option<FileKey>, offset: u64, buf: &mut [u8]) -> Vec<Range<usize>> {
    if buf.is_empty() { return Vec::new(); }
    let end = offset + buf.len() as u64;
    let hits: Vec<(u64, Arc<SpanData>)> = match (key, lock().as_ref()) {
        (Some(k), Some(s)) if s.key == k => {
            // The span starting at or before `offset`, then every one starting inside.
            let first = s.spans.range(..=offset).next_back().map(|(&o, d)| (o, Arc::clone(d)));
            first.into_iter().chain(s.spans.range(offset + 1..end).map(|(&o, d)| (o, Arc::clone(d))))
                .filter(|(o, d)| o + d.len() as u64 > offset)
                .collect()
        }
        _ => Vec::new(),
    };
    // Copy outside the lock (a span released meanwhile stays alive through its Arc).
    let mut holes = Vec::new();
    let mut at = 0usize; // next byte of `buf` not yet accounted for
    for (o, d) in hits {
        let lo = o.max(offset);
        let hi = (o + d.len() as u64).min(end);
        let (blo, bhi) = ((lo - offset) as usize, (hi - offset) as usize);
        if blo > at { holes.push(at..blo); }
        buf[blo..bhi].copy_from_slice(&d[(lo - o) as usize..(hi - o) as usize]);
        at = bhi;
    }
    if at < buf.len() { holes.push(at..buf.len()); }
    holes
}

/// Release the spans of `key`'s file that end at or before `offset`.
pub fn release_before(key: Option<FileKey>, offset: u64) {
    let mut g = lock();
    let Some(s) = g.as_mut() else { return };
    if Some(s.key) != key { return; }
    let gone: Vec<u64> = s.spans.iter()
        .take_while(|(o, d)| **o + d.len() as u64 <= offset)
        .map(|(&o, _)| o).collect();
    for o in gone {
        if let Some(d) = s.spans.remove(&o) { s.bytes -= d.len(); }
    }
    if s.spans.is_empty() { *g = None; }
}

/// Release everything held.
pub fn release_all() {
    *lock() = None;
}

/// Bytes held.
pub fn held_bytes() -> usize {
    lock().as_ref().map_or(0, |s| s.bytes)
}

/// The held spans as file byte ranges, ascending (tests, diagnostics).
pub fn held_spans() -> Vec<Range<u64>> {
    lock().as_ref().map_or_else(Vec::new, |s|
        s.spans.iter().map(|(&o, d)| o..o + d.len() as u64).collect())
}

/// Release the spans furthest from the head (those read last) until at most
/// `max_bytes` are held. Returns the bytes released.
pub fn trim_to(max_bytes: usize) -> usize {
    let mut g = lock();
    let Some(s) = g.as_mut() else { return 0 };
    let mut freed = 0;
    while s.bytes > max_bytes {
        let Some((&o, _)) = s.spans.iter().next_back() else { break };
        if let Some(d) = s.spans.remove(&o) { s.bytes -= d.len(); freed += d.len(); }
    }
    if s.spans.is_empty() { *g = None; }
    freed
}

/// The end of the held span of `key`'s file that covers `offset`, or `offset` when
/// none does; and the start of the first held span past it (`u64::MAX` if none).
fn coverage_at(key: FileKey, offset: u64) -> (u64, u64) {
    let g = lock();
    let Some(s) = g.as_ref().filter(|s| s.key == key) else { return (offset, u64::MAX) };
    let covered = s.spans.range(..=offset).next_back()
        .map(|(&o, d)| o + d.len() as u64).filter(|&e| e > offset).unwrap_or(offset);
    let next = s.spans.range(covered.max(offset + 1)..).next().map_or(u64::MAX, |(&o, _)| o);
    (covered, next)
}

/// Bytes `window_io` reads at a time: small enough that its memory checks between
/// reads come every ~0.13 s at the ~120 MB/s of a container's disk.
const PREFETCH_SPAN: usize = if cfg!(test) { 4 << 10 } else { 16 << 20 };

/// Room (`room_mb`) `window_io` never reads into and restores by releasing spans, and
/// that the interpolation keeps beyond each group's own allocations (`trim_to_room`), MB.
pub const ROOM_FLOOR_MB: f64 = PREFETCH_RESERVE_MB;

/// How often `window_io` checks memory once it has stopped reading.
const WATCH_INTERVAL: std::time::Duration = std::time::Duration::from_millis(if cfg!(test) { 1 } else { 50 });

/// What `window_io` sees of memory: the room (`room_mb`) and this process's RSS, MB.
#[derive(Clone, Copy, Debug)]
pub struct MemProbe {
    pub room_mb: Option<f64>,
    pub rss_mb: f64,
}

/// The live `MemProbe`.
pub fn mem_probe() -> MemProbe {
    MemProbe { room_mb: room_mb(), rss_mb: crate::log::rss_mb() }
}

/// What `window_io` did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindowIo {
    /// Bytes read into held spans.
    pub read: usize,
    /// The highest RSS it saw net of the held spans, MB (the window's working set).
    pub peak_net_mb: f64,
    /// Bytes of held spans it released to keep `ROOM_FLOOR_MB` of room.
    pub released: usize,
}

/// The window loop's companion to a window's HMM, until `cancel` is set (then
/// unparked). A reader thread reads the bytes of `file` in `start..end` that no span
/// holds, in order from `start`, `PREFETCH_SPAN` bytes at a time, within `limit`
/// bytes: on a run that drops its pages from the page cache (`Plan::drop_cache`) into
/// held spans — while the room stays above `ROOM_FLOOR_MB` — dropping each span's
/// cache copy (`drop_cached`); otherwise it only warms the page cache, up to `limit`
/// bytes past `start`, skipping the spans the cache mostly holds already (when it can
/// tell: `Residency`). This
/// thread, never blocked by a read, checks memory every `WATCH_INTERVAL`: it records
/// the window's working set (`WindowIo::peak_net_mb`) and, whenever the room falls
/// under `ROOM_FLOOR_MB` (the HMM grew more than the limit allowed for), releases
/// spans from the far end to restore it. A read error or a file whose identity
/// changes ends the reading (the interpolation then reads the rest itself). `probe`
/// is `mem_probe` outside tests.
pub fn window_io(file: Option<&std::fs::File>, start: u64, end: u64, limit: usize,
                 cancel: &std::sync::atomic::AtomicBool, probe: &(dyn Fn() -> MemProbe + Sync)) -> WindowIo {
    window_io_with(file, start, end, limit, cancel, probe, &|f, buf, at| super::tiled::read_file_at(f, buf, at))
}

/// `window_io` reading through `read` (tests block it).
fn window_io_with(file: Option<&std::fs::File>, start: u64, end: u64, limit: usize,
                  cancel: &std::sync::atomic::AtomicBool, probe: &(dyn Fn() -> MemProbe + Sync),
                  read: &(dyn Fn(&std::fs::File, &mut [u8], u64) -> std::io::Result<()> + Sync)) -> WindowIo {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
    let read_total = AtomicUsize::new(0);
    let stop_reading = AtomicBool::new(false);
    let mut out = WindowIo::default();
    let check = |out: &mut WindowIo| {
        let p = probe();
        out.peak_net_mb = out.peak_net_mb.max(p.rss_mb - held_bytes() as f64 / 1048576.0);
        if let Some(room) = p.room_mb && room < ROOM_FLOOR_MB {
            let short = ((ROOM_FLOOR_MB - room) * 1048576.0) as usize;
            out.released += trim_to(held_bytes().saturating_sub(short));
        }
    };
    std::thread::scope(|sc| {
        let reader = file.and_then(|file| FileKey::of(file).map(|key| (file, key))).map(|(file, key)| {
            let (read_total, stop_reading) = (&read_total, &stop_reading);
            sc.spawn(move || {
                // Without page-cache drops (`SELPHI_PANEL_DROP_CACHE`) the page cache
                // would keep a copy of every span held here, and the duplicates push other
                // cached stripes out (one user's chr10, cached whole after its download,
                // read 0.8 GB on the old path and 4.3 GB holding spans): there the reader
                // only warms the cache, reading the spans it does not mostly hold into a
                // scratch buffer, and the interpolation reads them from the cache.
                let warm_only = !drops_pages_of(file);
                let residency = if warm_only { Residency::of(file) } else { None };
                let mut scratch: Vec<u8> = Vec::new();
                let mut pos = start;
                while pos < end && !cancel.load(Relaxed) && !stop_reading.load(Relaxed) {
                    let (covered, next) = coverage_at(key, pos);
                    if covered > pos { pos = covered; continue; }
                    let len = (end.min(next) - pos).min(PREFETCH_SPAN as u64) as usize;
                    if warm_only {
                        // The limit bounds how far past `start` the cache is warmed, cached
                        // spans included: past it, warming would push nearer stripes out.
                        if (pos - start) as usize + len > limit { break; }
                        if residency.as_ref().is_some_and(|r| r.mostly_cached(pos, len)) { pos += len as u64; continue; }
                        if scratch.len() < len { scratch.resize(len, 0); }
                        if read(file, &mut scratch[..len], pos).is_err() { break; }
                    } else {
                        if len > limit.saturating_sub(held_bytes()) { break; }
                        if probe().room_mb.is_some_and(|r| r - (len as f64 / 1048576.0) < ROOM_FLOOR_MB) { break; }
                        let Some(mut buf) = SpanData::anon(len) else { break };
                        if read(file, buf.as_mut_slice(), pos).is_err() || !put(file, key, pos, buf) { break; }
                        drop_cached(file, pos, pos + len as u64);
                    }
                    read_total.fetch_add(len, Relaxed);
                    pos += len as u64;
                }
            })
        });
        while !cancel.load(Relaxed) {
            check(&mut out);
            std::thread::park_timeout(WATCH_INTERVAL);
        }
        stop_reading.store(true, Relaxed);
        if let Some(h) = reader { h.join().expect("panel prefetch reader panicked"); }
        check(&mut out);
    });
    out.read = read_total.load(Relaxed);
    out
}

/// Release the spans furthest from the head until the room (`room_mb`) is at least
/// `floor_mb` or nothing is held. Returns the bytes released.
pub fn trim_to_room(floor_mb: f64) -> usize {
    let Some(room) = room_mb() else { return 0 };
    if room >= floor_mb { return 0; }
    let short = ((floor_mb - room) * 1048576.0) as usize;
    trim_to(held_bytes().saturating_sub(short))
}

/// The panel file of a run that is cache-limited (`set_limited`), and whether pages of
/// it are dropped from the page cache: those the run holds in RAM or has read for the
/// last time (`drop_cached`, `consumed_before`), so the cache keeps stripes still to
/// be read. One file at a time.
static LIMITED: Mutex<Option<(FileKey, bool)>> = Mutex::new(None);
/// Bytes of the limited file before this offset were dropped by `consumed_before`.
static CONSUMED: Mutex<u64> = Mutex::new(0);

/// Mark the panel at `path` cache-limited (`Plan::limited`), dropping its pages from
/// the page cache when `drop_cache` (`Plan::drop_cache`); or no file (`None`).
pub fn set_limited(path: Option<&std::path::Path>, drop_cache: bool) {
    *LIMITED.lock().unwrap_or_else(|e| e.into_inner()) = path.and_then(FileKey::of_path).map(|k| (k, drop_cache));
    *CONSUMED.lock().unwrap_or_else(|e| e.into_inner()) = 0;
}

/// A panel file is marked cache-limited (`set_limited`).
pub fn limited() -> bool {
    LIMITED.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// `file` is the panel marked cache-limited, with page-cache drops.
fn drops_pages_of(file: &std::fs::File) -> bool {
    let l = *LIMITED.lock().unwrap_or_else(|e| e.into_inner());
    matches!(l, Some((k, true)) if FileKey::of(file) == Some(k))
}

/// Drop `file`'s pages in `start..end` from the page cache (`posix_fadvise`
/// DONTNEED; the kernel keeps dirty, mapped or busy pages), when `file` is the panel
/// marked cache-limited with drops; otherwise, and off Linux, nothing.
pub fn drop_cached(file: &std::fs::File, start: u64, end: u64) {
    if drops_pages_of(file) { fadvise_dontneed(file, start, end); }
}

/// `posix_fadvise(DONTNEED)` over `start..end` of `file` (Linux; elsewhere nothing).
fn fadvise_dontneed(file: &std::fs::File, start: u64, end: u64) {
    if end <= start { return; }
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        // SAFETY: plain syscall on an open descriptor; advice only, no memory touched.
        unsafe {
            libc::posix_fadvise(file.as_raw_fd(), start as libc::off_t, (end - start) as libc::off_t,
                libc::POSIX_FADV_DONTNEED);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
}

/// Which pages of a file the page cache holds (`mincore` over a read-only mapping that
/// is never accessed: only its address is used, so no page is faulted in). Linux only.
struct Residency {
    #[cfg(target_os = "linux")]
    map: memmap2::MmapRaw,
    #[cfg(target_os = "linux")]
    page: usize,
}

impl Residency {
    fn of(file: &std::fs::File) -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::MetadataExt;
            // Linux reports every page of a file the caller neither owns nor may write
            // as resident: unknown there.
            // SAFETY: geteuid has no preconditions.
            if file.metadata().ok()?.uid() != unsafe { libc::geteuid() } { return None; }
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
            if page <= 0 { return None; }
            let map = memmap2::MmapOptions::new().map_raw_read_only(file).ok()?;
            Some(Residency { map, page: page as usize })
        }
        #[cfg(not(target_os = "linux"))]
        { let _ = file; None }
    }

    /// At least half the pages of `pos..pos + len` are in the page cache (false when
    /// that cannot be told).
    fn mostly_cached(&self, pos: u64, len: usize) -> bool {
        #[cfg(target_os = "linux")]
        {
            let page = self.page;
            let (lo, hi) = ((pos as usize) / page * page, (pos as usize).saturating_add(len).min(self.map.len()));
            if hi <= lo { return false; }
            let mut vec = vec![0u8; (hi - lo).div_ceil(page)];
            // SAFETY: lo..hi lies inside the mapping and lo is page-aligned; `vec` holds
            // one byte per page of the range; the mapping's memory is not accessed.
            let ok = unsafe {
                libc::mincore(self.map.as_ptr().add(lo) as *mut libc::c_void, hi - lo, vec.as_mut_ptr())
            } == 0;
            ok && 2 * vec.iter().filter(|&&v| v & 1 == 1).count() >= vec.len()
        }
        #[cfg(not(target_os = "linux"))]
        { let _ = (pos, len); false }
    }
}

/// The interpolation's reads have moved to `offset` of the panel at `path` and never
/// go back: spans ending at or before it are released and, when the panel is marked
/// cache-limited with drops, its pages before it are dropped from the page cache.
/// (Not for the sample-batched output, which reads each window once per sample batch.)
pub fn consumed_before(path: &std::path::Path, offset: u64) {
    if held_bytes() == 0 && !limited() { return; }
    let Ok(file) = std::fs::File::open(path) else { return };
    release_before(FileKey::of(&file), offset);
    if !drops_pages_of(&file) { return; }
    let from = {
        let mut c = CONSUMED.lock().unwrap_or_else(|e| e.into_inner());
        if offset <= *c { return; }
        std::mem::replace(&mut *c, offset)
    };
    drop_cached(&file, from, offset);
}

/// Room for this process to grow, in MB: `MemAvailable`, and the room under the
/// cgroup memory limit when that is less (`None` if neither is known).
pub fn room_mb() -> Option<f64> {
    match (crate::log::available_ram_mb(), crate::log::cgroup_headroom_mb()) {
        (Some(a), Some(c)) => Some(a.min(c)),
        (a, c) => a.or(c),
    }
}

/// How a chip run treats a tiled panel it reads twice (`plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    /// The page cache cannot be relied on to keep the panel's tiles between the
    /// two reads: they exceed the room the process has (`room_mb`) once the stages in
    /// between have grown (`GROWTH_PER_BITMATRIX`). Then windows prefetch
    /// (`set_limited`), and pages may be dropped from the cache (`drop_cache`).
    pub limited: bool,
    /// Bytes of head-most extraction reads to leave held in RAM.
    pub hold: usize,
    /// Bytes of the extraction's reads after the held ones (head first) to leave in
    /// the page cache; the rest are dropped from it as they are decoded.
    pub keep_cached: u64,
    /// Drop pages from the page cache (`drop_cached`) on a limited run. They are the
    /// inode's pages, so also gone for any other process reading the same panel:
    /// only when the panel is declared the run's own (`SELPHI_PANEL_DROP_CACHE=1`, as
    /// a container task that downloaded it can).
    pub drop_cache: bool,
}

/// Share of the room above `RETAIN_RESERVE_MB` the extraction holds under
/// `SELPHI_PANEL_RETAIN_MB=auto` (`plan`). Not the default: one user x the
/// 75,552-haplotype chr2 panel in a 6 GiB cgroup read the same bytes from disk
/// holding 1.07 GB (8,773 MB) as holding nothing (8,768 MB), with the peak 1.07 GB
/// higher — under one memory limit held spans and the page cache are zero-sum once
/// the cache drops its copies, and the cache after a tail-first extraction already
/// keeps the head.
pub const RETAIN_FRACTION: f64 = 0.33;
/// Room `plan` leaves alone, MB.
pub const RETAIN_RESERVE_MB: f64 = 1024.0;
/// What the stages between the extraction and the interpolation grow by, in units of
/// the chip-site bitmatrix the extraction builds (`plan`): one user x the
/// 75,552-haplotype chr2 panel grew 1.7 GB over a 511 MB bitmatrix in phasing (its
/// common-site copy, the few-target sweep index).
pub const GROWTH_PER_BITMATRIX: f64 = 3.5;
/// Share of the room above the expected growth and `PREFETCH_RESERVE_MB` a window's
/// prefetch may fill (`window_prefetch_limit`).
pub const PREFETCH_FRACTION: f64 = 0.75;
/// Room `window_prefetch_limit` leaves alone, MB.
pub const PREFETCH_RESERVE_MB: f64 = 512.0;

/// The plan for a panel whose tiles take `tile_bytes`, extracted into a chip-site
/// bitmatrix of `bitmatrix_bytes`, from the room the process has now.
/// `SELPHI_PANEL_CACHE=limited|plenty` overrides the `limited` test.
/// `SELPHI_PANEL_RETAIN_MB` sets the held bytes: none by default, `auto` =
/// `RETAIN_FRACTION` of the room above `RETAIN_RESERVE_MB` and the stages' growth
/// (`GROWTH_PER_BITMATRIX`) on a limited run; any hold is capped at half of that room. On a limited run the page cache is left the next
/// reads up to what the room keeps once the held bytes and the stages' growth
/// (`GROWTH_PER_BITMATRIX`) are taken out — the cgroup squeezes the cache to that
/// before the interpolation starts, and pages beyond it would only push out the head
/// the interpolation reads first (under a classic LRU the tail pages the extraction
/// found cached are activated by that second touch and would outlive the head it read
/// once). Otherwise nothing is held or dropped: a host with the memory to spare keeps
/// the panel in its page cache.
pub fn plan(tile_bytes: u64, bitmatrix_bytes: u64) -> Plan {
    plan_with(tile_bytes, bitmatrix_bytes, room_mb(), crate::config::raw("SELPHI_PANEL_RETAIN_MB").as_deref(),
        crate::config::raw("SELPHI_PANEL_CACHE").as_deref(), crate::config::is_one("SELPHI_PANEL_DROP_CACHE"))
}

/// `plan` for a room of `room` MB and the knobs' values.
fn plan_with(tile_bytes: u64, bitmatrix_bytes: u64, room: Option<f64>, retain_mb: Option<&str>,
             cache: Option<&str>, drop_cache: bool) -> Plan {
    // Unset: none (measured: under one memory limit held spans and the page cache
    // are zero-sum, see `RETAIN_FRACTION`); "auto": the formula; else MB.
    let hold = match retain_mb {
        Some("auto") => None,
        Some(v) => Some(v.trim().parse::<usize>().unwrap_or(0) << 20),
        None => Some(0),
    };
    let tiles = match cache {
        Some("plenty") => return Plan { limited: false, hold: 0, keep_cached: u64::MAX, drop_cache: false },
        Some("limited") => u64::MAX,
        _ => tile_bytes,
    };
    let mut p = plan_for(tiles, bitmatrix_bytes, room.unwrap_or(if tiles == u64::MAX { 0.0 } else { f64::INFINITY }), hold);
    p.drop_cache &= drop_cache;
    p
}

/// `plan` for a room of `room_mb`, holding `hold` bytes when given (capped).
pub fn plan_for(tile_bytes: u64, bitmatrix_bytes: u64, room_mb: f64, hold: Option<usize>) -> Plan {
    let room = room_mb.max(0.0) * 1048576.0;
    // What the bitmatrix and what phasing builds on it take: the stages between the
    // two reads. The tiles fit the cache that leaves: not limited (one user's chrX,
    // 3.5 GB of tiles in a 6 GiB cgroup, read nothing from disk a second time).
    let growth = GROWTH_PER_BITMATRIX * bitmatrix_bytes as f64;
    if tile_bytes as f64 <= room - growth {
        return Plan { limited: false, hold: 0, keep_cached: u64::MAX, drop_cache: false };
    }
    // The room above the reserve once that growth is set aside: what a hold lives in.
    let above = ((room_mb - RETAIN_RESERVE_MB).max(0.0) * 1048576.0 - growth).max(0.0);
    let hold = hold.unwrap_or((above * RETAIN_FRACTION) as usize).min((above * 0.5) as usize);
    let keep = room - hold as f64 - growth;
    Plan { limited: true, hold, keep_cached: keep.max(0.0) as u64, drop_cache: true }
}

/// What a window's HMM is expected to grow by (`window_prefetch_limit`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Growth {
    /// Measured on earlier windows (`WindowIo::peak_net_mb`), scaled to this one, MB.
    Measured(f64),
    /// The run's peak so far over its current RSS: the first window after phasing,
    /// whose peak bounds the HMM's (conservatively).
    PeakSoFar,
    /// No bound known (the first window of a run that did not phase): no prefetch.
    Unknown,
}

/// How many bytes the store may hold once a window's prefetch (`window_io`, run
/// during the window's HMM, from the window's first stripe on) has filled it;
/// 0 = no prefetch (the window then preloads its first stripes as before).
/// `SELPHI_PANEL_PREFETCH_MB` sets it. Default: nothing unless the run is
/// `limited`; then what is held plus `PREFETCH_FRACTION` of the room left once the
/// window's `growth` (plus `INTERP_RESERVE_MB` for the interpolation's buffers when
/// measured) and `PREFETCH_RESERVE_MB` are set aside; nothing for `Growth::Unknown`.
pub fn window_prefetch_limit(growth: Growth) -> usize {
    if let Some(mb) = crate::config::usize_opt("SELPHI_PANEL_PREFETCH_MB") { return mb << 20; }
    if !limited() { return 0; }
    let Some(room) = room_mb() else { return 0 };
    let growth = match growth {
        Growth::Measured(g) => g + INTERP_RESERVE_MB,
        Growth::PeakSoFar => (crate::log::peak_mem_mb() - crate::log::rss_mb()).max(0.0),
        Growth::Unknown => return 0,
    };
    prefetch_limit_for(held_bytes(), room, growth)
}

/// The interpolation's buffers beyond the HMM's growth (`window_prefetch_limit`), MB:
/// its two compressed reads (`COMP_READ_BYTES` each), a group's values and formatted
/// output (`GROUP_BYTES` each) and decoded stripes. Leaving them out of the prefetch
/// keeps the interpolation's per-group reservation from releasing what was prefetched.
pub const INTERP_RESERVE_MB: f64 = 1024.0;

/// `window_prefetch_limit` for `held` bytes held, `room_mb` of room and an expected
/// growth of `growth_mb`.
pub fn prefetch_limit_for(held: usize, room_mb: f64, growth_mb: f64) -> usize {
    let free = (room_mb - growth_mb - PREFETCH_RESERVE_MB).max(0.0) * PREFETCH_FRACTION;
    held + (free * 1048576.0) as usize
}

#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    fn file_bytes(n: usize) -> Vec<u8> { (0..n).map(|i| (i * 7 + i / 251) as u8).collect() }

    /// A file holding `bytes`, open, and its key.
    fn file_with(dir: &std::path::Path, name: &str, bytes: &[u8]) -> (std::fs::File, FileKey) {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let f = std::fs::File::open(&path).unwrap();
        let k = FileKey::of(&f).unwrap();
        (f, k)
    }

    #[test]
    fn fill_copies_held_spans_and_reports_the_holes() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(1000);
        let (file, k) = file_with(dir.path(), "a", &f);
        let (_, other) = file_with(dir.path(), "b", &f);
        assert_ne!(k, other);
        assert!(put(&file, k, 100, SpanData::Heap(f[100..200].to_vec())));
        assert!(put(&file, k, 300, SpanData::Heap(f[300..350].to_vec())));
        assert!(put(&file, k, 350, SpanData::Heap(f[350..400].to_vec())));
        assert!(!put(&file, k, 150, SpanData::Heap(f[150..250].to_vec()))); // overlaps the first: not added
        assert_eq!(held_bytes(), 200);
        for (lo, hi) in [(0usize, 1000usize), (120, 130), (90, 110), (150, 320), (200, 300), (390, 1000), (0, 50), (100, 400)] {
            let mut buf = vec![0xEEu8; hi - lo];
            let holes = fill(Some(k), lo as u64, &mut buf);
            let mut covered = vec![true; hi - lo];
            for h in &holes { for i in h.clone() { covered[i] = false; } }
            for i in 0..hi - lo {
                let held = (100..200).contains(&(lo + i)) || (300..400).contains(&(lo + i));
                assert_eq!(covered[i], held, "[{lo},{hi}) byte {i}");
                if held { assert_eq!(buf[i], f[lo + i]); } else { assert_eq!(buf[i], 0xEE); }
            }
            assert!(holes.windows(2).all(|w| w[0].end < w[1].start), "{holes:?}");
        }
        // Another file is never served; an empty read touches nothing.
        let mut buf = vec![0u8; 100];
        assert_eq!(fill(Some(other), 100, &mut buf), vec![0..100]);
        assert_eq!(fill(None, 100, &mut buf), vec![0..100]);
        for at in [0u64, 99, 100, 150, 400, u64::MAX] { assert!(fill(Some(k), at, &mut []).is_empty()); }
        // The same path rewritten (new times) is another file.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.path().join("a"), &f).unwrap();
        let k2 = FileKey::of_path(&dir.path().join("a")).unwrap();
        assert_ne!(k2, k);
        assert_eq!(fill(Some(k2), 100, &mut buf), vec![0..100]);
        release_all();
    }

    #[test]
    fn release_before_drops_only_spans_behind_the_offset() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(1000);
        let (file, k) = file_with(dir.path(), "a", &f);
        let (other_file, other) = file_with(dir.path(), "b", &f);
        for o in [0usize, 100, 200, 300] { assert!(put(&file, k, o as u64, SpanData::Heap(f[o..o + 100].to_vec()))); }
        release_before(Some(other), 1000);
        assert_eq!(held_bytes(), 400);
        release_before(Some(k), 150);
        assert_eq!(held_spans(), vec![100..200, 200..300, 300..400]);
        release_before(Some(k), 200);
        assert_eq!(held_spans(), vec![200..300, 300..400]);
        assert_eq!(trim_to(150), 100);
        assert_eq!(held_spans(), vec![200..300]);
        // A span of another file replaces the store.
        assert!(put(&other_file, other, 0, SpanData::Heap(f[..10].to_vec())));
        assert_eq!(held_spans(), vec![0..10]);
        release_before(Some(other), 10);
        assert_eq!(held_bytes(), 0);
        release_all();
    }

    /// Only the file marked limited has its pages dropped; the consumed watermark
    /// only moves forward; unmarking clears it.
    #[test]
    fn page_cache_drops_follow_the_limited_file_only() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(10_000);
        let (a, _) = file_with(dir.path(), "a", &f);
        let (b, _) = file_with(dir.path(), "b", &f);
        set_limited(None, false);
        assert!(!limited() && !drops_pages_of(&a));
        set_limited(Some(&dir.path().join("a")), false);
        assert!(limited() && !drops_pages_of(&a) && !drops_pages_of(&b));
        set_limited(Some(&dir.path().join("a")), true);
        assert!(limited() && drops_pages_of(&a) && !drops_pages_of(&b));
        consumed_before(&dir.path().join("a"), 4096);
        assert_eq!(*CONSUMED.lock().unwrap(), 4096);
        consumed_before(&dir.path().join("a"), 100);
        assert_eq!(*CONSUMED.lock().unwrap(), 4096);
        consumed_before(&dir.path().join("b"), 8192);
        assert_eq!(*CONSUMED.lock().unwrap(), 4096);
        drop_cached(&a, 0, 10_000);
        drop_cached(&b, 0, 10_000);
        set_limited(None, false);
        assert!(!limited() && *CONSUMED.lock().unwrap() == 0);
    }

    fn plenty() -> MemProbe { MemProbe { room_mb: Some(1e9), rss_mb: 100.0 } }

    /// `window_io` fills the gaps between held spans in order from its start, with the
    /// file's bytes, within the limit, and then watches until cancelled.
    #[test]
    fn window_io_reads_the_gaps_in_order_within_the_limit() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f.bin");
        let f = file_bytes(100_000);
        std::fs::write(&path, &f).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let k = FileKey::of(&file).unwrap();
        let yes = std::sync::atomic::AtomicBool::new(true);
        // Drops on: every span is read, cached or not (see the residency test below).
        set_limited(Some(&path), true);
        assert!(put(&file, k, 10_000, SpanData::Heap(f[10_000..12_000].to_vec())));
        assert!(put(&file, k, 30_000, SpanData::Heap(f[30_000..31_000].to_vec())));
        // Unlimited: everything in 5,000..60,000 ends up held, nothing outside it.
        // (Cancelled from the start would read nothing: run it on a thread.)
        // It reads, then watches until cancelled: run it on a thread, wait until it
        // holds what it should, then stop it.
        let run = |start: u64, end: u64, limit: usize, expect_held: usize| {
            let stop = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|sc| {
                let h = sc.spawn(|| window_io(Some(&file), start, end, limit, &stop, &plenty));
                let t0 = std::time::Instant::now();
                while held_bytes() < expect_held && t0.elapsed().as_secs() < 30 { std::thread::yield_now(); }
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                h.thread().unpark();
                h.join().unwrap()
            })
        };
        let io = run(5_000, 60_000, usize::MAX, 55_000);
        assert_eq!(io.read, 55_000 - 3_000);
        assert_eq!(io.released, 0);
        let mut buf = vec![0u8; 100_000];
        assert_eq!(fill(Some(k), 0, &mut buf), vec![0..5_000, 60_000..100_000]);
        assert_eq!(&buf[5_000..60_000], &f[5_000..60_000]);
        assert!(held_spans().iter().all(|sp| sp.end - sp.start <= PREFETCH_SPAN as u64));
        // A limit stops it before the span that would cross it.
        release_all();
        let io = run(0, 100_000, 3 * PREFETCH_SPAN + 10, 3 * PREFETCH_SPAN);
        assert_eq!(io.read, 3 * PREFETCH_SPAN);
        assert_eq!(held_spans().last().unwrap().end, 3 * PREFETCH_SPAN as u64);
        // Cancelled before it starts: it reads nothing, holds what it found.
        assert_eq!(window_io(Some(&file), 0, 100_000, usize::MAX, &yes, &plenty).read, 0);
        assert_eq!(held_bytes(), 3 * PREFETCH_SPAN);
        // No file: it only watches; its working set is RSS net of the held spans.
        let io = window_io(None, 0, 100_000, usize::MAX, &yes, &|| MemProbe { room_mb: Some(1e9), rss_mb: 500.0 });
        assert_eq!(io.read, 0);
        assert!((io.peak_net_mb - (500.0 - held_bytes() as f64 / 1048576.0)).abs() < 1e-9);
        release_all();
        set_limited(None, false);
    }

    /// `window_io` reads nothing that would leave less than `ROOM_FLOOR_MB` of room,
    /// and releases spans from the far end when the room falls under it.
    #[test]
    fn window_io_keeps_the_room_floor() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(100_000);
        let (file, k) = file_with(dir.path(), "f", &f);
        set_limited(Some(&dir.path().join("f")), true);
        let yes = std::sync::atomic::AtomicBool::new(true);
        let no = std::sync::atomic::AtomicBool::new(false);
        // At the floor: nothing read (it is not cancelled, but stops reading; cancel to end).
        let tight = || MemProbe { room_mb: Some(ROOM_FLOOR_MB), rss_mb: 100.0 };
        std::thread::scope(|sc| {
            let h = sc.spawn(|| window_io(Some(&file), 0, 100_000, usize::MAX, &no, &tight));
            std::thread::sleep(std::time::Duration::from_millis(20));
            no.store(true, std::sync::atomic::Ordering::Relaxed);
            h.thread().unpark();
            assert_eq!(h.join().unwrap().read, 0);
        });
        // Under the floor by one span's worth: the far spans go, the head stays.
        for o in (0..10).map(|i| i * 4096u64) { assert!(put(&file, k, o, SpanData::Heap(f[o as usize..o as usize + 4096].to_vec()))); }
        let short = 2.0 * 4096.0 / 1048576.0;
        let io = window_io(None, 0, 0, 0, &yes, &|| MemProbe { room_mb: Some(ROOM_FLOOR_MB - short), rss_mb: 100.0 });
        assert_eq!(io.released, 2 * 4096);
        assert_eq!(held_spans().last().unwrap().end, 8 * 4096);
        assert_eq!(held_spans().first().unwrap().start, 0);
        release_all();
        set_limited(None, false);
    }

    /// Pressure that arrives while `window_io` watches (after its reads, before it is
    /// cancelled) releases spans from the far end until the floor is back.
    #[test]
    fn window_io_releases_spans_when_pressure_arrives_while_watching() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(100_000);
        let (file, _) = file_with(dir.path(), "f", &f);
        set_limited(Some(&dir.path().join("f")), true);
        // Room in KB, set by the test while the thread runs; released spans give it back.
        let room_kb = std::sync::atomic::AtomicU64::new(1 << 40);
        let probe = || MemProbe {
            room_mb: Some(room_kb.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1024.0
                + (100_000 - held_bytes().min(100_000)) as f64 / 1048576.0),
            rss_mb: 100.0 };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let io = std::thread::scope(|sc| {
            let h = sc.spawn(|| window_io(Some(&file), 0, 100_000, usize::MAX, &stop, &probe));
            let t0 = std::time::Instant::now();
            while held_bytes() < 100_000 && t0.elapsed().as_secs() < 30 { std::thread::yield_now(); }
            assert_eq!(held_bytes(), 100_000);
            // Three spans short of the floor: the three furthest go, while it watches.
            let short_kb = (3 * PREFETCH_SPAN / 1024) as u64;
            room_kb.store((ROOM_FLOOR_MB * 1024.0) as u64 - short_kb, std::sync::atomic::Ordering::Relaxed);
            while held_bytes() == 100_000 && t0.elapsed().as_secs() < 30 { std::thread::yield_now(); }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            h.thread().unpark();
            h.join().unwrap()
        });
        assert_eq!(io.read, 100_000);
        // Whole spans from the far end until 3 spans' worth are back: 1,696 (the last,
        // short one: 100,000 = 24 x 4,096 + 1,696) + 3 x 4,096.
        assert_eq!(io.released, 1696 + 3 * PREFETCH_SPAN, "{}", io.released);
        assert_eq!(held_spans().first().unwrap().start, 0);
        assert_eq!(held_bytes(), 100_000 - io.released);
        release_all();
        set_limited(None, false);
    }

    /// A read that blocks does not block the memory checks: pressure arriving while the
    /// reader is stuck releases spans already held, and cancelling ends the run once
    /// the read returns.
    #[test]
    fn window_io_watches_while_a_read_blocks() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(100_000);
        let (file, _) = file_with(dir.path(), "f", &f);
        set_limited(Some(&dir.path().join("f")), true);
        // The reader's third read blocks until the gate opens.
        let reads = std::sync::atomic::AtomicUsize::new(0);
        let gate = std::sync::atomic::AtomicBool::new(false);
        let read = |fl: &std::fs::File, buf: &mut [u8], at: u64| -> std::io::Result<()> {
            if reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 2 {
                while !gate.load(std::sync::atomic::Ordering::SeqCst) { std::thread::yield_now(); }
            }
            super::super::tiled::read_file_at(fl, buf, at)
        };
        let room_kb = std::sync::atomic::AtomicU64::new(1 << 40);
        let probe = || MemProbe { room_mb: Some(room_kb.load(std::sync::atomic::Ordering::SeqCst) as f64 / 1024.0
            + (2 * PREFETCH_SPAN - held_bytes().min(2 * PREFETCH_SPAN)) as f64 / 1048576.0), rss_mb: 100.0 };
        let stop = std::sync::atomic::AtomicBool::new(false);
        let io = std::thread::scope(|sc| {
            let h = sc.spawn(|| window_io_with(Some(&file), 0, 100_000, usize::MAX, &stop, &probe, &read));
            let t0 = std::time::Instant::now();
            // Two spans held, the third read stuck.
            while (held_bytes() < 2 * PREFETCH_SPAN || reads.load(std::sync::atomic::Ordering::SeqCst) < 3)
                && t0.elapsed().as_secs() < 30 { std::thread::yield_now(); }
            assert_eq!(held_bytes(), 2 * PREFETCH_SPAN);
            // One span short of the floor: the monitor releases the far span while the read blocks.
            room_kb.store((ROOM_FLOOR_MB * 1024.0) as u64 - (PREFETCH_SPAN / 1024) as u64, std::sync::atomic::Ordering::SeqCst);
            while held_bytes() == 2 * PREFETCH_SPAN && t0.elapsed().as_secs() < 30 { std::thread::yield_now(); }
            assert_eq!(held_spans(), vec![0..PREFETCH_SPAN as u64]);
            room_kb.store(1 << 40, std::sync::atomic::Ordering::SeqCst);
            stop.store(true, std::sync::atomic::Ordering::SeqCst);
            h.thread().unpark();
            gate.store(true, std::sync::atomic::Ordering::SeqCst);
            h.join().unwrap()
        });
        assert_eq!(io.released, PREFETCH_SPAN);
        assert!(io.read >= 2 * PREFETCH_SPAN);
        release_all();
        set_limited(None, false);
    }

    /// Without page-cache drops nothing is held: spans the cache mostly holds are
    /// skipped, the others are read once to warm the cache.
    #[test]
    fn window_io_without_drops_only_warms_the_cache() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        set_limited(None, false);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        let f = file_bytes(64 * 4096);
        std::fs::write(&path, &f).unwrap();
        std::fs::File::open(&path).unwrap().sync_all().unwrap();
        let run = |file: &std::fs::File| {
            let stop = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|sc| {
                let h = sc.spawn(|| window_io(Some(file), 0, f.len() as u64, usize::MAX, &stop, &plenty));
                std::thread::sleep(std::time::Duration::from_millis(300));
                stop.store(true, std::sync::atomic::Ordering::Relaxed);
                h.thread().unpark();
                h.join().unwrap()
            })
        };
        let file = std::fs::File::open(&path).unwrap();
        let r = Residency::of(&file);
        // Cached (just read): nothing read, nothing held.
        let _ = std::fs::read(&path).unwrap();
        if r.as_ref().is_some_and(|r| r.mostly_cached(0, f.len())) {
            assert_eq!(run(&file).read, 0);
        }
        assert_eq!(held_bytes(), 0);
        // Evicted: read (warming the cache), still nothing held.
        fadvise_dontneed(&file, 0, f.len() as u64);
        if r.as_ref().is_some_and(|r| !r.mostly_cached(0, f.len())) {
            assert!(run(&file).read > 0);
            assert!(r.as_ref().unwrap().mostly_cached(0, f.len()));
        }
        assert_eq!(held_bytes(), 0);
        // A cached prefix counts toward the limit: with a limit of 16 spans and the
        // first 8 cached, only 8 cold spans are read.
        if r.is_some() {
            let span = PREFETCH_SPAN as u64;
            let _ = std::fs::read(&path).unwrap();
            fadvise_dontneed(&file, 8 * span, f.len() as u64);
            if r.as_ref().unwrap().mostly_cached(0, 8 * span as usize) && !r.as_ref().unwrap().mostly_cached(8 * span, 8 * span as usize) {
                let stop = std::sync::atomic::AtomicBool::new(false);
                let io = std::thread::scope(|sc| {
                    let h = sc.spawn(|| window_io(Some(&file), 0, f.len() as u64, 16 * PREFETCH_SPAN, &stop, &plenty));
                    std::thread::sleep(std::time::Duration::from_millis(300));
                    stop.store(true, std::sync::atomic::Ordering::Relaxed);
                    h.thread().unpark();
                    h.join().unwrap()
                });
                assert!(io.read <= 8 * PREFETCH_SPAN, "{}", io.read);
            }
        }
        release_all();
    }

    /// Bytes read before a file was rewritten are not held under its new identity.
    #[test]
    fn put_refuses_bytes_of_a_rewritten_file() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(10_000);
        let (file, k) = file_with(dir.path(), "f", &f);
        let old = f[0..100].to_vec();
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Rewritten in place (same inode, new times) between the read and the put.
        std::fs::write(dir.path().join("f"), file_bytes(10_000).iter().map(|b| b ^ 1).collect::<Vec<u8>>()).unwrap();
        assert!(!put(&file, k, 0, SpanData::Heap(old)));
        assert_eq!(held_bytes(), 0);
        release_all();
    }

    /// Fills and releases from several threads at once see either a span's bytes or
    /// a hole, never anything else.
    #[test]
    fn concurrent_fill_and_release_serve_only_the_files_bytes() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        release_all();
        let dir = tempfile::tempdir().unwrap();
        let f = file_bytes(400_000);
        let (file, k) = file_with(dir.path(), "f", &f);
        for o in (0..100).map(|i| i * 4000u64) { assert!(put(&file, k, o, SpanData::Heap(f[o as usize..o as usize + 4000].to_vec()))); }
        std::thread::scope(|sc| {
            sc.spawn(|| for at in (0..100).map(|i| i * 4000u64) { release_before(Some(k), at); std::thread::yield_now(); });
            for t in 0..3u64 {
                let f = &f;
                sc.spawn(move || {
                    for i in 0..200u64 {
                        let lo = ((i * 7919 + t * 104_729) % 390_000) as usize;
                        let mut buf = vec![0xA5u8; 9_000];
                        let holes = fill(Some(k), lo as u64, &mut buf);
                        let mut hole = vec![false; buf.len()];
                        for h in holes { for j in h { hole[j] = true; } }
                        for j in 0..buf.len() { if !hole[j] { assert_eq!(buf[j], f[lo + j]); } }
                    }
                });
            }
        });
        release_all();
    }

    #[test]
    fn budget_is_zero_when_the_panel_fits_the_cache() {
        let plenty = Plan { limited: false, hold: 0, keep_cached: u64::MAX, drop_cache: false };
        // 1.4 GB of tiles in a 6 GiB container with ~5 GB of room: the page cache keeps them.
        assert_eq!(plan_for(1_432_000_000, 124_000_000, 5000.0, None), plenty);
        // 3.5 GB (one user's chrX) in ~4.6 GB of room with a 190 MB bitmatrix: still kept.
        assert_eq!(plan_for(3_506_000_000, 190_000_000, 4600.0, None), plenty);
        // Any host with tens of GB free.
        assert_eq!(plan_for(7_500_000_000, 511_000_000, 100_000.0, None), plenty);
        // 7.5 GB in ~4.6 GB of room: a third of what is above 1 GB held; cached, what
        // the room keeps beyond it and 3.5 bitmatrices.
        let p = plan_for(7_500_000_000, 511_000_000, 4600.0, None);
        let above = (4600.0 - RETAIN_RESERVE_MB) * 1048576.0 - 3.5 * 511e6;
        let hold = (above * RETAIN_FRACTION) as usize;
        let keep = (4600.0 * 1048576.0 - hold as f64 - 3.5 * 511e6) as u64;
        assert_eq!(p, Plan { limited: true, hold, keep_cached: keep, drop_cache: true });
        // An explicit hold is capped at half the room above the reserve and growth.
        assert_eq!(plan_for(7_500_000_000, 511_000_000, 4600.0, Some(6 << 30)).hold, (above * 0.5) as usize);
        // The knobs: plenty is never limited, even when the growth exceeds the room;
        // limited always is; drops only when asked; unknown room is not limited.
        assert!(!plan_with(0, 1_500_000_000, Some(1000.0), None, Some("plenty"), true).limited);
        assert!(plan_with(1, 0, Some(1e9), None, Some("limited"), false).limited);
        assert!(!plan_with(7_500_000_000, 511_000_000, Some(4600.0), None, None, false).drop_cache);
        assert!(plan_with(7_500_000_000, 511_000_000, Some(4600.0), None, None, true).drop_cache);
        assert!(!plan_with(7_500_000_000, 511_000_000, None, None, None, true).limited);
        assert_eq!(plan_with(7_500_000_000, 511_000_000, Some(4600.0), None, None, false).hold, 0);
        assert!(plan_with(7_500_000_000, 511_000_000, Some(4600.0), Some("auto"), None, false).hold > 0);
        // A bitmatrix whose growth takes the whole room: nothing held.
        assert_eq!(plan_for(7_500_000_000, 1_500_000_000, 4600.0, Some(6 << 30)).hold, 0);
        // No room above the reserve: limited, nothing held or kept cached.
        assert_eq!(plan_for(7_500_000_000, 511_000_000, 900.0, None), Plan { limited: true, hold: 0, keep_cached: 0, drop_cache: true });
        // An explicit hold: the cache keeps the room's worth beyond it.
        let p = plan_for(7_500_000_000, 511_000_000, 4600.0, Some(0));
        assert_eq!(p, Plan { limited: true, hold: 0, keep_cached: (4600.0 * 1048576.0 - 3.5 * 511e6) as u64, drop_cache: true });
        // Window prefetch: what is held, plus three quarters of what is left above
        // the expected regrowth and the reserve.
        assert_eq!(prefetch_limit_for(100, 4000.0, 1000.0), 100 + (2488.0 * 0.75 * 1048576.0) as usize);
        assert_eq!(prefetch_limit_for(100, 1500.0, 1000.0), 100);
    }
}
