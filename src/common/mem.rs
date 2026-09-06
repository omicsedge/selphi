//! Give freed memory back between pipeline stages.
//!
//! mimalloc keeps a freed page on the heap of the thread that allocated it and
//! purges it lazily, the next time THAT thread does allocator work. The
//! imputation window loop frees most of a window's memory — the per-target
//! weight CSRs, several GB at biobank scale — on the main thread, while the rayon
//! workers that allocated it sit idle until the next window. Nothing purges, RSS
//! never comes down, and the next window's allocations land on top.
//!
//! Measured on MESA 200 × TOPMed chr20: `MIMALLOC_PURGE_DELAY=0` (purge on every
//! free) took the peak from 11.34 GB to 8.88 GB — the `win2 start` RSS marker
//! read 4.1 GB against 9.96 — at +8% wall from the purge churn on every free.
//! This does the same surgically: one forced collect on every worker thread at
//! each window boundary, where the big frees have just happened. mimalloc's own
//! documentation of `mi_collect` describes exactly this shape: "when a long
//! running thread allocates a lot of blocks that are freed by other threads it
//! may improve resource usage by calling this every once in a while".

/// Force every rayon worker (and the calling thread) to return its freed pages
/// to the OS. Cheap — a few ms per call — and a no-op without the mimalloc
/// feature. `SELPHI_NO_MI_COLLECT=1` disables it for A/B.
pub fn release_freed_memory() {
    #[cfg(feature = "mimalloc")]
    {
        use std::sync::OnceLock;
        static OFF: OnceLock<bool> = OnceLock::new();
        if *OFF.get_or_init(|| crate::config::is_one("SELPHI_NO_MI_COLLECT")) { return; }
        // SAFETY: mi_collect only walks the calling thread's heap and the
        // allocator's abandoned-segment list; it frees nothing that is in use.
        rayon::broadcast(|_| unsafe { libmimalloc_sys::mi_collect(true) });
        unsafe { libmimalloc_sys::mi_collect(true) };
    }
}
