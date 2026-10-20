//! Byte admission for nested single-sample work and pressure-based reclamation
//! for whole-sample Rayon waves. Blocking permit waits stay outside Rayon.

use std::sync::{Condvar, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Whole samples stay on Rayon when there is enough independent sample work.
/// The single-sample path retains byte admission and optional nested work.
#[derive(Clone, Copy, Default)]
pub(crate) enum SampleExecution<'a> {
    #[default]
    Unbudgeted,
    Admitted(&'a MemoryBudget),
    Waves(&'a WaveReclaimer),
}

impl<'a> SampleExecution<'a> {
    pub fn permits(self) -> Option<&'a MemoryBudget> {
        match self { Self::Admitted(memory) => Some(memory), _ => None }
    }

    pub fn finish_sample(self) {
        if let Self::Waves(reclaimer) = self { reclaimer.finish_sample(); }
    }

    /// Resident size above which a whole-sample wave's phasing returns the
    /// snapshots it has consumed before its call ends: the reclaimer's
    /// threshold. Other executions keep them until the call ends.
    pub fn release_above(self) -> Option<usize> {
        match self { Self::Waves(reclaimer) => Some(reclaimer.threshold), _ => None }
    }
}

/// Release allocator residency only under pressure, after all of a sample's
/// large scratch has died on its allocating worker. This is an RSS trigger,
/// not a hard process limit: concurrent live allocations cannot be collected.
/// Leave a quarter of the target for work already running on other workers.
pub(crate) struct WaveReclaimer {
    threshold: usize,
    samples: AtomicUsize,
    collections: AtomicUsize,
}

impl WaveReclaimer {
    pub fn new(target: usize) -> Self {
        Self { threshold: target - target / 4,
            samples: AtomicUsize::new(0), collections: AtomicUsize::new(0) }
    }

    fn collect_if_needed(&self, rss: Option<usize>, collect: impl FnOnce()) {
        self.samples.fetch_add(1, Ordering::Relaxed);
        if rss.is_some_and(|rss| rss >= self.threshold) {
            collect();
            self.collections.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn finish_sample(&self) {
        self.collect_if_needed(resident_bytes(), collect_local);
    }

    pub fn counts(&self) -> (usize, usize) {
        (self.samples.load(Ordering::Relaxed), self.collections.load(Ordering::Relaxed))
    }
}

/// Read only this process. A peak-RSS fallback would keep collection enabled
/// forever after one peak, so platforms without a current reading return None.
pub(crate) fn resident_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let pages = statm.split_whitespace().nth(1)?.parse::<usize>().ok()?;
        // SAFETY: sysconf has no pointer arguments or side effects here.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 { return None; }
        pages.checked_mul(page_size as usize)
    }
    #[cfg(not(target_os = "linux"))]
    { None }
}

#[derive(Default)]
struct Usage { fixed: usize, work: usize, work_jobs: usize, waiting: usize, peak: usize }

pub(crate) struct MemoryBudget {
    limit: usize,
    fixed_limit: usize,
    shared: usize,
    usage: Mutex<Usage>,
    changed: Condvar,
    #[cfg(test)]
    allow_parallel_fb: bool,
    #[cfg(test)]
    parallel_fb_grants: std::sync::atomic::AtomicUsize,
}

pub(crate) struct Permit<'a> { budget: &'a MemoryBudget, bytes: usize, fixed: bool }

impl MemoryBudget {
    pub fn new(limit: usize, shared: usize) -> Self {
        assert!(limit > shared, "lcWGS memory budget must exceed shared process storage");
        Self { limit, fixed_limit: shared + (limit - shared) / 2, shared,
            usage: Mutex::new(Usage { fixed: shared, ..Usage::default() }), changed: Condvar::new(),
            #[cfg(test)]
            allow_parallel_fb: true,
            #[cfg(test)]
            parallel_fb_grants: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn work_limit(&self) -> usize { self.limit - self.fixed_limit }

    pub fn try_chunk(&self, bytes: usize) -> Option<Permit<'_>> {
        assert!(bytes <= self.fixed_limit - self.shared,
            "one lcWGS chunk exceeds the state allowance; increase LCWGS_MEM_BUDGET_GB");
        let mut used = self.usage.lock().unwrap();
        if used.fixed + bytes > self.fixed_limit
            || bytes > self.limit.saturating_sub(used.fixed + used.work) { return None; }
        used.fixed += bytes;
        used.peak = used.peak.max(used.fixed + used.work);
        Some(Permit { budget: self, bytes, fixed: true })
    }

    #[cfg(test)]
    fn try_work(&self, bytes: usize) -> Option<Permit<'_>> {
        let mut used = self.usage.lock().unwrap();
        self.admit_work(&mut used, bytes)
    }

    fn admit_work(&self, used: &mut Usage, bytes: usize) -> Option<Permit<'_>> {
        if bytes > self.limit.saturating_sub(used.fixed + used.work) { return None; }
        used.work += bytes;
        used.work_jobs += 1;
        used.peak = used.peak.max(used.fixed + used.work);
        Some(Permit { budget: self, bytes, fixed: false })
    }

    /// Prefer whole sample jobs when they already fill the pool. A nested join
    /// can suspend its parent's packs and permit while executing another job;
    /// with a full queue this adds storage and delays the iteration barrier.
    /// Rows need no extra bytes, so memory waiters alone must not disable them.
    pub fn parallel_rows(&self) -> bool {
        let used = self.usage.lock().unwrap();
        used.work_jobs < rayon::current_num_threads()
    }

    /// Optional parallel work never waits on a worker. A denied request uses
    /// the sequential computation under the existing primary permit.
    pub fn try_parallel_fb(&self, bytes: usize) -> Option<Permit<'_>> {
        #[cfg(test)]
        if !self.allow_parallel_fb { return None; }
        let mut used = self.usage.lock().unwrap();
        if used.waiting != 0 || used.work_jobs >= rayon::current_num_threads() { return None; }
        let permit = self.admit_work(&mut used, bytes)?;
        #[cfg(test)]
        self.parallel_fb_grants.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(permit)
    }

    #[cfg(test)]
    pub fn set_parallel_fb(&mut self, enabled: bool) { self.allow_parallel_fb = enabled; }

    #[cfg(test)]
    pub fn parallel_fb_grants(&self) -> usize {
        self.parallel_fb_grants.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn work(&self, bytes: usize) -> Permit<'_> {
        assert!(rayon::current_thread_index().is_none(), "memory admission must run outside Rayon");
        assert!(bytes <= self.work_limit(),
            "one lcWGS sample exceeds the work allowance; increase LCWGS_MEM_BUDGET_GB");
        let mut used = self.usage.lock().unwrap();
        used.waiting += 1;
        while bytes > self.limit.saturating_sub(used.fixed + used.work) {
            used = self.changed.wait(used).unwrap();
        }
        used.waiting -= 1;
        self.admit_work(&mut used, bytes).unwrap()
    }

    pub fn run<R: Send>(&self, bytes: usize, f: impl FnOnce() -> R + Send) -> R {
        let _permit = self.work(bytes);
        rayon::join(|| { let out = f(); collect_local(); out }, || ()).0
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut used = self.budget.usage.lock().unwrap();
        if self.fixed { used.fixed -= self.bytes; } else {
            used.work -= self.bytes;
            used.work_jobs -= 1;
        }
        self.budget.changed.notify_all();
    }
}

/// Return pages on the allocating worker before its allowance can be reused.
pub(crate) fn collect_local() {
    #[cfg(feature = "mimalloc")]
    unsafe { libmimalloc_sys::mi_collect(true); }
}

/// Bounded native coordinators keep permit waits off the compute pool. The
/// result order is the input order even when tasks finish out of order.
pub(crate) fn map_samples<T: Send>(
    n: usize, workers: usize, f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let next = AtomicUsize::new(0);
    let output = Mutex::new((0..n).map(|_| None).collect::<Vec<_>>());
    std::thread::scope(|scope| {
        for _ in 0..n.min(workers.max(1)) {
            let (f, next, output) = (&f, &next, &output);
            scope.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n { break; }
                let value = f(i);
                output.lock().unwrap()[i] = Some(value);
                collect_local();
            });
        }
    });
    output.into_inner().unwrap().into_iter().map(Option::unwrap).collect()
}

/// Admit fitting chunks as soon as a completed core has been merged. A permit
/// includes the returned result until the coordinator finishes that merge.
pub(crate) fn run_chunks<T: Send>(
    costs: &[usize], budget: &MemoryBudget, max_live: usize,
    process: impl Fn(usize) -> T + Sync, mut merge: impl FnMut(usize, T),
) {
    use std::sync::mpsc;
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::channel();
        let mut pending: Vec<_> = (0..costs.len()).collect();
        pending.sort_by_key(|&i| std::cmp::Reverse(costs[i]));
        let mut live = 0;
        while live != 0 || !pending.is_empty() {
            let mut i = 0;
            while live < max_live.max(1) && i < pending.len() {
                let c = pending[i];
                if let Some(permit) = budget.try_chunk(costs[c]) {
                    pending.remove(i);
                    live += 1;
                    let (tx, process) = (tx.clone(), &process);
                    scope.spawn(move || {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| process(c)));
                        collect_local();
                        let _ = tx.send((c, result, permit));
                    });
                } else { i += 1; }
            }
            assert!(live > 0, "no lcWGS chunk fits its memory allowance");
            let (c, result, permit) = rx.recv().unwrap();
            match result {
                Ok(result) => merge(c, result),
                Err(panic) => std::panic::resume_unwind(panic),
            }
            collect_local();
            drop(permit);
            live -= 1;
        }
    });
}

/// Two packed axes, compact HL/map arrays, checkpointed FB, and local lists.
/// The factor of two covers vector capacity growth; retained caches are charged
/// separately for every worker, including idle workers.
pub(crate) fn fb_bytes(sites: usize, k: usize) -> usize {
    let stride = ((sites as f64).sqrt().ceil() as usize).max(1);
    let checkpoints = sites.div_ceil(stride);
    2 * (sites * (2 * k.div_ceil(64) * 8 + 96)
        + (checkpoints + stride + 4) * k * 4)
}

/// The second phaser of a two-pass rephase: its K x 8 state and copies of the
/// per-site typing, plus the per-boundary HProbs and the forward sums stored at
/// meeting points.
pub(crate) fn two_pass_phase_bytes(sites: usize, k: usize) -> usize {
    sites * 160 + k * 64 + 4096
}

/// The K-wide snapshot size is counted from the actual sampled alleles before
/// allocation. Other phaser metadata and its pack have bounded site/K axes.
pub(crate) fn phase_bytes(sites: usize, k: usize, snapshots: usize) -> usize {
    snapshots + 2 * (sites * (k.div_ceil(64) * 8 + 160) + k * 128)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wave_collection_follows_current_pressure_without_sticking_at_a_past_peak() {
        let reclaimer = WaveReclaimer::new(400);
        let calls = AtomicUsize::new(0);
        for rss in [None, Some(299), Some(300), Some(450), Some(200)] {
            reclaimer.collect_if_needed(rss, || { calls.fetch_add(1, Ordering::Relaxed); });
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        assert_eq!(reclaimer.counts(), (5, 2));
        assert!(SampleExecution::Waves(&reclaimer).permits().is_none());
    }

    #[test]
    fn permits_charge_idle_state_and_release_on_unwind() {
        let b = MemoryBudget::new(1000, 100);
        let c = b.try_chunk(400).unwrap();
        assert!(b.try_chunk(51).is_none());
        let w = b.try_work(500).unwrap();
        assert!(b.try_work(1).is_none());
        drop(w);
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _p = b.try_work(500).unwrap();
            panic!("release");
        }));
        assert_eq!(b.usage.lock().unwrap().work, 0);
        drop(c);
        assert_eq!(b.usage.lock().unwrap().fixed, 100);
        assert!(b.usage.lock().unwrap().peak <= 1000);
    }

    #[test]
    fn chunk_admission_counts_work_using_free_state_space() {
        let b = MemoryBudget::new(1000, 100);
        let a = b.try_work(300).unwrap();
        let z = b.try_work(300).unwrap();
        assert!(b.try_chunk(350).is_none());
        drop(a);
        assert!(b.try_chunk(350).is_some());
        drop(z);
        assert!(b.usage.lock().unwrap().peak <= 1000);
    }

    #[test]
    fn nested_work_yields_to_full_pool_and_waiting_samples() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
        pool.install(|| {
            let b = MemoryBudget::new(1000, 0);
            let primary = b.try_work(200).unwrap();
            assert!(b.parallel_rows());
            let extra = b.try_parallel_fb(200).unwrap();
            assert!(!b.parallel_rows());
            assert!(b.try_parallel_fb(1).is_none());
            drop(extra);
            b.usage.lock().unwrap().waiting = 1;
            assert!(b.parallel_rows());
            assert!(b.try_parallel_fb(1).is_none());
            b.usage.lock().unwrap().waiting = 0;
            assert!(b.try_parallel_fb(801).is_none());
            assert!(b.parallel_rows());
            drop(primary);
            assert_eq!(b.usage.lock().unwrap().work_jobs, 0);
        });
    }

    #[test]
    fn byte_admission_keeps_more_than_three_chunks_runnable() {
        let b = MemoryBudget::new(1000, 0);
        let chunks: Vec<_> = (0..6).map(|_| b.try_chunk(80).unwrap()).collect();
        assert!(b.try_chunk(80).is_none());
        let work = b.try_work(500).unwrap();
        assert!(b.try_work(21).is_none());
        drop(work);
        drop(chunks);
        assert_eq!(b.usage.lock().unwrap().fixed, 0);
        assert!(b.usage.lock().unwrap().peak <= 1000);
    }

    #[test]
    fn oversized_work_is_rejected_before_execution() {
        let b = MemoryBudget::new(100, 0);
        let ran = std::sync::atomic::AtomicBool::new(false);
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            b.run(51, || ran.store(true, std::sync::atomic::Ordering::Relaxed));
        }));
        assert!(err.is_err());
        assert!(!ran.load(std::sync::atomic::Ordering::Relaxed));
    }

    #[test]
    fn completion_admits_replacement_before_slow_chunk_finishes() {
        let budget = MemoryBudget::new(100, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let rx = Mutex::new(rx);
        let mut order = Vec::new();
        run_chunks(&[25, 25, 25], &budget, 2, |i| {
            if i == 0 { rx.lock().unwrap().recv_timeout(std::time::Duration::from_secs(10)).unwrap(); }
            if i == 2 { tx.send(()).unwrap(); }
            i
        }, |i, value| { assert_eq!(i, value); order.push(i); });
        assert_eq!(order[0], 1);
        assert_eq!(order.len(), 3);
        assert_eq!(budget.usage.lock().unwrap().fixed, 0);
    }

    #[test]
    fn waiting_coordinators_leave_one_compute_worker_free() {
        let b = MemoryBudget::new(100, 0);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let got = map_samples(12, 4, |i| {
            let _permit = b.work(50);
            pool.install(|| i * i)
        });
        assert_eq!(got, (0..12).map(|i| i * i).collect::<Vec<_>>());
        assert_eq!(b.usage.lock().unwrap().work, 0);
    }
}
