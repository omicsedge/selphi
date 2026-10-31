//! Byte admission for nested single-sample work and pressure-based reclamation
//! for whole-sample Rayon waves. Blocking permit waits stay outside Rayon.

use std::sync::{Arc, Condvar, Mutex};
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
struct Usage {
    fixed: usize, work: usize, work_jobs: usize, waiting: usize, alone_waiting: usize, peak: usize,
    /// Reservations held for a chunk not yet admitted (chunk, bytes), part of `fixed`.
    reserved: Vec<(usize, usize)>,
}

pub(crate) struct MemoryBudget {
    limit: usize,
    fixed_limit: usize,
    shared: usize,
    usage: Mutex<Usage>,
    changed: Condvar,
    /// An explicit LCWGS_MEM_BUDGET_GB is a hard limit: a chunk or task above its half stops the
    /// run. The default allowance is sized from the average chunk, so there such an item runs
    /// alone instead (see [`MemoryBudget::lenient`]).
    strict: bool,
    oversized: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    allow_parallel_fb: bool,
    #[cfg(test)]
    parallel_fb_grants: std::sync::atomic::AtomicUsize,
}

pub(crate) struct Permit<'a> { budget: &'a MemoryBudget, bytes: usize, fixed: bool, reserved_for: Option<usize> }

impl MemoryBudget {
    pub fn new(limit: usize, shared: usize) -> Self {
        assert!(limit > shared, "lcWGS memory budget must exceed shared process storage");
        Self { limit, fixed_limit: shared + (limit - shared) / 2, shared,
            usage: Mutex::new(Usage { fixed: shared, ..Usage::default() }), changed: Condvar::new(),
            strict: true, oversized: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            allow_parallel_fb: true,
            #[cfg(test)]
            parallel_fb_grants: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// The default allowance. Calls clustered in a few chunks (an exome, a variant-only file) put
    /// most sites in one chunk, whose state or phasing work can exceed its half of an allowance
    /// sized from the average chunk: such a chunk is admitted only when no other chunk is live,
    /// and such work only when no other work runs, so the run goes on above the allowance.
    pub fn lenient(mut self) -> Self { self.strict = false; self }

    pub fn work_limit(&self) -> usize { self.limit - self.fixed_limit }

    /// Whether a chunk of `bytes` state can be admitted at all: always under the
    /// default (lenient) allowance, where an oversized chunk runs alone; under an
    /// explicit budget only within the chunk half of it ([`Self::try_chunk`]
    /// refuses the rest).
    pub fn admits_chunk(&self, bytes: usize) -> bool {
        !self.strict || bytes <= self.fixed_limit - self.shared
    }

    fn note_oversized(&self, what: &str, bytes: usize) {
        if !self.oversized.swap(true, std::sync::atomic::Ordering::Relaxed) {
            crate::selphi_info!("  lcWGS: one {} needs {:.2} GB, above its share of the allowance; it runs alone",
                what, bytes as f64 / 1e9);
        }
    }

    /// Charge `bytes` of state held for chunk `chunk` before it starts (its panel
    /// side built ahead, see `pipeline::PanelAhead`) until the permit drops:
    /// never waits or refuses (the caller checked the memory itself). Other
    /// chunks' admission counts it; `chunk`'s own does not
    /// ([`Self::try_chunk_for`]), which it would otherwise wait for forever.
    pub fn reserve(&self, chunk: usize, bytes: usize) -> Permit<'_> {
        let mut used = self.usage.lock().unwrap();
        used.fixed += bytes;
        used.reserved.push((chunk, bytes));
        used.peak = used.peak.max(used.fixed + used.work);
        self.changed.notify_all();
        Permit { budget: self, bytes, fixed: true, reserved_for: Some(chunk) }
    }

    #[cfg(test)]
    pub fn try_chunk(&self, bytes: usize) -> Option<Permit<'_>> {
        self.try_chunk_for(usize::MAX, bytes)
    }

    /// Admit chunk `chunk`'s state, not counting what is reserved for it.
    pub fn try_chunk_for(&self, chunk: usize, bytes: usize) -> Option<Permit<'_>> {
        let oversized = bytes > self.fixed_limit - self.shared;
        assert!(!oversized || !self.strict,
            "one lcWGS chunk exceeds the state allowance; increase LCWGS_MEM_BUDGET_GB");
        let mut used = self.usage.lock().unwrap();
        let own: usize = used.reserved.iter().filter(|&&(c, _)| c == chunk).map(|&(_, b)| b).sum();
        let fixed = used.fixed - own;
        if oversized {
            if fixed != self.shared || used.work != 0 { return None; }
            self.note_oversized("chunk", bytes);
        } else if fixed + bytes > self.fixed_limit
            || bytes > self.limit.saturating_sub(fixed + used.work) { return None; }
        used.fixed += bytes;
        used.peak = used.peak.max(used.fixed + used.work);
        Some(Permit { budget: self, bytes, fixed: true, reserved_for: None })
    }

    #[cfg(test)]
    fn try_work(&self, bytes: usize) -> Option<Permit<'_>> {
        let mut used = self.usage.lock().unwrap();
        self.admit_work(&mut used, bytes)
    }

    fn admit_work(&self, used: &mut Usage, bytes: usize) -> Option<Permit<'_>> {
        if bytes > self.limit.saturating_sub(used.fixed + used.work) { return None; }
        Some(self.charge_work(used, bytes))
    }

    fn charge_work(&self, used: &mut Usage, bytes: usize) -> Permit<'_> {
        used.work += bytes;
        used.work_jobs += 1;
        used.peak = used.peak.max(used.fixed + used.work);
        Permit { budget: self, bytes, fixed: false, reserved_for: None }
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
        assert!(!self.strict || bytes <= self.work_limit(),
            "one lcWGS sample exceeds the work allowance; increase LCWGS_MEM_BUDGET_GB");
        let mut used = self.usage.lock().unwrap();
        used.waiting += 1;
        // Lenient only: work that cannot fit beside the live chunks even alone waits for the other
        // work to finish, and later requests wait behind it so that it is not starved. A request
        // that fitted when it arrived becomes such work at a later wake-up when the live state
        // grew meanwhile (a chunk admitted, a panel built ahead reserved): it could otherwise wait
        // for room that never comes. Once alone it stays alone until admitted, keeping its
        // precedence over later requests.
        let mut alone = false;
        loop {
            if !alone && !self.strict && bytes > self.limit.saturating_sub(used.fixed) {
                used.alone_waiting += 1;
                alone = true;
            }
            if alone {
                if used.work == 0 {
                    used.alone_waiting -= 1;
                    used.waiting -= 1;
                    self.note_oversized("task", bytes);
                    return self.charge_work(&mut used, bytes);
                }
            } else if used.alone_waiting == 0 && bytes <= self.limit.saturating_sub(used.fixed + used.work) {
                used.waiting -= 1;
                return self.admit_work(&mut used, bytes).unwrap();
            }
            used = self.changed.wait(used).unwrap();
        }
    }

    pub fn run<R: Send>(&self, bytes: usize, f: impl FnOnce() -> R + Send) -> R {
        let permit = self.work(bytes);
        self.run_with(permit, f)
    }

    /// The permit [`Self::run`] would wait for, when it can be granted at once
    /// without overtaking a waiting request; `None` otherwise.
    pub fn try_permit(&self, bytes: usize) -> Option<Permit<'_>> {
        assert!(rayon::current_thread_index().is_none(), "memory admission must run outside Rayon");
        assert!(!self.strict || bytes <= self.work_limit(),
            "one lcWGS sample exceeds the work allowance; increase LCWGS_MEM_BUDGET_GB");
        let mut used = self.usage.lock().unwrap();
        if used.waiting != 0 || used.alone_waiting != 0 { return None; }
        self.admit_work(&mut used, bytes)
    }

    /// Run `f` under a permit granted by [`Self::try_permit`] (as [`Self::run`] does),
    /// on this thread's chain pool when it has one (see [`run_chunks`]).
    pub fn run_with<R: Send>(&self, permit: Permit<'_>, f: impl FnOnce() -> R + Send) -> R {
        let _permit = permit;
        let job = || { let out = f(); collect_local(); out };
        match chain_pool() {
            Some(pool) => in_chain_pool(Some(pool), job),
            None => rayon::join(job, || ()).0,
        }
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut used = self.budget.usage.lock().unwrap();
        if let Some(c) = self.reserved_for {
            let i = used.reserved.iter().position(|&r| r == (c, self.bytes)).expect("a held reservation");
            used.reserved.swap_remove(i);
        }
        if self.fixed { used.fixed -= self.bytes; } else {
            used.work -= self.bytes;
            used.work_jobs -= 1;
        }
        self.budget.changed.notify_all();
    }
}

thread_local! {
    /// The pool of the Gibbs chain this thread works for (see [`run_chunks`]).
    static CHAIN_POOL: std::cell::RefCell<Option<Arc<rayon::ThreadPool>>> = const { std::cell::RefCell::new(None) };
}

/// This thread's chain pool (see [`run_chunks`]): the native threads a chain
/// spawns take it along, so that their parallel work stays in it.
pub(crate) fn chain_pool() -> Option<Arc<rayon::ThreadPool>> {
    CHAIN_POOL.with(|p| p.borrow().clone())
}

/// Run `f` here with `pool` as this thread's chain pool (restored after, also on unwinding).
pub(crate) fn with_chain_pool<R>(pool: Option<Arc<rayon::ThreadPool>>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<Arc<rayon::ThreadPool>>);
    impl Drop for Restore {
        fn drop(&mut self) { CHAIN_POOL.with(|p| *p.borrow_mut() = self.0.take()); }
    }
    let _restore = Restore(CHAIN_POOL.with(|p| std::mem::replace(&mut *p.borrow_mut(), pool)));
    f()
}

/// Run `f` on a worker of `pool`, which is its chain pool there; `None` runs it here.
pub(crate) fn in_chain_pool<R: Send>(pool: Option<Arc<rayon::ThreadPool>>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => {
            let own = pool.clone();
            pool.install(move || with_chain_pool(Some(own), f))
        }
        None => f(),
    }
}

/// Have every worker of this thread's chain pool release its kernel scratch
/// and return the pages of blocks other threads freed (results and packs cross
/// threads): an idle worker would hold them until it next allocates, and the
/// pool's workers take turns at the forward-backwards. Each worker does it when
/// it next looks for work (the chain's coordinator waits on its worker, so a
/// blocking broadcast could wait for it forever). Nothing outside a chain pool.
pub(crate) fn collect_chain_pool() {
    if CHAIN_POOL.with(|p| p.borrow().is_some()) {
        let collect = || { super::hmm::release_worker_scratch(); collect_local(); };
        rayon::spawn_broadcast(move |_| collect());
        collect();
    }
}

/// A Gibbs chain's own pool (see [`run_chunks`]): `threads` compute workers
/// and one for the chain's coordinator, which runs its loop there and waits
/// on it while its sample threads work. `None` when it cannot be built (the
/// chain then runs in the global pool, as before).
pub(crate) fn new_chain_pool(threads: usize) -> Option<Arc<rayon::ThreadPool>> {
    rayon::ThreadPoolBuilder::new().num_threads(threads + 1).stack_size(16 << 20)
        .thread_name(|i| format!("lcwgs-chain-{i}")).build().ok().map(Arc::new)
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
    let pool = chain_pool();
    std::thread::scope(|scope| {
        for _ in 0..n.min(workers.max(1)) {
            let (f, next, output, pool) = (&f, &next, &output, pool.clone());
            scope.spawn(move || with_chain_pool(pool, || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                if i >= n { break; }
                let value = f(i);
                output.lock().unwrap()[i] = Some(value);
                collect_local();
            }));
        }
    });
    output.into_inner().unwrap().into_iter().map(Option::unwrap).collect()
}

/// Admit fitting chunks as soon as a completed core has been merged. A permit
/// includes the returned result until the coordinator finishes that merge.
///
/// Each Gibbs chain (a chunk's, and the deep pass of a concurrent SPLIT, see
/// `pipeline::run_chunked_gibbs`) runs in a pool of its own ([`new_chain_pool`];
/// idle workers sleep). In one shared pool a worker waiting inside a join takes
/// any queued job, so a chain stalled behind another chunk's whole
/// forward-backward or phasing call, its own calls queued behind busy workers,
/// and a chain's coordinator, waiting in a join, could run the other SPLIT
/// chain on its stack and wait for it there at their next selection sweep.
/// Which thread runs a job does not change any result.
pub(crate) fn run_chunks<T: Send>(
    costs: &[usize], budget: &MemoryBudget, max_live: usize, threads: usize,
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
                if let Some(permit) = budget.try_chunk_for(c, costs[c]) {
                    pending.remove(i);
                    live += 1;
                    let (tx, process) = (tx.clone(), &process);
                    scope.spawn(move || {
                        let pool = new_chain_pool(threads);
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| in_chain_pool(pool, || process(c))));
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
    fn reservation_counts_for_other_chunks_not_its_own() {
        let b = MemoryBudget::new(1000, 100);
        let r = b.reserve(7, 300);
        assert!(b.try_chunk_for(3, 201).is_none(), "another chunk sees the reservation");
        let own = b.try_chunk_for(7, 400).expect("its own chunk does not wait for it");
        drop(own);
        // An oversized chunk runs alone: its own reservation does not count as company.
        let lenient = MemoryBudget::new(1000, 100).lenient();
        let r2 = lenient.reserve(5, 50);
        assert!(lenient.try_chunk_for(4, 600).is_none());
        let big = lenient.try_chunk_for(5, 600).expect("alone but for its own reservation");
        drop((big, r2, r));
        assert_eq!(b.usage.lock().unwrap().fixed, 100);
        assert!(b.usage.lock().unwrap().reserved.is_empty());
        assert_eq!(lenient.usage.lock().unwrap().fixed, 100);
    }

    /// Work waiting beside other work becomes work that runs alone when a later
    /// reservation leaves it no room beside the live chunks (it waited forever);
    /// work that waits alone keeps its precedence when a chunk ends meanwhile.
    #[test]
    fn waiting_work_turns_alone_after_a_reservation() {
        let b = MemoryBudget::new(1000, 0).lenient();
        let a = b.try_chunk_for(0, 400).unwrap();
        let c = b.try_chunk_for(2, 50).unwrap();
        let x = b.try_work(200).unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let b = &b;
            s.spawn(move || { let _w = b.work(540); tx.send(()).unwrap(); });
            while b.usage.lock().unwrap().waiting == 0 { std::thread::yield_now(); }
            let r = b.reserve(1, 100);
            drop(c);
            drop(x);
            rx.recv_timeout(std::time::Duration::from_secs(20)).expect("the waiting work was admitted");
            drop(r);
        });
        drop(a);
        {
            let used = b.usage.lock().unwrap();
            assert_eq!((used.fixed, used.work, used.waiting, used.alone_waiting), (0, 0, 0, 0));
        }
        // Alone stays alone: a chunk ending while it waits does not let later work overtake it.
        let big = b.try_chunk_for(0, 500).unwrap();
        let x = b.try_work(400).unwrap();
        std::thread::scope(|s| {
            let b = &b;
            let h = s.spawn(move || { let _w = b.work(600); });
            while b.usage.lock().unwrap().alone_waiting == 0 { std::thread::yield_now(); }
            drop(big);
            assert!(b.try_permit(200).is_none(), "a waiting alone request goes first");
            drop(x);
            h.join().unwrap();
        });
        let used = b.usage.lock().unwrap();
        assert_eq!((used.fixed, used.work, used.waiting, used.alone_waiting), (0, 0, 0, 0));
    }

    #[test]
    fn try_permit_never_waits_nor_overtakes() {
        let b = MemoryBudget::new(1000, 0);
        let p = b.try_permit(450).expect("fits");
        let q = b.try_permit(450).expect("fits");
        assert!(b.try_permit(101).is_none(), "does not fit");
        assert_eq!(b.run_with(b.try_permit(100).unwrap(), || 7), 7);
        drop(q);
        b.usage.lock().unwrap().waiting = 1;
        assert!(b.try_permit(1).is_none(), "a waiting request goes first");
        b.usage.lock().unwrap().waiting = 0;
        drop(p);
        assert_eq!(b.usage.lock().unwrap().work, 0);
    }

    #[test]
    fn admits_chunk_matches_try_chunk_admission() {
        let strict = MemoryBudget::new(1000, 100);
        assert!(strict.admits_chunk(450) && !strict.admits_chunk(451));
        assert!(strict.try_chunk(450).is_some());
        let lenient = MemoryBudget::new(1000, 100).lenient();
        assert!(lenient.admits_chunk(451) && lenient.admits_chunk(10_000));
        assert!(lenient.try_chunk(10_000).is_some());
    }

    #[test]
    fn lenient_budget_runs_an_oversized_chunk_and_its_work_alone() {
        let b = MemoryBudget::new(1000, 100).lenient();
        let big = b.try_chunk(600).unwrap();
        assert!(b.try_chunk(10).is_none());
        let w = b.work(900);
        assert!(b.try_work(1).is_none());
        drop(w);
        drop(big);
        let small = b.try_chunk(100).unwrap();
        assert!(b.try_chunk(600).is_none());
        drop(small);
        assert_eq!(b.usage.lock().unwrap().fixed, 100);
        assert_eq!(b.usage.lock().unwrap().work, 0);
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

    /// The two Gibbs chains of a concurrent SPLIT (a chunk's coordinator and a
    /// thread it spawns) each work in a pool of their own, meet at every round
    /// (as at each selection sweep) and collect after it. In one shared pool the
    /// first chain's coordinator, waiting inside a join, could take the second
    /// chain's queued job, run it on its own stack and wait for it there at the
    /// rendezvous; and a blocking collection waited for a worker that waited for
    /// the other chain. Eight chunks, each with fresh pools.
    #[test]
    fn chain_pools_meet_and_collect_without_deadlock() {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let budget = MemoryBudget::new(1 << 30, 0);
            let mut seen = Vec::new();
            run_chunks(&[1; 8], &budget, 2, 1, |c| {
                assert!(chain_pool().is_some(), "a chunk's chain runs in its own pool");
                let meet = std::sync::Barrier::new(2);
                let chain = |id: usize| -> usize {
                    use rayon::prelude::*;
                    let mut sum = 0;
                    for round in 0..20 {
                        // A join the coordinator waits in while the other chain is queued
                        // (its first round outlasts spawning the other chain's thread).
                        sum += (0..64usize).into_par_iter().map(|i| {
                            if round == 0 { std::thread::sleep(std::time::Duration::from_micros(200)); }
                            i * round
                        }).sum::<usize>();
                        meet.wait();
                        sum += map_samples(1, 1, |_| in_chain_pool(chain_pool(), || {
                            (0..64usize).into_par_iter().map(|i| i + id).sum::<usize>()
                        }))[0];
                        collect_chain_pool();
                    }
                    sum
                };
                let (a, b) = std::thread::scope(|scope| {
                    let other = scope.spawn(|| in_chain_pool(new_chain_pool(1), || chain(1)));
                    (chain(0), other.join().unwrap())
                });
                (c, a + b)
            }, |c, (cc, v)| { assert_eq!(c, cc); seen.push(v); });
            tx.send(seen).unwrap();
        });
        let seen = rx.recv_timeout(std::time::Duration::from_secs(120)).expect("chains met and collected (no deadlock)");
        assert_eq!(seen.len(), 8);
    }

    #[test]
    fn completion_admits_replacement_before_slow_chunk_finishes() {
        let budget = MemoryBudget::new(100, 0);
        let (tx, rx) = std::sync::mpsc::channel();
        let rx = Mutex::new(rx);
        let mut order = Vec::new();
        run_chunks(&[25, 25, 25], &budget, 2, 2, |i| {
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
