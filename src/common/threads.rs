//! The threads an lcWGS run starts beside the global Rayon pool, counted while
//! they live against the room the process's thread limits leave.
//!
//! One-sample runs give each Gibbs chain a pool of its own (`lcwgs::memory::run_chunks`),
//! beside the threads the run starts anyway: chunk and SPLIT coordinators, sample
//! threads (`lcwgs::memory::map_samples`), selection decoders, the panel-ahead
//! builder, the panel's stripe readers (`srp::tiled`), and the BGZF readers and
//! writers (their threads held while they live, [`ThreadBudget::hold`]). Where
//! `RLIMIT_NPROC` (counted over every thread of the user) or the cgroup's
//! `pids.max` leaves little room, the pools give way: one is granted only while
//! the threads alive, its own and a reserve for the other threads fit that room,
//! else its chain runs as it did before the pools; readers and writers take fewer
//! threads or none. A thread the OS still refuses is replaced by the asking thread
//! doing the work itself (decoders, the panel-ahead builder and stripe read-ahead
//! are skipped), and no pool is granted after that. Every thread is counted until
//! it has finished its work, so a dropped pool's workers hold their share until
//! they exit, and the chunk stage ends when they have ([`ThreadBudget::wait_pools_exited`]).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

/// Compute workers the live chain pools hold together, per thread of the run.
const CHAIN_WORKER_BUDGET: usize = 8;
/// Compute workers a chain gets at least (its two concurrent passes and a
/// block task), budget or not.
const CHAIN_MIN_WORKERS: usize = 3;
/// Room kept for native threads when a pool is granted, per thread of the run:
/// as much as the pools may take together.
const NATIVE_RESERVE: usize = CHAIN_WORKER_BUDGET;

/// The threads one run starts through [`global`] (tests make their own).
pub(crate) struct ThreadBudget {
    /// Threads started here that have not finished.
    live: AtomicUsize,
    /// Compute workers of the chain pools, while they live.
    pool_workers: AtomicUsize,
    /// Threads the process could still start when the first one was asked for.
    room: OnceLock<usize>,
    /// The OS refused a thread: no pool from now on.
    refused: AtomicBool,
    /// Pool grants one at a time.
    grant: Mutex<()>,
    /// A pool was refused (said once in the log).
    noted: AtomicBool,
    /// Threads of the pools built here that have not exited.
    pool_threads: Mutex<usize>,
    pools_exited: Condvar,
    /// The pool threads' handles and kernel thread ids (set when they start),
    /// joined by [`Self::wait_pools_exited`].
    pool_handles: Mutex<Vec<(std::thread::JoinHandle<()>, std::sync::Arc<AtomicUsize>)>>,
}

static GLOBAL: ThreadBudget = ThreadBudget::new();

#[cfg(test)]
thread_local! {
    /// Tests: this thread's native spawns are refused as by the OS, and no room is left for holds.
    static REFUSE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: run `f` with this thread's native spawns refused and its holds denied (budgets untouched).
#[cfg(test)]
pub(crate) fn refusing<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) { REFUSE.with(|r| r.set(self.0)); }
    }
    let _restore = Restore(REFUSE.with(|r| r.replace(true)));
    f()
}

/// The run's budget.
pub(crate) fn global() -> &'static ThreadBudget { &GLOBAL }

/// Counts a thread off when its work ends (also on unwinding).
struct Finished<'a> { budget: &'a ThreadBudget, compute: bool, pool: bool }

impl Drop for Finished<'_> {
    fn drop(&mut self) {
        self.budget.live.fetch_sub(1, Ordering::Relaxed);
        if self.compute { self.budget.pool_workers.fetch_sub(1, Ordering::Relaxed); }
        if self.pool { self.budget.pools_lose(1); }
    }
}

/// Threads a library starts (a BGZF reader's or writer's), counted while it lives
/// ([`ThreadBudget::hold`]).
pub(crate) struct Held { budget: &'static ThreadBudget, n: usize }

impl Drop for Held {
    fn drop(&mut self) { self.budget.live.fetch_sub(self.n, Ordering::Relaxed); }
}

impl ThreadBudget {
    pub(crate) const fn new() -> Self {
        ThreadBudget { live: AtomicUsize::new(0), pool_workers: AtomicUsize::new(0), room: OnceLock::new(),
            refused: AtomicBool::new(false), grant: Mutex::new(()), noted: AtomicBool::new(false),
            pool_threads: Mutex::new(0), pools_exited: Condvar::new(), pool_handles: Mutex::new(Vec::new()) }
    }

    /// A budget whose room is `room` threads (tests).
    #[cfg(test)]
    pub(crate) fn with_room(room: usize) -> Self {
        let b = Self::new();
        b.room.set(room).unwrap();
        b
    }

    fn room(&self) -> usize { *self.room.get_or_init(measure_room) }

    /// Threads started here that have not finished.
    #[cfg(test)]
    pub(crate) fn live(&self) -> usize { self.live.load(Ordering::Relaxed) }

    /// `n` threads a library is about to start, counted while the guard lives,
    /// when they fit the room; `None` when they do not (the caller then starts
    /// fewer or none).
    pub(crate) fn hold(&'static self, n: usize) -> Option<Held> {
        #[cfg(test)]
        if REFUSE.with(std::cell::Cell::get) { return None; }
        let room = self.room();
        let mut live = self.live.load(Ordering::Relaxed);
        loop {
            if live.saturating_add(n) > room { return None; }
            match self.live.compare_exchange_weak(live, live + n, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(Held { budget: self, n }),
                Err(now) => live = now,
            }
        }
    }

    /// A library's worker count up to `max` (it starts one thread more) that
    /// fits the room, and its hold: `max`, else one, else one unheld (as before
    /// the budget; the caller may avoid the threads altogether).
    pub(crate) fn hold_workers(&'static self, max: usize) -> (usize, Option<Held>) {
        let max = max.max(1);
        if let Some(h) = self.hold(max + 1) { return (max, Some(h)); }
        (1, self.hold(2))
    }

    fn pools_lose(&self, n: usize) {
        let mut left = self.pool_threads.lock().unwrap_or_else(|e| e.into_inner());
        *left -= n;
        if *left == 0 { self.pools_exited.notify_all(); }
    }

    /// Wait until the threads of every pool built here have exited (a dropped
    /// pool's workers leave on their own time), at most `limit`: the chunk stage
    /// ends with the threads it started gone. The threads that have finished
    /// are joined, and then waited for until the process no longer counts them
    /// (the kernel releases a thread's slot after the join can return).
    pub(crate) fn wait_pools_exited(&self, limit: std::time::Duration) {
        let t0 = std::time::Instant::now();
        let left = self.pool_threads.lock().unwrap_or_else(|e| e.into_inner());
        drop(self.pools_exited.wait_timeout_while(left, limit, |n| *n > 0));
        let finished: Vec<_> = {
            let mut handles = self.pool_handles.lock().unwrap_or_else(|e| e.into_inner());
            let (done, running) = std::mem::take(&mut *handles).into_iter().partition(|(h, _)| h.is_finished());
            *handles = running;
            done
        };
        for (h, tid) in finished {
            let _ = h.join();
            // Gone from /proc once the kernel has released it (its slot freed before that).
            let task = format!("/proc/self/task/{}", tid.load(Ordering::Relaxed));
            while tid.load(Ordering::Relaxed) != 0 && std::path::Path::new(&task).exists() && t0.elapsed() < limit {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    }

    /// Start a native thread in `scope`, counted until `f` returns. An error
    /// (the OS refused it) is for the caller to absorb.
    pub(crate) fn spawn_scoped<'scope, 'env, T: Send + 'scope>(
        &'scope self, scope: &'scope std::thread::Scope<'scope, 'env>, name: &str,
        f: impl FnOnce() -> T + Send + 'scope,
    ) -> std::io::Result<std::thread::ScopedJoinHandle<'scope, T>> {
        #[cfg(test)]
        if REFUSE.with(std::cell::Cell::get) { return Err(std::io::ErrorKind::WouldBlock.into()); }
        self.room();
        self.live.fetch_add(1, Ordering::Relaxed);
        let spawned = std::thread::Builder::new().name(name.into()).spawn_scoped(scope, move || {
            let _done = Finished { budget: self, compute: false, pool: false };
            f()
        });
        if spawned.is_err() {
            self.live.fetch_sub(1, Ordering::Relaxed);
            self.refused.store(true, Ordering::Relaxed);
        }
        spawned
    }

    /// The compute workers of a new chain pool for a run of `threads` threads:
    /// the run's `threads` while the live pools hold fewer than
    /// [`CHAIN_WORKER_BUDGET`] per thread (the first chains, which are the
    /// largest chunks), then what is left of that, at least [`CHAIN_MIN_WORKERS`]
    /// (or `threads`, when fewer); `None` when that and the pool's coordinator
    /// do not fit the room beside the native threads' reserve.
    fn chain_workers(&self, threads: usize) -> Option<usize> {
        let share = chain_share(threads, self.pool_workers.load(Ordering::Relaxed));
        let free = self.room().saturating_sub(self.live.load(Ordering::Relaxed))
            .saturating_sub(NATIVE_RESERVE.saturating_mul(threads));
        let workers = share.min(free.saturating_sub(1));
        (workers >= CHAIN_MIN_WORKERS.min(threads)).then_some(workers)
    }

    /// A Gibbs chain's pool: `builder` with its compute workers
    /// ([`Self::chain_workers`]) and one more for the chain's coordinator.
    /// `None` when the room does not allow it or the OS refused a thread.
    pub(crate) fn chain_pool(&'static self, threads: usize, builder: rayon::ThreadPoolBuilder) -> Option<(rayon::ThreadPool, usize)> {
        let threads = threads.max(1);
        let _grant = self.grant.lock().unwrap_or_else(|e| e.into_inner());
        let workers = (!self.refused.load(Ordering::Relaxed)).then(|| self.chain_workers(threads)).flatten();
        let pool = workers.and_then(|w| self.build(builder, w + 1, w).map(|pool| (pool, w)));
        if pool.is_none() && !self.noted.swap(true, Ordering::Relaxed) {
            crate::selphi_info!("  lcWGS: the process's thread limits leave room for {} more threads ({} running here): Gibbs chains run without pools of their own",
                self.room().saturating_sub(self.live.load(Ordering::Relaxed)), self.live.load(Ordering::Relaxed));
        }
        pool
    }

    /// A pool of `n` threads (no compute share), when they fit the room beside
    /// the native threads' reserve for a run of `threads` threads.
    pub(crate) fn helper_pool(&'static self, threads: usize, n: usize, builder: rayon::ThreadPoolBuilder) -> Option<rayon::ThreadPool> {
        let _grant = self.grant.lock().unwrap_or_else(|e| e.into_inner());
        if self.refused.load(Ordering::Relaxed) { return None; }
        let free = self.room().saturating_sub(self.live.load(Ordering::Relaxed))
            .saturating_sub(NATIVE_RESERVE.saturating_mul(threads.max(1)));
        if n > free { return None; }
        self.build(builder, n, 0)
    }

    /// Build `builder` with `n` threads, of which the first `compute` count as
    /// pool compute workers; each is counted until it exits (a dropped pool's
    /// workers may still finish queued jobs). The threads of a build the OS cut
    /// short are counted off as they exit, the ones never started at once.
    fn build(&'static self, builder: rayon::ThreadPoolBuilder, n: usize, compute: usize) -> Option<rayon::ThreadPool> {
        self.live.fetch_add(n, Ordering::Relaxed);
        self.pool_workers.fetch_add(compute, Ordering::Relaxed);
        *self.pool_threads.lock().unwrap_or_else(|e| e.into_inner()) += n;
        let mut started = 0usize;
        let built = builder.num_threads(n).spawn_handler(|t| {
            let mut b = std::thread::Builder::new();
            if let Some(name) = t.name() { b = b.name(name.to_owned()); }
            if let Some(size) = t.stack_size() { b = b.stack_size(size); }
            let done = Finished { budget: self, compute: t.index() < compute, pool: true };
            let tid = std::sync::Arc::new(AtomicUsize::new(0));
            let own = tid.clone();
            let h = b.spawn(move || {
                let _done = done;
                #[cfg(target_os = "linux")]
                own.store(unsafe { libc::gettid() } as usize, Ordering::Relaxed);
                #[cfg(not(target_os = "linux"))]
                let _ = own;
                t.run()
            })?;
            self.pool_handles.lock().unwrap_or_else(|e| e.into_inner()).push((h, tid));
            started += 1;
            Ok(())
        }).build();
        match built {
            Ok(pool) => Some(pool),
            Err(_) => {
                // A refused spawn dropped its `Finished` (counted off); the rest never started.
                let never = n.saturating_sub(started + 1);
                self.live.fetch_sub(never, Ordering::Relaxed);
                self.pool_workers.fetch_sub(compute.saturating_sub((started + 1).min(compute)), Ordering::Relaxed);
                if never > 0 { self.pools_lose(never); }
                self.refused.store(true, Ordering::Relaxed);
                None
            }
        }
    }
}

/// [`ThreadBudget::chain_workers`] before the room: `threads` while the budget
/// allows, then what is left of it, and at least [`CHAIN_MIN_WORKERS`] (or
/// `threads`, when fewer).
fn chain_share(threads: usize, live: usize) -> usize {
    threads.min((CHAIN_WORKER_BUDGET * threads).saturating_sub(live)).max(CHAIN_MIN_WORKERS.min(threads))
}

/// Threads the process can still start: what `RLIMIT_NPROC` leaves over every
/// thread of this real user and what the cgroups' `pids.max` leave; unbounded
/// without either.
#[cfg(target_os = "linux")]
fn measure_room() -> usize {
    let mut room = usize::MAX;
    let uid = unsafe { libc::getuid() };
    let mut rl = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    // Root's threads are not held to RLIMIT_NPROC.
    if uid != 0 && unsafe { libc::getrlimit(libc::RLIMIT_NPROC, &mut rl) } == 0 && rl.rlim_cur != libc::RLIM_INFINITY {
        let limit = usize::try_from(rl.rlim_cur).unwrap_or(usize::MAX);
        room = room.min(limit.saturating_sub(user_threads(uid)));
    }
    if let Some(free) = cgroup_pids_room() { room = room.min(free); }
    room
}

#[cfg(not(target_os = "linux"))]
fn measure_room() -> usize { usize::MAX }

/// Threads of the processes whose real user is `uid` (`/proc`).
#[cfg(target_os = "linux")]
fn user_threads(uid: u32) -> usize {
    let Ok(dir) = std::fs::read_dir("/proc") else { return 0 };
    dir.flatten().filter(|e| e.file_name().to_str().is_some_and(|s| s.bytes().all(|b| b.is_ascii_digit())))
        .filter_map(|e| std::fs::read_to_string(e.path().join("status")).ok())
        .filter_map(|s| status_uid_threads(&s))
        .filter(|&(u, _)| u == uid)
        .map(|(_, n)| n).sum()
}

/// The real uid and thread count of a `/proc/<pid>/status`.
fn status_uid_threads(status: &str) -> Option<(u32, usize)> {
    let field = |key: &str| status.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.split_whitespace().next());
    Some((field("Uid:")?.parse().ok()?, field("Threads:")?.parse().ok()?))
}

/// The tasks this process's cgroups (v2, or v1's pids hierarchy) and their
/// parents still allow: the least `pids.max - pids.current` among them.
#[cfg(target_os = "linux")]
fn cgroup_pids_room() -> Option<usize> {
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let (root, path) = groups.lines().find_map(|l| {
        let mut f = l.splitn(3, ':');
        let (_, ctl, path) = (f.next()?, f.next()?, f.next()?);
        if ctl.is_empty() { Some(("/sys/fs/cgroup".to_owned(), path.to_owned())) }
        else if ctl.split(',').any(|c| c == "pids") { Some(("/sys/fs/cgroup/pids".to_owned(), path.to_owned())) }
        else { None }
    })?;
    let mut dir = std::path::PathBuf::from(format!("{root}{path}"));
    let mut room: Option<usize> = None;
    loop {
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
        if let (Some(max), Some(cur)) = (read("pids.max").as_deref().and_then(pids_max), read("pids.current")) {
            let free = max.saturating_sub(cur.trim().parse().unwrap_or(0));
            room = Some(room.map_or(free, |r| r.min(free)));
        }
        if dir.as_os_str().len() <= root.len() || !dir.pop() { break; }
    }
    room
}

/// A `pids.max` value: `None` for `max` (no limit).
fn pids_max(s: &str) -> Option<usize> { s.trim().parse().ok() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_and_pids_parsing() {
        let s = "Name:\tselphi\nUid:\t1000\t1000\t1000\t1000\nGid:\t1000\t1000\t1000\t1000\nThreads:\t17\n";
        assert_eq!(status_uid_threads(s), Some((1000, 17)));
        assert_eq!(status_uid_threads("Name:\tx\n"), None);
        assert_eq!(pids_max("max\n"), None);
        assert_eq!(pids_max("96\n"), Some(96));
    }

    /// The first chains get the run's thread count, later ones what is left of
    /// eight per thread, and every chain at least three compute workers (or
    /// the run's thread count, when fewer).
    #[test]
    fn chain_pools_share_a_bounded_worker_budget() {
        assert_eq!(chain_share(16, 0), 16);
        assert_eq!(chain_share(16, 112), 16);
        assert_eq!(chain_share(16, 118), 10);
        assert_eq!(chain_share(16, 128), 3);
        assert_eq!(chain_share(16, 500), 3);
        assert_eq!(chain_share(2, 16), 2);
        assert_eq!(chain_share(1, 0), 1);
        assert_eq!(chain_share(1, 7), 1);
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::new()));
        let (pool, workers) = budget.chain_pool(2, rayon::ThreadPoolBuilder::new()).expect("a pool");
        assert_eq!(workers, 2);
        assert_eq!(pool.current_num_threads(), 3, "two compute workers beside the coordinator");
    }

    /// A pool is granted only within the room left beside the native threads'
    /// reserve (shrunk to fit, down to the minimum share); natives are counted
    /// but never refused by the budget.
    #[test]
    fn chain_pools_fit_the_room_beside_the_native_reserve() {
        let threads = 4;
        let reserve = NATIVE_RESERVE * threads;
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::with_room(reserve + 3)));
        assert!(budget.chain_pool(threads, rayon::ThreadPoolBuilder::new()).is_none(), "3 compute + 1 coordinator do not fit 3");
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::with_room(reserve + 4)));
        let (pool, workers) = budget.chain_pool(threads, rayon::ThreadPoolBuilder::new()).expect("3 + 1 fit 4");
        assert_eq!((workers, pool.current_num_threads()), (3, 4));
        assert_eq!(budget.live(), 4);
        assert!(budget.chain_pool(threads, rayon::ThreadPoolBuilder::new()).is_none(), "the room is taken");
        // Natives are counted whatever the room.
        std::thread::scope(|s| {
            let h = budget.spawn_scoped(s, "native", || budget.live()).unwrap();
            assert_eq!(h.join().unwrap(), 5);
        });
        assert_eq!(budget.live(), 4);
        assert!(budget.helper_pool(threads, 1, rayon::ThreadPoolBuilder::new()).is_none());
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::with_room(usize::MAX)));
        let (_, workers) = budget.chain_pool(threads, rayon::ThreadPoolBuilder::new()).unwrap();
        assert_eq!(workers, threads, "no limit: the share alone");
    }

    /// A dropped pool's workers keep their share until they exit: a worker
    /// still running a queued job after the drop is counted until it ends.
    #[test]
    fn a_dropped_pools_threads_count_until_they_exit() {
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::new()));
        let (pool, workers) = budget.chain_pool(2, rayon::ThreadPoolBuilder::new()).unwrap();
        assert_eq!((budget.live(), budget.pool_workers.load(Ordering::Relaxed)), (workers + 1, workers));
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();
        pool.spawn(move || { started_tx.send(()).unwrap(); go_rx.recv().unwrap(); });
        started_rx.recv().unwrap();
        drop(pool);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(budget.live() >= 1, "the busy worker still counts after the drop");
        go_tx.send(()).unwrap();
        budget.wait_pools_exited(std::time::Duration::from_secs(10));
        assert_eq!((budget.live(), budget.pool_workers.load(Ordering::Relaxed)), (0, 0), "workers counted off as they exit");
        assert!(budget.pool_handles.lock().unwrap().is_empty(), "every exited worker joined");
    }

    /// A library's threads are held while they fit the room: all of them, else
    /// one worker's, else none.
    #[test]
    fn library_threads_are_held_within_the_room() {
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::with_room(7)));
        let (w, a) = budget.hold_workers(4);
        assert_eq!((w, a.is_some(), budget.live()), (4, true, 5));
        let (w, b) = budget.hold_workers(4);
        assert_eq!((w, b.is_some(), budget.live()), (1, true, 7));
        let (w, c) = budget.hold_workers(4);
        assert_eq!((w, c.is_none(), budget.live()), (1, true, 7));
        drop((a, b));
        assert_eq!(budget.live(), 0);
    }

    /// A pool the OS cuts short (a thread refused) is no pool: the threads it
    /// started are counted off as they exit, and no pool is granted after it.
    #[test]
    fn a_refused_pool_thread_counts_off_and_stops_pools() {
        let budget: &'static ThreadBudget = Box::leak(Box::new(ThreadBudget::new()));
        // A stack no thread can have: the first spawn fails.
        let builder = rayon::ThreadPoolBuilder::new().stack_size(usize::MAX - 4096);
        assert!(budget.chain_pool(4, builder).is_none());
        assert_eq!((budget.live(), budget.pool_workers.load(Ordering::Relaxed)), (0, 0));
        assert!(budget.chain_pool(4, rayon::ThreadPoolBuilder::new()).is_none(), "no pool after a refusal");
        std::thread::scope(|s| assert_eq!(budget.spawn_scoped(s, "native", || 7).unwrap().join().unwrap(), 7));
    }
}
