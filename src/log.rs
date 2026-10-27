//! Global logger: writes to both stderr and a `.log` file.
//!
//! Two levels: info (always shown) and debug (only with `--debug` or `SELPHI_DEBUG=1`).
//! Use the macros `selphi_info!`, `selphi_debug!`, `selphi_step!`, `selphi_error!`.

use std::io::{Write, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Color: ANSI is emitted ONLY when stderr is a real terminal, so AWS/CloudWatch,
// files, and pipes get clean plain text (and the `.log` file is always plain —
// write_log strips ANSI before writing it). NO_COLOR (the cross-tool standard)
// disables it. No progress bars anywhere: every log line is static, append-only.
// ---------------------------------------------------------------------------
static USE_COLOR: LazyLock<bool> = LazyLock::new(|| {
    std::io::stderr().is_terminal() && !crate::config::present("NO_COLOR")
});

#[inline]
fn sgr(code: &str, s: &str) -> String {
    if *USE_COLOR { format!("\x1b[{code}m{s}\x1b[0m") } else { s.to_string() }
}
#[inline] pub fn dim(s: &str) -> String { sgr("2", s) }
#[inline] pub fn bold(s: &str) -> String { sgr("1", s) }
#[inline] pub fn red(s: &str) -> String { sgr("1;31", s) }
#[inline] pub fn green(s: &str) -> String { sgr("32", s) }
#[inline] pub fn yellow(s: &str) -> String { sgr("33", s) }
#[inline] pub fn cyan(s: &str) -> String { sgr("1;36", s) }

/// Human-readable byte size: `1.2 GB` / `142.0 MB` / `9.8 KB` / `512 B`.
pub fn fmt_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1e9 { format!("{:.1} GB", f / 1e9) }
    else if f >= 1e6 { format!("{:.1} MB", f / 1e6) }
    else if f >= 1e3 { format!("{:.1} KB", f / 1e3) }
    else { format!("{} B", b) }
}

/// Integer with thousands separators: `1070401` -> `1,070,401`.
pub fn fmt_thousands(n: u64) -> String {
    let s = n.to_string();
    let len = s.len();
    let mut out = String::with_capacity(len + len / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) { out.push(','); }
        out.push(c);
    }
    out
}

/// Strip ANSI SGR escape sequences (for the plain-text `.log` file).
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&n) = chars.peek() { chars.next(); if n.is_ascii_alphabetic() { break; } }
            }
        } else {
            out.push(c);
        }
    }
    out
}

struct LoggerInner {
    file: Option<std::fs::File>,
    start: Instant,
    debug_dir: PathBuf,
}

static LOGGER: LazyLock<Mutex<LoggerInner>> = LazyLock::new(|| {
    Mutex::new(LoggerInner {
        file: None,
        start: Instant::now(),
        debug_dir: PathBuf::from("debug"),
    })
});

static DEBUG_FLAG: AtomicBool = AtomicBool::new(false);

/// Initialize the global logger. Call once from main() after parsing args.
/// `debug_dir` is derived from `log_path` parent: `{out_dir}/debug/`.
pub fn init(log_path: &Path, debug: bool) {
    let file = std::fs::File::create(log_path).ok();

    // Debug dir: sibling of log file, e.g. results/debug/
    let debug_dir = log_path.parent()
        .unwrap_or(Path::new("."))
        .join("debug");

    let mut inner = LOGGER.lock().unwrap();
    inner.file = file;
    inner.start = Instant::now();
    inner.debug_dir = debug_dir;

    // Set debug flag from CLI or env var
    let env_debug = crate::config::is_one("SELPHI_DEBUG");
    DEBUG_FLAG.store(debug || env_debug, Ordering::Relaxed);
}

/// Initialize logger for stderr-only output (no log file).
/// Used by modes that don't produce output files (e.g. --index-stats).
pub fn init_stderr_only() {
    let mut inner = LOGGER.lock().unwrap();
    inner.file = None;
    inner.start = Instant::now();
}

/// Check if debug mode is enabled (lock-free).
#[inline]
pub fn is_debug() -> bool {
    DEBUG_FLAG.load(Ordering::Relaxed)
}

/// Get the debug output directory. Creates it if needed (only call when is_debug()).
pub fn debug_dir() -> PathBuf {
    let dir = LOGGER.lock().unwrap().debug_dir.clone();
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Seconds elapsed since logger initialization.
pub fn elapsed_secs() -> f64 {
    LOGGER.lock().unwrap().start.elapsed().as_secs_f64()
}

/// getrusage(RUSAGE_SELF) — portable (Linux + macOS) CPU-time and peak-RSS
/// source, used as the non-Linux fallback for the /proc-based readers below.
fn getrusage_self() -> Option<libc::rusage> {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 { Some(ru) } else { None }
    }
}

/// Total CPU time (user+system, all threads) in seconds.
/// Linux: /proc/self/stat (utime+stime in clock ticks); otherwise getrusage.
pub fn cpu_time_secs() -> f64 {
    if let Ok(stat) = std::fs::read_to_string("/proc/self/stat") {
        // Fields are space-separated. Field 14 = utime, 15 = stime (1-indexed).
        // But field 2 (comm) can contain spaces in parens, so find closing paren first.
        if let Some(pos) = stat.rfind(')') {
            let rest = &stat[pos + 2..]; // skip ") "
            let fields: Vec<&str> = rest.split_whitespace().collect();
            // Fields after comm: state(0), ppid(1), ..., utime(11), stime(12)
            if fields.len() > 12 {
                let utime: u64 = fields[11].parse().unwrap_or(0);
                let stime: u64 = fields[12].parse().unwrap_or(0);
                let ticks_per_sec = 100.0; // sysconf(_SC_CLK_TCK), almost always 100 on Linux
                return (utime + stime) as f64 / ticks_per_sec;
            }
        }
    }
    // Non-Linux (macOS): user+system time via getrusage.
    if let Some(ru) = getrusage_self() {
        let u = ru.ru_utime.tv_sec as f64 + ru.ru_utime.tv_usec as f64 / 1e6;
        let s = ru.ru_stime.tv_sec as f64 + ru.ru_stime.tv_usec as f64 / 1e6;
        return u + s;
    }
    0.0
}

/// Peak resident set size in MB (Linux /proc/self/status).
/// Current resident set (VmRSS, MB) — as opposed to `peak_mem_mb`'s high-water
/// mark. For placing an allocation in time: the peak tells you how much, this
/// tells you when.
pub fn rss_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmRSS:") {
                let kb: f64 = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
    }
    0.0
}

pub fn peak_mem_mb() -> f64 {
    // Linux: /proc/self/status VmHWM
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if line.starts_with("VmHWM:") {
                let kb: f64 = line.split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
    }
    // macOS: getrusage ru_maxrss is the true peak (VmHWM analogue), not the
    // current RSS `ps` reports. On Darwin ru_maxrss is in BYTES (Linux: KB).
    #[cfg(target_os = "macos")]
    if let Some(ru) = getrusage_self() {
        return ru.ru_maxrss as f64 / (1024.0 * 1024.0);
    }
    0.0
}

/// RAM the kernel says a new allocation can actually take right now, in MB —
/// `MemAvailable`, not `MemTotal`. The difference matters on a shared box: this
/// one routinely has several imputation runs on it, and a decision made against
/// total RAM would happily double-book memory another job is already using.
/// `MemAvailable` already discounts this process's own RSS (it is resident) and
/// counts reclaimable page cache as available, which is the right answer for
/// "can I grow by X without swapping".
///
/// Returns `None` when the kernel does not publish it (pre-3.14 Linux, macOS),
/// so callers can fall back rather than silently treat 0 as "no room".
pub fn available_ram_mb() -> Option<f64> {
    let info = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in info.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// Room left under this process's cgroup memory limits, in MB: the least, over
/// its cgroup and every ancestor that sets a limit, of the limit minus that
/// group's usage (`memory.max` - `memory.current` on cgroup v2,
/// `memory.limit_in_bytes` - `memory.usage_in_bytes` on v1), not counting the
/// group's inactive file cache, which the kernel reclaims before it fails an
/// allocation. A container's limit can sit far below the host's `MemAvailable`.
/// The cache matters: a prod imputation task downloads its panel (7.5 GB for
/// chr2) into a 6 GiB container, so the usage alone reads as a full container.
/// `None` when no group on the path sets a limit (or the files cannot be read).
pub fn cgroup_headroom_mb() -> Option<f64> {
    let read = |p: &str| std::fs::read_to_string(p).ok().and_then(|s| s.trim().parse::<f64>().ok());
    let groups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    // (mount dir, limit file, usage file, inactive-cache key in memory.stat, group path)
    // per hierarchy that holds memory
    let mut probes: Vec<(String, &str, &str, &str, String)> = Vec::new();
    for line in groups.lines() {
        let mut f = line.splitn(3, ':');
        let (_, ctl, path) = (f.next()?, f.next()?, f.next()?.trim().to_string());
        if ctl.is_empty() {
            probes.push(("/sys/fs/cgroup".into(), "memory.max", "memory.current", "inactive_file", path));
        } else if ctl.split(',').any(|c| c == "memory") {
            probes.push(("/sys/fs/cgroup/memory".into(), "memory.limit_in_bytes", "memory.usage_in_bytes",
                         "total_inactive_file", path));
        }
    }
    let room = |dir: &str, lim: &str, cur: &str, key: &str| -> Option<f64> {
        let (l, u) = (read(&format!("{dir}/{lim}"))?, read(&format!("{dir}/{cur}"))?);
        // v1 reports "no limit" as a huge page-aligned number
        if l >= (1u64 << 60) as f64 { return None; }
        let cache = std::fs::read_to_string(format!("{dir}/memory.stat")).ok()
            .map_or(0.0, |s| stat_value(&s, key)).min(u);
        Some(((l - (u - cache)) / 1048576.0).max(0.0))
    };
    let mut best: Option<f64> = None;
    for (mount, lim, cur, key, path) in probes {
        // the group and each ancestor, up to the mount root
        let mut p = path.trim_end_matches('/').to_string();
        loop {
            if let Some(r) = room(&format!("{mount}{p}"), lim, cur, key) {
                best = Some(best.map_or(r, |b: f64| b.min(r)));
            }
            match p.rfind('/') { Some(k) => p.truncate(k), None => break }
            if p.is_empty() {
                // the mount root itself
                if let Some(r) = room(&mount, lim, cur, key) {
                    best = Some(best.map_or(r, |b: f64| b.min(r)));
                }
                break;
            }
        }
    }
    best
}

/// The value of `key` in a cgroup `memory.stat` file (`key value` lines), 0 when absent.
fn stat_value(stat: &str, key: &str) -> f64 {
    stat.lines()
        .find_map(|l| l.strip_prefix(key).and_then(|v| v.strip_prefix(' ')))
        .and_then(|v| v.trim().parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// Total system RAM in MB. Works on Linux (/proc/meminfo) and macOS (sysctl).
pub fn system_ram_mb() -> f64 {
    // Linux
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        for line in info.lines() {
            if line.starts_with("MemTotal:") {
                let kb: f64 = line.split_whitespace()
                    .nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                return kb / 1024.0;
            }
        }
    }
    // macOS
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl").arg("-n").arg("hw.memsize").output();
        if let Ok(o) = out {
            if let Ok(s) = String::from_utf8(o.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0);
                }
            }
        }
    }
    0.0
}

/// Estimate peak memory for imputation/phasing and warn if insufficient.
/// Call after loading SRP metadata + target intersection.
///
/// ADVISORY ONLY: warns (does not abort, does not reduce threads) if the
/// estimate exceeds system RAM — the estimator deliberately over-estimates
/// (see compute_estimate_mb), so aborting would falsely block valid runs.
/// Returns n_threads unchanged. If the active rayon pool
/// has more threads, the caller should wrap heavy work in
/// `rayon::ThreadPoolBuilder::new().num_threads(N).build().unwrap().install(...)`.
pub fn estimate_and_warn(
    n_chip: usize, n_ref: usize, n_ref_variants: usize, n_samples: usize, n_threads: usize,
    needs_phasing: bool,
) -> usize {
    estimate_and_warn_with_mc(n_chip, n_ref, n_ref_variants, n_samples, n_threads, 2500, 0, needs_phasing)
}

/// Same as [`estimate_and_warn`] but caller passes the actual resolved
/// `max_candidates` and `--sample-batch-size` (in haps; 0 disables batching).
/// Both shape the estimate materially:
/// - `max_candidates` scales the per-thread PBWT reduced array linearly.
/// - `target_batch_size_haps`, when > 0, caps the in-flight weights /
///   posterior buffers from `n_haps × per_buffer` to
///   `target_batch_size_haps × per_buffer` — the entire point of batched
///   output.
pub fn estimate_and_warn_with_mc(
    n_chip: usize, n_ref: usize, n_ref_variants: usize, n_samples: usize, n_threads_init: usize,
    max_candidates_in: usize, target_batch_size_haps: usize,
    needs_phasing: bool,
) -> usize {
    // Mem cap: never let estimate exceed this fraction of system RAM at startup.
    let sys_ram_mb = system_ram_mb();

    // Compute estimate for a given thread count.
    let est_for_threads = |n_threads: usize| -> f64 {
        compute_estimate_mb(n_chip, n_ref, n_ref_variants, n_samples, n_threads,
            max_candidates_in, target_batch_size_haps, needs_phasing)
    };

    // NOTE: the static estimator overcounts I/O buffer transients (vcf_mb,
    // stripe_decomp_mb, interp_mb) that the streaming output pipeline never
    // actually holds simultaneously. Empirically on MESA 5K chr20 the peak is
    // ~120 GB at mc=2500 and ~121 GB at mc=20K while the estimator returns
    // ~158-162 GB. We therefore WARN at >100% of sys_ram but do NOT abort —
    // the real safety net is an OS-level memory cap (ulimit -v or cgroup)
    // around the selphi invocation. Aborting on the inflated estimate would
    // refuse to run on configurations that empirically fit fine.
    let n_threads = n_threads_init;
    if sys_ram_mb > 0.0 && est_for_threads(n_threads) > sys_ram_mb {
        crate::selphi_info!(
            "  ⚠ Conservative estimate {:.1} GB > system RAM {:.1} GB at --threads {}",
            est_for_threads(n_threads) / 1024.0, sys_ram_mb / 1024.0, n_threads,
        );
        crate::selphi_info!(
            "  ⚠ Estimator is pessimistic on I/O buffers. Real peak is typically 60–80% of this.",
        );
        crate::selphi_info!(
            "  ⚠ For hard OOM protection, wrap in `ulimit -v {:.0}000000 && selphi ...`",
            sys_ram_mb * 0.9 / 1024.0,
        );
    }
    let total_mb = est_for_threads(n_threads);
    let total_gb = total_mb / 1024.0;
    let sys_gb = sys_ram_mb / 1024.0;
    let _ = (n_chip, n_ref, n_samples, needs_phasing); // silence unused on log-only path
    let batch_note = if target_batch_size_haps > 0 {
        format!(", sample-batch-size={}", target_batch_size_haps / 2)
    } else { String::new() };
    crate::selphi_info!("  Resources: {:.1} GB estimated, {:.1} GB system RAM, {} threads (mc={}{})",
        total_gb, sys_gb, n_threads, max_candidates_in, batch_note);

    n_threads
}

fn compute_estimate_mb(
    n_chip: usize, n_ref: usize, n_ref_variants: usize, n_samples: usize, n_threads: usize,
    max_candidates: usize, target_batch_size_haps: usize, needs_phasing: bool,
) -> f64 {
    let n_haps = n_samples * 2;
    let n_ref_words = n_ref.div_ceil(64);
    let ref_bm_mb = (n_chip * n_ref_words * 8) as f64 / 1e6;
    let targ_mb = (n_chip * n_haps) as f64 / 1e6;
    let preload_mb = 500.0;
    // The loaded panel's variant table, resident for the whole run: the compact
    // index (srp::index: positions, one REF/ALT arena with offsets, each chromosome
    // name once) plus the variant-ID and original-ID columns (an arena + offsets
    // each). RSS right after "Loaded SRP: 17,900,635 variants" on TOPMed chr20 is
    // 1,396 MiB = ~82 B/variant (v3 chr20 450 MiB / 4.63M = ~102 B, 1KG chr22
    // 180 MiB / 1.07M = ~176 B, the difference being fixed reader overhead). 82 is
    // used. The per-variant String layout it replaced was 4,455 MiB on that load
    // (this term charged 241 B then).
    let srp_table_mb = (n_ref_variants as f64 * 82.0) / 1e6;

    // Per-thread (= per in-flight target) working set of the PBWT + HMM stage.
    //
    // Full dense storage costs `window_chip_vars * max_candidates * 12`:
    // an n_window_sites x n_candidates BYTE allele matrix for the PBWT
    // (n_candidates saturates at max_candidates), plus two dense
    // n_sites x n_states f32 HMM matrices — 1.75 GB/thread predicted,
    // 1.78 measured on MESA 100 x TOPMed chr20. To avoid these matrices, the PBWT
    // gathers one allele row per site (pbwt::AlleleRows), the forward is
    // sqrt(n)-checkpointed and the weights stream to CSR. What is left per
    // thread is one allele row (m bytes), the PBWT workspace (five m-wide i32
    // arrays + an n_ref-wide i64), the forward checkpoints (~2 sqrt(n_sites) x
    // n_states f32) and the per-site match lists (~30 haps/site, four copies at
    // worst). Measured after the change: the same rig peaks at 15.4 GB against
    // 43.3 before, i.e. the thread-scaled term collapsed from ~1.78 GB to noise.
    let window_chip_vars = (n_chip / 14).clamp(100, 5000);
    let m_cand = max_candidates.max(2500).min(n_ref) + n_haps;
    let states_est_thread = (max_candidates.max(1) as f64).min(n_ref as f64).min(3000.0);
    let hmm_per_thread_mb = (
        m_cand as f64                                            // allele row (u8)
        + 5.0 * m_cand as f64 * 4.0 + n_ref as f64 * 8.0        // PbwtWorkspace
        + 2.0 * (window_chip_vars as f64).sqrt() * states_est_thread * 4.0  // fwd checkpoints
        + window_chip_vars as f64 * 30.0 * 8.0 * 4.0             // match lists
    ) / 1e6;
    let hmm_mb = n_threads as f64 * hmm_per_thread_mb;

    let n_chip_window = n_chip.min(15000);
    // Kept CSR entries per chip row. Was 100; SELPHI_PRUNE_DIAG measures 14-29 on
    // MESA 100/400 x TOPMed chr20 (mean_csr_nnz_per_row), and the 1/(n_states+1)
    // threshold makes anything near 100 implausible. 32 is the honest upper end.
    let csr_nnz_per_row = 32.0;
    let per_weights_mb = (n_chip_window as f64 * 4.0 + n_chip_window as f64 * csr_nnz_per_row * 8.0) / 1e6;
    // With --sample-batch-size, weights and per-hap posterior are bounded to
    // one batch at a time (next batch starts only after the current one's
    // CSRs are streamed to its per-batch writer and dropped).
    let in_flight_haps = if target_batch_size_haps > 0 {
        target_batch_size_haps.min(n_haps)
    } else {
        n_haps
    };
    let weights_mb = (in_flight_haps as f64) * per_weights_mb;
    // Cross-window prior, now SPARSE: (i64 hap id, f64 weight) pairs, at most
    // n_states of them, rather than a dense f64 per reference haplotype. n_states
    // is bounded by the candidate set, not by the panel, so on a large panel this
    // is one to two orders of magnitude smaller than the dense form it replaced
    // (MESA 5K chr20, 171,054 haps x 10,000 target haps: 13.7 GB dense, ~0.4 GB
    // sparse). Estimated with the measured mean state count, capped by the panel
    // since the sparse form can never exceed one entry per haplotype.
    let states_est = (max_candidates.max(1) as f64).min(n_ref as f64);
    let hap_posterior_mb = (in_flight_haps as f64 * states_est * 16.0) / 1e6;
    let tile_cols = n_ref.div_ceil(4096);
    // Stripes per interpolation batch. Mirrors io/pipeline.rs exactly: a 2 GiB
    // cap over (decompressed stripes + result tiles), applied per TILE so no
    // single chip gap can exceed it (the chr20 centromere against TOPMed spans
    // 993 stripes, exceeding a 96-stripe cap when batched as one interval).
    let decomp_tile_bytes = 500usize * 1024;
    let bytes_per_stripe = tile_cols * decomp_tile_bytes;
    let result_bytes_per_stripe = in_flight_haps * 1024 * 4;
    let cap_stripes = ((2usize << 30) / (bytes_per_stripe + result_bytes_per_stripe).max(1)).max(4);
    let stripes_per_batch = cap_stripes.min((n_chip * 100).div_ceil(1024)).max(1);
    let stripe_decomp_mb = (stripes_per_batch * bytes_per_stripe) as f64 / 1e6;
    // Interpolation and VCF/BCF encode buffers scale with the in-flight hap
    // count (or sample count) and are also bounded by --sample-batch-size.
    let in_flight_samples = in_flight_haps / 2;
    // The interpolation batch is memory-CAPPED in io/pipeline.rs (mem_cap = 2 GiB
    // over decompressed stripes + result tiles, stripes per batch adapt to n_haps),
    // so this term cannot exceed that cap however large the cohort. The uncapped
    // form (haps x 1024 x 4 x 300 stripes) over-counted by ~10 GB at 5,000 samples.
    let interp_mb = ((in_flight_haps * 1024 * 4 * stripes_per_batch) as f64 / 1e6).min(2048.0);
    let vcf_mb = ((in_flight_samples as f64 * 12.0 * stripes_per_batch as f64 * 1024.0 * 2.0) / 1e6).min(2_000.0);
    let bgzf_mb = (n_threads as f64 * 64.0 * 1024.0) / 1e6;
    let thread_local_mb = n_threads as f64 * 20.0;
    let overhead_mb = 300.0;
    let mut total_mb = srp_table_mb + ref_bm_mb + targ_mb + preload_mb + hmm_mb + weights_mb
        + hap_posterior_mb + stripe_decomp_mb + interp_mb + vcf_mb + bgzf_mb
        + thread_local_mb + overhead_mb;
    if needs_phasing {
        // Phasing-only extras. Sized for one phasing window (the phasing
        // engine operates on similar window scales as the imputation HMM —
        // reuse the same `window_chip_vars` derivation).
        let n_total_haps = n_ref + n_haps;
        let hap_bits_mb = (n_total_haps as f64 * window_chip_vars as f64 / 8.0) / 1e6;
        let n_steps = (window_chip_vars / 10).max(1);
        let ibs_mb = (n_steps * n_samples * 100 * 4) as f64 / 1e6;
        let sample_arrays_mb = (window_chip_vars * n_samples * 12) as f64 / 1e6;
        let coded_mb = (n_steps * n_total_haps * 4) as f64 / 1e6;
        let global_mb = (n_chip * n_haps + n_chip * n_samples * 4) as f64 / 1e6;
        total_mb += hap_bits_mb + ibs_mb + sample_arrays_mb + coded_mb + global_mb;
    }
    total_mb
}


/// Print the startup banner.
pub fn print_banner(version: &str) {
    let art = cyan(r#"
  ___ ___ _    ___ _  _ ___
 / __| __| |  | _ \ || |_ _|
 \__ \ _|| |__|  _/ __ || |
 |___/___|____|_| |_||_|___|"#);
    let tag = format!("      {} {} {}",
        dim(&format!("v{version}")), '\u{1F980}', green("SelfDecode\u{2122}"));
    write_log(&format!("{art}\n{tag}\n"));
}

/// Write a message to both stderr and the log file.
pub fn write_log(msg: &str) {
    eprintln!("{}", msg);
    if let Ok(mut inner) = LOGGER.lock() && let Some(ref mut f) = inner.file {
        // The log file is always plain text — strip any color the terminal got.
        let plain = if msg.contains('\u{1b}') { strip_ansi(msg) } else { msg.to_string() };
        let _ = writeln!(f, "{}", plain);
        let _ = f.flush();
    }
}

/// Write a timestamped step message with memory.
pub fn write_step(msg: &str) {
    let elapsed = elapsed_secs();
    let mem = peak_mem_mb();
    // Dim the timing/memory suffix so the message reads first.
    let timing = dim(&format!("[{:.1}s | {:.0} MB]", elapsed, mem));
    let line = format!("  {:<60} {}", msg, timing);
    write_log(&line);
}

/// Info: always shown.
#[macro_export]
macro_rules! selphi_info {
    ($($arg:tt)*) => {
        $crate::log::write_log(&format!($($arg)*))
    };
}

/// Debug: only shown when --debug or SELPHI_DEBUG=1.
#[macro_export]
macro_rules! selphi_debug {
    ($($arg:tt)*) => {
        if $crate::log::is_debug() {
            $crate::log::write_log(&format!($($arg)*))
        }
    };
}

/// Timestamped progress step: always shown.
#[macro_export]
macro_rules! selphi_step {
    ($($arg:tt)*) => {
        $crate::log::write_step(&format!($($arg)*))
    };
}

/// Error: always shown, prefixed.
#[macro_export]
macro_rules! selphi_error {
    ($($arg:tt)*) => {
        $crate::log::write_log(&$crate::log::red(&format!("ERROR: {}", format!($($arg)*))))
    };
}

#[cfg(test)]
mod tests {
    use super::stat_value;

    #[test]
    fn stat_value_reads_the_exact_key() {
        // v2 lists active_file before inactive_file, v1 total_inactive_file after inactive_file
        let v2 = "anon 1048576\nfile 7340032\nactive_file 2097152\ninactive_file 5242880\n";
        assert_eq!(stat_value(v2, "inactive_file"), 5242880.0);
        let v1 = "cache 9\ninactive_file 4\ntotal_inactive_file 6\n";
        assert_eq!(stat_value(v1, "total_inactive_file"), 6.0);
        assert_eq!(stat_value(v1, "inactive_file"), 4.0);
        assert_eq!(stat_value("anon 3\n", "inactive_file"), 0.0);
    }
}
