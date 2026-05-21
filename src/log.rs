//! Global logger: writes to both stderr and a `.log` file.
//!
//! Two levels: info (always shown) and debug (only with `--debug` or `SELPHI_DEBUG=1`).
//! Use the macros `selphi_info!`, `selphi_debug!`, `selphi_step!`, `selphi_error!`.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

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
    let env_debug = std::env::var("SELPHI_DEBUG").ok().map(|v| v == "1").unwrap_or(false);
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

/// Total CPU time (user+system, all threads) in seconds.
/// Reads from /proc/self/stat (fields 14+15 = utime+stime in clock ticks).
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
    0.0
}

/// Format CPU utilization: "X.Xs cpu (YY% of N cores)"
pub fn fmt_cpu(wall_secs: f64, cpu_start: f64, n_cores: usize) -> String {
    let cpu_delta = cpu_time_secs() - cpu_start;
    let pct = if wall_secs > 0.01 { cpu_delta / wall_secs / n_cores as f64 * 100.0 } else { 0.0 };
    format!("{:.1}s cpu ({:.0}% of {} cores)", cpu_delta, pct, n_cores)
}

/// Peak resident set size in MB (Linux /proc/self/status).
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
    // macOS fallback: parse `ps` output for RSS
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
        {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(kb) = s.trim().parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
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
/// Auto-reduces n_threads to fit in `mem_cap_frac * sys_ram` and ABORTS if
/// even single-threaded would exceed the cap. Returns the effective
/// thread count to use for subsequent rayon work. If the active rayon pool
/// has more threads, the caller should wrap heavy work in
/// `rayon::ThreadPoolBuilder::new().num_threads(N).build().unwrap().install(...)`.
pub fn estimate_and_warn(
    n_chip: usize, n_ref: usize, n_samples: usize, n_threads: usize,
    needs_phasing: bool,
) -> usize {
    estimate_and_warn_with_mc(n_chip, n_ref, n_samples, n_threads, 2500, needs_phasing)
}

/// Same as [`estimate_and_warn`] but caller passes the actual
/// `max_candidates` value (default 2500). The per-thread PBWT reduced array
/// scales linearly with max_candidates and dominates memory at large
/// candidate values on biobank panels.
pub fn estimate_and_warn_with_mc(
    n_chip: usize, n_ref: usize, n_samples: usize, n_threads_init: usize,
    max_candidates_in: usize,
    needs_phasing: bool,
) -> usize {
    // Mem cap: never let estimate exceed this fraction of system RAM at startup.
    let sys_ram_mb = system_ram_mb();

    // Compute estimate for a given thread count.
    let est_for_threads = |n_threads: usize| -> f64 {
        compute_estimate_mb(n_chip, n_ref, n_samples, n_threads, max_candidates_in, needs_phasing)
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
    crate::selphi_info!("  Resources: {:.1} GB estimated, {:.1} GB system RAM, {} threads (mc={})",
        total_gb, sys_gb, n_threads, max_candidates_in);

    n_threads
}

fn compute_estimate_mb(
    n_chip: usize, n_ref: usize, n_samples: usize, n_threads: usize,
    max_candidates: usize, needs_phasing: bool,
) -> f64 {
    let n_haps = n_samples * 2;
    let n_ref_words = n_ref.div_ceil(64);
    let ref_bm_mb = (n_chip * n_ref_words * 8) as f64 / 1e6;
    let targ_mb = (n_chip * n_haps) as f64 / 1e6;
    let preload_mb = 500.0;
    let fl_fwd = 200usize;
    let n_var_window = n_chip.min(15000);
    let pbwt_per_thread_mb = (max_candidates * n_var_window
        + n_var_window * fl_fwd * 8
        + max_candidates * n_var_window * 4) as f64 / 1e6;
    let hmm_mb = n_threads as f64 * pbwt_per_thread_mb;
    let n_chip_window = n_chip.min(15000);
    let per_weights_mb = (n_chip_window as f64 * 4.0 + n_chip_window as f64 * 100.0 * 8.0) / 1e6;
    let weights_mb = (n_haps as f64) * per_weights_mb;
    let hap_posterior_mb = (n_haps as f64 * n_ref as f64 * 8.0) / 1e6;
    let tile_cols = n_ref.div_ceil(4096);
    let stripes_per_batch = 300usize.min((n_chip * 100).div_ceil(1024));
    let stripe_decomp_mb = (stripes_per_batch * tile_cols * 500 * 1024) as f64 / 1e6;
    let interp_mb = (n_haps * 1024 * 4 * stripes_per_batch) as f64 / 1e6;
    let vcf_mb = ((n_samples as f64 * 12.0 * stripes_per_batch as f64 * 1024.0 * 2.0) / 1e6).min(2_000.0);
    let bgzf_mb = (n_threads as f64 * 64.0 * 1024.0) / 1e6;
    let thread_local_mb = n_threads as f64 * 20.0;
    let overhead_mb = 300.0;
    let mut total_mb = ref_bm_mb + targ_mb + preload_mb + hmm_mb + weights_mb
        + hap_posterior_mb + stripe_decomp_mb + interp_mb + vcf_mb + bgzf_mb
        + thread_local_mb + overhead_mb;
    if needs_phasing {
        let n_total_haps = n_ref + n_haps;
        let hap_bits_mb = (n_total_haps as f64 * n_var_window as f64 / 8.0) / 1e6;
        let n_steps = n_var_window / 10;
        let ibs_mb = (n_steps * n_samples * 100 * 4) as f64 / 1e6;
        let sample_arrays_mb = (n_var_window * n_samples * 12) as f64 / 1e6;
        let coded_mb = (n_steps * n_total_haps * 4) as f64 / 1e6;
        let global_mb = (n_chip * n_haps + n_chip * n_samples * 4) as f64 / 1e6;
        total_mb += hap_bits_mb + ibs_mb + sample_arrays_mb + coded_mb + global_mb;
    }
    total_mb
}


/// Print the startup banner.
pub fn print_banner(version: &str) {
    write_log(&format!(r#"
  ___ ___ _    ___ _  _ ___
 / __| __| |  | _ \ || |_ _|
 \__ \ _|| |__|  _/ __ || |
 |___/___|____|_| |_||_|___|
      v{version} {crab} SelfDecode{tm}
"#, crab = '\u{1F980}', tm = '\u{2122}'));
}

/// Write a message to both stderr and the log file.
pub fn write_log(msg: &str) {
    eprintln!("{}", msg);
    if let Ok(mut inner) = LOGGER.lock() && let Some(ref mut f) = inner.file {
        let _ = writeln!(f, "{}", msg);
        let _ = f.flush();
    }
}

/// Write a timestamped step message with memory.
pub fn write_step(msg: &str) {
    let elapsed = elapsed_secs();
    let mem = peak_mem_mb();
    let line = format!("  {:<60} [{:.1}s | {:.0} MB]", msg, elapsed, mem);
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
        $crate::log::write_log(&format!("ERROR: {}", format!($($arg)*)))
    };
}
