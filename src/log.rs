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
pub fn estimate_and_warn(
    n_chip: usize, n_ref: usize, n_samples: usize, n_threads: usize,
    needs_phasing: bool,
) {
    let n_haps = n_samples * 2;
    let n_ref_words = n_ref.div_ceil(64);

    // ref_bm: bitmatrix (1 bit per allele)
    let ref_bm_mb = (n_chip * n_ref_words * 8) as f64 / 1e6;

    // targ_alleles
    let targ_mb = (n_chip * n_haps) as f64 / 1e6;

    // SRP preload (capped at ~500 MB compressed)
    let preload_mb = 500.0;

    // Per-thread PBWT + HMM buffers:
    // - reduced array: max_candidates × n_chip_per_window × 1 byte
    // - forward result: n_chip × fl_fwd × 8 bytes (haps + lens)
    // - HMM forward matrix: n_states × n_chip × 4 bytes
    // - thread-local reuse buffers
    let max_candidates = 2500usize;
    let fl_fwd = 200usize;
    let n_var_window = n_chip.min(15000); // typical max window size
    let pbwt_per_thread_mb = (max_candidates * n_var_window   // reduced array
        + n_var_window * fl_fwd * 8                            // FwdResult (haps + lens)
        + max_candidates * n_var_window * 4                    // HMM forward matrix
        ) as f64 / 1e6;
    let hmm_mb = n_threads as f64 * pbwt_per_thread_mb;

    // all_weights per window: each target holds a sparse CSR over the chip
    // window with ~100-200 non-zero entries per chip row. At biobank scale
    // this dominates peak memory.
    //   indptr: (n_chip_window + 1) × 4 bytes
    //   indices + data: nnz × 8 bytes, nnz ≈ n_chip_window × ~100
    let n_chip_window = n_chip.min(15000);
    let per_weights_mb = (n_chip_window as f64 * 4.0 + n_chip_window as f64 * 100.0 * 8.0) / 1e6;
    let weights_mb = (n_haps as f64) * per_weights_mb;

    // hap_posterior: n_haps × n_ref × 8 bytes, held after each non-final window.
    // Skipped on the final window by the HMM (compute_posterior=false), but
    // consumes peak for all but the last.
    let hap_posterior_mb = (n_haps as f64 * n_ref as f64 * 8.0) / 1e6;

    // Interpolation: batch of decompressed stripes + alt_probs results.
    // Stripe tiles: ~500 KB per stripe × n_tile_cols. Capped at 2 GB.
    // alt_probs per batch: n_haps × TILE_ROWS × 4 bytes × stripes_per_batch.
    // With mem cap, batch holds ~300 stripes max.
    let tile_cols = n_ref.div_ceil(4096);
    let stripes_per_batch = 300usize.min((n_chip * 100).div_ceil(1024)); // rough estimate
    let stripe_decomp_mb = (stripes_per_batch * tile_cols * 500 * 1024) as f64 / 1e6;
    let interp_mb = (n_haps * 1024 * 4 * stripes_per_batch) as f64 / 1e6;

    // VCF/BCF output: BGZF compressor queues + format strings.
    // With many samples, VCF lines are ~(12 bytes × n_samples) per variant.
    // Multiple tiles in flight simultaneously: ~2× stripes_per_batch.
    let vcf_mb = (n_samples as f64 * 12.0 * stripes_per_batch as f64 * 1024.0 * 2.0) / 1e6;

    // BGZF writer internal buffers: ~64 KB per compressor thread × n_threads
    let bgzf_mb = (n_threads as f64 * 64.0 * 1024.0) / 1e6;

    // Thread-local buffers (PBWT workspace, HMM reuse, tile decode) persist across calls
    let thread_local_mb = n_threads as f64 * 20.0;

    let overhead_mb = 300.0; // OS, allocator fragmentation, misc

    let mut total_mb = ref_bm_mb + targ_mb + preload_mb + hmm_mb + weights_mb
        + hap_posterior_mb
        + stripe_decomp_mb + interp_mb + vcf_mb + bgzf_mb + thread_local_mb + overhead_mb;

    // Phasing adds significant memory
    if needs_phasing {
        let n_total_haps = n_ref + n_haps;
        // hap_bits: all haplotypes packed in bits, per window
        let hap_bits_mb = (n_total_haps as f64 * n_var_window as f64 / 8.0) / 1e6;
        // IBS result arrays: n_coded_steps × n_samples × n_candidates × 4 bytes
        let n_steps = n_var_window / 10; // ~1 step per 10 variants
        let ibs_mb = (n_steps * n_samples * 100 * 4) as f64 / 1e6;
        // Per-sample arrays: confidence, resolved, locked, ref_alleles
        let sample_arrays_mb = (n_var_window * n_samples * 12) as f64 / 1e6;
        // Coded steps precomputation: n_steps × n_total_haps × 4 bytes
        let coded_mb = (n_steps * n_total_haps * 4) as f64 / 1e6;
        // global_phased + global_confidence
        let global_mb = (n_chip * n_haps + n_chip * n_samples * 4) as f64 / 1e6;
        total_mb += hap_bits_mb + ibs_mb + sample_arrays_mb + coded_mb + global_mb;
    }

    let sys_ram = system_ram_mb();
    let total_gb = total_mb / 1024.0;
    let sys_gb = sys_ram / 1024.0;

    crate::selphi_info!("  Resources: {:.1} GB estimated, {:.1} GB system RAM, {} threads",
        total_gb, sys_gb, n_threads);
    crate::selphi_info!("    ref_bm={:.0} MB  target={:.0} MB  preload={:.0} MB  hmm={:.0} MB  weights={:.0} MB  posterior={:.0} MB  interp={:.0} MB  vcf={:.0} MB",
        ref_bm_mb, targ_mb, preload_mb, hmm_mb, weights_mb, hap_posterior_mb, interp_mb + stripe_decomp_mb, vcf_mb);

    if sys_ram > 0.0 && total_mb > sys_ram * 0.9 {
        crate::selphi_info!("  ⚠ WARNING: estimated memory ({:.1} GB) exceeds 90% of system RAM ({:.1} GB)",
            total_gb, sys_gb);
        crate::selphi_info!("    The system may run out of memory. Consider:");
        crate::selphi_info!("    - Using fewer threads (--threads {})", (n_threads / 2).max(1));
        crate::selphi_info!("    - Running on a machine with more RAM");
        if n_ref > 100_000 {
            crate::selphi_info!("    - Large panel ({} haplotypes): ref bitmatrix alone = {:.1} GB",
                n_ref, ref_bm_mb / 1024.0);
        }
    }
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
