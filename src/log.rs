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
    0.0
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
    if let Ok(mut inner) = LOGGER.lock() {
        if let Some(ref mut f) = inner.file {
            let _ = writeln!(f, "{}", msg);
            let _ = f.flush();
        }
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
