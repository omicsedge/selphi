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
