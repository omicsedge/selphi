//! Multi-chromosome imputation: single input VCF → per-chr impute → merged output.
//!
//! Automatically discovers reference panels and genetic maps per chromosome,
//! splits the input by contig, imputes each independently, and concatenates
//! the results into a single output file.

use std::path::{Path, PathBuf};
use std::io;
use selphi::selphi_info;

/// Configuration for multi-chr imputation.
pub struct MultiChrConfig<'a> {
    pub input: &'a str,
    pub refpanel_dir: &'a str,
    pub map_dir: &'a str,
    pub out: &'a str,
    pub threads: usize,
    pub extra_args: Vec<String>,
}

/// Discover which chromosomes are available in both the input VCF and the ref panel dir.
fn discover_chromosomes(input: &str, refpanel_dir: &str, map_dir: &str) -> io::Result<Vec<ChrInfo>> {
    // Get contigs from input VCF
    let output = std::process::Command::new("bcftools")
        .args(["query", "-f", "%CHROM\\n", input])
        .output()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("bcftools not found: {}", e)))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut input_chrs: Vec<String> = stdout.lines()
        .map(|s| s.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    // Sort chromosomes naturally: 1,2,...,22,X,Y,MT
    input_chrs.sort_by(|a, b| chr_sort_key(a).cmp(&chr_sort_key(b)));

    // Match with available ref panels and maps
    let mut result = Vec::new();
    for chr in &input_chrs {
        let srp = find_file(refpanel_dir, chr, &["srp"]);
        let map = find_file(map_dir, chr, &["map"]);
        if let (Some(srp_path), Some(map_path)) = (srp, map) {
            result.push(ChrInfo {
                name: chr.clone(),
                refpanel: srp_path,
                map: map_path,
            });
        }
    }
    Ok(result)
}

struct ChrInfo {
    name: String,
    refpanel: PathBuf,
    map: PathBuf,
}

/// Find a file matching chr pattern in a directory.
fn find_file(dir: &str, chr: &str, extensions: &[&str]) -> Option<PathBuf> {
    let dir_path = Path::new(dir);
    if !dir_path.is_dir() { return None; }

    // Try common naming patterns (prefer _v2 over plain)
    for ext in extensions {
        for pattern in &[
            format!("chr{}_v2.{}", chr, ext),
            format!("chr{}.{}", chr, ext),
            format!("{}.{}", chr, ext),
        ] {
            let path = dir_path.join(pattern);
            if path.exists() { return Some(path); }
        }
    }

    // Fallback: glob for *chr{N}*
    if let Ok(entries) = std::fs::read_dir(dir_path) {
        let needle = format!("chr{}", chr);
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.contains(&needle) {
                for ext in extensions {
                    if name.ends_with(ext) {
                        return Some(entry.path());
                    }
                }
            }
        }
    }
    None
}

fn chr_sort_key(chr: &str) -> (u8, u32) {
    let s = chr.strip_prefix("chr").unwrap_or(chr);
    match s {
        "X" => (1, 23), "Y" => (1, 24), "MT" | "M" => (1, 25),
        _ => (0, s.parse::<u32>().unwrap_or(99)),
    }
}

/// Run multi-chromosome imputation.
pub fn run(config: &MultiChrConfig) -> io::Result<()> {
    let start = std::time::Instant::now();

    selphi::log::init_stderr_only();
    selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
    selphi_info!("  mode:     multi-chr imputation\n");
    selphi_info!("  input:    {}", config.input);
    selphi_info!("  panels:   {}", config.refpanel_dir);
    selphi_info!("  maps:     {}", config.map_dir);
    selphi_info!("  output:   {}", config.out);
    selphi_info!("  threads:  {}\n", config.threads);

    // Discover chromosomes
    let chrs = discover_chromosomes(config.input, config.refpanel_dir, config.map_dir)?;
    if chrs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound,
            "No chromosomes found with matching ref panels and maps"));
    }
    selphi_info!("  Found {} chromosomes: {}",
        chrs.len(),
        chrs.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", "));
    selphi_info!("");

    let exe = std::env::current_exe()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("Cannot find executable: {}", e)))?;

    // Process each chromosome
    let mut per_chr_outputs: Vec<PathBuf> = Vec::new();
    let t_fmt = format!("--threads={}", config.threads);

    for (i, chr) in chrs.iter().enumerate() {
        let chr_out = format!("{}_{}", config.out, chr.name);

        selphi_info!("  [{}/{}] chr{}: {} → {}",
            i + 1, chrs.len(), chr.name,
            chr.refpanel.display(), chr_out);

        // Extract this chr from input
        let chr_input = format!("{}_input_{}.vcf.gz", config.out, chr.name);
        let extract = std::process::Command::new("bcftools")
            .args(["view", "-r", &chr.name, config.input, "-Oz", "-o", &chr_input, "--threads", "4"])
            .output()?;
        if !extract.status.success() {
            selphi_info!("    \x1b[31mFAIL\x1b[0m  bcftools extract failed");
            continue;
        }
        // Index
        let _ = std::process::Command::new("bcftools")
            .args(["index", &chr_input, "--threads", "4"])
            .output();

        // Run selphi
        let mut args = vec![
            "--refpanel", chr.refpanel.to_str().unwrap_or(""),
            "--input", &chr_input,
            "--map", chr.map.to_str().unwrap_or(""),
            "--out", &chr_out,
            &t_fmt,
        ];
        // Add extra args
        let extra_refs: Vec<&str> = config.extra_args.iter().map(|s| s.as_str()).collect();
        args.extend_from_slice(&extra_refs);

        let result = std::process::Command::new(&exe)
            .args(&args)
            .output()?;

        let vcf_out = PathBuf::from(format!("{}.vcf.gz", chr_out));
        if result.status.success() && vcf_out.exists() {
            selphi_info!("    \x1b[32mOK\x1b[0m");
            per_chr_outputs.push(vcf_out);
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            let last = stderr.lines().last().unwrap_or("unknown error");
            selphi_info!("    \x1b[31mFAIL\x1b[0m  {}", last);
        }

        // Cleanup temp input
        let _ = std::fs::remove_file(&chr_input);
        let _ = std::fs::remove_file(format!("{}.tbi", chr_input));
        let _ = std::fs::remove_file(format!("{}.csi", chr_input));
    }

    // Concatenate all per-chr outputs
    if per_chr_outputs.is_empty() {
        return Err(io::Error::new(io::ErrorKind::Other, "No chromosomes were successfully imputed"));
    }

    let final_out = format!("{}.vcf.gz", config.out);
    selphi_info!("\n  Concatenating {} chromosomes → {}", per_chr_outputs.len(), final_out);

    let mut concat_args = vec!["concat".to_string()];
    for p in &per_chr_outputs {
        concat_args.push(p.to_string_lossy().to_string());
    }
    concat_args.extend_from_slice(&["-Oz".to_string(), "-o".to_string(), final_out.clone(),
        "--threads".to_string(), config.threads.to_string()]);

    let concat = std::process::Command::new("bcftools")
        .args(&concat_args)
        .output()?;

    if !concat.status.success() {
        return Err(io::Error::new(io::ErrorKind::Other, "bcftools concat failed"));
    }

    // Index final output
    let _ = std::process::Command::new("bcftools")
        .args(["index", &final_out, "--threads", "4"])
        .output();

    // Cleanup per-chr files
    for p in &per_chr_outputs {
        let _ = std::fs::remove_file(p);
        let _ = std::fs::remove_file(format!("{}.tbi", p.display()));
        let _ = std::fs::remove_file(p.with_extension("log"));
    }

    let elapsed = start.elapsed().as_secs_f64();
    selphi_info!("\n  Total: {:.0}s | {} chromosomes | {}", elapsed, chrs.len(), final_out);
    selphi_info!("");

    Ok(())
}
