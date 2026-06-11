//! Built-in self-test: exercises all output formats and code paths.
//!
//! Zero external dependencies — all validation is native (no bcftools required).
//! Usage: `selphi --self-test --refpanel X --input Y --map Z [--truth T] --out prefix`

use std::path::{Path, PathBuf};
use selphi::selphi_info;

pub struct SelfTestConfig<'a> {
    pub refpanel: &'a str,
    pub input: &'a str,
    pub map: &'a str,
    pub out_base: &'a str,
    pub truth: Option<&'a str>,
    pub threads: usize,
}

/// Run a single selphi subprocess test.
fn run_one(name: &str, cli_args: &[&str], pass: &mut u32, fail: &mut u32) {
    let exe = std::env::current_exe().expect("cannot find own executable");
    let output = std::process::Command::new(exe).args(cli_args).output();
    match output {
        Ok(o) if o.status.success() => {
            *pass += 1;
            selphi_info!("  {}  {}", selphi::log::green("PASS"), name);
        }
        Ok(o) => {
            *fail += 1;
            let stderr = String::from_utf8_lossy(&o.stderr);
            let last = stderr.lines().last().unwrap_or("");
            selphi_info!("  {}  {}  ({})", selphi::log::red("FAIL"), name, last);
        }
        Err(e) => {
            *fail += 1;
            selphi_info!("  {}  {}  ({})", selphi::log::red("FAIL"), name, e);
        }
    }
}

/// Native VCF validation: decompress and count header + data lines.
/// Returns (n_samples, n_variants) or error.
fn validate_vcf(path: &Path) -> Result<(usize, usize), String> {
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("open: {}", e))?;
    let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
    let mut raw = Vec::new();
    bgzf.read_to_end(&mut raw).map_err(|e| format!("decompress: {}", e))?;

    let mut n_samples = 0;
    let mut n_variants = 0;
    for line in raw.split(|&b| b == b'\n') {
        if line.is_empty() { continue; }
        if line.starts_with(b"#CHROM") {
            n_samples = line.split(|&b| b == b'\t').count().saturating_sub(9);
        } else if !line.starts_with(b"#") {
            n_variants += 1;
        }
    }
    if n_samples == 0 { return Err("no #CHROM header found".into()); }
    if n_variants == 0 { return Err("no data lines found".into()); }
    Ok((n_samples, n_variants))
}

/// Native BCF validation: read header and count records.
fn validate_bcf(path: &Path) -> Result<(usize, usize), String> {
    let header = selphi::srp::bcf_reader::read_header_only(path)
        .map_err(|e| format!("BCF header: {}", e))?;
    if header.n_samples == 0 { return Err("no samples".into()); }

    // Count records by reading raw BCF
    use std::io::Read;
    let file = std::fs::File::open(path).map_err(|e| format!("open: {}", e))?;
    let mut bgzf = noodles_bgzf::io::Reader::new(std::io::BufReader::new(file));
    let mut magic = [0u8; 5]; bgzf.read_exact(&mut magic).map_err(|e| format!("{}", e))?;
    let mut buf4 = [0u8; 4];
    bgzf.read_exact(&mut buf4).map_err(|e| format!("{}", e))?;
    let hl = u32::from_le_bytes(buf4) as usize;
    let mut hdr = vec![0u8; hl]; bgzf.read_exact(&mut hdr).map_err(|e| format!("{}", e))?;

    let mut n_variants = 0usize;
    loop {
        if bgzf.read_exact(&mut buf4).is_err() { break; }
        let ls = u32::from_le_bytes(buf4) as usize;
        if ls == 0 { break; }
        if bgzf.read_exact(&mut buf4).is_err() { break; }
        let li = u32::from_le_bytes(buf4) as usize;
        let mut skip = vec![0u8; ls + li];
        if bgzf.read_exact(&mut skip).is_err() { break; }
        n_variants += 1;
    }
    if n_variants == 0 { return Err("no records".into()); }
    Ok((header.n_samples, n_variants))
}

/// Native TBI/CSI index validation: decompress and check structure.
fn validate_index(path: &Path) -> Result<(String, usize), String> {
    let raw = std::fs::read(path).map_err(|e| format!("read: {}", e))?;
    // Decompress BGZF
    use std::io::Read;
    let mut bgzf = noodles_bgzf::io::Reader::new(&raw[..]);
    let mut data = Vec::new();
    bgzf.read_to_end(&mut data).map_err(|e| format!("decompress: {}", e))?;

    if data.len() < 8 { return Err("too small".into()); }
    let magic = &data[0..4];
    let (format, n_ref_offset) = match magic {
        b"TBI\x01" => ("TBI", 4),
        b"CSI\x01" => ("CSI", 4 + 4 + 4 + 4), // skip min_shift, depth, l_aux
        _ => return Err(format!("unknown magic {:?}", &magic[..4])),
    };

    // For CSI, skip l_aux bytes
    let mut off = n_ref_offset;
    if format == "CSI" {
        if data.len() < 16 { return Err("CSI too small".into()); }
        let l_aux = i32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
        off = 16 + l_aux;
    }
    if off + 4 > data.len() { return Err("truncated".into()); }
    let n_ref = i32::from_le_bytes(data[off..off+4].try_into().unwrap()) as usize;
    if n_ref == 0 { return Err("n_ref=0".into()); }

    Ok((format.to_string(), n_ref))
}

/// Check that a file exists and has non-zero size.
fn check_file(name: &str, path: &Path, _pass: &mut u32, fail: &mut u32) -> bool {
    if path.exists() {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if size > 0 {
            return true;
        }
    }
    *fail += 1;
    selphi_info!("  {}  {} (file missing or empty)", selphi::log::red("FAIL"), name);
    false
}

pub fn run(config: &SelfTestConfig) -> u32 {
    let SelfTestConfig { refpanel, input, map, out_base, truth, threads } = config;
    let t = format!("--threads={}", threads);

    let log_path = PathBuf::from(out_base).with_extension("log");
    selphi::log::init(&log_path, false);
    selphi::log::print_banner(env!("CARGO_PKG_VERSION"));
    selphi_info!("  mode:     self-test");
    selphi_info!("  refpanel: {}", refpanel);
    selphi_info!("  input:    {}", input);
    selphi_info!("  map:      {}", map);
    if let Some(t) = truth { selphi_info!("  truth:    {}", t); }
    selphi_info!("  log:      {}\n", log_path.display());

    let mut pass = 0u32;
    let mut fail = 0u32;

    macro_rules! test {
        ($name:expr, $($arg:expr),+ $(,)?) => {
            run_one($name, &[$($arg),+], &mut pass, &mut fail)
        };
    }

    // ── Pipeline tests ──────────────────────────────────────────────

    // 1. Phase-only. Force phasing so the test produces a VCF even when the
    // smoke-test input happens to already be phased.
    let out = format!("{}_phase", out_base);
    test!("phase-only (haploid)",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t,
        "--phase-only", "--force-phasing");

    // 2. VCF imputation
    let out = format!("{}_vcf", out_base);
    test!("impute → VCF",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t);

    // 3. BCF imputation
    let out = format!("{}_bcf", out_base);
    test!("impute → BCF",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t, "--bcf");

    // 4. Parquet
    let out = format!("{}_parquet", out_base);
    test!("impute → Parquet",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t, "--parquet");

    // 5. PGEN
    let out = format!("{}_pgen", out_base);
    test!("impute → PGEN",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t, "--pgen");

    // 6. SelfDecode
    let out = format!("{}_sd", out_base);
    test!("impute → SelfDecode",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t, "--selfdecode");

    // 7. Pre-phased input — reuse the phase-only output (produced above with
    // --force-phasing, so it exists regardless of input phase state).
    let phased_vcf = format!("{}_phase.vcf.gz", out_base);
    let out = format!("{}_prephased", out_base);
    test!("pre-phased input",
        "--refpanel", refpanel, "--input", &phased_vcf, "--map", map, "--out", &out, &t);

    // 8. Evaluate (requires --truth)
    if let Some(truth_path) = truth {
        let vcf_out = format!("{}_vcf.vcf.gz", out_base);
        let out = format!("{}_eval", out_base);
        test!("evaluate R²",
            "--evaluate", &vcf_out, "--truth", truth_path, "--out", &out, &t);
    }

    // ── Output validation (native, no bcftools) ─────────────────────

    // 9. VCF output: decompress and count samples + variants
    let vcf_path = PathBuf::from(format!("{}_vcf.vcf.gz", out_base));
    if check_file("VCF output exists", &vcf_path, &mut pass, &mut fail) {
        match validate_vcf(&vcf_path) {
            Ok((ns, nv)) => {
                pass += 1;
                selphi_info!("  {}  VCF valid ({} samples, {} variants)", selphi::log::green("PASS"), ns, nv);
            }
            Err(e) => {
                fail += 1;
                selphi_info!("  {}  VCF invalid ({})", selphi::log::red("FAIL"), e);
            }
        }
    }

    // 10. BCF output: read header + count records
    let bcf_path = PathBuf::from(format!("{}_bcf.bcf", out_base));
    if check_file("BCF output exists", &bcf_path, &mut pass, &mut fail) {
        match validate_bcf(&bcf_path) {
            Ok((ns, nv)) => {
                pass += 1;
                selphi_info!("  {}  BCF valid ({} samples, {} variants)", selphi::log::green("PASS"), ns, nv);
            }
            Err(e) => {
                fail += 1;
                selphi_info!("  {}  BCF invalid ({})", selphi::log::red("FAIL"), e);
            }
        }
    }

    // ── Index validation (native) ───────────────────────────────────

    // 11. TBI index: rebuild and verify structure
    if vcf_path.exists() {
        let tbi_path = PathBuf::from(format!("{}.tbi", vcf_path.display()));
        let _ = std::fs::remove_file(&tbi_path);
        match selphi::io::indexing::index_file(&vcf_path) {
            Ok(()) if tbi_path.exists() => {
                match validate_index(&tbi_path) {
                    Ok((fmt, n_ref)) => {
                        pass += 1;
                        selphi_info!("  {}  TBI index valid ({}, {} contig{})", selphi::log::green("PASS"), fmt, n_ref,
                            if n_ref != 1 { "s" } else { "" });
                    }
                    Err(e) => {
                        fail += 1;
                        selphi_info!("  {}  TBI index corrupt ({})", selphi::log::red("FAIL"), e);
                    }
                }
            }
            _ => {
                fail += 1;
                selphi_info!("  {}  TBI index build failed", selphi::log::red("FAIL"));
            }
        }
    }

    // 12. CSI index: rebuild and verify structure
    if bcf_path.exists() {
        let csi_path = PathBuf::from(format!("{}.csi", bcf_path.display()));
        let _ = std::fs::remove_file(&csi_path);
        match selphi::io::indexing::index_file(&bcf_path) {
            Ok(()) if csi_path.exists() => {
                match validate_index(&csi_path) {
                    Ok((fmt, n_ref)) => {
                        pass += 1;
                        selphi_info!("  {}  CSI index valid ({}, {} contig{})", selphi::log::green("PASS"), fmt, n_ref,
                            if n_ref != 1 { "s" } else { "" });
                    }
                    Err(e) => {
                        fail += 1;
                        selphi_info!("  {}  CSI index corrupt ({})", selphi::log::red("FAIL"), e);
                    }
                }
            }
            _ => {
                fail += 1;
                selphi_info!("  {}  CSI index build failed", selphi::log::red("FAIL"));
            }
        }
    }

    // ── Parquet / PGEN / SelfDecode file existence ──────────────────

    let pq_path = PathBuf::from(format!("{}_parquet.parquet", out_base));
    if check_file("Parquet output exists", &pq_path, &mut pass, &mut fail) {
        pass += 1;
        let sz = std::fs::metadata(&pq_path).map(|m| m.len()).unwrap_or(0);
        selphi_info!("  {}  Parquet file ({:.1} MB)", selphi::log::green("PASS"), sz as f64 / 1e6);
    }

    let pgen_path = PathBuf::from(format!("{}_pgen.pgen", out_base));
    let pvar_path = PathBuf::from(format!("{}_pgen.pvar", out_base));
    let psam_path = PathBuf::from(format!("{}_pgen.psam", out_base));
    if check_file("PGEN output exists", &pgen_path, &mut pass, &mut fail) {
        if pvar_path.exists() && psam_path.exists() {
            pass += 1;
            selphi_info!("  {}  PGEN triplet (.pgen + .pvar + .psam)", selphi::log::green("PASS"));
        } else {
            fail += 1;
            selphi_info!("  {}  PGEN missing .pvar or .psam", selphi::log::red("FAIL"));
        }
    }

    let sd_path = PathBuf::from(format!("{}_sd.selfdecode.zip", out_base));
    if check_file("SelfDecode output exists", &sd_path, &mut pass, &mut fail) {
        pass += 1;
        let sz = std::fs::metadata(&sd_path).map(|m| m.len()).unwrap_or(0);
        selphi_info!("  {}  SelfDecode ZIP ({:.1} MB)", selphi::log::green("PASS"), sz as f64 / 1e6);
    }

    // ── Summary ─────────────────────────────────────────────────────
    selphi_info!("");
    if fail == 0 {
        selphi_info!("All {} tests passed.", pass);
    } else {
        selphi_info!("{} passed, {} FAILED.", pass, fail);
    }

    fail
}
