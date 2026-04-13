//! Built-in self-test: exercises all output formats and code paths.
//!
//! Usage: `selphi --self-test --refpanel X --input Y --map Z [--truth T] --out prefix`

use std::path::{Path, PathBuf};
use selphi::selphi_info;

/// Configuration for the self-test, extracted from CLI args.
pub struct SelfTestConfig<'a> {
    pub refpanel: &'a str,
    pub input: &'a str,
    pub map: &'a str,
    pub out_base: &'a str,
    pub truth: Option<&'a str>,
    pub threads: usize,
}

/// Run a single test by spawning selphi as a subprocess.
fn run_one(name: &str, cli_args: &[&str], pass: &mut u32, fail: &mut u32) {
    let exe = std::env::current_exe().expect("cannot find own executable");
    let output = std::process::Command::new(exe).args(cli_args).output();
    match output {
        Ok(o) if o.status.success() => {
            *pass += 1;
            selphi_info!("  \x1b[32mPASS\x1b[0m  {}", name);
        }
        Ok(o) => {
            *fail += 1;
            let stderr = String::from_utf8_lossy(&o.stderr);
            let last = stderr.lines().last().unwrap_or("");
            selphi_info!("  \x1b[31mFAIL\x1b[0m  {}  ({})", name, last);
        }
        Err(e) => {
            *fail += 1;
            selphi_info!("  \x1b[31mFAIL\x1b[0m  {}  ({})", name, e);
        }
    }
}

/// Run all self-tests. Returns the number of failures (0 = all pass).
pub fn run(config: &SelfTestConfig) -> u32 {
    let SelfTestConfig { refpanel, input, map, out_base, truth, threads } = config;
    let t = format!("--threads={}", threads);

    // Initialize logger + banner (same pattern as other modes)
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

    // 1. Phase-only
    let out = format!("{}_phase", out_base);
    test!("phase-only (haploid)",
        "--refpanel", refpanel, "--input", input, "--map", map, "--out", &out, &t, "--phase-only");

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

    // 7. Pre-phased input (uses phase-only output from test 1)
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

    // 9. BCF readability via bcftools (external, optional)
    let bcf_path = format!("{}_bcf.bcf", out_base);
    if Path::new(&bcf_path).exists() {
        let query = std::process::Command::new("bcftools")
            .args(["view", "-H", &bcf_path])
            .output();
        match query {
            Ok(o) if o.status.success() => {
                let n = o.stdout.iter().filter(|&&b| b == b'\n').count();
                if n > 0 {
                    pass += 1;
                    selphi_info!("  \x1b[32mPASS\x1b[0m  BCF readable ({} variants)", n);
                } else {
                    fail += 1;
                    selphi_info!("  \x1b[31mFAIL\x1b[0m  BCF empty");
                }
            }
            _ => {
                fail += 1;
                selphi_info!("  \x1b[31mFAIL\x1b[0m  BCF read (bcftools not found?)");
            }
        }
    }

    // Summary
    selphi_info!("");
    if fail == 0 {
        selphi_info!("All {} tests passed.", pass);
    } else {
        selphi_info!("{} passed, {} FAILED.", pass, fail);
    }

    fail
}
