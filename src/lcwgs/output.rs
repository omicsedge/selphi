//! VCF.gz writer for lcWGS imputation output.
//!
//! Emits one record per shared variant with `GT:DS:GP`:
//! - `GT` — hard genotype call from the max-posterior genotype (phased "|"
//!   when the het orientation is determined, else "/").
//! - `DS` — ALT dosage E[ALT count] ∈ [0, 2].
//! - `GP` — three genotype posteriors P(0/0), P(0/1), P(1/1).
//!
//! Variant identity (chrom/pos/ref/alt) comes from the shared-panel order in
//! `LcwgsOutput.variants`, so the VCF is in genomic order and matches the
//! reference panel coordinates.

use std::io::{BufWriter, Write};

use super::pipeline::LcwgsOutput;

/// Write the lcWGS imputation result as a bgzipped VCF.
pub fn write_lcwgs_vcf(out: &LcwgsOutput, output_path: &std::path::Path) -> std::io::Result<()> {
    let n_samples = out.sample_ids.len();
    let n_var = out.n_variants;

    let file = std::fs::File::create(output_path)?;
    let bgzf = noodles_bgzf::io::multithreaded_writer::Builder::default()
        .set_worker_count(std::num::NonZeroUsize::new(4).unwrap())
        .build_from_writer(file);
    let mut w = BufWriter::with_capacity(8 << 20, bgzf);

    // Header
    writeln!(w, "##fileformat=VCFv4.2")?;
    writeln!(w, "##source=Selphi_v{}_lcWGS SelfDecode\u{2122}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w, "##FILTER=<ID=PASS,Description=\"All filters passed\">")?;
    writeln!(w, "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT allele frequency\">")?;
    writeln!(w, "##INFO=<ID=RAF,Number=A,Type=Float,Description=\"Reference-panel ALT allele frequency\">")?;
    writeln!(w, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;
    writeln!(w, "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Estimated ALT dose [0,2]\">")?;
    writeln!(w, "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Estimated genotype posteriors\">")?;
    write!(w, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
    for s in &out.sample_ids { write!(w, "\t{}", s)?; }
    writeln!(w)?;

    let mut line = String::with_capacity(n_samples * 24);
    for v in 0..n_var {
        let (chrom, pos, r, a) = &out.variants[v];
        // INFO AF = mean dosage / 2 across samples
        let mut ds_sum = 0.0f64;
        for s in 0..n_samples { ds_sum += out.dosage[v * n_samples + s] as f64; }
        let af = if n_samples > 0 { ds_sum / (2.0 * n_samples as f64) } else { 0.0 };

        line.clear();
        line.push_str(chrom);
        line.push('\t');
        line.push_str(&pos.to_string());
        line.push_str("\t.\t");
        line.push_str(r);
        line.push('\t');
        line.push_str(a);
        line.push_str("\t.\tPASS\tAF=");
        line.push_str(&format!("{:.4}", af));
        line.push_str("\tGT:DS:GP");

        for s in 0..n_samples {
            let ds = out.dosage[v * n_samples + s];
            let g_off = (v * n_samples + s) * 3;
            let g0 = out.gp[g_off];
            let g1 = out.gp[g_off + 1];
            let g2 = out.gp[g_off + 2];
            // Hard call = argmax genotype.
            let (gt, _) = if g0 >= g1 && g0 >= g2 {
                ("0/0", g0)
            } else if g2 >= g0 && g2 >= g1 {
                ("1/1", g2)
            } else {
                ("0/1", g1)
            };
            line.push('\t');
            line.push_str(gt);
            line.push(':');
            line.push_str(&format!("{:.3}", ds));
            line.push(':');
            line.push_str(&format!("{:.3},{:.3},{:.3}", g0, g1, g2));
        }
        writeln!(w, "{}", line)?;
    }

    w.flush()?;
    Ok(())
}
