//! Validation/usage example: native BAM genotype-likelihood pileup.
//! Usage: bam_gl_check <bam> <chrom> <sites.tsv>
//! sites.tsv: one line per site, `pos<TAB>ref<TAB>alt` (1-based pos).
//! Prints `pos<TAB>PL0,PL1,PL2<TAB>` per site (Phred, min=0), to compare vs
//! `bcftools mpileup` PL on the same BAM/sites.
use selphi::lcwgs::bam_pileup::{pileup_bams, PileupParams};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (bam, chrom, sites) = (&a[1], &a[2], &a[3]);
    let mut pos = Vec::new();
    let mut rb = Vec::new();
    let mut ab = Vec::new();
    let mut snp = Vec::new();
    for line in std::fs::read_to_string(sites).unwrap().lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 3 { continue; }
        let p: i64 = match f[0].parse() { Ok(v) => v, Err(_) => continue };
        pos.push(p);
        rb.push(f[1].as_bytes()[0]);
        ab.push(f[2].as_bytes()[0]);
        snp.push(f[1].len() == 1 && f[2].len() == 1);
    }
    let r = pileup_bams(&[bam.clone()], chrom, &pos, &rb, &ab, &snp, PileupParams::default()).unwrap();
    for i in 0..pos.len() {
        let (g0, g1, g2) = (r.gl3[i * 3], r.gl3[i * 3 + 1], r.gl3[i * 3 + 2]);
        let mx = g0.max(g1).max(g2);
        let pl = |g: f32| if mx > 0.0 { ((-10.0 * (g / mx).log10()).round() as i64).clamp(0, 255) } else { 0 };
        println!("{}\t{},{},{}", pos[i], pl(g0), pl(g1), pl(g2));
    }
}
