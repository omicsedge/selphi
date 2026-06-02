//! Validation: read_target_vcf must return IDENTICAL (samples, markers,
//! genotypes, is_phased) for a VCF target and the same data as binary BCF —
//! i.e. the BCF dispatch (read_target_bcf) matches the VCF text parser.
//! Usage: target_read_check <panel.srp> <target.vcf.gz> <target.bcf>
use selphi::io::target_io::{read_target_vcf, read_cohort_vcf, read_target_vcf_multi_chr};
use selphi::srp::SrpReader;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let srp = SrpReader::open(&a[1], 0).expect("open srp");
    let (sv, mv, gv, pv) = read_target_vcf(&a[2], &srp);
    let (sb, mb, gb, pb) = read_target_vcf(&a[3], &srp);

    // Cross-reader GT-parse consistency: read_cohort_vcf + read_target_vcf_multi_chr
    // share split_vcf_fields + parse_gt_region with read_target_vcf, so on the same
    // VCF their (sample_names, genotypes-in-order, is_phased) must match exactly
    // (markers differ only in hash/id, which we don't compare here).
    let (sc, _mc, gc, pc) = read_cohort_vcf(&a[2]);
    let (sm, by_chr, pm) = read_target_vcf_multi_chr(&a[2]);
    let gm: Vec<_> = by_chr.values().flat_map(|(_, g)| g.iter().cloned()).collect();
    let cohort_ok = sc == sv && gc == gv && pc == pv;
    let multichr_ok = sm == sv && gm == gv && pm == pv;
    println!("cross-reader: cohort_ok={} multichr_ok={}", cohort_ok, multichr_ok);
    let mut marker_diff = 0;
    for (x, y) in mv.iter().zip(mb.iter()) {
        if x.chrom != y.chrom || x.pos != y.pos || x.ref_allele != y.ref_allele
            || x.alt_allele != y.alt_allele || x.ref_hash != y.ref_hash || x.alt_hash != y.alt_hash {
            marker_diff += 1;
        }
    }
    // Biallelic-projection invariant: every extracted allele must be 0 or 1
    // (multiallelic index 2+ must be clamped at the GT parse, else chip
    // passthrough emits a GT allele beyond the single output ALT).
    let max_v = gv.iter().flatten().flat_map(|g| g.iter()).copied().max().unwrap_or(0);
    let max_b = gb.iter().flatten().flat_map(|g| g.iter()).copied().max().unwrap_or(0);
    println!("samples: {} vs {}", sv.len(), sb.len());
    println!("markers: {} vs {} (field mismatch {})", mv.len(), mb.len(), marker_diff);
    println!("genotypes_eq: {}", gv == gb);
    println!("is_phased: {} vs {}", pv, pb);
    println!("max_allele: vcf={} bcf={} (must be <=1)", max_v, max_b);
    let pass = sv == sb && mv.len() == mb.len() && marker_diff == 0 && gv == gb && pv == pb
        && max_v <= 1 && max_b <= 1 && cohort_ok && multichr_ok;
    println!("{}", if pass { "PASS: BCF reader == VCF reader" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
