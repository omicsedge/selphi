//! Validation: read_target_vcf must return IDENTICAL (samples, markers,
//! genotypes, is_phased) for a VCF target and the same data as binary BCF —
//! i.e. the BCF dispatch (read_target_bcf) matches the VCF text parser.
//! Usage: target_read_check <panel.srp> <target.vcf.gz> <target.bcf>
use selphi::io::target_io::read_target_vcf;
use selphi::srp::SrpReader;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let srp = SrpReader::open(&a[1], 0).expect("open srp");
    let (sv, mv, gv, pv) = read_target_vcf(&a[2], &srp);
    let (sb, mb, gb, pb) = read_target_vcf(&a[3], &srp);
    let mut marker_diff = 0;
    for (x, y) in mv.iter().zip(mb.iter()) {
        if x.chrom != y.chrom || x.pos != y.pos || x.ref_allele != y.ref_allele
            || x.alt_allele != y.alt_allele || x.ref_hash != y.ref_hash || x.alt_hash != y.alt_hash {
            marker_diff += 1;
        }
    }
    println!("samples: {} vs {}", sv.len(), sb.len());
    println!("markers: {} vs {} (field mismatch {})", mv.len(), mb.len(), marker_diff);
    println!("genotypes_eq: {}", gv == gb);
    println!("is_phased: {} vs {}", pv, pb);
    let pass = sv == sb && mv.len() == mb.len() && marker_diff == 0 && gv == gb && pv == pb;
    println!("{}", if pass { "PASS: BCF reader == VCF reader" } else { "FAIL" });
    std::process::exit(if pass { 0 } else { 1 });
}
