//! Faithful port of GLIMPSE2 `variant.h` / `variant_map.h`.
//!
//! A reference-panel variant with PHRED counts and cM position. `mac()` is the
//! minor-allele count used by the common/rare split and the sparse PBWT.

#[derive(Clone, Debug)]
pub struct Variant {
    pub bp: i64,
    pub id: String,
    pub ref_a: String,
    pub alt_a: String,
    /// Phasing-HMM variant type (set per iteration from the current genotype).
    pub vtype: i8,
    /// Index into the polymorphic-site arrays.
    pub idx: i32,
    /// Reference-allele count in the panel.
    pub cref: u32,
    /// Alternate-allele count in the panel.
    pub calt: u32,
    /// Genetic position (cM), f64 to match GLIMPSE2.
    pub cm: f64,
    /// Low-quality flag (very-soft GL site routed through the emission-skip path).
    pub lq: bool,
}

impl Variant {
    /// Minor-allele count = min(cref, calt). (variant.h `getMAC`)
    #[inline]
    pub fn mac(&self) -> u32 {
        self.cref.min(self.calt)
    }
    /// Alternate allele frequency (calt / (cref+calt)).
    #[inline]
    pub fn af(&self) -> f64 {
        let n = (self.cref + self.calt) as f64;
        if n > 0.0 {
            self.calt as f64 / n
        } else {
            0.0
        }
    }
}

/// Ordered variants + helpers (cM lookup, common/rare classification).
#[derive(Default)]
pub struct VariantMap {
    pub vars: Vec<Variant>,
}

impl VariantMap {
    pub fn new() -> Self {
        VariantMap { vars: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.vars.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vars.is_empty()
    }
    /// Is variant `i` common under the SPARSE_MAF split (AF in [maf, 1-maf]).
    #[inline]
    pub fn is_common(&self, i: usize, sparse_maf: f64) -> bool {
        let af = self.vars[i].af();
        af >= sparse_maf && af <= 1.0 - sparse_maf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mac_and_af() {
        let v = Variant {
            bp: 100,
            id: "rs1".into(),
            ref_a: "A".into(),
            alt_a: "G".into(),
            vtype: 0,
            idx: 0,
            cref: 90,
            calt: 10,
            cm: 1.0,
            lq: false,
        };
        assert_eq!(v.mac(), 10);
        assert!((v.af() - 0.1).abs() < 1e-12);
    }
}
