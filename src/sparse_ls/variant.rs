//! Reference-panel variant map for the GLIMPSE2 model.
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
    /// Genetic position (cM), f64 to match the GLIMPSE2 model.
    pub cm: f64,
    /// Low-quality flag (very-soft GL site routed through the emission-skip path).
    pub lq: bool,
}

impl Variant {
    /// Minor-allele count = min(cref, calt).
    #[inline]
    pub fn mac(&self) -> u32 {
        self.cref.min(self.calt)
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mac() {
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
    }
}
