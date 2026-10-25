//! Compact in-memory form of an SRP panel's per-variant sections: the variant index
//! (chromosome, position, REF, ALT) and the newline-delimited ID columns.
//!
//! The readers used to hold one `Variant { chr: String, pos, ref_allele: String,
//! alt_allele: String }` per variant and one `String` per ID: three heap blocks plus an
//! 80-byte record per variant, and a 24-byte header plus a heap block per ID. On chr20 of
//! the 75,552-haplotype v3 panel (4.63 M variants) that was ~1.1 GB of a 2.6 GB one-user
//! run. Here every column is one allocation: positions in a `Vec<i64>`, REF and ALT in one
//! byte arena with offsets, the chromosome name once per distinct name (a per-variant
//! number only when a panel's index holds more than one name), and each ID column one
//! arena with offsets. Readers hand out `&str` borrowed from the arenas; nothing is
//! copied on access. The on-disk format is unchanged.

use std::ops::Index;

/// End offsets into an arena: u32 while the arena is below 4 GiB, u64 once it grows past
/// (a column of that size is not expected, but it must not be refused or wrap).
#[derive(Clone, Debug)]
enum Ends {
    Narrow(Vec<u32>),
    Wide(Vec<u64>),
}

impl Ends {
    fn with_capacity(n: usize) -> Self {
        let mut v = Vec::with_capacity(n);
        v.push(0);
        Ends::Narrow(v)
    }

    #[inline(always)]
    fn at(&self, i: usize) -> usize {
        match self {
            Ends::Narrow(v) => v[i] as usize,
            Ends::Wide(v) => v[i] as usize,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Ends::Narrow(v) => v.len(),
            Ends::Wide(v) => v.len(),
        }
    }

    #[inline]
    fn push(&mut self, end: usize) {
        match self {
            Ends::Narrow(v) => match u32::try_from(end) {
                Ok(e) => v.push(e),
                Err(_) => {
                    let mut w: Vec<u64> = Vec::with_capacity(v.capacity().max(v.len() + 1));
                    w.extend(v.iter().map(|&e| e as u64));
                    w.push(end as u64);
                    *self = Ends::Wide(w);
                }
            },
            Ends::Wide(v) => v.push(end as u64),
        }
    }

    fn shrink_to_fit(&mut self) {
        match self {
            Ends::Narrow(v) => v.shrink_to_fit(),
            Ends::Wide(v) => v.shrink_to_fit(),
        }
    }

    fn heap_bytes(&self) -> usize {
        match self {
            Ends::Narrow(v) => v.capacity() * 4,
            Ends::Wide(v) => v.capacity() * 8,
        }
    }
}

// ---------------------------------------------------------------------------
// StrColumn
// ---------------------------------------------------------------------------

/// A column of strings held in one arena: entry `i` is `text[ends[i]..ends[i + 1]]`.
/// Indexing (`col[i]`) gives a `&str` and panics out of range, as `Vec<String>` did.
#[derive(Clone)]
pub struct StrColumn {
    text: String,
    ends: Ends,
}

impl Default for StrColumn {
    fn default() -> Self { Self::new() }
}

impl StrColumn {
    pub fn new() -> Self { Self::with_capacity(0, 0) }

    /// An empty column with room for `n` entries of `bytes` bytes in total.
    pub fn with_capacity(n: usize, bytes: usize) -> Self {
        StrColumn { text: String::with_capacity(bytes), ends: Ends::with_capacity(n + 1) }
    }

    /// The pieces of `text` between `'\n'`s, exactly as `text.split('\n')` gives them
    /// (a trailing separator ends with an empty piece; `""` is one empty piece), without
    /// the empty ones when `filter_empty`. The separators are squeezed out of `text` in
    /// place, so the column costs the text's own allocation plus its offsets.
    pub fn split_lines(text: String, filter_empty: bool) -> Self {
        let mut bytes = text.into_bytes();
        let n_pieces = memchr::memchr_iter(b'\n', &bytes).count() + 1;
        let mut ends = Ends::with_capacity(n_pieces + 1);
        let (mut r, mut w) = (0usize, 0usize);
        loop {
            let sep = memchr::memchr(b'\n', &bytes[r..]).map(|k| r + k);
            let end = sep.unwrap_or(bytes.len());
            let len = end - r;
            if len > 0 {
                if w != r { bytes.copy_within(r..end, w); }
                w += len;
            }
            if len > 0 || !filter_empty { ends.push(w); }
            match sep {
                Some(s) => r = s + 1,
                None => break,
            }
        }
        bytes.truncate(w);
        bytes.shrink_to_fit();
        ends.shrink_to_fit();
        // Only ASCII '\n' bytes were removed, at piece boundaries: still valid UTF-8 and
        // every offset is a character boundary.
        let text = String::from_utf8(bytes).expect("removing '\\n' keeps UTF-8 valid");
        StrColumn { text, ends }
    }

    #[inline]
    pub fn len(&self) -> usize { self.ends.len() - 1 }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }

    #[inline]
    pub fn get(&self, i: usize) -> Option<&str> {
        if i < self.len() { Some(self.at(i)) } else { None }
    }

    #[inline(always)]
    fn at(&self, i: usize) -> &str { &self.text[self.ends.at(i)..self.ends.at(i + 1)] }

    /// Append one entry.
    #[inline]
    pub fn push(&mut self, s: &str) {
        self.text.push_str(s);
        self.ends.push(self.text.len());
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &str> + DoubleEndedIterator + '_ {
        (0..self.len()).map(move |i| self.at(i))
    }

    pub fn shrink_to_fit(&mut self) {
        self.text.shrink_to_fit();
        self.ends.shrink_to_fit();
    }

    /// Bytes this column holds on the heap.
    pub fn heap_bytes(&self) -> usize { self.text.capacity() + self.ends.heap_bytes() }
}

impl Index<usize> for StrColumn {
    type Output = str;
    #[inline]
    fn index(&self, i: usize) -> &str {
        assert!(i < self.len(), "index {i} out of range for a column of {} strings", self.len());
        self.at(i)
    }
}

impl<S: AsRef<str>> FromIterator<S> for StrColumn {
    fn from_iter<I: IntoIterator<Item = S>>(iter: I) -> Self {
        let mut c = StrColumn::new();
        for s in iter { c.push(s.as_ref()); }
        c
    }
}

impl std::fmt::Debug for StrColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl PartialEq for StrColumn {
    fn eq(&self, other: &Self) -> bool { self.len() == other.len() && self.iter().eq(other.iter()) }
}

// ---------------------------------------------------------------------------
// VariantIndex
// ---------------------------------------------------------------------------

/// One panel variant, borrowed from a [`VariantIndex`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VariantRef<'a> {
    pub chr: &'a str,
    pub pos: i64,
    pub ref_allele: &'a str,
    pub alt_allele: &'a str,
}

/// The panel's variant index, column by column: positions contiguous, REF and ALT of
/// variant `i` at entries `2i` and `2i + 1` of one allele arena, and the chromosome as
/// one name per distinct name (`chrom_of` is filled only when there is more than one).
#[derive(Clone, Default)]
pub struct VariantIndex {
    pos: Vec<i64>,
    alleles: StrColumn,
    chroms: Vec<String>,
    /// Per variant, its name in `chroms`; empty while every variant carries `chroms[0]`.
    chrom_of: Vec<u32>,
}

impl VariantIndex {
    pub fn new() -> Self { Self::default() }

    /// An empty index with room for `n` variants of `allele_bytes` REF + ALT bytes in total.
    pub fn with_capacity(n: usize, allele_bytes: usize) -> Self {
        VariantIndex {
            pos: Vec::with_capacity(n),
            alleles: StrColumn::with_capacity(2 * n, allele_bytes),
            chroms: Vec::new(),
            chrom_of: Vec::new(),
        }
    }

    /// Append one variant.
    pub fn push(&mut self, chr: &str, pos: i64, ref_allele: &str, alt_allele: &str) {
        let n = self.pos.len();
        // The previous variant's name first: an index is sorted, so it changes rarely.
        let prev = if n == 0 { None } else if self.chrom_of.is_empty() { Some(0) } else { Some(self.chrom_of[n - 1] as usize) };
        let ci = match prev {
            Some(p) if self.chroms[p] == chr => p,
            _ => match self.chroms.iter().position(|c| c == chr) {
                Some(ci) => ci,
                None => { self.chroms.push(chr.to_string()); self.chroms.len() - 1 }
            },
        };
        if !self.chrom_of.is_empty() || ci != 0 {
            if self.chrom_of.is_empty() { self.chrom_of = vec![0; n]; }
            self.chrom_of.push(u32::try_from(ci).expect("more than 2^32 chromosome names"));
        }
        self.pos.push(pos);
        self.alleles.push(ref_allele);
        self.alleles.push(alt_allele);
    }

    #[inline]
    pub fn len(&self) -> usize { self.pos.len() }

    #[inline]
    pub fn is_empty(&self) -> bool { self.pos.is_empty() }

    /// Position of variant `i`.
    #[inline(always)]
    pub fn pos(&self, i: usize) -> i64 { self.pos[i] }

    /// Every variant's position, in panel order.
    #[inline]
    pub fn positions(&self) -> &[i64] { &self.pos }

    /// Chromosome of variant `i`.
    #[inline]
    pub fn chr(&self, i: usize) -> &str {
        assert!(i < self.len(), "variant {i} out of range for an index of {}", self.len());
        let ci = if self.chrom_of.is_empty() { 0 } else { self.chrom_of[i] as usize };
        &self.chroms[ci]
    }

    /// REF allele of variant `i` (as the index stores it: a hashed panel holds hashes).
    #[inline]
    pub fn ref_allele(&self, i: usize) -> &str {
        assert!(i < self.len(), "variant {i} out of range for an index of {}", self.len());
        self.alleles.at(2 * i)
    }

    /// ALT allele of variant `i` (as the index stores it).
    #[inline]
    pub fn alt_allele(&self, i: usize) -> &str {
        assert!(i < self.len(), "variant {i} out of range for an index of {}", self.len());
        self.alleles.at(2 * i + 1)
    }

    /// Variant `i`, borrowed.
    #[inline]
    pub fn get(&self, i: usize) -> VariantRef<'_> {
        VariantRef { chr: self.chr(i), pos: self.pos[i], ref_allele: self.ref_allele(i), alt_allele: self.alt_allele(i) }
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = VariantRef<'_>> + DoubleEndedIterator + '_ {
        (0..self.len()).map(move |i| self.get(i))
    }

    /// The distinct chromosome names, in order of first appearance.
    pub fn chromosomes(&self) -> &[String] { &self.chroms }

    pub fn shrink_to_fit(&mut self) {
        self.pos.shrink_to_fit();
        self.alleles.shrink_to_fit();
        self.chroms.shrink_to_fit();
        self.chrom_of.shrink_to_fit();
    }

    /// Bytes this index holds on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.pos.capacity() * 8 + self.alleles.heap_bytes()
            + self.chroms.iter().map(|c| c.capacity() + 24).sum::<usize>() + self.chrom_of.capacity() * 4
    }
}

impl std::fmt::Debug for VariantIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a> FromIterator<VariantRef<'a>> for VariantIndex {
    fn from_iter<I: IntoIterator<Item = VariantRef<'a>>>(iter: I) -> Self {
        let mut ix = VariantIndex::new();
        for v in iter { ix.push(v.chr, v.pos, v.ref_allele, v.alt_allele); }
        ix
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_lines_matches_str_split() {
        for text in ["", "\n", "a", "a\n", "\na", "a\n\nb", "rs1\nrs2\n\nrs4\n", "\n\n\n", "x-1-A-C\ny-2-ÄÖ-Ü\n"] {
            for filter in [false, true] {
                let want: Vec<&str> = text.split('\n').filter(|s| !filter || !s.is_empty()).collect();
                let col = StrColumn::split_lines(text.to_string(), filter);
                assert_eq!(col.len(), want.len(), "{text:?} filter={filter}");
                assert_eq!(col.iter().collect::<Vec<_>>(), want, "{text:?} filter={filter}");
                for (i, w) in want.iter().enumerate() {
                    assert_eq!(&col[i], *w);
                    assert_eq!(col.get(i), Some(*w));
                }
                assert_eq!(col.get(want.len()), None);
                assert_eq!(col, want.iter().collect::<StrColumn>());
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn column_index_out_of_range_panics() {
        let c: StrColumn = ["a", "b"].into_iter().collect();
        let _ = &c[2];
    }

    #[test]
    fn offsets_widen_past_4_gib() {
        let mut e = Ends::with_capacity(3);
        e.push(10);
        e.push(u32::MAX as usize);
        assert!(matches!(e, Ends::Narrow(_)));
        e.push(u32::MAX as usize + 7);
        assert!(matches!(e, Ends::Wide(_)));
        assert_eq!((0..4).map(|i| e.at(i)).collect::<Vec<_>>(), vec![0, 10, u32::MAX as usize, u32::MAX as usize + 7]);
    }

    /// Positions, alleles (a 255-byte and a 300-byte one, which the index writes escaped)
    /// and chromosome names, including the empty name and a second name mid-index, come
    /// back as pushed; the on-disk index parses into the same columns.
    #[test]
    fn variant_index_round_trip() {
        let long = "A".repeat(300);
        let exact = "G".repeat(255);
        let recs: Vec<(&str, i64, &str, &str)> = vec![
            ("22", 100, "A", "C"), ("22", 100, "A", "AT"), ("22", 205, long.as_str(), "A"),
            ("22", 300, "T", exact.as_str()), ("", 0, "", ""), ("chr22_KI270879v1_alt", 7, "C", "G"),
            ("22", i64::MAX, "N", "<DEL>"),
        ];
        let mut ix = VariantIndex::new();
        for r in &recs { ix.push(r.0, r.1, r.2, r.3); }
        let mut vbin = Vec::new();
        for r in &recs { crate::srp::helpers::push_variant_vbin(&mut vbin, r.1, r.0, r.2, r.3).unwrap(); }
        crate::srp::helpers::finish_vbin(&mut vbin, recs.len());
        assert_eq!(&vbin[..8], &crate::srp::helpers::LONG_ALLELE_MARKER, "a field of 255 B or more marks the index");
        let parsed = crate::srp::helpers::parse_variants_bin(&vbin, recs.len()).unwrap();
        for v in [&ix, &parsed] {
            assert_eq!(v.len(), recs.len());
            assert_eq!(v.positions(), recs.iter().map(|r| r.1).collect::<Vec<_>>().as_slice());
            for (i, r) in recs.iter().enumerate() {
                assert_eq!((v.chr(i), v.pos(i), v.ref_allele(i), v.alt_allele(i)), *r, "variant {i}");
                assert_eq!(v.get(i), VariantRef { chr: r.0, pos: r.1, ref_allele: r.2, alt_allele: r.3 });
            }
            assert_eq!(v.chromosomes(), &["22".to_string(), String::new(), "chr22_KI270879v1_alt".to_string()]);
            assert_eq!(v.iter().count(), recs.len());
        }
        assert_eq!(parsed.iter().collect::<VariantIndex>().iter().collect::<Vec<_>>(), ix.iter().collect::<Vec<_>>());

        // The empty-chromosome row alone (a panel of one such row) and an empty index.
        let mut s = VariantIndex::new();
        s.push("", 0, "", "");
        assert_eq!((s.len(), s.chr(0), s.pos(0), s.ref_allele(0), s.alt_allele(0)), (1, "", 0, "", ""));
        let e = crate::srp::helpers::parse_variants_bin(&[], 0).unwrap();
        assert!(e.is_empty() && e.positions().is_empty() && e.chromosomes().is_empty());
    }

    /// One chromosome name: no per-variant chromosome column, and an index of SNPs costs
    /// its position, two allele bytes and two offsets per variant.
    #[test]
    fn single_chromosome_index_is_compact() {
        let n = 10_000usize;
        let mut vbin = Vec::new();
        for i in 0..n { crate::srp::helpers::push_variant_vbin(&mut vbin, 1000 + i as i64, "20", "A", "G").unwrap(); }
        crate::srp::helpers::finish_vbin(&mut vbin, n);
        let v = crate::srp::helpers::parse_variants_bin(&vbin, n).unwrap();
        assert!(v.chrom_of.is_empty());
        assert_eq!(v.chr(n - 1), "20");
        assert!(v.heap_bytes() <= n * (8 + 2 + 2 * 4) + 64, "{} bytes for {n} SNPs", v.heap_bytes());
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn variant_out_of_range_panics() {
        let mut ix = VariantIndex::new();
        ix.push("1", 5, "A", "C");
        let _ = ix.ref_allele(1);
    }
}
