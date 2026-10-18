//! Bounded integer membership without sorting or hashing. The bitmap is small
//! (one bit per site/haplotype) and reused on each worker; result order is chosen
//! explicitly by the caller and never depends on bitmap insertion order.

use std::cell::RefCell;

#[derive(Default)]
pub(crate) struct Membership {
    words: Vec<u64>,
}

impl Membership {
    fn reset(&mut self, n: usize) {
        self.words.resize(n.div_ceil(64), 0);
        self.words.fill(0);
    }

    pub fn insert(&mut self, i: usize) -> bool {
        let word = &mut self.words[i / 64];
        let bit = 1u64 << (i % 64);
        let fresh = *word & bit == 0;
        *word |= bit;
        fresh
    }

    pub fn contains(&self, i: usize) -> bool {
        self.words[i / 64] & (1u64 << (i % 64)) != 0
    }

    pub fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(i, &word)| {
            let mut rest = word;
            std::iter::from_fn(move || {
                if rest == 0 { return None; }
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                Some(i * 64 + bit)
            })
        })
    }
}

thread_local! {
    static MEMBERSHIP: RefCell<Membership> = const { RefCell::new(Membership { words: Vec::new() }) };
}

pub(crate) fn with_membership<R>(n: usize, f: impl FnOnce(&mut Membership) -> R) -> R {
    MEMBERSHIP.with(|scratch| {
        let mut bits = scratch.borrow_mut();
        bits.reset(n);
        f(&mut bits)
    })
}

/// A cap means first-occurrence priority; no cap means an ascending union.
/// Stop consuming the input as soon as the first K distinct IDs are known.
pub(crate) fn ordered_or_sorted_union(
    ids: impl IntoIterator<Item = u32>, n: usize, cap: Option<usize>,
) -> Vec<u32> {
    with_membership(n, |seen| {
        if let Some(cap) = cap {
            let mut out = Vec::new();
            if cap == 0 { return out; }
            for id in ids {
                if seen.insert(id as usize) {
                    out.push(id);
                    if out.len() == cap { break; }
                }
            }
            out
        } else {
            for id in ids { seen.insert(id as usize); }
            seen.indices().map(|i| i as u32).collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unions_match_old_dedup_and_truncation() {
        use crate::lcwgs::hmm::reference_pack::Xs;
        let mut rng = Xs(0x3C6E_F372_FE94_F82B);
        for n in [1, 7, 63, 64, 65, 4796, 9] {
            for _ in 0..40 {
                let ids: Vec<u32> = (0..rng.below((n * 4) as u64)).map(|_| rng.below(n as u64) as u32).collect();
                for cap in [None, Some(0), Some(1), Some(n / 2), Some(n), Some(n * 2)] {
                    let mut expected = ids.clone();
                    if let Some(cap) = cap {
                        let mut seen = std::collections::HashSet::new();
                        expected.retain(|&id| seen.insert(id));
                        expected.truncate(cap);
                    } else {
                        expected.sort_unstable();
                        expected.dedup();
                    }
                    assert_eq!(ordered_or_sorted_union(ids.iter().copied(), n, cap), expected);
                }
            }
        }
        assert!(ordered_or_sorted_union([], 0, None).is_empty());
    }
}
