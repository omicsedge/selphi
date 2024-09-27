from typing import Iterable, List, Tuple
from pathlib import Path
from logging import Logger

from scipy import sparse
import numpy as np

from modules.hmm_utils import HMM
from modules.load_data import SparseCompositeMatchesNpz


def _filter_hap_indices(row_haps: np.ndarray, kept_haps: np.ndarray) -> np.ndarray:
    filtered = np.intersect1d(row_haps, kept_haps)
    if filtered.size:
        return filtered
    return row_haps


class CompositePanelMaskFilter:
    """
    Apply filters based on:
    - number of matches for each chip site in the composite ref panel
       (`haps_freqs_array_norm`)
    - estimated number of haploids that should be taken for each chip site
       (`nc_thresh`)
    """

    def __init__(
        self,
        target_hap: Tuple[str, int],
        npz_dir: Path,
        shape: Tuple[int],
        logger: Logger,
        kept_matches: int = 50,
    ):
        self.target_hap = target_hap
        self.npz_dir = npz_dir
        self.shape = shape
        self.logger = logger
        self.kept_matches = kept_matches

    @staticmethod
    def _expand_matches(
        matches: Iterable[Tuple[int]], keep: np.ndarray, hap: int, counts: np.ndarray
    ) -> np.ndarray:
        """Get variants to keep for a haplotype"""
        var_inds = np.hstack(
            [
                np.arange(start, stop + 1, dtype=np.int32)
                for start, stop in matches
                if (np.less_equal(start, keep) & np.less_equal(keep, stop)).any()
            ]
        )
        counts[hap] = var_inds.size
        return var_inds

    def sparse_matrix(self) -> sparse.csr_matrix:
        match_matrix = SparseCompositeMatchesNpz(
            *self.target_hap, self.npz_dir, self.shape, self.logger
        )
        best_matches = [
            match_matrix.get_row_matches(row)[: self.kept_matches]
            for row in range(self.shape[1])
        ]
        haps, counts = np.unique(np.hstack(best_matches), return_counts=True)
        best_haps = haps[counts > 1]
        filtered_matches = [_filter_hap_indices(row, best_haps) for row in best_matches]
        del best_matches
        indptr = np.append(
            [0], np.cumsum([matches.size for matches in filtered_matches])
        )
        indices = np.hstack(filtered_matches)
        del filtered_matches
        matrix = sparse.csc_matrix(
            (np.full(indptr[-1], True, dtype=np.bool_), indices, indptr),
            shape=self.shape,
        ).tocsr()
        counts = np.zeros(self.shape[0], dtype=int)
        indices = np.hstack(
            [
                self._expand_matches(
                    match_matrix.get_hap_matches(hap), matrix[hap].indices, hap, counts
                )
                for hap in np.unique(indices)
            ]
        )
        del matrix
        indptr = np.append([0], np.cumsum(counts))
        return sparse.csr_matrix(
            (np.full(indptr[-1], True, dtype=np.bool_), indices, indptr),
            shape=self.shape,
        ).transpose()

    def haplotype_id_lists(self) -> np.ndarray:
        return self.sparse_matrix().tolil().rows.copy()


def calculate_weights(
    target_hap: Tuple[str, int],
    chip_cM_coordinates: np.ndarray,
    npz_dir: Path,
    shape: Tuple[int],
    output_dir: Path,
    output_breaks: List[Tuple[int]],
    logger: Logger,
    est_ne: int = 1000000,
) -> None:
    """
    Load pbwt matches from npz and calculate weights for imputation
    """
    ordered_hap_indices = CompositePanelMaskFilter(
        target_hap, npz_dir, shape, logger
    ).haplotype_id_lists()

    haps, counts = np.unique(np.hstack(ordered_hap_indices), return_counts=True)
    cutoff = min(np.percentile(counts, 10), counts.max() - 1)
    best_haps = haps[counts > cutoff]

    HMM(
        [_filter_hap_indices(row, best_haps) for row in ordered_hap_indices],
        chip_cM_coordinates,
        output_dir,
        target_hap,
        output_breaks,
        shape,
        est_ne=est_ne,
        pErr=cutoff / shape[1],
    ).run()
