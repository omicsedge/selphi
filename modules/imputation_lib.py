from typing import List, Tuple
from pathlib import Path
from logging import Logger

from scipy import sparse
import numpy as np

from modules.hmm_utils import pRecomb, setFwdValues_SPARSE, setBwdValues_SPARSE
from modules.load_data import load_sparse_comp_matches_hybrid_npz
from modules.utils import get_std


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

    def _haps_freqs_array_norm(
        self, matches_row: sparse.csr_matrix, tmin: float = 0.1, tmax: float = 1
    ) -> sparse.csc_matrix:
        """
        First, calculates what is the number of matches for each chip variant in
            the new composite ref panel
        Second, normalizes the results linearly between `tmin` and `tmax`,
            yielding a useful multiplicative coefficient which scale down ref haploids
            that have less total matches with the test sample
        """
        if not 0 <= tmin < tmax <= 1:
            raise ValueError(
                "tmin and tmax parameters must be between 0 and 1, "
                "and tmax has to be greater than tmin"
            )
        freqs: np.ndarray = matches_row.getnnz(axis=1)
        rmin: int = freqs.min()
        rmax: int = freqs.max()
        return sparse.csr_matrix(
            ((freqs - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin
        ).transpose()

    def _normalize_matches(self, matches_row: sparse.csr_matrix) -> sparse.csc_matrix:
        """Multiply matches by normalized haplotype frequencies"""
        return matches_row.multiply(self._haps_freqs_array_norm(matches_row)).tocsc()

    def _thresh(
        self, matches_row: sparse.csr_matrix, X: sparse.csc_matrix
    ) -> np.ndarray:
        averages = matches_row.sum(axis=0) / matches_row.getnnz(axis=0)
        std_thresh: np.ndarray = get_std(
            np.array(averages)[0], matches_row.data.min(), self.shape[1]
        )
        std_ = np.array([np.std(X.getcol(index).data) for index in range(X.shape[1])])
        return np.array(X.max(axis=0) - std_thresh * std_)[0]

    def _best_matches(self, matches_row: sparse.csr_matrix) -> List[np.ndarray]:
        """Generate all best matches at all variants"""
        normalized = self._normalize_matches(matches_row)
        thresh = self._thresh(matches_row, normalized)
        return [
            row.indices[
                np.argsort(row.data)[::-1][
                    : min(self.kept_matches, (row.data >= thresh[idx]).sum())
                ]
            ].copy()
            for idx, row in enumerate(normalized.transpose())
        ]

    def _expand_matches(
        self, row, keep: np.ndarray, hap: int, counts: np.ndarray
    ) -> np.ndarray:
        """Get variants to keep for a haplotype"""
        var_inds = np.hstack(
            [
                run
                for run in np.split(row, np.where(np.diff(row) != 1)[0] + 1)
                if (np.less_equal(run[0], keep) & np.less_equal(keep, run[-1])).any()
            ]
        )
        counts[hap] = var_inds.size
        return var_inds

    def sparse_matrix(self) -> sparse.csr_matrix:
        matches_row: sparse.csr_matrix = load_sparse_comp_matches_hybrid_npz(
            *self.target_hap, self.npz_dir, self.shape, self.logger
        )
        best_matches = self._best_matches(matches_row)
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
        del indptr
        counts = np.zeros(self.shape[0], dtype=int)
        indices = np.hstack(
            [
                self._expand_matches(
                    matches_row[hap].indices, matrix[hap].indices, hap, counts
                )
                for hap in np.unique(indices)
            ]
        )
        del matches_row
        del matrix
        indptr = np.append([0], np.cumsum(counts))
        return sparse.csr_matrix(
            (np.full(indptr[-1], True, dtype=np.bool_), indices, indptr),
            shape=self.shape,
        ).transpose()

    def haplotype_id_lists(self) -> np.ndarray:
        return self.sparse_matrix().tolil().rows.copy()


def run_hmm(
    chip_sites_n: int,
    ordered_hap_indices: np.ndarray,
    distances_cm: np.ndarray,
    num_hid: int = 9000,
    est_ne: int = 1000000,
    pErr: float = 0.0001,
) -> sparse.csr_matrix:
    """
    Runs the forward and backward runs of the forward-backward algorithm,
        to get probabilities for imputation
    """
    nHaps = np.array([len(row) for row in ordered_hap_indices])
    pRecomb_arr: np.ndarray = pRecomb(distances_cm, num_hid=num_hid, ne=est_ne) / nHaps

    matrix_ = setFwdValues_SPARSE(
        chip_sites_n, ordered_hap_indices, pRecomb_arr, num_hid=num_hid, pErr=pErr
    )
    return setBwdValues_SPARSE(
        matrix_,
        chip_sites_n,
        ordered_hap_indices,
        pRecomb_arr,
        nHaps,
        num_hid=num_hid,
        pErr=pErr,
    )


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
    Processing as one chunk (CHUNK_SIZE = chip_sites_n)
    """
    ordered_hap_indices = CompositePanelMaskFilter(
        target_hap, npz_dir, shape, logger
    ).haplotype_id_lists()

    haps, counts = np.unique(np.hstack(ordered_hap_indices), return_counts=True)
    cutoff = min(np.percentile(counts, 10), counts.max() - 1)
    best_haps = haps[counts > cutoff]

    weight_matrix = run_hmm(
        shape[1],
        [_filter_hap_indices(row, best_haps) for row in ordered_hap_indices],
        chip_cM_coordinates,
        num_hid=shape[0],
        est_ne=est_ne,
        pErr=cutoff / shape[1],
    )
    del ordered_hap_indices

    for start, stop in output_breaks:
        sparse.save_npz(
            output_dir.joinpath(str(start), f"{target_hap[0]}_{target_hap[1]}.npz"),
            weight_matrix[start:stop, :],
        )
