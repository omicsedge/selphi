from typing import List, Tuple
from pathlib import Path
from scipy import sparse
import numpy as np

from modules.hmm_utils import pRecomb, setFwdValues_SPARSE, setBwdValues_SPARSE
from modules.load_data import load_sparse_comp_matches_hybrid_npz
from modules.utils import get_std


def calculate_haploid_frequencies_SPARSE(
    comp_matches_hybrid: sparse.csc_matrix,  # Match length where there is match length else 0
    CHUNK_SIZE: int,
    tmin: float = 0.1,
    tmax: float = 1,
) -> np.ndarray:
    """
    First, calculates what is the number of matches for each chip variant in the new composite ref panel
    Second, normalizes the results linearly between `tmin` and `tmax`,
        yielding a useful multiplicative coefficient which scale down ref haploids
            that have less total matches with the test sample
    """
    if not (0 <= tmin < tmax <= 1):
        raise ValueError(
            "tmin and tmax parameters must be between 0 and 1, and tmax has to be greater than tmin"
        )
    freqs: np.ndarray = np.vstack(
        [
            comp_matches_hybrid[:, start : start + CHUNK_SIZE].getnnz(axis=1)
            for start in range(0, comp_matches_hybrid.shape[1], CHUNK_SIZE)
        ]
    )
    rmin: np.ndarray = np.amin(freqs, axis=1)
    rmax: np.ndarray = np.amax(freqs, axis=1)
    return ((freqs - rmin) / (rmax - rmin)) * (tmax - tmin) + tmin


def calculate_haploid_count_threshold_SPARSE(
    comp_matches_hybrid: sparse.csc_matrix,
    haps_freqs_array_norm: np.ndarray,
    CHUNK_SIZE: int,
) -> np.ndarray:
    """
    1. calculates coefficients:
        - `std_thresh`
        An affine transformed (squeezed, stretched, moved) power function (`get_std`),
            which calculates a coefficient for each variant,
                from the average matches with the new composite ref panel
        - `lengthos`
        - standard deviations
    2. and uses them to calculate a final number of haploids taken for each chip index,
        using a threshold based on
            the number of total matches between the input sample and a reference haploid
    """
    averages = comp_matches_hybrid.sum(axis=0) / comp_matches_hybrid.getnnz(axis=0)
    std_thresh: np.ndarray = get_std(np.array(averages)[0])
    X: sparse.csc_matrix = sparse.hstack(
        [
            comp_matches_hybrid[
                :, chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE
            ].multiply(sparse.csr_matrix(haps_freqs_array_norm[chunk, :]).transpose())
            for chunk in range(haps_freqs_array_norm.shape[0])
        ]
    ).tocsc()
    std_ = np.array([np.std(row.data) for row in X.transpose()])
    final_thresh = np.array(X.max(axis=0) - std_thresh * std_)[0]
    return np.array(
        [
            np.where(X.getcol(index).data >= final_thresh[index], 1, 0).sum()
            for index in range(X.shape[1])
        ]
    ).copy()


class CompositePanelMaskFilter:
    """
    Apply filters based on:
    - number of matches for each chip site in the composite ref panel
       (`haps_freqs_array_norm_dict`)
    - estimated number of haploids that should be taken for each chip site
       (`nc_thresh`)
    """

    def __init__(
        self,
        matches: sparse.csc_matrix,
        haps_freqs_array_norm: np.ndarray,
        nc_thresh: np.ndarray,
        CHUNK_SIZE: int,
        kept_matches: int = 50,
    ):
        self.matches_row = matches.tocsr()
        self.haps_freqs_array_norm = haps_freqs_array_norm
        self.nc_thresh = np.clip(nc_thresh, 1, kept_matches)
        self.CHUNK_SIZE = CHUNK_SIZE

    def _normalize_matches(self) -> sparse.csc_matrix:
        """Multiply matches by normalized haplotype frequencies"""
        return sparse.hstack(
            [
                self.matches_row[
                    :, chunk * self.CHUNK_SIZE : (chunk + 1) * self.CHUNK_SIZE
                ].multiply(sparse.csr_matrix(row).transpose())
                for chunk, row in enumerate(self.haps_freqs_array_norm)
            ]
        ).tocsc()

    def _best_matches(self) -> List[np.ndarray]:
        """Generate all best matches at all variants"""
        return [
            row.indices[np.argsort(row.data)[::-1][: self.nc_thresh[chip_index]]].copy()
            for chip_index, row in enumerate(self._normalize_matches().transpose())
        ]

    def _expand_matches(
        self, keep: np.ndarray, hap: int, counts: np.ndarray
    ) -> np.ndarray:
        """Get variants to keep for a haplotype"""
        row = self.matches_row[hap].indices
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
        best_matches = self._best_matches()
        indptr = np.append([0], np.cumsum([len(matches) for matches in best_matches]))
        matrix = sparse.csc_matrix(
            (
                np.full(indptr[-1], True, dtype=np.bool_),
                np.hstack(best_matches),
                indptr,
            ),
            shape=self.matches_row.shape,
        ).tocsr()
        del best_matches
        counts = np.zeros(self.matches_row.shape[0], dtype=int)
        indices = np.hstack(
            [
                self._expand_matches(matrix[hap].indices, hap, counts)
                for hap in np.where(matrix.getnnz(axis=1) > 0)[0]
            ]
        )
        indptr = np.append([0], np.cumsum(counts))
        return sparse.csr_matrix(
            (np.full(indptr[-1], True, dtype=np.bool_), indices, indptr),
            shape=self.matches_row.shape,
        ).transpose()

    def haplotype_id_lists(self) -> np.ndarray:
        return self.sparse_matrix().tolil().rows.copy()


def run_hmm(
    chip_sites_n: int,
    ordered_hap_indices: np.ndarray,
    distances_cm: np.ndarray,
    num_hid: int = 9000,
    est_ne: int = 1000000,
) -> sparse.csr_matrix:
    """
    Runs the forward and backward runs of the forward-backward algorithm,
        to get probabilities for imputation
    """
    nHaps = np.array([len(row) for row in ordered_hap_indices])
    pRecomb_arr: np.ndarray = pRecomb(distances_cm, num_hid=num_hid, ne=est_ne) / nHaps

    matrix_ = setFwdValues_SPARSE(
        chip_sites_n, ordered_hap_indices, pRecomb_arr, num_hid=num_hid
    )
    return setBwdValues_SPARSE(
        matrix_,
        chip_sites_n,
        ordered_hap_indices,
        pRecomb_arr,
        nHaps,
        num_hid=num_hid,
    )


def calculate_weights(
    target_hap: Tuple[str, int],
    chip_cM_coordinates: np.ndarray,
    npz_dir: Path,
    shape: Tuple[int],
    output_dir: Path,
    output_breaks: List[Tuple[int]],
    est_ne: int = 1000000,
) -> None:
    """
    Load pbwt matches from npz and calculate weights for imputation
    Processing as one chunk (CHUNK_SIZE = chip_sites_n)
    """
    comp_matches_hybrid: sparse.csc_matrix = load_sparse_comp_matches_hybrid_npz(
        *target_hap, npz_dir, shape
    )

    haps_freqs_array_norm: np.ndarray = calculate_haploid_frequencies_SPARSE(
        comp_matches_hybrid, shape[1]
    )
    nc_thresh: np.ndarray = calculate_haploid_count_threshold_SPARSE(
        comp_matches_hybrid, haps_freqs_array_norm, shape[1]
    )
    ordered_hap_indices = CompositePanelMaskFilter(
        comp_matches_hybrid, haps_freqs_array_norm, nc_thresh, shape[1]
    ).haplotype_id_lists()

    weight_matrix = run_hmm(
        shape[1],
        ordered_hap_indices,
        chip_cM_coordinates,
        num_hid=shape[0],
        est_ne=est_ne,
    )

    for start, stop in output_breaks:
        sparse.save_npz(
            output_dir.joinpath(str(start), f"{target_hap[0]}_{target_hap[1]}.npz"),
            weight_matrix[start:stop, :],
        )
