from typing import Iterator, List, Tuple
from pathlib import Path
from datetime import datetime
from scipy import sparse
import numpy as np
from tqdm import tqdm

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
    data: sparse.csc_matrix = comp_matches_hybrid.copy()
    data.eliminate_zeros()
    freqs: np.ndarray = np.vstack(
        [
            data[:, start : start + CHUNK_SIZE].getnnz(axis=1)
            for start in range(0, data.shape[1], CHUNK_SIZE)
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
    )


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
        self.matches = matches
        self.matches_row = matches.tocsr()
        self.haps_freqs_array_norm = haps_freqs_array_norm
        self.nc_thresh = np.clip(nc_thresh, 0, kept_matches)
        self.CHUNK_SIZE = CHUNK_SIZE

    @staticmethod
    def stack(arrays: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrays, axis=0)

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

    def _best_matches(self) -> Iterator[np.ndarray]:
        """Generate all best matches at all variants"""
        return (
            row.indices[np.argsort(row.data)[::-1][: self.nc_thresh[chip_index]]]
            for chip_index, row in enumerate(self._normalize_matches().transpose())
        )

    def _get_variant_coordinates(
        self, chip_index: int, best_matches: np.ndarray
    ) -> np.ndarray:
        """Get coordinates to keep for a variant"""
        subset: sparse.coo_matrix = self.matches_row[best_matches, :].tocoo()
        cols_inds = subset.col
        stepsize = 1
        breakpoints: np.ndarray = np.where(np.diff(cols_inds) != stepsize)[0]
        intervals: np.ndarray = np.row_stack(
            (
                subset.row[np.append([0], breakpoints + 1)],
                cols_inds[np.append([0], breakpoints + 1)],
                cols_inds[np.append(breakpoints, -1)],
            )
        )
        selected: np.ndarray = intervals[
            :,
            np.intersect1d(
                np.where(intervals[1, :] < chip_index + 1),
                np.where(intervals[2, :] >= chip_index),
            ),
        ]
        return np.vstack(
            (
                np.repeat(
                    best_matches[selected[0, :]], selected[2, :] - selected[1, :] + 1
                ),
                np.concatenate([np.arange(row[1], row[2] + 1) for row in selected.T]),
            )
        ).T

    def coordinates_array(self) -> np.ndarray:
        """Generate 2d array of coordinates to keep"""
        return self.stack(
            [
                self._get_variant_coordinates(*match)
                for match in tqdm(enumerate(self._best_matches()))
            ]
        )

    def sparse_matrix(self) -> sparse.csc_matrix:
        """Generate sparse matrix filter"""
        print("Getting coordinates to keep: ", datetime.now())
        coordinates = self.coordinates_array()
        print("Creating sparse matrix from coordinates: ", datetime.now())
        return sparse.coo_matrix(
            (
                np.full_like(coordinates[:, 0], True, dtype=np.bool_),
                ([coordinates[:, 0], coordinates[:, 1]]),
            ),
            shape=self.matches.shape,
            dtype=np.bool_,
        ).tocsc()


def form_haploid_ids_lists_NEW(composite_: sparse.csc_matrix) -> np.ndarray:
    return np.array([row.indices for row in composite_.transpose()], dtype=object)


def run_hmm(
    chip_sites_n: int,
    ordered_hap_indices: np.ndarray,
    distances_cm: np.ndarray,
    num_hid: int = 9000,
) -> sparse.csr_matrix:
    """
    Runs the forward and backward runs of the forward-backward algorithm,
        to get probabilities for imputation
    """
    nHaps = np.array([np.unique(row).size for row in ordered_hap_indices])
    pRecomb_arr: np.ndarray = pRecomb(distances_cm, num_hid=num_hid) / nHaps

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
    target_hap: Tuple[str, int], chip_cM_coordinates: np.ndarray, npz_dir: Path
) -> sparse.csr_matrix:
    """
    Load pbwt matches from npz and calculate weights for imputation
    Processing as one chunk (CHUNK_SIZE = chip_sites_n)
    """
    comp_matches_hybrid: sparse.csc_matrix = load_sparse_comp_matches_hybrid_npz(
        *target_hap, npz_dir, fl=25
    )
    refpan_haps_n = comp_matches_hybrid.shape[0]
    chip_sites_n = comp_matches_hybrid.shape[1]

    haps_freqs_array_norm: np.ndarray = calculate_haploid_frequencies_SPARSE(
        comp_matches_hybrid, chip_sites_n
    )
    nc_thresh: np.ndarray = calculate_haploid_count_threshold_SPARSE(
        comp_matches_hybrid, haps_freqs_array_norm, chip_sites_n
    )
    composite_: sparse._csc.csc_matrix = CompositePanelMaskFilter(
        comp_matches_hybrid, haps_freqs_array_norm, nc_thresh, chip_sites_n
    ).sparse_matrix()

    del haps_freqs_array_norm
    del nc_thresh

    ordered_hap_indices = form_haploid_ids_lists_NEW(composite_)
    del composite_

    return run_hmm(
        chip_sites_n,
        ordered_hap_indices,
        chip_cM_coordinates,
        num_hid=refpan_haps_n,
    )
