from typing import Iterable, List, Tuple
from pathlib import Path
from logging import Logger

import pandas as pd
import numpy as np
from numba import njit
from scipy import sparse
from tqdm import tqdm

from modules.utils import get_std


@njit
def _normalize_freqs(freqs: np.ndarray) -> np.ndarray:
    """
    Normalize the results linearly between 0.1 and 1, scaling down
    reference haplotypes that have less total matches with the test sample
    """
    rmin = freqs.min()
    return ((freqs - rmin) / (freqs.max() - rmin)) * 0.9 + 0.1


@njit
def _calculate_threshold(
    max_normed: np.ndarray, sd_normed: np.ndarray, std_length: np.ndarray
) -> np.ndarray:
    """Calculate minimum normalized length to retain a match"""
    return max_normed - sd_normed * std_length


def load_and_interpolate_genetic_map(
    genetic_map_path: str, chip_BPs: List[int]
) -> np.ndarray:
    """
    Loads the genetic map, linearly interpolates the cM coordinates for chip BP positions.

    Takes:
     - `genetic_map_path` - path to the genetic map in the "PLINK-format"
     - `chip_BPs` - list of BP positions (e.g. `792461`)

    Returns two lists:
        - `chip_BP_positions` - chip (input) BP positions
        - `chip_cM_coordinates` - corresponding cM coordinates for the chip positions,
             inferred from the genetic map
    """
    progress = tqdm(
        total=4,
        desc=" [map] Loading and interpolating genetic map",
        ncols=75,
        bar_format="{desc}:\t\t\t\t{percentage:3.0f}% in {elapsed}",
        colour="#808080",
    )
    genetic_map = pd.read_csv(
        genetic_map_path,
        sep=" ",
        comment="#",
        header=None,
        usecols=[2, 3],
        names=["cM", "pos"],
        index_col="pos",
    )
    progress.update(1)
    genetic_map = genetic_map.join(pd.DataFrame(chip_BPs).set_index(0), how="outer")
    progress.update(1)
    genetic_map["cM"] = genetic_map["cM"].interpolate("index").bfill()
    progress.update(1)
    genetic_map = genetic_map.reset_index().rename({"index": "pos"}, axis=1)
    genetic_map = genetic_map[genetic_map["pos"].isin(chip_BPs)].reset_index(drop=True)
    chip_cM_coordinates = genetic_map.cM.to_numpy()  # type: ignore
    progress.update(1)
    progress.close()

    assert chip_cM_coordinates.size == len(chip_BPs)
    return chip_cM_coordinates


class SparseCompositeMatchesNpz:
    """Class for loading and processing pbwt matches"""

    def __init__(
        self,
        sample_name: str,
        hap: int,
        npz_dir: Path,
        shape: Tuple[int],
        logger: Logger,
    ) -> None:
        matrix: sparse.csr_matrix = sparse.load_npz(
            npz_dir.joinpath(f"parallel_haploid_mat_{sample_name}_{hap}.npz")
        ).tocsr()
        if matrix.shape != shape:
            raise IndexError(
                f"Sparse matrix for {sample_name}_{hap} is the wrong shape. "
                f"Expected shape: {shape}. Actual shape: {matrix.shape}"
            )
        self.n_haps, self.n_var = matrix.shape
        self.starts = matrix.indices.copy()
        self.stops = (matrix.indices + matrix.data - 1).copy()

        # cache row indices
        self.row_indices = np.fromiter(
            [self._get_row_indices(row) for row in range(self.n_var)], dtype=np.ndarray
        )

        # handle variants with no pbwt matches
        missing = np.array(
            [idx for idx, row in enumerate(self.row_indices) if row.size == 0]
        )
        if missing.size >= 15:
            logger.warning(
                f"\nSample {sample_name} haplotype {hap} had no matches at "
                f"{missing.size} variants"
            )
        added_freq = np.zeros((self.n_haps, 1), dtype=np.int32)
        if missing[0] == 0:
            # work backwards from first variant with a match
            start = np.nonzero(np.diff(missing) > 1)[0][0] + 1
            self.starts[self.starts == start] = 0
            self.row_indices[:start] = self.row_indices[start]
            added_freq[
                np.searchsorted(matrix.indptr, self.row_indices[start], side="right")
                - 1
            ] = start
        else:
            start = 0
        # work forwards, extending matches forward
        for i in missing[start:]:
            self.stops[self.stops == i - 1] = i
            self.row_indices[i] = self.row_indices[i - 1]
            added_freq[
                np.searchsorted(matrix.indptr, self.row_indices[i - 1], side="right")
                - 1
            ] += 1

        # normalize data by haplotype frequencies with tmin = 0.1 and tmax = 1
        normed_freqs = _normalize_freqs(matrix.sum(axis=1).A + added_freq)
        self.normed: np.ndarray = matrix.multiply(normed_freqs).tocsr().data

        # calculate metrics for each variant
        avg_len = np.zeros((self.n_var,), dtype=np.float64)
        sd_normed = np.zeros((self.n_var,), dtype=np.float64)
        max_normed = np.zeros((self.n_var,), dtype=np.float64)
        for row in range(self.n_var):
            avg_len[row] = matrix.data[self.row_indices[row]].mean()
            row_norm = self.normed[self.row_indices[row]]
            sd_normed[row] = row_norm.std()
            max_normed[row] = row_norm.max()

        # calculate minimum threshold to keep a match for each variant
        self.threshold = _calculate_threshold(
            max_normed, sd_normed, get_std(avg_len, matrix.data.min(), self.n_var)
        )

        self.indptr = matrix.indptr

    def _get_row_indices(self, row: int) -> np.ndarray:
        """
        Get indices of nonzero values for variant.
        Indices point to starts, stops, and normed, not haps.
        """
        return ((self.starts <= row) * (self.stops >= row)).nonzero()[0]

    def get_row_matches(self, row: int) -> np.ndarray:
        """
        Find the matches that exceed the threshold in a given row
        and return in order of best to worst
        """
        match_indices = self.row_indices[row][
            self.normed[self.row_indices[row]] >= self.threshold[row]
        ]
        haps = np.searchsorted(self.indptr, match_indices, side="right") - 1
        return haps[np.argsort(self.normed[match_indices])[::-1]]

    def get_hap_matches(self, hap: int) -> Iterable[Tuple[int]]:
        """Return tuples with ends of hap matches"""
        return zip(
            self.starts[self.indptr[hap] : self.indptr[hap + 1]],
            self.stops[self.indptr[hap] : self.indptr[hap + 1]],
        )
