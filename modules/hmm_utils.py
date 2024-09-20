from typing import List, Tuple
from pathlib import Path

from scipy import sparse
import numpy as np
from numba import njit
from zstd import compress, uncompress


@njit
def _calculate_pRecomb(
    distances_cm: np.ndarray, nHaps: np.ndarray, num_hid: int = 9000, ne: int = 1000000
) -> Tuple[np.ndarray]:
    """return pRecomb between obs and obs - 1, pRecomb between obs and obs + 1"""
    min_value = 0.0000001
    min_recomb = [1 - np.exp((min_value * -0.04 * ne) / num_hid)]
    dm_array = np.diff(distances_cm)
    dm_array[dm_array == 0] = min_value
    recomb_array = 1 - np.exp((dm_array * -0.04 * ne) / num_hid)
    return (
        np.append(min_recomb, recomb_array) / nHaps,
        np.append(recomb_array, min_recomb) / nHaps,
    )


class HMM:
    """
    Class to compute haplotype weights for imputation
    using forward-backward Hidden Markov Model
    """

    def __init__(
        self,
        ordered_matches: np.ndarray,
        distances_cm: np.ndarray,
        output_dir: Path,
        target_hap: Tuple[str, int],
        output_breaks: List[Tuple[int]],
        shape: Tuple[int],
        est_ne: int = 1000000,
        pErr: float = 0.0001,
    ):
        self.ordered_matches = ordered_matches
        self.n_hid, self.n_sites = shape
        self.nHaps = np.array([len(row) for row in ordered_matches])
        self.f_pRecomb, self.r_pRecomb = _calculate_pRecomb(
            distances_cm, self.nHaps, num_hid=self.n_hid, ne=est_ne
        )
        self.pErr = pErr
        self.pNoErr = 1 - pErr

        self.basename = f"{target_hap[0]}_{target_hap[1]}.npz"
        self.output_dir = output_dir
        self.output_breaks = output_breaks

        # reduce to only matching haplotypes
        self.matches = np.unique(np.hstack(ordered_matches))
        trans_ = np.zeros(self.n_hid, dtype=np.int32)
        trans_[self.matches] = np.arange(self.matches.size)

        self.dense_matches = np.zeros((self.n_sites, self.nHaps.max()), dtype=np.int32)
        for c in range(self.n_sites):
            self.dense_matches[c, : self.nHaps[c]] = trans_[ordered_matches[c]]

    def _calculate_row(
        self, last_row: np.ndarray, row_matches: np.ndarray, p_recomb: float
    ) -> np.ndarray:
        "Compute new row of alpha/beta values from existing row"
        em = np.full_like(last_row, self.pErr)
        em[row_matches] = self.pNoErr
        last_row[row_matches] += p_recomb
        return em * last_row

    def _chunk_fwd_values(
        self, init_alpha: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        """
        Compute forward values for chunk of matrix and return first row of next chunk.
        Start and stop are indices of weight matrix and are chip index + 1.
        For the first chunk, init_alpha is 1 / nHaps for matches at first variant.
        For all other chunks, init_alpha was calculated with start - 1 as chip_idx.
        """
        is_last = stop == self.output_breaks[-1][-1]
        fwd_block = np.zeros((stop - start, self.matches.size), dtype=np.float64)
        fwd_block[0, :] = init_alpha.copy()

        chunk_idx = 0
        for chip_idx in range(start, stop - 1 - int(is_last)):
            fwd_block[chunk_idx + 1, :] = self._calculate_row(
                fwd_block[chunk_idx, :].copy(),
                self.dense_matches[chip_idx, : self.nHaps[chip_idx]],
                self.f_pRecomb[chip_idx],
            )
            chunk_idx += 1

        if is_last:
            fwd_block[-1, self.dense_matches[-1, : self.nHaps[-1]]] = 1 / self.nHaps[-1]

        init_alpha[:] = fwd_block[-1, :].copy()
        return fwd_block

    def _chunk_bwd_values(
        self, init_beta: np.ndarray, start: int, stop: int
    ) -> np.ndarray:
        """
        Compute backward values for chunk of matrix and return last row of next chunk.
        Start and stop are indices of weight matrix and are chip index + 1.
        For the last chunk, init_beta is 1 / nHaps in last row.
        For all other chunks, init_beta was calculated with stop - 2 as chip_idx.
        """
        is_first = start == 0
        bwd_block = np.zeros((stop - start, self.matches.size), dtype=np.float64)
        bwd_block[-1, :] = init_beta.copy()

        chunk_idx = stop - start - 1
        for chip_idx in range(stop - 3, start - 2 + int(is_first), -1):
            bwd_block[chunk_idx - 1, :] = self._calculate_row(
                bwd_block[chunk_idx, :].copy(),
                self.dense_matches[chip_idx, : self.nHaps[chip_idx]],
                self.r_pRecomb[chip_idx],
            )
            chunk_idx -= 1

        if is_first:
            bwd_block[0, :] = 1 / self.nHaps[0]

        init_beta[:] = bwd_block[0, :].copy()
        return bwd_block

    def run(self) -> None:
        """
        Runs the forward and backward runs of the forward-backward algorithm
        """
        # calculate forward blocks
        alpha = np.zeros((self.matches.size,), dtype=np.float64)
        alpha[self.dense_matches[0, : self.nHaps[0]]] = 1 / self.nHaps[0]
        fwd_blocks = [
            compress(self._chunk_fwd_values(alpha, *chunk).tobytes())
            for chunk in self.output_breaks
        ]

        # move backward to calculate and save final weights
        beta = np.full_like(alpha, 1 / self.nHaps[-1])
        for (start, stop), fwd_block in reversed(
            [*zip(self.output_breaks, fwd_blocks)]
        ):
            bwd_block = self._chunk_bwd_values(beta, start, stop)
            weights = bwd_block * np.frombuffer(
                uncompress(fwd_block), dtype=np.float64
            ).reshape(bwd_block.shape)
            if start == 0:
                weights[0, self.dense_matches[0, : self.nHaps[0]]] = 1 / self.nHaps[0]
                weights[1, :] = weights[0, :]
            if stop == self.output_breaks[-1][-1]:
                weights[-1, self.dense_matches[-1, : self.nHaps[-1]]] = (
                    1 / self.nHaps[-1]
                )
                weights[-2, :] = weights[-1, :]

            weights = weights / weights.sum(axis=1, keepdims=True)
            weights[weights < 1 / (self.matches.size + 1)] = 0

            # create sparse matrix of entire reference panel
            sparse_weights = sparse.lil_matrix(
                (weights.shape[0], self.n_hid), dtype=np.float64
            )
            sparse_weights[:, self.matches] = weights
            # save to disk for interpolation
            sparse.save_npz(
                self.output_dir.joinpath(str(start), self.basename),
                sparse_weights.tocsr(),
            )
