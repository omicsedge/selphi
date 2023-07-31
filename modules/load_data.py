from typing import List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import sparse
from numba import njit
from tqdm import tqdm

######################################
#                                    #
#            GLOBAL DATA             #
#                                    #
######################################


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
        - `chip_cM_coordinates` - corresponding cM coordinates for the chip positions, inferred from the genetic map
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
    genetic_map["cM"] = genetic_map["cM"].interpolate("index").fillna(method="bfill")
    progress.update(1)
    genetic_map = genetic_map.reset_index().rename({"index": "pos"}, axis=1)
    genetic_map = genetic_map[genetic_map["pos"].isin(chip_BPs)].reset_index(drop=True)
    chip_cM_coordinates = genetic_map.cM.to_numpy()  # type: ignore
    progress.update(1)
    progress.close()

    assert chip_cM_coordinates.size == len(chip_BPs)
    return chip_cM_coordinates


@njit
def _expand_match(row: Tuple[int]) -> np.ndarray:
    return np.vstack(
        (np.arange(row[0], sum(row)), np.full(row[1], row[1]))
    ).T

def load_sparse_comp_matches_hybrid_npz(
    sample_name: str, hap: int, npz_dir: Path, shape: Tuple[int], fl: int = 25
) -> sparse.csc_matrix:
    npz_path = npz_dir.joinpath(f"parallel_haploid_mat_{sample_name}_{hap}.npz")
    x: sparse.csr_matrix = sparse.load_npz(npz_path).tocsr()
    if x.shape != shape:
        raise IndexError(
            f"Sparse matrix for {sample_name}_{hap} is the wrong shape. "
            f"Expected shape: {shape}. Actual shape: {x.shape}"
        )

    indptr = np.append([0], np.cumsum(x.sum(axis=1)))

    expanded = np.vstack(
        [_expand_match(row) for row in zip(x.indices, x.data)]
    )
    x = sparse.csr_matrix(
        (expanded[:, 1], expanded[:, 0], indptr), shape=x.shape
    ).tocsc()
    del indptr
    del expanded

    x_lil: sparse.lil_matrix = x.tolil()

    # handle variants with no matches
    missing: np.ndarray = np.where(x.getnnz(axis=0) == 0)[0]
    assert missing.size < 15

    if missing[0] == 0:
        start = np.where(np.diff(missing) > 1)[0][0] + 1
        # work backwards from first variant with a match
        for i in missing[:start][::-1]:
            x_lil[x[:, i + 1].indices, i] = x[:, i + 1].data + 1
            x = x_lil.tocsc()
    else:
        start = 0

    # work forwards, extending matches forward
    for i in missing[start:]:
        x_lil[x[:, i - 1].indices, i] = x[:, i - 1].data + 1
        x = x_lil.tocsc()
    del x
    return x_lil.tocsc()
