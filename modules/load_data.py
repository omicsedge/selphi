from typing import List
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
        desc="Loading and interpolating genetic map",
        ncols=75,
        bar_format="{desc}:\t\t{percentage:3.0f}% {bar}\t{elapsed}",
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
def _expand_match(row: np.ndarray) -> np.ndarray:
    return np.vstack(
        (
            np.full(row[2], row[0]),
            np.arange(row[1], row[1] + row[2]),
            np.full(row[2], row[2]),
        )
    ).T


@njit
def coo_to_array(rows: np.ndarray, columns: np.ndarray, data: np.ndarray) -> np.ndarray:
    return np.vstack((rows, columns, data)).T


def load_sparse_comp_matches_hybrid_npz(
    sample_name: str, hap: int, npz_dir: Path, fl: int = 25
) -> sparse.csc_matrix:
    npz_path = npz_dir.joinpath(f"parallel_haploid_mat_{sample_name}_{hap}.npz")
    x: sparse.csc_matrix = sparse.load_npz(npz_path).tocsc()
    keep = np.vstack(
        [
            np.column_stack(
                (
                    np.full_like(x[:, i].data[-fl:], i),
                    x[:, i].indices[np.argsort(x[:, i].data)[-fl:]],
                )
            )
            for i in range(x.shape[1])
        ]
    )
    mask = sparse.coo_matrix(
        (np.full_like(keep[:, 0], True, dtype=np.bool_), (keep[:, 1], keep[:, 0])),
        shape=x.shape,
        dtype=np.bool_,
    )
    x = x.multiply(mask).tocoo()

    expanded = np.vstack(
        [_expand_match(row) for row in coo_to_array(x.row, x.col, x.data)]
    )
    x = sparse.coo_matrix(
        (expanded[:, 2], (expanded[:, 0], expanded[:, 1])), shape=x.shape
    ).tocsc()
    x_lil: sparse.lil_matrix = x.tolil()

    missing_counter = 0
    for i in np.where(x.getnnz(axis=0) == 0)[0]:
        x_lil[x[:, i - 1].indices, i] = x[:, i - 1].data + 1
        x = x_lil.tocsc()
        missing_counter += 1

    assert missing_counter < 15
    return x_lil.tocsc()
