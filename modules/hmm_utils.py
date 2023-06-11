from scipy import sparse
import numpy as np
from tqdm import trange
from numba import njit


@njit
def pRecomb(
    distances_cm: np.ndarray, num_hid: int = 9000, ne: int = 1000000
) -> np.ndarray:
    """
    pRecomb between obs and and obs-1
    """
    dm_array = np.append([0], np.diff(distances_cm))
    dm_array[np.where(dm_array == 0)] = 0.0000001
    return 1 - np.exp((dm_array * -0.04 * ne) / num_hid)


# @njit
def setFwdValues_SPARSE(
    num_obs: int,  # number of variants
    ordered_matches: np.ndarray,
    pRecomb_arr: np.ndarray,
    num_hid: int = 9000,  # number of ref haplotypes
) -> sparse.lil_matrix:
    """
    set forward values
    """
    # Create compressed forward matrix
    chunk_compression = 30
    matrix_ = sparse.lil_matrix(
        ((num_obs - 1) // chunk_compression, num_hid), dtype=np.float64
    )

    pErr = 0.0001
    pNoErr = 1 - pErr

    alpha = np.zeros((num_hid,), dtype=np.float64)
    alpha[ordered_matches[0]] = 1 / ordered_matches[0].size

    for m in trange(1, num_obs):
        if m % chunk_compression == 0:
            matrix_[(m // chunk_compression) - 1, :] = alpha

        alpha[ordered_matches[m]] += pRecomb_arr[m]

        em = np.full((num_hid,), pErr, dtype=np.float64)
        em[ordered_matches[m]] = pNoErr
        alpha = em * alpha

    return matrix_


def run_forward_block(
    init_probs: np.ndarray,
    num_obs: int,
    aci: int,
    chunk_compression: int,
    ordered_matches_matrix: np.ndarray,
    pRecomb_arr: np.ndarray,
    nHaps: np.ndarray,
    initial: bool = False,
) -> np.ndarray:
    if (num_obs - aci) > chunk_compression:
        if not initial:
            end = chunk_compression
        else:
            end = chunk_compression - 1
    else:
        end = num_obs - aci

    num_hid = init_probs.shape[1]
    alpha = np.zeros((end, num_hid), dtype=np.float64)
    alpha[0, :] = init_probs

    pErr = 0.0001
    pNoErr = 1 - pErr

    for m in range(end - 1):
        aci_ = m + aci

        shift = np.zeros((num_hid,), dtype=np.float64)
        shift[ordered_matches_matrix[aci_, : nHaps[aci_]]] = pRecomb_arr[aci_]

        em = np.full(  # maybe no need to cr8 an array, just use pErr
            (num_hid,), pErr, dtype=np.float64
        )
        em[ordered_matches_matrix[aci_, : nHaps[aci_]]] = pNoErr

        alpha[m + 1, :] = em * (alpha[m, :] + shift)

    return alpha


@njit
def run_backward_calc(
    aci: int,
    pRecomb_arr: np.ndarray,
    ordered_matches_matrix: np.ndarray,
    nHaps: np.ndarray,
    forward_decomp_block: np.ndarray,
    beta: np.ndarray,
) -> None:
    pErr = 0.0001
    pNoErr = 1 - pErr

    for fbi in range(forward_decomp_block.shape[0] - 2, -2, -1):
        aci_ = aci + fbi

        beta[ordered_matches_matrix[aci_, : nHaps[aci_]]] += pRecomb_arr[aci_]
        row = forward_decomp_block[fbi + 1, :] * beta

        em = np.full((forward_decomp_block.shape[1],), pErr, dtype=np.float64)
        em[ordered_matches_matrix[aci_ - 1, : nHaps[aci_ - 1]]] = pNoErr
        beta[:] = beta * em

        forward_decomp_block[fbi + 1, :] = row / sum(row)


def setBwdValues_SPARSE(
    matrix_: sparse.lil_matrix,
    num_obs: int,
    ordered_matches: np.ndarray,
    pRecomb_arr: np.ndarray,
    nHaps: np.ndarray,
    num_hid: int = 9000,
) -> sparse.csr_matrix:
    """
    set backward values
    """
    chunk_compression = 30
    # reduce to only matching haplotypes
    matches = np.unique(matrix_.tocoo().col)
    trans_ = np.zeros(num_hid, dtype=np.int32)
    trans_[matches] = np.arange(matches.size)

    # create final weight matrix (Sparse)
    weight_matrix = sparse.lil_matrix((num_obs + 2, num_hid), dtype=np.float64)
    weight_matrix[-1, ordered_matches[-1]] = 1 / ordered_matches[-1].size
    weight_matrix[0, ordered_matches[0]] = 1 / ordered_matches[0].size
    weight_matrix[1, ordered_matches[0]] = 1 / ordered_matches[0].size

    pErr = 0.0001
    pNoErr = 1 - pErr

    # Loop over the matrix_ from back to front
    # rci for reverse compressed index
    ordered_matches_matrix = np.zeros((num_obs, nHaps.max()), dtype=np.int32)
    for c in range(num_obs):
        ordered_matches_matrix[c, : nHaps[c]] = trans_[ordered_matches[c]]

    matrix__ = matrix_[:, matches]
    beta = np.full((matches.size,), 1 / nHaps[-1], dtype=np.float64)

    for rci in trange(matrix__.shape[0] - 1, -1, -1):
        aci = (rci + 1) * chunk_compression  # aci for actual_chip_index

        forward_decomp_block = run_forward_block(
            matrix__[rci, :].toarray(),
            num_obs,
            aci,
            chunk_compression,
            ordered_matches_matrix,
            pRecomb_arr,
            nHaps,
        )

        run_backward_calc(
            aci,
            pRecomb_arr,
            ordered_matches_matrix,
            nHaps,
            forward_decomp_block,
            beta,
        )

        for fbi in range(forward_decomp_block.shape[0] - 1, -1, -1):
            temp_array = forward_decomp_block[fbi, :].copy()
            temp_array[temp_array < 1 / num_hid] = 0
            weight_matrix[aci + fbi, matches] = temp_array.copy()

    # Run Last iteration
    forward_decomp_block = run_forward_block(
        weight_matrix[1, matches].toarray(),
        num_obs,
        1,
        chunk_compression,
        ordered_matches_matrix,
        pRecomb_arr,
        nHaps,
        initial=True,
    )

    for aci in range(1, chunk_compression - 2):
        beta[ordered_matches_matrix[aci, : nHaps[aci]]] += pRecomb_arr[aci]
        row = forward_decomp_block[aci + 1, :] * beta

        em = np.full((forward_decomp_block.shape[1],), pErr, dtype=np.float64)
        em[ordered_matches_matrix[aci - 1, : nHaps[aci - 1]]] = pNoErr
        beta[:] = beta * em

        forward_decomp_block[aci + 1, :] = row / sum(row)

        # put values in the sparse matrix on the fly
        temp_array = forward_decomp_block[aci + 1, :].copy()
        temp_array[temp_array < 1 / num_hid] = 0
        weight_matrix[aci + 1, matches] = temp_array.copy()

    return weight_matrix.tocsr()
