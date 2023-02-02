import math
from typing import List, Set, Tuple, TypeVar, Union, Iterable, Dict, Literal

import numpy as np


def pRecomb(
    obs: int,
    distances_cm: List[float],
    variable_range: bool = False,
    num_hid: int = 9000,
    N_range: None = None,
    ne: int = 1000000,
):
    """
    pRecomb betweem obs and and obs-1
    """
    # if not variable_range:
    #     N = num_hid
    # else:
    #     N = N_range[obs - 1]
    N = num_hid
    dm = distances_cm[obs] - distances_cm[obs - 1]
    if dm == 0:
        dm = 0.0000001
    tao_m = 1 - math.exp((-0.04 * dm * ne) / N)

    return tao_m


def setFwdValues(
    alpha: np.ndarray,
    num_obs: int,
    ordered_matches: Dict[int, List[np.int64]],
    distances_cm: List[float],
    variable_range: bool = False,
    N_range: None= None,
    num_hid: int = 9000,
):
    """
    set forward values
    """
    nRefHaps = num_hid
    lastSum = 1
    start = 0
    end = num_obs
    nHaps = [len(np.unique(ordered_matches[x])) for x in range(0, num_obs)]
    pErr = 0.0001
    pNoErr = 1 - pErr

    alpha[0, :] = 1 / alpha.shape[1]

    alpha[1, ordered_matches[0]] = 1 / len(ordered_matches[0])

    for m in range(start+1, end):
        pRecomb_var = pRecomb(
            m,
            distances_cm,
            variable_range=variable_range,
            N_range=N_range,
            num_hid=num_hid,
        )
        shift = np.zeros((nRefHaps,), dtype=np.float64)
        shift[ordered_matches[m]] = pRecomb_var / nHaps[m]
        scale = (1 - pRecomb_var) / lastSum
        em = np.zeros((nRefHaps,), dtype=np.float64) # maybe no need to cr8 an array, just use pErr
        em = em + pErr
        em[ordered_matches[m]] = pNoErr
        alpha[m+1, :] = em * (1 * alpha[m+1 - 1, :] + shift)
        lastSum = sum(alpha[m+1, :])

    alpha[-1, :] = 1 / alpha.shape[1]

    return alpha


def setBwdValues(
    alpha: np.ndarray,
    num_obs: int,
    ordered_matches: Dict[int, List[np.int64]],
    distances_cm: List[float],
    variable_range: bool = False,
    N_range: None= None,
    num_hid: int = 9000,
):
    """
    set backward values
    """
    nRefHaps = num_hid
    lastSum = 1
    end = num_obs
    beta = np.zeros((nRefHaps, 1), dtype=np.float64)
    nHaps = [len(np.unique(ordered_matches[x])) for x in range(0, num_obs)]
    beta[:, 0] = 1 / nHaps[-1]
    pErr = 0.0001
    pNoErr = 1 - pErr
    for m in range(end - 2, 0, -1):
        pRecomb_var = pRecomb(
            m,
            distances_cm,
            variable_range=variable_range,
            N_range=N_range,
            num_hid=num_hid,
        )
        shift = np.zeros((nRefHaps,), dtype=np.float64)
        shift[ordered_matches[m]] = pRecomb_var / nHaps[m]
        scale = (1 - pRecomb_var) / lastSum
        stateSum = 0
        beta[:, 0] = 1 * beta[:, 0] + shift
        alpha[m+1, :] = alpha[m+1, :] * beta[:, 0]
        stateSum = sum(alpha[m+1, :])
        em = np.zeros((nRefHaps,), dtype=np.float64)  # maybe no need to cr8 an array, just use pErr
        em = em + pErr
        em[ordered_matches[m-1]] = pNoErr
        beta[:, 0] = beta[:, 0] * em
        alpha[m+1, :] = alpha[m+1, :] / stateSum
    return alpha
