import math
from tqdm import trange

import numpy as np

def pRecomb(
    obs,
    distances_cm,
    variable_range=False,
    num_hid=9000,
    N_range=None,
    ne=1000000,
):
    """
    pRecomb betweem obs and and obs-1
    """
    if not variable_range:
        N = num_hid
    else:
        N = N_range[obs - 1]
    dm = distances_cm[obs] - distances_cm[obs - 1]
    if dm == 0:
        dm = 0.0000001
    tao_m = 1 - math.exp((-0.04 * dm * ne) / N)

    return tao_m


def setFwdValues(
    num_obs,
    ordered_matches,
    length_matches_normalized,
    distances_cm,
    variable_range=False,
    N_range=None,
    num_hid=9000,
):
    """
    set forward values
    """
    nRefHaps = num_hid
    lastSum = 1
    start = 0
    end = num_obs
    alpha = np.zeros((num_obs, nRefHaps))
    alpha[start, ordered_matches[0]] = 1 / len(ordered_matches[0])
    nHaps = [len(np.unique(ordered_matches[x])) for x in range(0, num_obs)]
    pErr = 0.0001
    pNoErr = 1 - pErr

    for m in trange(1, end):
        pRecomb_var = pRecomb(
            m,
            distances_cm,
            variable_range=variable_range,
            N_range=N_range,
            num_hid=num_hid,
        )
        shift = np.zeros((nRefHaps,))
        shift[ordered_matches[m]] = pRecomb_var / nHaps[m]
        scale = (1 - pRecomb_var) / lastSum
        em = np.zeros((nRefHaps,))
        em = em + pErr
        em[ordered_matches[m]] = -np.array(length_matches_normalized[m]) + pNoErr
        alpha[m, :] = em * (1 * alpha[m - 1, :] + shift)
        lastSum = sum(alpha[m, :])
    return alpha


def setBwdValues(
    alpha,
    num_obs,
    ordered_matches,
    length_matches_normalized,
    distances_cm,
    variable_range=False,
    N_range=None,
    num_hid=9000,
):
    """
    set backward values
    """
    nRefHaps = num_hid
    lastSum = 1
    end = num_obs
    beta = np.zeros((nRefHaps, 1))
    nHaps = [len(np.unique(ordered_matches[x])) for x in range(0, num_obs)]
    beta[:, 0] = 1 / nHaps[-1]
    pErr = 0.0001
    pNoErr = 1 - pErr
    for m in trange(end - 2, 0, -1):
        pRecomb_var = pRecomb(
            m,
            distances_cm,
            variable_range=variable_range,
            N_range=N_range,
            num_hid=num_hid,
        )
        shift = np.zeros((nRefHaps,))
        shift[ordered_matches[m]] = pRecomb_var / nHaps[m]
        scale = (1 - pRecomb_var) / lastSum
        stateSum = 0
        beta[:, 0] = 1 * beta[:, 0] + shift
        alpha[m, :] = alpha[m, :] * beta[:, 0]
        stateSum = sum(alpha[m, :])
        em = np.zeros((nRefHaps,))
        em = em + pErr
        em[ordered_matches[m-1]] = -np.array(length_matches_normalized[m-1]) + pNoErr
        beta[:, 0] = beta[:, 0] * em
        alpha[m, :] = alpha[m, :] / stateSum
    return alpha
