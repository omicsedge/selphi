
import numpy as np
from hmm_utils import setFwdValues, setBwdValues
from utils import (
    interpolate_parallel,
    interpolate_parallel_packed,
)

"""
Testing different variations of methods for filtering hap indicies before HMM
"""

def run_hmm(
    original_indicies,
    ref_panel_full_array,
    num_obs,
    ordered_hap_indices,
    distances_cm,
    BI,
    BJ,
    length_matches_normalized,
    chr_length,
    num_hid=9000,
    variable_range=False,
    start_range=2000,
    end_range=10000,
    start_imputation=0,
    end_imputation=-1,
):
    """
    RUNs the forward and backward runs of the forward-backward algorithm
    """

    if end_imputation == -1:
        end_imputation = len(original_indicies) -1

    if not variable_range:
        alpha = setFwdValues(
            num_obs, ordered_hap_indices, length_matches_normalized, distances_cm, num_hid=num_hid
        )
        post = setBwdValues(
            alpha.copy(), num_obs, ordered_hap_indices, length_matches_normalized, distances_cm, num_hid=num_hid
        )
        resultoo_fb: np.ndarray = interpolate_parallel_packed(post.T, original_indicies, ref_panel_full_array, chr_length, start_imputation, end_imputation)
    else:
        return None
    return resultoo_fb

def run_hmm_variable_N(
    original_indicies,
    ref_panel_full_array,
    num_obs,
    ordered_hap_indices,
    distances_cm,
    N_range,
    length_matches_normalized,
    start_imputation=0,
    end_imputation=-1
):
    """
    RUNs the forward and backward runs of the forward-backward algorithm
    """

    alpha = setFwdValues(
        num_obs, ordered_hap_indices, length_matches_normalized, distances_cm, variable_range=True, N_range=N_range
    )
    post = setBwdValues(
        alpha.copy(),
        num_obs,
        ordered_hap_indices,
        length_matches_normalized,
        distances_cm,
        variable_range=True,
        N_range=N_range,
    )
    resultoo_fb = interpolate_parallel(post.T, original_indicies, ref_panel_full_array, start_imputation, end_imputation)
    return resultoo_fb
