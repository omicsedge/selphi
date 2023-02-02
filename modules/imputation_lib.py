from typing import Dict, List, Literal
import time

import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange
from numba import njit
import numba as nb
from numba.core import types
from numba.typed import Dict as nbDict, List as nbList


# from modules.data_utils import get_sample_index, remove_all_samples
from modules.hmm_utils import setFwdValues, setBwdValues
from modules.utils import (
    interpolate_parallel,
    impute_interpolate,
    interpolate_parallel_packed_prev,
    interpolate_parallel_packed_old,
    BidiBurrowsWheelerLibrary,
    get_std,
    BInBJ,
    BiDiPBWT as BiDiPBWT_utils,
    BInBJ_onelib,
)
# from modules.kuklog import kuklog_timestamp
# from modules.devutils import var_dump # separate line so i can comment/uncomment var_dump lines conveniently
# from modules.devutils import var_load # separate line so i can comment/uncomment var_load lines conveniently






# @njit
# def BInBJ(
#     num_hid,
#     num_chip_vars,
#     forward_pbwt_matches,
#     ppa_matrix,
#     backward_pbwt_matches,
#     rev_ppa_matrix
# ):
#     BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
#     BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

#     for chip_var in range(num_chip_vars):
#         forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
#         forward_pbwt_index=ppa_matrix[:,chip_var]
#         backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
#         backward_pbwt_index=rev_ppa_matrix[:,chip_var]
#         BI[:,chip_var] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
#         BJ[:,chip_var] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]

#     return BI, BJ




def BiDiPBWT(
    combined_ref_panel_chip: np.ndarray,
    num_hid: int,
    hap: Literal[0,1],
    kuklog_timestamp_func = lambda s: None,
):
    """
    Call BiDiPBWT - the main data structure for imputation.

    Input:
     - `ref_panel_chip_array` - the reference panel without test samples with chip sites only
     - `combined_ref_panel_chip` - the same reference panel but with the test sample (+2 haploids) (chip sites only)
    """
    # print("Building BI and BJ: main BiDiPBWT data structures...")
    start = time.time()
    BI_TEST, BJ_TEST = BInBJ(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap, num_hid)
    # BI_TEST, BJ_TEST = BiDiPBWT_utils(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap, num_hid)
    # print(f"{(time.time() - start) :8.3f} seconds")
    return BI_TEST, BJ_TEST


    # ppa_matrix_TEST, forward_pbwt_matches_TEST, rev_ppa_matrix_TEST, backward_pbwt_matches_TEST = BInBJ(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap, num_hid)
    
    
    # bidi_pbwt = BidiBurrowsWheelerLibrary(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap)
    # kuklog_timestamp_func(f"1 - instantiated library with the data")
    # ppa_matrix = bidi_pbwt.getForward_Ppa()
    # # div_matrix = bidi_pbwt.getForward_Div()
    # rev_ppa_matrix = bidi_pbwt.getBackward_Ppa()
    # # rev_div_matrix = bidi_pbwt.getBackward_Div()
    # kuklog_timestamp_func(f"2 - got forward and backward PPA and DEV matrices")

    # forward_pbwt_matches, forward_pbwt_hap_indices = bidi_pbwt.getForward_matches_indices()
    # backward_pbwt_matches, backward_pbwt_hap_indices = bidi_pbwt.getBackward_matches_indices()
    # kuklog_timestamp_func(f"3 - calculated matches_indices")

    # num_chip_vars = ppa_matrix.shape[1]
    # # num_hid = ref_panel_chip_array.shape[1]

    # # print("Building BI and BJ: main BiDiPBWT data structures")

    # BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    # BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    # print("Build BI BJ: main imputation DS")
    # # forward_pbwt_index = ppa_matrix.argsort(axis=0)
    # # backward_pbwt_index = np.flip(rev_ppa_matrix, axis=1).argsort(axis=0)
    # for chip_var in trange(0,num_chip_vars):
    #     forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
    #     forward_pbwt_index=ppa_matrix[:,chip_var]
    #     backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
    #     backward_pbwt_index=np.flip(rev_ppa_matrix, axis=1)[:,chip_var]
    #     BI[:,chip_var] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
    #     BJ[:,chip_var] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]
    
    # # BI, BJ = BInBJ(
    # #     num_hid,
    # #     num_chip_vars,
    # #     forward_pbwt_matches,
    # #     ppa_matrix,
    # #     backward_pbwt_matches,
    # #     np.flip(rev_ppa_matrix, axis=1)
    # # )
    # kuklog_timestamp_func(f"4 - calculated BI & BJ")

    # """
    # Naive pathetic attempt to vectorize the calculation of BI and BJ 
    # """
    # # BI_new = np.zeros((num_hid,num_chip_vars))
    # # BJ_new = np.zeros((num_hid,num_chip_vars))
    # # forward_pbwt_index = ppa_matrix.argsort(axis=0)
    # # backward_pbwt_index = np.flip(rev_ppa_matrix, axis=1).argsort(axis=0)
    # # for chip_var in range(0,num_chip_vars):
    # #     BI_new[:,chip_var] = forward_pbwt_matches[:,chip_var][forward_pbwt_index[:, chip_var]][:num_hid]
    # #     BJ_new[:,chip_var] = backward_pbwt_matches[:,chip_var][backward_pbwt_index[:, chip_var]][:num_hid]

    # # ppa_matrix                     .argsort(axis=0)[:, chip_var]
    # # np.flip(rev_ppa_matrix, axis=1).argsort(axis=0)[:, chip_var]

    # # forward_pbwt_matches [ppa_matrix                     .argsort(axis=0)]
    # # backward_pbwt_matches[np.flip(rev_ppa_matrix, axis=1).argsort(axis=0)]

    # # BI = forward_pbwt_matches [forward_pbwt_index.argsort()][:num_hid]
    # # BJ = backward_pbwt_matches[backward_pbwt_index.argsort()][:num_hid]


    # """
    # Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
    #     then treat is as matching all the samples
    # """
    # for chip_var in trange(0,num_chip_vars):

    #     x = np.unique(BI[:,chip_var])
    #     if len(x) == 1 and x[0] == 0:
    #         BI[:,chip_var] = 1
    #         BJ[:,chip_var] = 1

    # kuklog_timestamp_func(f"5 - cleaned up BI & BJ")


    # # print(f"{(ppa_matrix == ppa_matrix_TEST).all()=}")
    # # print(f"{(forward_pbwt_matches == forward_pbwt_matches_TEST).all()=}")
    # # print(f"{(rev_ppa_matrix == rev_ppa_matrix_TEST).all()=}")
    # # print(f"{(backward_pbwt_matches == backward_pbwt_matches_TEST).all()=}")


    # print(f"{(BI == BI_TEST).all()=}")
    # print(f"{(BJ == BJ_TEST).all()=}")
    
    # print(f"")

    # return BI, BJ


def create_composite_ref_panel(
    BI: np.ndarray,
    BJ: np.ndarray,
    fl: int = 13,
):
    """
    Creates the initial composite panel from the results of the BiDiPBWT.
    Masks the matches array: take only `fl` haplotypes for each variant.
    E.g. if `fl == BI.shape[0]`, then this is identity function.

    Returns:
     - `matches` - combined output pair from BiDiPBWT
     - `composite_` - resulting mask that was used to filter the `matches`
     - `comp_matches_hybrid` - `matches` masked with `composite_`
    """
    matches: np.ndarray = BI + BJ -1

    # print("Creating initial composite panel")
    composite_ = np.zeros(matches.shape, dtype=np.bool_) # mask (only 0s and 1s)
    best_matches = {}
    for chip_index in range(0, matches.shape[1]):
        best_matches[chip_index] = list(np.argsort(matches[:,chip_index])[::-1][:fl])
        for hap_index in best_matches[chip_index]:
            composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
            composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

    comp_matches_hybrid: np.ndarray = (composite_ * matches).astype(matches.dtype.type)

    return matches, composite_, comp_matches_hybrid


# @njit
def create_composite_ref_panel_np(
    BI: np.ndarray,
    BJ: np.ndarray,
    fl: int = 13,
):
    """
    Creates the initial composite panel from the results of the BiDiPBWT.
    Masks the matches array: take only `fl` haplotypes for each variant.
    E.g. if `fl == BI.shape[0]`, then this is identity function.

    Returns:
     - `matches` - combined output pair from BiDiPBWT
     - `composite_` - resulting mask that was used to filter the `matches`
     - `comp_matches_hybrid` - `matches` masked with `composite_`
    """
    matches: np.ndarray = BI + BJ -1

    # print("Creating initial composite panel")
    composite_ = np.zeros(matches.shape, dtype=np.bool_) # mask (only 0s and 1s)
    best_matches = np.zeros((matches.shape[1], fl), dtype=np.int64)
    for chip_index in range(0, matches.shape[1]):
        best_matches[chip_index] = np.argsort(matches[:,chip_index])[::-1][:fl]
        for hap_index in best_matches[chip_index]:
            composite_[hap_index ,chip_index:(chip_index + BJ[hap_index, chip_index])] = 1
            composite_[hap_index ,(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

    comp_matches_hybrid: np.ndarray = (composite_ * matches).astype(matches.dtype.type)

    return matches, composite_, comp_matches_hybrid





@njit
def run_BiDiPBWT_and_cr8_composite_refpanel(
    combined_ref_panel_chip: np.ndarray,
    num_hid: int,
    hap: Literal[0,1],
    fl: int = 13,
    # kuklog_timestamp_func = lambda s: None,
):

    assert hap == 0 or hap == 1

    """ #1
    Call BiDiPBWT - the main data structure for imputation.

    Input:
     - `ref_panel_chip_array` - the reference panel without test samples with chip sites only
     - `combined_ref_panel_chip` - the same reference panel but with the test sample (+2 haploids) (chip sites only)
    """
    # print("Building BI and BJ: main BiDiPBWT data structures...")
    # start = time.time()
    BI, BJ = BInBJ(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap, num_hid)
    # print(f"{(time.time() - start) :8.3f} seconds")
    # print("%8.3f seconds" % (time.time() - start))



    """ #2
    Creates the initial composite panel from the results of the BiDiPBWT.
    Masks the matches array: take only `fl` haplotypes for each variant.
    E.g. if `fl == BI.shape[0]`, then this is identity function.

    Returns:
     - `matches` - combined output pair from BiDiPBWT
     - `composite_` - resulting mask that was used to filter the `matches`
     - `comp_matches_hybrid` - `matches` masked with `composite_`
    """
    matches: np.ndarray = BI + BJ -1

    # print("Creating initial composite panel")
    composite_ = np.zeros(matches.shape, dtype=np.bool_) # mask (only 0s and 1s)
    best_matches = {}
    for chip_index in range(0, matches.shape[1]):
        best_matches[chip_index] = np.argsort(matches[:,chip_index])[::-1][:fl]
        for hap_index in best_matches[chip_index]:
            composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
            composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

    comp_matches_hybrid: np.ndarray = (composite_ * matches).astype(matches.dtype.type)

    return BI, BJ, matches, composite_, comp_matches_hybrid





def calculate_haploid_frequencies(
    matches: np.ndarray,
    comp_matches_hybrid: np.ndarray,
    CHUNK_SIZE: int,
    tmin: float = 0.1,
    tmax: float = 1,
):
    """
    First, calculates what is the number of matches for each chip variant in the new composite ref panel
    
    Second, normalizes the results linearly between `tmin` and `tmax`,
        yielding a useful multiplicative coefficient which scale down ref haploids
            that have less total matches with the test sample
    """
    assert 0 <= tmin < tmax <= 1, "wtf"

    # print("Calculating Haploid frequencies")
    hap_freq = {}
    for i in range(0,matches.shape[1]):
        chunk_index = int(i//CHUNK_SIZE)
        hap_freq.setdefault(chunk_index, {})
        
        indexoo = np.flatnonzero(comp_matches_hybrid[:,i])
        for indexo in indexoo:
            hap_freq[chunk_index][indexo] = hap_freq[chunk_index].get(indexo, 0) + 1

    haps_freqs_array_norm_dict: Dict[int, np.ndarray] = {}
    for chunk_index in list(hap_freq.keys()):
        
        haps_freqs_array = np.zeros(matches.shape[0])
        for key, item in hap_freq[chunk_index].items():
            haps_freqs_array[key] = item

        rmin = min(haps_freqs_array)
        rmax = max(haps_freqs_array)
        # tmin = 0.1
        # tmax = 1
        haps_freqs_array_norm = (((haps_freqs_array - rmin)/(rmax - rmin)) * (tmax - tmin) + tmin)
        haps_freqs_array_norm_dict[chunk_index] = haps_freqs_array_norm.copy()


    return haps_freqs_array_norm_dict



def calculate_haploid_count_threshold(
    matches: np.ndarray,
    composite_: np.ndarray,
    comp_matches_hybrid: np.ndarray,
    haps_freqs_array_norm_dict: Dict[int, np.ndarray],
    CHUNK_SIZE: int,
):
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

    # 1. STD_THRESH
    averages = []
    # comp_matches_hybrid = (composite_ * matches).astype(matches.dtype.type)
    for i in range(0,matches.shape[1]):
        averages.append(np.average(comp_matches_hybrid[np.flatnonzero(comp_matches_hybrid[:,i]),i]))

    std_thresh = list(get_std(np.array(averages)))

    # 2. 
    lengthos = [] # a list of np vectors 
    std_ = [] # a list of stds for ^^^ numpy vectors
    final_thresh = [] # a threshold that is used to calculate nc_thresh 
    nc_thresh: List[int] = [] # how many haploids should go into each chip variant
    # print("Calculating haploid count thresholds for each chip variant")
    for i in range(0,matches.shape[1]):
        chunk_index = int(i//CHUNK_SIZE)
        indexoo = np.flatnonzero(comp_matches_hybrid[:,i] * haps_freqs_array_norm_dict[chunk_index])
        X = comp_matches_hybrid[indexoo,i] * haps_freqs_array_norm_dict[chunk_index][indexoo]
        lengthos.append(X[np.argsort(X)[::-1]])
        std_.append(np.std(X[np.argsort(X)[::-1]]))
        final_thresh.append(max(lengthos[i]) - std_thresh[i]*std_[i])
        nc_thresh.append(
            len(np.flatnonzero(lengthos[i] >= final_thresh[i]))
        )

    # number of variants
    assert len(lengthos) == len(std_) == len(final_thresh) == len(nc_thresh) == matches.shape[1]

    return nc_thresh


def apply_filters_to_composite_panel_mask(
    matches: np.ndarray,
    # composite_: np.ndarray,
    # comp_matches_hybrid: np.ndarray,
    BI: np.ndarray,
    BJ: np.ndarray,

    # new filters
    haps_freqs_array_norm_dict: Dict[int, np.ndarray],
    nc_thresh: List[int],

    CHUNK_SIZE: int,
):
    """
    Apply filters based on:
     - number of matches for each chip site in the composite ref panel (`haps_freqs_array_norm_dict`)
     - estimated number of haploids that should be taken for each chip site (`nc_thresh`)
    """
    composite_ = np.zeros(matches.shape, dtype=np.bool_)
    best_matches = {}
    for chip_index in range(0, matches.shape[1]):
        chunk_index = int(chip_index//CHUNK_SIZE)
        xooi = matches[:,chip_index] * haps_freqs_array_norm_dict[chunk_index] # using comp_matches_hybrid instead of matches may improve accuracy
        best_matches[chip_index] = list(np.argsort(xooi)[::-1][:nc_thresh[chip_index]])
        for hap_index in best_matches[chip_index]:
            composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
            composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

    return composite_



def form_haploid_ids_lists(
    matches: np.ndarray,
    composite_: np.ndarray,
):
    """
    Forms a list of ids of haploids taken for each chip variant
        in a dict `ordered_matches_test__`
    """
    ordered_matches_test__: Dict[int, List[np.int64]] = {} # a list of ids of haploids taken for each chip variant
    comp_to_plot = np.zeros(composite_.shape, dtype=np.bool_)
    for i in range(matches.shape[1]):
        xooi = matches[:,i]
        sorting_key = list(xooi.argsort()[::-1])
    #     sorting_key = list(np.argsort(haps_freqs_array_norm)[::-1])

        uniqes = list(np.where(composite_[:,i] == 1)[0])
        ordered_matches_test__[i] = sorted(uniqes,key=sorting_key.index)
        comp_to_plot[ordered_matches_test__[i],i] = 1
        if len(ordered_matches_test__[i]) == 0:
            ordered_matches_test__[i] = list(np.where(matches[:,i] != 0)[0])

    return ordered_matches_test__


def run_hmm(
    original_indicies: List[int],
    full_full_ref_panel_vcfgz_path: str,
    chip_sites_n: int,
    ordered_hap_indices_list: List[Dict[int, List[np.int64]]],
    distances_cm: List[float],
    BI,
    BJ,
    chr_length: int,
    num_hid=9000,
    variable_range=False,
    start_range=2000,
    end_range=10000,
    start_imputation=0,
    end_imputation=-1,
    kuklog_timestamp_func=lambda s: None,
):
    """
    RUNs the forward and backward runs of the forward-backward algorithm
    """


    # ordered_hap_indices_list = ordered_matches_list
    # distances_cm = chip_cM_coordinates
    # imputing_samples_haploids_indices = imputing_samples_haploids_indices 
    # reference_haploids_indices = refpan_haploid_indices


    if end_imputation == -1:
        end_imputation = len(original_indicies) -1

    if not variable_range:

        # print("running forward and backward algorithms for all input haploids")
        weight_matrices: List[np.ndarray] = []

        for ordered_hap_indices in ordered_hap_indices_list:
            alpha = np.zeros((chip_sites_n+2, num_hid), dtype=np.float64)
            alpha = setFwdValues(
                alpha, chip_sites_n, ordered_hap_indices, distances_cm, num_hid=num_hid
            )
            kuklog_timestamp_func(f"forward values")
            post = setBwdValues(
                alpha, chip_sites_n, ordered_hap_indices, distances_cm, num_hid=num_hid
            )
            weight_matrices.append(post.T)
            kuklog_timestamp_func(f"backward values")

        # print("imputing/interpolating")
        # smpls2takeA = np.array([len(ordered_hap_indices[i]) for i in range(len(ordered_hap_indices))])
        # smpls2takeB = post.astype(bool).sum(axis=1)
        # for hap in [0,1]:
        #     var_dump(f"opts01-05_dev1.3.1/HG02330/{hap}/07_weight_matrix.pkl", weight_matrices[hap])
        #     # post_true = var_load(f"opts01-04_dev1/HG02330/{hap}/07_post.pkl")
        #     print(f"dumped weight_matrix for HG02330/{hap}")
        # exit(0)
        resultsoo_fb = impute_interpolate(
            weight_matrices,
            original_indicies,
            full_full_ref_panel_vcfgz_path,
            chr_length,
            start_imputation,
            end_imputation
        )
        kuklog_timestamp_func(f"imputation via interpolate")
    else:
        raise RuntimeError("this condition under run_hmm shouldn't have happened")
        return None
    return resultsoo_fb


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
    raise NotImplementedError("This function needs changing in accord to the updates of the f-b algo and interpolation imputation part since opt04 (stage #1 of optimizations)")

    alpha = setFwdValues(
        num_obs, ordered_hap_indices, distances_cm, variable_range=True, N_range=N_range
    )
    post = setBwdValues(
        alpha.copy(),
        num_obs,
        ordered_hap_indices,
        distances_cm,
        variable_range=True,
        N_range=N_range,
    )
    resultoo_fb = interpolate_parallel(post.T, original_indicies, ref_panel_full_array, start_imputation, end_imputation)
    return resultoo_fb


