from typing import Dict, List, Literal
import time

import numpy as np
from numba.typed import Dict as nbDict, List as nbList


# from modules.data_utils import get_sample_index, remove_all_samples
from modules.hmm_utils import setFwdValues, setBwdValues
from modules.utils import (
    impute_interpolate,
    get_std,
    BInBJ,
)



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
    BI, BJ = BInBJ(combined_ref_panel_chip.T.astype(np.int8), num_hid+hap, num_hid)
    return BI, BJ



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
    if not (0 <= tmin < tmax <= 1):
        raise ValueError("tmin and tmax parameters must be betwee 0 and 1, and tmax has to be greater than tmin")

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
    Runs the forward and backward runs of the forward-backward algorithm,
        then imputes using the resulting probabilities
    """

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

