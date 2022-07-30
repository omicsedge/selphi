from typing import Dict, List, Literal
import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange

from data_utils import get_sample_index, remove_all_samples
from hmm_utils import setFwdValues, setBwdValues
from utils import (
    interpolate_parallel,
    interpolate_parallel_packed,
    BidiBurrowsWheelerLibrary,
    get_std,
    skip_duplicates,
    forced_open
)



def BiDiPBWT(
    combined_ref_panel_chip: np.ndarray,
    ref_panel_chip_array: np.ndarray,
    hap: Literal[0,1],
):
    """
    Call BiDiPBWT - the main data structure for imputation.

    Input:
     - `ref_panel_chip_array` - the reference panel without test samples with chip sites only
     - `combined_ref_panel_chip` - the same reference panel but with the test sample (+2 haploids) (chip sites only)
    """
    bidi_pbwt = BidiBurrowsWheelerLibrary(combined_ref_panel_chip.T.astype(np.int8), ref_panel_chip_array.shape[1]+hap)
    ppa_matrix = bidi_pbwt.getForward_Ppa()
    div_matrix = bidi_pbwt.getForward_Div()
    rev_ppa_matrix = bidi_pbwt.getBackward_Ppa()
    rev_div_matrix = bidi_pbwt.getBackward_Div()

    forward_pbwt_matches, forward_pbwt_hap_indices = bidi_pbwt.getForward_matches_indices()
    backward_pbwt_matches, backward_pbwt_hap_indices = bidi_pbwt.getBackward_matches_indices()

    num_chip_vars = ppa_matrix.shape[1]
    num_hid = ref_panel_chip_array.shape[1]

    BI = np.zeros((num_hid,num_chip_vars))
    BJ = np.zeros((num_hid,num_chip_vars))

    print("Build BI BJ: main imputation DS")
    for chip_var in trange(0,num_chip_vars):
        forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
        forward_pbwt_index=ppa_matrix[:,chip_var]
        backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
        backward_pbwt_index=np.flip(rev_ppa_matrix, axis=1)[:,chip_var]
        BI[:,chip_var] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
        BJ[:,chip_var] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]


    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in trange(0,num_chip_vars):

        x = np.unique(BI[:,chip_var])
        if len(x) == 1 and x[0] == 0:
            BI[:,chip_var] = 1
            BJ[:,chip_var] = 1


    return BI, BJ



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
    fl = 13 

    print("Creating initial composite panel")
    composite_ = np.zeros(matches.shape) # mask (only 0s and 1s)
    best_matches = {}
    for chip_index in trange(0, matches.shape[1]):
        best_matches[chip_index] = list(np.argsort(matches[:,chip_index])[::-1][:fl])
        for hap_index in best_matches[chip_index]:
            composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
            composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

    comp_matches_hybrid: np.ndarray = (composite_ * matches).astype(int)

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
    assert 0 <= tmin < tmax <= 1, "wtf"

    print("Calculating Haploid frequencies")
    hap_freq = {}
    for i in trange(0,matches.shape[1]):
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
        tmin = 0.1
        tmax = 1
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
    hybrid = (composite_ * matches).astype(int)
    for i in range(0,matches.shape[1]):
        averages.append(np.average(hybrid[np.flatnonzero(hybrid[:,i]),i]))

    std_thresh = list(get_std(np.array(averages)))

    # 2. 
    lengthos = [] # a list of np vectors 
    std_ = [] # a list of stds for ^^^ numpy vectors
    final_thresh = [] # a threshold that is used to calculate nc_thresh 
    nc_thresh: List[int] = [] # how many haploids should go into each chip variant
    print("Calculating haploid count thresholds for each chip variant")
    for i in trange(0,matches.shape[1]):
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
    composite_: np.ndarray,
    comp_matches_hybrid: np.ndarray,
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
    composite_ = np.zeros(matches.shape)
    best_matches = {}
    for chip_index in trange(0, matches.shape[1]):
        chunk_index = int(chip_index//CHUNK_SIZE)
        xooi = matches[:,chip_index] * haps_freqs_array_norm_dict[chunk_index]
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

    Also, calculates a useless dict `length_matches_normalized` which value doesn't affect downstream calculations
    """
    ordered_matches_test__: Dict[int, List] = {} # a list of ids of haploids taken for each chip variant
    comp_to_plot = np.zeros(composite_.shape)
    for i in trange(matches.shape[1]):
        xooi = matches[:,i]
        sorting_key = list(xooi.argsort()[::-1])
    #     sorting_key = list(np.argsort(haps_freqs_array_norm)[::-1])

        uniqes = list(np.where(composite_[:,i] == 1)[0])
        ordered_matches_test__[i] = sorted(uniqes,key=sorting_key.index)
        comp_to_plot[ordered_matches_test__[i],i] = 1
        if len(ordered_matches_test__[i]) == 0:
            ordered_matches_test__[i] = list(np.where(matches[:,i] != 0)[0])
    
    # useless block: calculates a dictionary which value doesn't affect calculations
    length_matches = {}
    length_matches_normalized: Dict[int, List] = {}
    for i in range(0,matches.shape[1]):
        for j in ordered_matches_test__[i]:
            length_matches.setdefault(i, []).append(int(matches[j, i]))
        rmin = min(length_matches[i])
        rmax = max(length_matches[i])
        if rmax == rmin:
            rmin = rmin-1
        tmin = 0
        tmax = 0
        length_matches_normalized[i] = list(np.round((((np.array(length_matches[i]) - rmin)/(rmax - rmin)) * (tmax - tmin) + tmin),5))
    
    return ordered_matches_test__, length_matches_normalized


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


