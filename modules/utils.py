from functools import lru_cache
from typing import List, Set, Tuple, TypeVar, Union, Iterable, Dict, Literal
from collections import OrderedDict
import time
import random


#from black import Result
import numpy as np
from tqdm import trange, tqdm
from numba import njit, int8, int16, int32, int64

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass # type: ignore


from modules.data_utils import get_sample_index
# from modules.IBDs_extractor import IBDs_extractor
# from modules.IBDs_extractor_restored_sep14 import IBDs_extractor
from modules.vcfgz_reader import vcfgz_reader
from modules.devutils import var_dump





def dbg(val, name:str):
    if isinstance(val, np.ndarray):
        print(f"{name} ({val.shape}) dtype={val.dtype.type}")
    elif isinstance(val, list):
        if len(val) > 4:
            print(f"{name}: list[{len(val)}] = [{val[0]}, {val[1]}, {val[2]}, ...{val[-1]}]")
        else:
            print(f"{name} = {val}")
    elif isinstance(val, tuple):
        if len(val) > 4:
            print(f"{name}: tuple[{len(val)}] = ({val[0]}, {val[1]}, {val[2]}, ...{val[-1]})")
        else:
            print(f"{name} = {val}")
    else:
        print(f"{name} = {val}")


def interpolate_parallel(
    weight_matrix,
    original_indicies,
    original_ref_panel,
    start,
    end,
):  
    from joblib import Parallel, delayed
    def parallel_interpolate(start, end):
        full_constructed_panel = np.zeros((original_ref_panel.shape[0], 2), dtype=np.float32)
        if end == len(original_indicies):
            end = len(original_indicies) -1
        for i in trange(start, end):
            temp_haps = []
            for j in range(0,weight_matrix.shape[0]):
                if weight_matrix[j, i+1] < 1/200 and weight_matrix[j, i] < 1/weight_matrix.shape[0]:
                    continue
                # if (weight_matrix[j, i],weight_matrix[j, i+1]) in temp_haps:
                #     continue
                # temp_haps.append((weight_matrix[j, i],weight_matrix[j, i+1]))
                probs = np.linspace(
                    weight_matrix[j, i],
                    weight_matrix[j, i+1],
                    num=original_indicies[i+1] - original_indicies[i],
                    endpoint=False
                )
                # print(probs)
                # One Dosage
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],0] = (
                    original_ref_panel[original_indicies[i]:original_indicies[i+1],j] * probs+ 
                    full_constructed_panel[original_indicies[i]:original_indicies[i+1],0]
                )
                # Zero Dosage
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],1] = (
                    np.logical_not(original_ref_panel[original_indicies[i]:original_indicies[i+1],j]) * probs+
                    full_constructed_panel[original_indicies[i]:original_indicies[i+1],1]
                )
        result = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)[original_indicies[start]:original_indicies[end]]
        return result
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n][0], lst[i:i + n][-1] + 1


    full_constructed_panel = np.zeros((original_ref_panel.shape[0], 2), dtype=np.float32)
    ##############################
    temp_haps = []
    i = 0
    for j in range(0,weight_matrix.shape[0]):
        if weight_matrix[j, i] < 1/weight_matrix.shape[0]:
            continue
        probs = np.linspace(
            1/weight_matrix.shape[0],
            weight_matrix[j, i],
            num=original_indicies[i],
            endpoint=False
        )
        # print(probs)
        # One Dosage
        full_constructed_panel[0:original_indicies[i],0] = (
            original_ref_panel[0:original_indicies[i],j] * probs+ 
            full_constructed_panel[0:original_indicies[i],0]
        )
        # Zero Dosage
        full_constructed_panel[0:original_indicies[i],1] = (
            np.logical_not(original_ref_panel[0:original_indicies[i],j]) * probs+
            full_constructed_panel[0:original_indicies[i],1]
        )
        
    result_1 = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)[0:original_indicies[i]]
    ###########################################
    temp_haps = []

    for j in range(0,weight_matrix.shape[0]):
        if weight_matrix[j, -1] < 1/weight_matrix.shape[0]:
            continue
        probs = np.linspace(
            weight_matrix[j, -1],
            1/weight_matrix.shape[0],
            num=original_ref_panel.shape[0] - original_indicies[-1],
            endpoint=False
        )
        # print(probs)
        # One Dosage
        full_constructed_panel[original_indicies[-1]:,0] = (
            original_ref_panel[original_indicies[-1]:,j] * probs+ 
            full_constructed_panel[original_indicies[-1]:,0]
        )
        # Zero Dosage
        full_constructed_panel[original_indicies[-1]:,1] = (
            np.logical_not(original_ref_panel[original_indicies[-1]:,j]) * probs+
            full_constructed_panel[original_indicies[-1]:,1]
        )
        
    result_2 = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)[original_indicies[-1]:]
    #####################
    X =  np.concatenate([result_1, *Parallel(n_jobs=8)(delayed(parallel_interpolate)(start, end) for (start, end) in list(chunks(range(0, weight_matrix.shape[1]), 5000))), result_2]) # type: ignore
    return X


def unpack(
    ref_panel_full_array_full_packed: np.ndarray,
    hap: int,
    orig_index_0: int,
    orig_index_1: int,
    chr_length: int
):
    nbits = 8

    index_0 = int(orig_index_0/nbits)
    if orig_index_1 == -1:
        index_1 = ref_panel_full_array_full_packed.shape[0]
    else:   
        index_1 = int(orig_index_1/nbits) + 1

    mapped_orig_index_0 = orig_index_0 - (index_0 * nbits)
    if orig_index_1 == -1:
        mapped_orig_index_1 = chr_length - (index_0 * nbits)

    else:
        mapped_orig_index_1 = orig_index_1 - (index_0 * nbits)
    return np.unpackbits(ref_panel_full_array_full_packed[index_0:index_1,hap])[mapped_orig_index_0:mapped_orig_index_1]


def interpolate_parallel_packed_old(
    weight_matrix: np.ndarray,
    original_indicies: List[int],
    original_ref_panel: np.ndarray,
    chr_length: int,
    start: int,
    end: int,
):
    # number of reference panel haploids
    assert original_ref_panel.shape[1] == weight_matrix.shape[0], "input shapes have to be consistent"
    # number of chip variants
    assert len(original_indicies) == weight_matrix.shape[1], "input shapes have to be consistent"

    from joblib import Parallel, delayed
    def interpolate_probs_betw_chip_sites_in_range(start: int, end: int):
        full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
        if end == len(original_indicies):
            end = len(original_indicies) -1
        for i in trange(start, end): # chip variant
            temp_haps = []
            for j in range(0,weight_matrix.shape[0]): # ref panel sample
                if weight_matrix[j, i+1] < 1/200 and weight_matrix[j, i] < 1/weight_matrix.shape[0]:
                    continue
                # if (weight_matrix[j, i],weight_matrix[j, i+1]) in temp_haps:
                #     continue
                # temp_haps.append((weight_matrix[j, i],weight_matrix[j, i+1]))
                probs = np.linspace(
                    weight_matrix[j, i],
                    weight_matrix[j, i+1],
                    num=original_indicies[i+1] - original_indicies[i],
                    endpoint=False
                )
                # print(probs)
                jth_haploid_segment = unpack(original_ref_panel, j, original_indicies[i], original_indicies[i+1], chr_length)
                # One Dosage
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],0] += jth_haploid_segment * probs
                # Zero Dosage
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],1] += np.logical_not(jth_haploid_segment) * probs
        one_dsg = full_constructed_panel[original_indicies[start]:original_indicies[end],0] 
        zero_dsg = full_constructed_panel[original_indicies[start]:original_indicies[end],1]
        result: np.ndarray = (
            full_constructed_panel[original_indicies[start]:original_indicies[end],0] 
                                > 
            full_constructed_panel[original_indicies[start]:original_indicies[end],1]
        ).astype(np.bool_)

        return result
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n][0], lst[i:i + n][-1] + 1


    full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
    ##############################
    temp_haps = []
    i = 0
    for j in range(0,weight_matrix.shape[0]):
        if weight_matrix[j, i] < 1/weight_matrix.shape[0]:
            continue
        probs = np.linspace(
            1/weight_matrix.shape[0],
            weight_matrix[j, i],
            num=original_indicies[i],
            endpoint=False
        )
        # print(probs)
        jth_haploid_segment = unpack(original_ref_panel, j,0, original_indicies[i], chr_length)
        # One Dosage
        full_constructed_panel[0:original_indicies[i],0] += jth_haploid_segment * probs
        # Zero Dosage
        full_constructed_panel[0:original_indicies[i],1] += np.logical_not(jth_haploid_segment) * probs
        
    result_1 = (
        full_constructed_panel[0:original_indicies[i],0] > full_constructed_panel[0:original_indicies[i],1]
    ).astype(np.bool_)
    ###########################################
    temp_haps = []

    for j in range(0,weight_matrix.shape[0]):
        if weight_matrix[j, -1] < 1/weight_matrix.shape[0]:
            continue
        probs = np.linspace(
            weight_matrix[j, -1],
            1/weight_matrix.shape[0],
            num=chr_length - original_indicies[-1],
            endpoint=False
        )
        # print(probs)
        jth_haploid_segment = unpack(original_ref_panel, j, original_indicies[-1], -1, chr_length)
        # One Dosage
        full_constructed_panel[original_indicies[-1]:,0] += jth_haploid_segment * probs
        # Zero Dosage
        full_constructed_panel[original_indicies[-1]:,1] += np.logical_not(jth_haploid_segment) * probs
        
    result_2 = (
        full_constructed_panel[original_indicies[-1]:,0] > full_constructed_panel[original_indicies[-1]:,1]
    ).astype(np.bool_)


    #####################
    X = np.concatenate([
        result_1,
        # in parallel
        *Parallel(n_jobs=8)(
            delayed(interpolate_probs_betw_chip_sites_in_range)(start, end) 
                for (start, end) in list(chunks(range(0, weight_matrix.shape[1]), 5000))
        ), # type: ignore
        # # sequentially
        # *[
        #     interpolate_probs_betw_chip_sites_in_range(start, end) for (start, end) in list(chunks(range(0, weight_matrix.shape[1]), 5000))
        # ],
        result_2,
    ])
    return X



def impute_interpolate(
    weight_matrices: List[np.ndarray],
    original_indicies: List[int],
    full_full_ref_panel_vcfgz_path: str,
    chr_length: int,
    start: int,
    end: int,
) -> np.ndarray:

    if len(weight_matrices) == 0:
        return np.array([])

    """
    *n* chip sites divide the full sequence into *n+1* segments,
     iff the first and the last full-sequence sites are not in the chip sites.
    *n* chip sites divide the full sequence into *n* segments,
     iff either first or the last full-sequence site is in the chip sites.
    *n* chip sites divide the full sequence into *n-1* segments,
     iff both first and the last full-sequence site are in the chip sites.
    """
    segment_points: List[int] = [0, *original_indicies, chr_length]

    sticks_out_from_left : bool = False
    sticks_out_from_right: bool = False

    # TODO: run interpolation/imputation for leftmost and rightmost parts only when these conditions hold
    if original_indicies[0] != 0: # ref panel has sites before the first chip site
        sticks_out_from_left = True
    if original_indicies[-1] != chr_length-1: # ref panel has sites after the last chip site
        sticks_out_from_right = True

    total_ref_haploids = weight_matrices[0].shape[0]
    total_input_haploids = len(weight_matrices)

    Xs = np.zeros((total_input_haploids, chr_length), dtype=np.int8)
    # target_full_array = np.zeros((chr_length, total_input_haploids), dtype=np.int8)

    refpanel = vcfgz_reader(full_full_ref_panel_vcfgz_path)
    # refpanel = IBDs_extractor(full_full_ref_panel_vcfgz_path)
    batch_size = 15_000 # smaller batch -> less RAM usage, but more iterations (more CPU time)


    # # for testing/debugging
    # probs_collections: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = []
    # for in_hap in range(len(weight_matrices)):
    #     # a dict element contains: probs_interpolations, interpolated_ref_haps
    #     probs_collection: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    #     probs_collections.append(probs_collection)



    def linspace_selected(weight_matrix: np.ndarray, chip_site_i: int, to_keep: List[int]):
        """
        interpolation of probabilities via linspace happens only for selected haploids.
        Other haploids are additively ignored (think: "are set to 0")
        """
        return np.linspace(
            weight_matrix[to_keep, chip_site_i],
            weight_matrix[to_keep, chip_site_i+1],
            num=segment_points[chip_site_i+1]-segment_points[chip_site_i],
            endpoint=False,
        )

    """
    Conditions on which to filter ref haploids before interpolation differ for different intervals.
    """

    def linspace_inner(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        """for intervals *between* the chip sites"""
        haploids_to_keep = np.where(~(
            (weight_matrix[:, chip_site_i+1] < 1/200) & (weight_matrix[:, chip_site_i] < 1/total_ref_haploids)
        ))[0]
        return [linspace_selected(weight_matrix, chip_site_i, haploids_to_keep), haploids_to_keep]

    def linspace_leftmost(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        """if it exists, for the interval starting from first ref panel site and up to the first chip site"""
        haploids_to_keep = np.where(~(
            weight_matrix[:, 1] < 1/total_ref_haploids
        ))[0]
        return [linspace_selected(weight_matrix, 0, haploids_to_keep), haploids_to_keep]

    def linspace_rightmost(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        """if it exists, for the interval starting from the last chip site and up to the first chip site"""
        haploids_to_keep = np.where(~(
            weight_matrix[:, -2] < 1/total_ref_haploids
        ))[0]
        return [linspace_selected(weight_matrix, -2, haploids_to_keep), haploids_to_keep]



    def iterate_batch(batch: Union[np.ndarray,None], batch_start_pos: int, batch_end_pos: int):
        """
        Reads new batch from the table of the vcf.gz file.
        Fills in the target_full_array with the true sequence for the test input haploids
        Outputs:
         - the refilled batch (new read into the same np.array)
         - start position on the full sequence
         - end position on the fill sequence
        """
        batch, lines_read_n = refpanel.readlines(lines_n=batch_size, refill=batch)
        # batch, lines_read_n = refpanel.readlines3(lines_n=batch_size, total_haploids_n=6404, refill=batch)

        batch_start_pos = batch_end_pos
        batch_end_pos   = batch_end_pos+lines_read_n
        # target_full_array[batch_start_pos:batch_end_pos, :] = batch[:lines_read_n, imputing_samples_haploids_indices]

        return batch, batch_start_pos, batch_end_pos

    def iterate_chips(
        chip_site_i: int, chip_site1_pos: int, chip_site2_pos: int,
        linspace_func = linspace_inner,
    ) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], int, int]:
        """
        Interpolates probabilities on the next interval between two chip sites
        Outputs:
         - the list of interpolated probabilities between the chip sites, for each input haploid
         - start position on the full sequence (position of the first chip site)
         - end position on the fill sequence (position of the second chip site)
        """
        if chip_site_i == len(segment_points)-1:
            linspace_func = linspace_rightmost

        probs_interpolations, interpolated_ref_haps = zip(*[linspace_func(weight_matrix, chip_site_i) for weight_matrix in weight_matrices])
        # # for testing/debugging
        # for in_hap in range(len(probs_collections)):
        #     probs_collections[in_hap][chip_site_i] = probs_interpolations[in_hap], interpolated_ref_haps[in_hap]

        return probs_interpolations, interpolated_ref_haps, segment_points[chip_site_i], segment_points[chip_site_i+1]

    def impute_interval(start: int, end: int, probs_interpolations: List[np.ndarray], interpolated_ref_haps: Tuple[np.ndarray,...], batch_interval: np.ndarray):
        for in_hap, probs_interpolation in enumerate(probs_interpolations):
            Xs[in_hap, start:end] = (
                (probs_interpolation * batch_interval[:,interpolated_ref_haps[in_hap]]).sum(axis=1) 
                                    > 
                (probs_interpolation * np.logical_not(batch_interval[:,interpolated_ref_haps[in_hap]])).sum(axis=1)
            )
        del batch_interval
        del probs_interpolations

    try:
        def impute_in_steps():
            """
            iterating through:
                - the intervals between chip sites, and
                - the batches of the reference panel table read from the vcfgz

            Those may overlap. Also:
                - a batch may contain several intervals between the chip sites.
                - an interval between the chip sites may contain several batches.

            Therefore batches and intervals between the chip sites are iterated independently,
                but in the same loop. The processing happens on every intersection.
            """

            iB = 0 # Batch iterator
            iC = 0 # interval between the Chip sites iterator

            sB = eB = 0 # start and end positions of the vcf Batch on the full sequence
            sC = eC = 0 # start and end positions of the Chip sites on the full sequence

            inter_s = inter_e = 0 # intersection start and end positions


            batch, sB, eB = iterate_batch(None, sB, eB)
            iB += 1
            probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC, linspace_func=linspace_leftmost)
            iC += 1

            debug_pos_counter = 0
            debug_timer = time.time()

            while (True):
                # 1. Cutting out the intersection of the (ref panel) batch interval and the chip sites interval
                ( inter_s, inter_e ) = ( max(sB, sC), min(eB, eC) )

                # 2. Working on the intersection: performing imputation on the interval
                if inter_e - inter_s > 0:
                    impute_interval(
                        start=inter_s, end=inter_e,
                        probs_interpolations = [p[inter_s-sC : inter_e-sC] for p in probs_interpolations],
                        interpolated_ref_haps = interpolated_ref_haps,
                        # batch_interval = batch[inter_s-sB : inter_e-sB, reference_haploids_indices][:, internal_haploid_order],
                        batch_interval = batch[inter_s-sB : inter_e-sB],
                    )

                # # [some logging while not in prod]
                # if inter_e - debug_pos_counter >= 120_000:
                #     debug_pos_counter = inter_e
                #     debug_timer_round = time.time()
                #     print(f"[{(debug_timer_round - debug_timer):8.3f} s] iC={iC:6}, iB={iB:5}, intersect ({inter_s:7}, {inter_e:7}), batch({sB:7}, {eB:7}), chips({sC:7}, {eC:7})")
                #     debug_timer = debug_timer_round

                # 4. Iteration: iterating samples chip sites interval or reading a new batch from the ref panel
                if eB == eC:
                    batch, sB, eB = iterate_batch(batch, sB, eB)
                    iB += 1
                    probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC)
                    iC += 1 
                else: # which interval ends sooner?
                    if eB < eC:
                        batch, sB, eB = iterate_batch(batch, sB, eB)
                        iB += 1
                    else:
                        probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC)
                        iC += 1

        impute_in_steps()


    except IndexError as e:
        """
        Reached the end of the full sequence
        """
        # print(f"imputation loop broke with: IndexError: {e}")
        pass


    refpanel.close()

    return Xs






def interpolate_parallel_packed_prev(
    weight_matrices: List[np.ndarray],
    original_indicies: List[int],
    full_full_ref_panel_vcfgz_path: str,
    chr_length: int,
    start: int,
    end: int,
    imputing_samples_haploids_indices: List[int],
    reference_haploids_indices: List[int],
    internal_haploid_order: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    # # number of reference panel haploids
    # assert original_ref_panel.shape[1] == weight_matrix.shape[0], "input shapes have to be consistent"
    # # number of chip variants
    # assert len(original_indicies) == weight_matrix.shape[1], "input shapes have to be consistent"

    if len(weight_matrices) == 0:
        return np.array([]), np.array([])

    """
    *n* chip sites divide the full sequence into *n+1* segments,
     iff the first and the last full-sequence sites are not in the chip sites.
    *n* chip sites divide the full sequence into *n* segments,
     iff either first or the last full-sequence site is in the chip sites.
    *n* chip sites divide the full sequence into *n-1* segments,
     iff both first and the last full-sequence site are in the chip sites.
    """
    segment_points: List[int] = [0, *original_indicies, chr_length]

    sticks_out_from_left : bool = False
    sticks_out_from_right: bool = False

    # TODO: run imputation/interpolation for leftmost and rightmost parts only with those conditions
    if original_indicies[0] != 0: # ref panel has variants before the first chip variant
        sticks_out_from_left = True
    if original_indicies[-1] != chr_length-1: # ref panel has variants after the last chip variant
        sticks_out_from_right = True

    total_ref_haploids = weight_matrices[0].shape[0]
    total_input_haploids = len(weight_matrices)

    Xs = np.zeros((total_input_haploids, chr_length), dtype=np.int8)
    target_full_array = np.zeros((chr_length, total_input_haploids), dtype=np.int8)

    # refpanel = vcfgz_reader(full_full_ref_panel_vcfgz_path)
    refpanel = IBDs_extractor(full_full_ref_panel_vcfgz_path)
    batch_size = 15_000




    # probs_collections: List[Dict[int, Tuple[np.ndarray, np.ndarray]]] = []
    # for in_hap in range(len(weight_matrices)):
    #     # a dict element contains: probs_interpolations, interpolated_ref_haps
    #     probs_collection: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    #     probs_collections.append(probs_collection)




    def linspace_selected(weight_matrix: np.ndarray, chip_site_i: int, to_keep: List[int]):
        return np.linspace(
            weight_matrix[to_keep, chip_site_i],
            weight_matrix[to_keep, chip_site_i+1],
            num=segment_points[chip_site_i+1]-segment_points[chip_site_i],
            endpoint=False,
        )

    def linspace_inner(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        to_keep = np.where(~(
            (weight_matrix[:, chip_site_i+1] < 1/200) & (weight_matrix[:, chip_site_i] < 1/total_ref_haploids)
        ))[0]

        return [linspace_selected(weight_matrix, chip_site_i, to_keep), to_keep]

    def linspace_leftmost(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        to_keep = np.where(~(
            weight_matrix[:, 1] < 1/total_ref_haploids
        ))[0]

        return [linspace_selected(weight_matrix, 0, to_keep), to_keep]

    def linspace_rightmost(weight_matrix: np.ndarray, chip_site_i: int) -> List[np.ndarray]:
        to_keep = np.where(~(
            weight_matrix[:, -2] < 1/total_ref_haploids
        ))[0]

        return [linspace_selected(weight_matrix, -2, to_keep), to_keep]


    def iterate_batch(batch: Union[np.ndarray,None], batch_start_pos: int, batch_end_pos: int):
        """
        Reads new batch from the table of the vcf.gz file.
        Fills in the target_full_array with the true sequence for the test input haploids
        Outputs:
         - the refilled batch (new read into the same np.array)
         - start position on the full sequence
         - end position on the fill sequence
        """
        # batch, lines_read_n = refpanel.readlines(lines_n=batch_size, refill=batch)
        batch, lines_read_n = refpanel.readlines3(lines_n=batch_size, total_haploids_n=6404, refill=batch)

        batch_start_pos = batch_end_pos
        batch_end_pos   = batch_end_pos+lines_read_n
        target_full_array[batch_start_pos:batch_end_pos, :] = batch[:lines_read_n, imputing_samples_haploids_indices]

        return batch, batch_start_pos, batch_end_pos

    def iterate_chips(
        chip_site_i: int, chip_site1_pos: int, chip_site2_pos: int,
        linspace_func = linspace_inner,
    ) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], int, int]:
        """
        Interpolates probabilities on the next interval between two chip sites
        Outputs:
         - the list of interpolated probabilities between the chip sites, for each input haploid
         - start position on the full sequence (position of the first chip site)
         - end position on the fill sequence (position of the second chip site)
        """
        # print(f"in iterate_chips({chip_site_i=}, {chip_site1_pos=}, {chip_site2_pos=})")

        # panels_under_construction = [np.linspace(
        #     weight_matrix[:, chip_site_i-1],
        #     weight_matrix[:, chip_site_i],
        #     num=segment_points[chip_site_i]-segment_points[chip_site_i-1],
        #     endpoint=False,
        # ) for weight_matrix in weight_matrices]

        # panels_under_construction: List[np.ndarray] = []

        # for i, weight_matrix in enumerate(weight_matrices):
        #     wm_m1 = weight_matrix[:, chip_site_i-1]
        #     wm = weight_matrix[:, chip_site_i]
        #     interspace = np.linspace(
        #         weight_matrix[:, chip_site_i-1],
        #         weight_matrix[:, chip_site_i],
        #         num=segment_points[chip_site_i]-segment_points[chip_site_i-1],
        #         endpoint=False,
        #     )
        #     panels_under_construction.append(interspace)
        #     print(
        #         f"    {i=} [{segment_points[chip_site_i-1]}, {segment_points[chip_site_i]}), "
        #         f"wm_m1 = ({wm_m1.shape}) {wm_m1.dtype}, wm = ({wm.shape}) {wm.dtype}, "
        #         f"interspace = ({interspace.shape}) {interspace.dtype}"
        #     )
        #     # print(f"      {=}")
        if chip_site_i == len(segment_points)-1:
            linspace_func = linspace_rightmost

        probs_interpolations, interpolated_ref_haps = zip(*[linspace_func(weight_matrix, chip_site_i) for weight_matrix in weight_matrices])
        # for in_hap in range(len(probs_collections)):
        #     probs_collections[in_hap][chip_site_i] = probs_interpolations[in_hap], interpolated_ref_haps[in_hap]

        return probs_interpolations, interpolated_ref_haps, segment_points[chip_site_i], segment_points[chip_site_i+1]

    def construct_on_intersection(start: int, end: int, probs_interpolations: List[np.ndarray], interpolated_ref_haps: Tuple[np.ndarray,...], batch_interval: np.ndarray):
        # print(
        #     f"construct_on_intersection("
        #     # f"({start},{end}), "
        #     f"probs_interpolations=arrays[{len(probs_interpolations)}({probs_interpolations[0].shape}) {probs_interpolations[0].dtype}, "
        #     f"batch_interval=({batch_interval.shape}) {batch_interval.dtype}",
        #     f")"
        # )
        for in_hap, probs_interpolation in enumerate(probs_interpolations):
            Xs[in_hap, start:end] = \
                (probs_interpolation * batch_interval[:,interpolated_ref_haps[in_hap]]).sum(axis=1) \
                                    > \
                (probs_interpolation * np.logical_not(batch_interval[:,interpolated_ref_haps[in_hap]])).sum(axis=1)
            # # One Dosage
            # panels_under_construction[start:end, 2*in_hap]   = (probs_interpolation * batch_interval).sum(axis=1)
            # # Zero Dosage
            # panels_under_construction[start:end, 2*in_hap+1] = (probs_interpolation * np.logical_not(batch_interval)).sum(axis=1)


    try:
        """
        iterating through:
            - the intervals between chip sites, and
            - the batches of the reference panel table read from the vcfgz

        Those may overlap. Also:
            - a batch may contain several intervals between the chip sites.
            - an interval between the chip sites may contain several batches.

        Therefore batches and intervals between the chip sites are iterated independently,
            but in the same loop. The processing happens on every intersection.
        """

        iB = 0 # Batch iterator
        # iC = 1 # interval between the Chip sites iterator
        iC = 0 # interval between the Chip sites iterator

        sB = eB = 0 # start and end positions of the vcf batch on the full sequence
        sC = eC = 0 # start and end positions of the chip sites on the full sequence

        inter_s = inter_e = 0 # intersection start and end positions


        batch, sB, eB = iterate_batch(None, sB, eB)
        # print(f"iB:{iB}, {batch.shape=}, {batch.dtype.type}")
        iB += 1
        probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC, linspace_func=linspace_leftmost)
        # dbg(probs_interpolations, "probs_interpolations")
        # dbg(interpolated_ref_haps, "interpolated_ref_haps")
        # dbg(sC, "sC")
        # dbg(eC, "eC")
        iC += 1

        debug_pos_counter = 0
        debug_timer = time.time()

        while (True):
            ( inter_s, inter_e ) = ( max(sB, sC), min(eB, eC) )
            # print(f"iC={iC:6}, iB={iB:5}, intersect ({inter_s:7}, {inter_e:7}), batch({sB:7}, {eB:7}), chips({sC:7}, {eC:7})")

            if inter_e - inter_s > 0:
                construct_on_intersection(
                    start=inter_s, end=inter_e,
                    probs_interpolations = [p[inter_s-sC : inter_e-sC] for p in probs_interpolations],
                    interpolated_ref_haps=interpolated_ref_haps,
                    batch_interval = batch[inter_s-sB : inter_e-sB, reference_haploids_indices][:, internal_haploid_order],
                )

            # some logging
            if inter_e - debug_pos_counter >= 60_000:
                debug_pos_counter = inter_e
                debug_timer_round = time.time()
                print(f"[{(debug_timer_round - debug_timer):8.3f} s] iC={iC:6}, iB={iB:5}, intersect ({inter_s:7}, {inter_e:7}), batch({sB:7}, {eB:7}), chips({sC:7}, {eC:7})")
                debug_timer = debug_timer_round

            if eB == eC:
                batch, sB, eB = iterate_batch(batch, sB, eB)
                # print(f"iB:{iB}, {batch.shape=}, {batch.dtype.type}")
                iB += 1
                probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC)
                iC += 1 
            else: # which interval ends sooner?
                if eB < eC:
                    batch, sB, eB = iterate_batch(batch, sB, eB)
                    # print(f"iB:{iB}, {batch.shape=}, {batch.dtype.type}")
                    iB += 1
                else:
                    probs_interpolations, interpolated_ref_haps, sC, eC = iterate_chips(iC, sC, eC)
                    iC += 1

            # if iC > 10:
            #     raise RuntimeError(f'break at {iC=}') 
            # if iB > 10:
            #     raise RuntimeError(f'break at {iB=}')


    except IndexError as e:
        """
        Reached the end of the full sequence
        """
        # print(f"imputation loop broke with: IndexError: {e}")
        pass


    # print(f"dumping probs_collection to \"opts01-05_dev1.3.1/HG02330/{{hap}}/07debug_probs_collections_v2.pkl\" ...")
    # for hap in (0,1):
    #     var_dump(f"opts01-05_dev1.3.1/HG02330/{hap}/07debug_probs_collections_v2.pkl", probs_collections[hap])
    # print(f"dumped probs_collection")

    refpanel.close()

    return Xs, target_full_array










def interpolate_parallel_packed_prev_prev(
    weight_matrix: np.ndarray,
    original_indicies: List[int],
    full_full_ref_panel_vcfgz_path: str,
    chr_length: int,
    start: int,
    end: int,
):
    # # number of reference panel haploids
    # assert original_ref_panel.shape[1] == weight_matrix.shape[0], "input shapes have to be consistent"
    # # number of chip variants
    # assert len(original_indicies) == weight_matrix.shape[1], "input shapes have to be consistent"

    """
    *n* chip sites divide the full sequence into *n+1* segments,
     iff the first and the last full-sequence sites are not in the chip sites.
    *n* chip sites divide the full sequence into *n* segments,
     iff either first or the last full-sequence site is in the chip sites.
    *n* chip sites divide the full sequence into *n-1* segments,
     iff both first and the last full-sequence site are in the chip sites.
    """
    segment_points: List[int] = [0, *original_indicies, chr_length]

    total_haploids = weight_matrix.shape[0]

    sticks_out_from_left = False
    sticks_out_from_right = False

    if original_indicies[0] != 0:
        # there are variants in the reference panel that go before the first chip variant
        sticks_out_from_left = True
        # segment_points.append(0)
        # weight_matrix_extended = np.concatenate([default_weights, weight_matrix_extended], axis=1)
    if original_indicies[-1] != chr_length-1:
        # there are variants in the reference panel that go after the last chip variant
        sticks_out_from_right = True
        # segment_points.append(chr_length)
        # weight_matrix_extended = np.concatenate([weight_matrix_extended, default_weights], axis=1)


    # full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)


    f = IBDs_extractor(full_full_ref_panel_vcfgz_path) # type: ignore # no IBDs_extractor
    segment_length = 15_000
    # batch = 

    

    def interpolate_probs_on_full_panel(
        panel_under_construction: np.ndarray,
        site: int,
        ref_panel_hap: int
    ):
        i_from = segment_points[site]
        i_to = segment_points[site+1]

        probs = np.linspace(
            weight_matrix[ref_panel_hap, site],
            weight_matrix[ref_panel_hap, site+1],
            num=i_to - i_from,
            endpoint=False
        )

        haploid_segment = unpack(original_ref_panel, ref_panel_hap, i_from, i_to, chr_length) # type: ignore # yes, there's no original_ref_panel

        # One Dosage
        panel_under_construction[i_from:i_to,0] += haploid_segment * probs # panel_under_construction[0:407,0] += haploid_segment * probs
        # Zero Dosage
        panel_under_construction[i_from:i_to,1] += np.logical_not(haploid_segment) * probs


    from joblib import Parallel, delayed
    def interpolate_probs_betw_chip_sites_in_range(start: int, end: int):
        full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
        for i in trange(start, end): # 0 in (0, 1)
            for j in range(0,weight_matrix.shape[0]): # j in range(0,5820)
                if weight_matrix[j, i+1] < 1/200 and weight_matrix[j, i] < 1/weight_matrix.shape[0]:
                    continue
                interpolate_probs_on_full_panel(
                    full_constructed_panel,
                    site=i,
                    ref_panel_hap=j,
                )

        result: np.ndarray = (
            full_constructed_panel[segment_points[start]:segment_points[end],0] 
                                > 
            full_constructed_panel[segment_points[start]:segment_points[end],1]
        ).astype(np.bool_)

        return result
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n][0], lst[i:i + n][-1] + 1

    segment_sticking_out_from_left = np.array([])
    segment_sticking_out_from_right = np.array([])

    print("interpolating probabilities from forward and backward algorithms, and imputing")


    if sticks_out_from_left:
        ##############################
        full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
        for j in range(0, weight_matrix.shape[0]):
            if weight_matrix[j, 1] < 1/weight_matrix.shape[0]:
                continue

            interpolate_probs_on_full_panel(
                full_constructed_panel,
                site=0,
                ref_panel_hap=j,
            )

        segment_sticking_out_from_left = (
            full_constructed_panel[segment_points[0]:segment_points[1],0]
                        >
            full_constructed_panel[segment_points[0]:segment_points[1],1]
        ).astype(np.bool_)

    if sticks_out_from_right:
        ##############################
        full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
        for j in range(0, weight_matrix.shape[0]):
            if weight_matrix[j, -2] < 1/weight_matrix.shape[0]:
                continue

            interpolate_probs_on_full_panel(
                full_constructed_panel,
                site=-2,
                ref_panel_hap=j,
            )

        segment_sticking_out_from_right = (
            full_constructed_panel[segment_points[-2]:segment_points[-1],0]
                        >
            full_constructed_panel[segment_points[-2]:segment_points[-1],1]
        ).astype(np.bool_)



    X = np.concatenate([
        segment_sticking_out_from_left,
        # in parallel
        *Parallel(n_jobs=8)(
            delayed(interpolate_probs_betw_chip_sites_in_range)(start, end)
                for (start, end) in list(chunks(
                    range(
                        0 + sticks_out_from_left,
                        weight_matrix.shape[1] - 1 - sticks_out_from_right,
                    ), 5000
                ))
        ), # type: ignore
        # # sequentially
        # *[
        #     interpolate_probs_betw_chip_sites_in_range(start, end)
        #         for (start, end) in list(chunks(
        #             range(
        #                 0 + sticks_out_from_left,
        #                 weight_matrix.shape[1] - 1 - sticks_out_from_right,
        #             ), 5000
        #         ))
        # ],
        segment_sticking_out_from_right,
    ])
    return X










#####################################
###          DATA_LOAD            ###
###                               ###
#####################################
import collections

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def deduplicate(chip_id_list, full_id_list, chip_positions):
    duplicate_pos = ([item for item, count in collections.Counter(chip_positions).items() if count > 1])
    duplicate_indexs = []
    none = [[duplicate_indexs.append(y) for y in list_duplicates_of(chip_positions, x)] for x in duplicate_pos]
    chip_id_list = list(np.delete(np.array(chip_id_list),duplicate_indexs))
    original_indicies = [full_id_list.index(x) for x in chip_id_list]
    return (chip_id_list, original_indicies)


#####################################
###          PLOT_RESULTS         ###
###                               ###
#####################################
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def get_beagle_res(file_name):
    beagle_array = pd.read_csv(file_name, sep="\t", header=None, comment="#")
    beagle_array.drop([0,1,3,4,5,6,7,8],axis=1,inplace=True)
    concat_beagle = pd.concat([
        beagle_array.applymap(lambda x: str(x).split(":")[0].replace("/","|").split("|")[0]),
        beagle_array.applymap(lambda x: str(x).split(":")[0].replace("/","|").split("|")[-1]),
    ],axis=1)
    concat_beagle.drop([2],axis=1,inplace=True)
    concat_beagle.columns = [0,1] # type: ignore
    return concat_beagle

def plot_results(
    samples,
    folder_dict,
    ref_panel_full_array_full,
    target_full_array,
    original_indicies,
):

    START = 0
    END = -1
    column = ["ID","Population","SuperPopulation","E-Beagle","E-Selphi","A-Beagle","A-Selphi"]
    table = pd.DataFrame(np.zeros([len(samples),len(column)]),columns=column)
    for c, sample in enumerate(samples):
        sample_index = get_sample_index(sample)
        target_full_array[:,0] = ref_panel_full_array_full[:,sample_index[0]]
        target_full_array[:,1] = ref_panel_full_array_full[:,sample_index[1]]
        info_sample = pd.read_table("/home/ubuntu/selphi/data/igsr_samples_short.tsv")
        population = info_sample[info_sample["Sample name"] == sample]["Population name"].unique()[0]
        superPopulation = info_sample[info_sample["Sample name"] == sample]["Superpopulation name"].unique()[0]
        print("Sample:",sample)
        print("Population:",population)
        print("SuperPopulation:",superPopulation)

        table.loc[c,"ID"] = sample
        table.loc[c,"Population"] = population
        table.loc[c,"SuperPopulation"] = superPopulation

        combined_target_full = target_full_array[:,0] + target_full_array[:,1]

        results = {}
        res = {}
        new_x = target_full_array[:,0] + target_full_array[:,1]

        for key in folder_dict.keys():
            if key != "beagle":
                with open(f"{folder_dict[key]}saved_dictionary_{sample}_new_method.pkl", "rb") as fp:   # Unpickling
                    results[key] = pickle.load(fp)
                    sample_0 = results[key][sample][0]
                    sample_1 = results[key][sample][1]
                    results[key] = {sample:[sample_0,sample_1]}
            else:
                results[key] = get_beagle_res(f"{folder_dict[key]}{sample}_BEAGLE_5.4_mapYES_neYES_validated.vcf.gz")
                beagle_0 = results[key][0].to_numpy().astype(int)
                beagle_1 = results[key][1].to_numpy().astype(int)
                results[key] = {sample:[beagle_0,beagle_1]}
            
            arr__1 = results[key][sample][0]
            arr__2 = results[key][sample][1]

            y = arr__1 + arr__2
            errors = int(y.shape - np.sum(new_x == y))
            accuracy = round((1-(errors)/len(arr__1))*100,4)
            print(f"Software: {key}")
            print(f"ERROR:    {errors}")
            print(f"Accuracy: {accuracy}")
            
            if key == "beagle":
                table.loc[c,"E-Beagle"] = errors
                table.loc[c,"A-Beagle"] = accuracy
            elif key == "selphi HMM":
                table.loc[c,"E-Selphi"] = errors
                table.loc[c,"A-Selphi"] = accuracy

            combined_results = arr__1 + arr__2
            
            STEP_LENGTH = 500
            MAX_LENGTH = y.shape[0]-STEP_LENGTH
            lengths = list(range(STEP_LENGTH,MAX_LENGTH,STEP_LENGTH))
            for LENGTH in lengths:
                res.setdefault(key, []).append(LENGTH - np.sum( combined_target_full[:LENGTH] == combined_results[:LENGTH] ))
        print()
        fig, ax = plt.subplots(figsize=(16,8))    # to create the picture plot empty  
        for key in folder_dict.keys(): 
            
            plt.plot(lengths, res[key], label = f"{key}") # type: ignore

        ax.set_xlabel("Segment Lengths",fontsize=10)
        ax.set_ylabel("Wrongly imputed variants",fontsize=10)
        plt.rcParams['axes.facecolor']='white' # type: ignore
        plt.rcParams['savefig.facecolor']='white' # type: ignore
        plt.legend()

        plt.show()
    table[["E-Beagle","E-Selphi"]] = table[["E-Beagle","E-Selphi"]].astype(int)
    table.to_csv(f"results_{len(samples)}_sample.tsv",sep="\t",index=True)


# def plot_results(
#     samples,
#     folder_dict,
#     ref_panel_full_array_full,
#     target_full_array,
#     original_indicies,
# ):

#     START = 0
#     END = -1
#     for c, sample in enumerate(samples):
#         sample_index = get_sample_index(sample)
#         target_full_array[:,0] = ref_panel_full_array_full[:,sample_index[0]]
#         target_full_array[:,1] = ref_panel_full_array_full[:,sample_index[1]]
#         info_sample = pd.read_table("/home/ubuntu/selphi/data/igsr_samples_short.tsv")

#         print("Sample:",sample)
#         print("Population:",info_sample[info_sample["Sample name"] == sample]["Population name"].unique()[0])
#         print("SuperPopulation:",info_sample[info_sample["Sample name"] == sample]["Superpopulation name"].unique()[0])

#         control = get_beagle_res(f"/home/ubuntu/selphi/data/validation_data/genome_individuals/{sample}.individual.30x.full_genome_20.vcf.gz")
#         control_0 = control[0].to_numpy().astype(int)
#         control_1 = control[1].to_numpy().astype(int)

#         if np.array_equal(target_full_array[:,0],control_0) and np.array_equal(target_full_array[:,1],control_1):
#             print("Control: Passed!") 
#         else:
#             print("Control: Failed! \nERROR: Beagle and Selphi are using 2 different input data. Results will be misleading! :(")

#         indicies_for_comparison = np.where(
#         (np.sum(ref_panel_full_array_full,axis=1) >0) & 
#         # (np.sum(ref_panel_full_array_full,axis=1) > 0)  &
#         (np.sum(target_full_array,axis=1) >= 0))[0]
#         combined_target_full = target_full_array[:,0] + target_full_array[:,1]

#         results = {}
#         res = {}
#         new_x = target_full_array[:,0] + target_full_array[:,1]

#         for key in folder_dict.keys():
#             if key != "beagle":
#                 with open(f"{folder_dict[key]}saved_dictionary_{sample}_new_method.pkl", "rb") as fp:   # Unpickling
#                     results[key] = pickle.load(fp)
#                     sample_0 = results[key][sample][0]
#                     sample_1 = results[key][sample][1]
#                     results[key] = {sample:[sample_0,sample_1]}
#             else:
#                 results[key] = get_beagle_res(f"{folder_dict[key]}{sample}_BEAGLE_5.4_mapYES_neYES.vcf.gz")
#                 beagle_0 = results[key][0].to_numpy().astype(int)
#                 beagle_1 = results[key][1].to_numpy().astype(int)
#                 results[key] = {sample:[beagle_0,beagle_1]}
            
#             arr__1 = results[key][sample][0]
#             arr__2 = results[key][sample][1]

#             y = arr__1 + arr__2
#             print(f"Software: {key}")
#             print(f"ERROR:    {int(y.shape - np.sum(new_x == y))}")
#             print(f"Accuracy: {round((1-(int(y.shape - np.sum(new_x == y)))/len(arr__1))*100,4)}")
#             combined_results = arr__1 + arr__2
            
#             STEP_LENGTH = 500
#             MAX_LENGTH = y.shape[0]-STEP_LENGTH
#             lengths = list(range(STEP_LENGTH,MAX_LENGTH,STEP_LENGTH))
#             for LENGTH in lengths:
#                 res.setdefault(key, []).append(LENGTH - np.sum( combined_target_full[:LENGTH] == combined_results[:LENGTH] ))
#         print()
#         fig, ax = plt.subplots(figsize=(16,8))    # to create the picture plot empty  
#         for key in folder_dict.keys(): 
            
#             plt.plot(lengths, res[key], label = f"{key}")

#         ax.set_xlabel("Segment Lengths",fontsize=10)
#         ax.set_ylabel("Wrongly imputed variants",fontsize=10)
#         plt.rcParams['axes.facecolor']='white'
#         plt.rcParams['savefig.facecolor']='white'
#         plt.legend()

#         plt.show()
            # plt.savefig(f'image_name_{sample}_new.jpg',format="jpg",dpi=200,transparent=True,bbox_inches='tight')

#####################################
###            B-PBWT             ###
###      Bidirectional PBWT       ###
#####################################



@njit
def numba_full_unrollable(shape, dtype, flipped_matrix_to_subtract):
    matrix = np.empty(shape, dtype=dtype)
    n0, n1 = matrix.shape[0], matrix.shape[1]
    # matrix[:] = np.arange(1, shape[1] + 1)  # doing np.empty and then this because np.full is not fully supported by numba

    for i in range(n0):
        for j in range(n1):
            matrix[i,j] = 1+j - flipped_matrix_to_subtract[i, n1-j-1]

    return matrix







@njit(fastmath=True) #type:ignore # njit *is* callable
def PBWT_forth(haplotypeList: np.ndarray, Target: int, num_hid: int):
    library_forw = createBWLibrary(haplotypeList[:, ::-1])

    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba

    num_chip_vars = ppa_matrix.shape[1]

    B = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    for cv in range(num_chip_vars):
        cvf = num_chip_vars - cv - 1 # chip variant flipped

        # forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
        # # forward_pbwt_index=ppa_matrix[:,chip_var]
        # backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
        # # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
        # BI[:,cv] = forward_pbwt_matches[ppa_matrix[:,cv].argsort(), cv][:num_hid]
        # BJ[:,cv] = backward_pbwt_matches[rev_ppa_matrix[:,::-1][:,cv].argsort(), cv][:num_hid]

        div_matrix_cv = np.empty(nHaps, dtype=library_forw.div.dtype.type)
        for i in range(nHaps):
            div_matrix_cv[i] = 1+cv - library_forw.div[i, num_chip_vars-cv-1]

        forward_pbwt_matches_ = replace_col(
            # div_matrix[:, cv],
            div_matrix_cv,
            cv,
            np.where(ppa_matrix[:, cv] == Target)[0][0]
        )
        forward_pbwt_index=ppa_matrix[:,cv]

        B[:,cv] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]



    num_chip_vars = ppa_matrix.shape[1]
    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in range(num_chip_vars):

        x = np.unique(B[:,chip_var])
        if len(x) == 1 and x[0] == 0:
            B[:,chip_var] = 1
            B[:,chip_var] = 1

    return B



@njit(fastmath=True) #type:ignore # njit *is* callable
def PBWT_back(haplotypeList: np.ndarray, Target: int, num_hid: int):
    library_forw = createBWLibrary(haplotypeList)

    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba

    num_chip_vars = ppa_matrix.shape[1]

    B = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    for cv in range(num_chip_vars):
        cvf = num_chip_vars - cv - 1 # chip variant flipped

        # forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
        # # forward_pbwt_index=ppa_matrix[:,chip_var]
        # backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
        # # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
        # BI[:,cv] = forward_pbwt_matches[ppa_matrix[:,cv].argsort(), cv][:num_hid]
        # BJ[:,cv] = backward_pbwt_matches[rev_ppa_matrix[:,::-1][:,cv].argsort(), cv][:num_hid]

        div_matrix_cvf = np.empty(nHaps, dtype=library_forw.div.dtype.type)
        for i in range(nHaps):
            div_matrix_cvf[i] = 1+cvf - library_forw.div[i, num_chip_vars-cvf-1]

        forward_pbwt_matches_ = replace_col(
            # div_matrix[:, cvf],
            div_matrix_cvf,
            cvf,
            np.where(ppa_matrix[:, cvf] == Target)[0][0]
        )
        forward_pbwt_index=ppa_matrix[:,cvf]

        B[:,cv] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]



    num_chip_vars = ppa_matrix.shape[1]
    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in range(num_chip_vars):

        x = np.unique(B[:,chip_var])
        if len(x) == 1 and x[0] == 0:
            B[:,chip_var] = 1
            B[:,chip_var] = 1

    return B



@njit(fastmath=True) #type:ignore # njit *is* callable
def BiDiPBWT(haplotypeList: np.ndarray, Target: int, num_hid: int):
    BI = PBWT_forth(haplotypeList, Target, num_hid)
    BJ = PBWT_back (haplotypeList,          Target, num_hid)

    return BI, BJ



@njit(fastmath=True) #type:ignore # njit *is* callable
def BInBJ_onelib(haplotypeList: np.ndarray, Target: int, num_hid: int):
    # DOESN'T WORK.
    # createBWLibrary(haplotypeList[:, ::-1]) != createBWLibrary(haplotypeList)[:, ::-1]
    library_forw = createBWLibrary(haplotypeList[:, ::-1])
    library_back = createBWLibrary(haplotypeList)
    print("library_back.ppa == library_forw.ppa[:, ::-1] = ", (library_back.ppa == library_forw.ppa[:, ::-1]).all())
    print("library_back.div == library_forw.div[:, ::-1] = ", (library_back.div == library_forw.div[:, ::-1]).all())

    ppa_where = np.where(library_back.ppa != library_forw.ppa[:, ::-1])
    div_where = np.where(library_back.div != library_forw.div[:, ::-1])

    print("np.where(library_back.ppa != library_forw.ppa[:, ::-1]) = ", ppa_where , "shape=(",np.unique(ppa_where[0]).shape[0],np.unique(ppa_where[1]).shape[0],")")
    print("np.where(library_back.div != library_forw.div[:, ::-1]) = ", div_where , "shape=(",np.unique(div_where[0]).shape[0],np.unique(div_where[1]).shape[0],")")


    print("library_forw.ppa[0:5,0:5] = ", library_forw.ppa[0:5,0:5])
    print("library_back.ppa[0:5,0:5] = ", library_back.ppa[0:5,0:5])
    print("library_forw.ppa[:, ::-1][0:5,0:5] = ", library_forw.ppa[:, ::-1][0:5,0:5])
    print("library_forw.ppa[::-1, :][0:5,0:5] = ", library_forw.ppa[::-1, :][0:5,0:5])
    print("library_forw.ppa[::-1, ::-1][0:5,0:5] = ", library_forw.ppa[::-1, ::-1][0:5,0:5])
    print("")

    print("library_forw.div[0:5,0:5] = ", library_forw.div[0:5,0:5])
    print("library_back.div[0:5,0:5] = ", library_back.div[0:5,0:5])
    print("library_forw.div[:, ::-1][0:5,0:5] = ", library_forw.div[:, ::-1][0:5,0:5])
    print("library_forw.div[::-1, :][0:5,0:5] = ", library_forw.div[::-1, :][0:5,0:5])
    print("library_forw.div[::-1, ::-1][0:5,0:5] = ", library_forw.div[::-1, ::-1][0:5,0:5])
    print("")

    print("library_back.ppa == library_forw.ppa[:, ::-1] = ", (library_back.ppa == library_forw.ppa[:, ::-1]).all())
    print("library_back.div == library_forw.div[:, ::-1] = ", (library_back.div == library_forw.div[:, ::-1]).all())

    ppa_where = np.where(library_back.ppa != library_forw.ppa[:, ::-1])
    div_where = np.where(library_back.div != library_forw.div[:, ::-1])

    print("np.where(library_back.ppa != library_forw.ppa[:, ::-1]) = ", ppa_where , "shape=(",np.unique(ppa_where[0]).shape[0],np.unique(ppa_where[1]).shape[0],")")
    print("np.where(library_back.div != library_forw.div[:, ::-1]) = ", div_where , "shape=(",np.unique(div_where[0]).shape[0],np.unique(div_where[1]).shape[0],")")




    # print("np.where(library_back.ppa != library_forw.ppa[:, ::-1]) = ", np.where(library_back.ppa != library_forw.ppa[:, ::-1]))
    # print("np.where(library_back.div != library_forw.div[:, ::-1]) = ", np.where(library_back.div != library_forw.div[:, ::-1]))


    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba

    rev_ppa_matrix = library_forw.ppa

    num_chip_vars = ppa_matrix.shape[1]

    BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    for cv in range(num_chip_vars):
        cvf = num_chip_vars - cv - 1 # chip variant flipped

        div_matrix_cv = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            div_matrix_cv[i] = 1+cv - library_forw.div[i, cvf]

        forward_pbwt_matches_ = replace_col(
            # div_matrix[:, cv],
            div_matrix_cv,
            cv,
            np.where(ppa_matrix[:, cv] == Target)[0][0]
        )
        forward_pbwt_index=ppa_matrix[:,cv]

        rev_div_matrix_cvf = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            rev_div_matrix_cvf[i] = 1+cvf - library_forw.div[i, cvf]

        backward_pbwt_matches_ = replace_col(
            # rev_div_matrix[:,cvf],
            rev_div_matrix_cvf,
            cvf,
            np.where(rev_ppa_matrix[:, cvf] == Target)[0][0]
        )
        backward_pbwt_index=rev_ppa_matrix[:,::-1][:,cv]

        BI[:,cv] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
        BJ[:,cv] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]



    num_chip_vars = ppa_matrix.shape[1]
    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in range(num_chip_vars):

        x = np.unique(BI[:,chip_var])
        if len(x) == 1 and x[0] == 0:
            BI[:,chip_var] = 1
            BJ[:,chip_var] = 1



    return BI, BJ




@njit(fastmath=True) #type:ignore # njit *is* callable
def BInBJ(haplotypeList: np.ndarray, Target: int, num_hid: int):
    library_forw = createBWLibrary(haplotypeList[:, ::-1])
    library_back = createBWLibrary(haplotypeList)
    
    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba
    # div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_forw.div.dtype.type
    # ) - library_forw.div[:, ::-1]

    # div_matrix = np.empty(
    #     Shape, dtype=library_forw.div.dtype.type
    # )
    # div_matrix[:] = np.arange(1, Shape[1] + 1)  # doing np.empty and then this because np.full is not fully supported by numba
    # div_matrix -= library_forw.div[:, ::-1]

    ####div_matrix = numba_full_unrollable(Shape, library_forw.div.dtype.type, library_forw.div)
    # div_matrix -= library_forw.div[:, ::-1]

    rev_ppa_matrix = library_back.ppa[:, ::-1]
    # rev_div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_back.div.dtype.type
    # ) - library_back.div[:, ::-1]

    # rev_div_matrix = np.empty(
    #     Shape, dtype=library_back.div.dtype.type
    # )
    # rev_div_matrix[:] = np.arange(1, Shape[1] + 1)
    # rev_div_matrix -= library_back.div[:, ::-1]

    ####rev_div_matrix = numba_full_unrollable(Shape, library_back.div.dtype.type, library_back.div)
    # rev_div_matrix -= library_back.div[:, ::-1]


    # return ppa_matrix, div_matrix, rev_ppa_matrix, rev_div_matrix

    # forward_pbwt_matches, forward_pbwt_hap_indices = get_matches_indices(
    #     ppa_matrix, div_matrix, Target
    # )
    # backward_pbwt_matches, backward_pbwt_hap_indices = get_matches_indices(
    #     rev_ppa_matrix, rev_div_matrix, Target
    # )
    # backward_pbwt_matches = backward_pbwt_matches[:, ::-1]

    # return ppa_matrix, forward_pbwt_matches, rev_ppa_matrix, backward_pbwt_matches

    num_chip_vars = ppa_matrix.shape[1]

    BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    # for chip_var in range(num_chip_vars):
    #     forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
    #     forward_pbwt_index=ppa_matrix[:,chip_var]
    #     backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
    #     backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
    #     BI[:,chip_var] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
    #     BJ[:,chip_var] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]

    for cv in range(num_chip_vars):
        cvf = num_chip_vars - cv - 1 # chip variant flipped

        # forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
        # # forward_pbwt_index=ppa_matrix[:,chip_var]
        # backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
        # # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
        # BI[:,cv] = forward_pbwt_matches[ppa_matrix[:,cv].argsort(), cv][:num_hid]
        # BJ[:,cv] = backward_pbwt_matches[rev_ppa_matrix[:,::-1][:,cv].argsort(), cv][:num_hid]

        div_matrix_cv = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            div_matrix_cv[i] = 1+cv - library_forw.div[i, num_chip_vars-cv-1]

        forward_pbwt_matches_ = replace_col(
            # div_matrix[:, cv],
            div_matrix_cv,
            cv,
            np.where(ppa_matrix[:, cv] == Target)[0][0]
        )
        forward_pbwt_index=ppa_matrix[:,cv]

        rev_div_matrix_cvf = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            rev_div_matrix_cvf[i] = 1+cvf - library_back.div[i, num_chip_vars-cvf-1]

        backward_pbwt_matches_ = replace_col(
            # rev_div_matrix[:,cvf],
            rev_div_matrix_cvf,
            cvf,
            np.where(rev_ppa_matrix[:, cvf] == Target)[0][0]
        )
        backward_pbwt_index=rev_ppa_matrix[:,::-1][:,cv]

        BI[:,cv] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
        BJ[:,cv] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]



    num_chip_vars = ppa_matrix.shape[1]
    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in range(num_chip_vars):

        x = np.unique(BI[:,chip_var])
        if len(x) == 1 and x[0] == 0:
            BI[:,chip_var] = 1
            BJ[:,chip_var] = 1



    return BI, BJ




@njit(fastmath=True) #type:ignore # njit *is* callable
def BInBJ_both(haplotypeList: np.ndarray, Target0: int, Target1: int, num_hid: int):
    library_back = createBWLibrary(haplotypeList)
    # library_forw = createBWLibrary(np.flip(np.array(haplotypeList), axis=1))
    library_forw = createBWLibrary(haplotypeList[:, ::-1])
    
    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba
    # div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_forw.div.dtype.type
    # ) - library_forw.div[:, ::-1]

    # div_matrix = np.empty(
    #     Shape, dtype=library_forw.div.dtype.type
    # )
    # div_matrix[:] = np.arange(1, Shape[1] + 1)  # doing np.empty and then this because np.full is not fully supported by numba
    # div_matrix -= library_forw.div[:, ::-1]

    ###div_matrix = numba_full_unrollable(Shape, library_forw.div.dtype.type, library_forw.div)
    # div_matrix -= library_forw.div[:, ::-1]

    rev_ppa_matrix = library_back.ppa[:, ::-1]
    # rev_div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_back.div.dtype.type
    # ) - library_back.div[:, ::-1]

    # rev_div_matrix = np.empty(
    #     Shape, dtype=library_back.div.dtype.type
    # )
    # rev_div_matrix[:] = np.arange(1, Shape[1] + 1)
    # rev_div_matrix -= library_back.div[:, ::-1]

    ###rev_div_matrix = numba_full_unrollable(Shape, library_back.div.dtype.type, library_back.div)
    # rev_div_matrix -= library_back.div[:, ::-1]


    # return ppa_matrix, div_matrix, rev_ppa_matrix, rev_div_matrix

    num_chip_vars = ppa_matrix.shape[1]

    def BInBJ_one(target):
        # forward_pbwt_matches, forward_pbwt_hap_indices = get_matches_indices(
        #     ppa_matrix, div_matrix, target
        # )
        # backward_pbwt_matches, backward_pbwt_hap_indices = get_matches_indices(
        #     rev_ppa_matrix, rev_div_matrix, target
        # )
        # backward_pbwt_matches = backward_pbwt_matches[:, ::-1]

        BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
        BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
        for cv in range(num_chip_vars):
            cvf = num_chip_vars - cv - 1

            # forward_pbwt_matches_=forward_pbwt_matches[:,chip_var]
            # # forward_pbwt_index=ppa_matrix[:,chip_var]
            # backward_pbwt_matches_=backward_pbwt_matches[:,chip_var]
            # # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
            # BI[:,cv] = forward_pbwt_matches[ppa_matrix[:,cv].argsort(), cv][:num_hid]
            # BJ[:,cv] = backward_pbwt_matches[rev_ppa_matrix[:,::-1][:,cv].argsort(), cv][:num_hid]


            div_matrix_cv = np.empty(nHaps, dtype=library_forw.div.dtype.type)
            for i in range(nHaps):
                div_matrix_cv[i] = 1+cv - library_forw.div[i, num_chip_vars-cv-1]

            forward_pbwt_matches_ = replace_col(
                # div_matrix[:, cv],
                div_matrix_cv,
                cv,
                np.where(ppa_matrix[:, cv] == target)[0][0]
            )
            forward_pbwt_index=ppa_matrix[:,cv]

            rev_div_matrix_cvf = np.empty(nHaps, dtype=library_back.div.dtype.type)
            for i in range(nHaps):
                rev_div_matrix_cvf[i] = 1+cvf - library_back.div[i, num_chip_vars-cvf-1]

            backward_pbwt_matches_ = replace_col(
                # rev_div_matrix[:,cvf],
                rev_div_matrix_cvf,
                cvf,
                np.where(rev_ppa_matrix[:, cvf] == target)[0][0]
            )
            backward_pbwt_index=rev_ppa_matrix[:,::-1][:,cv]

            BI[:,cv] = forward_pbwt_matches_[forward_pbwt_index.argsort()][:num_hid]
            BJ[:,cv] = backward_pbwt_matches_[backward_pbwt_index.argsort()][:num_hid]

        """
        Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
            then treat is as matching all the samples
        """
        for chip_var in range(num_chip_vars):

            x = np.unique(BI[:,chip_var])
            if len(x) == 1 and x[0] == 0:
                BI[:,chip_var] = 1
                BJ[:,chip_var] = 1

        return BI, BJ



    BI0, BJ0 = BInBJ_one(Target0)
    BI1, BJ1 = BInBJ_one(Target1)


    return BI0, BJ0, BI1, BJ1





@njit
def BInBJ_paired(haplotypeList: np.ndarray, Target0: int, Target1: int, num_hid: int):
    library_back = createBWLibrary(haplotypeList)
    # library_forw = createBWLibrary(np.flip(np.array(haplotypeList), axis=1))
    library_forw = createBWLibrary(haplotypeList[:, ::-1])
    
    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba
    # div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_forw.div.dtype.type
    # ) - library_forw.div[:, ::-1]

    # div_matrix = np.empty(
    #     Shape, dtype=library_forw.div.dtype.type
    # )
    # div_matrix[:] = np.arange(1, Shape[1] + 1)  # doing np.empty and then this because np.full is not fully supported by numba
    # div_matrix -= library_forw.div[:, ::-1]

    div_matrix = numba_full_unrollable(Shape, library_forw.div.dtype.type, library_forw.div)
    # div_matrix -= library_forw.div[:, ::-1]

    rev_ppa_matrix = library_back.ppa[:, ::-1]
    # rev_div_matrix = np.full(
    #     Shape, np.arange(1, Shape[1] + 1), dtype=library_back.div.dtype.type
    # ) - library_back.div[:, ::-1]

    # rev_div_matrix = np.empty(
    #     Shape, dtype=library_back.div.dtype.type
    # )
    # rev_div_matrix[:] = np.arange(1, Shape[1] + 1)
    # rev_div_matrix -= library_back.div[:, ::-1]

    rev_div_matrix = numba_full_unrollable(Shape, library_back.div.dtype.type, library_back.div)
    # rev_div_matrix -= library_back.div[:, ::-1]


    # return ppa_matrix, div_matrix, rev_ppa_matrix, rev_div_matrix

    forward_pbwt_matches0, forward_pbwt_hap_indices0, forward_pbwt_matches1, forward_pbwt_hap_indices1 = get_matches_indices_pair(
        ppa_matrix, div_matrix, Target0, Target1
    )
    backward_pbwt_matches0, backward_pbwt_hap_indices0, backward_pbwt_matches1, backward_pbwt_hap_indices1 = get_matches_indices_pair(
        rev_ppa_matrix, rev_div_matrix, Target0, Target1
    )
    backward_pbwt_matches0 = backward_pbwt_matches0[:, ::-1]
    backward_pbwt_matches1 = backward_pbwt_matches1[:, ::-1]

    # return ppa_matrix, forward_pbwt_matches, rev_ppa_matrix, backward_pbwt_matches

    num_chip_vars = ppa_matrix.shape[1]

    BI0 = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    BJ0 = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    for chip_var in range(num_chip_vars):
        forward_pbwt_matches_0=forward_pbwt_matches0[:,chip_var]
        # forward_pbwt_index=ppa_matrix[:,chip_var]
        backward_pbwt_matches_0=backward_pbwt_matches0[:,chip_var]
        # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
        BI0[:,chip_var] = forward_pbwt_matches_0[ppa_matrix[:,chip_var].argsort()][:num_hid]
        BJ0[:,chip_var] = backward_pbwt_matches_0[rev_ppa_matrix[:,::-1][:,chip_var].argsort()][:num_hid]


    BI1 = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    BJ1 = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    for chip_var in range(num_chip_vars):
        forward_pbwt_matches_1=forward_pbwt_matches1[:,chip_var]
        # forward_pbwt_index=ppa_matrix[:,chip_var]
        backward_pbwt_matches_1=backward_pbwt_matches1[:,chip_var]
        # backward_pbwt_index=rev_ppa_matrix[:,::-1][:,chip_var]
        BI1[:,chip_var] = forward_pbwt_matches_1[ppa_matrix[:,chip_var].argsort()][:num_hid]
        BJ1[:,chip_var] = backward_pbwt_matches_1[rev_ppa_matrix[:,::-1][:,chip_var].argsort()][:num_hid]


    num_chip_vars = ppa_matrix.shape[1]
    """
    Data cleaning: If a chip variant doesn't have _any_ matches in the reference panel,
        then treat is as matching all the samples
    """
    for chip_var in range(num_chip_vars):

        x0 = np.unique(BI0[:,chip_var])
        x1 = np.unique(BI1[:,chip_var])
        if len(x0) == 1 and x0[0] == 0:
            BI0[:,chip_var] = 1
            BJ0[:,chip_var] = 1
        if len(x1) == 1 and x1[0] == 0:
            BI0[:,chip_var] = 1
            BJ0[:,chip_var] = 1



    return BI0, BJ0, BI1, BJ1









class BidiBurrowsWheelerLibrary:
    def __init__(self, haplotypeList: np.ndarray, Target: int):
        self.library_back = createBWLibrary(np.array(haplotypeList))
        self.library_forw = createBWLibrary(np.flip(np.array(haplotypeList), axis=1))
        # BInBJ(haplotypeList, Target)
        self.hapList = haplotypeList
        self.target = Target
        self.Shape = self.hapList.shape
        self.nHaps = len(haplotypeList)

    def getForward_Ppa(self):
        return np.flip(self.library_forw.ppa, axis=1)

    def getBackward_Ppa(self):
        return np.flip(self.library_back.ppa, axis=1)

    def getForward_Div(self):
        return np.full(
            self.Shape, np.arange(1, self.Shape[1] + 1), dtype=self.library_forw.div.dtype.type
        ) - np.flip(self.library_forw.div, axis=1)

    def getBackward_Div(self):
        return np.full(
            self.Shape, np.arange(1, self.Shape[1] + 1), dtype=self.library_back.div.dtype.type
        ) - np.flip(self.library_back.div, axis=1)

    def getForward_matches_indices(self):
        return get_matches_indices(
            self.getForward_Ppa(), self.getForward_Div(), self.target
        )

    def getBackward_matches_indices(self):
        matches_indices = get_matches_indices(
            self.getBackward_Ppa(), self.getBackward_Div(), self.target
        )
        return np.flip(
            matches_indices[0],
            axis=1,
        ), np.flip(
            matches_indices[1]
        )


jit_BidiBurrowsWheelerLibrary_s = OrderedDict()
jit_BidiBurrowsWheelerLibrary_s["ppa"] = int16[:, :]
jit_BidiBurrowsWheelerLibrary_s["div"] = int32[:, :]


@jitclass(jit_BidiBurrowsWheelerLibrary_s)
class jit_BidiBurrowsWheelerLibrary:
    def __init__(self, ppa, div):
        self.ppa = ppa
        self.div = div


@njit
def createBWLibrary(haps):

    # Definitions.
    # haps : a list of haplotypes
    # ppa : an ordering of haps in lexographic order (ppa_matrix)
    # div : Number of loci of a[i,j+k] == a[i,-1, j+k] (number of mactches) the length of the longest match between each (sorted) haplotype and the previous (sorted) haplotype.

    nHaps = haps.shape[0]
    nLoci = haps.shape[1]
    ppa = np.full(haps.shape, 0, dtype=np.int16)
    div = np.full(haps.shape, 0, dtype=np.int32)

    nZerosArray = np.full(nLoci, 0, dtype=np.int16)

    zeros = np.full(nHaps, 0, dtype=np.int16)
    ones = np.full(nHaps, 0, dtype=np.int16)
    dZeros = np.full(nHaps, 0, dtype=np.int32)
    dOnes = np.full(nHaps, 0, dtype=np.int32)

    nZeros = 0
    nOnes = 0
    for j in range(nHaps):
        if haps[j, nLoci - 1] == 0:
            zeros[nZeros] = j
            if nZeros == 0:
                dZeros[nZeros] = 0
            else:
                dZeros[nZeros] = 1
            nZeros += 1
        else:
            ones[nOnes] = j
            if nOnes == 0:
                dOnes[nOnes] = 0
            else:
                dOnes[nOnes] = 1
            nOnes += 1
    if nZeros > 0:
        ppa[0:nZeros, nLoci - 1] = zeros[0:nZeros]
        div[0:nZeros, nLoci - 1] = dZeros[0:nZeros]

    if nOnes > 0:
        ppa[nZeros:nHaps, nLoci - 1] = ones[0:nOnes]
        div[nZeros:nHaps, nLoci - 1] = dOnes[0:nOnes]

    nZerosArray[nLoci - 1] = nZeros

    for i in range(nLoci - 2, -1, -1):
        zeros = np.full(nHaps, 0, dtype=np.int16)
        ones = np.full(nHaps, 0, dtype=np.int16)
        dZeros = np.full(nHaps, 0, dtype=np.int32)
        dOnes = np.full(nHaps, 0, dtype=np.int32)

        nZeros = 0
        nOnes = 0

        dZerosTmp = -1  # This is a hack.
        dOnesTmp = -1

        for j in range(nHaps):

            dZerosTmp = min(dZerosTmp, div[j, i + 1])
            dOnesTmp = min(dOnesTmp, div[j, i + 1])
            if haps[ppa[j, i + 1], i] == 0:
                zeros[nZeros] = ppa[j, i + 1]
                dZeros[nZeros] = dZerosTmp + 1
                nZeros += 1
                dZerosTmp = nLoci
            else:
                ones[nOnes] = ppa[j, i + 1]
                dOnes[nOnes] = dOnesTmp + 1
                nOnes += 1
                dOnesTmp = nLoci

        if nZeros > 0:
            ppa[0:nZeros, i] = zeros[0:nZeros]
            div[0:nZeros, i] = dZeros[0:nZeros]

        if nOnes > 0:
            ppa[nZeros:nHaps, i] = ones[0:nOnes]
            div[nZeros:nHaps, i] = dOnes[0:nOnes]
        nZerosArray[i] = nZeros

    # I'm going to be a wee bit sloppy in creating zeroOccPrev
    # Not defined at 0 so start at 1.
    zeroOccPrev = np.full(haps.shape, 0, dtype=np.int16)

    for i in range(1, nLoci):
        count = 0
        for j in range(0, nHaps):
            if haps[ppa[j, i], i - 1] == 0:
                count += 1
            zeroOccPrev[j, i] = count

    library = jit_BidiBurrowsWheelerLibrary(ppa, div)
    return library



"""
because numba v0.54 and less don't support np.maximum.accumulate()
"""
@njit
def numba_maximum_accumulate(A):
    r = np.empty(len(A))
    t = -np.inf
    for i in range(len(A)):
        t = np.maximum(t, A[i])
        r[i] = t
    return r


@njit(fastmath=True) #type:ignore # njit *is* callable
def replace_col(array, col_index, hap_index):
    acc = np.zeros(array.shape[0], dtype=np.int32)
    acc[:hap_index + 1] = np.flip(numba_maximum_accumulate(np.flip(array[: hap_index + 1])))
    acc[:hap_index] = acc[1 : hap_index + 1]

    acc[hap_index+1:] = numba_maximum_accumulate(array[hap_index + 1:])
    return col_index + 1 - acc



# Function to replace values with match lengths
@njit(fastmath=True) #type:ignore # njit *is* callable
def get_matches_indices(ppa_matrix, div_matrix, target=6401):
    forward_pbwt_matches = np.zeros(ppa_matrix.shape, dtype=np.int32)
    forward_pbwt_hap_indices = np.zeros(forward_pbwt_matches.shape[1], dtype=np.int64)
    for i in range(0, forward_pbwt_matches.shape[1]):
        hap_index = np.where(ppa_matrix[:, i] == target)[0][0] # gets the first matching index (there should be only 1 though)
        forward_pbwt_matches[:, i] = replace_col(div_matrix[:, i], i, hap_index)
        forward_pbwt_hap_indices[i] = hap_index
    return (forward_pbwt_matches, forward_pbwt_hap_indices)


@njit
def get_matches_indices_pair(ppa_matrix, div_matrix, target0=6401, target1=6402):
    n0, n1 = ppa_matrix.shape[0], ppa_matrix.shape[1]
    forward_pbwt_matches0 = np.zeros((n0, n1), dtype=np.int32)
    forward_pbwt_hap_indices0 = np.zeros(n1, dtype=np.int64)
    forward_pbwt_matches1 = np.zeros((n0, n1), dtype=np.int32)
    forward_pbwt_hap_indices1 = np.zeros(n1, dtype=np.int64)
    
    for i in range(n1):
        hap_index0 = np.where(ppa_matrix[:, i] == target0)[0][0] # gets the first matching index (there should be only 1 though)
        hap_index1 = np.where(ppa_matrix[:, i] == target1)[0][0] # gets the first matching index (there should be only 1 though)
        forward_pbwt_matches0[:, i] = replace_col(div_matrix[:, i], i, hap_index0)
        forward_pbwt_matches1[:, i] = replace_col(div_matrix[:, i], i, hap_index1)
        forward_pbwt_hap_indices0[i] = hap_index0
        forward_pbwt_hap_indices1[i] = hap_index1
    return (forward_pbwt_matches0, forward_pbwt_hap_indices0, forward_pbwt_matches1, forward_pbwt_hap_indices1)



def get_std(avg_length, a=25):
    # convert average length into number on x axis
    rmin_ = 10000
    rmax_ = 1
    tmin_ = 0
    tmax_ = 1
    avg_length = ((avg_length - rmin_)/(rmax_ - rmin_)) * (tmax_ - tmin_) + tmin_

    std_not_normed = a*avg_length**(a-1.)

    rmin_ = 0
    rmax_ = a*1**(a-1.)
    tmin_ = 0.2
    tmax_ = 3
    return ((std_not_normed - rmin_)/(rmax_ - rmin_)) * (tmax_ - tmin_) + tmin_



# def internal_haploid_order(sample_i: int, samples_total: int):
#     return sample_i, sample_i+samples_total
# def vcf_haploid_order(sample_i: int):
#     return 2*sample_i, 2*sample_i + 1

# def internal_haploid_order_to_vcf(haploid_i: int, samples_total: int):
#     """converts haploid index in the internal order (where haps0 go first, then haps1) to the ordinary order in vcf"""
#     return int(haploid_i / 2) + samples_total*(haploid_i % 2)

# def vcf_haploid_order_to_internal(haploid_i: int, samples_total: int):
#     """converts haploid index in the vcf to the 'internal order' used originally in selphi (haps0 first, then haps1)"""
#     return 2*(haploid_i % samples_total) + int(haploid_i / samples_total)

def vcf_haploid_order_to_internal(haploid_i: int, samples_total: int):
    """leaves the order of haploids unchanged for the internal calculations"""
    return haploid_i

# @lru_cache(5)
# def get_shuffled_indices_list(length: int, seed: Union[int,None] = None):
#     indices = list(range(length))
#     random.Random(seed).shuffle(indices)
#     return indices
# def vcf_haploid_order_to_internal(haploid_i: int, samples_total: int):
#     """intoduces a shuffled haploid order to be used internally"""
#     return get_shuffled_indices_list(samples_total*2, seed=6)[haploid_i]

