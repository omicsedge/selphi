from typing import List, Set, Tuple, TypeVar, Union, Iterable, Dict, Literal
from collections import OrderedDict
import time


#from black import Result
import numpy as np
from tqdm import trange, tqdm
from numba import njit, int8, int32, int64

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass # type: ignore


from modules.data_utils import get_sample_index
from modules.vcfgz_reader import vcfgz_reader





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

    # this array hold the imputed samples
    Xs = np.zeros((total_input_haploids, chr_length), dtype=np.int8)

    refpanel = vcfgz_reader(full_full_ref_panel_vcfgz_path)
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

        batch_start_pos = batch_end_pos
        batch_end_pos   = batch_end_pos+lines_read_n

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
                - the intervals between the input (chip) sites, and
                - the batches of the reference panel table read from the vcfgz

            Those may overlap. Also:
                - a batch may contain several intervals between the chip sites.
                - an interval between the chip sites may contain several batches.

            Therefore we iterate batches and intervals between the chip sites independently,
                in the same loop. We repeat imputation on every intersection.
            """

            iB = 0 # reference panel Batch iterator
            iC = 0 # interval between the input (Chip) sites iterator

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

                # 2. Performing imputation on the intersection interval
                if inter_e - inter_s > 0:
                    impute_interval(
                        start=inter_s, end=inter_e,
                        probs_interpolations = [p[inter_s-sC : inter_e-sC] for p in probs_interpolations],
                        interpolated_ref_haps = interpolated_ref_haps,
                        batch_interval = batch[inter_s-sB : inter_e-sB],
                    )

                # # [some logging while not in prod]
                # if inter_e - debug_pos_counter >= 120_000:
                #     debug_pos_counter = inter_e
                #     debug_timer_round = time.time()
                #     print(f"[{(debug_timer_round - debug_timer):8.3f} s] iC={iC:6}, iB={iB:5}, intersect ({inter_s:7}, {inter_e:7}), batch({sB:7}, {eB:7}), chips({sC:7}, {eC:7})")
                #     debug_timer = debug_timer_round

                # 3. Iteration: iterating samples chip sites interval or reading a new batch from the ref panel
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
def BInBJ(haplotypeList: np.ndarray, Target: int, num_hid: int):
    library_forw = createBWLibrary(haplotypeList[:, ::-1])
    library_back = createBWLibrary(haplotypeList)
    
    Shape = haplotypeList.shape
    nHaps = haplotypeList.shape[0]

    ppa_matrix = library_forw.ppa[:, ::-1] # using this because np.flip(a, axis=1) is not supported by numba

    rev_ppa_matrix = library_back.ppa[:, ::-1]

    num_chip_vars = ppa_matrix.shape[1]

    BI = np.zeros((num_hid,num_chip_vars), dtype=np.int32)
    BJ = np.zeros((num_hid,num_chip_vars), dtype=np.int32)

    for cv in range(num_chip_vars):
        cvf = num_chip_vars - cv - 1 # chip variant flipped

        div_matrix_cv = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            div_matrix_cv[i] = 1+cv - library_forw.div[i, num_chip_vars-cv-1]

        forward_pbwt_matches_ = replace_col(
            div_matrix_cv,
            cv,
            np.where(ppa_matrix[:, cv] == Target)[0][0]
        )
        forward_pbwt_index=ppa_matrix[:,cv]

        rev_div_matrix_cvf = np.empty(nHaps, dtype=np.int32)
        for i in range(nHaps):
            rev_div_matrix_cvf[i] = 1+cvf - library_back.div[i, num_chip_vars-cvf-1]

        backward_pbwt_matches_ = replace_col(
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



jit_BidiBurrowsWheelerLibrary_s = OrderedDict()
jit_BidiBurrowsWheelerLibrary_s["ppa"] = int32[:, :]
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
    ppa = np.full(haps.shape, 0, dtype=np.int32)
    div = np.full(haps.shape, 0, dtype=np.int32)

    nZerosArray = np.full(nLoci, 0, dtype=np.int32)

    zeros = np.full(nHaps, 0, dtype=np.int32)
    ones = np.full(nHaps, 0, dtype=np.int32)
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
        zeros = np.full(nHaps, 0, dtype=np.int32)
        ones = np.full(nHaps, 0, dtype=np.int32)
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
    zeroOccPrev = np.full(haps.shape, 0, dtype=np.int32)

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

