from hmm_utils import forward, create_composite_chip_panel
import numpy as np
from tqdm import trange, tqdm
from collections import OrderedDict
from numba import njit, int8, int64

try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass

ordered_matches = OrderedDict()
ordered_matches_reverse = OrderedDict()

def get_best_break_points(ordered_matches, BJ, num=25):
    haps = [ordered_matches[x][0] for x in range(0,len(ordered_matches.keys()))]
    temp = []
    haps_ = []
    for i in haps:
        if i not in temp:
            temp.append(i)
            haps_.append(i)
        else:
            haps_.append(-1)
    matches = []
    for c,i in enumerate(haps_):
        if i != -1:
            matches.append(BJ[i,c])
        else:
            matches.append(-1)
    return list(np.argsort(np.array(matches)))[::-1][:num]


def recursive_again(
    BJ,
    starting_index=0,
    number_of_matches=0,
    haps_matches_tuple_list=[],
):
    global ordered_matches
    # print(starting_index)
    # print(number_of_matches)
    if starting_index == 0 and number_of_matches == 0:
        current_index = 0
    else:
        current_index = int(starting_index + number_of_matches)
    
    if current_index == BJ.shape[1]:
        return haps_matches_tuple_list
    hap_index = int(ordered_matches[current_index][0])
    num_matches = int(BJ[hap_index, current_index])
    if num_matches == 0:
        num_matches = 1
    haps_matches_tuple_list.append((hap_index, current_index, num_matches))
    # print(haps_matches_tuple_list)
    current_index = int(current_index)
    # if len(np.unique(BJ[ordered_matches[current_index][1:], current_index])) > 2:
    ordered_matches[current_index] = ordered_matches[current_index][1:] + [
        ordered_matches[current_index][0]
    ]
    return recursive_again(BJ, current_index, num_matches, haps_matches_tuple_list)


def reverse_recursive_again(
    BI,
    starting_index=14777,
    number_of_matches=0,
    haps_matches_tuple_list=[],
):
    global ordered_matches_reverse

    if starting_index == 14777 and number_of_matches == 0:
        current_index = BI.shape[1] - 1
    else:
        current_index = int(starting_index - number_of_matches)
    # print(current_index)
    if current_index == 0:
        return haps_matches_tuple_list
    hap_index = int(ordered_matches_reverse[current_index][0])
    num_matches = int(BI[hap_index, current_index]) - 1
    if num_matches == 0:
        num_matches = 1
    if num_matches == -1:
        num_matches = 1
    # print(hap_index)
    # print(num_matches)
    # print("--")
    haps_matches_tuple_list.append((hap_index, current_index, num_matches))
    # print(haps_matches_tuple_list)
    # if (
    #     len(np.unique(BI[ordered_matches_reverse[current_index][1:], current_index]))
    #     > 2
    # ):
    ordered_matches_reverse[current_index] = ordered_matches_reverse[current_index][
        1:
    ] + [ordered_matches_reverse[current_index][0]]
    return reverse_recursive_again(
        BI, current_index, num_matches, haps_matches_tuple_list
    )


def construct_full_hap(original_ref_panel, original_indicies, haps_matches_tuple_list):
    """ """
    full_constructed_hap = []
    full_constructed_hap.append(
        original_ref_panel[
            : original_indicies[0],
            haps_matches_tuple_list[0][0],
        ]
    )
    # hapso = [x[0] for x in haps_matches_tuple_list]
    for hap, current_index, matches in haps_matches_tuple_list:
        # print(current_index)
        # print(matches)
        # print('--')
        if current_index + matches != 14778:
            full_constructed_hap.append(
                original_ref_panel[
                    original_indicies[current_index] : original_indicies[
                        current_index + matches
                    ],
                    hap,
                ]
                * matches
            )
        else:
            full_constructed_hap.append(
                original_ref_panel[
                    original_indicies[current_index] : original_indicies[
                        -1
                    ],
                    hap,
                ]
                * matches
            )
    full_constructed_hap.append(
        original_ref_panel[
            original_indicies[-1] :,
            hap,
        ]
    )

    return np.concatenate(full_constructed_hap)


def construct_full_hap_matches(
    original_ref_panel, original_indicies, haps_matches_tuple_list
):
    """ """
    hap = original_ref_panel.shape[1] - 1
    full_constructed_hap = []
    full_constructed_hap.append(
        original_ref_panel[
            : original_indicies[0],
            hap,
        ]
    )
    # hapso = [x[0] for x in haps_matches_tuple_list]
    for _, current_index, matches in haps_matches_tuple_list:

        # print(current_index)
        # print(matches)
        # print('--')
        if current_index + matches != 14778:
            full_constructed_hap.append(
                original_ref_panel[
                    original_indicies[current_index] : original_indicies[
                        current_index + matches
                    ],
                    hap,
                ]
                * matches
            )
        else:
            full_constructed_hap.append(
                original_ref_panel[
                    original_indicies[current_index] : original_indicies[
                        -1
                    ],
                    hap,
                ]
                * matches
            )
    full_constructed_hap.append(
        original_ref_panel[
            original_indicies[-1] :,
            hap,
        ]
    )

    return np.concatenate(full_constructed_hap)


def create_composite_ref_panel(
    original_ref_panel,
    BJ,
    BI,
    original_indicies,
    matches_sum_indxs,
    indicies_max_matches,
    haploid_number=400,
):
    """
    This function create a composite reference panel
    """
    global ordered_matches
    global ordered_matches_reverse

    from collections import defaultdict

    ordered_hap_indices = {}
    ordered_matches = {}
    lengtho = 25

    for i in trange(0, BJ.shape[1]):
        y = np.where(
            BJ[:,i] >= np.percentile(BJ[:,i],0)
        )[0]
        # if i < 14700 and i > 100:
        #     y =  np.where(
        #         BJ[y,i] >= 10
        #     )[0]
        #     y =  np.where(
        #         BI[y,i] >= 10
        #     )[0]
        if len(list(y)) < 5:
            y =  np.where(
                BJ[:,i] >= 0
            )[0]
        x = BJ[y, i]
        new_index = y[np.argsort(x)[::-1]]  # Descending
        ordered_matches[i] = BJ[new_index, i][:lengtho]
        ordered_hap_indices[i] = new_index.copy()[:lengtho]

    for i in trange(0, BJ.shape[1]):
        d = defaultdict(list)
        for key, val in np.concatenate(
            [
                ordered_matches[i].reshape(len(ordered_matches[i]), 1),
                ordered_hap_indices[i].reshape(len(ordered_hap_indices[i]), 1),
            ],
            axis=1,
        ):
            d[key].append((val, np.where(matches_sum_indxs == val)[0][0]))
        sorted_by_second = {x: sorted(d[x], key=lambda tup: tup[1]) for x in d}
        ordered_matches[i] = [
            int(dd[0]) for e in sorted_by_second for dd in sorted_by_second[e]
        ]

    for i in trange(0, BI.shape[1]):
        y = np.where(BI[:,i] >= np.percentile(BI[:,i],0))[0]
        # if i < 14700 and i > 100:
        #     y =  np.where(
        #         BI[y,i] >= 10
        #     )[0]
        #     y =  np.where(
        #         BJ[y,i] >= 10
        #     )[0]
        if len(list(y)) < 5:
            y =  np.where(
                BI[:,i] >= 0
            )[0]
        x = BI[y,i]
        new_index = y[np.argsort(x)[::-1]]  # Descending
        ordered_matches_reverse[i] = BI[new_index, i][:lengtho]
        ordered_hap_indices[i] = new_index.copy()[:lengtho]

    for i in trange(0, BJ.shape[1]):
        d = defaultdict(list)
        for key, val in np.concatenate(
            [
                ordered_matches_reverse[i].reshape(len(ordered_matches_reverse[i]), 1),
                ordered_hap_indices[i].reshape(len(ordered_hap_indices[i]), 1),
            ],
            axis=1,
        ):
            d[key].append((val, np.where(matches_sum_indxs == val)[0][0]))
        sorted_by_second = {x: sorted(d[x], key=lambda tup: tup[1]) for x in d}
        ordered_matches_reverse[i] = [
            int(dd[0]) for e in sorted_by_second for dd in sorted_by_second[e]
        ]

    indicies_max_matches = get_best_break_points(ordered_matches, BJ)
    indicies_max_matches = np.array(indicies_max_matches)
    indicies_max_matches = np.delete(
        indicies_max_matches, np.where(indicies_max_matches == 0), axis=0
    )

    # full_constructed_panel = np.zeros(
    #     (1647102, haploid_number + len(indicies_max_matches)), dtype=np.int16
    # )
    # matches_for_haps = np.ones(
    #     (1647102, haploid_number + len(indicies_max_matches)), dtype=np.int16
    # )
    all_haps = []
    for co, starting_idx in enumerate(tqdm(indicies_max_matches)):
        final_haps = []

        # print(starting_idx)
        haps_matches_tuple_list_reverse = reverse_recursive_again(
            BI, starting_idx, 0, []
        )
        temp_tuple = list(haps_matches_tuple_list_reverse[-1])
        temp_tuple[1] = 0
        haps_matches_tuple_list_reverse.append(tuple(temp_tuple))
        haps_matches_tuple_list_reverse_final = []
        [
            haps_matches_tuple_list_reverse_final.insert(
                0, [x[0], haps_matches_tuple_list_reverse[c + 1][1], x[-1]]
            )
            for c, x in enumerate(haps_matches_tuple_list_reverse[:-1])
        ]
        final_haps.extend(haps_matches_tuple_list_reverse_final)
        

        haps_matches_tuple_list = recursive_again(BJ.astype(int), starting_idx, 0, [])
        final_haps.extend(haps_matches_tuple_list)
        # if co == 66:
        #     print(final_haps)
        # print(len(final_haps))
        # print(final_haps)
        all_haps.append(final_haps)
        # print(final_haps)
        # full_constructed_panel[:, co] = construct_full_hap(
        #     original_ref_panel, original_indicies, final_haps
        # )
        # matches_for_haps[:, co] = construct_full_hap_matches(
        #     matches_for_haps, original_indicies, final_haps
        # )

    # for j in trange(
    #     len(indicies_max_matches), len(indicies_max_matches) + haploid_number
    # ):
    #     haps_matches_tuple_list = recursive_again(BJ, 0, 0, [])
    #     # print(haps_matches_tuple_list)
    #     # break
    #     full_constructed_panel[:, j] = construct_full_hap(
    #         original_ref_panel, original_indicies, haps_matches_tuple_list
    #     )
    #     matches_for_haps[:, j] = construct_full_hap_matches(
    #         matches_for_haps, original_indicies, haps_matches_tuple_list
    #     )
    #     all_haps.append(haps_matches_tuple_list)
    return (all_haps, None, None)


def create_weight_matrix(all_haps):

    composite_chip_panel, composite_chip_matches = create_composite_chip_panel(all_haps, 14778)
    weight_matrix_lists = []
    # weight_matrix_lists.append(np.full((composite_chip_matches.shape[0],1), 1/composite_chip_matches.shape[0]))
    [weight_matrix_lists.append(np.expand_dims((composite_chip_matches[:,i]/sum(composite_chip_matches[:,i])),axis=1)) for i in range(0,composite_chip_matches.shape[1])]
    weight_matrix = np.concatenate(weight_matrix_lists, axis=1)
    return (composite_chip_panel, weight_matrix)

def interpolate(
    all_haps,
    original_indicies,
    original_ref_panel,
):  
    full_constructed_panel = np.zeros((1647102, 2), dtype=np.float32)
    composite_chip_panel, weight_matrix = create_weight_matrix(all_haps)


    for i in trange(0,weight_matrix.shape[1]-1):
        temp_haps = []
        for j in range(0,weight_matrix.shape[0]):
            # if (weight_matrix[j, i],weight_matrix[j, i+1]) in temp_haps:
            #     continue
            # temp_haps.append((weight_matrix[j, i],weight_matrix[j, i+1]))
            probs = np.linspace(
                weight_matrix[j, i],
                weight_matrix[j, i+1],
                num=original_indicies[i+1] - original_indicies[i],
                endpoint=False
            )
            # One Dosage
            full_constructed_panel[original_indicies[i]:original_indicies[i+1],0] = (
                original_ref_panel[original_indicies[i]:original_indicies[i+1],composite_chip_panel[j,i]] * probs * 0.5+
                original_ref_panel[original_indicies[i]:original_indicies[i+1],composite_chip_panel[j,i + 1]] * probs * 0.5+ 
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],0]
            )
            # Zero Dosage
            full_constructed_panel[original_indicies[i]:original_indicies[i+1],1] = (
                np.logical_not(original_ref_panel[original_indicies[i]:original_indicies[i+1],composite_chip_panel[j,i]]) * probs * 0.5+
                np.logical_not(original_ref_panel[original_indicies[i]:original_indicies[i+1],composite_chip_panel[j,i +1]]) * probs * 0.5+
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],1]
            )
    result = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)
    return result


def interpolate_(
    weight_matrix,
    original_indicies,
    original_ref_panel,
):  
    full_constructed_panel = np.zeros((1647102, 2), dtype=np.float32)


    for i in trange(0,14777):
        temp_haps = []
        for j in range(0,weight_matrix.shape[0]):
            if weight_matrix[j, i+1] < 1/weight_matrix.shape[0] and weight_matrix[j, i] < 1/weight_matrix.shape[0]:
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
    result = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)
    return result

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
from data_utils import get_sample_index

def get_beagle_res(file_name):
    beagle_array = pd.read_csv(file_name, sep="\t", header=None, comment="#")
    beagle_array.drop([0,1,3,4,5,6,7,8],axis=1,inplace=True)
    concat_beagle = pd.concat([
        beagle_array.applymap(lambda x: str(x).split(":")[0].replace("/","|").split("|")[0]),
        beagle_array.applymap(lambda x: str(x).split(":")[0].replace("/","|").split("|")[-1]),
    ],axis=1)
    concat_beagle.drop([2],axis=1,inplace=True)
    concat_beagle.columns = [0,1]
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
    for c, sample in enumerate(samples):
        sample_index = get_sample_index(sample)
        target_full_array[:,0] = ref_panel_full_array_full[:,sample_index[0]]
        target_full_array[:,1] = ref_panel_full_array_full[:,sample_index[1]]

        indicies_for_comparison = np.where(
        (np.sum(ref_panel_full_array_full,axis=1) >0) & 
        # (np.sum(ref_panel_full_array_full,axis=1) > 0)  &
        (np.sum(target_full_array,axis=1) >= 0))[0]
        combined_target_full = target_full_array[:,0] + target_full_array[:,1]

        print(sample)
        results = {}
        res = {}
        new_x = target_full_array[:,0] + target_full_array[:,1]
        for key in folder_dict.keys():
            if key != "beagle":
                with open(f"{folder_dict[key]}saved_dictionary_{sample}_new_method.pkl", "rb") as fp:   # Unpickling
                    results[key] = pickle.load(fp)
            else:
                results[key] = get_beagle_res(f"{folder_dict[key]}{sample}_BEAGLE_5.4_mapYES_neYES.vcf.gz")
                beagle_0 = results[key][0].to_numpy().astype(int)
                beagle_1 = results[key][1].to_numpy().astype(int)
                results[key] = {sample:[beagle_0,beagle_1]}
            
            arr__1 = results[key][sample][0]
            arr__2 = results[key][sample][1]
            # arr__1[original_indicies] = target_full_array[original_indicies,0]
            # arr__2[original_indicies] = target_full_array[original_indicies,1]

            y = arr__1 + arr__2
            print(f"COMBINED {key} ERROR: {int(y.shape - np.sum(new_x == y))}\nTOTAL IMPUTED SEGMENT LENGTH: {y.shape[0]}")
            print()
            combined_results = arr__1 + arr__2
            
            STEP_LENGTH = 500
            MAX_LENGTH = y.shape[0]-STEP_LENGTH
            lengths = list(range(STEP_LENGTH,MAX_LENGTH,STEP_LENGTH))
            for LENGTH in lengths:
                res.setdefault(key, []).append(LENGTH - np.sum( combined_target_full[:LENGTH] == combined_results[:LENGTH] ))

        fig, ax = plt.subplots(figsize=(16,8))    # to create the picture plot empty  
        for key in folder_dict.keys(): 
            
            plt.plot(lengths, res[key], label = f"{key}")

        ax.set_xlabel("Segment Lengths",fontsize=10)
        ax.set_ylabel("Wrongly imputed variants",fontsize=10)
        plt.rcParams['axes.facecolor']='white'
        plt.rcParams['savefig.facecolor']='white'
        plt.legend()

        plt.show()
            # plt.savefig(f'image_name_{sample}_new.jpg',format="jpg",dpi=200,transparent=True,bbox_inches='tight')

#####################################
###            B-PBWT             ###
###      Bidirectional PBWT       ###
#####################################


class BidiBurrowsWheelerLibrary:
    def __init__(self, haplotypeList, Target):
        self.library_back = createBWLibrary(np.array(haplotypeList))
        self.library_forw = createBWLibrary(np.flip(np.array(haplotypeList), axis=1))
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
            self.Shape, np.arange(1, self.Shape[1] + 1), dtype=np.int64
        ) - np.flip(self.library_forw.div, axis=1)

    def getBackward_Div(self):
        return np.full(
            self.Shape, np.arange(1, self.Shape[1] + 1), dtype=np.int64
        ) - np.flip(self.library_back.div, axis=1)

    def getForward_matches_indices(self):
        return get_matches_indices(
            self.getForward_Ppa(), self.getForward_Div(), self.target
        )

    def getBackward_matches_indices(self):
        return np.flip(
            get_matches_indices(
                self.getBackward_Ppa(), self.getBackward_Div(), self.target
            )[0],
            axis=1,
        ), list(
            np.flip(
                np.array(
                    get_matches_indices(
                        self.getBackward_Ppa(), self.getBackward_Div(), self.target
                    )[1]
                )
            )
        )


jit_BidiBurrowsWheelerLibrary_s = OrderedDict()
jit_BidiBurrowsWheelerLibrary_s["ppa"] = int64[:, :]
jit_BidiBurrowsWheelerLibrary_s["div"] = int64[:, :]
jit_BidiBurrowsWheelerLibrary_s["zeroOccPrev"] = int64[:, :]
jit_BidiBurrowsWheelerLibrary_s["nZeros"] = int64[:]
jit_BidiBurrowsWheelerLibrary_s["haps"] = int8[:, :]


@jitclass(jit_BidiBurrowsWheelerLibrary_s)
class jit_BidiBurrowsWheelerLibrary:
    def __init__(self, ppa, div, nZeros, zeroOccPrev, haps):
        self.ppa = ppa
        self.div = div
        self.nZeros = nZeros
        self.zeroOccPrev = zeroOccPrev
        self.haps = haps

    def getValues(self):
        return (self.ppa, self.div, self.nZeros, self.zeroOccPrev, self.haps)


@njit
def createBWLibrary(haps):

    # Definitions.
    # haps : a list of haplotypes
    # ppa : an ordering of haps in lexographic order (ppa_matrix)
    # div : Number of loci of a[i,j+k] == a[i,-1, j+k] (number of mactches) the length of the longest match between each (sorted) haplotype and the previous (sorted) haplotype.

    nHaps = haps.shape[0]
    nLoci = haps.shape[1]
    ppa = np.full(haps.shape, 0, dtype=np.int64)
    div = np.full(haps.shape, 0, dtype=np.int64)

    nZerosArray = np.full(nLoci, 0, dtype=np.int64)

    zeros = np.full(nHaps, 0, dtype=np.int64)
    ones = np.full(nHaps, 0, dtype=np.int64)
    dZeros = np.full(nHaps, 0, dtype=np.int64)
    dOnes = np.full(nHaps, 0, dtype=np.int64)

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
        zeros = np.full(nHaps, 0, dtype=np.int64)
        ones = np.full(nHaps, 0, dtype=np.int64)
        dZeros = np.full(nHaps, 0, dtype=np.int64)
        dOnes = np.full(nHaps, 0, dtype=np.int64)

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
    zeroOccPrev = np.full(haps.shape, 0, dtype=np.int64)

    for i in range(1, nLoci):
        count = 0
        for j in range(0, nHaps):
            if haps[ppa[j, i], i - 1] == 0:
                count += 1
            zeroOccPrev[j, i] = count

    library = jit_BidiBurrowsWheelerLibrary(ppa, div, nZerosArray, zeroOccPrev, haps)
    return library


def replace_col(array, col_index, hap_index):
    """
    Expects array
    """

    def accumelate_before(index, array):
        arrayo = np.flip(np.maximum.accumulate(np.flip(array[: index + 1])))
        return arrayo

    def accumelate_after(index, array):
        arrayo = np.maximum.accumulate(array[index + 1 :])
        return arrayo

    acc_before = accumelate_before(hap_index, array)
    acc_before[:hap_index] = acc_before[1 : hap_index + 1]
    return (
        col_index
        + 1
        - np.concatenate(
            [
                acc_before,
                accumelate_after(hap_index, array),
            ]
        )
    )


# Function to replace values with match lengths
def get_matches_indices(ppa_matrix, div_matrix, target=6401):
    forward_pbwt_matches = np.zeros(ppa_matrix.shape)
    forward_pbwt_hap_indices = list()
    for i in range(0, forward_pbwt_matches.shape[1]):
        hap_index = int(np.where(ppa_matrix[:, i] == target)[0])
        forward_pbwt_matches[:, i] = replace_col(div_matrix[:, i], i, hap_index)
        forward_pbwt_hap_indices.append(hap_index)
    return (forward_pbwt_matches.astype(int), forward_pbwt_hap_indices)
