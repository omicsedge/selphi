from modules.data_utils import get_sample_index
import os

#from black import Result
import numpy as np
from tqdm import trange, tqdm
from collections import OrderedDict
from numba import njit, int8, int64


try:
    from numba.experimental import jitclass
except ModuleNotFoundError:
    from numba import jitclass


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
    X =  np.concatenate([result_1, *Parallel(n_jobs=8)(delayed(parallel_interpolate)(start, end) for (start, end) in list(chunks(range(0, weight_matrix.shape[1]), 5000))), result_2])
    return X


def unpack(ref_panel_full_array_full_packed, hap, orig_index_0, orig_index_1, chr_length):
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


def interpolate_parallel_packed(
    weight_matrix,
    original_indicies,
    original_ref_panel,
    chr_length,
    start,
    end,
):  
    from joblib import Parallel, delayed
    def parallel_interpolate(start, end):
        full_constructed_panel = np.zeros((chr_length, 2), dtype=np.float32)
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
                    unpack(original_ref_panel, j, original_indicies[i], original_indicies[i+1], chr_length) * probs+ 
                    full_constructed_panel[original_indicies[i]:original_indicies[i+1],0]
                )
                # Zero Dosage
                full_constructed_panel[original_indicies[i]:original_indicies[i+1],1] = (
                    np.logical_not(unpack(original_ref_panel, j, original_indicies[i], original_indicies[i+1], chr_length)) * probs+
                    full_constructed_panel[original_indicies[i]:original_indicies[i+1],1]
                )
        result = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)[original_indicies[start]:original_indicies[end]]
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
        # One Dosage
        full_constructed_panel[0:original_indicies[i],0] = (
            unpack(original_ref_panel, j,0, original_indicies[i], chr_length) * probs+ 
            full_constructed_panel[0:original_indicies[i],0]
        )
        # Zero Dosage
        full_constructed_panel[0:original_indicies[i],1] = (
            np.logical_not(unpack(original_ref_panel, j,0, original_indicies[i], chr_length)) * probs+
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
            num=chr_length - original_indicies[-1],
            endpoint=False
        )
        # print(probs)
        # One Dosage
        full_constructed_panel[original_indicies[-1]:,0] = (
            unpack(original_ref_panel, j, original_indicies[-1], -1, chr_length) * probs+ 
            full_constructed_panel[original_indicies[-1]:,0]
        )
        # Zero Dosage
        full_constructed_panel[original_indicies[-1]:,1] = (
            np.logical_not(unpack(original_ref_panel, j, original_indicies[-1], -1, chr_length)) * probs+
            full_constructed_panel[original_indicies[-1]:,1]
        )
        
    result_2 = (full_constructed_panel[:,0] > full_constructed_panel[:,1]).astype(np.int16)[original_indicies[-1]:]
    #####################
    X =  np.concatenate([result_1, *Parallel(n_jobs=8)(delayed(parallel_interpolate)(start, end) for (start, end) in list(chunks(range(0, weight_matrix.shape[1]), 5000))), result_2])
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
            
            plt.plot(lengths, res[key], label = f"{key}")

        ax.set_xlabel("Segment Lengths",fontsize=10)
        ax.set_ylabel("Wrongly imputed variants",fontsize=10)
        plt.rcParams['axes.facecolor']='white'
        plt.rcParams['savefig.facecolor']='white'
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


class BidiBurrowsWheelerLibrary:
    def __init__(self, haplotypeList: np.ndarray, Target: int):
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
        matches_indices = get_matches_indices(
            self.getBackward_Ppa(), self.getBackward_Div(), self.target
        )
        return np.flip(
            matches_indices[0],
            axis=1,
        ), list(
            np.flip(
                np.array(
                    matches_indices[1]
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


def forced_open(file, mode='r'):
    """
    Wrapper around standard python `open`.
    If the file's leaf directory doesn't exist, it creates all intermediate-level directories.
    """
    the_dir = os.path.dirname(os.path.abspath(file))
    if not os.path.isdir(the_dir):
        os.makedirs(the_dir)
    return open(file, mode)

def skip_duplicates(seq):
    """Returns a list of only first occurrences of unique values from the input sequence"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

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
