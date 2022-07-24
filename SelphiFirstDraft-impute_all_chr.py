#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange

from utils import BidiBurrowsWheelerLibrary
from data_utils import get_sample_index, remove_all_samples

from variations import run_hmm


# In[4]:


# PARAMS
chrom = "chr1"
SI_data_folder = f"./new_data/SI_data/{chrom}"


# In[5]:


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


# In[6]:


with open("./data/beagle_data/imputed_samples.txt",mode="r") as f:
    lines = f.readlines()
    samples = [line.strip() for line in lines]


# In[7]:

samples = f7(samples)
len(samples)


# In[9]:


# Load Data

with open(f"{SI_data_folder}/chip_id_list.txt", "rb") as fp:   # Unpickling
    chip_id_list = pickle.load(fp)
with open(f"{SI_data_folder}/full_id_list_full.txt", "rb") as fp:   # Unpickling
    full_id_list = pickle.load(fp)
with open(f"{SI_data_folder}/original_indicies_full.txt", "rb") as fp:   # Unpickling
    original_indicies = pickle.load(fp)
chr_length = len(full_id_list)


# In[10]:


# LOAD GENETIC MAP
import os
# PUT genetic map path here that beagle uses
genetic_map_path = f"/home/nikita/s3-sd-platform-dev/efs/service/genome-files-processing/resource_files/genetic_map_plink/plink.{chrom}.GRCh38.map"

def get_cm(recomb_maps, pos):
    # TODO: Change default pop
    for i in list(recomb_maps["default"].keys()):
        if pos <= i:
            return recomb_maps["default"][i]
    return list(recomb_maps["default"].keys())[-1]

map_pops = ["default",]
recomb_maps = {x:z for (x,z) in [[x, {k:v for (k,v) in [[row[3],row[2]] for _, row in pd.read_csv(genetic_map_path, sep=" ", comment="#").iterrows()]}] for x in map_pops]}
chip_positions_dedup = [int(x.split('-')[1]) for x in chip_id_list]
num_obs = len(chip_positions_dedup)
distances_cm = [get_cm(recomb_maps, x) for x in chip_positions_dedup]


# In[11]:


genetic_map = pd.read_csv(genetic_map_path, sep=" ", comment="#",header=None,usecols=[0,2,3])
genetic_map.columns = ['chr','cM','pos']
genetic_map.set_index('pos',inplace=True)
genetic_map = genetic_map.join(pd.DataFrame(chip_positions_dedup).set_index(0),how='outer')
genetic_map['cM'] = genetic_map['cM'].interpolate('index').fillna(method='bfill')
genetic_map['chr'] = genetic_map.chr.fillna(20.0).astype(int)
genetic_map = genetic_map.reset_index().rename({'index':'pos'},axis=1)
genetic_map = genetic_map[genetic_map['pos'].isin(chip_positions_dedup)].reset_index(drop=True)
distances_cm = genetic_map.cM
distances_cm = list(distances_cm)

print("processed genetic map:")
print(genetic_map)


# In[14]:


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


# In[13]:


ref_panel_full_array_full_packed = zarr.load("./new_data/SI_data/chr1/reference_panel.30x.hg38_chr1_noinfo_full_nppacked.zip")

print("loaded the full reference panel")
print(f"shape: {ref_panel_full_array_full_packed.shape}")
print(ref_panel_full_array_full_packed.__repr__())

# In[15]:

import time
start_time = time.time()

from data_utils import remove_all_samples
CHUNK_SIZE = num_obs
HAPS = [0,1]

for sample in samples[12:]:
    print(sample)
    full_res_ = {}
    sample_index = get_sample_index(sample, samples_txt_path=f"./new_data/SI_data/samples.txt")
    target_full_array = np.zeros((chr_length,2))
    target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
    target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

    ref_panel_full_array = remove_all_samples(ref_panel_full_array_full_packed)
#     ref_panel_full_array = remove_all_samples(ref_panel_full_array_full)
#     ref_panel_full_array = zarr.load("./new_data/SI_data/chr1/reference_panel.30x.hg38_chr1_noinfo_full_nppacked.zip")




#     ref_panel_chip_array = ref_panel_full_array[original_indicies,:]
    ref_panel_chip_array = zarr.load(f"{SI_data_folder}/reference_panel.30x.hg38_{chrom}_noinfo_chip_variants_292samples_removed.zip")
    
    target_chip_array = target_full_array[original_indicies,:]
    combined_ref_panel_chip = np.concatenate([ref_panel_chip_array,target_chip_array],axis=1)
    print("Concatenated shape: ", combined_ref_panel_chip.shape)

    for hap in HAPS:

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

            
        print("temporary fix: for chip_variants with no matches, all are taken as matches")
        for chip_var in trange(0,num_chip_vars):
            
            x = np.unique(BI[:,chip_var])
            if len(x) == 1 and x[0] == 0:
                BI[:,chip_var] = 1
                BJ[:,chip_var] = 1

        matches = BI + BJ -1
        fl = 13
        
        print("Creating initial composite panel")
        composite_ = np.zeros(matches.shape)
        best_matches = {}
        for chip_index in trange(0, matches.shape[1]):
            best_matches[chip_index] = list(np.argsort(matches[:,chip_index])[::-1][:fl])
            for hap_index in best_matches[chip_index]:
                composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
                composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1

#         WOW = ((BI == 1).astype(int) * (BJ > 10).astype(int) * BJ)
#         for i in trange(0, matches.shape[1]):
#             x = np.unique(WOW[:,i], return_counts=True)
#             if len(x[0]) < 25:
#                 continue

#             perc = np.percentile(x[1][1:],99)
#             index_ = np.where(x[1][1:] > perc)[0]
#             repeats = x[0][1:][index_]
#             hapsi = []
#             [hapsi.extend(np.where(BJ[:,i] == x)[0]) for x in range(0,len(repeats))]
#             for hapso in hapsi:
#                 composite_[hapso,i: i + int(BJ[hapso, i])] = 0

        comp_matches_hybrid = (composite_ * matches).astype(int)
        
        print("Calculating Haploid frequencies")
        hap_freq = {}
        for i in trange(0,matches.shape[1]):
            chunk_index = int(i//CHUNK_SIZE)
            hap_freq.setdefault(chunk_index, {})
            
            indexoo = np.flatnonzero(comp_matches_hybrid[:,i])
            for indexo in indexoo:
                hap_freq[chunk_index][indexo] = hap_freq[chunk_index].get(indexo, 0) + 1

        haps_freqs_array_norm_dict = {}
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
            
        # STD_THRESH
        averages = []
        hybrid = (composite_ * matches).astype(int)
        for i in range(0,matches.shape[1]):
            averages.append(np.average(hybrid[np.flatnonzero(hybrid[:,i]),i]))
    
        std_thresh = list(get_std(np.array(averages)))


        std_ = []
        lengthos = []
        final_thresh = []
        nc_thresh = []
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

        composite_ = np.zeros(matches.shape)
        best_matches = {}
        for chip_index in trange(0, matches.shape[1]):
            chunk_index = int(chip_index//CHUNK_SIZE)
            xooi = matches[:,chip_index] * haps_freqs_array_norm_dict[chunk_index]
            best_matches[chip_index] = list(np.argsort(xooi)[::-1][:nc_thresh[chip_index]])
            for hap_index in best_matches[chip_index]:
                composite_[hap_index ,chip_index:int(chip_index + BJ[hap_index, chip_index])] = 1
                composite_[hap_index ,int(chip_index - BI[hap_index, chip_index] + 1):chip_index+1] = 1
    
        
        ordered_matches_test__ = {}
        comp_to_plot = np.zeros(composite_.shape)
        for i in trange(BJ.shape[1]):
            xooi = matches[:,i]
            sorting_key = list(xooi.argsort()[::-1])
        #     sorting_key = list(np.argsort(haps_freqs_array_norm)[::-1])

            uniqes = list(np.where(composite_[:,i] == 1)[0])
            ordered_matches_test__[i] = sorted(uniqes,key=sorting_key.index)
            comp_to_plot[ordered_matches_test__[i],i] = 1
            if len(ordered_matches_test__[i]) == 0:
                ordered_matches_test__[i] = list(np.where(matches[:,i] != 0)[0])
        
        length_matches = {}
        length_matches_normalized = {}
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
        
        print("Start Imputing")
        resultoo_fb = run_hmm(
            original_indicies,
            ref_panel_full_array,
            num_obs,
            ordered_matches_test__,
            distances_cm,
            BI, BJ,
            # THIS VARIABLE IS USELESS, YOU CAN REFACTOR AND REMOVE
            length_matches_normalized,
            chr_length,
            num_hid=matches.shape[0],
            
#             imputed_chr_length=
        )
#         resultoo_fb = run_hmm_variable_N(
#             original_indicies,
#             ref_panel_full_array,
#             num_obs,
#             ordered_matches_test__,
#             distances_cm,
#             nc_thresh,
#             length_matches_normalized,
#         )
        START = original_indicies[0]
        END = original_indicies[-1]
        print("DONE")

        new_x = target_full_array[:,hap]
        y = resultoo_fb
        print(int(y.shape - np.sum(new_x == y)))

        full_res_.setdefault(sample, []).append(resultoo_fb.copy())

        if hap == 1:
            print("saving results")
            with open(f'./method_first_draft/saved_dictionary_{str(sample)}_new_method_{chrom}.pkl', 'wb') as f:
                pickle.dump(full_res_, f)
            arr__1 = full_res_[sample][0]
            arr__2 = full_res_[sample][1]
            final_arr = arr__1 + arr__2

            y_1 = target_full_array[:,0]
            y_2 = target_full_array[:,1]
            final_y = y_1 + y_2
            print("Full results")
            print(int(final_arr.shape - np.sum(final_y == final_arr)))

print("--- %s seconds ---" % (time.time() - start_time))

# In[ ]:


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


# In[ ]:


sample = "HG02429"
results = get_beagle_res(f"./data/beagle_data/292_BEAGLE_5.4_mapYES_neYES_validated_chr1_{sample}.vcf.gz")
beagle_0 = results[0].to_numpy().astype(int)
beagle_1 = results[1].to_numpy().astype(int)
final_arr_beagle = beagle_0 + beagle_1

sample_index = get_sample_index(sample, samples_txt_path=f"./new_data/SI_data/samples.txt")
target_full_array = np.zeros((chr_length,2))
target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

y_1 = target_full_array[:,0]
y_2 = target_full_array[:,1]
final_y = y_1 + y_2
        
print(int(final_arr_beagle.shape - np.sum(final_y == final_arr_beagle)))


# In[ ]:


# THIS PLOT DOESN"T WORK YET NOW

# PLOT results

# del plot_results
import utils_no_mod
# importlib.reload(utils_no_mod)
from utils_no_mod import plot_results
folder_dict = {
    "beagle":"/home/ubuntu/selphi/data/validation_data/beagle_validated/",
#     "selphi HMM":"./",
    "selphi HMM":"/home/ubuntu/selphi/method_first_draft/",
}

# OVERWRITE SAMPLES
# samples = [
#     "HG02470",
# ]

plot_results(
    samples,
    folder_dict,
    ref_panel_full_array_full,
    target_full_array,
    original_indicies,
)

