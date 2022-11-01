from functools import lru_cache
from typing import Dict, List, Literal, OrderedDict
import os
from collections import OrderedDict


import pandas as pd
import numpy as np
import zarr
import pickle
from tqdm import trange


from modules.data_utils import get_sample_index, remove_all_samples, load_samples_list
from modules.utils import (
    vcf_haploid_order_to_internal,
)





######################################
#                                    #
#            GLOBAL DATA             #
#                                    #
######################################



def load_test_samples_indices(
    test_samples_names_path:str = "./data/beagle_data/imputed_samples.txt",
    all_samples_names_path:str = "./data/SI_data/samples.txt",
):
    """
    Creates two dictionaries:
     - one maps test sample names to their corresponding indicies in the full reference panel,
     - another one maps all other sample names that are used as the actual reference panel

    Makes sure the test samples are ordered as in the file
    """
    test_samples_list = load_samples_list(os.path.abspath(test_samples_names_path)) # abspath is needed for proper caching
    full_samples_list = load_samples_list(os.path.abspath(all_samples_names_path))  # abspath is needed for proper caching

    test_samples_indices: OrderedDict[str,int] = OrderedDict()
    refpan_samples_indices: OrderedDict[str,int] = OrderedDict()

    for idx, smpl in enumerate(full_samples_list):
        if smpl in test_samples_list:
            # test_samples_indices[smpl] = idx
            pass
        else:
            refpan_samples_indices[smpl] = idx

    for idx, smpl in enumerate(test_samples_list):
        test_samples_indices[smpl] = full_samples_list.index(smpl)

    return test_samples_indices, refpan_samples_indices



def load_variants_ids(
    chip_id_list_file: str,
    full_id_list_file: str,
    original_indicies_file: str,
):
    """
    `chip_id_list` contains ids available in the input sample

    `full_id_list` contains ids available in samples from the reference panel

    all ids in `chip_id_list` are included in `full_id_list`, while `original_indicies`
    is the list of indicies of ids in the `chip_id_list` as they appear in `full_id_list`
    """
    with open(chip_id_list_file, "rb") as fp:   # Unpickling
        chip_id_list: List[str] = pickle.load(fp)
    with open(full_id_list_file, "rb") as fp:   # Unpickling
        full_id_list: List[str] = pickle.load(fp)
    with open(original_indicies_file, "rb") as fp:   # Unpickling
        original_indicies: List[int] = pickle.load(fp)

    assert len(chip_id_list) == len(original_indicies)
    for i in range(len(original_indicies)):
        oi = original_indicies[i]
        assert chip_id_list[i] == full_id_list[oi]

    return chip_id_list, full_id_list, original_indicies


def load_and_interpolate_genetic_map(
    genetic_map_path: str,
    chip_id_list: List[str],
):
    """
    Loads the genetic map, linearly interpolates the cM coordinates for chip BP positions.

    Takes:
     - `genetic_map_path` - path to the genetic map in the "PLINK-format"
     - `chip_id_list` - list of chip variant ids (e.g. `'1-792461-G-A'`)

    Returns two lists:
        - `chip_BP_positions` - chip (input) BP positions
        - `chip_cM_coordinates` - corresponding cM coordinates for the chip positions, inferred from the genetic map
    """
    chip_BP_positions = [int(x.split('-')[1]) for x in chip_id_list]

    genetic_map = pd.read_csv(genetic_map_path, sep=" ", comment="#",header=None,usecols=[0,2,3])
    genetic_map.columns = ['chr','cM','pos'] # type: ignore
    genetic_map.set_index('pos',inplace=True)
    genetic_map = genetic_map.join(pd.DataFrame(chip_BP_positions).set_index(0),how='outer')
    genetic_map['cM'] = genetic_map['cM'].interpolate('index').fillna(method='bfill')
    genetic_map['chr'] = genetic_map.chr.fillna(20.0).astype(int)
    genetic_map = genetic_map.reset_index().rename({'index':'pos'},axis=1)
    genetic_map = genetic_map[genetic_map['pos'].isin(chip_BP_positions)].reset_index(drop=True)
    chip_cM_coordinates = genetic_map.cM # type: ignore
    chip_cM_coordinates: List[float] = list(chip_cM_coordinates)

    assert len(chip_cM_coordinates) == len(chip_BP_positions)
    return chip_BP_positions, chip_cM_coordinates


def load_reference_panels(
    full_dataset_file: str,
    chip_sites_dataset_file: str,
    ref_samples_list_file: str,
    samples_tobe_removed_list_file: str,
):
    """
    Loads/creates 3 datasets:
     - the full packed original full-sequenced dataset (includes true test samples)
     - the same dataset, but without test samples (i.e. full-sequenced reference panel)
     - the reference panel without test samples (i.e. reference panel with chip sites only)
    """
    #1
    ref_panel_full_array_full_packed: np.ndarray = zarr.load(full_dataset_file) # type: ignore # enforce the type
    #2
    ref_panel_full_array = remove_all_samples(ref_panel_full_array_full_packed, ref_samples_list_file, samples_tobe_removed_list_file)
    #3
    ref_panel_chip_array: np.ndarray = zarr.load(chip_sites_dataset_file) # type: ignore # enforce the type

    assert ref_panel_full_array.shape[0] == ref_panel_full_array_full_packed.shape[0] # number of sites
    assert ref_panel_full_array.shape[1] <= ref_panel_full_array_full_packed.shape[1] # number of samples
    return ref_panel_full_array_full_packed, ref_panel_full_array, ref_panel_chip_array


def load_chip_reference_panel(
    chip_sites_dataset_file: str,
    refpan_haploid_indices: List[int],
):
    """
    Loads/creates 2 chip-sites datasets:
     - the full reference panel (includes test samples) 
     - the "blank" "combined" reference panel: 
          the reference panel without test samples, with 2 blank places for the input sample haploids
    """
    #1
    full_ref_panel_chip_array: np.ndarray = zarr.load(chip_sites_dataset_file) # type: ignore # enforce the type

    #2
    num_hid = len(refpan_haploid_indices)
    blank_combined_ref_panel_chip: np.ndarray = np.zeros(
        (full_ref_panel_chip_array.shape[0], num_hid+2),
        dtype=full_ref_panel_chip_array.dtype.type
    )
    internal_order = [vcf_haploid_order_to_internal(i, int(num_hid/2)) for i in range(num_hid)]
    blank_combined_ref_panel_chip[:, :num_hid] = full_ref_panel_chip_array[:, refpan_haploid_indices][:, internal_order]

    return full_ref_panel_chip_array, blank_combined_ref_panel_chip


@lru_cache(1)
def load_full_reference_panels(
    full_dataset_file: str,
    ref_samples_list_file: str,
    samples_tobe_removed_list_file: str,
):
    """
    Loads/creates 2 datasets:
     - the full packed original full-sequenced dataset (includes true test samples)
     - the same dataset, but without test samples (i.e. full-sequenced reference panel)
    """
    #1
    ref_panel_full_array_full_packed: np.ndarray = zarr.load(full_dataset_file) # type: ignore # enforce the type
    #2
    ref_panel_full_array = remove_all_samples(ref_panel_full_array_full_packed, ref_samples_list_file, samples_tobe_removed_list_file)

    assert ref_panel_full_array.shape[0] == ref_panel_full_array_full_packed.shape[0] # number of sites
    assert ref_panel_full_array.shape[1] <= ref_panel_full_array_full_packed.shape[1] # number of samples
    return ref_panel_full_array_full_packed, ref_panel_full_array




######################################
#                                    #
#        SAMPLE-SPECIFIC DATA        #
#                                    #
######################################


def load_sample(
    sample: str,
    chr_length: int,
    original_indicies: List[int],
    ref_panel_full_array_full_packed: np.ndarray,
    ref_panel_chip_array: np.ndarray,
):
    """
    Loading two datasets of the test sample: 
     - the true full-sequenced sample,
     - the unimputed one (chip sites only)

    And creates a combined array:
     - the reference panel haploids + the test sample (2 haploids) (chip sites only)

    """
    sample_index = get_sample_index(sample, samples_txt_path=f"./data/SI_data/samples.txt")
    target_full_array = np.zeros((chr_length, 2), dtype=np.int8)
    target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
    target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

    target_chip_array: np.ndarray = target_full_array[original_indicies,:]
    combined_ref_panel_chip: np.ndarray = np.concatenate([ref_panel_chip_array,target_chip_array],axis=1)

    print("Concatenated shape (the reference panel haploids + the test sample (chip sites only)): ", combined_ref_panel_chip.shape)

    return target_full_array, target_chip_array, combined_ref_panel_chip





def combine_ref_panel_chip(
    test_sample_index: int,
    full_ref_panel_chip_array: np.ndarray,
    ref_panel_chip_array: np.ndarray,
):
    """
    Creates a "combined" reference panel with chip sites, consisting of
        the reference panel haploids + the test sample.

    Do not use, it's suboptimal.
    """

    target_chip_array: np.ndarray = full_ref_panel_chip_array[:,test_sample_index]
    combined_ref_panel_chip: np.ndarray = np.concatenate([ref_panel_chip_array,target_chip_array],axis=1)
    print("Concatenated shape (the reference panel haploids + the test sample (chip sites only)): ", combined_ref_panel_chip.shape)
    return combined_ref_panel_chip


def load_sample_fullsequence(
    sample: str,
    chr_length: int,
    original_indicies: List[int],
    ref_panel_full_array_full_packed: np.ndarray,
    ref_panel_chip_array: np.ndarray,
):
    """
    Old

    Do not use
    """
    sample_index = get_sample_index(sample, samples_txt_path=f"./data/SI_data/samples.txt")
    target_full_array = np.zeros((chr_length,2))
    target_full_array[:,0] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[0]])[:chr_length]
    target_full_array[:,1] = np.unpackbits(ref_panel_full_array_full_packed[:,sample_index[1]])[:chr_length]

    return target_full_array



